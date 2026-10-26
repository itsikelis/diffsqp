"""SQP subproblem QP solved by the bsqp ADMM kernels (C++/OpenMP or CUDA).

Drop-in replacement of `admm_qp_solve` / `admm_qp_resume` (same arguments,
same AdmmSolution / AdmmLog outputs). Layout: diffsqp stores `horizon` states
and `horizon - 1` controls, bsqp's `horizon` is the number of controls, so the
tensors are passed as they are.

Differences with the torch implementation:
- The dual residual is the QP stationarity (OSQP style), not
  ||M^T rho (z - z_prev)||, so termination and rho updates differ slightly.
- Equality rows (lb == ub) always get rho_bar * 1e3 (torch: rho_max on reset).
- The tolerance schedules (admm_*_tolerance_final) are not supported.
- A failed factorization is only detected through non-finite residuals; on the
  CPU, Eigen's Cholesky of an indefinite R' gives finite values, so it is not
  flagged there.
"""

import dataclasses
from typing import NamedTuple

import bsqp
import torch

from diffsqp.solvers.admm_qp import UNBOUNDED, constraint_map
from diffsqp.solvers.lqr import MAX_REG_RETRIES
from diffsqp.types import AdmmLog, AdmmSolution

# Dual regularization of the underactuation KKT blocks, as in lqr.py
DUAL_REG = 1e-8


class BsqpState(NamedTuple):
    """bsqp problem and solution of the last ADMM call (AdmmLog.factorization)."""

    problem: bsqp.AdmmProblem
    params: bsqp.AdmmParameters
    solution: bsqp.AdmmSolution


def bsqp_parameters(parameters):
    if (
        parameters.admm_abs_tolerance_final > 0.0
        or parameters.admm_rel_tolerance_final > 0.0
    ):
        raise ValueError("admm_backend='bsqp' does not support tolerance schedules")
    return bsqp.AdmmParameters(
        lqr=bsqp.LqrParameters(gamma=0.0, eps=DUAL_REG),
        eps_abs=parameters.admm_abs_tolerance,
        eps_rel=parameters.admm_rel_tolerance,
        max_iter=parameters.admm_max_iter,
        rho0=parameters.admm_rho_init,
        rho_min=float(parameters.admm_rho_min),
        rho_max=float(parameters.admm_rho_max),
        sigma=parameters.admm_sigma,
        alpha=parameters.admm_alpha,
        check_every=parameters.admm_check_every,
        rho_update_every=parameters.admm_rho_update_iter_freq,
        rho_strategy=(
            bsqp.RhoStrategy.OSQP
            if parameters.admm_update_rho
            else bsqp.RhoStrategy.FIXED
        ),
        osqp_thr=parameters.admm_adaptive_rho_tolerance,
    )


def _regularized_R(R, reg):
    """R + reg I per batch element (bsqp's gamma is a single scalar)."""
    eye = torch.eye(R.shape[-1], dtype=R.dtype, device=R.device)
    return R + reg.view(-1, 1, 1, 1).to(R.dtype) * eye


def bsqp_problem(lqr_mat, lqr_reg):
    """bsqp QP of the linearization; bounds beyond UNBOUNDED become infinite."""
    B = lqr_mat.Q.shape[0]
    lb = lqr_mat.lb.expand(B, *lqr_mat.lb.shape)
    ub = lqr_mat.ub.expand(B, *lqr_mat.ub.shape)
    inf = torch.tensor(float("inf"), dtype=lb.dtype, device=lb.device)
    lb = torch.where(lb <= -UNBOUNDED, -inf, lb)
    ub = torch.where(ub >= UNBOUNDED, inf, ub)
    return bsqp.AdmmProblem(
        A=lqr_mat.A,
        B=lqr_mat.B,
        b=lqr_mat.b,
        Q=lqr_mat.Q,
        q=lqr_mat.q,
        R=_regularized_R(lqr_mat.R, lqr_reg),
        r=lqr_mat.r,
        S=lqr_mat.S,
        C=lqr_mat.C,
        D=lqr_mat.D,
        d=lqr_mat.d,
        M=lqr_mat.M,
        N=lqr_mat.N,
        n=lqr_mat.n,
        lb=lb,
        ub=ub,
    )


def _unconstrained(bprob, params):
    fields = dataclasses.fields(bsqp.LqrProblem)
    lqr = bsqp.LqrProblem(**{f.name: getattr(bprob, f.name) for f in fields})
    return bsqp.riccati(lqr, params.lqr)


def _warm_start(problem, parameters, lqr_mat, bprob, unconstrained, previous_solution):
    """bsqp solution buffers initialized as `admm_qp.new_solution`."""
    sol = bsqp.AdmmSolution.zeros(bprob)
    if parameters.admm_warm_start_unconstrained:
        sol.dx.copy_(unconstrained.dx)
        sol.du.copy_(unconstrained.du)

    if parameters.admm_warm_start_z:
        z = constraint_map(lqr_mat, sol.dx, sol.du)
        sol.z.copy_(torch.clamp(z, bprob.lb - bprob.n, bprob.ub - bprob.n))

    # rho_bar <= 0 means params.rho0
    sol.rho_bar.fill_(-1.0)
    if previous_solution is not None:
        if not parameters.admm_reset_ksi:
            sol.ineq_mult.copy_(previous_solution.ksi)
        if not parameters.admm_reset_rho:
            sol.rho_bar.copy_(previous_solution.rho_common)
    return sol


def _tensor_fields(obj):
    for f in dataclasses.fields(obj):
        value = getattr(obj, f.name)
        if value is not None:
            yield f.name, value


def _index(obj, idx):
    """Batch elements `idx` of a bsqp problem / solution dataclass."""
    fields = {name: t[idx].contiguous() for name, t in _tensor_fields(obj)}
    return dataclasses.replace(obj, **fields)


def _scatter(dst, src, idx):
    for name, t in _tensor_fields(dst):
        t[idx] = getattr(src, name)


def _failed(sol):
    bad = sol.status == bsqp.QpStatus.FAILED
    return bad | ~torch.isfinite(sol.dx).flatten(1).all(dim=1)


def _admm(bprob, params, sol, lqr_mat, lqr_reg):
    """bsqp ADMM; failed elements are retried with 10x the regularization
    (in place, so it persists for the caller, as in `lqr_factorize`)."""
    start = dataclasses.replace(sol, **{n: t.clone() for n, t in _tensor_fields(sol)})
    bsqp.admm(bprob, params, sol)
    for _ in range(MAX_REG_RETRIES):
        failed = _failed(sol)
        if not failed.any():
            break
        idx = failed.nonzero().squeeze(1)
        lqr_reg[idx] *= 10.0
        bprob.R[idx] = _regularized_R(lqr_mat.R[idx], lqr_reg[idx])
        sub = _index(start, idx)
        bsqp.admm(_index(bprob, idx), params, sub)
        _scatter(sol, sub, idx)
    return _failed(sol)


def _to_diffsqp(sol, n_h):
    mu = sol.mu.clone()
    # diffsqp convention: no multiplier on the fixed initial state
    mu[:, 0] = 0.0
    nu = sol.nu if sol.nu is not None else sol.du.new_zeros(*sol.du.shape[:2], n_h)
    return AdmmSolution(
        dx=sol.dx,
        du=sol.du,
        mu=mu,
        nu=nu,
        z=sol.z,
        ksi=sol.ineq_mult,
        rho=sol.rho,
        rho_inv=1.0 / sol.rho,
        rho_common=sol.rho_bar,
    )


def bsqp_qp_solve(problem, parameters, lqr_mat, lqr_reg, previous_solution=None):
    """Same as `admm_qp_solve`, with the bsqp ADMM."""
    params = bsqp_parameters(parameters)
    bprob = bsqp_problem(lqr_mat, lqr_reg)
    n_h = problem.n_h if lqr_mat.C is not None else 0

    unconstrained = None
    if parameters.admm_warm_start_unconstrained or bprob.n_c == 0:
        unconstrained = _unconstrained(bprob, params)

    if bprob.n_c == 0:
        failed = ~torch.isfinite(unconstrained.dx).flatten(1).all(dim=1)
        mu = unconstrained.mu.clone()
        mu[:, 0] = 0.0
        nu = unconstrained.nu
        if nu is None:
            nu = unconstrained.du.new_zeros(*unconstrained.du.shape[:2], n_h)
        empty = lqr_mat.n
        solution = AdmmSolution(
            unconstrained.dx, unconstrained.du, mu, nu, empty, empty, empty, empty,
            torch.full_like(lqr_reg, parameters.admm_rho_init),
        )
        return solution, AdmmLog(iterations=1, failed=failed, factorization=None)

    sol = _warm_start(
        problem, parameters, lqr_mat, bprob, unconstrained, previous_solution
    )
    failed = _admm(bprob, params, sol, lqr_mat, lqr_reg)
    log = AdmmLog(
        iterations=int(sol.iters.max()),
        failed=failed,
        factorization=BsqpState(bprob, params, sol),
    )
    return _to_diffsqp(sol, n_h), log


def bsqp_qp_resume(problem, parameters, lqr_mat, lqr_reg, solution, log, active, max_iter):
    """Same as `admm_qp_resume`: more ADMM iterations for the `active`
    elements, warm-started from their current iterate (z, multipliers, rho)."""
    state = log.factorization
    params = dataclasses.replace(state.params, max_iter=max_iter)
    idx = active.nonzero().squeeze(1)
    sol = state.solution
    sub = _index(sol, idx)
    bsqp.admm(_index(state.problem, idx), params, sub)
    _scatter(sol, sub, idx)
    n_h = problem.n_h if lqr_mat.C is not None else 0
    log = AdmmLog(
        iterations=log.iterations + int(sub.iters.max()),
        failed=log.failed | _failed(sol),
        factorization=state,
    )
    return _to_diffsqp(sol, n_h), log


def bsqp_equality_correction(state: BsqpState, gaps, h):
    """(dx, du) of the ADMM-augmented LQR with zero cost gradients and the
    dynamics / underactuation offsets (gaps, h): the second-order correction
    of `sqp._second_order_correction`."""
    prob, rho, sigma = state.problem, state.solution.rho, state.params.sigma
    M, N = prob.M, prob.N
    Mt_rho = (M * rho.unsqueeze(-1)).mT
    Nt_rho = (N * rho[:, :-1].unsqueeze(-1)).mT
    eye_x = torch.eye(M.shape[-1], dtype=M.dtype, device=M.device)
    eye_u = torch.eye(N.shape[-1], dtype=N.dtype, device=N.device)
    lqr = bsqp.LqrProblem(
        A=prob.A,
        B=prob.B,
        b=gaps,
        Q=prob.Q + Mt_rho @ M + sigma * eye_x,
        q=torch.zeros_like(prob.q),
        R=prob.R + Nt_rho @ N + sigma * eye_u,
        r=torch.zeros_like(prob.r),
        S=prob.S + Nt_rho @ M[:, :-1],
        C=prob.C,
        D=prob.D,
        d=h,
    )
    correction = bsqp.riccati(lqr, state.params.lqr)
    return correction.dx, correction.du
