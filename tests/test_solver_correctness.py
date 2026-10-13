"""Correctness checks for the LQR, ADMM and SQP solvers.

    uv run python tests/test_solver_correctness.py
    uv run --with pytest pytest tests

All checks run in float64 on the CPU.
"""

import contextlib
import io
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(__file__))
import toy_problems  # noqa: E402

from diffsqp.solvers import admm_qp_solve, sqp_solve, SqpParameters  # noqa: E402
from diffsqp.solvers.lqr import lqr_factorize, lqr_solve_factorized  # noqa: E402
from diffsqp.types import QpParameters  # noqa: E402

DTYPE = torch.float64


def _random_lq(batch_size, horizon, n_x, n_u, n_h=0, n_g=0, seed=0):
    """Random well-posed LQ problem, optionally with box-type stage constraints."""
    g = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=g, dtype=DTYPE)

    def spd(*shape, n):
        X = rand(*shape, n, n)
        return X @ X.mT / n + 0.1 * torch.eye(n, dtype=DTYPE)

    B_, H = batch_size, horizon
    A = torch.eye(n_x, dtype=DTYPE) + 0.1 * rand(B_, H - 1, n_x, n_x)
    Bm = 0.1 * rand(B_, H - 1, n_x, n_u)
    Q = spd(B_, H, n=n_x)
    R = spd(B_, H - 1, n=n_u)
    S = 0.05 * rand(B_, H - 1, n_u, n_x)
    C = D = d = None
    if n_h > 0:
        C = rand(B_, H - 1, n_h, n_x)
        D = rand(B_, H - 1, n_h, n_u)
        d = 0.1 * rand(B_, H - 1, n_h)

    M = rand(B_, H, n_g, n_x)
    N = rand(B_, H - 1, n_g, n_u)
    n = 0.1 * rand(B_, H, n_g)
    # Feasible for the seeds used below, with 8 to 11 active rows at the optimum
    lb = -1.0 * torch.ones(H, n_g, dtype=DTYPE)
    ub = 0.5 * torch.ones(H, n_g, dtype=DTYPE)

    mat = QpParameters(
        Q=Q, q=rand(B_, H, n_x), R=R, r=rand(B_, H - 1, n_u), S=S,
        A=A, B=Bm, b=0.1 * rand(B_, H - 1, n_x), C=C, D=D, d=d,
        M=M, N=N, n=n, lb=lb, ub=ub,
    )
    problem = SimpleNamespace(
        batch_size=B_, horizon=H, n_x=n_x, n_u=n_u, n_h=n_h
    )
    return problem, mat


def _dense_qp(mat, b_idx):
    """Cost Hessian/gradient and equality constraints of one batch element.

    Variables are [dx_0, du_0, dx_1, du_1, ..., dx_{H-1}].
    """
    H, n_x = mat.q.shape[1], mat.q.shape[2]
    n_u = mat.r.shape[2]
    nz = H * n_x + (H - 1) * n_u

    def ix(k):
        return slice(k * (n_x + n_u), k * (n_x + n_u) + n_x)

    def iu(k):
        return slice(k * (n_x + n_u) + n_x, (k + 1) * (n_x + n_u))

    P = torch.zeros(nz, nz, dtype=DTYPE)
    c = torch.zeros(nz, dtype=DTYPE)
    for k in range(H):
        P[ix(k), ix(k)] = mat.Q[b_idx, k]
        c[ix(k)] = mat.q[b_idx, k]
    for k in range(H - 1):
        P[iu(k), iu(k)] = mat.R[b_idx, k]
        P[iu(k), ix(k)] = mat.S[b_idx, k]
        P[ix(k), iu(k)] = mat.S[b_idx, k].T
        c[iu(k)] = mat.r[b_idx, k]

    rows, rhs = [], []
    # dx_0 = 0
    E = torch.zeros(n_x, nz, dtype=DTYPE)
    E[:, ix(0)] = torch.eye(n_x, dtype=DTYPE)
    rows.append(E)
    rhs.append(torch.zeros(n_x, dtype=DTYPE))
    for k in range(H - 1):
        # dx_{k+1} - A dx_k - B du_k = b_k
        E = torch.zeros(n_x, nz, dtype=DTYPE)
        E[:, ix(k + 1)] = torch.eye(n_x, dtype=DTYPE)
        E[:, ix(k)] = -mat.A[b_idx, k]
        E[:, iu(k)] = -mat.B[b_idx, k]
        rows.append(E)
        rhs.append(mat.b[b_idx, k])
        if mat.C is not None:
            # C dx_k + D du_k = -d_k
            n_h = mat.C.shape[2]
            E = torch.zeros(n_h, nz, dtype=DTYPE)
            E[:, ix(k)] = mat.C[b_idx, k]
            E[:, iu(k)] = mat.D[b_idx, k]
            rows.append(E)
            rhs.append(-mat.d[b_idx, k])

    return P, c, torch.cat(rows), torch.cat(rhs), ix, iu


def _solve_dense_kkt(mat, b_idx):
    P, c, E, e, ix, iu = _dense_qp(mat, b_idx)
    nz, ne = P.shape[0], E.shape[0]
    KKT = torch.zeros(nz + ne, nz + ne, dtype=DTYPE)
    KKT[:nz, :nz] = P
    KKT[:nz, nz:] = E.T
    KKT[nz:, :nz] = E
    sol = torch.linalg.solve(KKT, torch.cat([-c, e]))[:nz]
    H = mat.q.shape[1]
    dx = torch.stack([sol[ix(k)] for k in range(H)])
    du = torch.stack([sol[iu(k)] for k in range(H - 1)])
    return dx, du


def test_lqr_matches_dense_kkt():
    for n_h in [0, 1]:
        for parallel_scan in [False, True]:
            problem, mat = _random_lq(3, 9, n_x=4, n_u=2, n_h=n_h, seed=n_h)
            reg = torch.zeros(3, dtype=DTYPE)
            fact = lqr_factorize(problem, mat, reg, parallel_scan=parallel_scan)
            sol = lqr_solve_factorized(fact, mat.q, mat.r)
            assert fact.success.all()
            for b in range(3):
                dx, du = _solve_dense_kkt(mat, b)
                # The saddle point system carries a -1e-8 dual regularization
                torch.testing.assert_close(sol.dx[b], dx, rtol=1e-6, atol=1e-7)
                torch.testing.assert_close(sol.du[b], du, rtol=1e-6, atol=1e-7)


def test_lqr_failure_is_contained():
    """A batch element whose factorization keeps failing takes no step, and
    the other elements are solved normally (instead of a RuntimeError)."""
    problem, mat = _random_lq(3, 6, n_x=3, n_u=2, seed=3)
    R = mat.R.clone()
    R[1] = -1e9 * torch.eye(2, dtype=DTYPE)  # indefinite beyond any regularization
    mat = mat._replace(R=R)
    # Only the failing element is regularized (and escalated, up to 1e5)
    reg = torch.tensor([0.0, 1e-5, 0.0], dtype=DTYPE)
    fact = lqr_factorize(problem, mat, reg)
    sol = lqr_solve_factorized(fact, mat.q, mat.r)

    assert fact.success.tolist() == [True, False, True]
    assert torch.all(sol.dx[1] == 0) and torch.all(sol.du[1] == 0)
    assert torch.isfinite(sol.dx).all()
    for b in [0, 2]:
        dx, du = _solve_dense_kkt(mat, b)
        torch.testing.assert_close(sol.dx[b], dx, rtol=1e-6, atol=1e-7)


def test_admm_matches_cvxpy():
    try:
        import cvxpy as cp
    except ImportError:
        print("  cvxpy not installed, skipping")
        return

    problem, mat = _random_lq(2, 8, n_x=3, n_u=2, n_g=3, seed=4)
    args = toy_problems.base_sqp_args(
        admm_max_iter=4000,
        admm_update_rho=True,
        admm_rho_update_iter_freq=25,
        admm_rho_init=1.0,
        admm_abs_tolerance=1e-10,
        admm_rel_tolerance=1e-10,
        admm_warm_start_unconstrained=True,
    )
    params = SqpParameters(**args)
    params.admm_rho_min = params.admm_rho_min.to(DTYPE)
    params.admm_rho_max = params.admm_rho_max.to(DTYPE)
    reg = torch.zeros(2, dtype=DTYPE)
    solution, log = admm_qp_solve(problem, params, mat, reg)

    for b in range(2):
        P, c, E, e, ix, iu = _dense_qp(mat, b)
        z = cp.Variable(P.shape[0])
        H = mat.q.shape[1]
        cons = [E.numpy() @ z == e.numpy()]
        for k in range(H):
            g = mat.M[b, k].numpy() @ z[ix(k)] + mat.n[b, k].numpy()
            if k < H - 1:
                g = g + mat.N[b, k].numpy() @ z[iu(k)]
            cons += [g >= mat.lb[k].numpy(), g <= mat.ub[k].numpy()]
        P_sym = 0.5 * (P + P.T)
        obj = 0.5 * cp.quad_form(z, cp.psd_wrap(P_sym.numpy())) + c.numpy() @ z
        qp = cp.Problem(cp.Minimize(obj), cons)
        qp.solve(solver=cp.CLARABEL)
        assert qp.status == "optimal", qp.status
        z_ref = torch.tensor(z.value, dtype=DTYPE)
        dx_ref = torch.stack([z_ref[ix(k)] for k in range(H)])
        du_ref = torch.stack([z_ref[iu(k)] for k in range(H - 1)])
        torch.testing.assert_close(solution.dx[b], dx_ref, rtol=0, atol=1e-5)
        torch.testing.assert_close(solution.du[b], du_ref, rtol=0, atol=1e-5)


def test_second_order_correction_reduces_residual():
    """Near a solution, the corrected step leaves a much smaller dynamics /
    underactuation residual than the plain full step (second order)."""
    from diffsqp.solvers.sqp import _second_order_correction, _step

    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(DTYPE)
    try:
        for name in ["cartpole_forward", "cartpole_inverse"]:
            problem, guess, args = toy_problems.PROBLEMS[name](batch_size=8, seed=2)
            args.update(sqp_max_iter=8, ls_function="merit_adaptive", admm_max_iter=200)
            params = SqpParameters(**args)
            with contextlib.redirect_stdout(io.StringIO()):
                guess, _ = sqp_solve(problem, params, guess)

            mat = problem.linearize(guess)
            direction, log = admm_qp_solve(
                problem, params, mat, torch.full((8,), 1e-5), None
            )
            alpha = torch.ones(8)
            trial = _step(guess, alpha, direction)
            corrected = _step(
                guess, alpha, _second_order_correction(problem, direction, log, trial)
            )

            def residual(s):
                gaps, h = problem.equality_residuals(s.x, s.u)
                r = gaps.abs().amax(dim=(1, 2))
                if h is not None:
                    r = torch.maximum(r, h.abs().amax(dim=(1, 2)))
                return r

            ratio = residual(trial) / residual(corrected)
            assert ratio.min() > 2.0, (name, ratio)
    finally:
        torch.set_default_dtype(default_dtype)


# Final per-element cost after 3 SQP iterations (batch 3, seed 1, 20 ADMM
# iterations), produced by the solver before the speed rework (commit e5103b2).
BASELINE_COSTS = {
    ("double_integrator", "filter"): [107.8782285581, 112.4315331139, 114.7908582517],
    ("double_integrator", "merit"): [154.2651355559, 159.6652430077, 163.5501948458],
    ("cartpole_forward", "filter"): [2169.7005921247, 2140.0780337553, 2128.0939154546],
    ("cartpole_forward", "merit"): [2216058.260706628, 2216047.6910032886, 2216042.8920226833],
    ("cartpole_inverse", "filter"): [342.7362343506, 344.5914872677, 346.8483243031],
    ("cartpole_inverse", "merit"): [293495.0448454912, 323141.756093174, 281908.1237251367],
}


def test_sqp_matches_baseline():
    """Same iterates as the original implementation (legacy line searches)."""
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(DTYPE)
    try:
        for (name, ls), expected in BASELINE_COSTS.items():
            for parallel_scan in [False, True]:
                # The references were computed with a 0.5 s double integrator
                kwargs = {"tf": 0.5} if name == "double_integrator" else {}
                problem, guess, args = toy_problems.PROBLEMS[name](
                    batch_size=3, seed=1, **kwargs
                )
                args.update(
                    sqp_max_iter=3,
                    sqp_cost_eps=-1.0,
                    admm_max_iter=20,
                    ls_function=ls,
                    lqr_parallel_scan=parallel_scan,
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    _, log = sqp_solve(problem, SqpParameters(**args), guess)
                torch.testing.assert_close(
                    torch.tensor(log.cost_hist[-1]),
                    torch.tensor(expected),
                    rtol=1e-8,
                    atol=0.0,
                    msg=f"{name} / {ls} / scan={parallel_scan}",
                )
    finally:
        torch.set_default_dtype(default_dtype)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, fn in tests:
        print(f"{name} ...", flush=True)
        fn()
        print("  ok")
