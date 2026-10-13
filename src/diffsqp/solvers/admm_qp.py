from copy import copy
import torch
import torch.nn.functional as F

from diffsqp.solvers.lqr import lqr_factorize, lqr_solve_factorized
from diffsqp.types import AdmmSolution, AdmmLog
from diffsqp.utils.math import bmv, bmtv

# Constraint rows with |lb - ub| below this are treated as equalities
EQ_TOL = 1e-6
# Constraint rows with both bounds beyond this are treated as unbounded
UNBOUNDED = 1e6


def constraint_map(mat, dx, du):
    """M dx + N du for all stages: (B, H, G). The terminal stage has no N du."""
    return bmv(mat.M, dx) + F.pad(bmv(mat.N, du), (0, 0, 0, 1))


def transposed_inf_norm(mat, v):
    """max(||M^T v||_inf, ||N^T v||_inf) over all stages: (B,)."""
    Mt_v = bmtv(mat.M, v)
    Nt_v = bmtv(mat.N, v[:, :-1])
    return torch.maximum(
        torch.abs(Mt_v).amax(dim=(1, 2)), torch.abs(Nt_v).amax(dim=(1, 2))
    )


def rho_rows(parameters, rho_common, lb, ub, eq_rho):
    """Per-row penalties (B, H, G) from the per-env scalar rho_common (B,)."""
    is_unbounded = (lb <= -UNBOUNDED) & (ub >= UNBOUNDED)
    is_eq = torch.abs(lb - ub) < EQ_TOL

    rho = rho_common.view(-1, 1, 1).expand(-1, *lb.shape)
    rho = torch.where(is_eq, eq_rho, rho)
    rho = torch.where(is_unbounded, parameters.admm_rho_min, rho)
    return rho


def update_rho(parameters, solution, residuals, dones, lb, ub):
    norm_prim, norm_prim_rel, norm_dual, norm_dual_rel = residuals
    rho_common = solution.rho_common

    scale = torch.sqrt((norm_prim * norm_dual_rel) / (norm_dual * norm_prim_rel + 1e-8))
    rho_estimate = torch.clamp(
        scale * rho_common,
        min=parameters.admm_rho_min,
        max=parameters.admm_rho_max,
    )

    # Evaluate which batch elements breach the tolerance threshold
    tol = parameters.admm_adaptive_rho_tolerance
    update_mask = (rho_estimate > rho_common * tol) | (rho_estimate < rho_common / tol)

    # Mask out environments that are already done
    update_mask = update_mask & ~dones

    # If no batch elements require an update, exit early
    if not update_mask.any().item():
        return False

    if parameters.verbose:
        print("Updating rho")

    # Update rho only for the flagged batch items
    rho_common = torch.where(update_mask, rho_estimate, rho_common)
    eq_rho = (1e3 * rho_common).view(-1, 1, 1)
    new_rho = rho_rows(parameters, rho_common, lb, ub, eq_rho)

    mask = update_mask.view(-1, 1, 1)
    solution.rho = torch.where(mask, new_rho, solution.rho)
    solution.rho_inv = torch.where(mask, 1.0 / new_rho, solution.rho_inv)
    solution.rho_common = rho_common
    return True


def check_termination(max_iter, admm_iter, residuals, abs_tol, rel_tol):
    norm_prim, norm_prim_rel, norm_dual, norm_dual_rel = residuals

    # Maximum iterations reached
    if admm_iter == max_iter - 1:
        return True

    tol_prim = abs_tol + rel_tol * norm_prim_rel
    tol_dual = abs_tol + rel_tol * norm_dual_rel

    return (norm_prim <= tol_prim) & (norm_dual <= tol_dual)


def proximal_step_and_residuals(
    parameters, solution, mat, lqr_solution, lb_n, ub_n, z_prev, dones
):
    """ADMM relaxation, projection and dual update for all stages at once.

    Batch elements in `dones` keep their values; their residuals are -inf.
    Returns the residuals and the updated z_prev.
    """
    alpha = parameters.admm_alpha
    keep = dones.view(-1, 1, 1)
    dx_hat, du_hat = lqr_solution.dx, lqr_solution.du

    # dx = alpha * dx_hat + (1-alpha) * dx
    new_dx = alpha * dx_hat + (1.0 - alpha) * solution.dx
    solution.dx = torch.where(keep, solution.dx, new_dx)
    new_du = alpha * du_hat + (1.0 - alpha) * solution.du
    solution.du = torch.where(keep, solution.du, new_du)

    # z_hat = alpha * (M * dx_hat + N * du_hat) + (1-alpha) * z
    z_hat = alpha * constraint_map(mat, dx_hat, du_hat) + (1.0 - alpha) * solution.z

    # z = clamp(z_hat + rho_inv * ksi, lb - n, ub - n)
    new_z = torch.clamp(z_hat + solution.rho_inv * solution.ksi, lb_n, ub_n)
    z = torch.where(keep, solution.z, new_z)

    # ksi = ksi + rho * (z_hat - z)
    new_ksi = solution.ksi + solution.rho * (z_hat - z)
    ksi = torch.where(keep, solution.ksi, new_ksi)
    solution.z, solution.ksi = z, ksi

    ### Residual Calculation ###
    # r_dual = max(M^T * rho * (z - z_prev), N^T * rho * (z - z_prev))
    norm_dual = transposed_inf_norm(mat, solution.rho * (z - z_prev))

    # r_prim = M * dx + N * du - z
    MdxNdu = constraint_map(mat, solution.dx, solution.du)
    norm_prim = torch.abs(MdxNdu - z).amax(dim=(1, 2))

    # r_prim_rel = max(M * dx + N * du, z)
    norm_prim_rel = torch.maximum(
        torch.abs(MdxNdu).amax(dim=(1, 2)), torch.abs(z).amax(dim=(1, 2))
    )

    # r_dual_rel = max(M^T * ksi, N^T * ksi)
    norm_dual_rel = transposed_inf_norm(mat, ksi)

    residuals = tuple(
        torch.where(dones, -float("inf"), res)
        for res in (norm_prim, norm_prim_rel, norm_dual, norm_dual_rel)
    )

    # IMPORTANT: Update z_prev only for unfinished environments
    z_prev = torch.where(keep, z_prev, z)
    return residuals, z_prev


def augmented_hessians(parameters, lqr_mat, rho):
    """
    Qc = Q + M^T * diag(rho) * M + sigma * I
    Rc = R + N^T * diag(rho) * N + sigma * I
    Sc = S + N^T * diag(rho) * M
    """
    sigma = parameters.admm_sigma
    M, N = lqr_mat.M, lqr_mat.N
    n_x, n_u = M.shape[-1], N.shape[-1]

    # Contiguous transposes: a transposed view makes batched cuBLAS ~10x slower
    Mt_rho = (M * rho.unsqueeze(-1)).mT.contiguous()
    Nt_rho = (N * rho[:, :-1].unsqueeze(-1)).mT.contiguous()
    Qc = lqr_mat.Q + Mt_rho @ M + sigma * torch.eye(n_x)
    Rc = lqr_mat.R + Nt_rho @ N + sigma * torch.eye(n_u)
    Sc = lqr_mat.S + Nt_rho @ M[:, :-1]
    return Qc, Rc, Sc


def augmented_gradients(parameters, lqr_mat, solution):
    """
    qc = q + M^T * ksi - M^T * diag(rho) * z - sigma * dx_prev
    rc = r + N^T * ksi - N^T * diag(rho) * z - sigma * du_prev
    """
    sigma = parameters.admm_sigma
    w = solution.ksi - solution.rho * solution.z
    qc = lqr_mat.q + bmtv(lqr_mat.M, w) - sigma * solution.dx
    rc = lqr_mat.r + bmtv(lqr_mat.N, w[:, :-1]) - sigma * solution.du
    return qc, rc


def initialize_tolerances(parameters):
    abs_tol = copy(parameters.admm_abs_tolerance)
    rel_tol = copy(parameters.admm_rel_tolerance)

    update_steps = (
        parameters.admm_tolerance_update_steps
        if parameters.admm_tolerance_update_steps > 0
        else parameters.admm_max_iter
    )

    abs_tol_step = 0.0
    if parameters.admm_abs_tolerance_final > 0.0:
        abs_tol_step = (
            parameters.admm_abs_tolerance_final - parameters.admm_abs_tolerance
        ) / update_steps

    rel_tol_step = 0.0
    if parameters.admm_rel_tolerance_final > 0.0:
        rel_tol_step = (
            parameters.admm_rel_tolerance_final - parameters.admm_rel_tolerance
        ) / update_steps

    return abs_tol, rel_tol, abs_tol_step, rel_tol_step


def reset_rho(problem, parameters, lb, ub):
    rho_common = torch.full((problem.batch_size,), parameters.admm_rho_init)
    rho = rho_rows(parameters, rho_common, lb, ub, parameters.admm_rho_max)
    return rho, 1.0 / rho, rho_common


def new_solution(problem, parameters, lqr_mat, unconstrained, previous_solution):
    batch_size = problem.batch_size
    horizon = problem.horizon
    n_x = problem.n_x
    n_u = problem.n_u
    n_h = problem.n_h
    lb, ub = lqr_mat.lb, lqr_mat.ub
    G = lb.shape[-1]

    # Equality Lagrange multipliers and consensus variables are always reset
    mu = torch.zeros((batch_size, horizon, n_x))
    nu = torch.zeros((batch_size, horizon - 1, n_h))
    z = torch.zeros((batch_size, horizon, G))

    # Determine dx, du
    if parameters.admm_warm_start_unconstrained:
        dx, du = unconstrained.dx, unconstrained.du
    else:
        dx = torch.zeros((batch_size, horizon, n_x))
        du = torch.zeros((batch_size, horizon - 1, n_u))

    if parameters.admm_warm_start_z:
        # Consensus variables consistent with the warm start (dx, du)
        z = torch.clamp(
            constraint_map(lqr_mat, dx, du), lqr_mat.lb - lqr_mat.n, lqr_mat.ub - lqr_mat.n
        )

    # Determine inequality Lagrange multipliers (ksi) and rho
    if previous_solution is None:
        ksi = torch.zeros((batch_size, horizon, G))
        rho, rho_inv, rho_common = reset_rho(problem, parameters, lb, ub)
    else:
        if parameters.admm_reset_ksi:
            ksi = torch.zeros((batch_size, horizon, G))
        else:
            ksi = previous_solution.ksi

        if parameters.admm_reset_rho:
            rho, rho_inv, rho_common = reset_rho(problem, parameters, lb, ub)
        else:
            rho = previous_solution.rho
            rho_inv = previous_solution.rho_inv
            rho_common = previous_solution.rho_common

    return AdmmSolution(dx, du, mu, nu, z, ksi, rho, rho_inv, rho_common)


def admm_qp_solve(problem, parameters, lqr_mat, lqr_reg, previous_solution=None):
    """Solve the SQP subproblem QP with ADMM, all stages at once.

    The LQR factorization only depends on rho, so it is computed once and
    reused until rho changes; each iteration then costs two linear recursions.
    """
    G = lqr_mat.lb.shape[-1]

    # Unconstrained solution: warm start, or the answer without constraints
    unconstrained = None
    if parameters.admm_warm_start_unconstrained or G == 0:
        fact = lqr_factorize(
            problem, lqr_mat, lqr_reg, parallel_scan=parameters.lqr_parallel_scan
        )
        unconstrained = lqr_solve_factorized(fact, lqr_mat.q, lqr_mat.r)

    # Prepare solution struct
    solution = new_solution(
        problem, parameters, lqr_mat, unconstrained, previous_solution
    )

    if G == 0:
        solution.dx, solution.du = unconstrained.dx, unconstrained.du
        solution.mu, solution.nu = unconstrained.mu, unconstrained.nu
        return solution, AdmmLog(iterations=1, failed=~fact.success, factorization=fact)

    # Boolean mask to track terminated envs
    dones = torch.zeros(problem.batch_size, dtype=torch.bool)
    z_prev = solution.z.clone()
    return _admm_iterations(
        problem,
        parameters,
        lqr_mat,
        lqr_reg,
        solution,
        dones,
        z_prev,
        None,
        parameters.admm_max_iter,
    )


def admm_qp_resume(problem, parameters, lqr_mat, lqr_reg, solution, log, active, max_iter):
    """Continue ADMM from `solution` for the `active` batch elements only.

    Used when the step of an element is not accurate enough for the SQP
    (it does not reduce the linearized constraint violation).
    """
    return _admm_iterations(
        problem,
        parameters,
        lqr_mat,
        lqr_reg,
        solution,
        ~active,
        solution.z.clone(),
        log.factorization,
        max_iter,
        log.iterations,
    )


def _admm_iterations(
    problem,
    parameters,
    lqr_mat,
    lqr_reg,
    solution,
    dones,
    z_prev,
    fact,
    max_iter,
    iterations_done=0,
):
    lb_n = lqr_mat.lb - lqr_mat.n
    ub_n = lqr_mat.ub - lqr_mat.n
    rho_changed = fact is None

    # Initialize termination tolerances
    abs_tol, rel_tol, abs_tol_step, rel_tol_step = initialize_tolerances(parameters)

    # Main ADMM loop
    for admm_iter in range(max_iter):
        # Refactorize only when the quadratic terms change
        if rho_changed:
            Qc, Rc, Sc = augmented_hessians(parameters, lqr_mat, solution.rho)
            fact = lqr_factorize(
                problem,
                lqr_mat,
                lqr_reg,
                Q=Qc,
                R=Rc,
                S=Sc,
                parallel_scan=parameters.lqr_parallel_scan,
            )
            rho_changed = False

        # Solve LQR
        qc, rc = augmented_gradients(parameters, lqr_mat, solution)
        lqr_solution = lqr_solve_factorized(fact, qc, rc)

        # Multipliers of the dynamics and underactuation constraints
        keep = dones.view(-1, 1, 1)
        solution.mu = torch.where(keep, solution.mu, lqr_solution.mu)
        solution.nu = torch.where(keep, solution.nu, lqr_solution.nu)

        # Proximal step and residual update
        residuals, z_prev = proximal_step_and_residuals(
            parameters, solution, lqr_mat, lqr_solution, lb_n, ub_n, z_prev, dones
        )

        # Check for termination
        new_dones = check_termination(
            max_iter,
            admm_iter,
            residuals,
            abs_tol,
            rel_tol,
        )

        dones = dones | new_dones

        if torch.all(dones):
            break

        # Update rho here
        if (
            parameters.admm_update_rho
            and admm_iter % parameters.admm_rho_update_iter_freq == 0
        ):
            rho_changed = update_rho(
                parameters, solution, residuals, dones, lqr_mat.lb, lqr_mat.ub
            )

        # Tighten absolute tolerance
        if (
            parameters.admm_abs_tolerance_final > 0.0
            and abs_tol > parameters.admm_abs_tolerance_final
        ):
            abs_tol += abs_tol_step

        # Tighten relative tolerance
        if (
            parameters.admm_rel_tolerance_final > 0.0
            and rel_tol > parameters.admm_rel_tolerance_final
        ):
            rel_tol += rel_tol_step

    log = AdmmLog(
        iterations=iterations_done + admm_iter + 1,
        failed=~fact.success,
        factorization=fact,
    )
    return solution, log
