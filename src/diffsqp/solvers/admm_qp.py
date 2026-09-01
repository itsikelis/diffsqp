from copy import copy
import torch

from diffsqp.solvers import lqr_solve
from diffsqp.types import AdmmSolution, AdmmLog, QpParameters


def update_rho(problem, parameters, solution, residuals):
    horizon = problem.horizon
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

    # If no batch elements require an update, exit early
    if not update_mask.any().item():
        return False

    print("Updating rho")

    # Update rho_common only for the flagged batch items
    rho_common[update_mask] = rho_estimate[update_mask]
    rho_common_m = rho_common[update_mask].unsqueeze(1)

    for k in range(horizon):
        lb, ub = problem.g_bounds(k)
        n_g = lb.shape[-1]  # Get the number of constraints (e.g., 7)

        # Identify constraint types based on bounds
        is_unbounded = (lb <= -1e6) & (ub >= 1e6)
        is_eq = torch.abs(lb - ub) < 1e-6

        new_rho_k = rho_common_m.expand(-1, n_g).clone()
        new_rho_k = torch.where(is_eq, 1e3 * rho_common_m, new_rho_k)
        new_rho_k = torch.where(is_unbounded, parameters.admm_rho_min, new_rho_k)

        solution.rho[k][update_mask] = new_rho_k
        solution.rho_inv[k][update_mask] = 1.0 / new_rho_k

    print(solution.rho[5])
    return True


def check_termination(parameters, admm_iter, residuals, abs_tol, rel_tol):
    norm_prim, norm_prim_rel, norm_dual, norm_dual_rel = residuals

    # print(
    #     "primal: ",
    #     norm_prim.tolist(),
    #     "primal rel: ",
    #     norm_prim_rel.tolist(),
    #     "dual: ",
    #     norm_dual.tolist(),
    #     "dual rel: ",
    #     norm_dual_rel.tolist(),
    # )
    #
    # Maximum iterations reached
    if admm_iter == parameters.admm_max_iter - 1:
        return True

    tol_prim = abs_tol + rel_tol * norm_prim_rel
    tol_dual = abs_tol + rel_tol * norm_dual_rel

    # print(tol_prim, tol_dual)

    if torch.all(norm_prim <= tol_prim) and torch.all(norm_dual <= tol_dual):
        return True

    return False


def proximal_step_and_residuals(
    problem, parameters, solution, lqr_mat, lqr_solution, z_prev
):
    alpha = parameters.admm_alpha
    batch_size = problem.batch_size

    # Initial residual values
    norm_prim = -float("inf") * torch.ones((batch_size))
    norm_prim_rel = -float("inf") * torch.ones((batch_size))
    norm_dual = -float("inf") * torch.ones((batch_size))
    norm_dual_rel = -float("inf") * torch.ones((batch_size))

    horizon = problem.horizon
    for k in range(horizon):
        lb, ub = problem.g_bounds(k)
        Diag_rho = torch.diag_embed(solution.rho[k])
        Diag_rho_inv = torch.diag_embed(solution.rho_inv[k])

        M_k = lqr_mat.M[k]
        dx_hat_k = lqr_solution.dx[:, k]

        if k < horizon - 1:
            N_k = lqr_mat.N[k]
            du_hat_k = lqr_solution.du[:, k]

        # ------------------------------------- #
        # dx = alpha * dx_hat + (1-alpha) * dx  #
        # ------------------------------------- #
        solution.dx[:, k] *= 1.0 - alpha
        solution.dx[:, k] += alpha * dx_hat_k

        # ------------------------------------- #
        # du = alpha * du_hat + (1-alpha) * du  #
        # ------------------------------------- #
        if k < horizon - 1:
            solution.du[:, k] *= 1.0 - alpha
            solution.du[:, k] += alpha * du_hat_k

        # If no generic stage constraints -> continue
        if M_k is None:
            continue

        # --------------------------------------------------------- #
        # z_hat = alpha * (M * dx_hat + N * du_hat) + (1-alpha) * z #
        # --------------------------------------------------------- #
        z_hat = torch.einsum("...ij,...j->...i", M_k, dx_hat_k)
        if k < horizon - 1:
            z_hat += torch.einsum("...ij,...j->...i", N_k, du_hat_k)
        z_hat *= alpha
        z_hat += (1.0 - alpha) * (solution.z[k])

        # ---------------------------------------- #
        # z = clamp(z_hat + rho_inv * ksi, lb, ub) #
        # ---------------------------------------- #
        solution.z[k] = torch.clamp(
            z_hat + torch.einsum("...ii,...i->...i", Diag_rho_inv, solution.ksi[k]),
            lb,
            ub,
        )

        # ----------------------------- #
        # ksi = ksi + rho ∘ (z_hat - z) #
        # ----------------------------- #
        solution.ksi[k] = solution.ksi[k] + torch.einsum(
            "...ii,...i->...i", Diag_rho, z_hat - solution.z[k]
        )

        ### Residual Calculation ###
        dx_k = solution.dx[:, k]
        if k < horizon - 1:
            du_k = solution.du[:, k]
        z_k = solution.z[k]
        ksi_k = solution.ksi[k]

        # ---------------------------------- #
        # z_diff_scaled = rho * (z - z_prev) #
        # ---------------------------------- #
        z_diff = z_k - z_prev[k]
        z_diff_scaled_k = torch.einsum("...ii,...i->...i", Diag_rho, z_diff)

        # ---------------------------------------------------------------- #
        # r_dual = = max(r_dual, M^T * z_diff_scaled, N^T * z_diff_scaled) #
        # ---------------------------------------------------------------- #
        r_dual_x = torch.einsum("...ji,...j->...i", M_k, z_diff_scaled_k)
        norm_dual = torch.maximum(
            norm_dual,
            torch.norm(r_dual_x, p=float("inf"), dim=1),
        )
        if k < horizon - 1:
            r_dual_u = torch.einsum("...ji,...j->...i", N_k, z_diff_scaled_k)
            norm_dual = torch.maximum(
                norm_dual, torch.norm(r_dual_u, p=float("inf"), dim=1)
            )

        # ---------------------------- #
        # r_prim = M * dx + N * du - z #
        # ---------------------------- #
        MdxNdu = torch.einsum("...ij,...j->...i", M_k, dx_k)
        if k < horizon - 1:
            MdxNdu += torch.einsum("...ij,...j->...i", N_k, du_k)

        prim_res_k = MdxNdu - z_k

        norm_prim = torch.maximum(
            norm_prim, torch.norm(prim_res_k, p=float("inf"), dim=1)
        )

        # -------------------------------------------- #
        # r_prim_rel = max(r_prim, M * dx + N * du, z) #
        # -------------------------------------------- #
        norm_prim_rel = torch.maximum(
            norm_prim_rel,
            torch.maximum(
                torch.norm(MdxNdu, p=float("inf"), dim=1),
                torch.norm(z_k, p=float("inf"), dim=1),
            ),
        )

        # ---------------------------------------------- #
        # r_dual_rel = max(r_dual, M^T * ksi, N^T * ksi) #
        # ---------------------------------------------- #
        x_k_rel = torch.einsum("...ji,...j->...i", M_k, ksi_k)
        norm_dual_rel = torch.maximum(
            norm_dual_rel,
            torch.norm(x_k_rel, p=float("inf"), dim=1),
        )
        if k < horizon - 1:
            u_k_rel = torch.einsum("...ji,...j->...i", N_k, ksi_k)
            norm_dual_rel = torch.maximum(
                norm_dual_rel,
                torch.norm(u_k_rel, p=float("inf"), dim=1),
            )

        # IMPORTANT: Update z_prev
        z_prev[k].copy_(z_k)

    return norm_prim, norm_prim_rel, norm_dual, norm_dual_rel


def update_constrained_matrices(
    problem, parameters, constr_mat, solution, lqr_mat, rho_changed
):
    """
    Qc = Q + M^T * diag(rho) * M + sigma * I
    qc = q + M^T * ksi - M^T * diag(rho) * z - sigma * dx_prev
    Rc = R + N^T * diag(rho) * N + sigma * I
    rc = r + N^T * ksi - N^T * diag(rho) * z - sigma * du_prev
    Sc = S + N^T * diag(rho) * M
    """
    horizon = problem.horizon
    n_x = problem.n_x
    n_u = problem.n_u

    sigma = parameters.admm_sigma

    Qc = torch.zeros(lqr_mat.Q.shape)
    qc = torch.zeros(lqr_mat.q.shape)
    Rc = torch.zeros(lqr_mat.R.shape)
    rc = torch.zeros(lqr_mat.r.shape)
    Sc = torch.zeros(lqr_mat.S.shape)

    for k in range(horizon):
        Q_k = lqr_mat.Q[:, k]
        q_k = lqr_mat.q[:, k]
        M_k = lqr_mat.M[k]

        dx_prev_k = solution.dx[:, k]
        z_k = solution.z[k]
        ksi_k = solution.ksi[k]

        Diag_rho = torch.diag_embed(solution.rho[k])

        if rho_changed:
            Qc[:, k] = (
                Q_k
                + torch.einsum("...ki,...kk,...kj->...ij", M_k, Diag_rho, M_k)
                + sigma * torch.eye(n_x)
            )
        else:
            Qc[:, k] = constr_mat.Q[:, k]

        qc[:, k] = (
            q_k
            + torch.einsum("...ij,...i->...j", M_k, ksi_k)
            - torch.einsum("...ij,...ii,...i->...j", M_k, Diag_rho, z_k)
            - sigma * dx_prev_k
        )

        if k < horizon - 1:
            R_k = lqr_mat.R[:, k]
            r_k = lqr_mat.r[:, k]
            S_k = lqr_mat.S[:, k]
            N_k = lqr_mat.N[k]

            du_prev_k = solution.du[:, k]

            if rho_changed:
                Rc[:, k] = (
                    R_k
                    + torch.einsum("...ki,...kk,...kj->...ij", N_k, Diag_rho, N_k)
                    + sigma * torch.eye(n_u)
                )
                Sc[:, k] = S_k + torch.einsum(
                    "...ki,...kk,...kj->...ij", N_k, Diag_rho, M_k
                )
            else:
                Rc[:, k] = constr_mat.R[:, k]
                Sc[:, k] = constr_mat.S[:, k]

            rc[:, k] = (
                r_k
                + torch.einsum("...ij,...i->...j", N_k, ksi_k)
                - torch.einsum("...ij,...ii,...i->...j", N_k, Diag_rho, z_k)
                - sigma * du_prev_k
            )

    return QpParameters(
        Q=Qc,
        q=qc,
        R=Rc,
        r=rc,
        S=Sc,
        A=lqr_mat.A,
        B=lqr_mat.B,
        b=lqr_mat.b,
        C=lqr_mat.C,
        D=lqr_mat.D,
        d=lqr_mat.d,
        M=None,
        N=None,
        n=None,
    )


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


def get_unconstrained_solution(problem, lqr_mat):
    unconstr_solution = lqr_solve(problem, lqr_mat)
    return unconstr_solution.dx, unconstr_solution.du


def reset_rho(problem, parameters):
    horizon = problem.horizon
    batch_size = problem.batch_size
    rho = [None] * horizon
    rho_inv = [None] * horizon

    rho_common = torch.full((problem.batch_size,), parameters.admm_rho_init)

    for k in range(horizon):
        lb, ub = problem.g_bounds(k)
        n_g = lb.shape[-1]  # Get the number of constraints (e.g., 7)

        # Identify constraint types based on bounds
        is_unbounded = (lb <= -1e6) & (ub >= 1e6)
        is_eq = torch.abs(lb - ub) < 1e-7

        new_rho_k = torch.full((batch_size, n_g), parameters.admm_rho_init)
        new_rho_k = torch.where(is_eq, parameters.admm_rho_max, new_rho_k)
        new_rho_k = torch.where(is_unbounded, parameters.admm_rho_min, new_rho_k)

        rho[k] = new_rho_k
        rho_inv[k] = 1.0 / rho[k]

    return rho, rho_inv, rho_common


def new_solution(problem, parameters, lqr_mat, previous_solution):
    batch_size = problem.batch_size
    horizon = problem.horizon
    n_x = problem.n_x
    n_u = problem.n_u
    n_h = problem.n_h

    # Equality Lagrange multipliers and consensus variables are always reset
    mu = torch.zeros((batch_size, horizon, n_x))
    nu = torch.zeros((batch_size, horizon - 1, n_h))
    z = [torch.zeros((batch_size, problem.n_g(k))) for k in range(horizon)]

    # Determine dx, du
    if parameters.admm_warm_start_unconstrained:
        dx, du = get_unconstrained_solution(problem, lqr_mat)
    else:
        dx = torch.zeros((batch_size, horizon, n_x))
        du = torch.zeros((batch_size, horizon - 1, n_u))

    # Determine inequality Lagrange multipliers (ksi) and rho
    if previous_solution is None:
        ksi = [torch.zeros((batch_size, problem.n_g(k))) for k in range(horizon)]
        rho, rho_inv, rho_common = reset_rho(problem, parameters)
    else:
        if parameters.admm_reset_ksi:
            ksi = [torch.zeros((batch_size, problem.n_g(k))) for k in range(horizon)]
        else:
            ksi = previous_solution.ksi

        if parameters.admm_reset_rho:
            rho, rho_inv, rho_common = reset_rho(problem, parameters)
        else:
            rho = previous_solution.rho
            rho_inv = previous_solution.rho_inv
            rho_common = previous_solution.rho_common

    return AdmmSolution(dx, du, mu, nu, z, ksi, rho, rho_inv, rho_common)


def admm_qp_solve(problem, parameters, lqr_mat, previous_solution=None):
    # Initialize things
    constr_mat = None
    rho_changed = True
    z_prev = [
        torch.zeros((problem.batch_size, problem.n_g(k)))
        for k in range(problem.horizon)
    ]

    # Prepare solution struct
    solution = new_solution(problem, parameters, lqr_mat, previous_solution)

    # Initialize termination tolerances
    abs_tol, rel_tol, abs_tol_step, rel_tol_step = initialize_tolerances(parameters)

    # Main ADMM loop
    for admm_iter in range(parameters.admm_max_iter):
        # Create constrained problem matrices
        constr_mat = update_constrained_matrices(
            problem, parameters, constr_mat, solution, lqr_mat, rho_changed
        )
        rho_changed = False  # ALWAYS set to false after constrained matrix calculation

        # Solve LQR
        lqr_solution = lqr_solve(problem, constr_mat)

        # Proximal step and residual update
        residuals = proximal_step_and_residuals(
            problem, parameters, solution, lqr_mat, lqr_solution, z_prev
        )

        # Check for termination
        if check_termination(
            parameters,
            admm_iter,
            residuals,
            abs_tol,
            rel_tol,
        ):
            log = AdmmLog(
                iterations=admm_iter + 1,
            )
            return solution, log

        # Update rho here
        if (
            parameters.admm_update_rho
            and admm_iter % parameters.admm_rho_update_iter_freq == 0
        ):
            rho_changed = update_rho(problem, parameters, solution, residuals)

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
