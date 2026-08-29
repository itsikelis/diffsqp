import torch
from copy import copy

from diffsqp.solvers import lqr_solve

from diffsqp.types import AdmmSolution, AdmmLog


def get_constrained_qp_matrices(Q_k, R_k, S_k, M_k, N_k, Diag_rho, sigma):
    n_x = Q_k.shape[-1]
    n_u = R_k.shape[-1]

    # Q = Q +M^T * diag(rho) * M + sigma * I
    Q_k_ = (
        Q_k
        + torch.einsum("...ki,kk,...kj->...ij", M_k, Diag_rho, M_k)
        + sigma * torch.eye(n_x)
    )

    # R = R + N^T * diag(rho) * N + sigma * I
    R_k_ = (
        R_k
        + torch.einsum("...ki,kk,...kj->...ij", N_k, Diag_rho, N_k)
        + sigma * torch.eye(n_u)
    )

    # S = S + N^T * diag(rho) * M
    S_k_ = S_k + torch.einsum("...ki,kk,...kj->...ij", N_k, Diag_rho, M_k)

    # print("diag(rho): ", Diag_rho)
    # print("Before R_k: ", R_k)
    # print("After R_k: ", R_k_)
    #
    # print("Before S_k: ", S_k)
    # print("After S_k: ", S_k_)
    # print("---------------------")

    return Q_k_, R_k_, S_k_


def check_admm_termination(
    parameters,
    admm_iter,
    norm_prim,
    norm_prim_rel,
    norm_dual,
    norm_dual_rel,
    current_abs_tol,
    current_rel_tol,
):
    # Maximum iterations reached
    if admm_iter == parameters.admm_max_iter - 1:
        return True

    print(
        "primal: ",
        norm_prim.item(),
        "primal rel: ",
        norm_prim_rel.item(),
        "dual: ",
        norm_dual.item(),
        "dual rel: ",
        norm_dual_rel.item(),
    )

    tol_prim = current_abs_tol + current_rel_tol * norm_prim_rel
    tol_dual = current_abs_tol + current_rel_tol * norm_dual_rel

    if torch.all(norm_prim <= tol_prim) and torch.all(norm_dual <= tol_dual):
        return True

    return False


def get_constrained_qp_vectors(
    q_k, r_k, M_k, N_k, z_k, ksi_k, Diag_rho, sigma, dx_prev_k, du_prev_k
):
    # q_k = q_k + M^T * ksi - M^T * diag(rho) * z - sigma * dx_prev
    q_k_ = (
        q_k
        + torch.einsum("...ij,...i->...j", M_k, ksi_k)
        - torch.einsum("...ij,ii,...i->...j", M_k, Diag_rho, z_k)
        - sigma * dx_prev_k
    )

    # r_k = r_k + N^T * y - N^T * diag(rho) * z - sigma * du_prev
    r_k_ = (
        r_k
        + torch.einsum("...ij,...i->...j", N_k, ksi_k)
        - torch.einsum("...ij,ii,...i->...j", N_k, Diag_rho, z_k)
        - sigma * du_prev_k
    )

    # print("Before q_k: ", q_k)
    # print("After q_k: ", q_k_)
    #
    # print("Before r_k: ", r_k)
    # print("After r_k: ", r_k_)
    # print("---------------------")

    return q_k_, r_k_


def admm_qp_solve(problem, parameters, mat, previous_solution=None):
    batch_size = problem.batch_size
    horizon = problem.horizon
    n_x = problem.n_x
    n_u = problem.n_u
    n_h = problem.n_h

    ## Get ADMM corrections ##
    rho_ineq = copy(parameters.admm_rho_ineq)
    rho_eq = copy(parameters.admm_rho_eq)
    rho = [None] * problem.horizon
    rho_inv = [None] * problem.horizon
    rho_changed = True

    z_prev = [torch.zeros((batch_size, problem.n_g(k))) for k in range(horizon)]

    admm_solution = AdmmSolution(
        dx=torch.zeros((batch_size, horizon, n_x)),
        du=torch.zeros((batch_size, horizon - 1, n_u)),
        mu=torch.zeros((batch_size, horizon, n_x)),
        nu=torch.zeros((batch_size, horizon - 1, n_h)),
        z=[torch.zeros((batch_size, problem.n_g(k))) for k in range(horizon)],
        ksi=[torch.zeros((batch_size, problem.n_g(k))) for k in range(horizon)],
    )
    if previous_solution is not None:
        admm_solution.z[:] = previous_solution.z[:]
        admm_solution.ksi[:] = previous_solution.ksi[:]

    # Calculate rho
    for k in range(horizon):
        rho_vec = []
        for c in problem.constraints[k]:
            if c.is_equality:
                rho_vec += [rho_eq for _ in range(c.n_g)]
            else:
                rho_vec += [rho_ineq for _ in range(c.n_g)]

        rho[k] = torch.tensor(rho_vec)
        rho_inv[k] = 1.0 / rho[k]

    # --- Tolerance Tightening Setup ---
    current_abs_tol = parameters.admm_abs_tolerance
    current_rel_tol = parameters.admm_rel_tolerance

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
    # ----------------------------------

    # Cache matrices
    orig_Q = mat.Q.clone()
    orig_R = mat.R.clone()
    orig_S = mat.S.clone()
    orig_q = mat.q.clone()
    orig_r = mat.r.clone()

    for admm_iter in range(parameters.admm_max_iter):
        # Max residual values
        norm_prim = -float("inf") * torch.ones((batch_size))
        norm_prim_rel = -float("inf") * torch.ones((batch_size))
        norm_dual = -float("inf") * torch.ones((batch_size))
        norm_dual_rel = -float("inf") * torch.ones((batch_size))

        for k in range(horizon):
            Q_k = orig_Q[:, k]
            q_k = orig_q[:, k]
            M_k = mat.M[k]
            z_k = admm_solution.z[k]
            ksi_k = admm_solution.ksi[k]
            dx_prev_k = admm_solution.dx[:, k]
            Diag_rho = torch.diag(rho[k])
            sigma = parameters.admm_sigma
            n_x = Q_k.shape[-1]

            # Q = Q +M^T * diag(rho) * M + sigma * I
            mat.Q[:, k] = (
                Q_k
                + torch.einsum("...ki,kk,...kj->...ij", M_k, Diag_rho, M_k)
                + sigma * torch.eye(n_x)
            )

            mat.q[:, k] = (
                q_k
                + torch.einsum("...ij,...i->...j", M_k, ksi_k)
                - torch.einsum("...ij,ii,...i->...j", M_k, Diag_rho, z_k)
                - sigma * dx_prev_k
            )

            if k < horizon - 1:
                R_k = orig_R[:, k]
                r_k = orig_r[:, k]
                S_k = orig_S[:, k]
                N_k = mat.N[k]
                du_prev_k = admm_solution.du[:, k]
                n_u = R_k.shape[-1]
                # R = R + N^T * diag(rho) * N + sigma * I
                mat.R[:, k] = (
                    R_k
                    + torch.einsum("...ki,kk,...kj->...ij", N_k, Diag_rho, N_k)
                    + sigma * torch.eye(n_u)
                )
                # S = S + N^T * diag(rho) * M
                mat.S[:, k] = S_k + torch.einsum(
                    "...ki,kk,...kj->...ij", N_k, Diag_rho, M_k
                )

                # r_k = r_k + N^T * y - N^T * diag(rho) * z - sigma * du_prev
                mat.r[:, k] = (
                    r_k
                    + torch.einsum("...ij,...i->...j", N_k, ksi_k)
                    - torch.einsum("...ij,ii,...i->...j", N_k, Diag_rho, z_k)
                    - sigma * du_prev_k
                )

        # Solve LQR to get dx_hat, du_hat
        lqr_solution = lqr_solve(problem, mat)

        # ADMM step
        for k in range(horizon):
            Diag_rho = torch.diag(rho[k])
            alpha = parameters.admm_alpha
            lb, ub = problem.g_bounds(k)

            dx_hat_k = lqr_solution.dx[:, k]
            M_k = mat.M[k]
            Q_k = mat.Q[:, k]
            q_k = mat.q[:, k]

            if k < horizon - 1:
                du_hat_k = lqr_solution.du[:, k]
                N_k = mat.N[k]
                R_k = mat.R[:, k]
                r_k = mat.r[:, k]
                S_k = mat.S[:, k]

            # ------------------------------------- #
            # dx = alpha * dx_hat + (1-alpha) * dx  #
            # ------------------------------------- #
            admm_solution.dx[:, k] *= 1.0 - alpha
            admm_solution.dx[:, k] += alpha * dx_hat_k

            # ------------------------------------- #
            # du = alpha * du_hat + (1-alpha) * du  #
            # ------------------------------------- #
            if k < horizon - 1:
                admm_solution.du[:, k] *= 1.0 - alpha
                admm_solution.du[:, k] += alpha * du_hat_k

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
            z_hat += (1.0 - alpha) * (admm_solution.z[k])

            # ---------------------------------------- #
            # z = clamp(z_hat + rho_inv * ksi, lb, ub) #
            # ---------------------------------------- #
            admm_solution.z[k] = torch.clamp(
                z_hat + rho_inv[k] * admm_solution.ksi[k], lb, ub
            )

            # ----------------------------- #
            # ksi = ksi + rho ∘ (z_hat - z) #
            # ----------------------------- #
            admm_solution.ksi[k] = admm_solution.ksi[k] + rho[k] * (
                z_hat - admm_solution.z[k]
            )

            ### Residual Calculation ###

            dx_k = admm_solution.dx[:, k]
            if k < horizon - 1:
                du_k = admm_solution.du[:, k]
            z_k = admm_solution.z[k]
            ksi_k = admm_solution.ksi[k]

            # ---------------------------------- #
            # z_diff_scaled = rho * (z - z_prev) #
            # ---------------------------------- #
            z_diff = z_k - z_prev[k]
            z_diff_scaled_k = torch.einsum("...ij,...j->...i", Diag_rho, z_diff)

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

            z_prev[k] = z_k.detach().clone()

        # Check ADMM termination
        if check_admm_termination(
            parameters,
            admm_iter,
            norm_prim,
            norm_prim_rel,
            norm_dual,
            norm_dual_rel,
            current_abs_tol,
            current_rel_tol,
        ):
            log = AdmmLog(
                rho=rho[k],
                iterations=admm_iter + 1,
            )
            return admm_solution, log

        # Tighten absolute tolerance
        if (
            parameters.admm_abs_tolerance_final > 0.0
            and current_abs_tol > parameters.admm_abs_tolerance_final
        ):
            current_abs_tol += abs_tol_step

        # Tighten relative tolerance
        if (
            parameters.admm_rel_tolerance_final > 0.0
            and current_rel_tol > parameters.rel_tolerance_final
        ):
            current_rel_tol += rel_tol_step
