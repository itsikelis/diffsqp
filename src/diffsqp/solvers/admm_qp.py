import torch
from copy import copy

from diffsqp.solvers import lqr_solve

from diffsqp.types import AdmmSolution


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


def check_admm_termination(parameters, admm_iter, r_prim):
    # Maximum iterations reached
    if admm_iter == parameters.admm_max_iter - 1:
        return True

    # print(r_prim <= parameters.admm_eps)
    # if torch.all(r_prim <= parameters.admm_eps):
    #     return True

    # O'Donoghue et. al. inspired termination
    # primal_residual =
    # dual_residual =

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


def admm_solve(problem, parameters, mat, previous_solution=None):
    batch_size = problem.n_batch
    horizon = problem.horizon
    n_x = problem.n_x
    n_u = problem.n_u
    n_h = problem.n_h

    ## Get ADMM corrections ##
    rho = copy(parameters.admm_rho)
    rho_inv = 1.0 / rho
    sigma = parameters.admm_sigma
    rho_changed = True

    if previous_solution is None:
        admm_solution = AdmmSolution(
            dx=torch.zeros((batch_size, horizon, n_x)),
            du=torch.zeros((batch_size, horizon - 1, n_u)),
            mu=torch.zeros((batch_size, horizon, n_x)),
            nu=torch.zeros((batch_size, horizon - 1, n_h)),
            z=[torch.zeros((batch_size, problem.n_g(k))) for k in range(horizon)],
            ksi=[torch.zeros((batch_size, problem.n_g(k))) for k in range(horizon)],
        )
    else:
        admm_solution = previous_solution

    for admm_iter in range(parameters.admm_max_iter):
        print("ADMM Iter: ", admm_iter)

        r_prim = -float("inf") * torch.ones((batch_size))
        r_dual = -float("inf") * torch.ones((batch_size))

        # TODO: Add rho_changed option
        for k in range(horizon - 1):

            n_g = mat.M[k].shape[-2]
            Diag_rho = rho * torch.eye(n_g)

            mat.Q[:, k], mat.R[:, k], mat.S[:, k] = get_constrained_qp_matrices(
                mat.Q[:, k],
                mat.R[:, k],
                mat.S[:, k],
                mat.M[k],
                mat.N[k],
                Diag_rho,
                sigma,
            )

            mat.q[:, k], mat.r[:, k] = get_constrained_qp_vectors(
                mat.q[:, k],
                mat.r[:, k],
                mat.M[k],
                mat.N[k],
                admm_solution.z[k],
                admm_solution.ksi[k],
                Diag_rho,
                sigma,
                admm_solution.dx[:, k],
                admm_solution.du[:, k],
            )

        # Solve LQR to get dx_hat, du_hat
        lqr_solution = lqr_solve(
            problem,
            mat.Q,
            mat.q,
            mat.R,
            mat.r,
            mat.S,
            mat.A,
            mat.B,
            mat.b,
            mat.C,
            mat.D,
            mat.d,
        )

        # ADMM step
        for k in range(horizon):
            alpha = parameters.admm_alpha
            lb, ub = problem.g_bounds(k)

            dx_hat = lqr_solution.dx[:, k]
            M_k = mat.M[k]
            Q_k = mat.Q[:, k]
            q_k = mat.q[:, k]

            if k < horizon - 1:
                du_hat = lqr_solution.du[:, k]
                N_k = mat.N[k]
                R_k = mat.R[:, k]
                r_k = mat.r[:, k]
                S_k = mat.S[:, k]

            # ------------------------------------- #
            # dx = alpha * dx_hat + (1-alpha) * dx  #
            # ------------------------------------- #
            admm_solution.dx[:, k] *= 1.0 - alpha
            admm_solution.dx[:, k] += alpha * dx_hat

            # ------------------------------------- #
            # du = alpha * du_hat + (1-alpha) * du  #
            # ------------------------------------- #
            if k < horizon - 1:
                admm_solution.du[:, k] *= 1.0 - alpha
                admm_solution.du[:, k] += alpha * du_hat

            # If no generic stage constraints -> continue
            if M_k is None:
                continue

            # --------------------------------------------------------- #
            # z_hat = alpha * (M * dx_hat + N * du_hat) + (1-alpha) * z #
            # --------------------------------------------------------- #
            z_hat = torch.einsum("...ij,...j->...i", M_k, dx_hat)
            if k < horizon - 1:
                z_hat += torch.einsum("...ij,...j->...i", N_k, du_hat)
            z_hat *= alpha
            z_hat += (1.0 - alpha) * (admm_solution.z[k])

            # ---------------------------------------- #
            # z = clamp(z_hat + rho_inv * ksi, lb, ub) #
            # ---------------------------------------- #
            admm_solution.z[k] = torch.clamp(
                z_hat + rho_inv * admm_solution.ksi[k], lb, ub
            )

            # ----------------------------- #
            # ksi = ksi + rho ∘ (z_hat - z) #
            # ----------------------------- #
            admm_solution.ksi[k] = admm_solution.ksi[k] + rho * (
                z_hat - admm_solution.z[k]
            )

            # Calculate residuals
            dx_k = admm_solution.dx[:, k]
            du_k = admm_solution.du[:, k]
            ksi_k = admm_solution.ksi[k]

            # -----------------------------  #
            # r_prim = |M * dx + N * du - z| #
            # -----------------------------  #
            r_prim_k = torch.einsum("...ij,...j->...i", M_k, dx_k)
            if k < horizon - 1:
                r_prim_k += torch.einsum("...ij,...j->...i", N_k, du_k)
            r_prim_k -= admm_solution.z[k]
            r_prim_k = torch.norm(r_prim_k, p=float("inf"), dim=1)

            # Store largest residual overall
            r_prim = torch.maximum(r_prim, r_prim_k)

            # ------------------------------------------------------ #
            # r_dual_x = |rho * (Q * dx + S^T * du + q - M^T * ksi)| #
            # r_dual_u = |rho * (R * du + S * dx + r - N^T * ksi)|   #
            # ------------------------------------------------------ #
            r_dual_x_k = torch.einsum("...ij,...j->...i", Q_k, dx_k)
            if k < horizon - 1:
                r_dual_x_k += torch.einsum("...ij,...i->...j", S_k, du_k)
            r_dual_x_k += q_k
            r_dual_x_k -= torch.einsum("...ij,...i->...j", M_k, ksi_k)

            # print(torch.norm(r_dual_x_k, p=float("inf"), dim=1))

            if k < horizon - 1:
                r_dual_u_k = torch.einsum("...ij,...j->...i", R_k, du_k)
                r_dual_u_k += torch.einsum("...ij,...j->...i", S_k, dx_k)
                r_dual_u_k += r_k
                r_dual_u_k -= torch.einsum("...ij,...i->...j", N_k, -ksi_k)

        # Check ADMM termination
        # print(r_prim)
        if check_admm_termination(parameters, admm_iter, r_prim):
            return admm_solution
