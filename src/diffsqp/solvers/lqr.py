import torch

from diffsqp.problems import Problem
from diffsqp.utils.math import mm, mv, tran
from diffsqp.types import QpParameters, AdmmSolution


def lqr_solve(problem: Problem, mat, reg):
    K, k, P, p = lqr_backward_pass_(
        problem,
        reg,
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
    result = lqr_forward_pass_(
        problem,
        K,
        k,
        P,
        p,
        mat.A,
        mat.B,
        mat.b,
    )

    return result


def lqr_backward_pass_(problem: Problem, reg, Q, q, R, r, S, A, B, b, C, D, d):
    # Wrap the backward pass in a retry loop
    max_retries = 10
    for attempt in range(max_retries):
        batch_size = problem.batch_size
        horizon = problem.horizon
        n_x = problem.n_x
        n_u = problem.n_u
        n_h = problem.n_h

        K = torch.zeros((batch_size, horizon - 1, n_u + n_h, n_x))
        k = torch.zeros((batch_size, horizon - 1, n_u + n_h))

        P = torch.zeros((batch_size, horizon, n_x, n_x))
        p = torch.zeros((batch_size, horizon, n_x))

        P[:, -1], p[:, -1] = Q[:, -1], q[:, -1]

        backward_fail = False
        for i in reversed(range(horizon - 1)):
            Q_i, q_i, R_i, r_i, S_i = (Q[:, i], q[:, i], R[:, i], r[:, i], S[:, i])

            A_i, B_i, b_i = A[:, i], B[:, i], b[:, i]

            C_i, D_i, d_i = None, None, None
            if problem.n_h != 0:
                C_i, D_i, d_i = C[:, i], D[:, i], d[:, i]

            K_i, k_i, P_i, p_i, success_mask = lqr_step_backward_(
                reg=reg,
                Q=Q_i,
                q=q_i,
                R=R_i,
                r=r_i,
                S=S_i,
                P_next=P[:, i + 1],
                p_next=p[:, i + 1],
                A=A_i,
                B=B_i,
                b=b_i,
                C=C_i,
                D=D_i,
                d=d_i,
            )

            if not success_mask.all():
                reg[~success_mask] *= 10.0
                backward_fail = True
                break  # Break out of the time-step loop, retry the whole trajectory

            K[:, i], k[:, i], P[:, i], p[:, i] = K_i, k_i, P_i, p_i

        if not backward_fail:
            return K, k, P, p

    raise RuntimeError("LQR Backward pass failed: Max regularization retries reached.")


def lqr_forward_pass_(problem: Problem, K, k, P, p, A, B, b):
    # TODO: Add initial state optimization as an option
    batch_size = problem.batch_size
    horizon = problem.horizon
    n_x = problem.n_x
    n_u = problem.n_u
    n_h = problem.n_h

    dx = torch.zeros((batch_size, horizon, n_x))
    du = torch.zeros((batch_size, horizon - 1, n_u))
    # Lagrange multipliers of the actuation part
    mu = torch.zeros((batch_size, horizon, n_x))
    # Lagrange multipliers of the underactuation part
    nu = torch.zeros((batch_size, horizon - 1, n_h))

    for i in range(horizon - 1):
        dx0 = dx[:, i]

        A_i, B_i, b_i = A[:, i], B[:, i], b[:, i]

        K_i, k_i = K[:, i], k[:, i]

        (
            dx[:, i + 1],
            du[:, i],
            mu[:, i + 1],
            nu[:, i],
        ) = lqr_step_forward_(
            x=dx0,
            K=K_i,
            k=k_i,
            P_next=P[:, i + 1],
            p_next=p[:, i + 1],
            A=A_i,
            B=B_i,
            b=b_i,
        )

    return AdmmSolution(
        dx=dx,
        du=du,
        mu=mu,
        nu=nu,
        z=None,
        ksi=None,
        rho=None,
        rho_inv=None,
        rho_common=None,
    )


def lqr_step_backward_(
    reg, Q, q, R, r, S, P_next, p_next, A, B, b, C=None, D=None, d=None
):
    # Create Q_, q_, R_, r_, S_
    # Pre-transpose matrices
    AT = tran(A)
    BT = tran(B)
    ST = tran(S)

    # cache term to reuse in the calculations
    l = mv(P_next, b) + p_next

    ## TODO: Optimize these with einsum for quadratics
    Q_ = Q + mm(AT, mm(P_next, A))
    q_ = q + mv(AT, l)
    R_ = R + mm(BT, mm(P_next, B))
    r_ = r + mv(BT, l)
    S_ = S + mm(BT, mm(P_next, A))

    # Apply first regularization
    n_u = R_.shape[-1]
    reg_I = reg.view(-1, 1, 1) * torch.eye(
        n_u,
    ).unsqueeze(0)
    R_ = R_ + reg_I

    # Check Positive Definiteness (Cholesky)
    # This is the most critical check for descent directions!
    L_primal, info_chol = torch.linalg.cholesky_ex(R_)
    primal_success = info_chol == 0

    if not primal_success.all():
        print("Primal pos def test failed")
        return None, None, None, None, primal_success

    if C is not None:
        n_h = D.shape[-2]
        dim = n_u + n_h

        R_ext = torch.zeros((*R_.shape[:-2], dim, dim))
        R_ext[..., :n_u, :n_u] = R_
        R_ext[..., n_u:, :n_u] = D
        R_ext[..., :n_u, n_u:] = D.transpose(-2, -1)

        # Apply dual regularization in the extended R
        # R_ext[..., n_u:, n_u:] = -1e-8 * torch.eye(n_h)
        dual_reg = -1e-8 * torch.eye(n_h, device=R_.device, dtype=R_.dtype).unsqueeze(0)
        R_ext[..., n_u:, n_u:] = dual_reg

        R_ = R_ext
        r_ = torch.cat([r_, d], dim=-1)
        S_ = torch.cat([S_, C], dim=-2)
        S_T = tran(S_)

        # Solve via LU because R_ext may be symmetric indefinite
        LU, pivots, info_lu = torch.linalg.lu_factor_ex(R_ext)
        lu_success = info_lu == 0

        if not lu_success.all():
            print("LU failed")
            # Catches extreme NaNs/Infs
            return None, None, None, None, lu_success

        K = torch.linalg.lu_solve(LU, pivots, -S_)
        k = torch.linalg.lu_solve(LU, pivots, -r_.unsqueeze(-1)).squeeze(-1)
    else:
        S_T = tran(S_)
        K = torch.cholesky_solve(-S_, L_primal)
        k = torch.cholesky_solve(-r_.unsqueeze(-1), L_primal).squeeze(-1)

    # Check for ill-conditioning: too big gains
    max_k_norm = torch.max(torch.abs(k), dim=-1).values
    well_conditioned = (max_k_norm < 1e6) & ~torch.isnan(max_k_norm)

    if not well_conditioned.all():
        print("K not well conditioned")
        return None, None, None, None, well_conditioned

    # Compute P, p
    P = Q_ + mm(S_T, K)
    p = q_ + mv(S_T, k)

    success_mask = torch.ones(R_.shape[0], dtype=torch.bool, device=R_.device)
    return K, k, P, p, success_mask


def lqr_step_forward_(x, K, k, P_next, p_next, A, B, b):
    n_u = B.shape[-1]
    u_ = mv(K, x) + k
    u = u_[..., :n_u]
    nu = u_[..., n_u:]

    x_next = mv(A, x) + mv(B, u) + b

    pi = mv(P_next, x_next) + p_next

    return x_next, u, pi, nu
