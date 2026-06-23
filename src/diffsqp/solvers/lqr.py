import torch

from diffsqp.problems import Problem
from diffsqp.utils.math import mm, mv, tran
from diffsqp.types import Trajectory, QpParameters, QpSolution


def lqr_backward_pass(problem: Problem, matrices: QpParameters):
    batch_size = problem.n_batch
    horizon = problem.horizon
    n_x = problem.n_x
    n_u = problem.n_u
    n_h = problem.n_h

    K = torch.zeros((batch_size, horizon - 1, n_u + n_h, n_x))
    k = torch.zeros((batch_size, horizon - 1, n_u + n_h))

    P = torch.zeros((batch_size, horizon, n_x, n_x))
    p = torch.zeros((batch_size, horizon, n_x))

    P[:, -1], p[:, -1] = matrices.Q[:, -1], matrices.q[:, -1]

    for i in reversed(range(horizon - 1)):
        Q_i, q_i, R_i, r_i, S_i = (
            matrices.Q[:, i],
            matrices.q[:, i],
            matrices.R[:, i],
            matrices.r[:, i],
            matrices.S[:, i],
        )

        A_i, B_i, b_i = matrices.A[:, i], matrices.B[:, i], matrices.b[:, i]

        C_i, D_i, d_i = None, None, None
        if problem.inverse_dynamics:
            C_i, D_i, d_i = matrices.C[:, i], matrices.D[:, i], matrices.d[:, i]

        (
            K[:, i],
            k[:, i],
            P[:, i],
            p[:, i],
        ) = lqr_step_backward_(
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

    return K, k, P, p


def lqr_forward_pass(problem: Problem, K, k, P, p, A, B, b):
    # TODO: Add initial state optimization as an option
    batch_size = problem.n_batch
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

    return QpSolution(dx=dx, du=du, mu=mu, nu=nu, lam=None)


def lqr_step_backward_(Q, q, R, r, S, P_next, p_next, A, B, b, C=None, D=None, d=None):
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

    if C is not None:
        n_u = R_.shape[-2]
        n_h = D.shape[-2]
        dim = n_u + n_h

        R_ext = torch.zeros((*R_.shape[:-2], dim, dim))
        R_ext[..., :n_u, :n_u] = R_
        R_ext[..., n_u:, :n_u] = D
        R_ext[..., :n_u, n_u:] = D.transpose(-2, -1)
        R_ = R_ext

        r_ = torch.cat([r_, d], dim=-1)

        S_ = torch.cat([S_, C], dim=-2)
    S_T = tran(S_)

    # Compute K, k
    K = torch.linalg.solve(R_, -S_)
    k = torch.linalg.solve(R_, -r_)

    # Compute P, p
    P = Q_ + mm(S_T, K)
    p = q_ + mv(S_T, k)

    return K, k, P, p


def lqr_step_forward_(x, K, k, P_next, p_next, A, B, b):
    n_u = B.shape[-1]
    u_ = mv(K, x) + k
    u = u_[..., :n_u]
    nu = u_[..., n_u:]

    x_next = mv(A, x) + mv(B, u) + b

    pi = mv(P_next, x_next) + p_next

    return x_next, u, pi, nu
