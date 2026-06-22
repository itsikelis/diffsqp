import torch

from diffsqp.problems import Problem
from diffsqp.utils.math import mm, mv, tran
from diffsqp.types import Trajectory


class Lqr:
    def __init__(self, prob: Problem) -> None:
        self.prob = prob
        self.horizon = self.prob.horizon
        self.nB = self.prob.states[:, 0].shape[0]

        self.A = [None] * (self.horizon - 1)
        self.B = [None] * (self.horizon - 1)
        self.b = [None] * (self.horizon - 1)

        self.K = [None] * (self.horizon - 1)
        self.k = [None] * (self.horizon - 1)

        self.P = [None] * self.horizon
        self.p = [None] * self.horizon

        self.delta_x = torch.zeros((self.nB, self.horizon, self.prob.n_x))
        self.delta_u = torch.zeros((self.nB, self.horizon - 1, self.prob.n_u))
        # Lagrange multipliers of the actuation part
        self.mu = torch.zeros((self.nB, self.horizon, self.prob.n_x))
        # Lagrange multipliers of the underactuation part
        self.nu = torch.zeros((self.nB, self.horizon - 1, self.prob.n_h))

    def solve(self, current_guess: Trajectory):
        # TODO: Time these
        self.backward_pass_()
        self.forward_pass_()

        # Return results
        return self.delta_x, self.delta_u, self.mu, self.nu

    def backward_pass_(self):
        x_N = self.prob.states[:, self.horizon - 1]
        self.P[-1], self.p[-1] = self.calc_final_cost_terms_(x_N)

        for i in range(self.horizon - 2, -1, -1):
            x_lin = self.prob.states[:, i]
            u_lin = self.prob.controls[:, i]
            x_next = self.prob.states[:, i + 1]

            Q, R, S, q, r = self.calc_linearized_cost_terms_(i, x_lin, u_lin)
            self.A[i], self.B[i], self.b[i] = self.calc_linearized_dynamics_terms_(
                x_lin, u_lin, x_next, self.prob.dynamics
            )
            C, D, e = self.calc_linearized_underactuation_terms_(x_lin, u_lin)

            (
                self.K[i],
                self.k[i],
                self.P[i],
                self.p[i],
            ) = self.step_backward_(
                Q=Q,
                q=q,
                R=R,
                r=r,
                S=S,
                P_next=self.P[i + 1],
                p_next=self.p[i + 1],
                A=self.A[i],
                B=self.B[i],
                b=self.b[i],
                C=C,
                D=D,
                d=e,
            )

    def forward_pass_(self):
        # TODO: Add initial state optimization as an option
        nx = self.prob.n_x
        self.delta_x[:, 0] = torch.zeros([self.nB, nx])
        for i in range(self.horizon - 1):
            x_lin = self.prob.states[:, i]
            u_lin = self.prob.controls[:, i]
            x_next = self.prob.states[:, i + 1]

            delta_x0 = self.delta_x[:, i]

            (
                self.delta_x[:, i + 1],
                self.delta_u[:, i],
                self.mu[:, i + 1],
                self.nu[:, i],
            ) = self.step_forward_(
                x=delta_x0,
                K=self.K[i],
                k=self.k[i],
                P_next=self.P[i + 1],
                p_next=self.p[i + 1],
                A=self.A[i],
                B=self.B[i],
                b=self.b[i],
            )

    def step_backward_(
        self, Q, q, R, r, S, P_next, p_next, A, B, b, C=None, D=None, d=None
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

        if C is not None:
            n_u = R_.shape[-2]
            n_g = D.shape[-2]
            dim = n_u + n_g

            R_ext = torch.zeros((*R_.shape[:-2], dim, dim))
            R_ext[..., :n_u, :n_u] = R_
            R_ext[..., n_u:, :n_u] = D
            R_ext[..., :n_u, n_u:] = D.transpose(-2, -1)
            R_ = R_ext

            r_ = torch.cat([r_, d], dim=-1)

            S_ = torch.cat([S_, C], dim=-2)
        S_T = tran(S_)

        # Assert shapes of R_, S_, Q_, q_
        # nB = R.shape[:-2]
        # nx = A.shape[-2]
        # nu = B.shape[-1]
        # ng = D.shape[-2]
        # assert Q_.shape == torch.Size([*nB, nx, nx])
        # assert q_.shape == torch.Size([*nB, nx])
        # assert R_.shape == torch.Size([*nB, nu + ng, nu + ng])
        # assert r_.shape == torch.Size([*nB, nu + ng])

        # Compute K, k
        K = torch.linalg.solve(R_, -S_)
        k = torch.linalg.solve(R_, -r_)

        # Compute P, p
        P = Q_ + mm(S_T, K)
        p = q_ + mv(S_T, k)

        return K, k, P, p

    def step_forward_(self, x, K, k, P_next, p_next, A, B, b):
        n_u = B.shape[-1]
        u_ = mv(K, x) + k
        u = u_[..., :n_u]
        nu = u_[..., n_u:]

        x_next = mv(A, x) + mv(B, u) + b

        pi = mv(P_next, x_next) + p_next

        # Sanity checks
        # nB = self.nB
        # n_x = self.prob.n_x
        # n_u = self.prob.n_u
        # assert delta_x.shape == torch.Size([nB, n_x])
        # assert delta_u.shape == torch.Size([nB, n_u])

        return x_next, u, pi, nu

    def calc_linearized_cost_terms_(self, stage_idx, x_lin, u_lin):
        Q = self.prob.lxx(stage_idx, x_lin, u_lin)
        q = self.prob.lx(stage_idx, x_lin, u_lin)
        R = self.prob.luu(stage_idx, x_lin, u_lin)
        r = self.prob.lu(stage_idx, x_lin, u_lin)
        S = self.prob.lux(stage_idx, x_lin, u_lin)
        return Q, R, S, q, r

    def calc_linearized_dynamics_terms_(self, x_lin, u_lin, x_next, dynamics):
        x_pred = dynamics.f(x_lin, u_lin, self.prob.dt)
        b = x_pred - x_next
        A = dynamics.fx(x_lin, u_lin, self.prob.dt)
        B = dynamics.fu(x_lin, u_lin, self.prob.dt)
        return A, B, b

    def calc_linearized_underactuation_terms_(self, x_lin, u_lin):
        if self.prob.underactuation is None:
            return None, None, None

        C = self.prob.underactuation.hx(x_lin, u_lin)
        D = self.prob.underactuation.hu(x_lin, u_lin)
        e = self.prob.underactuation.h(x_lin, u_lin)
        return C, D, e

    def calc_final_cost_terms_(self, x_N):
        P = self.prob.lxx(-1, x_N)
        p = self.prob.lx(-1, x_N)
        return P, p
