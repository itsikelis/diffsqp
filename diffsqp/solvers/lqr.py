import torch

from diffsqp.problems import Problem
from diffsqp.utils.math import mm, mv


class Lqr:
    def __init__(self, prob: Problem) -> None:
        self.prob = prob
        self.horizon = self.prob.horizon
        self.n_batch = self.prob.states[0].shape[0]
        self.n_state = self.prob.n_state
        self.n_ctrl = self.prob.n_ctrl

        self.K = [None] * (self.horizon - 1)
        self.k = [None] * (self.horizon - 1)

        self.V = [None] * self.horizon
        self.v = [None] * self.horizon

        self.Dx = [None] * self.horizon
        self.Du = [None] * (self.horizon - 1)

        # Only for API cohesion, does nothing (for now)
        self.delta_lambda = [torch.zeros((self.n_batch, self.n_state))] * (self.horizon)

    def solve(self):
        self.backward_pass_()
        self.forward_pass_()

        # Return corrections
        return self.Dx, self.Du

    def backward_pass_(self):
        x_N = self.prob.states[self.horizon - 1]
        self.V[-1], self.v[-1] = self.calc_final_cost_terms_(x_N, self.prob.costs[-1])

        for i in range(self.horizon - 2, -1, -1):
            x_lin = self.prob.states[i]
            u_lin = self.prob.controls[i]
            x_next = self.prob.states[i + 1]

            Q, R, S, q, r = self.calc_linearized_cost_terms_(
                x_lin, u_lin, self.prob.costs[i]
            )
            A, B, b = self.calc_linearized_dynamic_terms_(
                x_lin, u_lin, x_next, self.prob.stage_dynamics[i]
            )

            (self.V[i], self.v[i], self.K[i], self.k[i]) = self.riccati_backward_(
                Q=Q, q=q, R=R, r=r, S=S, A=A, B=B, b=b, V=self.V[i + 1], v=self.v[i + 1]
            )

    def forward_pass_(self):
        # TODO: Add initial state optimization as an option
        nx = self.prob.n_state
        self.Dx[0] = torch.zeros([self.n_batch, nx])
        for i in range(self.horizon - 1):
            x_lin = self.prob.states[i]
            u_lin = self.prob.controls[i]
            x_next = self.prob.states[i + 1]

            Dx0 = self.Dx[i]
            K = self.K[i]
            k = self.k[i]
            A, B, b = self.calc_linearized_dynamic_terms_(
                x_lin, u_lin, x_next, self.prob.stage_dynamics[i]
            )
            self.Dx[i + 1], self.Du[i] = self.riccati_forward_(
                Dx0=Dx0, K=K, k=k, A=A, B=B, b=b
            )

    def riccati_backward_(self, Q, q, R, r, S, A, B, b, V, v):
        AT = torch.transpose(A, 1, 2)
        BT = torch.transpose(B, 1, 2)
        ST = torch.transpose(S, 1, 2)

        Q_ = Q + mm(AT, mm(V, A))
        l = mv(V, b) + v
        q_ = q + mv(AT, l)
        R_ = R + mm(BT, mm(V, B))
        r_ = r + mv(BT, l)
        S_ = S + mm(BT, mm(V, A))

        S_T = S_.transpose(1, 2)
        K_ = torch.linalg.solve(R_, -S_)
        k_ = torch.linalg.solve(R_, -r_)

        V_ = Q_ - mm(S_T, -K_)
        v_ = q_ - mv(S_T, -k_)

        # Sanity checks
        nB = self.n_batch
        nx = self.prob.n_state
        nu = self.prob.n_ctrl
        assert Q_.shape == torch.Size([nB, nx, nx])
        assert q_.shape == torch.Size([nB, nx])
        assert R_.shape == torch.Size([nB, nu, nu])
        assert r_.shape == torch.Size([nB, nu])
        assert K_.shape == torch.Size([nB, nu, nx])
        assert k_.shape == torch.Size([nB, nu])
        assert V_.shape == torch.Size([nB, nx, nx])
        assert v_.shape == torch.Size([nB, nx])

        return V_, v_, K_, k_

    def riccati_forward_(self, Dx0, K, k, A, B, b):
        Du = mv(K, Dx0) + k
        Dx = mv(A, Dx0) + mv(B, Du) + b

        # Sanity checks
        nB = self.n_batch
        nx = self.prob.n_state
        nu = self.prob.n_ctrl
        assert Dx.shape == torch.Size([nB, nx])
        assert Du.shape == torch.Size([nB, nu])

        return Dx, Du

    def calc_linearized_cost_terms_(self, x_lin, u_lin, c):
        Q = c.lxx(x_lin, u_lin)
        q = c.lx(x_lin, u_lin)
        R = c.luu(x_lin, u_lin)
        r = c.lu(x_lin, u_lin)
        S = c.lux(x_lin, u_lin)
        return Q, R, S, q, r

    def calc_linearized_dynamic_terms_(self, x_lin, u_lin, x_next, dynamics):
        x_pred = dynamics.f(x_lin, u_lin, self.prob.dt)
        b = x_pred - x_next
        A = dynamics.fx(x_lin, u_lin, self.prob.dt)
        B = dynamics.fu(x_lin, u_lin, self.prob.dt)
        return A, B, b

    def calc_final_cost_terms_(self, x_N, final_cost):
        V = final_cost.lxx(x_N)
        v = final_cost.lx(x_N)
        return V, v
