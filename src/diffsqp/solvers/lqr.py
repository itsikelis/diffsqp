import torch

from diffsqp.problems import Problem
from diffsqp.utils.math import mm, mv


class Lqr:
    def __init__(self, prob: Problem) -> None:
        self.prob = prob
        self.horizon = self.prob.horizon
        self.nB = self.prob.states[0].shape[0]
        self.nx = self.prob.nx
        self.nu = self.prob.nu

        self.A = [None] * (self.horizon - 1)
        self.B = [None] * (self.horizon - 1)
        self.b = [None] * (self.horizon - 1)

        self.K = [None] * (self.horizon - 1)
        self.k = [None] * (self.horizon - 1)

        self.K_lam = [None] * (self.horizon - 1)
        self.k_lam = [None] * (self.horizon - 1)

        self.V = [None] * self.horizon
        self.v = [None] * self.horizon

        self.Dx = [None] * self.horizon
        self.Du = [None] * (self.horizon - 1)
        # Lagrange multipliers of the actuation part
        self.Dpi = [None] * (self.horizon - 1)
        # Lagrange multipliers of the underactuation part
        self.Dlam = [None] * (self.horizon - 1)

    def solve(self):
        self.backward_pass_()
        self.forward_pass_()

        # Return corrections
        return self.Dx, self.Du, self.Dpi, self.Dlam

    def backward_pass_(self):
        x_N = self.prob.states[self.horizon - 1]
        self.V[-1], self.v[-1] = self.calc_final_cost_terms_(x_N)
        for i in range(self.horizon - 2, -1, -1):
            x_lin = self.prob.states[i]
            u_lin = self.prob.controls[i]
            x_next = self.prob.states[i + 1]

            Q, R, S, q, r = self.calc_linearized_cost_terms_(i, x_lin, u_lin)
            self.A[i], self.B[i], self.b[i] = self.calc_linearized_dynamics_terms_(
                x_lin, u_lin, x_next, self.prob.stage_dynamics[i]
            )

            C = None
            D = None
            e = None
            if self.prob.constraints[i]:
                C, D, e = self.calc_linearized_constraint_terms_(
                    x_lin, u_lin, x_next, self.prob.constraints[i]
                )

            (
                self.K[i],
                self.k[i],
                self.V[i],
                self.v[i],
                self.K_lam[i],
                self.k_lam[i],
            ) = self.riccati_backward_(
                Q=Q,
                q=q,
                R=R,
                r=r,
                S=S,
                V=self.V[i + 1],
                v=self.v[i + 1],
                A=self.A[i],
                B=self.B[i],
                b=self.b[i],
                C=C,
                D=D,
                e=e,
            )

    def forward_pass_(self):
        # TODO: Add initial state optimization as an option
        nx = self.prob.nx
        self.Dx[0] = torch.zeros([self.nB, nx])
        for i in range(self.horizon - 1):
            x_lin = self.prob.states[i]
            u_lin = self.prob.controls[i]
            x_next = self.prob.states[i + 1]

            Dx0 = self.Dx[i]
            K = self.K[i]
            k = self.k[i]
            K_lam = self.K_lam[i]
            k_lam = self.k_lam[i]
            V = self.V[i + 1]
            v = self.v[i + 1]

            A = self.A[i]
            B = self.B[i]
            b = self.b[i]

            self.Dx[i + 1], self.Du[i], self.Dpi[i], self.Dlam[i] = (
                self.riccati_forward_(
                    Dx0=Dx0, K=K, k=k, A=A, B=B, b=b, K_lam=K_lam, k_lam=k_lam, V=V, v=v
                )
            )

            # print("Cx + Du + e = ", (mv(C, self.Dx[i]) + mv(D, self.Du[i]) + e)[0, 0])

    def riccati_backward_(self, Q, q, R, r, S, V, v, A, B, b, C, D, e):
        nB = Q.shape[0]
        nx = Q.shape[1]
        nu = R.shape[2]
        ng = None if D is None else D.shape[1]  # n of equality constraints

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

        K_ = None
        k_ = None
        K_lam = None
        k_lam = None
        V_ = None
        v_ = None
        if C is None:
            K_ = torch.linalg.solve(R_, -S_)
            k_ = torch.linalg.solve(R_, -r_)
            V_ = Q_ + mm(S_T, K_)
            v_ = q_ + mv(S_T, k_)
        else:
            S_ext = torch.cat([S_, C], dim=1)
            r_ext = torch.cat([r_, e], dim=1)
            S_extT = S_ext.transpose(1, 2)

            dim = R_.shape[1] + D.shape[1]
            R_ext = torch.zeros((nB, dim, dim))
            R_ext[:, 0:nu, 0:nu] = R_
            R_ext[:, nu:, 0:nu] = D
            R_ext[:, 0:nu, nu:] = torch.transpose(D, 1, 2)

            K_ext = torch.linalg.solve(R_ext, -S_ext)
            k_ext = torch.linalg.solve(R_ext, -r_ext)
            # Sanity check
            nB = self.nB
            nx = self.prob.nx
            nu = self.prob.nu
            assert K_ext.shape == torch.Size([nB, nu + ng, nx])
            assert k_ext.shape == torch.Size([nB, nu + ng])

            K_ = K_ext[:, 0:nu, :]
            k_ = k_ext[:, 0:nu]

            K_lam = K_ext[:, nu:, :]
            k_lam = k_ext[:, nu:]

            V_ = Q_ + mm(S_extT, K_ext)
            v_ = q_ + mv(S_extT, k_ext)

        # Sanity checks
        nB = self.nB
        nx = self.prob.nx
        nu = self.prob.nu
        assert Q_.shape == torch.Size([nB, nx, nx])
        assert q_.shape == torch.Size([nB, nx])
        assert R_.shape == torch.Size([nB, nu, nu])
        assert r_.shape == torch.Size([nB, nu])
        assert K_.shape == torch.Size([nB, nu, nx])
        assert k_.shape == torch.Size([nB, nu])
        assert V_.shape == torch.Size([nB, nx, nx])
        assert v_.shape == torch.Size([nB, nx])
        if K_lam is not None:
            assert K_lam.shape == torch.Size([nB, ng, nx])
            assert k_lam.shape == torch.Size([nB, ng])

        return K_, k_, V_, v_, K_lam, k_lam

    def riccati_forward_(self, Dx0, K, k, A, B, b, K_lam, k_lam, V, v):
        Du = mv(K, Dx0) + k
        Dx = mv(A, Dx0) + mv(B, Du) + b
        Dpi = mv(V, Dx) + v
        Dlam = None
        if K_lam is not None:
            Dlam = mv(K_lam, Dx0) + k_lam

        # Sanity checks
        nB = self.nB
        nx = self.prob.nx
        nu = self.prob.nu
        assert Dx.shape == torch.Size([nB, nx])
        assert Du.shape == torch.Size([nB, nu])

        return Dx, Du, Dpi, Dlam

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

    def calc_linearized_constraint_terms_(self, x_lin, u_lin, x_next, dynamics):
        C = dynamics.gx(x_lin, u_lin)
        D = dynamics.gu(x_lin, u_lin)
        e = dynamics.g(x_lin, u_lin)
        return C, D, e

    def calc_final_cost_terms_(self, x_N):
        V = self.prob.lxx(-1, x_N)
        v = self.prob.lx(-1, x_N)
        return V, v
