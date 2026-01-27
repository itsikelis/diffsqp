import torch
from torch import bmm

from diffsqp.problems import Problem


class Lqr:
    def __init__(self, prob: Problem) -> None:
        self.prob = prob
        self.horizon = self.prob.horizon
        self.n_batch = self.prob.variables.shape[0]

        # For the backward pass
        self.V = [None] * self.horizon
        self.u = [None] * self.horizon
        self.h = [None] * (self.horizon - 1)
        self.H = [None] * (self.horizon - 1)
        self.G = [None] * (self.horizon - 1)
        self.K = [None] * (self.horizon - 1)
        self.k = [None] * (self.horizon - 1)
        self.gamma = [None] * (self.horizon - 1)

        # For the forward pass
        self.delta_x = [None] * self.horizon
        self.delta_u = [None] * (self.horizon - 1)
        self.lagrange_mult = [None] * (self.horizon - 1)

    # def ensure_batch(t): return t.expand(n_batch, -1, -1) if t.dim() == 2 else t
    #
    # A, B, Q, R, Qf = map(ensure_batch, [A, B, Q, R, Qf])

    def solve(self):
        # 1. Backward Pass
        self.backward_pass()

        # 2. Forward Pass
        self.forward_pass()

        # Return corrections
        return self.delta_x, self.delta_u

    def backward_pass(self):
        x_F = self.prob.state(self.horizon - 1)
        self.V[-1] = self.prob.costs[-1].lxx(x_F, None)
        self.u[-1] = self.prob.costs[-1].lx(x_F, None)

        # Loop backwards in horizon
        for k in range(self.horizon - 2, -1, -1):
            nB = self.n_batch
            nx = self.prob.n_state
            nu = self.prob.n_ctrl

            x_k = self.prob.state(k)
            u_k = self.prob.control(k)
            x_pred = self.prob.stage_dynamics[k].f(x_k, u_k, self.prob.dt)
            self.gamma[k] = x_pred - self.prob.state(k + 1)

            Q = self.prob.costs[k].lxx(x_k, u_k)
            q = self.prob.costs[k].lx(x_k, u_k)

            S = self.prob.costs[k].lxu(x_k, u_k)
            ST = torch.transpose(S, 1, 2)
            R = self.prob.costs[k].luu(x_k, u_k)

            A = self.prob.stage_dynamics[k].fx(x_k, u_k, self.prob.dt)
            AT = torch.transpose(A, 1, 2)

            B = self.prob.stage_dynamics[k].fu(x_k, u_k, self.prob.dt)
            BT = torch.transpose(B, 1, 2)

            r_k = self.prob.costs[k].lu(x_k, u_k)

            assert self.gamma[k].shape == torch.Size([nB, nx])
            assert self.V[k + 1].shape == torch.Size([nB, nx, nx])
            assert r_k.shape == torch.Size([nB, nu])
            assert Q.shape == torch.Size([nB, nx, nx])
            assert q.shape == torch.Size([nB, nx])
            assert S.shape == torch.Size([nB, nx, nu])
            assert ST.shape == torch.Size([nB, nu, nx])
            assert R.shape == torch.Size([nB, nu, nu])
            assert A.shape == torch.Size([nB, nx, nx])
            assert AT.shape == torch.Size([nB, nx, nx])
            assert B.shape == torch.Size([nB, nx, nu])
            assert BT.shape == torch.Size([nB, nu, nx])

            # Calculate h_k
            prod1 = bmm(self.V[k + 1], self.gamma[k].unsqueeze(2)).squeeze(2)
            prod1 = self.u[k + 1] + prod1
            prod2 = bmm(BT, prod1.unsqueeze(2)).squeeze(2)
            self.h[k] = r_k + prod2

            assert self.h[k].shape == torch.Size([nB, nu])
            # Calculate G_k
            prod1 = bmm(self.V[k + 1], A)
            prod2 = bmm(BT, prod1)
            self.G[k] = ST + prod2

            assert self.G[k].shape == torch.Size([nB, nu, nx])
            # Calculate H_k
            prod1 = bmm(self.V[k + 1], B)
            prod2 = bmm(BT, prod1)
            self.H[k] = R + prod2

            assert self.H[k].shape == torch.Size([nB, nu, nu])
            # Calculate K_k, k_k
            Hk_inv = torch.inverse(self.H[k])
            self.K[k] = -bmm(Hk_inv, self.G[k])
            K_kT = torch.transpose(self.K[k], 1, 2)
            self.k[k] = -bmm(Hk_inv, self.h[k].unsqueeze(2)).squeeze(2)

            assert self.K[k].shape == torch.Size([nB, nu, nx])
            assert self.k[k].shape == torch.Size([nB, nu])

            # Calculate V_k
            prod1 = bmm(AT, bmm(self.V[k + 1], A))
            prod2 = bmm(K_kT, bmm(self.H[k], self.K[k]))
            self.V[k] = Q + prod1 - prod2

            assert self.V[k].shape == torch.Size([nB, nx, nx])
            # Calculate u_k
            prod1 = bmm(K_kT, r_k.unsqueeze(2)).squeeze(2)
            prod2 = torch.transpose(A + bmm(B, self.K[k]), 1, 2)
            prod3 = bmm(self.V[k + 1], self.gamma[k].unsqueeze(2)).squeeze(2)
            prod3 = self.u[k + 1] + prod3

            self.u[k] = q + prod1 + bmm(prod2, prod3.unsqueeze(2)).squeeze(2)

            assert self.u[k].shape == torch.Size([nB, nx])

    def forward_pass(self):
        nB = self.n_batch
        nx = self.prob.n_state
        nu = self.prob.n_ctrl

        self.delta_x[0] = torch.zeros([nB, nx])
        for k in range(self.horizon - 1):
            x_k = self.prob.state(k)
            u_k = self.prob.control(k)
            A_k = self.prob.stage_dynamics[k].fx(x_k, u_k, self.prob.dt)
            B_k = self.prob.stage_dynamics[k].fu(x_k, u_k, self.prob.dt)
            term1 = bmm(
                A_k + bmm(B_k, self.K[k]), self.delta_x[k].unsqueeze(2)
            ).squeeze(2)
            term2 = bmm(B_k, self.k[k].unsqueeze(2)).squeeze(2)
            term3 = self.gamma[k]
            self.delta_x[k + 1] = term1 + term2 + term3

            term1 = bmm(self.K[k], self.delta_x[k].unsqueeze(2)).squeeze(2)
            term2 = self.k[k]
            self.delta_u[k] = term1 + term2

            term1 = bmm(self.V[k], self.delta_x[k].unsqueeze(2)).squeeze(2)
            term2 = self.u[k]
            self.lagrange_mult[k] = term1 + term2

            assert self.delta_x[k + 1].shape == torch.Size([nB, nx])
            assert self.delta_u[k].shape == torch.Size([nB, nu])
            assert self.lagrange_mult[k].shape == torch.Size([nB, nx])
