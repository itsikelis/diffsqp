import torch

from diffsqp.costs import Cost


class LqrCost(Cost):
    def __init__(self, Q, R, x_des=None, u_des=None):
        self.n_batch = Q.shape[0]
        self.n_state = Q.shape[1]
        self.n_ctrl = R.shape[1]
        self.Q = Q
        self.R = R

        if x_des is None:
            x_des = torch.zeros((self.n_batch, self.n_state))
        if u_des is None:
            u_des = torch.zeros((self.n_batch, self.n_ctrl))
        self.x_des = x_des
        self.u_des = u_des

    def l(self, x, u):
        x_term = torch.bmm(
            torch.bmm(torch.transpose((x - self.x_des).unsqueeze(2), 1, 2), self.Q),
            (x - self.x_des).unsqueeze(2),
        ).squeeze(2)
        u_term = torch.bmm(
            torch.bmm(torch.transpose((u - self.u_des).unsqueeze(2), 1, 2), self.R),
            (u - self.u_des).unsqueeze(2),
        ).squeeze(2)
        return 0.5 * (x_term + u_term)

    def lx(self, x, u):
        """Gradient w.r.t x (B, n_state, 1)"""
        return torch.bmm(self.Q, (x - self.x_des).unsqueeze(2)).squeeze(2)

    def lu(self, x, u):
        """Gradient w.r.t u (B, n_ctrl, 1)"""
        return torch.bmm(self.R, (u - self.u_des).unsqueeze(2)).squeeze(2)

    def lxx(self, x, u):
        """Hessian w.r.t xx (B, n_state, n_state)"""
        return self.Q

    def luu(self, x, u):
        """Hessian w.r.t uu (B, n_ctrl, n_ctrl)"""
        return self.R

    def lux(self, x, u):
        """Hessian w.r.t ux (B, n_ctrl, n_state)"""
        return torch.zeros(
            self.n_batch, self.n_ctrl, self.n_state, device=x.device, dtype=x.dtype
        )

    def lxu(self, x, u):
        """Hessian w.r.t xu (B, n_state, n_ctrl)"""
        return torch.zeros(
            self.n_batch, self.n_state, self.n_ctrl, device=x.device, dtype=x.dtype
        )
