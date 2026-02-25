import torch


class TerminalCost:
    def __init__(self, Q, x_des=None):
        self.n_batch = Q.shape[0]
        self.n_state = Q.shape[1]
        self.Q = Q
        if x_des is None:
            x_des = torch.zeros((n_batch, n_state))

        self.x_des = x_des

    def l(self, x):
        x_term = torch.bmm(
            torch.bmm(torch.transpose((x - self.x_des).unsqueeze(2), 1, 2), self.Q),
            (x - self.x_des).unsqueeze(2),
        ).squeeze(1, 2)
        return 0.5 * (x_term)

    def lx(self, x):
        """Gradient w.r.t x (B, n_state, 1)"""
        return torch.bmm(self.Q, (x - self.x_des).unsqueeze(2)).squeeze(2)

    def lxx(self, x):
        """Hessian w.r.t xx (B, n_state, n_state)"""
        return self.Q
