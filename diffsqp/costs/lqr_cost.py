import torch

from diffsqp.costs import Cost


class LqrCost(Cost):
    def __init__(self, Q, R):
        self.Q = Q
        self.R = R

    def l(self, x, u):
        x_term = torch.bmm(torch.bmm(torch.transpose(x, 1, 2), self.Q), x)
        u_term = torch.bmm(torch.bmm(torch.transpose(u, 1, 2), self.R), u)
        return 0.5 * (x_term + u_term)

    def lx(self, x, u):
        deriv = torch.bmm(self.Q, x)
        return torch.transpose(deriv, 1, 2)

    def lu(self, x, u):
        deriv = torch.bmm(self.R, u)
        return torch.transpose(deriv, 1, 2)

    def lxx(self, x, u):
        return self.Q

    def luu(self, x, u):
        return self.R

    def lxu(self, x, u):
        return 0.0
