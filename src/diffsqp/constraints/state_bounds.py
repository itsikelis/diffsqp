import torch
from torch import sin, cos

from diffsqp.constraints import GenericConstraint


class StateBounds(GenericConstraint):
    def __init__(self, n_x, n_u, lb, ub):
        super().__init__(n_g=n_x, n_x=n_x, n_u=n_u, lb=lb, ub=ub)

    def g(self, x: torch.Tensor, u: torch.Tensor):
        res = x
        return res

    def gx(self, x: torch.Tensor, u: torch.Tensor):
        batch_size = x.shape[0]
        n_g = self.n_g
        n_x = self.n_x
        grad = torch.zeros((batch_size, n_g, n_x))
        grad[:] = torch.eye(n_x)
        return grad

    def gu(self, x: torch.Tensor, u: torch.Tensor):
        batch_size = x.shape[0]
        n_g = self.n_g
        n_u = self.n_u
        grad = torch.zeros((batch_size, n_g, n_u))
        return grad
