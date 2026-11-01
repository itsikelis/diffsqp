import torch
from torch import sin, cos

from diffsqp.constraints import GenericConstraint


class ControlBounds(GenericConstraint):
    stage_batched = True

    def __init__(self, n_x, n_u, lb, ub):
        super().__init__(n_g=n_u, n_x=n_x, n_u=n_u, lb=lb, ub=ub)

    def g(self, x: torch.Tensor, u: torch.Tensor):
        res = u
        return res

    def gx(self, x: torch.Tensor, u: torch.Tensor):
        return torch.zeros((*x.shape[:-1], self.n_g, self.n_x), dtype=x.dtype, device=x.device)

    def gu(self, x: torch.Tensor, u: torch.Tensor):
        eye = torch.eye(self.n_u, dtype=x.dtype, device=x.device)
        return eye.expand(*x.shape[:-1], self.n_g, self.n_u)
