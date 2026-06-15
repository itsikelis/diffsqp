import torch
from torch import sin, cos

from diffsqp.constraints import GenericConstraint


class ControlEqualityConstraint(GenericConstraint):
    def __init__(
        self,
        nx,
        nu,
        constr_value: torch.Tensor,
    ):
        self.ng = nu
        self.nx = nx
        self.nu = nu

        self.constr_u = constr_u
        self.constr_value = constr_value

    def g(self, x: torch.Tensor, u: torch.Tensor):
        nB = x.shape[0]
        res = u - self.constr_value
        return res

    def gx(self, x: torch.Tensor, u: torch.Tensor):
        nB = x.shape[0]
        ng = self.ng
        nx = self.nx
        grad = torch.zeros((nB, ng, nx))
        return grad

    def gu(self, x: torch.Tensor, u: torch.Tensor):
        nB = x.shape[0]
        ng = self.ng
        nu = self.nu
        grad = torch.zeros((nB, ng, nu))
        grad[:] = torch.eye(nu)
        return grad
