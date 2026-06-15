import torch
from torch import sin, cos

from diffsqp.constraints import UnderactuationConstraint
from diffsqp.dynamics import CartPoleParameters


class CartPoleUnderactuation(UnderactuationConstraint):
    def __init__(self, params: CartPoleParameters):
        super().__init__(ng=1, nx=4, nu=2)
        self.p = params

    def h(self, x: torch.Tensor, u: torch.Tensor):
        mc = self.p.mc
        mp = self.p.mp
        lp = self.p.lp
        grav = self.p.grav

        th = x[..., 1:2]
        dds = u[..., 0:1]
        ddth = u[..., 1:2]

        sth = torch.sin(th)
        cth = torch.cos(th)

        res = mp * lp * cth * dds + mp * lp * lp * ddth + mp * grav * lp * sth
        return res

    def hx(self, x: torch.Tensor, u: torch.Tensor):
        nB = x.shape[0]
        ng = self.ng
        nx = self.nx

        mp = self.p.mp
        lp = self.p.lp
        grav = self.p.grav
        th = x[:, 1:2]
        dds = u[:, 0:1]

        sth = torch.sin(th)
        cth = torch.cos(th)

        grad = torch.zeros((nB, ng, nx))
        # print((-mp * lp * sth * dds + mp * grav * lp * cth).shape)
        # exit()
        grad[:, 0:1, 1:2] = (-mp * lp * sth * dds + mp * grav * lp * cth).unsqueeze(2)
        return grad

    def hu(self, x: torch.Tensor, u: torch.Tensor):
        nB = x.shape[0]
        ng = self.ng
        nu = self.nu

        mp = self.p.mp
        lp = self.p.lp
        grav = self.p.grav
        th = x[:, 1:2]
        sth = torch.sin(th)
        cth = torch.cos(th)

        grad = torch.zeros((nB, ng, nu))
        grad[:, 0:1, 0:1] = (mp * lp * cth).unsqueeze(2)
        grad[:, 0:1, 1:] = mp * lp * lp
        return grad
