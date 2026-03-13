import torch
from torch import sin, cos

from diffsqp.dynamics import Dynamics


class CartPoleInverseDynamics(Dynamics):
    def __init__(self, mc: float, mp: float, lp: float, grav: float = 9.81):
        super().__init__(nx=4, nu=2, nq=2, nv=2)
        self.type = "inverse"
        self.ng = 1

        self.mc = mc
        self.mp = mp
        self.lp = lp
        self.grav = grav

    def g(self, x: torch.Tensor, u: torch.Tensor):
        n_batch = x.shape[0]
        mc = self.mc
        mp = self.mp
        lp = self.lp
        grav = self.grav

        th = x[:, 1:2]
        dds = u[:, 0:1]
        ddth = u[:, 1:2]

        sth = torch.sin(th)
        cth = torch.cos(th)

        res = mp * lp * cth * dds + mp * lp * lp * ddth + mp * grav * lp * sth
        return res

    def gx(self, x: torch.Tensor, u: torch.Tensor):
        n_batch = x.shape[0]
        ng = self.ng
        nx = self.nx

        mp = self.mp
        lp = self.lp
        grav = self.grav
        th = x[:, 1:2]
        dds = u[:, 0:1]

        sth = torch.sin(th)
        cth = torch.cos(th)

        grad = torch.zeros((n_batch, ng, nx))
        # print((-mp * lp * sth * dds + mp * grav * lp * cth).shape)
        # exit()
        grad[:, 0:1, 1:2] = (-mp * lp * sth * dds + mp * grav * lp * cth).unsqueeze(2)
        return grad

    def gu(self, x: torch.Tensor, u: torch.Tensor):
        n_batch = x.shape[0]
        ng = self.ng
        nu = self.nu

        mp = self.mp
        lp = self.lp
        grav = self.grav
        th = x[:, 1:2]
        sth = torch.sin(th)
        cth = torch.cos(th)

        grad = torch.zeros((n_batch, ng, nu))
        grad[:, 0:1, 0:1] = (mp * lp * cth).unsqueeze(2)
        grad[:, 0:1, 1:] = mp * lp * lp
        return grad
