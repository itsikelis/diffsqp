import torch
from torch import sin, cos

from diffsqp.dynamics import Dynamics


class AcrobotInverseDynamics(Dynamics):
    def __init__(
        self,
        m1: float,
        m2: float,
        l1: float,
        l2: float,
        lc1: float,
        lc2: float,
        grav: float = 9.81,
        I1=None,
        I2=None,
    ):
        super().__init__(nx=4, nu=2, nq=2, nv=2)
        self.type = "inverse"
        self.ng = 1

        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.lc1 = lc1
        self.lc2 = lc2
        self.grav = grav

        self.I1 = I1 if I1 is not None else (self.m1 * self.l1**2) / 3.0
        self.I2 = I2 if I1 is not None else (self.m2 * self.l2**2) / 3.0

    def g(self, x: torch.Tensor, u: torch.Tensor):
        n_batch = x.shape[0]
        m1 = self.m1
        m2 = self.m2
        l1 = self.l1
        lc1 = self.lc1
        lc2 = self.lc2
        grav = self.grav
        I1 = self.I1
        I2 = self.I2

        th1 = x[:, 0:1]
        th2 = x[:, 1:2]
        dth1 = x[:, 2:3]
        dth2 = x[:, 3:4]
        ddth1 = u[:, 0:1]
        ddth2 = u[:, 1:2]

        c2 = torch.cos(th2)
        s2 = torch.sin(th2)
        s1 = torch.sin(th1)
        s12 = torch.sin(th1 + th2)

        mult1 = I1 + I2 + m2 + l1 * l1 + 2.0 * m2 * l1 * lc2 * c2
        mult2 = I2 + m2 * l1 * lc2 * c2
        mult3 = -2.0 * m2 * l1 * lc2 * s2 * dth2
        mult4 = -m2 * l1 * lc2 * s2 * dth2
        sum5 = -m1 * grav * lc1 * s1 - m2 * grav * (l1 * s1 + lc2 * s12)
        res = mult1 * ddth1 + mult2 * ddth2 + mult3 * dth1 + mult4 * dth2 + sum5
        return res

    def gx(self, x: torch.Tensor, u: torch.Tensor):
        n_batch = x.shape[0]
        ng = self.ng
        nx = self.nx

        m1 = self.m1
        m2 = self.m2
        l1 = self.l1
        l2 = self.l2
        lc1 = self.lc1
        lc2 = self.lc2
        grav = self.grav
        I1 = self.I1
        I2 = self.I2

        th1 = x[:, 0:1]
        th2 = x[:, 1:2]
        dth1 = x[:, 2:3]
        dth2 = x[:, 3:4]
        ddth1 = u[:, 0:1]
        ddth2 = u[:, 1:2]

        c1 = torch.cos(th1)
        c2 = torch.cos(th2)
        s1 = torch.sin(th1)
        s2 = torch.sin(th2)
        c12 = torch.cos(th1 + th2)

        grad = torch.zeros((n_batch, ng, nx))

        dres_dth1 = -m1 * grav * lc1 * c1 - m2 * grav * (l1 * c1 + lc2 * c12)

        d_mult1_dth2 = -2.0 * m2 * l1 * lc2 * s2
        d_mult2_dth2 = -m2 * l1 * lc2 * s2
        d_mult3_dth2 = -2.0 * m2 * l1 * lc2 * c2 * dth2
        d_mult4_dth2 = -m2 * l1 * lc2 * c2 * dth2
        d_sum5_dth2 = -m2 * grav * lc2 * c12

        dres_dth2 = (
            d_mult1_dth2 * ddth1
            + d_mult2_dth2 * ddth2
            + d_mult3_dth2 * dth1
            + d_mult4_dth2 * dth2
            + d_sum5_dth2
        )

        dres_d_dth1 = -2.0 * m2 * l1 * lc2 * s2 * dth2

        dres_d_dth2 = -2.0 * m2 * l1 * lc2 * s2 * (dth1 + dth2)

        grad[:, 0:1, 0:1] = dres_dth1.unsqueeze(2)
        grad[:, 0:1, 1:2] = dres_dth2.unsqueeze(2)
        grad[:, 0:1, 2:3] = dres_d_dth1.unsqueeze(2)
        grad[:, 0:1, 3:4] = dres_d_dth2.unsqueeze(2)
        return grad

    def gu(self, x: torch.Tensor, u: torch.Tensor):
        n_batch = x.shape[0]
        ng = self.ng
        nu = self.nu

        m1 = self.m1
        m2 = self.m2
        l1 = self.l1
        l2 = self.l2
        lc1 = self.lc1
        lc2 = self.lc2
        grav = self.grav
        I1 = self.I1
        I2 = self.I2

        th1 = x[:, 0:1]
        th2 = x[:, 1:2]
        dth1 = x[:, 2:3]
        dth2 = x[:, 3:4]
        ddth1 = u[:, 0:1]
        ddth2 = u[:, 1:2]

        c1 = torch.cos(th1)
        c2 = torch.cos(th2)
        s2 = torch.sin(th2)
        s1 = torch.sin(th1)
        s12 = torch.sin(th1 + th2)

        mult1 = I1 + I2 + m2 + l1 * l1 + 2.0 * m2 * l1 * lc2 * c2
        mult2 = I2 + m2 * l1 * lc2 * c2

        grad = torch.zeros((n_batch, ng, nu))
        grad[:, 0:1, 0:1] = mult1.unsqueeze(2)
        grad[:, 0:1, 1:2] = mult2.unsqueeze(2)
        return grad
