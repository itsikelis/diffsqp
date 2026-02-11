import torch
from torch import sin, cos

from diffsqp.dynamics import Dynamics


class CartPoleDynamics(Dynamics):
    def __init__(self, mc: float, mp: float, lp: float, grav: float = 9.81):
        self.nx = 4
        self.nu = 1
        self.mc = mc
        self.mp = mp
        self.lp = lp
        self.grav = grav

    def fc(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Continuous time dynamics: x_dot = fc(x_k, u_k)
        """
        mc = self.mc
        mp = self.mp
        lp = self.lp
        grav = self.grav

        s = x[:, 0:1]
        th = x[:, 1:2]
        ds = x[:, 2:3]
        dth = x[:, 3:4]
        fx = u[:, 0:1]

        dds = (fx + mp * sin(th) * (lp * dth**2 + grav * cos(th))) / (
            mc + mp * (sin(th) ** 2)
        )

        ddth = (
            -fx * cos(th)
            - mp * lp * dth**2 * cos(th) * sin(th)
            - (mc + mp) * grav * sin(th)
        ) / (lp * (mc + mp * sin(th) ** 2))

        return torch.cat([ds, dth, dds, ddth], dim=1)

    def fcx(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        dfc/dx matrix: n_x x n_x
        """
        batch_size = x.shape[0]

        mc = self.mc
        mp = self.mp
        lp = self.lp
        grav = self.grav

        s = x[:, 0:1]
        th = x[:, 1:2]
        ds = x[:, 2:3]
        dth = x[:, 3:4]
        fx = u[:, 0:1]
        sth = sin(th)
        cth = cos(th)

        # ds_dot / dth
        numerator_a = (lp * mp * cos(th) * dth * dth) * (
            mc + mp * (1 - cos(th) * cos(th))
        ) - (lp * mp * sin(th) * dth * dth) * (mp * 2 * cos(th) * sin(th))
        numerator_b = -mp * 2 * cos(th) * sin(th) * fx
        numerator_c = (mp * grav * cos(2 * th)) * (
            mc + mp * (1 - cos(th) * cos(th))
        ) - (mp * grav * cos(th) * sin(th)) * (mp * 2 * cos(th) * sin(th))
        denominator = (mc + mp * (1 - cos(th) * cos(th))) ** 2

        dfc3_dth = (numerator_a + numerator_b + numerator_c) / denominator

        # dth_dot / dth
        numerator_a = -(lp * mp * dth * dth * cos(2 * th)) * (
            lp * mc + lp * mp * (1 - cos(th) * cos(th))
        ) + (lp * mp * cos(th) * sin(th) * dth * dth) * (
            lp * mp * 2 * cos(th) * sin(th)
        )
        numerator_b = (u * sin(th)) * (lp * mc + lp * mp * (1 - cos(th) * cos(th))) + (
            u * cos(th)
        ) * (lp * mp * 2 * cos(th) * sin(th))
        numerator_c = -((mc + mp) * grav * cos(th)) * (
            lp * mc + lp * mp * (1 - cos(th) * cos(th))
        ) + ((mc + mp) * grav * sin(th)) * (lp * mp * 2 * cos(th) * sin(th))
        denominator = (lp * mc + lp * mp * (1 - cos(th) * cos(th))) ** 2

        dfc4_dth = (numerator_a + numerator_b + numerator_c) / denominator

        # ds_dot / dth_dot
        numerator = 2 * dth * lp * mp * sth
        denominator = mc + mp * sth**2
        dfc3_ddth = numerator / denominator

        # dth_dot / dth_dot
        numerator = -(mp * lp * 2.0 * dth * cth * sth)
        denominator = lp * (mc + mp * sth**2)
        dfc4_ddth = numerator / denominator

        A = torch.zeros((batch_size, self.nx, self.nx), device=x.device)
        A[:, 0, :] = torch.tensor([0.0, 0.0, 1.0, 0.0])
        A[:, 1, :] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        A[:, 2, 1] = dfc3_dth.squeeze(1)
        A[:, 3, 1] = dfc4_dth.squeeze(1)
        A[:, 2, 3] = dfc3_ddth.squeeze(1)
        A[:, 3, 3] = dfc4_ddth.squeeze(1)

        return A

    def fcu(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        dfc/du matrix: n_x x n_u
        """
        batch_size = x.shape[0]

        mc = self.mc
        mp = self.mp
        lp = self.lp
        grav = self.grav

        s = x[:, 0:1]
        th = x[:, 1:2]
        ds = x[:, 2:3]
        dth = x[:, 3:4]
        fx = u[:, 0:1]
        sth = sin(th)
        cth = cos(th)

        dfc3_du = 1 / (mc + mp * sth**2)

        numerator = -cos(th)
        denominator = lp * mc + lp * mp * sth**2
        dfc4_du = numerator / denominator

        B = torch.zeros((batch_size, self.nx, self.nu), device=x.device)
        B[:, 2, :] = dfc3_du
        B[:, 3, :] = dfc4_du

        return B
