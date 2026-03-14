import torch
from torch import sin, cos

from diffsqp.dynamics import Dynamics


class AcrobotDynamics(Dynamics):
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

        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.lc1 = lc1
        self.lc2 = lc2
        self.grav = grav

        self.I1 = I1 if I1 is not None else (self.m1 * self.l1**2) / 3.0
        self.I2 = I2 if I1 is not None else (self.m2 * self.l2**2) / 3.0

    def fc(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Continuous time dynamics: x_dot = fc(x_k, u_k)
        """
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
        tau = u[:, 0:1]

        s1 = sin(th1)
        s2 = sin(th2)
        c2 = cos(th2)
        s12 = sin(th1 + th2)

        denom = I1 * I2 + I2 * l1**2 * m2 - l1**2 * lc2**2 * m2**2 * c2**2
        nom1 = -I2 * (
            grav * l1 * m2 * s1
            + grav * lc1 * m1 * s1
            + grav * lc2 * m2 * s12
            - l1 * lc2 * m2 * (2.0 * dth1 + dth2) * s2 * dth2
        )
        nom2 = (I2 + l1 * lc2 * m2 * c2) * (
            grav * lc2 * m2 * s12 + l1 * lc2 * m2 * s2 * dth1**2 - tau
        )
        ddth1 = (nom1 + nom2) / denom

        # denom = I1 * I2 + I2 * l1**2 * m2 - l1**2 * lc2**2 * m2**2 * c2**2
        nom1 = (I2 + l1 * lc2 * m2 * c2) * (
            grav * l1 * m2 * s1
            + grav * lc1 * m1 * s1
            + grav * lc2 * m2 * s12
            - l1 * lc2 * m2 * (2.0 * dth1 + dth2) * s2 * dth2
        )
        nom2 = -(grav * lc2 * m2 * s12 + l1 * lc2 * m2 * s2 * dth2 * dth2 - tau) * (
            I1 + I2 + l1 * l1 * m2 + 2.0 * l1 * lc2 * m2 * c2
        )
        ddth2 = (nom1 + nom2) / denom

        return torch.cat([dth1, dth2, ddth1, ddth2], dim=1)

    def fcx(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        dfc/dx matrix: n_x x n_x
        """
        nB = x.shape[0]

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
        tau = u[:, 0:1]

        s1 = sin(th1)
        s2 = sin(th2)
        c2 = cos(th2)
        s12 = sin(th1 + th2)
        c1 = cos(th1)
        c12 = cos(th1 + th2)

        d_nom1_th1 = -I2 * (
            grav * l1 * m2 * c1 + grav * lc1 * m1 * c1 + grav * lc2 * m2 * c12
        )
        d_nom2_th1 = (I2 + l1 * lc2 * m2 * c2) * (grav * lc2 * m2 * c12)
        denom = I1 * I2 + I2 * l1 * l1 * m2 - l1 * l1 * lc2 * lc2 * m2 * m2 * c2 * c2

        # DDTH1_TH1
        d_ddth1_th1 = (d_nom1_th1 + d_nom2_th1) / denom

        d_denom_th2 = 2 * l1 * l1 * lc2 * lc2 * m2 * m2 * c2 * s2
        d_nom1_th2 = -I2 * (
            grav * lc2 * m2 * c12 - l1 * lc2 * m2 * (2 * dth1 + dth2) * c2 * dth2
        )
        A = I2 + l1 * lc2 * m2 * c2
        B = grav * lc2 * m2 * s12 + l1 * lc2 * m2 * s2 * dth1 * dth1 - tau
        dA_th2 = -l1 * lc2 * m2 * s2
        dB_th2 = grav * lc2 * m2 * c12 + l1 * lc2 * m2 * c2 * dth1 * dth1
        d_nom2_th2 = dA_th2 * B + A * dB_th2
        nom1 = -I2 * (
            grav * l1 * m2 * s1
            + grav * lc1 * m1 * s1
            + grav * lc2 * m2 * s12
            - l1 * lc2 * m2 * (2 * dth1 + dth2) * s2 * dth2
        )
        nom2 = (I2 + l1 * lc2 * m2 * c2) * (
            grav * lc2 * m2 * s12 + l1 * lc2 * m2 * s2 * dth1 * dth1 - tau
        )
        nom_total = nom1 + nom2
        d_nom_total_th2 = d_nom1_th2 + d_nom2_th2

        # DDTH1_TH2
        d_ddth1_th2 = (d_nom_total_th2 * denom - nom_total * d_denom_th2) / (
            denom * denom
        )

        d_nom1_dth1 = -I2 * (-l1 * lc2 * m2 * 2 * s2 * dth2)
        d_nom2_dth1 = (I2 + l1 * lc2 * m2 * c2) * (l1 * lc2 * m2 * s2 * 2 * dth1)

        # DDTH1_DTH1
        d_ddth1_dth1 = (d_nom1_dth1 + d_nom2_dth1) / denom

        d_nom1_dth2 = -I2 * (-l1 * lc2 * m2 * (2 * dth1 + 2 * dth2) * s2)
        d_nom2_dth2 = 0

        # DDTH1_DTH2
        d_ddth1_dth2 = (d_nom1_dth2 + d_nom2_dth2) / denom

        d_nom1_th1 = (I2 + l1 * lc2 * m2 * c2) * (
            grav * l1 * m2 * c1 + grav * lc1 * m1 * c1 + grav * lc2 * m2 * c12
        )
        d_nom2_th1 = -(grav * lc2 * m2 * c12) * (
            I1 + I2 + l1 * l1 * m2 + 2.0 * l1 * lc2 * m2 * c2
        )

        # DDTH2_TH1
        d_ddth2_th1 = (d_nom1_th1 + d_nom2_th1) / denom

        d_denom_th2 = 2.0 * l1 * l1 * lc2 * lc2 * m2 * m2 * c2 * s2
        A1 = I2 + l1 * lc2 * m2 * c2
        B1 = (
            grav * l1 * m2 * s1
            + grav * lc1 * m1 * s1
            + grav * lc2 * m2 * s12
            - l1 * lc2 * m2 * (2.0 * dth1 + dth2) * s2 * dth2
        )
        dA1_th2 = -l1 * lc2 * m2 * s2
        dB1_th2 = (
            grav * lc2 * m2 * c12 - l1 * lc2 * m2 * (2.0 * dth1 + dth2) * c2 * dth2
        )
        d_nom1_th2 = dA1_th2 * B1 + A1 * dB1_th2
        A2 = -(grav * lc2 * m2 * s12 + l1 * lc2 * m2 * s2 * dth2 * dth2 - tau)
        B2 = I1 + I2 + l1 * l1 * m2 + 2.0 * l1 * lc2 * m2 * c2
        dA2_th2 = -(grav * lc2 * m2 * c12 + l1 * lc2 * m2 * c2 * dth2 * dth2)
        dB2_th2 = -2.0 * l1 * lc2 * m2 * s2
        d_nom2_th2 = dA2_th2 * B2 + A2 * dB2_th2
        nom1 = (I2 + l1 * lc2 * m2 * c2) * (
            grav * l1 * m2 * s1
            + grav * lc1 * m1 * s1
            + grav * lc2 * m2 * s12
            - l1 * lc2 * m2 * (2.0 * dth1 + dth2) * s2 * dth2
        )
        nom2 = -(grav * lc2 * m2 * s12 + l1 * lc2 * m2 * s2 * dth2 * dth2 - tau) * (
            I1 + I2 + l1 * l1 * m2 + 2.0 * l1 * lc2 * m2 * c2
        )
        nom_total_2 = nom1 + nom2
        d_nom_total_2_th2 = d_nom1_th2 + d_nom2_th2

        # DDTH2_TH2
        d_ddth2_th2 = (d_nom_total_2_th2 * denom - nom_total_2 * d_denom_th2) / (
            denom * denom
        )

        d_nom1_dth1 = (I2 + l1 * lc2 * m2 * c2) * (-l1 * lc2 * m2 * 2.0 * s2 * dth2)
        d_nom2_dth1 = 0.0

        # DDTH2_DTH2
        d_ddth2_dth1 = (d_nom1_dth1 + d_nom2_dth1) / denom

        d_nom1_dth2 = (I2 + l1 * lc2 * m2 * c2) * (
            -l1 * lc2 * m2 * s2 * (2.0 * dth1 + 2.0 * dth2)
        )
        d_nom2_dth2 = -(l1 * lc2 * m2 * s2 * 2.0 * dth2) * (
            I1 + I2 + l1 * l1 * m2 + 2.0 * l1 * lc2 * m2 * c2
        )

        # DDTH2_DTH2
        d_ddth2_dth2 = (d_nom1_dth2 + d_nom2_dth2) / denom

        A = torch.zeros((nB, self.nx, self.nx), device=x.device)
        A[:, 0, :] = torch.tensor([0.0, 0.0, 1.0, 0.0])
        A[:, 1, :] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        A[:, 2, 0] = d_ddth1_th1.squeeze(1)
        A[:, 2, 1] = d_ddth1_th2.squeeze(1)
        A[:, 2, 2] = d_ddth1_dth1.squeeze(1)
        A[:, 2, 3] = d_ddth1_dth2.squeeze(1)
        A[:, 3, 0] = d_ddth2_th1.squeeze(1)
        A[:, 3, 1] = d_ddth2_th2.squeeze(1)
        A[:, 3, 2] = d_ddth2_dth1.squeeze(1)
        A[:, 3, 3] = d_ddth2_dth2.squeeze(1)

        return A

    def fcu(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        dfc/du matrix: n_x x n_u
        """
        nB = x.shape[0]

        m2 = self.m2
        l1 = self.l1
        lc2 = self.lc2
        I1 = self.I1
        I2 = self.I2

        th2 = x[:, 1:2]
        tau = u[:, 0:1]

        c2 = cos(th2)

        denom = I1 * I2 + I2 * l1 * l1 * m2 - l1 * l1 * lc2 * lc2 * m2 * m2 * c2 * c2
        d_nom2_1_tau = -(I2 + l1 * lc2 * m2 * c2)
        d_ddth1_tau = d_nom2_1_tau / denom

        d_nom2_2_tau = I1 + I2 + l1 * l1 * m2 + 2.0 * l1 * lc2 * m2 * c2
        d_ddth2_tau = d_nom2_2_tau / denom

        B = torch.zeros((nB, self.nx, self.nu), device=x.device)
        B[:, 2, :] = d_ddth1_tau
        B[:, 3, :] = d_ddth2_tau

        return B
