import torch
from diffsqp.costs import Cost
from diffsqp.utils.math import mm, mv


class LqrCost(Cost):
    ## Classic Linear Quadratic Regulator (LQR) cost function.
    ##
    ## (x - x_des)^T @ Q @ (x - x_des) + (u - u_des)^T @ R @ (u - u_des)
    ##
    def __init__(
        self,
        Q: torch.Tensor,
        R: torch.Tensor = None,
        x_des: torch.Tensor = None,
        u_des: torch.Tensor = None,
    ) -> None:
        self.nB = Q.shape[0]
        self.nx = Q.shape[1]

        self.Q = Q
        self.x_des = x_des if x_des is not None else torch.zeros(Q.shape[:-1])

        if R is not None:
            self.nu = R.shape[1]
            self.R = R
            self.u_des = u_des if u_des is not None else torch.zeros(R.shape[:-1])
        else:
            self.R = None

    def l(self, x: torch.Tensor, u: torch.Tensor = None):
        # Einstein summation of (x - xd) ^ T @ Q @ (x - xd)
        diff_x = x - self.x_des
        x_term = torch.einsum("...i, ...ij, ...j -> ...", diff_x, self.Q, diff_x)

        if self.R is None:
            return 0.5 * x_term
        else:
            diff_u = u - self.u_des
            u_term = torch.einsum("...i, ...ij, ...j -> ...", diff_u, self.R, diff_u)
            return 0.5 * (x_term + u_term)

    def lx(self, x, u=None):
        """Gradient w.r.t x (B, nx, 1)"""
        return mv(self.Q, x - self.x_des)

    def lu(self, x, u):
        """Gradient w.r.t u (B, nu, 1)"""
        return mv(self.R, u - self.u_des)

    def lxx(self, x, u=None):
        """Hessian w.r.t xx (B, nx, nx)"""
        return self.Q

    def luu(self, x, u):
        """Hessian w.r.t uu (B, nu, nu)"""
        return self.R

    def lux(self, x, u):
        """Hessian w.r.t ux (B, nu, nx)"""
        return torch.zeros(self.nB, self.nu, self.nx, device=x.device, dtype=x.dtype)

    def lxu(self, x, u):
        """Hessian w.r.t xu (B, nx, nu)"""
        return torch.zeros(self.nB, self.nx, self.nu, device=x.device, dtype=x.dtype)
