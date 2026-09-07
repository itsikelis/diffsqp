import torch


class Dynamics:
    def __init__(self, nx, nu, nq, nv, use_semi_implicit=True):
        self.nx = nx
        self.nu = nu
        self.nq = nq
        self.nv = nv

        self.use_semi_implicit = use_semi_implicit

    def f(self, x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Discrete-time dynamics: x_k+1 = f(x_k, u_k)
        """
        x_dot = self.fc(x, u)

        if not self.use_semi_implicit:
            return x + dt * self.fc(x, u)

        E = self.calc_semi_impl_matrix_(dt)
        return x + torch.einsum("ij,...j->...i", E, x_dot)

    def fx(self, x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
        """
        State Jacobian: df/dx
        """

        if not self.use_semi_implicit:
            return torch.eye(self.nx) + dt * self.fcx(x, u)

        E = self.calc_semi_impl_matrix_(dt)

        fcx_val = self.fcx(x, u)
        e_fcx = torch.einsum("ij,...jk->...ik", E, fcx_val)

        return e_fcx + torch.eye(self.nx)

    def fu(self, x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Control Jacobian: df/du
        """

        if not self.use_semi_implicit:
            return dt * self.fcu(x, u)

        E = self.calc_semi_impl_matrix_(dt)

        fcu_val = self.fcu(x, u)
        return torch.einsum("ij,...jk->...ik", E, fcu_val)

    def fc(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Continuous time dynamics: x_dot = fc(x_k, u_k)
        """
        q_dot = x[..., self.nq :]
        q_ddot = u[..., :]

        return torch.cat([q_dot, q_ddot], dim=-1)

    def fcx(self, x, u) -> torch.Tensor:
        """
        Gradient of continuous time dynamics w.r.t x
        """
        n_B = x.shape[:-1]
        # TODO: Remove batch dimension from grad
        grad = torch.zeros(*n_B, self.nx, self.nx)
        grad[..., 0 : self.nv, self.nv :] = torch.eye(self.nv)
        return grad

    def fcu(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Gradient of continuous time dynamics w.r.t u
        """
        n_B = x.shape[:-1]
        # TODO: Remove batch dimension from grad
        grad = torch.zeros(*n_B, self.nx, self.nu)
        grad[..., self.nv :, :] = torch.eye(self.nu)
        return grad

    def calc_semi_impl_matrix_(self, dt):
        E = torch.zeros((self.nx, self.nx))
        E[0 : self.nq, 0 : self.nq] = torch.eye(self.nq) * dt
        E[0 : self.nq, self.nq :] = torch.eye(self.nq) * dt * dt
        E[self.nq :, self.nq :] = torch.eye(self.nq) * dt
        return E
