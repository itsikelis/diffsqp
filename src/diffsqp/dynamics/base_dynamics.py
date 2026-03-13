import torch

from diffsqp.utils.math import mm, mv


class Dynamics:
    def __init__(self, nx, nu, nq, nv):
        self.nx = nx
        self.nu = nu
        self.nq = nq
        self.nv = nv

    def f(self, x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Discrete-time dynamics: x_k+1 = f(x_k, u_k)
        """
        x_dot = self.fc(x, u).clone()
        # return x + dt * self.fc(x, u)

        nB = x.shape[0]
        E = self.calc_semi_impl_matrix_(dt)
        return x + mv(E, x_dot)

    def fx(self, x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
        """
        State Jacobian: df/dx
        """
        # return torch.add(dt * self.fcx(x, u), torch.eye(self.nx))

        nB = x.shape[0]
        E = self.calc_semi_impl_matrix_(dt)
        return torch.add(mm(E, self.fcx(x, u)), torch.eye(self.nx))

    def fu(self, x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Control Jacobian: df/du
        """
        # return dt * self.fcu(x, u)

        nB = x.shape[0]
        E = self.calc_semi_impl_matrix_(dt)
        return mm(E, self.fcu(x, u))

    def fc(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Continuous time dynamics: x_dot = fc(x_k, u_k)
        """
        q_dot = x[:, self.nq :]
        q_ddot = u[:, :]

        return torch.cat([q_dot, q_ddot], dim=1)

    def fcx(self, x, u) -> torch.Tensor:
        """
        Gradient of continuous time dynamics w.r.t x
        """
        n_batch = x.shape[0]
        # TODO: Remove batch dimension from grad
        grad = torch.zeros(self.nx, self.nx).repeat(n_batch, 1, 1)
        grad[:, 0 : self.nv, self.nv :] = torch.eye(self.nv)
        return grad

    def fcu(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Gradient of continuous time dynamics w.r.t u
        """
        n_batch = x.shape[0]
        # TODO: Remove batch dimension from grad
        grad = torch.zeros(self.nx, self.nu).repeat(n_batch, 1, 1)
        grad[:, self.nv :, :] = torch.eye(self.nu)
        return grad

    def calc_semi_impl_matrix_(self, dt):
        E = torch.zeros((self.nx, self.nx))
        E[0 : self.nq, 0 : self.nq] = torch.eye(self.nq) * dt
        E[0 : self.nq, self.nq :] = torch.eye(self.nq) * dt * dt
        E[self.nq :, self.nq :] = torch.eye(self.nq) * dt
        return E
