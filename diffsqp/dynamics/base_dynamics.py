import torch
from abc import ABC, abstractmethod

from diffsqp.utils.math import mm, mv


class Dynamics(ABC):
    """
    Abstract Base Class for robotic and physical system dynamics.
    Expects states 'x' of shape (N, state_dim) and controls 'u' of shape (N, control_dim).
    """

    def f(self, x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Discrete-time dynamics: x_k+1 = f(x_k, u_k)
        """
        x_dot = self.fc(x, u).clone()

        # return x + dt * self.fc(x, u)

        nB = x.shape[0]
        E = self.calc_semi_impl_matrix_(dt).repeat(nB, 1, 1)
        return x + mv(E, x_dot)

    def fx(self, x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
        """
        State Jacobian: df/dx
        """
        # return torch.add(dt * self.fcx(x, u), torch.eye(self.nx))

        nB = x.shape[0]
        E = self.calc_semi_impl_matrix_(dt).repeat(nB, 1, 1)
        return torch.add(mm(E, self.fcx(x, u).clone()), torch.eye(self.nx))

    def fu(self, x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Control Jacobian: df/du
        """
        # return dt * self.fcu(x, u)

        nB = x.shape[0]
        E = self.calc_semi_impl_matrix_(dt).repeat(nB, 1, 1)
        return mm(E, self.fcu(x, u).clone())

    @abstractmethod
    def fc(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Continuous time dynamics: x_dot = fc(x_k, u_k)
        """
        raise NotImplementedError

    @abstractmethod
    def fcx(self, x, u) -> torch.Tensor:
        """
        Derivative of continuous time dynamics w.r.t x
        """
        raise NotImplementedError

    @abstractmethod
    def fcu(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Derivative of continuous time dynamics w.r.t u
        """
        raise NotImplementedError

    def calc_semi_impl_matrix_(self, dt):
        E = torch.zeros((self.nx, self.nx))
        E[0 : self.nq, 0 : self.nq] = torch.eye(self.nq) * dt
        E[0 : self.nq, self.nq :] = torch.eye(self.nq) * dt * dt
        E[self.nq :, self.nq :] = torch.eye(self.nq) * dt
        return E
