import torch
from abc import ABC, abstractmethod


class Dynamics(ABC):
    """
    Abstract Base Class for robotic and physical system dynamics.
    Expects states 'x' of shape (N, state_dim) and controls 'u' of shape (N, control_dim).
    """

    def f(self, x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Discrete-time dynamics: x_k+1 = f(x_k, u_k)
        """
        x_dot = self.fc(x, u)
        return x + dt * self.fc(x, u)

    def fx(self, x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
        """
        State Jacobian: df/dx
        """
        n_state = x.shape[1]
        return torch.add(dt * self.fcx(x, u), torch.eye(n_state))

    def fu(self, x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Control Jacobian: df/du
        """
        n_state = x.shape[1]
        return dt * self.fcu(x, u)

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
