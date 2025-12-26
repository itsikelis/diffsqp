import torch
from abc import ABC, abstractmethod


class Dynamics(ABC):
    """
    Abstract Base Class for robotic and physical system dynamics.
    Expects states 'x' of shape (N, state_dim) and controls 'u' of shape (N, control_dim).
    """

    @abstractmethod
    def f(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Continuous-time dynamics: x_dot = f(x, u)
        """
        pass

    def step(self, x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Discrete-time step using RK4 Integration.
        x_next = x + (dt/6) * (k1 + 2k2 + 2k3 + k4)
        """
        k1 = self.f(x, u)
        k2 = self.f(x + 0.5 * dt * k1, u)
        k3 = self.f(x + 0.5 * dt * k2, u)
        k4 = self.f(x + dt * k3, u)

        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    @abstractmethod
    def fx(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        State Jacobian: df/dx
        """
        pass

    @abstractmethod
    def fu(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Control Jacobian: df/du
        """
        pass
