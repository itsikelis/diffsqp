import torch
from abc import ABC, abstractmethod


class Cost(ABC):
    """
    Abstract Base Class for an SQP cost.
    """

    @abstractmethod
    def l(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Cost
        """
        pass

    @abstractmethod
    def lx(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Cost gradient w.r.t x
        """
        pass

    @abstractmethod
    def lu(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Cost gradient w.r.t u
        """
        pass

    @abstractmethod
    def lxx(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Cost Hessian w.r.t xx
        """
        pass

    @abstractmethod
    def luu(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Cost Hessian w.r.t uu
        """
        pass

    @abstractmethod
    def lux(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Cost Hessian w.r.t xu
        """
        pass
