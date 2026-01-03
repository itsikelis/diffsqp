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
        raise NotImplementedError

    @abstractmethod
    def lx(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Cost gradient w.r.t x
        """
        raise NotImplementedError

    @abstractmethod
    def lu(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Cost gradient w.r.t u
        """
        raise NotImplementedError

    @abstractmethod
    def lxx(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Cost Hessian w.r.t xx
        """
        raise NotImplementedError

    @abstractmethod
    def luu(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Cost Hessian w.r.t uu
        """
        raise NotImplementedError

    @abstractmethod
    def lux(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Cost Hessian w.r.t ux
        """
        raise NotImplementedError

    @abstractmethod
    def lxu(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Cost Hessian w.r.t xu
        """
        raise NotImplementedError
