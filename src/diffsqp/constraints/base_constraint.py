import torch
from abc import ABC, abstractmethod

from diffsqp.utils.math import mm, mv


class Constraint(ABC):
    def __init__(self, ng, nx, nu, type="equality"):
        self.type = type
        self.ng = ng
        self.nx = nx
        self.nu = nu

    @abstractmethod
    def g(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def gx(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def gu(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
