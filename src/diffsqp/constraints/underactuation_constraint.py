import torch
from abc import ABC, abstractmethod

from diffsqp.utils.math import mm, mv


class UnderactuationConstraint(ABC):
    def __init__(self, ng, nx, nu):
        self.type = "equality"
        self.ng = ng
        self.nx = nx
        self.nu = nu

    @abstractmethod
    def h(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def hx(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def hu(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
