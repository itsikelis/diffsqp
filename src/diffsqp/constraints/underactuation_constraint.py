import torch
from abc import ABC, abstractmethod

from diffsqp.utils.math import mm, mv


class UnderactuationConstraint(ABC):
    def __init__(self, n_h, n_x, n_u):
        self.type = "equality"
        self.n_h = n_h
        self.n_x = n_x
        self.n_u = n_u

    @abstractmethod
    def h(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def hx(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def hu(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
