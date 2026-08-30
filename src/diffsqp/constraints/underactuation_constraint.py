import torch
from typing import Optional
from abc import ABC, abstractmethod
from diffsqp.constraints import GenericConstraint

from diffsqp.utils.math import mm, mv


class UnderactuationConstraint(GenericConstraint):
    def __init__(self, n_h, n_x, n_u, tol=1e-4):
        lb = torch.full((n_h,), -tol)
        ub = torch.full((n_h,), tol)
        super().__init__(n_g=n_h, n_x=n_x, n_u=n_u, lb=lb, ub=ub, is_equality=True)

        self.n_h = n_h
        self.n_x = n_x
        self.n_u = n_u

    def g(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.h(x, u)

    def gx(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.hx(x, u)

    def gu(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.hu(x, u)

    @abstractmethod
    def h(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def hx(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def hu(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
