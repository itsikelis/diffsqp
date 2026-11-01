import torch
from typing import Optional
from abc import ABC, abstractmethod

from diffsqp.utils.math import mm, mv


class GenericConstraint(ABC):
    # g, gx and gu accept stage-major inputs x (S, B, n_x), u (S, B, n_u), so a
    # constraint shared by all stages is evaluated once (see Problem.linearize)
    stage_batched = False

    def __init__(self, n_g, n_x, n_u, lb: torch.Tensor, ub: torch.Tensor):
        self.n_g = n_g
        self.n_x = n_x
        self.n_u = n_u
        self.ub = ub
        self.lb = lb

    @abstractmethod
    def g(self, x: torch.Tensor, u: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def gx(self, x: torch.Tensor, u: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def gu(self, x: torch.Tensor, u: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError
