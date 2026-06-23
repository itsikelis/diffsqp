from typing import NamedTuple
import torch


class QpSolution(NamedTuple):
    dx: torch.Tensor
    du: torch.Tensor
    mu: torch.Tensor
    nu: torch.Tensor
    lam: torch.Tensor
