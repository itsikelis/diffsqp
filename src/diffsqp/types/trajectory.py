from typing import NamedTuple
import torch


# Store current trajecotry
class Trajectory(NamedTuple):
    x: torch.Tensor
    u: torch.Tensor
    mu: torch.Tensor
    nu: torch.Tensor
    lam: torch.Tensor | None
