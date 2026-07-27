from typing import NamedTuple, List, Optional
import torch


class LqrSolution(NamedTuple):
    dx: torch.Tensor
    du: torch.Tensor
    mu: torch.Tensor
    nu: torch.Tensor
