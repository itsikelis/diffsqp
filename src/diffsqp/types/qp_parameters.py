from typing import NamedTuple
import torch


class QpParameters(NamedTuple):
    Q: torch.Tensor
    q: torch.Tensor
    R: torch.Tensor
    r: torch.Tensor
    S: torch.Tensor
    A: torch.Tensor
    B: torch.Tensor
    b: torch.Tensor
    C: torch.Tensor
    D: torch.Tensor
    d: torch.Tensor
