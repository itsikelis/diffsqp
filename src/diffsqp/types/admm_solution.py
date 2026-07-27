from typing import NamedTuple, List, Optional
import torch


class AdmmSolution(NamedTuple):
    dx: torch.Tensor
    du: torch.Tensor
    mu: torch.Tensor
    nu: torch.Tensor
    z: List[Optional[torch.Tensor]]
    ksi: List[Optional[torch.Tensor]]
