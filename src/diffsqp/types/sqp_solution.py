from typing import NamedTuple, Tuple, List, Optional
import torch


# Store current sqp result
class SqpSolution(NamedTuple):
    x: torch.Tensor
    u: torch.Tensor
    mu: torch.Tensor
    nu: torch.Tensor
    ksi: List[Optional[torch.Tensor]]
