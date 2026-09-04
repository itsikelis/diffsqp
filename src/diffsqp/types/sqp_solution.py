from dataclasses import dataclass
from typing import Tuple, List, Optional
import torch


# Store current sqp result
@dataclass
class SqpSolution:
    x: torch.Tensor
    u: torch.Tensor
    mu: torch.Tensor
    nu: torch.Tensor
    ksi: List[Optional[torch.Tensor]]
