from dataclasses import dataclass
from typing import List, Optional
import torch


@dataclass
class AdmmSolution:
    dx: torch.Tensor
    du: torch.Tensor
    mu: torch.Tensor
    nu: torch.Tensor
    z: List[Optional[torch.Tensor]]
    ksi: List[Optional[torch.Tensor]]
    rho: List[Optional[torch.Tensor]]
    rho_inv: List[Optional[torch.Tensor]]
    rho_common: Optional[torch.Tensor]
