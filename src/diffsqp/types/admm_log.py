from typing import NamedTuple, List, Optional
import torch


class AdmmLog(NamedTuple):
    rho: torch.Tensor
    iterations: torch.Tensor

    def __str__(self) -> str:
        return (
            f"=== ADMM Log ===\n"
            f" rho         : {self.rho}\n"
            f" Total Iters : {self.iterations}\n"
            f"======================"
        )
