from typing import NamedTuple, List, Optional
import torch


class AdmmLog(NamedTuple):
    iterations: torch.Tensor

    def __str__(self) -> str:
        return (
            f"=== ADMM Log ===\n"
            f" Total Iters : {self.iterations}\n"
            f"======================"
        )
