from typing import NamedTuple, Optional
import torch


class AdmmLog(NamedTuple):
    iterations: int
    # (B,) True where the LQR factorization failed, even after regularization
    failed: Optional[torch.Tensor] = None
    # Last LQR factorization, to resume ADMM without refactorizing
    factorization: Optional[object] = None

    def __str__(self) -> str:
        return (
            f"=== ADMM Log ===\n"
            f" Total Iters : {self.iterations}\n"
            f"======================"
        )
