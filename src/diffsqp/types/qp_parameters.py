from typing import NamedTuple, Optional
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
    # Stage constraints lb <= M dx + N du + n <= ub, padded to the largest stage
    # dimension G: M (B, H, G, n_x), N (B, H-1, G, n_u), n (B, H, G) and
    # lb, ub (H, G). Padded rows are zero with +-PAD_BOUND bounds.
    M: Optional[torch.Tensor]
    N: Optional[torch.Tensor]
    n: Optional[torch.Tensor]
    lb: Optional[torch.Tensor] = None
    ub: Optional[torch.Tensor] = None
