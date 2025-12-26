import dataclass
import torch


@dataclass
class QPStage:
    A: torch.Tensor  #  A: $n_x \times n_x$
    B: torch.Tensor  # B: $n_x \times n_u$
    gamma: torch.Tensor  # $n_x \times 1$
    Qxx: torch.Tensor
    Qxu: torch.Tensor
    Quu: torch.Tensor
    qx: torch.Tensor  # qx: $n_x \times 1$
    qu: torch.Tensor  # qu: $n_u \times 1$
    D: torch.Tensor  # D: $n_c \times n_x$
    E: torch.Tensor  # E: $n_c \times n_u$
    cl: torch.Tensor  # cl: $n_c \times 1$
    cu: torch.Tensor  # cu: $n_c \times 1$


@dataclass
class Dimensions:
    horizon: int

    # Total values for states, controls and constraints
    dim_state: int
    dim_control: int
    num_constraints: int

    # Total values per stage
    stage_state: torch.Tensor
    stage_control: torch.Tensor
    stage_constraint: torch.Tensor
    nq: torch.Tensor
    nv: torch.Tensor
    njoints: torch.Tensor


@dataclass
class QPData:
    dims: Dimensions
    stages: List[QPStage]
    Dt: torch.Tensor
    Qxx_t: torch.Tensor
    qx_t: torch.Tensor
    cl_t: torch.Tensor
    cu_t: torch.Tensor


def equality_constrained_lqr(**kwargs):
    """
    Batch-friendly Equality Constrained Finite Horizon LQR.

    Shapes:
        A: (n_batch, n, n) or (n, n)
        B: (n_batch, n, m) or (n, m)
        Q, R, Qf: Same logic as A and B
        x0, x_target: (n_batch, n, 1)
    """

    A = kwargs["A"]
    B = kwargs["B"]
    x0 = kwargs["x0"]
    # Qx, Qu, Qxx, Quu, R, Qf, x0, x_target, ns
    # Ensure all matrices have a batch dimension
    n_batch = x0.shape[0]

    def ensure_batch(t):
        return t.expand(n_batch, -1, -1) if t.dim() == 2 else t

    A, B, Q, R, Qf = map(ensure_batch, [A, B, Q, R, Qf])

    n, m = B.shape[1], B.shape[2]
    s = torch.zeros((n_batch, (ns - 1) * n, 1))

    V = [None] * ns
    u = [None] * ns
    h = [None] * (ns - 1)
    H = [None] * (ns - 1)

    V[-1] = Vxx_f
    u[-1] = qx_f

    # 1. Backward Pass (Vectorized over Batch)
    for k in range(ns - 1, -1):
        pass

    # 2. Forward Pass (Batch Simulation)

    return torch.stack(states, dim=1), torch.stack(inputs, dim=1)
