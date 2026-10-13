import torch


def mm(A, B):
    return torch.einsum("...ij,...jk->...ik", A, B)


def mv(A, b):
    # return (A @ b.unsqueeze(2)).squeeze(2)
    return torch.einsum("...ij,...jk->...i", A, b.unsqueeze(2))


def tran(A):
    return torch.einsum("...ij->...ji", A)


def inf_norm(A):
    return torch.norm(A, p=float("inf"), dim=-1)


def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    """Convert a quaternion into a rotation matrix."""
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    qw2, qx2, qy2, qz2 = qw * qw, qx * qx, qy * qy, qz * qz

    row0 = torch.stack(
        [qw2 + qx2 - qy2 - qz2, 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        dim=-1,
    )
    row1 = torch.stack(
        [2 * (qx * qy + qw * qz), qw2 - qx2 + qy2 - qz2, 2 * (qy * qz - qw * qx)],
        dim=-1,
    )
    row2 = torch.stack(
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), qw2 - qx2 - qy2 + qz2],
        dim=-1,
    )
    return torch.stack([row0, row1, row2], dim=-2)


def q_left(q: torch.Tensor) -> torch.Tensor:
    """Batched Left quaternion product"""
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    row0 = torch.stack([qw, -qx, -qy, -qz], dim=-1)
    row1 = torch.stack([qx, qw, -qz, qy], dim=-1)
    row2 = torch.stack([qy, qz, qw, -qx], dim=-1)
    row3 = torch.stack([qz, -qy, qx, qw], dim=-1)
    return torch.stack([row0, row1, row2, row3], dim=-2)


def bmv(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """A @ x for stacks of small matrices: (..., m, n) x (..., n) -> (..., m).

    On CUDA, batched cuBLAS gemv is several times slower than a broadcast
    multiply-and-sum for tiny matrices, so use the latter there.
    """
    if A.is_cuda and A.shape[-1] * A.shape[-2] <= 64:
        return torch.sum(A * x.unsqueeze(-2), dim=-1)
    return (A @ x.unsqueeze(-1)).squeeze(-1)


def bmtv(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """A^T @ x for stacks of small matrices: (..., m, n) x (..., m) -> (..., n).

    On CUDA, a matmul with the transposed view is as slow as the broadcast
    multiply-and-sum even for 13x13 matrices, and much slower for tiny ones.
    """
    if A.is_cuda:
        return torch.sum(A * x.unsqueeze(-1), dim=-2)
    return (A.mT @ x.unsqueeze(-1)).squeeze(-1)


def bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """A @ B for stacks of small matrices, broadcast-and-sum on CUDA when tiny."""
    if A.is_cuda and A.shape[-2] * A.shape[-1] * B.shape[-1] <= 64:
        return torch.sum(A.unsqueeze(-1) * B.unsqueeze(-3), dim=-2)
    return A @ B


def stage_major(T: torch.Tensor):
    """Split (B, H, ...) into H contiguous (B, ...) tensors.

    Slicing T[:, k] gives strided batches, for which cuBLAS picks slow kernels.
    """
    return T.transpose(0, 1).contiguous().unbind(0)
