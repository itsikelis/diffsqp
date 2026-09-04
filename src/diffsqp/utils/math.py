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
