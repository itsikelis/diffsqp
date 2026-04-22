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
