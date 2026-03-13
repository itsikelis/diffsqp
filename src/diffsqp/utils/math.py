import torch


def mm(A, B):
    if len(A.shape) == 2:
        return A @ B
    else:
        return torch.bmm(A, B)


def mv(A, b):
    if len(A.shape) == 2:
        return (A @ b.unsqueeze(2)).squeeze(2)
    else:
        return torch.bmm(A, b.unsqueeze(2)).squeeze(2)
