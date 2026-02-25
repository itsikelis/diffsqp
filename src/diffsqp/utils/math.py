import torch


def mm(A, B):
    return torch.bmm(A, B)


def mv(A, b):
    return torch.bmm(A, b.unsqueeze(2)).squeeze(2)
