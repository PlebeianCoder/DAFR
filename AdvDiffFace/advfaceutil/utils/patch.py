__all__ = ["patch_torch"]

from typing import Iterable

import torch


def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, Iterable):
        for elem in x:
            zero_gradients(elem)


def patch_torch():
    import torch.autograd.gradcheck

    setattr(torch.autograd.gradcheck, "zero_gradients", zero_gradients)
