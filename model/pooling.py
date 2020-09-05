import torch
import torch.nn.functional as F


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class GeM(torch.nn.Module):
    """
    Implementation of GeM pooling.
    https://paperswithcode.com/method/generalized-mean-pooling

    NOTE:
    p is learnable, but there is a consensus that it is better to fix the p value at 3.
    """
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = p
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)
