"""
Example 1a
Let us improve on example 1 by doing away with the silly looping
"""

import torch
import torch.nn as nn
import numpy as np


class BasicAttentionV(nn.Module):
    def __init__(self, din, dk, dv):
        super().__init__()
        self.din = din
        self.dk = dk
        self.dv = dv
        self.q = nn.Conv2d(din, dk, 1, bias=False)
        self.k = nn.Conv2d(din, dk, 1, bias=False)
        self.v = nn.Conv2d(din, dv, 1, bias=False)

    def forward(self, x, y=None):
        # Self attention has no y input, so get it from x
        y = y or x
        assert y.shape[-2] == x.shape[-2]
        assert y.shape[-1] == x.shape[-1]
        b, _, h, w = x.shape

        Q = self.q(x).view(b, self.dk, h*w).transpose(1, 2)
        K = self.k(x).view(b, self.dk, h*w)
        V = self.v(y).view(b, self.dv, h*w).transpose(1, 2)

        # Calculate the softmaxes with shape (b, h*w, h*w).
        S = torch.matmul(Q, K)
        # Calculate the softmax over the rows and columns
        S = torch.softmax(S, dim=-1)
        # The queries are the rows and the keys are the columns.
        z = torch.matmul(S, V).transpose(1, 2)

        return z.view(b, self.dv, h, w)


if __name__ == '__main__':
    from example1 import BasicAttention
    x = torch.randn(1, 16, 100, 100, device='cuda')
    torch.manual_seed(42)
    att1 = BasicAttentionV(16, 32, 24).cuda()
    torch.manual_seed(42)
    att2 = BasicAttention(16, 32, 24).cuda()
    y1 = att1(x)
    y2 = att2(x)
    print((y1-y2).abs().sum())
