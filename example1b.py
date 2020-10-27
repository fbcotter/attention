"""
Example 1b
Let us add multi headed attention
"""

import torch
import torch.nn as nn
import numpy as np


class MultiHeadedAttention(nn.Module):
    def __init__(self, din, dk, dv, n_heads=1):
        super().__init__()
        assert dk % n_heads == 0
        assert dv % n_heads == 0
        self.din = din
        self.dk = dk
        self.dv = dv
        self.n_heads = n_heads

        self.q = nn.Conv2d(din, dk, 1, bias=False)
        self.k = nn.Conv2d(din, dk, 1, bias=False)
        self.v = nn.Conv2d(din, dv, 1, bias=False)

    def forward(self, x, y=None):
        # Self attention has no y input, so get it from x
        y = y or x
        assert y.shape[-2] == x.shape[-2]
        assert y.shape[-1] == x.shape[-1]
        b, _, h, w = x.shape

        Q = self.q(x).view(b, self.n_heads, self.dk//self.n_heads, h*w).transpose(2, 3)
        K = self.k(x).view(b, self.n_heads, self.dk//self.n_heads, h*w)
        V = self.v(y).view(b, self.n_heads, self.dv//self.n_heads, h*w).transpose(2, 3)

        # Calculate the softmaxes with shape (b, n, h*w, h*w).
        S = torch.matmul(Q, K)

        S = torch.softmax(S, dim=-1)

        # V has shape (b, n, v, l), transpose it to shape (b, n, l, v) then take S*V
        z = torch.matmul(S, V).transpose(2, 3)

        return z.reshape(b, self.dv, h, w)


if __name__ == '__main__':
    from example1a import BasicAttentionV
    x = torch.randn(1, 16, 100, 100, device='cuda')
    att1 = BasicAttentionV(16, 32, 32).cuda()
    att2 = MultiHeadedAttention(16, 64, 64, n_heads=2).cuda()
    # Copy over the first group's weights
    att2.q.weight.data[:32] = att1.q.weight.data
    att2.k.weight.data[:32] = att1.k.weight.data
    att2.v.weight.data[:32] = att1.v.weight.data
    y1 = att1(x)
    y2 = att2(x)
    print((y1-y2[:, :32]).abs().sum())
