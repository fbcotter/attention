"""
Example 1
Let us build a simple attention layer, ignoring all the problems with it being too expensive to do on
images. We will also not build a multi-headed layer just yet, or worry about positional encoding.
"""

import torch
import torch.nn as nn
import numpy as np


class BasicAttention(nn.Module):
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

        Q = self.q(x)
        K = self.k(x)
        V = self.v(y)

        # Let us naively implement equation 2 from https://arxiv.org/pdf/1906.05909.pdf using a neighbourhood that is
        # the entire image region
        Sab = torch.zeros(b, h, w, device=x.device)
        z = torch.zeros(b, self.dv, h, w, device=x.device)
        for i in range(h):
            for j in range(w):
                # Take Qij -> shape (N, Dk) and K -> shape (N, Dk, H, W) and multiply over the Dk dimensions to get (N, H, W)
                Sab = torch.einsum('bk,bkhw->bhw', Q[..., i, j], K)

                # Take the spatial softmax over the last 2 dimensions
                Sab = torch.softmax(Sab.view(b, h*w), dim=-1).view(b, h, w)

                # Take Sab -> shape (N, H, W) and V -> shape(N, Dv, H, W) and multiply and sum over the the spatial dimensions
                z[..., i, j] = torch.einsum('bhw,bvhw->bv', Sab, V)

        return z


if __name__ == '__main__':
    x = torch.randn(1, 16, 100, 100, device='cuda')
    # Make the dk and dv different for funsies
    att = BasicAttention(16, 32, 24).cuda()
    y = att(x)
    print(y.shape)
