import torch
from torch import nn, Tensor
import torch.nn.functional as F

from os.path import join, realpath
from sys import path
path.append(realpath(join(__file__, '..', '..')))
from attn.linear_attention import (LinearSelfAttention, Projection,
                                   LinearAttention, MultiheadLinearAttention)
del join, realpath


def test_forward_LinearSelfAttention():
    attention = LinearSelfAttention(word_size=512, embed_dim=64,
                                    n=3, k=2)

    inp = torch.cat([torch.randn(1, 512),
                     torch.randn(1, 512),
                     torch.randn(1, 512)], dim=0)

    out = attention(inp)
    print(out.shape) #torch.Size([3, 64])


def test_forward_LinearAttention():
    for n in 3, 4:
        proj_E = Projection(n=n, k=2)
        proj_F = Projection(n=n, k=2)
        attention = LinearAttention(word_size=512, embed_dim=64,
                                    proj_E=proj_E, proj_F=proj_F)

        inp = torch.cat([torch.randn(1, 512),
                        torch.randn(1, 512),
                        torch.randn(1, 512)], dim=0)

        out = attention(inp)
        
        print(f"n= {n} |", f"output.shape {out.shape}") #torch.Size([3, 64])
        assert inp.shape[0] == out.shape[0]

def test_MultiheadLinearAttention():
    proj_E = Projection(8, 4)
    proj_F = Projection(8, 4)
    attention = MultiheadLinearAttention(
        word_size= 512, embed_dim= 64,
        proj_E= proj_E, proj_F= proj_F,
        sharing= 'not-share')
        # sharing= 'headwise')
        # sharing= 'key-value')
        # sharing= 'layerwise')

    inp = torch.cat([torch.randn(1, 512),
                     torch.randn(1, 512),
                     torch.randn(1, 512)], dim=0)

    out = attention(inp)
    print(out.shape) #torch.Size([3, 64])

if __name__ == '__main__':
    test_forward_LinearAttention()
