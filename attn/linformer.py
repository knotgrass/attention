from typing import Optional
from torch import nn, Tensor
import torch.nn.functional as F
from .attention import Attention


class LinearSelfAttention(Attention):
    r"""
    https://arxiv.org/abs/2006.04768
    """
    def __init__(self, word_size: int = 512, embed_dim: int = 64,
                 n:int=..., k:Optional[int] = None) -> None:
        super().__init__(word_size, embed_dim)
        if k is None:
            k = n // 4
        self.k = k
        self.n = n # sequence length
        self.prof_E = nn.Linear(in_features=n, out_features=k, bias=True)
        self.prof_F = nn.Linear(in_features=n, out_features=k, bias=True)

    def forward(self, x:Tensor) -> Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Handle a smaller dimension than expected
        padding = 0
        if q.shape[1] < self.n:
            padding = self.n - q.shape[1]
            pad_dims = (0, 0, 0, padding)
            q = F.pad(q, pad_dims)
            k = F.pad(k, pad_dims)
            v = F.pad(v, pad_dims)

        k_projected = self.prof_E(k.transpose(-2, -1)).transpose(-2, -1)
        v_projected = self.prof_F(v.transpose(-2, -1)).transpose(-2, -1)

        z = self.self_attention(q, k_projected, v_projected)
        # z = F.scaled_dot_product_attention(q, k_projected, v_projected)
        return z[:, :-padding, :] if padding > 0 else z
