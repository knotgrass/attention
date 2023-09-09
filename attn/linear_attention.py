import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional
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


class Projection(nn.Module):
    def __init__(self, n:int, k:int) -> None:
        self.k = k
        self.n = n # sequence length
        self.proj = nn.Linear(in_features=n, out_features=k)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class LinearAttention(Attention):
    r"""
    https://arxiv.org/abs/2006.04768
    """
    def __init__(self, word_size: int = 512, embed_dim: int = 64,
                 proj_E:Projection=..., proj_F:Projection=...) -> None:
        super().__init__(word_size, embed_dim)
        assert proj_E.n == proj_F.n
        assert proj_E.k == proj_F.k

        self.k = proj_F.k
        self.n = proj_F.n # sequence length
        self.proj_E = proj_E
        self.proj_F = proj_F

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

        k_projected = self.proj_E(k.transpose(-2, -1)).transpose(-2, -1)
        v_projected = self.proj_F(v.transpose(-2, -1)).transpose(-2, -1)

        z = self.self_attention(q, k_projected, v_projected)
        # z = F.scaled_dot_product_attention(q, k_projected, v_projected)
        return z[:, :-padding, :] if padding > 0 else z


class MultiheadLinearAttention(nn.Module):
    def __init__(self, word_size: int = 512, embed_dim: int = 64, n_head:int=12,
                 proj_E:Projection=..., proj_F:Projection=...,
                 sharing:str='not-share') -> None:
        assert sharing in ('not-share', 'headwise', 'key-value', 'layerwise')
        assert proj_E.n == proj_F.n
        assert proj_E.k == proj_F.k

        super().__init__()
        self.k = proj_E.k
        self.n = proj_E.n # sequence length
        self.sharing = sharing

        self.n_head = n_head
        self.embed_dim = embed_dim
        self.dim_K = torch.tensor(embed_dim)
        self.proj = nn.Parameter(torch.empty(embed_dim * n_head, embed_dim))
        nn.init.xavier_uniform_(self.proj)

        if sharing == 'not-share':
            self.proj_E = None
            self.proj_F = None
            self.multihead = nn.ModuleList([
                LinearAttention(word_size, embed_dim,
                                Projection(self.n, self.k),
                                Projection(self.n, self.k)) for _ in range(n_head)
            ])

        elif sharing == 'headwise':
            r"""
            Headwise sharing: for each layer, we share two projection matrices E and F such that
            Ei = E and Fi = F across all heads i.
            """
            self.proj_E = proj_E
            self.proj_F = proj_F
            self.multihead = nn.ModuleList([
                LinearAttention(word_size, embed_dim, proj_E, proj_F) for _ in range(n_head)
            ])

        elif sharing == 'key-value':
            r"""
            Key-value sharing: we do headwise sharing, with the additional constraint of sharing the
            key and value projections. For each layer, we create a single projection matrix E such that
            Ei = Fi = E for each key-value projection matrix across all head i.
            """
            self.proj_E = proj_E
            self.proj_F = None
            self.multihead = nn.ModuleList([
                LinearAttention(word_size, embed_dim, proj_E, proj_E) for _ in range(n_head)
            ])

        elif sharing == 'layerwise':
            r"""
            Layerwise sharing: we use a single projection matrix E across all layers, for all heads, and
            for both key and value.
            """
            self.proj_E = proj_E
            self.proj_F = None
            self.multihead = nn.ModuleList([
                LinearAttention(word_size, embed_dim, proj_E, proj_E) for _ in range(n_head)
            ])

    def forward(self, x: Tensor) -> Tensor:
        Z_s = torch.cat([head(x) for head in self.multihead], dim=1)
        Z = torch.matmul(Z_s, self.proj)
        return Z
