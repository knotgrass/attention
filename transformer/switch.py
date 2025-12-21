import torch
import torch.nn as nn
import torch.nn.functional as F
from attn.attention import MultiheadAttention
# -----------------------------
# Feed-Forward Expert
# -----------------------------
class FFNExpert(nn.Module):
    """
    A standard Feed Forward Neural Network acting as a single 'Expert' E_i(x).
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)


    def forward(self, x: torch.Tensor):
        """
        x: (T_i, d_model)
        T_i = number of tokens routed to this expert

        Math:
            E_i(x) = W2 · σ(W1 · x)
        """
        return self.fc2(F.gelu(self.fc1(x)))


# -----------------------------
# Switch (Top-1) Router
# -----------------------------
class SwitchRouter(nn.Module):
    """
    Router producing top-1 expert assignment
    """
    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.router = nn.Linear(d_model, num_experts)

    def forward(self, x) -> torch.return_types.max:
        """
        x: (B, S, d_model)

        Returns:
            expert_index: (B, S)
            gate_value:   (B, S)

        Math:
            h(x) = W_r x
            p(x) = softmax(h(x))
        """
        # h(x): (B, S, N)
        logits = self.router(x)                  # (batch, seq, num_experts)
        probs = F.softmax(logits, dim=-1)

        # top-1 routing
        gate_value, expert_idx = torch.max(probs, dim=-1)
        return expert_idx, gate_value

# -----------------------------
# Mixture-of-Experts FFN
# -----------------------------
class SwitchFFN(nn.Module):

    def __init__(self, d_model: int, d_ff: int, num_experts: int):
        super().__init__()

        # Router / gate
        self.router = SwitchRouter(d_model, num_experts)

        # Set of experts {E_1, ..., E_N}
        self.experts = nn.ModuleList(
            [FFNExpert(d_model, d_ff) for _ in range(num_experts)]
        )

    def forward(self, x):
        """
        x: (B, S, d_model)

        Output:
            y: (B, S, d_model)

        Math (Switch, top-1):
            y(x) = p_i(x) · E_i(x)
        """

        # --------------------------------------------------
        # Routing
        # --------------------------------------------------
        # expert_index: (B, S)
        # gate_value:   (B, S)
        expert_idx, gate_value = self.router(x)

        # Output buffer
        # y: (B, S, d_model)
        y = torch.zeros_like(x)

        # --------------------------------------------------
        # Dispatch tokens to experts
        # --------------------------------------------------
        for i, expert in enumerate(self.experts):
            # mask: (B, S)
            # True if token is routed to this expert
            token_mask = (expert_idx == i)

            if token_mask.any():
                # x_i: (T_i, d_model)
                x_i = x[token_mask]

                # E_i(x): (T_i, d_model)
                expert_outp = expert(x_i)

                # scatter back
                y[token_mask] = expert_outp

        # --------------------------------------------------
        # Apply gate value
        # --------------------------------------------------
        # gate_value.unsqueeze(-1): (B, S, 1)
        #
        # Math:
        #   y(x) = p_i(x) · E_i(x)
        y = y * gate_value.unsqueeze(-1)

        return y


# -----------------------------
# Switch Transformer Block: Attention + MoE
# -----------------------------
class SwitchTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        head_dim: int,
        num_heads: int,
        num_experts: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model == head_dim * num_heads

        self.attn = MultiheadAttention(d_model, head_dim, num_heads)
        self.switch_ffn = SwitchFFN(d_model, d_ff, num_experts)

        # Pre-norm LayerNorms
        # https://x.com/viplismism/status/2000517100071420358?s=20
        self.attn_ln = nn.LayerNorm(d_model)
        self.ffn_ln = nn.LayerNorm(d_model)

        # Residual dropouts
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_dropout  = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, S, d_model)

        Pre-Norm Math:
            x₁ = x + Attn(LN(x))
            x₂ = x₁ + SwitchFFN(LN(x₁))
        """
        # Self-Attention (pre-norm)
        attn_out = self.attn(self.attn_ln(x))
        x = x + self.attn_dropout(attn_out)

        # Switch FFN (pre-norm)
        ffn_out = self.switch_ffn(self.ffn_ln(x))
        x = x + self.ffn_dropout(ffn_out)
        return x


if __name__ == "__main__":
    x = torch.randn(2, 8, 512)  # (batch, seq, d_model)
    block = SwitchTransformer(
        d_model=512,
        d_ff=2048,
        num_heads=8,
        num_experts=4,
        dropout=0.1
    )
    y = block(x)
    print(y.shape)  # (2, 8, 512)
