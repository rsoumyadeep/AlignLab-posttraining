"""Feed-forward networks: the classic MLP and SwiGLU.

WHY A FEED-FORWARD BLOCK AT ALL. Attention MIXES information across positions,
but it is (given the weights) a linear combination of values - it has no
position-wise nonlinearity of its own. The FFN provides that: it processes each
position INDEPENDENTLY and gives the block its representational depth. The
usual division of labour is "attention routes information, the FFN processes
it", and in most Transformers the FFN holds roughly two thirds of the
parameters.

    classic   FFN(x) = W_2 · act(W_1 x + b_1) + b_2         d_ff = 4 * d_model
    SwiGLU    FFN(x) = W_down (Swish(W_gate x) * W_up x)    d_ff = 8/3 * d_model
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """The original Transformer FFN: expand, activate, project back.

    d_ff = 4 * d_model is convention from the 2017 paper, not derivation.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        activation: str = "gelu",
        bias: bool = True,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else 4 * d_model

        self.w_1 = nn.Linear(d_model, self.d_ff, bias=bias)
        self.w_2 = nn.Linear(self.d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout_p)

        acts = {"relu": F.relu, "gelu": F.gelu, "silu": F.silu}
        if activation not in acts:
            raise ValueError(f"unknown activation {activation!r}, expected {list(acts)}")
        self.activation_name = activation
        self.activation = acts[activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, d_model] -> [B, T, d_ff] -> [B, T, d_model]."""
        return self.dropout(self.w_2(self.activation(self.w_1(x))))

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_ff={self.d_ff}, act={self.activation_name}"


class SwiGLU(nn.Module):
    """SwiGLU feed-forward. Used by Llama, Qwen, Mistral, PaLM.

        SwiGLU(x) = W_down ( Swish(W_gate x)  ⊙  (W_up x) )
        Swish(z)  = z · sigmoid(z)                    (a.k.a. SiLU)

    TWO IDEAS COMBINED.

    1. Swish/SiLU instead of ReLU. ReLU is exactly zero for all negative
       inputs, so those units receive no gradient - the "dying ReLU" problem.
       Swish is smooth and only ASYMPTOTICALLY zero, so gradient still flows
       for negative inputs. It is also non-monotonic (it dips slightly below
       zero near z ≈ -1.28), which appears to help.

    2. A GATING branch. Rather than one projection through a nonlinearity,
       compute TWO projections and multiply them elementwise. One branch
       (gated by Swish) acts as a learned, input-dependent filter on the other.
       This is multiplicative rather than additive interaction, which a plain
       MLP cannot express at the same width.

    WHY THREE MATRICES AND THE 2/3 WIDTH. The gating costs an extra projection:
    W_gate, W_up, W_down instead of W_1, W_2. To keep the parameter count
    comparable to a classic FFN at d_ff = 4·d_model, the hidden width is scaled
    by 2/3:

        classic: 2 · d_model · (4·d_model)         = 8 · d_model²
        SwiGLU:  3 · d_model · (8/3 · d_model)     = 8 · d_model²

    So "8/3 · d_model" is not mystical - it is exactly the width that makes the
    three-matrix gated form cost the same as the two-matrix form. Real models
    then round it to a hardware-friendly multiple, which is why Qwen's
    intermediate_size is not exactly 8/3 · hidden_size. That actual value is
    read from the config in WP9 rather than assumed here.

    Honest note: the original GLU-variants paper offers no principled
    explanation for why gating helps - it is an empirical result. Claiming a
    clean theoretical justification would be overstating what is known.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        bias: bool = False,
        dropout_p: float = 0.0,
        multiple_of: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        if d_ff is None:
            d_ff = int(2 * (4 * d_model) / 3)  # 8/3 * d_model
            if multiple_of > 1:  # round up to a hardware-friendly width
                d_ff = multiple_of * ((d_ff + multiple_of - 1) // multiple_of)
        self.d_ff = d_ff

        self.w_gate = nn.Linear(d_model, self.d_ff, bias=bias)  # gated branch
        self.w_up = nn.Linear(d_model, self.d_ff, bias=bias)  # value branch
        self.w_down = nn.Linear(self.d_ff, d_model, bias=bias)  # back to d_model
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, d_model] -> [B, T, d_model].

            gate  [B, T, d_ff]   Swish(W_gate x)
            up    [B, T, d_ff]   W_up x
            prod  [B, T, d_ff]   elementwise product - the gating
            out   [B, T, d_model]
        """
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.dropout(self.w_down(gate * up))

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_ff={self.d_ff}"
