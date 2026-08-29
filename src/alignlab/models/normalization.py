"""LayerNorm and RMSNorm, from first principles.

WHY NORMALISE AT ALL. A deep residual stack adds contributions layer after
layer, so activation magnitudes drift upward with depth. Downstream layers then
see inputs at a scale their weights were not initialised for, gradients scale
badly, and training destabilises. Normalisation pins the scale at every layer,
which is what makes deep stacks trainable at all.

WHY NOT BATCHNORM. BatchNorm normalises across the BATCH axis, so each example's
output depends on the other examples in its batch. For language models that is
fatal: sequences have different lengths (padding pollutes the statistics), batch
size varies, and at generation time the batch is often 1 - there are no batch
statistics. LayerNorm and RMSNorm normalise across the FEATURE axis of each
token independently, so a token's output depends only on itself.

    LayerNorm   y = g * (x - mu) / sqrt(var + eps) + b     mu, var over features
    RMSNorm     y = g * x / sqrt(mean(x^2) + eps)          no mean, no bias
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Educational LayerNorm - the same function as torch.nn.LayerNorm.

        mu    = mean(x)                 over the last axis
        var   = mean((x - mu)^2)        BIASED estimator, i.e. /N not /(N-1)
        y     = gamma * (x - mu)/sqrt(var + eps) + beta

    Two details that matter:

    * The variance uses the BIASED (population) estimator. torch's
      ``nn.LayerNorm`` does the same. Using the unbiased /(N-1) form would
      produce a small systematic mismatch against every reference
      implementation - a difference big enough to fail a tolerance check while
      looking like "just numerics".

    * eps sits INSIDE the square root, ``sqrt(var + eps)``, not outside. This
      matters when var is tiny: adding eps after the sqrt would barely change
      the denominator, whereas the standard form bounds it away from zero.

    Parameters: 2 * normalized_shape (gain and bias).
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5, bias: bool = True) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [..., normalized_shape] -> same shape."""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + self.eps)

        out = normalized * self.weight
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}, bias={self.bias is not None}"


class RMSNorm(nn.Module):
    """RMSNorm (Zhang & Sennrich, 2019). Used by Llama, Qwen, Mistral, T5.

        rms(x) = sqrt(mean(x^2) + eps)
        y      = gamma * x / rms(x)

    THE DIFFERENCE FROM LAYERNORM, precisely: RMSNorm drops the mean
    subtraction (re-centring) and keeps only the rescaling. It also usually
    drops the bias.

    WHY THAT IS DEFENSIBLE. The original RMSNorm claim is that LayerNorm's
    benefit comes almost entirely from re-SCALING, not re-CENTRING. If true,
    the mean subtraction is doing little work and can be removed.

    WHAT IT BUYS:
      * fewer operations - no mean pass, no subtraction. Two reduction passes
        over the feature axis become one.
      * fewer parameters - no bias, so d_model rather than 2*d_model.
      * simpler backward pass, which matters at scale.

    WHAT IT COSTS:
      * the output is no longer mean-zero. RMSNorm normalises MAGNITUDE only,
        so any constant offset in x survives. Whether that matters is an
        empirical question, and the field's answer - by adoption - is that it
        largely does not.

    NOTE ON PRECISION: the computation is done in float32 even under autocast.
    In bf16, squaring activations can overflow or lose precision badly, and a
    normalisation layer is exactly where that error propagates everywhere.
    Real implementations (Llama, Qwen) do the same.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # rsqrt is cheaper than 1/sqrt and is what production kernels use.
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [..., normalized_shape] -> same shape."""
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}"
