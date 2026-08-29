"""Scaled dot-product attention in NumPy.

Reference implementation #2 of 3. Same algorithm as
``alignlab.models.attention``, expressed without any autograd machinery, so
that the computation is visible as pure arithmetic on arrays.

Purpose: an independent check on the PyTorch version. If both agree to within
float64 tolerance, a whole class of bug (wrong transpose, wrong softmax axis,
mask applied in the wrong place) is ruled out - because it is unlikely the same
mistake would be made identically twice in two different array libraries.

Shapes match the PyTorch version exactly:
    query [..., T_q, d_k]   key [..., T_k, d_k]   value [..., T_k, d_v]
"""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax.

    Subtracting the row max before exponentiating is not cosmetic. Raw logits
    of, say, 800 overflow float64's exp (which caps near 709), giving inf/inf
    -> nan. Subtracting the max makes the largest exponent exactly 0, so the
    biggest term is exp(0) == 1 and nothing overflows. The result is identical
    in exact arithmetic because
        softmax(x)_i = e^{x_i - c} / sum_j e^{x_j - c}
    for any constant c - the c cancels.

    Masked rows are handled too: entries at -inf give exp(-inf) == 0.
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    # Where an entire row is -inf the max is -inf and (-inf) - (-inf) = nan.
    # Causal masking never produces such a row (position t can always see
    # itself), but guard anyway so a bad mask fails loudly rather than as nan.
    x_max = np.where(np.isfinite(x_max), x_max, 0.0)
    shifted = x - x_max
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: np.ndarray | None = None,
    scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Attention(Q, K, V) = softmax(Q Kᵀ / sqrt(d_k)) V.

    Args:
        query/key/value: see module docstring.
        mask: broadcastable to [..., T_q, T_k]; **True means KEEP**.
        scale: overrides 1/sqrt(d_k); pass 1.0 to disable scaling.

    Returns:
        (context [..., T_q, d_v], weights [..., T_q, T_k])
    """
    d_k = query.shape[-1]
    if key.shape[-1] != d_k:
        raise ValueError(f"query d_k {d_k} != key d_k {key.shape[-1]}")
    if key.shape[-2] != value.shape[-2]:
        raise ValueError(f"key T_k {key.shape[-2]} != value T_k {value.shape[-2]}")

    # 1. similarity  [..., T_q, d_k] @ [..., d_k, T_k] -> [..., T_q, T_k]
    scores = query @ np.swapaxes(key, -2, -1)

    # 2. scale
    scale = (1.0 / np.sqrt(d_k)) if scale is None else scale
    scores = scores * scale

    # 3. mask before softmax
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)

    # 4. normalise over the key axis
    weights = softmax(scores, axis=-1)

    # 5. weighted average of values
    context = weights @ value

    return context, weights


def causal_mask(seq_len: int, key_len: int | None = None) -> np.ndarray:
    """[T_q, T_k] boolean mask; True means allowed. Mirrors the torch version."""
    key_len = seq_len if key_len is None else key_len
    if key_len < seq_len:
        raise ValueError(f"key_len ({key_len}) must be >= seq_len ({seq_len})")
    offset = key_len - seq_len
    q_idx = np.arange(seq_len)[:, None] + offset
    k_idx = np.arange(key_len)[None, :]
    return k_idx <= q_idx
