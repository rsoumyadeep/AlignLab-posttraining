"""Scaled dot-product attention, from first principles (PyTorch).

This is the reference implementation for AlignLab Phase 2. It is written to be
*read*, not to be fast: every step of

    Q = X W_Q,  K = X W_K,  V = X W_V
    Attention(Q, K, V) = softmax( Q Kᵀ / sqrt(d_k) ) V

appears as its own line with its shape stated. A fused kernel
(``F.scaled_dot_product_attention``) computes the same function far faster and
is used in Phase 2 only as a *comparison target* - never as a substitute for
understanding, because it hides exactly the mechanics this phase teaches.

SHAPE CONTRACT (see shapes.py for the naming convention)

    X        [B, T, d_model]     input token representations
    W_Q      [d_model, H*d_k]    query projection
    W_K      [d_model, H*d_k]    key projection
    W_V      [d_model, H*d_v]    value projection
    Q        [B, H, T_q, d_k]    after projection + head split
    K        [B, H, T_k, d_k]
    V        [B, H, T_k, d_v]
    scores   [B, H, T_q, T_k]    Q @ Kᵀ / sqrt(d_k)
    weights  [B, H, T_q, T_k]    softmax over the LAST axis (keys)
    context  [B, H, T_q, d_v]    weights @ V
    merged   [B, T_q, H*d_v]     heads concatenated
    output   [B, T_q, d_model]   after the output projection W_O

WHY THE PIECES ARE THE WAY THEY ARE - the questions this file must answer:

* Why three SEPARATE projections W_Q, W_K, W_V?
  Because a token plays three different roles. As a *query* it asks "what am I
  looking for?"; as a *key* it advertises "what do I offer?"; as a *value* it
  carries "what do I actually contribute if selected?". Tying the projections
  forces one vector to serve all three roles.

  Concretely, if W_Q == W_K then scores = X W (X W)ᵀ = (XW)(XW)ᵀ is a Gram
  matrix: SYMMETRIC (score(i,j) == score(j,i)) and with a strong diagonal,
  since a vector's largest inner product is usually with itself. Attention
  would be forced to be mutually reciprocal - "if I attend to you, you must
  attend to me equally" - which is the wrong prior for language, where "it"
  should attend strongly to "the cat" without "the cat" attending equally back.
  Verified in tests/test_attention.py::test_tied_qk_produces_symmetric_scores.

* Why divide by sqrt(d_k)?
  If the components of q and k are independent with mean 0 and variance 1, then
  q·k = sum over d_k terms, each with variance 1, so Var(q·k) = d_k and the
  logits have standard deviation sqrt(d_k). As d_k grows the logits spread out,
  softmax saturates towards one-hot, and the gradient of softmax vanishes
  (d softmax/d logit -> 0 when one probability -> 1). Dividing by sqrt(d_k)
  restores unit variance and keeps softmax in its responsive range.
  Measured in scripts/experiments/e1_scaling.py.

* Why is softmax over the KEY axis (the last one)?
  Each query must produce a probability distribution over the positions it can
  attend to, and those weights must sum to 1 so the output is a weighted
  AVERAGE of values (a convex combination). Softmaxing over queries instead
  would make each key's attention sum to 1 across queries, which answers a
  question nobody asked and does not produce a well-scaled output.

* Why does the mask go BEFORE the softmax?
  Masking after the softmax (zeroing the weights) would leave the remaining
  weights summing to less than 1 - the output would be shrunk toward zero by an
  amount that varies with position. Adding -inf before the softmax makes those
  entries exactly 0 AFTER normalisation while the surviving weights still sum
  to 1. Verified in tests/test_attention.py::test_masked_rows_still_sum_to_one.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from alignlab.models.shapes import assert_shape

# Value added to masked-out logits. -inf is mathematically what we want
# (exp(-inf) == 0), and PyTorch's softmax handles it correctly as long as at
# least one entry per row is unmasked - which causal masking guarantees, since
# position t can always attend to itself.
MASK_VALUE = float("-inf")


def causal_mask(
    seq_len: int,
    device: torch.device | str | None = None,
    key_len: int | None = None,
) -> torch.Tensor:
    """Build a boolean causal mask. True means "allowed to attend".

    Args:
        seq_len: number of query positions T_q.
        device: where to build it.
        key_len: number of key positions T_k. Defaults to seq_len. When
            T_k > T_q (the KV-cache case, where we have one new query but many
            cached keys) the queries are taken to be the LAST T_q positions of
            the key sequence, which is what incremental decoding needs.

    Returns:
        [T_q, T_k] bool tensor, lower-triangular in the square case.

    Why causal masking is required for autoregressive language modelling:
    the training objective asks the model to predict token t+1 from tokens
    <= t, evaluated at every position in parallel. Without the mask, position t
    can read position t+1 - the very token it is being asked to predict - and
    the task collapses into copying. Loss drops dramatically and the model
    learns nothing useful. Measured in scripts/experiments/e2_causal_leakage.py.
    """
    key_len = seq_len if key_len is None else key_len
    if key_len < seq_len:
        raise ValueError(
            f"key_len ({key_len}) must be >= seq_len ({seq_len}): a query "
            "cannot have fewer keys than its own position implies"
        )

    # Absolute index of each query within the key sequence. With a KV cache,
    # T_q new queries sit at the END of the T_k keys.
    offset = key_len - seq_len
    q_idx = torch.arange(seq_len, device=device).unsqueeze(1) + offset  # [T_q, 1]
    k_idx = torch.arange(key_len, device=device).unsqueeze(0)  # [1, T_k]
    return k_idx <= q_idx  # [T_q, T_k]


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    scale: float | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention(Q, K, V) = softmax(Q Kᵀ / sqrt(d_k)) V.

    Deliberately written step by step. Works for any number of leading batch
    dimensions, so it serves both [B, T, d] and [B, H, T, d] callers.

    Args:
        query: [..., T_q, d_k]
        key:   [..., T_k, d_k]
        value: [..., T_k, d_v]
        mask:  broadcastable to [..., T_q, T_k]. **True means KEEP.**
        scale: overrides 1/sqrt(d_k). Passing 1.0 disables scaling, which is
            what experiment E1 uses to show why the scale factor exists.
        dropout_p / training: attention dropout, applied to the weights.

    Returns:
        (context, weights)
            context [..., T_q, d_v]
            weights [..., T_q, T_k]  - returned because they are the single
                most useful diagnostic object in the whole architecture.
    """
    d_k = query.shape[-1]
    if key.shape[-1] != d_k:
        raise ValueError(
            f"query d_k ({d_k}) must equal key d_k ({key.shape[-1]}): they are "
            "compared by inner product"
        )
    if key.shape[-2] != value.shape[-2]:
        raise ValueError(
            f"key T_k ({key.shape[-2]}) must equal value T_k "
            f"({value.shape[-2]}): every key must have a matching value"
        )

    # --- 1. similarity: how much does each query match each key? ------------
    # [..., T_q, d_k] @ [..., d_k, T_k] -> [..., T_q, T_k]
    scores = query @ key.transpose(-2, -1)

    # --- 2. scale, so softmax stays in its responsive range ------------------
    scale = (1.0 / math.sqrt(d_k)) if scale is None else scale
    scores = scores * scale

    # --- 3. mask BEFORE softmax, so surviving weights still sum to 1 --------
    if mask is not None:
        scores = scores.masked_fill(~mask, MASK_VALUE)

    # --- 4. normalise over the KEY axis -> a distribution per query ----------
    weights = torch.softmax(scores, dim=-1)

    if dropout_p > 0.0 and training:
        weights = F.dropout(weights, p=dropout_p, training=True)

    # --- 5. weighted average of the values -----------------------------------
    # [..., T_q, T_k] @ [..., T_k, d_v] -> [..., T_q, d_v]
    context = weights @ value

    return context, weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional MQA / GQA, written for reading.

    WHY MULTIPLE HEADS?
    A single softmax produces ONE distribution per query, so a single head can
    only average one "kind" of relationship at a time. Language needs several
    at once: syntactic agreement, coreference, local n-gram context, topic. By
    projecting into H lower-dimensional subspaces and attending independently
    in each, the layer can represent H relationships simultaneously and then
    combine them through W_O.

    Note the parameter count is unchanged: H heads of size d_k = d_model / H
    cost the same as one head of size d_model. Multiple heads buy
    *specialisation*, not capacity - which is the point most explanations miss.

    KV-HEAD SHARING (set n_kv_heads):
        n_kv_heads == n_heads  -> MHA  (every query head has its own K/V)
        n_kv_heads == 1        -> MQA  (all query heads share ONE K/V head)
        1 < n_kv_heads < n_heads -> GQA (groups of query heads share a K/V head)
    Covered in detail in models/attention_variants documentation; the code path
    is here because it is a three-line change to the projections plus one
    repeat_interleave.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        bias: bool = False,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        if n_heads % n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({n_heads}) must be divisible by n_kv_heads "
                f"({n_kv_heads}) so query heads split into equal groups"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # query heads per KV head
        self.d_k = d_model // n_heads
        self.dropout_p = dropout_p

        # Separate projections - see the module docstring for why.
        # Q keeps all H heads; K and V only produce n_kv_heads heads.
        self.w_q = nn.Linear(d_model, n_heads * self.d_k, bias=bias)
        self.w_k = nn.Linear(d_model, n_kv_heads * self.d_k, bias=bias)
        self.w_v = nn.Linear(d_model, n_kv_heads * self.d_k, bias=bias)
        self.w_o = nn.Linear(n_heads * self.d_k, d_model, bias=bias)

    def _split_heads(self, x: torch.Tensor, n_heads: int) -> torch.Tensor:
        """[B, T, n_heads*d_k] -> [B, n_heads, T, d_k].

        The transpose is what puts the head axis next to the batch axis, so the
        subsequent matmuls treat each head as an independent problem.
        """
        B, T, _ = x.shape
        x = x.view(B, T, n_heads, self.d_k)  # [B, T, H, d_k]
        return x.transpose(1, 2)  # [B, H, T, d_k]

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, H, T, d_v] -> [B, T, H*d_v].

        ``.contiguous()`` is required: ``transpose`` returns a non-contiguous
        view, and ``view`` refuses to reinterpret non-contiguous memory. This
        is the single most common shape bug in hand-written attention.
        """
        B, H, T, d_v = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * d_v)

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Expand [B, H_kv, T, d] -> [B, H_kv*n_rep, T, d] for GQA/MQA.

        Each KV head is repeated for the query heads in its group. Note this
        materialises the expansion, so it does NOT save memory during the
        matmul - the saving is in the KV CACHE and in the projection
        parameters, which is the point of GQA.
        """
        if n_rep == 1:
            return x
        return x.repeat_interleave(n_rep, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, d_model]
            mask: broadcastable to [B, H, T_q, T_k]; True means keep.
        Returns:
            [B, T, d_model], and the attention weights if requested.
        """
        B, T, _ = x.shape
        assert_shape(x, (None, None, self.d_model), "x")

        # --- projections --------------------------------------------------
        q = self._split_heads(self.w_q(x), self.n_heads)  # [B, H,    T, d_k]
        k = self._split_heads(self.w_k(x), self.n_kv_heads)  # [B, H_kv, T, d_k]
        v = self._split_heads(self.w_v(x), self.n_kv_heads)  # [B, H_kv, T, d_k]

        # --- share K/V across query-head groups (no-op for MHA) ------------
        k = self._repeat_kv(k, self.n_rep)  # [B, H, T, d_k]
        v = self._repeat_kv(v, self.n_rep)  # [B, H, T, d_k]

        assert_shape(q, (B, self.n_heads, T, self.d_k), "q")
        assert_shape(k, (B, self.n_heads, T, self.d_k), "k")

        # --- attention ------------------------------------------------------
        context, weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout_p=self.dropout_p, training=self.training
        )
        assert_shape(context, (B, self.n_heads, T, self.d_k), "context")

        # --- merge heads and project ---------------------------------------
        merged = self._merge_heads(context)  # [B, T, H*d_k] == [B, T, d_model]
        out = self.w_o(merged)  # [B, T, d_model]
        assert_shape(out, (B, T, self.d_model), "output")

        if return_weights:
            return out, weights
        return out

    def extra_repr(self) -> str:
        kind = (
            "MHA"
            if self.n_kv_heads == self.n_heads
            else "MQA"
            if self.n_kv_heads == 1
            else "GQA"
        )
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"n_kv_heads={self.n_kv_heads}, d_k={self.d_k}, kind={kind}"
        )
