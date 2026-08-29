"""Attention correctness, shapes, masking and numerical stability.

These are PROPERTY tests as much as value tests. A value test says "this input
gives that output"; a property test says "no input may ever violate this rule",
and it is properties that catch the bugs which matter here - a wrong softmax
axis, a mask applied after normalisation, a transpose in the wrong place.

Platform/dtype are recorded in the equivalence suite
(tests/test_attention_equivalence.py), which is where tolerances live.
"""

from __future__ import annotations

import math

import pytest
import torch

from alignlab.models.attention import (
    MultiHeadAttention,
    causal_mask,
    scaled_dot_product_attention,
)
from alignlab.models.shapes import assert_shape

# Small, fixed dimensions so every shape below can be read at a glance.
B, T, D_MODEL, H = 2, 6, 16, 4
D_K = D_MODEL // H


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(1234)


# --------------------------------------------------------------------------
# shapes - the Phase 2 discipline, made executable
# --------------------------------------------------------------------------


def test_sdpa_shapes_full_chain() -> None:
    """Every shape in the attention chain, asserted explicitly.

    X       [B, T, d_model]
    Q,K,V   [B, H, T, d_k]
    scores  [B, H, T, T]
    context [B, H, T, d_k]
    """
    q = torch.randn(B, H, T, D_K)
    k = torch.randn(B, H, T, D_K)
    v = torch.randn(B, H, T, D_K)

    context, weights = scaled_dot_product_attention(q, k, v)

    assert_shape(q, (B, H, T, D_K), "q")
    assert_shape(weights, (B, H, T, T), "weights")
    assert_shape(context, (B, H, T, D_K), "context")


def test_mha_output_shape_matches_input() -> None:
    """A residual stream only works if attention is shape-preserving."""
    x = torch.randn(B, T, D_MODEL)
    out = MultiHeadAttention(D_MODEL, H)(x)
    assert_shape(out, (B, T, D_MODEL), "mha output")


def test_attention_supports_different_query_and_key_lengths() -> None:
    """Cross-attention and KV-cache decoding both need T_q != T_k."""
    t_q, t_k = 1, 9
    q = torch.randn(B, H, t_q, D_K)
    k = torch.randn(B, H, t_k, D_K)
    v = torch.randn(B, H, t_k, D_K)

    context, weights = scaled_dot_product_attention(q, k, v)
    assert_shape(weights, (B, H, t_q, t_k), "weights")
    assert_shape(context, (B, H, t_q, D_K), "context")


def test_head_split_merge_round_trip() -> None:
    """_split_heads then _merge_heads must be the identity.

    This is where `.contiguous()` bugs live: transpose returns a non-contiguous
    view and view() refuses to reinterpret it.
    """
    mha = MultiHeadAttention(D_MODEL, H)
    x = torch.randn(B, T, D_MODEL)

    split = mha._split_heads(x, H)
    assert_shape(split, (B, H, T, D_K), "split")
    merged = mha._merge_heads(split)

    assert torch.equal(merged, x)


# --------------------------------------------------------------------------
# softmax over the KEY axis
# --------------------------------------------------------------------------


def test_weights_form_a_distribution_over_keys() -> None:
    """Each QUERY produces a distribution over KEYS: rows sum to 1, all >= 0.

    If softmax were applied over the query axis instead, the COLUMNS would sum
    to 1 and this test would fail.
    """
    q, k, v = (torch.randn(B, H, T, D_K) for _ in range(3))
    _, weights = scaled_dot_product_attention(q, k, v)

    assert torch.all(weights >= 0)
    torch.testing.assert_close(
        weights.sum(dim=-1), torch.ones(B, H, T), rtol=1e-6, atol=1e-6
    )


def test_output_is_a_convex_combination_of_values() -> None:
    """Attention output must lie within the convex hull of the values.

    A direct consequence of weights being non-negative and summing to 1. If it
    fails, either the normalisation axis or the final matmul is wrong.
    """
    q = torch.randn(1, 1, 4, D_K)
    k = torch.randn(1, 1, 5, D_K)
    v = torch.randn(1, 1, 5, D_K)

    context, _ = scaled_dot_product_attention(q, k, v)

    assert torch.all(context <= v.max(dim=-2, keepdim=True).values + 1e-5)
    assert torch.all(context >= v.min(dim=-2, keepdim=True).values - 1e-5)


# --------------------------------------------------------------------------
# causal masking
# --------------------------------------------------------------------------


def test_causal_mask_is_lower_triangular() -> None:
    mask = causal_mask(5)
    assert_shape(mask, (5, 5), "causal mask")
    assert torch.equal(mask, torch.ones(5, 5, dtype=torch.bool).tril())


def test_no_attention_weight_to_future_positions() -> None:
    """THE causal property: weight to any future key is exactly 0."""
    q, k, v = (torch.randn(B, H, T, D_K) for _ in range(3))
    mask = causal_mask(T)

    _, weights = scaled_dot_product_attention(q, k, v, mask=mask)

    future = weights.triu(diagonal=1)
    assert torch.all(future == 0), f"leaked into the future: max {future.max()}"


def test_masked_rows_still_sum_to_one() -> None:
    """Why the mask goes BEFORE softmax.

    Masking after softmax would zero some weights and leave the row summing to
    less than 1, shrinking the output by a position-dependent amount. Applying
    -inf before softmax keeps every row a proper distribution.
    """
    q, k, v = (torch.randn(B, H, T, D_K) for _ in range(3))
    _, weights = scaled_dot_product_attention(q, k, v, mask=causal_mask(T))

    torch.testing.assert_close(
        weights.sum(dim=-1), torch.ones(B, H, T), rtol=1e-6, atol=1e-6
    )
    # position 0 sees only itself, so its whole distribution is on itself
    torch.testing.assert_close(
        weights[..., 0, 0], torch.ones(B, H), rtol=1e-6, atol=1e-6
    )


def test_output_at_position_t_ignores_later_tokens() -> None:
    """The strongest causality test: change the FUTURE, output must not move.

    A weight-level check can pass while information still leaks through some
    other path. This checks the actual observable.
    """
    mha = MultiHeadAttention(D_MODEL, H)
    mha.eval()

    x1 = torch.randn(1, T, D_MODEL)
    x2 = x1.clone()
    x2[:, T // 2 :, :] = torch.randn(1, T - T // 2, D_MODEL)  # rewrite the future

    mask = causal_mask(T)
    with torch.no_grad():
        out1 = mha(x1, mask=mask)
        out2 = mha(x2, mask=mask)

    # positions before the edit must be bit-for-bit unaffected
    torch.testing.assert_close(
        out1[:, : T // 2], out2[:, : T // 2], rtol=1e-6, atol=1e-6
    )
    # and the edited region must actually differ, else the test proves nothing
    assert not torch.allclose(out1[:, T // 2 :], out2[:, T // 2 :])


def test_causal_mask_with_kv_cache_offset() -> None:
    """With T_k > T_q the queries are the LAST T_q positions of the keys."""
    mask = causal_mask(seq_len=1, key_len=5)
    assert_shape(mask, (1, 5), "cached mask")
    assert torch.all(mask), "the newest query may attend to all cached keys"

    mask2 = causal_mask(seq_len=2, key_len=5)
    assert mask2[0].tolist() == [True, True, True, True, False]
    assert mask2[1].tolist() == [True, True, True, True, True]


def test_key_len_shorter_than_seq_len_raises() -> None:
    with pytest.raises(ValueError, match="must be >="):
        causal_mask(seq_len=5, key_len=3)


# --------------------------------------------------------------------------
# the scaling factor
# --------------------------------------------------------------------------


def test_default_scale_is_one_over_sqrt_dk() -> None:
    """Verified against an explicitly scaled computation."""
    q = torch.randn(1, 1, 3, D_K)
    k = torch.randn(1, 1, 3, D_K)
    v = torch.randn(1, 1, 3, D_K)

    _, auto = scaled_dot_product_attention(q, k, v)
    _, manual = scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(D_K))
    torch.testing.assert_close(auto, manual, rtol=1e-7, atol=1e-7)


def test_unscaled_attention_saturates_at_large_dk() -> None:
    """WHY the scale factor exists, as a test rather than an assertion.

    Var(q·k) = d_k for unit-variance inputs, so unscaled logits have standard
    deviation sqrt(d_k). At large d_k softmax saturates towards one-hot and its
    gradient vanishes. Scaling restores unit variance.

    Measured across d_k in scripts/experiments/e1_scaling.py.
    """
    torch.manual_seed(0)
    d_k_large = 512
    q = torch.randn(1, 1, 8, d_k_large)
    k = torch.randn(1, 1, 8, d_k_large)
    v = torch.randn(1, 1, 8, d_k_large)

    _, unscaled = scaled_dot_product_attention(q, k, v, scale=1.0)
    _, scaled = scaled_dot_product_attention(q, k, v)

    assert unscaled.max() > 0.99, "expected near-one-hot without scaling"
    assert scaled.max() < unscaled.max(), "scaling must reduce saturation"


# --------------------------------------------------------------------------
# separate Q/K/V projections
# --------------------------------------------------------------------------


def test_tied_qk_produces_symmetric_scores() -> None:
    """WHY W_Q and W_K must be separate.

    With W_Q == W_K the score matrix is (XW)(XW)ᵀ - a Gram matrix, hence
    symmetric. Attention would be forced to be mutually reciprocal, which is
    the wrong prior for language: "it" should attend to "the cat" without "the
    cat" attending equally back.
    """
    torch.manual_seed(0)
    x = torch.randn(1, T, D_MODEL)
    w = torch.nn.Linear(D_MODEL, D_MODEL, bias=False)

    projected = w(x)  # used as BOTH query and key
    scores = projected @ projected.transpose(-2, -1)

    torch.testing.assert_close(scores, scores.transpose(-2, -1), rtol=1e-5, atol=1e-5)

    # ...whereas independent projections are NOT symmetric
    w_q = torch.nn.Linear(D_MODEL, D_MODEL, bias=False)
    w_k = torch.nn.Linear(D_MODEL, D_MODEL, bias=False)
    free = w_q(x) @ w_k(x).transpose(-2, -1)
    assert not torch.allclose(free, free.transpose(-2, -1), rtol=1e-3, atol=1e-3)


def test_qkv_projections_are_independent_parameters() -> None:
    mha = MultiHeadAttention(D_MODEL, H)
    assert mha.w_q.weight is not mha.w_k.weight
    assert mha.w_q.weight is not mha.w_v.weight
    assert not torch.equal(mha.w_q.weight, mha.w_k.weight)


# --------------------------------------------------------------------------
# numerical stability and gradients
# --------------------------------------------------------------------------


def test_large_logits_do_not_overflow() -> None:
    """torch.softmax subtracts the row max internally; verify no nan/inf."""
    q = torch.full((1, 1, 4, D_K), 100.0)
    k = torch.full((1, 1, 4, D_K), 100.0)
    v = torch.randn(1, 1, 4, D_K)

    context, weights = scaled_dot_product_attention(q, k, v)
    assert torch.isfinite(context).all()
    assert torch.isfinite(weights).all()


def test_gradients_flow_to_all_projections() -> None:
    """Every projection must receive a finite, non-zero gradient."""
    mha = MultiHeadAttention(D_MODEL, H)
    x = torch.randn(B, T, D_MODEL, requires_grad=True)

    mha(x, mask=causal_mask(T)).sum().backward()

    for name, param in mha.named_parameters():
        assert param.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} gradient not finite"
        assert param.grad.abs().sum() > 0, f"{name} gradient is all zero"


# --------------------------------------------------------------------------
# permutation equivariance - motivates positional encoding (WP3)
# --------------------------------------------------------------------------


def test_attention_without_mask_is_permutation_equivariant() -> None:
    """Attention alone has NO notion of order. This is why PE exists.

    Permuting the input tokens permutes the output identically - so an
    unmasked, unpositioned attention layer cannot distinguish "dog bites man"
    from "man bites dog". Experiment E6.
    """
    mha = MultiHeadAttention(D_MODEL, H)
    mha.eval()

    x = torch.randn(1, T, D_MODEL)
    perm = torch.randperm(T)

    with torch.no_grad():
        out_then_perm = mha(x)[:, perm]
        perm_then_out = mha(x[:, perm])

    torch.testing.assert_close(out_then_perm, perm_then_out, rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------


def test_mismatched_dk_raises() -> None:
    with pytest.raises(ValueError, match="d_k"):
        scaled_dot_product_attention(
            torch.randn(1, 1, 3, 8), torch.randn(1, 1, 3, 4), torch.randn(1, 1, 3, 4)
        )


def test_mismatched_kv_length_raises() -> None:
    with pytest.raises(ValueError, match="T_k"):
        scaled_dot_product_attention(
            torch.randn(1, 1, 3, 4), torch.randn(1, 1, 5, 4), torch.randn(1, 1, 3, 4)
        )


def test_d_model_not_divisible_by_heads_raises() -> None:
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(d_model=10, n_heads=4)
