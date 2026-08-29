"""MHA vs MQA vs GQA.

The three differ in exactly one number: how many KEY/VALUE heads exist for a
given number of QUERY heads.

    n_kv_heads == n_heads      MHA   every query head has its own K/V
    n_kv_heads == 1            MQA   all query heads share ONE K/V head
    1 < n_kv_heads < n_heads   GQA   groups of query heads share a K/V head

Everything else - the scores, the softmax, the mask, the output projection -
is identical. The most important test here is that GQA with
n_kv_heads == n_heads reduces EXACTLY to MHA, because that proves the shared
code path has not quietly changed the maths for the MHA case.
"""

from __future__ import annotations

import pytest
import torch

from alignlab.models.attention import MultiHeadAttention, causal_mask
from alignlab.models.shapes import assert_shape

B, T, D_MODEL, H = 2, 6, 64, 8
D_K = D_MODEL // H  # 8


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(4242)


# --------------------------------------------------------------------------
# construction and classification
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_kv_heads,kind",
    [(H, "MHA"), (1, "MQA"), (2, "GQA"), (4, "GQA")],
)
def test_variant_is_classified_correctly(n_kv_heads: int, kind: str) -> None:
    mha = MultiHeadAttention(D_MODEL, H, n_kv_heads=n_kv_heads)
    assert kind in mha.extra_repr()
    assert mha.n_rep == H // n_kv_heads


def test_n_heads_must_be_divisible_by_n_kv_heads() -> None:
    """Query heads must split into EQUAL groups."""
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(D_MODEL, n_heads=8, n_kv_heads=3)


# --------------------------------------------------------------------------
# THE key equivalence: GQA(n_kv=n_h) is exactly MHA
# --------------------------------------------------------------------------


def test_gqa_with_full_kv_heads_is_exactly_mha() -> None:
    """The shared code path must not alter the MHA case.

    Same weights, same input; the GQA path (with n_rep == 1, so repeat_kv is a
    no-op) must produce bit-identical output to plain MHA.
    """
    torch.manual_seed(0)
    mha = MultiHeadAttention(D_MODEL, H, n_kv_heads=None)  # defaults to H
    gqa = MultiHeadAttention(D_MODEL, H, n_kv_heads=H)
    gqa.load_state_dict(mha.state_dict())
    mha.eval(), gqa.eval()

    x = torch.randn(B, T, D_MODEL)
    with torch.no_grad():
        a = mha(x, mask=causal_mask(T))
        b = gqa(x, mask=causal_mask(T))

    assert torch.equal(a, b), "GQA with n_kv_heads == n_heads must reduce to MHA"


# --------------------------------------------------------------------------
# shapes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_kv_heads", [1, 2, 4, 8])
def test_output_shape_is_variant_independent(n_kv_heads: int) -> None:
    """Whatever the KV sharing, the block is shape-preserving."""
    layer = MultiHeadAttention(D_MODEL, H, n_kv_heads=n_kv_heads)
    x = torch.randn(B, T, D_MODEL)
    out, weights = layer(x, mask=causal_mask(T), return_weights=True)

    assert_shape(out, (B, T, D_MODEL), "output")
    # weights always cover ALL query heads - sharing happens on K/V, not Q
    assert_shape(weights, (B, H, T, T), "weights")


@pytest.mark.parametrize("n_kv_heads", [1, 2, 8])
def test_kv_projection_shapes(n_kv_heads: int) -> None:
    """W_K and W_V shrink with n_kv_heads; W_Q and W_O do not."""
    layer = MultiHeadAttention(D_MODEL, H, n_kv_heads=n_kv_heads)

    assert layer.w_q.weight.shape == (H * D_K, D_MODEL)
    assert layer.w_k.weight.shape == (n_kv_heads * D_K, D_MODEL)
    assert layer.w_v.weight.shape == (n_kv_heads * D_K, D_MODEL)
    assert layer.w_o.weight.shape == (D_MODEL, H * D_K)


def test_repeat_kv_expands_groups_correctly() -> None:
    """Each KV head must serve a CONTIGUOUS group of query heads.

    repeat_interleave gives [kv0, kv0, kv1, kv1]; a plain repeat would give
    [kv0, kv1, kv0, kv1], silently pairing query heads with the wrong group.
    This test pins the correct one.
    """
    x = torch.arange(2.0).reshape(1, 2, 1, 1)  # two KV heads: values 0 and 1
    expanded = MultiHeadAttention._repeat_kv(x, n_rep=2)

    assert_shape(expanded, (1, 4, 1, 1), "expanded")
    assert expanded.flatten().tolist() == [0.0, 0.0, 1.0, 1.0]


def test_repeat_kv_is_noop_when_n_rep_is_one() -> None:
    x = torch.randn(1, 4, 3, 8)
    assert MultiHeadAttention._repeat_kv(x, 1) is x


# --------------------------------------------------------------------------
# MQA: all query heads share one K/V head
# --------------------------------------------------------------------------


def test_mqa_all_heads_share_one_kv() -> None:
    """With one KV head, every query head attends over the SAME keys.

    The attention weights still differ per head, because the QUERIES differ -
    sharing K/V does not collapse the heads into one.
    """
    layer = MultiHeadAttention(D_MODEL, H, n_kv_heads=1)
    layer.eval()
    x = torch.randn(1, T, D_MODEL)

    with torch.no_grad():
        _, weights = layer(x, mask=causal_mask(T), return_weights=True)

    assert_shape(weights, (1, H, T, T), "weights")
    # heads must still be distinguishable
    assert not torch.allclose(weights[0, 0], weights[0, 1], atol=1e-6)


# --------------------------------------------------------------------------
# parameter and KV-cache accounting - the reason GQA exists
# --------------------------------------------------------------------------


def _param_count(layer: MultiHeadAttention) -> int:
    return sum(p.numel() for p in layer.parameters())


def test_parameter_count_shrinks_with_fewer_kv_heads() -> None:
    counts = {
        kv: _param_count(MultiHeadAttention(D_MODEL, H, n_kv_heads=kv))
        for kv in (8, 4, 2, 1)
    }
    # strictly decreasing as KV heads are shared
    assert counts[8] > counts[4] > counts[2] > counts[1]

    # exact arithmetic: W_Q and W_O are fixed at d_model^2; W_K and W_V are
    # each d_model * (n_kv_heads * d_k)
    for kv, total in counts.items():
        expected = 2 * D_MODEL * D_MODEL + 2 * D_MODEL * (kv * D_K)
        assert total == expected, f"n_kv_heads={kv}: {total} != {expected}"


def test_kv_cache_bytes_scale_with_kv_heads() -> None:
    """The real motivation for GQA/MQA.

    A KV cache stores K and V for every past position:
        2 * B * n_kv_heads * T * d_k * bytes_per_element
    It scales with n_kv_heads, NOT n_heads - so MQA cuts the cache by a factor
    of n_heads. At long context this dominates inference memory.
    """
    batch, seq, bytes_per = 1, 4096, 2  # bf16

    def cache_bytes(n_kv: int) -> int:
        return 2 * batch * n_kv * seq * D_K * bytes_per

    mha_bytes = cache_bytes(H)  # n_kv = 8
    gqa_bytes = cache_bytes(2)  # n_kv = 2
    mqa_bytes = cache_bytes(1)  # n_kv = 1

    # The cache is exactly linear in n_kv_heads, so the ratios are the ratios
    # of the head counts - nothing subtler than that.
    assert mha_bytes == (H // 2) * gqa_bytes, "8 KV heads vs 2 is a factor of 4"
    assert gqa_bytes == 2 * mqa_bytes, "2 KV heads vs 1 is a factor of 2"
    assert mha_bytes == H * mqa_bytes, "MQA cuts the cache by exactly n_heads"

    # concrete numbers for the documentation, at T=4096, d_k=8, bf16
    assert (mha_bytes, gqa_bytes, mqa_bytes) == (1_048_576, 262_144, 131_072)


# --------------------------------------------------------------------------
# behaviour is preserved across variants
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_kv_heads", [1, 2, 8])
def test_causality_holds_for_every_variant(n_kv_heads: int) -> None:
    """KV sharing must not break the causal guarantee."""
    layer = MultiHeadAttention(D_MODEL, H, n_kv_heads=n_kv_heads)
    x = torch.randn(B, T, D_MODEL)

    _, weights = layer(x, mask=causal_mask(T), return_weights=True)
    assert torch.all(weights.triu(diagonal=1) == 0)


@pytest.mark.parametrize("n_kv_heads", [1, 2, 8])
def test_gradients_flow_for_every_variant(n_kv_heads: int) -> None:
    layer = MultiHeadAttention(D_MODEL, H, n_kv_heads=n_kv_heads)
    x = torch.randn(B, T, D_MODEL)

    layer(x, mask=causal_mask(T)).sum().backward()

    for name, param in layer.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
        assert param.grad.abs().sum() > 0, f"{name} gradient is all zero"
