"""Positional encodings: learned absolute, sinusoidal, RoPE, ALiBi.

The centrepiece is test_rope_inner_product_depends_only_on_relative_distance.
RoPE's entire claim is that attention scores become a function of (m - n) with
absolute position cancelling; that is a mathematical property, so it can be
checked numerically rather than argued about. A sign or pairing-convention
error - the classic RoPE bug - fails that test while leaving shapes and norms
perfectly correct.
"""

from __future__ import annotations

import math

import pytest
import torch

from alignlab.models.positional import (
    ALiBiBias,
    LearnedAbsolutePositionalEmbedding,
    RotaryPositionalEmbedding,
    SinusoidalPositionalEncoding,
)
from alignlab.models.shapes import assert_shape

B, T, D_MODEL, H = 2, 8, 32, 4
HEAD_DIM = D_MODEL // H  # 8


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(7)


# ==========================================================================
# learned absolute
# ==========================================================================


def test_learned_pe_shape_and_addition() -> None:
    pe = LearnedAbsolutePositionalEmbedding(max_seq_len=16, d_model=D_MODEL)
    x = torch.zeros(B, T, D_MODEL)
    out = pe(x)

    assert_shape(out, (B, T, D_MODEL), "learned PE output")
    # with x == 0 the output IS the embedding, identical across the batch
    torch.testing.assert_close(out[0], out[1])


def test_learned_pe_is_trainable() -> None:
    pe = LearnedAbsolutePositionalEmbedding(16, D_MODEL)
    pe(torch.zeros(1, T, D_MODEL)).sum().backward()
    assert pe.embedding.weight.grad is not None
    assert pe.embedding.weight.grad.abs().sum() > 0


def test_learned_pe_hard_fails_beyond_max_len() -> None:
    """The defining limitation: no value exists for an unseen position.

    Not "degrades gracefully" - undefined. This is the concrete reason
    absolute learned embeddings cannot extrapolate.
    """
    pe = LearnedAbsolutePositionalEmbedding(max_seq_len=4, d_model=D_MODEL)
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        pe(torch.zeros(1, 5, D_MODEL))


# ==========================================================================
# sinusoidal
# ==========================================================================


def test_sinusoidal_matches_the_published_formula() -> None:
    """PE(pos,2i)=sin(pos/base^(2i/d)), PE(pos,2i+1)=cos(same).

    Checked against a direct scalar transcription of the formula, so a bug in
    the vectorised construction cannot hide.
    """
    d_model, base, max_len = 8, 10000.0, 6
    pe = SinusoidalPositionalEncoding(max_len, d_model, base=base)

    for pos in range(max_len):
        for i in range(d_model // 2):
            angle = pos / (base ** (2 * i / d_model))
            assert math.isclose(pe.pe[pos, 2 * i].item(), math.sin(angle), abs_tol=1e-6)
            assert math.isclose(
                pe.pe[pos, 2 * i + 1].item(), math.cos(angle), abs_tol=1e-6
            )


def test_sinusoidal_is_bounded() -> None:
    """sin/cos in [-1,1], so PE cannot swamp the token embedding."""
    pe = SinusoidalPositionalEncoding(512, 64)
    assert pe.pe.abs().max() <= 1.0 + 1e-6


def test_sinusoidal_positions_are_unique() -> None:
    pe = SinusoidalPositionalEncoding(128, 64)
    codes = pe.pe
    dist = torch.cdist(codes, codes)
    dist.fill_diagonal_(float("inf"))
    assert dist.min() > 1e-3, "two positions received near-identical codes"


def test_sinusoidal_offset_is_a_fixed_linear_rotation() -> None:
    """The property the derivation turns on.

    For a fixed offset k, moving from pos to pos+k rotates each (sin, cos) pair
    by w*k - a matrix that depends on k but NOT on pos. That is what lets a
    linear map implement "shift by k" uniformly.
    """
    d_model, base = 16, 10000.0
    pe = SinusoidalPositionalEncoding(64, d_model, base=base).pe
    k = 5

    for i in range(d_model // 2):
        w = 1.0 / (base ** (2 * i / d_model))
        c, s = math.cos(w * k), math.sin(w * k)
        for pos in (0, 3, 17, 40):
            sin_p, cos_p = pe[pos, 2 * i].item(), pe[pos, 2 * i + 1].item()
            sin_pk = pe[pos + k, 2 * i].item()
            cos_pk = pe[pos + k, 2 * i + 1].item()
            # [sin(w(p+k))] = [ cos(wk)  sin(wk)] [sin(wp)]
            # [cos(w(p+k))]   [-sin(wk)  cos(wk)] [cos(wp)]
            assert math.isclose(sin_pk, c * sin_p + s * cos_p, abs_tol=1e-5)
            assert math.isclose(cos_pk, -s * sin_p + c * cos_p, abs_tol=1e-5)


def test_sinusoidal_is_a_buffer_not_a_parameter() -> None:
    """Fixed, but must move with .to(device) and be saved. That means buffer."""
    pe = SinusoidalPositionalEncoding(32, D_MODEL)
    assert "pe" in dict(pe.named_buffers())
    assert "pe" not in dict(pe.named_parameters())
    assert "pe" in pe.state_dict()


def test_sinusoidal_similarity_decays_with_distance() -> None:
    """Experiment E5b: nearby positions have more similar codes."""
    pe = SinusoidalPositionalEncoding(128, 64).pe
    ref = pe[64]
    near = torch.dot(ref, pe[65]).item()
    far = torch.dot(ref, pe[120]).item()
    assert near > far


# ==========================================================================
# RoPE - the important ones
# ==========================================================================


def test_rope_shapes_unchanged() -> None:
    rope = RotaryPositionalEmbedding(HEAD_DIM)
    q = torch.randn(B, H, T, HEAD_DIM)
    k = torch.randn(B, H, T, HEAD_DIM)

    q_r, k_r = rope(q, k)
    assert_shape(q_r, (B, H, T, HEAD_DIM), "rotated q")
    assert_shape(k_r, (B, H, T, HEAD_DIM), "rotated k")


def test_rope_preserves_norm() -> None:
    """Rotations are orthogonal, so RoPE must not change vector magnitude.

    If it does, the attention logit scale silently shifts - which would
    interact badly with the 1/sqrt(d_k) calibration from WP1.
    """
    rope = RotaryPositionalEmbedding(HEAD_DIM)
    q = torch.randn(B, H, T, HEAD_DIM)
    q_r, _ = rope(q, q.clone())

    torch.testing.assert_close(
        q_r.norm(dim=-1), q.norm(dim=-1), rtol=1e-5, atol=1e-5
    )


def test_rope_at_position_zero_is_identity() -> None:
    """Rotation by angle 0 changes nothing."""
    rope = RotaryPositionalEmbedding(HEAD_DIM)
    q = torch.randn(1, 1, 1, HEAD_DIM)
    q_r, _ = rope(q, q.clone(), offset=0)
    torch.testing.assert_close(q_r, q, rtol=1e-6, atol=1e-6)


def test_rope_inner_product_depends_only_on_relative_distance() -> None:
    """THE RoPE property, and experiment E5.

    <R(m)q, R(n)k> must depend only on (m - n). So for a fixed gap, sliding
    BOTH positions along the sequence must leave the inner product unchanged.

    A sign error or a mismatched pairing convention - the classic RoPE bugs -
    leaves shapes and norms correct but breaks exactly this.
    """
    rope = RotaryPositionalEmbedding(HEAD_DIM, max_seq_len=256)
    torch.manual_seed(0)
    q = torch.randn(1, 1, 1, HEAD_DIM)
    k = torch.randn(1, 1, 1, HEAD_DIM)

    gap = 3
    products = []
    for m in (0, 5, 20, 57, 100):
        q_r, _ = rope(q, q.clone(), offset=m + gap)
        _, k_r = rope(k.clone(), k, offset=m)
        products.append(float((q_r * k_r).sum()))

    spread = max(products) - min(products)
    print(f"\n  RoPE inner products at fixed gap={gap}: {products}")
    print(f"  spread = {spread:.3e}")
    assert spread < 1e-5, f"absolute position did not cancel; spread {spread}"


def test_rope_inner_product_changes_with_different_distance() -> None:
    """Guard for the test above: it must not pass by being constant."""
    rope = RotaryPositionalEmbedding(HEAD_DIM, max_seq_len=256)
    torch.manual_seed(0)
    q = torch.randn(1, 1, 1, HEAD_DIM)
    k = torch.randn(1, 1, 1, HEAD_DIM)

    def product(gap: int) -> float:
        q_r, _ = rope(q, q.clone(), offset=gap)
        _, k_r = rope(k.clone(), k, offset=0)
        return float((q_r * k_r).sum())

    assert abs(product(1) - product(7)) > 1e-3


def test_rope_offset_shifts_positions() -> None:
    """offset is what makes cached generation correct.

    Rotating a single token with offset=5 must equal rotating a 6-token
    sequence and taking the last position.
    """
    rope = RotaryPositionalEmbedding(HEAD_DIM, max_seq_len=64)
    torch.manual_seed(0)
    full = torch.randn(1, 1, 6, HEAD_DIM)

    rotated_full, _ = rope(full, full.clone(), offset=0)
    last_alone, _ = rope(full[:, :, 5:, :], full[:, :, 5:, :].clone(), offset=5)

    torch.testing.assert_close(
        rotated_full[:, :, 5:, :], last_alone, rtol=1e-5, atol=1e-5
    )


def test_rope_rejects_odd_head_dim() -> None:
    with pytest.raises(ValueError, match="even"):
        RotaryPositionalEmbedding(head_dim=7)


def test_rope_tables_are_non_persistent_buffers() -> None:
    """Recomputable from head_dim/base, so they need not bloat a checkpoint."""
    rope = RotaryPositionalEmbedding(HEAD_DIM)
    assert "cos_cached" in dict(rope.named_buffers())
    assert "cos_cached" not in rope.state_dict()


def test_rope_extends_table_beyond_initial_max() -> None:
    rope = RotaryPositionalEmbedding(HEAD_DIM, max_seq_len=4)
    q = torch.randn(1, 1, 10, HEAD_DIM)
    q_r, _ = rope(q, q.clone())
    assert_shape(q_r, (1, 1, 10, HEAD_DIM), "rotated beyond initial table")


# ==========================================================================
# ALiBi
# ==========================================================================


def test_alibi_shape_and_diagonal() -> None:
    alibi = ALiBiBias(n_heads=H)
    bias = alibi(T)

    assert_shape(bias, (1, H, T, T), "alibi bias")
    # a token is distance 0 from itself, so no penalty on the diagonal
    for h in range(H):
        torch.testing.assert_close(
            bias[0, h].diagonal(), torch.zeros(T), rtol=1e-6, atol=1e-6
        )


def test_alibi_penalty_grows_with_distance() -> None:
    """More distant keys are penalised more - a monotone decay."""
    bias = ALiBiBias(n_heads=H)(T)
    row = bias[0, 0, T - 1]  # last query, all keys
    for j in range(T - 1):
        assert row[j] < row[j + 1], "penalty must increase with distance"


def test_alibi_slopes_are_geometric_and_distinct() -> None:
    """Each head gets a different range: small slope looks far, large looks near."""
    alibi = ALiBiBias(n_heads=8)
    slopes = alibi.slopes
    assert slopes.numel() == 8
    assert len(set(slopes.tolist())) == 8, "heads must have distinct ranges"
    assert torch.all(slopes > 0)
    ratios = slopes[1:] / slopes[:-1]
    torch.testing.assert_close(ratios, ratios[0].expand_as(ratios), rtol=1e-4, atol=1e-6)


def test_alibi_handles_non_power_of_two_heads() -> None:
    for n in (3, 6, 12):
        assert ALiBiBias(n_heads=n).slopes.numel() == n


def test_alibi_uses_no_learned_parameters() -> None:
    """ALiBi adds zero parameters - part of why it is cheap."""
    assert sum(p.numel() for p in ALiBiBias(n_heads=H).parameters()) == 0


# ==========================================================================
# absolute vs relative - the organising comparison
# ==========================================================================


def test_absolute_schemes_change_the_representation_relative_ones_do_not() -> None:
    """The structural difference, made concrete.

    Absolute PE ADDS to the token embedding, so the residual stream itself
    changes. RoPE leaves the embedding untouched and acts on Q/K inside
    attention; ALiBi leaves the vectors alone entirely and edits scores.
    """
    x = torch.randn(1, T, D_MODEL)

    sinusoidal = SinusoidalPositionalEncoding(64, D_MODEL)
    assert not torch.allclose(sinusoidal(x), x), "absolute PE must modify x"

    # RoPE never sees x at all - it takes q and k
    rope = RotaryPositionalEmbedding(HEAD_DIM)
    assert not hasattr(rope, "forward_embedding")

    # ALiBi produces an additive score bias, not a vector modification
    assert_shape(ALiBiBias(H)(T), (1, H, T, T), "alibi")
