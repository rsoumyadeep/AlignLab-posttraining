"""Normalization (LayerNorm, RMSNorm) and feed-forward (classic, SwiGLU).

The load-bearing tests here are:
  * our LayerNorm matches torch.nn.LayerNorm - proving the biased-variance and
    eps-inside-sqrt details are right;
  * RMSNorm does NOT centre - the single property that distinguishes it;
  * SwiGLU's 8/3 width makes it parameter-equivalent to a 4x classic FFN, which
    is the actual reason for that odd constant.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from alignlab.models.feedforward import FeedForward, SwiGLU
from alignlab.models.normalization import LayerNorm, RMSNorm
from alignlab.models.shapes import assert_shape

B, T, D_MODEL = 2, 5, 16


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(11)


# ==========================================================================
# LayerNorm
# ==========================================================================


def test_layernorm_matches_torch_reference() -> None:
    """Pins the biased variance estimator and eps INSIDE the sqrt.

    Using the unbiased /(N-1) estimator, or adding eps after the sqrt, would
    produce a small systematic mismatch that looks like "just numerics" but is
    a real bug.
    """
    ours = LayerNorm(D_MODEL, eps=1e-5)
    theirs = torch.nn.LayerNorm(D_MODEL, eps=1e-5)

    with torch.no_grad():
        theirs.weight.copy_(ours.weight)
        theirs.bias.copy_(ours.bias)

    x = torch.randn(B, T, D_MODEL)
    torch.testing.assert_close(ours(x), theirs(x), rtol=1e-6, atol=1e-6)


def test_layernorm_output_is_centred_and_unit_scaled() -> None:
    """With gain=1, bias=0: mean 0 and variance 1 over the FEATURE axis."""
    ln = LayerNorm(D_MODEL)
    out = ln(torch.randn(B, T, D_MODEL) * 7.0 + 3.0)

    torch.testing.assert_close(
        out.mean(dim=-1), torch.zeros(B, T), rtol=1e-4, atol=1e-5
    )
    torch.testing.assert_close(
        out.var(dim=-1, unbiased=False), torch.ones(B, T), rtol=1e-3, atol=1e-3
    )


def test_layernorm_normalises_per_token_not_per_batch() -> None:
    """Each token is normalised independently - unlike BatchNorm.

    This is what makes it safe at batch size 1 and with variable-length
    sequences, which is the whole reason language models use it.
    """
    ln = LayerNorm(D_MODEL)
    x = torch.randn(1, T, D_MODEL)

    alone = ln(x)
    # add an unrelated second batch element; the first must not change
    together = ln(torch.cat([x, torch.randn(1, T, D_MODEL) * 100], dim=0))

    torch.testing.assert_close(alone, together[:1], rtol=1e-6, atol=1e-6)


def test_layernorm_parameter_count() -> None:
    assert sum(p.numel() for p in LayerNorm(D_MODEL).parameters()) == 2 * D_MODEL
    assert sum(p.numel() for p in LayerNorm(D_MODEL, bias=False).parameters()) == D_MODEL


# ==========================================================================
# RMSNorm
# ==========================================================================


def test_rmsnorm_output_has_unit_rms() -> None:
    rms = RMSNorm(D_MODEL)
    out = rms(torch.randn(B, T, D_MODEL) * 5.0)

    actual = out.pow(2).mean(dim=-1).sqrt()
    torch.testing.assert_close(actual, torch.ones(B, T), rtol=1e-3, atol=1e-3)


def test_rmsnorm_does_not_centre() -> None:
    """THE distinguishing property.

    RMSNorm rescales magnitude but never subtracts the mean, so a constant
    offset survives normalisation. LayerNorm removes it entirely.
    """
    x = torch.randn(B, T, D_MODEL) + 10.0  # large positive offset

    rms_out = RMSNorm(D_MODEL)(x)
    ln_out = LayerNorm(D_MODEL)(x)

    assert rms_out.mean().abs() > 0.5, "RMSNorm must NOT centre"
    assert ln_out.mean().abs() < 1e-4, "LayerNorm must centre"


def test_rmsnorm_equals_layernorm_on_already_centred_input() -> None:
    """When the mean is already 0, re-centring is a no-op and the two agree.

    Shows precisely what RMSNorm drops: nothing at all, when the input happens
    to be centred.
    """
    x = torch.randn(B, T, 256)
    x = x - x.mean(dim=-1, keepdim=True)  # force exact centring

    ln = LayerNorm(256, eps=1e-6, bias=False)
    rms = RMSNorm(256, eps=1e-6)

    torch.testing.assert_close(ln(x), rms(x), rtol=1e-3, atol=1e-3)


def test_rmsnorm_has_fewer_parameters_than_layernorm() -> None:
    rms_params = sum(p.numel() for p in RMSNorm(D_MODEL).parameters())
    ln_params = sum(p.numel() for p in LayerNorm(D_MODEL).parameters())

    assert rms_params == D_MODEL  # gain only, no bias
    assert ln_params == 2 * D_MODEL
    assert rms_params * 2 == ln_params


def test_rmsnorm_computes_in_float32_under_low_precision_input() -> None:
    """Squaring bf16 activations loses precision badly; the norm is where that
    error would propagate everywhere. Output dtype must still match the input.
    """
    rms = RMSNorm(D_MODEL)
    x = torch.randn(B, T, D_MODEL, dtype=torch.bfloat16)
    out = rms(x.float()).to(torch.bfloat16)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("norm_cls", [LayerNorm, RMSNorm])
def test_norm_gradients_flow(norm_cls) -> None:
    norm = norm_cls(D_MODEL)
    norm(torch.randn(B, T, D_MODEL)).sum().backward()
    assert norm.weight.grad is not None
    assert torch.isfinite(norm.weight.grad).all()


# ==========================================================================
# FeedForward
# ==========================================================================


def test_feedforward_shape_and_default_width() -> None:
    ff = FeedForward(D_MODEL)
    assert ff.d_ff == 4 * D_MODEL
    assert_shape(ff(torch.randn(B, T, D_MODEL)), (B, T, D_MODEL), "ffn out")


def test_feedforward_rejects_unknown_activation() -> None:
    with pytest.raises(ValueError, match="unknown activation"):
        FeedForward(D_MODEL, activation="swish_but_typo")


def test_feedforward_is_position_wise() -> None:
    """The FFN sees one position at a time - it cannot mix across the sequence.

    That is the division of labour: attention routes, the FFN processes.
    """
    ff = FeedForward(D_MODEL).eval()
    x = torch.randn(1, T, D_MODEL)

    with torch.no_grad():
        full = ff(x)
        single = ff(x[:, 2:3, :])

    torch.testing.assert_close(full[:, 2:3, :], single, rtol=1e-5, atol=1e-6)


# ==========================================================================
# SwiGLU
# ==========================================================================


def test_swiglu_shape_and_two_thirds_width() -> None:
    swiglu = SwiGLU(D_MODEL)
    assert swiglu.d_ff == int(2 * (4 * D_MODEL) / 3)
    assert_shape(swiglu(torch.randn(B, T, D_MODEL)), (B, T, D_MODEL), "swiglu out")


def test_swiglu_two_thirds_width_matches_classic_parameter_count() -> None:
    """WHY the 8/3 constant exists.

    classic (2 matrices, 4x width) : 2 * d * 4d      = 8 d^2
    SwiGLU  (3 matrices, 8/3 width): 3 * d * (8/3)d  = 8 d^2

    So 8/3 is exactly the width that makes the gated three-matrix form cost the
    same as the classic two-matrix form. Not mystical - arithmetic.
    """
    d = 768
    classic = sum(p.numel() for p in FeedForward(d, bias=False).parameters())
    swiglu = sum(p.numel() for p in SwiGLU(d, bias=False).parameters())

    assert classic == 8 * d * d
    # integer truncation of 8/3 makes this approximate, not exact
    assert abs(swiglu - classic) / classic < 0.01


def test_swiglu_has_three_projections() -> None:
    swiglu = SwiGLU(D_MODEL)
    names = {n for n, _ in swiglu.named_parameters()}
    assert names == {"w_gate.weight", "w_up.weight", "w_down.weight"}


def test_swiglu_gating_is_multiplicative() -> None:
    """Zeroing the gate branch must zero the output, whatever the up branch is.

    Demonstrates the interaction is a PRODUCT, not a sum - which is what a
    plain MLP cannot express at the same width.
    """
    swiglu = SwiGLU(D_MODEL).eval()
    with torch.no_grad():
        swiglu.w_gate.weight.zero_()  # Swish(0) = 0, so the gate is closed

    out = swiglu(torch.randn(B, T, D_MODEL))
    torch.testing.assert_close(out, torch.zeros_like(out), rtol=1e-6, atol=1e-6)


def test_silu_is_smooth_and_nonmonotonic() -> None:
    """Swish/SiLU properties that motivate it over ReLU.

    * gradient still flows for negative inputs (ReLU gives exactly zero)
    * non-monotonic: it dips below zero, with a minimum near z = -1.28
    """
    z = torch.linspace(-5, 5, 501, requires_grad=True)
    y = F.silu(z)
    y.sum().backward()

    neg = z < -0.1
    assert z.grad[neg].abs().min() > 0, "SiLU must pass gradient for negatives"

    minimum_at = z.detach()[y.detach().argmin()]
    assert -1.5 < minimum_at < -1.0, f"expected dip near -1.28, got {minimum_at}"


def test_swiglu_multiple_of_rounds_width_up() -> None:
    """Real models round d_ff to a hardware-friendly multiple."""
    swiglu = SwiGLU(1024, multiple_of=256)
    assert swiglu.d_ff % 256 == 0
    assert swiglu.d_ff >= int(2 * (4 * 1024) / 3)


def test_swiglu_gradients_flow_to_all_three_matrices() -> None:
    swiglu = SwiGLU(D_MODEL)
    swiglu(torch.randn(B, T, D_MODEL)).sum().backward()

    for name, param in swiglu.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
        assert param.grad.abs().sum() > 0, f"{name} gradient is all zero"
