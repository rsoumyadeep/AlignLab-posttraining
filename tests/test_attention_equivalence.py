"""Cross-verification: PyTorch vs NumPy vs pure Python attention.

Three independent implementations of the SAME algorithm. If all three agree,
a whole class of bug is ruled out - a wrong transpose, a softmax over the wrong
axis, a mask applied after normalisation - because it is implausible that the
identical mistake would be made three times in three different substrates.

TOLERANCE POLICY (declared in the Phase 2 plan BEFORE implementation, and not
to be loosened to force agreement).

    comparison                        dtype     rtol    atol
    ------------------------------------------------------------
    PyTorch vs NumPy                  float64   1e-12   1e-12
    PyTorch vs NumPy                  float32   1e-5    1e-6
    pure Python vs NumPy              float64   1e-12   1e-12
    educational vs F.sdpa (fused)     float32   1e-5    1e-5

Rationale: all three run the same operation order in exact arithmetic, so at
float64 they should agree to near machine epsilon (~2.2e-16); 1e-12 leaves room
for accumulation over the contraction without hiding a real error. The float32
row is looser because a fused kernel may reassociate the reduction - the same
Tier C phenomenon measured across machines in Phase 1C.

Python floats ARE IEEE-754 doubles, so the pure-Python implementation is
compared against float64, never float32.

Platform and dtype are recorded by test_record_platform_and_dtype below, so a
tolerance can always be read against the environment that produced it.
"""

from __future__ import annotations

import platform
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from alignlab.models import attention_numpy as anp
from alignlab.models import attention_pure as apy
from alignlab.models.attention import causal_mask, scaled_dot_product_attention

# --- pre-declared tolerances ----------------------------------------------
RTOL_F64, ATOL_F64 = 1e-12, 1e-12
RTOL_F32, ATOL_F32 = 1e-5, 1e-6
RTOL_FUSED, ATOL_FUSED = 1e-5, 1e-5

# Tiny dimensions: pure Python is ~1000x slower than the array versions and is
# a correctness reference, never a benchmark.
T_Q, T_K, D_K, D_V = 4, 4, 8, 8

SEED = 20260829


def _inputs_numpy(t_q=T_Q, t_k=T_K, d_k=D_K, d_v=D_V):
    """Deterministic float64 inputs shared by all three implementations."""
    rng = np.random.default_rng(SEED)
    q = rng.standard_normal((t_q, d_k))
    k = rng.standard_normal((t_k, d_k))
    v = rng.standard_normal((t_k, d_v))
    return q, k, v


def _to_lists(a: np.ndarray) -> list[list[float]]:
    return [[float(x) for x in row] for row in a]


def _max_abs_diff(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


# --------------------------------------------------------------------------
# environment record
# --------------------------------------------------------------------------


def test_record_platform_and_dtype() -> None:
    """Not an assertion so much as a record, printed with -s.

    A tolerance is only meaningful alongside the platform and dtype that
    produced it.
    """
    print("\n--- equivalence test environment ---")
    print(f"  platform : {platform.platform()}")
    print(f"  python   : {sys.version.split()[0]}")
    print(f"  torch    : {torch.__version__}")
    print(f"  numpy    : {np.__version__}")
    print(f"  dtypes   : torch float64/float32, numpy float64, pure python float64")
    print(f"  tolerance: f64 rtol={RTOL_F64} atol={ATOL_F64}")
    print(f"             f32 rtol={RTOL_F32} atol={ATOL_F32}")
    assert torch.__version__


# --------------------------------------------------------------------------
# PyTorch vs NumPy
# --------------------------------------------------------------------------


def test_torch_matches_numpy_float64_unmasked() -> None:
    q, k, v = _inputs_numpy()

    ctx_np, w_np = anp.scaled_dot_product_attention(q, k, v)
    ctx_pt, w_pt = scaled_dot_product_attention(
        torch.from_numpy(q), torch.from_numpy(k), torch.from_numpy(v)
    )

    assert ctx_pt.dtype == torch.float64
    np.testing.assert_allclose(ctx_pt.numpy(), ctx_np, rtol=RTOL_F64, atol=ATOL_F64)
    np.testing.assert_allclose(w_pt.numpy(), w_np, rtol=RTOL_F64, atol=ATOL_F64)


def test_torch_matches_numpy_float64_causal() -> None:
    q, k, v = _inputs_numpy()

    ctx_np, w_np = anp.scaled_dot_product_attention(
        q, k, v, mask=anp.causal_mask(T_Q, T_K)
    )
    ctx_pt, w_pt = scaled_dot_product_attention(
        torch.from_numpy(q),
        torch.from_numpy(k),
        torch.from_numpy(v),
        mask=causal_mask(T_Q, key_len=T_K),
    )

    np.testing.assert_allclose(ctx_pt.numpy(), ctx_np, rtol=RTOL_F64, atol=ATOL_F64)
    np.testing.assert_allclose(w_pt.numpy(), w_np, rtol=RTOL_F64, atol=ATOL_F64)


def test_torch_matches_numpy_float32_looser() -> None:
    """float32 needs the looser tolerance - documented, not hidden."""
    q, k, v = _inputs_numpy()
    q32, k32, v32 = (a.astype(np.float32) for a in (q, k, v))

    ctx_np, _ = anp.scaled_dot_product_attention(q32, k32, v32)
    ctx_pt, _ = scaled_dot_product_attention(
        torch.from_numpy(q32), torch.from_numpy(k32), torch.from_numpy(v32)
    )

    np.testing.assert_allclose(ctx_pt.numpy(), ctx_np, rtol=RTOL_F32, atol=ATOL_F32)


# --------------------------------------------------------------------------
# pure Python vs NumPy
# --------------------------------------------------------------------------


def test_pure_python_matches_numpy_unmasked() -> None:
    q, k, v = _inputs_numpy()

    ctx_np, w_np = anp.scaled_dot_product_attention(q, k, v)
    ctx_py, w_py = apy.scaled_dot_product_attention(
        _to_lists(q), _to_lists(k), _to_lists(v)
    )

    np.testing.assert_allclose(np.array(ctx_py), ctx_np, rtol=RTOL_F64, atol=ATOL_F64)
    np.testing.assert_allclose(np.array(w_py), w_np, rtol=RTOL_F64, atol=ATOL_F64)


def test_pure_python_matches_numpy_causal() -> None:
    q, k, v = _inputs_numpy()

    ctx_np, w_np = anp.scaled_dot_product_attention(
        q, k, v, mask=anp.causal_mask(T_Q, T_K)
    )
    ctx_py, w_py = apy.scaled_dot_product_attention(
        _to_lists(q), _to_lists(k), _to_lists(v), mask=apy.causal_mask(T_Q, T_K)
    )

    np.testing.assert_allclose(np.array(ctx_py), ctx_np, rtol=RTOL_F64, atol=ATOL_F64)
    np.testing.assert_allclose(np.array(w_py), w_np, rtol=RTOL_F64, atol=ATOL_F64)


def test_all_three_agree() -> None:
    """The headline claim of WP1, with the actual differences reported."""
    q, k, v = _inputs_numpy()
    mask_np = anp.causal_mask(T_Q, T_K)

    ctx_np, _ = anp.scaled_dot_product_attention(q, k, v, mask=mask_np)
    ctx_pt, _ = scaled_dot_product_attention(
        torch.from_numpy(q),
        torch.from_numpy(k),
        torch.from_numpy(v),
        mask=causal_mask(T_Q, key_len=T_K),
    )
    ctx_py, _ = apy.scaled_dot_product_attention(
        _to_lists(q), _to_lists(k), _to_lists(v), mask=apy.causal_mask(T_Q, T_K)
    )

    d_pt_np = _max_abs_diff(ctx_pt.numpy(), ctx_np)
    d_py_np = _max_abs_diff(ctx_py, ctx_np)
    d_pt_py = _max_abs_diff(ctx_pt.numpy(), ctx_py)

    print(
        f"\n  max|torch-numpy| = {d_pt_np:.3e}"
        f"\n  max|pure -numpy| = {d_py_np:.3e}"
        f"\n  max|torch-pure | = {d_pt_py:.3e}"
        f"\n  tolerance atol  = {ATOL_F64:.1e}"
    )

    assert d_pt_np < ATOL_F64
    assert d_py_np < ATOL_F64
    assert d_pt_py < ATOL_F64


# --------------------------------------------------------------------------
# multi-head: pure Python slicing vs PyTorch reshape
# --------------------------------------------------------------------------


def test_pure_python_multihead_matches_torch_reshape() -> None:
    """Proves the head split is just a slice of the feature axis.

    The PyTorch version uses view+transpose; the pure-Python version slices
    columns. Agreement shows these are the same operation, which is the point
    that makes the reshape stop being mysterious.
    """
    n_heads, d_head = 2, 4
    width = n_heads * d_head
    rng = np.random.default_rng(SEED)
    q = rng.standard_normal((T_Q, width))
    k = rng.standard_normal((T_K, width))
    v = rng.standard_normal((T_K, width))

    merged_py, _ = apy.multi_head_attention(
        _to_lists(q), _to_lists(k), _to_lists(v), n_heads=n_heads,
        mask=apy.causal_mask(T_Q, T_K),
    )

    def split(a: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(a).unsqueeze(0)  # [1, T, width]
        return t.view(1, a.shape[0], n_heads, d_head).transpose(1, 2)

    ctx_pt, _ = scaled_dot_product_attention(
        split(q), split(k), split(v), mask=causal_mask(T_Q, key_len=T_K)
    )
    merged_pt = ctx_pt.transpose(1, 2).contiguous().view(T_Q, width)

    np.testing.assert_allclose(
        merged_pt.numpy(), np.array(merged_py), rtol=RTOL_F64, atol=ATOL_F64
    )


# --------------------------------------------------------------------------
# educational vs PyTorch's fused kernel
# --------------------------------------------------------------------------


def test_educational_matches_fused_sdpa() -> None:
    """Our step-by-step version must match torch's fused implementation.

    F.scaled_dot_product_attention is used in Phase 2 ONLY as a comparison
    target - never as a substitute for the explicit implementation, because it
    hides the mechanics this phase teaches.

    Note the flipped mask convention: ours uses True == keep; torch's
    attn_mask uses True == keep as well for bool masks, but is_causal=True is
    the idiomatic path and is what we compare against here.
    """
    torch.manual_seed(SEED)
    q = torch.randn(2, 4, T_Q, D_K)
    k = torch.randn(2, 4, T_Q, D_K)
    v = torch.randn(2, 4, T_Q, D_V)

    ours, _ = scaled_dot_product_attention(q, k, v, mask=causal_mask(T_Q))
    theirs = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    diff = float((ours - theirs).abs().max())
    print(f"\n  max|ours - F.sdpa| = {diff:.3e} (tolerance {ATOL_FUSED:.1e})")
    torch.testing.assert_close(ours, theirs, rtol=RTOL_FUSED, atol=ATOL_FUSED)


# --------------------------------------------------------------------------
# shared behaviour across implementations
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1.0, 0.5, None])
def test_scale_parameter_consistent_across_implementations(scale) -> None:
    q, k, v = _inputs_numpy()

    ctx_np, _ = anp.scaled_dot_product_attention(q, k, v, scale=scale)
    ctx_pt, _ = scaled_dot_product_attention(
        torch.from_numpy(q), torch.from_numpy(k), torch.from_numpy(v), scale=scale
    )
    ctx_py, _ = apy.scaled_dot_product_attention(
        _to_lists(q), _to_lists(k), _to_lists(v), scale=scale
    )

    np.testing.assert_allclose(ctx_pt.numpy(), ctx_np, rtol=RTOL_F64, atol=ATOL_F64)
    np.testing.assert_allclose(np.array(ctx_py), ctx_np, rtol=RTOL_F64, atol=ATOL_F64)


def test_all_implementations_produce_distributions() -> None:
    q, k, v = _inputs_numpy()
    mask_np = anp.causal_mask(T_Q, T_K)

    _, w_np = anp.scaled_dot_product_attention(q, k, v, mask=mask_np)
    _, w_py = apy.scaled_dot_product_attention(
        _to_lists(q), _to_lists(k), _to_lists(v), mask=apy.causal_mask(T_Q, T_K)
    )

    np.testing.assert_allclose(w_np.sum(axis=-1), 1.0, rtol=1e-12, atol=1e-12)
    for row in w_py:
        assert abs(sum(row) - 1.0) < 1e-12


def test_pure_python_rejects_fully_masked_row() -> None:
    """A query attending to nothing has no defined distribution - fail loudly."""
    q, k, v = _inputs_numpy(t_q=1, t_k=2)
    with pytest.raises(ValueError, match="masked"):
        apy.scaled_dot_product_attention(
            _to_lists(q), _to_lists(k), _to_lists(v), mask=[[False, False]]
        )
