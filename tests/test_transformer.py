"""Decoder-only Transformer: shapes, causality, gradients, parameter accounting."""

from __future__ import annotations

import pytest
import torch

from alignlab.models.shapes import assert_shape
from alignlab.models.transformer import (
    DecoderBlock,
    DecoderOnlyTransformer,
    TransformerConfig,
)

VOCAB, D_MODEL, LAYERS, HEADS, T, B = 64, 64, 2, 4, 8, 2


def make_config(**overrides) -> TransformerConfig:
    base = dict(
        vocab_size=VOCAB,
        d_model=D_MODEL,
        n_layers=LAYERS,
        n_heads=HEADS,
        max_seq_len=32,
    )
    base.update(overrides)
    return TransformerConfig(**base)


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(3)


# ==========================================================================
# config
# ==========================================================================


def test_config_rejects_indivisible_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        make_config(d_model=10, n_heads=4)


def test_config_defaults_to_mha() -> None:
    cfg = make_config()
    assert cfg.n_kv_heads == cfg.n_heads
    assert cfg.head_dim == D_MODEL // HEADS


def test_config_rejects_bad_kv_head_grouping() -> None:
    with pytest.raises(ValueError, match="divisible"):
        make_config(n_heads=4, n_kv_heads=3)


# ==========================================================================
# shapes and the forward path
# ==========================================================================


def test_forward_produces_logits_of_the_right_shape() -> None:
    model = DecoderOnlyTransformer(make_config())
    ids = torch.randint(0, VOCAB, (B, T))

    logits, loss = model(ids)
    assert_shape(logits, (B, T, VOCAB), "logits")
    assert loss is None


def test_loss_is_computed_when_targets_are_given() -> None:
    model = DecoderOnlyTransformer(make_config())
    ids = torch.randint(0, VOCAB, (B, T))

    _, loss = model(ids, targets=ids)
    assert loss is not None
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_untrained_loss_is_near_ln_vocab() -> None:
    """A randomly initialised LM should be roughly uniform over the vocab.

    ln(V) is the sanity target. Being far BELOW it at init would suggest the
    logits are not actually uniform - e.g. a leak or a bad init.
    """
    import math

    cfg = make_config(vocab_size=1000, tie_embeddings=False)
    model = DecoderOnlyTransformer(cfg)
    ids = torch.randint(0, 1000, (4, 16))

    _, loss = model(ids, targets=ids)
    assert abs(float(loss) - math.log(1000)) < 0.5


def test_sequence_longer_than_max_raises() -> None:
    model = DecoderOnlyTransformer(make_config(max_seq_len=8))
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        model(torch.randint(0, VOCAB, (1, 9)))


def test_block_is_shape_preserving() -> None:
    block = DecoderBlock(make_config())
    x = torch.randn(B, T, D_MODEL)
    assert_shape(block(x), (B, T, D_MODEL), "block output")


# ==========================================================================
# causality of the FULL model - the property that matters most
# ==========================================================================


def test_full_model_is_causal() -> None:
    """Changing token t must not alter logits at any position < t.

    This is the end-to-end causality guarantee. A single mis-shaped mask
    anywhere in the stack would break it, so this catches what per-layer tests
    might miss.
    """
    model = DecoderOnlyTransformer(make_config()).eval()

    ids = torch.randint(0, VOCAB, (1, T))
    modified = ids.clone()
    modified[0, T - 1] = (modified[0, T - 1] + 1) % VOCAB  # change the LAST token

    with torch.no_grad():
        a, _ = model(ids)
        b, _ = model(modified)

    torch.testing.assert_close(a[:, : T - 1], b[:, : T - 1], rtol=1e-5, atol=1e-5)
    assert not torch.allclose(a[:, T - 1], b[:, T - 1])


# ==========================================================================
# residual / pre-norm structure
# ==========================================================================


def test_gradients_reach_the_first_layer() -> None:
    """Pre-norm keeps a clean additive path from embedding to output.

    If the residual structure were broken, early-layer gradients would be
    vanishing or absent.
    """
    model = DecoderOnlyTransformer(make_config(n_layers=4))
    ids = torch.randint(0, VOCAB, (B, T))

    _, loss = model(ids, targets=ids)
    loss.backward()

    first = model.blocks[0]
    for name, param in first.named_parameters():
        assert param.grad is not None, f"block0.{name} has no gradient"
        assert torch.isfinite(param.grad).all()
        assert param.grad.abs().sum() > 0, f"block0.{name} gradient is all zero"


def test_every_parameter_receives_gradient() -> None:
    model = DecoderOnlyTransformer(make_config())
    _, loss = model(torch.randint(0, VOCAB, (B, T)), targets=torch.randint(0, VOCAB, (B, T)))
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"


# ==========================================================================
# weight tying and parameter accounting
# ==========================================================================


def test_weight_tying_shares_the_matrix() -> None:
    model = DecoderOnlyTransformer(make_config(tie_embeddings=True))
    assert model.lm_head.weight is model.token_embedding.weight


def test_weight_tying_saves_exactly_vocab_times_dmodel() -> None:
    tied = DecoderOnlyTransformer(make_config(tie_embeddings=True)).num_parameters()
    untied = DecoderOnlyTransformer(make_config(tie_embeddings=False)).num_parameters()
    assert untied - tied == VOCAB * D_MODEL


def test_parameter_breakdown_sums_correctly() -> None:
    model = DecoderOnlyTransformer(make_config(tie_embeddings=True))
    b = model.parameter_breakdown()

    assert b["all_blocks"] == b["per_block"] * LAYERS
    assert b["lm_head"] == 0  # tied
    assert b["total"] == b["embedding"] + b["all_blocks"] + b["final_norm"]


@pytest.mark.parametrize("n_kv_heads", [1, 2, 4])
def test_model_works_for_every_attention_variant(n_kv_heads: int) -> None:
    model = DecoderOnlyTransformer(make_config(n_kv_heads=n_kv_heads))
    logits, loss = model(
        torch.randint(0, VOCAB, (B, T)), targets=torch.randint(0, VOCAB, (B, T))
    )
    assert_shape(logits, (B, T, VOCAB), "logits")
    assert torch.isfinite(loss)


def test_fewer_kv_heads_means_fewer_parameters() -> None:
    mha = DecoderOnlyTransformer(make_config(n_kv_heads=4)).num_parameters()
    mqa = DecoderOnlyTransformer(make_config(n_kv_heads=1)).num_parameters()
    assert mqa < mha
