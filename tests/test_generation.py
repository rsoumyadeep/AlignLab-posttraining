"""Decoding strategies and autoregressive generation.

The strategies are pure functions of the logits, so they are tested against
HAND-COMPUTED cases with no model involved. That is deliberate: a test that
runs a model and checks "something plausible came out" would not catch an
off-by-one in the top-p cutoff, which is exactly the kind of bug that lives
here.
"""

from __future__ import annotations

import math

import pytest
import torch

from alignlab.models.generation import (
    apply_temperature,
    beam_search,
    generate,
    greedy_select,
    sample_from_logits,
    top_k_filter,
    top_p_filter,
)
from alignlab.models.shapes import assert_shape
from alignlab.models.transformer import DecoderOnlyTransformer, TransformerConfig

VOCAB = 32
NEG_INF = float("-inf")


def make_model() -> DecoderOnlyTransformer:
    torch.manual_seed(0)
    cfg = TransformerConfig(
        vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=4, max_seq_len=64
    )
    return DecoderOnlyTransformer(cfg).eval()


# ==========================================================================
# greedy
# ==========================================================================


def test_greedy_picks_the_argmax() -> None:
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0]])
    assert greedy_select(logits).item() == 1


def test_greedy_is_deterministic() -> None:
    model = make_model()
    prompt = torch.randint(0, VOCAB, (1, 3))
    a = generate(model, prompt, max_new_tokens=8, greedy=True)
    b = generate(model, prompt, max_new_tokens=8, greedy=True)
    assert torch.equal(a, b)


# ==========================================================================
# temperature
# ==========================================================================


def test_temperature_one_is_a_noop() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    torch.testing.assert_close(apply_temperature(logits, 1.0), logits)


def test_low_temperature_sharpens_high_temperature_flattens() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]])

    sharp = torch.softmax(apply_temperature(logits, 0.5), dim=-1)
    normal = torch.softmax(logits, dim=-1)
    flat = torch.softmax(apply_temperature(logits, 5.0), dim=-1)

    assert sharp.max() > normal.max() > flat.max()
    # flattening moves the distribution toward uniform = 1/3
    assert abs(float(flat.max()) - 1 / 3) < abs(float(normal.max()) - 1 / 3)


def test_temperature_preserves_ranking() -> None:
    """Temperature rescales confidence; it never reorders tokens."""
    logits = torch.randn(1, VOCAB)
    base = logits.argsort(descending=True)
    for t in (0.1, 0.5, 2.0, 10.0):
        assert torch.equal(apply_temperature(logits, t).argsort(descending=True), base)


def test_temperature_zero_is_argmax() -> None:
    """The T -> 0 limit, handled explicitly to avoid dividing by zero."""
    logits = torch.tensor([[1.0, 5.0, 3.0]])
    out = apply_temperature(logits, 0.0)
    assert out.argmax().item() == 1
    assert torch.softmax(out, dim=-1)[0, 1].item() == pytest.approx(1.0)


def test_negative_temperature_raises() -> None:
    with pytest.raises(ValueError, match="temperature"):
        apply_temperature(torch.zeros(1, 3), -1.0)


# ==========================================================================
# top-k  (hand-computed)
# ==========================================================================


def test_top_k_keeps_exactly_k_tokens() -> None:
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
    filtered = top_k_filter(logits, k=2)

    # keeps 5.0 and 4.0; masks 3.0, 2.0, 1.0
    assert filtered[0, 1].item() == 5.0
    assert filtered[0, 4].item() == 4.0
    assert filtered[0, 0] == NEG_INF
    assert filtered[0, 2] == NEG_INF
    assert filtered[0, 3] == NEG_INF
    assert int(torch.isfinite(filtered).sum()) == 2


def test_top_k_larger_than_vocab_keeps_everything() -> None:
    logits = torch.randn(1, 5)
    torch.testing.assert_close(top_k_filter(logits, k=99), logits)


def test_top_k_one_equals_greedy() -> None:
    logits = torch.randn(1, VOCAB)
    filtered = top_k_filter(logits, k=1)
    assert int(torch.isfinite(filtered).sum()) == 1
    assert filtered.argmax() == logits.argmax()


def test_top_k_zero_raises() -> None:
    with pytest.raises(ValueError, match="top_k"):
        top_k_filter(torch.zeros(1, 5), k=0)


# ==========================================================================
# top-p  (hand-computed - this is where off-by-one bugs live)
# ==========================================================================


def test_top_p_keeps_the_crossing_token() -> None:
    """Hand-computed nucleus.

    Logits chosen so the softmax is exactly [0.5, 0.3, 0.2] (to 3dp):
    with p=0.7, cumulative goes 0.5, 0.8 -> the SECOND token crosses 0.7 and
    must be KEPT, so the nucleus is {0, 1} and only token 2 is masked.

    Dropping the crossing token instead would retain less than p and, for a
    peaked distribution, could empty the set entirely.
    """
    probs = torch.tensor([0.5, 0.3, 0.2])
    logits = probs.log().unsqueeze(0)

    filtered = top_p_filter(logits, p=0.7)

    assert torch.isfinite(filtered[0, 0])
    assert torch.isfinite(filtered[0, 1]), "the crossing token must be kept"
    assert filtered[0, 2] == NEG_INF


def test_top_p_one_keeps_everything() -> None:
    logits = torch.randn(1, 8)
    filtered = top_p_filter(logits, p=1.0)
    assert bool(torch.isfinite(filtered).all())


def test_top_p_never_empties_the_nucleus() -> None:
    """Even a tiny p must keep at least the top token, or sampling is undefined."""
    for p in (0.01, 0.001):
        filtered = top_p_filter(torch.randn(1, VOCAB), p=p)
        assert int(torch.isfinite(filtered).sum()) >= 1


def test_top_p_is_adaptive_where_top_k_is_not() -> None:
    """The reason nucleus sampling exists.

    A PEAKED distribution should keep few tokens; a FLAT one should keep many.
    top-k would keep exactly k in both cases.
    """
    peaked = torch.tensor([[10.0] + [0.0] * (VOCAB - 1)])
    flat = torch.zeros(1, VOCAB)

    n_peaked = int(torch.isfinite(top_p_filter(peaked, p=0.9)).sum())
    n_flat = int(torch.isfinite(top_p_filter(flat, p=0.9)).sum())

    assert n_peaked < n_flat, f"peaked kept {n_peaked}, flat kept {n_flat}"
    assert n_peaked == 1


def test_top_p_out_of_range_raises() -> None:
    for bad in (0.0, 1.5, -0.2):
        with pytest.raises(ValueError, match="top_p"):
            top_p_filter(torch.zeros(1, 4), p=bad)


# ==========================================================================
# sampling pipeline
# ==========================================================================


def test_sampling_never_selects_a_filtered_token() -> None:
    """Masked tokens get probability exactly 0, so must never be drawn."""
    logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
    gen = torch.Generator().manual_seed(0)

    for _ in range(200):
        token = sample_from_logits(logits, temperature=1.0, top_k=2, generator=gen)
        assert token.item() in (0, 1)


def test_sampling_is_reproducible_with_a_seeded_generator() -> None:
    logits = torch.randn(1, VOCAB)
    a = sample_from_logits(logits, generator=torch.Generator().manual_seed(7))
    b = sample_from_logits(logits, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


def test_very_low_temperature_approaches_greedy() -> None:
    logits = torch.randn(1, VOCAB)
    gen = torch.Generator().manual_seed(0)
    token = sample_from_logits(logits, temperature=0.01, generator=gen)
    assert token.item() == int(logits.argmax())


# ==========================================================================
# generation loop
# ==========================================================================


def test_generate_appends_the_requested_number_of_tokens() -> None:
    model = make_model()
    prompt = torch.randint(0, VOCAB, (2, 5))
    out = generate(model, prompt, max_new_tokens=7, greedy=True)

    assert_shape(out, (2, 12), "generated")
    assert torch.equal(out[:, :5], prompt), "the prompt must be preserved"


def test_generate_stops_at_eos() -> None:
    model = make_model()
    prompt = torch.randint(0, VOCAB, (1, 2))

    with torch.no_grad():
        logits, _ = model(prompt)
        first = int(logits[0, -1].argmax())

    out = generate(model, prompt, max_new_tokens=20, greedy=True, eos_token_id=first)
    assert out.shape[1] == prompt.shape[1] + 1


def test_generated_tokens_are_in_vocab_range() -> None:
    model = make_model()
    out = generate(
        model, torch.randint(0, VOCAB, (1, 3)), max_new_tokens=15, top_k=5, top_p=0.9
    )
    assert int(out.min()) >= 0
    assert int(out.max()) < VOCAB


# ==========================================================================
# beam search
# ==========================================================================


def test_beam_search_returns_a_sequence_of_the_right_length() -> None:
    model = make_model()
    prompt = torch.randint(0, VOCAB, (1, 3))
    out = beam_search(model, prompt, max_new_tokens=6, beam_width=3)

    assert_shape(out, (1, 9), "beam output")
    assert torch.equal(out[0, :3], prompt[0])


def test_beam_width_one_equals_greedy() -> None:
    """A single beam has nothing to compare against, so it degenerates."""
    model = make_model()
    prompt = torch.randint(0, VOCAB, (1, 3))

    beam = beam_search(model, prompt, max_new_tokens=6, beam_width=1)
    greedy = generate(model, prompt, max_new_tokens=6, greedy=True)

    assert torch.equal(beam, greedy)


def test_beam_search_scores_at_least_as_well_as_greedy() -> None:
    """The point of beam search: it explores, so it should not score worse.

    Score is total log-probability of the generated continuation.
    """
    model = make_model()
    prompt = torch.randint(0, VOCAB, (1, 4))

    def sequence_logprob(seq: torch.Tensor) -> float:
        with torch.no_grad():
            logits, _ = model(seq[:, :-1])
        log_probs = torch.log_softmax(logits, dim=-1)
        targets = seq[:, 1:]
        picked = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return float(picked[:, prompt.shape[1] - 1 :].sum())

    beam = beam_search(model, prompt, max_new_tokens=8, beam_width=4)
    greedy = generate(model, prompt, max_new_tokens=8, greedy=True)

    assert sequence_logprob(beam) >= sequence_logprob(greedy) - 1e-4


def test_beam_search_rejects_batch_greater_than_one() -> None:
    model = make_model()
    with pytest.raises(ValueError, match="batch size 1"):
        beam_search(model, torch.randint(0, VOCAB, (2, 3)), max_new_tokens=2)


# ==========================================================================
# regression: the top-p unsort bug found by experiment E9
# ==========================================================================


def test_top_p_unsorts_correctly() -> None:
    """REGRESSION TEST for a real bug, found by E9 and not by the tests above.

    top_p_filter sorts, thresholds, then must UNSORT back to vocabulary order.
    The original implementation used ``sorted_idx.argsort()`` as the scatter
    index - the inverse permutation, which is right for a gather and wrong for
    a scatter. Surviving logits were written to the WRONG vocabulary positions,
    so the nucleus contained arbitrary tokens instead of the most probable ones.

    Every earlier top-p test missed it because they used ALREADY-DESCENDING
    logits, where sorted_idx is the identity and argsort(identity) is also the
    identity - making the bug invisible. This test uses deliberately unsorted
    input.

    The visible symptom in E9: top-p=0.9 produced ungrammatical output with a
    total log-probability of -674 against -25 for plain temperature sampling.
    """
    logits = torch.tensor([[1.0, 5.0, 2.0, 4.0, 0.0]])  # NOT sorted
    # softmax ~ [0.013, 0.693, 0.035, 0.255, 0.005]; p=0.9 needs {1, 3}

    filtered = top_p_filter(logits, p=0.9)

    assert filtered[0, 1].item() == 5.0, "highest logit must be kept, in place"
    assert filtered[0, 3].item() == 4.0, "second highest must be kept, in place"
    for masked in (0, 2, 4):
        assert filtered[0, masked] == NEG_INF, f"index {masked} should be masked"


def test_top_p_preserves_values_at_their_original_indices() -> None:
    """Whatever survives must keep BOTH its value and its vocabulary index."""
    torch.manual_seed(0)
    logits = torch.randn(1, 20)

    filtered = top_p_filter(logits, p=0.8)

    kept = torch.isfinite(filtered)
    torch.testing.assert_close(filtered[kept], logits[kept])
    # and the kept set must be exactly the top-scoring tokens
    n_kept = int(kept.sum())
    expected = set(logits.argsort(descending=True)[0, :n_kept].tolist())
    assert set(torch.nonzero(kept[0]).flatten().tolist()) == expected


def test_top_p_and_top_k_agree_when_they_select_the_same_count() -> None:
    """Cross-check between the two filters - they must not disagree."""
    torch.manual_seed(1)
    logits = torch.randn(1, 30)

    p_filtered = top_p_filter(logits, p=0.75)
    n = int(torch.isfinite(p_filtered).sum())
    k_filtered = top_k_filter(logits, k=n)

    assert torch.equal(torch.isfinite(p_filtered), torch.isfinite(k_filtered))
