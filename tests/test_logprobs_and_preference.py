"""Tests for Phase 5's preference and log-probability infrastructure.

Phase 6's DPO will be built directly on this arithmetic, so it is tested here
in isolation - where a wrong shift or a wrong denominator is a failing assert
rather than a plausible-looking loss curve.

Offline and CPU-only. Nothing downloads a model or a dataset.
"""

from __future__ import annotations

import math

import pytest
import torch

from alignlab.logprobs import (
    approximate_kl,
    logprob_ratio,
    sequence_kl,
    sequence_logprobs,
    token_kl,
    token_logprobs,
)
from alignlab.masking import IGNORE_INDEX
from alignlab.preference import (
    audit_preferences,
    bradley_terry_loss,
    bradley_terry_probability,
    is_wellformed_preference,
    shares_prompt,
    to_preference_triple,
)

USER = {"role": "user", "content": "What is 2+2?"}
GOOD = {"role": "assistant", "content": "It is 4."}
BAD = {"role": "assistant", "content": "Purple."}


def row(chosen_text="It is 4.", rejected_text="Purple.", sc=8.0, sr=3.0, prompt="What is 2+2?"):
    return {
        "prompt": prompt,
        "chosen": [{"role": "user", "content": prompt},
                   {"role": "assistant", "content": chosen_text}],
        "rejected": [{"role": "user", "content": prompt},
                     {"role": "assistant", "content": rejected_text}],
        "score_chosen": sc,
        "score_rejected": sr,
    }


class _FakeDataset:
    """Minimal indexable stand-in so the audit can be tested without datasets."""

    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, i):
        return self._rows[i]


# --------------------------------------------------------------- log-probabilities


class TestTokenLogprobs:
    def test_shapes_lose_one_position_to_the_shift(self):
        logits = torch.randn(2, 6, 10)
        labels = torch.randint(0, 10, (2, 6))
        picked, mask = token_logprobs(logits, labels)
        assert picked.shape == (2, 5)
        assert mask.shape == (2, 5)

    def test_masked_positions_contribute_zero(self):
        logits = torch.randn(1, 5, 10)
        labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 3, 4, 5]])
        picked, mask = token_logprobs(logits, labels)
        # after the shift, labels are [IGNORE, 3, 4, 5] -> first is masked
        assert not mask[0, 0]
        assert picked[0, 0] == 0.0
        assert mask[0, 1:].all()

    def test_values_match_a_hand_computed_log_softmax(self):
        """Indexing after the shift, spelled out.

        labels        = [IGNORE, 2, 5]        (T = 3)
        shift_labels  = [2, 5]                (T-1 = 2)  <- labels[0] is GONE
        so picked[0] scores label 2 from logits[0], and picked[1] scores
        label 5 from logits[1].

        An earlier version of this test expected picked[0] == 0, treating the
        leading IGNORE as if it survived the shift. It does not - it is
        dropped, not masked. The module was right; the test was wrong, which is
        precisely the off-by-one token_logprobs exists to centralise.
        """
        torch.manual_seed(0)
        logits = torch.randn(1, 3, 7)
        labels = torch.tensor([[IGNORE_INDEX, 2, 5]])
        picked, mask = token_logprobs(logits, labels)

        assert picked.shape == (1, 2)
        assert mask.all()  # both surviving positions are scored

        expected_0 = torch.log_softmax(logits[0, 0].float(), -1)[2]
        expected_1 = torch.log_softmax(logits[0, 1].float(), -1)[5]
        assert picked[0, 0] == pytest.approx(float(expected_0), abs=1e-6)
        assert picked[0, 1] == pytest.approx(float(expected_1), abs=1e-6)

    def test_logprobs_are_negative(self):
        picked, mask = token_logprobs(torch.randn(2, 5, 20), torch.randint(0, 20, (2, 5)))
        assert (picked[mask] < 0).all()

    def test_a_confident_model_gives_logprob_near_zero(self):
        logits = torch.full((1, 2, 5), -50.0)
        logits[0, 0, 3] = 50.0
        picked, _ = token_logprobs(logits, torch.tensor([[0, 3]]))
        assert picked[0, 0] == pytest.approx(0.0, abs=1e-5)

    @pytest.mark.parametrize("bad", [torch.randn(3, 4), torch.randn(2, 3, 4, 5)])
    def test_wrong_logit_rank_rejected(self, bad):
        with pytest.raises(ValueError, match=r"logits must be \[B, T, V\]"):
            token_logprobs(bad, torch.zeros(2, 3, dtype=torch.long))

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValueError, match="disagree"):
            token_logprobs(torch.randn(1, 5, 10), torch.zeros(1, 4, dtype=torch.long))


class TestSequenceLogprobs:
    def test_sum_equals_the_sum_of_token_logprobs(self):
        torch.manual_seed(1)
        logits = torch.randn(2, 6, 12)
        labels = torch.randint(0, 12, (2, 6))
        picked, _ = token_logprobs(logits, labels)
        scores = sequence_logprobs(logits, labels)
        torch.testing.assert_close(scores.sum_logprob, picked.sum(-1))

    def test_mean_divides_by_CONTRIBUTING_tokens_not_labelled_ones(self):
        """The Phase 3 E12 mistake, asserted so it cannot recur."""
        logits = torch.randn(1, 5, 10)
        labels = torch.tensor([[7, 8, 9, IGNORE_INDEX, IGNORE_INDEX]])
        scores = sequence_logprobs(logits, labels)
        # 3 labelled positions, but the shift drops position 0 -> 2 contribute
        assert int((labels != IGNORE_INDEX).sum()) == 3
        assert int(scores.n_tokens[0]) == 2
        assert scores.mean_logprob[0] == pytest.approx(
            float(scores.sum_logprob[0]) / 2, abs=1e-6
        )

    def test_leading_mask_does_not_lose_a_token(self):
        """A prompt-masked prefix is unaffected: position 0 was ignored anyway."""
        logits = torch.randn(1, 5, 10)
        labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 3, 4, 5]])
        scores = sequence_logprobs(logits, labels)
        assert int(scores.n_tokens[0]) == 3

    def test_longer_completions_have_more_negative_sums(self):
        """Length sensitivity of the SUM - the reason mean is also provided."""
        torch.manual_seed(2)
        logits = torch.randn(2, 9, 10)
        labels = torch.full((2, 9), IGNORE_INDEX)
        labels[0, 1:3] = torch.randint(0, 10, (2,))   # 2 tokens
        labels[1, 1:9] = torch.randint(0, 10, (8,))   # 8 tokens
        scores = sequence_logprobs(logits, labels)
        assert scores.sum_logprob[1] < scores.sum_logprob[0]
        assert int(scores.n_tokens[0]) == 2 and int(scores.n_tokens[1]) == 8

    def test_all_masked_sequence_does_not_divide_by_zero(self):
        logits = torch.randn(1, 4, 10)
        labels = torch.full((1, 4), IGNORE_INDEX)
        scores = sequence_logprobs(logits, labels)
        assert int(scores.n_tokens[0]) == 0
        assert torch.isfinite(scores.mean_logprob).all()


class TestLogprobRatio:
    def test_identical_models_give_zero_ratio(self):
        """The DPO implicit reward of an untrained policy is exactly zero."""
        torch.manual_seed(3)
        logits = torch.randn(2, 6, 12)
        labels = torch.randint(0, 12, (2, 6))
        scores = sequence_logprobs(logits, labels)
        ratio = logprob_ratio(scores, scores)
        torch.testing.assert_close(ratio, torch.zeros_like(ratio))

    def test_ratio_is_positive_when_the_policy_prefers_the_sequence(self):
        torch.manual_seed(4)
        labels = torch.randint(0, 8, (1, 5))
        ref_logits = torch.randn(1, 5, 8)
        pol_logits = ref_logits.clone()
        for t in range(4):
            pol_logits[0, t, labels[0, t + 1]] += 5.0
        policy = sequence_logprobs(pol_logits, labels)
        reference = sequence_logprobs(ref_logits, labels)
        assert float(logprob_ratio(policy, reference)[0]) > 0

    def test_length_normalisation_changes_the_ratio(self):
        torch.manual_seed(5)
        labels = torch.randint(0, 8, (1, 7))
        policy = sequence_logprobs(torch.randn(1, 7, 8), labels)
        reference = sequence_logprobs(torch.randn(1, 7, 8), labels)
        assert not torch.allclose(
            logprob_ratio(policy, reference),
            logprob_ratio(policy, reference, length_normalise=True),
        )


class TestKL:
    def test_kl_of_a_distribution_with_itself_is_zero(self):
        torch.manual_seed(6)
        logits = torch.randn(2, 5, 11)
        mask = torch.ones(2, 5, dtype=torch.bool)
        kl = token_kl(logits, logits, mask)
        torch.testing.assert_close(kl, torch.zeros_like(kl), atol=1e-6, rtol=0)

    def test_kl_is_non_negative(self):
        torch.manual_seed(7)
        kl = token_kl(torch.randn(2, 5, 11), torch.randn(2, 5, 11),
                      torch.ones(2, 5, dtype=torch.bool))
        assert (kl >= -1e-6).all()

    def test_kl_is_NOT_symmetric(self):
        """The direction is a modelling choice, not a formality."""
        torch.manual_seed(8)
        p, q = torch.randn(1, 3, 9), torch.randn(1, 3, 9)
        mask = torch.ones(1, 3, dtype=torch.bool)
        assert not torch.allclose(token_kl(p, q, mask), token_kl(q, p, mask))

    def test_kl_grows_as_the_policy_moves_away(self):
        torch.manual_seed(9)
        reference = torch.randn(1, 4, 10)
        mask = torch.ones(1, 4, dtype=torch.bool)
        previous = -1.0
        for scale in (0.0, 0.5, 2.0, 8.0):
            policy = reference + scale * torch.randn(1, 4, 10)
            value = float(token_kl(policy, reference, mask).mean())
            assert value >= previous - 1e-6
            previous = value

    def test_masked_positions_do_not_contribute(self):
        torch.manual_seed(10)
        p, q = torch.randn(1, 4, 10), torch.randn(1, 4, 10)
        mask = torch.tensor([[True, False, True, False]])
        kl = token_kl(p, q, mask)
        assert kl[0, 1] == 0.0 and kl[0, 3] == 0.0

    def test_sequence_kl_uses_the_same_positions_as_the_logprobs(self):
        torch.manual_seed(11)
        labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 3, 4, 5]])
        policy_logits = torch.randn(1, 5, 10)
        reference_logits = torch.randn(1, 5, 10)
        _, n_kl = sequence_kl(policy_logits, reference_logits, labels)
        scores = sequence_logprobs(policy_logits, labels)
        assert int(n_kl[0]) == int(scores.n_tokens[0]) == 3

    def test_shape_mismatch_rejected(self):
        with pytest.raises(ValueError, match="differ in shape"):
            token_kl(torch.randn(1, 3, 9), torch.randn(1, 3, 8),
                     torch.ones(1, 3, dtype=torch.bool))

    def test_k3_estimator_is_non_negative_and_zero_at_equality(self):
        """k3 is the estimator PPO code actually uses."""
        equal = approximate_kl(torch.zeros(5), torch.zeros(5))
        torch.testing.assert_close(equal, torch.zeros(5))
        torch.manual_seed(12)
        value = approximate_kl(torch.randn(1000), torch.randn(1000))
        assert (value >= -1e-6).all()


# ------------------------------------------------------------------- preference


class TestBradleyTerry:
    def test_equal_rewards_give_one_half(self):
        assert bradley_terry_probability(2.0, 2.0) == pytest.approx(0.5)

    def test_only_the_difference_matters(self):
        """Preference data cannot identify an absolute reward scale."""
        a = bradley_terry_probability(1.0, 0.0)
        b = bradley_terry_probability(101.0, 100.0)
        assert a == pytest.approx(b)

    def test_probability_increases_with_the_margin(self):
        values = [bradley_terry_probability(m, 0.0) for m in (-2, -1, 0, 1, 2)]
        assert values == sorted(values)

    def test_it_saturates(self):
        assert bradley_terry_probability(20.0, 0.0) == pytest.approx(1.0, abs=1e-6)

    def test_loss_is_the_negative_log_of_the_probability(self):
        loss = float(bradley_terry_loss(torch.tensor(1.5), torch.tensor(0.5)))
        prob = bradley_terry_probability(1.5, 0.5)
        assert loss == pytest.approx(-math.log(prob), abs=1e-6)

    def test_loss_at_zero_margin_is_log_two(self):
        loss = float(bradley_terry_loss(torch.tensor(0.0), torch.tensor(0.0)))
        assert loss == pytest.approx(math.log(2), abs=1e-6)

    def test_loss_decreases_as_the_margin_grows(self):
        losses = [
            float(bradley_terry_loss(torch.tensor(m), torch.tensor(0.0)))
            for m in (-1.0, 0.0, 1.0, 4.0)
        ]
        assert losses == sorted(losses, reverse=True)


class TestPreferenceConversion:
    def test_splits_prompt_from_both_responses(self):
        out = to_preference_triple(row())
        assert out["prompt"] == [USER]
        assert out["chosen"][0]["content"] == "It is 4."
        assert out["rejected"][0]["content"] == "Purple."

    def test_boundary_whitespace_is_stripped_from_both(self):
        """The Phase 3 tokenization bug applies to chosen AND rejected."""
        out = to_preference_triple(row(chosen_text="\n\nIt is 4.", rejected_text="\nPurple."))
        assert out["chosen"][0]["content"] == "It is 4."
        assert out["rejected"][0]["content"] == "Purple."

    def test_conversation_not_ending_in_assistant_is_rejected(self):
        bad = row()
        bad["chosen"] = [USER, USER]
        assert to_preference_triple(bad)["prompt"] == []

    def test_too_short_is_rejected(self):
        bad = row()
        bad["rejected"] = [USER]
        assert to_preference_triple(bad)["prompt"] == []

    def test_shares_prompt_accepts_a_matching_pair(self):
        assert shares_prompt(row())

    def test_shares_prompt_rejects_a_mismatched_pair(self):
        bad = row()
        bad["rejected"][0] = {"role": "user", "content": "A different question"}
        assert not shares_prompt(bad)


class TestWellformed:
    def test_accepts_a_good_triple(self):
        assert is_wellformed_preference(to_preference_triple(row()))

    def test_rejects_IDENTICAL_responses(self):
        """Asks the model to prefer a string over itself; log-ratio is 0."""
        out = to_preference_triple(row(chosen_text="same", rejected_text="same"))
        assert not is_wellformed_preference(out)

    @pytest.mark.parametrize("text", ["", "   ", "\n\t"])
    def test_rejects_empty_chosen(self, text):
        assert not is_wellformed_preference(to_preference_triple(row(chosen_text=text)))

    @pytest.mark.parametrize("text", ["", "  "])
    def test_rejects_empty_rejected(self, text):
        assert not is_wellformed_preference(to_preference_triple(row(rejected_text=text)))

    def test_rejects_blank_prompt(self):
        assert not is_wellformed_preference(to_preference_triple(row(prompt="   ")))


class TestAudit:
    def test_counts_ties(self):
        data = _FakeDataset([row(sc=5.0, sr=5.0), row(sc=8.0, sr=2.0)])
        audit = audit_preferences(data)
        assert audit.ties == 1
        assert audit.tie_fraction == pytest.approx(0.5)

    def test_counts_inverted_pairs(self):
        audit = audit_preferences(_FakeDataset([row(sc=1.0, sr=9.0)]))
        assert audit.inverted == 1

    def test_counts_identical_responses(self):
        audit = audit_preferences(
            _FakeDataset([row(chosen_text="same", rejected_text="same")])
        )
        assert audit.identical_responses == 1

    def test_counts_empty_responses(self):
        audit = audit_preferences(_FakeDataset([row(rejected_text="")]))
        assert audit.empty_responses == 1

    def test_detects_prompt_mismatch(self):
        bad = row()
        bad["rejected"][0] = {"role": "user", "content": "different"}
        assert audit_preferences(_FakeDataset([bad])).prompt_prefix_mismatches == 1

    def test_measures_length_bias(self):
        data = _FakeDataset([
            row(chosen_text="a" * 100, rejected_text="b" * 10) for _ in range(10)
        ])
        audit = audit_preferences(data)
        assert audit.chosen_longer_fraction == pytest.approx(1.0)
        assert any("LENGTH BIAS" in n for n in audit.notes)

    def test_high_tie_rate_is_flagged(self):
        data = _FakeDataset([row(sc=5.0, sr=5.0) for _ in range(10)])
        assert any("TIES" in n for n in audit_preferences(data).notes)

    def test_clean_data_produces_no_notes(self):
        data = _FakeDataset([
            row(chosen_text="short", rejected_text="a much longer rejected answer")
            for _ in range(10)
        ])
        assert audit_preferences(data).notes == []

    def test_margins_are_measured(self):
        audit = audit_preferences(_FakeDataset([row(sc=8.0, sr=3.0), row(sc=6.0, sr=5.0)]))
        assert audit.margin_mean == pytest.approx(3.0)
        assert audit.margin_median == pytest.approx(3.0)

    def test_to_dict_is_json_friendly(self):
        data = audit_preferences(_FakeDataset([row()])).to_dict()
        assert "tie_fraction" in data and isinstance(data["rows"], int)
