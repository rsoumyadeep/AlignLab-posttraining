"""Tests for the first-principles DPO loss.

The algorithm is tested here in isolation, where a wrong sign or a swapped
branch is a failing assert rather than a training run that produces a
plausible-looking loss curve and the wrong model.

Several tests assert properties the DERIVATION implies - that the loss at
initialisation is exactly log 2, that beta scales the logit linearly, that
swapping chosen and rejected negates the logit. Those are the checks that would
catch a misread of the algebra, which unit tests of the code against itself
would not.

CPU, float64 where exactness is asserted.
"""

from __future__ import annotations

import math

import pytest
import torch

from alignlab.dpo import (
    LOSS_AT_INIT,
    dpo_loss,
    implicit_rewards,
    preference_accuracy,
    verify_reference_is_frozen,
    verify_zero_reward_at_init,
)
from alignlab.logprobs import SequenceScores


def scores(sums, means=None, tokens=None) -> SequenceScores:
    """Build a SequenceScores directly, so the loss is tested without a model."""
    total = torch.tensor(sums, dtype=torch.float64)
    n = torch.tensor(tokens if tokens is not None else [10] * len(sums))
    mean = (
        torch.tensor(means, dtype=torch.float64)
        if means is not None
        else total / n.clamp(min=1)
    )
    return SequenceScores(sum_logprob=total, mean_logprob=mean, n_tokens=n)


class TestImplicitRewards:
    def test_reward_is_beta_times_the_log_ratio(self):
        pc, pr = scores([-10.0]), scores([-20.0])
        rc, rr = scores([-12.0]), scores([-18.0])
        chosen, rejected = implicit_rewards(pc, pr, rc, rr, beta=0.5)
        assert float(chosen[0]) == pytest.approx(0.5 * (-10.0 - -12.0))
        assert float(rejected[0]) == pytest.approx(0.5 * (-20.0 - -18.0))

    def test_zero_when_policy_equals_reference(self):
        s = scores([-10.0, -20.0])
        chosen, rejected = implicit_rewards(s, s, s, s, beta=0.1)
        assert torch.all(chosen == 0)
        assert torch.all(rejected == 0)

    def test_beta_scales_rewards_linearly(self):
        pc, pr = scores([-10.0]), scores([-20.0])
        rc, rr = scores([-12.0]), scores([-18.0])
        a, _ = implicit_rewards(pc, pr, rc, rr, beta=0.1)
        b, _ = implicit_rewards(pc, pr, rc, rr, beta=0.5)
        assert float(b[0]) == pytest.approx(5 * float(a[0]))

    def test_length_normalise_uses_the_mean_branch(self):
        pc = scores([-100.0], means=[-2.0], tokens=[50])
        pr = scores([-20.0], means=[-4.0], tokens=[5])
        rc = scores([-110.0], means=[-2.2], tokens=[50])
        rr = scores([-25.0], means=[-5.0], tokens=[5])

        summed, _ = implicit_rewards(pc, pr, rc, rr, beta=1.0)
        meaned, _ = implicit_rewards(pc, pr, rc, rr, beta=1.0, length_normalise=True)
        assert float(summed[0]) == pytest.approx(10.0)
        assert float(meaned[0]) == pytest.approx(0.2)


class TestDPOLoss:
    def test_loss_at_initialisation_is_exactly_log_two(self):
        """π == π_ref ⟹ both rewards 0 ⟹ L = −log σ(0) = log 2."""
        s = scores([-10.0, -20.0, -30.0])
        loss, stats = dpo_loss(s, s, s, s, beta=0.1)
        assert float(loss) == pytest.approx(math.log(2), abs=1e-12)
        assert float(loss) == pytest.approx(LOSS_AT_INIT, abs=1e-12)
        assert stats.reward_margin == pytest.approx(0.0)

    def test_loss_falls_when_the_policy_prefers_chosen(self):
        rc, rr = scores([-10.0]), scores([-10.0])
        neutral, _ = dpo_loss(scores([-10.0]), scores([-10.0]), rc, rr, beta=0.1)
        better, _ = dpo_loss(scores([-8.0]), scores([-12.0]), rc, rr, beta=0.1)
        assert float(better) < float(neutral)

    def test_loss_rises_when_the_policy_prefers_rejected(self):
        rc, rr = scores([-10.0]), scores([-10.0])
        neutral, _ = dpo_loss(scores([-10.0]), scores([-10.0]), rc, rr, beta=0.1)
        worse, _ = dpo_loss(scores([-12.0]), scores([-8.0]), rc, rr, beta=0.1)
        assert float(worse) > float(neutral)

    def test_swapping_chosen_and_rejected_negates_the_logit(self):
        """A property of the algebra, not of the implementation."""
        pc, pr = scores([-8.0]), scores([-12.0])
        rc, rr = scores([-10.0]), scores([-10.0])
        forward, _ = dpo_loss(pc, pr, rc, rr, beta=0.2)
        swapped, _ = dpo_loss(pr, pc, rr, rc, beta=0.2)
        # L(z) = -logsigmoid(z), L(-z) = -logsigmoid(-z); L(z)+L(-z) = -log σ(z)σ(-z)
        chosen, rejected = implicit_rewards(pc, pr, rc, rr, beta=0.2)
        z = float(chosen[0] - rejected[0])
        assert float(forward) == pytest.approx(-math.log(1 / (1 + math.exp(-z))))
        assert float(swapped) == pytest.approx(-math.log(1 / (1 + math.exp(z))))

    def test_beta_scales_the_logit_linearly(self):
        pc, pr = scores([-8.0]), scores([-12.0])
        rc, rr = scores([-10.0]), scores([-10.0])
        _, a = dpo_loss(pc, pr, rc, rr, beta=0.1)
        _, b = dpo_loss(pc, pr, rc, rr, beta=0.4)
        assert b.reward_margin == pytest.approx(4 * a.reward_margin)

    def test_larger_beta_saturates_sooner(self):
        """The mechanism behind beta as a KL strength.

        Here the log-ratio difference is fixed at
            (−8 − −10) − (−12 − −10) = 2 − (−2) = 4
        so the sigmoid argument is exactly 4β and the loss is −logsigmoid(4β).
        Asserting those exact values rather than a threshold, because a
        threshold is a guess: an earlier version of this test asserted
        loss < 0.01 at β=1, and the true value is 0.0181.
        """
        pc, pr = scores([-8.0]), scores([-12.0])
        rc, rr = scores([-10.0]), scores([-10.0])

        for beta in (0.01, 0.1, 1.0):
            loss = float(dpo_loss(pc, pr, rc, rr, beta=beta)[0])
            expected = -math.log(1 / (1 + math.exp(-4 * beta)))
            assert loss == pytest.approx(expected, abs=1e-9)

        losses = [float(dpo_loss(pc, pr, rc, rr, beta=b)[0]) for b in (0.01, 0.1, 1.0)]
        assert losses == sorted(losses, reverse=True)
        # β=1 is effectively saturated; β=0.01 is barely distinguishable from
        # the log-2 starting point.
        assert losses[0] == pytest.approx(0.6733, abs=1e-3)
        assert losses[-1] == pytest.approx(0.0181, abs=1e-3)

    def test_numerically_stable_for_a_confidently_wrong_policy(self):
        """log(sigmoid(x)) would underflow to -inf here; logsigmoid does not."""
        pc, pr = scores([-1000.0]), scores([0.0])
        rc, rr = scores([0.0]), scores([0.0])
        loss, _ = dpo_loss(pc, pr, rc, rr, beta=1.0)
        assert torch.isfinite(loss)
        assert float(loss) > 100

    def test_gradients_flow_to_the_policy_only(self):
        policy_c = torch.tensor([-8.0], requires_grad=True)
        policy_r = torch.tensor([-12.0], requires_grad=True)
        ref_c = torch.tensor([-10.0])
        ref_r = torch.tensor([-10.0])
        n = torch.tensor([10])

        loss, _ = dpo_loss(
            SequenceScores(policy_c, policy_c / n, n),
            SequenceScores(policy_r, policy_r / n, n),
            SequenceScores(ref_c, ref_c / n, n),
            SequenceScores(ref_r, ref_r / n, n),
            beta=0.1,
        )
        loss.backward()
        assert policy_c.grad is not None and policy_r.grad is not None
        # pushing chosen UP reduces the loss -> negative gradient on chosen
        assert float(policy_c.grad) < 0
        assert float(policy_r.grad) > 0

    def test_negative_beta_rejected(self):
        s = scores([-10.0])
        with pytest.raises(ValueError, match="beta must be positive"):
            dpo_loss(s, s, s, s, beta=-0.1)

    def test_zero_beta_rejected(self):
        s = scores([-10.0])
        with pytest.raises(ValueError, match="beta must be positive"):
            dpo_loss(s, s, s, s, beta=0.0)


class TestStats:
    def test_reward_accuracy_counts_pairs_the_policy_ranks_correctly(self):
        pc = scores([-8.0, -12.0, -9.0])
        pr = scores([-12.0, -8.0, -10.0])
        rc = scores([-10.0, -10.0, -10.0])
        rr = scores([-10.0, -10.0, -10.0])
        _, stats = dpo_loss(pc, pr, rc, rr, beta=0.1)
        assert stats.reward_accuracy == pytest.approx(2 / 3)

    def test_stats_record_token_counts_for_length_tracking(self):
        pc = scores([-80.0], tokens=[40])
        pr = scores([-20.0], tokens=[10])
        rc, rr = scores([-80.0], tokens=[40]), scores([-20.0], tokens=[10])
        _, stats = dpo_loss(pc, pr, rc, rr, beta=0.1)
        assert stats.chosen_tokens == pytest.approx(40.0)
        assert stats.rejected_tokens == pytest.approx(10.0)

    def test_stats_are_json_friendly(self):
        s = scores([-10.0])
        _, stats = dpo_loss(s, s, s, s, beta=0.1)
        data = stats.to_dict()
        assert isinstance(data["loss"], float)
        assert "reward_accuracy" in data


class TestPreferenceAccuracy:
    def test_sum_and_mean_can_disagree(self):
        """The Phase 5 finding, reproduced as a unit test.

        A long chosen response with a BETTER per-token mean can still lose on
        the SUM, purely because it has more tokens.
        """
        chosen = scores([-100.0], means=[-2.0], tokens=[50])
        rejected = scores([-20.0], means=[-4.0], tokens=[5])

        by_sum = preference_accuracy(chosen, rejected)
        by_mean = preference_accuracy(chosen, rejected, length_normalise=True)
        assert float(by_sum[0]) == 0.0   # SUM prefers the short rejected one
        assert float(by_mean[0]) == 1.0  # MEAN prefers the chosen one

    def test_sum_is_the_default(self):
        chosen = scores([-100.0], means=[-2.0], tokens=[50])
        rejected = scores([-20.0], means=[-4.0], tokens=[5])
        assert torch.equal(
            preference_accuracy(chosen, rejected),
            preference_accuracy(chosen, rejected, length_normalise=False),
        )

    def test_accuracy_over_a_batch(self):
        chosen = scores([-8.0, -12.0, -9.0])
        rejected = scores([-12.0, -8.0, -10.0])
        assert float(preference_accuracy(chosen, rejected).mean()) == pytest.approx(2 / 3)


class TestReferenceFrozen:
    def test_accepts_a_properly_frozen_model(self):
        model = torch.nn.Linear(4, 4)
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        report = verify_reference_is_frozen(model)
        assert report["frozen"] and report["n_trainable"] == 0

    def test_rejects_a_trainable_reference(self):
        model = torch.nn.Linear(4, 4).eval()
        with pytest.raises(RuntimeError, match="trainable parameters"):
            verify_reference_is_frozen(model)

    def test_rejects_a_reference_left_in_train_mode(self):
        model = torch.nn.Linear(4, 4)
        for p in model.parameters():
            p.requires_grad_(False)
        model.train()
        with pytest.raises(RuntimeError, match="train\\(\\) mode"):
            verify_reference_is_frozen(model)


class TestZeroRewardAtInit:
    def test_reports_zero_and_log_two(self):
        s = scores([-10.0, -20.0])
        report = verify_zero_reward_at_init(s, s, s, s, beta=0.1)
        assert report["rewards_zero"]
        assert report["loss_matches_log2"]
        assert report["max_abs_chosen_reward"] == 0.0

    def test_detects_a_mismatched_reference(self):
        policy = scores([-10.0])
        wrong_reference = scores([-11.0])
        report = verify_zero_reward_at_init(
            policy, policy, wrong_reference, wrong_reference, beta=0.1
        )
        assert not report["rewards_zero"]


class TestDPOConfigComposition:
    """The Phase 6 config must encode the pre-registered decisions."""

    @staticmethod
    def _compose(overrides=None):
        from hydra import compose, initialize_config_dir
        from alignlab.config_schema import register_configs
        from alignlab.dpo_train import CONFIG_DIR

        register_configs()
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            return compose(config_name="dpo", overrides=overrides or [])

    def test_composes_with_defaults(self):
        cfg = self._compose()
        assert cfg.model.id == "Qwen/Qwen2.5-1.5B"
        assert cfg.sft_run_name == "sft-qwen1p5b-noRobots-001"

    def test_objective_defaults_to_SUM_the_published_one(self):
        """length_normalise=True would silently optimise a different objective."""
        assert self._compose().dpo.length_normalise is False

    def test_beta_default_is_the_preregistered_reference_point(self):
        assert self._compose().dpo.beta == 0.1

    def test_all_preregistered_betas_compose(self):
        for beta in (0.01, 0.1, 0.5):
            assert self._compose([f"dpo.beta={beta}"]).dpo.beta == pytest.approx(beta)

    def test_ties_are_kept_unchanged_from_phase_5(self):
        assert self._compose().drop_ties is False

    def test_seed_is_unchanged_across_phases(self):
        assert self._compose().reproducibility.seed == 42

    def test_preference_splits_are_the_prefs_ones(self):
        cfg = self._compose()
        assert cfg.preference_train_split == "train_prefs"
        assert cfg.preference_eval_split == "test_prefs"

    def test_storage_guard_defaults_on(self):
        assert self._compose().storage_guard is True

    def test_smoke_profile_caps_steps(self):
        assert self._compose(["dpo=smoke"]).dpo.max_steps == 8

    def test_dpo_learning_rate_is_far_below_sft(self):
        """DPO starts from a tuned model; 2e-5 would degrade it quickly."""
        assert self._compose().dpo.learning_rate < 1e-5


class TestMatchesPaperReferenceImplementation:
    """Our loss vs the reference code printed in the DPO paper's Appendix B.

    The paper's HTML full text was ACTUALLY INSPECTED (arxiv.org/html/2305.18290v3,
    2026-08-30). Appendix B gives this reference implementation verbatim:

        pi_logratios  = pi_yw_logps - pi_yl_logps
        ref_logratios = ref_yw_logps - ref_yl_logps
        losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios))
        rewards = beta * (pi_logps - ref_logps).detach()

    Ours groups the terms differently - per-response implicit rewards, then
    their difference:

        r_w = beta * (pi_w - ref_w)
        r_l = beta * (pi_l - ref_l)
        losses = -F.logsigmoid(r_w - r_l)

    These are algebraically identical:
        beta*((pi_w - pi_l) - (ref_w - ref_l))
      = beta*(pi_w - ref_w) - beta*(pi_l - ref_l)

    Asserting it numerically is what turns "algebraically identical" from a
    claim into a check. Our grouping was chosen independently, before the
    appendix was read, because it makes the per-response implicit reward a
    first-class quantity that can be logged.
    """

    @staticmethod
    def paper_reference_loss(pi_w, pi_l, ref_w, ref_l, beta):
        """Verbatim structure from the paper's Appendix B."""
        pi_logratios = pi_w - pi_l
        ref_logratios = ref_w - ref_l
        return -torch.nn.functional.logsigmoid(beta * (pi_logratios - ref_logratios))

    @pytest.mark.parametrize("beta", [0.01, 0.1, 0.5, 1.0])
    def test_loss_matches_the_paper_reference(self, beta):
        torch.manual_seed(0)
        pi_w = torch.randn(8, dtype=torch.float64) * 10 - 40
        pi_l = torch.randn(8, dtype=torch.float64) * 10 - 40
        ref_w = torch.randn(8, dtype=torch.float64) * 10 - 40
        ref_l = torch.randn(8, dtype=torch.float64) * 10 - 40
        n = torch.full((8,), 20)

        ours, _ = dpo_loss(
            SequenceScores(pi_w, pi_w / n, n),
            SequenceScores(pi_l, pi_l / n, n),
            SequenceScores(ref_w, ref_w / n, n),
            SequenceScores(ref_l, ref_l / n, n),
            beta=beta,
        )
        theirs = self.paper_reference_loss(pi_w, pi_l, ref_w, ref_l, beta).mean()
        assert float(ours) == pytest.approx(float(theirs), abs=1e-12)

    def test_implicit_rewards_match_the_paper_definition(self):
        """Paper: rewards = beta * (pi_logps - ref_logps)."""
        torch.manual_seed(1)
        pi_w = torch.randn(4, dtype=torch.float64)
        pi_l = torch.randn(4, dtype=torch.float64)
        ref_w = torch.randn(4, dtype=torch.float64)
        ref_l = torch.randn(4, dtype=torch.float64)
        n = torch.full((4,), 10)
        beta = 0.3

        chosen, rejected = implicit_rewards(
            SequenceScores(pi_w, pi_w / n, n),
            SequenceScores(pi_l, pi_l / n, n),
            SequenceScores(ref_w, ref_w / n, n),
            SequenceScores(ref_l, ref_l / n, n),
            beta=beta,
        )
        torch.testing.assert_close(chosen, beta * (pi_w - ref_w))
        torch.testing.assert_close(rejected, beta * (pi_l - ref_l))

    def test_paper_default_beta_is_in_our_preregistered_set(self):
        """The paper uses beta=0.1 by default and beta=0.5 for TL;DR.

        Both are in our pre-registered {0.01, 0.1, 0.5} - which was committed
        BEFORE the appendix was read. Recorded as independent corroboration,
        not as a reason to change the sweep.
        """
        assert 0.1 in (0.01, 0.1, 0.5)
        assert 0.5 in (0.01, 0.1, 0.5)
