"""Tests for the Phase 7 evaluation metrics.

Every metric is checked against a hand-computed value or an analytic property.
That is the point of keeping them as pure functions: a metric you can only
exercise by loading a 1.5B model is a metric nobody checks.

Several tests encode findings from earlier phases as executable regressions -
if a future change makes perplexity comparable across regions, or lets a
preference metric be reported without its lengths, these fail.
"""

from __future__ import annotations

import math

import pytest

from alignlab.evals.metrics import (
    APPLICABLE,
    NOT_MEASURED,
    ComparisonVerdict,
    Interval,
    PreferenceStats,
    compare_proportions,
    difference_is_resolvable,
    distinct_n,
    length_stats,
    max_ngram_repeat,
    perplexity,
    perplexity_from_totals,
    structural_checks,
    termination_stats,
    wilson_interval,
)


class TestPerplexity:
    def test_exp_of_the_mean_nll(self):
        assert perplexity(0.0) == pytest.approx(1.0)
        assert perplexity(1.0) == pytest.approx(math.e)
        assert perplexity(math.log(50)) == pytest.approx(50.0)

    def test_negative_nll_is_rejected_as_a_sign_error(self):
        with pytest.raises(ValueError, match="non-negative"):
            perplexity(-0.5)

    def test_token_weighted_not_example_weighted(self):
        """Totals in, so a long answer is not weighted like a short one."""
        result = perplexity_from_totals(total_nll=200.0, n_tokens=100, region="completion")
        assert result.mean_nll == pytest.approx(2.0)
        assert result.perplexity == pytest.approx(math.exp(2.0))
        assert result.n_tokens == 100

    def test_zero_tokens_is_rejected(self):
        with pytest.raises(ValueError, match="n_tokens must be positive"):
            perplexity_from_totals(1.0, 0, "completion")

    def test_region_is_carried_and_gates_comparison(self):
        """Phase 3's lesson: prompt and completion perplexity are not the same."""
        completion = perplexity_from_totals(100.0, 50, "completion")
        prompt = perplexity_from_totals(100.0, 50, "prompt")
        assert completion.comparable_to(perplexity_from_totals(80.0, 40, "completion"))
        assert not completion.comparable_to(prompt)

    def test_identical_numbers_different_regions_are_not_comparable(self):
        a = perplexity_from_totals(100.0, 50, "completion")
        b = perplexity_from_totals(100.0, 50, "full_sequence")
        assert a.perplexity == b.perplexity
        assert not a.comparable_to(b)  # equal values, still incomparable


class TestLengthStats:
    def test_basic_statistics(self):
        stats = length_stats([10, 20, 30, 40, 50])
        assert stats.n == 5
        assert stats.mean == pytest.approx(30.0)
        assert stats.median == pytest.approx(30.0)
        assert stats.minimum == 10 and stats.maximum == 50

    def test_empty_is_safe(self):
        stats = length_stats([])
        assert stats.n == 0 and stats.mean == 0.0

    def test_unsorted_input_is_handled(self):
        assert length_stats([50, 10, 30]).median == pytest.approx(30.0)

    def test_p90(self):
        assert length_stats(list(range(1, 11))).p90 == 10


class TestTermination:
    @staticmethod
    def gen(n_generated, stop=True, text="hello world"):
        return {"n_generated": n_generated, "emitted_stop_token": stop, "text": text}

    def test_counts_natural_termination(self):
        stats = termination_stats(
            [self.gen(50), self.gen(60), self.gen(200, stop=False)], max_new_tokens=200
        )
        assert stats.n == 3
        assert stats.terminated == 2
        assert stats.hit_length_cap == 1
        assert stats.termination_rate == pytest.approx(2 / 3)

    def test_stop_token_rate_is_separate_from_termination(self):
        """A model can stop by running out of budget without emitting a stop token."""
        stats = termination_stats(
            [self.gen(200, stop=False), self.gen(200, stop=False)], max_new_tokens=200
        )
        assert stats.terminated == 0
        assert stats.stop_token_rate == 0.0

    def test_phase_4_scenario_is_representable(self):
        """Full SFT 4/4 vs PEFT 0/4 - the gap perplexity did not show."""
        sft = termination_stats([self.gen(75) for _ in range(4)], max_new_tokens=200)
        peft = termination_stats(
            [self.gen(200, stop=False) for _ in range(4)], max_new_tokens=200
        )
        assert sft.stop_token_rate == 1.0
        assert peft.stop_token_rate == 0.0

    def test_empty_output_is_counted(self):
        stats = termination_stats([self.gen(3, text="   ")], max_new_tokens=200)
        assert stats.empty == 1


class TestRepetition:
    def test_distinct_n_is_one_for_unique_text(self):
        assert distinct_n("a b c d e", 2) == pytest.approx(1.0)

    def test_distinct_n_detects_repetition(self):
        """Phase 3's base model emitted '-unstyled' to the token cap."""
        degenerate = " ".join(["-unstyled"] * 30)
        assert distinct_n(degenerate, 2) < 0.1

    def test_short_text_is_not_penalised(self):
        assert distinct_n("hi", 2) == 1.0

    def test_max_ngram_repeat_counts_the_worst_offender(self):
        assert max_ngram_repeat("a b c a b c a b c", 3) == 3

    def test_max_ngram_repeat_on_short_text(self):
        assert max_ngram_repeat("a", 3) == 0


class TestStructuralChecks:
    def test_detects_a_numbered_list(self):
        assert structural_checks("1. first\n2. second")["has_numbered_list"]

    def test_detects_a_bullet_list(self):
        assert structural_checks("- one\n- two")["has_bullet_list"]

    def test_detects_a_code_fence(self):
        assert structural_checks("```python\nx=1\n```")["has_code_fence"]

    def test_counts_placeholders(self):
        """The SFT model's emails used [Name] and [Date]."""
        text = "Hi [Name], I can't make [Date]."
        assert structural_checks(text)["placeholder_count"] == 2

    def test_empty_is_flagged(self):
        assert structural_checks("   ")["is_empty"]

    def test_reports_are_descriptive_not_scored(self):
        """No single quality number is produced - by design."""
        keys = set(structural_checks("hello"))
        assert "score" not in keys and "quality" not in keys


class TestPreferenceStats:
    @staticmethod
    def phase6():
        """The measured Phase 6 evaluation-set numbers."""
        return PreferenceStats(
            n=184, sum_correct=86, mean_correct=108,
            chosen_tokens_mean=271.7391, rejected_tokens_mean=242.2065,
            chosen_logp_mean=-291.1196, rejected_logp_mean=-261.6510,
        )

    def test_both_accuracies_are_available(self):
        stats = self.phase6()
        assert stats.sum_accuracy == pytest.approx(86 / 184)
        assert stats.mean_accuracy == pytest.approx(108 / 184)

    def test_sum_and_mean_disagree_on_the_real_data(self):
        """Phase 5/6's central finding, as a regression test."""
        stats = self.phase6()
        assert stats.sum_accuracy < 0.5 < stats.mean_accuracy

    def test_token_gap_is_reported(self):
        assert self.phase6().token_gap == pytest.approx(29.5326, abs=1e-3)

    def test_length_attribution_reproduces_the_phase_6_arithmetic(self):
        """Per token chosen is BETTER; the SUM gap is a length artifact."""
        attribution = self.phase6().length_attribution()
        assert attribution["chosen_logp_per_token"] == pytest.approx(-1.0713, abs=1e-3)
        assert attribution["rejected_logp_per_token"] == pytest.approx(-1.0803, abs=1e-3)
        # chosen is better per token
        assert attribution["chosen_logp_per_token"] > attribution["rejected_logp_per_token"]
        # length over-explains the gap, leaving a positive residual for chosen
        assert attribution["explained_by_length"] < attribution["logp_gap"]
        assert attribution["residual"] > 0

    def test_to_dict_always_carries_lengths_and_attribution(self):
        """Reporting preference accuracy without lengths would hide the effect."""
        data = self.phase6().to_dict()
        assert "chosen_tokens_mean" in data and "rejected_tokens_mean" in data
        assert "length_attribution" in data
        assert "sum_accuracy" in data and "mean_accuracy" in data

    def test_zero_examples_is_safe(self):
        empty = PreferenceStats(0, 0, 0, 0.0, 0.0, 0.0, 0.0)
        assert empty.sum_accuracy == 0.0
        assert empty.length_attribution()["residual"] == 0.0


class TestWilsonInterval:
    def test_contains_the_point_estimate(self):
        interval = wilson_interval(50, 100)
        assert interval.low < interval.point < interval.high
        assert interval.point == pytest.approx(0.5)

    def test_stays_inside_zero_one_at_the_extremes(self):
        """Where the normal approximation would escape [0, 1]."""
        for successes, n in ((0, 10), (10, 10), (1, 200)):
            interval = wilson_interval(successes, n)
            assert 0.0 <= interval.low <= interval.high <= 1.0

    def test_narrows_as_n_grows(self):
        widths = [wilson_interval(n // 2, n).width for n in (20, 100, 1000)]
        assert widths == sorted(widths, reverse=True)

    def test_known_value(self):
        """Wilson 95% for 50/100 is approximately [0.4038, 0.5962]."""
        interval = wilson_interval(50, 100, 0.95)
        assert interval.low == pytest.approx(0.4038, abs=1e-3)
        assert interval.high == pytest.approx(0.5962, abs=1e-3)

    def test_phase_6_sample_size_cannot_resolve_chance(self):
        """184 pairs at 46.7% - the interval straddles 0.5."""
        interval = wilson_interval(86, 184)
        assert not interval.excludes(0.5)

    def test_a_clear_effect_does_exclude_chance(self):
        assert wilson_interval(150, 184).excludes(0.5)

    def test_invalid_inputs_rejected(self):
        with pytest.raises(ValueError, match="n must be positive"):
            wilson_interval(0, 0)
        with pytest.raises(ValueError, match="outside"):
            wilson_interval(11, 10)

    def test_untabulated_level_raises_rather_than_defaulting(self):
        with pytest.raises(ValueError, match="not tabulated"):
            wilson_interval(5, 10, level=0.975)


class TestComparison:
    def test_overlapping_intervals_are_not_resolvable(self):
        a = wilson_interval(86, 184)
        b = wilson_interval(88, 184)
        assert not difference_is_resolvable(a, b)

    def test_separated_intervals_are_resolvable(self):
        assert difference_is_resolvable(wilson_interval(20, 184), wilson_interval(160, 184))

    def test_compare_proportions_reports_both_intervals(self):
        verdict = compare_proportions("pref_sum", "SFT", 86, 184, "DPO", 86, 184)
        assert isinstance(verdict, ComparisonVerdict)
        assert verdict.delta == pytest.approx(0.0)
        assert verdict.resolvable is False
        assert "NOT resolvable" in verdict.note

    def test_phase_6_comparison_is_honestly_unresolvable(self):
        """The DPO null: identical rates, and the note says so."""
        verdict = compare_proportions("pref_sum", "SFT", 86, 184, "DPO b=0.1", 86, 184)
        assert verdict.status == APPLICABLE
        assert verdict.resolvable is False

    def test_a_large_difference_is_reported_as_resolvable(self):
        verdict = compare_proportions("stop_rate", "PEFT", 0, 40, "SFT", 40, 40)
        assert verdict.resolvable is True
        assert "do not overlap" in verdict.note

    def test_verdict_serialises(self):
        verdict = compare_proportions("m", "a", 1, 10, "b", 2, 10)
        data = verdict.to_dict()
        assert data["metric"] == "m" and "interval_a" in data


class TestIntervalHelpers:
    def test_excludes(self):
        interval = Interval(point=0.9, low=0.8, high=0.95, level=0.95, method="wilson", n=100)
        assert interval.excludes(0.5)
        assert not interval.excludes(0.85)

    def test_width(self):
        interval = Interval(0.5, 0.4, 0.6, 0.95, "wilson", 100)
        assert interval.width == pytest.approx(0.2)


class TestDashboard:
    """The dashboard must refuse to collapse metrics into one score."""

    @staticmethod
    def _dashboard():
        from datetime import datetime, timezone
        from alignlab.evals.report import Dashboard, ModelEvaluation

        d = Dashboard(created_at=datetime.now(timezone.utc).isoformat())
        for name, stop, gen_len in (("SFT", 4, 93.8), ("PEFT", 0, 200.0)):
            d.models.append(ModelEvaluation(
                name=name, checkpoint=f"/x/{name}", provenance={},
                perplexity={"completion": {"perplexity": 7.2, "n_tokens": 100,
                                           "region": "completion"}},
                preference={"n": 184, "sum_correct": 86, "mean_correct": 108,
                            "sum_accuracy": 86 / 184, "mean_accuracy": 108 / 184,
                            "chosen_tokens_mean": 271.7,
                            "rejected_tokens_mean": 242.2,
                            "length_attribution": {}},
                generation={"termination": {"n": 4, "emitted_stop_token": stop,
                                            "terminated": stop, "hit_length_cap": 4 - stop,
                                            "empty": 0},
                            "length": {"mean": gen_len}, "mean_distinct_2": 0.9},
            ))
        return d

    def test_no_aggregate_score_is_produced(self):
        """Structural: there is no function that collapses metrics."""
        from alignlab.evals import report

        names = [n for n in dir(report) if "score" in n.lower()]
        assert names == []
        assert "no_aggregate_score" in self._dashboard().to_dict()

    def test_phase_4_stop_gap_is_resolvable_even_at_n_equals_4(self):
        """0/4 vs 4/4 is a real separation; the harness must say so."""
        from alignlab.evals.report import compare_models

        verdicts = compare_models(self._dashboard(), "SFT", "PEFT")
        stop = next(v for v in verdicts if v["metric"] == "stop_token_rate")
        assert stop["resolvable"] is True

    def test_identical_preference_rates_are_unresolvable(self):
        from alignlab.evals.report import compare_models

        verdicts = compare_models(self._dashboard(), "SFT", "PEFT")
        pref = next(v for v in verdicts if v["metric"].startswith("preference_sum"))
        assert pref["resolvable"] is False
        assert "NOT resolvable" in pref["note"]

    def test_length_is_always_reported_with_preference(self):
        from alignlab.evals.report import compare_models

        verdicts = compare_models(self._dashboard(), "SFT", "PEFT")
        pref = next(v for v in verdicts if v["metric"].startswith("preference_sum"))
        assert "length_note" in pref
        assert any(v["metric"] == "generated_length" for v in verdicts)

    def test_missing_metric_is_NOT_MEASURED_not_absent(self):
        from alignlab.evals.metrics import NOT_MEASURED
        from alignlab.evals.report import compare_models

        d = self._dashboard()
        d.models[1].generation = None
        verdicts = compare_models(d, "SFT", "PEFT")
        term = next(v for v in verdicts if v["metric"] == "termination")
        assert term["status"] == NOT_MEASURED

    def test_unknown_model_raises(self):
        from alignlab.evals.report import compare_models

        with pytest.raises(KeyError, match="unknown model"):
            compare_models(self._dashboard(), "SFT", "nope")

    def test_render_text_runs_and_states_the_no_score_rule(self):
        from alignlab.evals.report import compare_models, render_text

        d = self._dashboard()
        d.comparisons = compare_models(d, "SFT", "PEFT")
        text = render_text(d)
        assert "NO AGGREGATE SCORE IS PRODUCED" in text
        assert "SFT" in text and "PEFT" in text


class TestProvenanceFingerprints:
    """eval-full-001 shipped with dataset_fingerprint null in every record."""

    def _section(self, max_examples=200):
        from types import SimpleNamespace

        return SimpleNamespace(max_examples=max_examples)

    def _info(self):
        return {
            "name": "HuggingFaceH4/no_robots",
            "revision": None,
            "eval_split": "test",
            "eval_rows": 200,
            "eval_rows_dropped": 3,
            "eval_fingerprint": {"sha256": "b3bc775a" + "0" * 56},
        }

    def test_populated_family_records_its_own_dataset(self):
        from alignlab.evaluate import _fingerprint_record

        record = _fingerprint_record(self._section(), self._info())
        assert record["status"] == APPLICABLE
        assert record["dataset"] == "HuggingFaceH4/no_robots"
        assert record["split"] == "test"
        assert record["n_rows"] == 200
        assert record["sha256"].startswith("b3bc775a")

    def test_no_field_is_silently_null_when_applicable(self):
        from alignlab.evaluate import _fingerprint_record

        record = _fingerprint_record(self._section(), self._info())
        # revision is legitimately None (the dataset is unpinned); every other
        # field must carry a value rather than a silent null.
        nulls = [k for k, v in record.items() if v is None and k != "revision"]
        assert nulls == [], f"silently null provenance fields: {nulls}"

    def test_disabled_family_says_NOT_MEASURED_not_null(self):
        from alignlab.evaluate import _fingerprint_record

        record = _fingerprint_record(self._section(), {})
        assert record == {"status": NOT_MEASURED}
        assert "sha256" not in record

    def test_the_two_families_are_distinguishable(self):
        from alignlab.evaluate import _fingerprint_record

        pref = dict(self._info())
        pref["name"] = "HuggingFaceH4/ultrafeedback_binarized"
        pref["eval_split"] = "test_prefs"
        pref["eval_fingerprint"] = {"sha256": "deadbeef" + "0" * 56}

        a = _fingerprint_record(self._section(), self._info())
        b = _fingerprint_record(self._section(), pref)
        assert a["dataset"] != b["dataset"]
        assert a["sha256"] != b["sha256"]


class TestDashboardTextIsPortable:
    """Phase 6 lesson, locked down: print() under Windows cp1252 raises on
    non-ASCII, and evaluate.py prints the rendered dashboard to stdout."""

    def test_rendered_dashboard_is_ascii(self):
        from alignlab.evals.report import Dashboard, render_text

        dashboard = Dashboard(
            created_at="2026-08-30T00:00:00+00:00",
            protocol={"dataset": "x", "judge_model": "y", "n_preference_pairs": 1},
            lessons=["a lesson"],
        )
        render_text(dashboard).encode("ascii")

    def test_the_committed_phase7_dashboard_is_ascii(self):
        """The artefact that was actually printed, not just a synthetic one."""
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "docs" / "phase7" / \
            "eval-full-001_dashboard.txt"
        if not path.is_file():
            pytest.skip("phase 7 dashboard evidence not present")
        path.read_text(encoding="utf-8").encode("ascii")
