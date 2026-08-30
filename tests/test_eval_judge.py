"""Tests for the LLM-as-judge, using a scripted fake judge.

No model is loaded. The point is to verify the ORDER-HANDLING and AGGREGATION
logic, which is where a judge harness actually goes wrong: mapping a swapped
presentation back to the original labels, refusing to count position-flips as
preference, and not manufacturing precision from ties.

A judge harness that has only ever been run against a real model is a harness
whose position-bias correction nobody has checked.
"""

from __future__ import annotations

import pytest

from alignlab.evals.judge import (
    JUDGE_TEMPLATE,
    JudgeResult,
    judge_pairs,
    parse_verdict,
)


class ScriptedJudge:
    """A fake judge whose behaviour is chosen per test.

    ``mode``:
        "always_first"   always picks whichever answer is shown FIRST
                         (pure position bias)
        "always_second"  always picks whichever is shown SECOND
        "prefers_text"   always picks the answer containing `target`
        "always_tie"     always ties
        "garbage"        returns unparseable text
    """

    def __init__(self, mode: str, target: str = ""):
        self.mode = mode
        self.target = target
        self.calls = 0

    # the judge_pairs helper calls generate() through _ask; we stub _ask instead
    def reply(self, rendered_prompt: str) -> str:
        self.calls += 1
        if self.mode == "always_first":
            return "A"
        if self.mode == "always_second":
            return "B"
        if self.mode == "always_tie":
            return "TIE"
        if self.mode == "garbage":
            return "I think perhaps neither is clearly stronger, honestly"
        if self.mode == "prefers_text":
            # Which slot holds the target string?
            a_start = rendered_prompt.index("Answer A:")
            b_start = rendered_prompt.index("Answer B:")
            slot_a = rendered_prompt[a_start:b_start]
            return "A" if self.target in slot_a else "B"
        raise AssertionError(f"unknown mode {self.mode}")


@pytest.fixture
def patched_ask(monkeypatch):
    """Route judge._ask to the scripted judge."""
    holder = {}

    def fake_ask(judge_model, judge_tokenizer, prompt, device, max_new_tokens=8):
        return holder["judge"].reply(prompt)

    monkeypatch.setattr("alignlab.evals.judge._ask", fake_ask)
    return holder


def run(holder, judge, n=6, answers_a=None, answers_b=None, seed=42):
    holder["judge"] = judge
    prompts = [f"question {i}" for i in range(n)]
    a = answers_a or [f"alpha answer {i}" for i in range(n)]
    b = answers_b or [f"bravo answer {i}" for i in range(n)]
    return judge_pairs(
        judge_model=None, judge_tokenizer=None,
        prompts=prompts, answers_a=a, answers_b=b,
        device=None, model_a="A-model", model_b="B-model",
        judge_model_id="scripted", seed=seed,
    )


class TestParseVerdict:
    @pytest.mark.parametrize("text,expected", [
        ("A", "a"), ("B", "b"), ("TIE", "tie"),
        ("  a  ", "a"), ("Answer: B", "b"), ("tie", "tie"),
    ])
    def test_parses(self, text, expected):
        assert parse_verdict(text) == expected

    def test_unparseable_is_not_guessed(self):
        """Silently mapping a bad reply to a tie would invent data."""
        assert parse_verdict("hmm, hard to say") == "unparsed"
        assert parse_verdict("") == "unparsed"


class TestPositionBias:
    def test_a_pure_position_judge_produces_zero_decided_verdicts(self, patched_ask):
        """The headline property: position bias must NOT become a win rate.

        A judge that always picks whichever answer is shown first agrees with
        itself on NO pair once the orders are swapped, so every pair is
        inconsistent and none counts as a preference.
        """
        result = run(patched_ask, ScriptedJudge("always_first"), n=6)
        assert result.inconsistent == 6
        assert result.decided == 0
        assert result.position_bias_rate == pytest.approx(1.0)
        assert result.win_rate_b() is None

    def test_always_second_is_equally_caught(self, patched_ask):
        result = run(patched_ask, ScriptedJudge("always_second"), n=6)
        assert result.inconsistent == 6
        assert result.decided == 0

    def test_a_content_judge_is_consistent_across_orders(self, patched_ask):
        """A judge that reads the text agrees with itself regardless of order."""
        result = run(patched_ask, ScriptedJudge("prefers_text", target="bravo"), n=6)
        assert result.inconsistent == 0
        assert result.b_wins == 6
        assert result.a_wins == 0
        assert result.position_bias_rate == 0.0

    def test_content_judge_favouring_a(self, patched_ask):
        result = run(patched_ask, ScriptedJudge("prefers_text", target="alpha"), n=6)
        assert result.a_wins == 6 and result.b_wins == 0

    def test_verdicts_are_normalised_to_original_labels(self, patched_ask):
        """Both presentations must be reported in terms of the ORIGINAL a/b."""
        result = run(patched_ask, ScriptedJudge("prefers_text", target="bravo"), n=3)
        for v in result.verdicts:
            assert v["verdict_first"] == "b"
            assert v["verdict_second"] == "b"
            assert v["consistent"] is True

    def test_every_pair_is_judged_exactly_twice(self, patched_ask):
        judge = ScriptedJudge("prefers_text", target="bravo")
        run(patched_ask, judge, n=5)
        assert judge.calls == 10


class TestAggregation:
    def test_ties_are_counted_not_split(self, patched_ask):
        """Splitting ties as half-wins would manufacture precision."""
        result = run(patched_ask, ScriptedJudge("always_tie"), n=8)
        assert result.ties == 8
        assert result.a_wins == 0 and result.b_wins == 0
        assert result.decided == 0
        assert result.win_rate_b() is None

    def test_unparsed_replies_are_counted_as_inconsistent(self, patched_ask):
        result = run(patched_ask, ScriptedJudge("garbage"), n=4)
        assert result.inconsistent == 4
        assert result.settings["unparsed_replies"] == 4

    def test_win_rate_is_over_DECIDED_pairs_only(self, patched_ask):
        result = run(patched_ask, ScriptedJudge("prefers_text", target="bravo"), n=10)
        rate = result.win_rate_b()
        assert rate["point"] == pytest.approx(1.0)
        assert rate["n"] == 10

    def test_win_rate_carries_an_interval_and_a_verdict(self, patched_ask):
        result = run(patched_ask, ScriptedJudge("prefers_text", target="bravo"), n=10)
        rate = result.win_rate_b()
        assert "low" in rate and "high" in rate
        assert "interpretation" in rate
        assert rate["excludes_chance"] is True

    def test_small_samples_report_as_unresolvable(self):
        """A 3-2 split on 5 pairs must not read as a finding."""
        result = JudgeResult(
            model_a="a", model_b="b", judge_model="j",
            n_pairs=5, a_wins=2, b_wins=3, ties=0, inconsistent=0,
        )
        rate = result.win_rate_b()
        assert rate["excludes_chance"] is False
        assert "NOT resolvable" in rate["interpretation"]


class TestProvenanceAndLimitations:
    def test_the_prompt_is_recorded(self, patched_ask):
        result = run(patched_ask, ScriptedJudge("always_tie"), n=2)
        assert result.judge_prompt == JUDGE_TEMPLATE
        assert "impartial evaluator" in result.judge_system

    def test_settings_record_both_orders_and_greedy(self, patched_ask):
        result = run(patched_ask, ScriptedJudge("always_tie"), n=2)
        assert result.settings["both_orders"] is True
        assert result.settings["decoding"] == "greedy"

    def test_limitations_are_attached_to_every_result(self, patched_ask):
        """Not just in prose - the JSON carries them."""
        result = run(patched_ask, ScriptedJudge("always_tie"), n=2)
        assert result.limitations
        joined = " ".join(result.limitations)
        assert "NOT ground truth" in joined
        assert "SAME model family" in joined

    def test_to_dict_exposes_position_bias(self, patched_ask):
        data = run(patched_ask, ScriptedJudge("always_first"), n=4).to_dict()
        assert data["position_bias_rate"] == pytest.approx(1.0)
        assert data["decided"] == 0

    def test_seed_changes_presentation_order_not_the_outcome(self, patched_ask):
        """A content-reading judge gives the same answer under any seed."""
        a = run(patched_ask, ScriptedJudge("prefers_text", target="bravo"), n=8, seed=1)
        b = run(patched_ask, ScriptedJudge("prefers_text", target="bravo"), n=8, seed=999)
        assert a.b_wins == b.b_wins == 8


class TestInputValidation:
    def test_mismatched_lengths_rejected(self, patched_ask):
        patched_ask["judge"] = ScriptedJudge("always_tie")
        with pytest.raises(ValueError, match="same length"):
            judge_pairs(
                None, None, ["p1", "p2"], ["a1"], ["b1", "b2"],
                None, "A", "B", "scripted",
            )
