"""Evaluation dashboard — comparison without collapsing.

PROJECT_INSTRUCTIONS: *"Do not collapse fundamentally different metrics into
one arbitrary 'quality score.'"*

That is enforced structurally here: there is no aggregate score, and no
function that produces one. Four phases of this project demonstrated why —
perplexity, stop-token behaviour and preference accuracy repeatedly disagreed,
and each disagreement was the finding. A weighted average would have erased all
four.

WHAT THE DASHBOARD DOES INSTEAD:

  * reports every metric side by side, per model
  * carries each metric's APPLICABLE / NOT_APPLICABLE / NOT_MEASURED status
  * attaches a Wilson interval to every rate
  * says "NOT resolvable at this sample size" when the intervals overlap
  * always shows length beside any preference or quality number
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from alignlab.evals.metrics import (
    APPLICABLE,
    NOT_APPLICABLE,
    NOT_MEASURED,
    compare_proportions,
    wilson_interval,
)
from alignlab.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ModelEvaluation:
    """Everything measured about one model."""

    name: str
    checkpoint: str
    provenance: dict
    perplexity: dict = field(default_factory=dict)      # region -> result dict
    preference: dict | None = None
    generation: dict | None = None
    statuses: dict = field(default_factory=dict)        # metric -> status
    notes: list[str] = field(default_factory=list)

    def status_of(self, metric: str) -> str:
        return self.statuses.get(metric, NOT_MEASURED)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "checkpoint": self.checkpoint,
            "provenance": self.provenance,
            "perplexity": self.perplexity,
            "preference": self.preference,
            "generation": self.generation,
            "statuses": self.statuses,
            "notes": self.notes,
        }


@dataclass
class Dashboard:
    """A full evaluation pass over several models, plus pairwise comparisons."""

    created_at: str
    models: list[ModelEvaluation] = field(default_factory=list)
    comparisons: list[dict] = field(default_factory=list)
    judge_results: list[dict] = field(default_factory=list)
    protocol: dict = field(default_factory=dict)
    lessons: list[str] = field(default_factory=list)

    def by_name(self, name: str) -> ModelEvaluation | None:
        for model in self.models:
            if model.name == name:
                return model
        return None

    def to_dict(self) -> dict:
        return {
            "created_at": self.created_at,
            "protocol": self.protocol,
            "models": [m.to_dict() for m in self.models],
            "comparisons": self.comparisons,
            "judge_results": self.judge_results,
            "lessons": self.lessons,
            # Stated in the artifact itself, not only in documentation.
            "no_aggregate_score": (
                "This report deliberately contains no single quality score. "
                "Phases 3-6 each found two metrics disagreeing, and each "
                "disagreement was the finding."
            ),
        }

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8"
        )
        logger.info("Wrote evaluation dashboard: %s", path)
        return path


def compare_models(
    dashboard: Dashboard, baseline: str, candidate: str
) -> list[dict]:
    """Pairwise comparison, with an explicit status per metric.

    A metric is NOT_APPLICABLE rather than absent when it cannot be compared -
    for example a perplexity measured over a different token region, which
    Phase 3 showed is a different quantity wearing the same name.
    """
    a = dashboard.by_name(baseline)
    b = dashboard.by_name(candidate)
    if a is None or b is None:
        raise KeyError(f"unknown model in comparison: {baseline} vs {candidate}")

    verdicts: list[dict] = []

    # --- perplexity, per region, only where regions match
    for region in sorted(set(a.perplexity) | set(b.perplexity)):
        pa, pb = a.perplexity.get(region), b.perplexity.get(region)
        if pa is None or pb is None:
            verdicts.append({
                "metric": f"perplexity[{region}]",
                "model_a": baseline, "model_b": candidate,
                "status": NOT_MEASURED,
                "note": f"{region} not measured for "
                        f"{baseline if pa is None else candidate}",
            })
            continue
        verdicts.append({
            "metric": f"perplexity[{region}]",
            "model_a": baseline, "model_b": candidate,
            "value_a": pa["perplexity"], "value_b": pb["perplexity"],
            "delta": pb["perplexity"] - pa["perplexity"],
            "status": APPLICABLE,
            "note": (
                f"token-weighted over {pa['n_tokens']}/{pb['n_tokens']} "
                f"{region} tokens; perplexities are comparable only within a region"
            ),
        })

    # --- preference accuracy, both objectives, with intervals
    if a.preference and b.preference:
        for label, key in (
            ("preference_sum (PRIMARY)", "sum_correct"),
            ("preference_mean (diagnostic)", "mean_correct"),
        ):
            verdict = compare_proportions(
                label, baseline, a.preference[key], a.preference["n"],
                candidate, b.preference[key], b.preference["n"],
            )
            data = verdict.to_dict()
            data["length_note"] = (
                f"chosen/rejected tokens: "
                f"{a.preference['chosen_tokens_mean']:.1f}/"
                f"{a.preference['rejected_tokens_mean']:.1f} - "
                f"length must be read alongside this number"
            )
            verdicts.append(data)
    else:
        verdicts.append({
            "metric": "preference", "model_a": baseline, "model_b": candidate,
            "status": NOT_MEASURED, "note": "preference data not evaluated for one side",
        })

    # --- termination, the behavioural metric perplexity misses
    if a.generation and b.generation:
        ta, tb = a.generation["termination"], b.generation["termination"]
        verdict = compare_proportions(
            "stop_token_rate", baseline, ta["emitted_stop_token"], ta["n"],
            candidate, tb["emitted_stop_token"], tb["n"],
        ).to_dict()
        verdict["note"] += (
            " | BEHAVIOURAL metric: Phase 4 found perplexity within 3.5% while "
            "this was 0/4 vs 4/4"
        )
        verdicts.append(verdict)

        verdicts.append({
            "metric": "generated_length",
            "model_a": baseline, "model_b": candidate,
            "value_a": a.generation["length"]["mean"],
            "value_b": b.generation["length"]["mean"],
            "delta": b.generation["length"]["mean"] - a.generation["length"]["mean"],
            "status": APPLICABLE,
            "note": (
                "reported for every comparison - Phases 5 and 6 both found "
                "length driving a number that looked like quality"
            ),
        })
    else:
        verdicts.append({
            "metric": "termination", "model_a": baseline, "model_b": candidate,
            "status": NOT_MEASURED, "note": "generation not run for one side",
        })

    return verdicts


def render_text(dashboard: Dashboard) -> str:
    """A human-readable dashboard. No aggregate score, by design."""
    lines: list[str] = []
    add = lines.append

    add("=" * 108)
    add("ALIGNLAB EVALUATION DASHBOARD")
    add("=" * 108)
    protocol = dashboard.protocol
    add(f"  created      : {dashboard.created_at}")
    for key in ("dataset", "eval_fingerprint", "n_preference_pairs",
                "n_generation_prompts", "judge_model", "git_commit"):
        if protocol.get(key) is not None:
            add(f"  {key:<13}: {protocol[key]}")

    # ---------------------------------------------------------- per model
    add("")
    add("-" * 108)
    add("PER-MODEL METRICS   (no aggregate score - see the note at the end)")
    add("-" * 108)
    header = (f"  {'model':<24} {'ppl[compl]':>11} {'ppl[full]':>10} "
              f"{'pref SUM':>10} {'pref MEAN':>10} {'stop':>7} {'gen len':>8} "
              f"{'distinct2':>10}")
    add(header)

    def fmt(value, spec, fallback="n/a"):
        """Format a value, or a placeholder when it was not measured."""
        return format(value, spec) if value is not None else fallback

    for m in dashboard.models:
        compl = m.perplexity.get("completion", {}).get("perplexity")
        full = m.perplexity.get("full_sequence", {}).get("perplexity")
        pref = m.preference or {}
        gen = m.generation or {}
        term = gen.get("termination", {})
        length = gen.get("length", {})

        sum_acc = pref.get("sum_accuracy")
        mean_acc = pref.get("mean_accuracy")
        stop_cell = (
            f"{term['emitted_stop_token']}/{term['n']}" if term else "n/a"
        )
        add(
            f"  {m.name:<24} "
            f"{fmt(compl, '.3f'):>11} "
            f"{fmt(full, '.3f'):>10} "
            f"{(fmt(100 * sum_acc, '.1f') + '%') if sum_acc is not None else 'n/a':>10} "
            f"{(fmt(100 * mean_acc, '.1f') + '%') if mean_acc is not None else 'n/a':>10} "
            f"{stop_cell:>7} "
            f"{fmt(length.get('mean'), '.1f'):>8} "
            f"{fmt(gen.get('mean_distinct_2'), '.3f'):>10}"
        )

    # ------------------------------------------------- length attribution
    add("")
    add("-" * 108)
    add("LENGTH ATTRIBUTION on the preference set   (Phase 6's decisive check)")
    add("-" * 108)
    for m in dashboard.models:
        if not m.preference:
            continue
        attribution = m.preference.get("length_attribution", {})
        if not attribution:
            continue
        add(f"  {m.name}")
        add(f"    chosen/token {attribution['chosen_logp_per_token']:+.4f}  "
            f"rejected/token {attribution['rejected_logp_per_token']:+.4f}  "
            f"gap {attribution['logp_gap']:+.2f}  "
            f"explained by length {attribution['explained_by_length']:+.2f}  "
            f"residual {attribution['residual']:+.2f}")

    # ------------------------------------------------------- comparisons
    if dashboard.comparisons:
        add("")
        add("-" * 108)
        add("PAIRWISE COMPARISONS   (Wilson 95% intervals; overlap = not resolvable)")
        add("-" * 108)
        for verdict in dashboard.comparisons:
            status = verdict.get("status", APPLICABLE)
            if status != APPLICABLE:
                add(f"  [{status}] {verdict['metric']}: {verdict.get('note','')}")
                continue
            resolvable = verdict.get("resolvable")
            marker = "" if resolvable is None else ("RESOLVED" if resolvable else "unresolved")
            add(f"  {verdict['metric']:<32} {verdict['model_a']} -> {verdict['model_b']}: "
                f"{verdict.get('value_a', float('nan')):.4f} -> "
                f"{verdict.get('value_b', float('nan')):.4f} "
                f"({verdict.get('delta', 0):+.4f}) {marker}")
            if verdict.get("note"):
                add(f"      {verdict['note']}")

    # -------------------------------------------------------------- judge
    if dashboard.judge_results:
        add("")
        add("-" * 108)
        add("LLM-AS-JUDGE   (NOT ground truth)")
        add("-" * 108)
        for judged in dashboard.judge_results:
            add(f"  {judged['model_a']} vs {judged['model_b']}  (judge: {judged['judge_model']})")
            add(f"    a={judged['a_wins']} b={judged['b_wins']} tie={judged['ties']} "
                f"inconsistent={judged['inconsistent']} of {judged['n_pairs']}")
            add(f"    position bias rate: {100*judged['position_bias_rate']:.1f}%")
            rate = judged.get("win_rate_b")
            if rate:
                add(f"    win rate (B, decided only): {rate['point']:.3f} "
                    f"[{rate['low']:.3f}, {rate['high']:.3f}] -> {rate['interpretation']}")
            else:
                add("    win rate: NOT COMPUTABLE - no consistently decided pairs")

    # ------------------------------------------------------------ lessons
    if dashboard.lessons:
        add("")
        add("-" * 108)
        add("EVALUATION-DESIGN LESSONS CARRIED FORWARD")
        add("-" * 108)
        for lesson in dashboard.lessons:
            add(f"  - {lesson}")

    add("")
    add("=" * 108)
    add("NO AGGREGATE SCORE IS PRODUCED. Phases 3-6 each found two metrics")
    add("disagreeing, and each disagreement was the finding. A weighted average")
    add("would have erased all four.")
    add("=" * 108)
    return "\n".join(lines)
