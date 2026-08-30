"""AlignLab evaluation subsystem (Phase 7).

Built on the Phase 1 skeleton in ``alignlab.evaluation`` (EvalResult,
Evaluator, EvalReport, run_evaluation), which already had the right shape:
results carry the manifest of the run that produced them, and an evaluator
that raises is recorded as NOT_TESTED rather than aborting the pass.

Phase 7 adds the concrete metrics, the evaluators, an LLM-as-judge, and a
comparison layer that refuses to collapse different metrics into one score.


NOT to be confused with ``alignlab.evaluation`` (singular), which is the Phase 1
evaluator protocol that train.py drives. This package is the metrics subsystem.
See that module's docstring for why both names survive.
"""

from alignlab.evals.metrics import (
    APPLICABLE,
    NOT_APPLICABLE,
    NOT_MEASURED,
    ComparisonVerdict,
    Interval,
    LengthStats,
    PerplexityResult,
    PreferenceStats,
    TerminationStats,
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

__all__ = [
    "APPLICABLE", "NOT_APPLICABLE", "NOT_MEASURED",
    "ComparisonVerdict", "Interval", "LengthStats", "PerplexityResult",
    "PreferenceStats", "TerminationStats",
    "compare_proportions", "difference_is_resolvable", "distinct_n",
    "length_stats", "max_ngram_repeat", "perplexity", "perplexity_from_totals",
    "structural_checks", "termination_stats", "wilson_interval",
]
