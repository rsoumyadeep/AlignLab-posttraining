"""Evaluation metrics — pure functions, no models, no I/O.

Separated from the evaluators that call them so every metric can be tested
against hand-computed values. A metric that can only be exercised by loading a
1.5B model is a metric nobody checks.

THE DESIGN LESSON THIS MODULE ENCODES. Four phases produced the same finding in
four different costumes:

  Phase 3  a loss can be a mean over the wrong token population - and an
           unmasked loss is not comparable to a masked one in EITHER direction
  Phase 4  perplexity said the PEFT arms were within 3.5% of full SFT while
           stop-token behaviour said 0/4 versus 4/4
  Phase 5  the same preference comparison gave 47.2% (SUM) and 58.3% (MEAN)
  Phase 6  the SUM metric was dominated by a ~29-nat length gap, not by quality

So every metric here carries the information needed to notice when it is
measuring something other than what its name suggests: token counts alongside
losses, lengths alongside preference rates, and an explicit denominator
wherever a mean is taken.

STATUS VOCABULARY, extending PROJECT_INSTRUCTIONS section 15 for comparisons:

    APPLICABLE      the metric is meaningful for this model/comparison
    NOT_APPLICABLE  the metric is not meaningful here (e.g. preference
                    accuracy for a model never trained on preferences is
                    computable but not informative as a claim about training)
    NOT_MEASURED    it could apply, but was not run
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Sequence

APPLICABLE = "APPLICABLE"
NOT_APPLICABLE = "NOT_APPLICABLE"
NOT_MEASURED = "NOT_MEASURED"


# --------------------------------------------------------------- perplexity


def perplexity(mean_nll: float) -> float:
    """``exp(mean negative log-likelihood per token)``.

    THE NUMBER IS MEANINGLESS WITHOUT ITS DENOMINATOR. `exp` of a mean over
    completion tokens and `exp` of a mean over all tokens are different
    quantities, and Phase 3 measured how different: on the base model, the
    prompt region scored 6.3314 and the completion region 3.6295 - a factor of
    1.7 in nats. Callers must say which they exponentiated; `PerplexityResult`
    forces them to.

    Raises on a negative NLL, which cannot arise from a correct computation and
    almost always means a sign error.
    """
    if mean_nll < 0:
        raise ValueError(
            f"mean NLL must be non-negative, got {mean_nll}. A negative value "
            f"usually means log-probabilities were passed where negative "
            f"log-probabilities were expected."
        )
    return math.exp(mean_nll)


@dataclass(frozen=True)
class PerplexityResult:
    """Perplexity with the denominator that defines it."""

    perplexity: float
    mean_nll: float
    total_nll: float
    n_tokens: int
    # "completion" | "full_sequence" | "prompt" - never omitted
    region: str

    def to_dict(self) -> dict:
        return asdict(self)

    def comparable_to(self, other: "PerplexityResult") -> bool:
        """Two perplexities are comparable only over the same token region."""
        return self.region == other.region


def perplexity_from_totals(
    total_nll: float, n_tokens: int, region: str
) -> PerplexityResult:
    """Token-weighted perplexity from accumulated totals.

    Token-weighted, not example-weighted: averaging per-example means would
    weight a 5-token answer the same as a 500-token one.
    """
    if n_tokens <= 0:
        raise ValueError(f"n_tokens must be positive, got {n_tokens}")
    mean = total_nll / n_tokens
    return PerplexityResult(
        perplexity=perplexity(mean),
        mean_nll=mean,
        total_nll=total_nll,
        n_tokens=n_tokens,
        region=region,
    )


# ------------------------------------------------------------------ lengths


@dataclass(frozen=True)
class LengthStats:
    """Response-length distribution.

    Reported ALONGSIDE every preference and quality metric, because Phase 5 and
    Phase 6 both found length driving a number that looked like quality.
    """

    n: int
    mean: float
    median: float
    minimum: int
    maximum: int
    p90: int

    def to_dict(self) -> dict:
        return asdict(self)


def length_stats(lengths: Sequence[int]) -> LengthStats:
    if not lengths:
        return LengthStats(0, 0.0, 0.0, 0, 0, 0)
    ordered = sorted(lengths)
    n = len(ordered)
    return LengthStats(
        n=n,
        mean=sum(ordered) / n,
        median=float(ordered[n // 2]),
        minimum=ordered[0],
        maximum=ordered[-1],
        p90=ordered[min(n - 1, int(0.9 * n))],
    )


# ------------------------------------------------------- termination / format


@dataclass(frozen=True)
class TerminationStats:
    """Behavioural checks on generated text.

    PHASE 4'S LESSON, MADE REUSABLE. Perplexity put the PEFT arms within 3.5%
    of full SFT; stop-token behaviour put them at 0/4 against 4/4. A model that
    never stops is unusable regardless of its perplexity, and no likelihood
    metric reveals it.

    These are BEHAVIOURAL metrics. They say whether output is well-formed, not
    whether it is good.
    """

    n: int
    terminated: int
    emitted_stop_token: int
    hit_length_cap: int
    empty: int

    @property
    def termination_rate(self) -> float:
        return self.terminated / self.n if self.n else 0.0

    @property
    def stop_token_rate(self) -> float:
        return self.emitted_stop_token / self.n if self.n else 0.0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["termination_rate"] = round(self.termination_rate, 6)
        data["stop_token_rate"] = round(self.stop_token_rate, 6)
        return data


def termination_stats(
    generations: Sequence[dict],
    max_new_tokens: int,
) -> TerminationStats:
    """Summarise termination behaviour over a set of generations.

    Each generation dict needs ``n_generated``, ``emitted_stop_token`` and
    ``text``.
    """
    n = len(generations)
    hit_cap = sum(1 for g in generations if g["n_generated"] >= max_new_tokens)
    stop = sum(1 for g in generations if g.get("emitted_stop_token"))
    empty = sum(1 for g in generations if not (g.get("text") or "").strip())
    # "Terminated" means it stopped on its own, not that it ran out of budget.
    terminated = sum(
        1 for g in generations if g["n_generated"] < max_new_tokens
    )
    return TerminationStats(
        n=n,
        terminated=terminated,
        emitted_stop_token=stop,
        hit_length_cap=hit_cap,
        empty=empty,
    )


def distinct_n(text: str, n: int = 2) -> float:
    """Fraction of n-grams that are unique. Low values indicate repetition.

    Phase 3's base model produced ``-unstyled`` repeated to the token cap;
    distinct-2 catches that where perplexity did not.
    """
    tokens = text.split()
    if len(tokens) < n:
        return 1.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams)


def max_ngram_repeat(text: str, n: int = 3) -> int:
    """How many times the most frequent n-gram occurs."""
    tokens = text.split()
    if len(tokens) < n:
        return 0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return max(Counter(grams).values())


_PLACEHOLDER = re.compile(r"\[[A-Za-z][A-Za-z ]{0,30}\]")


def structural_checks(text: str) -> dict:
    """Cheap structure signals, reported not scored.

    These are descriptive. None of them is a quality judgement, and they are
    deliberately not combined into a single number.
    """
    stripped = text.strip()
    lines = [line for line in stripped.splitlines() if line.strip()]
    return {
        "chars": len(stripped),
        "words": len(stripped.split()),
        "lines": len(lines),
        "distinct_2": round(distinct_n(stripped, 2), 6),
        "distinct_3": round(distinct_n(stripped, 3), 6),
        "max_trigram_repeat": max_ngram_repeat(stripped, 3),
        "has_numbered_list": bool(re.search(r"^\s*\d+[.)]\s", stripped, re.M)),
        "has_bullet_list": bool(re.search(r"^\s*[-*•]\s", stripped, re.M)),
        "has_code_fence": "```" in stripped,
        "placeholder_count": len(_PLACEHOLDER.findall(stripped)),
        "is_empty": not stripped,
    }


# --------------------------------------------------------------- preference


@dataclass(frozen=True)
class PreferenceStats:
    """Preference accuracy under BOTH objectives, with lengths.

    PHASE 5 AND 6'S LESSON, MADE STRUCTURAL. Reporting one of these without the
    other, or either without lengths, hides the effect that dominated Phase 6:
    the SUM comparison was decided by a ~29-nat length gap, not by quality.

    `sum_accuracy` is the project's PRIMARY metric - it is the published DPO
    objective. `mean_accuracy` is a DIAGNOSTIC. That ordering is fixed by the
    Phase 6 pre-registration and is not a per-experiment choice.
    """

    n: int
    sum_correct: int
    mean_correct: int
    chosen_tokens_mean: float
    rejected_tokens_mean: float
    chosen_logp_mean: float
    rejected_logp_mean: float

    @property
    def sum_accuracy(self) -> float:
        return self.sum_correct / self.n if self.n else 0.0

    @property
    def mean_accuracy(self) -> float:
        return self.mean_correct / self.n if self.n else 0.0

    @property
    def token_gap(self) -> float:
        """Extra tokens carried by the chosen response, on average."""
        return self.chosen_tokens_mean - self.rejected_tokens_mean

    @property
    def logp_gap(self) -> float:
        return self.chosen_logp_mean - self.rejected_logp_mean

    def length_attribution(self) -> dict:
        """How much of the summed log-prob gap is explained by length alone?

        Phase 6's decisive calculation, generalised: multiply the extra token count by
        the rejected response's per-token log-probability. What remains is the
        part attributable to something other than length.
        """
        if self.rejected_tokens_mean <= 0:
            return {"explained_by_length": 0.0, "residual": self.logp_gap}
        per_token = self.rejected_logp_mean / self.rejected_tokens_mean
        explained = self.token_gap * per_token
        return {
            "chosen_logp_per_token": (
                self.chosen_logp_mean / self.chosen_tokens_mean
                if self.chosen_tokens_mean
                else 0.0
            ),
            "rejected_logp_per_token": per_token,
            "logp_gap": self.logp_gap,
            "explained_by_length": explained,
            "residual": self.logp_gap - explained,
        }

    def to_dict(self) -> dict:
        data = asdict(self)
        data["sum_accuracy"] = round(self.sum_accuracy, 6)
        data["mean_accuracy"] = round(self.mean_accuracy, 6)
        data["token_gap"] = round(self.token_gap, 4)
        data["logp_gap"] = round(self.logp_gap, 4)
        data["length_attribution"] = self.length_attribution()
        return data


# ------------------------------------------------------------- uncertainty


@dataclass(frozen=True)
class Interval:
    """A confidence interval, with the method that produced it named."""

    point: float
    low: float
    high: float
    level: float
    method: str
    n: int

    @property
    def width(self) -> float:
        return self.high - self.low

    def excludes(self, value: float) -> bool:
        """Does the interval exclude a reference value (e.g. 0.5 for chance)?"""
        return value < self.low or value > self.high

    def to_dict(self) -> dict:
        data = asdict(self)
        data["width"] = round(self.width, 6)
        return data


def wilson_interval(successes: int, n: int, level: float = 0.95) -> Interval:
    """Wilson score interval for a proportion.

    WHY WILSON AND NOT THE NORMAL APPROXIMATION. The textbook
    ``p ± z·sqrt(p(1-p)/n)`` misbehaves badly at small n and near 0 or 1 - it
    can produce bounds outside [0, 1], and its coverage is poor exactly where
    this project operates (184 preference pairs, 4 qualitative prompts, ~50
    judge comparisons). Wilson stays inside [0, 1] and has far better small-
    sample coverage.

    This is the honest alternative to inventing significance. PROJECT_
    INSTRUCTIONS forbids manufacturing statistical claims; reporting an
    interval and letting it be wide is the correct response to a small sample.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0 <= successes <= n:
        raise ValueError(f"successes {successes} outside [0, {n}]")

    z = _z_for(level)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = (z / denominator) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return Interval(
        point=p,
        low=max(0.0, centre - margin),
        high=min(1.0, centre + margin),
        level=level,
        method="wilson",
        n=n,
    )


def _z_for(level: float) -> float:
    """Two-sided normal quantile for the common levels.

    A small table rather than a dependency on scipy, which this project does
    not install. Unsupported levels raise rather than silently using 1.96.
    """
    table = {0.80: 1.2815515655446004, 0.90: 1.6448536269514722,
             0.95: 1.959963984540054, 0.99: 2.5758293035489004}
    if level not in table:
        raise ValueError(
            f"confidence level {level} not tabulated; supported: {sorted(table)}"
        )
    return table[level]


def difference_is_resolvable(a: Interval, b: Interval) -> bool:
    """Do two proportion intervals fail to overlap?

    NON-OVERLAP IS A CONSERVATIVE TEST, NOT A SIGNIFICANCE TEST. Overlapping
    intervals do not prove no difference, and this function deliberately does
    not return a p-value. It answers one narrow question: is the observed gap
    larger than the uncertainty on either side? If not, the honest report is
    "not resolvable at this sample size".
    """
    return a.high < b.low or b.high < a.low


@dataclass
class ComparisonVerdict:
    """The outcome of comparing one metric between two models."""

    metric: str
    model_a: str
    model_b: str
    value_a: float
    value_b: float
    delta: float
    status: str = APPLICABLE
    interval_a: dict | None = None
    interval_b: dict | None = None
    resolvable: bool | None = None
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def compare_proportions(
    metric: str,
    model_a: str,
    successes_a: int,
    n_a: int,
    model_b: str,
    successes_b: int,
    n_b: int,
    level: float = 0.95,
) -> ComparisonVerdict:
    """Compare two rates, with intervals and an explicit resolvability verdict."""
    interval_a = wilson_interval(successes_a, n_a, level)
    interval_b = wilson_interval(successes_b, n_b, level)
    resolvable = difference_is_resolvable(interval_a, interval_b)
    note = (
        "intervals do not overlap - the difference exceeds the uncertainty"
        if resolvable
        else f"intervals overlap at n={n_a}/{n_b} - NOT resolvable at this sample size"
    )
    return ComparisonVerdict(
        metric=metric,
        model_a=model_a,
        model_b=model_b,
        value_a=interval_a.point,
        value_b=interval_b.point,
        delta=interval_b.point - interval_a.point,
        interval_a=interval_a.to_dict(),
        interval_b=interval_b.to_dict(),
        resolvable=resolvable,
        note=note,
    )
