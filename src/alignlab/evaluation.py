"""Evaluation harness skeleton.

PROJECT_INSTRUCTIONS section 7 is explicit that evaluation is NOT merely a
final phase: a minimal evaluation pipeline should exist early and be
re-runnable throughout the project. This module is that early skeleton.

What exists now (Phase 1): the interfaces, the result record, the runner, and
JSON persistence. Verified against a trivial arithmetic evaluator in
tests/test_evaluation.py.

What does NOT exist yet: perplexity, generation quality, instruction-following,
preference win-rate and LLM-as-judge. Those are Phase 7 (with a baseline
captured before Phase 3 training begins) and are marked DEFERRED. This file
deliberately does not stub them out with fake numbers - an evaluator that
returns a plausible-looking constant is worse than no evaluator at all.

The design constraint that matters: results carry the manifest of the run that
produced them, so a metric can never drift away from the model, machine and
commit it describes.


NAMING (Phase 8 audit). This module and the ``alignlab.evals`` package have
confusingly similar names and are NOT the same thing:

    alignlab.evaluation   THIS module. The Phase 1 evaluator PROTOCOL - a
                          registry of named evaluators, run by train.py, where
                          a failing evaluator is recorded NOT_TESTED rather
                          than aborting the run.
    alignlab.evals        The Phase 7 measurement SUBSYSTEM - perplexity,
                          preference statistics, generation behaviour, the
                          two-order judge and the dashboard.

They were deliberately not merged: one is a harness contract, the other is a
body of metrics, and the Phase 7 subsystem was built ON this skeleton rather
than replacing it. The names were left alone as well - renaming a module that
completed Phase 3-6 training code imports would churn working experimental
paths for a cosmetic gain, which the Phase 8 brief explicitly rules out. The
ambiguity is documented here and in alignlab/evals/__init__.py instead.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from alignlab.logging_utils import get_logger

logger = get_logger(__name__)

RESULTS_FILE_NAME = "eval_results.json"


@dataclass
class EvalResult:
    """The outcome of one evaluator.

    status uses the vocabulary from PROJECT_INSTRUCTIONS section 15. An
    evaluator that did not actually run reports NOT_TESTED and carries no
    metrics - it never reports a placeholder value.
    """

    name: str
    metrics: dict[str, float] = field(default_factory=dict)
    status: str = "MEASURED"
    n_examples: int | None = None
    notes: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Evaluator(Protocol):
    """Anything that can produce an EvalResult."""

    name: str

    def evaluate(self, model: Any, **kwargs: Any) -> EvalResult: ...


@dataclass
class EvalReport:
    """A full evaluation pass: several evaluators against one model."""

    run_name: str
    created_at: str
    model_description: str
    results: list[EvalResult] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, directory: str | Path, file_name: str = RESULTS_FILE_NAME) -> Path:
        """Persist the report as JSON."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / file_name
        path.write_text(
            json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8"
        )
        logger.info("Wrote evaluation report: %s", path)
        return path

    def metric(self, evaluator_name: str, metric_name: str) -> float | None:
        """Look up a single metric, or None if it was not measured."""
        for result in self.results:
            if result.name == evaluator_name:
                return result.metrics.get(metric_name)
        return None


def run_evaluation(
    model: Any,
    evaluators: list[Evaluator],
    run_name: str,
    model_description: str = "unspecified",
    manifest: dict[str, Any] | None = None,
    **kwargs: Any,
) -> EvalReport:
    """Run every evaluator against one model and collect the results.

    An evaluator that raises does not abort the pass. It is recorded with
    status NOT_TESTED and the exception text, because losing four working
    metrics to one broken evaluator is a bad trade - and because a silently
    missing metric is exactly the kind of gap that later gets mistaken for a
    measured zero.
    """
    report = EvalReport(
        run_name=run_name,
        created_at=datetime.now(timezone.utc).isoformat(),
        model_description=model_description,
        manifest=manifest or {},
    )

    for evaluator in evaluators:
        name = getattr(evaluator, "name", type(evaluator).__name__)
        logger.info("Running evaluator: %s", name)
        try:
            report.results.append(evaluator.evaluate(model, **kwargs))
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
            logger.exception("Evaluator %s failed", name)
            report.results.append(
                EvalResult(
                    name=name,
                    metrics={},
                    status="NOT_TESTED",
                    notes=f"Evaluator raised {type(exc).__name__}: {exc}",
                )
            )

    return report


def load_report(path: str | Path) -> dict[str, Any]:
    """Read an evaluation report back from disk."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
