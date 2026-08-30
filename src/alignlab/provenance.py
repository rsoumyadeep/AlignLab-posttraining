"""Mechanical audit of what a run actually recorded about itself.

Phase 8 asks whether every important experiment can be reconstructed from
recorded metadata. That question deserves a program rather than a paragraph:
prose in a README claiming "we record provenance" is exactly the kind of
statement that rots silently, and Phase 7 produced a concrete instance - a
dashboard shipped with ``dataset_fingerprint: null`` in all five records while
the surrounding documentation described provenance as complete.

So this module walks the run artefacts on disk and reports, per field, whether
the value is there.

The vocabulary is deliberately narrow:

    RECORDED        the value is present in the artefact
    NOT_RECORDED    the field applies to this run and is absent
    NOT_APPLICABLE  the field cannot apply (no checkpoint for an eval-only run)

**A historical gap is reported, never repaired.** Nothing here infers a missing
git commit from a timestamp, or a dtype from a checkpoint's file size. An old
run that did not record its seed is a run whose seed is unknown, and saying so
is the only honest option - reconstructing it from memory would manufacture
provenance, which is worse than lacking it.

Two artefact shapes are understood, because the project produces two:

    run_manifest.json            training runs (Phases 3-6)
    evaluation_dashboard.json    evaluation passes (Phase 7)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

RECORDED = "RECORDED"
NOT_RECORDED = "NOT_RECORDED"
NOT_APPLICABLE = "NOT_APPLICABLE"

TRAINING_MANIFEST = "run_manifest.json"
EVAL_DASHBOARD = "evaluation_dashboard.json"


def _dig(obj: Any, *path: str) -> Any:
    """Follow a key path, returning None the moment it stops resolving."""
    for key in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
        if obj is None:
            return None
    return obj


def _first(*values: Any) -> Any:
    """The first value that is neither None nor an empty container."""
    for value in values:
        if value is None:
            continue
        if isinstance(value, (dict, list, str)) and len(value) == 0:
            continue
        return value
    return None


@dataclass(frozen=True)
class FieldStatus:
    """One provenance field's presence in one artefact."""

    name: str
    status: str
    summary: str | None = None

    @property
    def recorded(self) -> bool:
        return self.status == RECORDED

    def to_dict(self) -> dict:
        return {"field": self.name, "status": self.status, "summary": self.summary}


@dataclass
class RunProvenance:
    """The provenance audit of a single run artefact."""

    run_name: str
    artefact: str
    kind: str
    fields: list[FieldStatus] = field(default_factory=list)

    @property
    def missing(self) -> list[str]:
        return [f.name for f in self.fields if f.status == NOT_RECORDED]

    @property
    def not_applicable(self) -> list[str]:
        return [f.name for f in self.fields if f.status == NOT_APPLICABLE]

    @property
    def complete(self) -> bool:
        """True when every APPLICABLE field is recorded."""
        return not self.missing

    def to_dict(self) -> dict:
        return {
            "run_name": self.run_name,
            "artefact": self.artefact,
            "kind": self.kind,
            "complete": self.complete,
            "missing": self.missing,
            "not_applicable": self.not_applicable,
            "fields": [f.to_dict() for f in self.fields],
        }


# --------------------------------------------------------------------------
# Field locators
# --------------------------------------------------------------------------
#
# Each entry maps a required field to a function returning the recorded value,
# or None when absent. Locators must NOT invent a fallback: returning "unknown"
# instead of None would report a gap as if it were filled.

Locator = Callable[[dict], Any]


def _training_locators() -> dict[str, Locator]:
    return {
        "git_commit": lambda m: _dig(m, "git", "commit"),
        "git_dirty_flag": lambda m: (
            _dig(m, "git", "dirty") if isinstance(_dig(m, "git"), dict)
            and "dirty" in m.get("git", {}) else None
        ),
        "model_revision": lambda m: _first(
            _dig(m, "notes", "model_revision"),
            _dig(m, "config", "model", "revision"),
        ),
        "dataset_fingerprint": lambda m: _dig(
            m, "notes", "dataset", "train_fingerprint", "sha256"
        ),
        "eval_fingerprint": lambda m: _dig(
            m, "notes", "dataset", "eval_fingerprint", "sha256"
        ),
        "seed": lambda m: _first(m.get("seed"), _dig(m, "config", "reproducibility", "seed")),
        "configuration": lambda m: m.get("config"),
        "config_digest": lambda m: m.get("config_sha256"),
        "hardware": lambda m: m.get("hardware"),
        "software_versions": lambda m: m.get("packages"),
        "dtype": lambda m: _first(
            _dig(m, "notes", "dtype"), _dig(m, "config", "model", "dtype")
        ),
        "training_parameters": lambda m: _first(
            _dig(m, "config", "sft"), _dig(m, "config", "dpo"), _dig(m, "config", "train")
        ),
        "checkpoint_identity": lambda m: _first(
            _dig(m, "notes", "checkpoint"),
            _dig(m, "paths", "checkpoint_root"),
        ),
    }


def _eval_locators() -> dict[str, Locator]:
    """Locators for an evaluation dashboard.

    Reads the per-model provenance records rather than the protocol header: the
    header carried both dataset fingerprints even in the run whose per-model
    records were null, so auditing the header alone would have missed the very
    gap this module exists to catch.
    """

    def models(d: dict) -> list[dict]:
        return [m for m in d.get("models", []) if m.get("provenance")]

    def every(d: dict, *path: str) -> Any:
        found = [_dig(m.get("provenance", {}), *path) for m in models(d)]
        return found if found and all(v is not None for v in found) else None

    return {
        "git_commit": lambda d: every(d, "git_commit"),
        "git_dirty_flag": lambda d: (
            [m["provenance"].get("git_dirty") for m in models(d)]
            if models(d) and all("git_dirty" in m["provenance"] for m in models(d))
            else None
        ),
        # A fine-tuned checkpoint is a local directory and has no hub revision,
        # so requiring one from every model would report a false gap. What pins
        # the lineage is the BASE model's revision, which every derived
        # checkpoint descends from.
        "model_revision": lambda d: _first(
            _dig(d, "protocol", "base_revision"),
            every(d, "model_revision"),
        ),
        "dataset_fingerprint": lambda d: _first(
            every(d, "extras", "fingerprints", "perplexity", "sha256"),
            every(d, "dataset_fingerprint"),
        ),
        "eval_fingerprint": lambda d: _first(
            every(d, "extras", "fingerprints", "preference", "sha256"),
            every(d, "eval_fingerprint"),
        ),
        "seed": lambda d: _dig(d, "protocol", "seed"),
        "configuration": lambda d: d.get("protocol"),
        "hardware": lambda d: every(d, "hardware"),
        "software_versions": lambda d: every(d, "hardware", "torch_version"),
        "dtype": lambda d: every(d, "dtype"),
        "evaluation_parameters": lambda d: every(d, "generation"),
        "checkpoint_identity": lambda d: every(d, "checkpoint"),
    }


# Fields that simply cannot apply to a given artefact kind. Marking these
# NOT_APPLICABLE rather than NOT_RECORDED keeps the "missing" list meaningful:
# an evaluation pass has no training parameters, and reporting that as a gap
# every time would train the reader to ignore the report.
_NOT_APPLICABLE_BY_KIND = {
    "training": ("evaluation_parameters",),
    "evaluation": ("training_parameters", "config_digest"),
}

# Fields that only exist once a run has actually loaded a model and data.
_MODEL_DEPENDENT_FIELDS = (
    "model_revision",
    "dataset_fingerprint",
    "eval_fingerprint",
    "dtype",
    "training_parameters",
)

REQUIRED_FIELDS = (
    "git_commit",
    "git_dirty_flag",
    "model_revision",
    "dataset_fingerprint",
    "eval_fingerprint",
    "seed",
    "configuration",
    "config_digest",
    "hardware",
    "software_versions",
    "dtype",
    "training_parameters",
    "evaluation_parameters",
    "checkpoint_identity",
)


def _summarise(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return value if len(value) <= 60 else value[:57] + "..."
    if isinstance(value, dict):
        return f"{len(value)} keys"
    if isinstance(value, list):
        unique = {json.dumps(v, sort_keys=True, default=str) for v in value}
        if len(unique) == 1:
            return _summarise(value[0])
        return f"{len(value)} values, {len(unique)} distinct"
    return str(value)


def _model_free_run(manifest: dict, kind: str) -> bool:
    """True for an infrastructure run that never loaded a model or dataset.

    Phase 1 produced several: a wandb wiring check, a SIGUSR1 preemption test,
    a path-resolution check. They have no model revision and no dataset
    fingerprint to record, so listing those as GAPS would be wrong - and worse,
    it would pad the report with eight false positives and teach the reader to
    skim past the one real one.

    Detected from artefact content rather than the run's name: a name-based
    rule would silently mis-classify the next run somebody calls "smoke".
    """
    if kind != "training":
        return False
    notes = manifest.get("notes") or {}
    return not (notes.get("model_id") or notes.get("dataset"))


def audit_manifest(manifest: dict, kind: str, run_name: str = "",
                   artefact: str = "") -> RunProvenance:
    """Audit one loaded artefact.

    ``kind`` is "training" or "evaluation"; it selects the locator set and the
    fields that cannot apply.
    """
    if kind not in ("training", "evaluation"):
        raise ValueError(f"unknown artefact kind: {kind!r}")

    locators = _training_locators() if kind == "training" else _eval_locators()
    inapplicable = _NOT_APPLICABLE_BY_KIND[kind]
    if _model_free_run(manifest, kind):
        inapplicable = inapplicable + _MODEL_DEPENDENT_FIELDS

    statuses: list[FieldStatus] = []
    for name in REQUIRED_FIELDS:
        if name in inapplicable:
            statuses.append(FieldStatus(name, NOT_APPLICABLE))
            continue
        locator = locators.get(name)
        value = locator(manifest) if locator else None
        if value is None:
            statuses.append(FieldStatus(name, NOT_RECORDED))
        else:
            statuses.append(FieldStatus(name, RECORDED, _summarise(value)))

    return RunProvenance(
        run_name=run_name or manifest.get("run_name", "?"),
        artefact=artefact,
        kind=kind,
        fields=statuses,
    )


def audit_run_directory(directory: Path) -> list[RunProvenance]:
    """Audit every artefact in one run directory (may hold both kinds)."""
    results: list[RunProvenance] = []
    for filename, kind in ((TRAINING_MANIFEST, "training"),
                           (EVAL_DASHBOARD, "evaluation")):
        path = directory / filename
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # An unreadable artefact is a provenance gap, not a crash: the
            # audit must still report on the runs it CAN read.
            results.append(RunProvenance(
                run_name=directory.name, artefact=str(path), kind=kind,
                fields=[FieldStatus(name, NOT_RECORDED, f"unreadable: {exc}")
                        for name in REQUIRED_FIELDS],
            ))
            continue
        results.append(audit_manifest(
            payload, kind, run_name=directory.name, artefact=str(path)
        ))
    return results


def audit_runs(output_root: Path) -> list[RunProvenance]:
    """Audit every run directory beneath ``output_root``, sorted by name."""
    root = Path(output_root)
    if not root.is_dir():
        return []
    results: list[RunProvenance] = []
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        results.extend(audit_run_directory(directory))
    return results


def render_provenance_report(audits: Iterable[RunProvenance]) -> str:
    """A plain-text table. ASCII only - see the Phase 6 cp1252 lesson."""
    audits = list(audits)
    lines: list[str] = []
    lines.append("=" * 100)
    lines.append("PROVENANCE AUDIT")
    lines.append("=" * 100)
    lines.append("  RECORDED / NOT_RECORDED / NOT_APPLICABLE")
    lines.append("  A gap is reported, never reconstructed.")
    lines.append("")

    if not audits:
        lines.append("  no run artefacts found")
        return "\n".join(lines)

    width = max(len(a.run_name) for a in audits) + 2
    lines.append(f"  {'run':<{width}} {'kind':<11} {'complete':<9} missing")
    lines.append("  " + "-" * 96)
    for audit in audits:
        missing = ", ".join(audit.missing) if audit.missing else "-"
        lines.append(
            f"  {audit.run_name:<{width}} {audit.kind:<11} "
            f"{'yes' if audit.complete else 'NO':<9} {missing}"
        )

    complete = sum(1 for a in audits if a.complete)
    lines.append("")
    lines.append(f"  {complete}/{len(audits)} artefacts record every applicable field")

    gaps: dict[str, int] = {}
    for audit in audits:
        for name in audit.missing:
            gaps[name] = gaps.get(name, 0) + 1
    if gaps:
        lines.append("")
        lines.append("  gaps by field (NOT RECORDED - not recoverable after the fact):")
        for name, count in sorted(gaps.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {name:<26} {count} run(s)")
    lines.append("=" * 100)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """``python -m alignlab.provenance [output_root]``

    Exit code 1 when any artefact is missing an applicable field, so this can
    gate CI. It reports; it never edits an artefact.
    """
    import argparse

    from alignlab.paths import output_root as default_output_root

    parser = argparse.ArgumentParser(description="Audit run provenance.")
    parser.add_argument(
        "output_root", nargs="?", default=None,
        help="directory of run directories (default: the configured output root)",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit JSON instead of the table"
    )
    args = parser.parse_args(argv)

    root = Path(args.output_root) if args.output_root else default_output_root()
    audits = audit_runs(root)

    if args.json:
        print(json.dumps([a.to_dict() for a in audits], indent=1))
    else:
        print(render_provenance_report(audits))

    return 0 if all(a.complete for a in audits) else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
