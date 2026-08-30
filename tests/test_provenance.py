"""Behavioural tests for the provenance auditor.

The auditor's whole value is that it reports gaps honestly. So the tests are
mostly about what it must NOT do: invent a value, hide a real gap, or pad the
report with false ones.
"""

from __future__ import annotations

import json

import pytest

from alignlab.provenance import (
    NOT_APPLICABLE,
    NOT_RECORDED,
    RECORDED,
    REQUIRED_FIELDS,
    audit_manifest,
    audit_run_directory,
    audit_runs,
    render_provenance_report,
)


def complete_training_manifest() -> dict:
    """The shape a real Phase 3-6 run writes (trimmed to the audited fields)."""
    return {
        "run_name": "sft-qwen1p5b-noRobots-001",
        "seed": 42,
        "config": {
            "reproducibility": {"seed": 42},
            "model": {"id": "Qwen/Qwen2.5-1.5B", "revision": "8faed761", "dtype": "auto"},
            "sft": {"learning_rate": 2e-05, "num_train_epochs": 1.0},
        },
        "config_sha256": "e14d1d86",
        "git": {"commit": "3c66757f", "branch": "phase-3-sft", "dirty": True},
        "hardware": {"device": "cuda", "torch_version": "2.6.0+cu124"},
        "packages": {"torch": "2.6.0+cu124", "transformers": "5.16.1"},
        "paths": {"checkpoint_root": "/data/AlignLab/checkpoints"},
        "notes": {
            "model_id": "Qwen/Qwen2.5-1.5B",
            "model_revision": "8faed761",
            "dtype": "torch.bfloat16",
            "dataset": {
                "name": "HuggingFaceH4/no_robots",
                "train_fingerprint": {"sha256": "17bebb17"},
                "eval_fingerprint": {"sha256": "762160ae"},
            },
        },
    }


def complete_eval_dashboard() -> dict:
    """The shape produced AFTER the Phase 7 fingerprint fix."""
    provenance = {
        "model_revision": "8faed761",
        "checkpoint": "Qwen/Qwen2.5-1.5B",
        "dtype": "torch.bfloat16",
        "git_commit": "2e2a51f1",
        "git_dirty": False,
        "hardware": {"device": "cuda", "torch_version": "2.6.0+cu124"},
        "generation": {"max_new_tokens": 256, "do_sample": False, "seed": 42},
        "extras": {
            "fingerprints": {
                "perplexity": {"sha256": "f92fdec3"},
                "preference": {"sha256": "b3bc775a"},
            }
        },
    }
    return {
        "protocol": {"seed": 42, "base_revision": "8faed761", "run_name": "eval-x"},
        "models": [
            {"name": "base", "provenance": dict(provenance)},
            {"name": "SFT", "provenance": dict(provenance)},
        ],
    }


class TestCompleteArtefacts:
    def test_complete_training_manifest_has_no_gaps(self):
        audit = audit_manifest(complete_training_manifest(), "training")
        assert audit.complete, f"unexpected gaps: {audit.missing}"
        assert audit.missing == []

    def test_complete_eval_dashboard_has_no_gaps(self):
        audit = audit_manifest(complete_eval_dashboard(), "evaluation")
        assert audit.complete, f"unexpected gaps: {audit.missing}"

    def test_every_required_field_gets_a_status(self):
        audit = audit_manifest(complete_training_manifest(), "training")
        assert [f.name for f in audit.fields] == list(REQUIRED_FIELDS)
        assert all(f.status in (RECORDED, NOT_RECORDED, NOT_APPLICABLE)
                   for f in audit.fields)


class TestGapsAreReported:
    def test_missing_field_is_NOT_RECORDED(self):
        manifest = complete_training_manifest()
        del manifest["git"]
        audit = audit_manifest(manifest, "training")
        assert "git_commit" in audit.missing
        assert not audit.complete

    def test_the_phase7_dataset_fingerprint_bug_is_caught(self):
        """eval-full-001 shipped dataset_fingerprint null in every record.

        This is the exact defect the auditor was written to detect, so it is
        pinned with the historical shape rather than a synthetic one.
        """
        dashboard = complete_eval_dashboard()
        for model in dashboard["models"]:
            model["provenance"].pop("extras")
            model["provenance"]["dataset_fingerprint"] = None
            model["provenance"]["eval_fingerprint"] = "b3bc775a"

        audit = audit_manifest(dashboard, "evaluation")
        assert "dataset_fingerprint" in audit.missing
        assert "eval_fingerprint" not in audit.missing, (
            "the preference fingerprint WAS recorded; reporting it as missing "
            "would overstate the gap"
        )

    def test_a_gap_in_one_model_is_a_gap_for_the_run(self):
        """Four models recording a value and one not is not 'recorded'."""
        dashboard = complete_eval_dashboard()
        dashboard["models"][1]["provenance"]["dtype"] = None
        audit = audit_manifest(dashboard, "evaluation")
        assert "dtype" in audit.missing

    def test_null_is_not_rescued_into_a_placeholder(self):
        """A locator must never substitute 'unknown' for an absent value."""
        manifest = complete_training_manifest()
        manifest["notes"]["dtype"] = None
        manifest["config"]["model"]["dtype"] = None
        audit = audit_manifest(manifest, "training")
        dtype = next(f for f in audit.fields if f.name == "dtype")
        assert dtype.status == NOT_RECORDED
        assert dtype.summary is None


class TestApplicability:
    def test_eval_run_has_no_training_parameters(self):
        audit = audit_manifest(complete_eval_dashboard(), "evaluation")
        assert "training_parameters" in audit.not_applicable
        assert "training_parameters" not in audit.missing

    def test_training_run_has_no_evaluation_parameters(self):
        audit = audit_manifest(complete_training_manifest(), "training")
        assert "evaluation_parameters" in audit.not_applicable

    def test_infrastructure_run_does_not_produce_false_gaps(self):
        """A wiring check that never loaded a model has nothing to record.

        Before this rule the audit listed 8 such runs as incomplete, which
        would have buried the single real gap.
        """
        manifest = {
            "run_name": "wandb-offline-test",
            "seed": 42,
            "config": {"reproducibility": {"seed": 42}},
            "config_sha256": "abc",
            "git": {"commit": "deadbeef", "dirty": False},
            "hardware": {"device": "cpu"},
            "packages": {"torch": "2.6.0"},
            "paths": {"checkpoint_root": "/data/ckpt"},
            "notes": {"phase": 1},
        }
        audit = audit_manifest(manifest, "training")
        assert audit.complete, f"false gaps on an infra run: {audit.missing}"
        assert "model_revision" in audit.not_applicable
        assert "dataset_fingerprint" in audit.not_applicable

    def test_a_real_training_run_is_not_excused_by_that_rule(self):
        """The exemption must not swallow a genuine gap."""
        manifest = complete_training_manifest()
        manifest["notes"]["dataset"]["train_fingerprint"] = {}
        audit = audit_manifest(manifest, "training")
        assert "dataset_fingerprint" in audit.missing
        assert "dataset_fingerprint" not in audit.not_applicable

    def test_local_checkpoint_without_hub_revision_is_not_a_gap(self):
        """A fine-tuned checkpoint is a path, not a hub revision.

        The base revision pins the lineage; demanding one per model reported a
        false gap on every evaluation run.
        """
        dashboard = complete_eval_dashboard()
        dashboard["models"][1]["provenance"]["model_revision"] = None
        dashboard["models"][1]["provenance"]["checkpoint"] = "checkpoints/sft/final"
        audit = audit_manifest(dashboard, "evaluation")
        assert "model_revision" not in audit.missing


class TestRobustness:
    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            audit_manifest({}, "guesswork")

    def test_unreadable_artefact_is_reported_not_raised(self, tmp_path):
        run = tmp_path / "broken-run"
        run.mkdir()
        (run / "run_manifest.json").write_text("{not json", encoding="utf-8")

        audits = audit_run_directory(run)

        assert len(audits) == 1
        assert not audits[0].complete
        assert "unreadable" in (audits[0].fields[0].summary or "")

    def test_one_broken_run_does_not_hide_the_others(self, tmp_path):
        good = tmp_path / "good-run"
        good.mkdir()
        (good / "run_manifest.json").write_text(
            json.dumps(complete_training_manifest()), encoding="utf-8"
        )
        bad = tmp_path / "bad-run"
        bad.mkdir()
        (bad / "run_manifest.json").write_text("{", encoding="utf-8")

        audits = audit_runs(tmp_path)

        assert {a.run_name for a in audits} == {"good-run", "bad-run"}
        assert next(a for a in audits if a.run_name == "good-run").complete

    def test_missing_root_returns_empty_rather_than_raising(self, tmp_path):
        assert audit_runs(tmp_path / "does-not-exist") == []

    def test_directory_without_artefacts_is_skipped(self, tmp_path):
        (tmp_path / "empty-run").mkdir()
        assert audit_runs(tmp_path) == []


class TestReport:
    def test_report_names_the_gap_and_the_run(self, tmp_path):
        run = tmp_path / "eval-full-001"
        run.mkdir()
        dashboard = complete_eval_dashboard()
        for model in dashboard["models"]:
            model["provenance"].pop("extras")
        (run / "evaluation_dashboard.json").write_text(
            json.dumps(dashboard), encoding="utf-8"
        )

        text = render_provenance_report(audit_runs(tmp_path))

        assert "eval-full-001" in text
        assert "dataset_fingerprint" in text
        assert "NOT RECORDED" in text

    def test_report_is_ascii(self, tmp_path):
        """Phase 6 lesson: print() under Windows cp1252 raises on non-ASCII."""
        run = tmp_path / "r"
        run.mkdir()
        (run / "run_manifest.json").write_text(
            json.dumps(complete_training_manifest()), encoding="utf-8"
        )
        text = render_provenance_report(audit_runs(tmp_path))
        text.encode("ascii")  # raises if not

    def test_empty_report_says_so(self):
        assert "no run artefacts found" in render_provenance_report([])
