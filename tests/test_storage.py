"""Tests for the disk pre-flight guard.

The guard's whole value is that it REFUSES. A guard that can only be observed
to pass has not been tested, so most of these tests assert on the refusal path
and on the arithmetic that decides it.

The arithmetic is separated from the filesystem deliberately: estimates are
pure functions and can be asserted exactly, while the filesystem-facing part
is tested against tmp_path and against deliberately absurd requests rather
than by trying to actually fill a disk.
"""

from __future__ import annotations

import pytest

from alignlab.storage import (
    DEFAULT_HEADROOM_FRACTION,
    GIB,
    MIN_HEADROOM_BYTES,
    DiskStatus,
    InsufficientStorage,
    disk_status,
    estimate_checkpoint_bytes,
    estimate_lora_checkpoint_bytes,
    headroom_bytes,
    require_free_space,
    require_free_space_for_checkpoints,
)

# Qwen2.5-1.5B, MEASURED from the real checkpoint by
# scripts/experiments/e11_weights_reconciliation.py.
#
# This constant previously held 1,543,656,960 - Phase 2's config-only estimate,
# which omitted the 28 x 2048 attention QKV bias parameters. E11 falsified that
# number against the downloaded weights. The wrong value survived here until
# the storage estimates were checked against a real checkpoint file and came up
# 57,344 params short, which is the same error propagating one file further.
QWEN_PARAMS = 1_543_714_304


class TestEstimates:
    def test_weights_only_bf16_is_two_bytes_per_param(self):
        assert estimate_checkpoint_bytes(1000, "bfloat16", optimizer=None) == 2000

    def test_adamw_adds_eight_bytes_and_master_adds_four(self):
        # bf16 params (2) + exp_avg (4) + exp_avg_sq (4) + fp32 master (4) = 14
        assert estimate_checkpoint_bytes(1000, "bfloat16", "adamw") == 14_000

    def test_master_weights_can_be_excluded(self):
        assert (
            estimate_checkpoint_bytes(1000, "bfloat16", "adamw", master_weights=False)
            == 10_000
        )

    def test_sgd_is_cheaper_than_adamw(self):
        sgd = estimate_checkpoint_bytes(1000, "bfloat16", "sgd")
        adamw = estimate_checkpoint_bytes(1000, "bfloat16", "adamw")
        assert sgd < adamw
        assert sgd == 1000 * (2 + 4 + 4)

    def test_qwen_full_checkpoint_is_about_twenty_gib(self):
        """The number that justifies keep_last_checkpoints=1.

        Asserted as a range, not a point: the claim being protected is "this
        is tens of GiB, not hundreds of MiB", and a tight equality assertion
        would only be re-stating the arithmetic above.
        """
        gib = estimate_checkpoint_bytes(QWEN_PARAMS, "bfloat16", "adamw") / GIB
        assert 19.0 < gib < 22.0

    def test_lora_checkpoint_is_orders_of_magnitude_smaller(self):
        """The measurement that will justify raising keep_last in Phase 4."""
        full = estimate_checkpoint_bytes(QWEN_PARAMS, "bfloat16", "adamw")
        lora = estimate_lora_checkpoint_bytes(9_231_360)
        assert full / lora > 100

    def test_unknown_optimizer_is_rejected_not_guessed(self):
        with pytest.raises(ValueError, match="unknown optimizer"):
            estimate_checkpoint_bytes(1000, optimizer="lion")

    def test_negative_params_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            estimate_checkpoint_bytes(-1)

    def test_unknown_dtype_falls_back_to_four_bytes(self):
        """An unrecognised dtype must over-estimate, never under-estimate."""
        assert estimate_checkpoint_bytes(1000, "float64", optimizer=None) == 4000


class TestHeadroom:
    def test_small_requests_get_the_floor_not_the_fraction(self):
        assert headroom_bytes(1024) == MIN_HEADROOM_BYTES

    def test_large_requests_get_the_fraction(self):
        required = 100 * GIB
        assert headroom_bytes(required) == int(required * DEFAULT_HEADROOM_FRACTION)

    def test_headroom_is_never_below_the_floor(self):
        for n in (0, 1, GIB, 50 * GIB):
            assert headroom_bytes(n) >= MIN_HEADROOM_BYTES


class TestDiskStatus:
    def test_reads_a_real_directory(self, tmp_path):
        status = disk_status(tmp_path)
        assert status.total_bytes > 0
        assert status.free_bytes >= 0
        assert 0.0 <= status.percent_used <= 100.0

    def test_measures_nearest_existing_ancestor_for_a_missing_path(self, tmp_path):
        """Pre-flight must work for a directory we are about to create."""
        missing = tmp_path / "does" / "not" / "exist" / "yet"
        status = disk_status(missing)
        assert status.total_bytes > 0
        assert status.path == str(missing.resolve())

    def test_to_dict_is_json_friendly_and_carries_derived_fields(self, tmp_path):
        data = disk_status(tmp_path).to_dict()
        assert {"free_gib", "total_gib", "percent_used"} <= set(data)
        assert isinstance(data["free_bytes"], int)

    def test_percent_used_handles_zero_total(self):
        assert DiskStatus("x", 0, 0, 0).percent_used == 0.0


class TestRequireFreeSpace:
    def test_passes_when_there_is_room(self, tmp_path):
        status = require_free_space(tmp_path, 1024, label="tiny")
        assert isinstance(status, DiskStatus)

    def test_refuses_an_impossible_request(self, tmp_path):
        with pytest.raises(InsufficientStorage, match="INSUFFICIENT STORAGE"):
            require_free_space(tmp_path, 10**18, label="absurd")

    def test_refusal_names_the_shortfall_and_the_operation(self, tmp_path):
        with pytest.raises(InsufficientStorage) as exc:
            require_free_space(tmp_path, 10**18, label="my-operation")
        message = str(exc.value)
        assert "my-operation" in message
        assert "short by" in message

    def test_refusal_suggests_lora_and_retention_as_remedies(self, tmp_path):
        """The error must be actionable, not merely correct."""
        with pytest.raises(InsufficientStorage) as exc:
            require_free_space(tmp_path, 10**18)
        message = str(exc.value)
        assert "keep_last_checkpoints" in message
        assert "LoRA" in message

    def test_override_proceeds_instead_of_raising(self, tmp_path):
        status = require_free_space(
            tmp_path, 10**18, label="override", allow_override=True
        )
        assert isinstance(status, DiskStatus)

    def test_headroom_is_actually_enforced(self, tmp_path):
        """A request that fits ONLY without headroom must still be refused."""
        free = disk_status(tmp_path).free_bytes
        # Ask for everything free minus a little - fits bare, fails with headroom.
        request = max(free - (MIN_HEADROOM_BYTES // 2), 1)
        with pytest.raises(InsufficientStorage):
            require_free_space(tmp_path, request, label="no-headroom")


class TestCheckpointGuard:
    def test_budgets_keep_last_plus_one(self, tmp_path):
        """Rotation writes before deleting, so peak is keep_last + 1."""
        with pytest.raises(InsufficientStorage) as exc:
            require_free_space_for_checkpoints(
                tmp_path, n_params=10**12, keep_last=1
            )
        # 1e12 params * 14 bytes * 2 checkpoints - astronomically over any disk.
        assert "2 x" not in str(exc.value)  # phrasing check: label shows "1+1 x"
        assert "1+1 x" in str(exc.value)

    def test_passes_for_a_tiny_model(self, tmp_path):
        status = require_free_space_for_checkpoints(
            tmp_path, n_params=1000, keep_last=1
        )
        assert isinstance(status, DiskStatus)

    def test_keep_last_zero_still_budgets_one_checkpoint(self, tmp_path):
        with pytest.raises(InsufficientStorage) as exc:
            require_free_space_for_checkpoints(
                tmp_path, n_params=10**12, keep_last=0
            )
        assert "0+1 x" in str(exc.value)

    def test_more_retention_needs_more_space(self):
        from alignlab.storage import estimate_checkpoint_bytes as est

        one = est(QWEN_PARAMS) * 2
        three = est(QWEN_PARAMS) * 4
        assert three > one


class TestEstimateAgainstMeasurement:
    """The estimator versus a checkpoint that was actually written.

    Measured on csrslave, 2026-08-30, from a real TRL/transformers bf16 run:
        model.safetensors  3,087,467,144 B
        optimizer.pt       6,175,148,456 B
        total              9,262,615,600 B  (8.63 GiB, 6.00 bytes/param)
    """

    MEASURED_TOTAL = 9_262_615_600
    MEASURED_MODEL = 3_087_467_144
    MEASURED_OPTIMIZER = 6_175_148_456

    def test_measured_checkpoint_is_six_bytes_per_param(self):
        assert self.MEASURED_TOTAL / QWEN_PARAMS == pytest.approx(6.0, abs=0.01)

    def test_bf16_params_account_for_the_model_file(self):
        """2 bytes/param, to within the safetensors header (~38 KB, 0.001%)."""
        estimate = estimate_checkpoint_bytes(QWEN_PARAMS, "bfloat16", optimizer=None)
        overhead = self.MEASURED_MODEL - estimate
        assert 0 < overhead < 100_000, f"unexpected overhead {overhead}"

    def test_bf16_optimizer_states_account_for_the_optimizer_file(self):
        """AdamW's two moments in bf16 = 4 bytes/param, to within pickle overhead.

        This is the finding that explains the 2.33x over-estimate: the moments
        are stored in bf16, not fp32, and no fp32 master copy is written.
        """
        estimate = 4 * QWEN_PARAMS
        overhead = self.MEASURED_OPTIMIZER - estimate
        assert 0 < overhead < 1_000_000, f"unexpected overhead {overhead}"

    def test_tuned_estimate_matches_the_measurement_to_within_overhead(self):
        tuned = estimate_checkpoint_bytes(
            QWEN_PARAMS,
            "bfloat16",
            "adamw",
            master_weights=False,
            optimizer_state_dtype="bfloat16",
        )
        assert tuned == pytest.approx(self.MEASURED_TOTAL, rel=1e-4)
        assert tuned <= self.MEASURED_TOTAL  # container overhead is not modelled

    def test_default_over_predicts_and_that_is_the_safe_direction(self):
        """A disk guard must never under-predict; 2.3x high is acceptable."""
        default = estimate_checkpoint_bytes(QWEN_PARAMS)
        assert default > self.MEASURED_TOTAL
        assert default / self.MEASURED_TOTAL == pytest.approx(2.33, abs=0.05)
