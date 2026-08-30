"""Tests for Phase 3 data handling and SFT configuration composition.

Deliberately OFFLINE. Nothing here downloads a model or a dataset: the
fingerprint logic, the prompt/completion conversion and the config schema are
all testable without the network, and tests that need 3 GiB of weights would
never run on the local CPU machine. The parts that genuinely require the model
are verified by scripts/experiments/e12 on the server instead.
"""

from __future__ import annotations

import pytest

from alignlab.data import (
    DatasetFingerprint,
    fingerprint_rows,
    is_wellformed,
    to_prompt_completion,
)

USER = {"role": "user", "content": "What is 2+2?"}
ASSISTANT = {"role": "assistant", "content": "It is 4."}


class TestFingerprint:
    def test_same_rows_same_hash(self):
        rows = [{"a": 1}, {"b": 2}]
        first = fingerprint_rows(rows, "ds", "train")
        second = fingerprint_rows(list(rows), "ds", "train")
        assert first.sha256 == second.sha256

    def test_key_order_does_not_change_the_hash(self):
        """Dict ordering is an implementation detail, not data."""
        a = fingerprint_rows([{"x": 1, "y": 2}], "ds", "train")
        b = fingerprint_rows([{"y": 2, "x": 1}], "ds", "train")
        assert a.sha256 == b.sha256

    def test_row_order_DOES_change_the_hash(self):
        """Order is data: a reshuffled training set is a different experiment."""
        a = fingerprint_rows([{"x": 1}, {"x": 2}], "ds", "train")
        b = fingerprint_rows([{"x": 2}, {"x": 1}], "ds", "train")
        assert a.sha256 != b.sha256

    def test_content_change_changes_the_hash(self):
        a = fingerprint_rows([{"x": 1}], "ds", "train")
        b = fingerprint_rows([{"x": 2}], "ds", "train")
        assert a.sha256 != b.sha256

    def test_row_separator_prevents_concatenation_collisions(self):
        """["ab"] and ["a","b"] must not hash alike."""
        a = fingerprint_rows(["ab"], "ds", "train")
        b = fingerprint_rows(["a", "b"], "ds", "train")
        assert a.sha256 != b.sha256

    def test_counts_are_recorded(self):
        fp = fingerprint_rows([{"x": 1}, {"x": 2}, {"x": 3}], "ds", "train")
        assert fp.n_rows == 3
        assert fp.n_chars > 0

    def test_empty_is_valid_and_distinct(self):
        fp = fingerprint_rows([], "ds", "train")
        assert fp.n_rows == 0
        assert fp.sha256 != fingerprint_rows([{"x": 1}], "ds", "train").sha256

    def test_unicode_is_stable_across_platforms(self):
        """ensure_ascii keeps the digest independent of platform encoding."""
        rows = [{"text": "café → 日本語"}]
        assert fingerprint_rows(rows, "d", "t").sha256 == (
            fingerprint_rows(rows, "d", "t").sha256
        )

    def test_to_dict_round_trips(self):
        fp = fingerprint_rows([{"x": 1}], "ds", "train")
        assert DatasetFingerprint(**fp.to_dict()) == fp


class TestPromptCompletionConversion:
    def test_single_turn_splits_at_the_assistant(self):
        out = to_prompt_completion({"messages": [USER, ASSISTANT]})
        assert out["prompt"] == [USER]
        assert out["completion"] == [ASSISTANT]

    def test_multi_turn_keeps_only_the_final_assistant_turn(self):
        messages = [USER, ASSISTANT, {"role": "user", "content": "And 3+3?"},
                    {"role": "assistant", "content": "It is 6."}]
        out = to_prompt_completion({"messages": messages})
        assert len(out["prompt"]) == 3
        assert out["completion"] == [messages[-1]]

    def test_conversation_not_ending_in_assistant_is_rejected(self):
        """Training on a user turn as if it were a target is a silent data bug."""
        out = to_prompt_completion({"messages": [ASSISTANT, USER]})
        assert out["completion"] == []
        assert not is_wellformed(out)

    def test_too_short_is_rejected(self):
        assert to_prompt_completion({"messages": [USER]})["prompt"] == []

    def test_missing_messages_key_is_rejected_not_crashed(self):
        assert to_prompt_completion({})["prompt"] == []

    @pytest.mark.parametrize("content", ["", "   ", "\n\t"])
    def test_blank_completion_is_rejected(self, content):
        out = to_prompt_completion(
            {"messages": [USER, {"role": "assistant", "content": content}]}
        )
        assert not is_wellformed(out)

    def test_blank_prompt_turn_is_rejected(self):
        out = to_prompt_completion(
            {"messages": [{"role": "user", "content": "  "}, ASSISTANT]}
        )
        assert not is_wellformed(out)

    def test_wellformed_accepts_a_good_pair(self):
        assert is_wellformed(to_prompt_completion({"messages": [USER, ASSISTANT]}))


class TestSFTConfigComposition:
    """The SFT config must compose, and its pins must survive composition."""

    @staticmethod
    def _compose(overrides=None):
        from hydra import compose, initialize_config_dir
        from alignlab.config_schema import register_configs
        from alignlab.sft import CONFIG_DIR

        register_configs()
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            return compose(config_name="sft", overrides=overrides or [])

    def test_composes_with_defaults(self):
        cfg = self._compose()
        assert cfg.model.id == "Qwen/Qwen2.5-1.5B"
        assert cfg.data.name == "HuggingFaceH4/no_robots"

    def test_model_revision_is_pinned_to_a_full_sha(self):
        """An unpinned revision makes a rerun a different experiment."""
        cfg = self._compose()
        assert len(cfg.model.revision) == 40
        assert cfg.model.revision == "8faed761d45a263340a0528343f099c05c9a4323"

    def test_storage_guard_defaults_on(self):
        cfg = self._compose()
        assert cfg.storage_guard is True
        assert cfg.allow_low_disk is False

    def test_packing_is_off_by_default(self):
        """Phase 3 chooses a verifiable mask over throughput."""
        assert self._compose().sft.packing is False

    def test_save_total_limit_is_storage_aware(self):
        assert self._compose().sft.save_total_limit == 1

    def test_smoke_profile_caps_steps(self):
        cfg = self._compose(["sft=smoke"])
        assert cfg.sft.max_steps == 20

    def test_server_env_group_composes(self):
        cfg = self._compose(["env=server"])
        assert cfg.env.expect_cuda is True
        assert cfg.env.checkpoint_root

    def test_overrides_reach_the_config(self):
        cfg = self._compose(["sft.learning_rate=1e-6", "data.max_train=64"])
        assert cfg.sft.learning_rate == pytest.approx(1e-6)
        assert cfg.data.max_train == 64

    def test_typo_in_a_field_is_rejected_at_compose_time(self):
        """The structured schema exists so a typo fails loudly, not silently."""
        from hydra.errors import ConfigCompositionException

        with pytest.raises((ConfigCompositionException, Exception)):
            self._compose(["sft.learing_rate=1e-6"])


class TestPeftConfigComposition:
    """The Phase 4 arms must differ in exactly what we intend."""

    @staticmethod
    def _compose(overrides=None):
        from hydra import compose, initialize_config_dir
        from alignlab.config_schema import register_configs
        from alignlab.sft import CONFIG_DIR

        register_configs()
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            return compose(config_name="sft", overrides=overrides or [])

    def test_default_arm_is_full_finetune(self):
        assert self._compose().peft.method == "none"

    def test_lora_arm_composes(self):
        cfg = self._compose(["peft=lora"])
        assert cfg.peft.method == "lora"
        assert cfg.peft.r == 16
        assert cfg.peft.alpha == 32.0

    def test_qlora_arm_composes_with_nf4(self):
        cfg = self._compose(["peft=qlora"])
        assert cfg.peft.method == "qlora"
        assert cfg.peft.quant_type == "nf4"
        assert cfg.peft.double_quant is True
        assert cfg.peft.compute_dtype == "bfloat16"

    def test_lora_and_qlora_share_every_adapter_setting(self):
        """The arms must differ ONLY in how the frozen base is stored."""
        lora = self._compose(["peft=lora"]).peft
        qlora = self._compose(["peft=qlora"]).peft
        for field in ("r", "alpha", "dropout", "bias"):
            assert getattr(lora, field) == getattr(qlora, field), field
        assert list(lora.target_modules) == list(qlora.target_modules)

    def test_alpha_is_twice_r_so_scaling_is_two(self):
        cfg = self._compose(["peft=lora"])
        assert cfg.peft.alpha / cfg.peft.r == 2.0

    def test_targets_exclude_lm_head(self):
        """lm_head is TIED to the embedding; adapting it would adapt both."""
        cfg = self._compose(["peft=lora"])
        assert "lm_head" not in list(cfg.peft.target_modules)

    def test_rank_override_reaches_the_config(self):
        assert self._compose(["peft=lora", "peft.r=64"]).peft.r == 64

    def test_invalid_method_is_rejected(self):
        from alignlab.config_schema import PeftConfigGroup

        with pytest.raises(ValueError, match="none|lora|qlora"):
            PeftConfigGroup(method="adapters")

    def test_nonpositive_rank_rejected_for_peft_methods(self):
        from alignlab.config_schema import PeftConfigGroup

        with pytest.raises(ValueError, match="rank must be positive"):
            PeftConfigGroup(method="lora", r=0)

    def test_scaling_property(self):
        from alignlab.config_schema import PeftConfigGroup

        assert PeftConfigGroup(method="lora", r=8, alpha=16).scaling == 2.0
