"""Tests for the first-principles LoRA implementation.

These correspond one-to-one with the properties Phase 4 is required to verify:
the base stays frozen, only A/B train, the adapted model is IDENTICAL to the
base at initialisation, gradients flow only where intended, the parameter count
is right, save/load round-trips, merging works, and merged inference agrees
with adapter inference.

Everything here runs on CPU in float32 and finishes in seconds - the point is
to test the MECHANISM, which does not depend on a GPU. The real Qwen numbers
are measured separately on the server.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from alignlab.lora import (
    LoRAConfig,
    LoRALinear,
    apply_lora,
    count_parameters,
    load_lora_state_dict,
    lora_state_dict,
    merge_all,
    unmerge_all,
)


class TinyModel(nn.Module):
    """A stand-in with Qwen-like module NAMES so target matching is exercised."""

    def __init__(self, d: int = 32, d_ff: int = 64) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d // 2)
        self.v_proj = nn.Linear(d, d // 2)
        self.o_proj = nn.Linear(d, d)
        self.gate_proj = nn.Linear(d, d_ff)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        # Every targeted projection must be ON the forward path. An earlier
        # version of this model used only q_proj and o_proj, which made
        # test_gradients_reach_A_and_B_only fail for k_proj/v_proj - correctly,
        # since an uncalled module cannot receive a gradient. The bug was in
        # the test model, not in LoRA.
        h = self.norm(x)
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        return self.o_proj(q + torch.cat([k, v], dim=-1))


@pytest.fixture
def model():
    torch.manual_seed(0)
    return TinyModel()


class TestConfig:
    def test_scaling_is_alpha_over_r(self):
        assert LoRAConfig(r=8, alpha=16).scaling == 2.0
        assert LoRAConfig(r=16, alpha=16).scaling == 1.0

    def test_alpha_equals_2r_gives_scale_2_at_every_rank(self):
        """The common convention: scale stays fixed as capacity changes."""
        for r in (1, 4, 16, 64):
            assert LoRAConfig(r=r, alpha=2 * r).scaling == 2.0

    @pytest.mark.parametrize("r", [0, -1])
    def test_nonpositive_rank_rejected(self, r):
        with pytest.raises(ValueError, match="rank must be positive"):
            LoRAConfig(r=r)

    def test_nonpositive_alpha_rejected(self):
        with pytest.raises(ValueError, match="alpha must be positive"):
            LoRAConfig(alpha=0)


class TestInitialisation:
    def test_B_is_zero_and_A_is_not(self):
        layer = LoRALinear(nn.Linear(16, 8), r=4, alpha=8)
        assert torch.all(layer.lora_B.weight == 0)
        assert not torch.all(layer.lora_A.weight == 0)

    def test_delta_weight_is_exactly_zero_at_init(self):
        layer = LoRALinear(nn.Linear(16, 8), r=4, alpha=8)
        assert torch.all(layer.delta_weight == 0)

    def test_output_is_IDENTICAL_to_base_at_init(self):
        """The headline property: adapting must not perturb the model."""
        base = nn.Linear(16, 8)
        x = torch.randn(4, 16)
        expected = base(x).clone()
        layer = LoRALinear(base, r=4, alpha=8)
        assert torch.equal(layer(x), expected)

    def test_delta_weight_has_the_base_weight_shape(self):
        layer = LoRALinear(nn.Linear(16, 8), r=4, alpha=8)
        assert layer.delta_weight.shape == layer.base.weight.shape

    def test_wrapping_a_non_linear_is_rejected(self):
        with pytest.raises(TypeError, match="wraps nn.Linear"):
            LoRALinear(nn.LayerNorm(8), r=4, alpha=8)


class TestFreezing:
    def test_base_weight_is_frozen_by_construction(self):
        layer = LoRALinear(nn.Linear(16, 8), r=4, alpha=8)
        assert not layer.base.weight.requires_grad
        assert not layer.base.bias.requires_grad

    def test_adapters_are_trainable(self):
        layer = LoRALinear(nn.Linear(16, 8), r=4, alpha=8)
        assert layer.lora_A.weight.requires_grad
        assert layer.lora_B.weight.requires_grad

    def test_apply_lora_freezes_everything_else(self, model):
        apply_lora(model, LoRAConfig(r=4, alpha=8))
        for name, param in model.named_parameters():
            if "lora_" in name:
                assert param.requires_grad, name
            else:
                assert not param.requires_grad, name

    def test_untargeted_module_is_frozen_but_still_present(self, model):
        """gate_proj is not a target: frozen, and NOT wrapped."""
        apply_lora(model, LoRAConfig(r=4, alpha=8))
        assert isinstance(model.gate_proj, nn.Linear)
        assert not isinstance(model.gate_proj, LoRALinear)
        assert not model.gate_proj.weight.requires_grad


class TestGradientFlow:
    def test_gradients_reach_A_and_B_only(self, model):
        apply_lora(model, LoRAConfig(r=4, alpha=8))
        out = model(torch.randn(2, 32))
        out.sum().backward()

        for name, param in model.named_parameters():
            if "lora_" in name:
                assert param.grad is not None, f"{name} got no gradient"
            else:
                assert param.grad is None, f"{name} got a gradient but is frozen"

    def test_B_receives_gradient_at_init_but_A_does_not(self):
        """Why B=0/A random is the only initialisation that works.

        dL/dA is proportional to B, which is zero at step 0, so A gets a zero
        gradient. dL/dB is proportional to A, which is not zero. B therefore
        moves first. If BOTH were zero neither would ever move - the saddle
        point the module docstring warns about.
        """
        layer = LoRALinear(nn.Linear(16, 8), r=4, alpha=8)
        layer(torch.randn(4, 16)).sum().backward()
        assert torch.all(layer.lora_A.weight.grad == 0)
        assert not torch.all(layer.lora_B.weight.grad == 0)

    def test_both_zero_is_a_dead_saddle_point(self):
        """Demonstrates the failure mode we avoid, rather than merely asserting it."""
        layer = LoRALinear(nn.Linear(16, 8), r=4, alpha=8)
        nn.init.zeros_(layer.lora_A.weight)  # deliberately break the init
        layer(torch.randn(4, 16)).sum().backward()
        assert torch.all(layer.lora_A.weight.grad == 0)
        assert torch.all(layer.lora_B.weight.grad == 0)

    def test_base_weight_is_unchanged_by_an_optimiser_step(self, model):
        apply_lora(model, LoRAConfig(r=4, alpha=8))
        before = model.q_proj.base.weight.detach().clone()

        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
        for _ in range(3):
            opt.zero_grad()
            model(torch.randn(4, 32)).sum().backward()
            opt.step()

        assert torch.equal(model.q_proj.base.weight, before)
        # and the adapter DID move, so the test is not vacuous
        assert not torch.all(model.q_proj.lora_B.weight == 0)


class TestParameterCounts:
    def test_adapter_params_are_r_times_din_plus_dout(self):
        layer = LoRALinear(nn.Linear(64, 32), r=8, alpha=16)
        expected = 8 * 64 + 32 * 8
        actual = layer.lora_A.weight.numel() + layer.lora_B.weight.numel()
        assert actual == expected

    def test_lora_is_far_cheaper_than_the_dense_update(self):
        d_in, d_out, r = 1536, 1536, 16
        dense = d_in * d_out
        lora = r * (d_in + d_out)
        assert lora < dense / 40

    def test_counts_are_consistent(self, model):
        apply_lora(model, LoRAConfig(r=4, alpha=8))
        counts = count_parameters(model)
        assert counts["total"] == counts["trainable"] + counts["frozen"]
        assert 0 < counts["trainable_fraction"] < 1

    def test_summary_trainable_equals_adapter_params(self, model):
        summary = apply_lora(model, LoRAConfig(r=4, alpha=8))
        assert summary["trainable_params"] == summary["adapter_params"]

    def test_higher_rank_means_more_trainable_params(self):
        counts = []
        for r in (2, 8, 32):
            torch.manual_seed(0)
            m = TinyModel()
            counts.append(apply_lora(m, LoRAConfig(r=r, alpha=2 * r))["trainable_params"])
        assert counts == sorted(counts)
        assert counts[0] < counts[1] < counts[2]

    def test_wrapping_does_not_change_total_base_params(self, model):
        before = sum(p.numel() for p in model.parameters())
        summary = apply_lora(model, LoRAConfig(r=4, alpha=8))
        after = sum(p.numel() for p in model.parameters())
        assert after == before + summary["adapter_params"]


class TestTargeting:
    def test_only_named_targets_are_wrapped(self, model):
        apply_lora(model, LoRAConfig(r=4, alpha=8, target_modules=("q_proj", "v_proj")))
        assert isinstance(model.q_proj, LoRALinear)
        assert isinstance(model.v_proj, LoRALinear)
        assert not isinstance(model.k_proj, LoRALinear)
        assert not isinstance(model.o_proj, LoRALinear)

    def test_no_match_raises_rather_than_silently_doing_nothing(self, model):
        """A typo in target_modules must not produce a model that trains nothing."""
        with pytest.raises(ValueError, match="no nn.Linear modules matched"):
            apply_lora(model, LoRAConfig(target_modules=("does_not_exist",)))

    def test_matching_is_on_the_leaf_name_not_a_substring(self):
        class Odd(nn.Module):
            def __init__(self):
                super().__init__()
                self.not_q_proj_really = nn.Linear(8, 8)
                self.q_proj = nn.Linear(8, 8)

        m = Odd()
        apply_lora(m, LoRAConfig(r=2, alpha=4, target_modules=("q_proj",)))
        assert isinstance(m.q_proj, LoRALinear)
        assert not isinstance(m.not_q_proj_really, LoRALinear)

    def test_matched_count_is_reported(self, model):
        summary = apply_lora(model, LoRAConfig(r=4, alpha=8))
        assert summary["matched_modules"] == 4  # q, k, v, o


class TestMerge:
    def test_merged_output_matches_adapter_output(self):
        """The property that makes LoRA free at inference."""
        torch.manual_seed(0)
        layer = LoRALinear(nn.Linear(32, 16), r=4, alpha=8)
        nn.init.normal_(layer.lora_B.weight, std=0.05)  # make the adapter non-trivial

        x = torch.randn(8, 32)
        before = layer(x).clone()
        layer.merge()
        after = layer(x)

        assert layer.merged
        torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)

    def test_merge_actually_changes_the_base_weight(self):
        layer = LoRALinear(nn.Linear(32, 16), r=4, alpha=8)
        nn.init.normal_(layer.lora_B.weight, std=0.05)
        before = layer.base.weight.detach().clone()
        layer.merge()
        assert not torch.equal(layer.base.weight, before)

    def test_merge_is_a_noop_at_init_because_delta_is_zero(self):
        layer = LoRALinear(nn.Linear(32, 16), r=4, alpha=8)
        before = layer.base.weight.detach().clone()
        layer.merge()
        assert torch.equal(layer.base.weight, before)

    def test_unmerge_restores_the_original_weight(self):
        """Exact only to float tolerance - subtraction is not bitwise reversible."""
        torch.manual_seed(0)
        layer = LoRALinear(nn.Linear(32, 16), r=4, alpha=8)
        nn.init.normal_(layer.lora_B.weight, std=0.05)
        original = layer.base.weight.detach().clone()

        layer.merge()
        layer.unmerge()

        assert not layer.merged
        torch.testing.assert_close(layer.base.weight, original, rtol=1e-6, atol=1e-7)

    def test_double_merge_does_not_double_the_update(self):
        torch.manual_seed(0)
        layer = LoRALinear(nn.Linear(32, 16), r=4, alpha=8)
        nn.init.normal_(layer.lora_B.weight, std=0.05)
        layer.merge()
        once = layer.base.weight.detach().clone()
        layer.merge()  # guarded no-op
        assert torch.equal(layer.base.weight, once)

    def test_unmerge_without_merge_is_a_noop(self):
        layer = LoRALinear(nn.Linear(32, 16), r=4, alpha=8)
        before = layer.base.weight.detach().clone()
        layer.unmerge()
        assert torch.equal(layer.base.weight, before)

    def test_merge_all_and_unmerge_all_report_counts(self, model):
        apply_lora(model, LoRAConfig(r=4, alpha=8))
        assert merge_all(model) == 4
        assert unmerge_all(model) == 4

    def test_whole_model_merged_output_matches(self, model):
        apply_lora(model, LoRAConfig(r=4, alpha=8))
        for m in model.modules():
            if isinstance(m, LoRALinear):
                nn.init.normal_(m.lora_B.weight, std=0.02)

        x = torch.randn(4, 32)
        before = model(x).clone()
        merge_all(model)
        torch.testing.assert_close(model(x), before, rtol=1e-5, atol=1e-6)


class TestSaveLoad:
    def test_state_dict_contains_only_adapter_tensors(self, model):
        apply_lora(model, LoRAConfig(r=4, alpha=8))
        state = lora_state_dict(model)
        assert state
        assert all("lora_A" in k or "lora_B" in k for k in state)
        assert not any("base" in k for k in state)

    def test_adapter_state_is_far_smaller_than_the_full_state(self, model):
        apply_lora(model, LoRAConfig(r=4, alpha=8))
        adapter = sum(t.numel() for t in lora_state_dict(model).values())
        full = sum(p.numel() for p in model.parameters())
        assert adapter < full / 5

    def test_round_trip_restores_behaviour_exactly(self, model):
        apply_lora(model, LoRAConfig(r=4, alpha=8))
        for m in model.modules():
            if isinstance(m, LoRALinear):
                nn.init.normal_(m.lora_B.weight, std=0.02)

        x = torch.randn(4, 32)
        expected = model(x).clone()
        state = lora_state_dict(model)

        # A fresh model with the SAME base weights but re-initialised adapters.
        torch.manual_seed(0)
        fresh = TinyModel()
        apply_lora(fresh, LoRAConfig(r=4, alpha=8))
        assert not torch.allclose(fresh(x), expected)  # differs before loading

        load_lora_state_dict(fresh, state)
        torch.testing.assert_close(fresh(x), expected, rtol=1e-6, atol=1e-7)

    def test_round_trip_through_a_file(self, model, tmp_path):
        apply_lora(model, LoRAConfig(r=4, alpha=8))
        for m in model.modules():
            if isinstance(m, LoRALinear):
                nn.init.normal_(m.lora_B.weight, std=0.02)
        x = torch.randn(4, 32)
        expected = model(x).clone()

        path = tmp_path / "adapter.pt"
        torch.save(lora_state_dict(model), path)

        torch.manual_seed(0)
        fresh = TinyModel()
        apply_lora(fresh, LoRAConfig(r=4, alpha=8))
        load_lora_state_dict(fresh, torch.load(path, weights_only=True))
        torch.testing.assert_close(fresh(x), expected, rtol=1e-6, atol=1e-7)

    def test_unknown_key_is_rejected(self, model):
        apply_lora(model, LoRAConfig(r=4, alpha=8))
        state = lora_state_dict(model)
        state["not_a_real_param"] = torch.zeros(1)
        with pytest.raises(KeyError, match="not in the model"):
            load_lora_state_dict(model, state)

    def test_partial_state_is_rejected(self, model):
        """A silent partial load leaves B=0 layers contributing nothing."""
        apply_lora(model, LoRAConfig(r=4, alpha=8))
        state = lora_state_dict(model)
        state.pop(next(iter(state)))
        with pytest.raises(KeyError, match="missing adapter keys"):
            load_lora_state_dict(model, state)


class TestMathematicalIdentity:
    def test_forward_equals_base_plus_scaled_BAx(self):
        """Assert the implementation IS the equation, not merely near it."""
        torch.manual_seed(0)
        base = nn.Linear(32, 16)
        layer = LoRALinear(base, r=4, alpha=8)
        nn.init.normal_(layer.lora_B.weight, std=0.05)

        x = torch.randn(8, 32)
        manual = base(x) + layer.scaling * (x @ layer.lora_A.weight.T @ layer.lora_B.weight.T)
        torch.testing.assert_close(layer(x), manual, rtol=1e-5, atol=1e-6)

    def test_delta_weight_equals_scaling_times_B_matmul_A(self):
        layer = LoRALinear(nn.Linear(32, 16), r=4, alpha=8)
        nn.init.normal_(layer.lora_B.weight, std=0.05)
        expected = layer.scaling * (layer.lora_B.weight @ layer.lora_A.weight)
        torch.testing.assert_close(layer.delta_weight, expected)

    def test_delta_weight_rank_is_at_most_r(self):
        """The defining constraint: BA cannot have rank above r."""
        torch.manual_seed(0)
        layer = LoRALinear(nn.Linear(64, 64), r=4, alpha=8)
        nn.init.normal_(layer.lora_B.weight, std=0.1)
        rank = torch.linalg.matrix_rank(layer.delta_weight.float())
        assert rank <= 4

    def test_scaling_changes_the_update_magnitude_linearly(self):
        torch.manual_seed(0)
        magnitudes = []
        for alpha in (8.0, 16.0, 32.0):
            torch.manual_seed(0)
            layer = LoRALinear(nn.Linear(32, 16), r=8, alpha=alpha)
            nn.init.normal_(layer.lora_B.weight, std=0.05)
            magnitudes.append(float(layer.delta_weight.abs().mean()))
        assert magnitudes[1] == pytest.approx(2 * magnitudes[0], rel=1e-5)
        assert magnitudes[2] == pytest.approx(4 * magnitudes[0], rel=1e-5)
