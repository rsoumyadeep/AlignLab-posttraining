# Phase 4 — Code Explanation

Explains the code that exists. Every claim was observed on the running system.

---

## 1. Files

| File | Responsibility |
|---|---|
| `src/alignlab/lora.py` | **first-principles LoRA**: `LoRALinear`, `apply_lora`, merge/unmerge, adapter save/load |
| `src/alignlab/peft_setup.py` | translate AlignLab config → `peft.LoraConfig` / `BitsAndBytesConfig`; parameter accounting |
| `src/alignlab/config_schema.py` | `PeftConfigGroup` — the three arms |
| `configs/peft/{none,lora,qlora}.yaml` | the arms, differing in exactly one thing |
| `scripts/experiments/e15_bitsandbytes_environment.py` | does 4-bit actually work here? |
| `scripts/experiments/e16_lora_targets.py` | real module names and adapter costs |
| `scripts/experiments/e17_svd_rank_selection.py` | **rank selection from a real ΔW** |
| `scripts/experiments/e18_educational_vs_peft.py` | ours vs the library |
| `scripts/experiments/e19_peft_comparison.py` | full SFT vs LoRA vs QLoRA |
| `tests/test_lora.py` | 46 tests on the educational implementation |

---

## 2. `LoRALinear` — the core mechanism

```
x        : [..., d_in]
base(x)  : [..., d_out]        frozen
A        : [r, d_in]           lora_A.weight, Kaiming-uniform
A x      : [..., r]
B        : [d_out, r]          lora_B.weight, ZERO
B A x    : [..., d_out]
output   : base(x) + (alpha/r) · B A x
```

**It wraps rather than replaces.** `self.base` is the original `nn.Linear`,
kept intact. That is what makes `merge`/`unmerge` exact and what guarantees the
original weights are always recoverable.

**Freezing happens in `__init__`,** not in the caller. `apply_lora` then
freezes *everything* first and lets only the new adapters be trainable — in
that order, so a parameter that was already trainable for some other reason
cannot silently ride along.

### `merge()` / `unmerge()`

`merge()` folds `delta_weight` into `base.weight` in place, after which the
module is an ordinary linear layer with **no adapter overhead at inference**.
Guarded against double-merging, which would add the update twice.

`unmerge()` is exact only to float tolerance — subtraction is not bitwise
reversible. The test asserts a tolerance and says why.

### `lora_state_dict` / `load_lora_state_dict`

Extracts *only* adapter tensors. Saving `model.state_dict()` instead would
produce a full-size checkpoint and quietly discard LoRA's storage advantage.

Loading is **strict in both directions**: unknown keys are rejected, and
missing adapter keys are rejected. A silently partial load would leave some
layers at `B = 0`, contributing nothing — a model that looks "almost right"
rather than broken.

---

## 3. Equation → code

| Maths | Code |
|---|---|
| `ΔW = BA` | `lora_B.weight @ lora_A.weight` |
| `α/r` | `self.scaling` |
| `W' = W + (α/r)BA` | `delta_weight`, added into `base.weight` on `merge()` |
| `W'x = Wx + (α/r)BAx` | `forward()` |
| rank(ΔW) ≤ r | asserted by `test_delta_weight_rank_is_at_most_r` |

`ΔW = UΣVᵀ`, `ΔW_r = U_rΣ_rV_rᵀ` and the energy curve live in `e17`, computed
in **float64** because ΔW is a difference of bf16 tensors and its tail sits near
the precision floor.

---

## 4. Design decisions

| Decision | Alternative | Why |
|---|---|---|
| wrap the base module | replace it | merge/unmerge stay exact; originals recoverable |
| `B = 0`, `A` random | both random / both zero | both-zero is a dead saddle; both-random perturbs the pretrained model |
| `α = 2r` | fixed `α` | keeps scaling constant across ranks, so a rank sweep is not also an LR sweep |
| freeze all, then unfreeze adapters | freeze targets only | a stray trainable parameter cannot ride along |
| match on the **leaf** module name | substring match | `q_proj` must not match `not_q_proj_thing` |
| raise when no module matches | warn | a typo'd target must not yield a model that trains nothing |
| `peft` for the real runs | our own | ours is the *educational* implementation; E18 shows they agree, and peft handles checkpointing/dtype details |
| NF4 over FP4 | FP4 | E15 measured 24.6% lower relative error on normal data |
| refuse when bitsandbytes is missing | fall back to bf16 | a silent fallback produces memory numbers that are wrong in the most misleading way |
| arm-aware storage guard | one figure | guarding LoRA with the 20 GiB full-FT estimate would refuse runs that fit easily |
| correct `Params4bit` packing | trust `numel()` | it returns *bytes*; uncorrected, the manifest would halve a 1.5B model |

---

## 5. What `peft_setup` adds around the libraries

- **`build_quantization_config`** — returns `BitsAndBytesConfig` only for
  `qlora`, and **raises** if bitsandbytes is absent.
- **`build_peft_config`** — returns `peft.LoraConfig`, or raises rather than
  silently training every parameter.
- **`summarise_trainable`** — counts by iterating the model rather than
  trusting peft's printout, and corrects for 4-bit packing.
- **`describe_peft`** — a manifest-ready record, including
  `"compute_is_4bit": False`, because that is the fact people get wrong.

`sft.py` additionally **refuses to train** if a PEFT arm somehow has ≥50% of
parameters trainable — i.e. if adapter wrapping silently failed and the "PEFT"
run is secretly a full fine-tune.

---

## 6. Measurements, and where they came from

| Measurement | Value | Source |
|---|---|---|
| native lib loaded | `libbitsandbytes_cuda124.so` | E15 |
| NF4 vs FP4 relative error | 0.0912 vs 0.1210 (**24.6% better**) | E15 |
| `Linear4bit` vs bf16 `nn.Linear` | rel. 0.0910 | E15 |
| 4-bit packed storage ratio | exactly **4.00×** | E15 |
| `nn.Linear` modules in Qwen2.5-1.5B | **197** (196 + `lm_head`) | E16 |
| attention targets matched | **112** matrices | E16 |
| trainable at r=16 (attention) | **4,358,144** = 0.2823% | E16 |
| predicted vs actual trainable | **identical** | E16 |
| `‖ΔW‖/‖W‖` after 1 epoch | 0.0013 – 0.0050 | E17 |
| rank for 90% of ΔW's energy | **~730 of 1536** | E17 |
| noise control, same statistic | 783 | E17 |
| energy captured at r=16 | **10.69%** | E17 |
| break-even rank, 1536×1536 | **768** | executed examples |
| ours vs peft: logits, weights | **0.000e+00** | E18 |

---

## 7. Bugs and corrections

### 7a. A test model that could not receive gradients — caught on first run
`test_gradients_reach_A_and_B_only` failed for `k_proj`/`v_proj`. Correctly:
`TinyModel.forward` used only `q_proj` and `o_proj`, so the other two were
never called. **The bug was in the test model, not in LoRA.** The model now
exercises all four projections.

### 7b. Arithmetic error in E17's own text
`730 × 3072` was printed as 2,240,160; it is **2,242,560**. Caught by executing
the parameter-cost example, not by re-reading. The conclusion was unaffected
(4.9% saving either way) but the number was wrong.

### 7c. `triton` installed but unimportable
`import triton` failed with `ModuleNotFoundError: No module named 'setuptools'`
— triton needs setuptools at import time and the venv had none. Fixed by
installing `setuptools` into the AlignLab venv. Latent before Phase 4 because
nothing had imported triton.

### 7d. `tokenizer.additional_special_tokens` removed in transformers 5.x
Raises `AttributeError`. `get_added_vocab()` is the replacement. The fifth
transformers-5 API break this project has hit by running into it.

---

## 8. Test map

| Claim | Test |
|---|---|
| scaling is α/r; α=2r fixes it at 2 | `TestConfig` (4) |
| B=0, ΔW=0, output identical to base | `TestInitialisation` (5) |
| base frozen, adapters trainable | `TestFreezing` (4) |
| gradients only to A/B; both-zero is a dead saddle | `TestGradientFlow` (4) |
| parameter counts and scaling with r | `TestParameterCounts` (6) |
| leaf-name matching; no-match raises | `TestTargeting` (4) |
| merge/unmerge, idempotence, whole-model | `TestMerge` (8) |
| adapter-only state; strict load | `TestSaveLoad` (6) |
| forward IS the equation; rank ≤ r | `TestMathematicalIdentity` (4) |
| the three arms compose correctly | `TestPeftConfigComposition` (10) |

Local: **403 passed, 2 skipped**.
