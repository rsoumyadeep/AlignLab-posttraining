# Phase 4 Report — Parameter-Efficient Fine-Tuning (LoRA / QLoRA)

**Date:** 2026-08-30 · **Branch:** `phase-4-peft`
**Machine:** `csrslave`, 1 × RTX A6000 (48 GiB, cc 8.6), bf16
**Model:** `Qwen/Qwen2.5-1.5B` @ `8faed761d45a263340a0528343f099c05c9a4323` (PINNED)

---

## 1. Environment Verification (first action, as instructed)

| | |
|---|---|
| GPU | 2 × NVIDIA RTX A6000, 47.53 GiB each, cc **8.6**, driver 550.144.03 |
| CUDA (torch) | **12.4** · cuDNN 9.1.0 · bf16 supported |
| PyTorch | **2.6.0+cu124** |
| `nvcc` | **NOT PRESENT** — no source builds possible, no root |
| bitsandbytes | **0.50.2** — installed into the AlignLab venv only |
| peft | **0.20.0** |
| setuptools | 84.0.0 — installed because `triton` 3.2.0 was present but **unimportable without it** |
| Disk before/after | 80 GiB free; +0.1 GiB for three wheels |

**Compatibility result: PASS.** bitsandbytes loaded
`libbitsandbytes_cuda124.so` — a **prebuilt** kernel matching torch's CUDA 12.4
exactly, so the missing `nvcc` is irrelevant. This was a live question: the same
constraint made `flash-attn` unbuildable in Phase 2.

**Isolation confirmed.** Installed with `VIRTUAL_ENV=.venv`; the system
`python3` still reports `ModuleNotFoundError: No module named 'bitsandbytes'`.
No system-wide or unrelated environment was modified.

**E15 verified by execution, not by import** (an import succeeds in plenty of
broken installations):

| check | result |
|---|---|
| H1 kernel matches torch CUDA | **PASS** |
| H2 error small but non-zero | **PASS** — NF4 mean abs 0.07277 |
| H3 NF4 beats FP4 on normal data | **PASS** — rel. 0.0912 vs 0.1210 (**24.6% better**) |
| H4 4-bit storage ratio | **PASS** — exactly **4.00×** |
| `Linear4bit` vs bf16 `nn.Linear` | rel. 0.0910 |

---

## 2. What Was Built

| Module | Contents |
|---|---|
| `src/alignlab/lora.py` | **first-principles LoRA** — `LoRALinear`, `apply_lora`, merge/unmerge, adapter save/load |
| `src/alignlab/peft_setup.py` | config → `peft.LoraConfig` / `BitsAndBytesConfig`; parameter accounting incl. 4-bit packing |
| `configs/peft/{none,lora,qlora}.yaml` | three arms differing in exactly one thing |
| `tests/test_lora.py` | **46 tests** |
| E15–E20 | six experiments |

Tests: **290 → 405** (403 passed / 2 skipped local; 404 / 1 server).

---

## 3. LoRA From First Principles — What Was Verified

Every property Phase 4 required, all tested:

| Requirement | Result |
|---|---|
| base weight frozen | **VERIFIED** — frozen in `__init__`; unchanged after optimiser steps |
| A/B trainable | **VERIFIED** |
| output identical at init | **VERIFIED — bitwise identical** to base (B = 0 ⟹ ΔW = 0) |
| gradients only through intended params | **VERIFIED** — every non-adapter param has `grad is None` |
| parameter count correct | **VERIFIED** — `r·(d_in+d_out)`, exact |
| adapter save/load | **VERIFIED** — round-trips through a file; strict in both directions |
| adapter merge | **VERIFIED** — output preserved; double-merge guarded |
| merged == adapter inference | **VERIFIED** (fp32); **see §7c for bf16** |
| `rank(ΔW) ≤ r` | **VERIFIED** |
| forward IS the equation | **VERIFIED** against a hand-written `base(x) + s·B A xᵀ` |

**Initialisation, verified not asserted.** B = 0 with A random is the only
workable choice: both-zero is a dead saddle (both gradients vanish — tested by
deliberately breaking the init), both-random silently perturbs the pretrained
model. Measured: `dL/dA = 0` and `dL/dB ≠ 0` at step 0, so B moves first.

### E18 — educational vs library

| check | result |
|---|---|
| same modules adapted | 8 vs 8, identical |
| same trainable count | 7,168 vs 7,168 |
| logits, matched weights | **0.000e+00** |
| both == base at init | both exactly 0 |
| merged weights vs `merge_and_unload()` | **0.000e+00** |

Exact bitwise agreement. This is what turns "self-consistent" into "correct" —
46 passing tests alone would not have caught a transposed factor.

---

## 4. LoRA Target Modules (E16) — inspected, not assumed

**197 `nn.Linear` modules found**, enumerated from the loaded model:

| leaf | count | in → out | bias |
|---|---:|---|---|
| `lm_head` | 1 | 1536 → 151936 | no |
| `gate_proj`/`up_proj` | 28 each | 1536 → 8960 | no |
| `down_proj` | 28 | 8960 → 1536 | no |
| `q_proj`/`o_proj` | 28 each | 1536 → 1536 | q **yes**, o no |
| `k_proj`/`v_proj` | 28 each | 1536 → **256** | **yes** |

`lm_head` is **excluded from every target set**: it is TIED to the input
embedding, so adapting it would adapt the embedding too. *(This decision turns
out to explain §6's headline finding.)*

**Target set chosen: `q_proj, k_proj, v_proj, o_proj` — 112 matrices.**

| rank | trainable | % of model | adapter (fp32) |
|---:|---:|---:|---:|
| 4 | 1,089,536 | 0.0706% | 4.2 MiB |
| **16** | **4,358,144** | **0.2823%** | **16.6 MiB** |
| 64 | 17,432,576 | 1.1293% | 66.5 MiB |

Cross-check: predicted **4,358,144** == actual **4,358,144**. Adapter file on
disk 17,462,432 B = predicted 17,432,576 + safetensors header.

---

## 5. SVD Rank Selection (E17) — the Tier-1 deliverable

Analysed a **real** ΔW = `W_sft − W_base` from Phase 3's full fine-tune, with
two controls (same-norm Gaussian; the pretrained weight).

**H2 was DISPROVED**, and that is the result.

Layer 13 `q_proj` (1536×1536):

| rank | energy | noise control | recon. error | LoRA params |
|---:|---:|---:|---:|---:|
| 16 | **10.69%** | 3.95% | 0.9450 | 49,152 |
| 64 | **22.95%** | 14.57% | 0.8778 | 196,608 |
| 256 | 54.53% | 46.92% | 0.6743 | 786,432 |

Rank for **90% energy: ~730 of 1536**. The noise control needs **783** — real
structure, but a **7% margin**, not an order of magnitude.

**Break-even for a 1536×1536 matrix is rank 768.** So the naive procedure
recommends r=730, costing 2,242,560 against the dense 2,359,296 — a **4.9%
saving**. For the narrow GQA projections, rank 256 already costs *more* than
dense.

**Why this does not sink LoRA.** LoRA's premise is not "full FT's ΔW is
low-rank"; it is "**some** low-rank ΔW suffices". Unconstrained gradient descent
has no incentive to be low-rank, so rank 730 measures how unconstrained the
optimiser was.

**Also measured (H4, held strongly):** `‖ΔW‖/‖W‖` = **0.0013–0.0050**. One
epoch moves the weights by about a quarter of one percent.

**PCA connection**, verified numerically (executed): covariance eigenvalues
**identical** to `σ²/(n−1)`; principal directions match right singular vectors
to `|cos| = 1.0000000000`; centring changes the first/second singular-value
ratio from **45.7 → 1.5**. ΔW needs no centring — it is a matrix difference,
not a data cloud, so calling its SVD "PCA" is a category error.

Six reasons the SVD reading is a heuristic are recorded in E17 and the study
note.

---

## 6. Full SFT vs LoRA vs QLoRA (E19)

**Controlled:** model + revision, dataset (**identical fingerprint
`17bebb1758a66958` across all arms — verified**), objective, `max_length=1024`,
packing off, batch 4×8=32, 1 epoch/296 steps, seed 42, evaluation set and code
path, hardware.

**Not controllable:** learning rate. LoRA needs a higher LR (adapters start at
zero). Holding it fixed measures LR sensitivity; changing it breaks the control.
**Both were run.**

### Training — MEASURED

| arm | lr | trainable | % | peak VRAM | time | samp/s | eval loss | checkpoint |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| full SFT | 2e-5 | 1,543,714,304 | 100% | **14.42 GiB** | 1168 s | 8.08 | **1.9893** | 2.886 GiB |
| LoRA r=16 | 2e-4 | 4,358,144 | 0.2815% | 4.72 GiB | 1055 s | 8.95 | 2.0402 | 0.027 GiB |
| LoRA r=16 | 2e-5 | 4,358,144 | 0.2815% | 4.72 GiB | 1053 s | 8.96 | 2.0718 | 0.027 GiB |
| QLoRA r=16 | 2e-4 | 4,358,144 | 0.2815% | **2.88 GiB** | 1044 s | 9.04 | 2.0769 | 0.019 GiB |

*(Full-SFT peak VRAM measured by a 12-step probe at identical batch geometry —
peak VRAM depends on geometry and optimizer state, not step count. Phase 3's
run predates the instrumentation. Its checkpoint was deleted after measuring.)*

**Memory: full SFT 3.05× LoRA, 5.01× QLoRA. Checkpoint: 107× smaller.**

| hypothesis | outcome |
|---|---|
| H1 all PEFT arms 4,358,144 trainable | **HOLDS** |
| H2 QLoRA VRAM < LoRA | **HOLDS** — 2.88 vs 4.72 GiB |
| H3 QLoRA slower than LoRA | **DISPROVED** — 1044 vs 1055 s, 1% *faster* |
| H4 adapter >100× smaller | **HOLDS** — 107× |
| H5 full SFT < LoRA@2e-4 < LoRA@2e-5 on eval loss | **HOLDS** |

**On H3, the honest reading is not "QLoRA is faster."** A 1% gap at one run per
arm is indistinguishable from run-to-run variation; dequantization overhead is
simply **not detectable at this scale**. The smoke runs *did* show QLoRA 16%
slower (16.25 s vs 13.99 s over 20 steps), where the one-off cost of quantizing
1.5B weights at load dominates and does not represent steady state.

**On H5, the LR mattered.** The LR effect (0.032 nats) is ~60% of the
LoRA-vs-full-FT gap (0.051 nats). Reporting only the LR-matched arm would have
**overstated LoRA's cost by about half**.

### Evaluation — MEASURED (Phase 3's methodology, unchanged)

| model | completion ppl | stopped | emitted `<\|im_end\|>` |
|---|---:|---:|---:|
| base (untrained) | 8.541 | 1/4 | 0/4 |
| **full SFT** | **7.199** | **4/4** | **4/4** |
| LoRA @2e-4 | 7.449 | 0/4 | 0/4 |
| LoRA @2e-5 | 7.624 | 0/4 | 0/4 |
| QLoRA @2e-4 | 7.511 | 0/4 | 0/4 |

---

## 7. Unexpected Findings

### 7a. **No PEFT arm learned to stop — and perplexity almost hides it**

Perplexity separates the arms by 3.5–6%. Stop-token behaviour separates them
**completely**: full fine-tuning terminates on 4/4 prompts, **no PEFT arm on
any**. All three write fluent, on-topic answers and run to the token cap.

**E20 found the mechanism.** At the position where `<|im_end|>` should appear:

| model | P(`<\|im_end\|>`) | median rank | argmax |
|---|---:|---:|---:|
| base | 0.00000 | 31,600 | 0.0% |
| **full SFT** | **0.39513** | **1** | **79.0%** |
| LoRA @2e-4 | 0.00021 | 160 | 0.0% |
| LoRA @2e-5 | 0.00006 | 2,217 | 0.0% |
| QLoRA @2e-4 | 0.00019 | 194 | 0.0% |

LoRA **did** learn something (rank 31,600 → 160, 74.8× the base probability)
and still lost decisively — full SFT is ~1,880× higher.

**The structural explanation.** Emitting a rare token means moving its *logit*,
produced by `lm_head` — which is **tied to the embedding** and therefore
**excluded from our target set**. Full fine-tuning moved that matrix by
relative **0.0136**, against E17's **0.0013–0.0050** for attention: the
**largest relative change of any matrix measured in this project**, and exactly
the one LoRA cannot reach.

> **LoRA's capability ceiling is set by WHICH matrices it can reach, not only by
> rank.** A behaviour whose mechanism lives outside the target set is
> unreachable at any rank.

**NOT CONFIRMED.** This is consistent with the hypothesis, not proof. The
decisive test — a LoRA variant adapting the MLP or an untied head — is one more
~18-minute run and is **deferred, not skipped** (§10).

### 7b. E17's H2 disproved — the real ΔW is not low-rank (§5)

### 7c. **Merging a LoRA adapter in bf16 is lossy**

| dtype | merged vs unmerged logits, max | mean |
|---|---:|---:|
| bf16 | **5.625e-01** | 5.00e-02 |
| float32 | 6.998e-05 | 4.70e-06 |

~8,000× worse. Cause: `‖ΔW‖/‖W‖ ≈ 0.003` sits at the resolution of bf16's
~8-bit mantissa, so most of the update rounds away when added to the much
larger base weight.

E18 verified merge exactness **in float32**, and that result stands. What did
not transfer is the assumption that it holds in the dtype these models are
served in. Found by derisking E19 on a real adapter. **E19 was changed to
evaluate unmerged**, so a merge artefact is not attributed to a training arm.

Generation was character-identical across all four merged/unmerged × dtype
combinations on the probe prompt — "large logit difference" and "different
behaviour" are not the same claim.

### 7d. QLoRA's adapter is smaller than LoRA's (0.019 vs 0.027 GiB)

Same 4,358,144 parameters. With a 4-bit base, peft stores adapters in the bf16
compute dtype rather than fp32 — half the bytes.

---

## 8. Bugs / Corrections

**8a. A test model that could not receive gradients.** `test_gradients_reach_A_and_B_only`
failed on first run for `k_proj`/`v_proj`. Correctly: `TinyModel.forward` used
only `q_proj`/`o_proj`. **The bug was in the test model, not in LoRA.**

**8b. Arithmetic error in E17's own text.** `730 × 3072` printed as 2,240,160;
it is **2,242,560**. Caught by *executing* the parameter-cost example.
Conclusion unaffected.

**8c. Hard-coded server paths in E19 — and I pushed the failure.**
`tests/test_no_hardcoded_paths` caught `/data/home/rsoumyadeep/...` in argparse
defaults; three tests failed. I did not notice because the commit chain was
`pytest | tail -2 && git commit`, so the pipeline's exit status was `tail`'s
success, not pytest's failure. **Three failing tests were pushed.** Both fixed;
the lesson — never pipe a gate through another command — is recorded.

**8d. `triton` installed but unimportable.** `ModuleNotFoundError: No module
named 'setuptools'`. Latent until Phase 4, since nothing had imported triton.

**8e. `tokenizer.additional_special_tokens` removed in transformers 5.x.**
`get_added_vocab()` is the replacement — the **fifth** transformers-5 API break
this project has hit by running into it.

**8f. `merge_precision` NameError in E19.** An earlier patch did not land
(shell escaping), so E19 crashed writing its payload *after* printing all
results. Fixed and re-run.

---

## 9. Resources Actually Inspected

**Two papers, PARTIALLY INSPECTED** (abstracts fetched programmatically; full
PDFs **NOT** read): LoRA (2106.09685), QLoRA (2305.14314).

**Four inspected by EXECUTION** (stronger than reading, and machine-specific):
bitsandbytes 0.50.2, peft 0.20.0, TRL's peft/quantization path,
Qwen2.5-1.5B's real module tree.

**Eckart–Young and the PCA↔SVD identity** are recorded as **NOT READ FROM A
SOURCE** and verified numerically instead, rather than given a citation we did
not open.

Two places our measurements meet the papers:

- LoRA's "**10,000×** fewer trainable parameters" — we measure **354×**
  (attention r=16) and 2,834× (q/v r=4). Not a contradiction: dense cost grows
  as `d²`, adapter cost as `r·d`, so the ratio scales with width and GPT-3 175B
  is ~100× wider. **Quoting 10,000× for a 1.5B model would be wrong.**
- QLoRA's "NF4 is information theoretically optimal for normally distributed
  weights" — E15 measured NF4 **24.6% better** than FP4. Consistent with the
  claim; a two-way comparison cannot establish optimality.

**NOT INSPECTED:** both full papers, Intrinsic Dimensionality (directly
relevant to E17), LoRA-FA/DoRA/rsLoRA, peft source in depth, GPTQ/AWQ.
**No claim here rests on them** — including E17's disproof, which is stated from
our own measurement without the literature that would contextualise it.

---

## 10. Limitations

1. **Single seed, one run per arm.** No variance estimate. The 1% H3 gap and
   the 0.05-nat H5 gaps are not distinguishable from noise on this evidence.
2. **Learning rate confounded with arm** — mitigated by running both, not solved.
3. **No rank sweep.** r=16 is a starting point, not tuned — the very thing §5
   says you should sweep.
4. **Only one target set trained.** `attention_mlp` was costed (E16) but never
   trained, so §7a's structural explanation is supported, not confirmed.
5. **In-distribution evaluation only** — no_robots' own test split.
6. **4 qualitative prompts**, greedy. Indicative, not a measurement.
7. **No LLM-as-judge** — Phase 7.
8. **Paged optimizers not used** (unnecessary at 2.88 GiB) — a QLoRA component
   we did not exercise.
9. **NF4-vs-FP4 measured on synthetic normal data**, not real weights.
10. **Single GPU.** The second A6000 was used only to parallelise evaluation.
11. **Adapter resume NOT TESTED** end-to-end.

---

## 11. What Phase 4 Does **Not** Establish

- That LoRA/QLoRA are "as good as" full fine-tuning. On this dataset they are
  **measurably worse** on the one behaviour a user would notice most.
- That r=16 or the attention-only target set is a good choice — neither was swept.
- That the `lm_head` explanation for §7a is **causal**.
- Anything about DPO, reward modelling, or preference learning.
- That any result generalises beyond no_robots and this model.

---

## 12. Git / Synchronization State

```
Branch : phase-4-peft   (local + GitHub + server, same commit)
Base   : d5cda3a (end of Phase 3, on phase-3-sft)
Remote : git@github.com:rsoumyadeep/AlignLab-posttraining.git  [PRIVATE]
Merged : none. main at d21c070; phase-1-foundation at 71ed4e6; phase-3-sft at d5cda3a.
Working tree: clean on both machines
```

No history rewritten, no force-push. **`checkpoint-295` (8.7 GiB) and `final/`
(2.9 GiB) preserved as instructed** — verified present after all Phase 4 work.
The only deletion was the VRAM probe's own checkpoint, created and removed
within this phase. `/data`: 80 GiB free.

---

## 13. Explain-Back Checkpoints Deferred

Phase 4 adds **Drill 1 (LoRA via SVD)**, **Drill 2 (PCA)** and **Checkpoint 16
(LoRA initialisation)**, plus `PYTORCH_CONCEPTS` My Understanding,
`LEARNING_RESOURCES` × 6 Own Words, and the `INTERVIEW_DEFENSE` Explain Back.

**Running total: 20 checkpoints + 33 Own Words + Drills 3–5.** All marked
`DEFERRED — USER EXPLAIN-BACK REQUIRED`. None answered on the USER's behalf.

---

**PHASE 4 STATUS: ENGINEERING, EXPERIMENTS, VERIFICATION, TRAINING AND
DOCUMENTATION COMPLETE — USER EXPLAIN-BACK CHECKPOINTS DEFERRED BY INSTRUCTION.**

**DPO NOT STARTED. Phase 5 NOT STARTED.**
