# Phase 6 Report — Direct Preference Optimization

**Date:** 2026-08-30 · **Branch:** `phase-6-dpo`
**Machine:** `csrslave`, 1 × RTX A6000, bf16
**Policy & reference:** Phase 3 SFT checkpoint, from `Qwen/Qwen2.5-1.5B` @ `8faed761…`

**Headline:** the pre-registered sweep produced a **null result on the primary
metric**, and the cause is now measured rather than guessed. §7 is the finding
that matters.

---

## 1. What Was Built

| Component | Status |
|---|---|
| `src/alignlab/dpo.py` — implicit rewards, DPO loss, stats, 2 safety checks | **IMPLEMENTED (ours, not TRL)** |
| `src/alignlab/dpo_train.py` — the training loop | **IMPLEMENTED (ours, not TRL)** |
| `configs/dpo.yaml`, `configs/dpo/{default,smoke}.yaml` | **IMPLEMENTED** |
| `tests/test_dpo.py` — 41 tests | **IMPLEMENTED** |
| `docs/phase6/BETA_PREREGISTRATION.md` | **committed 13:15:12, before any DPO code** |
| E23 — SFT vs DPO evaluation | **MEASURED** |
| `PYTORCH_CONCEPTS/examples/dpo_mechanics_examples.py` | **EXECUTED** |

Tests: **460 → 501** (500 passed / 1 skipped server; 499 / 2 local).

**TRL's `DPOTrainer` was not used.** The loss and the loop are both ours.

---

## 2. What Was Verified

- **The loss matches the paper's own reference implementation to 1e-12** at
  β ∈ {0.01, 0.1, 0.5, 1.0}. Appendix B was read verbatim; our grouping
  (per-response implicit rewards, then their difference) differs from theirs
  but is algebraically identical, and was written **before** the appendix was
  read.
- **Loss at initialisation is exactly `log 2`** — verified in unit tests and on
  every real run (`0.6931471824645996`).
- **Implicit reward is exactly 0 when π = π_ref** — `max_abs_reward = 0.0` on
  all four runs. The trainer **refuses to start** otherwise.
- **Reference is frozen**: 0 trainable parameters, `eval()` mode, checked at
  runtime and in tests. A trainable reference raises.
- **Gradients flow to the policy only**, with the correct signs.
- **`logsigmoid` stability**: finite at a −1000 log-probability gap, where
  `log(sigmoid(x))` returns `−inf`.
- **Prefix consistency**: 0 violations in 400 sampled sequences per run.
- **Properties of the algebra**, not just the code: β scales the logit
  linearly; swapping chosen/rejected negates it; `rank(ΔW) ≤ r` analogue —
  larger β saturates, verified against closed-form values.

---

## 3. What Was Measured

### Training configuration (identical across all pre-registered arms)

| | |
|---|---|
| Policy start | `checkpoints/sft-qwen1p5b-noRobots-001/final` |
| Reference | the **same** checkpoint, frozen |
| Model revision | `8faed761d45a263340a0528343f099c05c9a4323` (PINNED) |
| Dataset | `HuggingFaceH4/ultrafeedback_binarized`, `train_prefs`/`test_prefs` |
| Train fingerprint | `1e48de32d0d5ece3b345948b…` |
| Eval fingerprint | `b3bc775a6db554cc75715221…` |
| Ties | **kept** (`drop_ties=False`) — unchanged from Phase 5 |
| Objective | **SUM** (published) |
| Pairs | 1,869 train / 184 eval usable (131 / 16 dropped as over-length) |
| Steps | 116 (1 pair/forward × 16 accumulation) |
| LR | 5e-7, cosine, warmup 0.1, AdamW, grad-clip 1.0 |
| Seed | 42 · dtype bf16 · gradient checkpointing on |
| Runtime | 636–645 s per arm |
| **Peak VRAM** | **18.56 GiB** (vs ~14.4 GiB for Phase 3 full SFT) |
| Final model | 2.89 GiB per arm |

### 4. Beta Experiment — the pre-registered sweep

| model | **SUM (primary)** | MEAN (diag) | reward acc | margin | KL median | gen len | im_end |
|---|---:|---:|---:|---:|---:|---:|---:|
| SFT baseline | **46.7%** | 58.7% | — | 0.0000 | 0.0000 | 93.8 | 4/4 |
| DPO β=0.01 | **46.7%** | 58.7% | 51.1% | −0.0001 | 0.0008 | 95.8 | 4/4 |
| DPO β=0.1 | **46.7%** | 58.7% | 51.1% | +0.0041 | 0.0008 | 96.0 | 4/4 |
| DPO β=0.5 | **46.7%** | 58.7% | 54.3% | +0.0090 | 0.0008 | 95.8 | 4/4 |

**Preference accuracy did not change by a single pair out of 184 in any arm.**

The reward margin **is** monotone in β (−0.0001, +0.0041, +0.0090), and β=0.1
moved `log π(chosen)` up (+0.0199) and `log π(rejected)` down (−0.0206). The
mechanism works; the magnitude does not.

### Hypothesis outcomes

| | prediction | outcome |
|---|---|---|
| **H1** | all β reduce loss below `log 2` | **NOT SUPPORTED** — logged losses are single-pair noise (§9a); eval margins moved only for β ≥ 0.1 |
| **H2** | SUM accuracy exceeds 47.2% for some β | **DISPROVED** — unchanged at 46.7% everywhere |
| **H3** | KL monotone decreasing in β | **NOT SUPPORTED** — all three at 0.0008 |
| **H4** | length increases for some β | **DISPROVED in the sweep** (+2%); **CONFIRMED in the post-hoc run** (+30%) — see §6 |
| **H5** | SUM and MEAN may diverge | **NOT OBSERVED in the sweep**; observed post-hoc (SUM flat, MEAN +0.5p) |

---

## 5. Baseline vs DPO

The pre-DPO baseline was re-measured **in-process, before the first optimiser
step**, on the same batches and the same code path: **46.7% SUM**. Phase 5
measured 47.2% on a different subsample — consistent.

Qualitative outputs across the sweep are near-identical to the baseline: one
phrase differs ("for that time" → "for the same time"). Stop-token behaviour is
intact at 4/4 in every arm — the capability Phase 4 found LoRA could not learn.

---

## 6. Post-hoc diagnostic — **NOT pre-registered**

The sweep's null had an unidentified cause, so **one** additional run was
launched *after* seeing those results, at **10× the learning rate** (5e-6,
β=0.1, everything else identical). It is labelled `[POST-HOC]` in every output
and **cannot displace the primary result**.

| model | SUM | MEAN | reward acc | margin | KL median | **gen len** |
|---|---:|---:|---:|---:|---:|---:|
| SFT baseline | 46.7% | 58.7% | — | 0.0000 | 0.0000 | 93.8 |
| DPO β=0.1 (lr 5e-7) | 46.7% | 58.7% | 51.1% | +0.0041 | 0.0008 | 96.0 |
| **[POST-HOC] β=0.1, lr 5e-6** | **46.7%** | **59.2%** | **65.2%** | **+0.0700** | 0.0017 | **122.0** |

At 10× the learning rate the policy moved **~49× further**
(`log π(chosen)` +0.9773 vs +0.0199), the reward margin grew **17×**, and
reward accuracy reached **65.2%**.

**Two things followed, and both matter:**

1. **SUM preference accuracy still did not move.** Not one pair.
2. **Generated length rose 30%** (93.8 → 122.0 tokens, max 146 → 206) — the
   length trap Phase 5 predicted, materialising as soon as training had enough
   signal to exploit it. **H4 is confirmed here**, having been disproved in the
   under-powered sweep.

---

## 7. The finding — why SUM accuracy cannot move

This is Phase 6's real result, and it is arithmetic, not speculation.

**Measured on the evaluation set:**

| | chosen | rejected |
|---|---:|---:|
| mean summed log-prob | −291.12 | −261.65 |
| mean tokens | 271.7 | 242.2 |
| **mean per-token log-prob** | **−1.0713** | **−1.0803** |

**Per token, the chosen response is genuinely more likely** (−1.0713 vs
−1.0803). The SUM comparison inverts that verdict purely because chosen carries
**29.5 more tokens**:

```
SUM gap (chosen − rejected)      = −29.47 nats
explained by length alone        = −31.90 nats
residual once length is removed  =  +2.43 nats   (chosen is BETTER)
```

> **The 46.7% SUM baseline is a length artifact, and the artifact is ~29 nats
> deep.** MEAN removes the length term and reports 58.7% — which is why Phase 5
> saw an 11-point swing on identical models and data.

**And that is why DPO cannot move it.** To flip the *average* pair under SUM,
the policy must shift the chosen-vs-rejected log-probability gap by ~29 nats:

| run | relative movement achieved | shortfall |
|---|---:|---:|
| β=0.1, lr 5e-7 | +0.0405 nats | **728× short** |
| [POST-HOC] lr 5e-6 | +0.7004 nats | **42× short** |

Even the 10× run is 42× short. Closing a 29-nat gap would require distorting
the model far beyond anything these runs approach — and the MEAN metric, which
*did* move (+0.5p), is the one where a 0.7-nat shift is meaningful.

**So the null is not "DPO didn't train."** DPO trained, in the correct
direction, monotonically in β. The primary metric is simply **dominated by a
property of the data rather than of the model** — the same lesson as Phase 3's
masked loss, Phase 4's stop-token gap, and Phase 5's baseline swing.

---

## 8. Length Analysis (mandatory)

| model | gen mean | gen max | chosen tok | rejected tok | im_end | hit cap |
|---|---:|---:|---:|---:|---:|---:|
| SFT baseline | 93.8 | 146 | 271.7 | 242.2 | 4/4 | 0/4 |
| β=0.01 | 95.8 | 145 | 271.7 | 242.2 | 4/4 | 0/4 |
| β=0.1 | 96.0 | 146 | 271.7 | 242.2 | 4/4 | 0/4 |
| β=0.5 | 95.8 | 145 | 271.7 | 242.2 | 4/4 | 0/4 |
| **[POST-HOC] lr 5e-6** | **122.0** | **206** | 271.7 | 242.2 | 4/4 | 0/4 |

- Sweep: **+2%** — no pathology, and no opportunity for one (the model barely moved).
- Post-hoc: **+30%** — the predicted pathology, appearing exactly when training
  had enough signal.
- **No arm lost the stop token.** 4/4 throughout, 0/4 hit the length cap.

`chosen tok` / `rejected tok` are properties of the evaluation *data*, constant
by construction — included because they are the 29.5-token gap driving §7.

---

## 9. Bugs / Corrections

**9a. Our per-step logging is single-pair, not a window average.**
`history` records the last pair of each gradient-accumulation window, so a
±0.01 margin there is **noise, not trend**. Found while reading the sweep logs.
The training code was **deliberately not patched mid-sweep** so all arms ran
identical code. All conclusions use the 184-pair eval figures instead.

**9b. `reward_accuracy` "0.0000 → 0.5109" is not a 51-point gain.** At baseline
every margin is *exactly* zero, so a `margin > 0` test counts none. The
baseline is **degenerate, not terrible**.

**9c. A test threshold I invented.** `test_larger_beta_saturates_sooner`
asserted `loss < 0.01` at β=1; the true value is **0.0181**. Replaced with the
closed-form values, since the logit there is exactly `4β`. **The code was right;
the assertion was a guess.**

**9d. A float32 claim written from expectation.** I wrote that `sigmoid`
underflows at x = −80. Executing showed `log(sigmoid(-80)) = -80.000000`; it
underflows at **−200**. Corrected, with the error recorded.

**9e. Two mis-transcribed table values** in the PyTorch note (0.691148/0.673284
against the executed 0.691149/0.673347).

---

## 10. Resources Actually Inspected

**DPO paper (arXiv 2305.18290), HTML full text — PARTIALLY INSPECTED**, and
this is an **upgrade** from Phase 5's abstract-only status:

- **Read in full:** the section/appendix heading list; §3 Preliminaries
  including Eq. 3 and Eq. 4 with surrounding text; **Appendix B** (implementation
  details, reference code, hyperparameters).
- **NOT read:** §§1, 2, 5, 6, 7; Appendices A.1–A.3, A.5, A.6, C, D.

Claims obtained include Eq. 4 and its partition function, the statement that
*"β is a parameter controlling the deviation from the base reference policy
π_ref, namely the initial SFT model"*, and the hyperparameters: *"we use β = 0.1,
batch size of 64 and the RMSprop optimizer with a learning rate of 1e-6 …
For TL;DR summarization, we use β = 0.5."*

**Our deviation, recorded not hidden:** AdamW at 5e-7 with effective batch 16.
**These results are not a replication.**

**Read but NOT resolved:** the rendered Eq. 21 in Appendix A.4 shows the
gradient's sigmoid argument with `y_l` and `y_w` in the reverse order from
Eq. 7. Possibly a rendering artefact, a local sign convention, or an erratum.
Nothing here depends on it — our loss is verified against Appendix B's code.

**NOT INSPECTED:** the paper's theory and experiment sections; TRL's
`DPOTrainer`; IPO/KTO/ORPO; Bradley & Terry (1952). **We make no claim about
DPO's stability or win rates** — those are §6 results we did not read.

---

## 11. Documentation Created

| Folder | File |
|---|---|
| `STUDY_WITH_CLAUDE/` | `phase6/01_dpo_theory_and_results.md` |
| `CODE_EXPLANATION/` | `phase6/README.md` |
| `PYTORCH_CONCEPTS/` | `pytorch-dpo-mechanics.md` + executed example |
| `LEARNING_RESOURCES/` | `phase6_resources.md` |
| `INTERVIEW_DEFENSE/` | `phase6_dpo.md` — 11 questions |
| `docs/phase6/` | pre-registration, this report, 5 evidence JSONs |

---

## 12. Limitations

1. **Training budget is the binding constraint.** 116 steps × 16 pairs = 1,856
   pairs at lr 5e-7, against the paper's batch-64 RMSprop at 1e-6.
2. **One seed per arm.** No variance estimate.
3. **No LR sweep pre-registered.** β was swept; LR turned out to matter more —
   established only post-hoc.
4. **Per-step logging is single-pair** (§9a).
5. **In-distribution evaluation only** — UltraFeedback's own test split.
6. **184 eval pairs.** A 1-pair flip is 0.5 percentage points.
7. **4 qualitative prompts**, greedy.
8. **No LLM-as-judge** — Phase 7.
9. **Ties kept** (11.9%), length imbalance **not** addressed.
10. **Not a replication** of the paper (different optimizer, LR, batch).
11. **The post-hoc run is post-hoc** — one run, not pre-registered, and not
    evidence for a pre-registered claim.

---

## 13. What The Results Prove

- Our DPO implementation is **correct**: matches the paper's reference code to
  1e-12, gives exactly `log 2` at initialisation, and moves the policy in the
  correct direction, monotonically in β.
- The safety checks work: frozen reference, zero implicit reward at init, both
  enforced at runtime on every run.
- At this budget, DPO changed the policy by **KL 0.0008** from the reference —
  ~250× less than one epoch of SFT moved from the base.
- **The 46.7% SUM baseline is a length artifact ~29 nats deep**, and per-token
  the chosen responses are genuinely more likely (+2.43 nats residual).
- **Higher LR produces the predicted length pathology** (+30% generation
  length) while still not moving the SUM metric.

## 14. What The Results Do **Not** Prove

- **Not** that DPO fails, or that it cannot improve this model. The budget was
  too small, and the diagnostic shows the policy moves readily with more LR.
- **Not** that β is unimportant — the margin was monotone in β; the arms simply
  all landed below the metric's resolution.
- **Not** anything about assistant quality. Preference accuracy on
  UltraFeedback measures fit to UltraFeedback.
- **Not** that MEAN is the "right" objective. It is a diagnostic; the published
  objective is SUM, and Phase 6 trained SUM.
- **Not** that 30% length growth is harmful — no quality evaluation was run on
  the post-hoc model.
- **No causal claim** — one seed, one dataset, one model.

---

## 15. Tests

**501 total: 500 passed / 1 skipped (server); 499 / 2 skipped (local).**
Phase 6 added **41**: implicit rewards (4), loss and its algebra (10), stats
(3), SUM-vs-MEAN divergence (3), frozen reference (3), zero reward at init (2),
config pre-registration (10), **paper-reference agreement (6)**.

---

## 16. Git / Synchronization State

```
Branch : phase-6-dpo   (local + GitHub + server, same commit)
Base   : e6bf65d (end of Phase 5)
Remote : git@github.com:rsoumyadeep/AlignLab-posttraining.git  [PRIVATE]
Merged : none. main d21c070; phase-1-foundation 71ed4e6;
         phase-3-sft d5cda3a; phase-4-peft 08bdf19; phase-5-preference e6bf65d
Working tree: clean on both machines
```

No history rewritten, no force-push. **Phases 4 and 5 not reopened.**
**`checkpoint-295` (8.7 GiB) and `final/` (2.9 GiB) preserved** — verified.
Four DPO models retained at 2.9 GiB each. `/data`: **65 GiB free**.

---

## 17. USER Explain-Back Checkpoints

Phase 6 adds **Checkpoints 20–22** (derive the DPO loss; interpret a null
result; what β trades off), plus `PYTORCH_CONCEPTS` My Understanding,
`LEARNING_RESOURCES` × 3 Own Words, and the `INTERVIEW_DEFENSE` Explain Back.

**Running total: 26 checkpoints + 42 Own Words + Drills 3–5.** All marked
`DEFERRED — USER EXPLAIN-BACK REQUIRED`. None answered on the USER's behalf.

---

## 18. Phase 7 Prerequisites

**Available:**
1. Five models to evaluate: base, SFT, LoRA, QLoRA, and four DPO variants.
2. A working evaluation harness (E13 → E19 → E23) using one consistent
   methodology across four phases.
3. Length tracking already instrumented.
4. 65 GiB free; both GPUs idle.

**Decisions Phase 7 must make:**
5. **A judge model for LLM-as-judge** — none chosen; Phase 5 and 6 deliberately
   avoided it.
6. **An out-of-distribution evaluation set.** Every result so far is
   in-distribution and therefore partly circular.
7. **Whether to report SUM or MEAN preference accuracy** — §7 shows they measure
   different things, and SUM is length-dominated on this data.
8. **Whether to re-run DPO with adequate budget.** The post-hoc diagnostic
   suggests it would move; that would be a **new pre-registered experiment**,
   not a Phase 6 amendment.
9. **Multiple seeds**, to separate small effects from noise.

**Still NOT TESTED:** W&B online, adapter resume end-to-end, TRL's `DPOTrainer`.

---

**PHASE 6 STATUS: DPO IMPLEMENTED FROM FIRST PRINCIPLES, VERIFIED AGAINST THE
PAPER'S REFERENCE CODE, TRAINED, EVALUATED AND DOCUMENTED. THE PRE-REGISTERED
SWEEP RETURNED A NULL ON THE PRIMARY METRIC, AND THE CAUSE IS MEASURED.
USER EXPLAIN-BACK CHECKPOINTS DEFERRED BY INSTRUCTION.**

**Phase 7 NOT STARTED.**
