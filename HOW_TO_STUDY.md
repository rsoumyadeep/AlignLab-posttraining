# HOW TO STUDY AlignLab

**Purpose.** This file answers one question: *how do I actually learn this
project well enough to defend it under hostile technical questioning?*

It is **not** a build plan. Nothing in this document asks you to implement a
feature, invent an experiment, retrain a model, or edit a result. AlignLab's
implementation is complete (Phases 1–8, `docs/FINAL_STUDY_HANDOFF.md` §1). The
work left is *yours*: understanding, deriving, locating, and defending.

**Relationship to `STUDY_SCHEDULE.md`.** That file is the *calendar* — ten days,
1–10 September 2026, with concepts and checklists per day. This file is the
*method* — for each phase, what to learn, in what order, from which file, to
what depth, against which experiment, and under which interview attack. Use both:
`STUDY_SCHEDULE.md` tells you *when*, `HOW_TO_STUDY.md` tells you *how*.

**Source of truth.** Every number, filename, experiment ID and verdict below was
read out of this repository. Where the repository is silent, this file says so.
Where two repository documents disagree, §14 records the disagreement rather than
picking a winner.

**Verified while writing this file (2 September 2026):**

| Check | Result |
|---|---|
| `python -m pytest -q` (local, Windows CPU) | **607 passed, 3 skipped**, 73 s |
| Skips | CUDA-only test, SIGUSR1 (not on Windows), `wandb` not installed |
| Branch | `phase-7-evaluation` |

---

## Table of contents

1. [The four things you must never blur](#1-the-four-things-you-must-never-blur)
2. [Vocabulary you must use exactly](#2-vocabulary-you-must-use-exactly)
3. [The dependency graph](#3-the-dependency-graph)
4. [The study loop](#4-the-study-loop)
5. [Phase 1 — engineering foundation](#5-phase-1--engineering-foundation)
6. [Phase 2 — transformer components](#6-phase-2--transformer-components)
7. [Phase 3 — supervised fine-tuning](#7-phase-3--supervised-fine-tuning)
8. [Phase 4 — PEFT: LoRA, SVD, QLoRA](#8-phase-4--peft-lora-svd-qlora)
9. [Phase 5 — preference learning foundations](#9-phase-5--preference-learning-foundations)
10. [Phase 6 — DPO](#10-phase-6--dpo)
11. [Phase 7 — evaluation](#11-phase-7--evaluation)
12. [Phase 8 — reproducibility and project defence](#12-phase-8--reproducibility-and-project-defence)
13. [Interview chains](#13-interview-chains)
14. [Discrepancies in the repository you should know about](#14-discrepancies-in-the-repository-you-should-know-about)
15. [High-ROI ranking](#15-high-roi-ranking)
16. [Don't waste time on these](#16-dont-waste-time-on-these)
17. [Active-recall checkpoint bank](#17-active-recall-checkpoint-bank)
18. [Final mastery test](#18-final-mastery-test)
19. [How to use this file](#19-how-to-use-this-file)

---

## 1. The four things you must never blur

Almost every way an interview goes wrong here is a collapse of one of these four
into another. Keep them separate in your speech, not just in your head.

| Layer | What it is | Example |
|---|---|---|
| **(1) IMPLEMENTED** | Code that exists in this repository and runs | `src/alignlab/lora.py` implements `W + (α/r)·BA` with `B=0` at init, and `tests/test_lora.py` asserts it |
| **(2) MEASURED** | A number produced by a committed run, traceable to an artefact | E17 measured, on layer 13 `q_proj`, that r=16 captures **10.69%** of ΔW's energy and 90% needs rank **~730 of 1536** |
| **(3) CONCLUDED** | An interpretation the project chose to draw from (2) | "The measured update is not low-rank, so the popular explanation of *why* LoRA works is not supported by this data" |
| **(4) MY DEFENCE JOB** | What *you* must be able to say when challenged | "…and that does **not** show LoRA is ineffective: LoRA needs *some* low-rank update to suffice, not full fine-tuning's update to *be* low-rank" |

A worked contrast, because this is the single most-tested chain in the project:

- **General theory (not AlignLab's claim):** "LoRA is low rank."
- **(1) Implemented:** `LoRALinear`, `apply_lora`, merge/unmerge, and an SVD
  analysis script `scripts/experiments/e17_svd_rank_selection.py`.
- **(2) Measured:** on the real ΔW = W_sft − W_base from Phase 3, layer 13
  `q_proj` (1536×1536): rank for 90% energy ≈ **730**; a same-norm Gaussian
  control needs **783**; r=16 captures **10.69%**; ‖ΔW‖/‖W‖ ≈ **0.0013–0.0050**.
- **(3) Concluded:** H2 **DISPROVED**. The project forbids itself from ever
  saying "LoRA works because fine-tuning updates are low rank"
  (`docs/EXPERIMENT_REGISTRY.md` E17, "Consequence").
- **(4) Your job:** explain why the disproof is about the *explanation*, not the
  *method*; explain why r=730 is useless anyway (break-even rank for 1536×1536 is
  **768**, so the naive recipe saves 4.9%); and explain what experiment would
  settle it.

Say the layer out loud when you answer. "We measured…" and "I'd conclude…" are
different sentences and interviewers notice which one you use.

---

## 2. Vocabulary you must use exactly

The repository uses two controlled vocabularies. Using them correctly is itself
a signal; using them loosely destroys the project's main selling point.

**Verdict vocabulary** (`docs/EXPERIMENT_REGISTRY.md`):

| Word | Meaning | A real example |
|---|---|---|
| **HOLDS** | Hypothesis survived the test | E1, E3, E5, E7, E21 |
| **DISPROVED** | Stated hypothesis failed | E11, E12 H5, E17 H2, E19 H3, E22 H4 |
| **NOT CONFIRMED** | Consistent with the evidence, not decisively tested | **E20** |
| **MEASURED** | Measurement with no prior hypothesis | E13, E14, E16, `eval-full-001` |
| **UNTESTABLE AS STATED** | The criterion could not decide the question | E22 H1 |
| **NULL RESULT** | Preregistered comparison that moved nothing | β sweep, E23 |
| **POST-HOC** | Run after seeing results, outside preregistration | `diag-dpo-beta0.1-lr5e-6` |

**Evidence vocabulary** (`README.md` §17): `IMPLEMENTED` · `VERIFIED` ·
`MEASURED` · `RECORDED` · `NOT CONFIRMED` · `UNVERIFIED` · `NOT TESTED` ·
`DEFERRED`.

**Three sentences you must never say** (from `docs/FINAL_STUDY_HANDOFF.md` §8):

1. "LoRA works because fine-tuning updates are low rank." — E17 disproved it here.
2. "DPO improved the model." — SUM accuracy was 86/184 for the baseline *and*
   every arm.
3. "LoRA fails to stop **because** `lm_head` is outside the target set." — E20 is
   **NOT CONFIRMED**; say "consistent with".

---

## 3. The dependency graph

Adjusted to what this repository actually contains. Read top to bottom; the
arrows are hard prerequisites, the bracketed notes say why.

```
        PHASE 1 — engineering substrate
        seeding · manifests · provenance · storage · paths · preemption
        [why first: every later number is only meaningful because a manifest pins it]
                     │
                     ▼
        TENSOR / PYTORCH MECHANICS
        view vs reshape vs transpose · contiguous · broadcasting · the shift
        [why: the project's own recurring bug class was mis-indexed shifts — E12 6d]
                     │
                     ▼
        PHASE 2a — SELF-ATTENTION
        QKᵀ/√d_k · softmax over keys · causal mask before softmax
        [E1, E2, E3]
                     │
        ┌────────────┴─────────────┐
        ▼                          ▼
  MHA / MQA / GQA            NORMALISATION + FFN
  [E4 · KV-cache size]       RMSNorm, SwiGLU [E7]
        │                          │
        └────────────┬─────────────┘
                     ▼
        POSITION — sinusoidal · RoPE · ALiBi   [E5, E6]
                     │
                     ▼
        PHASE 2b — DECODER-ONLY STACK
        residual stream · pre-norm · lm_head TIED to embedding
        [the tying is what makes Phase 4's E20 possible — learn it here]
                     │
                     ▼
        GENERATION + KV CACHE + DECODING   [E8, E9, E10]
                     │
                     ▼
        PHASE 3 — SFT
        ChatML · causal-LM loss · the logits/labels shift · loss masking
        [E12, E13, E14 — and the tokenization-boundary bug]
                     │
        ┌────────────┴─────────────┐
        ▼                          ▼
  PHASE 4 — PEFT             (perplexity as a metric,
  LoRA · SVD/PCA · QLoRA      carried forward to Phase 7)
  [E15–E20]
        │
        ▼
        PHASE 5 — PREFERENCE FOUNDATIONS
        preference data · Bradley-Terry · KL · reference model
        RLHF + PPO are CONCEPTUAL ONLY here
        [E21, E22 — SUM vs MEAN discovered]
                     │
                     ▼
        PHASE 6 — DPO
        derivation · β · preregistration · the null result
        [β sweep, E23, diag-dpo-beta0.1-lr5e-6]
                     │
                     ▼
        PHASE 7 — EVALUATION
        region-scoped perplexity · stop-token · distinct-2 ·
        Wilson intervals · length attribution · two-order LLM judge
        [eval-full-001]
                     │
                     ▼
        PHASE 8 — PROJECT DEFENCE
        provenance audit · limitations · what the project does NOT claim
```

**Three back-edges worth noticing**, because interviewers love them:

- Phase 2's `lm_head`-tied-to-embedding fact is the *entire* basis of Phase 4's
  E20 hypothesis. If you skip it in Phase 2 you cannot defend Phase 4.
- Phase 3's E12 H5 disproof ("the prompt region is *harder*") is why
  `PerplexityResult` carries a `region` field in Phase 7. Design follows finding.
- Phase 5's E22 H4 disproof ("SUM baseline is below chance") reframes *all* of
  Phase 6. Study Phase 5 before Phase 6 or the null result will look like a
  failure instead of an arithmetic consequence.

---

## 4. The study loop

Run this loop per concept, not per file. A file you have read is not a concept
you can defend.

```
 1. LEARN        Read the theory once. STUDY_WITH_CLAUDE/ or an external source.
 2. DERIVE       Close the book. Reproduce the equation on paper.
 3. DIMENSIONS   Say every tensor shape aloud, start to finish, no pauses.
 4. MINIMAL IMPL Write the 10-line version from memory. No IDE completion.
 5. REPO CODE    Open the actual file. Find the line that is your equation.
 6. TEST         Open the test that pins it. Ask: what would break if I deleted
                 this line? Which test fails?
 7. EXPERIMENT   Open the experiment script's docstring — it states objective,
                 hypothesis and config BEFORE any code. Read it before results.
 8. RESULT       Open the committed JSON in docs/phase*/ and find the number
                 yourself. Do not quote the prose around it.
 9. INTERPRET    Say what the number supports, in the project's own vocabulary.
10. LIMITATION   Say what it does NOT support. Check docs/LIMITATIONS.md.
11. EXPLAIN-BACK Write your own answer into the STUDY_WITH_CLAUDE/ Checkpoint.
12. DEFEND       Attempt the INTERVIEW_DEFENSE question CLOSED-BOOK, then read
                 the model answer to check — never to learn.
13. NEXT
```

**Rule for step 12:** if you read the model answer before attempting your own,
you have converted an active-recall exercise into passive reading and you will
overestimate your readiness. This is the highest-value rule in the file.

**Depth levels** used throughout:

- **FOUNDATIONAL** — you must be able to state it correctly and use it.
- **INTERVIEW-READY** — you must be able to derive it on a whiteboard, cold.
- **PROJECT-READY** — INTERVIEW-READY, *plus* you can point to the AlignLab
  implementation, the experiment, the number, and the limitation.
- **DEEP / OPTIONAL** — worth knowing, not worth blocking on.

**Priority levels:** **MUST STUDY** · **SHOULD STUDY** · **NICE TO HAVE**.

---

## 5. Phase 1 — engineering foundation

### 5.1 Phase purpose

Phase 1 (split 1A local / 1B server) built the substrate that makes every later
number *citable*: seeding, run manifests, provenance auditing, storage guards,
path resolution, preemption handling, and a deliberate two-environment split
(Windows CPU for development, Linux A6000 for results).

**Why it exists:** the project's claim is not "we got good numbers", it is "every
number traces to a run whose exact conditions were recorded". Without Phase 1 the
rest is anecdote.

**What depends on it:** everything. Phase 8's provenance audit (17/18 artefacts)
is Phase 1's guarantee being *checked* rather than asserted.

### 5.2 Prerequisites

- Basic Python packaging and `sys.path` / editable installs.
- What a PRNG seed is and which libraries have independent RNGs (Python `random`,
  NumPy, `torch` CPU, `torch` CUDA).
- Enough Git to read a SHA, a dirty flag, and a branch.
- What a POSIX signal is (only for `preemption.py`).

Nothing mathematical is needed here.

### 5.3 Concepts, simple → hard

| # | Concept | Understand | Derive | Implement from scratch | Depth | Priority |
|---|---|---|---|---|---|---|
| 1 | Config composition (Hydra groups) | Why config groups beat argparse for experiment matrices | — | — | FOUNDATIONAL | SHOULD |
| 2 | Seeding across libraries | Which RNGs exist and why seeding one is not enough | — | A `set_seed` covering 4 RNGs | PROJECT-READY | **MUST** |
| 3 | Reproducibility Tiers A/B/C | What each tier claims and what it refuses to claim | — | — | PROJECT-READY | **MUST** |
| 4 | Run manifests | What must be recorded for a run to be reconstructable | — | — | PROJECT-READY | **MUST** |
| 5 | Checkpoint atomicity | Why write-temp-then-`os.replace` | — | — | INTERVIEW-READY | SHOULD |
| 6 | Storage estimation | Bytes per parameter × (weights + grads + optimizer states) | **Yes** | A checkpoint-size estimator | INTERVIEW-READY | SHOULD |
| 7 | Preemption / SIGUSR1 | Why a handler sets a flag and does nothing else | — | — | INTERVIEW-READY | NICE |

### 5.4 First-principles targets

Derivable without notes:

- **Checkpoint size arithmetic.** For a full fine-tune in bf16 with an Adam-family
  optimizer: parameters + gradients + two optimizer moments, and which of those
  are kept in fp32. Then reconcile against the repository's measured figures:
  full-SFT checkpoint **2.886 GiB**, an intermediate `checkpoint-295` at **8.7 G**,
  LoRA adapter **0.027 GiB**, QLoRA adapter **0.019 GiB**
  (`docs/FINAL_STUDY_HANDOFF.md` §9, `docs/phase4/PHASE_4_REPORT.md` §6).
- **Why a signal handler must not do work.** Reason from re-entrancy, not from
  memory of the doc.

Not worth deriving: Hydra's resolution order, logging configuration.

### 5.5 Repository implementation

| Concept | Inspect |
|---|---|
| Seeding + Tier A/B/C | `src/alignlab/seeding.py` (`set_seed`, `capture_rng_state`, `restore_rng_state`); the tier definitions are in the module docstring |
| Manifests | `src/alignlab/manifest.py` (`git_info`, `package_versions`, `config_hash`, `capture_environment`, `RunManifest.write`) |
| Provenance auditing | `src/alignlab/provenance.py` (`RunProvenance`, `audit_manifest`, `_training_locators`, `_eval_locators`, `main`) |
| Storage guard | `src/alignlab/storage.py` (`estimate_checkpoint_bytes`, `estimate_lora_checkpoint_bytes`, `require_free_space_for_checkpoints`, `InsufficientStorage`) |
| Paths | `src/alignlab/paths.py` (`repo_root`, `output_root`, `checkpoint_root`, `configure_hf_cache`, `_rebind_hub_cache`) |
| Preemption | `src/alignlab/preemption.py` (`PreemptionHandler._on_signal`, `should_stop`, `_requeue`) |
| Tests | `tests/test_seeding.py`, `test_checkpoint.py`, `test_storage.py`, `test_provenance.py`, `test_preemption.py`, `test_paths_logging_device.py`, `test_no_hardcoded_paths.py` |
| Docs | `docs/phase1/PHASE_1A_REPORT.md`, `PHASE_1B_REPORT.md`, `STORAGE_POLICY.md`; `STUDY_WITH_CLAUDE/phase1_foundation.md`, `phase1b_two_environments.md`; `CODE_EXPLANATION/phase1/README.md` |

### 5.6 Code-reading strategy

Line-by-line study is warranted for exactly two things here:

1. **`paths.configure_hf_cache` and `_rebind_hub_cache`.** This is where Phase 8's
   headline engineering bug lived: the function was a **no-op in every entrypoint**
   because `huggingface_hub` freezes `HF_HUB_CACHE` at import time, costing
   **18.34 GB** of duplicate weights. Read the fix and then read
   `docs/phase8/PHASE_8_REPORT.md` on it. This is a genuinely good "tell me about
   a bug you found" story.
2. **`provenance._eval_locators`.** It shows how a "field is recorded" claim is
   turned into a machine-checkable assertion, which is the difference between a
   reproducibility *policy* and a reproducibility *guarantee*.

Skim everything else.

### 5.7 Experiments to understand

Phase 1 has no E-numbered experiments. What it has instead:

- **Tier A bitwise reproducibility across two separate Python processes** —
  VERIFIED (`docs/phase1/PHASE_1A_REPORT.md` §2). Understand what "across separate
  processes" adds over "twice in one process".
- **SIGUSR1 handling** — VERIFIED on the server (`docs/phase1/PHASE_1B_REPORT.md`).
- **Cross-machine bitwise agreement** — **NOT ACHIEVABLE / NOT CLAIMED**. Only
  Tier C applies between local CPU-fp32 and server Ampere-bf16.
- **W&B online mode, SLURM requeue, multi-GPU RNG restore** — **NOT TESTED** (no
  API key, no scheduler, no second GPU in use).
- **The deliberately broken server config** — read
  `INTERVIEW_DEFENSE/phase1_engineering.md` Q7 for why.

### 5.8 Can I defend this?

**A. Basic.** What does a run manifest record, and why each field?

**B. First principles.** Why is seeding one RNG insufficient? Enumerate the RNGs.

**C. Mathematical.** Estimate the disk cost of one full-parameter checkpoint of a
1.54B model, then reconcile with the measured 2.886 GiB and the 8.7 G intermediate.

**D. Implementation.** What does `os.replace` guarantee that `open(...).write()`
does not?

**E. Why this design.** Hydra over argparse — and when would that be wrong?
(`phase1_engineering.md` Q5.) Why single-GPU default on a 2×A6000 box? (Q8.)

**F. Experiment.** How was SIGUSR1 actually verified rather than assumed?

**G. Results.** What exactly does Tier A claim, and what does Tier B claim
*instead of* Tier A?

**H. Failure/bug.** `configure_hf_cache` — what was the bug, why did it survive so
long, and what did it cost?

**I. Skeptical follow-up.** *"Your checkpoint round-trip test passes. So resume is
correct?"* — A strong answer must name at least one thing that still differs after
a resume, explain why the test cannot see it, and avoid claiming the test proves
more than it does (`phase1_engineering.md` Q2, Q4).

> **MY EXPLANATION** (write your own; do not read the model answers first):
>
> _______________________________________________

---

## 6. Phase 2 — transformer components

### 6.1 Phase purpose

Build every architectural component from first principles, in three independent
implementations where it matters, and *measure* the claims that textbooks assert.
Phase 2 is where AlignLab earns the right to say "I understand transformers"
rather than "I have used one".

**Why it exists:** to make later phases legible. You cannot reason about LoRA
target modules without knowing what `q_proj` is; you cannot reason about the
stop-token failure without knowing that `lm_head` is tied to the embedding.

**What depends on it:** Phases 3–7, all of them.

### 6.2 Prerequisites

- **Linear algebra:** matrix multiplication as inner products; transpose;
  orthogonality; the Frobenius norm; what a rotation matrix is in 2-D.
- **Probability:** softmax as a distribution; entropy; variance of a sum of
  independent terms (this is *exactly* the √d_k argument).
- **Calculus:** the Jacobian of softmax, at least qualitatively — you need to know
  that a saturated softmax has a near-zero Jacobian.
- **PyTorch:** `view` vs `reshape` vs `transpose` vs `permute`, `.contiguous()`,
  broadcasting rules, `masked_fill`, `torch.softmax(dim=...)`, buffers vs
  parameters, `register_buffer(persistent=False)`.

Study the PyTorch list *first*: `PYTORCH_CONCEPTS/pytorch-tensor-manipulation.md`
plus `examples/tensor_reshaping_examples.py`.

### 6.3 Concepts, simple → hard

| # | Concept | Understand | Derive | Implement from scratch | Depth | Priority |
|---|---|---|---|---|---|---|
| 1 | Tensor shape tracking | Every reshape in the head split | — | Head split/merge round trip | PROJECT-READY | **MUST** |
| 2 | Softmax | Distribution over the *key* axis; numerical stability | **Yes** | 3 lines | FOUNDATIONAL | **MUST** |
| 3 | Scaled dot-product attention | Q,K,V as three separate projections | **Yes** | **Yes, cold** | PROJECT-READY | **MUST** |
| 4 | The √d_k scale | The variance argument | **Yes** | — | PROJECT-READY | **MUST** |
| 5 | Causal masking | Why mask *before* softmax; `-inf` not 0 | **Yes** | The mask itself | PROJECT-READY | **MUST** |
| 6 | Multi-head attention | Why H subspaces, and the output projection | **Yes** (param count) | **Yes** | PROJECT-READY | **MUST** |
| 7 | MQA / GQA | KV-cache size as a function of H_kv | **Yes** | `_repeat_kv` | PROJECT-READY | **MUST** |
| 8 | LayerNorm vs RMSNorm | What centring costs and what it removes | **Yes** | Both | INTERVIEW-READY | **MUST** |
| 9 | SwiGLU / gated FFN | Three matrices, not two; the width ratio | **Yes** | **Yes** | INTERVIEW-READY | SHOULD |
| 10 | Sinusoidal PE | Why sin/cos at geometric frequencies | **Yes** | **Yes** | INTERVIEW-READY | SHOULD |
| 11 | RoPE | Rotation in 2-D pairs; relative-position property | **Yes** | `_rotate_half` + apply | PROJECT-READY | **MUST** |
| 12 | ALiBi | Linear distance penalty per head | Outline only | — | DEEP / OPTIONAL | NICE |
| 13 | Decoder-only stack | Residual stream, pre-norm, weight tying | — | Block assembly | PROJECT-READY | **MUST** |
| 14 | KV cache | What is cached, what is recomputed | **Yes** (bytes) | The cache | PROJECT-READY | **MUST** |
| 15 | Decoding strategies | greedy / temperature / top-k / top-p / beam | — | top-p correctly | INTERVIEW-READY | SHOULD |
| 16 | Flash-attention concepts | Tiling, online softmax, IO-awareness | Outline | — | INTERVIEW-READY | SHOULD |
| 17 | Parameter arithmetic | Reproduce a real checkpoint's count | **Yes** | — | PROJECT-READY | **MUST** |

### 6.4 First-principles targets

Derive cold, on paper, no notes:

- **Attention.** From `X ∈ ℝ^{B×T×d_model}` to `softmax(QKᵀ/√d_k)V`, stating why
  W_Q, W_K, W_V are three separate matrices and what breaks if `W_Q = W_K`.
- **The √d_k scale.** If q and k have i.i.d. zero-mean unit-variance components,
  the dot product over d_k terms has variance d_k, so its standard deviation grows
  as √d_k. Dividing by √d_k restores unit scale. Then connect to E1's numbers.
- **Causal mask.** Why `masked_fill(-inf)` before softmax keeps rows summing to 1,
  and why masking *after* softmax does not.
- **MHA dimensions.** `[B,T,d_model] → [B,T,H,d_head] → [B,H,T,d_head]`, and the
  reverse merge. State when `.contiguous()` is required and why.
- **KV-cache bytes.** `2 · B · H_kv · T · d_head · bytes_per_element · n_layers`.
  Then reproduce E4's 100% → 6.25% for H_kv 16 → 1.
- **Sinusoidal PE.** The published formula and why an offset is a fixed linear
  rotation (this is the property `tests/test_positional.py` pins).
- **RoPE.** Rotating each (2i, 2i+1) pair by angle `m·θ_i` makes the inner product
  between positions m and n depend only on `m − n`. Show it for one pair.
- **RMSNorm vs LayerNorm.** Write both. Say precisely which statistic RMSNorm
  declines to compute.
- **SwiGLU.** `SwiGLU(x) = (SiLU(W_gate x) ⊙ W_up x) W_down`. Count its parameters
  against a 2-matrix FFN at equal width.
- **Qwen2.5-1.5B parameter count.** From `config.json` values, reproduce
  **1,543,714,304**. E11 is the story of getting this wrong.

Do **not** bother deriving: ALiBi's slope schedule, beam-search bookkeeping.

### 6.5 Repository implementation

| Concept | Files, classes, functions |
|---|---|
| Attention core | `src/alignlab/models/attention.py` — `causal_mask()`, `scaled_dot_product_attention()`, `MultiHeadAttention` (`_split_heads`, `_merge_heads`, `_repeat_kv`, `forward`) |
| Independent implementations | `models/attention_numpy.py`, `models/attention_pure.py` |
| Shape assertions | `models/shapes.py` — `assert_shape`, `describe` |
| Normalisation | `models/normalization.py` — `LayerNorm`, `RMSNorm._norm` |
| FFN | `models/feedforward.py` — `FeedForward`, `SwiGLU` |
| Position | `models/positional.py` — `SinusoidalPositionalEncoding._build`, `RotaryPositionalEmbedding._rotate_half` / `forward`, `ALiBiBias._slopes` |
| Stack | `models/transformer.py` — `TransformerConfig`, `CausalSelfAttention`, `DecoderBlock`, `DecoderOnlyTransformer`, `parameter_breakdown()` |
| KV cache | `models/kv_cache.py` — `KVCache.update`, `memory_bytes`, `predicted_bytes` |
| Generation | `models/generation.py` — `greedy_select`, `apply_temperature`, `top_k_filter`, **`top_p_filter`** (site of the E9 bug), `sample_from_logits`, `generate`, `beam_search` |
| Experiments | `scripts/experiments/e1_scaling.py`, `e2_causal_leakage.py`, `e4_attention_variants.py`, `e7_norm_comparison.py`, `e8_kv_cache.py`, `e9_decoding.py`, `e10_sdpa_backends.py`, `e11_weights_reconciliation.py`, `wp9_qwen_reconciliation.py`, `tiny_lm.py` |
| Tests | `tests/test_attention.py`, `test_attention_equivalence.py`, `test_attention_variants.py`, `test_normalization_ffn.py`, `test_positional.py`, `test_kv_cache.py`, `test_transformer.py`, `test_generation.py` |
| Results | `docs/phase2/PHASE_2_REPORT.md`, `docs/phase2/gpu_experiment_results_2026-08-29.txt` |
| Study docs | `STUDY_WITH_CLAUDE/phase2/01`–`09`; `CODE_EXPLANATION/phase2/README.md`; `INTERVIEW_DEFENSE/phase2_transformers.md` |

**Note on where E3, E5, E6 live.** There is no `e3_*.py`, `e5_*.py` or `e6_*.py`
script. Those results come from the test suites and are reported in
`docs/phase2/PHASE_2_REPORT.md` §2–§3. Verified: E5's spread is computed at
`tests/test_positional.py:205` inside
`test_rope_inner_product_depends_only_on_relative_distance`; E6's guard is
`tests/test_kv_cache.py::test_rope_offset_is_what_makes_the_cache_correct`; E3's
agreement comes from `tests/test_attention_equivalence.py`. Knowing this is a
small, real detail that shows you actually opened the repository.

### 6.6 Code-reading strategy

**The five code paths that deserve line-by-line study in Phase 2:**

1. **`scaled_dot_product_attention`** (`attention.py:126`). Five numbered steps in
   the source map one-to-one onto the equation. For each step, name the shape
   before and after. Then ask: why does `scale=1.0` exist as an option? (Because
   E1 needs to disable scaling.)
2. **`causal_mask`** (`attention.py:86`). The `offset = key_len - seq_len` line is
   the entire KV-cache correctness story. Trace it for `T_q=1, T_k=17`.
3. **`MultiHeadAttention._repeat_kv`** (`attention.py:271`). This is where GQA
   becomes MHA-shaped again. Confirm that repeating does **not** change the
   *cache* size — only the compute shape. That distinction is the whole of E4.
4. **`RotaryPositionalEmbedding.forward`** (`positional.py:222`). Find the position
   offset argument; that is what E6 guards.
5. **`top_p_filter`** (`generation.py:93`). Read it knowing the bug: the original
   used `sorted_idx.argsort()` — the inverse permutation, correct for `gather`,
   wrong for `scatter` — so surviving logits landed at wrong vocabulary positions.
   Five tests missed it because they all used already-descending logits, where the
   permutation is the identity. E9 found it. This is the best "why an experiment
   beats a unit test" story in the repository.

**Reading order per concept:** equation → shapes → repo function → the test that
would fail if you deleted the key line → the experiment → the number.

### 6.7 Experiments you must understand

**E1 — attention scaling.**
*Why:* the √d_k scale is universally quoted and rarely justified.
*Hypothesis:* without scaling, logit variance grows with d_k, softmax saturates,
gradients vanish.
*Setup:* d_k swept 4 → 1024 (256×), scaled and unscaled arms.
*Measured:* unscaled — logit std **1.85 → 31.50** (tracking √d_k), max prob
**0.42 → 0.992**, entropy **1.78 → 0.028**, Jacobian mass **0.708 → 0.014**.
Scaled — all four flat.
*Supports:* saturation is measurable and the vanishing gradient is *visible* in
the Jacobian rather than inferred.
*Does NOT prove:* anything about training dynamics at scale.
*Correction inside the experiment:* the original metric was `grad_norm`, which was
confounded — it rose then fell rather than shrinking monotonically. It was replaced
by `jacobian_mass`, **and the bad metric is kept in the script** rather than
deleted (`docs/phase2/PHASE_2_REPORT.md` §6c). That is a defensible answer to "how
do you handle a metric that misled you".
*Verdict:* **HOLDS.**

**E2 — causal-mask leakage.**
*Hypothesis:* removing the mask lets the model see future tokens, so training loss
collapses without real learning.
*Measured:* train loss **0.2907 masked vs 0.0088 unmasked — 33.2× lower when
leaking.**
*Supports:* the founding lesson of the whole project — **a dramatically better
loss can mean a broken experiment.** It is why SFT later gained three pre-flight
audits that refuse to start.
*Does NOT prove:* that all large loss drops are bugs.
*Verdict:* **HOLDS.**

**E3 — three independent implementations.**
*Measured:* torch–numpy **3.331e-16**, pure–numpy **1.110e-16**, torch–pure
**4.441e-16** (tol 1e-12); vs `F.scaled_dot_product_attention` **3.576e-07** in
fp32 (tol 1e-5).
*Supports:* the three implementations compute the *same function*.
*Does NOT prove:* that the function is the right one. Three implementations of the
same misunderstanding agree perfectly. Say this unprompted.
*Verdict:* **HOLDS.**

**E4 — MHA / MQA / GQA cost.**
*Hypothesis:* reducing KV heads shrinks the KV cache proportionally and reduces
latency.
*Measured:* H_kv 16 → 1: parameters **100% → 53.12%**, **KV cache 100% → 6.25%**,
CPU latency **160.4 → 117.4 ms**, **GPU latency flat (0.86 → 0.81 ms)**.
*Supports:* the memory saving is architectural and transfers.
*Does NOT prove:* a latency benefit. On the A6000 at this model size there was
none. Report it as measured, never generalised.
*Verdict:* **HOLDS on CPU, NOT on GPU.**

**E5 / E6 — RoPE.**
*E5 measured:* attention-logit spread **2.384e-07** across positions 0–100 at fixed
gap — invariance to absolute position.
*E6:* a guard. Forcing the cached-decoding position offset to 0 **does** change the
output, proving the offset is genuinely wired and the test could fail on broken
code. Understand *why a guard that cannot fail is worthless*.
*Verdict:* both **HOLD.**

**E7 — RMSNorm vs LayerNorm.**
*Measured:* **0.28×** the cost on CPU, **0.17–0.22×** on GPU. Centring check:
LayerNorm output mean **−0.000000**, RMSNorm output mean **+0.995010**.
*Supports:* RMSNorm is not "LayerNorm minus a redundant step" — the mean it
declines to remove is large.
*Does NOT prove:* quality equivalence.
*Correction inside the experiment:* the first version compared only against the
project's own unfused LayerNorm, which made the speedup an artefact. A fused
baseline is now permanent (`PHASE_2_REPORT.md` §6d).
*Verdict:* **HOLDS.**

**E8 — KV cache.**
*Measured:* CPU correctness **8.345e-07**; no-cache **6.16 → 15.52 ms/token**;
cached **flat ~5.4 ms**; speedup **2.96×**. GPU speedup **1.05×**.
*Supports:* the cache flattens per-token cost on CPU.
*Interpretation the project chose:* the GPU figure is **overhead-bound at this
model size**, not a refutation.
*Does NOT prove:* GPU behaviour at scale.
*Verdict:* **HOLDS on CPU; overhead-bound on GPU.**

**E9 — five decoding strategies.**
*Setup, and this matters:* run on the **tiny educational LM** (`tiny_lm.py`), seed
20260829, prompt `"the cat "`, 96 new tokens, 8 samples per stochastic setting —
**not** on Qwen. Say so before an interviewer asks.
*Measured:* greedy distinct-4 **0.257** / grammatical **0.750** / logprob
**−18.71**; top-p 0.9 → **0.566** / **0.863** / **−23.30**; temperature 2.0 →
**0.667** / **0.323** / **−80.67**.
*The real finding:* **the experiment found a genuine bug in the project's own
`top_p_filter`** (see §6.6 item 5).
*Verdict:* **MEASURED + defect found.**

**E10 — SDPA backends (the Flash-attention claim).**
*Scope, precisely:* this tests **PyTorch's `F.scaled_dot_product_attention`
backends**, not a hand-written Flash kernel. `flash-attn` is unbuildable on the
server (no nvcc, no root), recorded in Phase 2's limitations.
*Measured:* GPU, T=4096, bf16 — MATH **82.397 ms / 9568.3 MiB**, FLASH **1.478 ms /
65.0 MiB** → **56× faster, 147× less memory**. Numerical difference **1.562e-02 =
2× bf16 epsilon**. MiB per token: MATH **0.369 → 2.336** (quadratic), FLASH
**0.0159 constant** (linear).
*Supports:* the IO-aware kernel computes the same function with linear memory.
*Does NOT prove:* numerics beyond 2× bf16 epsilon.
*Verdict:* **HOLDS.**

**E11 — Qwen parameter reconciliation.**
*Hypothesis (Phase 2):* our formula reproduces the published counts.
*Measured:* predicted **1,543,656,960** vs actual **1,543,714,304** — short by
exactly **57,344 = 28 × (1536 + 256 + 256)**, the attention QKV **biases**.
*The sharpest part of the story:* Phase 2 had **already listed** "Qwen has QKV
bias" among its wrong predictions — the fact never reached the arithmetic. Error
0.0037%, invisible behind the model card's "1.54B".
*Consequence:* the wrong prediction is **preserved**, not edited, in `docs/phase2/`.
*Verdict:* **DISPROVED, then corrected.**

### 6.8 Can I defend this?

**A. Basic.** What are Q, K and V, and why three matrices instead of one?

**B. First principles.** Why divide by √d_k? A strong answer must (i) give the
intuition, (ii) give the variance argument explicitly, (iii) cite E1's four
numbers, (iv) say where the vanishing gradient became *visible* (the Jacobian mass,
0.708 → 0.014), and (v) state the limitation — E1 says nothing about training
dynamics at scale.

**C. Mathematical.** Derive the KV-cache byte formula and reproduce 100% → 6.25%.
Separately: reproduce 1,543,714,304 from `config.json`.

**D. Implementation.** Narrate every shape from X to output in MHA, including the
output projection, without pausing. Then: when is `.contiguous()` required, and why?

**E. Why this design.** Why does `scaled_dot_product_attention` take a `scale`
override at all? Why does `causal_mask` take `key_len` separately from `seq_len`?

**F. Experiment.** Why were three attention implementations written, and what does
their 1e-16 agreement establish? What does it *not* establish?

**G. Results interpretation.** GQA cut the KV cache to 6.25% but GPU latency was
flat. What is the correct thing to claim?

**H. Failure/bug.** Tell the `top_p_filter` story: what the bug was, why five unit
tests missed it, and what found it.

**I. Skeptical follow-ups.**
- *"Your GPU KV-cache speedup was 1.05×. Doesn't that mean KV caching is useless?"*
  A strong answer explains overhead-bound behaviour at 1.5B on an A6000, refuses to
  generalise in either direction, and distinguishes the memory claim from the
  latency claim.
- *"You said Flash attention is 56× faster. Did you implement Flash attention?"*
  A strong answer says **no** — PyTorch's SDPA backends were measured — and states
  why (`flash-attn` unbuildable on the server).
- *"Three implementations agreeing to 1e-16 — that proves your attention is
  correct, right?"* The answer must be a clean "no", with the reason.

> **MY EXPLANATION** (Checkpoints 1–12 live in `STUDY_WITH_CLAUDE/phase2/`):
>
> _______________________________________________

---

## 7. Phase 3 — supervised fine-tuning

### 7.1 Phase purpose

Take the **base** Qwen2.5-1.5B (not Instruct) and instruction-tune it on
`HuggingFaceH4/no_robots` with a **verified** completion-only loss mask, then
measure what changed.

**Why it exists:** SFT is where an alignment pipeline starts, and the loss mask is
the one place where a silent bug produces plausible-looking numbers. Phase 3's real
deliverable is not the model — it is the *verification* that the loss is computed
on the tokens the project claims.

**What depends on it:** Phase 4 (LoRA/QLoRA compare against this run and reuse its
evaluation code path), Phase 5 (the SFT model is DPO's reference), Phase 7
(perplexity regions come from E12's disproof).

### 7.2 Prerequisites

- Cross-entropy for a categorical distribution; perplexity as `exp(mean NLL)`.
- The **causal-LM shift**: logits at position t predict token t+1, so labels are
  shifted left by one. Know that `contributing positions = (labels[:,1:] != -100)`.
- Tokenization: BPE merges, and why a token boundary is not a character boundary.
- ChatML: `<|im_start|>` / `<|im_end|>` turn structure.
- PyTorch: `F.cross_entropy(logits, targets, ignore_index=-100)` takes **logits**,
  and which axis it reduces over.

Read `PYTORCH_CONCEPTS/pytorch-loss-masking-and-shifts.md` and run
`examples/loss_masking_examples.py` before opening `masking.py`.

### 7.3 Concepts, simple → hard

| # | Concept | Understand | Derive | Implement from scratch | Depth | Priority |
|---|---|---|---|---|---|---|
| 1 | Instruction tuning | What SFT changes vs pretraining | — | — | FOUNDATIONAL | **MUST** |
| 2 | ChatML templating | Prompt/completion boundary as *tokens* | — | — | PROJECT-READY | **MUST** |
| 3 | Causal-LM loss + shift | Which position predicts which token | **Yes** | **Yes** | PROJECT-READY | **MUST** |
| 4 | `IGNORE_INDEX = -100` | Why −100 specifically | — | — | FOUNDATIONAL | **MUST** |
| 5 | Completion-only masking | Which tokens contribute, and why | **Yes** | **Yes** | PROJECT-READY | **MUST** |
| 6 | Token-region perplexity | Why prompt and completion perplexity are not comparable | **Yes** | — | PROJECT-READY | **MUST** |
| 7 | Truncation | What truncation does to the mask | — | — | INTERVIEW-READY | SHOULD |
| 8 | Packing (and why it is OFF) | Throughput vs mask verifiability | — | — | INTERVIEW-READY | SHOULD |
| 9 | Dataset fingerprinting | Why a content hash beats a version string | — | — | INTERVIEW-READY | SHOULD |

### 7.4 First-principles targets

- **Work the shift by hand.** Take an 8-token sequence with a 3-token prompt.
  Write `input_ids`, `labels`, the shifted pair, and count contributing positions.
  Then explain E12's bug 6d: the decomposition used `(labels != -100).sum()` when
  the correct count is `(labels[:,1:] != -100).sum()` — the shift drops position 0.
  With 25 rather than 26 prompt positions, the decomposition agreed to **1.6e-07**.
- **Perplexity from NLL.** `ppl = exp(mean NLL)`, token-weighted not
  example-weighted. Derive why token-weighting is the right choice and what changes
  if you average per example instead.
- **Why an unmasked loss is not comparable to a masked one in *either* direction.**
  It is a mean over a **different token population**. E12 H5 is the proof.

### 7.5 Repository implementation

| Concept | Inspect |
|---|---|
| Mask construction | `src/alignlab/masking.py` — `IGNORE_INDEX` (line 49), `render_prompt`, `render_full`, `check_prefix_consistency`, `expected_labels`, `describe_mask`, `compare_masks`, `MaskReport` |
| Training entrypoint | `src/alignlab/sft.py` — `resolve_dtype`, **`audit_prefix_consistency`**, **`verify_mask_on_real_batch`**, **`audit_truncation`**, `run_sft` |
| Data | `src/alignlab/data.py` — `to_prompt_completion`, `_strip_boundary_whitespace` (the boundary-bug fix), `is_wellformed`, `fingerprint_rows`, `DatasetFingerprint` |
| Configs | `configs/sft/default.yaml` (read the comments — `packing: false` and `completion_only_loss: null` both carry their rationale), `configs/data/no_robots.yaml`, `configs/model/qwen2_5_1_5b.yaml` |
| Experiments | `scripts/experiments/e12_loss_masking.py`, `e13_sft_before_after.py`, `e14_region_decomposition.py` |
| Tests | `tests/test_data_and_sft_config.py`, `tests/test_smoke_train.py` |
| Artefacts | `docs/phase3/e13_before_after.json`, `e14_region_decomposition.json`, `sft_run_manifest.json`, `sft_run_summary.json` |
| Docs | `docs/phase3/PHASE_3_REPORT.md`; `STUDY_WITH_CLAUDE/phase3/01_sft_and_loss_masking.md`; `CODE_EXPLANATION/phase3/README.md`; `INTERVIEW_DEFENSE/phase3_sft.md` |

### 7.6 Code-reading strategy

**Line-by-line, in this order:**

1. **`masking.expected_labels`** — the independent re-derivation of the label
   vector. This is what makes E12's H2 ("TRL's boundary agrees with an independent
   computation") testable at all. Note that it builds labels from *token counts*,
   not from string offsets.
2. **`masking.check_prefix_consistency`** — the function that caught the
   tokenization-boundary bug. Read it asking: *what assumption is this refusing to
   trust?*
3. **`sft.audit_prefix_consistency` / `verify_mask_on_real_batch` /
   `audit_truncation`** — three pre-flight audits that **refuse to start training**.
   Understand why these exist *given* that E12 already passed
   (`INTERVIEW_DEFENSE/phase3_sft.md` Q9 is exactly this challenge).
4. **`data._strip_boundary_whitespace`** — five lines that fix a real bug. Read the
   docstring for why stripping is a fix and not a workaround.

**A useful exercise:** delete (mentally) the shift in the decomposition and predict
which number moves and by how much. Then check against E12's 1.6e-07.

### 7.7 Experiments you must understand

**E12 — loss masking on the real model.**
*Hypotheses as written:* H1 masked and unmasked losses differ · H2 the TRL boundary
agrees with an independent computation · H3 no example is fully masked · H4 the
decomposition reproduces the reported loss · **H5 the prompt region is *easier*
than the completion region.**
*Measured:* H1–H4 **HOLD**, the decomposition agreeing to **1.6e-07** *after* the
counting fix. **H5 DISPROVED** — prompt CE **6.3314** vs completion **3.6295**.
*Why H5 failed:* the base model had **never seen ChatML**. The template tokens are
the hardest thing in the sequence.
*Supports:* an unmasked loss is not reliably higher *or* lower than a masked one —
it is a mean over a different token population.
*Consequence in code:* `PerplexityResult.region` exists in Phase 7 because of this.
*Does NOT prove:* which region is "harder" in general.
*Defects found:* the `labels[:,1:]` mis-count (6d), and a stale `datasets` cache
(6c) — see below.
*Verdict:* **H1–H4 HOLD · H5 DISPROVED.**

**The two Phase 3 bugs you should be able to tell as a story:**

- **The tokenization-boundary bug (6b).** The ChatML prompt ends with token `198`
  (a lone newline). A completion that *starts* with newlines lets BPE merge them
  into token `1406` (`'\n\n\n'`), so the prompt/completion boundary lands **mid
  token** — half prompt, half answer. **3 of 200 rows, 1.5%.** Found by
  `check_prefix_consistency`, a check that exists *because the assumption was
  chosen not to be trusted*. TRL warns and proceeds anyway.
- **The second-order bug (6c).** The fix appeared to do nothing: `datasets.map()`
  silently served a **stale cache**, returning the same three indices. What proved
  the fix had finally landed was the content fingerprint changing,
  `b7dff71c…` → `dc6fd746…`. `load_from_cache_file=False` now forces recomputation.
  This is a better bug story than the first one, because the failure mode was *"my
  correct fix looked ineffective"*.

**E13 — SFT before/after, 200 held-out examples.**
*Measured:* completion perplexity **8.541 → 7.199 (−15.7%)** over **38,831**
completion tokens; stop-token emission **0/4 → 4/4**.
*Verdict:* **MEASURED.**
*Note:* Phase 7 re-measured on a different subset and got **8.824 → 7.398 (−16.2%)**
and **0/6 → 6/6**. Both are real; they are different evaluation subsets. Do not mix
the pairs.

**E14 — control-region decomposition, 100 examples.**
*Measured* (`docs/phase3/e14_region_decomposition.json`), NLL by region, base → SFT:

| region | base NLL | SFT NLL | absolute drop | tokens |
|---|---:|---:|---:|---:|
| template | 8.8727 | 8.1303 | **0.7425** | 1,521 |
| user text | 2.6402 | 2.5207 | 0.1195 | 11,193 |
| completion | 2.1243 | 1.9729 | 0.1514 | 21,001 |

The template region — 1,521 tokens, the ChatML scaffolding — improved by far the
most in absolute terms, and accounts for **45.8%** of the prompt-region gain
(`template_share_of_prompt_gain`).
*What this supports:* it is consistent with E12 H5's explanation — the model was
learning the *format* it had never seen. It also shows why denominators matter: a
region with 1,521 tokens and a region with 21,001 tokens contribute very
differently to any pooled average.
*What it does NOT prove:* that format learning is *all* SFT did — the completion
region improved too.
*Verdict:* **MEASURED.**

**The training run itself:** `sft-qwen1p5b-noRobots-001`, one epoch of `no_robots`
(9,499 training rows), batch 4×8 = 32, 296 optimiser steps, lr 2e-5 cosine, seed
42, `max_length=1024`, packing off, bf16 on one A6000. Peak VRAM **14.42 GiB**
(measured later by a 12-step probe at identical batch geometry, because Phase 3's
run predates the instrumentation — see `docs/phase4/fullsft_vram_probe.log`).

### 7.8 Can I defend this?

**A. Basic.** Which tokens contribute to the SFT loss, and which do not?

**B. First principles.** Why is a masked loss not comparable to an unmasked loss —
in *either* direction?

**C. Mathematical.** Work the shift on a short sequence; count contributing
positions; explain the off-by-one that E12 had to fix.

**D. Implementation.** Where is the mask built? What does
`check_prefix_consistency` compare, and against what?

**E. Why this design.** Why is packing off? Why do three pre-flight audits exist
when E12 already passed? (`phase3_sft.md` Q8, Q9.)

**F. Experiment.** State E12's five hypotheses as written, and which failed.

**G. Results interpretation.** *"A colleague's SFT loss is 0.8 and yours is 2.1,
same model, same data. Who is winning?"* (`phase3_sft.md` Q3.) A strong answer
must get to *token population* and *masking convention* before it says anything
about quality.

**H. Failure/bug.** Tell the boundary bug and the stale-cache bug, and say what
proved the fix landed.

**I. Skeptical follow-ups.**
- *"Perplexity dropped 15.7%. Your model is better, then?"* A strong answer names
  the in-distribution problem (`no_robots` test split is the training
  distribution), and pivots to the stop-token result as the stronger evidence.
- *"Only 3 rows out of 200 had the boundary bug. Why does 1.5% matter?"* A strong
  answer separates *effect size* from *class of failure*, and notes that the check
  was cheap while the failure was silent.

> **MY EXPLANATION** (Checkpoints 13–15, `STUDY_WITH_CLAUDE/phase3/`):
>
> _______________________________________________

---

## 8. Phase 4 — PEFT: LoRA, SVD, QLoRA

### 8.1 Phase purpose

Implement LoRA from first principles, verify it against `peft`, quantify what PEFT
costs and saves against Phase 3's full fine-tune, and interrogate the *standard
explanation* of why LoRA works by measuring the real update.

**Why it exists:** this is the project's highest-value phase for interviews. It
contains the parameter arithmetic, the SVD/PCA material, the quantization material,
and the project's most important negative result.

**What depends on it:** Phase 7 evaluates the LoRA and QLoRA checkpoints; the
stop-token failure discovered here reappears there.

### 8.2 Prerequisites

- **Linear algebra:** matrix rank; the SVD `A = UΣVᵀ`; the Eckart–Young theorem
  (best rank-k approximation in Frobenius/spectral norm); Frobenius norm as
  `√Σσᵢ²`; eigendecomposition of a covariance matrix.
- **Optimisation:** constrained maximisation with a Lagrange multiplier — you need
  this to derive PCA properly.
- **Numerics:** IEEE floating point — sign/exponent/mantissa; what fp32, fp16 and
  **bf16** each spend their bits on; machine epsilon.
- **PyTorch:** `nn.Linear` weight shape convention (`[out_features, in_features]`),
  `requires_grad_`, `torch.no_grad()`, in-place `.data` updates.

Read `PYTORCH_CONCEPTS/pytorch-quantization-and-lowrank.md` and run
`examples/svd_pca_examples.py`.

### 8.3 Concepts, simple → hard

| # | Concept | Understand | Derive | Implement from scratch | Depth | Priority |
|---|---|---|---|---|---|---|
| 1 | LoRA equation | `W'x = Wx + (α/r)·BAx` | **Yes** | **Yes, cold** | PROJECT-READY | **MUST** |
| 2 | LoRA initialisation | Why `B=0`, `A` random; both gradients | **Yes** | — | PROJECT-READY | **MUST** |
| 3 | LoRA parameter count | `r·(d_in + d_out)` per matrix | **Yes** | Reproduce **4,358,144** | PROJECT-READY | **MUST** |
| 4 | `α/r` scaling | Why α = 2r makes a rank sweep not also an LR sweep | **Yes** | — | INTERVIEW-READY | **MUST** |
| 5 | Target-module selection | Why `q,k,v,o` and why **not** `lm_head` | — | — | PROJECT-READY | **MUST** |
| 6 | Merging / unmerging | What merging buys, what it costs | — | **Yes** | PROJECT-READY | **MUST** |
| 7 | SVD | `A = UΣVᵀ`; energy uses **squared** singular values | **Yes** | Rank-k truncation | PROJECT-READY | **MUST** |
| 8 | Break-even rank | `r* = (d_in·d_out)/(d_in + d_out)` | **Yes** | — | INTERVIEW-READY | **MUST** |
| 9 | PCA, and PCA ↔ SVD | Variance maximisation → Lagrangian → eigenproblem | **Yes** | — | PROJECT-READY | **MUST** |
| 10 | Float formats | fp32 / fp16 / bf16 — exponent vs mantissa | **Yes** | — | PROJECT-READY | **MUST** |
| 11 | Quantization basics | Scale, zero-point, symmetric vs asymmetric | **Yes** | Round-trip quantizer | INTERVIEW-READY | **MUST** (external theory) |
| 12 | NF4 and block-wise quant | NF4 is **storage**, not arithmetic | — | — | PROJECT-READY | **MUST** |
| 13 | Double quantization | Quantizing the block constants | — | — | INTERVIEW-READY | SHOULD |
| 14 | QLoRA memory model | Where the saving actually comes from | **Yes** | — | PROJECT-READY | **MUST** |

### 8.4 First-principles targets

- **The LoRA parameter count, module by module.** Target set is
  `q_proj, k_proj, v_proj, o_proj`, **112 matrices** across 28 layers. Shapes:
  `q_proj` and `o_proj` are 1536→1536; `k_proj` and `v_proj` are 1536→**256**
  (GQA). At r=16, adapter params per matrix = `r·(d_in + d_out)`. Compute:
  - `q_proj`: 16·(1536+1536) = 49,152 · 28 layers
  - `o_proj`: 16·(1536+1536) = 49,152 · 28
  - `k_proj`: 16·(1536+256) = 28,672 · 28
  - `v_proj`: 16·(1536+256) = 28,672 · 28
  Total must come to **4,358,144**. Do this on paper until it is automatic — it is
  the single most likely live-arithmetic question in the whole project.
- **The two percentages.** `4,358,144 / 1,543,714,304 = 0.2823%`
  (`docs/phase4/PHASE_4_REPORT.md` §4) and
  `4,358,144 / (1,543,714,304 + 4,358,144) = 0.2815%` (E19's table, which uses the
  logical total *including* the adapters). Both appear in the repository. Being
  able to explain *why they differ* is a small, very convincing detail.
- **Both gradients at initialisation.** With `B = 0`: `∂L/∂A ∝ Bᵀ(...) = 0`, while
  `∂L/∂B ∝ (...)Aᵀ ≠ 0`. So B moves first and A follows. Then say what goes wrong
  with **both zero** (a dead saddle — the repository has a test named exactly
  `test_both_zero_is_a_dead_saddle_point`) and with **both random** (the model no
  longer starts at the pretrained function).
- **Break-even rank.** LoRA costs `r(d_in + d_out)`; dense costs `d_in·d_out`.
  Setting them equal gives `r* = (d_in·d_out)/(d_in + d_out)`. For 1536×1536,
  `r* = 768`. Now say what that implies for a "use SVD to pick the rank" recipe.
- **PCA from scratch.** Maximise `wᵀΣw` subject to `wᵀw = 1`; form the Lagrangian
  `wᵀΣw − λ(wᵀw − 1)`; differentiate; get `Σw = λw`. Then connect: for centred data
  `X`, the covariance eigenvalues equal `σ²/(n−1)` from `X`'s SVD, and the principal
  directions are the right singular vectors.
- **Why ΔW's SVD is not PCA.** ΔW is a matrix *difference*, not a data cloud, so
  there is no centring step and calling its SVD "PCA" is a category error. The
  repository verified this numerically: centring changes the first/second
  singular-value ratio from **45.7 → 1.5**.
- **Explained energy.** Fraction of energy at rank k is `Σ_{i≤k} σᵢ² / Σ_i σᵢ²` —
  **squared**. This is E17's H1 and it HOLDS.

### 8.5 Repository implementation

| Concept | Inspect |
|---|---|
| LoRA | `src/alignlab/lora.py` — `LoRAConfig.scaling`, `LoRALinear.__init__` (freezing at construction), `reset_parameters`, `delta_weight`, `forward`, `merge`, `unmerge`, `_iter_target_linears`, `apply_lora`, `lora_state_dict`, `count_parameters` |
| Library path | `src/alignlab/peft_setup.py` — `build_quantization_config`, `build_peft_config`, `describe_peft`, `summarise_trainable` |
| Configs | `configs/peft/lora.yaml`, `qlora.yaml`, `none.yaml` — **read every comment**; they carry the reasoning (r=16 is "a STARTING POINT, not a value derived from the SVD analysis"; NF4 chosen over FP4 on measured relative error 0.0912 vs 0.1210) |
| Experiments | `scripts/experiments/e15_bitsandbytes_environment.py`, `e16_lora_targets.py`, `e17_svd_rank_selection.py`, `e18_educational_vs_peft.py`, `e19_peft_comparison.py`, `e20_stop_token_gap.py` |
| Tests | `tests/test_lora.py` — read the class names: `TestInitialisation`, `TestFreezing`, `TestGradientFlow`, `TestParameterCounts` |
| Artefacts | `docs/phase4/e15_*.json`, `e16_lora_targets.json`, `e17_svd_rank_selection.json`, `e18_educational_vs_peft.json`, `e19_peft_comparison.json`, `e20_stop_token_gap.json`, `fullsft_vram_probe.log`, three `*_summary.json` run summaries |
| Docs | `docs/phase4/PHASE_4_REPORT.md`; `STUDY_WITH_CLAUDE/phase4/01_lora_svd_and_pca.md`; `LEARNING_RESOURCES/phase4_resources.md`; `INTERVIEW_DEFENSE/phase4_peft.md`; `HOW_TO_RUN.md` §9.3, §9.4 |

### 8.6 Code-reading strategy

**Line-by-line, in this order:**

1. **`LoRALinear.__init__`** — note that the base is *frozen at construction*, not
   left to the caller. Note `bias=False` on both adapter layers and the stated
   reason: a bias would add a term outside `BAx`, so the module would stop
   implementing the equation it claims to.
2. **`LoRALinear.forward` and `delta_weight`** — confirm `scaling = α/r` appears in
   both, consistently, and that `forward` short-circuits when merged (merging twice
   would double the update).
3. **`merge` / `unmerge`** — read the docstring's honesty: unmerge is exact only up
   to floating-point round-off, and the tests assert a *tolerance*, not equality.
   This is the hook for the bf16 merge finding.
4. **`_iter_target_linears` and `apply_lora`** — how the target set is matched
   against qualified module names, and where everything else gets frozen.
5. **`e17_svd_rank_selection.py`** — read the docstring for the hypotheses, then
   find where the **two controls** are computed (same-norm Gaussian, and the
   pretrained weight itself). The controls are the methodological point.

### 8.7 Experiments you must understand

**E15 — does bitsandbytes actually work here?**
*Why:* an import succeeding proves nothing about a CUDA kernel running.
*Measured:* verified **by executing a quantized matmul**, not by importing. A
`Linear4bit` forward agreed with a bf16 `nn.Linear` to **9.1% relative error** —
which is quantization error in the *weights*, not reduced-precision arithmetic.
NF4 measured at **24.6% lower relative error than FP4** on normally-distributed
data (0.0912 vs 0.1210). Related finding: `triton` installed but unimportable
(missing `setuptools`).
*Verdict:* **HOLDS.**

**E16 — which modules does LoRA target?**
*Approach:* enumerated the loaded module tree instead of assuming the conventional
set applies.
*Measured:* **197 `nn.Linear` modules** total. `lm_head` 1536→151936; `gate_proj`
and `up_proj` 1536→8960 (28 each); `down_proj` 8960→1536 (28); `q_proj`/`o_proj`
1536→1536 (28 each, q has bias); `k_proj`/`v_proj` 1536→**256** (28 each, both with
bias). Target set `q,k,v,o` = **112 matrices**. At r=16: **4,358,144** trainable =
**0.2823%**, adapter 16.6 MiB. Predicted count == actual count; the file on disk is
17,462,432 B = 17,432,576 predicted + safetensors header.
*Key decision:* `lm_head` is **excluded from every target set** because it is
**tied to the input embedding** — adapting it would adapt the embedding too.
*This decision is what makes E20 possible.*
*Verdict:* **MEASURED.**

**E17 — SVD rank selection. The project's most important negative result.**
*Hypotheses:* H1 explained energy uses **squared** singular values ·
**H2 the real full-fine-tuning ΔW is approximately low-rank, which is why LoRA
works.**
*Setup:* ΔW = `W_sft − W_base` from Phase 3's full fine-tune, on **10 sampled
modules** (layers 0, 13, 27; attention projections plus two MLP matrices), with
**two controls**: a same-norm Gaussian, and the pretrained weight itself.
*Measured*, layer 13 `q_proj` (1536×1536):

| rank | energy | noise control | recon. error | LoRA params |
|---:|---:|---:|---:|---:|
| 16 | **10.69%** | 3.95% | 0.9450 | 49,152 |
| 64 | 22.95% | 14.57% | 0.8778 | 196,608 |
| 256 | 54.53% | 46.92% | 0.6743 | 786,432 |

Rank for **90% energy ≈ 730 of 1536**; the noise control needs **783** — real
structure, but only a **~7% margin**, not an order of magnitude.
Also measured (H4): `‖ΔW‖/‖W‖` = **0.0013–0.0050**. One epoch moved the weights by
about a quarter of one percent.
*Concluded:* **H2 DISPROVED.** The measured update is not low-rank.
*The arithmetic that finishes the argument:* break-even rank for 1536×1536 is
**768**. So a naive "pick the rank from the SVD" recipe would recommend r≈730,
costing 2,242,560 parameters against a dense 2,359,296 — a **4.9% saving**. For the
narrow GQA projections (1536→256), rank 256 already costs *more* than dense.
*What it does NOT prove:* that LoRA is ineffective. **LoRA needs *some* low-rank
update to suffice, not full fine-tuning's update to *be* low-rank.** Unconstrained
gradient descent has no incentive to produce a low-rank update, so rank 730
measures how unconstrained the optimiser was — not what LoRA requires.
*Verdict:* **H1 HOLDS · H2 DISPROVED.**

**E18 — our LoRA against `peft`.**
*Measured:* agreement **in float32**; merged vs adapter max difference **2.682e-07**.
*What did not transfer:* the same agreement in **bf16** — see E19.
*Verdict:* **HOLDS (fp32).**

**The bf16 merge finding (Phase 4 §7c).**
*Measured:* merged-vs-unmerged logits differ by **6.875e-01** max (mean 6.103e-02)
in bf16, versus **6.330e-05** in fp32 — **12,460× worse**.
*Mechanism:* `‖ΔW‖/‖W‖ ≈ 0.003` sits at the resolution of bf16's mantissa, so
`W + ΔW` rounds back toward `W`.
*Consequence:* E19 was changed to evaluate the **unmerged** adapter, so a merge
artefact is not attributed to a training arm. This is a first-class "we changed the
protocol after finding a defect" story.

**E19 — full SFT vs LoRA vs QLoRA.**
*Controlled:* model + revision, dataset (identical fingerprint `17bebb1758a66958`
across all arms — verified), objective, `max_length=1024`, packing off, batch
4×8=32, 1 epoch / 296 steps, seed 42, evaluation set and code path, hardware.
*Not controllable:* learning rate — LoRA conventionally needs a higher one because
adapters start at zero. **Both were run** rather than choosing.

| arm | lr | trainable | peak VRAM | time | eval loss | checkpoint |
|---|---|---:|---:|---:|---:|---:|
| full SFT | 2e-5 | 1,543,714,304 | **14.42 GiB** | 1168 s | **1.9893** | 2.886 GiB |
| LoRA r=16 | 2e-4 | 4,358,144 | 4.72 GiB | 1055 s | 2.0402 | 0.027 GiB |
| LoRA r=16 | 2e-5 | 4,358,144 | 4.72 GiB | 1053 s | 2.0718 | 0.027 GiB |
| QLoRA r=16 | 2e-4 | 4,358,144 | **2.88 GiB** | 1044 s | 2.0769 | 0.019 GiB |

Memory: full SFT is **3.05×** LoRA and **5.01×** QLoRA. Checkpoints **107×**
smaller.
*Evaluation* (Phase 3's code path, unchanged): base 8.541 (0/4 stop-token) · full
SFT **7.199 (4/4)** · LoRA@2e-4 7.449 (0/4) · LoRA@2e-5 7.624 (0/4) · QLoRA@2e-4
7.511 (0/4).
*The finding none of the hypotheses anticipated:* **perplexity separates the arms
by ~3.5%; stop-token behaviour separates them completely.** A single metric would
have concluded the methods are interchangeable here.
*The LR confound, quantified:* the LR effect is **0.032 nats**, about 60% of the
LoRA-vs-full-FT gap of **0.051 nats**. Reporting only the LR-matched arm would have
**overstated LoRA's cost by roughly half**.
*On timing:* QLoRA ran 1044 s vs LoRA's 1055 s — 1% *faster*. **The honest reading
is not "QLoRA is faster."** One run per arm, single seed, no error bar;
dequantization overhead is simply not detectable at this scale. (The 20-step smoke
runs *did* show QLoRA 16% slower, where the one-off load-time quantization cost
dominates.)
*Verdict:* see §14 — the hypothesis **numbering** for E19 differs between the
experiment script and the registry. Learn the script's numbering; it is primary.

**E20 — why do the PEFT models never stop?**
*Candidate explanations, before measuring:* H1 LoRA learned **nothing** about the
stop token · H2 LoRA learned **something but not enough** (probability rose but
stayed below argmax) · **H3 the structural explanation** — emitting a rare token
requires moving that token's **logit**, produced by `lm_head`, which is tied to the
embedding and therefore outside LoRA's target set.
*Measured*, at the position where `<|im_end|>` (id 151645) should be emitted, over
100 held-out examples:

| model | mean P(`<\|im_end\|>`) | median rank | argmax |
|---|---:|---:|---:|
| base | 0.0000028 | 31,600 | 0.0% |
| **full SFT** | **0.39513** | **1** | **79.0%** |
| LoRA @2e-4 | 0.00021 | 160 | 0.0% |
| LoRA @2e-5 | 0.000056 | 2,217 | 0.0% |
| QLoRA @2e-4 | 0.00019 | 194 | 0.0% |

`lm_head` movement under full fine-tuning: Frobenius base **399.894**, delta
**5.440**, relative **0.013604** — the largest relative change of any matrix
measured in the project.
*Outcome:* **H1 false, H2 true.** LoRA learned *something* (rank 31,600 → 160) and
still never wins the argmax. H3 is **structural rather than directly measurable
here**; what the script *can* do is bound how much of the effect was unavailable to
LoRA.
*Verdict:* **NOT CONFIRMED.** Say **"consistent with"**, never "because".
*The decisive test:* a LoRA variant adapting the MLP, or an untied output head. One
~18-minute run. **DEFERRED, not skipped.** The registry calls it "the single most
valuable outstanding experiment in the project".
*This is the best "what would you do next" answer you have. Have it ready.*

### 8.8 Can I defend this?

**A. Basic.** What is LoRA and what problem does it solve?

**B. First principles.** Why does a low-rank factorisation reduce parameters, and
by how much? At what rank does it stop saving?

**C. Mathematical.** (i) Reproduce 4,358,144 module by module, live. (ii) Derive
PCA from variance maximisation. (iii) State the PCA ↔ SVD relationship. (iv) Show
why explained energy uses **squared** singular values.

**D. Implementation.** Why is `B` zero at init? Write both gradients. What breaks
with both-zero and with both-random? Why does `LoRALinear` freeze the base in
`__init__` rather than leaving it to the caller?

**E. Why this design.** Why `α = 2r`? Why exclude `lm_head` from the target set?
Why evaluate the **unmerged** adapter in E19?

**F. Experiment.** Why did E17 need a **same-norm random control**? What would the
result have looked like without it?

**G. Results interpretation.** *"90% of ΔW's energy needs rank ~730, and you
trained at r=16 which captures 10.69%. So your adapters were far too small?"* A
strong answer separates "what full fine-tuning did" from "what LoRA needs", cites
the break-even rank of 768, and notes that r=16 still reached within 3.6%
perplexity of full SFT.

**H. Failure/bug.** The bf16 merge: what is the number, what is the mechanism, and
what did you change because of it?

**I. Skeptical follow-ups.**
- *"Your SVD result seems to undermine LoRA. Defend it."* (`phase4_peft.md` Q8.)
- *"LoRA claims 10,000× fewer trainable parameters. Do you see that?"* (Q6.) The
  answer here is **354×**, and the reason is a scale effect, not a contradiction.
- *"You say `lm_head` being frozen explains the stop-token failure. Prove it."*
  The only correct answer is that you **cannot** — E20 is NOT CONFIRMED — followed
  immediately by the exact experiment that would settle it and its cost.
- *"QLoRA was faster in your table. So use QLoRA always?"* Reject the premise: 1%,
  one seed, no error bar; QLoRA's claim is **memory**.

> **MY EXPLANATION** (Checkpoint 16, `STUDY_WITH_CLAUDE/phase4/`):
>
> _______________________________________________

---

## 9. Phase 5 — preference learning foundations

### 9.1 Phase purpose

Prepare everything DPO needs — preference data, sequence log-probabilities, KL
machinery, a frozen reference model — and **measure the baseline before optimising
it**. Phase 5 also studies RLHF and PPO as *reasoning*, without implementing them.

**Why it exists:** because measuring where a metric *starts* is what turned Phase
6's null from a mystery into arithmetic.

**What depends on it:** all of Phase 6, and the preference metrics in Phase 7.

### 9.2 Scope boundary — say this before you are asked

**RLHF with a learned reward model, and PPO, are CONCEPTUAL ONLY in AlignLab —
NOT IMPLEMENTED.** No reward model was trained; there is no PPO loop, no value
network, no rollout machinery in `src/`.

What **is** implemented and tested:

- `preference.bradley_terry_probability` and `preference.bradley_terry_loss` — the
  Bradley-Terry math as standalone, unit-tested functions.
- `logprobs.token_logprobs`, `sequence_logprobs`, `logprob_ratio`, `token_kl`,
  `sequence_kl`, `approximate_kl` (the **k3** estimator).
- `dpo.verify_reference_is_frozen`, `dpo.verify_zero_reward_at_init`.
- `preference.audit_preferences`, `load_preference_dataset`.

So the honest sentence is: *"Bradley-Terry is implemented as a loss function and
tested; no reward model was trained, and PPO was studied but not built."*

### 9.3 Prerequisites

- **Probability:** log-probabilities; why summing log-probs over tokens gives a
  sequence log-probability; the logistic function.
- **Information theory:** KL divergence, its asymmetry, its non-negativity, and
  what a Monte-Carlo estimator of it looks like.
- **RL vocabulary (for the conceptual part):** policy, reward, advantage, value
  function, on-policy vs off-policy, importance ratio, clipped surrogate.
- **PyTorch:** `log_softmax` then `gather` (never `log(softmax(...))`),
  `model.eval()` and `torch.no_grad()`, `requires_grad_(False)`.

Read `PYTORCH_CONCEPTS/pytorch-logprobs-and-kl.md`; run
`examples/logprob_and_kl_examples.py`.

### 9.4 Concepts, simple → hard

| # | Concept | Understand | Derive | Implement from scratch | Depth | Priority |
|---|---|---|---|---|---|---|
| 1 | Preference data shape | prompt / chosen / rejected; ties | — | — | FOUNDATIONAL | **MUST** |
| 2 | Sequence log-probability | SUM vs MEAN over tokens | **Yes** | **Yes** | PROJECT-READY | **MUST** |
| 3 | Bradley-Terry | `P(a≻b) = σ(r_a − r_b)` | **Yes** | **Yes** | PROJECT-READY | **MUST** |
| 4 | Reward-model loss | BT → `−log σ(r_w − r_l)` | **Yes** | — | INTERVIEW-READY | **MUST** |
| 5 | RLHF pipeline | Every model in it, and what each costs | — | — | INTERVIEW-READY (CONCEPTUAL ONLY) | **MUST** |
| 6 | PPO | Clipped surrogate; why a value net; the rollout loop | Outline | — | INTERVIEW-READY (CONCEPTUAL ONLY) | **MUST** |
| 7 | KL regularisation | What it prevents; which direction | **Yes** | Exact KL | PROJECT-READY | **MUST** |
| 8 | KL estimators (k1, k3) | Variance vs bias; float32 reality | **Yes** | k3 | PROJECT-READY | SHOULD |
| 9 | Reference model | Why frozen, why `eval()` | — | — | PROJECT-READY | **MUST** |
| 10 | Length bias in preference data | Why SUM is length-sensitive | **Yes** | — | PROJECT-READY | **MUST** |

### 9.5 First-principles targets

- **Bradley-Terry.** From "each item has a latent score" to
  `P(a ≻ b) = exp(r_a)/(exp(r_a)+exp(r_b)) = σ(r_a − r_b)`, then to the maximum-
  likelihood loss `−log σ(r_w − r_l)`. State the assumptions (transitivity,
  independence of annotations, a single scalar quality axis).
- **KL, both directions.** `KL(p‖q) = Σ p log(p/q)`. Say which one RLHF penalises
  and what the *other* direction would encourage.
- **Why summed log-probability is length-sensitive.** Each token contributes a
  negative number, so longer sequences have lower sums *mechanically*. This is the
  seed of the whole SUM/MEAN story.
- **The k3 estimator.** Understand why it is mathematically non-negative, and then
  why float32 cancellation produced negatives anyway.

### 9.6 Repository implementation

| Concept | Inspect |
|---|---|
| Log-probs | `src/alignlab/logprobs.py` — `token_logprobs`, `sequence_logprobs` (note `SequenceScores` carries `sum_logprob`, `mean_logprob`, `n_tokens`), `logprob_ratio`, `token_kl`, `sequence_kl`, `approximate_kl` |
| Preference data | `src/alignlab/preference.py` — `to_preference_triple`, `shares_prompt`, `is_wellformed_preference`, `audit_preferences`, `PreferenceAudit.tie_fraction`, `load_preference_dataset`, `bradley_terry_probability`, `bradley_terry_loss` |
| Reference-model checks | `src/alignlab/dpo.py` — `verify_reference_is_frozen`, `verify_zero_reward_at_init` |
| Experiments | `scripts/experiments/e21_preference_data_readiness.py`, `e22_reference_model_and_kl.py` |
| Tests | `tests/test_logprobs_and_preference.py` — especially `test_mean_divides_by_CONTRIBUTING_tokens_not_labelled_ones`, `test_leading_mask_does_not_lose_a_token`, `test_kl_is_NOT_symmetric` |
| Artefacts | `docs/phase5/e21_preference_readiness.json`, `e22_reference_and_kl.json` |
| Docs | `docs/phase5/PHASE_5_REPORT.md`; `STUDY_WITH_CLAUDE/phase5/01_rlhf_ppo_and_dpo.md`; `INTERVIEW_DEFENSE/phase5_rlhf_dpo.md`; `LEARNING_RESOURCES/phase5_resources.md` |

### 9.7 Code-reading strategy

1. **`sequence_logprobs`** — find where `n_tokens` is computed and confirm it counts
   **contributing** positions after the shift, not labelled ones. A test pins this;
   understand why the distinction bit the project in Phase 3 too.
2. **`approximate_kl`** — locate the k3 form and compare it against the exact
   `token_kl`.
3. **`audit_preferences`** — see what it checks before the data is allowed near a
   training loop, and match those checks to E21's four hypotheses.

### 9.8 Experiments you must understand

**E21 — is the preference data usable?**
*Dataset:* `HuggingFaceH4/ultrafeedback_binarized`.
*Hypotheses:* H1 chosen and rejected share a prompt · H2 the token prefix matches
for both responses · H3 there is a length bias in tokens · H4 loading is
reproducible.
*Measured:* **all four HOLD.** **11.9% ties.** Chosen responses are **56.5% longer**
in tokens.
*Supports:* the data is structurally sound for DPO.
*Does NOT prove:* that the data is *good* — only that it loads consistently and has
the structure the objective assumes.
*Read H3 as the warning it turned out to be.* Before opening Phase 6, predict what
a 56.5% length gap does to a SUM objective.
*Verdict:* **HOLDS — READY for Phase 6.**

**E22 — reference model, KL, and the pre-DPO baseline. 108 usable pairs.**
*Hypotheses:* H1 KL(SFT‖base) > 0 and small · H2 KL(LoRA) < KL(SFT) · H3 the
implicit reward is exactly 0 when π = π_ref · **H4 the SFT model already prefers
the chosen response more than half the time by SUM** · H5 MEAN favours chosen more
than SUM.
*Measured:*

| policy | KL(policy‖base) mean | **median** | max |
|---|---:|---:|---:|
| full SFT | 0.6547 | **0.2044** | 7.3349 |
| LoRA @2e-4 | 0.6233 | 0.2008 | — |
| QLoRA @2e-4 | 0.5564 | 0.1589 | — |

The distribution is heavily right-skewed (mean ≈ 3× median), so **the median is the
number to quote**: one epoch of SFT moved a typical token's distribution by ~0.20
nats. *Remember this — the "250× under-budget" claim in Phase 6 compares against
**0.2044**, the median, not 0.6547.*

| model | prefers chosen, **SUM** | prefers chosen, **MEAN** |
|---|---:|---:|
| full SFT | **51/108 = 47.2%** | **63/108 = 58.3%** |
| LoRA @2e-4 | 50/108 = 46.3% | 62/108 = 57.4% |
| QLoRA @2e-4 | 50/108 = 46.3% | 62/108 = 57.4% |

*Outcomes:* H1 HOLDS (with a caveat its criterion hid) · H2 HOLDS (0.6233 < 0.6547)
· H3 **HOLDS EXACTLY** — implicit reward `0.000e+00` when π = π_ref · **H4
DISPROVED — 47.2%, below chance** · H5 HOLDS (58.3% vs 47.2%, an **11.1-point
swing** on identical models and data).
*Why H4's failure matters:* **the metric DPO optimises did not start above chance.**
This reframes all of Phase 6 before Phase 6 runs.
*Method note, and a good story:* a claimed non-negativity property of the **k3** KL
estimator was **falsified by executing it** — **11 negatives in 200,000 float32
samples** (min −2.980e-08), **zero** in float64. Mathematically non-negative;
floating-point cancellation in practice. Also measured: `k1` has **194×** the
variance of `k3`.
*Criterion defect, recorded not hidden:* H1's criterion `0 < mean < 1` could not
test the word "small" it was meant to test — recorded as **UNTESTABLE AS STATED**.
*Verdict:* **H4 DISPROVED.**

### 9.9 Can I defend this?

**A. Basic.** What does preference data look like, and what does DPO need from it?

**B. First principles.** Derive Bradley-Terry and turn it into a reward-model loss.

**C. Mathematical.** Write KL both ways. Explain which direction RLHF penalises and
what the other direction would do.

**D. Implementation.** Where does the project compute sequence log-probabilities,
and why does `mean_logprob` divide by *contributing* tokens?

**E. Why this design.** Why must the reference model be frozen **and** in `eval()`?
Name a failure mode for each requirement separately.

**F. Experiment.** Why measure the preference baseline *before* training? What
would have happened to the Phase 6 write-up without E22?

**G. Results interpretation.** Explain the 11.1-point SUM/MEAN swing on identical
models and data.

**H. Failure/bug.** The k3 non-negativity claim: what was claimed, how was it
falsified, and what is the correct statement now?

**I. Skeptical follow-ups.**
- *"You didn't implement PPO. Isn't your DPO-vs-PPO comparison hollow?"*
  (`phase5_rlhf_dpo.md` Q11.) A strong answer concedes the scope boundary
  immediately, states what *was* implemented, and defends the choice on cost and
  on what the project was trying to learn.
- *"When would PPO still be the right call?"* (Q9.) You need a real answer here,
  not a dismissal.
- *"Your KL was 0.65 for SFT. That sounds large."* A strong answer immediately
  distinguishes mean from median on a right-skewed distribution.

> **MY EXPLANATION** (Checkpoints 17–19, `STUDY_WITH_CLAUDE/phase5/`):
>
> _______________________________________________

---

## 10. Phase 6 — DPO

### 10.1 Phase purpose

Implement DPO from the published formula, **preregister** the β sweep, run it, and
report the result honestly — which turned out to mean reporting a null.

**Why it exists:** to close the loop from preference data to a trained policy
without a reward model, and to demonstrate preregistered experimental practice.

**What depends on it:** Phase 7 evaluates the DPO checkpoint and independently
replicates the null.

### 10.2 Prerequisites

Everything from Phase 5, plus:

- **Constrained optimisation:** the KL-regularised objective and its closed-form
  optimum `π*(y|x) ∝ π_ref(y|x)·exp(r(x,y)/β)`.
- **Partition functions:** what `Z(x)` is and why a *difference* of two terms
  containing it cancels.
- `logsigmoid` and why it beats `log(sigmoid(x))` numerically.

Read `PYTORCH_CONCEPTS/pytorch-dpo-mechanics.md`; run
`examples/dpo_mechanics_examples.py`.

### 10.3 Concepts, simple → hard

| # | Concept | Understand | Derive | Implement from scratch | Depth | Priority |
|---|---|---|---|---|---|---|
| 1 | The DPO reparameterisation | Reward as a log-ratio to the reference | **Yes** | — | PROJECT-READY | **MUST** |
| 2 | The DPO loss | `−log σ(β(Δlog π_θ − Δlog π_ref))` | **Yes, cold** | **Yes** | PROJECT-READY | **MUST** |
| 3 | Where `log Z(x)` cancels | The step that makes DPO possible | **Yes** | — | PROJECT-READY | **MUST** |
| 4 | Loss = ln 2 at init | Why, and why it is a good sanity check | **Yes** | — | PROJECT-READY | **MUST** |
| 5 | β | What it trades off; why small β both keeps and constrains | **Yes** | — | PROJECT-READY | **MUST** |
| 6 | Implicit reward | `β(log π_θ − log π_ref)` | **Yes** | **Yes** | PROJECT-READY | **MUST** |
| 7 | SUM vs MEAN | Published objective vs diagnostic | **Yes** | — | PROJECT-READY | **MUST** |
| 8 | Length attribution | Decomposing a SUM gap into length + residual | **Yes** | **Yes** | PROJECT-READY | **MUST** |
| 9 | Preregistration | What it protects against | — | — | INTERVIEW-READY | **MUST** |
| 10 | Reading a null result | Quantifying the shortfall rather than excusing it | — | — | PROJECT-READY | **MUST** |

### 10.4 First-principles targets

**Derive the DPO objective, cold.** The chain, with no steps skipped:

1. Start from `max_π E[r(x,y)] − β·KL(π ‖ π_ref)`.
2. Its optimum is `π*(y|x) = (1/Z(x))·π_ref(y|x)·exp(r(x,y)/β)`.
3. Solve for the reward: `r(x,y) = β·log(π*(y|x)/π_ref(y|x)) + β·log Z(x)`.
4. Substitute into Bradley-Terry: `P(y_w ≻ y_l) = σ(r_w − r_l)`.
5. **`β log Z(x)` appears in both terms and cancels**, because Z depends only on x
   and both responses share the prompt. This is the step to be able to say out loud.
6. Maximum likelihood gives
   `L = −log σ( β[(log π_θ(y_w|x) − log π_ref(y_w|x)) − (log π_θ(y_l|x) − log π_ref(y_l|x))] )`.

**Why the loss is exactly `ln 2 = 0.6931471805599453` at initialisation.** At
init, π_θ = π_ref, so both implicit rewards are 0, the logit is 0, and
`−log σ(0) = log 2`. The repository asserts this in
`tests/test_dpo.py::test_loss_at_initialisation_is_exactly_log_two`, and E22's H3
verified the implicit reward is `0.000e+00` — **exactly**, not approximately.

**The length decomposition.** Given mean summed log-probs and mean token counts for
chosen and rejected, split the SUM gap into a part explained by the token-count
difference at the average per-token rate, and a residual. Reproduce
**−29.47 = −31.90 + 2.43** from the Phase 6 numbers.

### 10.5 Repository implementation

| Concept | Inspect |
|---|---|
| Loss | `src/alignlab/dpo.py` — `implicit_rewards` (note the `length_normalise` branch), `dpo_loss` (note `F.logsigmoid` and the stated reason), `DPOBatchStats`, `preference_accuracy` |
| Guards | `dpo.verify_reference_is_frozen`, `dpo.verify_zero_reward_at_init` |
| Training loop | `src/alignlab/dpo_train.py` — `resolve_policy_path`, `tokenize_pair`, `score_pair`, `evaluate`, `run_dpo` |
| Config | `configs/dpo/default.yaml` — read every comment: β=0.1 as the preregistered reference point, `length_normalise: false` = the **published** objective, lr 5e-7 as convention with **no LR sweep run**, batch 1 × grad-accum 16 |
| Preregistration | `docs/phase6/BETA_PREREGISTRATION.md` — **read this before the report**, every time |
| Tests | `tests/test_dpo.py` — `TestImplicitRewards`, `TestDPOLoss`, `TestPreferenceAccuracy::test_sum_and_mean_can_disagree` |
| Artefacts | `docs/phase6/e23_dpo_evaluation.json`, `dpo-beta{0.01,0.1,0.5}-sum_summary.json`, `diag-dpo-beta0.1-lr5e-6_summary.json` |
| Docs | `docs/phase6/PHASE_6_REPORT.md` §7–§8 (the two most important sections); `STUDY_WITH_CLAUDE/phase6/01_dpo_theory_and_results.md`; `INTERVIEW_DEFENSE/phase6_dpo.md`; `HOW_TO_RUN.md` §7.2, §7.3, §9.2 |

### 10.6 Code-reading strategy

1. **`dpo_loss`** — map each line to a symbol in your derivation. Confirm the four
   `SequenceScores` inputs (policy chosen/rejected, reference chosen/rejected) and
   that the loss is `−logsigmoid(chosen_reward − rejected_reward)`.
2. **`implicit_rewards`** — the `length_normalise` flag is the entire SUM/MEAN
   story in one branch. Note that the default (`False`) is the **published**
   objective and MEAN is labelled a **diagnostic**.
3. **`DPOBatchStats.reward_accuracy`** — read its comment: the fraction of pairs the
   policy already ranks correctly, *and it is not implied by the loss falling*.
   That distinction is what let Phase 6 say "DPO trained, in the correct direction"
   while the headline accuracy did not move.
4. **`dpo_train.evaluate`** — see how SUM and MEAN are both computed on every pass,
   so one can never be reported without the other.

### 10.7 Experiments you must understand

**The preregistered β sweep — `dpo-beta0.01-sum`, `dpo-beta0.1-sum`,
`dpo-beta0.5-sum`.**
*Preregistration:* `docs/phase6/BETA_PREREGISTRATION.md`, committed **13:15:12,
before any DPO code existed**. It fixed β ∈ {0.01, 0.1, 0.5}, SUM as the primary
objective, MEAN as a diagnostic, five hypotheses, and the criteria for declaring a
negative result — all in advance, so β could not be chosen after seeing results.
*Result:* **NULL.** Under the preregistered configuration and training budget, DPO
did **not** change SUM preference accuracy on the evaluation set: **86/184 before
and after.**
*The mechanism, measured rather than excused:*

| | chosen | rejected |
|---|---:|---:|
| mean summed log-prob | −291.12 | −261.65 |
| mean tokens | 271.7 | 242.2 |
| **mean per-token log-prob** | **−1.0713** | **−1.0803** |

```
SUM gap (chosen − rejected)      = −29.47 nats
explained by length alone        = −31.90 nats
residual once length is removed  =  +2.43 nats   (chosen is BETTER per token)
```

To flip the *average* pair under SUM the policy must move the gap by ~29 nats.

| run | movement achieved | shortfall |
|---|---:|---:|
| β=0.1, lr 5e-7 | +0.0405 nats | **728× short** |
| [POST-HOC] lr 5e-6 | +0.7004 nats | **42× short** |

KL from the reference was ~**0.0008**, against SFT's **0.2044** median — roughly
**250× under-budget**. The policy barely moved.
*Concluded:* **NULL RESULT, preserved as the primary conclusion.**
*What it is NOT:* "DPO does not work." It is a statement about **this configuration
and this budget**. It also cannot distinguish β values — the budget was too small
for that.
*Independently replicated* in `eval-full-001` on a different evaluation path.

**E23 — SFT vs DPO evaluation.** The one numbered Phase 6 experiment, planned back
in Phase 5 as the comparison that would decide whether DPO changed anything.

| arm | SUM | MEAN | KL from reference |
|---|---|---|---:|
| SFT (baseline) | 0.4674 (86/184) | 0.5870 | 0.000000 |
| DPO β=0.01 | 0.4674 (86/184) | 0.5870 | 0.000801 |
| DPO β=0.1 | 0.4674 (86/184) | 0.5870 | 0.000799 |
| DPO β=0.5 | 0.4674 (86/184) | 0.5870 | 0.000803 |

**Byte-identical, not merely close.** Wilson intervals overlap completely; the
report says **"NOT resolvable at this sample size"** rather than "+0.0%".
*Verdict:* **NULL RESULT.**

**How the project checked the null was not a bug.** The weights *did* change:
perplexity moved in the **4th decimal**, greedy generation differed on **4 of 6**
prompts in Phase 7, and KL from the reference was non-zero and **monotonic in β**.
DPO trained; the metric could not see it. Be ready to give this answer unprompted —
it is the difference between "a null result" and "a broken pipeline".

**Post-hoc 10× learning rate — `diag-dpo-beta0.1-lr5e-6`.**
*Status:* **POST-HOC.** Run after seeing the sweep's result, therefore **not**
covered by the preregistration, and labelled as such everywhere.
*Measured:* gap shifted by **0.7004 nats** — still **42× short**. Generated output
**~30% longer** (mean length 93.8 → **122.0**, max 146 → 206) — the predicted length
pathology, appearing exactly when training had enough signal. KL **0.002365**
(versus ~0.0008 preregistered; still ~86× under SFT's 0.2044). **SUM accuracy
unchanged at 86/184.** MEAN moved 0.5870 → **0.5924** — one pair's worth.
*Verdict:* **POST-HOC — does not replace or overwrite the preregistered result.**
*Also note:* no arm lost the stop token — 4/4 throughout, 0/4 hit the length cap.

### 10.8 Can I defend this?

**A. Basic.** What is DPO, and what does it remove from the RLHF pipeline?

**B. First principles.** Derive it from the KL-constrained optimum, showing exactly
where `log Z(x)` cancels and why it is allowed to.

**C. Mathematical.** (i) Why is the loss exactly `ln 2` at init? (ii) Reproduce
−29.47 = −31.90 + 2.43. (iii) What does β multiply, and what happens to the
gradient as β → 0 and β → ∞?

**D. Implementation.** Why `F.logsigmoid` instead of `log(sigmoid(...))`? Name the
input case where it matters most.

**E. Why this design.** Why is SUM the primary objective and MEAN only a
diagnostic, given that Phase 5 already knew MEAN looked better? (Because SUM is the
**published** objective and the preregistration fixed it. Choosing MEAN after
seeing 58.3% > 47.2% would be exactly the practice preregistration exists to
prevent.)

**F. Experiment.** How does anyone know β was not chosen after the fact? Name the
artefact and the timestamp.

**G. Results interpretation.** State the null in the project's own words, and then
quantify it two ways (728× on the metric, ~250× on KL).

**H. Failure/bug.** How did you rule out "DPO silently didn't train"?

**I. Skeptical follow-ups.**
- *"Your headline result is that nothing happened. Why should I be impressed?"*
  (`phase6_dpo.md` Q10.)
- *"Isn't 'the metric was wrong' just an excuse for DPO not working?"*
  (`phase8_project_defence.md` Q5.) The strongest counter is Phase 7's finding that
  the SUM/MEAN inversion appears on the **untrained base model** too — so it cannot
  be an artefact of any training stage.
- *"DPO changed the weights. Doesn't that mean it improved the model?"* A strong
  answer separates *the policy moved* from *the policy improved*, and notes that
  the only aggregate that moved at all was MEAN, by one pair, in a post-hoc run.

> **MY EXPLANATION** (Checkpoints 20–22, `STUDY_WITH_CLAUDE/phase6/`):
>
> _______________________________________________

---

## 11. Phase 7 — evaluation

### 11.1 Phase purpose

Build an evaluation subsystem that measures five models on one pass and **refuses
to produce an aggregate quality score**, then run it and report what is and is not
resolvable.

**Why it exists:** because in every phase from 3 to 6, two metrics disagreed and
the disagreement was the finding. A weighted average would have erased all four.

**What depends on it:** Phase 8, and every claim you will make in an interview.

### 11.2 Prerequisites

- **Statistics:** binomial proportion; why the textbook (Wald) interval fails at
  small n or extreme p; the **Wilson** interval; what "overlapping intervals" does
  and does not tell you.
- **Metrics:** perplexity as `exp(mean NLL)`; n-gram diversity (distinct-n).
- **Experimental design:** position bias, order-randomisation, positive controls,
  in-distribution vs out-of-distribution evaluation.

Read `PYTORCH_CONCEPTS/pytorch-evaluation-mechanics.md`; run
`examples/evaluation_mechanics_examples.py`.

### 11.3 Concepts, simple → hard

| # | Concept | Understand | Derive | Implement from scratch | Depth | Priority |
|---|---|---|---|---|---|---|
| 1 | Perplexity, precisely | token-weighted `exp(mean NLL)` | **Yes** | **Yes** | PROJECT-READY | **MUST** |
| 2 | Region-scoped perplexity | Why two regions are not comparable | **Yes** | — | PROJECT-READY | **MUST** |
| 3 | Stop-token rate vs termination rate | Why they are separate metrics | — | — | PROJECT-READY | **MUST** |
| 4 | distinct-2 | What it detects, what it does not measure | — | **Yes** | INTERVIEW-READY | **MUST** |
| 5 | Wilson interval | Why not the Wald interval | **Yes** | **Yes** | PROJECT-READY | **MUST** |
| 6 | Resolvability | Answering "can this data decide?" instead of a p-value | — | — | PROJECT-READY | **MUST** |
| 7 | Length attribution | Splitting a SUM gap into length + residual | **Yes** | **Yes** | PROJECT-READY | **MUST** |
| 8 | LLM-as-judge, two-order protocol | Why every pair is judged twice | — | — | PROJECT-READY | **MUST** |
| 9 | Position bias as a first-class metric | Why it is reported, not corrected away | — | — | PROJECT-READY | **MUST** |
| 10 | Evaluation leakage | Why every number here is in-distribution | — | — | PROJECT-READY | **MUST** |

### 11.4 First-principles targets

- **Wilson interval.** Know why it is derived by inverting the score test rather
  than plugging p̂ into the normal approximation, and why that fixes the behaviour
  at p̂ = 0 or 1 — which matters here because stop-token rates are literally 0/6 and
  6/6.
- **Why overlapping intervals is a weak test.** Understand that non-overlap implies
  a significant difference but overlap does **not** imply no difference. Then note
  what the project chose to do: `difference_is_resolvable()` returns **no p-value**
  and answers one narrow question.
- **Perplexity from token totals.** `exp(total_nll / total_tokens)` and why that is
  not the mean of per-example perplexities.

### 11.5 Repository implementation

| Concept | Inspect |
|---|---|
| Metrics | `src/alignlab/evals/metrics.py` — `perplexity`, `PerplexityResult.comparable_to` (returns False across regions **even for identical values**), `perplexity_from_totals`, `TerminationStats` (separate `termination_rate` and `stop_token_rate`), `distinct_n`, `max_ngram_repeat`, `structural_checks`, `PreferenceStats.length_attribution`, `wilson_interval`, `Interval.excludes`, `difference_is_resolvable`, `compare_proportions` |
| Judge | `src/alignlab/evals/judge.py` — `JUDGE_SYSTEM`, `JUDGE_TEMPLATE`, `parse_verdict`, `JudgeResult.position_bias_rate`, `win_rate_b`, `judge_pairs` |
| Report | `src/alignlab/evals/report.py` — `ModelEvaluation.status_of` (APPLICABLE / NOT_APPLICABLE / NOT_MEASURED), `Dashboard.to_dict` (emits `no_aggregate_score`), `compare_models`, `render_text` |
| Runners | `src/alignlab/evals/runners.py` — `GenerationSettings`, `EvalProvenance`, `build_provenance`, `completion_perplexity`, `preference_evaluation`, `generation_evaluation` |
| Entrypoint | `src/alignlab/evaluate.py` — `GENERATION_PROMPTS` (the six fixed prompts) |
| Config | `configs/eval/default.yaml` — the five models, the four comparisons, and the comments explaining greedy decoding and the same-family judge risk |
| Tests | `tests/test_eval_metrics.py` — `test_identical_numbers_different_regions_are_not_comparable`, `test_no_aggregate_score_is_produced`, `test_reports_are_descriptive_not_scored`; `tests/test_eval_judge.py` |
| Artefacts | `docs/phase7/eval-full-001_dashboard.json`, `.txt`, `eval-full-001_run.log.txt` |
| Docs | `docs/phase7/PHASE_7_REPORT.md`, `EVAL_RESULTS.md`; `STUDY_WITH_CLAUDE/phase7/01_evaluation_theory.md`; `INTERVIEW_DEFENSE/phase7_evaluation.md` |

### 11.6 Code-reading strategy

1. **`PerplexityResult.comparable_to`** — three lines that encode a Phase 3 finding
   as a type-level guard. Ask: what bug class does this make impossible?
2. **`PreferenceStats`** — confirm it **cannot serialise SUM without MEAN**, or
   either without token counts, and that `length_attribution()` runs automatically.
   This is design-following-finding again.
3. **`wilson_interval` and `difference_is_resolvable`** — note the deliberate
   absence of a p-value, and read the docstrings for why.
4. **`judge_pairs`** — find where each pair is presented in both orders and where a
   disagreement causes the pair to be **excluded** rather than counted as half a
   win.
5. **`report.Dashboard.to_dict`** — find `no_aggregate_score` and then find the test
   that asserts it. A design constraint enforced by a test is a strong talking
   point.

### 11.7 The experiment — `eval-full-001`

*Scope:* five models (base, SFT, LoRA r=16, QLoRA r=16, DPO β=0.1) in one pass —
two perplexity regions, preference SUM and MEAN with length attribution, six
generation prompts, and a two-order LLM judge.
*Sets:* `no_robots` test, 150 examples, **27,127 completion tokens**;
`ultrafeedback_binarized` `test_prefs`, 200 rows → **184 usable**; generation greedy,
`max_new_tokens=256`, seed 42; judge `Qwen/Qwen2.5-7B-Instruct` **@ `main` (not a
SHA)**.

**Per-model results:**

| model | ppl[compl] | ppl[full] | pref SUM | pref MEAN | stop | gen len | distinct-2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 8.824 | 14.993 | 45.7% | 61.4% | 0/6 | 212.3 | 0.523 |
| **SFT** | **7.398** | **12.284** | 46.7% | 58.7% | **6/6** | 117.3 | **0.839** |
| LoRA r=16 | 7.665 | 13.600 | 46.7% | 60.9% | 0/6 | 256.0 | 0.449 |
| QLoRA r=16 | 7.733 | 13.732 | 46.7% | 61.4% | 0/6 | 256.0 | 0.572 |
| DPO b=0.1 | 7.398 | 12.287 | 46.7% | 58.7% | **6/6** | 118.2 | 0.831 |

**The headline number: of 24 pairwise comparisons, 3 were resolvable at 95%
confidence — all three were stop-token rates.**

**Finding 1 — SFT is the only stage that moved anything visible.** Perplexity
8.824 → 7.398 (**−16.2%**), stop **0/6 → 6/6 (RESOLVED)**, generation length
212.3 → 117.3, distinct-2 0.523 → 0.839. The base model's failures are qualitative
and severe: `.DrawString(...)` repeated to the cap; `-unstyled` about 128 times; one
English prompt answered in Chinese. **Note the metric-design point:** the base model
*terminated* twice out of six but emitted the ChatML turn terminator **zero** times.
A single "did it finish" metric would have scored it 33% and hidden what SFT taught.

**Finding 2 — the PEFT arms reproduce Phase 4's failure, and the generations show
why it is worse than "runs long".** LoRA and QLoRA hit the 256-token cap on 6/6
prompts (min = max = median = 256). Past the answer they **fall back into the chat
template**, emitting `You are a helpful assistant.` repeatedly. With no `<|im_end|>`
in their learned behaviour, the likeliest continuation after a finished answer is
the next ChatML turn. distinct-2 ranks them **last of all five models** (LoRA
0.449, QLoRA 0.572). **This failure is invisible to every quantitative metric here
and obvious in the text** — that is `HOW_TO_RUN.md` §18 principle 8, made concrete.

**Finding 3 — DPO changed the weights and no aggregate metric.** Perplexity moved
in the **4th decimal** (7.398 vs 7.398 at completion, 12.284 vs 12.287 at full),
generation differs on **4 of 6** prompts, and nothing aggregate changed. This
**replicates the Phase 6 null on a different evaluation path with a different
subset.**

**Finding 4 — the SUM/MEAN inversion is a property of the metric.**

| model | chosen/token | rejected/token | SUM gap | explained by length | residual |
|---|---:|---:|---:|---:|---:|
| **base (untrained)** | −1.1470 | −1.1607 | −30.57 | −34.28 | **+3.71** |
| SFT | −1.0713 | −1.0803 | −29.47 | −31.90 | **+2.43** |
| LoRA r=16 | −1.1001 | −1.1098 | −30.13 | −32.78 | **+2.65** |
| QLoRA r=16 | −1.0924 | −1.1005 | −30.30 | −32.50 | **+2.20** |
| DPO β=0.1 | −1.0712 | −1.0804 | −29.43 | −31.91 | **+2.48** |

Every model prefers the chosen response **per token**. Every model's SUM comparison
**inverts** that verdict. The residual is positive in **all five rows, including the
untrained base model**, which has had no preference training of any kind. **The
inversion is not caused by any training stage.**

*This table is the single strongest piece of evidence you own against the "the
metric excuse" challenge. Memorise the base-model row.*

**Finding 5 — the judge did not confirm the one thing everything else agrees on.**

| | base vs SFT | SFT vs DPO |
|---|---|---|
| position-inconsistent | **2 of 6 (33.3%)** | 1 of 6 (16.7%) |
| decided | 4 | 2 |
| win rate (B) | 0.750 **[0.301, 0.954]** | 0.500 [0.095, 0.905] |
| verdict | **NOT resolvable at n=4** | **NOT resolvable at n=2** |

Unparsed replies: **0**. Base emits `-unstyled` 128 times, SFT writes a correct
email, and the judge still could not resolve the comparison. On the cooking prompt —
where base answered in Chinese — it replied `A` in **both** orders: it picked
whichever it saw first, twice.
*An accidental positive control passed:* two SFT-vs-DPO pairs were byte-identical
and a third differed by two words; the judge returned `TIE` on all three, in both
orders, **unprompted**. Weak evidence, and it was not designed in — say both halves.
*Calibration from the literature* (MT-Bench, arXiv 2306.05685, Table 2, read
directly): GPT-4 is self-consistent on only **65.0%** of pairs, Claude-v1 on
**23.8%**. A high bias rate from a 7B judge is the expected result. **No comparison
with those figures is claimed — n=6.** The `LEARNING_RESOURCES/phase7_resources.md`
entry marks this **PARTIALLY INSPECTED**.
*What the judge does NOT establish:* anything about agreement with humans. No human
study was run; the judge is **same model family** as the models under test; and it
is pinned to `main`, **not a commit SHA** — a deviation from the project's practice
everywhere else. The judge prompt instructs it to ignore length, and **that was not
verified**.

**Defects this run found:** `dataset_fingerprint` **null in all five** per-model
provenance records, and `configure_hf_cache` a **no-op in every entrypoint**,
costing **18.34 GB** of duplicate weights.

### 11.8 Can I defend this?

**A. Basic.** How does AlignLab decide whether a model improved?

**B. First principles.** Why is perplexity insufficient? Use the Phase 4/7 numbers:
3.6% separation against 0/6 vs 6/6.

**C. Mathematical.** Derive the Wilson interval's motivation; explain why it beats
the Wald interval at 0/6 and 6/6. Compute a length attribution from raw totals.

**D. Implementation.** Where is the "no aggregate score" rule enforced, and by what?

**E. Why this design.** Why is termination separate from stop-token emission? Why
does `comparable_to` return False for identical values in different regions?

**F. Experiment.** Why judge every pair twice? What happens to a disagreeing pair,
and why is that better than counting it as half a win?

**G. Results interpretation.** 24 comparisons, 3 resolvable. Is that a failed
evaluation? A strong answer distinguishes *the instrument was too weak* from
*nothing happened*, and names which of the two applies to which comparison.

**H. Failure/bug.** The `dataset_fingerprint` null: what happened, and why was it
**not back-filled**?

**I. Skeptical follow-ups.**
- *"Your evaluation found almost nothing improved. Isn't that a failed project?"*
  (`phase7_evaluation.md` Q8.)
- *"What's the weakest part of your evaluation?"* (Q9.) The intended answer is in
  `docs/LIMITATIONS.md` §7: **everything is in-distribution**, and it is called the
  single largest gap in the project. Say it before the interviewer does.
- *"Your judge is from the same family as the models it judges. Why report it at
  all?"* (`phase8_project_defence.md` Q9.)

> **MY EXPLANATION** (Checkpoints 23–25, `STUDY_WITH_CLAUDE/phase7/`):
>
> _______________________________________________

---

## 12. Phase 8 — reproducibility and project defence

### 12.1 Phase purpose

Audit the repository against its own claims, fix what was found, clean up, and
write the documents a reader needs to judge whether the results are trustworthy.

### 12.2 Concepts

| # | Concept | Depth | Priority |
|---|---|---|---|
| 1 | Provenance auditing as an executable check | PROJECT-READY | **MUST** |
| 2 | Tier A / B / C reproducibility claims | PROJECT-READY | **MUST** |
| 3 | Environment drift between two machines | PROJECT-READY | **MUST** |
| 4 | Preregistration as a practice | INTERVIEW-READY | **MUST** |
| 5 | Preserving disproved hypotheses verbatim | PROJECT-READY | **MUST** |
| 6 | Knowing what a project does NOT claim | PROJECT-READY | **MUST** |

### 12.3 Repository implementation

- `python -m alignlab.provenance` — exits 1 if any field is missing. Run it; it is
  cheap and needs no model.
- `docs/phase8/PHASE_8_REPORT.md`, `docs/phase8/CLEANUP_RECORD.md`.
- `docs/LIMITATIONS.md` — 18 numbered items plus "what this project does not claim".
- `docs/EXPERIMENT_REGISTRY.md` — every hypothesis as written *before* the result.
- `INTERVIEW_DEFENSE/phase8_project_defence.md` — Q1–Q11.

### 12.4 Findings

**F13 — the HF-cache no-op.** `configure_hf_cache` was a no-op in every entrypoint
because `huggingface_hub` freezes `HF_HUB_CACHE` at import. Cost: **18.34 GB** of
duplicate weights, recovered. *Why it matters:* a documented guarantee that was
never actually enforced. The auditor now exists so the claim is *checked*.

**The permanent provenance gap.** `eval-full-001` recorded `dataset_fingerprint:
null` in all five per-model records. The code is fixed and a later run
(`eval-cachefix-001`) is clean, but that artefact's field is left **NOT RECORDED**.
It was **not back-filled**, because reconstructing provenance after the fact would
be worse than lacking it. The auditor reports **17/18**. Understand and be able to
defend that choice — it is a small ethics-of-measurement question and interviewers
like it.

**Environment drift.** The two machines run different torch, transformers and
Python versions. Tier B "holds only in the direction that matters": every *result*
comes from the server and every run manifest records that machine's exact versions,
so any run can be reconstructed from its own record.

### 12.5 Can I defend this?

- *"How do I know your numbers are real?"* (`phase8_project_defence.md` Q7.) A
  strong answer names the manifest fields, the auditor, the pinned model revision
  `8faed761…`, the dataset fingerprints, and a linear history with no force-pushes
  and no rebases (92 commits at the handoff's `6c11e75`; 95 today — check before
  quoting a count).
- *"Why not back-fill the missing fingerprint?"* State the principle, not just the
  decision.
- *"Your local and server environments differ. Doesn't that break reproducibility?"*
  A strong answer uses the Tier vocabulary precisely and does not overclaim Tier A.

---

## 13. Interview chains

Each chain is a drill-down an interviewer can run. Work through one chain per
session, out loud, timing yourself. **The answers are not written here on purpose.**
What follows each arrow is the question and, in brackets, what a strong answer must
contain.

### Chain 1 — Attention

1. What is self-attention? *[three projections; a weighted average of values; weights from query-key similarity]*
2. Write it. *[softmax(QKᵀ/√d_k)V, with shapes]*
3. Why three separate matrices? *[what breaks if W_Q = W_K — symmetric scores]*
4. Why divide by √d_k? *[variance-of-a-dot-product argument, explicitly]*
5. Prove it matters. *[E1's four numbers, including Jacobian mass 0.708 → 0.014]*
6. What happens if you remove the causal mask? *[E2: 0.2907 → 0.0088, 33.2×, and the lesson]*
7. Why mask *before* softmax? *[surviving weights must still sum to 1]*
8. What does softmax's axis have to be, and why? *[key axis; what breaks otherwise]*
9. You have three implementations agreeing to 1e-16. Are you correct? *[no — same function, not the right function]*
10. Now make it multi-head. Every shape. *[the full chain including the output projection]*
11. Now make it GQA. What changes? *[H_kv; `_repeat_kv`; cache size vs compute shape]*
12. Quantify the saving. *[E4: params 53.12%, cache 6.25%]*
13. Did latency improve? *[CPU yes 160.4→117.4 ms; **GPU flat** 0.86→0.81 ms — refuse to generalise]*
14. Where does position come from? *[RoPE; rotation of 2-D pairs]*
15. Prove RoPE is relative. *[E5: spread 2.384e-07 at fixed gap]*
16. How do you know your RoPE offset is wired into cached decoding? *[E6 — a guard that can fail]*
17. What would you test next? *[your own answer — long-context extrapolation is explicitly outside E5's scope]*

### Chain 2 — SFT

1. What is SFT and how does it differ from pretraining?
2. What is the loss? *[causal LM cross-entropy]*
3. Which tokens contribute? *[completion only; `IGNORE_INDEX = -100`]*
4. Show me the shift. *[work it on a short sequence; `labels[:,1:]`]*
5. How do you know your mask is right? *[E12 H2 and H4; the independent `expected_labels`]*
6. Your decomposition failed by 1.8e-02 first. What was wrong? *[the position count, not the mask]*
7. Is a masked loss higher or lower than an unmasked one? *[neither reliably — different token population]*
8. What did E12 H5 predict, and what happened? *[prompt easier; **DISPROVED**, 6.3314 vs 3.6295]*
9. Why was the prompt harder? *[base model had never seen ChatML]*
10. What did that change in the codebase? *[`PerplexityResult.region`]*
11. Tell me a subtle bug. *[tokenization boundary: token 198 → 1406, 3/200 rows]*
12. Your fix looked ineffective. Why? *[stale `datasets` cache; the fingerprint change proved it landed]*
13. What did SFT actually buy? *[E13: 8.541 → 7.199, −15.7%; stop 0/4 → 4/4]*
14. Is the model better? *[in-distribution caveat first, then the behavioural evidence]*
15. Why is packing off? *[mask verifiability over throughput]*
16. What would you do differently? *[your own answer]*

### Chain 3 — LoRA

1. What is LoRA? *[`W'x = Wx + (α/r)·BAx`, base frozen]*
2. Why low rank? *[parameter count; gradient/optimizer memory]*
3. Why does low rank reduce parameters? *[r(d_in + d_out) vs d_in·d_out]*
4. Calculate the count for this project. *[module by module to **4,358,144**]*
5. That's what percent? *[0.2823% or 0.2815% — and why both appear]*
6. Why r=16? *[a starting point; **no rank sweep was run** — say it]*
7. Could SVD choose the rank? *[explain the recipe, then why it fails here]*
8. Derive SVD's role. *[Eckart–Young; energy uses squared singular values]*
9. What did your SVD experiment find? *[E17: 90% at rank ~730/1536; r=16 = 10.69%; H2 **DISPROVED**]*
10. So LoRA shouldn't work? *[suffice vs be; and it *did* work — within 3.6% perplexity]*
11. Why is r=730 useless anyway? *[break-even 768; 4.9% saving; narrow GQA matrices cost more]*
12. Why did you need a random control? *[the noise control needs 783 — a 7% margin, not an order of magnitude]*
13. What matrices did you target? *[q,k,v,o — 112 of 197 `nn.Linear`]*
14. Why not `lm_head`? *[tied to the embedding]*
15. What happened to stopping behaviour? *[0/4 and 0/6 for every PEFT arm; full SFT 4/4 and 6/6]*
16. Why? *[E20 — **consistent with** reachability; `lm_head` relative movement 0.0136; **NOT CONFIRMED**]*
17. What experiment would distinguish rank limitation from target-matrix reachability? *[a LoRA variant on the MLP or an untied head; ~18 minutes; **DEFERRED**]*
18. Why haven't you run it? *[your own answer — be honest]*

### Chain 4 — Quantization / QLoRA

1. What is quantization? *[map a wide numeric range onto few bits]*
2. Symmetric vs asymmetric? *[scale only vs scale + zero-point]* — **mark this as external theory: AlignLab did not measure it.**
3. fp32 vs fp16 vs bf16? *[exponent vs mantissa; bf16 keeps fp32's range and spends mantissa]*
4. Why does that matter here? *[the bf16 merge: 6.875e-01 vs 6.330e-05, **12,460× worse**]*
5. Why exactly? *[‖ΔW‖/‖W‖ ≈ 0.003 sits at bf16's mantissa resolution]*
6. What did you change because of it? *[E19 evaluates **unmerged**]*
7. What is NF4? *[quantization levels at the quantiles of a normal distribution]*
8. Is QLoRA 4-bit arithmetic? *[**no** — 4-bit **storage**, bf16 compute; dequantize before every matmul]*
9. How do you know the kernel actually ran? *[E15 — executed a quantized matmul, not an import check]*
10. What did QLoRA save? *[peak VRAM 2.88 GiB vs LoRA 4.72 GiB; adapter 0.019 vs 0.027 GiB]*
11. Was it faster? *[**refuse the claim** — 1044 vs 1055 s, ~1%, one seed, no error bar]*
12. What did it cost in quality? *[7.733 vs 7.665 completion perplexity in Phase 7 — and both are 0/6 on stop]*
13. What is `Params4bit.numel()` actually returning? *[**bytes**, not elements]*
14. What would you measure next? *[your own answer]*

### Chain 5 — RLHF / PPO

1. Walk me through RLHF end to end. *[SFT model, preference data, reward model, policy, reference, value net]*
2. Where does the reward model come from? *[Bradley-Terry on preference pairs]*
3. Derive that loss. *[σ(r_w − r_l), then `−log σ(r_w − r_l)`]*
4. What does PPO optimise? *[clipped surrogate on the importance ratio, with a KL penalty]*
5. Why a value network? *[variance reduction via advantages]*
6. Why a reference model? *[KL anchor against reward hacking]*
7. Which KL direction, and why? *[be precise; say what the other direction encourages]*
8. Did you implement PPO? *[**No. CONCEPTUAL ONLY. Say it first, unprompted.**]*
9. Then what did you implement? *[Bradley-Terry functions, log-probs, exact + k3 KL, DPO loss and loop]*
10. Isn't the comparison hollow? *[`phase5_rlhf_dpo.md` Q11 — concede scope, defend the choice]*
11. When is PPO still right? *[Q9 — have a real answer]*
12. Your k3 estimator returned negatives. Explain. *[11 in 200,000 float32, min −2.98e-08, zero in float64]*

### Chain 6 — DPO

1. What is DPO? *[preference optimisation without a reward model or rollouts]*
2. Derive it. *[the five-step chain in §10.4]*
3. Where does `log Z(x)` cancel, and why is that legal? *[both responses share the prompt]*
4. What is the implicit reward? *[`β(log π_θ − log π_ref)`]*
5. What is the loss at initialisation? *[exactly `ln 2`; and E22 H3 verified `0.000e+00`]*
6. What does β control? *[the strength of the KL anchor; behaviour as β → 0 and β → ∞]*
7. Which β did you use, and how do I know you didn't pick it afterwards? *[preregistration, committed 13:15:12 before any DPO code existed]*
8. What happened? *[**86/184 for the baseline and all three arms — byte-identical**]*
9. Did DPO train at all? *[yes — 4th-decimal perplexity, 4/6 generations differ, KL monotonic in β]*
10. Then why did nothing move? *[~29-nat length gap; run moved 0.0405 → **728× short**; KL 0.0008 vs 0.2044 → **~250× under**]*
11. Isn't "the metric was wrong" an excuse? *[the untrained **base model** shows the same inversion, residual +3.71]*
12. What about the 10× LR run? *[**POST-HOC**; 0.7004 nats, still 42× short; output +30% longer; SUM unchanged]*
13. Why is that output-length increase interesting? *[the predicted length pathology appearing exactly when there was enough signal]*
14. So DPO doesn't work? *[**reject the generalisation** — a statement about this configuration and budget]*
15. What would you run next? *[a properly-budgeted run with KL in SFT's range; and a length-controlled evaluation]*

### Chain 7 — Evaluation

1. How do you know a model got better?
2. Why isn't perplexity enough? *[3.6% vs 0/6 vs 6/6]*
3. What did you measure instead? *[stop-token rate, distinct-2, length, direct generation inspection, judge]*
4. Why separate termination from stop-token emission? *[the base model terminated 2/6 and emitted `<|im_end|>` 0/6]*
5. How do you quantify uncertainty? *[Wilson intervals]*
6. Why Wilson and not the textbook interval? *[behaviour at 0/6 and 6/6]*
7. How many comparisons were resolvable? *[**3 of 24**, all stop-token rates]*
8. Is that a failed evaluation? *[weak instrument ≠ no effect; say which applies where]*
9. Why is there no overall score? *[a weighted average would have erased all four metric disagreements; a test enforces this]*
10. Tell me about your judge. *[two orders; disagreement excludes the pair]*
11. How biased was it? *[**33.3%** on base-vs-SFT; 16.7% on SFT-vs-DPO]*
12. Did it confirm your SFT result? *[**no** — 0.750 [0.301, 0.954], not resolvable at n=4]*
13. Is your judge ground truth? *[no — no human study, same family, pinned to `main` not a SHA, and the "ignore length" instruction was never verified]*
14. What is the weakest part of your evaluation? *[**everything is in-distribution** — say it first]*
15. What would you add first? *[an OOD benchmark; then multiple seeds; then a length-controlled win rate — note **AlpacaEval's length-controlled win rate was not inspected or used**]*

### Chain 8 — Reproducibility

1. Can I rerun your SFT and get your number? *[Tier vocabulary, immediately]*
2. What does Tier A claim? *[bitwise, same machine, same seed, same environment — verified across separate processes]*
3. Tier B? *[structural across machines — same code path, shapes, token accounting; verified by comparing manifests, not floats]*
4. Tier C? *[statistical only — CPU-fp32 vs Ampere-bf16]*
5. What is in a run manifest? *[git SHA + dirty flag, config hash, dataset and eval fingerprints, hardware, library versions, dtype, seed]*
6. How is that checked rather than asserted? *[`python -m alignlab.provenance`, exit 1 on a missing field]*
7. What does it currently report? *[**17/18** artefacts complete]*
8. What is the missing one? *[`eval-full-001`'s per-model `dataset_fingerprint`]*
9. Why not fix it? *[reconstructing provenance after the fact would be worse than lacking it]*
10. What is untested in your infrastructure? *[W&B online, SLURM requeue, multi-GPU RNG restore — **NOT TESTED**]*
11. What single change would most improve confidence in your results? *[multiple seeds — no result here has a training-variance estimate]*

---

## 14. Discrepancies in the repository you should know about

These are real, and an attentive interviewer reading your repository could find
them. Knowing them in advance turns a stumble into a strength. **None of them
changes a measured number.**

**14.1 E19's hypothesis numbering differs between two documents.**

- `scripts/experiments/e19_peft_comparison.py` (primary — the hypotheses were
  written in its docstring before the run) lists **H1** ~0.28% trainable · **H2**
  QLoRA VRAM < LoRA < full SFT · **H3 QLoRA is SLOWER per step than LoRA** ·
  **H4** adapters >100× smaller · **H5** eval-loss ordering. Outcome: H1, H2, H4, H5
  HOLD; **H3 DISPROVED** (QLoRA was 1% *faster*).
- `docs/EXPERIMENT_REGISTRY.md` and `docs/FINAL_STUDY_HANDOFF.md` (F6) paraphrase
  E19's hypotheses differently and label **H3** as *"LoRA matches full fine-tuning
  on held-out loss"* → DISPROVED.
- `docs/phase4/e19_peft_comparison.json` records only `H1`–`H4`.

Both statements about the *world* are supported by the data (QLoRA was not slower;
LoRA did not match full fine-tuning). Only the **labels** disagree. **Use the
script's numbering** and, if pressed, say plainly that the summary documents
renumber. Do not pretend the registry's H3 is what the experiment preregistered.

**14.2 Two different preference-evaluation sets, two different percentages.**
E22 used **108 usable pairs** → SUM 47.2% (51/108), MEAN 58.3% (63/108). Phase 6/7
used **184 usable pairs** → SUM 46.7% (86/184), MEAN 58.7%. Both appear in the
documents. Always name the set with the number.

**14.3 KL: mean vs median.** SFT's KL from base is **0.6547 mean / 0.2044 median**
on a heavily right-skewed distribution. The "~250× under-budget" comparison uses the
**median**. Quote the median and say why.

**14.4 E17's headline figures are per-module.** "90% energy at rank ~730 of 1536"
and "r=16 captures 10.69%" are **layer 13 `q_proj`**
(`docs/phase4/PHASE_4_REPORT.md` §5). Layer 0 `q_proj` in the JSON shows 90% at rank
**653** and r=16 energy **13.66%**. Ten modules were analysed. Say "on the modules
we sampled" and name one.

**14.5 The two trainable-parameter percentages.** 0.2823% (against the base model's
count) and 0.2815% (against the count including adapters). Both are in the
repository. See §8.4.

**14.6 The explain-back marker count.** `docs/phase8/PHASE_8_REPORT.md` §6 and §13
say "31 files carry unfilled markers"; the corrected figure is **42 files, 107
occurrences, 25 numbered Checkpoints** (`docs/FINAL_STUDY_HANDOFF.md` §1). The
Phase 8 number counted files not occurrences, searched four folders, used a narrower
pattern, and was taken before `phase8_project_defence.md` existed. The Phase 8
report was left as written, with the handoff carrying the correction — which is
itself the project's stated practice.

**14.7 `PROJECT_INSTRUCTIONS.txt` is an untracked duplicate** of
`PROJECT_INSTRUCTIONS.md` (identical size, 28,189 bytes). It is not part of the
tracked repository. Ignore it.

---

## 15. High-ROI ranking

Ranked by: how much AlignLab implementation backs it, how much measured evidence it
carries, how much `INTERVIEW_DEFENSE/` material exists, and how often it is actually
asked in ML/LLM interviews.

### TIER 1 — MUST MASTER

| Topic | Why the ROI is highest |
|---|---|
| **Tensor-dimension tracking** | Checks every other Tier-1 topic; the project's own recurring bug class was mis-indexed shifts (E12 6d). You cannot fake this on a whiteboard. |
| **Self-attention from scratch** | Three implementations in the repository, plus E1/E2/E3. Universally asked, and here you can answer with measurements instead of assertions. |
| **LoRA: equation, init, parameter count** | Own implementation, `peft` cross-check, exact arithmetic (4,358,144). This is the highest-density question area in modern LLM interviews. |
| **SVD, rank selection, and E17** | The project's most distinctive result. Lets you be the candidate who *disproved* a story everyone repeats. |
| **PCA and PCA ↔ SVD** | Classic derivation, verified numerically here (`|cos| = 1.0000000000`), and it connects directly to E17. |
| **SFT loss masking and the shift** | Two hypotheses (E12 H5), two real bugs, and a design consequence (`PerplexityResult.region`). |
| **DPO derivation** | Own implementation from the published formula, plus a preregistered sweep. The `log Z(x)` cancellation is a standard whiteboard ask. |
| **MHA / MQA / GQA + KV cache** | E4 and E8 give you real numbers *and* the discipline of refusing to generalise the latency result. |
| **Evaluation methodology and Wilson intervals** | "3 of 24 resolvable" is a memorable, defensible headline that shows statistical maturity. |

### TIER 2 — SHOULD MASTER

| Topic | Why |
|---|---|
| **Quantization mechanics, NF4, QLoRA** | E15 verified by execution; the storage-vs-arithmetic distinction is a frequent discriminator. |
| **bf16 / fp16 / fp32 and the merge failure** | 12,460× is a striking number with a clean mechanistic explanation. |
| **RoPE** | E5's invariance measurement plus E6's guard; asked constantly. |
| **Positional encodings generally** | Sinusoidal derivation is a standard whiteboard exercise. |
| **Bradley-Terry** | Short derivation, implemented and tested here. |
| **KL regularisation, reference models** | Needed for both PPO and DPO answers; E22 gives you real numbers. |
| **SUM vs MEAN and length attribution** | The project's second-most distinctive finding; the base-model row is your best evidence. |
| **LLM-as-judge limitations** | Position bias measured at 33.3% with a two-order protocol you can describe precisely. |
| **RMSNorm / SwiGLU** | Cheap to learn, frequently asked, E7 gives measured numbers. |

### TIER 3 — SUPPORTING KNOWLEDGE

| Topic | Why |
|---|---|
| **PPO mechanics** | CONCEPTUAL ONLY here. Know it well enough to explain and to justify choosing DPO; do not oversell. |
| **Flash-attention concepts** | You measured PyTorch's SDPA backends, not a kernel. Know tiling and online softmax; be precise about scope. |
| **Decoding strategies** | E9's value is mostly the bug story. |
| **Reproducibility engineering** | Strong differentiator with infrastructure-minded interviewers; less so in pure modelling interviews. |
| **Storage / preemption / Hydra** | Good "engineering maturity" colour; rarely the main event. |
| **ALiBi** | Know it exists and how it differs from RoPE. Nothing here depends on it. |

---

## 16. Don't waste time on these

Each item names *why* it is low-value **for your goal**, not that it is worthless.

**16.1 Rebuilding or re-running anything.** The project is complete and every
result is committed. Re-running SFT costs ~20 minutes of A6000 time and teaches you
nothing you cannot get from `sft_run_summary.json`. Read the artefacts instead.

**16.2 `beam_search` internals** (`generation.py:225`). Implemented, tested, and
not used by any headline result. Know what beam search is; skip the bookkeeping.

**16.3 ALiBi's slope schedule.** Implemented (`ALiBiBias._slopes`) but no AlignLab
experiment depends on it. Concept only.

**16.4 Hydra config resolution mechanics.** Know *why* config groups beat argparse
(`phase1_engineering.md` Q5). Do not study the resolution order.

**16.5 Logging, tracking, and W&B plumbing.** `tracking.py` and the three
`configs/tracking/*.yaml` exist; W&B **online** mode is **NOT TESTED** (no API key).
There are no rendered charts in the repository (`HOW_TO_RUN.md` §8: "There are no
plots"). Nothing here is interview material.

**16.6 SLURM / preemption requeue.** `preemption.py` is implemented and SIGUSR1 was
verified, but the **requeue path is NOT TESTED** — no working scheduler exists.
Know the design (handler sets a boolean, nothing more) and stop.

**16.7 PPO implementation detail.** It is **CONCEPTUAL ONLY**. Study it to the depth
of "explain the objective, name every model, justify choosing DPO instead". Do not
study GAE hyperparameters or PPO engineering tricks — you cannot back any of it with
this project and it invites questions you cannot answer from your own work.

**16.8 Symmetric vs asymmetric quantization, at length.** It is listed as a study
target (`docs/FINAL_STUDY_HANDOFF.md` §4 Q8) and is **not measured anywhere in
AlignLab**. Learn it as external theory to interview depth and **label it as
external** when you answer.

**16.9 Reading `INTERVIEW_DEFENSE/` answers early.** This is the most expensive
mistake available to you. It converts twenty-five active-recall exercises into
passive reading and makes you feel ready when you are not.

**16.10 Memorising `HOW_TO_RUN.md` commands.** 73 KB of operational detail. You need
§7 (interpreting results), §9 (reading tables), §10 (confounds) and §18
(interpretation principles). The rest is reference.

**16.11 Chasing the E20 experiment.** It is **DEFERRED** and requires GPU access.
Its interview value is entirely in *being able to specify it precisely* — the
variant, the cost (~18 minutes), and what each outcome would mean. That costs you
five minutes of thought, not a run.

---

## 17. Active-recall checkpoint bank

For **every** major concept, you must eventually be able to answer this fixed set
of twelve questions without notes. Copy the block, fill it in yourself, and keep
your answers in the `STUDY_WITH_CLAUDE/` Checkpoint files.

```
CONCEPT: ______________________

 1. What is it?
 2. Why do we need it?
 3. What is the mathematical formulation?
 4. What are the tensor dimensions?
 5. How is it implemented HERE (file, class, function)?
 6. Why was it implemented this way rather than another?
 7. Which AlignLab experiment tested it (real ID)?
 8. What happened (real numbers)?
 9. What does that result support?
10. What does it NOT prove?
11. What limitation remains?
12. How would I debug it if it broke?
```

### Progress checklist

Tick only when you have done the thing **without notes**. Do not tick "read it".

**Phase 1 — engineering**
- [ ] I can state what a run manifest records and why each field is there
- [ ] I can define Tier A, B and C and say which one this project actually claims where
- [ ] I can derive checkpoint size and reconcile it with 2.886 GiB / 0.027 GiB
- [ ] I can tell the `configure_hf_cache` bug story, including the 18.34 GB
- [ ] I can say what a passing checkpoint round-trip test does **not** prove

**Phase 2 — transformers**
- [ ] I can derive attention on paper, cold
- [ ] I can narrate every shape from X to output in MHA, without pausing
- [ ] I can give the variance argument for √d_k and quote E1's four numbers
- [ ] I can explain E2 and the lesson AlignLab took from it
- [ ] I can derive the KV-cache byte formula and reproduce 100% → 6.25%
- [ ] I can explain why E4's memory result transfers and its latency result does not
- [ ] I can derive sinusoidal PE
- [ ] I can explain RoPE mathematically and intuitively, and quote E5's 2.384e-07
- [ ] I can explain why E6 is a guard and why a guard that cannot fail is worthless
- [ ] I can write LayerNorm and RMSNorm and explain the +0.995010 mean
- [ ] I can write SwiGLU and account for Qwen's 5.83× width ratio
- [ ] I can draw the decoder-only stack from memory, including the tied `lm_head`
- [ ] I can explain the KV cache and why the GPU speedup was only 1.05×
- [ ] I can tell the `top_p_filter` bug story and say why five tests missed it
- [ ] I can reproduce 1,543,714,304 and explain E11's 57,344

**Phase 3 — SFT**
- [ ] I can say which tokens contribute to the loss and why
- [ ] I can work the logits/labels shift by hand and explain E12's off-by-one
- [ ] I can explain why H5 was DISPROVED and what it changed in the code
- [ ] I can tell the tokenization-boundary bug and the stale-cache bug
- [ ] I can quote E13 and E14 and say which is the stronger evidence, and why
- [ ] I can answer the "0.8 vs 2.1 loss" question correctly

**Phase 4 — PEFT**
- [ ] I can derive 4,358,144 module by module, live
- [ ] I can explain B=0 / A random, write both gradients, and name both failure modes
- [ ] I can derive PCA from variance maximisation
- [ ] I can state the PCA ↔ SVD relationship and why ΔW's SVD is not PCA
- [ ] I can state E17's result exactly and explain why it does not invalidate LoRA
- [ ] I can compute the break-even rank and explain why r≈730 saves almost nothing
- [ ] I can explain why a same-norm random control was necessary
- [ ] I can explain the bf16 merge failure and its mechanism
- [ ] I can state E19's controls, its LR confound, and refuse the "QLoRA is faster" claim
- [ ] I can state E20 using "consistent with", and specify the deferred decisive run

**Phase 5 — preference foundations**
- [ ] I can say, unprompted, that PPO and reward-model RLHF are NOT IMPLEMENTED here
- [ ] I can derive Bradley-Terry and its loss
- [ ] I can explain the KL constraint, its direction, and the role of the reference model
- [ ] I can quote E21's 11.9% ties and 56.5% length gap, and predict their consequence
- [ ] I can explain E22 H4's disproof and why it reframes all of Phase 6
- [ ] I can tell the k3-negativity story correctly

**Phase 6 — DPO**
- [ ] I can derive DPO cold, showing exactly where `log Z(x)` cancels
- [ ] I can explain why the loss is `ln 2` at init and why that is a useful check
- [ ] I can explain β and why the sweep could not distinguish β values
- [ ] I can reproduce −29.47 = −31.90 + 2.43 and interpret the positive residual
- [ ] I can state the null in the project's own words, and never as "DPO doesn't work"
- [ ] I can quantify the shortfall two ways (728×, ~250×)
- [ ] I can explain how the null was checked not to be a bug
- [ ] I can explain why the 10× LR run is labelled POST-HOC and what it adds

**Phase 7 — evaluation**
- [ ] I can explain why perplexity alone did not establish better behaviour
- [ ] I can explain Wilson intervals and why 3 of 24 comparisons were resolvable
- [ ] I can recite the length-attribution table's **base-model row** from memory
- [ ] I can describe the two-order judge protocol and what happens to a flipped pair
- [ ] I can say exactly what the judge did and did not establish
- [ ] I can explain why no aggregate score exists and how that is enforced
- [ ] I can name the in-distribution problem before being asked

**Phase 8 — defence**
- [ ] I can explain AlignLab end to end in two minutes
- [ ] I can name all eight findings in `docs/PROJECT_NARRATIVE.md`
- [ ] I can name five disproved hypotheses and what each one bought
- [ ] I can recite what the project does NOT claim
- [ ] I can defend the deliberately preserved wrong predictions
- [ ] I can name the three things I would do next, in priority order

---

## 18. Final mastery test

### What "I have learned AlignLab" means

You have learned it when, for any number in the repository, you can say **which
experiment produced it, what it supports, what it does not support, and what you
would run next** — without notes, and without strengthening any claim.

### A. Architecture
- [ ] Draw a decoder-only transformer from memory, unprompted
- [ ] Name every matrix in a Qwen2.5-1.5B block with its shape
- [ ] Explain why `lm_head` is tied to the embedding and what that caused downstream

### B. Mathematics
- [ ] Attention, from X to output
- [ ] The √d_k variance argument
- [ ] Sinusoidal PE
- [ ] RoPE's relative-position property
- [ ] KV-cache bytes
- [ ] LoRA parameter count → 4,358,144
- [ ] Break-even rank
- [ ] SVD energy with squared singular values
- [ ] PCA via the Lagrangian
- [ ] Bradley-Terry → its loss
- [ ] DPO from the KL-constrained optimum
- [ ] Length attribution of a SUM gap
- [ ] Wilson interval motivation

### C. PyTorch
- [ ] `view` / `reshape` / `transpose` / `permute` and when `.contiguous()` is needed
- [ ] The head split and merge as pure reshapes
- [ ] Broadcasting `[T,T]` over `[B,H,T,T]`
- [ ] `log_softmax` + `gather`, and why not `log(softmax(...))`
- [ ] `ignore_index=-100` and the label shift
- [ ] Freezing, `requires_grad_`, and where gradients flow in LoRA
- [ ] Buffers vs parameters, persistent vs non-persistent

### D. Transformer internals
- [ ] MHA / MQA / GQA parameters and cache sizes
- [ ] RMSNorm vs LayerNorm, with E7's numbers
- [ ] SwiGLU and the 5.83× ratio
- [ ] KV cache correctness and the position offset
- [ ] The five decoding strategies, and where the top-p bug was

### E. SFT
- [ ] Loss masking, end to end, with the shift
- [ ] The three pre-flight audits and why they exist
- [ ] E12's five hypotheses and the one that failed
- [ ] Both Phase 3 bugs, told as stories

### F. PEFT
- [ ] LoRA equation, init, scaling, merging
- [ ] The target set, and the `lm_head` exclusion and its consequence
- [ ] E17 stated exactly, plus the "suffice vs be" defence
- [ ] Quantization: storage vs arithmetic; NF4; the bf16 merge failure
- [ ] E19's controls, LR confound, and the metric-disagreement finding
- [ ] E20 as NOT CONFIRMED, with the decisive experiment specified

### G. Preference learning
- [ ] The RLHF pipeline (conceptual), named honestly as not implemented
- [ ] Bradley-Terry, KL, the reference model
- [ ] E21's length warning and E22's below-chance baseline

### H. DPO
- [ ] The derivation, β, the preregistration
- [ ] The null, quantified two ways
- [ ] The POST-HOC run and its correct status
- [ ] Why "DPO changed the weights" is not "DPO improved the model"

### I. Evaluation
- [ ] Region-scoped perplexity, stop-token rate, distinct-2
- [ ] Wilson intervals and resolvability
- [ ] Length attribution across all five models, including the base row
- [ ] The judge protocol and its measured limitations
- [ ] Why no aggregate score exists

### J. Reproducibility
- [ ] Tiers A/B/C, used precisely
- [ ] Manifests, fingerprints, the provenance auditor and its 17/18
- [ ] What is NOT TESTED, said plainly

### K. Experimental reasoning
- [ ] Preregistration and what it protects against
- [ ] Why a control is required before claiming a spectrum decays
- [ ] Why a guard that cannot fail is worthless
- [ ] Why an experiment found a bug five unit tests missed
- [ ] Why a disproved hypothesis is preserved verbatim

### L. Interview defence
- [ ] The eight findings of `docs/PROJECT_NARRATIVE.md`
- [ ] The nine claims in `docs/FINAL_STUDY_HANDOFF.md` §8 that you must never make
- [ ] The 18 items of `docs/LIMITATIONS.md`, at least by category
- [ ] The five things that would most change the picture

### Mock-interview progression

Run these in order. Do not advance until the previous round is comfortable.

**ROUND 1 — the project in 2 minutes.** Scope boundary (post-training a 1.5B base
model, not pretraining), what was built, and the two or three findings that matter.
*Pass condition:* no jargon dumping, no metric without its caveat, and you finish
inside 2 minutes.

**ROUND 2 — the project in 10 minutes.** Phase by phase, with real run names and
real numbers. *Pass condition:* you name at least three disproved hypotheses without
being asked, and you never overstate a result.

**ROUND 3 — deep technical questioning.** Eight questions drawn at random from
`INTERVIEW_DEFENSE/`, five minutes each, closed book. *Pass condition:* six of eight
answered to a level you would accept from a colleague.

**ROUND 4 — mathematical derivations.** On paper, cold: attention; sinusoidal PE;
the LoRA parameter count; PCA; the DPO objective including the `log Z(x)`
cancellation. *Pass condition:* all five, no notes, no algebra errors that change
the answer.

**ROUND 5 — code walkthrough.** Open `attention.py`, `lora.py`, `masking.py`,
`dpo.py`, `evals/metrics.py` and narrate what each key function does and why it is
written that way. *Pass condition:* you can point to the line implementing each
equation you derived in Round 4, and name the test that pins it.

**ROUND 6 — experimental skepticism.** Someone attacks each headline result in
turn. *Pass condition:* for every result you can state what it supports, what it
does not, and what you would run next — and you concede the in-distribution problem
before being pushed to.

**ROUND 7 — failure and bug discussion.** Tell six bug stories: the causal-mask
leak, the top-p permutation, the 57,344 parameter count, the tokenization boundary
plus stale cache, the bf16 merge, and the HF-cache no-op. *Pass condition:* each
story has a detection mechanism, a fix, and a consequence for the design.

**ROUND 8 — research defence.** "What is the most interesting thing here, why is it
interesting, what is the alternative explanation, and how would you settle it?"
*Pass condition:* you can run the full E17 → E20 argument — the low-rank explanation
failed, reachability is the better hypothesis, it is **NOT CONFIRMED**, and here is
the ~18-minute experiment that would settle it — without once saying "because".

---

## 19. How to use this file

**Every morning, in this order:**

1. Open `STUDY_SCHEDULE.md` and read today's row. That is *what* to study.
2. Open this file at the matching phase section. That is *how*.
3. Read §5.x/§6.x/… subsection **1** (purpose) and **2** (prerequisites). If a
   prerequisite is missing, fix that first — it will cost you the whole day
   otherwise.
4. Pick the first **MUST STUDY** concept you have not ticked in §17.
5. Run the 13-step study loop (§4) on it. Do not skip steps 2, 4 or 12.
6. Repeat until the day's MUST items are ticked. Then, and only then, SHOULD items.
7. End the day by attempting the phase's **"Can I defend this?"** section aloud,
   closed book. Mark each question **passed / shaky / failed**.
8. Carry every **shaky** and **failed** item into tomorrow morning, before new
   material.

**Every evening, five minutes:** pick one interview chain from §13 and run it out
loud, timed. Chains are cheap and they are what an interview actually feels like.

**When you have 30 minutes and no energy:** read one committed JSON artefact from
`docs/phase*/` and find one number in it yourself. Low effort, high retention.

**When you feel ready:** run a mock round from §18. If Round 3 is uncomfortable, you
are not ready, regardless of how much you have read.

**Rules that override everything else:**

- **Never fill in an explain-back with someone else's words.** The Checkpoints in
  `STUDY_WITH_CLAUDE/` are yours; this file deliberately leaves every one of them
  blank.
- **Never strengthen a finding.** Where the repository says "consistent with", say
  "consistent with". Where it says NOT CONFIRMED, say NOT CONFIRMED.
- **Never quote a number without its set.** 47.2% and 46.7% are different sets.
  0.6547 and 0.2044 are mean and median.
- **Always name the layer** when you use the four-way distinction of §1:
  implemented, measured, concluded, or what you personally need to defend.

---

**Related:** [[study-schedule]] · [[final-study-handoff]] · [[experiment-registry]]
· [[project-narrative]] · [[limitations]] · [[how-to-run]]
