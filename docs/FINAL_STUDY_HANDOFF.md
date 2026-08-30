# AlignLab — Final Study Handoff

**Not a new phase.** Implementation and engineering are complete. This document
exists to tell you how to *study and defend* the finished project later.

Everything here was verified against the repository at commit
`6c11e75591b34104e99e7134304ca8cf9eda26af` on branch `phase-7-evaluation`.
Nothing is inferred.

**Your explain-back sections remain untouched and unfilled.** See §6.

---

## 1. Verified final state

| Check | Result |
|---|---|
| branch | `phase-7-evaluation` |
| HEAD | `6c11e75591b34104e99e7134304ca8cf9eda26af` |
| LOCAL = GITHUB = SERVER | **yes** — all three at `6c11e75` |
| working tree clean | **yes** — 0 dirty files locally and on the server |
| unpushed commits | **0** |
| history rewritten | **no** — 92 commits, 0 force-pushes, 0 rebases |
| phase reports 1–8 | **all present** (Phase 1 is split 1A/1B) |
| `HOW_TO_RUN.md` | present |
| `README.md` | present |
| `docs/EXPERIMENT_REGISTRY.md` | present |
| `docs/PROJECT_NARRATIVE.md` | present |
| `docs/LIMITATIONS.md` | present |
| five documentation folders | present — 16 / 8 / 7 / 7 / 8 markdown files |
| tests | **607 passed / 3 skipped** local · **609 passed / 1 skipped** server |

### One discrepancy, reconciled

`docs/phase8/PHASE_8_REPORT.md` §6 and §13 say **"31 files carry unfilled
markers."** The correct current figures are in §6 below: **42 files, 107 marker
occurrences.**

The Phase 8 number is not wrong so much as narrower, and it has since moved:

- it counted **files, not occurrences**;
- it searched **four folders**, excluding `docs/` and `CODE_EXPLANATION/`;
- it used a narrower pattern that missed bare `USER CHECKPOINT` headers;
- it was measured **before `INTERVIEW_DEFENSE/phase8_project_defence.md` was
  created** in that same commit.

Re-running that exact command today returns **32** — the original 31 plus the
Phase 8 defence file. **No marker was filled and none was removed.** The Phase 8
report is left as written rather than silently edited; this note is the
correction.

---

## 2. Study roadmap

Ordering follows the requested tiers. **One adjustment, for a repository-backed
reason:** *tensor dimension tracking* (Tier 1 #5) is listed **first**, because
every other Tier-1 topic is checked by it, `PYTORCH_CONCEPTS/pytorch-tensor-manipulation.md`
already exists with executed examples, and the project's own recurring bug class
was mis-indexed shifts and mis-counted positions (E12's decomposition, and the
Phase 5 test that mis-indexed the leading `IGNORE_INDEX`).

Everything else keeps the requested order.

### TIER 1 — highest interview ROI

| # | Topic | Phase | Source | Study doc | Experiments | Key result | Interview | Checkpoint |
|---|---|---|---|---|---|---|---|---|
| 0 | **tensor dimension tracking** | 2–6 | `models/shapes.py`, `masking.py`, `logprobs.py` | `PYTORCH_CONCEPTS/pytorch-tensor-manipulation.md`, `pytorch-loss-masking-and-shifts.md` | E12 | decomposition agreed to **1.6e-07** after the shift fix | `phase3_sft.md` | 13–15 |
| 1 | **LoRA** | 4 | `src/alignlab/lora.py` | `STUDY_WITH_CLAUDE/phase4/01_lora_svd_and_pca.md` | E18, E19 | 4,358,144 trainable = **0.28%**; matches `peft` in fp32 | `INTERVIEW_DEFENSE/phase4_peft.md` | **16** |
| 2 | **SVD & LoRA rank selection** | 4 | `lora.py`, `scripts/experiments/e17_svd_rank_selection.py` | same | **E17** | 90% energy at rank **~730/1536**; r=16 captures **10.69%** → **H2 DISPROVED** | `phase4_peft.md` | 16 |
| 3 | **PCA ↔ SVD** | 4 | — | same | E17 | explained energy uses **squared** singular values (H1 HOLDS) | `phase4_peft.md` | 16 |
| 4 | **multi-head / self-attention from scratch** | 2 | `models/attention.py`, `attention_numpy.py`, `attention_pure.py` | `STUDY_WITH_CLAUDE/phase2/01_attention_from_first_principles.md` | E1, E2, E3 | 3 implementations agree to **1.1e-16 – 4.4e-16** | `INTERVIEW_DEFENSE/phase2_transformers.md` | **1, 3** |
| 5 | **quantization mechanics** | 4 | `src/alignlab/peft_setup.py` | `PYTORCH_CONCEPTS/pytorch-quantization-and-lowrank.md` | E15 | NF4 is **storage**, not arithmetic; `Params4bit.numel()` returns **bytes** | `phase4_peft.md` | 16 |
| 6 | **LoRA vs QLoRA** | 4 | `peft_setup.py`, `configs/peft/{lora,qlora}.yaml` | `STUDY_WITH_CLAUDE/phase4/01_lora_svd_and_pca.md` | **E19** | QLoRA **3,095,107,584 B** vs LoRA **5,073,415,168 B** peak VRAM | `phase4_peft.md` | 16 |
| 7 | **positional encoding / RoPE** | 2 | `models/positional.py` | `STUDY_WITH_CLAUDE/phase2/03_positional_encoding.md` | E5, E6 | fixed-gap spread **2.384e-07**; the offset guard **does** change output | `phase2_transformers.md` | **4, 5** |
| 8 | **GQA / MQA / MHA** | 2 | `models/attention.py` | `STUDY_WITH_CLAUDE/phase2/02_multihead_and_variants.md` | E4 | KV cache 100% → **6.25%**; **GPU latency flat** | `phase2_transformers.md` | **2, 6** |

### TIER 2

| # | Topic | Phase | Source | Study doc | Experiments | Key result | Checkpoint |
|---|---|---|---|---|---|---|---|
| 10 | decoder-only Transformer | 2 | `models/transformer.py` | `phase2/05_decoder_only_and_generation.md` | tiny LM | 334,080 params; loss 2.87 → 0.27 vs ln(17)=2.8332 | **7, 8** |
| 11 | causal masking | 2 | `models/attention.py` | `phase2/01_attention_from_first_principles.md` | **E2** | loss **0.2907 vs 0.0088 — 33.2× lower when leaking** | 1, 3 |
| 12 | KV caching | 2 | `models/kv_cache.py` | `phase2/06_kv_cache.md` | **E8** | **2.96×** CPU; **1.05×** GPU (overhead-bound) | **9** |
| 13 | RMSNorm vs LayerNorm | 2 | `models/normalization.py` | `phase2/04_normalization_and_ffn.md` | E7 | **0.28×** cost CPU; RMS mean **+0.995010** vs LN −0.000000 | **11, 12** |
| 14 | SwiGLU | 2 | `models/feedforward.py` | `phase2/04_normalization_and_ffn.md` | — | Qwen `d_ff/d_model` = **5.83×** | 11, 12 |
| 15 | autoregressive generation | 2 | `models/generation.py` | `phase2/05_decoder_only_and_generation.md` | E9 | — | 7, 8 |
| 16 | decoding strategies | 2 | `models/generation.py` | same | **E9** | greedy distinct-4 0.257 → top-p 0.9 **0.566**; **found a real top-p bug** | 7, 8 |
| 17 | SFT loss masking | 3 | `masking.py`, `sft.py` | `phase3/01_sft_and_loss_masking.md` | **E12** | prompt CE **6.3314** vs completion **3.6295** → **H5 DISPROVED** | **13, 14, 15** |

Also in Tier 2 territory and worth reading: `phase2/07_flash_attention_concepts.md`
(Checkpoint **10**) — E10 measured **56× faster, 147× less memory** at T=4096.

### TIER 3

| # | Topic | Phase | Source | Study doc | Experiments | Key result | Checkpoint |
|---|---|---|---|---|---|---|---|
| 18–20 | RLHF · reward models · **PPO** | 5 | **none — CONCEPTUAL, NOT IMPLEMENTED** | `phase5/01_rlhf_ppo_and_dpo.md` | — | study the *reasoning* for choosing DPO here | **17, 18, 19** |
| 21 | KL regularization | 5 | `src/alignlab/logprobs.py` | same | E22 | KL(SFT‖base) **0.6547** vs KL(LoRA) **0.6233**; `k3` non-negativity **falsified in float32** (11/200,000, min −2.98e-08) | 17–19 |
| 22 | Bradley-Terry | 5 | `src/alignlab/preference.py` | same | E21 | 11.9% ties; chosen **56.5% longer** | 17–19 |
| 23 | DPO | 6 | `dpo.py`, `dpo_train.py` | `phase6/01_dpo_theory_and_results.md` | β sweep, E23 | loss at init exactly **ln 2 = 0.6931471805599453** | **20, 21, 22** |
| 24 | DPO vs PPO | 5–6 | — | `phase5/01_rlhf_ppo_and_dpo.md` | — | no reward model, no rollout loop, no value network | 17–22 |
| 25 | DPO **beta** | 6 | `configs/dpo/default.yaml` | `phase6/01_dpo_theory_and_results.md` | **β sweep** | **pre-registered**; all three arms gave **86/184** | 20–22 |
| 26 | preference evaluation | 5–7 | `preference.py`, `evals/metrics.py` | `phase7/01_evaluation_theory.md` | E22, E23 | **SUM 47.2% vs MEAN 58.3%** | 20–25 |

### TIER 4

| # | Topic | Phase | Source | Study doc | Key result | Checkpoint |
|---|---|---|---|---|---|---|
| 27 | evaluation methodology | 7 | `evals/metrics.py`, `report.py` | `phase7/01_evaluation_theory.md` | **3 of 24** comparisons resolvable; **no aggregate score exists** | **23, 24, 25** |
| 28 | LLM-as-judge | 7 | `evals/judge.py` | same | position bias **33.3%**; win rate **[0.301, 0.954]** | 23–25 |
| 29 | reproducibility | 1, 8 | `seeding.py`, `manifest.py`, `provenance.py` | `PYTORCH_CONCEPTS/pytorch-rng-and-state.md` | **17/18** artefacts complete; Tier A/B/C distinction | — |
| 30 | experiment design | 3–6 | `scripts/experiments/` | `docs/EXPERIMENT_REGISTRY.md` | **pre-registration**; 5 hypotheses disproved | — |
| 31 | ML systems engineering | 1, 8 | `paths.py`, `storage.py`, `preemption.py` | `CODE_EXPLANATION/phase1/`, `phase8/` | storage guard; SIGUSR1 verified; the HF-cache no-op bug | — |

---

## 3. Findings to master

Each is stated as the reports state it. **Do not strengthen any of these.**

### F1 — Attention scaling (E1)
**What happened:** with `d_k` swept 4→1024 unscaled, logit std went 1.85→31.50
(tracking √d_k), max prob 0.42→0.992, entropy 1.78→0.028, Jacobian mass
0.708→0.014. Scaled: all four flat.
**Why it matters:** turns a hand-wave into a curve, and shows the vanishing
gradient in the Jacobian rather than by assertion.
**Evidence:** `docs/phase2/PHASE_2_REPORT.md` §3.
**Does not prove:** anything about training dynamics at scale.

### F2 — Causal-mask leakage (E2)
**What happened:** training loss **0.2907 masked vs 0.0088 unmasked — 33.2×
lower when leaking.**
**Why it matters:** the founding lesson of the project — a dramatically better
loss can mean a broken experiment. It is why SFT has three pre-flight audits
that refuse to start.
**Evidence:** `docs/phase2/PHASE_2_REPORT.md`.
**Does not prove:** that all large loss drops are bugs.

### F3 — Hardware-dependent timing (E4, E8)
**What happened:** GQA cut the KV cache to **6.25%** and CPU latency
160.4→117.4 ms, but **GPU latency was flat** (0.86→0.81 ms). The KV cache gave
**2.96×** on CPU and **1.05×** on GPU.
**Why it matters:** the *memory* results are architectural and transfer; the
*latency* results did not. The GPU figure is **overhead-bound at this model
size**, not a refutation.
**Evidence:** `docs/phase2/gpu_experiment_results_2026-08-29.txt`.
**Does not prove:** that GQA/KV caching do not help — only that they did not
help *latency* here.

### F4 — E17: the SVD/rank finding
**What happened:** SVD of the real full-fine-tuning ΔW. **H1 HOLDS** (explained
energy uses squared singular values). **H2 — "ΔW is approximately low-rank" —
DISPROVED**: 90% of energy needs rank **~730 of 1536**, while r=16 captures
**10.69%**. `relative_update` ≈ **0.0013**.
**Why it matters:** **this is the most important negative result in the
project.** The popular explanation of LoRA is not supported by the update
actually measured here.
**Evidence:** `docs/phase4/e17_svd_rank_selection.json`, `configs/peft/lora.yaml`.
**Does not prove:** that LoRA does not work — it did. LoRA needs a low-rank
update to **suffice**, not full fine-tuning's update to **be** low-rank.

### F5 — Phase 3 SFT result
**What happened:** completion perplexity **8.541 → 7.199 (−15.7%)** over 38,831
completion tokens (E13); stop-token emission **0/4 → 4/4**. Re-measured in Phase
7 on a different subset: **8.824 → 7.398 (−16.2%)**, **0/6 → 6/6**.
**Why it matters:** the project's one large behavioural win, and the only
comparison whose Wilson intervals are disjoint.
**Evidence:** `docs/phase3/e13_before_after.json`,
`docs/phase7/eval-full-001_dashboard.json`.
**Does not prove:** that SFT made a better assistant. The eval split is the
training distribution.

### F6 — Phase 4 LoRA/QLoRA comparison (E19)
**What happened:** 4,358,144 trainable parameters (**0.28%**); QLoRA peak VRAM
**3,095,107,584 B** vs LoRA **5,073,415,168 B**; eval loss 1.9893 (full SFT) vs
2.0402 / 2.0718 / 2.0769. **H1, H2, H4 HOLD; H3 DISPROVED** — LoRA did not match
full fine-tuning. Runtimes 1167.876 / 1054.845 / 1053.122 / 1044.005 s.
Separately: **merging in bf16 is 12,460× less exact than fp32.**
**Why it matters:** the savings are real and the quality cost is small — and the
LR confound was measured (0.032 nats against a 0.051-nat gap), so reporting only
the matched arm would have overstated LoRA's cost by about half.
**Evidence:** `docs/phase4/e19_peft_comparison.json`.
**Does not prove:** that QLoRA is *faster* — full SFT's peak VRAM is **NOT
RECORDED** in E19, and the runtime differences are small and single-seeded.

### F7 — E20: the stopping-token finding
**What happened:** all PEFT arms emitted `<|im_end|>` **0/4** (Phase 4) and
**0/6** (Phase 7) while full SFT emitted it 4/4 and 6/6. `lm_head` is **tied to
the embedding** and outside LoRA's target set; full fine-tuning moved that
matrix by relative **0.0136**, the largest relative change measured.
**Why it matters:** LoRA's ceiling may be set by **which matrices it can
reach**, not only by rank — and perplexity separated these models by only 3.5%.
**Evidence:** `docs/phase4/e20_stop_token_gap.json`; Phase 7 generations show
the models falling back into the chat template (`You are a helpful assistant.`
repeated to the cap).
**Does not prove:** the causal claim. **Status: NOT CONFIRMED.** The decisive
MLP-target / untied-head run (~18 min) was never executed.

### F8 — Phase 5 SUM vs MEAN (E22)
**What happened:** identical models and data gave **SUM 47.2%** (below chance)
and **MEAN 58.3%** — an **11.1-point swing**. **H4 DISPROVED.** Also: H3 holds
**exactly** (`0.000e+00` implicit reward when π = π_ref); H1's criterion is
recorded **UNTESTABLE AS STATED**.
**Why it matters:** the objective DPO would optimise did not start above chance,
which reframes all of Phase 6.
**Evidence:** `docs/phase5/e22_reference_and_kl.json`.
**Does not prove:** that the data is bad — per token, chosen is genuinely better.

### F9 — Phase 6 DPO null result
**What happened:** SUM accuracy **0.4674 = 86/184 for the SFT baseline and all
three β arms — byte-identical.** KL from reference ≈ **0.0008** against SFT's
**0.2044**.
**Why it matters:** a pre-registered negative result, with the mechanism
quantified rather than excused.
**Evidence:** `docs/phase6/e23_dpo_evaluation.json`,
`docs/phase6/BETA_PREREGISTRATION.md` (committed before any DPO code existed).
**Does not prove:** **"DPO does not work."** It is a statement about this
configuration and this budget. It also cannot distinguish β values — the budget
was too small for that.

### F10 — Phase 6 length effect
**What happened:** SUM gap **−29.47 nats**, explained by length **−31.90**,
residual **+2.43** — chosen is *better* per token. Flipping the average pair
needs ~29 nats; the run moved **0.04** (**728× short**). The **POST-HOC** 10× LR
run moved it **0.70** (**42× short**) and produced ~**30% longer** output.
**Why it matters:** the metric was measuring length.
**Evidence:** `docs/phase6/PHASE_6_REPORT.md` §7–8.
**Does not prove:** that length is the only obstacle — the KL shortfall is a
separate, independent problem.

### F11 — Phase 7 evaluation findings
**What happened:** **of 24 pairwise comparisons, 3 were resolvable — all
stop-token rates.** The length decomposition run on all five models gave a
**positive residual in every row, including the untrained base model**. DPO
changed the weights (perplexity in the 4th decimal, 4 of 6 generations differ)
and no aggregate metric.
**Why it matters:** the SUM/MEAN inversion is a property of the **metric**, not
of any training stage — the untrained base proves that. And the Phase 6 null
replicated independently.
**Evidence:** `docs/phase7/eval-full-001_dashboard.{json,txt}`,
`docs/phase7/EVAL_RESULTS.md`.
**Does not prove:** general quality. Everything is in-distribution.

### F12 — LLM-as-judge limitations
**What happened:** **33.3%** of base-vs-SFT pairs flipped on order swap, leaving
4 decided verdicts; SFT won 3, interval **[0.301, 0.954]** — includes 0.5.
Unparsed replies: 0. An unplanned control passed: `TIE` on two byte-identical
pairs, in both orders.
**Why it matters:** base emits `-unstyled` 128 times and SFT writes a correct
email, and the judge still could not resolve it. A one-order judge would have
returned a clean 6-pair win rate with a third of its verdicts decided by
position.
**Evidence:** `docs/phase7/eval-full-001_dashboard.json` → `judge_results`.
For calibration: MT-Bench Table 2 reports GPT-4 at **65.0%** self-consistency,
Claude-v1 at **23.8%** (`LEARNING_RESOURCES/phase7_resources.md`,
**PARTIALLY INSPECTED**).
**Does not prove:** anything about the judge's agreement with humans — **no
human study was run**, and the judge is **same-family** and pinned to `main`,
not a SHA.

### F13 — Phase 8 engineering / reproducibility findings
**What happened:** `configure_hf_cache` was a **no-op in every entrypoint**,
because `huggingface_hub` freezes `HF_HUB_CACHE` at import — costing **18.34 GB**
of duplicate weights. `dataset_fingerprint` was **null in all five** Phase 7
provenance records. The provenance auditor now reports **17/18** artefacts
complete, the one gap being that known bug, with the post-fix run clean.
**Why it matters:** a documented guarantee that was never actually enforced.
The auditor exists so the claim is checked rather than asserted.
**Evidence:** `docs/phase8/PHASE_8_REPORT.md`, `docs/phase8/CLEANUP_RECORD.md`.
**Does not prove:** that provenance is complete for history — the
`eval-full-001` gap is permanent and **was not back-filled**.

---

## 4. Questions I must eventually be able to answer

**Study targets. No answers are supplied here** — deeper preparation lives in
`INTERVIEW_DEFENSE/`, and your explain-back sections are deliberately empty.

**Attention and architecture** (`INTERVIEW_DEFENSE/phase2_transformers.md`)
1. Explain attention from scratch.
2. Code masked self-attention and track **every tensor shape**.
3. Explain MHA vs MQA vs GQA — parameters, KV cache, and what did *not* change.
4. Derive sinusoidal positional encoding.
5. Explain RoPE mathematically, and why it depends only on relative position.
6. Explain KV caching — what is cached, what is recomputed, and why the GPU
   speedup here was 1.05×.

**Quantization and low-rank** (`INTERVIEW_DEFENSE/phase4_peft.md`)
7. Explain quantization.
8. Explain symmetric vs asymmetric quantization.
9. Explain BF16 / FP16 / FP32 — and why bf16 merging was 12,460× less exact.
10. Explain LoRA mathematically.
11. **Derive** the LoRA parameter count, and reproduce **4,358,144**.
12. Explain how SVD *could* be used for rank selection.
13. **Explain why E17 complicates the naive SVD explanation of LoRA.**
14. Derive PCA from scratch.
15. Explain the PCA ↔ SVD relationship.
16. Explain why QLoRA saves memory — and why that is a storage claim.

**Training objectives** (`INTERVIEW_DEFENSE/phase3_sft.md`, `phase5_rlhf_dpo.md`, `phase6_dpo.md`)
17. Explain SFT loss masking, and the shift between logits and labels.
18. Explain RLHF end to end.
19. Explain PPO.
20. Explain DPO.
21. **Derive the DPO objective** from the KL-constrained optimum, showing where
    `log Z(x)` cancels.
22. Explain the role of the reference model, and why it must be frozen and in
    `eval()`.
23. Explain β — what it controls and why the sweep could not distinguish values.
24. Explain DPO vs PPO, and justify the choice made here.

**Evaluation** (`INTERVIEW_DEFENSE/phase7_evaluation.md`, `phase8_project_defence.md`)
25. Explain how AlignLab evaluates whether a model improved.
26. Explain why perplexity alone is insufficient — with the Phase 4/7 numbers.
27. Explain why LLM-as-judge is not ground truth.
28. Explain the major failures and corrections in AlignLab, and what each one
    changed about the design.

---

## 5. Explain-back inventory — VERIFIED COUNTS

**Nothing below has been filled in. All remain: DEFERRED — USER EXPLAIN-BACK
REQUIRED.**

Counted by matching `USER EXPLAIN-BACK REQUIRED`, `USER MUST WRITE THIS` or
`USER CHECKPOINT` across all documentation folders **and** `docs/`:

- **42 files** contain at least one marker
- **107 total marker occurrences**
- **25 numbered Checkpoints (1–25)**

| Location | Files | Occurrences |
|---|---:|---:|
| `STUDY_WITH_CLAUDE/` | 16 | 46 |
| `LEARNING_RESOURCES/` | 7 | 29 |
| `INTERVIEW_DEFENSE/` | 8 | 16 |
| `PYTORCH_CONCEPTS/` | 5 | 10 |
| `docs/phase*/` (phase reports) | 6 | 6 |
| **total** | **42** | **107** |

> **If you re-count later, expect 43 / 111.** This handoff file quotes the
> marker strings while describing them, so a naive `grep` over `docs/` now
> matches it too. Exclude `docs/FINAL_STUDY_HANDOFF.md` to reproduce the
> figures above.

`LEARNING_RESOURCES/phase2_resources.md` alone holds **18** — one "Own Words"
entry per external resource. `CODE_EXPLANATION/` contains **none** by design.

### The 25 numbered Checkpoints, by file

| Checkpoints | File | Topic |
|---|---|---|
| **1, 3** | `STUDY_WITH_CLAUDE/phase2/01_attention_from_first_principles.md` | attention, scaling, causal mask |
| **2, 6** | `phase2/02_multihead_and_variants.md` | MHA / MQA / GQA |
| **4, 5** | `phase2/03_positional_encoding.md` | sinusoidal, RoPE, ALiBi |
| **7, 8** | `phase2/05_decoder_only_and_generation.md` | decoder-only, generation |
| **9** | `phase2/06_kv_cache.md` | KV caching |
| **10** | `phase2/07_flash_attention_concepts.md` | Flash attention |
| **11, 12** | `phase2/04_normalization_and_ffn.md` | RMSNorm vs LayerNorm, SwiGLU |
| **13, 14, 15** | `phase3/01_sft_and_loss_masking.md` | SFT, loss masking |
| **16** | `phase4/01_lora_svd_and_pca.md` | LoRA, SVD, PCA |
| **17, 18, 19** | `phase5/01_rlhf_ppo_and_dpo.md` | RLHF, PPO, DPO |
| **20, 21, 22** | `phase6/01_dpo_theory_and_results.md` | DPO theory and results |
| **23, 24, 25** | `phase7/01_evaluation_theory.md` | evaluation, LLM-as-judge |

Additionally, `STUDY_WITH_CLAUDE/phase1_foundation.md` and
`phase1b_two_environments.md` carry unnumbered markers (3 each).

---

## 6. Experiment → study map

Real names and IDs only. Phases 6–7 stopped assigning E-numbers, so those rows
use run names.

| Experiment | Concept | Why run | Key result | What to learn | Does NOT establish |
|---|---|---|---|---|---|
| **E1** | attention scaling | is √d_k needed? | logit std 1.85→31.50; Jacobian 0.708→0.014 | saturation is measurable | anything at scale |
| **E2** | causal masking | what does the mask buy? | loss **33.2× lower when leaking** | a better loss can mean a bug | that all big drops are bugs |
| **E3** | 3 implementations | correctness | agree to **1.1e-16–4.4e-16** | agreement ≠ correctness | that the maths is right |
| **E4** | GQA/MQA | cost and saving | KV cache → **6.25%**; GPU latency flat | memory transfers, latency may not | that GQA never helps latency |
| **E5/E6** | RoPE | relative-position invariance | spread **2.384e-07**; offset guard live | why a guard must be able to fail | long-context extrapolation |
| **E7** | RMSNorm | is centring redundant? | **0.28×** cost; RMS mean **+0.995010** | the mean it skips is large | quality equivalence |
| **E8** | KV cache | does it flatten cost? | **2.96×** CPU, **1.05×** GPU | overhead-bound ≠ refuted | GPU behaviour at scale |
| **E9** | decoding | diversity vs quality | distinct-4 0.257→0.566 | experiments find bugs (**top-p**) | which decoder is "best" |
| **E10** | Flash attention | speed and memory | **56×**, **147×**; linear vs quadratic MiB/T | why memory scaling matters | numerics beyond 2× bf16 eps |
| **E11** | parameter arithmetic | does our formula match? | off by **57,344** (QKV biases) | check against the real checkpoint | anything about other models |
| **E12** | loss masking | is the mask right? | prompt **6.3314** vs completion **3.6295** | regions are not comparable | which region is "harder" generally |
| **E13** | SFT before/after | what did SFT change? | ppl **8.541→7.199**; stop **0/4→4/4** | behaviour beats likelihood | general quality |
| **E14** | region decomposition | where does loss live? | MEASURED | denominators matter | — |
| **E15** | bitsandbytes | does it really work? | quantised matmul **executed** | import ≠ working kernel | performance |
| **E16** | LoRA targets | which modules? | **112 of 197** `nn.Linear` matched | inspect, don't assume | that this set is optimal |
| **E17** | SVD rank selection | is ΔW low-rank? | 90% at rank **~730/1536**; r=16 = **10.69%** → **H2 DISPROVED** | the popular LoRA story failed | that LoRA doesn't work |
| **E18** | ours vs `peft` | is our LoRA right? | matches in **fp32** | fp32 agreement ≠ bf16 | bf16 correctness |
| **E19** | SFT vs LoRA vs QLoRA | what does PEFT cost? | 0.28% params; **H3 DISPROVED** | control what you can, report what you can't | that QLoRA is faster |
| **E20** | stop-token gap | why won't PEFT stop? | `lm_head` relative change **0.0136** | reachability ≠ rank | **NOT CONFIRMED** — no causal claim |
| **E21** | preference data | is it usable? | 11.9% ties; chosen **56.5% longer** | check data before optimising it | data quality |
| **E22** | reference & KL | where does the metric start? | **SUM 47.2%**, MEAN 58.3% → **H4 DISPROVED** | measure the baseline first | that the data is bad |
| **β sweep** (`dpo-beta{0.01,0.1,0.5}-sum`) | DPO | does DPO help? | **86/184 in every arm**; KL ~0.0008 | pre-registration | that DPO doesn't work |
| **E23** | SFT vs DPO eval | did anything change? | identical; **NOT resolvable** | say "unresolvable", not "+0.0%" | a β preference |
| **`diag-dpo-beta0.1-lr5e-6`** | 10× LR | is LR the blocker? | +0.70 nats, **42× short**; output +30% | label post-hoc work **POST-HOC** | anything preregistered |
| **`eval-full-001`** | full evaluation | measure all five | **3 of 24** resolvable; residual positive in **all five** rows | metrics disagree; no aggregate score | general quality |

---

## 7. Reading order

**A.** `HOW_TO_RUN.md` — §7 (interpreting results) and §9 (reading tables) are
the highest-value pages in the repository for study.
**B.** `docs/PROJECT_NARRATIVE.md` — the eight findings.
**C.** `docs/LIMITATIONS.md` — read **before** forming any conclusion.
**D.** `docs/EXPERIMENT_REGISTRY.md` — hypotheses as written *before* results.
**E.** The phase report for whatever you are studying (`docs/phase*/`).
**F.** `CODE_EXPLANATION/phase*/README.md` — how it is built.
**G.** `STUDY_WITH_CLAUDE/` — the theory, and where your Checkpoints live.
**H.** `PYTORCH_CONCEPTS/` — run the example scripts; all 7 execute.
**I.** The source file itself.
**J.** The committed artefact in `docs/phase*/` — the actual numbers.
**K.** `INTERVIEW_DEFENSE/`, ending with `phase8_project_defence.md`.

**Suggested cadence:** take one Tier-1 topic at a time, follow E→K for it, then
write its Checkpoint answer before moving on. The Checkpoints are ordered to
match Tier 1 → Tier 4 reasonably closely.

---

## 8. What NOT to claim in interviews

Verified: **none of these is asserted anywhere in the repository.** Every
occurrence found is an explicit disclaimer. Do not introduce them.

| Claim | Why it is unsupported |
|---|---|
| "LoRA works because full fine-tuning updates are low rank." | **E17 DISPROVED this** on the update measured here — 90% of energy needs rank ~730/1536. LoRA worked; the explanation did not. |
| "DPO improved the model." | SUM accuracy was **86/184 for the baseline and all three arms**, byte-identical. |
| "QLoRA is faster than LoRA." | Runtimes 1044.005 s vs 1054.845 s — a ~1% difference, **single seed, no error bar**. QLoRA's claim is **memory** (3.10 GB vs 5.07 GB), not speed. |
| "Perplexity proves the model is better." | Perplexity separated the PEFT arms from full SFT by **3.6%** while stop-token behaviour was **0/6 vs 6/6**. |
| "LLM-as-judge provides ground truth." | **33.3%** position bias; 4 decided verdicts; never validated against humans; same model family. |
| "AlignLab is production-ready." | 1.5B model, **one epoch**, 9,499 examples, single seed, entirely in-distribution evaluation. |
| "AlignLab proves the model is generally better." | **Every number is in-distribution.** No OOD set, no standard benchmark. |

Two more worth adding, from the same evidence:

| Claim | Why it is unsupported |
|---|---|
| "LoRA fails to stop **because** `lm_head` is outside the target set." | **E20 is NOT CONFIRMED.** Say "consistent with", not "because" — the decisive run was never done. |
| "GQA / KV caching make inference faster." | True for **memory**; on the A6000 latency was **flat** (E4) and the KV cache gave **1.05×** (E8). |

---

## 9. Final project snapshot

| | |
|---|---|
| **Base model** | `Qwen/Qwen2.5-1.5B` @ `8faed761d45a263340a0528343f099c05c9a4323` — **base, not Instruct**; 1,543,714,304 parameters |
| **Completed phases** | 1 (1A+1B), 2, 3, 4, 5, 6, 7, 8 — **all 8** |
| **Implemented** | attention ×3, RoPE/ALiBi/sinusoidal, RMSNorm, SwiGLU, decoder-only Transformer, KV cache, 5 decoding strategies, SFT, LoRA, QLoRA, Bradley-Terry, sequence log-probs, exact + `k3` KL, DPO loss and training loop, evaluation subsystem, provenance auditor |
| **Conceptual only** | **RLHF with a learned reward model, and PPO — NOT IMPLEMENTED** |
| **Main training runs** | `sft-qwen1p5b-noRobots-001`, `lora-r16-lr2e-4-001`, `lora-r16-lr2e-5-001`, `qlora-r16-lr2e-4-001`, `dpo-beta0.01-sum`, `dpo-beta0.1-sum`, `dpo-beta0.5-sum`, `diag-dpo-beta0.1-lr5e-6` (POST-HOC) |
| **Evaluation runs** | `eval-full-001` (5 models + judge), `eval-cachefix-001` (verification) |
| **Tests** | **607 passed / 3 skipped** local · **609 passed / 1 skipped** server |
| **Server hardware** | `csrslave` — **2 × NVIDIA RTX A6000, 49,140 MiB each**; runs used **one** (`CUDA_VISIBLE_DEVICES=0`), recorded as 47.53 GiB / `device_count: 1` in manifests |
| **Key checkpoints** | `sft-.../checkpoint-295` **8.7 G** · `sft-.../final` **2.9 G** · `dpo-beta0.1-sum/final` **2.9 G** · `lora-r16-lr2e-4-001/final` **28 M** · `qlora-r16-lr2e-4-001/final` **20 M** |
| **Storage** | `/data` at 100% capacity, **54 G free** (Phase 8 recovered 18.34 GB) |
| **Git SHA** | `6c11e75591b34104e99e7134304ca8cf9eda26af` |
| **Sync** | **LOCAL = GITHUB = SERVER**, all working trees clean, 0 unpushed, 92 commits, no history rewritten |
| **Deferred explain-backs** | **42 files, 107 occurrences, 25 numbered Checkpoints** |

### Known major limitations

Everything is **in-distribution** · **single seed (42)** everywhere · DPO budget
**~250× under** SFT's KL · one LoRA target set, **no rank sweep** · **the
MLP-target confirmation (E20) was never run** · no standard benchmark · judge
unvalidated, same-family, pinned to `main` · greedy decoding only · length
measured but **never controlled** · single GPU, single model, single size · one
permanent provenance gap in `eval-full-001` · **no production-quality claim**.

Full list: `docs/LIMITATIONS.md`.

---

**Related:** [[how-to-run]] · [[project-narrative]] · [[experiment-registry]] ·
[[limitations]] · [[phase8-report]]
