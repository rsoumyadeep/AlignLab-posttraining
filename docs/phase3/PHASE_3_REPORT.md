# Phase 3 Report — Supervised Fine-Tuning

**Date:** 2026-08-30 · **Branch:** `phase-3-sft`
**Machines:** local (Windows, CPU) + `csrslave` (1 × RTX A6000 used)
**Model:** `Qwen/Qwen2.5-1.5B` @ `8faed761d45a263340a0528343f099c05c9a4323` (PINNED)

---

## 1. What Was Built

**Modules** (`src/alignlab/`):

| Module | Contents |
|---|---|
| `storage.py` | disk pre-flight guard; checkpoint-size arithmetic; refuses rather than warns |
| `data.py` | instruction dataset → prompt/completion, split, content fingerprint |
| `masking.py` | loss-mask computation **independent of TRL**, prefix-consistency check |
| `sft.py` | SFT entrypoint: three pre-flight audits, manifest, resume, tracking |
| `config_schema.py` | `ModelConfig`, `DataConfig`, `SFTHyperParams`, `SFTExperimentConfig` |

**Configs:** `configs/sft.yaml` root + `model/`, `data/`, `sft/` groups
(`default` and `smoke` profiles).

**Scripts:** `fetch_qwen_weights.py` (pinned, guarded, idempotent, verified) ·
`e11` weights reconciliation · `e12` **loss-masking gate** · `e13` before/after ·
`e14` control-region decomposition.

**Tests:** +59 (290 → **349**).

**Documentation:** all five folders.

---

## 2. What Was Verified

- **The loss mask.** TRL's labels are **identical, position by position**, to a
  mask computed without importing TRL. Boundary 26 of 33 tokens; active region
  decodes to exactly `'It is 4.<|im_end|>\n'`.
- **The broken arm behaves as expected**: `completion_only_loss=False` makes
  33/33 tokens active instead of 7.
- **`completion_only_loss` auto-resolves to `True`** from the dataset shape
  (read in TRL source, then observed).
- **Prefix consistency: 0/300** sampled rows inconsistent after the fix.
- **Boundaries agree on real batches**: 3/3, re-checked before the first
  optimiser step of the training run.
- **Zero examples with no trainable tokens** across all 9,436 processed rows.
- **The corrected parameter formula reproduces the real checkpoint exactly**
  (1,543,714,304).
- **The disk guard fires**: needs 44.28 GiB, 94.1 GiB free → proceed; and
  refuses on impossible requests (tested).
- **Download integrity**: all 8 files byte-exact against the Hub manifest.
- **Resume path** implemented and exercised by checkpoint writes; **full
  interrupt/resume cycle NOT TESTED** on a real SFT run.
- **W&B offline** produced a run directory.
- Tests: **348 passed / 1 skipped** (server), **347 / 2 skipped** (local).

---

## 3. What Was Measured

### The training run — `sft-qwen1p5b-noRobots-001`

| | |
|---|---|
| Model | Qwen2.5-1.5B, **1,543,714,304** params, **100% trainable** (full FT) |
| Data | `HuggingFaceH4/no_robots`, **9,499** train / **200** eval |
| Train fingerprint | `17bebb1758a66958aaf4ba16…` |
| Eval fingerprint | `762160ae5ca3cde3c62874d4…` |
| Rows dropped by filter | **1 of 9,500** |
| Schedule | 296 steps, 1 epoch, effective batch 32, warmup 9 |
| LR | 2e-5 cosine, weight decay 0.0, grad clip 1.0 |
| Precision | bf16, `sdpa`, gradient checkpointing on |
| **Runtime** | **1,167.9 s (19.5 min)**, 8.08 samples/s, 0.253 steps/s |
| **Final train loss** | **2.0498** |
| **Final eval loss** | **1.9893** |
| Eval token accuracy | 0.5538 |
| VRAM | ~16.8 GiB of 48 GiB |

**Eval loss trajectory:** 2.045 → 2.007 → 1.992 → 1.990 → 1.989 → **1.989**
(steps 50/100/150/200/250/296). Flat after step 200.

### Tokenization and truncation, `max_length=1024`

| | |
|---|---|
| Examples | 9,436 |
| At the length limit | **100 (1.06%)** |
| **Zero-trainable-token examples** | **0** |
| Total tokens | 2,699,784 |
| Active (loss-bearing) tokens | 1,679,946 (**62.23%**) |
| Median length | 239 |

### Storage — estimate versus measurement

| | |
|---|---:|
| Weights download (manifest) | **2.886 GiB** (8 files) |
| Download time | 40.5 s |
| **Estimated** full checkpoint | **20.13 GiB** (14 B/param) |
| **Measured** full checkpoint | **8.63 GiB** (6.00 B/param) |
| — `model.safetensors` | 3,087,467,144 B = **2.00 B/param** |
| — `optimizer.pt` | 6,175,148,456 B = **4.00 B/param** |
| Final model (weights only) | 2.886 GiB |
| `/data` free, before → after | 94.1 → **79.5 GiB** |

The estimate over-predicts **2.33×**: AdamW's moments are stored in **bf16**,
and no fp32 master copy is written. The conservative default is **kept** — a
guard that under-predicts fails mid-write on a shared volume, which is not
symmetric with refusing a run that would have fitted.

### E13 — before/after, 200 held-out examples

| metric | base | SFT | change |
|---|---:|---:|---:|
| **completion perplexity** (TRAINED) | 8.541 | **7.199** | **−15.7%** |
| prompt perplexity (control) | 30.996 | 25.079 | −19.1% |
| generations stopping naturally | **1 / 4** | **4 / 4** | — |
| generations emitting `<\|im_end\|>` | **0 / 4** | **4 / 4** | — |

### E14 — decomposing the control region, 100 examples

| region | base NLL | SFT NLL | abs drop | rel | tokens |
|---|---:|---:|---:|---:|---:|
| template (ChatML scaffold) | 8.8727 | 8.1303 | **0.7425** | −8.4% | 1,521 |
| user text | 2.6402 | 2.5207 | 0.1195 | −4.5% | 11,193 |
| completion (trained) | 2.1243 | 1.9729 | 0.1514 | −7.1% | 21,001 |

Prompt-region NLL drop 0.1940; **45.8%** attributable to template tokens,
**54.2%** to user text.

---

## 4. Results — interpretation

**The qualitative change is the headline, not the perplexity.** Before SFT the
base model produced degenerate repetition (`-unstyled` repeated to the token
cap), answered an English prompt in Chinese, and stopped naturally once in four
attempts. After one epoch it produced coherent, correctly formatted English
answers — including a well-formed email with `[Name]`/`[Date]` placeholders —
and terminated cleanly **4/4** by emitting `<|im_end|>`, a token it had never
emitted before. That is a binary, measurable capability that one epoch bought.

**A 15.7% perplexity improvement is modest and should be read as such.** It
measures fit to no_robots' own test split. It is exactly what one epoch of
no_robots should buy and is **not** evidence of a better assistant in general.

**Eval loss was flat after step 200** — roughly two thirds of the way through
the single epoch. More epochs at this LR would likely add little without other
changes.

---

## 5. Unexpected Findings

1. **The untrained "control" region improved *more* than the trained one**
   (−19.1% vs −15.7%). Investigated in E14 rather than reported. A masked
   region is **not an experimental control**: masking removes the direct
   gradient, not the representational change, because every position is scored
   by the same shared parameters. Template tokens improved most in absolute NLL
   (0.74), but user text contributed the larger share of the total (54.2%)
   simply because there are ~7× more of them. **Within a single model there is
   no isolated control.** The control still tells us direction — prompt NLL
   rising sharply would have indicated drift; it fell, so nothing was damaged.

2. **The base model's tokenizer ships a ChatML template it was never trained
   on**, and its `eos_token` (`<|endoftext|>`, 151643) disagrees with the
   template's terminator (`<|im_end|>`, 151645). Both are already in the vocab,
   so no resize is needed — but generation must be told to accept both.

3. **E12's H5 was disproved.** We predicted the prompt would be *easier* than
   the answer, making an unmasked loss look better. Measured: prompt CE
   **6.3314** vs completion **3.6295** — the prompt is *harder*, because the
   base model has never seen ChatML and the opening tokens have no left
   context. The corrected lesson is stronger: an unmasked loss is a mean over a
   **different token population** and is not comparable in *either* direction.

4. **The disk estimate was 2.33× too conservative** — bf16 optimiser moments,
   no fp32 master copy.

5. **`assistant_only_loss` is unavailable** on Qwen2.5's stock template (no
   `{% generation %}` markers). TRL refuses outright.

---

## 6. Bugs / Corrections

**6a. Phase 2's parameter count was wrong by 57,344.** Config-only arithmetic
gave 1,543,656,960; the real checkpoint has **1,543,714,304**. The gap is
exactly `28 × (1536 + 256 + 256)` — the attention QKV biases. Phase 2 had
**already discovered** Qwen uses QKV bias and listed it among its wrong
predictions; the fact never reached the arithmetic. Error 0.0037%, invisible at
the card's "1.54B" — which is why the Phase 2 report's word *"exactly"* was
wrong. Found by E11. Corrected formula reproduces the checkpoint to the
parameter.

**6b. Completion whitespace broke the tokenization boundary.** 3/200 rows. The
ChatML prompt ends with token `198` (`'\n'`); a completion starting with
newlines lets BPE merge it into `1406` (`'\n\n\n'`), so the boundary lands on a
token that is half prompt and half answer. Found by `check_prefix_consistency`,
which exists precisely because TRL's boundary arithmetic rests on a prefix
assumption we chose to verify rather than trust. TRL warns about this but
proceeds anyway. Fixed by stripping — a fix, not a workaround: those newlines
are redundant with the template's own newline.

**6c. `datasets.map()` silently served a stale cache.** The fix in 6b appeared
to do nothing — the same three indices returned — while the function was
correct in isolation. `load_from_cache_file=False` now forces recomputation.
The content fingerprint changing (`b7dff71c…` → `dc6fd746…`) is what proved the
data had finally changed.

**6d. Our own loss decomposition mis-counted.** E12 arm D's cross-check failed
by 1.8e-02 on its first run. The masking was correct; the counting was not.
Contributing positions are `(labels[:, 1:] != -100).sum()`, not
`(labels != -100).sum()` — the shift drops position 0. With 25 rather than 26
prompt positions the decomposition agrees to **1.6e-07**.

**6e. `tests/test_storage.py` carried Phase 2's wrong parameter count** — 6a
propagating one file further, caught only when estimates were checked against a
real checkpoint file.

**6f. Five transformers 5.x / TRL 1.x API breaks**, each found by failure:
`cfg.rope_theta` → `cfg.rope_parameters["rope_theta"]`; `warmup_ratio` removed
(only `warmup_steps`); `logging_dir` removed; `max_seq_length` → `max_length`;
`tokenizer.additional_special_tokens` → `get_added_vocab()`.

**6g. Guessed dataset split names.** `train_sft`/`test_sft` inferred from the
repository's parquet filenames; the builder exposes `train`/`test`.

**6h. A false claim of ours, corrected.** We initially concluded Qwen's template
supported `assistant_only_loss` by substring-matching `"generation"` — which
occurs only as `add_generation_prompt`. **A grep is not a parser.**

**6i. Process failure on our side.** Draft figures were written into
`PYTORCH_CONCEPTS/pytorch-loss-masking-and-shifts.md` **before** the example was
executed, and were wrong (2.706560/2.653215 vs the actual 2.709972/2.436063).
Running the script replaced them, and the note records that it happened.

---

## 7. Resources Actually Inspected

Seven, on 2026-08-30 — dominated by **source code and configuration** rather
than papers, because three Phase 3 bugs came from assuming an API instead of
reading it.

TRL `sft_trainer.py` (specific lines) · Qwen2.5-1.5B model card at the pinned
revision · `config.json` + `tokenizer_config.json` · `Qwen2Config` under
transformers 5.16.1 (introspection only) · no_robots schema and distribution ·
the Hub file-tree API · PROJECT_INSTRUCTIONS.

**Explicitly NOT INSPECTED:** InstructGPT, Self-Instruct, LIMA, the Qwen2.5
technical report, TRL's packing implementation, the transformers 5.x migration
guide. **No claim here rests on them.**

### A Phase 2 open question, now resolved

Phase 2 recorded `max_position_embeddings = 131072` vs a card context of 32,768
as an **UNRESOLVED** discrepancy. Reading the card at the pinned revision shows
the numbers describe **different scopes**: the 128K claim sits in the
*series-level* introduction; the *per-repo* spec block for this checkpoint says
`Context Length: Full 32,768 tokens`.

**Operational decision:** treat **32,768** as the trustworthy trained context.
The 131072 is the architectural ceiling RoPE is configured for (consistent with
`rope_theta = 1e6`) and would need YaRN scaling to use. Not load-bearing here —
Phase 3 trained at 1,024.

---

## 8. Limitations

1. **One run, one seed.** No variance estimate. Nothing here distinguishes a
   real effect from run-to-run noise.
2. **No LR sweep.** 2e-5 is convention, not a tuned value.
3. **One epoch**, and eval loss was flat after step 200.
4. **Evaluation is in-distribution.** Perplexity on no_robots' own test split.
   "Better" is partly circular; no external benchmark was run.
5. **Only 4 qualitative prompts**, greedy-decoded. Indicative, not a measurement
   of instruction-following quality.
6. **No LLM-as-judge** — deferred to Phase 7, which needs a judge model.
7. **Multi-turn signal discarded** for ~8% of rows; `assistant_only_loss`
   unavailable.
8. **Resume NOT TESTED end-to-end** on a real SFT run — the code path exists and
   checkpoints were written, but no interrupt/restart cycle was exercised.
9. **Single GPU.** DDP still untouched; the second A6000 was idle.
10. **`packing=False`** — throughput left on the table deliberately.
11. **W&B online still NOT TESTED**; offline only.
12. The manifest for the training run records `dirty: True` (untracked
    `docs/phase3/`), which is honest but means the run was not from a pristine
    tree.

---

## 9. What Phase 3 Does **Not** Establish

- That the SFT model is a better **assistant** in general.
- That 2e-5 / 1 epoch / this dataset is a good configuration.
- Anything about LoRA, QLoRA, DPO, or reward modelling.
- That the perplexity gain would survive on out-of-distribution prompts.
- That the qualitative improvement generalises beyond the 4 prompts shown.
- Any **causal** claim beyond "training on this data produced these changes" —
  no ablation isolated *which* aspect of the data mattered.

---

## 10. Git / Synchronization State

```
Branch : phase-3-sft   (local + GitHub + server, all at the same commit)
Base   : 71ed4e6 (end of Phase 2, on phase-1-foundation)
Remote : git@github.com:rsoumyadeep/AlignLab-posttraining.git  [PRIVATE]
Merged : none. main remains at d21c070; phase-1-foundation at 71ed4e6.
Working tree: clean on both machines
```

**Phase 3 commits (15):** storage guard + fetcher · data/masking/E11 · E12 gate ·
E12 shift fix · SFT entrypoint + configs + tests · split-name fix · warmup fix ·
whitespace fix · cache fix · E13 + measured checkpoint size · docs ×2 · README ·
unused import · E14.

A **new branch** was created for Phase 3 rather than continuing on
`phase-1-foundation`, whose name was two phases stale. No history rewritten, no
force-push. Phase 1 and Phase 2 commits untouched.

---

## 11. Phase 4 Prerequisites

**The gate is OPEN.** PROJECT_INSTRUCTIONS §3 required verified loss masking
before PEFT; E12's H1–H3 pass and every run re-verifies. **Phase 4 may start.**

1. Install `peft` (currently **absent** on both machines) and, for QLoRA,
   `bitsandbytes` (**absent**, availability on the server **UNVERIFIED**).
2. `SFTTrainer` accepts `peft_config` and `quantization_config` directly —
   verified present in the 1.12.0 signature.
3. **Raise `save_total_limit`** for LoRA. `estimate_lora_checkpoint_bytes`
   already exists; a rank-16 adapter is ~100 MiB against 8.63 GiB, so the
   storage argument that forced `1` does not apply.
4. **Baseline for comparison is now recorded**: full FT = 1,543,714,304
   trainable (100%), 8.63 GiB resumable checkpoint, 19.5 min/epoch,
   ~16.8 GiB VRAM, completion ppl 7.199.
5. Disk: **79.5 GiB free**, still shared and still at 99%.
6. The Phase 3 SFT checkpoint is retained at
   `checkpoints/sft-qwen1p5b-noRobots-001/final` (2.9 GiB) as the DPO starting
   point for Phase 6. Its `checkpoint-295` (8.7 GiB) can be deleted once resume
   is no longer needed — **awaiting authorisation**.

---

## 12. Explain-Back Checkpoints Deferred

Per the standing instruction, theory is **DEFERRED** and no USER section was
filled. Phase 3 adds **three** new checkpoints:

| # | Checkpoint | Location |
|---|---|---|
| 13 | Which tokens contribute to the SFT loss, and why | `STUDY_WITH_CLAUDE/phase3/01` |
| 14 | The tokenization boundary and why prefix-consistency can fail | `STUDY_WITH_CLAUDE/phase3/01` |
| 15 | Truncation × completion-only masking | `STUDY_WITH_CLAUDE/phase3/01` |

Plus `PYTORCH_CONCEPTS/pytorch-loss-masking-and-shifts.md` (My Understanding),
`LEARNING_RESOURCES/phase3_resources.md` (7 × Own Words), and
`INTERVIEW_DEFENSE/phase3_sft.md` (Explain Back).

**Running total outstanding: 17 checkpoints + 27 Own Words entries + Drills 1–5.**
All marked `DEFERRED — USER EXPLAIN-BACK REQUIRED`. None was answered on the
USER's behalf.

---

**PHASE 3 STATUS: ENGINEERING, EXPERIMENTS, VERIFICATION, TRAINING AND
DOCUMENTATION COMPLETE — USER EXPLAIN-BACK CHECKPOINTS DEFERRED BY INSTRUCTION.**
