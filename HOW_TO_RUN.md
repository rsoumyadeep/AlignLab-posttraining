# HOW TO RUN AlignLab

A guided tour of the **empirical** side of AlignLab: how to run it, how to
reproduce the experiments, and — the part that matters more — how to read what
comes out.

Everything below was checked against this repository at commit `e1c5cb3`.
Where something could not be established from the repository or its history, it
is marked **NOT RECORDED**, **NOT VERIFIED**, **NOT TESTED**, **NOT CONFIGURED**
or **DEFERRED** rather than guessed.

**This document does not duplicate the five documentation folders.** Theory
lives in `STUDY_WITH_CLAUDE/`, code architecture in `CODE_EXPLANATION/`, PyTorch
mechanics in `PYTORCH_CONCEPTS/`, source audits in `LEARNING_RESOURCES/`, and
interview preparation in `INTERVIEW_DEFENSE/`. This file links to them.

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Repository structure](#2-repository-structure)
3. [Environment setup](#3-environment-setup)
4. [Basic running instructions](#4-basic-running-instructions)
5. [Configuration guide](#5-configuration-guide)
6. [Experiment reproduction guide](#6-experiment-reproduction-guide)
7. [How to understand experiment results](#7-how-to-understand-experiment-results)
8. [How to read the plots](#8-how-to-read-the-plots)
9. [Result table interpretation](#9-result-table-interpretation)
10. [Experiment comparison](#10-experiment-comparison)
11. [Visual inspection guide](#11-visual-inspection-guide)
12. [Troubleshooting](#12-troubleshooting)
13. [Server workflow](#13-server-workflow)
14. [Testing and verification](#14-testing-and-verification)
15. [Known limitations](#15-known-limitations)
16. [Reproducibility checklist](#16-reproducibility-checklist)
17. [Experiment → evidence cross-reference](#17-experiment--evidence-cross-reference)
18. [Interpretation principles](#18-interpretation-principles)
19. [How should I read AlignLab?](#19-how-should-i-read-alignlab)

---

## 1. Project overview

### What it is

A complete **post-training** pipeline for `Qwen/Qwen2.5-1.5B` — SFT, LoRA,
QLoRA, preference data handling, DPO, and an evaluation subsystem — together
with the experimental record of what each stage actually did.

### Objective

To answer *"how do you know the model actually got better?"* with measurements,
and to keep the answers that came back negative. **Five stated hypotheses were
disproved** and are preserved with their original wording.

### Scope boundary

**In scope:** everything after pretraining.
**Out of scope:** pretraining; RLHF with a learned reward model and PPO (studied
conceptually in Phase 5, **NOT IMPLEMENTED**); multi-GPU training; serving; any
claim of production model quality.

### Base model

`Qwen/Qwen2.5-1.5B` — the **base** model, not `-Instruct` — pinned to revision
`8faed761d45a263340a0528343f099c05c9a4323` in every config.

Starting from base is what makes the headline result a measurement: the base
model genuinely cannot emit the ChatML terminator, so `0/6 → 6/6` is
attributable to our SFT rather than to somebody else's.

### Data and model flow — as the repository actually works

```
Qwen/Qwen2.5-1.5B @ 8faed761  (frozen, pinned, downloaded once)
        |
        |  HuggingFaceH4/no_robots  (train 9,499 rows / test 200)
        v
  [ alignlab.sft ]  ------ peft=none ---> sft-qwen1p5b-noRobots-001/final  (2.9 G, full weights)
        |                                          |
        |------- peft=lora  ---> lora-r16-lr2e-4-001/final   (adapter, ~29 MB)
        |------- peft=qlora ---> qlora-r16-lr2e-4-001/final  (adapter, ~19 MB)
        |
        |  the three PEFT arms branch FROM THE BASE MODEL, not from the SFT
        |  checkpoint - they are alternatives to full SFT, not stages after it
        v
sft-qwen1p5b-noRobots-001/final
        |
        |  HuggingFaceH4/ultrafeedback_binarized (test_prefs; 200 rows -> 184 usable)
        |  used BOTH as DPO training data and as the preference eval set
        v
  [ alignlab.dpo_train ]  --> dpo-beta{0.01,0.1,0.5}-sum/final  (2.9 G each, full weights)
        |                     diag-dpo-beta0.1-lr5e-6/final     (POST-HOC)
        v
  [ alignlab.evaluate ]  --> outputs/<run>/evaluation_dashboard.{json,txt}
        loads all five models in one pass; adapters loaded UNMERGED
```

**Two things that are easy to get wrong about this diagram:**

1. **LoRA and QLoRA start from the base model, not from the SFT checkpoint.**
   They are *alternatives* to full SFT trained on the same data with the same
   objective, which is what makes E19 a controlled comparison.
2. **DPO starts from the full SFT checkpoint** (`.../final`), which is why that
   checkpoint is protected and why the Phase 6 reference model is that same
   SFT model frozen.

### Which phases require retraining, which reuse checkpoints

| Phase | Needs training? | Starts from |
|---|---|---|
| 1 — foundation | no real model | synthetic linear regression (`alignlab.train`) |
| 2 — transformer components | trains a **tiny** LM (334,080 params) | random init |
| 3 — SFT | **yes**, ~19.5 min | base model |
| 4 — PEFT | **yes**, 3 arms × ~17.5 min | base model |
| 5 — preference learning | **no** — measurement only | reuses SFT + LoRA checkpoints |
| 6 — DPO | **yes**, 4 arms × ~10.7 min | SFT `final/` |
| 7 — evaluation | **no** | reuses all five checkpoints |
| 8 — finalisation | **no** | documentation + audit only |

### Conceptual vs implemented

| | Status |
|---|---|
| Attention (3 implementations), RoPE/ALiBi, RMSNorm, SwiGLU, KV cache, 5 decoders | **IMPLEMENTED** |
| SFT, LoRA, QLoRA, DPO loss + training loop, evaluation subsystem | **IMPLEMENTED** |
| Bradley-Terry, sequence log-probs, exact KL and the `k3` estimator | **IMPLEMENTED** |
| **PPO / RLHF with a learned reward model** | **CONCEPTUAL — NOT IMPLEMENTED** |
| Flash attention | uses PyTorch SDPA backends; **no custom kernel** |
| MLP-target LoRA (E20's decisive test) | **NOT CONFIRMED — DEFERRED** |

### Major findings

Summarised in [`docs/PROJECT_NARRATIVE.md`](docs/PROJECT_NARRATIVE.md); read
§7 and §9 below for how to interpret them.

---

## 2. Repository structure

Verified present. Nothing is listed here that does not exist.

### Tracked source code

```
src/alignlab/
  __init__.py  paths.py  seeding.py  logging_utils.py  checkpoint.py
  manifest.py  tracking.py  preemption.py  storage.py  env_detect.py
  config_schema.py  device.py  evaluation.py  train.py
  data.py  masking.py  sft.py                    Phase 3
  lora.py  peft_setup.py                         Phase 4
  preference.py  logprobs.py                     Phase 5
  dpo.py  dpo_train.py                           Phase 6
  evals/  __init__.py metrics.py runners.py judge.py report.py
  evaluate.py                                    Phase 7
  provenance.py                                  Phase 8
  models/  attention.py attention_pure.py positional.py transformer.py
           generation.py  (+ normalization/FFN)   Phase 2

configs/          Hydra tree — 22 YAML files (see §5)
tests/            26 test modules
scripts/          env_report.py, fetch_qwen_config.py, fetch_qwen_weights.py,
                  server_probe.sh, experiments/ (22 scripts: E1-E23 plus
                  tiny_lm.py and wp9_qwen_reconciliation.py)
```

### Tracked documentation

```
README.md                     project overview, 20 sections
HOW_TO_RUN.md                 this file
PROJECT_INSTRUCTIONS.md       the authoritative project specification
docs/
  EXPERIMENT_REGISTRY.md      every experiment, hypothesis, verdict, evidence
  PROJECT_NARRATIVE.md        the eight findings that matter
  LIMITATIONS.md              what the project does not establish
  phase1/ .. phase8/          phase reports + committed evidence artefacts
STUDY_WITH_CLAUDE/   16 files   theory and derivations
CODE_EXPLANATION/     8 files   what the code actually does
PYTORCH_CONCEPTS/     7 notes + 7 executable example scripts
LEARNING_RESOURCES/   7 files   honest source audit
INTERVIEW_DEFENSE/    8 files   reasoning under challenge
```

**There is no `CLAUDE.md` in this repository.**

### Generated artefacts — NOT tracked

`.gitignore` excludes `outputs/`, `checkpoints/`, `runs/`, `artifacts/`,
`wandb/`, `*.pt`, `.hydra/`, `multirun/`, caches and `.venv/`. **Zero files
under `outputs/`, `checkpoints/` or `logs/` are tracked** — verified.

Evidence is therefore **copied deliberately** into `docs/phase*/` rather than
tracked in place. That is why `docs/phase7/eval-full-001_dashboard.json` exists
in git while `outputs/eval-full-001/evaluation_dashboard.json` does not.

### Server-only resources

| | Location on `csrslave` |
|---|---|
| repository | `/data/home/rsoumyadeep/AlignLab` |
| virtualenv | `.venv/` (Python 3.11.16) |
| checkpoints | `checkpoints/<run-name>/` — 14 runs |
| run outputs | `outputs/<run-name>/` — 18 runs |
| HF cache | `.cache/huggingface/` (configured, **not** `~/.cache`) |

Datasets and checkpoints exist **only** on the server. Cloning this repository
gives you code, configs, docs and committed evidence — not weights.

---

## 3. Environment setup

### The two environments are NOT identical

This is measured, not assumed:

| | Local (development) | Server (`csrslave`, training) |
|---|---|---|
| Python | 3.13.13 | **3.11.16** |
| torch | 2.12.1+cpu | **2.6.0+cu124** |
| transformers | 5.8.1 | **5.16.1** |
| GPU | GTX 1050, 4 GB, Pascal | **NVIDIA RTX A6000, 47.53 GiB, Ampere** |
| bf16 | no | **yes — VERIFIED** |
| Role | code, docs, CPU test suite | **all real training and evaluation** |

**Every result in this project came from the server.** The local machine cannot
train this model in any configuration — bf16 weights alone are ≈3.1 GB before
gradients, optimiser state or activations.

The drift is recorded as a **Tier B** (structural cross-machine
reproducibility) weakness in `README.md` §17. It does not affect any result,
because each run manifest pins the exact versions of the machine that produced
it.

### Install

```bash
uv venv --python 3.11 .venv
uv pip install -e ".[dev]"                     # pytest, pytest-cov
```

Optional dependency groups, as declared in `pyproject.toml`:

| Group | Contents | When |
|---|---|---|
| `dev` | `pytest>=8.0`, `pytest-cov>=4.1` | always |
| `train` | `transformers`, `datasets`, `accelerate`, `peft`, `trl` | Phases 3+ |
| `tracking` | `wandb>=0.16` | optional; `noop`/`offline` work without it |
| `quant` | `bitsandbytes>=0.43` | **server only** — no useful CPU path |

Server install, as actually used:

```bash
ssh <server>
cd /data/home/rsoumyadeep/AlignLab
uv venv --python 3.11 .venv
UV_CACHE_DIR=/tmp/uv-cache uv pip install --index-url https://download.pytorch.org/whl/cu124 torch
UV_CACHE_DIR=/tmp/uv-cache uv pip install -e ".[tracking,dev]"
```

`UV_CACHE_DIR=/tmp` matters: `/tmp` is a separate NVMe, so several GB of package
cache stays off the 100%-full `/data` volume. Delete it afterwards.

### Verify the environment

```bash
python scripts/env_report.py            # human-readable
python scripts/env_report.py --json     # machine-readable
```

Every line is an observation from the running process. Hardware that torch
cannot use is **not** reported as usable — which is why the local GTX 1050 shows
as present but unusable.

### Environment variables

| Variable | Purpose |
|---|---|
| `ALIGNLAB_OUTPUT_ROOT` | run directories (fallback when config is empty) |
| `ALIGNLAB_CKPT_ROOT` | checkpoints |
| `ALIGNLAB_CACHE_ROOT` | HF cache |
| `ALIGNLAB_RUN_NAME` | force a run name (used for resume) |
| `HF_HUB_CACHE`, `HF_DATASETS_CACHE` | set **by** `configure_hf_cache()`; an already-exported value outranks the config |

**`HF_HOME` is deliberately never set** — it also governs the credentials file,
and relocating it would break an existing login.

> **The Hugging Face cache trap, and why it matters to you.**
> `huggingface_hub` reads `HF_HUB_CACHE` **once, at import**. Every AlignLab
> entrypoint therefore calls `configure_hf_cache()` **before** importing
> `transformers`. Getting this wrong cost 18.34 GB of duplicate weights in
> Phase 7 (`docs/phase8/CLEANUP_RECORD.md`). If you write your own script,
> configure the cache first. A test asserts the ordering in all three
> entrypoints.

### Storage

`/data` on the server is **shared with 8 other users and sits at 100%
capacity**, currently with **54 GiB free**. Disk, not VRAM, is the binding
constraint.

`alignlab.storage` runs a pre-flight check and **refuses to start** a run whose
checkpoints would not fit. Measured sizes: a resumable full-SFT checkpoint is
**8.63 GiB**; the exported `final/` is 2.9 G; a LoRA adapter is ~29 MB.

---

## 4. Basic running instructions

Every command below was checked against the repository. **Nothing here is
invented by convention.**

### Tests — cheap, CPU, no network, no credentials

```bash
python -m pytest -q                              # full suite  (~34 s local)
python -m pytest tests/test_lora.py -q           # one module
python -m pytest -q -k "Wilson or Provenance"    # by name
python -m pytest -q -m "not slow"                # by marker
```

Markers declared in `pyproject.toml`: `slow`, `gpu`, `network`, `server`.

### Linting — **NOT CONFIGURED**

There is no `ruff`, `black`, `flake8` or `mypy` configuration in
`pyproject.toml`, and no linter in the `dev` extra. **There is no lint command
for this repository.** Reported as absent rather than invented.

### Phase 1 foundation smoke test — CPU, seconds

```bash
python -m alignlab.train env=local train.max_steps=50
```

Exercises config composition, run directory, logging, seeding, manifest
capture, tracking, preemption, the loop, checkpointing, resume and evaluation —
on a **tiny synthetic linear regression**. No tokenizer, no pretrained weights,
**Qwen is not downloaded**. Writes `outputs/<run>/`.

### Fetch the base model — server, ~3 GB download

```bash
python scripts/fetch_qwen_config.py      # config only, cheap
python scripts/fetch_qwen_weights.py     # 2.886 GiB, pinned revision
```

### SFT and PEFT — server, GPU, expensive

```bash
python -m alignlab.sft env=server                              # full SFT
python -m alignlab.sft env=server peft=lora                    # LoRA r=16
python -m alignlab.sft env=server peft=qlora                   # QLoRA NF4
python -m alignlab.sft env=server sft=smoke                    # 20 steps, minutes
python -m alignlab.sft env=server run_name=my-run sft.learning_rate=1e-5
```

Writes `outputs/<run>/` (logs, `run_manifest.json`, `metrics.jsonl`,
`sft_summary.json`) and `checkpoints/<run>/` (`checkpoint-N/`, `final/`).

**Resume** is automatic: `sft.resume: true` is the default, and the entrypoint
picks the latest `checkpoint-*` under the run's checkpoint directory. To resume
a specific run, give it the same `run_name`.

### DPO — server, GPU, ~10.7 min per arm

```bash
python -m alignlab.dpo_train env=server                        # beta=0.1 default
python -m alignlab.dpo_train env=server dpo.beta=0.01 run_name=dpo-beta0.01-sum
python -m alignlab.dpo_train env=server dpo=smoke              # cheap wiring check
```

Writes `outputs/<run>/dpo_summary.json` and `checkpoints/<run>/final/`.

### Evaluation — server, GPU

```bash
python -m alignlab.evaluate env=server eval=quick     # 2 models, no judge, minutes
python -m alignlab.evaluate env=server eval=default   # 5 models + judge, ~7 min
```

Writes `outputs/<run>/evaluation_dashboard.json` and `.txt`. The text dashboard
is also printed to stdout.

**The judge stage downloads Qwen2.5-7B-Instruct (~15 GB) if not cached.**
Check free space first.

### Provenance audit — cheap, CPU, no model needed

```bash
python -m alignlab.provenance                    # configured output root
python -m alignlab.provenance outputs            # explicit path
python -m alignlab.provenance --json
```

Exit code **1** if any artefact is missing an applicable field. This is the
command that verifies the reproducibility claims rather than asserting them.

### Individual experiments

All 22 live in `scripts/experiments/` (20 numbered `e*.py`, plus
`tiny_lm.py` and `wp9_qwen_reconciliation.py`). They import `alignlab`, so either
install the package or set `PYTHONPATH=src`:

```bash
PYTHONPATH=src python scripts/experiments/e1_scaling.py       # CPU, seconds — VERIFIED to run
PYTHONPATH=src python scripts/experiments/e19_peft_comparison.py --help
```

Nine take `argparse` options (E12, E13, E14, E17, E19, E20, E21, E22, E23);
the rest take none. Those with a `--out` flag default to writing their JSON
into `docs/phase*/`, which is how the committed evidence was produced.

### Retrieve artefacts from the server

```bash
scp <server>:/data/home/rsoumyadeep/AlignLab/outputs/<run>/evaluation_dashboard.json .
```

---

## 5. Configuration guide

Hydra structured configs with `ConfigStore` schema registration. Invalid
configurations fail at composition time rather than mid-run.

### Roots — three, deliberately separate

| Root | Entrypoint | Groups it composes |
|---|---|---|
| `configs/config.yaml` | `alignlab.train` | `env, train, logging, tracking` |
| `configs/sft.yaml` | `alignlab.sft` | `env, model, data, sft, peft, logging, tracking` |
| `configs/dpo.yaml` | `alignlab.dpo_train` | `env, model, dpo, logging, tracking` |
| `configs/eval.yaml` | `alignlab.evaluate` | `env, eval, logging` |

### Groups

| Group | Options | Notes |
|---|---|---|
| `env` | `local`, `server` | **the only place machine paths live** |
| `model` | `qwen2_5_1_5b` | id + pinned revision + dtype + attn impl |
| `data` | `no_robots` | dataset, splits, caps |
| `sft` | `default`, `smoke` | hyperparameters |
| `peft` | `none`, `lora`, `qlora` | the arm selector |
| `dpo` | `default`, `smoke` | β, objective, LR |
| `eval` | `default`, `quick` | model set, datasets, judge |
| `logging` | `default` | console + file levels |
| `tracking` | `noop`, `wandb_offline`, `wandb_online` | `wandb_online` is **NOT TESTED** (no API key) |

### Values that materially affect results

| Parameter | Default | Why it matters |
|---|---|---|
| `model.revision` | `8faed761…` | **pinned**; changing it invalidates every comparison |
| `sft.max_length` | 1024 | truncation changes the token population, so perplexity is not comparable across values |
| `sft.per_device_train_batch_size` × `gradient_accumulation_steps` | 4 × 8 = **32** | effective batch; changing it changes the optimisation trajectory |
| `sft.learning_rate` | 2e-5 | conventional, **not tuned** — no LR sweep was run |
| `sft.packing` | **false** | ON deliberately-off: packing raises throughput but makes the loss mask far harder to verify, and a verified mask is the point of Phase 3 |
| `sft.completion_only_loss` | `null` | lets TRL resolve from data shape; E12 **verifies** the resolution rather than assuming it |
| `sft.save_total_limit` | 1 | **storage-bound, not a quality decision** |
| `peft.r` | 16 | a starting point, **not** derived from the SVD analysis |
| `peft.alpha` | 32.0 (= 2r) | keeps scaling `alpha/r = 2` constant, so a rank sweep is not also an LR sweep |
| `peft.target_modules` | `q,k,v,o_proj` | **`lm_head` deliberately excluded** — it is tied to the embedding. This is the centre of E20. |
| `dpo.beta` | 0.1 | **pre-registered** candidate set {0.01, 0.1, 0.5} |
| `dpo.length_normalise` | **false** | false = SUM = the published objective. True = MEAN, a **diagnostic, never the headline** |
| `dpo.learning_rate` | 5e-7 | conventional, ~40× below SFT; **not tuned** |
| `eval.generation.do_sample` | **false** (greedy) | a comparison should depend on weights, not sampling noise |
| `reproducibility.seed` | 42 | single seed everywhere — see §15 |

### Parameters that break comparability

Changing any of these makes a run **not directly comparable** to the committed
results: `model.revision`, `data.*` (dataset, split, caps — the fingerprint
changes), `sft.max_length`, batch geometry, `num_train_epochs`,
`peft.target_modules`, `dpo.length_normalise`, `eval.generation.*`, and the
seed.

Overrides are passed on the command line, Hydra-style:

```bash
python -m alignlab.sft env=server peft=lora peft.r=64 sft=smoke
```

---

## 6. Experiment reproduction guide

The full registry — every experiment, hypothesis and verdict — is
[`docs/EXPERIMENT_REGISTRY.md`](docs/EXPERIMENT_REGISTRY.md). This section adds
the **reproduction** detail.

> **Naming, preserved as it actually is.** Phases 2–6 number experiments
> E1–E23. Phases 6 and 7 then stopped assigning numbers and identify work by
> run name. So the β sweep, the post-hoc diagnostic and the Phase 7 evaluation
> pass have **no E-number**, and none is invented here.

---

### Phase 2 — Transformer components

**A. Identification** — scripts `e1_scaling.py`, `e2_causal_leakage.py`,
`e4_attention_variants.py`, `e7_norm_comparison.py`, `e8_kv_cache.py`,
`e9_decoding.py`, `e10_sdpa_backends.py`, `tiny_lm.py`,
`wp9_qwen_reconciliation.py`. Report: `docs/phase2/PHASE_2_REPORT.md`.
Evidence: `docs/phase2/gpu_experiment_results_2026-08-29.txt`.

**B. Question** — do the standard architectural claims survive measurement?

**C. Hypotheses and verdicts** — E1 **HOLDS**; E2 **HOLDS**; E4 **HOLDS on CPU,
NOT on GPU**; E5/E6 **HOLD**; E7 **HOLDS**; E8 **HOLDS on CPU, overhead-bound on
GPU**; E9 **MEASURED** (+ found a real top-p bug in our own code); E10
**HOLDS**; E11 **DISPROVED, then corrected**.

**D. Setup** — E1: B=1, H=1, T=16, `d_k ∈ {4,16,64,256,1024}`, inputs N(0,1),
**float64** so what is measured is saturation rather than float32 rounding,
seed fixed in the script. E10: GPU, T=4096, bf16.

**E. Command**

```bash
PYTHONPATH=src python scripts/experiments/e1_scaling.py     # CPU — VERIFIED to run
PYTHONPATH=src python scripts/experiments/e10_sdpa_backends.py   # needs GPU
```

**F. Runtime** — E1 **MEASURED: seconds** (run during this documentation pass).
Other Phase 2 scripts: **NOT RECORDED** individually.

**G. Artefacts** — E1 prints a table and writes nothing. GPU results are
committed verbatim in `docs/phase2/gpu_experiment_results_2026-08-29.txt`.

**H. Controls** — fixed seed; float64 for E1; three independent attention
implementations cross-checked (torch/numpy/pure Python) and against
`F.scaled_dot_product_attention`.

**How to Explain This Experiment.** *Testing:* whether textbook claims about
attention hold when measured. *Design:* sweep one variable (`d_k`) across 256×
with everything else fixed, and measure four quantities rather than one.
*Observed:* unscaled logit std tracked √d_k (1.85→31.50) and Jacobian mass
collapsed 0.708→0.014; scaled stayed flat. *Why:* q·k sums `d_k` unit-variance
terms, so its variance is `d_k`. *Establishes:* the scale factor prevents
softmax saturation, and the vanishing gradient is visible in the Jacobian
rather than inferred. *Does not establish:* anything about training dynamics at
scale. *Next:* **NOT PLANNED** — Phase 2 is closed.
Deeper: `INTERVIEW_DEFENSE/phase2_transformers.md`.

---

### Phase 3 — Supervised fine-tuning

**A. Identification** — run `sft-qwen1p5b-noRobots-001`; scripts
`e12_loss_masking.py`, `e13_sft_before_after.py`, `e14_region_decomposition.py`.
Report `docs/phase3/PHASE_3_REPORT.md`. Artefacts: `sft_run_manifest.json`,
`sft_run_summary.json`, `e13_before_after.json`, `e14_region_decomposition.json`.

**B. Question** — does the loss mask do what TRL says, and what does one epoch
of SFT actually change?

**C. Hypotheses** — E12 H1–H4 **HOLD**; **H5 DISPROVED** (we predicted the
prompt region would be *easier*; it was **harder**).

**D. Setup** (from `sft_run_manifest.json`)

| | |
|---|---|
| model / revision | `Qwen/Qwen2.5-1.5B` @ `8faed761…` |
| parameters | 1,543,714,304 (all trainable) |
| dataset | `HuggingFaceH4/no_robots`, train 9,499 (1 dropped) / eval 200 |
| train fingerprint | `17bebb1758a66958aaf4ba166438c6c48edba46a78a104c05f2913570f011178` |
| eval fingerprint | `762160ae5ca3cde3c62874d4792576c85ad6d45b8ec6632e7b166bcec5f0b40e` |
| batch | 4 × 8 = 32 · `max_length` 1024 · packing off |
| optimiser | AdamW, LR 2e-5, cosine, warmup 0.03, `max_grad_norm` 1.0 |
| epochs / seed | 1.0 (296 steps) / 42 |
| hardware / dtype | 1× RTX A6000 / bf16 |
| VRAM | ~16.8 GiB of 48 |

**E. Command**

```bash
python -m alignlab.sft env=server run_name=sft-qwen1p5b-noRobots-001
PYTHONPATH=src python scripts/experiments/e13_sft_before_after.py
```

**F. Runtime** — **MEASURED 1167.876 s (~19.5 min)**, 8.08 samples/s.

**G. Artefacts** — `checkpoints/sft-qwen1p5b-noRobots-001/{checkpoint-295,final}`
(8.7 G / 2.9 G); `outputs/<run>/{run.log, run_manifest.json, metrics.jsonl,
sft_summary.json}`; committed copies in `docs/phase3/`.

**H. Controls** — pinned revision; dataset fingerprints recorded; seed 42; three
pre-flight audits (prefix consistency over 300 rows, loss mask verified on a
real batch against an independent computation, truncation audit) that **refuse
to start** the run on disagreement; git commit **and dirty flag** in the
manifest.

**How to Explain This Experiment.** *Testing:* whether the loss is computed on
the tokens we think, and what SFT changes. *Design:* verify the mask against an
independent implementation **before** training, because Phase 2's E2 showed a
masking bug produces a 33.2× *better* loss. *Observed:* boundaries agreed
token-for-token on every checked example; perplexity improved and stop-token
emission went 0/4 → 4/4. *Establishes:* the objective is correct and SFT taught
termination. *Does not establish:* general quality — the eval split is the
training distribution. *Next:* the PEFT comparison (Phase 4).
Deeper: `INTERVIEW_DEFENSE/phase3_sft.md`.

---

### Phase 4 — PEFT (LoRA / QLoRA)

**A. Identification** — runs `lora-r16-lr2e-4-001`, `lora-r16-lr2e-5-001`,
`qlora-r16-lr2e-4-001`; scripts `e15`…`e20`. Report
`docs/phase4/PHASE_4_REPORT.md`. Artefacts: `e15`–`e20` JSON, three run
summaries, `fullsft_vram_probe.log`.

**B. Question** — what do LoRA and QLoRA actually cost, and where is their
ceiling?

**C. Hypotheses** — E17 H1 **HOLDS**, **H2 DISPROVED**; E19 H1, H2, H4 **HOLD**,
**H3 DISPROVED**; E20 **NOT CONFIRMED**.

**D. Setup** — held fixed across arms: model + revision, dataset + fingerprint,
objective, `max_length` 1024, batch 4×8, 1 epoch (296 steps), seed 42, the same
eval split and code path, one A6000, bf16. **Could not be held fixed:** the
learning rate — so **both** were run (2e-5 matched, 2e-4 conventional).
Trainable parameters: **4,358,144** (0.28%). Targets `q,k,v,o_proj`; α=32.

**E. Command**

```bash
python -m alignlab.sft env=server peft=lora  run_name=lora-r16-lr2e-4-001 sft.learning_rate=2e-4
python -m alignlab.sft env=server peft=qlora run_name=qlora-r16-lr2e-4-001 sft.learning_rate=2e-4
PYTHONPATH=src python scripts/experiments/e19_peft_comparison.py
```

**F. Runtime — MEASURED** — full SFT 1167.876 s; LoRA@2e-4 1054.8454 s;
LoRA@2e-5 1053.1224 s; QLoRA@2e-4 1044.005 s. Throughput 8.08 → 9.04 samples/s.

**G. Artefacts** — adapters (~29 MB LoRA, ~19 MB QLoRA) vs 2.9 G full weights;
the six JSON evidence files in `docs/phase4/`.

**H. Controls** — `dataset_controlled: true` recorded in
`e19_peft_comparison.json`; identical eval code path (E13's); a test asserts
every adapter setting is shared between `lora` and `qlora` so the arms differ in
exactly one thing — **how the frozen base is stored**.

**How to Explain This Experiment.** *Testing:* the cost and the ceiling of
parameter-efficient fine-tuning. *Design:* hold everything fixed except the
PEFT method; where a variable could not be held fixed (LR), run both and report
both. *Observed:* 0.28% of parameters, QLoRA at ~61% of LoRA's peak VRAM, only
3.6% perplexity cost — **and 0/4 stop-token emission against full SFT's 4/4**.
*Why:* E20's hypothesis — emitting `<|im_end|>` requires moving its logit via
`lm_head`, which is **tied to the embedding** and outside the target set; full
fine-tuning moved that matrix by relative 0.0136, the largest of any matrix
measured. *Establishes:* the savings are real and a likelihood metric did not
reveal an unusable model. *Does not establish:* that reachability is the cause —
that is **NOT CONFIRMED**. *Next:* the deferred MLP-target run (~18 min).
Deeper: `INTERVIEW_DEFENSE/phase4_peft.md`.

---

### Phase 5 — Preference learning

**A. Identification** — scripts `e21_preference_data_readiness.py`,
`e22_reference_model_and_kl.py`. Report `docs/phase5/PHASE_5_REPORT.md`.
Artefacts `e21_preference_readiness.json`, `e22_reference_and_kl.json`.

**B. Question** — is the preference data usable, and where does the DPO metric
*start*?

**C. Hypotheses** — E21 H1–H4 **HOLD**. E22: H1 **HOLDS** (with a caveat the
criterion hid), H2 **HOLDS**, H3 **HOLDS EXACTLY** (`0.000e+00`), **H4
DISPROVED**, H5 **HOLDS**. E22's H1 criterion is additionally recorded as
**UNTESTABLE AS STATED** — `0 < mean < 1` could not test the word "small".

**D. Setup** — `HuggingFaceH4/ultrafeedback_binarized`, `test_prefs`; 200 rows →
**184 usable** after dropping ties (11.9% ties). Reuses the SFT and LoRA
checkpoints — **no training**.

**E. Command**

```bash
PYTHONPATH=src python scripts/experiments/e21_preference_data_readiness.py
PYTHONPATH=src python scripts/experiments/e22_reference_model_and_kl.py
```

**F. Runtime** — **NOT RECORDED**.

**G. Artefacts** — the two JSON files in `docs/phase5/`.

**H. Controls** — fixed eval set and fingerprint; frozen reference model in
`eval()`; the implicit reward verified to be **exactly zero** when π = π_ref.

**How to Explain This Experiment.** *Testing:* whether the preference metric is
sound before optimising it. *Design:* measure the baseline **before** writing
any DPO code, so a later null cannot be explained away after the fact.
*Observed:* the SFT model preferred the chosen response only **47.2%** of the
time by SUM — **below chance** — while MEAN gave **58.3%**. *Why:* chosen
responses are **56.5% longer**, and SUM is a sum over tokens. *Establishes:*
the objective DPO would optimise did not start above chance, which reframes all
of Phase 6. *Does not establish:* that the data is bad — per token the chosen
response is genuinely better. *Next:* Phase 6, entered knowing this.
Deeper: `INTERVIEW_DEFENSE/phase5_rlhf_dpo.md`.

---

### Phase 6 — DPO

**A. Identification** — runs `dpo-beta0.01-sum`, `dpo-beta0.1-sum`,
`dpo-beta0.5-sum`, and the **POST-HOC** `diag-dpo-beta0.1-lr5e-6`; script
`e23_dpo_evaluation.py`. Report `docs/phase6/PHASE_6_REPORT.md`.
**Pre-registration: `docs/phase6/BETA_PREREGISTRATION.md`.**

**B. Question** — does DPO improve preference accuracy over the SFT baseline?

**C. Hypothesis — PRE-REGISTERED, committed 13:15:12 before any DPO code
existed.** Fixed β ∈ {0.01, 0.1, 0.5}; **SUM primary**, MEAN diagnostic; five
hypotheses and the criteria for declaring a negative result, all written in
advance. **Verdict: NULL RESULT.**

**D. Setup** — starts from `sft-qwen1p5b-noRobots-001/final`; reference is that
same model frozen. β per arm; `length_normalise: false` (SUM); `max_length`
1024; batch 1 × 16 accumulation; LR 5e-7 cosine, warmup 0.1; 1 epoch; seed 42;
gradient checkpointing on; 184 usable pairs.

**E. Command**

```bash
python -m alignlab.dpo_train env=server dpo.beta=0.01 run_name=dpo-beta0.01-sum
python -m alignlab.dpo_train env=server dpo.beta=0.1  run_name=dpo-beta0.1-sum
python -m alignlab.dpo_train env=server dpo.beta=0.5  run_name=dpo-beta0.5-sum
PYTHONPATH=src python scripts/experiments/e23_dpo_evaluation.py
```

**F. Runtime — MEASURED** — 640.336 s (β=0.01), 644.785 s (β=0.1), 635.923 s
(β=0.5), 641.819 s (post-hoc). ~10.7 min each.

**G. Artefacts** — four `*_summary.json` and `e23_dpo_evaluation.json` in
`docs/phase6/`; four `final/` checkpoints (2.9 G each).

**H. Controls** — **pre-registration** (the git commit timestamp is the
evidence); identical eval set and fingerprint
(`b3bc775a6db554cc757152219fc9ffb2f54a016c6dd69635847df970224a7748`); two
pre-flight assertions that **refuse to train** — reference frozen, and implicit
reward exactly 0 at init, with loss exactly `ln 2 = 0.6931471805599453`.

**How to Explain This Experiment.** *Testing:* whether DPO moves the published
objective. *Design:* pre-register β and the negative-result criteria so neither
can be chosen after the fact. *Observed:* SUM accuracy **0.4674 = 86/184 for the
baseline and all three arms — byte-identical**; KL from reference ~**0.0008**
against SFT's 0.2044. *Why:* the SUM metric carries a ~29-nat length gap and
the run moved it 0.04 (**728× short**); the policy barely moved (**~250×
under-budget** in KL). *Establishes:* under this configuration and budget, DPO
did not change SUM preference accuracy. *Does not establish:* **that DPO does
not work** — that claim is not supported. *Next:* a KL-budgeted run.
Deeper: `INTERVIEW_DEFENSE/phase6_dpo.md`.

---

### Phase 7 — Evaluation

**A. Identification** — run `eval-full-001`; module `alignlab.evaluate`.
Report `docs/phase7/PHASE_7_REPORT.md`, results `docs/phase7/EVAL_RESULTS.md`.
Artefacts `eval-full-001_dashboard.{json,txt}`, `eval-full-001_run.log.txt`.

**B. Question** — measure all five models on one pass, with uncertainty.

**C. Hypothesis** — none; this is **MEASURED / descriptive**.

**D. Setup** (from the dashboard's provenance)

| | |
|---|---|
| git commit | `2e2a51f1a2dc5c251ddb7c79b62a847261510bfa`, `git_dirty: false` |
| perplexity set | `no_robots` test, 150 examples, **27,127 completion tokens**, fp `f92fdec3…` |
| preference set | `ultrafeedback_binarized` `test_prefs`, 200 → 184, fp `b3bc775a…` |
| generation | **6 fixed prompts**, greedy, `max_new_tokens` 256, seed 42 |
| judge | `Qwen/Qwen2.5-7B-Instruct` @ **`main` — not a SHA** |
| hardware / dtype | RTX A6000 47.53 GiB / bf16 |

**E. Command**

```bash
python -m alignlab.evaluate env=server eval=default run_name=eval-full-001
```

**F. Runtime — MEASURED ~7 min 1 s** (log timestamps 14:59:00 → 15:06:01, in
`eval-full-001_run.log.txt`). That excludes the one-off judge download, whose
duration is recorded in `LEARNING_RESOURCES/phase7_resources.md` but not in any
run artefact.

**G. Artefacts** — `evaluation_dashboard.json` (107,582 B, includes **every**
generation and every judge verdict), `.txt`, `run.log`.

**H. Controls** — greedy decoding; adapters loaded **unmerged**; every judge
pair judged **twice in both orders**; Wilson intervals on every rate; full
provenance per model.

**How to Explain This Experiment.** *Testing:* what each training stage actually
achieved, with honest uncertainty. *Design:* no aggregate score, both perplexity
regions, SUM and MEAN always together, and position-randomised judging — each
choice justified by a specific earlier failure. *Observed:* **of 24 pairwise
comparisons, 3 were resolvable — all stop-token rates.** *Establishes:* SFT
taught termination; the SUM/MEAN inversion holds on all five models including
the untrained base. *Does not establish:* general quality (everything is
in-distribution), or that SFT beats base **in the judge's opinion** (n=4
decided, interval includes 0.5). *Next:* an out-of-distribution set.
Deeper: `INTERVIEW_DEFENSE/phase7_evaluation.md`.

---

## 7. How to understand experiment results

The single most useful thing this project learned: **in every phase from 3 to 6,
two metrics disagreed, and the disagreement was the finding.** Read results
looking for that.

### 7.1 Perplexity vs behavioural termination

- **Question it answers:** how well does the model predict the reference
  completion tokens?
- **Metric:** `exp(mean NLL per token)`, token-weighted.
- **Good/bad:** lower is better — *within one token region, on one dataset*.
- **Expected:** SFT improves it. **Actual:** 8.824 → 7.398 (−16.2%).
- **The unexpected trend, and the lesson:** perplexity put the PEFT arms within
  **3.6%** of full SFT while stop-token behaviour was **0/6 vs 6/6**. A
  likelihood metric **cannot see a behaviour the reference tokens do not
  exercise**. Termination, refusal, format compliance and tool-call validity all
  fall in that gap.
- **Bug indicator:** a *dramatically* better loss. Phase 2's E2 measured
  **33.2× lower** loss from a leaking causal mask.
- **Confounders:** token region (prompt vs completion vs full — Phase 3
  measured 6.3314 vs 3.6295 on the *same forward pass*); the denominator; the
  dataset. **All perplexity here is in-distribution.**
- **Type:** confirmatory for distribution fit; **blind** to behaviour.

### 7.2 SUM vs MEAN preference accuracy — and length

- **Question:** does the model rank the preferred response above the rejected
  one?
- **Metric:** fraction of pairs where the chosen response scores higher. **SUM**
  is the published DPO objective; **MEAN** is per-token.
- **Expected:** a fine-tuned model should exceed 50%. **Actual: 47.2% by SUM —
  below chance — and 58.3% by MEAN.** An 11.1-point swing on identical models
  and data.
- **Why:** chosen responses are 56.5% longer, and SUM sums over tokens:

```
SUM gap                      = -29.47 nats
explained by length alone    = -31.90 nats
residual once length removed =  +2.43 nats   <- chosen is BETTER per token
```

- **What makes it conclusive:** Phase 7 ran this on all five models. **Positive
  residual in every row, including the untrained base model.** The inversion is
  a property of the **metric**, not of any training stage.
- **Legitimate conclusion:** SUM preference accuracy on this dataset is
  length-dominated. **Illegitimate:** "the model prefers worse answers."
- **Type:** diagnostic. `PreferenceStats` cannot serialise SUM without MEAN or
  either without token counts, so this cannot be reported one-sided.

### 7.3 The DPO null

- **Actual:** SUM accuracy **86/184 for the baseline and all three β arms —
  byte-identical**. KL from reference **0.0008** vs SFT's 0.2044.
- **Is it a bug?** Reasonable question, and it was checked: the weights *did*
  change (Phase 7 perplexity moved in the 4th decimal, greedy generation
  differed on 4 of 6 prompts). The policy moved — just ~250× less than SFT did.
- **Legitimate:** "under the preregistered configuration and budget, DPO did not
  change SUM preference accuracy." **Illegitimate:** "DPO does not work."
- **Type:** confirmatory, pre-registered, and **replicated independently** in
  Phase 7 on a different evaluation path.

### 7.4 LLM-as-judge

- **Metric:** win rate over **decided** pairs, plus a first-class
  `position_bias_rate`.
- **Actual:** **33.3%** of base-vs-SFT pairs flipped when the answers were
  swapped. 4 decided verdicts; SFT won 3; interval **[0.301, 0.954]** — includes
  0.5.
- **Read this correctly:** base emits `-unstyled` 128 times and SFT writes a
  correct email, and the judge **still** could not resolve it. That is the
  protocol working — a one-order judge would have returned a clean 6-pair win
  rate with a third of its verdicts decided by position.
- **Confounders:** verbosity bias; **same-family** judge (Qwen judging Qwen);
  judge pinned to `main`, not a SHA.
- **Illegitimate:** treating the judge as ground truth. No human agreement study
  was run.
- **Type:** exploratory.

### 7.5 Memory vs quality

QLoRA used **3,095,107,584 B** peak VRAM vs LoRA's **5,073,415,168 B** (~61%)
for the same 4,358,144 trainable parameters and a 0.9% perplexity difference.
**Memory reduction is not evidence of preserved capability** — both arms failed
termination identically.

### 7.6 Timing is hardware-dependent

E4: reducing KV heads 16→1 cut the KV cache to **6.25%** and CPU latency
160.4→117.4 ms — but **GPU latency was flat** (0.86→0.81 ms). E8: KV cache gave
**2.96×** on CPU and **1.05×** on GPU, which is **overhead-bound at this model
size**, not a refutation.

**Never generalise a timing result across hardware.** The *memory* results are
architectural and do transfer; the *latency* results did not.

---

## 8. How to read the plots

### There are no plots.

**This repository contains zero plots, figures or visualisations.** Verified
mechanically: no tracked `.png`/`.jpg`/`.svg`/`.pdf`/`.html` files, and no
reference to `matplotlib`, `seaborn`, `plotly`, `pyplot` or `savefig` anywhere
in `src/`, `tests/`, `scripts/`, `configs/` or `pyproject.toml`.

**No plot is described here, because describing one would mean inventing it.**

### What serves the role a plot usually plays

| Artefact | Format | Where |
|---|---|---|
| Evaluation dashboard | **ASCII text table**, also printed to stdout | `docs/phase7/eval-full-001_dashboard.txt` |
| Full evaluation record | JSON — every metric, generation and judge verdict | `docs/phase7/eval-full-001_dashboard.json` |
| Per-step training metrics | JSONL, one record per logged step | `outputs/<run>/metrics.jsonl` (**not tracked**) |
| Experiment results | JSON per experiment | `docs/phase*/e*.json` |
| Console tables | printed by the experiment scripts | not persisted unless redirected |

The text dashboard is **deliberately ASCII-only** — `alignlab.evaluate` prints
it to stdout, and on Windows a cp1252 console raises `UnicodeEncodeError` on
non-ASCII. Two tests enforce this, one of them against the committed Phase 7
artefact.

### W&B runs exist but contain no rendered charts

`outputs/<run>/wandb/offline-run-*/` directories exist for the training runs.
They were produced in **offline** mode and **never synced** — W&B online mode is
**NOT TESTED** (no API key on either machine). So no chart has ever been
rendered from them. Syncing would require an account and is out of scope.

### How to read the training curve without a plot

`outputs/<run>/metrics.jsonl` holds one JSON record per logged step
(`logging_steps: 5` for SFT). To inspect the loss trajectory:

```bash
python -c "
import json
for line in open('outputs/<run>/metrics.jsonl'):
    d = json.loads(line)
    if 'loss' in d: print(d.get('step'), d['loss'])
"
```

**What to look for** — grounded in what these runs actually did:

- **Healthy:** loss falls quickly then flattens. SFT's final train loss was
  2.0498 with eval loss 1.9893 — eval *below* train, normal for one epoch with
  dropout active during training.
- **Suspicious — check the mask first:** a loss that collapses toward zero.
  E2 measured **0.2907 vs 0.0088** — the 33× "better" number was the broken one.
- **Underfitting:** loss barely moves. The DPO runs are close to this by design
  — KL 0.0008 means the policy hardly moved, which the summaries confirm.
- **Instability/divergence:** spikes or NaNs. **Not observed** in any committed
  run.
- **Overfitting:** eval loss rising while train falls. **Not observable here** —
  one epoch, single eval points; **NOT MEASURED**.

**Note the honest limits:** with one epoch, one seed and eval every 50 steps,
these traces are too sparse to diagnose much. They are **descriptive**, not
diagnostic.

---

## 9. Result table interpretation

### 9.1 The Phase 7 per-model table

Source: `docs/phase7/eval-full-001_dashboard.txt`.

| model | ppl[compl] | ppl[full] | pref SUM | pref MEAN | stop | gen len | distinct-2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 8.824 | 14.993 | 45.7% | 61.4% | 0/6 | 212.3 | 0.523 |
| SFT | 7.398 | 12.284 | 46.7% | 58.7% | 6/6 | 117.3 | 0.839 |
| LoRA r=16 | 7.665 | 13.600 | 46.7% | 60.9% | 0/6 | 256.0 | 0.449 |
| QLoRA r=16 | 7.733 | 13.732 | 46.7% | 61.4% | 0/6 | 256.0 | 0.572 |
| DPO b=0.1 | 7.398 | 12.287 | 46.7% | 58.7% | 6/6 | 118.2 | 0.831 |

**How to read each column**

| Metric | Direction | Objective or proxy? | Length-sensitive? | Decoding-sensitive? | Can mislead? |
|---|---|---|---|---|---|
| `ppl[compl]` | lower better | **proxy** | no (token-weighted) | no | **yes** — blind to behaviour |
| `ppl[full]` | lower better | proxy | no | no | not comparable to `ppl[compl]` |
| `pref SUM` | higher better | **the DPO objective** | **YES, dominant** | no | **yes** |
| `pref MEAN` | higher better | **diagnostic only** | reduced | no | yes if quoted as the headline |
| `stop` | higher better | **direct behaviour** | no | **yes** | no |
| `gen len` | neither | descriptive | — | **yes** | reported precisely to expose length effects |
| `distinct-2` | higher better | crude proxy | somewhat | **yes** | detects degeneration, not quality |

**Magnitudes, honestly.** `ppl[compl]` 7.398 vs 7.665 is a **3.6%** difference
and **has no error bar** — single seed, no training-variance estimate. Do not
call it an improvement. By contrast **0/6 vs 6/6** is resolvable at n=6 because
the Wilson intervals ([0.000, 0.390] and [0.610, 1.000]) do not overlap.

**`ppl[compl]` and `ppl[full]` are never comparable.** They are means over
different token populations. `PerplexityResult.comparable_to()` returns `False`
across regions **even for identical values**, and there is a test for that.

**Statistical uncertainty was measured**: every rate carries a Wilson 95%
interval, and `difference_is_resolvable()` reports **"NOT resolvable at this
sample size"** when intervals overlap — deliberately returning **no p-value**,
because overlapping intervals do not prove no difference.

### 9.2 The β sweep table

Source: `docs/phase6/e23_dpo_evaluation.json`.

| arm | SUM | MEAN | KL from reference |
|---|---|---|---:|
| SFT (baseline) | 0.4674 (86/184) | 0.5870 | 0.000000 |
| DPO β=0.01 | 0.4674 (86/184) | 0.5870 | 0.000801 |
| DPO β=0.1 | 0.4674 (86/184) | 0.5870 | 0.000799 |
| DPO β=0.5 | 0.4674 (86/184) | 0.5870 | 0.000803 |
| **[POST-HOC]** β=0.1 lr5e-6 | 0.4674 (86/184) | 0.5924 | 0.002365 |

**Read the KL column first.** It is the sanity check: 0.0008 against SFT's
0.2044 says the policy barely moved, so identical accuracy is *expected*, not
mysterious. **Do not read the identical SUM values as "β doesn't matter"** —
this budget could not have distinguished β values.

The **POST-HOC** row is labelled as such and **must not replace** the
preregistered result. Its MEAN moved 0.5870 → 0.5924 — one pair's worth.

### 9.3 The PEFT comparison table

Source: `docs/phase4/e19_peft_comparison.json`.

| arm | trainable | peak VRAM (B) | runtime (s) | samples/s | eval loss |
|---|---:|---:|---:|---:|---:|
| full SFT | 1,543,714,304 | **NOT RECORDED** (null) | 1167.876 | 8.08 | 1.9893 |
| LoRA r=16 @2e-4 | 4,358,144 | 5,073,415,168 | 1054.845 | 8.945 | 2.0402 |
| LoRA r=16 @2e-5 | 4,358,144 | 5,073,415,168 | 1053.122 | 8.960 | 2.0718 |
| QLoRA r=16 @2e-4 | 4,358,144 | 3,095,107,584 | 1044.005 | 9.038 | 2.0769 |

**Full SFT's peak VRAM is `null` in the JSON — NOT RECORDED.** A separate probe
recorded ~16.8 GiB (`docs/phase4/fullsft_vram_probe.log`), from a different run.
Do not present the two as one controlled measurement.

**The learning-rate rows exist because the LR could not be held fixed.** The LR
effect (0.032 nats) against a LoRA-vs-full gap of 0.051 nats means reporting
only the matched arm would have **overstated LoRA's cost by about half**.

### 9.4 The SVD rank table (E17)

Source: `docs/phase4/e17_svd_rank_selection.json`, per module.

Recorded per matrix: `frob_dW`, `frob_W`, `relative_update`,
`top_singular_values`, `singular_value_decay_ratio`, and `energy_ranks` at
50/90/95/99%.

**How to read it:** `energy_ranks["90%"]` is the rank needed to capture 90% of
the update's energy — using **squared** singular values (E17 H1). For
`layers.0.self_attn.q_proj` it is **653 of 1536**, and `relative_update` is
**0.0013**.

**The finding:** the measured ΔW needs rank ~730 of 1536 for 90% of its energy,
while **r=16 captures 10.69%** (recorded in `configs/peft/lora.yaml`). So
**H2 — "the real ΔW is approximately low-rank" — is DISPROVED.**

**What that does and does not mean.** LoRA still worked. What failed is the
*explanation*: LoRA needs a low-rank update to **suffice**, not full
fine-tuning's update to **be** low-rank. **No document in this repository may
claim "LoRA works because fine-tuning updates are low rank."**

---

## 10. Experiment comparison

### Controls actually used

| Control | Where |
|---|---|
| same model + **pinned revision** | every config |
| same dataset + **sha256 fingerprint** over the actual rows | every training run's manifest |
| same objective and `max_length` | E19 across all PEFT arms |
| same batch geometry, epochs, seed (42) | E19, β sweep |
| same evaluation set **and the same eval code path** | E13 → E19 → E23 |
| same decoding (greedy, seed 42) | all Phase 7 generation |
| same hardware and dtype | all server runs |
| **same initialisation** (B=0 in LoRA) | adapted model starts exactly equal to base |
| git commit **and dirty flag** | every manifest |

### Known confounds — stated, not hidden

1. **Learning rate in E19.** LoRA conventionally needs a higher LR. Holding it
   fixed measures LR sensitivity; changing it breaks the control. **Both were
   run and both reported.**
2. **Full SFT's peak VRAM is NOT RECORDED** in E19; the ~16.8 GiB figure is from
   a separate probe run.
3. **The preference set is both DPO's training data and its evaluation data** —
   different splits, so no example-level leakage, but total distribution-level
   overlap.
4. **The judge is same-family** (Qwen judging Qwen) and pinned to `main`.
5. **Single seed everywhere**, so no comparison has a training-variance estimate.
6. **Phase 4 and Phase 7 perplexities differ** (8.541 vs 8.824 for base) because
   they use **different eval subsets** — 38,831 vs 27,127 completion tokens.
   Both correct; **not comparable**. The reconciliation is in
   `docs/phase7/EVAL_RESULTS.md` §4.

**Why changing several variables at once ruins the inference:** if LoRA had been
run with a different LR *and* a different rank *and* a different target set, the
0/6 termination failure could not have been attributed to any one of them. E20's
hypothesis is only formulable because everything except the PEFT method was
held fixed — and it is still only **NOT CONFIRMED**, because the one remaining
variable (the target set) has not been varied.

---

## 11. Visual inspection guide

**Qualitative evidence**, and clearly distinguished from the quantitative
tables. All generations are in `docs/phase7/eval-full-001_dashboard.json` under
each model's `generation.generations`.

```bash
python -c "
import json, textwrap
d = json.load(open('docs/phase7/eval-full-001_dashboard.json'))
for m in d['models']:
    print('='*70); print(m['name'])
    for g in m['generation']['generations'][:2]:
        print('-'*60)
        print('PROMPT:', g['prompt'][:80])
        print('n_gen=%s stop=%s' % (g['n_generated'], g['emitted_stop_token']))
        print(textwrap.fill(g['text'][:400].replace(chr(10), ' | '), 100))
"
```

### What to look for — grounded in what these models actually did

**1. Termination.** Does it stop, and does it stop *the right way*? Check
`emitted_stop_token` and `n_generated` separately. The base model **terminated
2/6 but emitted `<|im_end|>` 0/6** — it hit `<|endoftext|>` instead. A single
"did it finish" number would score it 33% and hide the thing SFT taught.

**2. Degeneration.** Base output includes `-unstyled` repeated ~128 times and
`.DrawString(...)` looped to the cap. `distinct-2` is the quantitative proxy
(base 0.523, LoRA 0.449, SFT 0.839) but reading the text is what tells you
*which kind* of repetition.

**3. Template regurgitation — the PEFT diagnostic.** Past the answer, LoRA and
QLoRA fall back into the chat template:

```
Dear [Name], ... Best regards, [Your Name] комф
You are a helpful assistant.TRGL
You are a helpful assistant.TRGL     [to the cap]
```

They emit the system prompt back. **No metric in this project captures this** —
it was found by reading the output, and it is what makes E20's reachability
hypothesis concrete.

**4. Conditioning fidelity.** Base answered an **English** cooking prompt **in
Chinese**. Structural correctness (does an "email" have a greeting and a
sign-off) is checked descriptively by `structural_checks` and **never scored**.

**5. Differences between checkpoints.** SFT vs DPO: **2 of 6 generations are
byte-identical**, 4 differ. That is how you can tell DPO changed the weights
even though no aggregate metric moved.

### Metrics deliberately NOT computed

**BLEU, ROUGE, METEOR, BERTScore, MMLU, HELM, AlpacaEval (including its
length-controlled variant), and FID are NOT MEASURED.** No human evaluation was
performed. No safety, refusal or factuality evaluation exists.

Consequently, every statement about *answer quality* in this project is either
(a) the length-confounded preference metric, (b) the unvalidated LLM judge, or
(c) **qualitative**. There is no quantitative ground-truth quality measurement,
and none should be inferred.

---

## 12. Troubleshooting

Only issues actually encountered in this project.

### Hugging Face cache lands in the wrong place

- **Symptom:** free disk drops sharply; the same model appears in two cache
  roots.
- **Cause:** `huggingface_hub` reads `HF_HUB_CACHE` **once, at import**. Setting
  it after importing `transformers` is a no-op.
- **Diagnose:** `du -sh ~/.cache/huggingface/hub/* | sort -h`
- **Fix:** call `configure_hf_cache()` **before** importing `transformers`. All
  three entrypoints do, and a test asserts the ordering. This cost 18.34 GB in
  Phase 7 — see `docs/phase8/CLEANUP_RECORD.md`.

### Disk full on `/data`

- **Symptom:** a run refuses to start, or a write fails mid-checkpoint.
- **Cause:** `/data` is shared with 8 users at 100% capacity.
- **Diagnose:** `df -h /data` · `du -sh checkpoints/*/ | sort -h`
- **Fix:** the storage guard should refuse *before* starting. Never delete
  another user's data; `~/.cache/huggingface` holds unrelated models.

### `datasets` serves a stale cache

- **Symptom:** a preprocessing fix appears to have no effect.
- **Cause:** `datasets.map()` reused a cached result.
- **Diagnose:** compare the **dataset fingerprint** across runs.
- **Fix:** `load_from_cache_file=False`. In Phase 3 the fingerprint changing
  (`b7dff71c` → `dc6fd746`) is what proved the fix landed.

### Adapter checkpoint fails to load

- **Symptom:** `AutoModelForCausalLM.from_pretrained` fails on a LoRA/QLoRA
  checkpoint.
- **Cause:** they are **adapters**, not full models.
- **Diagnose:** `ls checkpoints/<run>/final/` — an `adapter_config.json` means
  adapter.
- **Fix:** load with `PeftModel.from_pretrained`. `load_model()` in
  `evaluate.py` detects this and loads **unmerged** — merging in bf16 is
  12,460× less exact than fp32.

### `bitsandbytes` imports but does nothing

- **Symptom:** import succeeds; quantised training misbehaves.
- **Cause:** an import proves nothing about a CUDA kernel running.
- **Fix:** run E15, which **executes** a quantised matmul.
- **Related, recorded:** `triton` is installed but **unimportable** on the
  server (missing `setuptools`). Never needed.

### `completion_only_loss` silently resolves to `False`

- **Symptom:** loss looks fine; the model trains on prompt tokens too.
- **Diagnose:** the SFT pre-flight audit reports the resolved value.
- **Fix:** the run **refuses to start**. Do not disable that guard.

### A test "passes" because the gate was piped

- **Symptom:** failing tests get committed.
- **Cause:** `pytest | tail -2 && git commit` — the pipe's exit status is
  `tail`'s.
- **Fix:** never pipe a gate through another command. This happened in Phase 4
  and three failing tests were pushed.

### Unicode crash when printing

- **Symptom:** `UnicodeEncodeError: 'charmap' codec can't encode…`
- **Cause:** `print()` to a Windows cp1252 console with non-ASCII.
- **Fix:** the dashboard renderer is ASCII-only and two tests enforce it. In
  your own scripts, `.encode('ascii', 'replace')`.

### Server job dies when SSH drops

- **Symptom:** a long run stops when the connection closes.
- **Cause:** **SLURM is installed but NON-FUNCTIONAL** (`Slurmctld(primary) at
  csrmaster is DOWN`).
- **Fix:** run under `tmux`. Nothing enforces GPU allocation, so check
  `nvidia-smi` first — 8 users share the machine.

### An artefact is missing after a run

- **Diagnose:** `python -m alignlab.provenance outputs` — exit 1 and the
  offending field are reported.
- **Note:** `eval-full-001` has one **known, permanent** gap
  (`dataset_fingerprint`), left as produced rather than back-filled.

---

## 13. Server workflow

**No credentials appear in this repository.** Use your own SSH configuration;
`<server>` below is a placeholder for your SSH host alias.

```bash
ssh <server>
cd /data/home/rsoumyadeep/AlignLab
source .venv/bin/activate          # or call .venv/bin/python directly
```

### Launch and monitor

```bash
# check the shared GPU BEFORE launching - nothing reserves it
nvidia-smi

tmux new -s sft
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m alignlab.sft env=server 2>&1 | tee logs/sft.log
# detach: Ctrl-b d      reattach: tmux attach -t sft

tail -f logs/sft.log
watch -n 30 nvidia-smi
df -h /data
```

### Retrieve artefacts

```bash
scp <server>:/data/home/rsoumyadeep/AlignLab/outputs/<run>/evaluation_dashboard.json docs/phase7/
```

Evidence is copied into `docs/phase*/` deliberately, because `outputs/` and
`checkpoints/` are gitignored.

### Git synchronisation

The project maintains **LOCAL = GITHUB = SERVER**, and history is never
rewritten (no force-push, no rebase of pushed commits).

```bash
# local
git add -A && git commit -m "..." && git push origin <branch>
# server
cd /data/home/rsoumyadeep/AlignLab && git pull --ff-only origin <branch>
# verify
git rev-parse --short HEAD    # on each of the three
```

---

## 14. Testing and verification

```bash
python -m pytest -q                          # full suite
python -m pytest tests/test_dpo.py -q        # one module
python -m pytest -q -k Wilson                # by name
```

**There is no lint command — linting is NOT CONFIGURED** (§4).

### Current results

| | local | server |
|---|---|---|
| passed | **607** | **609** |
| skipped | 3 | 1 |

**The skips are genuine environmental impossibilities, and none was converted
into a passing test.** Locally: CUDA absent, `SIGUSR1` absent on Windows,
`wandb` not installed. On the server: one test that asserts the **CPU-only
failure mode** cannot run on a GPU machine.

### What the important tests actually protect

| Test area | Protects against |
|---|---|
| `test_seeding.py` | non-reproducible runs — asserts **bitwise** agreement across **separate interpreter processes**, not just two calls in one |
| `test_no_hardcoded_paths.py` | machine-specific paths in source — it caught a literal home path in a Phase 8 test of mine |
| `test_storage.py` | starting a run that cannot fit its checkpoints |
| `test_lora.py` | our LoRA diverging from `peft`; B=0 initialisation |
| `test_dpo.py` | the loss at init being anything other than `ln 2 = 0.6931471805599453`; non-zero implicit reward when π = π_ref |
| `test_logprobs_and_preference.py` | the log-prob **shift** being applied twice or not at all |
| `test_eval_metrics.py` | perplexity regions being compared; SUM emitted without MEAN; **the report exposing any aggregate score** |
| `test_eval_judge.py` | position bias — a scripted always-first judge must yield **zero** decided verdicts, not a 100% win rate |
| `test_provenance.py` | the auditor inventing a value, hiding a real gap, or padding the report with false ones |
| `test_paths_logging_device.py` | `configure_hf_cache` being called after `transformers` is imported — the ordering is asserted by source inspection |

**Two examples of tests verifying a mathematical claim rather than an
implementation detail:**

- The DPO loss at initialisation must be exactly `ln 2`, because when π = π_ref
  both implicit rewards are 0 and `-logsigmoid(0) = ln 2`. A test asserts the
  literal `0.6931471805599453`.
- Wilson intervals are checked against a **known value** (50/100 →
  `[0.4038, 0.5962]`) and for staying inside `[0,1]` at 0/10, 10/10 and 1/200,
  where the normal approximation escapes.

### Verify GPU

```bash
python scripts/env_report.py
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 15. Known limitations

Full list: [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md). The ones that most
change how you should read results:

| Limitation | Status | Why | Consequence |
|---|---|---|---|
| **All evaluation is in-distribution** | **NOT MEASURED** out-of-distribution | no OOD set was held out | **"better" is partly circular**; no generalisation claim is supported |
| Single seed (42) everywhere | **NOT MEASURED** | budget | no training-variance estimate; the 3.6% PEFT gap has no error bar |
| DPO budget 250× under SFT's KL | **MEASURED** | fixed epochs, not a KL target | the null is about this budget, **not** about DPO |
| One LoRA target set, r=16 only | **NOT TESTED** (other sets/ranks) | budget | rank/target effects unknown |
| **MLP-target LoRA confirmation** | **NOT CONFIRMED — DEFERRED** | ~18 min, never run | E20's hypothesis stays a hypothesis |
| No rank sweep | **NOT TESTED** | budget | the SVD analysis is not an empirical curve |
| DPO not replicated under the paper's setup | **NOT TESTED** | our config differs | says nothing about their result |
| LLM judge | **MEASURED** (33.3% position bias), **NOT VERIFIED** vs humans | no human study | not ground truth |
| Judge pinned to `main`, not a SHA | **RECORDED** | oversight | a rerun could get a different judge |
| No standard benchmark | **OUT OF SCOPE** | — | no comparability with published numbers |
| Greedy decoding only | **IMPLEMENTED**, by choice | isolate weights from sampling noise | behaviour under sampling **NOT MEASURED** |
| Length measured, never controlled | **MEASURED** | no length-matched set | preference results stay confounded |
| Single GPU, single model, single size | **OUT OF SCOPE** | — | nothing about scale |
| `eval-full-001` provenance gap | **NOT RECORDED**, permanent | Phase 7 bug, since fixed | that artefact's per-model field is absent; **not** back-filled |
| Cross-machine bitwise reproducibility | **NOT ACHIEVABLE / not claimed** | CPU-fp32 vs Ampere-bf16 | Tier C only |
| W&B online, SLURM requeue, multi-GPU RNG restore | **NOT TESTED** | no API key, no scheduler, no second GPU | untested code paths |
| **No production-quality claim** | — | — | 1.5B, one epoch, 9,499 examples |

---

## 16. Reproducibility checklist

```
[ ] correct repository commit          git rev-parse HEAD
[ ] clean working tree                 git status --porcelain   (empty)
[ ] correct Python environment         python --version      (server: 3.11.16)
[ ] correct dependency versions        python scripts/env_report.py
[ ] correct PyTorch/CUDA               server: torch 2.6.0+cu124, CUDA 12.4
[ ] GPU verified                       nvidia-smi ; torch.cuda.is_available()
[ ] free disk checked                  df -h /data       (storage guard also enforces)
[ ] base model present at pinned rev   8faed761d45a263340a0528343f099c05c9a4323
[ ] dataset available                  no_robots / ultrafeedback_binarized
[ ] dataset fingerprint verified       compare sha256 against the phase manifest
[ ] correct checkpoint selected        checkpoints/<run>/final/
[ ] correct config selected            env=server plus the group overrides
[ ] smoke test passed                  sft=smoke  or  eval=quick
[ ] test suite passed                  python -m pytest -q
[ ] training launched under tmux       nvidia-smi checked first
[ ] logs monitored                     tail -f logs/<run>.log
[ ] checkpoint produced                ls checkpoints/<run>/final/
[ ] evaluation executed                python -m alignlab.evaluate env=server eval=default
[ ] artifacts inspected                dashboard .txt and .json
[ ] provenance audited                 python -m alignlab.provenance      (exit 0)
[ ] results compared correctly         same fingerprint, region, decoding (see S10)
[ ] conclusions consistent with limits see S15 before claiming anything
```

*(No "plots generated" item — this project produces none.)*

---

## 17. Experiment → evidence cross-reference

Real run names throughout; **no invented IDs**.

| Phase | Experiment / run | Source | Report | Artefact | Key result | Interpretation |
|---|---|---|---|---|---|---|
| 2 | E1 | `scripts/experiments/e1_scaling.py` | `docs/phase2/PHASE_2_REPORT.md` | console | logit std 1.85→31.50; Jacobian 0.708→0.014 | **HOLDS** — the √d_k scale prevents saturation |
| 2 | E2 | `e2_causal_leakage.py` | phase2 | console | loss 0.2907 vs **0.0088** | **HOLDS** — a 33.2× better loss meant a broken mask |
| 2 | E4 | `e4_attention_variants.py` | phase2 | `gpu_experiment_results_*.txt` | KV cache → 6.25%; GPU latency flat | memory transfers, **latency does not** |
| 2 | E8 | `e8_kv_cache.py` | phase2 | same | 2.96× CPU, **1.05× GPU** | **overhead-bound** on GPU, not a refutation |
| 2 | E9 | `e9_decoding.py` | phase2 | console | distinct-4 0.257→0.566 | **MEASURED**; found a real top-p bug |
| 2 | E10 | `e10_sdpa_backends.py` | phase2 | same | **56× faster, 147× less memory** | **HOLDS** |
| 2/3 | E11 | `e11_weights_reconciliation.py` | phase2/3 | `docs/phase3/qwen_weights_download.json` | off by **57,344** (QKV biases) | **DISPROVED, then corrected** |
| 3 | E12 | `e12_loss_masking.py` | `docs/phase3/PHASE_3_REPORT.md` | manifest `loss_mask_verified` | prompt **6.3314** vs completion **3.6295** | **H5 DISPROVED**; regions are not comparable |
| 3 | `sft-qwen1p5b-noRobots-001` | `alignlab.sft` | phase3 | `sft_run_{manifest,summary}.json` | 1167.876 s; eval loss 1.9893 | the project's baseline model |
| 3 | E13 | `e13_sft_before_after.py` | phase3 | `e13_before_after.json` | ppl 8.541→**7.199**; stop 0/4→**4/4** | SFT taught termination |
| 3 | E14 | `e14_region_decomposition.py` | phase3 | `e14_region_decomposition.json` | region decomposition | **MEASURED** |
| 4 | E15 | `e15_bitsandbytes_environment.py` | `docs/phase4/PHASE_4_REPORT.md` | `e15_*.json` | quantised matmul **executed** | import ≠ working kernel |
| 4 | E16 | `e16_lora_targets.py` | phase4 | `e16_lora_targets.json` | 112 of 197 `nn.Linear` matched | targets inspected, not assumed |
| 4 | E17 | `e17_svd_rank_selection.py` | phase4 | `e17_svd_rank_selection.json` | 90% energy at rank ~730/1536; r=16 = **10.69%** | **H2 DISPROVED** — ΔW is not low-rank |
| 4 | E18 | `e18_educational_vs_peft.py` | phase4 | `e18_*.json` | matches `peft` in fp32 | **HOLDS (fp32 only)** |
| 4 | E19 + 3 runs | `e19_peft_comparison.py` | phase4 | `e19_peft_comparison.json` | 4,358,144 trainable; bf16 merge **12,460×** worse | **H3 DISPROVED**; evaluate unmerged |
| 4 | E20 | `e20_stop_token_gap.py` | phase4 | `e20_stop_token_gap.json` | `lm_head` relative change **0.0136** | **NOT CONFIRMED** — reachability |
| 5 | E21 | `e21_preference_data_readiness.py` | `docs/phase5/PHASE_5_REPORT.md` | `e21_preference_readiness.json` | 11.9% ties; chosen **56.5% longer** | data **READY** |
| 5 | E22 | `e22_reference_model_and_kl.py` | phase5 | `e22_reference_and_kl.json` | SUM **47.2%**, MEAN **58.3%** | **H4 DISPROVED** — below chance |
| 6 | β sweep (3 runs) | `alignlab.dpo_train` | `docs/phase6/PHASE_6_REPORT.md` | 3 × `*_summary.json` | KL ~**0.0008** | **NULL**; pre-registered |
| 6 | E23 | `e23_dpo_evaluation.py` | phase6 | `e23_dpo_evaluation.json` | **86/184 for every arm** | **NULL RESULT** |
| 6 | `diag-dpo-beta0.1-lr5e-6` | `alignlab.dpo_train` | phase6 | `diag-*_summary.json` | gap +0.70 nats; **42× short** | **POST-HOC** — never the headline |
| 7 | `eval-full-001` | `alignlab.evaluate` | `docs/phase7/PHASE_7_REPORT.md` | `eval-full-001_dashboard.{json,txt}` | **3 of 24** comparisons resolvable | see §7, §9 |
| 8 | provenance audit | `alignlab.provenance` | `docs/phase8/PHASE_8_REPORT.md` | — | **17/18** artefacts complete | the one gap is the known Phase 7 bug |
| 8 | cache cleanup | — | `docs/phase8/CLEANUP_RECORD.md` | — | **18.34 GB** recovered | verified before and after |

---

## 18. Interpretation principles

The project's empirical philosophy, each tied to a real result.

1. **A lower loss is not evidence of better behaviour.** E2: removing the causal
   mask made training loss **33.2× lower** and the model useless.
2. **A metric can be confounded by sequence length.** SUM preference accuracy
   inverts its own verdict on all five models, including the untrained base,
   purely because chosen responses are 56.5% longer.
3. **A successful training run does not imply successful alignment.** All three
   PEFT runs completed cleanly, converged, and produced models that never stop.
4. **Memory reduction does not imply equivalent quality.** QLoRA at ~61% of
   LoRA's VRAM failed termination identically.
5. **Agreement between implementations does not prove correctness.** Three
   attention implementations agreeing to 1e-16 shows they compute the *same*
   thing — E9 still found a real top-p bug, and E18's fp32 agreement did not
   transfer to bf16.
6. **A null result is a result.** The Phase 6 null is the primary conclusion of
   that phase, quantified (728× short on the metric, 250× under-budget in KL),
   not an absence of one.
7. **Do not rewrite a hypothesis after seeing the result.** Five were disproved
   and all five keep their original wording. β was **pre-registered** with a
   commit timestamp as evidence.
8. **Visual and quantitative evidence answer different questions.** The PEFT
   template-regurgitation failure is invisible to every metric here and obvious
   in the text.
9. **An LLM judge is not ground truth.** 33.3% position bias; unvalidated;
   same-family; it could not resolve base-vs-SFT.
10. **Correlation does not establish causation.** E20 is **NOT CONFIRMED** —
    `lm_head` being outside the target set is *consistent with* the termination
    failure, and calling it the cause would need the deferred experiment.
11. **One seed limits confidence.** No result here has a training-variance
    estimate; a 3.6% gap is not called an improvement.
12. **In-distribution evaluation limits generalisation.** Everything is measured
    on the distribution the models trained on, so "better" is partly circular —
    and the dashboard says so in its own output.

---

## 19. How should I read AlignLab?

**If you have 15 minutes:** `README.md` §15 (findings) →
`docs/PROJECT_NARRATIVE.md` → `docs/LIMITATIONS.md`.

**If you want to reproduce something:** this file §3 → §4 → §6 → §16.

**If you want to judge whether the results are trustworthy:**
`docs/EXPERIMENT_REGISTRY.md` (hypotheses as written, before results) →
§9 and §10 here → run `python -m alignlab.provenance`.

**Full path for a new engineer:**

1. `README.md` — what it is and what was found
2. **`HOW_TO_RUN.md`** (this file) — how to run it and how to read results
3. `docs/PROJECT_NARRATIVE.md` — the eight findings in depth
4. `docs/EXPERIMENT_REGISTRY.md` — every hypothesis and verdict
5. the relevant `docs/phase*/PHASE_*_REPORT.md`
6. the committed artefacts in `docs/phase*/` — the actual numbers
7. `scripts/experiments/e*.py` — each script's docstring states its objective,
   hypothesis and configuration before any code
8. `CODE_EXPLANATION/` — how the implementation is organised
9. `PYTORCH_CONCEPTS/` — the mechanics, with runnable examples
10. `STUDY_WITH_CLAUDE/` — the theory and derivations
11. `INTERVIEW_DEFENSE/` — the reasoning under challenge, ending with
    `phase8_project_defence.md`

**Read `docs/LIMITATIONS.md` before forming a conclusion about any number in
this repository.**

---

*Every command, path, config, experiment and number above was verified against
the repository at commit `e1c5cb3`. Where the repository had no answer, this
document says so rather than supplying one.*
