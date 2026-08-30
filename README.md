# AlignLab

An end-to-end **LLM post-training** framework built from first principles.

AlignLab does not pretrain a language model. Its scope begins after
pretraining, from the base checkpoint `Qwen/Qwen2.5-1.5B` (the **base** model,
not `-Instruct`):

```
Large-scale pretraining          [OUTSIDE ALIGNLAB]
        │
        ▼
Qwen/Qwen2.5-1.5B  ──►  SFT  ──►  PEFT (LoRA / QLoRA)  ──►  Preference
                                                             learning
                                                                │
                                        Evaluation  ◄──  DPO  ◄─┘
```

The goal is not a working checkpoint. It is understanding, first-principles
implementation, realistic ML engineering, and empirical evaluation — such that
the work can be reproduced, explained, defended and extended.

---

## Current status — Phase 3 (SFT) complete

**Phase 1** built the engineering foundation. **Phase 2** built the Transformer
components from first principles: attention in PyTorch/NumPy/pure Python,
RoPE/ALiBi/sinusoidal, RMSNorm, SwiGLU, a decoder-only model, a KV cache, and
five decoding strategies.

**Phase 3 fine-tunes the real model.** Qwen2.5-1.5B weights (2.886 GiB, pinned
revision `8faed761`) are downloaded, and the base model is supervised
fine-tuned on `HuggingFaceH4/no_robots` using TRL's `SFTTrainer` — with the
loss mask verified against an independent computation before every run, because
PROJECT_INSTRUCTIONS §3 makes that a precondition for PEFT.

No LoRA, no QLoRA, no DPO yet.

| Phase | Status |
|---|---|
| 1A — Foundation (local) | ✅ **COMPLETE** |
| 1B — Foundation (server) | ✅ **COMPLETE**, 110 passed / 1 skipped on GPU server |
| 1C — Version alignment, storage policy, path-wiring fix | ✅ **COMPLETE** |
| 2 — Transformer understanding | ✅ **COMPLETE** (impl/exp/docs); USER explain-backs deferred |
| 3 — SFT | ✅ **COMPLETE** (impl/exp/docs); USER explain-backs deferred |
| 4 — PEFT (LoRA / QLoRA) | ✅ **COMPLETE** (impl/exp/docs); USER explain-backs deferred |
| 5 — Preference learning / RLHF | ✅ **COMPLETE** (infra/measurements/docs); PPO conceptual only |
| 6 — DPO | ✅ **COMPLETE** (impl/sweep/docs); pre-registered sweep returned a characterised null |
| 7 — Evaluation | ⬜ skeleton only |
| 8 — Engineering polish | ⬜ not started |

---

## Two environments

AlignLab runs in two places with deliberately different roles.

| | Local (development) | Server (training) |
|---|---|---|
| Hardware | GTX 1050, 4 GB, Pascal | 2 × RTX A6000, 48 GB, Ampere |
| torch | **2.6.0+cpu** | **2.6.0+cu124** (CUDA 12.4, cuDNN 9.1.0) |
| bf16 | no | **yes — verified, matmul executed** |
| Role | code, docs, fast CPU tests | all real training |
| Status | ✅ verified | ✅ **verified on `csrslave`** |

The local 4 GB GPU cannot train Qwen2.5-1.5B in any configuration — bf16
weights alone are ≈3.1 GB before gradients, optimizer state or activations. It
is a development machine, permanently.

The server (`csrslave`) was audited live on 2026-08-29; verbatim probe output is
committed in `docs/phase1/`. Three findings shape the project:

- **SLURM is installed but NON-FUNCTIONAL** — `Slurmctld(primary) at csrmaster
  is DOWN`. Jobs run directly under `tmux`. Nothing enforces GPU allocation, and
  8 other users share the machine: check `nvidia-smi` before every launch.
- **`/data` is 99% full (~77 GB free)** and shared. Disk, not VRAM, is the
  binding constraint. Checkpoint rotation is capacity planning here.
- **SIGUSR1 preemption is verified** by external `kill -USR1` on a live run —
  no scheduler needed.

---

## Quickstart (local)

```bash
uv venv --python 3.11 .venv
uv pip install -e ".[tracking,dev]"          # local: CPU torch

python scripts/env_report.py          # what this machine actually has
python -m pytest -q                   # 499 passed, 2 skipped
python -m alignlab.train              # foundation smoke run (toy model)
```

Overrides use Hydra syntax:

```bash
python -m alignlab.train env=local train.max_steps=50 reproducibility.seed=7
python -m alignlab.train tracking=wandb_offline
```

### Fine-tuning (Phase 3 / Phase 4), server only

```bash
python -m alignlab.sft env=server                    # full-parameter SFT
python -m alignlab.sft env=server peft=lora          # LoRA  r=16, attention
python -m alignlab.sft env=server peft=qlora         # QLoRA 4-bit NF4 base
python -m alignlab.sft env=server peft=lora peft.r=64 sft=smoke
```

The three arms differ in exactly one thing — how the frozen base is stored —
and a test asserts that every adapter setting is shared between `lora` and
`qlora`. Every run re-verifies its own loss mask before the first optimiser
step and refuses to train if it disagrees with an independent computation.

### Server (`csrslave`)

```bash
ssh server && cd /data/home/rsoumyadeep/AlignLab
uv venv --python 3.11 .venv
UV_CACHE_DIR=/tmp/uv-cache uv pip install \n    --index-url https://download.pytorch.org/whl/cu124 torch
UV_CACHE_DIR=/tmp/uv-cache uv pip install -e ".[tracking,dev]"

.venv/bin/python -m pytest -q                       # 500 passed, 1 skipped
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m alignlab.train env=server
```

`UV_CACHE_DIR=/tmp` matters: `/tmp` is a separate 1.6 TB NVMe, so ~5 GB of
package cache stays off the 99%-full `/data` volume. Delete it afterwards.

---

## Layout

```
src/alignlab/       foundation modules (see CODE_EXPLANATION/phase1/)
src/alignlab/models/ Transformer components, Phase 2 (attention x3, RoPE/ALiBi,
                     RMSNorm, SwiGLU, decoder-only model, KV cache, decoding)
src/alignlab/lora.py first-principles LoRA (Phase 4), verified against peft
src/alignlab/preference.py  preference data + Bradley-Terry (Phase 5)
src/alignlab/logprobs.py    sequence log-probs, KL - the DPO/PPO arithmetic
src/alignlab/dpo.py         first-principles DPO loss (Phase 6), verified vs the paper
src/alignlab/dpo_train.py   DPO training loop - our loss, not TRL's trainer
src/alignlab/sft.py  SFT/LoRA/QLoRA entrypoint with three pre-flight audits
configs/            Hydra tree; env/ group absorbs machine differences
tests/              501 tests; no network or credentials needed
scripts/            env_report.py, server_probe.sh (executed on csrslave)
docs/phase1/        phase reports, storage policy, server probe evidence
docs/phase2/        Phase 2 report + verbatim GPU experiment output
docs/phase3/        Phase 3 report, weight-download evidence, before/after eval
docs/phase4/        Phase 4 report, bitsandbytes verification, SVD rank analysis
docs/phase5/        Phase 5 report, preference-data readiness, KL measurements
docs/phase6/        Phase 6 report, beta pre-registration, DPO sweep evidence

STUDY_WITH_CLAUDE/  theory, intuition, derivations + USER checkpoints
CODE_EXPLANATION/   what the code actually does (never imagined code)
PYTORCH_CONCEPTS/   concepts with EXECUTED examples and real output
LEARNING_RESOURCES/ honest audit of external resources
INTERVIEW_DEFENSE/  reasoning questions, not definitions
```

---

## Reproducibility, stated precisely

Three tiers, because claiming one guarantee where only another holds is the
most common way reproducibility claims become false:

- **Tier A — bitwise, same machine.** Same seed, same environment, bitwise
  identical. Achievable and tested across separate processes.
- **Tier B — structural, across machines.** Same commit, config hash, data
  version and seed → same code path and shapes. Verified by comparing manifests,
  not floats.
- **Tier C — statistical, across machines or dtypes.** Comparable in
  distribution only. Any CPU-fp32 vs Ampere-bf16 comparison is Tier C.

Every run writes a `run_manifest.json` recording the git SHA **and dirty flag**,
resolved config hash, hardware, library versions and seed.

---

## Evidence vocabulary

Used strictly and never interchangeably: `IMPLEMENTED` (code exists),
`VERIFIED` (tested, behaved as expected), `MEASURED` (a quantitative result was
actually obtained), `RECORDED`, `UNVERIFIED`, `NOT TESTED`, `DEFERRED`.

Currently `NOT TESTED`: W&B **online** mode (no API key on either machine), the
SLURM requeue path (no working scheduler exists to test against), and multi-GPU
RNG restore. **Now VERIFIED:** CUDA, bf16, GPU smoke run, real SIGUSR1
preemption, W&B offline.

**Resolved in Phase 1C:** both machines now run **torch 2.6.0**, differing only
in build variant (`+cpu` vs `+cu124`). Measured result: this did **not** produce
bitwise cross-machine agreement (~1e-7 divergence remains, platform/BLAS-level),
but it removes API and default drift between torch majors — which is what
actually threatened Tier B.

**Storage:** `/data` on the server is 99% full and shared with 8 users. See
[`docs/phase1/STORAGE_POLICY.md`](docs/phase1/STORAGE_POLICY.md). Phase 3
replaced the estimate with a **measurement**: a resumable full-SFT checkpoint
for Qwen2.5-1.5B is **8.63 GiB** (2.00 B/param bf16 weights + 4.00 B/param bf16
AdamW moments), not the ~15.5-20 GB estimated. `alignlab.storage` keeps the
conservative estimate as its default anyway — a guard that under-predicts fails
mid-write, which is not symmetric with refusing a run that would have fitted —
and now **refuses to start** a run whose checkpoints would not fit.

---

## License

MIT.
