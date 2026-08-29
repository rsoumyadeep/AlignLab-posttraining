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

## ⚠ Current status — Phase 1A complete

This repository currently contains **the engineering foundation only**.

**There is no modelling code yet.** No Qwen weights have been downloaded, no
tokenizer is used, no attention is implemented, and no training has been run.
The `train.py` entrypoint exercises the foundation against a deliberately tiny
synthetic linear regression.

| Phase | Status |
|---|---|
| 1A — Foundation (local) | ✅ **COMPLETE**, 101 tests passing |
| 1B — Foundation (server) | ⏸ **DEFERRED** — awaiting server access |
| 2 — Transformer understanding | ⬜ not started |
| 3 — SFT | ⬜ not started |
| 4 — PEFT (LoRA / QLoRA) | ⬜ not started |
| 5 — Preference learning / RLHF | ⬜ not started |
| 6 — DPO | ⬜ not started |
| 7 — Evaluation | ⬜ skeleton only |
| 8 — Engineering polish | ⬜ not started |

---

## Two environments

AlignLab runs in two places with deliberately different roles.

| | Local (development) | Server (training) |
|---|---|---|
| Hardware | GTX 1050, 4 GB, Pascal | 2 × RTX A6000, 48 GB, Ampere |
| torch | 2.13.0**+cpu** | not yet installed |
| bf16 / FlashAttention-2 | no | yes (hardware-eligible) |
| Role | code, docs, fast CPU tests | all real training |
| Status | ✅ verified | ⏸ **never accessed** |

The local 4 GB GPU cannot train Qwen2.5-1.5B in any configuration — bf16
weights alone are ≈3.1 GB before gradients, optimizer state or activations. It
is a development machine, permanently.

**Everything known about the server comes from a hardware report generated
2026-05-07.** That is a hardware reference, not evidence the machine is
reachable today. Run `scripts/server_probe.sh` before relying on any of it.

---

## Quickstart (local)

```bash
uv venv --python 3.11 .venv
uv pip install -e ".[tracking,dev]"

python scripts/env_report.py          # what this machine actually has
python -m pytest -q                   # 101 passed, 2 skipped
python -m alignlab.train              # foundation smoke run (toy model)
```

Overrides use Hydra syntax:

```bash
python -m alignlab.train env=local train.max_steps=50 reproducibility.seed=7
python -m alignlab.train tracking=wandb_offline
```

---

## Layout

```
src/alignlab/       foundation modules (see CODE_EXPLANATION/phase1/)
configs/            Hydra tree; env/ group absorbs machine differences
tests/              101 tests, CPU-only, no network or credentials needed
scripts/            env_report.py, server_probe.sh (NEVER EXECUTED)
docs/phase1/        phase reports

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

Currently `NOT TESTED`: W&B online mode, the SLURM requeue path, every GPU code
path, and `scripts/server_probe.sh`.

---

## License

MIT.
