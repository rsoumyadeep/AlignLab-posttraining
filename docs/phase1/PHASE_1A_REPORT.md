# Phase 1A Report

**Date:** 2026-08-29
**Branch:** `phase-1-foundation`
**Scope:** local development environment only. Phase 1B (server) is DEFERRED.

---

## 1. What Was Built

The engineering foundation, and nothing else. No modelling code.

**Source — `src/alignlab/` (12 modules)**

| Module | Responsibility |
|---|---|
| `paths.py` | All storage roots resolved from environment variables |
| `env_detect.py` | `local` vs `server` detection; platform description |
| `seeding.py` | `set_seed`, RNG capture/restore, Tier A/B/C definition |
| `logging_utils.py` | Console + per-run file logging, idempotent |
| `device.py` | Device resolution; hardware description |
| `checkpoint.py` | Atomic save, complete load, rotation, pointer recovery |
| `tracking.py` | `Tracker` protocol: noop / wandb-offline / wandb-online |
| `preemption.py` | Signal binding separated from checkpoint-and-exit logic |
| `manifest.py` | Run provenance to `run_manifest.json` |
| `config_schema.py` | Typed Hydra schema |
| `evaluation.py` | Rerunnable evaluation harness skeleton |
| `train.py` | Entrypoint wiring all of the above around a **toy** loop |

**Configuration — `configs/`:** root config plus `env` / `train` / `logging` /
`tracking` groups. `env/local.yaml` holds verified local values;
`env/server.yaml` holds Hydra `MISSING` for every path.

**Tests — `tests/`:** 8 files, 103 collected.

**Scripts:** `env_report.py` (executed), `server_probe.sh` (**never executed**).

**Documentation:** all five mandated folders, plus `README.md` and this report.

---

## 2. What Was Verified

`VERIFIED` = actually executed and behaved as expected.

- Hydra composition, CLI override, config-group switching
- Typed schema rejects an unknown key at compose time
- `env=server` composes structurally **but raises `MissingMandatoryValue`** on
  every path — the guard against fabricated server paths
- **Tier A bitwise reproducibility across two separate Python processes**
- A different seed produces a different result (seed is genuinely applied)
- RNG state capture/restore reproduces the exact stream
- Checkpoint round-trip: model, optimizer moments, scheduler, step, RNG state —
  compared with `torch.equal`, bitwise
- Atomic write leaves no `.tmp` debris
- Rotation keeps the newest N; 9-digit padding sorts numerically past step 10
- Stale `latest.txt` pointer recovers via filename order
- Logging writes console + file, file more verbose, no duplicate handlers on
  repeated setup, foreign (pytest) handlers left intact
- `device=cuda` on a CPU-only machine **raises** rather than silently degrading
- `describe_hardware` reports no GPU here — correct, since the installed torch
  cannot use the GTX 1050 that is physically present
- Tracker: `noop` and **`wandb` offline** (no credentials, no network)
- `metrics.jsonl` appends independently of the tracker backend
- Manifest captures git SHA + dirty flag, config hash, hardware, versions;
  env-var capture is an allowlist (no credential leakage)
- Config hash is key-order independent
- Preemption: signal registration, restoration, decision logic, one-shot
  checkpoint, SLURM detection from env
- Evaluation harness: results collected; a **broken evaluator is recorded as
  `NOT_TESTED` without discarding the metrics that worked**
- **No hard-coded machine-specific absolute paths** anywhere in
  `src/`, `tests/`, `configs/`, `scripts/` — enforced by a test
- End-to-end smoke: train → checkpoint → resume → weights match bitwise;
  simulated preemption stops early and leaves a resumable checkpoint

---

## 3. What Was Measured

Only foundation measurements. **No model quality metric exists.**

| Measurement | Value |
|---|---|
| Test suite | **101 passed, 2 skipped**, 103 collected |
| Suite wall time | **~21–22 s**, CPU |
| Local Python | 3.11.15 |
| Local torch | **2.13.0+cpu**, `cuda.is_available() == False` |
| Preemption signals available locally | `SIGTERM`, `SIGBREAK` (**no `SIGUSR1`**) |
| torch RNG state size | **5056-byte uint8 CPU tensor** |
| fp32 / bf16 / fp16 mantissa bits | ~23 / **~7** / ~10 |
| fp16 max | 65504 (1e30 → `inf`); bf16 max ≈ 3.39e38 |

`NOT MEASURED`: checkpoint size or write time at realistic model scale, GPU
memory, throughput, any language-model metric.

---

## 4. Results

The foundation runs end to end on CPU. A run produces `run.log`,
`resolved_config.yaml`, `run_manifest.json`, `metrics.jsonl`,
`eval_results.json`, and rotated checkpoints.

Resume verified from the CLI: a run stopped at step 20, re-invoked with
`max_steps=35`, resumed at 20 and appended steps 25/30/35, with rotation
retaining exactly the newest three checkpoints.

**Interpretation.** This demonstrates that the scaffolding holds together. It
demonstrates **nothing** about language modelling — the workload is a
four-parameter linear regression on synthetic data.

---

## 5. Unexpected Findings

1. **`hydra/job_logging: disabled` silently destroys logging.** It sets
   `disable_existing_loggers: true`, which disabled AlignLab's own logger. The
   run completed successfully and printed **nothing at all**. `none` is the
   correct value. A completely silent failure mode.
2. **torch RNG state is 5056 bytes**, not a small integer — concrete evidence
   that "just save the seed" cannot restore a stream mid-run.
3. **bf16 is *less* precise than fp16** (~7 vs ~10 mantissa bits). Its advantage
   is exponent *range*, not precision. Easy to state backwards.
4. **Comparing two optimizer `state_dict`s with `==` raises**
   `RuntimeError: Boolean value of Tensor with more than one value is
   ambiguous`. Nested tensors need `torch.equal` element-wise.
5. **Two consecutive attempts at a floating-point non-associativity example both
   failed**, each producing `0.0` for both orderings because fp32 rounding
   collapsed the groupings. The reasoning sounded right both times.

---

## 6. Corrections / Bugs Discovered

**Bug 1 — `.gitignore` case-collision (would have broken portability).**
The conventional virtualenv entry `ENV/` also matched `configs/env/`, because
Git is case-insensitive on Windows. The entire environment config group was
being silently excluded from version control — the foundation would have been
committed *without the two files that make it portable*, and would have looked
correct locally.
*Found by:* running `git status --ignored` as an explicit verification step
rather than reading the ignore file and assuming.
*Fix:* anchored `/​.venv/`, `/venv/`, `/ENV/` with a comment recording why.

**Bug 2 — NaN reported as if measured.**
Re-running a completed run resumed at the step budget, executed zero steps, and
returned `final_loss: nan`, which in a results table reads like a measurement
that came out badly.
*Fix:* `final_loss` is now `None`, `steps_this_invocation` makes the no-op
explicit, a warning is logged, and no redundant checkpoint is written (which
would have rotated a genuinely older one out of existence). Two regression
tests added.

**Test bugs, fixed:** optimizer state compared with `==` (→ `torch.equal`
element-wise); handler count assertion included pytest's own handlers (→ count
only AlignLab-tagged handlers, which also confirmed foreign handlers survive).

**Documentation bugs, fixed:** three PyTorch examples were wrong on first
execution (two non-associativity attempts, one dtype comparison using constants
that coincidentally round identically in bf16 and fp16). All three failures are
preserved in the document rather than quietly replaced.

---

## 7. Limitations

1. **Nothing has run on a GPU.** Every CUDA path is untested.
2. **Nothing has run on the server.** Phase 1B entirely deferred.
3. **W&B online is `NOT TESTED`** — needs an API key and outbound access.
4. **SLURM requeue is `IMPLEMENTED, NOT TESTED`** — no scheduler available.
5. **`SIGUSR1` delivery is untestable locally** — the signal does not exist on
   Windows. The test skips visibly and runs on POSIX.
6. **Toy workload only.** Checkpoint size/time, memory and throughput at 1.5B
   are unknown.
7. **No dataset versioning.** A real reproducibility gap; Phase 3 must add a
   fingerprint.
8. **Dataloader worker RNG is not captured**, so resume is not bit-exact for a
   multi-worker input pipeline.
9. **No distributed support** — no DDP, no distributed sampler, no sharded
   checkpointing.
10. **`weights_only=False` on load** — a deliberate trade to store RNG tuples
    and the config; safe only because we load our own files.
11. **Multi-GPU RNG-restore fallback has never executed** (needs 2 GPUs).

---

## 8. Resources Actually Inspected

**None.**

No paper, documentation page, repository, blog post or video was accessed
during Phase 1A. No browsing tool was used. The implementation was written from
working knowledge and verified by **executing it**.

`LEARNING_RESOURCES/resources.md` therefore lists every entry as
`NOT INSPECTED`, with the specific question each is expected to answer when it
is eventually opened.

The evidence base for Phase 1A is the 101 passing tests and the executed
example scripts — not citations.

---

## 9. Git / Synchronization State

Read from actual command output, not asserted.

```
Branch:   phase-1-foundation
HEAD:     95209dfbfa7ee443dfdf78e17c82152df5e81e52
          (95209df  docs(phase-1a): documentation folders, README and report)
Commits:  d21c070  baseline (inputs only: instructions, server report, hygiene)
          c8f3855  feat(phase-1a): engineering foundation
          95209df  docs(phase-1a): five documentation folders + README + report
Merged:   nothing merged to main; main remains at d21c070
Unmerged: phase-1-foundation is ahead of main and NOT merged
Remote:   none configured — no push has occurred, nothing is synchronized
          anywhere off this machine
Deferred: Phase 1B branch not created
```

**No remote exists.** No claim of synchronization is made.

---

## 10. Deferred Explain-Back Checkpoints

**None of the five mandatory drills is complete.** All belong to Phases 2–4.

Open USER checkpoints created in Phase 1A, all **UNCOMPLETED**:

| Location | Checkpoint |
|---|---|
| `STUDY_WITH_CLAUDE/phase1_foundation.md` §3 | *My Understanding* — why bitwise reproducibility holds on one machine but not across two |
| `STUDY_WITH_CLAUDE/phase1_foundation.md` §4 | *Explain Back* — what is silently wrong with a weights-only checkpoint |
| `STUDY_WITH_CLAUDE/phase1_foundation.md` §6 | *My Understanding* — why `???` beats a sensible default |
| `LEARNING_RESOURCES/resources.md` | *Own Words* — one entry per resource, after reading it |
| `INTERVIEW_DEFENSE/phase1_engineering.md` | *Explain Back* — Q2, Q4, Q8 aloud, plus two unanswered questions |

Per the explain-back protocol, these are **not** complete merely because the
material is documented. They require the USER to demonstrate understanding.

---

## 11. Confirmations

- **No server connection was attempted.** No SSH, no network call to
  `10.192.12.132` or any other host. `scripts/server_probe.sh` is committed and
  **has never been executed anywhere**.
- **No model was downloaded.** `Qwen/Qwen2.5-1.5B` is not present. The HF cache
  contains only pre-existing unrelated datasets. `transformers`, `datasets`,
  `peft` and `trl` are **not installed** in the venv.
- **No real training was performed.** The only training was a four-parameter
  linear regression on synthetic data, recorded in every manifest as
  `is_real_training: false`, `model_downloaded: false`.
- **Nothing from `server_system_info.txt` is marked VERIFIED.** It is cited only
  as `VERIFIED FROM SERVER REPORT (2026-05-07)` — a hardware reference, never
  evidence of current availability.
