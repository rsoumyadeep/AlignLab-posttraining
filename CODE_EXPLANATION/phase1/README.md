# Phase 1A — Code Explanation

Describes the code that **actually exists** at commit `c8f3855` on branch
`phase-1-foundation`. Nothing here describes planned or imagined code. If this
document and the source ever disagree, the source is right and this document is
stale — check `git log` on the files named below.

---

## 1. Files: what exists and why

| File | Responsibility |
|---|---|
| `src/alignlab/paths.py` | Resolve all storage roots from environment variables. The single place that knows *where things go*. |
| `src/alignlab/env_detect.py` | Decide whether we are `local` or `server`; describe the platform. |
| `src/alignlab/seeding.py` | Seed every RNG; capture and restore RNG state for checkpoints. |
| `src/alignlab/logging_utils.py` | Console + per-run file logging, idempotent. |
| `src/alignlab/device.py` | Resolve a device string to a `torch.device`; describe hardware torch can actually use. |
| `src/alignlab/checkpoint.py` | Atomic save, complete load, rotation, latest-pointer recovery. |
| `src/alignlab/tracking.py` | `Tracker` protocol with noop / W&B-offline / W&B-online backends. |
| `src/alignlab/preemption.py` | Catch a stop signal; checkpoint at a safe point; optional SLURM requeue. |
| `src/alignlab/manifest.py` | Capture full run provenance to `run_manifest.json`. |
| `src/alignlab/config_schema.py` | Typed Hydra schema so config typos fail at compose time. |
| `src/alignlab/evaluation.py` | Rerunnable evaluation harness skeleton. |
| `src/alignlab/train.py` | Entrypoint wiring all of the above around a **toy** training loop. |
| `configs/` | Hydra tree: root + `env` / `train` / `logging` / `tracking` groups. |
| `scripts/env_report.py` | Print observed environment facts. |
| `scripts/server_probe.sh` | Read-only server questionnaire. **NEVER EXECUTED.** |

---

## 2. Data flow

```
CLI args ──> Hydra compose ──> AlignLabConfig (typed)
                                     │
                                     ├─> run_name ──> run_dir()        [paths]
                                     │                    │
                                     │                    ├─> run.log  [logging]
                                     │                    ├─> resolved_config.yaml
                                     │                    ├─> run_manifest.json [manifest]
                                     │                    ├─> metrics.jsonl     [tracking]
                                     │                    └─> eval_results.json [evaluation]
                                     │
                                     ├─> set_seed()                    [seeding]
                                     ├─> resolve_device()              [device]
                                     ├─> build_tracker()               [tracking]
                                     │
                                     └─> training loop
                                              │
                                              ├─> save_checkpoint() ──> ckpt_root/<run>/step-*.pt
                                              ├─> handler.should_stop() [preemption]
                                              └─> run_evaluation()      [evaluation]
```

Two roots, deliberately separate: **run outputs** (small: logs, configs,
manifests, metrics) and **checkpoints** (large). On the server they may need to
live on different volumes, and separating them now costs nothing.

---

## 3. Module interaction

`paths` is the base — it imports nothing from AlignLab. `env_detect` depends on
`paths` for the env-var name only. `logging_utils` is imported by nearly
everything, so it also imports nothing from AlignLab (avoiding an import cycle).
`manifest` sits at the top, pulling from `device`, `env_detect` and `paths`.

The dependency graph is acyclic and shallow by design; a cycle here would
surface as a confusing `ImportError` at entrypoint start.

---

## 4. Key functions

### `paths.py`

- `repo_root()` — walks up from `__file__` looking for `pyproject.toml`.
- `output_root()` / `checkpoint_root()` / `cache_root()` — read
  `ALIGNLAB_OUTPUT_ROOT` / `ALIGNLAB_CKPT_ROOT` / `ALIGNLAB_CACHE_ROOT`,
  falling back to repo-relative defaults.
  **Edge case:** those fallbacks are correct locally and deliberately *wrong*
  on a shared server, which is why `env/server.yaml` makes the values mandatory.
- `generate_run_name(prefix)` — `<prefix>-YYYYmmdd-HHMMSS`; lexicographic order
  matches chronological order. `ALIGNLAB_RUN_NAME` overrides, which is what
  lets a resumed job write back into its original directory.

### `seeding.py`

- `set_seed(seed, deterministic=False) -> SeedReport` — seeds `random`, NumPy,
  torch CPU, all CUDA devices, and `PYTHONHASHSEED`.
  - **Input validation:** rejects non-`int`, and rejects `bool` explicitly —
    `bool` is a subclass of `int`, so `set_seed(True)` would otherwise silently
    seed with 1.
  - **`deterministic=True`** additionally sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`
    (required for deterministic CUDA matmuls on CUDA ≥ 10.2),
    `use_deterministic_algorithms(warn_only=True)`, and disables cuDNN
    benchmarking. Off by default because it costs throughput.
  - **Known limitation:** `PYTHONHASHSEED` set at runtime affects only
    subprocesses spawned afterwards, not the running interpreter.
- `capture_rng_state()` / `restore_rng_state()` — CUDA state restored only if
  captured *and* CUDA is available now, so a server checkpoint stays loadable on
  the local CPU machine.

### `checkpoint.py`

- `save_checkpoint(...) -> Path` — writes `step-{step:09d}.pt` via temp file +
  `os.replace`, then writes `latest.txt`.
  - **9-digit zero padding** so filenames sort numerically; without it `step-9`
    sorts after `step-10` and rotation deletes the wrong file. Tested.
  - **`latest.txt` is a text file, not a symlink** — symlinks need elevated
    privileges on Windows.
- `load_checkpoint(...) -> CheckpointPayload` — `map_location="cpu"` by default
  so a GPU-written checkpoint is inspectable locally.
  - **`weights_only=False`** is required: AlignLab checkpoints legitimately
    contain non-tensor objects (RNG state tuples, the config dict). Safe here
    because we only load checkpoints this project wrote. **Never point this at
    an untrusted file.**
- `latest_checkpoint(dir)` — prefers `latest.txt`, falls back to filename order
  if the pointer is missing or stale, so a run killed between the rename and the
  pointer write is still resumable. Tested.

### `preemption.py`

- `available_preemption_signals()` — `getattr(signal, name, None)` rather than a
  hard reference, so the module imports on Windows where `SIGUSR1` is absent.
- `PreemptionHandler._on_signal` — sets a flag and returns. **No I/O.** See
  the design decision below.
- `request_stop(reason)` — the OS-independent trigger that makes the logic
  testable without a real signal.
- `_requeue()` — invokes `scontrol requeue $SLURM_JOB_ID`.
  **STATUS: IMPLEMENTED, NOT TESTED.** No SLURM has been available.

### `manifest.py`

- `config_hash(config)` — SHA-256 over the config with sorted keys, so two
  structurally identical configs hash identically. This is the mechanism behind
  Tier B cross-machine reproducibility.
- `git_info()` — commit, branch, **dirty flag**, and the dirty file list. The
  dirty flag matters more than the SHA: a result from a dirty tree is not
  reproducible from the recorded commit, and the manifest must say so.
- `relevant_env_vars()` — an **allowlist**, not `os.environ`. Dumping the whole
  environment would leak credentials into a file meant to be shareable.

---

## 5. Mathematics → code

Phase 1 contains no ML mathematics; the toy loop optimises a plain MSE:

```
objective        L(w) = (1/N) Σ (x_i·w − y_i)²
   ↓ loss        criterion = torch.nn.MSELoss(); loss = criterion(model(x), y)
   ↓ gradient    loss.backward()                      ∂L/∂w into p.grad
   ↓ clipping    clip_grad_norm_(params, 1.0)         rescale if ‖g‖ > 1
   ↓ update      optimizer.step()                     AdamW
   ↓ schedule    scheduler.step()                     StepLR, ×0.5 every 10
```

This mapping is trivial *here* on purpose. The same skeleton is where the SFT
cross-entropy loss (Phase 3) and the DPO objective (Phase 6) will be
substituted, and those documents will fill this section properly.

---

## 6. Design decisions

**Hydra over argparse / pydantic-settings.** Chosen for config *groups* — the
`env` group is what swaps every machine-specific value with one CLI token.
Rejected: argparse (no composition, and 30+ flags is unreadable);
pydantic-settings (good validation, no group composition); a bare YAML file
(no override mechanism, no schema). Cost accepted: Hydra's working-directory
and logging behaviour needed explicit overriding — see below.

**`hydra/job_logging: none`.** Hydra's default job logging installs a *second*
file handler writing `train.log` into the working directory, giving two
divergent logs per run. First attempt used `disabled`, which sets
`disable_existing_loggers: true` and silently killed AlignLab's own logger —
the run produced *no output at all*. `none` (apply no logging config) is
correct. Preserved here because the failure mode was completely silent.

**A `Tracker` protocol rather than calling `wandb` directly.** The server's
outbound network access is UNVERIFIED. Training code must not hard-depend on a
reachable tracking service. The interface is deliberately four methods to avoid
premature abstraction. Rejected: calling `wandb` directly (untestable without
credentials); MLflow (heavier, no benefit at this scale).

**Signal handler sets a flag only.** See §5. Rejected: checkpointing inside the
handler — risks a corrupt file and an allocator deadlock, and would make the
logic untestable on Windows.

**Two lockfiles, one `pyproject.toml`.** The library versions must match across
machines; only the torch build variant may differ. Rejected: one lockfile
(cannot express `+cpu` vs CUDA); no lockfile (defeats reproducibility).

**`requires-python = ">=3.10"`.** The local venv is 3.11, but the server's
Python is **unverified** and Ubuntu 22.04 ships 3.10. Pinning `>=3.11` could
lock us out of the primary training environment. To be revisited after probing.

---

## 7. Edge cases handled

| Edge case | Handling | Tested |
|---|---|---|
| `SIGUSR1` absent (Windows) | `getattr` lookup; register what exists | yes |
| Process killed mid-checkpoint-write | temp file + `os.replace` | yes (no `.tmp` debris) |
| Stale `latest.txt` pointer | fall back to filename order | yes |
| Step ≥ 10 filename sorting | 9-digit zero padding | yes |
| `setup_logging` called repeatedly | tagged handlers removed first | yes |
| Foreign log handlers (pytest) | only tagged handlers removed | yes |
| `device=cuda` on a CPU-only box | raises, never silent CPU fallback | yes |
| CUDA RNG restored on a CPU machine | skipped with a warning | via code path |
| Device-count mismatch on RNG restore | caught, warned, not fatal | no — needs 2 GPUs |
| Resuming an already-complete run | 0 steps, `final_loss=None`, no new checkpoint | yes |
| Broken evaluator | recorded `NOT_TESTED`, others still run | yes |
| `wandb` requested but absent | falls back to noop **with a loud warning** | partial |
| Server paths unset | Hydra `MissingMandatoryValue` | yes |
| Config typo | compose-time failure | yes |

---

## 8. Test map

| Test file | Covers | Count |
|---|---|---|
| `test_seeding.py` | Tier A bitwise, incl. across separate processes | 10 |
| `test_checkpoint.py` | round-trip, rotation, atomicity, pointer recovery | 14 |
| `test_config.py` | composition, overrides, server MISSING guard | 10 |
| `test_paths_logging_device.py` | roots, env detection, logging, device | 25 |
| `test_preemption.py` | signals, decision logic, SLURM detection | 13 |
| `test_tracking_manifest_eval.py` | trackers, manifests, eval harness | 19 |
| `test_no_hardcoded_paths.py` | repo-wide absolute-path scan | 5 |
| `test_smoke_train.py` | end-to-end train → checkpoint → resume | 11 |

**Result at commit `c8f3855`: 101 passed, 2 skipped, ~21 s** on Windows CPU.
The 2 skips are visible in the `-ra` summary: one needs CUDA, one needs
`SIGUSR1`.

---

## 9. Staleness watch

Re-check this document when: the config schema changes, a storage root is
added, the checkpoint format version increments past 1, or `env/server.yaml`
is populated after the server probe.

**Related:** [[phase1-foundation]] · [[pytorch-rng-and-state]] ·
[[interview-phase1-engineering]]
