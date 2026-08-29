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

---

# Phase 1B — What Changed on the Server

Describes the code as of commit `b352048`. Phase 1A sections above remain
accurate; this section records only what Phase 1B altered.

## 10. `configs/env/server.yaml` — placeholders replaced

The three `???` (Hydra `MISSING`) values are now populated. Each is backed by a
line in the committed probe evidence.

| Key | Value | Evidence |
|---|---|---|
| `output_root` | `…/AlignLab/outputs` | repo cloned and verified at that path; `/data/home/rsoumyadeep  EXISTS, WRITABLE` |
| `checkpoint_root` | `…/AlignLab/checkpoints` | same volume; no separate scratch volume exists on this host |
| `cache_root` | `…/AlignLab/.cache/huggingface` | deliberately **not** `~/.cache/huggingface` — `~/.cache` is already 24 GB on a volume with 77 GB free |
| `device` | `auto` | `torch.cuda.is_available()` → `True` |
| `num_workers` | `4` | `nproc` → 64, but 8 other users are active; a polite fraction |
| `expect_cuda` | `true` | verified |

The file's header carries the full verified hardware block so the config is
self-documenting: a reader does not have to cross-reference the probe output to
know why a value is what it is.

## 11. Two tests changed — and why that was legitimate

Both changes were forced by the placeholder→verified transition. Neither was
made to silence a failure.

### `test_server_paths_are_mandatory_and_unset` → `test_server_paths_are_populated_and_absolute`

The original asserted every server path raised `MissingMandatoryValue`. Its own
docstring set the condition for change: *"only once `scripts/server_probe.sh`
has been run and the values are backed by real output."* That condition is met.

The invariant moves from **"must be absent"** to **"must be real"**: set,
absolute POSIX, and not a disguised placeholder (`???`, `TODO`, `CHANGEME`).
Two tests were added alongside it:

- `test_server_roots_are_distinct` — outputs, checkpoints and cache must not
  collide. Sharing a directory would let checkpoint rotation delete run logs,
  and would make the cache indistinguishable from results when pruning. That
  matters on a volume at 99%.
- `test_local_paths_remain_env_var_driven` — populating the *server* config
  must not tempt anyone into hard-coding the *local* one. Local stays empty,
  meaning "defer to `ALIGNLAB_*` env vars, then repo-relative".

### `test_no_hardcoded_absolute_paths` — scope corrected

This failed on the populated server config:

```
configs/env/server.yaml:47: output_root: /data/home/rsoumyadeep/AlignLab/outputs
```

**The test was wrong, not the config.** `configs/env/` is by design the single
place machine-specific values may live — that is the entire purpose of the env
config group. The scan forbade exactly what the design exists to permit. It had
passed in Phase 1A only because `server.yaml` held placeholders, so the rule was
never actually exercised against its intended exception.

Fix: `configs/env/` is excluded via an explicit `EXCLUDED_FROM_SCAN` tuple. The
guard remains fully in force for `src/`, `tests/`, `scripts/` and the rest of
`configs/`.

To stop that exclusion from silently widening later, a complementary test was
added — `test_machine_specific_paths_are_confined_to_env_group`. It re-runs the
scan with exclusions **off** and asserts every hit lies under `configs/env/`.
Adding a directory to the exclusion list therefore cannot quietly open a hole:
the confinement test would fail.

## 12. `requirements-server-cu124.txt` — new

Generated by `uv pip freeze` from the venv that actually ran the suite. Header
records host, Python, torch build, GPUs, driver, and the exact install commands.

It also records a **known divergence** rather than hiding it: local torch is
`2.13.0+cpu`, server torch is `2.6.0+cu124`. These differ in version, not just
build variant, because the cu124 index caps at 2.6.0 (verified by resolver
error). This weakens the Tier B guarantee and is flagged as unresolved.

## 13. Updated edge-case status

Rows from §7 whose status changed once real hardware was available:

| Edge case | Phase 1A | Phase 1B |
|---|---|---|
| `SIGUSR1` present | skipped on Windows | **VERIFIED** — external `kill -USR1` on a live run |
| GPU described when present | skipped (no CUDA) | **VERIFIED** — 2 × A6000, cc 8.6 |
| bf16 support | NOT TESTED | **VERIFIED** — flag `True` *and* a bf16 matmul executed |
| `device=cuda` raises without CUDA | verified locally | correctly **skips** on server (CUDA present) |
| CUDA RNG restore | code path only | still **NOT TESTED** — no multi-GPU restore exercised |
| Device-count mismatch on RNG restore | not tested | still **NOT TESTED** |

## 14. Test counts by environment

Same 106 tests collected on both machines; the difference is entirely
environmental and visible in the skip reasons.

| | Local (Windows, CPU) | Server (Linux, 2×A6000) |
|---|---|---|
| Collected | 106 | 106 |
| Passed | 104 | **105** |
| Skipped | 2 | 1 |
| Failed | 0 | 0 |
| Skip reasons | needs CUDA; `SIGUSR1` absent | CPU-only assertion (CUDA *is* present) |

The two tests skipped locally both **ran and passed** on the server. That is the
point of marking skips visibly rather than letting them pass silently.

**Related:** [[phase1b-two-environments]] · [[pytorch-rng-and-state]]

---

# Phase 1C — Version Alignment and Storage Wiring

## 15. `paths.configure_hf_cache()` — new

**Why it exists.** `configs/env/server.yaml` declared `cache_root`, but the
Hugging Face libraries read *environment variables*, not our config. Nothing
bridged the two: `manifest.py` only **recorded** `HF_HOME`; nothing **set** it.
The first Phase 3 download would therefore have gone to `~/.cache/huggingface`
— on the server, a directory already holding **5.3 GB of other projects'
models** on a volume with 77 GB free.

**What it sets, and what it deliberately does not:**

| Variable | Set? | Reason |
|---|---|---|
| `HF_HUB_CACHE` | ✅ `<cache_root>/hub` | where model blobs land |
| `HF_DATASETS_CACHE` | ✅ `<cache_root>/datasets` | where dataset arrow files land |
| `HF_HOME` | ❌ **never** | it also governs the stored-credentials file; relocating it would break an existing `huggingface-cli login` |

**Precedence:** an already-set variable is respected, never overwritten — an
operator who exported `HF_HUB_CACHE` deliberately outranks our config.

**Call site:** `train.train()`, immediately after the run directory is resolved
and **before** anything could trigger a download.

**Tests (4):** variables set correctly; `HF_HOME` untouched; existing value
respected; falls back to `cache_root` when no explicit root is given.

## 16. Manifest allowlist gap — found and fixed

The first wiring test passed, but inspecting a real `run_manifest.json` showed
`HF_HUB_CACHE: null` while `HF_DATASETS_CACHE` was recorded. The allowlist in
`manifest.relevant_env_vars()` listed `HF_HOME`, `HF_DATASETS_CACHE` and
`TRANSFORMERS_CACHE` — but **not the one variable AlignLab actually sets**.

A run could therefore have relocated its model cache with the manifest showing
nothing. `HF_HUB_CACHE` was added, with a regression test.

*Worth noting how it was found:* not by a failing test, but by reading the
output of a real run. The wiring test only asserted the environment; it did not
assert the manifest recorded it.

## 17. torch version alignment

`torch` in the local venv was downgraded **2.13.0+cpu → 2.6.0+cpu** so both
machines run the same minor version, differing only in build variant:

```
local  : torch 2.6.0+cpu     Python 3.11.15  Windows
server : torch 2.6.0+cu124   Python 3.11.16  Linux, 2 x RTX A6000
```

The server pins the version because it is the primary training environment and
the cu124 index caps at 2.6.0.

**Lockfile divergence after alignment** — now essentially only the intended one:

| Package | Local | Server |
|---|---|---|
| `torch` | 2.6.0+cpu | 2.6.0+cu124 |
| `filelock` | 3.32.4 | 3.32.3 |
| `colorama` | present | absent (Windows-only dep) |
| `setuptools` | present | absent |
| 14 × `nvidia-*`/`triton`/`cusparselt` | absent | present |

**What alignment bought, measured.** Not bitwise agreement. Re-running the
cross-machine CPU comparison after alignment still shows ~1e-7 divergence — 2 of
6 logged steps matched exactly, up from 1 of 6. Separately, local torch 2.6.0
produced *different* values from local torch 2.13.0 **on the same machine**, so
the version genuinely does affect arithmetic.

Conclusion: the residual divergence is **platform/BLAS-level** (Windows vs
Linux), not version-level. Alignment's real value is removing API and default
drift between two torch majors — which is what actually threatened the Tier B
structural guarantee. Tier C still governs cross-machine numerics.

**Related:** [[storage-policy]] · [[phase1b-two-environments]]
