# AlignLab Storage Policy

**Status:** Phase 1C. Audit is VERIFIED (measured on `csrslave`, 2026-08-29).
Forward projections are **ESTIMATES — NOT MEASURED** (arithmetic on parameter
counts, clearly marked as such).

**Nothing in this document has been deleted or executed.** Cleanup candidates
in §6 are PROPOSED and untouched.

---

## 1. Why this document exists

On the department server, **disk is the binding constraint — not VRAM.**

```
/dev/sda1  7.3T  6.8T  77G  99%  /data     ← home + all project data
/dev/nvme0n1p4  1.8T  101G  1.6T  6%  /    ← holds /tmp; user-writable there
tmpfs  252G  /dev/shm                      ← RAM-backed, volatile
```

`/data` is shared by at least 8 active users. There is **no quota tooling**
(`quota` and `repquota` are absent), so no per-user limit is discoverable and
no early warning exists. The failure mode is a full shared volume affecting
everyone, not a friendly error for us.

---

## 2. Current AlignLab footprint (VERIFIED)

| Item | Size | Location |
|---|---|---|
| **AlignLab total** | **5.1 GB** | `/data/home/rsoumyadeep/AlignLab` |
| `.venv` | **5.1 GB** | inside the project |
| `.git` | 560 KB | |
| `src` + `tests` + `configs` + `docs` | ~730 KB | |
| `checkpoints` (toy runs) | 216 KB | |
| `outputs` (toy runs) | 168 KB | |

The venv dominates, and is almost entirely CUDA:

| Package | Size |
|---|---|
| `nvidia/*` (CUDA runtime libs) | **2.7 GB** |
| `torch/` | 1.5 GB |
| `triton/` | 439 MB |
| `cusparselt/` | 203 MB |
| `wandb/` | 92 MB |

**AlignLab is currently responsible for 5.1 GB of the ~6.8 TB used.**

## 3. Cache locations (VERIFIED)

| Cache | Location | Size | Owner |
|---|---|---|---|
| **AlignLab HF cache** | `<repo>/.cache/huggingface` | **not yet created** | AlignLab |
| Default HF cache | `~/.cache/huggingface` | **5.3 GB** | other projects |
| uv cache | `~/.cache/uv` | **5.2 GB** | shared/other projects |
| pip cache | `~/.cache/pip` | **13 GB** | other projects |
| `~/.cache` total | | **24 GB** | mostly not AlignLab |

The pre-existing `~/.cache/huggingface` holds `models--microsoft--phi-2`,
`models--mistralai--Mistral-7B-Instruct-v0.2` and six Minari datasets. **None of
it is AlignLab's**, and none of it is ours to remove.

### A real gap found and closed in Phase 1C

`configs/env/server.yaml` declared `cache_root`, but **nothing wired it to the
Hugging Face libraries** — they read environment variables, not our config.
`manifest.py` only *recorded* `HF_HOME`; nothing *set* it.

Left alone, the first Phase 3 model download would have landed in
`~/.cache/huggingface`, adding ~3 GB to a directory already holding 5.3 GB of
someone else's models — hiding AlignLab's disk usage inside another project's
cache.

**Fixed:** `alignlab.paths.configure_hf_cache()` now sets `HF_HUB_CACHE` and
`HF_DATASETS_CACHE` from `cache_root`, and `train.py` calls it before anything
could trigger a download. Covered by four tests.

`HF_HOME` is deliberately **not** set: it also governs the stored-credentials
file, and relocating it would break an existing `huggingface-cli login`.
Setting only the two cache variables moves the large files and leaves
authentication where the user put it.

## 4. Projected requirements — ESTIMATES, NOT MEASURED

Arithmetic on Qwen2.5-1.5B (~1.54 B parameters). Nothing here has been
measured; no model has been downloaded.

### Model weights
| Item | Estimate |
|---|---|
| Qwen2.5-1.5B, bf16 safetensors | **~3.1 GB** |
| Tokenizer + config | < 20 MB |

### Checkpoints — the dangerous one

A *resumable* AlignLab checkpoint stores model + optimizer + scheduler + RNG.
For AdamW, the optimizer alone is two fp32 tensors per parameter:

| Component | Size |
|---|---|
| bf16 weights | 3.1 GB |
| AdamW `exp_avg` (fp32) | 6.2 GB |
| AdamW `exp_avg_sq` (fp32) | 6.2 GB |
| **Full-SFT resumable checkpoint** | **≈ 15.5 GB** (≈18.5 GB if weights fp32) |

> ⚠ **`keep_last_checkpoints: 3` × 15.5 GB ≈ 46 GB — more than half the free
> space, from a single run.** The current default is 3. For full-parameter SFT
> that default is unsafe on this volume and must be lowered.

| LoRA checkpoint (rank 16, attention projections) | **~50–200 MB** |

LoRA is roughly **100× cheaper to checkpoint** than full SFT. On this machine
that is a storage argument for LoRA, entirely separate from the usual memory
argument — and worth stating explicitly in the Phase 4 write-up.

### Datasets
| Item | Estimate |
|---|---|
| Instruction dataset (~50k examples), raw + tokenized | ~0.5–2 GB |
| Preference dataset (chosen/rejected) | ~0.5–2 GB |

### Experiment artifacts (MEASURED on toy runs, extrapolated)
Logs, manifests, resolved configs and `metrics.jsonl` are **KB per run**. Even
hundreds of runs stay under 1 GB. These are not a concern.

### W&B behaviour
Offline mode writes a run directory under the run's output dir — small
(metrics + metadata) **unless artifacts are logged**. `WandbTracker.log_artifact`
exists and would copy whatever it is given into W&B's store, **duplicating it on
disk**. Policy: **never log a full checkpoint as a W&B artifact on this
machine.** Adapters and metrics only.

## 5. Policy

1. **Model/dataset caches live under the project**, via `cache_root` →
   `HF_HUB_CACHE`/`HF_DATASETS_CACHE`. AlignLab's disk usage must be
   attributable to AlignLab, and prunable without touching other projects.
2. **`keep_last_checkpoints` is capacity planning, not hygiene.**
   - Full-parameter SFT: **`keep_last: 1`** (≈15.5 GB). Never 3.
   - LoRA/QLoRA: 3 is fine (~0.6 GB total).
   - Set it per-experiment, deliberately.
3. **Check free space before any run that downloads or checkpoints.**
   `df -h /data` — treat **< 20 GB free** as stop-and-reassess.
4. **Package caches go to `/tmp`**, which is a separate 1.6 TB NVMe:
   `UV_CACHE_DIR=/tmp/uv-cache uv pip install …`, removed afterwards. This kept
   5.1 GB off `/data` during Phase 1B. **VERIFIED effective.**
5. **`/tmp` for scratch, never for anything durable.** It is large and fast but
   volatile and world-shared. Never point `checkpoint_root` at it.
6. **`/dev/shm` (252 GB) for dataloader shared memory only** — RAM-backed;
   anything written there consumes RAM.
7. **Never log checkpoints as W&B artifacts** (see §4).
8. **Never write to `/data/data2`.** It is writable (`drwxrwxrwx`, owned
   `nobody`) but sits on the same filesystem, so it offers no extra space and
   has no clear ownership.
9. **Do not delete other projects' data**, including `~/.cache/*` and
   `miniconda3`, without explicit owner approval.
10. **Re-audit at each phase boundary** and record before/after in the phase
    report, as Phase 1B and 1C did.

### Configuration summary

| Setting | Local | Server |
|---|---|---|
| `output_root` | `<repo>/outputs` | `<repo>/outputs` |
| `checkpoint_root` | `<repo>/checkpoints` | `<repo>/checkpoints` |
| `cache_root` | `<repo>/.cache` | `<repo>/.cache/huggingface` |
| `HF_HUB_CACHE` | set from `cache_root` | set from `cache_root` |
| uv cache during install | default | **`/tmp/uv-cache`, then removed** |
| W&B | offline | offline |

## 6. Cleanup candidates — PROPOSED, NOT EXECUTED

**None of the following has been touched. Every item requires explicit
approval. Items marked ⛔ were placed off-limits by instruction.**

| # | Candidate | Size | Risk | Notes |
|---|---|---|---|---|
| 1 | `~/.cache/pip` | **13 GB** | **Low** | Pure download cache; pip refetches on demand. Largest single easy win. Belongs to other projects' installs, but is disposable by nature. |
| 2 | `~/.cache/uv` | **5.2 GB** | **Low** | Same reasoning. uv refetches. Note our Phase 1B install deliberately bypassed it. |
| 3 | Stale `/tmp/ant_maze*.xml` (yours) | small, many files | **Low** | Dozens of MuJoCo temp files from other work. On `/tmp`, so it frees **no `/data` space** — tidiness only. |
| 4 | `~/.cache/ms-playwright-go` | 128 MB | Low | Unrelated tooling cache. |
| 5 | `~/.cache/huggingface` (phi-2, Mistral-7B, Minari) | **5.3 GB** | **Medium** | Other projects' models. Re-downloadable but slow. **Only you know if still needed.** |
| 6 | ⛔ `miniconda3` | **47 GB** | — | **Explicitly excluded by instruction. Not proposed.** |
| 7 | ⛔ Anything else on `/data` | — | — | **Explicitly excluded by instruction.** |

**Realistic recovery from low-risk items (#1 + #2 + #4): ≈18.3 GB**, taking free
space from ~77 GB to ~95 GB. That alone would make a full-SFT run with
`keep_last: 1` comfortable.

**My recommendation, for your decision only:** approve #1 and #2. They are pure
download caches, regenerate automatically, and belong to no experiment. I have
not touched them and will not without your word.

**A caveat worth stating:** these caches are in *your* home directory but may be
shared with your other active projects (DiffuLab, thesis work). Clearing them
will make the next install in those projects slower. Nothing breaks.

## 7. What this policy does NOT solve

- **No quota visibility.** We cannot see a limit or get a warning.
- **Other users can fill `/data` at any time.** 77 GB free is not reserved for
  us; it is a shared snapshot.
- **No automated enforcement.** Rules 2 and 3 are conventions a human must
  follow; nothing in the code refuses to start a run when disk is low. A
  pre-flight disk check is a reasonable Phase 3 addition and is **not
  implemented**.
