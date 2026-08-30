# Phase 8 — Cleanup Record

**Date:** 2026-08-30 · **Authorised by:** user, Phase 8 brief, "CLEANUP DECISION"
**Scope authorised:** exactly two duplicate Qwen cache directories. Nothing else.

---

## What was deleted

| # | Path | Size |
|---|---|---:|
| 1 | `/data/home/rsoumyadeep/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct` | 15,242,823,988 B |
| 2 | `/data/home/rsoumyadeep/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B` | 3,098,984,611 B |

**Nothing else was touched.**

---

## Why they existed

Created by the Phase 7 bug recorded in `docs/phase7/PHASE_7_REPORT.md` §6.5:
`configure_hf_cache` set `HF_HUB_CACHE` *after* `transformers` was imported,
but `huggingface_hub` reads that variable once at import into module constants.
The call was therefore a no-op in every entrypoint and downloads landed in
`~/.cache/huggingface` regardless of the configured `cache_root`.

The bug is fixed (commit `4843ff3`); these directories are its residue.

---

## Pre-deletion verification

### 1. Exact paths resolved
Both under `$HOME/.cache/huggingface/hub`, distinct from AlignLab's configured
`cache_root` (`configs/env/server.yaml:58` →
`/data/home/rsoumyadeep/AlignLab/.cache/huggingface`).

### 2. Confirmed duplicates of the project-cache copies

Hugging Face blobs are **content-addressed**, so identical blob names mean
identical bytes.

| model | home | project | verdict |
|---|---|---|---|
| Qwen2.5-7B-Instruct | 11 blobs | 11 blobs | **identical blob sets** (same md5 of sorted listing), same revision `a09a3545…` |
| Qwen2.5-1.5B | 7 blobs | 8 blobs | **home is a strict SUBSET** — `comm -23` empty; project additionally holds `LICENSE` (11,343 B). Same revision `8faed761…`, byte-identical weight blob `a961db72…` |

The 1.5B copies were **not** identical, and that was checked before deleting
rather than assumed. Deleting the home copy loses nothing because every blob it
held is also in the project cache.

### 3. Not required by any retained artifact

- No file under `configs/`, `src/`, or any `adapter_config.json` references the
  home cache path.
- Both adapters name their base as the model id `Qwen/Qwen2.5-1.5B`, which
  resolves through whichever cache is in force — the project cache.
- `sft-qwen1p5b-noRobots-001/final` and `dpo-beta0.1-sum/final` are
  self-contained (`model.safetensors` present); they do not read the hub cache.

### 4. Unrelated data explicitly preserved

Everything else in that directory survived, verified by inventory before and
after: `models--microsoft--phi-2` (5.2 G), `models--mistralai--Mistral-7B-Instruct-v0.2`,
six `datasets--farama-minari--*`, `datasets--HuggingFaceH4--no_robots`,
`datasets--HuggingFaceH4--ultrafeedback_binarized`,
`datasets--trl-lib--ultrafeedback_binarized`, `CACHEDIR.TAG`.

### 5/6. Protected checkpoints untouched

`checkpoint-295` (8.7 G) and `final/` (2.9 G) are in
`AlignLab/checkpoints/`, a different tree entirely. Neither was involved.

---

## 7. Free space

| | bytes available on `/data` | human |
|---|---:|---|
| before | 38,898,704,384 | 37 G |
| after | 57,240,600,576 | **54 G** |
| **recovered** | **18,341,896,192** | **18.34 GB (17.08 GiB)** |

---

## Post-deletion verification

Run with `HF_HUB_OFFLINE=1`, so any missing file would have raised rather than
silently re-downloading:

```
cache in force: /data/home/rsoumyadeep/AlignLab/.cache/huggingface/hub
tokenizer OK, vocab 151665
base OK, params 1543714304
LoRA adapter OK
SFT checkpoint OK, params 1543714304
home hub before load test: 6056941943 bytes
home hub after  load test: 6056941943 bytes
NO RE-DOWNLOAD: home cache byte-identical
```

**The deletion broke nothing, and nothing was silently re-fetched to hide a
break.**

---

## Note for the future

The judge (Qwen2.5-7B-Instruct) now exists in **one** place. Re-running the
Phase 7 judge stage will use the project cache. If that copy is ever removed,
the judge would be re-downloaded — and it is pinned to `main`, not a commit
SHA, so a future download could differ. That limitation is recorded in
`docs/phase7/PHASE_7_REPORT.md` §9.4 and remains open.

**Related:** [[phase7-eval-results]] · [[phase8-report]]
