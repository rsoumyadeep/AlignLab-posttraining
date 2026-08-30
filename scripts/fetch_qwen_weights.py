"""Phase 3 - download the Qwen2.5-1.5B WEIGHTS at the pinned revision.

The Phase 2 companion script (scripts/fetch_qwen_config.py) downloaded
configuration metadata only and enforced a denylist that ABORTED on any weight
file. This script is the deliberate, Phase-3-authorised counterpart: it is the
one place in AlignLab permitted to download 2.9 GiB of parameters.

WHAT MAKES THIS SAFE TO RUN ON A SHARED, 99%-FULL VOLUME:

  1. the revision is PINNED to a commit SHA, so this is reproducible and
     cannot silently fetch a different model six months from now;
  2. the expected size is read from the Hub API FOR THAT EXACT REVISION and
     printed before anything is written - not estimated, not remembered;
  3. a disk pre-flight refuses to start unless the volume has the download
     plus headroom (alignlab.storage);
  4. the download is IDEMPOTENT - an already-complete snapshot is detected and
     re-verified rather than re-fetched, so re-running never duplicates 2.9 GiB;
  5. it downloads into AlignLab's CONFIGURED cache root, not
     ~/.cache/huggingface, so our disk usage is visible and separately
     prunable instead of hidden inside another project's cache;
  6. after the download every file's size is compared against the API manifest,
     and the result is written as machine-readable evidence.

WHY snapshot_download RATHER THAN from_pretrained. from_pretrained would
download and then immediately materialise a 3 GB model in RAM. Separating
"acquire the bytes" from "load the model" means the download can be verified,
recorded and reasoned about on its own, and means a failed load does not
imply a re-download.

Run:
    python scripts/fetch_qwen_weights.py                 # download + verify
    python scripts/fetch_qwen_weights.py --verify-only   # no network writes
    python scripts/fetch_qwen_weights.py --dry-run       # pre-flight only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from alignlab.logging_utils import get_logger
from alignlab.paths import cache_root, configure_hf_cache, repo_root
from alignlab.storage import GIB, InsufficientStorage, disk_status, require_free_space

logger = get_logger(__name__)

MODEL_ID = "Qwen/Qwen2.5-1.5B"

# PINNED. Resolved from the HF API on 2026-08-29 (Phase 2) and re-confirmed on
# 2026-08-30 to be identical to what `main` points at. Pinning costs nothing
# while they agree and protects the project the moment they diverge.
REVISION = "8faed761d45a263340a0528343f099c05c9a4323"

# Everything the model needs and nothing else. There are no other files at
# this revision, but naming them makes the intent explicit and means a file
# added upstream later is not silently pulled in.
ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "LICENSE",
]

EVIDENCE_DIR = repo_root() / "docs" / "phase3"


def api_manifest(model_id: str, revision: str) -> list[dict]:
    """Read the file tree for an exact revision from the Hub API.

    This is the ground truth the download is verified against. Fetching it for
    the pinned SHA rather than for `main` is the whole point: it describes the
    bytes we are actually asking for.
    """
    url = (
        f"https://huggingface.co/api/models/{model_id}/tree/{revision}"
        "?recursive=true"
    )
    with urllib.request.urlopen(url, timeout=60) as response:
        entries = json.load(response)
    return [e for e in entries if e.get("type") == "file"]


def expected_files(manifest: list[dict]) -> dict[str, int]:
    """Map filename -> size for the files we intend to download."""
    return {
        e["path"]: int(e.get("size") or 0)
        for e in manifest
        if e["path"] in ALLOW_PATTERNS
    }


def snapshot_dir() -> Path:
    """Where huggingface_hub will place this revision's snapshot."""
    org, name = MODEL_ID.split("/")
    configured = os.environ.get("HF_HUB_CACHE")
    hub = Path(configured) if configured else cache_root() / "hub"
    return hub / f"models--{org}--{name}" / "snapshots" / REVISION


def inspect_local(expected: dict[str, int]) -> tuple[dict[str, int], list[str]]:
    """Return (present file -> real size, list of missing/short filenames).

    Resolves symlinks: the HF cache stores blobs once and symlinks them into
    each snapshot, so ``stat`` must follow the link to measure real bytes.
    """
    directory = snapshot_dir()
    present: dict[str, int] = {}
    problems: list[str] = []

    for name, want in expected.items():
        path = directory / name
        if not path.exists():
            problems.append(f"{name}: MISSING")
            continue
        real = path.resolve().stat().st_size
        present[name] = real
        if real != want:
            problems.append(f"{name}: {real:,} bytes on disk, expected {want:,}")

    return present, problems


def write_evidence(payload: dict, filename: str) -> Path:
    """Persist a machine-readable record of what was actually downloaded."""
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE_DIR / filename
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run the pre-flight and print the plan; download nothing",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify an existing snapshot against the API manifest; download nothing",
    )
    parser.add_argument(
        "--allow-low-disk",
        action="store_true",
        help="proceed despite a failed disk pre-flight (recorded as a warning)",
    )
    args = parser.parse_args()

    applied = configure_hf_cache()
    hub_cache = os.environ.get("HF_HUB_CACHE", str(cache_root() / "hub"))

    print("=" * 78)
    print("Phase 3 - Qwen2.5-1.5B WEIGHTS")
    print("=" * 78)
    print(f"model     : {MODEL_ID}")
    print(f"revision  : {REVISION}  (PINNED)")
    print(f"hub cache : {hub_cache}")
    if applied:
        print(f"env set   : {', '.join(sorted(applied))}")
    else:
        print("env set   : none (HF cache variables were already set)")

    # ---------------------------------------------------------------- manifest
    print("\n--- expected files, from the Hub API at the PINNED revision ---")
    manifest = api_manifest(MODEL_ID, REVISION)
    expected = expected_files(manifest)
    total = sum(expected.values())
    for name in sorted(expected, key=lambda n: -expected[n]):
        print(f"  {expected[name]:>15,}  {name}")
    print(f"  {total:>15,}  TOTAL = {total / GIB:.3f} GiB")

    unexpected = [e["path"] for e in manifest if e["path"] not in ALLOW_PATTERNS]
    if unexpected:
        print(f"  not requested: {', '.join(sorted(unexpected))}")

    # ------------------------------------------------------------- idempotency
    present, problems = inspect_local(expected)
    complete = present and not problems
    print("\n--- local snapshot ---")
    print(f"  path    : {snapshot_dir()}")
    if complete:
        print(f"  status  : COMPLETE - all {len(present)} files present at expected sizes")
    elif present:
        print(f"  status  : PARTIAL - {len(present)}/{len(expected)} files")
        for p in problems:
            print(f"            {p}")
    else:
        print("  status  : ABSENT - nothing downloaded yet")

    if args.verify_only:
        print("\n--verify-only: no download attempted.")
        return 0 if complete else 1

    # ------------------------------------------------------------- pre-flight
    still_needed = total - sum(present.values())
    still_needed = max(still_needed, 0)
    print("\n--- disk pre-flight ---")
    print(f"  {disk_status(hub_cache)}")
    print(f"  still to fetch: {still_needed / GIB:.3f} GiB")

    try:
        require_free_space(
            hub_cache,
            still_needed,
            label=f"{MODEL_ID} weights @ {REVISION[:12]}",
            allow_override=args.allow_low_disk,
        )
    except InsufficientStorage as exc:
        print(f"\nREFUSING TO DOWNLOAD.\n{exc}")
        return 2

    if args.dry_run:
        print("\n--dry-run: pre-flight passed, nothing downloaded.")
        return 0

    if complete:
        print("\nAlready complete - nothing to download. (Idempotent re-run.)")
    else:
        # Imported here so --dry-run and --verify-only work without the
        # modelling stack installed.
        from huggingface_hub import snapshot_download

        print("\n--- downloading ---")
        started = datetime.now(timezone.utc)
        local_path = snapshot_download(
            repo_id=MODEL_ID,
            revision=REVISION,
            allow_patterns=ALLOW_PATTERNS,
        )
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        print(f"  snapshot: {local_path}")
        print(f"  elapsed : {elapsed:.1f} s")

    # ---------------------------------------------------------------- verify
    print("\n--- verification against the API manifest ---")
    present, problems = inspect_local(expected)
    for name in sorted(present):
        mark = "OK " if present[name] == expected[name] else "BAD"
        print(f"  {mark} {present[name]:>15,}  {name}")

    if problems:
        print("\nVERIFICATION FAILED:")
        for p in problems:
            print(f"  {p}")
        return 3

    print(f"\n  all {len(present)} files match the manifest exactly")

    after = disk_status(hub_cache)
    print(f"  {after}")

    evidence = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "revision": REVISION,
        "revision_pinned": True,
        "hub_cache": hub_cache,
        "snapshot_dir": str(snapshot_dir()),
        "expected_files": expected,
        "observed_files": present,
        "all_files_match": True,
        "total_bytes": total,
        "total_gib": round(total / GIB, 4),
        "disk_after": after.to_dict(),
    }
    path = write_evidence(evidence, "qwen_weights_download.json")
    print(f"  evidence written: {path.relative_to(repo_root())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
