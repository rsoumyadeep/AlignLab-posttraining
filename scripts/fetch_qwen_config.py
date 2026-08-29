"""WP9 - fetch Qwen2.5-1.5B CONFIGURATION METADATA ONLY. No weights.

SCOPE, and it is enforced rather than merely intended:

  * an ALLOWLIST of filenames is downloaded - nothing else, ever
  * a DENYLIST of weight extensions is checked before every request and
    aborts the script if a weight file is ever named
  * the total downloaded size is asserted to be tiny
  * the revision is PINNED to an exact commit SHA, so this is reproducible

WHY METADATA ONLY. Phase 2 is about understanding the architecture, and the
architecture is fully described by config.json. The 3.1 GB of weights belong
to Phase 3 and would consume scarce space on the department server for no
Phase 2 benefit.

WHY THIS RUNS AFTER WP1-WP8, NOT BEFORE. Reading the real config first would
teach us to copy numbers. Reading it last turns it into a TEST of
understanding: we predicted GQA and a SwiGLU width near 8/3 - does the actual
configuration agree?

Files fetched into the project's configured HF cache, mirroring the Hugging
Face layout so a later hf_hub_download finds them where it expects.

Run: python scripts/fetch_qwen_config.py
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from alignlab.paths import cache_root

MODEL_ID = "Qwen/Qwen2.5-1.5B"

# Pinned revision, resolved from the HF API on 2026-08-29 and recorded here so
# the download is reproducible rather than "whatever main points at today".
REVISION = "8faed761d45a263340a0528343f099c05c9a4323"

# Exactly what may be downloaded. Nothing outside this list is requested.
ALLOWED_FILES = (
    "config.json",  # the architecture - the whole point of WP9
    "generation_config.json",  # default decoding settings
    "tokenizer_config.json",  # special tokens, chat template presence
)

# Any filename matching these aborts the run. Belt and braces alongside the
# allowlist: if someone widens ALLOWED_FILES carelessly, this still stops it.
FORBIDDEN_SUFFIXES = (
    ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".h5", ".msgpack",
)

MAX_TOTAL_BYTES = 2 * 1024 * 1024  # 2 MiB; config files are a few KB


def _guard(filename: str) -> None:
    if filename not in ALLOWED_FILES:
        raise ValueError(f"REFUSING: {filename!r} is not in the allowlist")
    for suffix in FORBIDDEN_SUFFIXES:
        if filename.endswith(suffix):
            raise ValueError(f"REFUSING: {filename!r} looks like model weights")


def destination() -> Path:
    """Mirror the Hugging Face cache layout inside our configured cache root."""
    org, name = MODEL_ID.split("/")
    return cache_root() / "hub" / f"models--{org}--{name}" / "snapshots" / REVISION


def fetch(filename: str, out_dir: Path) -> int:
    _guard(filename)
    url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{filename}"
    out_path = out_dir / filename

    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()

    if len(payload) > MAX_TOTAL_BYTES:
        raise ValueError(
            f"REFUSING: {filename} is {len(payload):,} bytes, larger than the "
            f"{MAX_TOTAL_BYTES:,}-byte metadata limit - this is not a config file"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(payload)
    return len(payload)


def main() -> None:
    out_dir = destination()

    print("=" * 78)
    print("WP9 - Qwen2.5-1.5B configuration metadata")
    print("=" * 78)
    print(f"model    : {MODEL_ID}")
    print(f"revision : {REVISION}  (PINNED)")
    print(f"cache    : {out_dir}")
    print(f"allowed  : {', '.join(ALLOWED_FILES)}")
    print("NOT downloading: model.safetensors or any other weight file\n")

    total = 0
    for filename in ALLOWED_FILES:
        size = fetch(filename, out_dir)
        total += size
        print(f"  fetched {filename:<26} {size:>8,} bytes")

    print(f"\n  total {total:,} bytes ({total / 1024:.1f} KiB)")

    # Prove no weights landed anywhere in the destination.
    strays = [
        p.name
        for p in out_dir.rglob("*")
        if p.is_file() and any(p.name.endswith(s) for s in FORBIDDEN_SUFFIXES)
    ]
    print(f"  weight files present: {strays if strays else 'NONE'}")
    assert not strays, "a weight file was written - this must never happen"

    print("\n--- config.json ---")
    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    for key in sorted(config):
        print(f"  {key:<32} {config[key]}")


if __name__ == "__main__":
    main()
