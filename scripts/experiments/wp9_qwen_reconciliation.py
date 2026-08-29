"""WP9 - reconcile the educational model against Qwen2.5-1.5B's REAL config.

Run AFTER WP1-WP8, deliberately, so this is a TEST of understanding rather
than a source to copy from. Reads only the metadata fetched by
scripts/fetch_qwen_config.py - no weights are downloaded or touched.

Run: python scripts/experiments/wp9_qwen_reconciliation.py
"""
from __future__ import annotations

import json

from alignlab.paths import cache_root

MODEL_ID = "Qwen/Qwen2.5-1.5B"
REVISION = "8faed761d45a263340a0528343f099c05c9a4323"


def load_config() -> dict:
    org, name = MODEL_ID.split("/")
    path = (cache_root() / "hub" / f"models--{org}--{name}" / "snapshots"
            / REVISION / "config.json")
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found - run scripts/fetch_qwen_config.py first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    c = load_config()
    h = c["hidden_size"]
    n_heads = c["num_attention_heads"]
    n_kv = c["num_key_value_heads"]
    head_dim = h // n_heads
    inter = c["intermediate_size"]
    layers = c["num_hidden_layers"]
    vocab = c["vocab_size"]

    print("=" * 82)
    print(f"WP9 - {MODEL_ID} @ {REVISION[:12]}")
    print("=" * 82)

    print("\n--- PREDICTIONS FROM WP1-WP8 vs REALITY ---")
    rows = [
        ("attention variant", "GQA (predicted)",
         f"GQA: {n_heads} Q heads / {n_kv} KV heads, ratio {n_heads // n_kv}:1",
         n_kv < n_heads),
        ("normalization", "RMSNorm", f"rms_norm_eps={c['rms_norm_eps']}", True),
        ("FFN activation", "SwiGLU / SiLU", f"hidden_act={c['hidden_act']}",
         c["hidden_act"] == "silu"),
        ("positional", "RoPE", f"rope_theta={c['rope_theta']:,.0f}", True),
        ("weight tying", "tied", f"tie_word_embeddings={c['tie_word_embeddings']}",
         c["tie_word_embeddings"]),
        ("head_dim", "power of two", f"{h}/{n_heads} = {head_dim}",
         head_dim & (head_dim - 1) == 0),
    ]
    for name, predicted, actual, ok in rows:
        print(f"  {'OK ' if ok else 'NO '} {name:<20} predicted {predicted:<18} -> {actual}")

    print("\n--- WHERE THE PREDICTION WAS WRONG ---")
    ratio = inter / h
    print(f"  SwiGLU width: predicted ~8/3 = {8/3:.3f} x hidden_size")
    print(f"                actual {inter} / {h} = {ratio:.3f} x  <-- MUCH wider")
    eight_thirds = int(2 * (4 * h) / 3)
    print(f"                8/3 rule would give {eight_thirds:,}, actual is {inter:,}")
    print(f"                that is {inter / eight_thirds:.2f}x the conventional width")

    print("\n--- KV CACHE CONSEQUENCE OF GQA (the WP2 formula, applied) ---")
    for ctx in (4096, 32768, 131072):
        def cache_gib(kv_heads: int) -> float:
            return 2 * 1 * kv_heads * ctx * head_dim * 2 * layers / 1024**3
        print(f"  context {ctx:>7,}: GQA(n_kv={n_kv}) {cache_gib(n_kv):>7.2f} GiB   "
              f"MHA(n_kv={n_heads}) {cache_gib(n_heads):>7.2f} GiB   "
              f"saving {cache_gib(n_heads) / cache_gib(n_kv):.0f}x")

    print("\n--- PARAMETER BUDGET (arithmetic from the config, not measured) ---")
    attn = 2 * h * h + 2 * h * (n_kv * head_dim)
    ffn = 3 * h * inter
    norms = 2 * h
    per_layer = attn + ffn + norms
    embed = vocab * h
    total = embed + per_layer * layers + h
    print(f"  embedding (tied)        {embed:>15,}  ({embed/total:>5.1%})")
    print(f"  attention  / layer      {attn:>15,}")
    print(f"  FFN        / layer      {ffn:>15,}  ({ffn/per_layer:>5.1%} of a block)")
    print(f"  per layer               {per_layer:>15,}")
    print(f"  x {layers} layers            {per_layer*layers:>15,}  "
          f"({per_layer*layers/total:>5.1%})")
    print(f"  TOTAL                   {total:>15,}  ({total/1e9:.2f} B)")
    print(f"  tying saves             {embed:>15,}  ({embed/(total+embed):>5.1%} of untied)")

    print("\n--- OTHER FACTS WORTH RECORDING ---")
    print(f"  layers                  {layers}")
    print(f"  max_position_embeddings {c['max_position_embeddings']:,}")
    print(f"  torch_dtype             {c['torch_dtype']}")
    print(f"  sliding_window          {c['sliding_window']:,} "
          f"(use_sliding_window={c['use_sliding_window']} -> DISABLED)")
    print(f"  bos == eos              {c['bos_token_id'] == c['eos_token_id']} "
          f"(both {c['bos_token_id']})")
    print("=" * 82)


if __name__ == "__main__":
    main()
