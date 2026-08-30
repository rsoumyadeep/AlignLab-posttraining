"""E16 - which modules does LoRA actually target in Qwen2.5-1.5B?

PROJECT_INSTRUCTIONS for Phase 4: "Use the actual Qwen2.5-1.5B
architecture/configuration. Do not assume textbook module names. Explicitly
inspect and document which modules are targeted."

That instruction is not pedantry. Phase 2 predicted three architecture facts
wrongly, Phase 3's E11 found a fourth (the omitted QKV biases), and Phase 3 hit
a runtime failure from guessing dataset split names off filenames. The pattern
is consistent: assumptions about names and shapes are where this project's
errors come from. So this script enumerates every nn.Linear in the loaded model
rather than trusting the LoRA paper's naming or ours.

WHAT IT RECORDS, per PROJECT_INSTRUCTIONS:
    target modules, number of affected matrices, original parameter count,
    trainable parameter count, percentage trainable, resulting adapter size

HYPOTHESES, recorded before running:

    H1  the attention projections are named q_proj/k_proj/v_proj/o_proj and the
        MLP gate_proj/up_proj/down_proj, 7 per layer x 28 layers = 196 Linears
    H2  k_proj and v_proj are NOT square. Qwen uses GQA with 2 KV heads against
        12 query heads, so they map 1536 -> 256 while q_proj/o_proj are
        1536 -> 1536. A LoRA adapter's cost is r*(d_in + d_out), so the KV
        adapters are CHEAPER than the Q/O ones - the asymmetry GQA introduces
        into the base model shows up in the adapter budget too.
    H3  lm_head is tied to the embedding and should NOT be adapted; adapting it
        would also adapt the input embedding, which is not what "adapt the
        attention projections" means.

Run (server):
    python scripts/experiments/e16_lora_targets.py
"""

from __future__ import annotations

import json

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from alignlab.lora import LoRAConfig, apply_lora
from alignlab.paths import configure_hf_cache, repo_root

MODEL_ID = "Qwen/Qwen2.5-1.5B"
REVISION = "8faed761d45a263340a0528343f099c05c9a4323"

# Candidate target sets, from cheapest to most expensive. The LoRA paper adapts
# attention only; QLoRA's authors found adapting ALL linear layers mattered
# more than rank. Both are measured here rather than asserted.
TARGET_SETS = {
    "q_v_only": ("q_proj", "v_proj"),
    "attention": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "attention_mlp": (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ),
}

RANKS = (4, 8, 16, 32, 64)


def enumerate_linears(model) -> list[dict]:
    """Every nn.Linear in the model, with its real name and shape."""
    rows = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            rows.append(
                {
                    "name": name,
                    "leaf": name.rsplit(".", 1)[-1],
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                    "params": module.weight.numel()
                    + (module.bias.numel() if module.bias is not None else 0),
                    "has_bias": module.bias is not None,
                }
            )
    return rows


def adapter_params(d_in: int, d_out: int, r: int) -> int:
    """r*(d_in + d_out): A is r x d_in, B is d_out x r."""
    return r * (d_in + d_out)


def main() -> int:
    configure_hf_cache()

    config = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION)
    print("=" * 78)
    print("E16 - LoRA target modules in Qwen2.5-1.5B")
    print("=" * 78)
    print(f"  model    {MODEL_ID} @ {REVISION[:12]}")
    print(f"  layers   {config.num_hidden_layers}  hidden {config.hidden_size}  "
          f"ffn {config.intermediate_size}")
    print(f"  heads    {config.num_attention_heads} Q / {config.num_key_value_heads} KV "
          f"(GQA), head_dim {config.hidden_size // config.num_attention_heads}")

    # Loaded on CPU in float32 is unnecessary; meta device would avoid the
    # memory entirely, but we want real .numel() on real parameters, and bf16
    # on CPU is cheap enough at 1.5B.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, dtype=torch.bfloat16
    )
    total_params = sum(p.numel() for p in model.parameters())

    linears = enumerate_linears(model)
    print(f"\n--- every nn.Linear: {len(linears)} found ---")

    by_leaf: dict[str, list[dict]] = {}
    for row in linears:
        by_leaf.setdefault(row["leaf"], []).append(row)

    print(f"  {'leaf name':<14} {'count':>6} {'in':>7} {'out':>7} {'bias':>6} {'params each':>14}")
    for leaf in sorted(by_leaf, key=lambda k: -by_leaf[k][0]["params"]):
        rows = by_leaf[leaf]
        r0 = rows[0]
        shapes = {(x["in_features"], x["out_features"]) for x in rows}
        shape_note = "" if len(shapes) == 1 else "  <-- SHAPES VARY"
        print(f"  {leaf:<14} {len(rows):>6} {r0['in_features']:>7} "
              f"{r0['out_features']:>7} {str(r0['has_bias']):>6} "
              f"{r0['params']:>14,}{shape_note}")

    # ------------------------------------------------------------ hypotheses
    expected_leaves = {"q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"}
    found_leaves = set(by_leaf)
    h1 = expected_leaves <= found_leaves and len(linears) >= 196
    print(f"\n  H1 (expected names, >=196 Linears): {h1}")
    print(f"     found leaves: {sorted(found_leaves)}")
    if not expected_leaves <= found_leaves:
        print(f"     MISSING: {sorted(expected_leaves - found_leaves)}")

    q = by_leaf["q_proj"][0]
    k = by_leaf["k_proj"][0]
    h2 = k["out_features"] < q["out_features"]
    print(f"\n  H2 (KV projections are narrower than Q, because of GQA): {h2}")
    print(f"     q_proj {q['in_features']}->{q['out_features']}, "
          f"k_proj {k['in_features']}->{k['out_features']}")
    print(f"     adapter cost at r=16: q_proj {adapter_params(q['in_features'], q['out_features'], 16):,} "
          f"vs k_proj {adapter_params(k['in_features'], k['out_features'], 16):,}")

    h3 = "lm_head" not in found_leaves or config.tie_word_embeddings
    print(f"\n  H3 (lm_head tied, must not be adapted): tie_word_embeddings="
          f"{config.tie_word_embeddings}")
    print(f"     lm_head present as a Linear: {'lm_head' in found_leaves}")
    print("     It is EXCLUDED from every target set below: adapting a tied")
    print("     lm_head would also adapt the input embedding.")

    # ------------------------------------------------- cost of each target set
    print("\n" + "=" * 78)
    print("TARGET SETS x RANK - what each would cost")
    print("=" * 78)
    print(f"  base model: {total_params:,} total parameters")

    table = {}
    for set_name, targets in TARGET_SETS.items():
        matched = [row for row in linears if row["leaf"] in targets]
        per_rank = {}
        print(f"\n  --- {set_name}: {targets}")
        print(f"      matrices adapted: {len(matched)}")
        base_covered = sum(row["params"] for row in matched)
        print(f"      base params covered: {base_covered:,} "
              f"({100 * base_covered / total_params:.1f}% of the model)")
        print(f"      {'rank':>5} {'trainable':>14} {'% of model':>11} "
              f"{'adapter fp32':>14} {'adapter bf16':>14}")
        for r in RANKS:
            trainable = sum(
                adapter_params(row["in_features"], row["out_features"], r)
                for row in matched
            )
            per_rank[r] = {
                "trainable": trainable,
                "fraction": trainable / total_params,
                "bytes_fp32": trainable * 4,
                "bytes_bf16": trainable * 2,
            }
            print(f"      {r:>5} {trainable:>14,} {100 * trainable / total_params:>10.4f}% "
                  f"{trainable * 4 / 1024**2:>13.1f}M {trainable * 2 / 1024**2:>13.1f}M")
        table[set_name] = {
            "targets": list(targets),
            "matrices": len(matched),
            "base_params_covered": base_covered,
            "per_rank": per_rank,
        }

    # ------------------------------- verify against the real implementation
    print("\n" + "=" * 78)
    print("CROSS-CHECK: apply_lora on the real model (attention, r=16)")
    print("=" * 78)
    summary = apply_lora(model, LoRAConfig(r=16, alpha=32, target_modules=TARGET_SETS["attention"]))
    predicted = table["attention"]["per_rank"][16]["trainable"]
    print(f"  predicted trainable : {predicted:,}")
    print(f"  actual trainable    : {summary['trainable_params']:,}")
    print(f"  agree               : {predicted == summary['trainable_params']}")
    print(f"  matrices adapted    : {summary['matched_modules']}")
    print(f"  total after wrapping: {summary['total_params_after']:,}")
    print(f"  trainable fraction  : {100 * summary['trainable_fraction']:.4f}%")

    frozen_ok = all(
        not p.requires_grad
        for n, p in model.named_parameters()
        if "lora_" not in n
    )
    print(f"  every non-adapter parameter frozen: {frozen_ok}")

    print("\n--- WHAT THIS PROVES / DOES NOT PROVE ---")
    print("  PROVES     : the real module names and shapes, the exact adapter")
    print("               cost of each target set and rank, and that our")
    print("               implementation's count matches the arithmetic.")
    print("  DOES NOT   : say which target set or rank trains BETTER. Cost is")
    print("               not quality. Rank selection is E17; a quality")
    print("               comparison would need controlled training runs.")

    payload = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "total_params": total_params,
        "n_linear_modules": len(linears),
        "leaves": {k: {"count": len(v), "in": v[0]["in_features"],
                       "out": v[0]["out_features"], "bias": v[0]["has_bias"]}
                   for k, v in by_leaf.items()},
        "target_sets": table,
        "hypotheses": {"H1": bool(h1), "H2": bool(h2), "H3": bool(h3)},
        "cross_check": {
            "predicted_trainable": predicted,
            "actual_trainable": summary["trainable_params"],
            "agree": bool(predicted == summary["trainable_params"]),
            "all_non_adapter_frozen": bool(frozen_ok),
        },
    }
    out = repo_root() / "docs" / "phase4" / "e16_lora_targets.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n  evidence: {out.relative_to(repo_root())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
