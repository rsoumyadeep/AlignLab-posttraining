"""E18 - does our first-principles LoRA agree with the peft library's?

PROJECT_INSTRUCTIONS section 21 asks for a two-level approach: an educational
minimal implementation, verified, then a practical library implementation for
the realistic experiment. This script is the bridge that makes the pair
meaningful rather than merely coexisting.

WHY THIS MATTERS MORE THAN IT SOUNDS. `alignlab.lora` has 46 unit tests, so it
is internally consistent. But internal consistency proves only that it does
what WE think LoRA is. If our reading of the equation were subtly wrong -
scaling applied in the wrong place, B and A transposed, dropout on the wrong
branch - every one of those tests would still pass, and the educational
implementation would be teaching something false while the real training runs
(which use peft) did something else.

Agreement with an independent implementation is what turns "self-consistent"
into "correct".

THE METHOD. Build the SAME tiny Qwen2 twice. Adapt one with our LoRALinear and
one with peft. Copy the adapter weights across so the only remaining
difference is the code path. Then compare outputs.

    same weights + same equation  ->  same output, to float precision
    same weights + different eq   ->  divergence, and the diff localises it

HYPOTHESES, recorded before running:

    H1  both implementations adapt exactly the same set of modules
    H2  both report the same trainable parameter count
    H3  with matched adapter weights, outputs agree to float32 precision
    H4  both are exactly equal to the BASE model at initialisation, since both
        initialise B to zero
    H5  our merge() produces the same weights as peft's merge_and_unload()

Run:
    python scripts/experiments/e18_educational_vs_peft.py
"""

from __future__ import annotations

import json

import torch
from transformers import AutoModelForCausalLM, Qwen2Config

from alignlab.lora import LoRAConfig, LoRALinear, apply_lora, merge_all
from alignlab.paths import repo_root

R = 8
ALPHA = 16.0
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


def tiny_qwen() -> Qwen2Config:
    """A real Qwen2 architecture at a size that runs on CPU in a second.

    Real architecture rather than a toy nn.Module: the module NAMES and the
    GQA shape asymmetry (k/v narrower than q) are exactly what a target-matching
    bug would hide behind.
    """
    return Qwen2Config(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        vocab_size=256,
        tie_word_embeddings=True,
    )


def build_base(seed: int = 0):
    torch.manual_seed(seed)
    return AutoModelForCausalLM.from_config(tiny_qwen()).to(torch.float32).eval()


def adapted_module_names_ours(model) -> set[str]:
    return {
        name for name, module in model.named_modules() if isinstance(module, LoRALinear)
    }


def adapted_module_names_peft(model) -> set[str]:
    """peft renames wrapped modules to `<path>.base_layer`; recover the path."""
    names = set()
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "base_layer"):
            clean = name.replace("base_model.model.", "")
            names.add(clean)
    return names


def main() -> int:
    from peft import LoraConfig, get_peft_model

    print("=" * 78)
    print("E18 - first-principles LoRA vs the peft library")
    print("=" * 78)
    print(f"  r={R}  alpha={ALPHA}  scaling={ALPHA / R}  targets={TARGETS}")

    torch.manual_seed(123)
    input_ids = torch.randint(0, 256, (2, 16))

    # ------------------------------------------------------------- baseline
    base = build_base()
    with torch.no_grad():
        base_logits = base(input_ids).logits.clone()

    # ----------------------------------------------------------------- ours
    ours = build_base()
    ours_summary = apply_lora(ours, LoRAConfig(r=R, alpha=ALPHA, target_modules=TARGETS))
    ours_names = adapted_module_names_ours(ours)

    # ----------------------------------------------------------------- peft
    theirs_base = build_base()
    peft_config = LoraConfig(
        r=R,
        lora_alpha=ALPHA,
        lora_dropout=0.0,  # dropout would make the comparison stochastic
        target_modules=list(TARGETS),
        bias="none",
        task_type="CAUSAL_LM",
    )
    theirs = get_peft_model(theirs_base, peft_config)
    theirs_names = adapted_module_names_peft(theirs)

    # ------------------------------------------------------- H1: same modules
    print("\n--- H1: same modules adapted ---")
    print(f"  ours  : {len(ours_names)} modules")
    print(f"  peft  : {len(theirs_names)} modules")
    only_ours = sorted(ours_names - theirs_names)
    only_theirs = sorted(theirs_names - ours_names)
    h1 = ours_names == theirs_names
    print(f"  H1 holds: {h1}")
    if not h1:
        print(f"    only ours : {only_ours[:5]}")
        print(f"    only peft : {only_theirs[:5]}")

    # ---------------------------------------------------- H2: same param count
    ours_trainable = sum(p.numel() for p in ours.parameters() if p.requires_grad)
    theirs_trainable = sum(p.numel() for p in theirs.parameters() if p.requires_grad)
    print("\n--- H2: same trainable parameter count ---")
    print(f"  ours  : {ours_trainable:,}")
    print(f"  peft  : {theirs_trainable:,}")
    h2 = ours_trainable == theirs_trainable
    print(f"  H2 holds: {h2}")

    # ------------------------------------------- H4: both equal base at init
    print("\n--- H4: both equal the BASE model at initialisation ---")
    with torch.no_grad():
        ours_init = ours(input_ids).logits
        theirs_init = theirs(input_ids).logits
    d_ours = float((ours_init - base_logits).abs().max())
    d_theirs = float((theirs_init - base_logits).abs().max())
    print(f"  max|ours  - base| = {d_ours:.3e}")
    print(f"  max|peft  - base| = {d_theirs:.3e}")
    h4 = d_ours == 0.0 and d_theirs == 0.0
    print(f"  H4 holds (both EXACTLY zero, because B=0): {h4}")

    # ------------------------------- H3: matched weights -> matched outputs
    print("\n--- H3: with matched adapter weights, outputs agree ---")
    # Give ours non-trivial adapters, then copy them into peft's modules.
    torch.manual_seed(7)
    with torch.no_grad():
        for name in sorted(ours_names):
            mod = ours.get_submodule(name)
            torch.nn.init.normal_(mod.lora_A.weight, std=0.05)
            torch.nn.init.normal_(mod.lora_B.weight, std=0.05)

            target = theirs.get_submodule(f"base_model.model.{name}")
            target.lora_A["default"].weight.copy_(mod.lora_A.weight)
            target.lora_B["default"].weight.copy_(mod.lora_B.weight)

    with torch.no_grad():
        ours_out = ours(input_ids).logits
        theirs_out = theirs(input_ids).logits

    diff = (ours_out - theirs_out).abs()
    h3 = bool(diff.max() < 1e-5)
    print(f"  max|ours - peft| = {float(diff.max()):.3e}")
    print(f"  mean|ours - peft| = {float(diff.mean()):.3e}")
    print(f"  both differ from base (adapters are active): "
          f"{float((ours_out - base_logits).abs().max()):.3e}")
    print(f"  H3 holds (agree to 1e-5): {h3}")

    # ------------------------------------------------------- H5: merge agrees
    print("\n--- H5: merged weights agree ---")
    merged_ours = merge_all(ours)
    theirs_merged = theirs.merge_and_unload()

    max_weight_diff = 0.0
    compared = 0
    for name in sorted(ours_names):
        w_ours = ours.get_submodule(name).base.weight
        w_theirs = theirs_merged.get_submodule(name).weight
        max_weight_diff = max(max_weight_diff, float((w_ours - w_theirs).abs().max()))
        compared += 1

    with torch.no_grad():
        merged_out_ours = ours(input_ids).logits
        merged_out_theirs = theirs_merged(input_ids).logits
    merged_logit_diff = float((merged_out_ours - merged_out_theirs).abs().max())

    h5 = max_weight_diff < 1e-5 and merged_logit_diff < 1e-5
    print(f"  merged {merged_ours} modules (ours), compared {compared} weights")
    print(f"  max|W_ours - W_peft| after merge = {max_weight_diff:.3e}")
    print(f"  max|logits| after merge          = {merged_logit_diff:.3e}")
    print(f"  H5 holds: {h5}")

    # merged output must also equal the pre-merge adapter output
    premerge_diff = float((merged_out_ours - ours_out).abs().max())
    print(f"  merged == pre-merge adapter output: {premerge_diff:.3e}")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    checks = {
        "H1 same modules adapted     ": h1,
        "H2 same trainable count     ": h2,
        "H3 matched weights -> agree ": h3,
        "H4 both == base at init     ": h4,
        "H5 merge agrees             ": h5,
    }
    for label, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    all_pass = all(checks.values())
    print(f"\n  Educational implementation agrees with peft: {all_pass}")
    if all_pass:
        print("  The first-principles module is not merely self-consistent -")
        print("  it computes the same function as an independent implementation.")

    print("\n--- WHAT THIS PROVES / DOES NOT PROVE ---")
    print("  PROVES     : the two implementations compute the same function on")
    print("               the same weights, adapt the same modules, and merge")
    print("               identically.")
    print("  DOES NOT   : prove either is a good idea, or that peft's TRAINING")
    print("               behaviour (dropout, dtype casting, gradient")
    print("               checkpointing interaction) matches ours - dropout was")
    print("               set to 0 precisely to remove that variable.")

    payload = {
        "r": R, "alpha": ALPHA, "targets": list(TARGETS),
        "ours_trainable": ours_trainable,
        "peft_trainable": theirs_trainable,
        "modules_ours": sorted(ours_names),
        "modules_peft": sorted(theirs_names),
        "max_logit_diff": float(diff.max()),
        "max_weight_diff_after_merge": max_weight_diff,
        "hypotheses": {k.strip(): bool(v) for k, v in checks.items()},
        "all_pass": bool(all_pass),
    }
    out = repo_root() / "docs" / "phase4" / "e18_educational_vs_peft.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n  evidence: {out.relative_to(repo_root())}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
