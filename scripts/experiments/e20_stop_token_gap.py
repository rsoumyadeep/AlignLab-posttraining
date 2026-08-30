"""E20 - why did no PEFT arm learn to stop, when full fine-tuning did?

THE OBSERVATION (E19), which none of Phase 4's hypotheses anticipated:

    model              completion ppl   emitted <|im_end|>
    full SFT                    7.199                  4/4
    LoRA  r=16 @2e-4            7.449                  0/4
    LoRA  r=16 @2e-5            7.624                  0/4
    QLoRA r=16 @2e-4            7.511                  0/4

Perplexity puts the arms within 6% of each other. Stop-token behaviour
separates them completely. All three PEFT models write fluent, on-topic
answers and then run to the token cap.

PROJECT_INSTRUCTIONS section 18 says a surprising result means STOP AND
INVESTIGATE, so this script does, rather than reporting it as a curiosity.

CANDIDATE EXPLANATIONS, before measuring:

  H1  LoRA learned NOTHING about the stop token - P(<|im_end|>) at the correct
      position is unchanged from the base model.
  H2  LoRA learned SOMETHING but not enough - P(<|im_end|>) rose from the base
      but stayed below the argmax, so greedy decoding never picks it.
  H3  THE STRUCTURAL EXPLANATION. Emitting a specific rare token requires
      moving that token's LOGIT, and the logit is produced by `lm_head`.
      In Qwen2.5, `lm_head` is TIED to the input embedding, and our LoRA
      target set deliberately excludes it (E16 H3: adapting a tied lm_head
      would also adapt the embedding). Full fine-tuning updates that matrix
      directly; LoRA over q/k/v/o cannot touch it at all, and can only
      influence the logit indirectly through the residual stream.

H2 and H3 are compatible and would jointly explain the result: the adapter
shifts the hidden state somewhat, but with the output projection frozen it
cannot move one token's logit above a strong competitor.

WHAT WOULD DISTINGUISH THEM. If P(<|im_end|>) is unchanged -> H1. If it rose
but lost the argmax -> H2, and the size of the rise says how close it came.
H3 is structural rather than directly measurable here; what this script CAN do
is record how much of the base model's `lm_head` full fine-tuning actually
moved, which bounds how much of the effect was unavailable to LoRA.

WHAT THIS SCRIPT DOES NOT DO. It does not train a LoRA variant that adapts the
MLP or an untied output head. That is the experiment that would CONFIRM H3, it
costs another ~18-minute run per variant, and it is recorded as the obvious
next step rather than quietly skipped.

OUTCOMES, recorded after running. Measured at the position where <|im_end|>
should be emitted, over 100 held-out examples:

    model          P(<|im_end|>)   median rank   argmax
    base                 0.00000        31,600     0.0%
    full SFT             0.39513             1    79.0%
    LoRA  @2e-4          0.00021           160     0.0%
    LoRA  @2e-5          0.00006         2,217     0.0%
    QLoRA @2e-4          0.00019           194     0.0%

    H1  DISPROVED. LoRA did NOT learn nothing: it moved the stop token from
        rank 31,600 to rank 160, and its probability to 74.8x the base's.

    H2  CONFIRMED. It learned something and still lost decisively. Full
        fine-tuning reached P = 0.395 (140,607x the base) and rank 1, taking
        the argmax on 79% of examples. LoRA reached P = 0.00021 - about
        1,880x lower than full SFT - and rank 1 on none.

    H3  STRONGLY SUPPORTED, not proven. Full fine-tuning moved the tied
        embedding / lm_head matrix by relative 0.0136 (||dW|| 5.44 against
        ||W|| 399.89). For comparison, E17 measured the ATTENTION updates at
        relative 0.0013-0.0050. So the output projection received the LARGEST
        relative change of any matrix measured in this project - roughly 3-10x
        the attention matrices - and it is precisely the matrix our LoRA target
        set excludes.

THE LESSON, which is more general than the stop token. LoRA's capability
ceiling is set by WHICH matrices it can reach, not only by rank. A behaviour
whose mechanism lives in a matrix outside the target set is not reachable at
any rank. Here that behaviour was "stop talking", perplexity barely noticed,
and the target set - chosen for principled reasons (a tied lm_head cannot be
adapted without also adapting the input embedding) - is what caused it.

Run (server):
    python scripts/experiments/e20_stop_token_gap.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignlab.data import load_instruction_dataset
from alignlab.masking import expected_labels
from alignlab.paths import checkpoint_root, configure_hf_cache, repo_root

BASE_MODEL = "Qwen/Qwen2.5-1.5B"
REVISION = "8faed761d45a263340a0528343f099c05c9a4323"

ARMS = [
    ("full SFT", "sft-qwen1p5b-noRobots-001"),
    ("LoRA @2e-4", "lora-r16-lr2e-4-001"),
    ("LoRA @2e-5", "lora-r16-lr2e-5-001"),
    ("QLoRA @2e-4", "qlora-r16-lr2e-4-001"),
]


def load(run_name: str | None, ckpt_root: Path, device, dtype):
    if run_name is None:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, revision=REVISION, dtype=dtype
        )
        return model.to(device).eval()
    final = ckpt_root / run_name / "final"
    if (final / "adapter_config.json").exists():
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, revision=REVISION, dtype=dtype
        )
        # UNMERGED - bf16 merging is lossy (E19), and we are measuring
        # probabilities at the token level where that would matter.
        return PeftModel.from_pretrained(base, final).to(device).eval()
    return AutoModelForCausalLM.from_pretrained(final, dtype=dtype).to(device).eval()


def stop_token_stats(model, tokenizer, dataset, device, im_end: int, n: int) -> dict:
    """At the position where <|im_end|> SHOULD be emitted, what does the model think?

    For each eval example the trained sequence ends with the assistant's answer
    followed by <|im_end|>. The logits at the position immediately BEFORE that
    token are the model's prediction for it. We record the probability it
    assigns, its rank among all tokens, and whether it is the argmax - which is
    exactly what greedy decoding would need.
    """
    probs, ranks, argmax_hits = [], [], 0
    counted = 0

    for i in range(min(n, len(dataset))):
        row = dataset[i]
        try:
            ids, _, boundary = expected_labels(
                tokenizer, row["prompt"], row["completion"]
            )
        except ValueError:
            continue
        if im_end not in ids:
            continue
        pos = len(ids) - 1 - ids[::-1].index(im_end)  # last <|im_end|>
        if pos == 0:
            continue

        x = torch.tensor([ids[: pos + 1]], device=device)
        with torch.no_grad():
            logits = model(input_ids=x).logits[0, pos - 1].float()
        p = torch.softmax(logits, dim=-1)
        prob = float(p[im_end])
        rank = int((p > p[im_end]).sum()) + 1

        probs.append(prob)
        ranks.append(rank)
        argmax_hits += int(rank == 1)
        counted += 1

    if not counted:
        return {"counted": 0}
    return {
        "counted": counted,
        "mean_prob": sum(probs) / counted,
        "median_rank": sorted(ranks)[counted // 2],
        "argmax_fraction": argmax_hits / counted,
        "argmax_hits": argmax_hits,
    }


def lm_head_movement(ckpt_root: Path) -> dict | None:
    """How much did full fine-tuning move the output projection?

    This bounds the part of the effect LoRA structurally could not reproduce:
    our target set excludes lm_head, so whatever full fine-tuning achieved by
    moving it was unavailable to every PEFT arm.
    """
    final = ckpt_root / "sft-qwen1p5b-noRobots-001" / "final"
    if not final.exists():
        return None

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=REVISION, dtype=torch.bfloat16
    )
    sft = AutoModelForCausalLM.from_pretrained(final, dtype=torch.bfloat16)

    wb = base.get_input_embeddings().weight.detach().to(torch.float64)
    ws = sft.get_input_embeddings().weight.detach().to(torch.float64)
    delta = ws - wb

    out = {
        "frobenius_base": float(torch.linalg.norm(wb)),
        "frobenius_delta": float(torch.linalg.norm(delta)),
        "relative": float(torch.linalg.norm(delta) / torch.linalg.norm(wb)),
    }
    del base, sft
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-examples", type=int, default=100)
    parser.add_argument("--out", default="docs/phase4/e20_stop_token_gap.json")
    args = parser.parse_args()

    configure_hf_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    ckpt_root = checkpoint_root()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=REVISION)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    _, eval_ds, info = load_instruction_dataset(max_train=1, max_eval=args.eval_examples)

    print("=" * 84)
    print("E20 - the stop-token capability gap")
    print("=" * 84)
    print(f"  <|im_end|> id {im_end} | eval rows {len(eval_ds)} | "
          f"sha256 {info['eval_fingerprint']['sha256'][:16]}")
    print("\n  At the position where <|im_end|> should be emitted:")
    print(f"\n  {'model':<14} {'P(im_end)':>12} {'median rank':>12} {'argmax':>10} {'n':>5}")

    results = {}
    for label, run_name in [("base", None)] + ARMS:
        model = load(run_name, ckpt_root, device, dtype)
        stats = stop_token_stats(
            model, tokenizer, eval_ds, device, im_end, args.eval_examples
        )
        results[label] = stats
        if stats["counted"]:
            print(f"  {label:<14} {stats['mean_prob']:>12.5f} "
                  f"{stats['median_rank']:>12} "
                  f"{100 * stats['argmax_fraction']:>9.1f}% {stats['counted']:>5}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    base_p = results["base"]["mean_prob"]
    full_p = results["full SFT"]["mean_prob"]
    lora_p = results["LoRA @2e-4"]["mean_prob"]

    print("\n--- WHICH HYPOTHESIS? ---")
    print(f"  base       P(<|im_end|>) = {base_p:.5f}")
    print(f"  LoRA @2e-4 P(<|im_end|>) = {lora_p:.5f}  "
          f"({lora_p / base_p:.1f}x the base)" if base_p > 0 else "")
    print(f"  full SFT   P(<|im_end|>) = {full_p:.5f}  "
          f"({full_p / base_p:.1f}x the base)" if base_p > 0 else "")

    h1 = abs(lora_p - base_p) < 0.01 * max(base_p, 1e-9)
    h2 = lora_p > base_p and results["LoRA @2e-4"]["argmax_fraction"] < 0.5
    print(f"\n  H1 (LoRA learned nothing)      : {h1}")
    print(f"  H2 (learned some, lost argmax) : {h2}")
    if h2:
        print("     -> the adapter DID move the stop token's probability up, but")
        print("        not far enough to win greedy decoding.")

    print("\n--- H3: how much did full fine-tuning move the OUTPUT PROJECTION? ---")
    movement = lm_head_movement(ckpt_root)
    if movement:
        print(f"  ||W_embed||        {movement['frobenius_base']:.2f}")
        print(f"  ||dW_embed||       {movement['frobenius_delta']:.4f}")
        print(f"  relative           {movement['relative']:.5f}")
        print("\n  lm_head is TIED to this matrix, and our LoRA target set")
        print("  (q/k/v/o) excludes it. Full fine-tuning moved it; every PEFT arm")
        print("  here structurally could not. That is the leading explanation for")
        print("  a capability gap that perplexity barely registers.")
        print("\n  NOT CONFIRMED: this is consistent with H3, not proof of it. The")
        print("  decisive experiment is a LoRA variant adapting the MLP (or an")
        print("  untied output head) - one more ~18-minute run, recorded as the")
        print("  next step rather than skipped.")

    payload = {
        "im_end_id": im_end,
        "eval_fingerprint": info["eval_fingerprint"],
        "results": results,
        "lm_head_movement": movement,
        "hypotheses": {"H1_learned_nothing": bool(h1), "H2_learned_but_lost_argmax": bool(h2)},
    }
    out = repo_root() / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n  evidence: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
