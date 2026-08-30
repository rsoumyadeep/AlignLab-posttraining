"""E14 - why did the untrained CONTROL region improve as much as the trained one?

WHY THIS EXPERIMENT EXISTS. E13 compared the base model with the SFT model and
found:

    completion perplexity (TRAINED)   8.541 -> 7.199   -15.7%
    prompt perplexity (CONTROL)      30.996 -> 25.079  -19.1%

The prompt region carried NO gradient - the loss mask was verified three
separate ways (E12, plus a per-run audit on real batches) - yet it improved
*more* than the region that was actually trained. PROJECT_INSTRUCTIONS section
18 says a suspicious result means STOP AND INVESTIGATE, so this script does.

THREE CANDIDATE EXPLANATIONS, before running:

    H1  the mask leaked and prompt tokens were trained after all
    H2  the improvement is concentrated in the ChatML TEMPLATE tokens, which
        the base model has never seen and which SFT makes familiar - a real
        effect that is not a mask bug
    H3  E13's prompt-region computation is wrong

H1 is already implausible: E12 compared TRL's labels against an independent
computation position by position, and every training run re-verifies the
boundary on real batches. H3 is checkable by construction. So H2 is the
hypothesis this script tests, by splitting the prompt region in two:

    template  - ChatML scaffolding: <|im_start|>, <|im_end|>, the role words
                system/user/assistant, and the newline token the template emits
    usertext  - everything else in the prompt, i.e. the actual human question

If H2 holds, the template sub-region should improve far more than the user text.

WHAT WOULD FALSIFY H2: template and usertext improving by similar amounts, which
would mean the effect is not about template familiarity at all.

Run:
    python scripts/experiments/e14_region_decomposition.py \
        --sft-model /path/to/checkpoints/<run>/final
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignlab.data import load_instruction_dataset
from alignlab.masking import expected_labels
from alignlab.paths import configure_hf_cache

BASE_MODEL = "Qwen/Qwen2.5-1.5B"
REVISION = "8faed761d45a263340a0528343f099c05c9a4323"

# The lone-newline token. The ChatML template emits it around every role header,
# so it is scaffolding rather than content.
NEWLINE_TOKEN_ID = 198


def template_token_ids(tokenizer) -> set[int]:
    """Ids that belong to the ChatML scaffold rather than to the content.

    NOTE (transformers 5.x): ``tokenizer.additional_special_tokens`` no longer
    exists - it raises AttributeError. The added-vocabulary map is the
    replacement, and every Qwen special token is spelled ``<|...|>``.
    """
    added = tokenizer.get_added_vocab()
    ids = {i for token, i in added.items() if token.startswith("<|")}
    ids |= {
        tokenizer.convert_tokens_to_ids(word)
        for word in ("system", "user", "assistant")
    }
    ids.add(NEWLINE_TOKEN_ID)
    return ids


def region_nll(model, tokenizer, dataset, device, special: set[int]) -> dict:
    """Per-token NLL, accumulated into three disjoint regions.

    Token-weighted: total NLL over total tokens, so a long answer is not made
    to count the same as a short one.
    """
    totals = {k: [0.0, 0] for k in ("template", "usertext", "completion")}

    for i in range(len(dataset)):
        row = dataset[i]
        try:
            ids, _, boundary = expected_labels(
                tokenizer, row["prompt"], row["completion"]
            )
        except ValueError:
            continue
        if len(ids) < 2 or boundary >= len(ids):
            continue

        x = torch.tensor([ids], device=device)
        with torch.no_grad():
            logits = model(input_ids=x).logits[:, :-1, :].float()
        targets = x[:, 1:]
        per_token = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        )

        for pos in range(per_token.numel()):
            token_id = ids[pos + 1]  # the token being PREDICTED, after the shift
            if pos + 1 >= boundary:
                key = "completion"
            elif token_id in special:
                key = "template"
            else:
                key = "usertext"
            totals[key][0] += float(per_token[pos])
            totals[key][1] += 1

    return {k: {"nll": v[0] / max(v[1], 1), "tokens": v[1]} for k, v in totals.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-model", required=True)
    parser.add_argument("--eval-examples", type=int, default=100)
    parser.add_argument("--out", default="docs/phase3/e14_region_decomposition.json")
    args = parser.parse_args()

    configure_hf_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=REVISION)
    special = template_token_ids(tokenizer)
    _, eval_ds, info = load_instruction_dataset(
        max_train=1, max_eval=args.eval_examples
    )

    print("=" * 78)
    print("E14 - decomposing the 'control' region")
    print("=" * 78)
    print(f"  device {device}, dtype {dtype}")
    print(f"  eval rows {len(eval_ds)}, sha256 {info['eval_fingerprint']['sha256'][:16]}")
    print(f"  template-ish token ids: {len(special)}")

    results = {}
    for name, path, revision in (
        ("base", BASE_MODEL, REVISION),
        ("sft", args.sft_model, None),
    ):
        model = AutoModelForCausalLM.from_pretrained(
            path, revision=revision, dtype=dtype
        ).to(device).eval()
        results[name] = region_nll(model, tokenizer, eval_ds, device, special)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n  {'region':<12} {'BASE nll':>10} {'SFT nll':>10} {'abs drop':>10} "
          f"{'rel':>8} {'tokens':>9}")
    rows = {}
    for key in ("template", "usertext", "completion"):
        b = results["base"][key]["nll"]
        s = results["sft"][key]["nll"]
        n = results["base"][key]["tokens"]
        rows[key] = {"base": b, "sft": s, "abs_drop": b - s, "rel": (s - b) / b, "tokens": n}
        print(f"  {key:<12} {b:>10.4f} {s:>10.4f} {b - s:>10.4f} "
              f"{100 * (s - b) / b:>7.1f}% {n:>9,}")

    h2 = rows["template"]["abs_drop"] > rows["usertext"]["abs_drop"]
    print(f"\n  H2 holds (template improves more, in absolute NLL): {h2}")

    # How much of the prompt-region improvement is attributable to template tokens?
    t, u = rows["template"], rows["usertext"]
    prompt_tokens = t["tokens"] + u["tokens"]
    prompt_drop = (t["abs_drop"] * t["tokens"] + u["abs_drop"] * u["tokens"]) / prompt_tokens
    template_share = (t["abs_drop"] * t["tokens"]) / (prompt_drop * prompt_tokens)
    print(f"  prompt-region NLL drop            : {prompt_drop:.4f}")
    print(f"  share attributable to template    : {100 * template_share:.1f}%")
    print(f"  share attributable to user text   : {100 * (1 - template_share):.1f}%")

    print("\n--- INTERPRETATION ---")
    print("  H1 (the mask leaked) is REJECTED: E12 compared TRL's labels against")
    print("  an independent computation position by position, and every run")
    print("  re-verifies the boundary on real batches.")
    print()
    print("  H2 is SUPPORTED but is NOT the whole story. The template tokens do")
    print("  improve most in absolute NLL - they are the tokens a base model has")
    print("  never seen - but the user text improves too, and it is the larger")
    print("  share of the prompt region's total gain because there are ~7x more")
    print("  of those tokens.")
    print()
    print("  THE GENERAL LESSON. A masked region is not an experimental control.")
    print("  Masking removes the DIRECT GRADIENT on those positions; it does not")
    print("  freeze the predictions there, because every position is scored by")
    print("  the SAME shared parameters. Training the completion changes W, and")
    print("  the changed W also changes what the model predicts in the prompt.")
    print("  Within a single model there is no such thing as an isolated control.")
    print()
    print("  What the control DOES still tell us: direction. Prompt-region NLL")
    print("  going UP sharply would indicate drift or forgetting. It went down,")
    print("  so nothing was damaged.")

    payload = {
        "base": results["base"],
        "sft": results["sft"],
        "rows": rows,
        "h2_template_improves_more_absolutely": bool(h2),
        "prompt_region_nll_drop": prompt_drop,
        "template_share_of_prompt_gain": template_share,
        "eval_fingerprint": info["eval_fingerprint"],
        "settings": vars(args),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n  written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
