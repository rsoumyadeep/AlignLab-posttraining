"""E13 - before/after evaluation of the SFT model against the base model.

PROJECT_INSTRUCTIONS section 20 requires a before/after story for every phase
that changes model behaviour, and section 7 requires answering "How do you know
the model actually got better?" without relying on a single metric.

WHAT IS MEASURED

  1. COMPLETION-ONLY perplexity on the held-out eval split. Completion-only is
     the point: perplexity over the whole sequence would average in the
     templated prompt, which is not what training optimised and which E12
     showed behaves differently from the answer region. Masked exactly as
     training was.

  2. PROMPT-REGION perplexity, as a CONTROL. Training put no gradient on those
     tokens, so a large change there is a warning sign, not a success. Reporting
     only the metric that is supposed to improve is how one fools oneself.

  3. Qualitative generation from fixed prompts with a fixed seed, for both
     models, decoded identically. The base model is expected to continue rather
     than answer, and to fail to stop.

  4. Stop-token behaviour. The base model has never emitted <|im_end|>; the SFT
     model should have learned to. This is measurable (does generation
     terminate before max_new_tokens?) rather than impressionistic, and it is
     arguably the most concrete thing one epoch of SFT buys.

WHAT THIS DESIGN DELIBERATELY AVOIDS. A single headline number. Perplexity on
one dataset measures fit to THAT distribution, not instruction-following
quality, and a lower number here is not evidence of a better assistant. The
report separates MEASURED from INTERPRETED accordingly.

NO LLM-AS-JUDGE HERE. That belongs to Phase 7, needs a judge model this project
has not chosen, and would add an unvalidated dependency to a phase whose job is
to establish the baseline honestly.

Run:
    python scripts/experiments/e13_sft_before_after.py \
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
from alignlab.masking import IGNORE_INDEX, expected_labels
from alignlab.paths import configure_hf_cache

BASE_MODEL = "Qwen/Qwen2.5-1.5B"
REVISION = "8faed761d45a263340a0528343f099c05c9a4323"

# Fixed qualitative prompts. Deliberately spread across task types so a single
# capability does not dominate the impression. Held constant between models.
QUALITATIVE_PROMPTS = [
    "Write a two-sentence summary of why the sky appears blue.",
    "List three practical tips for someone learning to cook.",
    "What is the difference between a list and a tuple in Python?",
    "Write a short, friendly email declining a meeting invitation.",
]


def completion_and_prompt_nll(model, tokenizer, dataset, device, limit: int):
    """Return (completion NLL, prompt NLL, token counts) over a dataset.

    Token-weighted, not example-weighted: a mean of per-example means would
    silently weight a 5-token answer the same as a 500-token one. Accumulating
    total NLL and total tokens is the honest aggregate.
    """
    total_completion_nll = 0.0
    total_completion_tokens = 0
    total_prompt_nll = 0.0
    total_prompt_tokens = 0
    skipped = 0

    for i in range(min(limit, len(dataset))):
        row = dataset[i]
        try:
            ids, labels, boundary = expected_labels(
                tokenizer, row["prompt"], row["completion"]
            )
        except ValueError:
            skipped += 1
            continue
        if len(ids) < 2 or boundary >= len(ids):
            skipped += 1
            continue

        input_ids = torch.tensor([ids], device=device)
        completion_labels = torch.tensor([labels], device=device)
        prompt_labels = input_ids.clone()
        prompt_labels[:, boundary:] = IGNORE_INDEX

        with torch.no_grad():
            logits = model(input_ids=input_ids).logits

        shift_logits = logits[:, :-1, :].float()

        for labels_tensor, (nll_acc, tok_acc) in (
            (completion_labels, ("completion", None)),
            (prompt_labels, ("prompt", None)),
        ):
            shift_labels = labels_tensor[:, 1:]
            n = int((shift_labels != IGNORE_INDEX).sum())
            if n == 0:
                continue
            summed = float(
                F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                    reduction="sum",
                )
            )
            if nll_acc == "completion":
                total_completion_nll += summed
                total_completion_tokens += n
            else:
                total_prompt_nll += summed
                total_prompt_tokens += n

    return {
        "completion_nll": total_completion_nll / max(total_completion_tokens, 1),
        "completion_tokens": total_completion_tokens,
        "prompt_nll": total_prompt_nll / max(total_prompt_tokens, 1),
        "prompt_tokens": total_prompt_tokens,
        "examples_scored": min(limit, len(dataset)) - skipped,
        "examples_skipped": skipped,
    }


def generate_samples(model, tokenizer, device, max_new_tokens: int, seed: int):
    """Greedy generation from the fixed prompts, identical for both models.

    Greedy rather than sampled so the comparison is deterministic and the two
    models differ only in their weights. Sampling would add a second source of
    variation to a comparison that is trying to isolate one.
    """
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    results = []

    for prompt in QUALITATIVE_PROMPTS:
        torch.manual_seed(seed)
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        n_in = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                # Accept EITHER terminator. The base model only knows
                # <|endoftext|>; the SFT model should have learned <|im_end|>.
                eos_token_id=[im_end, tokenizer.eos_token_id],
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = out[0][n_in:]
        stopped = bool(
            len(generated) < max_new_tokens
            or generated[-1].item() in (im_end, tokenizer.eos_token_id)
        )
        results.append(
            {
                "prompt": prompt,
                "response": tokenizer.decode(generated, skip_special_tokens=True).strip(),
                "n_generated": int(len(generated)),
                "hit_length_cap": int(len(generated)) >= max_new_tokens,
                "stopped_naturally": stopped,
                "emitted_im_end": im_end in generated.tolist(),
            }
        )
    return results


def evaluate(name, model_path, revision, tokenizer, dataset, device, dtype, args):
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    print(f"  path/id : {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, revision=revision, dtype=dtype
    ).to(device).eval()

    nll = completion_and_prompt_nll(
        model, tokenizer, dataset, device, args.eval_examples
    )
    completion_ppl = float(torch.exp(torch.tensor(nll["completion_nll"])))
    prompt_ppl = float(torch.exp(torch.tensor(nll["prompt_nll"])))

    print(f"\n  completion NLL {nll['completion_nll']:.4f}  ppl {completion_ppl:9.3f}"
          f"  ({nll['completion_tokens']:,} tokens)   <- what training optimised")
    print(f"  prompt     NLL {nll['prompt_nll']:.4f}  ppl {prompt_ppl:9.3f}"
          f"  ({nll['prompt_tokens']:,} tokens)   <- CONTROL, untrained")
    print(f"  examples scored {nll['examples_scored']}, skipped {nll['examples_skipped']}")

    samples = generate_samples(model, tokenizer, device, args.max_new_tokens, args.seed)
    stopped = sum(s["stopped_naturally"] for s in samples)
    emitted = sum(s["emitted_im_end"] for s in samples)
    print(f"\n  generation: {stopped}/{len(samples)} stopped naturally, "
          f"{emitted}/{len(samples)} emitted <|im_end|>")
    for s in samples:
        print(f"\n  --- {s['prompt']}")
        body = s["response"][:400].replace("\n", "\n      ")
        print(f"      {body}")
        print(f"      [{s['n_generated']} tokens, stopped={s['stopped_naturally']}, "
              f"im_end={s['emitted_im_end']}]")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "name": name,
        "path": str(model_path),
        **nll,
        "completion_ppl": completion_ppl,
        "prompt_ppl": prompt_ppl,
        "samples": samples,
        "stopped_naturally": stopped,
        "emitted_im_end": emitted,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-model", required=True, help="path to the SFT final/ dir")
    parser.add_argument("--eval-examples", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="docs/phase3/e13_before_after.json")
    args = parser.parse_args()

    configure_hf_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=REVISION)
    _, eval_ds, info = load_instruction_dataset(max_train=1, max_eval=args.eval_examples)

    print("=" * 78)
    print("E13 - SFT before/after")
    print("=" * 78)
    print(f"  device {device}, dtype {dtype}, seed {args.seed}")
    print(f"  eval split: {info['eval_fingerprint']['n_rows']} rows, "
          f"sha256 {info['eval_fingerprint']['sha256'][:16]}")

    before = evaluate(
        "BEFORE - base Qwen2.5-1.5B", BASE_MODEL, REVISION, tokenizer, eval_ds,
        device, dtype, args,
    )
    after = evaluate(
        "AFTER - SFT", args.sft_model, None, tokenizer, eval_ds, device, dtype, args,
    )

    print("\n" + "=" * 78)
    print("COMPARISON")
    print("=" * 78)
    print(f"  {'metric':<34} {'base':>12} {'SFT':>12} {'change':>12}")
    for label, key in (
        ("completion perplexity (TRAINED)", "completion_ppl"),
        ("prompt perplexity (CONTROL)", "prompt_ppl"),
    ):
        b, a = before[key], after[key]
        print(f"  {label:<34} {b:>12.3f} {a:>12.3f} {100 * (a - b) / b:>11.1f}%")
    print(f"  {'stopped naturally':<34} {before['stopped_naturally']:>12} "
          f"{after['stopped_naturally']:>12}")
    print(f"  {'emitted <|im_end|>':<34} {before['emitted_im_end']:>12} "
          f"{after['emitted_im_end']:>12}")

    print("\n--- WHAT THIS SHOWS / DOES NOT SHOW ---")
    print("  MEASURED   : completion perplexity on a held-out split of the SAME")
    print("               dataset, and whether generation terminates.")
    print("  NOT SHOWN  : that the model is a better assistant in general. Lower")
    print("               perplexity on no_robots' test split means better fit to")
    print("               no_robots, which is what one epoch of it should buy.")
    print("  CONTROL    : the prompt region was never trained. A large change")
    print("               there would indicate drift, not success.")

    payload = {
        "base": before,
        "sft": after,
        "eval_fingerprint": info["eval_fingerprint"],
        "settings": vars(args),
        "device": str(device),
        "dtype": str(dtype),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n  written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
