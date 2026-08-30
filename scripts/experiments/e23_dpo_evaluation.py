"""E23 - SFT baseline vs DPO, across the pre-registered beta sweep.

THE EVALUATION SET AND METRICS WERE FIXED BEFORE TRAINING. See
docs/phase6/BETA_PREREGISTRATION.md, committed before any DPO code existed.
Nothing here was chosen after seeing a result.

PRIMARY METRIC
    Preference accuracy under the SUM of token log-probabilities - the
    published DPO objective, and the objective Phase 6 trained.
    Phase 5 baseline: 47.2%.

DIAGNOSTIC, NEVER THE HEADLINE
    The same accuracy under the per-token MEAN. Phase 5 measured 58.3% on
    identical models and data; the gap is length, not quality. Reporting the
    MEAN figure as though it were the published-objective result would be
    substituting a different objective under the same name.

LENGTH IS TRACKED THROUGHOUT, because Phase 5 measured chosen responses at
56.5% longer in tokens than rejected ones, and the SUM objective is
length-sensitive. "Prefer longer" is therefore a gradient direction DPO can
exploit, and a preference-accuracy gain accompanied by a large length increase
is not evidence of better answers.

WHAT IS MEASURED, per model:
    1. preference accuracy, SUM   (primary)
    2. preference accuracy, MEAN  (diagnostic)
    3. log-probability statistics on chosen and rejected
    4. implicit-reward accuracy and margin against the frozen SFT reference
    5. KL(policy || SFT reference) on the same sequences
    6. qualitative generations from FIXED prompts, greedy
    7. generated response LENGTH, and stop-token behaviour

HYPOTHESES: recorded in the pre-registration, restated here.
    H1 all betas reduce training loss below log 2 = 0.693147
    H2 SUM preference accuracy improves above 47.2% for at least one beta
    H3 KL from the reference is monotone decreasing in beta
    H4 generated length INCREASES for at least one beta
    H5 SUM and MEAN can move in different directions; if they do, the
       disagreement is preserved

Run (server, after the sweep):
    python scripts/experiments/e23_dpo_evaluation.py
"""

from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignlab.dpo import dpo_loss, preference_accuracy
from alignlab.logprobs import sequence_kl, sequence_logprobs
from alignlab.masking import expected_labels
from alignlab.paths import checkpoint_root, configure_hf_cache, output_root, repo_root
from alignlab.preference import load_preference_dataset
from alignlab.storage import GIB

BASE_MODEL = "Qwen/Qwen2.5-1.5B"
REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
SFT_RUN = "sft-qwen1p5b-noRobots-001"

# Pre-registered beta set.
BETAS = (0.01, 0.1, 0.5)

# The SAME four prompts used in Phase 3's E13 and Phase 4's E19, so
# qualitative behaviour is comparable across three phases.
QUALITATIVE_PROMPTS = [
    "Write a two-sentence summary of why the sky appears blue.",
    "List three practical tips for someone learning to cook.",
    "What is the difference between a list and a tuple in Python?",
    "Write a short, friendly email declining a meeting invitation.",
]


def load(path, device, dtype):
    if not path.exists():
        return None
    return AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()


def build_batches(tokenizer, dataset, device, max_length):
    batches = []
    for i in range(len(dataset)):
        row = dataset[i]
        try:
            c_ids, c_lab, _ = expected_labels(tokenizer, row["prompt"], row["chosen"])
            r_ids, r_lab, _ = expected_labels(tokenizer, row["prompt"], row["rejected"])
        except ValueError:
            continue
        if len(c_ids) > max_length or len(r_ids) > max_length:
            continue
        batches.append({
            "chosen_ids": torch.tensor([c_ids], device=device),
            "chosen_labels": torch.tensor([c_lab], device=device),
            "rejected_ids": torch.tensor([r_ids], device=device),
            "rejected_labels": torch.tensor([r_lab], device=device),
        })
    return batches


@torch.no_grad()
def score_model(model, reference, batches, beta):
    sum_correct = mean_correct = reward_correct = 0
    margins, kls = [], []
    chosen_lp, rejected_lp, chosen_tok, rejected_tok = [], [], [], []

    for b in batches:
        pc = sequence_logprobs(model(input_ids=b["chosen_ids"]).logits, b["chosen_labels"])
        pr = sequence_logprobs(model(input_ids=b["rejected_ids"]).logits, b["rejected_labels"])
        rc = sequence_logprobs(reference(input_ids=b["chosen_ids"]).logits, b["chosen_labels"])
        rr = sequence_logprobs(reference(input_ids=b["rejected_ids"]).logits, b["rejected_labels"])

        sum_correct += int(preference_accuracy(pc, pr)[0])
        mean_correct += int(preference_accuracy(pc, pr, length_normalise=True)[0])
        _, stats = dpo_loss(pc, pr, rc, rr, beta=beta)
        reward_correct += int(stats.reward_margin > 0)
        margins.append(stats.reward_margin)

        kl, _ = sequence_kl(
            model(input_ids=b["chosen_ids"]).logits,
            reference(input_ids=b["chosen_ids"]).logits,
            b["chosen_labels"],
        )
        kls.append(float(kl[0]))

        chosen_lp.append(float(pc.sum_logprob[0]))
        rejected_lp.append(float(pr.sum_logprob[0]))
        chosen_tok.append(int(pc.n_tokens[0]))
        rejected_tok.append(int(pr.n_tokens[0]))

    n = max(len(batches), 1)
    return {
        "pairs": len(batches),
        "preference_accuracy_sum": sum_correct / n,
        "preference_accuracy_mean": mean_correct / n,
        "reward_accuracy": reward_correct / n,
        "reward_margin": sum(margins) / n,
        "kl_from_reference_mean": sum(kls) / n,
        "kl_from_reference_median": sorted(kls)[n // 2],
        "chosen_logp_mean": sum(chosen_lp) / n,
        "rejected_logp_mean": sum(rejected_lp) / n,
        "chosen_tokens_mean": sum(chosen_tok) / n,
        "rejected_tokens_mean": sum(rejected_tok) / n,
    }


@torch.no_grad()
def generate(model, tokenizer, device, max_new_tokens):
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    out = []
    for prompt in QUALITATIVE_PROMPTS:
        torch.manual_seed(42)
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        n_in = inputs["input_ids"].shape[1]
        gen = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            eos_token_id=[im_end, tokenizer.eos_token_id],
            pad_token_id=tokenizer.pad_token_id,
        )
        body = gen[0][n_in:]
        out.append({
            "prompt": prompt,
            "response": tokenizer.decode(body, skip_special_tokens=True).strip(),
            "n_generated": int(len(body)),
            "emitted_im_end": im_end in body.tolist(),
            "hit_cap": int(len(body)) >= max_new_tokens,
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-examples", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--out", default="docs/phase6/e23_dpo_evaluation.json")
    args = parser.parse_args()

    configure_hf_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    ckpt_root = checkpoint_root()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=REVISION)
    _, eval_ds, info = load_preference_dataset(max_train=1, max_eval=args.eval_examples)

    print("=" * 92)
    print("E23 - SFT baseline vs DPO (pre-registered beta sweep)")
    print("=" * 92)
    print(f"  device {device}, dtype {dtype}")
    print(f"  eval split: {len(eval_ds)} pairs, sha256 "
          f"{info['eval_fingerprint']['sha256'][:16]}")
    print(f"  PRIMARY metric: preference accuracy under SUM (published objective)")
    print(f"  Phase 5 baseline: 47.2% SUM / 58.3% MEAN (diagnostic)")

    reference = load(ckpt_root / SFT_RUN / "final", device, dtype)
    if reference is None:
        print("  SFT checkpoint missing - cannot evaluate")
        return 2
    batches = build_batches(tokenizer, eval_ds, device, args.max_length)
    print(f"  usable pairs: {len(batches)}")

    models = [("SFT (baseline)", ckpt_root / SFT_RUN / "final", None)]
    for beta in BETAS:
        models.append((f"DPO beta={beta}", ckpt_root / f"dpo-beta{beta}-sum" / "final", beta))

    results = {}
    for label, path, beta in models:
        model = load(path, device, dtype)
        if model is None:
            print(f"  SKIP {label}: {path} missing")
            continue
        scored = score_model(model, reference, batches, beta if beta else 0.1)
        samples = generate(model, tokenizer, device, args.max_new_tokens)
        lengths = [s["n_generated"] for s in samples]
        scored["generation"] = {
            "samples": samples,
            "mean_length": sum(lengths) / len(lengths),
            "max_length": max(lengths),
            "emitted_im_end": sum(s["emitted_im_end"] for s in samples),
            "hit_cap": sum(s["hit_cap"] for s in samples),
        }
        results[label] = scored
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------- the table
    print("\n" + "=" * 92)
    print("PRIMARY: preference accuracy under SUM   |   DIAGNOSTIC: MEAN")
    print("=" * 92)
    print(f"  {'model':<18} {'SUM (primary)':>14} {'MEAN (diag)':>13} "
          f"{'reward acc':>11} {'margin':>9} {'KL median':>10}")
    for label, r in results.items():
        print(f"  {label:<18} {100 * r['preference_accuracy_sum']:>13.1f}% "
              f"{100 * r['preference_accuracy_mean']:>12.1f}% "
              f"{100 * r['reward_accuracy']:>10.1f}% "
              f"{r['reward_margin']:>+9.4f} {r['kl_from_reference_median']:>10.4f}")

    print("\n--- LENGTH ANALYSIS (mandatory: Phase 5 measured a strong length effect) ---")
    print(f"  {'model':<18} {'gen mean len':>13} {'gen max':>9} "
          f"{'im_end':>8} {'hit cap':>9} {'chosen tok':>11} {'rejected tok':>13}")
    for label, r in results.items():
        g = r["generation"]
        print(f"  {label:<18} {g['mean_length']:>13.1f} {g['max_length']:>9} "
              f"{g['emitted_im_end']}/4{'':>4} {g['hit_cap']}/4{'':>5} "
              f"{r['chosen_tokens_mean']:>11.1f} {r['rejected_tokens_mean']:>13.1f}")

    baseline = results.get("SFT (baseline)")
    if baseline:
        print("\n--- CHANGE FROM THE SFT BASELINE ---")
        b_len = baseline["generation"]["mean_length"]
        print(f"  {'model':<18} {'d SUM':>9} {'d MEAN':>9} {'d gen len':>11} "
              f"{'len ratio':>10}")
        for label, r in results.items():
            if label == "SFT (baseline)":
                continue
            d_sum = 100 * (r["preference_accuracy_sum"] - baseline["preference_accuracy_sum"])
            d_mean = 100 * (r["preference_accuracy_mean"] - baseline["preference_accuracy_mean"])
            d_len = r["generation"]["mean_length"] - b_len
            ratio = r["generation"]["mean_length"] / max(b_len, 1)
            print(f"  {label:<18} {d_sum:>+8.1f}p {d_mean:>+8.1f}p "
                  f"{d_len:>+10.1f} {ratio:>9.2f}x")

    # ----------------------------------------------------------- qualitative
    print("\n--- QUALITATIVE, same prompt across models, greedy ---")
    prompt = QUALITATIVE_PROMPTS[3]
    for label, r in results.items():
        s = next(x for x in r["generation"]["samples"] if x["prompt"] == prompt)
        body = s["response"][:230].replace("\n", " ")
        print(f"\n  [{label}] {s['n_generated']} tok, im_end={s['emitted_im_end']}")
        print(f"    {body}")

    print("\n" + "=" * 92)
    print("WHAT THIS SHOWS / DOES NOT SHOW")
    print("=" * 92)
    print("  MEASURED   : preference accuracy under both objectives, implicit")
    print("               reward statistics, KL from the SFT reference, and")
    print("               generated response length.")
    print("  NOT SHOWN  : that any model is a better assistant. Preference")
    print("               accuracy on UltraFeedback's test split measures fit to")
    print("               UltraFeedback's preferences.")
    print("  LENGTH     : if SUM accuracy rose while generated length rose")
    print("               sharply, the gain is CONFOUNDED with length and must")
    print("               not be reported as a quality improvement.")
    print("  DISAGREEMENT: where SUM and MEAN move differently, both are")
    print("               reported. Neither is resolved in favour of the other.")

    payload = {
        "eval_fingerprint": info["eval_fingerprint"],
        "pairs": len(batches),
        "phase5_baseline": {"sum": 0.472, "mean": 0.583},
        "results": results,
        "settings": vars(args),
    }
    out = repo_root() / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n  evidence: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
