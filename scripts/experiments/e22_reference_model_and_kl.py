"""E22 - the reference model, the KL constraint, and the pre-DPO baseline.

Phase 5 is mostly conceptual, but two of its central ideas can be MEASURED on
models this project already trained, rather than only described:

  THE KL CONSTRAINT. RLHF adds a penalty on KL(policy || reference) to stop the
  policy drifting away from the pretrained model while chasing reward. That is
  usually explained in words. Here it gets a number: how far did Phase 3's SFT
  actually move the policy from the base model, in the units the penalty is
  written in?

  THE REFERENCE MODEL. DPO needs a frozen pi_ref, and standard practice is to
  use the SFT checkpoint - the same model DPO starts from. That has an exact,
  checkable consequence: at initialisation pi = pi_ref, so DPO's implicit
  reward is EXACTLY ZERO for every sequence. If it is not zero, the reference
  is wired wrong, and this is the cheapest possible place to find that out.

  THE PRE-DPO BASELINE. Before any preference training, does the SFT model
  already assign higher log-probability to the preferred response than to the
  rejected one? Whatever that number is, it is the baseline Phase 6 must beat -
  and it must be measured BEFORE running DPO, otherwise there is nothing to
  compare against.

HYPOTHESES, recorded before running:

  H1  KL(SFT || base) > 0 and is small in absolute terms - one epoch of SFT
      perturbs rather than rewrites (E17 measured ||dW||/||W|| at 0.0013-0.0050)
  H2  KL(LoRA || base) < KL(SFT || base): LoRA changes 0.28% of parameters and
      cannot touch the output projection, so it should move the distribution
      less
  H3  with policy = reference = the SFT model, the DPO implicit reward is
      EXACTLY 0 for every sequence
  H4  the SFT model already prefers chosen over rejected on MORE than 50% of
      pairs - it was trained on instruction data, and preferred responses are
      generally better-formed, so some signal should exist before DPO
  H5  that preference rate is length-confounded: because the published DPO
      objective sums log-probabilities and chosen responses are longer
      (E21: 56.5% longer in tokens), the SUM comparison should favour chosen
      LESS than the length-normalised MEAN comparison does

WHAT THIS IS NOT. No DPO training. No PPO implementation. No reward model is
trained. This measures properties of existing models on preference data.

Run (server):
    python scripts/experiments/e22_reference_model_and_kl.py
"""

from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignlab.logprobs import logprob_ratio, sequence_kl, sequence_logprobs
from alignlab.masking import expected_labels
from alignlab.paths import checkpoint_root, configure_hf_cache, repo_root
from alignlab.preference import load_preference_dataset

BASE_MODEL = "Qwen/Qwen2.5-1.5B"
REVISION = "8faed761d45a263340a0528343f099c05c9a4323"

POLICIES = [
    ("full SFT", "sft-qwen1p5b-noRobots-001"),
    ("LoRA @2e-4", "lora-r16-lr2e-4-001"),
    ("QLoRA @2e-4", "qlora-r16-lr2e-4-001"),
]


def load_model(run_name: str | None, ckpt_root, device, dtype):
    if run_name is None:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, revision=REVISION, dtype=dtype
        )
        return model.to(device).eval()
    final = ckpt_root / run_name / "final"
    if not final.exists():
        return None
    if (final / "adapter_config.json").exists():
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, revision=REVISION, dtype=dtype
        )
        # UNMERGED: Phase 4 measured bf16 merging as lossy (12,460x worse than
        # fp32), and this script compares log-probabilities at token level.
        return PeftModel.from_pretrained(base, final).to(device).eval()
    return AutoModelForCausalLM.from_pretrained(final, dtype=dtype).to(device).eval()


def build_batch(tokenizer, row, key, device):
    """Tokenize one (prompt, response) pair into ids + completion-masked labels."""
    ids, labels, boundary = expected_labels(tokenizer, row["prompt"], row[key])
    return (
        torch.tensor([ids], device=device),
        torch.tensor([labels], device=device),
        boundary,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--out", default="docs/phase5/e22_reference_and_kl.json")
    args = parser.parse_args()

    configure_hf_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    ckpt_root = checkpoint_root()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=REVISION)
    _, eval_ds, info = load_preference_dataset(max_train=1, max_eval=args.examples)

    print("=" * 84)
    print("E22 - reference model, KL constraint, pre-DPO baseline")
    print("=" * 84)
    print(f"  device {device}, dtype {dtype}")
    print(f"  preference eval: {len(eval_ds)} pairs, "
          f"sha256 {info['eval_fingerprint']['sha256'][:16]}")

    # Pre-tokenize once so every model sees identical inputs.
    batches = []
    for i in range(len(eval_ds)):
        row = eval_ds[i]
        try:
            c_ids, c_lab, _ = build_batch(tokenizer, row, "chosen", device)
            r_ids, r_lab, _ = build_batch(tokenizer, row, "rejected", device)
        except ValueError:
            continue
        if c_ids.shape[1] > args.max_tokens or r_ids.shape[1] > args.max_tokens:
            continue
        batches.append((c_ids, c_lab, r_ids, r_lab))
    print(f"  usable pairs after length filter: {len(batches)}")

    reference = load_model(None, ckpt_root, device, dtype)

    results = {}
    for label, run_name in POLICIES:
        policy = load_model(run_name, ckpt_root, device, dtype)
        if policy is None:
            print(f"  SKIP {label}: checkpoint missing")
            continue

        kls, n_toks = [], []
        chosen_sum, rejected_sum = [], []
        chosen_mean, rejected_mean = [], []

        for c_ids, c_lab, r_ids, r_lab in batches:
            with torch.no_grad():
                p_logits = policy(input_ids=c_ids).logits
                r_logits = reference(input_ids=c_ids).logits
            kl, n = sequence_kl(p_logits, r_logits, c_lab)
            kls.append(float(kl[0]))
            n_toks.append(int(n[0]))

            with torch.no_grad():
                sc = sequence_logprobs(policy(input_ids=c_ids).logits, c_lab)
                sr = sequence_logprobs(policy(input_ids=r_ids).logits, r_lab)
            chosen_sum.append(float(sc.sum_logprob[0]))
            rejected_sum.append(float(sr.sum_logprob[0]))
            chosen_mean.append(float(sc.mean_logprob[0]))
            rejected_mean.append(float(sr.mean_logprob[0]))

        n = len(kls)
        prefers_sum = sum(1 for a, b in zip(chosen_sum, rejected_sum) if a > b)
        prefers_mean = sum(1 for a, b in zip(chosen_mean, rejected_mean) if a > b)

        results[label] = {
            "kl_mean": sum(kls) / n,
            "kl_median": sorted(kls)[n // 2],
            "kl_max": max(kls),
            "tokens_mean": sum(n_toks) / n,
            "prefers_chosen_sum": prefers_sum,
            "prefers_chosen_mean": prefers_mean,
            "pairs": n,
            "prefers_chosen_sum_fraction": prefers_sum / n,
            "prefers_chosen_mean_fraction": prefers_mean / n,
        }
        print(f"  {label:<14} KL(policy||base) mean {results[label]['kl_mean']:.4f} "
              f"median {results[label]['kl_median']:.4f}  "
              f"prefers chosen {prefers_sum}/{n} (sum) {prefers_mean}/{n} (mean)")

        del policy
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------- H1 / H2
    print("\n--- H1: did SFT move the policy away from the base at all? ---")
    sft = results.get("full SFT", {})
    h1 = 0.0 < sft.get("kl_mean", 0.0) < 1.0
    print(f"  KL(SFT || base): mean {sft.get('kl_mean', 0):.4f} nats/token, "
          f"max {sft.get('kl_max', 0):.4f}")
    print(f"  H1 holds (positive but small): {h1}")
    print("  This is the quantity an RLHF KL penalty is written in. One epoch of")
    print("  SFT is worth this much divergence per token - the scale a penalty")
    print("  coefficient would have to be chosen against.")

    print("\n--- H2: does LoRA move the distribution less than full SFT? ---")
    lora = results.get("LoRA @2e-4", {})
    h2 = lora.get("kl_mean", 9e9) < sft.get("kl_mean", 0.0)
    print(f"  KL(LoRA  || base) = {lora.get('kl_mean', 0):.4f}")
    print(f"  KL(SFT   || base) = {sft.get('kl_mean', 0):.4f}")
    print(f"  H2 holds: {h2}")

    # -------------------------------------------------------------------- H3
    print("\n--- H3: policy == reference gives an implicit reward of EXACTLY 0 ---")
    sft_model = load_model("sft-qwen1p5b-noRobots-001", ckpt_root, device, dtype)
    zeros, checked = [], 0
    for c_ids, c_lab, _, _ in batches[:20]:
        with torch.no_grad():
            logits = sft_model(input_ids=c_ids).logits
        scores = sequence_logprobs(logits, c_lab)
        zeros.append(float(logprob_ratio(scores, scores)[0]))
        checked += 1
    h3 = all(z == 0.0 for z in zeros)
    print(f"  checked {checked} sequences; max |implicit reward| = "
          f"{max(abs(z) for z in zeros):.3e}")
    print(f"  H3 holds (exactly zero): {h3}")
    print("  This is the wiring check that costs nothing and catches a reference")
    print("  model plumbed to the wrong checkpoint - a bug that otherwise shows")
    print("  up as a DPO run that trains but optimises the wrong objective.")
    del sft_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --------------------------------------------------------------- H4 / H5
    print("\n--- H4/H5: the pre-DPO baseline ---")
    print(f"  {'model':<14} {'prefers chosen (SUM)':>22} {'prefers chosen (MEAN)':>23}")
    for label, data in results.items():
        print(f"  {label:<14} "
              f"{data['prefers_chosen_sum']}/{data['pairs']} "
              f"({100 * data['prefers_chosen_sum_fraction']:5.1f}%){'':>6} "
              f"{data['prefers_chosen_mean']}/{data['pairs']} "
              f"({100 * data['prefers_chosen_mean_fraction']:5.1f}%)")

    h4 = sft.get("prefers_chosen_sum_fraction", 0) > 0.5
    h5 = sft.get("prefers_chosen_mean_fraction", 0) > sft.get(
        "prefers_chosen_sum_fraction", 0
    )
    print(f"\n  H4 holds (SFT already prefers chosen >50% by SUM): {h4}")
    print(f"  H5 holds (MEAN favours chosen more than SUM does): {h5}")
    print("\n  WHY H5 IS THE INTERESTING ONE. The published DPO objective uses the")
    print("  SUM, and chosen responses are longer (E21: 56.5% longer in tokens).")
    print("  A longer sequence has a more negative summed log-probability simply")
    print("  for being longer, so the SUM comparison is biased AGAINST the")
    print("  chosen response. Length normalisation removes that bias. Phase 6")
    print("  must record which it uses, because the two measure different things.")

    print("\n--- WHAT THIS PROVES / DOES NOT PROVE ---")
    print("  PROVES     : how far SFT moved the policy in KL from the base, that")
    print("               the reference wiring gives an exactly zero implicit")
    print("               reward at initialisation, and the pre-DPO preference")
    print("               baseline Phase 6 must beat.")
    print("  DOES NOT   : show that DPO will improve any of these. No preference")
    print("               training was run. It also does not establish that KL")
    print("               from the BASE is the right reference for Phase 6 - DPO")
    print("               will use the SFT model as pi_ref, and KL-from-base is")
    print("               measured here to give the constraint a scale.")

    payload = {
        "eval_fingerprint": info["eval_fingerprint"],
        "pairs": len(batches),
        "results": results,
        "hypotheses": {"H1": bool(h1), "H2": bool(h2), "H3": bool(h3),
                       "H4": bool(h4), "H5": bool(h5)},
    }
    out = repo_root() / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n  evidence: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
