"""E21 - is the preference data actually ready for DPO?

Phase 5's job is to prepare Phase 6, and "prepare" means verified, not
downloaded. This script checks the readiness list the phase requires:
preference schema, chosen/rejected structure, tokenizer behaviour, sequence
handling, fingerprinting, and reproducibility.

WHY EACH CHECK EARNS ITS PLACE — every one corresponds to a bug this project
has already been bitten by, or to an assumption DPO's arithmetic rests on.

  schema        Phase 3 shipped a bug from guessing split names off filenames.
                The splits here are train_prefs/test_prefs, NOT train/test, and
                that was verified against the live builder.

  prompt shared DPO compares log pi(y_w|x) against log pi(y_l|x) for the SAME
                x. If the two responses answered different prompts the
                comparison is meaningless. Checked, not assumed.

  token prefix  DPO scores completion tokens only, so the prompt's tokens must
                be a genuine PREFIX of both full sequences. Phase 3 found 1.5%
                of no_robots rows violating this because BPE merged the
                template's trailing newline into a completion beginning with
                whitespace. The same failure mode applies to BOTH responses
                here, so it is measured on both.

  lengths       DPO's published objective sums token log-probabilities, so it
                is length-sensitive - and the audit already measured chosen
                responses to be longer than rejected in 55.8% of pairs. This
                quantifies the same bias in TOKENS rather than characters, and
                measures how many pairs would be truncated at a given
                max_length, since truncation removes exactly the completion
                tokens DPO scores.

  fingerprint   the same content hash Phases 3 and 4 used, so a Phase 6 result
                can be traced to exact rows.

HYPOTHESES, recorded before running:

  H1  every pair shares its prompt between chosen and rejected
  H2  after whitespace stripping, the prompt is a token-prefix of BOTH the
      chosen and the rejected sequence, for every sampled pair
  H3  chosen completions are longer than rejected in TOKENS too, at roughly
      the 55.8% seen in characters
  H4  the filtered dataset is deterministic: two loads with the same seed
      produce the same fingerprint

WHAT THIS DOES NOT DO. It does not train, and it does not run DPO. No model is
fine-tuned in Phase 5.

Run (server):
    python scripts/experiments/e21_preference_data_readiness.py
"""

from __future__ import annotations

import argparse
import json

from transformers import AutoTokenizer

from alignlab.masking import check_prefix_consistency, render_full, render_prompt
from alignlab.paths import configure_hf_cache, repo_root
from alignlab.preference import audit_preferences, load_preference_dataset

BASE_MODEL = "Qwen/Qwen2.5-1.5B"
REVISION = "8faed761d45a263340a0528343f099c05c9a4323"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-train", type=int, default=2000)
    parser.add_argument("--max-eval", type=int, default=500)
    parser.add_argument("--token-sample", type=int, default=400)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--out", default="docs/phase5/e21_preference_readiness.json")
    args = parser.parse_args()

    configure_hf_cache()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=REVISION)

    print("=" * 80)
    print("E21 - preference data readiness for DPO")
    print("=" * 80)
    print(f"  tokenizer {BASE_MODEL} @ {REVISION[:12]}")

    train, evaluation, info = load_preference_dataset(
        max_train=args.max_train, max_eval=args.max_eval
    )

    print(f"\n--- dataset ---")
    print(f"  {info['name']}  splits {info['train_split']} / {info['eval_split']}")
    print(f"  raw train rows        : {info['train_rows_raw']:,}")
    print(f"  after filtering       : {len(train):,} (subsampled to {args.max_train})")
    print(f"  eval rows             : {len(evaluation):,}")
    print(f"  drop_ties             : {info['drop_ties']}  (explicit, not defaulted)")
    print(f"  train fingerprint     : {info['train_fingerprint']['sha256'][:32]}")
    print(f"  eval  fingerprint     : {info['eval_fingerprint']['sha256'][:32]}")

    raw_audit = info["audits"]["train_raw"]
    print(f"\n--- audit of the RAW train split (sampled {raw_audit['rows']}) ---")
    print(f"  score ties            : {raw_audit['ties']} "
          f"({100 * raw_audit['tie_fraction']:.1f}%)")
    print(f"  inverted pairs        : {raw_audit['inverted']}")
    print(f"  identical responses   : {raw_audit['identical_responses']}")
    print(f"  empty responses       : {raw_audit['empty_responses']}")
    print(f"  prompt mismatches     : {raw_audit['prompt_prefix_mismatches']}")
    print(f"  multi-turn            : {raw_audit['multi_turn']}")
    print(f"  score margin mean/med : {raw_audit['margin_mean']:.3f} / "
          f"{raw_audit['margin_median']:.3f}")
    print(f"  chosen/rejected chars : {raw_audit['chosen_chars_mean']:.0f} / "
          f"{raw_audit['rejected_chars_mean']:.0f}")
    print(f"  chosen longer         : {100 * raw_audit['chosen_longer_fraction']:.1f}%")
    for note in raw_audit["notes"]:
        print(f"  NOTE: {note}")

    # ------------------------------------------------ H1: shared prompt
    print("\n--- H1: chosen and rejected share the prompt ---")
    post_audit = audit_preferences(train, sample=None)
    h1 = post_audit.prompt_prefix_mismatches == 0
    print(f"  mismatches after filtering: {post_audit.prompt_prefix_mismatches} "
          f"of {post_audit.rows}")
    print(f"  identical responses       : {post_audit.identical_responses} "
          f"(filtered, so expect 0)")
    print(f"  H1 holds: {h1}")

    # -------------------------------- H2: token-prefix, for BOTH responses
    print("\n--- H2: prompt tokens are a PREFIX of both sequences ---")
    n = min(args.token_sample, len(train))
    bad_chosen, bad_rejected = [], []
    chosen_tokens, rejected_tokens, prompt_tokens = [], [], []

    for i in range(n):
        row = train[i]
        ok_c, n_prompt, n_full_c = check_prefix_consistency(
            tokenizer, row["prompt"], row["chosen"]
        )
        ok_r, _, n_full_r = check_prefix_consistency(
            tokenizer, row["prompt"], row["rejected"]
        )
        if not ok_c:
            bad_chosen.append(i)
        if not ok_r:
            bad_rejected.append(i)
        prompt_tokens.append(n_prompt)
        chosen_tokens.append(n_full_c - n_prompt)
        rejected_tokens.append(n_full_r - n_prompt)

    h2 = not bad_chosen and not bad_rejected
    print(f"  sampled                     : {n}")
    print(f"  chosen  prefix violations   : {len(bad_chosen)} {bad_chosen[:5]}")
    print(f"  rejected prefix violations  : {len(bad_rejected)} {bad_rejected[:5]}")
    print(f"  H2 holds: {h2}")
    if not h2:
        print("  A boundary-based completion mask would be wrong by a token on")
        print("  those rows - the same failure Phase 3 found in no_robots.")

    # ---------------------------------------------- H3: length bias in tokens
    print("\n--- H3: length bias, in TOKENS ---")
    chosen_longer = sum(1 for a, b in zip(chosen_tokens, rejected_tokens) if a > b)
    mean_c = sum(chosen_tokens) / n
    mean_r = sum(rejected_tokens) / n
    fraction = chosen_longer / n
    h3 = fraction > 0.5
    print(f"  prompt tokens   mean {sum(prompt_tokens) / n:8.1f}")
    print(f"  chosen  tokens  mean {mean_c:8.1f}  median "
          f"{sorted(chosen_tokens)[n // 2]:6d}")
    print(f"  rejected tokens mean {mean_r:8.1f}  median "
          f"{sorted(rejected_tokens)[n // 2]:6d}")
    print(f"  chosen longer in tokens : {chosen_longer}/{n} "
          f"({100 * fraction:.1f}%)")
    print(f"  H3 holds (>50%): {h3}")
    print("\n  WHY THIS MATTERS FOR DPO. The published objective SUMS token")
    print("  log-probabilities, so a longer completion is more negative simply")
    print("  for being longer. With chosen systematically longer, part of what")
    print("  DPO can learn here is 'be longer'. alignlab.logprobs offers a")
    print("  length-normalised ratio as an opt-in, and Phase 6 must record")
    print("  which it used.")

    # ------------------------------------------- truncation at max_length
    over = sum(
        1 for p, c, r in zip(prompt_tokens, chosen_tokens, rejected_tokens)
        if p + max(c, r) > args.max_length
    )
    print(f"\n--- sequence handling at max_length={args.max_length} ---")
    print(f"  pairs whose longer sequence exceeds the limit: {over}/{n} "
          f"({100 * over / n:.1f}%)")
    print("  Truncation removes tokens from the TAIL, which for a prompt/")
    print("  completion pair is exactly the completion DPO scores. A pair")
    print("  truncated asymmetrically compares two different amounts of text.")

    # ------------------------------------------------- H4: reproducibility
    print("\n--- H4: reproducibility ---")
    train2, eval2, info2 = load_preference_dataset(
        max_train=args.max_train, max_eval=args.max_eval
    )
    same_train = (
        info["train_fingerprint"]["sha256"] == info2["train_fingerprint"]["sha256"]
    )
    same_eval = info["eval_fingerprint"]["sha256"] == info2["eval_fingerprint"]["sha256"]
    h4 = same_train and same_eval
    print(f"  train fingerprint stable : {same_train}")
    print(f"  eval  fingerprint stable : {same_eval}")
    print(f"  H4 holds: {h4}")

    # ----------------------------------------------------------- one example
    print("\n--- one fully rendered pair (what DPO will actually score) ---")
    row = train[0]
    prompt_text = render_prompt(tokenizer, row["prompt"])
    print(f"  PROMPT ({len(tokenizer(prompt_text, add_special_tokens=False)['input_ids'])} tokens):")
    print("    " + prompt_text[:300].replace("\n", "\\n"))
    for key in ("chosen", "rejected"):
        full = render_full(tokenizer, row["prompt"], row[key])
        completion = full[len(prompt_text):]
        n_tok = len(tokenizer(full, add_special_tokens=False)["input_ids"])
        print(f"\n  {key.upper()} (total {n_tok} tokens), completion begins:")
        print("    " + completion[:220].replace("\n", "\\n"))

    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    checks = {
        "H1 chosen/rejected share the prompt": h1,
        "H2 prompt is a token-prefix of both": h2,
        "H3 length bias present in tokens   ": h3,
        "H4 loading is reproducible         ": h4,
    }
    for label, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    ready = h1 and h2 and h4  # H3 is a property to KNOW, not a gate
    print(f"\n  Preference data READY for Phase 6 DPO: {ready}")
    print("  (H3 is a measured property of the data, not a gate - the length")
    print("   bias is real and must be carried into Phase 6's interpretation.)")

    payload = {
        "dataset": info,
        "post_filter_audit": post_audit.to_dict(),
        "token_stats": {
            "sampled": n,
            "prompt_tokens_mean": sum(prompt_tokens) / n,
            "chosen_tokens_mean": mean_c,
            "rejected_tokens_mean": mean_r,
            "chosen_longer_fraction_tokens": fraction,
            "over_max_length": over,
            "max_length": args.max_length,
        },
        "prefix_violations": {"chosen": bad_chosen, "rejected": bad_rejected},
        "hypotheses": {k.strip(): bool(v) for k, v in checks.items()},
        "ready": bool(ready),
    }
    out = repo_root() / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n  evidence: {args.out}")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
