"""Executable examples for PYTORCH_CONCEPTS/pytorch-loss-masking-and-shifts.md.

Every number quoted in that note comes from running this file. Run it to
reproduce them:

    python PYTORCH_CONCEPTS/examples/loss_masking_examples.py

Seeded so the outputs are stable across runs and machines (CPU, float32).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE = -100


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def example_ignore_index() -> None:
    section("1. ignore_index changes the DENOMINATOR, not just the numerator")

    torch.manual_seed(0)
    logits = torch.randn(1, 4, 10)
    targets_all = torch.tensor([[1, 2, 3, 4]])
    targets_masked = torch.tensor([[IGNORE, IGNORE, 3, 4]])

    loss_all = F.cross_entropy(
        logits.reshape(-1, 10), targets_all.reshape(-1), ignore_index=IGNORE
    )
    loss_masked = F.cross_entropy(
        logits.reshape(-1, 10), targets_masked.reshape(-1), ignore_index=IGNORE
    )

    print(f"  cross_entropy(all)    = {loss_all:.6f}   (mean over 4)")
    print(f"  cross_entropy(masked) = {loss_masked:.6f}   (mean over 2)")

    # The masked loss must equal the mean of the two surviving positions.
    per_position = F.cross_entropy(
        logits.reshape(-1, 10), targets_all.reshape(-1), reduction="none"
    )
    manual = per_position[2:].mean()
    print(f"  mean of positions 2,3 = {manual:.6f}")
    print(f"  masked == manual      : {torch.allclose(loss_masked, manual)}")

    sum_all = F.cross_entropy(
        logits.reshape(-1, 10), targets_all.reshape(-1),
        ignore_index=IGNORE, reduction="sum",
    )
    sum_masked = F.cross_entropy(
        logits.reshape(-1, 10), targets_masked.reshape(-1),
        ignore_index=IGNORE, reduction="sum",
    )
    print(f"\n  sum(all)    = {sum_all:.6f}  over 4 terms")
    print(f"  sum(masked) = {sum_masked:.6f}  over 2 terms")
    print("\n  The masked loss is NOT the unmasked loss minus something.")
    print("  It is a different average, over a different population.")


def example_shift() -> None:
    section("2. The shift drops position 0's label")

    cases = {
        "prompt masked (leading -100)": [IGNORE, IGNORE, 3, 4, 5],
        "trailing mask (padding-like)": [7, 8, 9, IGNORE, IGNORE],
    }

    for name, raw in cases.items():
        labels = torch.tensor([raw])
        labelled = int((labels != IGNORE).sum())
        contributing = int((labels[:, 1:] != IGNORE).sum())
        print(f"\n  {name}")
        print(f"    labels                     : {raw}")
        print(f"    labelled positions         : {labelled}")
        print(f"    loss-contributing positions: {contributing}")
        if labelled != contributing:
            print("    ^^ position 0 was labelled but is dropped by the shift")

    print("\n  Correct : (labels[:, 1:] != -100).sum()")
    print("  WRONG   : (labels      != -100).sum()")


def example_aggregation() -> None:
    section("3. Token-weighted vs example-weighted aggregation")

    # Two examples: 2 tokens at NLL 1.0, 8 tokens at NLL 3.0.
    nlls = [1.0, 3.0]
    counts = [2, 8]

    example_weighted = sum(nlls) / len(nlls)
    token_weighted = sum(n * c for n, c in zip(nlls, counts)) / sum(counts)

    print(f"  example 1: {counts[0]} tokens, NLL/token {nlls[0]}")
    print(f"  example 2: {counts[1]} tokens, NLL/token {nlls[1]}")
    print(f"\n  example-weighted : {example_weighted:.6f}")
    print(f"  token-weighted   : {token_weighted:.6f}")
    print(f"  they differ by   : {100 * abs(token_weighted - example_weighted) / example_weighted:.1f}%")
    print("\n  Neither is wrong. Reporting one while describing the other is.")


def example_decomposition() -> None:
    section("4. A masked loss decomposes by token-weighted regions")

    torch.manual_seed(1)
    vocab, length = 50, 12
    logits = torch.randn(1, length, vocab)
    ids = torch.randint(0, vocab, (1, length))
    boundary = 7

    def masked_ce(labels):
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        n = int((shift_labels != IGNORE).sum())
        loss = F.cross_entropy(
            shift_logits.reshape(-1, vocab),
            shift_labels.reshape(-1),
            ignore_index=IGNORE,
        )
        return float(loss), n

    completion = ids.clone()
    completion[:, :boundary] = IGNORE
    prompt = ids.clone()
    prompt[:, boundary:] = IGNORE

    l_completion, n_completion = masked_ce(completion)
    l_prompt, n_prompt = masked_ce(prompt)
    l_all, n_all = masked_ce(ids.clone())

    print(f"  prompt region     : {n_prompt} counted, CE {l_prompt:.6f}")
    print(f"  completion region : {n_completion} counted, CE {l_completion:.6f}")
    print(f"  everything        : {n_all} counted, CE {l_all:.6f}")

    reconstructed = (l_prompt * n_prompt + l_completion * n_completion) / (
        n_prompt + n_completion
    )
    print(f"\n  token-weighted reconstruction : {reconstructed:.9f}")
    print(f"  measured over everything      : {l_all:.9f}")
    print(f"  difference                    : {abs(reconstructed - l_all):.2e}")
    print(f"  counts add up ({n_prompt} + {n_completion} = {n_all}) : "
          f"{n_prompt + n_completion == n_all}")
    print("\n  This cross-check is what caught a mis-count in E12's first run.")


def main() -> None:
    print("PyTorch loss masking and shifts - executed examples")
    print(f"torch {torch.__version__}, device cpu, dtype float32")
    example_ignore_index()
    example_shift()
    example_aggregation()
    example_decomposition()
    print()


if __name__ == "__main__":
    main()
