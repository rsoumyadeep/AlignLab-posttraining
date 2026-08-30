"""Executable examples for PYTORCH_CONCEPTS/pytorch-logprobs-and-kl.md.

Every number quoted in that note comes from running this file:

    python PYTORCH_CONCEPTS/examples/logprob_and_kl_examples.py

Seeded, CPU, float32.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from alignlab.logprobs import (
    approximate_kl,
    logprob_ratio,
    sequence_logprobs,
    token_kl,
    token_logprobs,
)
from alignlab.masking import IGNORE_INDEX
from alignlab.preference import bradley_terry_loss, bradley_terry_probability


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def example_gather_vs_cross_entropy() -> None:
    section("1. log-probability of a SPECIFIC token: gather on log_softmax")

    torch.manual_seed(0)
    logits = torch.randn(1, 4, 6)
    labels = torch.tensor([[1, 2, 3, 4]])

    picked, mask = token_logprobs(logits, labels)

    # cross_entropy returns the NEGATIVE log-probability of the target
    ce = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, 6), labels[:, 1:].reshape(-1), reduction="none"
    )
    print("  token_logprobs :", [round(float(v), 6) for v in picked[0]])
    print("  -cross_entropy :", [round(float(-v), 6) for v in ce])
    print("  identical      :", torch.allclose(picked[0], -ce, atol=1e-6))
    print("\n  cross_entropy IS negative log-probability of the target. They are")
    print("  the same quantity with opposite sign - worth knowing so you never")
    print("  compute one when you meant the other.")


def example_shift_and_counts() -> None:
    section("2. the shift, and the two different 'token counts'")

    torch.manual_seed(1)
    logits = torch.randn(1, 6, 8)

    for name, raw in (
        ("prompt-masked (leading ignore)", [IGNORE_INDEX, IGNORE_INDEX, 3, 4, 5, 6]),
        ("padding-masked (trailing)", [1, 2, 3, 4, IGNORE_INDEX, IGNORE_INDEX]),
    ):
        labels = torch.tensor([raw])
        scores = sequence_logprobs(logits, labels)
        labelled = int((labels != IGNORE_INDEX).sum())
        print(f"\n  {name}")
        print(f"    labels                : {raw}")
        print(f"    labelled positions    : {labelled}")
        print(f"    CONTRIBUTING positions: {int(scores.n_tokens[0])}")
        if labelled != int(scores.n_tokens[0]):
            print("    ^^ position 0 was labelled but is dropped by the shift")

    print("\n  Correct denominator: (labels[:, 1:] != -100).sum()")


def example_sum_vs_mean() -> None:
    section("3. SUM vs MEAN: why DPO's objective is length-sensitive")

    torch.manual_seed(2)
    vocab, length = 50, 21
    logits = torch.randn(2, length, vocab)

    labels = torch.full((2, length), IGNORE_INDEX)
    torch.manual_seed(3)
    labels[0, 1:6] = torch.randint(0, vocab, (5,))    # short completion
    labels[1, 1:21] = torch.randint(0, vocab, (20,))  # long completion

    scores = sequence_logprobs(logits, labels)
    print(f"  {'sequence':<12} {'tokens':>7} {'sum logp':>12} {'mean logp':>12}")
    for i, name in enumerate(("short", "long")):
        print(f"  {name:<12} {int(scores.n_tokens[i]):>7} "
              f"{float(scores.sum_logprob[i]):>12.4f} "
              f"{float(scores.mean_logprob[i]):>12.4f}")

    print("\n  The long sequence has a far more negative SUM purely because it")
    print("  has more tokens - the per-token means are comparable. On a random")
    print("  model the two means should be similar; the sums are not.")
    ratio = float(scores.sum_logprob[1] / scores.sum_logprob[0])
    print(f"  sum ratio long/short  : {ratio:.2f}x   (token ratio 4.00x)")


def example_kl_properties() -> None:
    section("4. KL: zero at equality, non-negative, NOT symmetric")

    torch.manual_seed(4)
    p = torch.randn(1, 1, 5)
    q = torch.randn(1, 1, 5)
    mask = torch.ones(1, 1, dtype=torch.bool)

    print(f"  KL(p||p) = {float(token_kl(p, p, mask)):.3e}")
    print(f"  KL(p||q) = {float(token_kl(p, q, mask)):.6f}")
    print(f"  KL(q||p) = {float(token_kl(q, p, mask)):.6f}")
    print(f"  symmetric: {torch.allclose(token_kl(p, q, mask), token_kl(q, p, mask))}")

    print("\n  Direction matters. KL(policy||reference) is large when the POLICY")
    print("  puts mass where the reference puts none - which is what reward")
    print("  hacking looks like in distribution space. That is why RLHF uses")
    print("  this direction and not the reverse.")

    print("\n  --- a concrete asymmetry ---")
    # reference is confident; policy is flat
    reference = torch.tensor([[[10.0, 0.0, 0.0, 0.0, 0.0]]])
    policy = torch.zeros(1, 1, 5)
    print(f"  reference confident, policy flat : KL(pol||ref) = "
          f"{float(token_kl(policy, reference, mask)):8.4f}")
    print(f"                                     KL(ref||pol) = "
          f"{float(token_kl(reference, policy, mask)):8.4f}")


def example_kl_estimators() -> None:
    section("5. the sampled KL estimator PPO actually uses")

    torch.manual_seed(5)
    # simulate: policy slightly different from reference, sampled tokens
    n = 200_000
    reference_lp = -torch.rand(n) * 3
    policy_lp = reference_lp + 0.1 * torch.randn(n)

    k1 = policy_lp - reference_lp
    k3 = approximate_kl(policy_lp, reference_lp)

    print(f"  samples          : {n:,}")
    print(f"  k1 mean          : {float(k1.mean()):+.6f}   var {float(k1.var()):.6f}")
    print(f"  k3 mean          : {float(k3.mean()):+.6f}   var {float(k3.var()):.6f}")
    print(f"  variance ratio k1/k3 : {float(k1.var() / k3.var()):,.0f}x")
    print(f"  k1 can be NEGATIVE: {bool((k1 < 0).any())}  (min {float(k1.min()):.4f})")

    negatives = int((k3 < 0).sum())
    print(f"\n  k3 negatives in float32 : {negatives} of {n:,}  "
          f"(min {float(k3.min()):.3e})")
    k3_double = approximate_kl(policy_lp.double(), reference_lp.double())
    print(f"  k3 negatives in float64 : {int((k3_double < 0).sum())} "
          f"(min {float(k3_double.min()):.3e})")
    print("\n  k3 is non-negative in EXACT arithmetic: f(x) = exp(-x) - 1 + x has")
    print("  f(0) = 0 and f'(x) = 1 - exp(-x), so x = 0 is its minimum. The few")
    print("  float32 negatives are catastrophic cancellation when |k1| is tiny -")
    print("  magnitude ~3e-08, and they vanish in float64. Worth knowing before")
    print("  asserting 'KL >= 0' with a strict comparison in a training loop.")
    print("\n  k1 is unbiased but high variance and genuinely negative for")
    print("  individual samples, which is awkward for a divergence. k3 is biased")
    print("  low, far lower variance, and non-negative up to float error - which")
    print("  is why PPO implementations use it.")


def example_bradley_terry() -> None:
    section("6. Bradley-Terry: only differences, and saturation")

    print(f"  {'r_w':>6} {'r_l':>6} {'P(w>l)':>10} {'loss':>10}")
    for rw, rl in ((0, 0), (1, 0), (2, 0), (5, 0), (101, 100), (0, 1)):
        p = bradley_terry_probability(float(rw), float(rl))
        loss = float(bradley_terry_loss(torch.tensor(float(rw)), torch.tensor(float(rl))))
        print(f"  {rw:>6} {rl:>6} {p:>10.6f} {loss:>10.6f}")

    print("\n  (1,0) and (101,100) are IDENTICAL - only the difference is")
    print("  identified, so preference data cannot pin down an absolute scale.")
    print("  At margin 5 the loss is already near zero: confident pairs stop")
    print("  contributing gradient.")


def example_implicit_reward() -> None:
    section("7. DPO's implicit reward is exactly 0 when policy == reference")

    torch.manual_seed(6)
    logits = torch.randn(3, 8, 20)
    labels = torch.randint(0, 20, (3, 8))
    labels[:, 0] = IGNORE_INDEX

    scores = sequence_logprobs(logits, labels)
    same = logprob_ratio(scores, scores)
    print(f"  policy == reference -> implicit reward: {[float(v) for v in same]}")
    print(f"  all exactly zero: {bool((same == 0).all())}")

    # now perturb the policy toward the labelled tokens
    policy_logits = logits.clone()
    for b in range(3):
        for t in range(7):
            if labels[b, t + 1] != IGNORE_INDEX:
                policy_logits[b, t, labels[b, t + 1]] += 3.0
    policy = sequence_logprobs(policy_logits, labels)
    moved = logprob_ratio(policy, scores)
    print(f"  policy favours these sequences -> {[round(float(v), 3) for v in moved]}")
    print(f"  all positive: {bool((moved > 0).all())}")
    print("\n  This is the check worth running before any DPO training: if the")
    print("  reference is wired to the wrong checkpoint, the first number is")
    print("  not zero, and the run optimises something other than intended.")


def main() -> None:
    print("log-probabilities and KL - executed examples")
    print(f"torch {torch.__version__}, cpu, float32")
    example_gather_vs_cross_entropy()
    example_shift_and_counts()
    example_sum_vs_mean()
    example_kl_properties()
    example_kl_estimators()
    example_bradley_terry()
    example_implicit_reward()
    print()


if __name__ == "__main__":
    main()
