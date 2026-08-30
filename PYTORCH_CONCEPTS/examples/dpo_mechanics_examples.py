"""Executable examples for PYTORCH_CONCEPTS/pytorch-dpo-mechanics.md.

Every number quoted in that note comes from running this file:

    python PYTORCH_CONCEPTS/examples/dpo_mechanics_examples.py

Seeded, CPU, float64 where exactness is asserted.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from alignlab.dpo import dpo_loss, implicit_rewards
from alignlab.logprobs import SequenceScores


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def make(sums, tokens):
    total = torch.tensor(sums, dtype=torch.float64)
    n = torch.tensor(tokens)
    return SequenceScores(total, total / n, n)


def example_logsigmoid_stability() -> None:
    section("1. logsigmoid vs log(sigmoid(x)) - where the naive form dies")

    print(f"  {'x':>8} {'log(sigmoid(x))':>18} {'F.logsigmoid(x)':>18}")
    for x in (0.0, -10.0, -30.0, -80.0, -200.0, -1000.0):
        t = torch.tensor(x, dtype=torch.float32)
        naive = torch.log(torch.sigmoid(t))
        stable = F.logsigmoid(t)
        print(f"  {x:>8.1f} {float(naive):>18.6f} {float(stable):>18.6f}")

    print("\n  MEASURED, not assumed: float32 sigmoid survives x = -80 here")
    print("  (log(sigmoid(-80)) = -80.000000). It is at x = -200 that it")
    print("  underflows to 0, so log(0) = -inf and the gradient is lost.")
    print("  logsigmoid computes -softplus(-x) directly and stays exact all")
    print("  the way to -1000.")
    print("\n  (An earlier draft of this note claimed -80 underflows. Running it")
    print("   showed otherwise. The threshold was written from expectation.)")
    print("\n  WHY THIS IS THE CASE THAT MATTERS FOR DPO: the sigmoid argument is")
    print("  strongly negative exactly when the policy CONFIDENTLY PREFERS THE")
    print("  REJECTED response - the pair with the most to learn from. The naive")
    print("  form would return -inf there and produce a NaN gradient.")


def example_beta_scaling() -> None:
    section("2. what beta does to the loss and its gradient")

    # A fixed log-ratio difference of 4 nats.
    print("  fixed log-ratio difference = (-8 - -10) - (-12 - -10) = 4 nats")
    print(f"\n  {'beta':>7} {'logit':>9} {'loss':>10} {'dL/dlogit':>12}")
    for beta in (0.001, 0.01, 0.1, 0.5, 1.0, 2.0):
        logit = torch.tensor(beta * 4.0, dtype=torch.float64, requires_grad=True)
        loss = -F.logsigmoid(logit)
        loss.backward()
        print(f"  {beta:>7.3f} {float(logit):>9.4f} {float(loss):>10.6f} "
              f"{float(logit.grad):>12.6f}")

    print("\n  At beta=0.001 the logit is 0.004 and the loss is 0.691 - essentially")
    print("  log 2 = 0.693147, the value at initialisation. The gradient w.r.t.")
    print("  the logit is near -0.5, its maximum, but the logit itself barely")
    print("  moves because beta multiplies the parameter gradient too.")
    print("\n  At beta=2 the logit is 8 and the loss is 0.0003: saturated, almost")
    print("  no gradient. beta trades 'how far the policy may move' against")
    print("  'how much signal remains once it has moved'.")


def example_gradient_accumulation() -> None:
    section("3. gradient accumulation == a larger batch, exactly")

    torch.manual_seed(0)
    weights = torch.randn(6, dtype=torch.float64, requires_grad=True)

    def loss_for(index):
        return (weights[index] ** 2) * (index + 1)

    # one big batch
    weights.grad = None
    total = sum(loss_for(i) for i in range(6)) / 6
    total.backward()
    batched = weights.grad.clone()

    # accumulated, one at a time
    weights.grad = None
    for i in range(6):
        (loss_for(i) / 6).backward()
    accumulated = weights.grad.clone()

    print(f"  batched     : {[round(float(v), 8) for v in batched]}")
    print(f"  accumulated : {[round(float(v), 8) for v in accumulated]}")
    print(f"  identical   : {torch.equal(batched, accumulated)}")
    print("\n  This is why AlignLab's DPO loop can process ONE pair per forward")
    print("  and still take an effective-batch-16 step: divide each pair's loss")
    print("  by the accumulation count before backward, and the accumulated")
    print("  gradient equals the batched one exactly.")
    print("\n  The catch: it is only equal if EVERY microbatch is divided by the")
    print("  SAME count. A final partial window divided by the full count")
    print("  under-weights those examples - which is why our loop only steps on")
    print("  a complete window.")


def example_frozen_reference_memory() -> None:
    section("4. what freezing actually saves")

    n = 1_543_714_304  # Qwen2.5-1.5B
    bf16 = 2

    print(f"  model: {n:,} parameters\n")
    print(f"  {'component':<34} {'bytes/param':>12} {'GiB':>10}")
    rows = [
        ("policy weights (bf16)", bf16, n * bf16),
        ("policy gradients (bf16)", bf16, n * bf16),
        ("AdamW moments (2 x bf16)", 2 * bf16, n * 2 * bf16),
        ("REFERENCE weights (bf16)", bf16, n * bf16),
        ("reference gradients", 0, 0),
        ("reference optimizer state", 0, 0),
    ]
    for name, per, total in rows:
        print(f"  {name:<34} {per:>12} {total / 1024**3:>10.2f}")

    trainable_side = n * (bf16 + bf16 + 2 * bf16)
    reference_side = n * bf16
    print(f"\n  policy side  : {trainable_side / 1024**3:.2f} GiB")
    print(f"  reference    : {reference_side / 1024**3:.2f} GiB  (weights only)")
    print(f"  ratio        : {trainable_side / reference_side:.1f}x")
    print("\n  A frozen model costs its WEIGHTS and nothing else - no gradients,")
    print("  no optimizer state. That is why DPO's second model is far cheaper")
    print("  than doubling the memory, and why PPO's FOUR models are still")
    print("  expensive: two of them are trained.")
    print("\n  MEASURED on the real run: peak VRAM 18.56 GiB, against ~14.4 GiB")
    print("  for the Phase 3 full SFT of the same model - the reference adds")
    print("  roughly its weight footprint, as predicted, plus activations for")
    print("  the extra forward passes.")


def example_implicit_reward_is_zero() -> None:
    section("5. the check that costs nothing")

    torch.manual_seed(1)
    s = make([-291.1, -261.6, -180.4], [270, 240, 165])

    chosen, rejected = implicit_rewards(s, s, s, s, beta=0.1)
    loss, stats = dpo_loss(s, s, s, s, beta=0.1)

    print(f"  policy == reference")
    print(f"    implicit reward, chosen  : {[float(v) for v in chosen]}")
    print(f"    implicit reward, rejected: {[float(v) for v in rejected]}")
    print(f"    loss                     : {float(loss):.16f}")
    print(f"    log 2                    : {0.6931471805599453:.16f}")
    print(f"    exact match              : {float(loss) == 0.6931471805599453}")

    print("\n  Note this holds for ANY log-probabilities, however large. The")
    print("  cancellation is structural, not numerical - which is what makes it")
    print("  a reliable wiring check rather than a coincidence.")

    wrong = make([-291.2, -261.6, -180.4], [270, 240, 165])
    c2, _ = implicit_rewards(s, s, wrong, wrong, beta=0.1)
    print(f"\n  reference off by 0.1 nats on ONE sequence:")
    print(f"    implicit reward, chosen  : {[round(float(v), 6) for v in c2]}")
    print(f"    detected                 : {bool((c2.abs() > 0).any())}")


def main() -> None:
    print("DPO mechanics - executed examples")
    print(f"torch {torch.__version__}, cpu")
    example_logsigmoid_stability()
    example_beta_scaling()
    example_gradient_accumulation()
    example_frozen_reference_memory()
    example_implicit_reward_is_zero()
    print()


if __name__ == "__main__":
    main()
