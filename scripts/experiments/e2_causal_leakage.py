"""E2 - what happens when the causal mask is removed.

OBJECTIVE   Demonstrate that causal masking is REQUIRED for autoregressive
            language modelling, not a stylistic choice.

HYPOTHESIS  Training computes the loss at every position in parallel: position
            t predicts token t+1. WITHOUT the mask, position t can attend to
            position t+1 - the very token it is being asked to predict. The
            task collapses into copying, so training loss should fall FAR
            below the masked run, and far below the irreducible entropy of the
            data. That low loss is LEAKAGE, not learning.

            The decisive consequence: the leaked model should be much worse at
            actual generation, because at generation time the future does not
            exist and the shortcut it learned is unavailable.

CONFIG      tiny educational LM on synthetic grammar, seed 20260829,
            334k params, 600 steps, identical in both arms except the mask.

WARNING FOR THE READER: a sudden dramatic loss drop is evidence of leakage far
more often than of success. PROJECT_INSTRUCTIONS section 18 - stop and
investigate.

Run: python scripts/experiments/e2_causal_leakage.py
"""
from __future__ import annotations

import torch

from tiny_lm import (
    VOCAB_SIZE,
    decode,
    encode,
    generate_corpus,
    restore_causal_masking,
    train_tiny_lm,
)
from alignlab.models.generation import generate

SEED = 20260829


def sample_text(model, prompt: str, n: int = 60) -> str:
    ids = torch.tensor([encode(prompt)], dtype=torch.long)
    out = generate(model, ids, max_new_tokens=n, greedy=True, use_cache=True)
    return decode(out[0].tolist())


def main() -> None:
    print("=" * 78)
    print("E2 - causal masking: required, or decorative?")
    print("=" * 78)

    try:
        print("\n--- ARM 1: causal mask ENABLED (correct) ---")
        causal_model, causal = train_tiny_lm(seed=SEED, causal=True, verbose=True)

        print("\n--- ARM 2: causal mask DISABLED (leakage) ---")
        leaky_model, leaky = train_tiny_lm(seed=SEED, causal=False, verbose=True)
    finally:
        restore_causal_masking()

    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)
    print(f"{'':<26}{'masked':>12}{'unmasked':>12}")
    print("-" * 50)
    print(f"{'final train loss':<26}{causal.final_train_loss:>12.4f}"
          f"{leaky.final_train_loss:>12.4f}")
    print(f"{'final val loss':<26}{causal.final_val_loss:>12.4f}"
          f"{leaky.final_val_loss:>12.4f}")
    print(f"{'uniform-guess loss':<26}{causal.uniform_loss:>12.4f}"
          f"{leaky.uniform_loss:>12.4f}")
    ratio = causal.final_train_loss / max(leaky.final_train_loss, 1e-9)
    print(f"\nunmasked loss is {ratio:.1f}x LOWER than masked")

    print("\n--- but can either one actually GENERATE? ---")
    prompt = "the cat "
    print(f"prompt: {prompt!r}\n")
    print(f"  masked   : {sample_text(causal_model, prompt)!r}")
    print(f"  unmasked : {sample_text(leaky_model, prompt)!r}")

    print("\n" + "=" * 78)
    print("INTERPRETATION")
    print("The unmasked model reports a far lower training loss while being")
    print("useless at generation. It learned to READ THE ANSWER, a shortcut")
    print("that does not exist at inference time. Lower loss, no capability.")
    print("=" * 78)


if __name__ == "__main__":
    main()
