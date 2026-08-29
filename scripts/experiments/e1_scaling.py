"""E1 - Why attention divides by sqrt(d_k).

OBJECTIVE
    Turn the variance argument for the 1/sqrt(d_k) factor into a measurement
    rather than an assertion.

HYPOTHESIS
    For q, k with independent unit-variance components, q·k is a sum of d_k
    terms each of variance 1, so Var(q·k) = d_k and the logits have standard
    deviation sqrt(d_k). Therefore, as d_k grows:
      * UNSCALED: logit spread grows like sqrt(d_k); softmax saturates towards
        one-hot; attention entropy collapses; the softmax Jacobian shrinks, so
        gradients through attention vanish.
      * SCALED by 1/sqrt(d_k): logit standard deviation stays ~1 regardless of
        d_k; entropy and gradients stay roughly flat.

CONFIGURATION
    B=1, H=1, T=16 queries/keys; d_k in {4, 16, 64, 256, 1024}
    inputs ~ N(0,1), float64 (so the effect measured is saturation, not
    float32 rounding); seed fixed below.

METRICS
    logit_std        empirical standard deviation of the scores
    max_prob         mean over queries of max_j softmax weight (1.0 == one-hot)
    entropy          mean Shannon entropy of the attention distribution, in
                     nats. ln(16) = 2.7726 is the uniform maximum.
    jacobian_mass    mean over queries of sum_j p_j(1 - p_j) = 1 - sum_j p_j^2.
                     This is the diagonal mass of the softmax Jacobian and is
                     the DIRECT measure of how responsive softmax still is. It
                     goes to 0 exactly when the distribution becomes one-hot.
    grad_norm        ||d(sum context)/d(query)||.

                     PREDICTION CORRECTION, PRESERVED: the original hypothesis
                     said grad_norm would shrink as d_k grew. It does NOT (see
                     the recorded results) - it rises and then falls. The
                     metric is CONFOUNDED: with scale=1.0 the backward pass
                     also carries the un-shrunk logit magnitudes, so growth in
                     the gradient's scale masks the loss of softmax
                     responsiveness. jacobian_mass was added afterwards as the
                     unconfounded measure. The original column is kept rather
                     than deleted, because the wrong choice of metric is the
                     instructive part.

Run:  python scripts/experiments/e1_scaling.py
"""

from __future__ import annotations

import math

import torch

from alignlab.models.attention import scaled_dot_product_attention
from alignlab.seeding import set_seed

SEED = 20260829
T = 16
D_K_VALUES = (4, 16, 64, 256, 1024)
DTYPE = torch.float64


def measure(d_k: int, scaled: bool) -> dict[str, float]:
    """One (d_k, scaling) cell of the experiment."""
    set_seed(SEED)

    q = torch.randn(1, 1, T, d_k, dtype=DTYPE, requires_grad=True)
    k = torch.randn(1, 1, T, d_k, dtype=DTYPE)
    v = torch.randn(1, 1, T, d_k, dtype=DTYPE)

    scale = None if scaled else 1.0  # None => 1/sqrt(d_k)
    context, weights = scaled_dot_product_attention(q, k, v, scale=scale)

    raw_scores = (q @ k.transpose(-2, -1)) * (
        (1.0 / math.sqrt(d_k)) if scaled else 1.0
    )

    context.sum().backward()

    probs = weights.detach()
    entropy = -(probs * torch.log(probs.clamp_min(1e-300))).sum(-1).mean()
    # sum_j p_j(1-p_j) = 1 - sum_j p_j^2 : softmax Jacobian diagonal mass.
    jacobian_mass = (1.0 - (probs**2).sum(dim=-1)).mean()

    return {
        "logit_std": float(raw_scores.detach().std()),
        "max_prob": float(probs.max(dim=-1).values.mean()),
        "entropy": float(entropy),
        "jacobian_mass": float(jacobian_mass),
        "grad_norm": float(q.grad.norm()),
    }


def main() -> None:
    print("=" * 78)
    print("E1 - the effect of the 1/sqrt(d_k) scaling factor on attention")
    print("=" * 78)
    print(f"torch {torch.__version__} | dtype {DTYPE} | T={T} | seed={SEED}")
    print(f"uniform-distribution entropy for T={T}: ln({T}) = {math.log(T):.4f} nats")

    for scaled in (False, True):
        label = "SCALED by 1/sqrt(d_k)" if scaled else "UNSCALED (scale = 1.0)"
        print(f"\n--- {label} ---")
        print(f"{'d_k':>6} {'logit_std':>12} {'max_prob':>10} "
              f"{'entropy':>10} {'jac_mass':>10} {'grad_norm':>12}")
        for d_k in D_K_VALUES:
            m = measure(d_k, scaled)
            print(
                f"{d_k:>6} {m['logit_std']:>12.4f} {m['max_prob']:>10.4f} "
                f"{m['entropy']:>10.4f} {m['jacobian_mass']:>10.4f} "
                f"{m['grad_norm']:>12.3e}"
            )

    print("\n" + "=" * 78)
    print("Read the UNSCALED block first: logit_std should track sqrt(d_k),")
    print("max_prob should approach 1.0, and entropy should collapse toward 0.")
    print("The SCALED block should stay roughly flat across every d_k.")
    print()
    print("jac_mass is the unconfounded saturation measure: it falls towards 0")
    print("as softmax becomes one-hot. grad_norm is CONFOUNDED - see the")
    print("prediction correction in this file's docstring.")
    print("=" * 78)


if __name__ == "__main__":
    main()
