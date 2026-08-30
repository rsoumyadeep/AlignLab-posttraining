"""Sequence log-probabilities and KL — the arithmetic under PPO and DPO.

This is the infrastructure Phase 6 trains on, built and tested in Phase 5 where
it can be verified in isolation rather than debugged inside a training loop.

WHY THIS TINY MODULE MATTERS. Every quantity in preference optimisation is
built from one thing: the log-probability a model assigns to a completion given
a prompt.

    reward modelling   a scalar head on top of the same hidden states
    PPO                ratio = exp(log pi(y|x) - log pi_old(y|x)), and a KL
                       penalty against a frozen reference
    DPO                beta * [ (log pi(y_w|x) - log pi_ref(y_w|x))
                              - (log pi(y_l|x) - log pi_ref(y_l|x)) ]

Get this wrong and every downstream number is wrong in a way that still trains
and still produces a plausible loss curve — the same failure mode Phase 3's
loss mask had.

THREE THINGS THAT ARE EASY TO GET WRONG, all handled explicitly here.

1. THE SHIFT. Position t's logits predict token t+1. So the log-probability of
   a sequence uses ``logits[:, :-1]`` against ``ids[:, 1:]``. Phase 3's E12
   already produced a wrong number by counting labels before the shift; the
   same off-by-one here would silently mis-score every sequence.

2. SUM VERSUS MEAN. DPO's objective as published uses the SUM of token
   log-probabilities over the completion, not the mean. That makes it
   length-sensitive: a longer completion has a more negative log-probability
   simply for being longer. Given that UltraFeedback's chosen responses are
   longer than its rejected ones in 55.8% of pairs, this interacts directly
   with a known bias in the data. Both are provided, the default is ``sum``
   to match the published objective, and the choice is recorded rather than
   hidden.

3. WHICH TOKENS COUNT. Only completion tokens. Prompt tokens are shared
   between chosen and rejected, so including them adds the same constant to
   both — harmless in the DPO difference, but NOT harmless for a raw
   log-probability, for perplexity, or for KL. The mask is passed in
   explicitly rather than inferred.

NOTHING HERE TRAINS ANYTHING. Phase 5 does not run DPO.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from alignlab.logging_utils import get_logger
from alignlab.masking import IGNORE_INDEX

logger = get_logger(__name__)


@dataclass(frozen=True)
class SequenceScores:
    """Per-sequence log-probabilities and the token counts behind them."""

    sum_logprob: torch.Tensor  # [B]
    mean_logprob: torch.Tensor  # [B]
    n_tokens: torch.Tensor  # [B] - tokens that actually contributed

    def to_dict(self) -> dict:
        return {
            "sum_logprob": self.sum_logprob.tolist(),
            "mean_logprob": self.mean_logprob.tolist(),
            "n_tokens": self.n_tokens.tolist(),
        }


def token_logprobs(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token log-probabilities of the labelled tokens, after the shift.

    Args:
        logits: ``[B, T, V]`` — the model's raw outputs.
        labels: ``[B, T]`` — token ids, with ``IGNORE_INDEX`` where a position
            should not be scored (prompt tokens and padding).

    Returns:
        ``(logprobs, mask)``, both ``[B, T-1]``. ``logprobs`` holds
        ``log p(label_t | context)`` at each scored position and 0 elsewhere;
        ``mask`` is True where the position was scored.

    The shift is applied here, once, so no caller has to remember it.
    """
    if logits.dim() != 3:
        raise ValueError(f"logits must be [B, T, V], got {tuple(logits.shape)}")
    if labels.dim() != 2:
        raise ValueError(f"labels must be [B, T], got {tuple(labels.shape)}")
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            f"logits {tuple(logits.shape[:2])} and labels {tuple(labels.shape)} "
            f"disagree on batch or sequence length"
        )

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != IGNORE_INDEX

    # gather needs a valid index everywhere, including at masked positions,
    # so clamp the ignore value to 0 and zero those entries afterwards.
    safe_labels = shift_labels.masked_fill(~mask, 0)

    logprobs = torch.log_softmax(shift_logits.float(), dim=-1)
    picked = torch.gather(logprobs, dim=2, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    return picked * mask, mask


def sequence_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> SequenceScores:
    """Sum and mean completion log-probability per sequence.

    ``mean_logprob`` divides by the number of CONTRIBUTING tokens, which is the
    count after the shift — not the number of labelled positions. Those differ
    whenever position 0 is labelled, and confusing them is exactly the mistake
    Phase 3's E12 made.
    """
    picked, mask = token_logprobs(logits, labels)
    n_tokens = mask.sum(dim=-1)
    total = picked.sum(dim=-1)
    mean = total / n_tokens.clamp(min=1)
    return SequenceScores(sum_logprob=total, mean_logprob=mean, n_tokens=n_tokens)


@torch.no_grad()
def score_sequences(
    model, input_ids: torch.Tensor, labels: torch.Tensor
) -> SequenceScores:
    """Run a model and score the labelled completion tokens.

    ``no_grad`` because Phase 5 only measures. A training use (Phase 6) needs
    gradients through the policy and must call the pieces directly.
    """
    attention_mask = torch.ones_like(input_ids)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    return sequence_logprobs(logits, labels)


def logprob_ratio(
    policy: SequenceScores, reference: SequenceScores, length_normalise: bool = False
) -> torch.Tensor:
    """``log pi(y|x) - log pi_ref(y|x)`` — the quantity DPO is built from.

    This single difference is what DPO calls an *implicit reward*: it says how
    much more (or less) likely the policy makes this completion than the frozen
    reference does. No reward model is involved.

    ``length_normalise=False`` matches the published objective. Setting it True
    uses mean-per-token instead, which removes the length sensitivity discussed
    in the module docstring — a deviation from the paper, so it is opt-in.
    """
    if length_normalise:
        return policy.mean_logprob - reference.mean_logprob
    return policy.sum_logprob - reference.sum_logprob


def token_kl(
    policy_logits: torch.Tensor, reference_logits: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Exact per-token KL( policy || reference ) over the full vocabulary.

    KL(p||q) = sum_v p(v) [ log p(v) - log q(v) ], computed over every
    vocabulary entry rather than estimated from samples.

    THE DIRECTION MATTERS AND IS NOT SYMMETRIC. KL(policy||reference) — the
    direction used here and in RLHF's penalty — is large when the policy puts
    mass where the reference puts none. That is precisely the failure the
    penalty exists to prevent: a policy drifting into text the pretrained model
    considers implausible, which is what reward hacking looks like in
    distribution space.

    Args:
        policy_logits / reference_logits: ``[B, T, V]``
        mask: ``[B, T]`` True where the position should count.
    """
    if policy_logits.shape != reference_logits.shape:
        raise ValueError(
            f"policy {tuple(policy_logits.shape)} and reference "
            f"{tuple(reference_logits.shape)} logits differ in shape"
        )
    policy_logprobs = torch.log_softmax(policy_logits.float(), dim=-1)
    reference_logprobs = torch.log_softmax(reference_logits.float(), dim=-1)
    kl = (policy_logprobs.exp() * (policy_logprobs - reference_logprobs)).sum(dim=-1)
    return kl * mask


def sequence_kl(
    policy_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean per-token KL over the scored positions of each sequence.

    Returns ``(kl_per_sequence, n_tokens)``, both ``[B]``.

    The shift is applied so the KL is measured at exactly the positions whose
    log-probabilities ``sequence_logprobs`` would score — otherwise the two
    numbers describe different token sets and cannot be discussed together.
    """
    shift_labels = labels[:, 1:]
    mask = shift_labels != IGNORE_INDEX
    kl = token_kl(policy_logits[:, :-1, :], reference_logits[:, :-1, :], mask)
    n_tokens = mask.sum(dim=-1)
    return kl.sum(dim=-1) / n_tokens.clamp(min=1), n_tokens


def approximate_kl(
    policy_logprobs: torch.Tensor, reference_logprobs: torch.Tensor
) -> torch.Tensor:
    """The sampled KL estimator PPO implementations actually use.

    Exact KL needs the full vocabulary distribution at every position, which is
    ``[B, T, 151936]`` for this model — too large to hold for a whole PPO batch.
    So practical implementations estimate it from the sampled tokens only:

        k1 = log pi(y) - log pi_ref(y)          unbiased, high variance
        k3 = exp(-k1) - 1 + k1                  biased low, much lower variance,
                                                and always non-negative

    This returns k3, which is what TRL and most PPO code use. It is included
    here because "PPO penalises KL" and "PPO penalises an estimator of KL whose
    properties differ from KL" are different statements, and the second is the
    true one.

    NOT USED by any experiment in Phase 5 — provided so the conceptual material
    has a concrete referent.
    """
    diff = policy_logprobs - reference_logprobs
    return torch.exp(-diff) - 1 + diff
