"""Direct Preference Optimization, from first principles.

PROJECT_INSTRUCTIONS for Phase 6: *"Implement DPO from first principles. The
core DPO computation must be independently understandable and tested. Do not
simply call a high-level TRL DPO trainer and consider the algorithm
implemented."*

This module is the algorithm. `alignlab.dpo_train` is the loop that uses it.
TRL is **not** used for the loss.

---------------------------------------------------------------- THE DERIVATION

RLHF maximises reward under a KL constraint against a frozen reference:

    max_π  E_{y~π}[ r(x,y) ]  −  β · KL( π(·|x) ‖ π_ref(·|x) )

This has a closed-form optimum (a standard result for KL-regularised reward
maximisation - the optimal policy is the reference tilted by the exponentiated
reward):

    π*(y|x) = (1 / Z(x)) · π_ref(y|x) · exp( r(x,y) / β )

Take logs and solve for the reward:

    r(x,y) = β · log[ π*(y|x) / π_ref(y|x) ]  +  β · log Z(x)

Now substitute into the Bradley-Terry preference model (implemented and tested
in `alignlab.preference`):

    P(y_w ≻ y_l | x) = σ( r(x,y_w) − r(x,y_l) )

The partition function `log Z(x)` depends ONLY on x. Both responses share the
same x, so it appears once with each sign and **cancels**:

    P(y_w ≻ y_l | x) = σ( β·[log π(y_w|x) − log π_ref(y_w|x)]
                        − β·[log π(y_l|x) − log π_ref(y_l|x)] )

Maximising the likelihood of the observed preferences gives the DPO loss:

    L = − log σ( β·[Δ_w − Δ_l] )       where Δ = log π_θ − log π_ref

**The cancellation is why both responses must share a prompt.** Phase 5's E21
verified that on 3000/3000 rows precisely because this step depends on it.

PROVENANCE, stated honestly. This derivation is written from standing
knowledge. The DPO paper's abstract was read (LEARNING_RESOURCES/phase5); the
**full paper was NOT**, so the derivation above is **not** cited to it. It is
verified for internal consistency by the tests in `tests/test_dpo.py`, which
check the properties the algebra implies.

------------------------------------------------------------- WHAT β CONTROLS

β multiplies the log-ratio difference INSIDE the sigmoid, so it sets how large
a gap is needed to saturate the loss:

    small β → sigmoid argument stays near 0, gradients persist, the policy can
              move far from π_ref. Weak effective KL constraint.
    large β → a small gap saturates, gradients vanish early, the policy stays
              near π_ref. Strong effective KL constraint.

------------------------------------------------------ SUM, NOT MEAN, AND WHY

The published objective uses the **SUM** of token log-probabilities over the
completion. `sum_logprob` is therefore the default here, and Phase 6 trains it.

That choice is not free, and Phase 5 measured its cost. UltraFeedback's chosen
responses are **56.5% longer in tokens** than its rejected ones, and a summed
log-probability is more negative for a longer sequence simply for being longer.
Measured consequence: the pre-DPO preference baseline is **47.2% under SUM**
and **58.3% under MEAN** - identical models, identical data, an 11-point swing
that straddles chance.

So "prefer the longer response" is a gradient direction this objective can
exploit. `length_normalise=True` computes the MEAN variant as a **diagnostic**;
it is never the primary metric, and using it silently would be substituting a
different objective under the same name.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from alignlab.logging_utils import get_logger
from alignlab.logprobs import SequenceScores

logger = get_logger(__name__)

# The loss at initialisation, when π == π_ref so both implicit rewards are 0:
#   L = −log σ(0) = −log(0.5) = log 2
LOSS_AT_INIT = 0.6931471805599453


@dataclass
class DPOBatchStats:
    """Everything worth logging from one DPO batch.

    Recorded per batch rather than derived later, because "the loss went down"
    is not by itself evidence that preferences were learned - the reward
    accuracy and margin are what say so.
    """

    loss: float
    chosen_reward: float
    rejected_reward: float
    reward_margin: float
    reward_accuracy: float
    policy_chosen_logp: float
    policy_rejected_logp: float
    reference_chosen_logp: float
    reference_rejected_logp: float
    chosen_tokens: float
    rejected_tokens: float

    def to_dict(self) -> dict:
        return asdict(self)


def implicit_rewards(
    policy_chosen: SequenceScores,
    policy_rejected: SequenceScores,
    reference_chosen: SequenceScores,
    reference_rejected: SequenceScores,
    beta: float,
    length_normalise: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DPO's implicit rewards: ``β · (log π_θ − log π_ref)`` for each response.

    There is no reward model. The "reward" is how much more likely the policy
    makes a response than the frozen reference does - which is the whole point
    of the reparameterisation.

    ``length_normalise=False`` (default) uses the SUM, matching the published
    objective. True uses the per-token MEAN and is a DIAGNOSTIC.
    """
    if length_normalise:
        chosen = beta * (policy_chosen.mean_logprob - reference_chosen.mean_logprob)
        rejected = beta * (
            policy_rejected.mean_logprob - reference_rejected.mean_logprob
        )
    else:
        chosen = beta * (policy_chosen.sum_logprob - reference_chosen.sum_logprob)
        rejected = beta * (policy_rejected.sum_logprob - reference_rejected.sum_logprob)
    return chosen, rejected


def dpo_loss(
    policy_chosen: SequenceScores,
    policy_rejected: SequenceScores,
    reference_chosen: SequenceScores,
    reference_rejected: SequenceScores,
    beta: float = 0.1,
    length_normalise: bool = False,
) -> tuple[torch.Tensor, DPOBatchStats]:
    """The DPO loss and its diagnostics.

        L = − log σ( r_w − r_l ),    r = β·(log π_θ − log π_ref)

    Uses ``logsigmoid`` rather than ``log(sigmoid(...))``: the latter
    underflows to ``-inf`` for a strongly negative argument, which happens
    exactly when the policy confidently prefers the REJECTED response - the
    case where the gradient matters most. ``F.logsigmoid`` is computed in a
    numerically stable form.

    Returns ``(loss, stats)`` with loss a scalar averaged over the batch.
    """
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")

    chosen_rewards, rejected_rewards = implicit_rewards(
        policy_chosen, policy_rejected, reference_chosen, reference_rejected,
        beta=beta, length_normalise=length_normalise,
    )

    logits = chosen_rewards - rejected_rewards
    losses = -F.logsigmoid(logits)

    stats = DPOBatchStats(
        loss=float(losses.mean()),
        chosen_reward=float(chosen_rewards.mean()),
        rejected_reward=float(rejected_rewards.mean()),
        reward_margin=float((chosen_rewards - rejected_rewards).mean()),
        # The fraction of pairs where the policy ALREADY ranks chosen above
        # rejected, in implicit-reward terms. This is what "learning the
        # preference" means, and it is not implied by the loss falling.
        reward_accuracy=float((logits > 0).float().mean()),
        policy_chosen_logp=float(policy_chosen.sum_logprob.mean()),
        policy_rejected_logp=float(policy_rejected.sum_logprob.mean()),
        reference_chosen_logp=float(reference_chosen.sum_logprob.mean()),
        reference_rejected_logp=float(reference_rejected.sum_logprob.mean()),
        chosen_tokens=float(policy_chosen.n_tokens.float().mean()),
        rejected_tokens=float(policy_rejected.n_tokens.float().mean()),
    )
    return losses.mean(), stats


def preference_accuracy(
    policy_chosen: SequenceScores,
    policy_rejected: SequenceScores,
    length_normalise: bool = False,
) -> torch.Tensor:
    """Does the POLICY ALONE assign higher log-probability to chosen?

    Deliberately NOT the same as `reward_accuracy`. That one compares implicit
    rewards, which involve the reference; this compares raw policy
    log-probabilities and is directly comparable to Phase 5's pre-DPO baseline
    (47.2% SUM / 58.3% MEAN).

    Keeping both is the point: a policy can improve its implicit-reward ranking
    while its raw log-probability ranking barely moves, and conflating them
    would hide that.
    """
    if length_normalise:
        return (policy_chosen.mean_logprob > policy_rejected.mean_logprob).float()
    return (policy_chosen.sum_logprob > policy_rejected.sum_logprob).float()


def verify_reference_is_frozen(reference_model) -> dict:
    """Assert the reference has no trainable parameters and is in eval mode.

    A reference model left trainable does not crash. It drifts toward the
    policy, the implicit rewards shrink toward zero, and the run optimises
    progressively less. PROJECT_INSTRUCTIONS requires this be tested
    explicitly, so it is checked at runtime as well as in unit tests.
    """
    trainable = [n for n, p in reference_model.named_parameters() if p.requires_grad]
    training_mode = reference_model.training
    report = {
        "n_trainable": len(trainable),
        "first_trainable": trainable[:5],
        "training_mode": training_mode,
        "frozen": not trainable and not training_mode,
    }
    if trainable:
        raise RuntimeError(
            f"reference model has {len(trainable)} trainable parameters "
            f"(e.g. {trainable[:3]}). It must be frozen: a drifting reference "
            f"silently weakens the objective instead of failing."
        )
    if training_mode:
        raise RuntimeError(
            "reference model is in train() mode; dropout would make its "
            "log-probabilities stochastic and the implicit rewards noisy."
        )
    return report


def verify_zero_reward_at_init(
    policy_chosen: SequenceScores,
    policy_rejected: SequenceScores,
    reference_chosen: SequenceScores,
    reference_rejected: SequenceScores,
    beta: float,
    tolerance: float = 0.0,
) -> dict:
    """When π == π_ref, both implicit rewards must be EXACTLY zero.

    Phase 5's E22 established this on the real checkpoint. It is re-verified
    here inside the DPO implementation itself, because the property that
    matters is not "the models are equal" but "this code path computes zero
    when they are" - a reference wired to the wrong checkpoint, or a
    sum/mean mismatch between the two branches, breaks the second while the
    first still holds.

    The loss at that point is exactly ``log 2``.
    """
    chosen, rejected = implicit_rewards(
        policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta
    )
    max_chosen = float(chosen.abs().max())
    max_rejected = float(rejected.abs().max())
    loss, _ = dpo_loss(
        policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta
    )
    return {
        "max_abs_chosen_reward": max_chosen,
        "max_abs_rejected_reward": max_rejected,
        "loss": float(loss),
        "expected_loss": LOSS_AT_INIT,
        "rewards_zero": max_chosen <= tolerance and max_rejected <= tolerance,
        "loss_matches_log2": abs(float(loss) - LOSS_AT_INIT) <= 1e-6,
    }
