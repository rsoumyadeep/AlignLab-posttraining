"""Preference data — the input to reward modelling and to DPO.

Phase 5 builds the infrastructure Phase 6 will train on. Nothing here trains
anything; it loads, validates, audits and fingerprints preference pairs.

THE SHAPE OF THE PROBLEM. A preference example is a prompt with two candidate
responses and a label saying which a human (or a model standing in for one)
preferred:

    prompt          x
    chosen          y_w   ("w" for win)
    rejected        y_l   ("l" for lose)

Both reward modelling and DPO consume exactly this. The Bradley-Terry model
says the probability that y_w beats y_l is

    P(y_w > y_l | x) = sigma( r(x, y_w) - r(x, y_l) )

so only the DIFFERENCE of scores is identified - which is why preference data
can never pin down an absolute reward scale, and why a reward model's raw
output is not meaningful on its own. See STUDY_WITH_CLAUDE/phase5.

WHY THE VALIDATION HERE IS NOT BOILERPLATE. Auditing 3,000 rows of
UltraFeedback before writing this module turned up four things that would each
have quietly damaged a DPO run:

  * 11.9% of pairs are score TIES - chosen and rejected were rated EQUALLY,
    yet the pair is presented as a preference. Bradley-Terry assumes a genuine
    ordering; a tie contributes a training signal that says "prefer y_w over
    y_l" when the annotator said no such thing.
  * 14 in 3,000 pairs have chosen text IDENTICAL to rejected text. For DPO the
    log-ratio of a pair with identical responses is exactly zero by
    construction, so the example asks the model to prefer a string over itself.
  * 2 in 3,000 responses are empty.
  * Chosen responses are LONGER than rejected in 55.8% of pairs (mean 1291 vs
    1132 characters). This is the well-known length bias in preference data,
    and it matters because a preference-trained model can learn "be longer"
    instead of "be better".

The first three are filtered (with the counts recorded). The fourth cannot be
filtered - it is a property of the data - so it is MEASURED and reported, so
that a later "DPO made responses longer" observation is interpretable rather
than surprising.

THE TOKENIZATION LESSON FROM PHASE 3 APPLIES HERE TOO. A completion beginning
with whitespace lets BPE merge across the prompt/completion boundary, so the
prompt's tokens stop being a prefix of the full sequence. DPO computes
log-probabilities over completion tokens only, exactly like SFT, so the same
boundary must hold. ``_strip_boundary_whitespace`` from ``alignlab.data`` is
reused rather than reimplemented.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

from alignlab.data import _strip_boundary_whitespace, fingerprint_dataset
from alignlab.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_DATASET = "HuggingFaceH4/ultrafeedback_binarized"
DEFAULT_REVISION = "main"

# VERIFIED against the live builder, not guessed from filenames. This dataset
# also exposes train_sft/test_sft/train_gen/test_gen; the PREFERENCE splits are
# the *_prefs ones. Phase 3 shipped a bug from guessing split names, so these
# were checked before being written down.
DEFAULT_TRAIN_SPLIT = "train_prefs"
DEFAULT_EVAL_SPLIT = "test_prefs"

EXPECTED_COLUMNS = {
    "prompt", "prompt_id", "chosen", "rejected", "score_chosen", "score_rejected",
}


@dataclass
class PreferenceAudit:
    """What the data actually looks like, measured rather than assumed."""

    rows: int
    ties: int
    inverted: int
    identical_responses: int
    empty_responses: int
    prompt_prefix_mismatches: int
    multi_turn: int
    margin_mean: float
    margin_median: float
    chosen_chars_mean: float
    rejected_chars_mean: float
    chosen_longer_fraction: float
    notes: list[str] = field(default_factory=list)

    @property
    def tie_fraction(self) -> float:
        return self.ties / self.rows if self.rows else 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tie_fraction"] = round(self.tie_fraction, 5)
        return data


def to_preference_triple(example: dict[str, Any]) -> dict[str, Any]:
    """UltraFeedback row -> TRL's conversational preference shape.

    Returns ``{"prompt": [...], "chosen": [...], "rejected": [...]}`` where
    prompt is the conversation up to (not including) the assistant turn, and
    chosen/rejected are single assistant turns.

    WHY SPLIT THE PROMPT OUT rather than passing the full conversations. TRL's
    DPOTrainer accepts either, but with an explicit prompt the boundary between
    scored and unscored tokens is a single index - the same property that made
    the SFT loss mask verifiable by hand in Phase 3. It also lets us CHECK that
    the prompt really is shared between the two responses, which is the
    assumption the whole comparison rests on: if chosen and rejected answered
    different questions, their log-probabilities would not be comparable.
    """
    chosen = example.get("chosen") or []
    rejected = example.get("rejected") or []
    if len(chosen) < 2 or len(rejected) < 2:
        return {"prompt": [], "chosen": [], "rejected": []}
    if chosen[-1].get("role") != "assistant" or rejected[-1].get("role") != "assistant":
        return {"prompt": [], "chosen": [], "rejected": []}

    prompt = list(chosen[:-1])

    chosen_turn = dict(chosen[-1])
    chosen_turn["content"] = _strip_boundary_whitespace(chosen_turn.get("content") or "")
    rejected_turn = dict(rejected[-1])
    rejected_turn["content"] = _strip_boundary_whitespace(
        rejected_turn.get("content") or ""
    )

    return {
        "prompt": prompt,
        "chosen": [chosen_turn],
        "rejected": [rejected_turn],
    }


def shares_prompt(example: dict[str, Any]) -> bool:
    """Do chosen and rejected answer the SAME prompt?

    Checked rather than assumed. A preference pair whose two responses address
    different prompts is not a preference at all, and their log-probabilities
    would not be comparable. Measured on UltraFeedback: 3000/3000 rows have
    ``prompt == chosen[0].content == rejected[0].content``.
    """
    chosen = example.get("chosen") or []
    rejected = example.get("rejected") or []
    if not chosen or not rejected:
        return False
    return [m.get("content") for m in chosen[:-1]] == [
        m.get("content") for m in rejected[:-1]
    ]


def is_wellformed_preference(example: dict[str, Any]) -> bool:
    """Whether a converted triple is usable for preference training.

    Rejects, with the reason each matters:

      * missing/empty prompt          - nothing to condition on
      * empty chosen or rejected      - DPO would score an empty sequence
      * IDENTICAL chosen and rejected - asks the model to prefer a string over
        itself; the log-ratio is zero by construction
    """
    prompt = example.get("prompt") or []
    chosen = example.get("chosen") or []
    rejected = example.get("rejected") or []

    if not prompt or len(chosen) != 1 or len(rejected) != 1:
        return False
    chosen_text = (chosen[0].get("content") or "").strip()
    rejected_text = (rejected[0].get("content") or "").strip()
    if not chosen_text or not rejected_text:
        return False
    if chosen_text == rejected_text:
        return False
    return all((m.get("content") or "").strip() for m in prompt)


def audit_preferences(dataset, sample: int | None = 3000) -> PreferenceAudit:
    """Measure the properties of a preference set that affect training.

    Sampled by default because the full UltraFeedback train split is 61,135
    rows and this reads every response's text.
    """
    n = len(dataset) if sample is None else min(sample, len(dataset))

    ties = inverted = identical = empty = mismatch = multi_turn = 0
    margins: list[float] = []
    chosen_chars: list[int] = []
    rejected_chars: list[int] = []
    chosen_longer = 0

    for i in range(n):
        row = dataset[i]
        chosen = row.get("chosen") or []
        rejected = row.get("rejected") or []

        score_c = row.get("score_chosen")
        score_r = row.get("score_rejected")
        if score_c is not None and score_r is not None:
            margins.append(float(score_c) - float(score_r))
            if score_c == score_r:
                ties += 1
            elif score_c < score_r:
                inverted += 1

        if len(chosen) > 2 or len(rejected) > 2:
            multi_turn += 1
        if not shares_prompt(row):
            mismatch += 1

        text_c = (chosen[-1].get("content") if chosen else "") or ""
        text_r = (rejected[-1].get("content") if rejected else "") or ""
        if not text_c.strip() or not text_r.strip():
            empty += 1
        if text_c.strip() and text_c.strip() == text_r.strip():
            identical += 1

        chosen_chars.append(len(text_c))
        rejected_chars.append(len(text_r))
        if len(text_c) > len(text_r):
            chosen_longer += 1

    notes: list[str] = []
    tie_fraction = ties / n if n else 0.0
    if tie_fraction > 0.05:
        notes.append(
            f"{100 * tie_fraction:.1f}% of pairs are score TIES - the annotator "
            f"rated both responses equally, yet the pair is used as a "
            f"preference. Bradley-Terry assumes a genuine ordering."
        )
    if identical:
        notes.append(
            f"{identical} pairs have IDENTICAL chosen and rejected text; their "
            f"log-ratio is zero by construction. Filtered."
        )
    if mismatch:
        notes.append(
            f"{mismatch} pairs do NOT share a prompt between chosen and "
            f"rejected - their log-probabilities are not comparable."
        )
    longer_fraction = chosen_longer / n if n else 0.0
    if longer_fraction > 0.52:
        notes.append(
            f"LENGTH BIAS: chosen is longer in {100 * longer_fraction:.1f}% of "
            f"pairs (mean {statistics.mean(chosen_chars):.0f} vs "
            f"{statistics.mean(rejected_chars):.0f} chars). A preference-trained "
            f"model can learn 'be longer' rather than 'be better'. This cannot "
            f"be filtered away - it is a property of the data."
        )

    audit = PreferenceAudit(
        rows=n,
        ties=ties,
        inverted=inverted,
        identical_responses=identical,
        empty_responses=empty,
        prompt_prefix_mismatches=mismatch,
        multi_turn=multi_turn,
        margin_mean=statistics.mean(margins) if margins else 0.0,
        margin_median=statistics.median(margins) if margins else 0.0,
        chosen_chars_mean=statistics.mean(chosen_chars) if chosen_chars else 0.0,
        rejected_chars_mean=statistics.mean(rejected_chars) if rejected_chars else 0.0,
        chosen_longer_fraction=longer_fraction,
        notes=notes,
    )
    for note in notes:
        logger.warning("preference audit: %s", note)
    return audit


def load_preference_dataset(
    name: str = DEFAULT_DATASET,
    revision: str = DEFAULT_REVISION,
    train_split: str = DEFAULT_TRAIN_SPLIT,
    eval_split: str = DEFAULT_EVAL_SPLIT,
    max_train: int | None = None,
    max_eval: int | None = 500,
    seed: int = 42,
    drop_ties: bool = False,
):
    """Load, convert, filter and fingerprint a preference dataset.

    Returns ``(train, eval, info)``.

    ``drop_ties`` defaults to **False**, deliberately. Ties are 11.9% of
    UltraFeedback and dropping them is a defensible choice - but it is a
    CHANGE TO THE DATA, so it must be an explicit decision recorded in the
    manifest rather than a silent default. The audit reports the tie rate
    either way.

    Subsampling is SHUFFLED with the seed, never head-truncated.
    """
    from datasets import load_dataset

    logger.info("loading %s@%s", name, revision)
    raw_train = load_dataset(name, split=train_split, revision=revision)
    raw_eval = load_dataset(name, split=eval_split, revision=revision)

    missing = EXPECTED_COLUMNS - set(raw_train.column_names)
    if missing:
        raise ValueError(
            f"{name} is missing expected columns {sorted(missing)}; found "
            f"{sorted(raw_train.column_names)}. Inspect the schema rather than "
            f"adapting blindly."
        )

    audits = {
        "train_raw": audit_preferences(raw_train).to_dict(),
        "eval_raw": audit_preferences(raw_eval).to_dict(),
    }

    def convert(dataset, split_name: str):
        before = len(dataset)
        if drop_ties:
            dataset = dataset.filter(
                lambda r: r["score_chosen"] != r["score_rejected"],
                desc=f"drop ties ({split_name})",
                load_from_cache_file=False,
            )
        after_ties = len(dataset)
        converted = dataset.map(
            to_preference_triple,
            remove_columns=dataset.column_names,
            desc=f"to preference triple ({split_name})",
            # Phase 3 lesson: datasets.map caches on disk and did NOT invalidate
            # when the mapping function changed, silently serving pre-fix rows.
            load_from_cache_file=False,
        )
        kept = converted.filter(
            is_wellformed_preference,
            desc=f"filter ({split_name})",
            load_from_cache_file=False,
        )
        logger.info(
            "%s: %d rows -> %d after tie-drop -> %d well-formed (dropped %d)",
            split_name, before, after_ties, len(kept), after_ties - len(kept),
        )
        return kept, before, after_ties

    train, train_before, train_after_ties = convert(raw_train, train_split)
    evaluation, eval_before, eval_after_ties = convert(raw_eval, eval_split)

    if max_train is not None and max_train < len(train):
        train = train.shuffle(seed=seed).select(range(max_train))
    if max_eval is not None and max_eval < len(evaluation):
        evaluation = evaluation.shuffle(seed=seed).select(range(max_eval))

    info = {
        "name": name,
        "revision": revision,
        "train_split": train_split,
        "eval_split": eval_split,
        "seed": seed,
        "drop_ties": drop_ties,
        "train_rows_raw": train_before,
        "train_rows_after_tie_drop": train_after_ties,
        "train_rows": len(train),
        "eval_rows_raw": eval_before,
        "eval_rows": len(evaluation),
        "audits": audits,
        "train_fingerprint": fingerprint_dataset(
            train, name, train_split, revision
        ).to_dict(),
        "eval_fingerprint": fingerprint_dataset(
            evaluation, name, eval_split, revision
        ).to_dict(),
    }
    logger.info(
        "preference train %d rows (%s), eval %d rows (%s)",
        len(train), info["train_fingerprint"]["sha256"][:16],
        len(evaluation), info["eval_fingerprint"]["sha256"][:16],
    )
    return train, evaluation, info


# ---------------------------------------------------------------- Bradley-Terry


def bradley_terry_probability(reward_chosen, reward_rejected):
    """P(chosen beats rejected) = sigmoid(r_w - r_l).

    THE MODEL BEHIND EVERYTHING IN PHASES 5 AND 6. A reward model is trained by
    maximising the likelihood of the observed preferences under this model, and
    DPO's loss is this same expression with the reward replaced by an implicit
    one defined through the policy and reference log-probabilities.

    Two properties worth seeing in code rather than only in prose:

      * only the DIFFERENCE matters. Adding a constant to every reward leaves
        every probability unchanged, so preference data cannot identify an
        absolute reward scale.
      * it saturates. A large margin gives a probability near 1 and a nearly
        flat gradient, so confident pairs stop contributing signal.

    Accepts floats or tensors.
    """
    try:
        import torch

        if isinstance(reward_chosen, torch.Tensor) or isinstance(
            reward_rejected, torch.Tensor
        ):
            return torch.sigmoid(reward_chosen - reward_rejected)
    except ImportError:  # pragma: no cover
        pass
    import math

    return 1.0 / (1.0 + math.exp(-(reward_chosen - reward_rejected)))


def bradley_terry_loss(reward_chosen, reward_rejected):
    """Negative log-likelihood of the observed preference: -log sigma(r_w - r_l).

    This is exactly the loss a reward model is trained with. Phase 6's DPO loss
    has the same shape, with ``beta * (log-ratio difference)`` in place of the
    reward difference - which is the single most useful thing to notice when
    asked how DPO relates to reward modelling.
    """
    import torch

    chosen = torch.as_tensor(reward_chosen)
    rejected = torch.as_tensor(reward_rejected)
    return -torch.nn.functional.logsigmoid(chosen - rejected)
