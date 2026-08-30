"""Loss-mask inspection - which tokens actually carry gradient.

PROJECT_INSTRUCTIONS section 3 makes this the gate for Phase 3:

    "Loss masking must be explicitly verified before PEFT is layered on top.
     The project must be able to explain: Which tokens contribute to the
     training loss and why?"

A loss mask is the easiest thing in supervised fine-tuning to get wrong and
the hardest to notice. Nothing crashes. The loss curve looks fine - better,
in fact, because predicting the prompt is easier than predicting the answer,
so a broken mask makes the numbers LOOK BETTER while training the model to do
the wrong thing. Phase 2's E2 demonstrated the identical failure mode for
causal masking: removing the mask improved training loss 33x and destroyed
generation. This module exists so the same class of bug cannot hide here.

THE METHOD: TWO INDEPENDENT COMPUTATIONS.

    ours  - tokenize the prompt alone, count the tokens, and assert that
            everything before that index is ignored and everything after it
            is trained
    TRL's - whatever the real training pipeline actually produces

Agreement between two independent computations is evidence. Inspecting only
TRL's output would verify nothing: a mask can be self-consistent and wrong.

THE PREFIX ASSUMPTION, WHICH WE CHECK RATHER THAN ASSUME. TRL builds its mask
as ``[0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))``. That
is only correct if tokenizing the prompt alone yields a genuine PREFIX of
tokenizing prompt+completion. BPE merges across boundaries, so this is not
guaranteed in general - it holds here because the ChatML boundary is a special
token. ``check_prefix_consistency`` verifies it instead of trusting it; when
it fails, the reported boundary is off by a token and the model trains on one
token of prompt (or misses one token of answer) forever, silently.

IGNORE_INDEX is -100 because that is torch.nn.CrossEntropyLoss's default
``ignore_index``: positions holding it contribute no loss and no gradient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from alignlab.logging_utils import get_logger

logger = get_logger(__name__)

IGNORE_INDEX = -100


@dataclass
class MaskReport:
    """A full account of one example's loss mask."""

    n_tokens: int
    n_active: int
    n_ignored: int
    boundary: int | None
    active_positions: list[int]
    ignored_text: str
    active_text: str
    contiguous_active_suffix: bool
    prefix_consistent: bool | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def active_fraction(self) -> float:
        return self.n_active / self.n_tokens if self.n_tokens else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_tokens": self.n_tokens,
            "n_active": self.n_active,
            "n_ignored": self.n_ignored,
            "boundary": self.boundary,
            "active_fraction": round(self.active_fraction, 6),
            "contiguous_active_suffix": self.contiguous_active_suffix,
            "prefix_consistent": self.prefix_consistent,
            "notes": list(self.notes),
        }

    def __str__(self) -> str:
        return (
            f"{self.n_active}/{self.n_tokens} tokens active "
            f"({self.active_fraction:.1%}), boundary at {self.boundary}"
        )


def render_prompt(tokenizer, prompt_messages: Sequence[dict]) -> str:
    """Render the prompt with a generation prompt appended.

    ``add_generation_prompt=True`` appends the ``<|im_start|>assistant\\n``
    header that cues the model to answer. It belongs to the PROMPT, not the
    completion: the model must be conditioned on it, never asked to predict
    it. Getting this wrong shifts the boundary by three tokens and is the
    single most common chat-SFT bug.
    """
    return tokenizer.apply_chat_template(
        list(prompt_messages), tokenize=False, add_generation_prompt=True
    )


def render_full(
    tokenizer, prompt_messages: Sequence[dict], completion_messages: Sequence[dict]
) -> str:
    """Render prompt + completion as the single sequence that gets trained."""
    return tokenizer.apply_chat_template(
        [*prompt_messages, *completion_messages],
        tokenize=False,
        add_generation_prompt=False,
    )


def check_prefix_consistency(
    tokenizer, prompt_messages: Sequence[dict], completion_messages: Sequence[dict]
) -> tuple[bool, int, int]:
    """Verify prompt tokens are a true prefix of prompt+completion tokens.

    Returns ``(is_prefix, n_prompt_tokens, n_full_tokens)``.

    If this returns False, every boundary-based mask - ours and TRL's alike -
    is off, and the correct response is to stop and investigate the tokenizer,
    not to adjust an index until the numbers look right.
    """
    prompt_ids = tokenizer(render_prompt(tokenizer, prompt_messages), add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(
        render_full(tokenizer, prompt_messages, completion_messages),
        add_special_tokens=False,
    )["input_ids"]

    is_prefix = len(prompt_ids) <= len(full_ids) and full_ids[: len(prompt_ids)] == prompt_ids
    return is_prefix, len(prompt_ids), len(full_ids)


def expected_labels(
    tokenizer, prompt_messages: Sequence[dict], completion_messages: Sequence[dict]
) -> tuple[list[int], list[int], int]:
    """Independently compute what the labels SHOULD be.

    Returns ``(input_ids, labels, boundary)``. Everything strictly before
    ``boundary`` is IGNORE_INDEX; everything from it onward is the token id.

    This deliberately does not import TRL. It is the second opinion.
    """
    is_prefix, n_prompt, _ = check_prefix_consistency(
        tokenizer, prompt_messages, completion_messages
    )
    if not is_prefix:
        raise ValueError(
            "prompt tokenization is NOT a prefix of prompt+completion "
            "tokenization; a boundary-based loss mask cannot be correct here. "
            "Investigate the tokenizer and chat template before training."
        )

    full_ids = tokenizer(
        render_full(tokenizer, prompt_messages, completion_messages),
        add_special_tokens=False,
    )["input_ids"]

    labels = [IGNORE_INDEX] * n_prompt + list(full_ids[n_prompt:])
    return list(full_ids), labels, n_prompt


def describe_mask(
    tokenizer,
    input_ids: Sequence[int],
    labels: Sequence[int],
    prefix_consistent: bool | None = None,
) -> MaskReport:
    """Turn a (input_ids, labels) pair into a human-checkable report.

    Decodes the ignored and active regions separately, which is what turns
    "the mask has 47 active positions" into "the mask trains exactly the
    assistant's answer" - a claim a person can actually falsify by reading it.
    """
    if len(input_ids) != len(labels):
        raise ValueError(
            f"input_ids ({len(input_ids)}) and labels ({len(labels)}) differ in length"
        )

    active = [i for i, label in enumerate(labels) if label != IGNORE_INDEX]
    ignored = [i for i, label in enumerate(labels) if label == IGNORE_INDEX]
    notes: list[str] = []

    boundary = active[0] if active else None
    contiguous = bool(active) and active == list(range(active[0], len(labels)))

    if not active:
        notes.append("NO ACTIVE POSITIONS - this example contributes zero loss")
    if active and ignored and min(ignored) > min(active):
        notes.append("ignored positions appear AFTER active ones (padding, or a bug)")
    if not contiguous and active:
        notes.append("active positions are not a contiguous suffix")

    # Where labels are shifted or padded, a label may not equal its input id.
    mismatched = [i for i in active if labels[i] != input_ids[i]]
    if mismatched:
        notes.append(
            f"{len(mismatched)} active labels differ from their input id "
            f"(expected if labels are pre-shifted)"
        )

    return MaskReport(
        n_tokens=len(input_ids),
        n_active=len(active),
        n_ignored=len(ignored),
        boundary=boundary,
        active_positions=active,
        ignored_text=tokenizer.decode([input_ids[i] for i in ignored]),
        active_text=tokenizer.decode([input_ids[i] for i in active]),
        contiguous_active_suffix=contiguous,
        prefix_consistent=prefix_consistent,
        notes=notes,
    )


def compare_masks(ours: Sequence[int], theirs: Sequence[int]) -> dict[str, Any]:
    """Compare two label sequences position by position.

    Reports not just whether they agree but WHERE they disagree, because an
    off-by-one at the boundary and a wholesale failure need different fixes.
    """
    if len(ours) != len(theirs):
        return {
            "identical": False,
            "reason": "different lengths",
            "len_ours": len(ours),
            "len_theirs": len(theirs),
        }

    disagreements = [i for i, (a, b) in enumerate(zip(ours, theirs)) if a != b]
    ours_active = {i for i, v in enumerate(ours) if v != IGNORE_INDEX}
    theirs_active = {i for i, v in enumerate(theirs) if v != IGNORE_INDEX}

    return {
        "identical": not disagreements,
        "n_disagreements": len(disagreements),
        "first_disagreements": disagreements[:10],
        "active_only_in_ours": sorted(ours_active - theirs_active)[:10],
        "active_only_in_theirs": sorted(theirs_active - ours_active)[:10],
        "n_active_ours": len(ours_active),
        "n_active_theirs": len(theirs_active),
    }
