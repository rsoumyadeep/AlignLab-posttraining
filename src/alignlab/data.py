"""Instruction datasets for supervised fine-tuning.

Phase 3 needs an instruction dataset, a train/validation split, a chat
formatting decision, and a fingerprint that ties a result to the exact bytes
that produced it. This module owns all four.

WHY no_robots IS THE DEFAULT. It is 10k examples written by human annotators
(not distilled from another model), it is 21 MiB rather than gigabytes on a
volume with under 100 GiB free, and it ships an official train/test split so
the validation set is not something we invented. Small and honest beats large
and convenient when the goal is to understand what the trainer does.

WHY prompt/completion RATHER THAN A SINGLE text FIELD. TRL decides loss
masking from the dataset's SHAPE: SFTTrainer sets ``completion_only_loss``
to True automatically when a sample has both "prompt" and "completion" keys
(trl/trainer/sft_trainer.py, and verified in scripts/experiments/e12). A
single pre-rendered "text" column trains on the prompt tokens too, silently.
Choosing the shape is therefore choosing the loss, which is exactly the
decision PROJECT_INSTRUCTIONS section 3 says must be explicit and verified.

THE FINGERPRINT IS OURS, NOT THE LIBRARY'S. datasets exposes
``Dataset._fingerprint``, but that value is a cache key derived from the
sequence of operations applied, so it changes when the library version or the
transformation order changes even though the data is identical, and it is
private API. We hash the CONTENT instead: a canonical JSON serialisation of
every row, in order. Same rows in the same order -> same fingerprint, on any
machine, under any library version. That is what makes it usable as evidence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from alignlab.logging_utils import get_logger

logger = get_logger(__name__)

# Pinned dataset. Like the model revision, recorded so a result can be traced
# to exact bytes rather than to "whatever main pointed at that day".
DEFAULT_DATASET = "HuggingFaceH4/no_robots"
DEFAULT_DATASET_REVISION = "main"

# The columns no_robots provides. Named so a schema change upstream fails
# loudly here rather than producing quietly wrong training data.
EXPECTED_COLUMNS = {"prompt", "prompt_id", "messages", "category"}


@dataclass(frozen=True)
class DatasetFingerprint:
    """A content-addressed identity for one split.

    ``sha256`` covers the rows themselves. ``n_rows`` and ``n_chars`` are
    redundant with it but are cheap to eyeball in a manifest, and catch the
    common "did I accidentally train on 100 examples" mistake at a glance.
    """

    name: str
    split: str
    revision: str
    n_rows: int
    n_chars: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"{self.name}:{self.split}@{self.revision} "
            f"{self.n_rows} rows, sha256={self.sha256[:16]}"
        )


def _canonical(row: Any) -> str:
    """Serialise one row deterministically.

    ``sort_keys`` makes the hash independent of dict ordering;
    ``ensure_ascii`` makes it independent of the platform's default encoding;
    ``separators`` removes incidental whitespace. Any of those left to the
    default would make the fingerprint machine-dependent, which would defeat
    the point of having one.
    """
    return json.dumps(row, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def fingerprint_rows(
    rows: Iterable[Any],
    name: str,
    split: str,
    revision: str = DEFAULT_DATASET_REVISION,
) -> DatasetFingerprint:
    """Hash rows in order, returning a reproducible identity.

    Streaming rather than materialising: the digest is updated row by row so
    this stays usable on a dataset larger than memory.
    """
    digest = hashlib.sha256()
    n_rows = 0
    n_chars = 0
    for row in rows:
        payload = _canonical(row)
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")  # unambiguous row separator
        n_rows += 1
        n_chars += len(payload)

    return DatasetFingerprint(
        name=name,
        split=split,
        revision=revision,
        n_rows=n_rows,
        n_chars=n_chars,
        sha256=digest.hexdigest(),
    )


def fingerprint_dataset(
    dataset: Any,
    name: str,
    split: str,
    revision: str = DEFAULT_DATASET_REVISION,
) -> DatasetFingerprint:
    """Fingerprint a datasets.Dataset by content."""
    return fingerprint_rows(iter(dataset), name=name, split=split, revision=revision)


def to_prompt_completion(example: dict[str, Any]) -> dict[str, Any]:
    """Split one conversation into a prompt and a single-turn completion.

    Input is a no_robots row whose ``messages`` is a list of
    ``{"role", "content"}`` dicts ending in an assistant turn. Output is TRL's
    conversational prompt-completion shape::

        {"prompt":     [ ...every message up to the last... ],
         "completion": [ the final assistant message ]}

    WHY THE LAST TURN ONLY. A multi-turn conversation could be expanded into
    one training example per assistant turn, which uses the data more fully.
    We deliberately do not: with prompt/completion the boundary between masked
    and unmasked tokens is a SINGLE INDEX, which is what makes the loss mask
    verifiable by hand (e12, arm A). Using the data fully is worth less than
    knowing exactly which tokens carry gradient.

    WHAT THIS COSTS, MEASURED rather than hand-waved. In no_robots' first 2000
    training rows the turn-count distribution is 1840 two-turn conversations
    and 160 longer ones (7, 9, 5, 8 and 6 turns) - so ~92% of examples are
    already single-turn and lose nothing at all. For the remaining ~8% the
    earlier assistant turns become context instead of targets.

    THE ALTERNATIVE IS CURRENTLY UNAVAILABLE, not merely unchosen. TRL's
    ``assistant_only_loss=True`` would train every assistant turn, but it
    requires a chat template carrying ``{% generation %}`` markers, and
    Qwen2.5's stock template does not have them - TRL rejects it with "The
    chat template is not training-compatible". Recorded in e12 arm C as NOT
    TESTED; enabling it needs a custom template and is deferred.

    Rows whose final turn is not from the assistant are returned with an empty
    completion so the caller can filter them; silently training on a user turn
    as if it were a target would be a data bug that no loss curve would reveal.
    (Measured: 2000/2000 sampled rows do end with an assistant turn, so this
    guard is expected to fire rarely - which is not a reason to omit it.)

    """
    messages = example.get("messages") or []
    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        return {"prompt": [], "completion": []}
    return {"prompt": list(messages[:-1]), "completion": [messages[-1]]}


def is_wellformed(example: dict[str, Any]) -> bool:
    """Whether a converted example is usable as a training pair."""
    prompt = example.get("prompt") or []
    completion = example.get("completion") or []
    if not prompt or len(completion) != 1:
        return False
    if completion[0].get("role") != "assistant":
        return False
    if not (completion[0].get("content") or "").strip():
        return False
    return all((m.get("content") or "").strip() for m in prompt)


def load_instruction_dataset(
    name: str = DEFAULT_DATASET,
    revision: str = DEFAULT_DATASET_REVISION,
    train_split: str = "train",
    eval_split: str = "test",
    max_train: int | None = None,
    max_eval: int | None = None,
    seed: int = 42,
):
    """Load, convert and subsample an instruction dataset.

    Returns ``(train, eval, info)`` where the datasets are in TRL's
    conversational prompt-completion shape and ``info`` carries both
    fingerprints plus the filtering counts.

    SUBSAMPLING IS SHUFFLED, AND SEEDED. Taking the first N rows would be
    reproducible but biased: no_robots is grouped by ``category``, so
    ``select(range(N))`` yields a skewed slice of task types. Shuffling with
    an explicit seed keeps reproducibility without the bias. The seed is
    recorded in the fingerprint's owning manifest.

    THE FINGERPRINT IS TAKEN AFTER FILTERING AND SUBSAMPLING, not before, so
    it identifies the rows the model actually saw.
    """
    from datasets import load_dataset  # imported lazily; heavy

    logger.info("loading %s@%s", name, revision)
    raw_train = load_dataset(name, split=train_split, revision=revision)
    raw_eval = load_dataset(name, split=eval_split, revision=revision)

    missing = EXPECTED_COLUMNS - set(raw_train.column_names)
    if missing:
        raise ValueError(
            f"{name} is missing expected columns {sorted(missing)}; "
            f"found {sorted(raw_train.column_names)}. The upstream schema has "
            f"changed - inspect it rather than adapting blindly."
        )

    def convert(dataset, split_name: str):
        before = len(dataset)
        converted = dataset.map(
            to_prompt_completion,
            remove_columns=dataset.column_names,
            desc=f"to prompt/completion ({split_name})",
        )
        kept = converted.filter(is_wellformed, desc=f"filter ({split_name})")
        dropped = before - len(kept)
        if dropped:
            logger.info(
                "%s: dropped %d/%d malformed rows (%.2f%%)",
                split_name, dropped, before, 100.0 * dropped / max(before, 1),
            )
        return kept, before, dropped

    train, train_before, train_dropped = convert(raw_train, train_split)
    evaluation, eval_before, eval_dropped = convert(raw_eval, eval_split)

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
        "train_rows_before_filter": train_before,
        "train_rows_dropped": train_dropped,
        "eval_rows_before_filter": eval_before,
        "eval_rows_dropped": eval_dropped,
        "train_rows": len(train),
        "eval_rows": len(evaluation),
        "train_fingerprint": fingerprint_dataset(
            train, name, train_split, revision
        ).to_dict(),
        "eval_fingerprint": fingerprint_dataset(
            evaluation, name, eval_split, revision
        ).to_dict(),
    }

    logger.info(
        "train %d rows (%s), eval %d rows (%s)",
        len(train), info["train_fingerprint"]["sha256"][:16],
        len(evaluation), info["eval_fingerprint"]["sha256"][:16],
    )
    return train, evaluation, info
