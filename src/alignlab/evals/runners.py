"""Model-facing evaluation runners.

The pure metrics live in `metrics.py`; this module is the layer that actually
touches a model. Kept separate so the arithmetic stays testable without a GPU.

WHAT THIS PROVIDES, all configuration-driven and none of it hard-coded to one
model or dataset:

    completion_perplexity   completion-only and full-sequence, labelled
    preference_evaluation   SUM (primary) and MEAN (diagnostic), with lengths
    generation_evaluation   fixed prompts, recorded settings, structural checks

EVERY RUNNER RECORDS ITS OWN PROVENANCE. A number without its model revision,
dataset fingerprint, tokenizer, dtype and generation settings cannot be
reproduced, and four phases of this project have shown that a metric detached
from its denominator is worse than no metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from alignlab.evals.metrics import (
    PerplexityResult,
    PreferenceStats,
    length_stats,
    perplexity_from_totals,
    structural_checks,
    termination_stats,
)
from alignlab.logging_utils import get_logger
from alignlab.logprobs import sequence_logprobs
from alignlab.masking import IGNORE_INDEX, expected_labels

logger = get_logger(__name__)


@dataclass
class GenerationSettings:
    """Everything that affects a generation, recorded so it can be repeated.

    Defaults are GREEDY (`do_sample=False`), which makes a comparison between
    two models depend only on their weights. Sampling adds a second source of
    variation to a comparison trying to isolate one.
    """

    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    seed: int = 42

    def to_dict(self) -> dict:
        return {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature if self.do_sample else None,
            "top_k": self.top_k if self.do_sample else None,
            "top_p": self.top_p if self.do_sample else None,
            "seed": self.seed,
            "decoding": "greedy" if not self.do_sample else "sampled",
        }

    def generate_kwargs(self) -> dict:
        """Only pass sampling parameters when sampling is actually on.

        Passing `temperature` with `do_sample=False` is silently ignored by
        transformers, which makes a config look meaningful when it is not.
        """
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }
        if self.do_sample:
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            if self.top_k is not None:
                kwargs["top_k"] = self.top_k
            if self.top_p is not None:
                kwargs["top_p"] = self.top_p
        return kwargs


@dataclass
class EvalProvenance:
    """The record that makes a result reproducible."""

    model_id: str
    model_revision: str | None
    checkpoint: str | None
    tokenizer: str
    dtype: str
    device: str
    dataset: str | None = None
    dataset_fingerprint: str | None = None
    eval_fingerprint: str | None = None
    generation: dict | None = None
    judge_model: str | None = None
    git_commit: str | None = None
    git_dirty: bool | None = None
    timestamp: str | None = None
    hardware: dict | None = None
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "checkpoint": self.checkpoint,
            "tokenizer": self.tokenizer,
            "dtype": self.dtype,
            "device": self.device,
            "dataset": self.dataset,
            "dataset_fingerprint": self.dataset_fingerprint,
            "eval_fingerprint": self.eval_fingerprint,
            "generation": self.generation,
            "judge_model": self.judge_model,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "timestamp": self.timestamp,
            "hardware": self.hardware,
            "extras": self.extras,
        }


def build_provenance(
    model_id: str,
    tokenizer_id: str,
    device: torch.device,
    dtype: torch.dtype,
    **kwargs: Any,
) -> EvalProvenance:
    """Capture provenance from the live process, guessing nothing."""
    from datetime import datetime, timezone

    from alignlab.device import describe_hardware
    from alignlab.manifest import git_info

    git = git_info() or {}
    return EvalProvenance(
        model_id=model_id,
        tokenizer=tokenizer_id,
        dtype=str(dtype),
        device=str(device),
        git_commit=git.get("commit"),
        git_dirty=git.get("dirty"),
        timestamp=datetime.now(timezone.utc).isoformat(),
        hardware=describe_hardware().as_dict(),
        **kwargs,
    )


# --------------------------------------------------------------- perplexity


@torch.no_grad()
def completion_perplexity(
    model,
    tokenizer,
    examples: Sequence[dict],
    device: torch.device,
    max_length: int = 1024,
    completion_key: str = "completion",
) -> dict[str, PerplexityResult]:
    """Completion-only AND full-sequence perplexity, both labelled.

    Both are returned deliberately. Phase 3 measured them differing by a factor
    of 1.7 in nats on the same model, and a report that quotes one without
    saying which invites exactly that confusion.

    Skipped examples are counted, never silently dropped.
    """
    totals = {"completion": [0.0, 0], "full_sequence": [0.0, 0]}
    skipped = 0

    for row in examples:
        try:
            ids, labels, _ = expected_labels(tokenizer, row["prompt"], row[completion_key])
        except (ValueError, KeyError):
            skipped += 1
            continue
        if len(ids) < 2 or len(ids) > max_length:
            skipped += 1
            continue

        input_ids = torch.tensor([ids], device=device)
        logits = model(input_ids=input_ids).logits[:, :-1, :].float()

        for region, target in (
            ("completion", torch.tensor([labels], device=device)[:, 1:]),
            ("full_sequence", input_ids[:, 1:]),
        ):
            n = int((target != IGNORE_INDEX).sum())
            if n == 0:
                continue
            total = float(
                F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                    reduction="sum",
                )
            )
            totals[region][0] += total
            totals[region][1] += n

    results = {}
    for region, (total, n) in totals.items():
        if n:
            results[region] = perplexity_from_totals(total, n, region)
    if skipped:
        logger.info("perplexity: skipped %d of %d examples", skipped, len(examples))
    return results


# --------------------------------------------------------------- preference


@torch.no_grad()
def preference_evaluation(
    model,
    tokenizer,
    pairs: Sequence[dict],
    device: torch.device,
    max_length: int = 1024,
) -> PreferenceStats:
    """Preference accuracy under SUM (primary) and MEAN (diagnostic).

    Both are always computed. `PreferenceStats` then reports them together with
    the token counts, so the length effect that dominated Phase 6 cannot be
    omitted from a report by accident.
    """
    sum_correct = mean_correct = 0
    chosen_lp, rejected_lp, chosen_tok, rejected_tok = [], [], [], []

    for row in pairs:
        try:
            c_ids, c_lab, _ = expected_labels(tokenizer, row["prompt"], row["chosen"])
            r_ids, r_lab, _ = expected_labels(tokenizer, row["prompt"], row["rejected"])
        except (ValueError, KeyError):
            continue
        if max(len(c_ids), len(r_ids)) > max_length or min(len(c_ids), len(r_ids)) < 2:
            continue

        chosen = sequence_logprobs(
            model(input_ids=torch.tensor([c_ids], device=device)).logits,
            torch.tensor([c_lab], device=device),
        )
        rejected = sequence_logprobs(
            model(input_ids=torch.tensor([r_ids], device=device)).logits,
            torch.tensor([r_lab], device=device),
        )

        sum_correct += int(chosen.sum_logprob[0] > rejected.sum_logprob[0])
        mean_correct += int(chosen.mean_logprob[0] > rejected.mean_logprob[0])
        chosen_lp.append(float(chosen.sum_logprob[0]))
        rejected_lp.append(float(rejected.sum_logprob[0]))
        chosen_tok.append(int(chosen.n_tokens[0]))
        rejected_tok.append(int(rejected.n_tokens[0]))

    n = len(chosen_lp)
    if n == 0:
        return PreferenceStats(0, 0, 0, 0.0, 0.0, 0.0, 0.0)
    return PreferenceStats(
        n=n,
        sum_correct=sum_correct,
        mean_correct=mean_correct,
        chosen_tokens_mean=sum(chosen_tok) / n,
        rejected_tokens_mean=sum(rejected_tok) / n,
        chosen_logp_mean=sum(chosen_lp) / n,
        rejected_logp_mean=sum(rejected_lp) / n,
    )


# --------------------------------------------------------------- generation


@torch.no_grad()
def generation_evaluation(
    model,
    tokenizer,
    prompts: Sequence[str],
    device: torch.device,
    settings: GenerationSettings,
    stop_token: str = "<|im_end|>",
) -> dict:
    """Generate from a FIXED prompt set and record everything about it.

    THE PROMPT SET IS PASSED IN, NOT CHOSEN HERE, and the full set is always
    returned. PROJECT_INSTRUCTIONS forbids cherry-picking: any documentation
    that shows a subset must state the selection rule, and the machine-readable
    output carries every generation so the selection can be checked.
    """
    stop_id = tokenizer.convert_tokens_to_ids(stop_token)
    eos_ids = [i for i in {stop_id, tokenizer.eos_token_id} if i is not None]

    generations = []
    for prompt in prompts:
        torch.manual_seed(settings.seed)
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        n_in = inputs["input_ids"].shape[1]
        output = model.generate(
            **inputs,
            **settings.generate_kwargs(),
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.pad_token_id,
        )
        body = output[0][n_in:]
        decoded = tokenizer.decode(body, skip_special_tokens=True).strip()
        generations.append(
            {
                "prompt": prompt,
                "text": decoded,
                "n_generated": int(len(body)),
                "emitted_stop_token": stop_id in body.tolist() if stop_id else False,
                "structure": structural_checks(decoded),
            }
        )

    termination = termination_stats(generations, settings.max_new_tokens)
    lengths = length_stats([g["n_generated"] for g in generations])
    return {
        "settings": settings.to_dict(),
        "n_prompts": len(prompts),
        "generations": generations,
        "termination": termination.to_dict(),
        "length": lengths.to_dict(),
        "mean_distinct_2": (
            sum(g["structure"]["distinct_2"] for g in generations) / len(generations)
            if generations
            else 0.0
        ),
    }
