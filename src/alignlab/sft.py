"""Phase 3 - supervised fine-tuning entrypoint.

This is the first AlignLab script that trains a real language model. It uses
Hugging Face TRL's SFTTrainer, which PROJECT_INSTRUCTIONS section 3 asks for
explicitly ("realistic industry engineering is itself an objective") while
also requiring that we understand what the trainer is doing rather than
treating it as a black box.

WHAT THIS MODULE ADDS AROUND SFTTrainer, and why each piece is not optional:

  storage guard      a full-parameter checkpoint here is ~20 GiB and the
                     server volume is shared and 99% full. The run refuses to
                     start rather than dying halfway through a write.

  mask verification  before the first optimiser step, the actual collated
                     batch is inspected and the prompt/completion boundary
                     re-derived independently. E12 proves the mechanism in
                     isolation; this proves it for THIS dataset, THIS
                     tokenizer, THIS config, on every run. A verification that
                     only ever ran once is a claim about the past.

  truncation audit   max_length silently drops the tail of long examples,
                     which for a completion-masked dataset can remove the
                     ENTIRE training signal from an example while leaving it
                     in the batch. Measured and reported, never assumed small.

  fingerprints       the manifest records the model revision AND a content
                     hash of the exact rows trained on, so a number can be
                     traced to the bytes that produced it.

The training loop itself is TRL's. Reimplementing it would teach less than
verifying it, and the educational implementations live in Phase 2's
src/alignlab/models/.

Usage:
    python -m alignlab.sft env=server
    python -m alignlab.sft env=server sft=smoke
    python -m alignlab.sft env=server sft=smoke tracking.mode=offline
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from alignlab.config_schema import register_configs
from alignlab.data import fingerprint_dataset, load_instruction_dataset
from alignlab.device import describe_hardware, resolve_device
from alignlab.logging_utils import get_logger, setup_logging
from alignlab.manifest import capture_environment
from alignlab.masking import (
    IGNORE_INDEX,
    check_prefix_consistency,
    describe_mask,
    expected_labels,
)
from alignlab.paths import (
    checkpoint_root,
    configure_hf_cache,
    describe_roots,
    generate_run_name,
    run_dir,
)
from alignlab.seeding import set_seed
from alignlab.storage import (
    GIB,
    disk_status,
    estimate_checkpoint_bytes,
    require_free_space_for_checkpoints,
)
from alignlab.tracking import build_tracker, write_metrics_jsonl

logger = get_logger(__name__)

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    """Resolve the model dtype, refusing to silently downgrade.

    "auto" means bfloat16 on a bf16-capable CUDA device and float32 otherwise.
    bfloat16 rather than float16 because it keeps float32's exponent range and
    therefore needs no loss scaling - the A6000s were VERIFIED bf16-capable in
    Phase 1B.
    """
    if requested != "auto":
        return getattr(torch, requested)
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def audit_prefix_consistency(tokenizer, dataset, sample: int = 300, seed: int = 0) -> dict:
    """Count rows where prompt tokens are NOT a prefix of prompt+completion.

    THIS AUDIT EXISTS BECAUSE THE FIRST REAL RUN FAILED IT. TRL emitted
    "Mismatch between tokenized prompt and the start of tokenized
    prompt+completion" and 3 of 200 rows were affected: completions beginning
    with newlines let BPE merge the template's trailing newline into the first
    completion token, so the mask boundary landed on a token that was half
    prompt and half answer. alignlab.data now strips that whitespace, and this
    audit is what proves the fix holds for the data actually being trained on.

    Sampled rather than exhaustive: this tokenizes each row twice more on top
    of what TRL already does, which is not free on 9,500 rows. A seeded sample
    of 300 detects a 1% defect rate with high probability while staying cheap
    enough to run on every training run.
    """
    import random

    n = len(dataset)
    indices = list(range(n))
    if n > sample:
        random.Random(seed).shuffle(indices)
        indices = indices[:sample]

    inconsistent = []
    for i in indices:
        row = dataset[i]
        ok, _, _ = check_prefix_consistency(tokenizer, row["prompt"], row["completion"])
        if not ok:
            inconsistent.append(i)

    result = {
        "rows_sampled": len(indices),
        "rows_total": n,
        "inconsistent": len(inconsistent),
        "inconsistent_rate": round(len(inconsistent) / max(len(indices), 1), 5),
        "inconsistent_indices": inconsistent[:20],
    }
    if inconsistent:
        logger.warning(
            "PREFIX INCONSISTENCY: %d/%d sampled rows tokenize such that the "
            "prompt is NOT a prefix of prompt+completion. Boundary-based loss "
            "masks are wrong by a token on those rows. Indices: %s",
            len(inconsistent), len(indices), inconsistent[:20],
        )
    else:
        logger.info(
            "prefix consistency: %d/%d sampled rows OK - boundary masks are sound",
            len(indices), len(indices),
        )
    return result


def verify_mask_on_real_batch(trainer, tokenizer, dataset, n_examples: int = 3) -> dict:
    """Re-verify the loss mask on the data this run will actually train on.

    E12 verifies the masking MECHANISM against a hand-built example. This
    verifies the mechanism as configured HERE - real dataset rows, real
    max_length, real collator - and it runs on every training run.

    The distinction matters: E12 passing says the machinery works; this says
    the machinery is wired to this experiment correctly. A truncation setting
    or a dataset schema change could break the second while the first still
    passes.
    """
    processed = trainer.train_dataset
    findings: list[dict] = []
    fully_masked = 0

    for i in range(min(n_examples, len(processed))):
        row = processed[i]
        report = describe_mask(tokenizer, row["input_ids"], row["labels"])
        raw = dataset[i]

        # Independent boundary, computed without TRL.
        try:
            _, our_labels, our_boundary = expected_labels(
                tokenizer, raw["prompt"], raw["completion"]
            )
            # Truncation makes lengths differ legitimately; compare the
            # boundary, which truncation of the TAIL does not move.
            boundary_agrees = report.boundary == our_boundary
        except ValueError as exc:
            our_boundary = None
            boundary_agrees = False
            logger.warning("independent boundary failed for example %d: %s", i, exc)

        if report.n_active == 0:
            fully_masked += 1

        findings.append(
            {
                "index": i,
                "n_tokens": report.n_tokens,
                "n_active": report.n_active,
                "active_fraction": round(report.active_fraction, 4),
                "trl_boundary": report.boundary,
                "independent_boundary": our_boundary,
                "boundary_agrees": bool(boundary_agrees),
                "active_text_head": report.active_text[:120],
            }
        )

    all_agree = all(f["boundary_agrees"] for f in findings)
    logger.info(
        "mask verification on real batches: %d/%d boundaries agree with the "
        "independent computation",
        sum(f["boundary_agrees"] for f in findings),
        len(findings),
    )
    for finding in findings:
        logger.info(
            "  example %d: %d/%d active (%.1f%%), boundary %s vs %s",
            finding["index"], finding["n_active"], finding["n_tokens"],
            100 * finding["active_fraction"], finding["trl_boundary"],
            finding["independent_boundary"],
        )

    return {
        "examples_checked": len(findings),
        "all_boundaries_agree": bool(all_agree),
        "fully_masked_examples": fully_masked,
        "findings": findings,
    }


def audit_truncation(trainer, max_length: int) -> dict:
    """Measure how much signal max_length is discarding.

    THIS IS NOT A FORMALITY. With completion-only masking, truncation removes
    tokens from the END of the sequence - which is exactly where the trainable
    completion lives. An example truncated hard enough keeps its prompt, loses
    its answer, and contributes ZERO gradient while still occupying a batch
    slot and diluting the mean loss. Reporting "0 active positions" counts is
    the only way to see that happening.
    """
    processed = trainer.train_dataset
    lengths = [len(row) for row in processed["input_ids"]]
    active_counts = [
        sum(1 for label in row if label != IGNORE_INDEX) for row in processed["labels"]
    ]

    at_limit = sum(1 for n in lengths if n >= max_length)
    zero_signal = sum(1 for n in active_counts if n == 0)
    total = len(lengths)

    audit = {
        "examples": total,
        "max_length": max_length,
        "at_or_over_limit": at_limit,
        "at_limit_fraction": round(at_limit / total, 4) if total else 0.0,
        "zero_active_examples": zero_signal,
        "zero_active_fraction": round(zero_signal / total, 4) if total else 0.0,
        "tokens_total": sum(lengths),
        "active_tokens_total": sum(active_counts),
        "active_token_fraction": (
            round(sum(active_counts) / sum(lengths), 4) if sum(lengths) else 0.0
        ),
        "length_p50": sorted(lengths)[total // 2] if total else 0,
        "length_max": max(lengths) if total else 0,
    }

    logger.info(
        "truncation audit: %d/%d examples at the %d-token limit (%.1f%%); "
        "%d examples have ZERO trainable tokens; %.1f%% of all tokens carry loss",
        at_limit, total, max_length, 100 * audit["at_limit_fraction"],
        zero_signal, 100 * audit["active_token_fraction"],
    )
    if zero_signal:
        logger.warning(
            "%d examples contribute NO gradient - their completions were "
            "truncated away entirely. Raise max_length or filter them.",
            zero_signal,
        )
    return audit


def run_sft(cfg: DictConfig) -> dict[str, Any]:
    """Execute one supervised fine-tuning run."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    run_name = cfg.run_name or generate_run_name(prefix=cfg.experiment)
    directory = run_dir(run_name, configured=cfg.env.output_root or None)
    setup_logging(
        level=cfg.logging.level, log_dir=directory, file_level=cfg.logging.file_level
    )

    logger.info("=" * 70)
    logger.info("AlignLab Phase 3 - supervised fine-tuning")
    logger.info("=" * 70)
    logger.info("Run: %s", run_name)
    logger.info("Output: %s", directory)

    hf_env = configure_hf_cache(cfg.env.cache_root or None)
    if hf_env:
        logger.info("HF cache configured: %s", hf_env)

    seed_report = set_seed(
        cfg.reproducibility.seed, deterministic=cfg.reproducibility.deterministic
    )
    logger.info("Seed: %s", seed_report)

    device = resolve_device(cfg.env.device)
    hardware = describe_hardware()
    dtype = resolve_dtype(cfg.model.dtype, device)
    logger.info("Device: %s | dtype: %s", device, dtype)
    if cfg.env.expect_cuda and not hardware.cuda_available:
        logger.warning(
            "env=%s expects CUDA but none is available - this run will be "
            "very slow and its timings are NOT comparable to a GPU run",
            cfg.env.name,
        )

    ckpt_dir = checkpoint_root(cfg.env.checkpoint_root or None) / run_name

    # ------------------------------------------------------- storage pre-flight
    # Before anything expensive. The parameter count is not yet known (the
    # model is not loaded), so the published 1.54B is used for the estimate and
    # the real figure is recorded afterwards.
    if cfg.storage_guard:
        logger.info("Disk before: %s", disk_status(ckpt_dir.parent))
        require_free_space_for_checkpoints(
            ckpt_dir.parent,
            n_params=1_543_714_304,
            keep_last=cfg.sft.save_total_limit,
            param_dtype="bfloat16",
            optimizer="adamw",
            label=f"{cfg.model.id} full-parameter SFT",
            allow_override=cfg.allow_low_disk,
        )
    else:
        logger.warning("storage_guard is DISABLED for this run")

    # ------------------------------------------------------------------- data
    train_ds, eval_ds, data_info = load_instruction_dataset(
        name=cfg.data.name,
        revision=cfg.data.revision,
        train_split=cfg.data.train_split,
        eval_split=cfg.data.eval_split,
        max_train=cfg.data.max_train,
        max_eval=cfg.data.max_eval,
        seed=cfg.reproducibility.seed,
    )
    logger.info("Train fingerprint: %s", data_info["train_fingerprint"]["sha256"])
    logger.info("Eval  fingerprint: %s", data_info["eval_fingerprint"]["sha256"])

    # -------------------------------------------------------- model, tokenizer
    logger.info("Loading %s @ %s", cfg.model.id, cfg.model.revision)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.id, revision=cfg.model.revision
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.id,
        revision=cfg.model.revision,
        dtype=dtype,
        attn_implementation=cfg.model.attn_implementation,
    )
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Parameters: %s total, %s trainable (%.2f%%)",
        f"{n_params:,}", f"{n_trainable:,}", 100.0 * n_trainable / n_params,
    )

    # A base model's EOS is <|endoftext|>, but the ChatML template terminates
    # assistant turns with <|im_end|>. The template already puts <|im_end|>
    # inside the trainable completion, so the model DOES learn to emit it - but
    # generation must be told to stop on it, which the eval script handles.
    logger.info(
        "eos_token=%r (id %s), pad_token=%r (id %s), im_end id=%s",
        tokenizer.eos_token, tokenizer.eos_token_id,
        tokenizer.pad_token, tokenizer.pad_token_id,
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
    )

    # --------------------------------------------------------------- training
    #
    # TRANSFORMERS 5.x MIGRATION FINDING. `warmup_ratio` no longer exists on
    # TrainingArguments/SFTConfig - only `warmup_steps` does. Passing it raises
    # TypeError: unexpected keyword argument 'warmup_ratio'. We keep the ratio
    # as AlignLab's knob because it is the portable one (it means the same
    # thing when the dataset size changes) and convert it here, which also
    # makes the resulting step count visible in the log instead of implicit.
    steps_per_epoch = max(
        1,
        len(train_ds)
        // (cfg.sft.per_device_train_batch_size * cfg.sft.gradient_accumulation_steps),
    )
    total_steps = (
        cfg.sft.max_steps
        if cfg.sft.max_steps and cfg.sft.max_steps > 0
        else int(steps_per_epoch * cfg.sft.num_train_epochs)
    )
    warmup_steps = int(round(cfg.sft.warmup_ratio * total_steps))
    logger.info(
        "Schedule: %d steps/epoch, %d total optimiser steps, warmup %d "
        "(ratio %.3f), effective batch %d sequences",
        steps_per_epoch, total_steps, warmup_steps, cfg.sft.warmup_ratio,
        cfg.sft.per_device_train_batch_size * cfg.sft.gradient_accumulation_steps,
    )

    sft_args = SFTConfig(
        output_dir=str(ckpt_dir),
        max_length=cfg.sft.max_length,
        packing=cfg.sft.packing,
        completion_only_loss=cfg.sft.completion_only_loss,
        per_device_train_batch_size=cfg.sft.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.sft.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.sft.gradient_accumulation_steps,
        learning_rate=cfg.sft.learning_rate,
        lr_scheduler_type=cfg.sft.lr_scheduler_type,
        warmup_steps=warmup_steps,
        weight_decay=cfg.sft.weight_decay,
        max_grad_norm=cfg.sft.max_grad_norm,
        num_train_epochs=cfg.sft.num_train_epochs,
        max_steps=cfg.sft.max_steps,
        logging_steps=cfg.sft.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.sft.eval_steps,
        save_strategy="steps",
        save_steps=cfg.sft.save_steps,
        save_total_limit=cfg.sft.save_total_limit,
        seed=cfg.sft.seed,
        data_seed=cfg.sft.seed,
        bf16=(dtype == torch.bfloat16),
        gradient_checkpointing=cfg.model.gradient_checkpointing,
        dataloader_num_workers=cfg.env.num_workers,
        report_to=[],  # AlignLab owns tracking; see alignlab.tracking
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    logger.info("TRL resolved completion_only_loss -> %s", trainer.completion_only_loss)

    # ------------------------------------------- verify BEFORE the first step
    prefix_audit = audit_prefix_consistency(
        tokenizer, train_ds, seed=cfg.reproducibility.seed
    )
    mask_check = verify_mask_on_real_batch(trainer, tokenizer, train_ds)
    truncation = audit_truncation(trainer, cfg.sft.max_length)

    if not mask_check["all_boundaries_agree"]:
        raise RuntimeError(
            "Loss-mask verification FAILED on real batches: TRL's boundary "
            "disagrees with the independent computation. Refusing to train. "
            "PROJECT_INSTRUCTIONS section 3 makes a verified mask a "
            "precondition, and a disagreement here means one of the two is "
            "wrong - investigate before adjusting anything."
        )
    if not trainer.completion_only_loss:
        raise RuntimeError(
            "completion_only_loss resolved to False - prompt tokens would "
            "carry gradient. Refusing to train. If this is deliberate (an "
            "ablation), set it explicitly and record why."
        )

    # -------------------------------------------------------------- manifest
    resolved = OmegaConf.to_container(cfg, resolve=True)
    manifest = capture_environment(
        run_name=run_name,
        config=resolved,
        seed=cfg.reproducibility.seed,
        roots=describe_roots(
            output=cfg.env.output_root or None,
            checkpoint=cfg.env.checkpoint_root or None,
            cache=cfg.env.cache_root or None,
        ),
        notes={
            "phase": 3,
            "model_id": cfg.model.id,
            "model_revision": cfg.model.revision,
            "model_parameters": n_params,
            "model_trainable_parameters": n_trainable,
            "dtype": str(dtype),
            "dataset": data_info,
            "loss_mask_verified": mask_check,
            "prefix_consistency": prefix_audit,
            "truncation_audit": truncation,
            "completion_only_loss": bool(trainer.completion_only_loss),
            "estimated_checkpoint_bytes": estimate_checkpoint_bytes(n_params),
            "deferred": "USER explain-back checkpoints - see docs/phase3",
        },
    )
    manifest_path = manifest.write(directory)
    logger.info("Manifest: %s", manifest_path)

    tracker = build_tracker(
        mode=cfg.tracking.mode,
        project=cfg.tracking.project,
        run_name=run_name,
        entity=cfg.tracking.entity,
        tags=list(cfg.tracking.tags),
        config=resolved,
        run_dir=directory,
    )

    # ---------------------------------------------------------------- resume
    resume_from = None
    if cfg.sft.resume and ckpt_dir.exists():
        candidates = sorted(
            (p for p in ckpt_dir.glob("checkpoint-*") if p.is_dir()),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if candidates:
            resume_from = str(candidates[-1])
            logger.info("Resuming from %s", resume_from)

    logger.info("Starting training")
    result = trainer.train(resume_from_checkpoint=resume_from)

    metrics = dict(result.metrics)
    logger.info("Training finished: %s", metrics)

    eval_metrics = trainer.evaluate()
    logger.info("Final evaluation: %s", eval_metrics)

    final_dir = ckpt_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info("Saved final model to %s", final_dir)

    actual_bytes = sum(
        p.stat().st_size for p in final_dir.rglob("*") if p.is_file()
    )
    logger.info(
        "Final model on disk: %.2f GiB (weights only, no optimiser state)",
        actual_bytes / GIB,
    )

    for key, value in {**metrics, **eval_metrics}.items():
        if isinstance(value, (int, float)):
            write_metrics_jsonl(directory / "metrics.jsonl", {key: value}, step=0)
    tracker.log_metrics({**metrics, **eval_metrics})
    tracker.finish()

    summary = {
        "run_name": run_name,
        "model_id": cfg.model.id,
        "model_revision": cfg.model.revision,
        "parameters": n_params,
        "trainable_parameters": n_trainable,
        "dataset": data_info,
        "train_metrics": metrics,
        "eval_metrics": eval_metrics,
        "loss_mask_verified": mask_check,
        "prefix_consistency": prefix_audit,
        "truncation_audit": truncation,
        "final_model_dir": str(final_dir),
        "final_model_bytes": actual_bytes,
        "estimated_full_checkpoint_bytes": estimate_checkpoint_bytes(n_params),
        "disk_after": disk_status(ckpt_dir.parent).to_dict(),
    }
    (directory / "sft_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    logger.info("Disk after: %s", disk_status(ckpt_dir.parent))
    return summary


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="sft")
def main(cfg: DictConfig) -> None:
    run_sft(cfg)


if __name__ == "__main__":
    register_configs()
    sys.exit(main())
