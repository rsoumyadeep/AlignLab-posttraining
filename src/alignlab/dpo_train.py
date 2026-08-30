"""Phase 6 — DPO training entrypoint.

The loss is **ours** (`alignlab.dpo`), and so is the loop. TRL's `DPOTrainer`
is deliberately not used: PROJECT_INSTRUCTIONS requires the core computation to
be independently understandable and tested, and a training loop that calls a
library trainer would demonstrate neither.

WHAT THIS LOOP DOES, per preference pair:

    1. tokenize prompt+chosen and prompt+rejected, masking prompt tokens
    2. POLICY   forward on both  -> log π_θ(y_w|x),  log π_θ(y_l|x)   [grad]
    3. REFERENCE forward on both -> log π_ref(y_w|x), log π_ref(y_l|x) [no grad]
    4. loss = −log σ( β·[Δ_w − Δ_l] )
    5. backward, accumulate, step

ONE PAIR PER FORWARD, BY DESIGN. Chosen and rejected have different lengths.
Batching them together needs padding plus a mask, and the interaction between
padding, the completion mask and the shift is exactly the machinery this phase
exists to make legible. Gradient accumulation reaches the effective batch size
instead. Throughput is not the objective.

TWO PRE-FLIGHT VERIFICATIONS, BEFORE THE FIRST OPTIMISER STEP:

    * the reference has zero trainable parameters and is in eval() mode
    * with π == π_ref (true at initialisation, since both are the same
      checkpoint), every implicit reward is EXACTLY zero and the loss is
      exactly log 2

The second is the one that catches a reference pointed at the wrong checkpoint
— a bug that otherwise trains happily while optimising the wrong objective.

Usage:
    python -m alignlab.dpo_train env=server dpo.beta=0.1
    python -m alignlab.dpo_train env=server dpo=smoke
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from alignlab.config_schema import register_configs
from alignlab.device import describe_hardware, resolve_device
from alignlab.dpo import (
    LOSS_AT_INIT,
    dpo_loss,
    preference_accuracy,
    verify_reference_is_frozen,
    verify_zero_reward_at_init,
)
from alignlab.logging_utils import get_logger, setup_logging
from alignlab.logprobs import sequence_logprobs
from alignlab.manifest import capture_environment
from alignlab.masking import check_prefix_consistency, expected_labels
from alignlab.paths import (
    checkpoint_root,
    configure_hf_cache,
    describe_roots,
    generate_run_name,
    run_dir,
)
from alignlab.preference import load_preference_dataset
from alignlab.seeding import set_seed
from alignlab.storage import GIB, disk_status, require_free_space
from alignlab.tracking import build_tracker, write_metrics_jsonl

logger = get_logger(__name__)

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


def resolve_policy_path(cfg) -> Path:
    if cfg.policy_checkpoint:
        return Path(cfg.policy_checkpoint)
    return checkpoint_root(cfg.env.checkpoint_root or None) / cfg.sft_run_name / "final"


def tokenize_pair(tokenizer, row, device, max_length: int):
    """Tokenize one preference pair into ids + completion-masked labels.

    Returns None when either sequence exceeds ``max_length``. Truncating would
    remove tokens from the tail — which for a prompt/completion pair is exactly
    the completion DPO scores — and would truncate chosen and rejected by
    different amounts, making their log-probabilities incomparable. Dropping is
    the honest choice; the count is recorded.
    """
    try:
        chosen_ids, chosen_labels, _ = expected_labels(
            tokenizer, row["prompt"], row["chosen"]
        )
        rejected_ids, rejected_labels, _ = expected_labels(
            tokenizer, row["prompt"], row["rejected"]
        )
    except ValueError:
        return None
    if len(chosen_ids) > max_length or len(rejected_ids) > max_length:
        return None
    if len(chosen_ids) < 2 or len(rejected_ids) < 2:
        return None
    return {
        "chosen_ids": torch.tensor([chosen_ids], device=device),
        "chosen_labels": torch.tensor([chosen_labels], device=device),
        "rejected_ids": torch.tensor([rejected_ids], device=device),
        "rejected_labels": torch.tensor([rejected_labels], device=device),
    }


def score_pair(model, batch, grad: bool):
    """Log-probabilities of chosen and rejected under one model."""
    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        chosen_logits = model(input_ids=batch["chosen_ids"]).logits
        rejected_logits = model(input_ids=batch["rejected_ids"]).logits
    return (
        sequence_logprobs(chosen_logits, batch["chosen_labels"]),
        sequence_logprobs(rejected_logits, batch["rejected_labels"]),
    )


@torch.no_grad()
def evaluate(policy, reference, batches, beta: float) -> dict:
    """Preference accuracy (SUM primary, MEAN diagnostic) and reward stats.

    Both variants are computed every time. Reporting only one would hide the
    disagreement Phase 5 measured, which the pre-registration requires be kept
    visible.
    """
    policy.eval()
    sum_correct = mean_correct = reward_correct = 0
    margins, chosen_logps, rejected_logps = [], [], []
    chosen_tokens, rejected_tokens = [], []

    for batch in batches:
        p_chosen, p_rejected = score_pair(policy, batch, grad=False)
        r_chosen, r_rejected = score_pair(reference, batch, grad=False)

        sum_correct += int(preference_accuracy(p_chosen, p_rejected)[0])
        mean_correct += int(
            preference_accuracy(p_chosen, p_rejected, length_normalise=True)[0]
        )
        _, stats = dpo_loss(p_chosen, p_rejected, r_chosen, r_rejected, beta=beta)
        reward_correct += int(stats.reward_accuracy > 0.5)
        margins.append(stats.reward_margin)
        chosen_logps.append(stats.policy_chosen_logp)
        rejected_logps.append(stats.policy_rejected_logp)
        chosen_tokens.append(stats.chosen_tokens)
        rejected_tokens.append(stats.rejected_tokens)

    policy.train()
    n = max(len(batches), 1)
    return {
        "pairs": len(batches),
        "preference_accuracy_sum": sum_correct / n,
        "preference_accuracy_mean": mean_correct / n,
        "reward_accuracy": reward_correct / n,
        "reward_margin": sum(margins) / n,
        "policy_chosen_logp": sum(chosen_logps) / n,
        "policy_rejected_logp": sum(rejected_logps) / n,
        "chosen_tokens": sum(chosen_tokens) / n,
        "rejected_tokens": sum(rejected_tokens) / n,
    }


def run_dpo(cfg: DictConfig) -> dict[str, Any]:
    # BEFORE transformers - huggingface_hub freezes HF_HUB_CACHE at import,
    # so configuring the cache afterwards is a no-op. See configure_hf_cache.
    hf_env = configure_hf_cache(cfg.env.cache_root or None)

    from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

    run_name = cfg.run_name or generate_run_name(prefix=cfg.experiment)
    directory = run_dir(run_name, configured=cfg.env.output_root or None)
    setup_logging(
        level=cfg.logging.level, log_dir=directory, file_level=cfg.logging.file_level
    )

    logger.info("=" * 70)
    logger.info("AlignLab Phase 6 - Direct Preference Optimization")
    logger.info("=" * 70)
    logger.info("Run: %s | beta=%s | objective=%s", run_name, cfg.dpo.beta,
                "MEAN (diagnostic)" if cfg.dpo.length_normalise else "SUM (published)")

    logger.info("HF cache: %s", hf_env["effective_hub_cache"])
    if "rebound_live_constant" in hf_env:
        logger.warning("huggingface_hub was already imported; rebound %s",
                       hf_env["rebound_live_constant"])
    seed_report = set_seed(
        cfg.reproducibility.seed, deterministic=cfg.reproducibility.deterministic
    )
    logger.info("Seed: %s", seed_report)

    device = resolve_device(cfg.env.device)
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    logger.info("Device: %s | dtype: %s", device, dtype)

    ckpt_dir = checkpoint_root(cfg.env.checkpoint_root or None) / run_name
    if cfg.storage_guard:
        logger.info("Disk before: %s", disk_status(ckpt_dir.parent))
        # Only the final model is written (no intermediate checkpoints), so the
        # budget is one bf16 copy plus headroom.
        require_free_space(
            ckpt_dir.parent,
            required_bytes=int(3.1 * GIB),
            label=f"DPO final model (beta={cfg.dpo.beta})",
            allow_override=cfg.allow_low_disk,
        )

    # ------------------------------------------------------------------ data
    train_ds, eval_ds, data_info = load_preference_dataset(
        name=cfg.preference_dataset,
        train_split=cfg.preference_train_split,
        eval_split=cfg.preference_eval_split,
        max_train=cfg.max_train,
        max_eval=cfg.max_eval,
        seed=cfg.reproducibility.seed,
        drop_ties=cfg.drop_ties,
    )
    logger.info("Train fingerprint: %s", data_info["train_fingerprint"]["sha256"])
    logger.info("Eval  fingerprint: %s", data_info["eval_fingerprint"]["sha256"])

    # --------------------------------------------------- policy and reference
    policy_path = resolve_policy_path(cfg)
    if not policy_path.exists():
        raise FileNotFoundError(f"policy checkpoint not found: {policy_path}")
    logger.info("Policy + reference start from: %s", policy_path)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.id, revision=cfg.model.revision)
    policy = AutoModelForCausalLM.from_pretrained(policy_path, dtype=dtype).to(device)
    reference = AutoModelForCausalLM.from_pretrained(policy_path, dtype=dtype).to(device)

    for param in reference.parameters():
        param.requires_grad_(False)
    reference.eval()
    frozen_report = verify_reference_is_frozen(reference)
    logger.info("Reference frozen: %s", frozen_report)

    if cfg.dpo.gradient_checkpointing:
        policy.gradient_checkpointing_enable()
        policy.config.use_cache = False
    policy.train()

    n_params = sum(p.numel() for p in policy.parameters())
    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logger.info("Policy: %s params, %s trainable", f"{n_params:,}", f"{n_trainable:,}")

    # ------------------------------------------------------------ tokenize
    def prepare(dataset, name):
        batches, dropped = [], 0
        for i in range(len(dataset)):
            item = tokenize_pair(tokenizer, dataset[i], device, cfg.dpo.max_length)
            if item is None:
                dropped += 1
            else:
                batches.append(item)
        logger.info("%s: %d usable pairs, %d dropped (over %d tokens or malformed)",
                    name, len(batches), dropped, cfg.dpo.max_length)
        return batches, dropped

    train_batches, train_dropped = prepare(train_ds, "train")
    eval_batches, eval_dropped = prepare(eval_ds, "eval")
    if not train_batches:
        raise RuntimeError("no usable training pairs after tokenization")

    # ------------------------------------ prefix consistency, sampled audit
    prefix_bad = 0
    for i in range(min(200, len(train_ds))):
        row = train_ds[i]
        ok_c, _, _ = check_prefix_consistency(tokenizer, row["prompt"], row["chosen"])
        ok_r, _, _ = check_prefix_consistency(tokenizer, row["prompt"], row["rejected"])
        prefix_bad += int(not ok_c) + int(not ok_r)
    logger.info("prefix consistency: %d violations in %d sampled sequences",
                prefix_bad, 2 * min(200, len(train_ds)))

    # -------------------------- PRE-FLIGHT: zero reward when policy==reference
    first = train_batches[0]
    p_chosen, p_rejected = score_pair(policy, first, grad=False)
    r_chosen, r_rejected = score_pair(reference, first, grad=False)
    zero_report = verify_zero_reward_at_init(
        p_chosen, p_rejected, r_chosen, r_rejected, beta=cfg.dpo.beta
    )
    logger.info("zero-reward check: %s", zero_report)
    if not zero_report["rewards_zero"]:
        raise RuntimeError(
            f"policy and reference start from the SAME checkpoint, so every "
            f"implicit reward must be exactly 0 - got max "
            f"{zero_report['max_abs_chosen_reward']:.3e}. The reference is "
            f"wired to the wrong model, or the sum/mean branches disagree. "
            f"Refusing to train."
        )
    if not zero_report["loss_matches_log2"]:
        raise RuntimeError(
            f"loss at initialisation is {zero_report['loss']:.6f}, expected "
            f"log 2 = {LOSS_AT_INIT:.6f}. Refusing to train."
        )

    # ------------------------------------------------- baseline BEFORE training
    logger.info("Measuring the pre-DPO baseline on the eval split...")
    baseline = evaluate(policy, reference, eval_batches, cfg.dpo.beta)
    logger.info("BASELINE (SUM primary): %.4f | MEAN diagnostic: %.4f",
                baseline["preference_accuracy_sum"], baseline["preference_accuracy_mean"])

    # ------------------------------------------------------------- optimiser
    steps_per_epoch = max(1, len(train_batches) // cfg.dpo.gradient_accumulation_steps)
    total_steps = (
        cfg.dpo.max_steps if cfg.dpo.max_steps and cfg.dpo.max_steps > 0
        else int(steps_per_epoch * cfg.dpo.num_train_epochs)
    )
    warmup_steps = int(round(cfg.dpo.warmup_ratio * total_steps))
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=cfg.dpo.learning_rate, weight_decay=cfg.dpo.weight_decay,
    )
    scheduler = get_scheduler(
        cfg.dpo.lr_scheduler_type, optimizer=optimizer,
        num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )
    logger.info("Schedule: %d optimiser steps, warmup %d, effective batch %d pairs",
                total_steps, warmup_steps, cfg.dpo.gradient_accumulation_steps)

    resolved = OmegaConf.to_container(cfg, resolve=True)
    manifest = capture_environment(
        run_name=run_name, config=resolved, seed=cfg.reproducibility.seed,
        roots=describe_roots(
            output=cfg.env.output_root or None,
            checkpoint=cfg.env.checkpoint_root or None,
            cache=cfg.env.cache_root or None,
        ),
        notes={
            "phase": 6, "algorithm": "DPO (first-principles, alignlab.dpo)",
            "beta": cfg.dpo.beta,
            "objective": "MEAN" if cfg.dpo.length_normalise else "SUM",
            "policy_checkpoint": str(policy_path),
            "reference_checkpoint": str(policy_path),
            "reference_frozen": frozen_report,
            "zero_reward_at_init": zero_report,
            "prefix_violations": prefix_bad,
            "dataset": data_info,
            "train_dropped": train_dropped, "eval_dropped": eval_dropped,
            "baseline": baseline,
            "deferred": "USER explain-back checkpoints - see docs/phase6",
        },
    )
    manifest.write(directory)

    tracker = build_tracker(
        mode=cfg.tracking.mode, project=cfg.tracking.project, run_name=run_name,
        entity=cfg.tracking.entity, tags=list(cfg.tracking.tags),
        config=resolved, run_dir=directory,
    )

    # ---------------------------------------------------------------- train
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    history = []
    step = 0
    accumulated = 0
    optimizer.zero_grad()

    logger.info("Starting DPO training")
    for index, batch in enumerate(train_batches):
        p_chosen, p_rejected = score_pair(policy, batch, grad=True)
        r_chosen, r_rejected = score_pair(reference, batch, grad=False)
        loss, stats = dpo_loss(
            p_chosen, p_rejected, r_chosen, r_rejected,
            beta=cfg.dpo.beta, length_normalise=cfg.dpo.length_normalise,
        )
        (loss / cfg.dpo.gradient_accumulation_steps).backward()
        accumulated += 1

        if accumulated == cfg.dpo.gradient_accumulation_steps:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.dpo.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            accumulated = 0
            step += 1

            if step % cfg.dpo.logging_steps == 0:
                record = {
                    "step": step, "loss": stats.loss,
                    "reward_margin": stats.reward_margin,
                    "reward_accuracy": stats.reward_accuracy,
                    "chosen_reward": stats.chosen_reward,
                    "rejected_reward": stats.rejected_reward,
                    "lr": scheduler.get_last_lr()[0],
                }
                history.append(record)
                logger.info(
                    "step %d/%d | loss %.4f | margin %+.4f | chosen_r %+.4f | "
                    "rejected_r %+.4f | lr %.2e",
                    step, total_steps, stats.loss, stats.reward_margin,
                    stats.chosen_reward, stats.rejected_reward,
                    scheduler.get_last_lr()[0],
                )
                write_metrics_jsonl(directory / "metrics.jsonl", record, step=step)
                tracker.log_metrics(record, step=step)

            if step >= total_steps:
                break

    elapsed = time.time() - started
    peak_vram = int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
    logger.info("Training finished in %.1f s | peak VRAM %.2f GiB",
                elapsed, peak_vram / GIB)

    # ------------------------------------------------------------- evaluate
    logger.info("Evaluating after DPO...")
    final = evaluate(policy, reference, eval_batches, cfg.dpo.beta)
    logger.info("AFTER (SUM primary): %.4f | MEAN diagnostic: %.4f",
                final["preference_accuracy_sum"], final["preference_accuracy_mean"])

    final_dir = ckpt_dir / "final"
    policy.config.use_cache = True
    policy.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    model_bytes = sum(p.stat().st_size for p in final_dir.rglob("*") if p.is_file())
    logger.info("Saved policy to %s (%.2f GiB)", final_dir, model_bytes / GIB)

    summary = {
        "run_name": run_name,
        "beta": cfg.dpo.beta,
        "objective": "MEAN" if cfg.dpo.length_normalise else "SUM",
        "policy_checkpoint": str(policy_path),
        "model_id": cfg.model.id, "model_revision": cfg.model.revision,
        "parameters": n_params, "trainable_parameters": n_trainable,
        "dataset": data_info,
        "train_pairs": len(train_batches), "eval_pairs": len(eval_batches),
        "train_dropped": train_dropped, "eval_dropped": eval_dropped,
        "prefix_violations": prefix_bad,
        "reference_frozen": frozen_report,
        "zero_reward_at_init": zero_report,
        "total_steps": step, "planned_steps": total_steps,
        "learning_rate": cfg.dpo.learning_rate,
        "gradient_accumulation_steps": cfg.dpo.gradient_accumulation_steps,
        "seed": cfg.reproducibility.seed,
        "dtype": str(dtype),
        "train_runtime_s": elapsed,
        "peak_vram_bytes": peak_vram,
        "final_model_bytes": model_bytes,
        "baseline": baseline,
        "final": final,
        "history": history,
        "disk_after": disk_status(ckpt_dir.parent).to_dict(),
    }
    (directory / "dpo_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    tracker.log_metrics({f"final_{k}": v for k, v in final.items() if isinstance(v, (int, float))})
    tracker.finish()
    logger.info("Disk after: %s", disk_status(ckpt_dir.parent))
    return summary


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="dpo")
def main(cfg: DictConfig) -> None:
    run_dpo(cfg)


if __name__ == "__main__":
    register_configs()
    sys.exit(main())
