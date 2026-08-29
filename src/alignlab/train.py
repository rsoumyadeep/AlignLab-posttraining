"""Phase 1 training entrypoint - the foundation smoke test.

WHAT THIS IS: an end-to-end exercise of every Phase 1 component wired together
in the order a real training script uses them - config composition, run
directory, logging, seeding, manifest capture, tracking, preemption handling,
the training loop, checkpointing, resume, and an evaluation pass.

WHAT THIS IS NOT: real training. The model here is a deliberately tiny linear
regression on synthetic data. No pretrained checkpoint is loaded, no tokenizer
is used, and no language modelling happens. Qwen2.5-1.5B is NOT downloaded by
this script. Real SFT arrives in Phase 3.

The toy model is the point. It makes the foundation verifiable in seconds on a
CPU laptop, so that when Phase 3 puts a 1.5B model on the GPU server, the
scaffolding underneath it has already been proven.

Usage:
    python -m alignlab.train
    python -m alignlab.train env=local train.max_steps=50
    python -m alignlab.train tracking.mode=offline reproducibility.seed=7
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from alignlab.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from alignlab.config_schema import register_configs
from alignlab.device import describe_hardware, resolve_device
from alignlab.evaluation import EvalResult, run_evaluation
from alignlab.logging_utils import get_logger, setup_logging
from alignlab.manifest import capture_environment
from alignlab.paths import (
    checkpoint_root,
    configure_hf_cache,
    generate_run_name,
    run_dir,
)
from alignlab.preemption import PreemptionHandler
from alignlab.seeding import set_seed
from alignlab.tracking import build_tracker, write_metrics_jsonl

logger = get_logger(__name__)

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


class ToyModel(torch.nn.Module):
    """A minimal linear model, used only to exercise the foundation.

    Small enough that a full train/checkpoint/resume cycle finishes in under a
    second on CPU, which is what makes it usable as a test.
    """

    def __init__(self, in_features: int = 4, out_features: int = 1) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def make_synthetic_batch(
    batch_size: int, in_features: int, device: torch.device, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate one batch from a fixed linear rule plus noise.

    Uses an explicit generator so batch generation is reproducible independently
    of whatever else consumed the global RNG.
    """
    x = torch.randn(batch_size, in_features, generator=generator)
    true_w = torch.linspace(1.0, 2.0, in_features).unsqueeze(1)
    y = x @ true_w + 0.1 * torch.randn(batch_size, 1, generator=generator)
    return x.to(device), y.to(device)


class MeanLossEvaluator:
    """Trivial evaluator: mean squared error over freshly generated batches.

    Exists to prove the evaluation harness runs end to end. It is NOT a
    language-model metric. Perplexity, win-rate and LLM-as-judge are DEFERRED
    to Phase 7.
    """

    name = "toy_mse"

    def __init__(self, batches: int = 5, batch_size: int = 32, in_features: int = 4):
        self.batches = batches
        self.batch_size = batch_size
        self.in_features = in_features

    def evaluate(self, model: torch.nn.Module, **kwargs: Any) -> EvalResult:
        device = next(model.parameters()).device
        generator = torch.Generator().manual_seed(12345)
        criterion = torch.nn.MSELoss()

        model.eval()
        total = 0.0
        with torch.no_grad():
            for _ in range(self.batches):
                x, y = make_synthetic_batch(
                    self.batch_size, self.in_features, device, generator
                )
                total += float(criterion(model(x), y))
        model.train()

        return EvalResult(
            name=self.name,
            metrics={"mse": total / self.batches},
            status="MEASURED",
            n_examples=self.batches * self.batch_size,
            notes="Synthetic regression MSE. Foundation check only, not an LLM metric.",
        )


def train(cfg: DictConfig) -> dict[str, Any]:
    """Run the Phase 1 smoke training loop. Returns a summary dict."""
    # -- run identity ------------------------------------------------------
    run_name = cfg.run_name or generate_run_name(prefix=cfg.experiment)
    directory = run_dir(run_name)

    setup_logging(
        level=cfg.logging.level, log_dir=directory, file_level=cfg.logging.file_level
    )
    logger.info("=" * 70)
    logger.info("AlignLab Phase 1 foundation smoke run: %s", run_name)
    logger.info("Run directory: %s", directory)
    logger.info("=" * 70)

    # -- storage policy ----------------------------------------------------
    # Must happen before anything could trigger a Hugging Face download.
    # Declaring cache_root in the config does nothing on its own; the HF
    # libraries read environment variables.
    hf_env = configure_hf_cache(cfg.env.cache_root or None)
    if hf_env:
        logger.info("Hugging Face cache directed to: %s", hf_env["HF_HUB_CACHE"])
    else:
        logger.info("Hugging Face cache left as already configured in the environment")

    # -- reproducibility ---------------------------------------------------
    seed_report = set_seed(
        cfg.reproducibility.seed, deterministic=cfg.reproducibility.deterministic
    )
    logger.info("Seeding: %s", seed_report)

    # -- device ------------------------------------------------------------
    device = resolve_device(cfg.env.device)
    hardware = describe_hardware()
    logger.info("Device: %s | CUDA available: %s", device, hardware.cuda_available)
    if cfg.env.expect_cuda and not hardware.cuda_available:
        logger.warning(
            "Config env=%s expects CUDA but torch reports none. Continuing on %s - "
            "results will NOT be comparable to GPU runs.",
            cfg.env.name,
            device,
        )

    # -- provenance --------------------------------------------------------
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved, dict)
    manifest = capture_environment(
        run_name=run_name,
        config=resolved,
        seed=cfg.reproducibility.seed,
        notes={
            "phase": "1A",
            "workload": "synthetic toy regression",
            "is_real_training": False,
            "model_downloaded": False,
        },
    )
    manifest.write(directory)
    OmegaConf.save(cfg, directory / "resolved_config.yaml")
    logger.info("Config SHA-256: %s", manifest.config_sha256)

    # -- tracking ----------------------------------------------------------
    tracker = build_tracker(
        mode=cfg.tracking.mode,
        project=cfg.tracking.project,
        run_name=run_name,
        entity=cfg.tracking.entity,
        tags=list(cfg.tracking.tags),
        run_dir=directory,
        config=resolved,
    )
    tracker.log_config(resolved)

    # -- model / optimizer -------------------------------------------------
    in_features = 4
    model = ToyModel(in_features=in_features).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = torch.nn.MSELoss()

    ckpt_dir = checkpoint_root() / run_name
    start_step = 0

    # -- resume ------------------------------------------------------------
    if cfg.train.resume:
        existing = latest_checkpoint(ckpt_dir)
        if existing is not None:
            payload = load_checkpoint(
                existing,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                map_location=device,
            )
            start_step = payload.step
            logger.info("Resumed from %s at step %d", existing, start_step)
        else:
            logger.info("No existing checkpoint in %s - starting fresh", ckpt_dir)

    # -- preemption --------------------------------------------------------
    def checkpoint_now() -> None:
        save_checkpoint(
            directory=ckpt_dir,
            step=current_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=resolved,
            keep_last=cfg.train.keep_last_checkpoints,
        )

    handler = PreemptionHandler(
        checkpoint_fn=checkpoint_now, requeue=cfg.preemption.requeue
    )
    if cfg.preemption.enabled:
        handler.register()

    # -- training loop -----------------------------------------------------
    generator = torch.Generator().manual_seed(cfg.reproducibility.seed)
    metrics_path = directory / "metrics.jsonl"
    current_step = start_step
    last_loss: float | None = None
    steps_this_invocation = 0
    stopped_early = False

    if start_step >= cfg.train.max_steps:
        # Resuming a run that already reached its step budget. Not an error,
        # but it must be stated: the loop below will not execute, so no loss is
        # produced and final_loss stays None rather than being reported as a
        # number that was never measured.
        logger.warning(
            "Resumed at step %d but train.max_steps is %d - nothing left to do. "
            "Raise train.max_steps to continue this run.",
            start_step,
            cfg.train.max_steps,
        )

    model.train()
    for step in range(start_step, cfg.train.max_steps):
        current_step = step + 1

        x, y = make_synthetic_batch(cfg.train.batch_size, in_features, device, generator)
        loss = criterion(model(x), y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.train.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()
        scheduler.step()

        last_loss = float(loss.detach())
        steps_this_invocation += 1

        if current_step % cfg.train.log_every == 0:
            payload = {"loss": last_loss, "lr": scheduler.get_last_lr()[0]}
            logger.info("step %d | loss %.6f | lr %.2e", current_step, *payload.values())
            tracker.log_metrics(payload, step=current_step)
            write_metrics_jsonl(metrics_path, payload, step=current_step)

        if current_step % cfg.train.checkpoint_every == 0:
            checkpoint_now()

        if handler.should_stop():
            logger.warning("Preemption detected at step %d", current_step)
            handler.handle()
            stopped_early = True
            break

    if cfg.preemption.enabled:
        handler.unregister()

    # -- final checkpoint --------------------------------------------------
    # Skipped when nothing ran: re-saving an identical checkpoint would rotate
    # a genuinely older one out of existence for no gain.
    if not stopped_early and steps_this_invocation > 0:
        final_path = save_checkpoint(
            directory=ckpt_dir,
            step=current_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=resolved,
            keep_last=cfg.train.keep_last_checkpoints,
        )
        logger.info("Final checkpoint: %s", final_path)

    # -- evaluation --------------------------------------------------------
    report = run_evaluation(
        model=model,
        evaluators=[MeanLossEvaluator(in_features=in_features)],
        run_name=run_name,
        model_description="ToyModel (synthetic regression) - NOT a language model",
        manifest={"config_sha256": manifest.config_sha256},
    )
    report.write(directory)
    for result in report.results:
        logger.info("eval %s: %s (%s)", result.name, result.metrics, result.status)
        if result.metrics:
            tracker.log_metrics(
                {f"eval/{k}": v for k, v in result.metrics.items()}, step=current_step
            )

    tracker.finish()

    summary = {
        "run_name": run_name,
        "run_dir": str(directory),
        "checkpoint_dir": str(ckpt_dir),
        "steps_completed": current_step,
        "steps_this_invocation": steps_this_invocation,
        # None, not NaN, when no step ran: a metric that was never measured
        # must not be reported as if it were a number.
        "final_loss": last_loss,
        "stopped_early": stopped_early,
        "config_sha256": manifest.config_sha256,
        "eval": {r.name: r.metrics for r in report.results},
    }
    logger.info("Run complete: %s", summary)
    return summary


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    register_configs()
    sys.exit(main())  # type: ignore[func-returns-value]
