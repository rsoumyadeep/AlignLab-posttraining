"""Phase 7 — the evaluation entrypoint.

Configuration-driven: the model set, datasets, prompt set, generation settings
and judge all come from `configs/eval/`. Adding a model to the comparison is a
one-line config change, not a code change.

    python -m alignlab.evaluate env=server
    python -m alignlab.evaluate env=server eval=quick
    python -m alignlab.evaluate env=server eval.judge.enabled=false

WHAT IT PRODUCES: one machine-readable dashboard JSON plus a rendered text
table, both carrying full provenance — model revision, checkpoint, dataset and
evaluation fingerprints, tokenizer, dtype, generation settings, judge model,
git commit and dirty flag, timestamp and hardware.

WHAT IT REFUSES TO PRODUCE: a single quality score. See `evals/report.py`.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from alignlab.device import resolve_device
from alignlab.evals.metrics import APPLICABLE, NOT_APPLICABLE, NOT_MEASURED
from alignlab.evals.report import Dashboard, ModelEvaluation, compare_models, render_text
from alignlab.evals.runners import (
    GenerationSettings,
    build_provenance,
    completion_perplexity,
    generation_evaluation,
    preference_evaluation,
)
from alignlab.logging_utils import get_logger, setup_logging
from alignlab.paths import (
    checkpoint_root,
    configure_hf_cache,
    generate_run_name,
    run_dir,
)
from alignlab.seeding import set_seed

logger = get_logger(__name__)

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")

# The fixed qualitative prompt set, unchanged since Phase 3's E13 so that
# behaviour is comparable across five phases. Every one is generated for every
# model, and the full set is written to the dashboard - documentation showing a
# subset must state its selection rule.
GENERATION_PROMPTS = [
    "Write a two-sentence summary of why the sky appears blue.",
    "List three practical tips for someone learning to cook.",
    "What is the difference between a list and a tuple in Python?",
    "Write a short, friendly email declining a meeting invitation.",
    "Explain what a hash table is to someone who has never programmed.",
    "Give me three ideas for a weekend project using a Raspberry Pi.",
]

# Lessons from earlier phases, carried into every evaluation report so the
# design rationale travels with the numbers.
EVALUATION_LESSONS = [
    "Phase 3: a loss is a mean over a token POPULATION. Masked and unmasked "
    "losses are not comparable in either direction - measured 6.3314 (prompt) "
    "vs 3.6295 (completion) on the same model.",
    "Phase 4: perplexity put the PEFT arms within 3.5% of full SFT while "
    "stop-token behaviour was 0/4 vs 4/4. A likelihood metric did not reveal "
    "an unusable model.",
    "Phase 5: the same preference comparison gave 47.2% (SUM) and 58.3% "
    "(MEAN) on identical models and data - an 11-point swing from length.",
    "Phase 6: the SUM preference metric was dominated by a ~29-nat length gap. "
    "Per token the chosen response was genuinely better; the SUM comparison "
    "inverted that verdict.",
    "Therefore: always report the denominator, always report length beside "
    "preference and quality, and never collapse metrics into one score.",
]


def resolve_checkpoint(spec: str, cfg, ckpt_root: Path) -> tuple[str, str | None]:
    """Map a config checkpoint spec to (path_or_id, revision)."""
    if spec == "base":
        return cfg.eval.base_model, cfg.eval.base_revision
    return str(ckpt_root / spec / "final"), None


def load_model(path: str, revision: str | None, device, dtype, cfg=None):
    """Load a full checkpoint, an adapter, or the pinned base model.

    ADAPTERS ARE LOADED UNMERGED. Phase 4 measured bf16 merging as lossy -
    merged-vs-unmerged logits differed by 6.875e-01 (mean 6.103e-02) in bf16
    against 6.330e-05 in float32, roughly 12,460x worse, because ||dW||/||W||
    is about 0.003 and sits at the resolution of bf16's mantissa. Evaluating a
    merged adapter would measure the adapter PLUS a merge artefact and
    attribute both to the training run.
    """
    from transformers import AutoModelForCausalLM

    if revision is not None:
        return (
            AutoModelForCausalLM.from_pretrained(path, revision=revision, dtype=dtype)
            .to(device)
            .eval()
        )

    directory = Path(path)
    if not directory.exists():
        return None

    if (directory / "adapter_config.json").exists():
        from peft import PeftModel

        base_id = cfg.eval.base_model if cfg else "Qwen/Qwen2.5-1.5B"
        base_rev = cfg.eval.base_revision if cfg else None
        base = AutoModelForCausalLM.from_pretrained(
            base_id, revision=base_rev, dtype=dtype
        )
        logger.info("  loading as ADAPTER (unmerged) on top of %s", base_id)
        return PeftModel.from_pretrained(base, str(directory)).to(device).eval()

    return AutoModelForCausalLM.from_pretrained(directory, dtype=dtype).to(device).eval()


def load_lm_examples(cfg, seed: int):
    """Instruction data for perplexity (prompt/completion shape)."""
    from alignlab.data import load_instruction_dataset

    _, evaluation, info = load_instruction_dataset(
        name=cfg.eval.perplexity.dataset,
        eval_split=cfg.eval.perplexity.split,
        max_train=1,
        max_eval=cfg.eval.perplexity.max_examples,
        seed=seed,
    )
    return list(evaluation), info


def load_preference_examples(cfg, seed: int):
    from alignlab.preference import load_preference_dataset

    _, evaluation, info = load_preference_dataset(
        name=cfg.eval.preference.dataset,
        eval_split=cfg.eval.preference.split,
        max_train=1,
        max_eval=cfg.eval.preference.max_examples,
        seed=seed,
    )
    return list(evaluation), info



def _fingerprint_record(section, info: dict) -> dict:
    """The dataset identity behind one metric family.

    Populated per family rather than once per run: the eval-full-001 dashboard
    carried ``dataset_fingerprint: null`` in every record while a single
    ``eval_fingerprint`` named the preference rows and sat beside the
    perplexity numbers as well. A provenance field that is silently null is the
    same failure this project's status vocabulary exists to prevent.
    """
    if not info:
        return {"status": NOT_MEASURED}
    fingerprint = info.get("eval_fingerprint", {})
    return {
        "status": APPLICABLE,
        "dataset": info.get("name"),
        "split": info.get("eval_split"),
        "revision": info.get("revision"),
        "n_rows": info.get("eval_rows"),
        "rows_dropped": info.get("eval_rows_dropped"),
        "sha256": fingerprint.get("sha256"),
        "max_examples": getattr(section, "max_examples", None),
    }


def run_evaluation_suite(cfg: DictConfig) -> dict[str, Any]:
    from transformers import AutoTokenizer

    run_name = cfg.run_name or generate_run_name(prefix=cfg.experiment)
    directory = run_dir(run_name, configured=cfg.env.output_root or None)
    setup_logging(level=cfg.logging.level, log_dir=directory,
                  file_level=cfg.logging.file_level)

    logger.info("=" * 70)
    logger.info("AlignLab Phase 7 - evaluation suite")
    logger.info("=" * 70)

    configure_hf_cache(cfg.env.cache_root or None)
    set_seed(cfg.reproducibility.seed, deterministic=cfg.reproducibility.deterministic)
    device = resolve_device(cfg.env.device)
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    ckpt_root = checkpoint_root(cfg.env.checkpoint_root or None)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.eval.base_model, revision=cfg.eval.base_revision
    )

    lm_examples, lm_info = ([], {})
    if cfg.eval.perplexity.enabled:
        lm_examples, lm_info = load_lm_examples(cfg, cfg.reproducibility.seed)
        logger.info("perplexity set: %d examples (%s)", len(lm_examples),
                    lm_info["eval_fingerprint"]["sha256"][:16])

    pref_examples, pref_info = ([], {})
    if cfg.eval.preference.enabled:
        pref_examples, pref_info = load_preference_examples(cfg, cfg.reproducibility.seed)
        logger.info("preference set: %d pairs (%s)", len(pref_examples),
                    pref_info["eval_fingerprint"]["sha256"][:16])

    fingerprints = {
        "perplexity": _fingerprint_record(cfg.eval.perplexity, lm_info),
        "preference": _fingerprint_record(cfg.eval.preference, pref_info),
    }

    settings = GenerationSettings(
        max_new_tokens=cfg.eval.generation.max_new_tokens,
        do_sample=cfg.eval.generation.do_sample,
        temperature=cfg.eval.generation.temperature,
        top_k=cfg.eval.generation.top_k,
        top_p=cfg.eval.generation.top_p,
        seed=cfg.eval.generation.seed,
    )

    dashboard = Dashboard(
        created_at=datetime.now(timezone.utc).isoformat(),
        protocol={
            "run_name": run_name,
            "base_model": cfg.eval.base_model,
            "base_revision": cfg.eval.base_revision,
            "dataset": cfg.eval.perplexity.dataset if cfg.eval.perplexity.enabled else None,
            "lm_eval_fingerprint": (
                lm_info.get("eval_fingerprint", {}).get("sha256") if lm_info else None
            ),
            "preference_dataset": (
                cfg.eval.preference.dataset if cfg.eval.preference.enabled else None
            ),
            "eval_fingerprint": (
                pref_info.get("eval_fingerprint", {}).get("sha256") if pref_info else None
            ),
            "n_preference_pairs": len(pref_examples),
            "n_lm_examples": len(lm_examples),
            "n_generation_prompts": len(GENERATION_PROMPTS),
            "generation_settings": settings.to_dict(),
            "generation_prompts": list(GENERATION_PROMPTS),
            "judge_model": cfg.eval.judge.model if cfg.eval.judge.enabled else None,
            "seed": cfg.reproducibility.seed,
            "dtype": str(dtype),
            "device": str(device),
        },
        lessons=list(EVALUATION_LESSONS),
    )

    # ------------------------------------------------------- per-model pass
    for entry in cfg.eval.models:
        path, revision = resolve_checkpoint(entry.checkpoint, cfg, ckpt_root)
        logger.info("evaluating %s (%s)", entry.name, path)
        model = load_model(path, revision, device, dtype, cfg)
        if model is None:
            logger.warning("checkpoint missing, skipping: %s", path)
            dashboard.models.append(ModelEvaluation(
                name=entry.name, checkpoint=str(path),
                provenance={}, statuses={"all": NOT_MEASURED},
                notes=[f"checkpoint not found: {path}"],
            ))
            continue

        provenance = build_provenance(
            model_id=cfg.eval.base_model,
            tokenizer_id=cfg.eval.base_model,
            device=device, dtype=dtype,
            model_revision=revision, checkpoint=str(path),
            dataset=cfg.eval.perplexity.dataset if cfg.eval.perplexity.enabled else None,
            dataset_fingerprint=fingerprints["perplexity"].get("sha256"),
            eval_fingerprint=fingerprints["preference"].get("sha256"),
            generation=settings.to_dict(),
            judge_model=cfg.eval.judge.model if cfg.eval.judge.enabled else None,
            # Two metric families read two different datasets. A single
            # eval_fingerprint field sitting beside both sets of numbers
            # invites attributing the perplexity to the preference rows.
            extras={"fingerprints": fingerprints},
        )

        statuses: dict[str, str] = {}
        perplexity_results: dict[str, dict] = {}
        if cfg.eval.perplexity.enabled and lm_examples:
            results = completion_perplexity(
                model, tokenizer, lm_examples, device,
                max_length=cfg.eval.perplexity.max_length,
            )
            perplexity_results = {k: v.to_dict() for k, v in results.items()}
            statuses["perplexity"] = APPLICABLE
        else:
            statuses["perplexity"] = NOT_MEASURED

        preference_result = None
        if cfg.eval.preference.enabled and pref_examples:
            stats = preference_evaluation(
                model, tokenizer, pref_examples, device,
                max_length=cfg.eval.preference.max_length,
            )
            preference_result = stats.to_dict()
            statuses["preference"] = APPLICABLE
        else:
            statuses["preference"] = NOT_MEASURED

        generation_result = None
        if cfg.eval.generation.enabled:
            generation_result = generation_evaluation(
                model, tokenizer, GENERATION_PROMPTS, device, settings,
            )
            statuses["generation"] = APPLICABLE
        else:
            statuses["generation"] = NOT_MEASURED

        dashboard.models.append(ModelEvaluation(
            name=entry.name, checkpoint=str(path),
            provenance=provenance.to_dict(),
            perplexity=perplexity_results,
            preference=preference_result,
            generation=generation_result,
            statuses=statuses,
        ))

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------- comparisons
    for pair in cfg.eval.comparisons:
        try:
            dashboard.comparisons.extend(
                compare_models(dashboard, pair.baseline, pair.candidate)
            )
        except KeyError as exc:
            logger.warning("skipping comparison: %s", exc)

    # ------------------------------------------------------------- judge
    if cfg.eval.judge.enabled and cfg.eval.judge.comparisons:
        run_judge(cfg, dashboard, device, dtype)

    # ------------------------------------------------------------ outputs
    json_path = dashboard.write(directory / "evaluation_dashboard.json")
    text = render_text(dashboard)
    (directory / "evaluation_dashboard.txt").write_text(text, encoding="utf-8")
    print(text)
    logger.info("dashboard: %s", json_path)
    return dashboard.to_dict()


def run_judge(cfg, dashboard: Dashboard, device, dtype) -> None:
    """Pairwise LLM-as-judge over the fixed generation prompts."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from alignlab.evals.judge import judge_pairs

    logger.info("loading judge: %s", cfg.eval.judge.model)
    judge_tokenizer = AutoTokenizer.from_pretrained(
        cfg.eval.judge.model, revision=cfg.eval.judge.revision
    )
    judge_model = AutoModelForCausalLM.from_pretrained(
        cfg.eval.judge.model, revision=cfg.eval.judge.revision, dtype=dtype
    ).to(device).eval()

    for pair in cfg.eval.judge.comparisons:
        a = dashboard.by_name(pair.baseline)
        b = dashboard.by_name(pair.candidate)
        if not (a and b and a.generation and b.generation):
            logger.warning("judge: skipping %s vs %s (generation missing)",
                           pair.baseline, pair.candidate)
            continue
        prompts = [g["prompt"] for g in a.generation["generations"]]
        answers_a = [g["text"] for g in a.generation["generations"]]
        answers_b = [g["text"] for g in b.generation["generations"]]
        result = judge_pairs(
            judge_model, judge_tokenizer, prompts, answers_a, answers_b,
            device, pair.baseline, pair.candidate,
            judge_model_id=cfg.eval.judge.model, seed=cfg.eval.judge.seed,
        )
        dashboard.judge_results.append(result.to_dict())

    del judge_model
    if device.type == "cuda":
        torch.cuda.empty_cache()


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="eval")
def main(cfg: DictConfig) -> None:
    run_evaluation_suite(cfg)


if __name__ == "__main__":
    sys.exit(main())
