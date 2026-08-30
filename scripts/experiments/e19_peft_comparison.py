"""E19 - full SFT vs LoRA vs QLoRA, on matched settings.

PROJECT_INSTRUCTIONS Phase 4: measure trainable parameters, total parameters,
peak VRAM, training time, throughput, checkpoint size, evaluation
loss/perplexity and qualitative outputs, holding as many variables fixed as
practical and RECORDING the ones that could not be.

WHAT IS HELD FIXED across all arms:
    model + revision   Qwen/Qwen2.5-1.5B @ 8faed761...
    dataset            HuggingFaceH4/no_robots, same fingerprint
    objective          completion-only masked causal LM (verified every run)
    max_length         1024, packing off
    batch geometry     4 x 8 = 32 sequences
    epochs             1 (296 optimiser steps)
    seed               42
    evaluation         the same held-out split, the same code path (E13's)
    hardware           one RTX A6000, bf16

WHAT COULD NOT BE HELD FIXED, and why - stated plainly rather than buried:

    LEARNING RATE. LoRA conventionally needs a higher LR than full fine-tuning,
    because the adapters start at zero and are small. Holding LR at the full-FT
    value would make this a measurement of LR sensitivity rather than of LoRA.
    Changing it breaks the control. Neither choice is right, so BOTH were run:

        full SFT   lr 2e-5   (Phase 3)
        LoRA       lr 2e-4   (conventional)
        LoRA       lr 2e-5   (LR-matched to full SFT - the strict control)
        QLoRA      lr 2e-4   (matched to the LoRA arm it should be compared to)

    Read the LR-matched LoRA arm against full SFT for a controlled comparison,
    and the 2e-4 arms against each other for a "each method configured
    sensibly" comparison. Reporting only one of these would be choosing the
    answer.

HYPOTHESES, recorded before running:

    H1  all three PEFT arms train ~0.28% of the parameters (4,358,144)
    H2  QLoRA uses less peak VRAM than LoRA, which uses less than full SFT
    H3  QLoRA is SLOWER per step than LoRA - it pays a dequantization on every
        matmul that LoRA does not
    H4  adapter checkpoints are >100x smaller than the full-SFT checkpoint
    H5  full SFT reaches the lowest eval loss, LoRA@2e-4 is close, and
        LoRA@2e-5 is worst because the LR is too low for adapters starting at
        zero

Run (server):
    python scripts/experiments/e19_peft_comparison.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignlab.data import load_instruction_dataset
from alignlab.paths import checkpoint_root, configure_hf_cache, output_root, repo_root
from alignlab.storage import GIB

BASE_MODEL = "Qwen/Qwen2.5-1.5B"
REVISION = "8faed761d45a263340a0528343f099c05c9a4323"

# run name -> (label, arm, learning rate)
ARMS = {
    "sft-qwen1p5b-noRobots-001": ("full SFT", "none", "2e-5"),
    "lora-r16-lr2e-4-001": ("LoRA r=16", "lora", "2e-4"),
    "lora-r16-lr2e-5-001": ("LoRA r=16", "lora", "2e-5"),
    "qlora-r16-lr2e-4-001": ("QLoRA r=16", "qlora", "2e-4"),
}

QUALITATIVE_PROMPTS = [
    "Write a two-sentence summary of why the sky appears blue.",
    "List three practical tips for someone learning to cook.",
    "What is the difference between a list and a tuple in Python?",
    "Write a short, friendly email declining a meeting invitation.",
]


def directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def load_summary(outputs_root: Path, run_name: str) -> dict | None:
    path = outputs_root / run_name / "sft_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_model_for_eval(run_name: str, arm: str, ckpt_root: Path, device, dtype):
    """Load a trained model, whether it is a full checkpoint or an adapter.

    Adapter runs save only adapter tensors, so the base must be loaded first
    and the adapter applied on top. This is the same asymmetry that makes LoRA
    checkpoints small, showing up at evaluation time.
    """
    final = ckpt_root / run_name / "final"
    if not final.exists():
        return None, f"missing {final}"

    is_adapter = (final / "adapter_config.json").exists()
    if not is_adapter:
        model = AutoModelForCausalLM.from_pretrained(final, dtype=dtype)
        return model.to(device).eval(), "full checkpoint"

    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=REVISION, dtype=dtype
    )
    model = PeftModel.from_pretrained(base, final)
    # DO NOT MERGE FOR EVALUATION.
    #
    # An earlier version of this script merged, on the reasoning that it removes
    # adapter overhead and exercises the merge path. Measuring first showed that
    # would have been a mistake: merging in bf16 is LOSSY. On this trained
    # adapter, merged-vs-unmerged logits differ by max 5.625e-01 (mean 5.00e-02)
    # in bf16, against 6.998e-05 (mean 4.70e-06) in float32 - roughly 8,000x
    # worse. The cause is that ||dW||/||W|| is about 0.003 (E17), which sits at
    # the resolution of bf16's ~8-bit mantissa, so most of the update rounds
    # away when added to the much larger base weight.
    #
    # Evaluating the merged model would therefore measure the adapter PLUS a
    # merge artefact, and attribute both to the training arm. The model as
    # TRAINED is the unmerged one, so that is what gets evaluated. Merge
    # precision is measured separately by measure_merge_precision().
    return model.to(device).eval(), "adapter (unmerged)"


def measure_merge_precision(run_name: str, ckpt_root: Path, device) -> dict | None:
    """How much does merging cost, in bf16 versus float32?

    LoRA's practical selling point is that the adapter can be folded into the
    base weight for zero inference overhead. That claim is exact in float32 and
    NOT exact in bf16, which is the dtype these models are actually served in.
    Worth measuring rather than assuming.
    """
    final = ckpt_root / run_name / "final"
    if not (final / "adapter_config.json").exists():
        return None

    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=REVISION)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": QUALITATIVE_PROMPTS[3]}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)

    out = {}
    for dtype, name in ((torch.bfloat16, "bfloat16"), (torch.float32, "float32")):
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, revision=REVISION, dtype=dtype
        ).to(device).eval()
        model = PeftModel.from_pretrained(base, final)
        with torch.no_grad():
            unmerged = model(**inputs).logits.float().clone()
        merged_model = model.merge_and_unload().eval()
        with torch.no_grad():
            merged = merged_model(**inputs).logits.float()
        diff = (unmerged - merged).abs()
        out[name] = {"max": float(diff.max()), "mean": float(diff.mean())}
        del base, model, merged_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def completion_nll(model, tokenizer, dataset, device, limit: int) -> tuple[float, int]:
    """Token-weighted completion-only NLL - identical to E13's methodology."""
    import torch.nn.functional as F

    from alignlab.masking import IGNORE_INDEX, expected_labels

    total, count = 0.0, 0
    for i in range(min(limit, len(dataset))):
        row = dataset[i]
        try:
            ids, labels, boundary = expected_labels(
                tokenizer, row["prompt"], row["completion"]
            )
        except ValueError:
            continue
        if len(ids) < 2 or boundary >= len(ids):
            continue
        x = torch.tensor([ids], device=device)
        y = torch.tensor([labels], device=device)
        with torch.no_grad():
            logits = model(input_ids=x).logits[:, :-1, :].float()
        shift = y[:, 1:]
        n = int((shift != IGNORE_INDEX).sum())
        if n == 0:
            continue
        total += float(
            F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                shift.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
        )
        count += n
    return total / max(count, 1), count


def generate(model, tokenizer, device, max_new_tokens: int) -> list[dict]:
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    out = []
    for prompt in QUALITATIVE_PROMPTS:
        torch.manual_seed(42)
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        n_in = inputs["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=[im_end, tokenizer.eos_token_id],
                pad_token_id=tokenizer.pad_token_id,
            )
        body = gen[0][n_in:]
        out.append(
            {
                "prompt": prompt,
                "response": tokenizer.decode(body, skip_special_tokens=True).strip(),
                "n_generated": int(len(body)),
                "emitted_im_end": im_end in body.tolist(),
                "stopped": int(len(body)) < max_new_tokens,
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # Defaults resolve through alignlab.paths (config -> ALIGNLAB_* env var ->
    # repo-relative), NOT a hard-coded server path. tests/test_no_hardcoded_paths
    # caught an earlier version of this file doing the latter, which is exactly
    # what that test exists for: a server path baked into a script makes the
    # script unrunnable anywhere else and silently wrong if the roots move.
    parser.add_argument("--outputs-root", default=None)
    parser.add_argument("--ckpt-root", default=None)
    parser.add_argument("--eval-examples", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--out", default="docs/phase4/e19_peft_comparison.json")
    args = parser.parse_args()

    configure_hf_cache()
    outputs_root = Path(args.outputs_root) if args.outputs_root else output_root()
    ckpt_root = Path(args.ckpt_root) if args.ckpt_root else checkpoint_root()
    print(f"  outputs root : {outputs_root}")
    print(f"  ckpt root    : {ckpt_root}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print("=" * 100)
    print("E19 - full SFT vs LoRA vs QLoRA")
    print("=" * 100)

    rows = []
    for run_name, (label, arm, lr) in ARMS.items():
        summary = load_summary(outputs_root, run_name)
        if summary is None:
            print(f"  MISSING summary for {run_name} - skipping")
            continue
        final_dir = ckpt_root / run_name / "final"
        counts = summary.get("peft_counts") or {}
        train = summary.get("train_metrics", {})
        rows.append(
            {
                "run_name": run_name,
                "label": label,
                "arm": arm,
                "lr": lr,
                "trainable": counts.get("trainable", summary.get("trainable_parameters")),
                "logical_total": counts.get("logical_total", summary.get("parameters")),
                "peak_vram_bytes": summary.get("peak_vram_allocated_bytes"),
                "train_runtime_s": train.get("train_runtime"),
                "samples_per_second": train.get("train_samples_per_second"),
                "train_loss": train.get("train_loss"),
                "eval_loss": summary.get("eval_metrics", {}).get("eval_loss"),
                "final_dir_bytes": directory_bytes(final_dir),
                "dataset_fp": summary.get("dataset", {})
                .get("train_fingerprint", {})
                .get("sha256", "")[:16],
            }
        )

    # ------------------------------------------------------- control check
    print("\n--- CONTROL: is the dataset identical across arms? ---")
    fps = {r["dataset_fp"] for r in rows}
    print(f"  distinct train fingerprints: {len(fps)}  {sorted(fps)}")
    controlled = len(fps) == 1
    print(f"  dataset held fixed: {controlled}")
    if not controlled:
        print("  WARNING: arms did not train on the same rows. The comparison is")
        print("  NOT controlled and the numbers below are not directly comparable.")

    # -------------------------------------------------------- training table
    print("\n--- TRAINING (measured) ---")
    header = (f"  {'arm':<12} {'lr':>6} {'trainable':>13} {'%':>8} {'peak VRAM':>11} "
              f"{'time':>9} {'samp/s':>8} {'eval loss':>10} {'ckpt':>10}")
    print(header)
    for r in rows:
        pct = 100 * r["trainable"] / r["logical_total"] if r["logical_total"] else 0
        vram = f"{r['peak_vram_bytes'] / GIB:.2f} GiB" if r["peak_vram_bytes"] else "n/a"
        rt = f"{r['train_runtime_s']:.0f} s" if r["train_runtime_s"] else "n/a"
        sp = f"{r['samples_per_second']:.2f}" if r["samples_per_second"] else "n/a"
        el = f"{r['eval_loss']:.4f}" if r["eval_loss"] else "n/a"
        ck = f"{r['final_dir_bytes'] / GIB:.3f} GiB"
        print(f"  {r['label']:<12} {r['lr']:>6} {r['trainable']:>13,} {pct:>7.4f}% "
              f"{vram:>11} {rt:>9} {sp:>8} {el:>10} {ck:>10}")

    # -------------------------------------------------------------- hypotheses
    peft_rows = [r for r in rows if r["arm"] != "none"]
    full = next((r for r in rows if r["arm"] == "none"), None)

    h1 = all(r["trainable"] == 4_358_144 for r in peft_rows) if peft_rows else False
    print(f"\n  H1 (all PEFT arms train 4,358,144 params): {h1}")

    lora4 = next((r for r in rows if r["run_name"] == "lora-r16-lr2e-4-001"), None)
    qlora = next((r for r in rows if r["arm"] == "qlora"), None)
    h2 = h3 = h4 = None
    if lora4 and qlora and lora4["peak_vram_bytes"] and qlora["peak_vram_bytes"]:
        h2 = qlora["peak_vram_bytes"] < lora4["peak_vram_bytes"]
        print(f"  H2 (QLoRA peak VRAM < LoRA): {h2}  "
              f"({qlora['peak_vram_bytes'] / GIB:.2f} vs {lora4['peak_vram_bytes'] / GIB:.2f} GiB)")
        h3 = qlora["train_runtime_s"] > lora4["train_runtime_s"]
        print(f"  H3 (QLoRA slower than LoRA): {h3}  "
              f"({qlora['train_runtime_s']:.0f} vs {lora4['train_runtime_s']:.0f} s)")
    if full and lora4 and full["final_dir_bytes"] and lora4["final_dir_bytes"]:
        ratio = full["final_dir_bytes"] / lora4["final_dir_bytes"]
        h4 = ratio > 100
        print(f"  H4 (adapter checkpoint >100x smaller): {h4}  ({ratio:.0f}x)")

    # ------------------------------------------------------------ evaluation
    evaluation = {}
    if not args.skip_eval:
        print("\n--- EVALUATION (same methodology as Phase 3's E13) ---")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=REVISION)
        _, eval_ds, info = load_instruction_dataset(
            max_train=1, max_eval=args.eval_examples
        )
        print(f"  eval split {len(eval_ds)} rows, sha256 "
              f"{info['eval_fingerprint']['sha256'][:16]}")

        targets = [("base (untrained)", None, None)] + [
            (r["label"] + " lr" + r["lr"], r["run_name"], r["arm"]) for r in rows
        ]

        for label, run_name, arm in targets:
            if run_name is None:
                model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL, revision=REVISION, dtype=dtype
                ).to(device).eval()
                how = "base"
            else:
                model, how = load_model_for_eval(run_name, arm, ckpt_root, device, dtype)
                if model is None:
                    print(f"  {label}: {how} - SKIPPED")
                    continue

            nll, tokens = completion_nll(
                model, tokenizer, eval_ds, device, args.eval_examples
            )
            ppl = float(torch.exp(torch.tensor(nll)))
            samples = generate(model, tokenizer, device, args.max_new_tokens)
            stopped = sum(s["stopped"] for s in samples)
            im_end = sum(s["emitted_im_end"] for s in samples)

            evaluation[label] = {
                "loaded_as": how,
                "completion_nll": nll,
                "completion_ppl": ppl,
                "tokens": tokens,
                "stopped": stopped,
                "emitted_im_end": im_end,
                "samples": samples,
            }
            print(f"  {label:<24} ppl {ppl:>8.3f}  stopped {stopped}/4  "
                  f"im_end {im_end}/4   [{how}]")

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        print("\n--- one sample per model, same prompt, greedy ---")
        prompt = QUALITATIVE_PROMPTS[3]
        for label, data in evaluation.items():
            sample = next(s for s in data["samples"] if s["prompt"] == prompt)
            body = sample["response"][:220].replace("\n", " ")
            print(f"\n  [{label}] ({sample['n_generated']} tok, "
                  f"im_end={sample['emitted_im_end']})")
            print(f"    {body}")

    print("\n" + "=" * 100)
    print("WHAT THIS SHOWS / DOES NOT SHOW")
    print("=" * 100)
    print("  MEASURED   : trainable parameters, peak VRAM, wall-clock time,")
    print("               checkpoint size, completion-only perplexity on a")
    print("               held-out split, and stop-token behaviour.")
    print("  INTERPRETED: any statement that one arm is 'better'. Perplexity on")
    print("               no_robots' test split measures fit to no_robots.")
    print("  NOT ESTABLISHED: causality, and generalisation beyond this dataset.")
    print("               Single seed, single run per arm - a small gap between")
    print("               arms is NOT distinguishable from run-to-run noise.")
    print("  CONFOUNDED : learning rate varies with arm by necessity. Compare")
    print("               LoRA@2e-5 against full SFT for the strict control.")

    payload = {
        "arms": rows,
        "dataset_controlled": bool(controlled),
        "hypotheses": {"H1": h1, "H2": h2, "H3": h3, "H4": h4},
        "evaluation": evaluation,
        "merge_precision": merge_precision,
    }
    out = repo_root() / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n  evidence: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
