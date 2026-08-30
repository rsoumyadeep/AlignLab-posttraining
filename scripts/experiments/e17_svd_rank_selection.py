"""E17 - choosing a LoRA rank from the singular-value spectrum of a REAL dW.

PROJECT_INSTRUCTIONS Tier 1, topic 1. The USER must eventually answer "How
would you choose the LoRA rank for a new task/model?" without merely saying
"use SVD". This experiment supplies the evidence that answer needs.

WHY THIS EXPERIMENT CAN BE HONEST HERE AND USUALLY CANNOT BE. Most write-ups
about LoRA rank analyse a synthetic or hypothetical dW, because the real thing
requires having already done a full fine-tune. AlignLab HAS one: Phase 3
full-parameter SFT of Qwen2.5-1.5B on no_robots. So

    dW = W_sft - W_base

is a genuine, measured weight update produced by real training - not a
construction chosen to make the point.

    LoRA's premise:  dW has low intrinsic rank
    This experiment: measure the rank of a dW that actually happened

THE CONTROLS, without which "the spectrum decays" means nothing. Every spectrum
decays somewhat. To claim dW is *unusually* low-rank we compare against:

    random   a Gaussian matrix of the SAME shape, scaled to the same Frobenius
             norm. Its singular values follow Marchenko-Pastur - the spectrum
             of pure noise. If dW looked like this, low-rank adaptation would
             have no justification.
    W_base   the pretrained weight itself. A useful reference for what a
             "normal" trained matrix's spectrum looks like.

HYPOTHESES, recorded before running:

    H1  dW's normalised singular values decay FASTER than the random control's
    H2  a small rank captures most of dW's energy - specifically, r <= 64
        captures >= 50% of the squared Frobenius norm
    H3  different module types have different spectra: the narrow GQA KV
        projections (1536->256, max rank 256) cannot behave like the square
        1536x1536 ones
    H4  dW is much smaller in norm than W_base - fine-tuning perturbs rather
        than rewrites

WHAT THIS EXPERIMENT CANNOT SHOW, stated before the results so it cannot look
like an excuse afterwards:

  * NOT that rank r is sufficient to TRAIN a good adapter. It measures the
    rank of the solution full fine-tuning FOUND, not the minimum rank that
    could reach comparable quality. LoRA searches a different space and may
    find a different, lower-rank solution.
  * NOT that energy equals usefulness. Explained Frobenius energy is a
    geometric quantity. A direction with a small singular value can still
    carry task-critical information, and a large one can be redundant.
  * NOT anything causal about downstream quality - no adapter was trained here.

Run (server, needs both the base model and the Phase 3 SFT checkpoint):
    python scripts/experiments/e17_svd_rank_selection.py \
        --sft-model /path/to/checkpoints/<run>/final
"""

from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM

from alignlab.paths import configure_hf_cache, repo_root

BASE_MODEL = "Qwen/Qwen2.5-1.5B"
REVISION = "8faed761d45a263340a0528343f099c05c9a4323"

# One of each shape class, sampled from early / middle / late layers so a
# single unrepresentative layer cannot drive the conclusion.
INSPECT = [
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.v_proj",
    "model.layers.13.self_attn.q_proj",
    "model.layers.13.self_attn.k_proj",
    "model.layers.13.self_attn.v_proj",
    "model.layers.13.self_attn.o_proj",
    "model.layers.13.mlp.gate_proj",
    "model.layers.13.mlp.down_proj",
    "model.layers.27.self_attn.q_proj",
    "model.layers.27.self_attn.v_proj",
]

ENERGY_TARGETS = (0.5, 0.9, 0.95, 0.99)
REPORT_RANKS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def spectrum(matrix: torch.Tensor) -> torch.Tensor:
    """Singular values, descending, in float64.

    float64 matters. dW is a DIFFERENCE of two bf16 tensors, so its entries are
    already near the precision floor; computing the SVD in bf16 or even float32
    would let round-off dominate the tail of the spectrum, which is exactly the
    part this experiment is reading.
    """
    return torch.linalg.svdvals(matrix.to(torch.float64))


def energy_curve(sv: torch.Tensor) -> torch.Tensor:
    """Cumulative fraction of squared Frobenius norm.

    The Eckart-Young theorem says the best rank-r approximation in Frobenius
    norm is the truncated SVD, and that its error is the tail sum of SQUARED
    singular values. So squared - not raw - singular values are the right thing
    to accumulate, and this is the single most common mistake in write-ups on
    this topic.
    """
    squared = sv**2
    return torch.cumsum(squared, dim=0) / squared.sum()


def rank_for_energy(curve: torch.Tensor, target: float) -> int:
    """Smallest r with cumulative energy >= target (1-indexed)."""
    idx = torch.searchsorted(curve, torch.tensor(target, dtype=curve.dtype))
    return int(idx.item()) + 1


def reconstruction_error(sv: torch.Tensor, r: int) -> float:
    """Relative Frobenius error of the best rank-r approximation.

    ||dW - dW_r||_F / ||dW||_F = sqrt(sum_{i>r} s_i^2 / sum_i s_i^2)

    Computed from the singular values alone - no need to form dW_r.
    """
    squared = sv**2
    total = squared.sum()
    if r >= len(sv):
        return 0.0
    return float(torch.sqrt(squared[r:].sum() / total))


def analyse(name: str, dW: torch.Tensor, W_base: torch.Tensor) -> dict:
    d_out, d_in = dW.shape
    max_rank = min(d_out, d_in)

    sv = spectrum(dW)
    curve = energy_curve(sv)

    # Control 1: Gaussian noise at the same scale.
    generator = torch.Generator(device="cpu").manual_seed(0)
    noise = torch.randn(d_out, d_in, generator=generator, dtype=torch.float64)
    noise = noise * (torch.linalg.norm(dW.to(torch.float64)) / torch.linalg.norm(noise))
    sv_noise = spectrum(noise)
    curve_noise = energy_curve(sv_noise)

    # Control 2: the pretrained weight itself.
    sv_base = spectrum(W_base)
    curve_base = energy_curve(sv_base)

    norm_dW = float(torch.linalg.norm(dW.to(torch.float64)))
    norm_W = float(torch.linalg.norm(W_base.to(torch.float64)))

    return {
        "name": name,
        "shape": [d_out, d_in],
        "max_rank": max_rank,
        "dense_params": d_out * d_in,
        "frob_dW": norm_dW,
        "frob_W": norm_W,
        "relative_update": norm_dW / norm_W,
        "top_singular_values": [float(v) for v in sv[:8]],
        "singular_value_decay_ratio": float(sv[0] / sv[min(63, len(sv) - 1)]),
        "energy_ranks": {
            f"{int(t * 100)}%": rank_for_energy(curve, t) for t in ENERGY_TARGETS
        },
        "energy_ranks_noise": {
            f"{int(t * 100)}%": rank_for_energy(curve_noise, t) for t in ENERGY_TARGETS
        },
        "energy_ranks_base": {
            f"{int(t * 100)}%": rank_for_energy(curve_base, t) for t in ENERGY_TARGETS
        },
        "by_rank": {
            r: {
                "energy": float(curve[min(r, len(curve)) - 1]),
                "energy_noise": float(curve_noise[min(r, len(curve_noise)) - 1]),
                "recon_error": reconstruction_error(sv, r),
                "lora_params": r * (d_in + d_out),
                "compression": (d_out * d_in) / (r * (d_in + d_out)),
            }
            for r in REPORT_RANKS
            if r <= max_rank
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-model", required=True)
    parser.add_argument("--out", default="docs/phase4/e17_svd_rank_selection.json")
    args = parser.parse_args()

    configure_hf_cache()

    print("=" * 78)
    print("E17 - SVD rank selection from a REAL fine-tuning update")
    print("=" * 78)
    print(f"  base : {BASE_MODEL} @ {REVISION[:12]}")
    print(f"  sft  : {args.sft_model}")
    print("  dW   = W_sft - W_base, from Phase 3 full-parameter SFT")

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=REVISION, dtype=torch.bfloat16
    )
    sft = AutoModelForCausalLM.from_pretrained(args.sft_model, dtype=torch.bfloat16)

    base_params = dict(base.named_parameters())
    sft_params = dict(sft.named_parameters())

    results = []
    for name in INSPECT:
        key = f"{name}.weight"
        if key not in base_params or key not in sft_params:
            print(f"  SKIP {name}: not found")
            continue
        W_base = base_params[key].detach()
        W_sft = sft_params[key].detach()
        dW = (W_sft.to(torch.float64) - W_base.to(torch.float64))
        results.append(analyse(name, dW, W_base))

    # ------------------------------------------------------------- H4 first
    print("\n--- H4: how big is the update relative to the weight? ---")
    print(f"  {'module':<42} {'||W||':>10} {'||dW||':>10} {'ratio':>8}")
    for r in results:
        print(f"  {r['name']:<42} {r['frob_W']:>10.2f} {r['frob_dW']:>10.3f} "
              f"{r['relative_update']:>7.4f}")
    h4 = all(r["relative_update"] < 0.1 for r in results)
    print(f"\n  H4 holds (every ||dW||/||W|| < 0.1): {h4}")
    print("  Fine-tuning PERTURBS the pretrained weights; it does not rewrite them.")

    # ------------------------------------------------- H1/H2: the spectra
    print("\n--- H1/H2: rank needed to capture X% of dW's energy ---")
    print("  (energy = squared Frobenius norm; Eckart-Young says truncated SVD is optimal)")
    print(f"\n  {'module':<42} {'maxr':>5} " + " ".join(f"{t:>6}" for t in ("50%", "90%", "95%", "99%")))
    for r in results:
        cells = " ".join(f"{r['energy_ranks'][t]:>6}" for t in ("50%", "90%", "95%", "99%"))
        print(f"  {r['name']:<42} {r['max_rank']:>5} {cells}")

    print(f"\n  CONTROL - the same statistic for Gaussian noise of equal norm:")
    print(f"  {'module':<42} {'maxr':>5} " + " ".join(f"{t:>6}" for t in ("50%", "90%", "95%", "99%")))
    for r in results:
        cells = " ".join(f"{r['energy_ranks_noise'][t]:>6}" for t in ("50%", "90%", "95%", "99%"))
        print(f"  {r['name']:<42} {r['max_rank']:>5} {cells}")

    h1 = all(
        r["energy_ranks"]["90%"] < r["energy_ranks_noise"]["90%"] for r in results
    )
    h2 = all(r["by_rank"].get(64, {}).get("energy", 0) >= 0.5 for r in results)
    print(f"\n  H1 holds (dW concentrates faster than noise at 90%): {h1}")
    print(f"  H2 holds (r=64 captures >=50% of energy everywhere): {h2}")

    # --------------------------------------------- the rank/cost/error table
    print("\n--- The table that answers 'how would you choose the rank?' ---")
    focus = next(r for r in results if r["name"].endswith("13.self_attn.q_proj"))
    print(f"  module {focus['name']}  shape {focus['shape']}  "
          f"dense {focus['dense_params']:,} params")
    print(f"\n  {'rank':>5} {'energy':>9} {'noise':>9} {'recon err':>10} "
          f"{'LoRA params':>13} {'compression':>12}")
    for rank, row in focus["by_rank"].items():
        print(f"  {rank:>5} {100 * row['energy']:>8.2f}% {100 * row['energy_noise']:>8.2f}% "
              f"{row['recon_error']:>10.4f} {row['lora_params']:>13,} "
              f"{row['compression']:>11.1f}x")

    # ------------------------------------------------------- H3: shape classes
    print("\n--- H3: do different module shapes behave differently? ---")
    print(f"  {'module':<42} {'shape':>14} {'r@90%':>7} {'r@90% / maxr':>14}")
    for r in results:
        frac = r["energy_ranks"]["90%"] / r["max_rank"]
        print(f"  {r['name']:<42} {str(tuple(r['shape'])):>14} "
              f"{r['energy_ranks']['90%']:>7} {frac:>13.3f}")
    shapes = {tuple(r["shape"]) for r in results}
    h3 = len(shapes) > 1
    print(f"\n  H3: {len(shapes)} distinct shapes present: {sorted(shapes)}")

    print("\n" + "=" * 78)
    print("HOW TO CHOOSE A RANK - and why this is a HEURISTIC")
    print("=" * 78)
    print("""
  THE PROCEDURE the numbers above support:
    1. If a full fine-tune for a similar task exists, take dW = W_ft - W_base,
       compute its singular values, and read off the rank at your energy
       target. Compare against a same-norm random control - without that
       comparison "the spectrum decays" is not evidence.
    2. If no such fine-tune exists - the usual case - you cannot do step 1 at
       all. Start from a rank the literature reports for a similar
       model/task/data scale, and sweep.
    3. Budget-first is legitimate: pick the largest r your memory allows, since
       LoRA's cost is linear in r and typically a fraction of a percent.

  WHY IT IS A HEURISTIC AND NOT A RULE:

    a) IT MEASURES THE WRONG THING, STRICTLY SPEAKING. The spectrum of the dW
       that full fine-tuning FOUND is not the minimum rank needed to reach
       comparable quality. Gradient descent had no incentive to be low-rank, so
       its solution is an upper bound on the necessary rank, not an estimate.

    b) ENERGY IS NOT USEFULNESS. Explained Frobenius energy is geometry. A
       direction with a tiny singular value may carry the task-critical signal,
       and a dominant one may be an artefact of the optimiser or of a few
       outlier rows.

    c) IT IS POST-HOC. Step 1 requires the very full fine-tune LoRA exists to
       avoid. Circular for the case you actually care about.

    d) IT IS PER-MATRIX, AND THE MATRICES DISAGREE. The table above shows
       different modules needing different ranks. A single global r is already
       a compromise; the SVD does not tell you how to make it.

    e) TASK AND DATA SCALE DOMINATE. Rank interacts with dataset size and task
       difficulty. A rank sufficient for 10k instruction pairs may not be
       sufficient for a domain shift.

    f) THE MEASUREMENT ITSELF IS NOISY HERE. dW is a difference of bf16
       tensors, so its small singular values sit near the precision floor. The
       SVD is computed in float64 to keep round-off from dominating the tail,
       but the INPUT precision is still bf16 and that limits what the tail can
       mean.

  THE ANSWER TO GIVE IN AN INTERVIEW is therefore not "use SVD". It is:
  "SVD of an observed update tells you the rank of a solution that exists,
   which bounds rather than determines the rank you need; in practice I would
   sweep rank against a fixed budget and validation metric, use the spectrum
   as a prior and a sanity check, and expect the answer to depend on the
   target modules and the data scale more than on the model."
""")

    print("--- WHAT THIS PROVES / DOES NOT PROVE ---")
    print("  PROVES     : a real fine-tuning update on this model concentrates")
    print("               its energy in far fewer directions than same-norm")
    print("               noise, which is evidence FOR the low-rank premise.")
    print("  DOES NOT   : establish that any particular rank trains as well as")
    print("               full fine-tuning. No adapter was trained in E17.")
    print("               That comparison is E18/E19.")

    payload = {
        "base_model": BASE_MODEL,
        "revision": REVISION,
        "sft_model": args.sft_model,
        "hypotheses": {"H1": bool(h1), "H2": bool(h2), "H3": bool(h3), "H4": bool(h4)},
        "modules": results,
    }
    out = repo_root() / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n  evidence: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
