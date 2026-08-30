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

OUTCOMES, recorded after the first run. The hypotheses above are preserved
exactly as written beforehand.

    H1  HOLDS, but only weakly. dW does concentrate faster than noise - yet
        the margin is small: on layer 13 q_proj, 90% of dW's energy needs rank
        730 against the noise control's 783. A 7% difference, not an order of
        magnitude.

    H2  **DISPROVED, and this is the most important result in E17.** r=64
        captures 22.95% of q_proj's energy, not the >=50% predicted. Reaching
        90% needs rank ~730 of a maximum 1536 - roughly HALF of full rank. At
        the rank people actually use, r=16, the truncated SVD reconstructs
        just 10.69% of the update's energy, with relative reconstruction error
        0.945.

    H3  HOLDS. Square attention projections concentrate relatively better
        (r@90% / max_rank ~ 0.41-0.48) than the narrow GQA KV projections
        (~0.65-0.77) or the MLP matrices (~0.77-0.78).

    H4  HOLDS strongly, and more so than expected: every ||dW|| / ||W|| is
        between 0.0013 and 0.0050. One epoch of SFT moves the weights by
        roughly a quarter of one percent.

WHAT H2's FAILURE ACTUALLY MEANS - the central lesson of this experiment.

It is tempting to read "dW is not low-rank" as "LoRA should not work". That
inference is wrong, and seeing why is the whole point.

    LoRA's premise is NOT "the dW that full fine-tuning produces is low-rank".
    It is "there EXISTS a low-rank dW achieving comparable task performance".

Those are different claims, and only the second is what LoRA needs. Full
fine-tuning's update is nearly full-rank because nothing in gradient descent
pushes it toward low rank - every direction that reduces the loss even
slightly gets some update, so the solution spreads across the whole spectrum.
Its rank is an artefact of the optimisation being unconstrained, not a
measurement of how much rank the task requires.

So the naive procedure - "SVD the update, read off the rank" - is not merely
imprecise. Applied to a real dW it gives ~730, which nobody uses, and which
would cost more parameters than the dense update it approximates. The
measurement below is evidence that the procedure is WRONG, and that is more
useful than a tidy confirmation would have been.

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
  START FROM WHAT WAS MEASURED, NOT FROM THE STORY.

  The naive procedure - "SVD the update, read off the rank at 90% energy" -
  returns ~730 here. Nobody uses rank 730: at that rank a LoRA adapter costs
  730 * (1536 + 1536) = 2,242,560 parameters against the dense update's
  1536 * 1536 = 2,359,296. It saves 4.9%. The break-even rank for a square
  1536x1536 matrix is 768 - above that a 'low-rank' factorisation costs MORE
  than the dense update it approximates, and 730 is just under it. The
  procedure, applied honestly to a real dW, recommends something absurd.

  So the first thing the spectrum tells you is that the naive reading is wrong.

  WHY IT IS WRONG. LoRA does not require that full fine-tuning's dW be
  low-rank. It requires that SOME low-rank dW does the job. Full fine-tuning's
  update is nearly full-rank because unconstrained gradient descent has no
  reason to be anything else - every loss-reducing direction gets some update.
  Rank ~730 measures how unconstrained the optimiser was, not how much rank
  the task needs.

  WHAT THE SPECTRUM IS STILL GOOD FOR:

    1. AS A CONTROLLED COMPARISON. dW concentrating faster than a same-norm
       random matrix (730 vs 783 at 90%) is real evidence of structure, even
       though the margin is modest. Without the control you would not know
       whether ANY decay was meaningful.
    2. AS A RELATIVE RANKING ACROSS MODULES. Square attention projections
       concentrate relatively better than MLP matrices here, which is a
       reason to prefer attention targets when the budget is tight - and it
       matches what the LoRA paper chose to adapt.
    3. AS A SANITY CHECK. A dW whose spectrum looked exactly like noise would
       be a warning that the fine-tune learned nothing structured.

  WHAT ACTUALLY DETERMINES THE RANK YOU SHOULD USE:

    a) A SWEEP AGAINST A VALIDATION METRIC. This is the honest answer. Rank is
       a hyperparameter; treat it as one. Cost is linear in r and tiny
       (r=16 attention-only is 0.28% of this model), so the sweep is cheap.
    b) THE TARGET MODULE SET, which matters more than r in the QLoRA authors'
       reported experience. Adapting all linear layers at low rank often beats
       adapting attention only at high rank, for the same budget.
    c) DATA SCALE AND TASK DISTANCE. A rank sufficient for 10k in-domain
       instruction pairs need not suffice for a domain shift.
    d) YOUR MEMORY BUDGET, which is a legitimate first constraint given (a).

  WHY THE SVD READING IS A HEURISTIC AND NOT A RULE:

    a) IT MEASURES A SOLUTION, NOT A REQUIREMENT - the point above.
    b) ENERGY IS NOT USEFULNESS. Explained Frobenius energy is geometry. A
       direction with a tiny singular value may carry the task-critical
       signal; a dominant one may be an optimiser artefact or a few outlier
       rows.
    c) IT IS POST-HOC AND CIRCULAR. It needs the very full fine-tune that
       LoRA exists to avoid.
    d) THE MATRICES DISAGREE. Different modules want different ranks; a single
       global r is already a compromise the SVD cannot resolve.
    e) THE MEASUREMENT IS PRECISION-LIMITED. dW is a difference of bf16
       tensors, so its small singular values sit near the precision floor. The
       SVD runs in float64 to keep round-off out of the tail, but the INPUT is
       bf16 and that bounds what the tail can mean.

  THE ANSWER TO GIVE IN AN INTERVIEW:

  "I would not choose it from an SVD. I measured this: on a real full
   fine-tune of Qwen2.5-1.5B, dW needs rank ~730 of 1536 for 90% of its
   energy, and r=16 captures under 11%. Yet r=16 adapters work. That tells you
   LoRA's premise is not 'the full-FT update is low-rank' - it is 'a low-rank
   update suffices', which is a different and weaker claim. So I treat rank as
   a hyperparameter and sweep it against validation loss under a fixed budget,
   spend my first extra parameters on WIDENING the target module set rather
   than raising r, and use the spectrum only as a sanity check and a relative
   ranking across module types."
""")

    print("--- WHAT THIS PROVES / DOES NOT PROVE ---")
    print("  PROVES     : a real fine-tuning update on this model is only")
    print("               MODESTLY more concentrated than same-norm noise, and")
    print("               is nowhere near low-rank: ~730 of 1536 for 90% energy.")
    print("               The naive 'SVD the update and read off r' procedure")
    print("               therefore recommends a rank nobody would use.")
    print("  DOES NOT   : show that LoRA fails. LoRA needs a low-rank update to")
    print("               SUFFICE, not full fine-tuning's update to BE low-rank.")
    print("               Whether r=16 trains competitively is E18/E19, where an")
    print("               adapter is actually trained. No adapter was trained here.")

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
