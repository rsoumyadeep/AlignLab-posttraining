# AlignLab — Experiment Registry

Every numbered experiment run in Phases 2–7, with its hypothesis **as stated
before the result was known**, what happened, and where the evidence lives.

**Hypotheses are recorded as they were written, including the ones that turned
out wrong.** Four were disproved and one was hidden by a badly-chosen criterion.
Those rows are the most useful in the table and none of them has been quietly
edited to match the outcome.

Verdict vocabulary: **HOLDS** · **DISPROVED** · **NOT CONFIRMED** (consistent
with the evidence but not decisively tested) · **MEASURED** (a measurement with
no prior hypothesis) · **UNTESTABLE AS STATED** (the criterion could not decide
the question).

**A naming inconsistency, recorded rather than tidied away.** Phases 2–6 number
experiments E1–E23; Phases 6 and 7 then stopped assigning numbers and identified
work by report section and run name instead. So the β sweep, the post-hoc
diagnostic and the Phase 7 evaluation pass have **no E-number**, and this
registry does not invent one for them — it uses their real run names. An earlier
draft of this file did invent "E24" and "E25"; they never existed in the
repository and were removed.

---

## Index

| ID | Phase | Subject | Verdict |
|---|---|---|---|
| [E1](#e1) | 2 | attention scaling by √d_k | HOLDS |
| [E2](#e2) | 2 | causal-mask leakage | HOLDS (33.2×) |
| [E3](#e3) | 2 | three-implementation agreement | HOLDS |
| [E4](#e4) | 2 | GQA/MQA cost | HOLDS (CPU), **not on GPU** |
| [E5](#e5) | 2 | RoPE relative-position invariance | HOLDS |
| [E6](#e6) | 2 | RoPE offset guard | HOLDS |
| [E7](#e7) | 2 | RMSNorm vs LayerNorm | HOLDS |
| [E8](#e8) | 2 | KV cache speedup | HOLDS (CPU), **overhead-bound on GPU** |
| [E9](#e9) | 2 | decoding strategies | MEASURED + **a real top-p bug** |
| [E10](#e10) | 2 | Flash vs math attention | HOLDS (56×, 147×) |
| [E11](#e11) | 2/3 | Qwen parameter arithmetic | **DISPROVED, then corrected** |
| [E12](#e12) | 3 | loss masking on the real model | H1–H4 HOLD · **H5 DISPROVED** |
| [E13](#e13) | 3 | SFT before/after | MEASURED |
| [E14](#e14) | 3 | control-region decomposition | MEASURED |
| [E15](#e15) | 4 | bitsandbytes environment | HOLDS |
| [E16](#e16) | 4 | LoRA target modules | MEASURED |
| [E17](#e17) | 4 | SVD rank selection | **H2 DISPROVED** |
| [E18](#e18) | 4 | our LoRA vs `peft` | HOLDS (fp32) |
| [E19](#e19) | 4 | full SFT vs LoRA vs QLoRA | H1,H2,H4 HOLD · **H3 DISPROVED** |
| [E20](#e20) | 4 | why PEFT never stops | **NOT CONFIRMED** |
| [E21](#e21) | 5 | preference data readiness | H1–H4 HOLD |
| [E22](#e22) | 5 | reference model, KL, baseline | H1,H2,H3,H5 HOLD · **H4 DISPROVED** |
| [beta-sweep](#beta-sweep) | 6 | DPO β sweep (preregistered) | **NULL RESULT** |
| [E23](#e23) | 6 | SFT vs DPO evaluation | **NULL RESULT** |
| [diag-lr](#diag-lr) | 6 | post-hoc 10× learning rate | POST-HOC |
| [eval-full-001](#eval-full-001) | 7 | full evaluation pass | MEASURED |

---

## Phase 2 — Transformer components

<a id="e1"></a>
### E1 — Does attention need the 1/√d_k scale?
- **Why:** the scale is universally quoted; the reason is usually asserted.
- **Hypothesis:** without scaling, logit variance grows with d_k, softmax
  saturates, and gradients vanish.
- **Config:** d_k swept 4 → 1024 (256×), scaled and unscaled arms.
- **Result:** unscaled — logit std **1.85 → 31.50** (tracking √d_k), max prob
  0.42 → **0.992**, entropy 1.78 → **0.028**, Jacobian mass **0.708 → 0.014**.
  Scaled — all four flat.
- **Learned:** saturation is measurable, and the vanishing gradient is visible
  in the Jacobian rather than inferred.
- **Verdict:** **HOLDS** · Evidence: `docs/phase2/PHASE_2_REPORT.md` §3

<a id="e2"></a>
### E2 — What does the causal mask actually buy?
- **Hypothesis:** removing it lets the model see future tokens, so training
  loss collapses without any real learning.
- **Result:** train loss **0.2907 masked vs 0.0088 unmasked — 33.2× lower when
  leaking.**
- **Learned:** the strongest single Phase 2 result, and the origin of a theme
  the whole project inherited: **a dramatically better loss can mean a broken
  experiment.**
- **Verdict:** **HOLDS** · Evidence: `docs/phase2/PHASE_2_REPORT.md`

<a id="e3"></a>
### E3 — Three independent attention implementations
- **Hypothesis:** PyTorch, NumPy and pure-Python implementations agree to
  floating-point tolerance; agreement is evidence of correctness.
- **Result:** torch–numpy **3.331e-16** · pure–numpy **1.110e-16** · torch–pure
  **4.441e-16** (tol 1e-12). Versus `F.scaled_dot_product_attention`:
  **3.576e-07** (fp32, tol 1e-5).
- **Verdict:** **HOLDS**

<a id="e4"></a>
### E4 — What do GQA and MQA cost and save?
- **Hypothesis:** reducing KV heads shrinks the KV cache proportionally and
  reduces latency.
- **Result:** H_kv 16 → 1: params 100% → 53.12%, **KV cache 100% → 6.25%**,
  CPU latency 160.4 → 117.4 ms. **On GPU, latency was flat (0.86 → 0.81 ms).**
- **Learned:** the memory saving is real and architectural; the latency saving
  is hardware-dependent and did **not** appear on the A6000 at this size.
- **Verdict:** **HOLDS on CPU, NOT on GPU** — recorded as measured, not
  generalised.

<a id="e5"></a>
### E5 — Does RoPE depend only on relative position?
- **Hypothesis:** for a fixed gap, the attention logit is invariant to absolute
  position.
- **Result:** spread **2.384e-07** across positions 0–100.
- **Verdict:** **HOLDS**

<a id="e6"></a>
### E6 — The RoPE offset guard
- **Hypothesis:** if the position offset is wired correctly, forcing it to 0
  during cached decoding must change the output. (A guard against a test that
  would pass on broken code.)
- **Result:** forcing offset 0 **does** change results.
- **Verdict:** **HOLDS** — the guard is live, not decorative.

<a id="e7"></a>
### E7 — RMSNorm vs LayerNorm
- **Hypothesis:** dropping the mean-centring makes RMSNorm cheaper, and the
  activations are not in fact zero-mean.
- **Result:** **0.28× the cost** on CPU, **0.17–0.22×** on GPU. Centring check:
  LayerNorm mean −0.000000, RMSNorm mean **+0.995010**.
- **Learned:** RMSNorm is not "LayerNorm without a redundant step" — the mean it
  declines to remove is large.
- **Verdict:** **HOLDS**

<a id="e8"></a>
### E8 — Does the KV cache do what it claims?
- **Hypothesis:** without a cache, per-token cost grows with sequence length;
  with one, it is flat.
- **Result:** CPU — correctness 8.345e-07; no-cache **6.16 → 15.52 ms/token**;
  cached **flat ~5.4 ms**; speedup **2.96×**. GPU — **1.05×**.
- **Learned:** the GPU result is **overhead-bound at this model size**, not a
  refutation. Reported as measured rather than explained away.
- **Verdict:** **HOLDS on CPU; overhead-bound on GPU**

<a id="e9"></a>
### E9 — Five decoding strategies
- **Why:** to measure the diversity/quality trade-off rather than assert it.
- **Result:** greedy distinct-4 0.257 / grammatical 0.750 / logprob −18.71 ·
  top-p 0.9 → 0.566 / **0.863** / −23.30 · temperature 2.0 → 0.667 / **0.323** /
  −80.67.
- **Learned:** **this experiment found a real bug in our own top-p
  implementation**, which is why it is in the registry as more than a
  measurement.
- **Verdict:** **MEASURED** + defect found

<a id="e10"></a>
### E10 — Flash attention vs the math path
- **Hypothesis:** Flash attention is faster and its memory is linear rather
  than quadratic in sequence length.
- **Result:** GPU, T=4096, bf16 — MATH **82.397 ms / 9568.3 MiB**, FLASH
  **1.478 ms / 65.0 MiB** → **56× faster, 147× less memory**. Numerical
  difference 1.562e-02 = **2× bf16 epsilon**. MiB per token: MATH 0.369 → 2.336
  (quadratic), FLASH **0.0159 constant** (linear).
- **Verdict:** **HOLDS**

<a id="e11"></a>
### E11 — Reconciling our arithmetic with the real Qwen2.5-1.5B
- **Hypothesis (Phase 2):** our parameter formula reproduces the published
  1.54B / 1.31B counts.
- **Result:** **DISPROVED first.** Phase 2's count was wrong by exactly
  **57,344 = 28 × (1536 + 256 + 256)** — the QKV **biases**, which Qwen has and
  our formula omitted. Corrected formula reproduces the checkpoint exactly.
- **Learned:** 6 of 7 architectural predictions confirmed; the one that failed
  failed for a specific, findable reason.
- **Verdict:** **DISPROVED, then corrected.** The original prediction is
  preserved in `docs/phase2/`; it was not edited to match.

---

## Phase 3 — Supervised fine-tuning

<a id="e12"></a>
### E12 — Loss masking on the real model
- **Hypotheses:** H1 masked and unmasked losses differ · H2 the TRL boundary
  agrees with an independent computation · H3 no example is fully masked ·
  H4 the decomposition reproduces the reported loss · **H5 the prompt region is
  *easier* than the completion region.**
- **Result:** H1–H4 **HOLD** (decomposition agreed to **1.6e-07** after a fix).
  **H5 DISPROVED:** prompt CE **6.3314** vs completion **3.6295** — the prompt
  is *harder*, because the base model had never seen ChatML.
- **Learned:** an unmasked loss is not reliably higher *or* lower. It is a mean
  over a **different token population**, so it is not comparable in either
  direction. This became `PerplexityResult.region` in Phase 7.
- **Verdict:** **H1–H4 HOLD · H5 DISPROVED** · Evidence:
  `docs/phase3/PHASE_3_REPORT.md`
- **Defects found:** our own decomposition mis-counted contributing positions
  (`labels[:,1:]`, not `labels`); a stale `datasets` cache served pre-fix rows
  and the fingerprint change proved the fix landed.

<a id="e13"></a>
### E13 — SFT before/after, 200 held-out examples
- **Result:** completion perplexity **8.541 → 7.199 (−15.7%)** over 38,831
  completion tokens; stop-token emission **0/4 → 4/4**.
- **Verdict:** **MEASURED** · Evidence: `docs/phase3/e13_before_after.json`

<a id="e14"></a>
### E14 — Decomposing the control region, 100 examples
- **Verdict:** **MEASURED** · Evidence: `docs/phase3/PHASE_3_REPORT.md` §E14

---

## Phase 4 — PEFT

<a id="e15"></a>
### E15 — Does bitsandbytes actually work here?
- **Why:** an import succeeding proves nothing about a CUDA kernel running.
- **Result:** **verified by execution, not by import.** Related finding:
  `triton` installed but unimportable (missing `setuptools`).
- **Verdict:** **HOLDS**

<a id="e16"></a>
### E16 — Which modules does LoRA target?
- **Approach:** inspected the actual module tree rather than assuming the
  conventional `q_proj,k_proj,v_proj,o_proj` set applies.
- **Verdict:** **MEASURED** · Evidence: `docs/phase4/PHASE_4_REPORT.md` §4

<a id="e17"></a>
### E17 — SVD rank selection (Tier-1 deliverable)
- **Hypotheses:** H1 explained energy uses **squared** singular values ·
  **H2 the real full-fine-tuning update ΔW is approximately low-rank, which is
  why LoRA works.**
- **Result:** H1 holds. **H2 DISPROVED — the measured ΔW is *not* low-rank.**
- **Learned:** **this is the most important negative result in the project.**
  The popular explanation of LoRA ("fine-tuning updates are intrinsically low
  rank, so a low-rank parameterisation suffices") is *not* supported by the
  update actually measured here. LoRA still worked; the usual story for *why*
  did not survive contact with the data.
- **Verdict:** **H2 DISPROVED** · Evidence: `docs/phase4/PHASE_4_REPORT.md` §5
- **Consequence:** no AlignLab document may claim "LoRA works because updates
  are low rank."

<a id="e18"></a>
### E18 — Our LoRA against `peft`
- **Hypothesis:** our first-principles implementation matches the library, and
  merge/unmerge round-trips exactly.
- **Result:** **HOLDS in float32.** What did **not** transfer is that it holds
  in the dtype these models are served in — see E19.
- **Verdict:** **HOLDS (fp32)**

<a id="e19"></a>
### E19 — Full SFT vs LoRA vs QLoRA
- **Hypotheses:** H1 PEFT trains far fewer parameters · H2 QLoRA uses less
  memory than LoRA · **H3 LoRA matches full fine-tuning on held-out loss** ·
  H4 the dataset is controlled across arms.
- **Result:** H1, H2, H4 **HOLD**. **H3 DISPROVED** — LoRA did not match full
  fine-tuning. Perplexities: full SFT **7.199**, LoRA@2e-4 **7.449**,
  LoRA@2e-5 **7.624**, QLoRA@2e-4 **7.511** (38,831 completion tokens).
  Stop-token emission: full SFT **4/4**, every PEFT arm **0/4**.
- **Learned:** the learning-rate confound was measured rather than assumed —
  the LR effect (0.032 nats) against a LoRA-vs-full gap of 0.051 nats means
  reporting only the LR-matched arm would have **overstated LoRA's cost by
  about half**.
- **Also found:** **merging a LoRA adapter in bf16 is lossy** — merged-vs-unmerged
  logits differ by **6.875e-01** (mean 6.103e-02) in bf16 versus **6.330e-05**
  in fp32, **12,460× worse**, because ‖ΔW‖/‖W‖ ≈ 0.003 sits at the resolution of
  bf16's mantissa. E19 was changed to evaluate **unmerged** so a merge artefact
  is not attributed to a training arm.
- **Verdict:** **H1, H2, H4 HOLD · H3 DISPROVED**

<a id="e20"></a>
### E20 — Why do the PEFT models never stop?
- **Hypothesis:** LoRA's capability ceiling is set by **which matrices it can
  reach**, not only by rank. Emitting `<|im_end|>` requires moving that token's
  logit, produced by `lm_head`, which is tied to the embedding and therefore
  outside LoRA's target set. Full fine-tuning moved that matrix by relative
  **0.0136**, the largest relative change of any matrix measured.
- **Verdict:** **NOT CONFIRMED.** Consistent with the hypothesis; not proof.
  The decisive test — a LoRA variant adapting the MLP or an untied head — is
  one ~18-minute run and is **deferred, not skipped.**
- **This remains the single most valuable outstanding experiment in the
  project.**

---

## Phase 5 — Preference learning

<a id="e21"></a>
### E21 — Is the preference data usable?
- **Hypotheses:** H1 chosen and rejected share a prompt · H2 the token prefix
  matches for both responses · H3 there is a length bias in tokens ·
  H4 the loading is reproducible.
- **Result:** **all four HOLD.** 11.9% ties; chosen responses **56.5% longer**
  in tokens.
- **Verdict:** **HOLDS — data READY for Phase 6**

<a id="e22"></a>
### E22 — Reference model, KL, and the pre-DPO baseline
- **Hypotheses:** H1 KL(SFT‖base) > 0 · H2 KL(LoRA) < KL(SFT) · H3 the implicit
  reward is exactly 0 when π = π_ref · **H4 the SFT model already prefers the
  chosen response more than half the time by SUM** · H5 MEAN favours chosen
  more than SUM.
- **Result:** H1 **HOLDS** (with a caveat the criterion hid) · H2 **HOLDS**
  (0.6233 < 0.6547) · H3 **HOLDS EXACTLY** (`0.000e+00`) · **H4 DISPROVED —
  47.2%, below chance** · H5 **HOLDS** (58.3% vs 47.2%).
- **Learned:** H4's failure is the most consequential Phase 5 finding and it
  reframes all of Phase 6: **the metric DPO optimises did not start above
  chance.** An 11.1-point swing between SUM and MEAN on identical models and
  data.
- **Verdict:** **H4 DISPROVED**
- **Method note:** a claimed non-negativity property of the k3 KL estimator was
  **falsified by executing it** — 11 negatives in 200,000 float32 samples
  (min −2.98e-08), zero in float64. Mathematically non-negative; floating-point
  cancellation in practice.
- **Criterion defect:** E22's H1 criterion `0 < mean < 1` could not test the
  word "small" it was meant to test — recorded as **UNTESTABLE AS STATED**.

---

## Phase 6 — DPO

<a id="beta-sweep"></a>
### The preregistered β sweep — runs `dpo-beta0.01-sum`, `dpo-beta0.1-sum`, `dpo-beta0.5-sum`
- **Preregistration:** `docs/phase6/BETA_PREREGISTRATION.md`, committed
  **13:15:12, before any DPO code existed.** Fixed β ∈ {0.01, 0.1, 0.5}, SUM as
  the primary objective, MEAN diagnostic, five hypotheses, and the criteria for
  declaring a negative result — all written down in advance so β could not be
  chosen after seeing results.
- **Result:** **NULL.** Under the preregistered configuration and training
  budget, **DPO did not change SUM preference accuracy on the evaluation set**
  (86/184 before and after).
- **Measured explanation, not an excuse:** the SUM metric carries a **~29-nat
  length gap**; flipping the average pair requires shifting it by ~29 nats, and
  the run achieved **0.04** — 728× short. KL from the reference was **0.0008**
  against SFT's 0.2044, roughly **250× under-budget**.
- **Verdict:** **NULL RESULT — preserved as the primary conclusion.**
- **This must not be restated as "DPO does not work."** It is a statement about
  this configuration and this budget.
- **Independently replicated in `eval-full-001`** on a different evaluation path.

<a id="e23"></a>
### E23 — SFT vs DPO evaluation
- **The one numbered Phase 6 experiment**, planned in Phase 5 as the comparison
  that would decide whether DPO changed anything.
- **Result:** SUM preference accuracy **0.4674 = 86/184 for the SFT baseline
  and for every DPO arm** — byte-identical, not merely close:

  | arm | SUM | MEAN | KL from reference |
  |---|---|---|---:|
  | SFT (baseline) | 0.4674 (86/184) | 0.5870 | 0.000000 |
  | DPO β=0.01 | 0.4674 (86/184) | 0.5870 | 0.000801 |
  | DPO β=0.1 | 0.4674 (86/184) | 0.5870 | 0.000799 |
  | DPO β=0.5 | 0.4674 (86/184) | 0.5870 | 0.000803 |

  Wilson intervals overlap completely; the difference is **NOT resolvable at
  this sample size**, and the report says exactly that rather than "+0.0%".
- **Verdict:** **NULL RESULT** · Evidence: `docs/phase6/e23_dpo_evaluation.json`

<a id="diag-lr"></a>
### Post-hoc 10× learning rate — run `diag-dpo-beta0.1-lr5e-6`
- **Status:** **POST-HOC.** Run after seeing the sweep's result, and therefore not
  covered by the preregistration.
- **Result:** shifted the gap by **0.70 nats** — still **42× short**. Generated
  output ~**30% longer**. KL from the reference **0.002365** (versus ~0.0008 for
  the preregistered arms — still ~86× under SFT's 0.2044).
- **SUM accuracy: 86/184 (0.4674) — unchanged, exactly as in every other arm.**
  MEAN moved 0.5870 → **0.5924**, one pair's worth.
- **Verdict:** **POST-HOC — does not replace or overwrite the preregistered
  result.**

---

## Phase 7 — Evaluation

<a id="eval-full-001"></a>
### The full evaluation pass — run `eval-full-001`
- **Scope:** five models (base, SFT, LoRA r=16, QLoRA r=16, DPO β=0.1) on one
  pass: two perplexity regions, preference SUM and MEAN with length
  attribution, 6 generation prompts, and a two-order LLM judge.
- **Results:** **of 24 pairwise comparisons, 3 were resolvable — all
  stop-token rates.**
  - SFT: perplexity **8.824 → 7.398 (−16.2%)**, stop **0/6 → 6/6** (RESOLVED),
    distinct-2 0.523 → 0.839.
  - LoRA/QLoRA: **0/6 stop**, cap hit on every prompt, and past the answer they
    **fall back into the chat template**, emitting `You are a helpful
    assistant.` repeatedly.
  - DPO: perplexity moved in the **4th decimal**, generation differs on 4 of 6
    prompts, and **no aggregate metric changed** — replicating the Phase 6
    null on a different evaluation path.
  - **Length attribution on all five models: every one prefers the chosen
    response per token, and every one's SUM comparison inverts that verdict.
    Positive residual in all five rows, including the untrained base model.**
    The inversion is a property of the **metric**, not of any training stage.
  - Judge: position bias **33.3%** on base-vs-SFT; SFT won 3 of 4 *decided*
    pairs, **[0.301, 0.954] — not resolvable**. An accidental positive control
    passed: the judge returned `TIE` on two byte-identical pairs in both orders,
    unprompted.
- **Verdict:** **MEASURED** · Evidence: `docs/phase7/EVAL_RESULTS.md`,
  `eval-full-001_dashboard.{json,txt}`, `eval-full-001_run.log.txt`
- **Defects found:** `dataset_fingerprint` null in all five provenance records;
  `configure_hf_cache` a no-op in every entrypoint, costing 18.34 GB of
  duplicate weights.

---

## What the disproved hypotheses cost, and bought

| Hypothesis | Believed | Measured | Consequence |
|---|---|---|---|
| E11 | our parameter formula was right | off by 57,344 (QKV biases) | formula corrected; prediction preserved |
| E12 H5 | prompt region is easier | **harder** (6.3314 vs 3.6295) | `PerplexityResult.region` exists because of this |
| E17 H2 | ΔW is low-rank — "why LoRA works" | **not low-rank** | the project may not use that explanation |
| E19 H3 | LoRA matches full fine-tuning | it did not | LR confound measured, both arms reported |
| E22 H4 | SFT prefers chosen >50% by SUM | **47.2%, below chance** | reframes all of Phase 6 |

**Five disproved hypotheses across six phases.** None was rewritten after the
fact; each is preserved with its original wording in the phase report that
produced it.

**Related:** [[phase2-report]] · [[phase3-report]] · [[phase4-report]] ·
[[phase5-report]] · [[phase6-report]] · [[phase7-report]] · [[project-narrative]]
