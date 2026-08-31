# AlignLab — CV / Resume Description

A guide for presenting AlignLab on a CV and defending it in interviews.
Every claim below is grounded in the repository; nothing here is new work.

---

## 1. Recommended One-Line Description

> **AlignLab** — an end-to-end **LLM post-training** pipeline for `Qwen2.5-1.5B`
> (**SFT → LoRA/QLoRA → DPO → evaluation**) with **Transformer internals, LoRA
> and the DPO objective implemented from first principles** and verified against
> reference implementations, plus a preregistered evaluation subsystem that
> reports **null and disproved results** as first-class findings.

---

## 2. Recommended CV Entry

**AlignLab — LLM Post-Training & Alignment Framework** · *PyTorch, Hugging Face,
Hydra, bitsandbytes* · [github.com/rsoumyadeep/AlignLab-posttraining](https://github.com/rsoumyadeep/AlignLab-posttraining)

- Implemented Transformer internals from scratch — **multi-head / grouped-query
  attention** (three independent implementations cross-verified against
  `F.scaled_dot_product_attention`), **RoPE**, **RMSNorm**, **SwiGLU**, **KV
  caching** and five decoding strategies; measured **56× speed / 147× memory**
  advantage of Flash vs. math attention at T=4096 and a KV-cache footprint drop
  to **6.25%** under GQA.
- Ran **completion-only supervised fine-tuning** of `Qwen/Qwen2.5-1.5B` (9,499
  instruction examples) behind three pre-flight audits that abort the run on a
  loss-mask mismatch: completion **perplexity 8.824 → 7.398 (−16.2%)** and
  stop-token emission **0/6 → 6/6**.
- Built **LoRA and QLoRA** from the update rule `W' = W + (α/r)·BA` and validated
  against `peft`: **4.36M trainable parameters (0.28% of the model)** and
  **4-bit NF4** quantization cutting peak VRAM **5.07 GB → 3.10 GB** for a
  **3.6%** perplexity cost; showed adapter merging must be done in **fp32**
  (**12,460×** lower merge error than bf16).
- Studied rank selection with **SVD / Eckart–Young**, explained-energy on
  **squared singular values** and the **PCA↔SVD** relationship, then measured the
  real full-fine-tuning **ΔW and disproved the hypothesis that it is
  approximately low-rank** — identifying *which matrices an adapter can reach*
  (a **tied `lm_head`** outside the LoRA target set) as the stronger explanation
  of the observed PEFT ceiling.
- Derived and implemented the **DPO** objective from the KL-constrained RLHF
  optimum with verified invariants (implicit reward **exactly 0** and loss
  **exactly ln 2** at initialisation); a **preregistered** β ∈ {0.01, 0.1, 0.5}
  sweep returned a **characterised null** — quantified as **KL 0.0008 vs SFT's
  0.2044 (~250× under-budget)** rather than reported as a failure.
- Designed a **14-metric evaluation suite with no aggregate score**: pure/impure
  separation, **Wilson intervals** on every rate, an explicit *"not resolvable at
  this sample size"* verdict (**3 of 24** comparisons resolvable), and an
  **LLM-as-judge** run in both presentation orders that exposed **33.3% position
  bias**; showed a summed-log-prob preference metric was tracking **token count**
  (chosen responses **56.5%** longer; **SUM 47.2% vs MEAN 58.3%**).
- Engineered for reproducibility: **Hydra structured configs**, per-run
  manifests (git SHA + dirty flag, config hash, dataset fingerprint, library
  versions, seed), a mechanical **provenance audit**, storage pre-flight,
  SIGUSR1 preemption handling, and a **607-test CPU suite** (3 skipped: CUDA, SIGUSR1, W&B) requiring no network
  or credentials.

*(Pick 3–5 of these for a real CV — see §7.)*

---

## 3. Interview-Catchy Keywords

**Multi-Head Attention** · **GQA/MQA** · **RoPE** · **ALiBi** · **RMSNorm** ·
**SwiGLU** · **KV Caching** · **Flash / SDPA attention** · **Causal masking** ·
**Decoding strategies** · **SFT** · **Completion-only loss masking** ·
**LoRA** · **QLoRA** · **NF4 Quantization** · **PEFT** · **SVD** ·
**Eckart–Young** · **PCA** · **Bradley–Terry preference model** ·
**RLHF (conceptual)** · **DPO** · **KL divergence & the k3 estimator** ·
**Reference model** · **LLM Evaluation** · **LLM-as-a-Judge** ·
**Position-bias control** · **Wilson confidence intervals** ·
**Preregistration** · **PyTorch** · **Hydra** · **Reproducible Experiments** ·
**Run provenance / manifests**

---

## 4. Claims I Can Defend

- I implemented **attention, RoPE, RMSNorm, SwiGLU, the KV cache and a
  decoder-only Transformer from first principles**, and verified them against
  PyTorch's own kernel and two independent re-implementations.
- I ran **full-parameter SFT** on a 1.5B base model and measured a **behavioural**
  change — stop-token emission **0/6 → 6/6** — alongside a **−16.2%**
  in-distribution completion perplexity.
- I implemented **LoRA from the maths** (B zero-initialised so the adapted model
  starts exactly equal to the base) and checked it against `peft`; and I know
  **why the merge must happen in fp32** — because ‖ΔW‖/‖W‖ ≈ 0.003 sits at bf16's
  mantissa resolution.
- I can explain **QLoRA** correctly: **NF4 is a storage format, not an arithmetic
  one**, and I measured what that buys (~**61%** of LoRA's peak VRAM at the same
  4,358,144 trainable parameters).
- I used **SVD/PCA properly** for rank analysis — including the break-even rank
  `r* = (d_in·d_out)/(d_in+d_out)` — and I **measured** the real ΔW instead of
  assuming the textbook story about it.
- I derived the **DPO loss from the KL-constrained RLHF objective** (where
  `log Z(x)` cancels) and implemented it with `F.logsigmoid`, plus my own
  training loop and pre-training invariant checks.
- I can state **why DPO rather than PPO** for this budget — no reward model, no
  rollout loop, no value network — while being explicit that **PPO was studied
  conceptually and not implemented**.
- I **preregistered** the β sweep before writing DPO code, so the hyperparameter
  could not be chosen after seeing results.
- I designed an evaluation that **refuses to over-claim**: no aggregate score,
  intervals on every rate, both judge orders, and length attribution — and it
  caught a preference metric that was measuring **token count**.
- I can defend **five disproved hypotheses**, including one about my own
  parameter arithmetic, all recorded in their original wording.

---

## 5. Claims I Should NOT Make

| ❌ Don't say | ✅ Say instead |
|---|---|
| "Trained an LLM from scratch" | "Built the **post-training** pipeline on a pretrained `Qwen2.5-1.5B`; pretraining is explicitly out of scope." |
| "DPO improved the model" | "A **preregistered** DPO sweep returned a **null**, which I characterised: KL 0.0008 vs SFT's 0.2044 — ~**250× under-budget**." |
| "LoRA works because the fine-tuning update is low-rank" | "I **measured** ΔW and it was **not** low-rank; the reachable target set — with `lm_head` tied to the embedding — is the stronger hypothesis (**not confirmed**; the decisive run is deferred)." |
| "Production-ready alignment framework" | "A research pipeline: one model, one size, single seeds, one epoch — **no claim of production model quality**." |
| "Implemented RLHF with PPO" | "**PPO studied conceptually**, not implemented; DPO is what I implemented and ran." |
| "The judge showed my model is better" | "The judge was **33.3% position-inconsistent** and could not resolve the comparison — I report that rather than the win rate." |
| "SFT made a better assistant" | "SFT substantially changed **termination behaviour** and in-distribution perplexity; broader quality was not established." |
| "LoRA matched full fine-tuning" | "LoRA cost only **3.6%** perplexity but emitted **0/6** stop tokens — the likelihood metric and usability disagreed." |
| "Benchmarked on standard suites" | "Bespoke, narrow, **in-distribution** evaluation — no MMLU/HELM/AlpacaEval." |

---

## 6. Recommended Interview Framing

> "Most fine-tuning projects call `TRL` and report a loss curve. AlignLab is
> about *understanding* the post-training pipeline, so I implemented the load-
> bearing parts myself — attention and the Transformer block, LoRA's update rule,
> the DPO objective derived from the KL-constrained RLHF optimum — and then
> verified each against a reference before trusting it. I ran the full path on
> `Qwen2.5-1.5B`: SFT, LoRA, QLoRA, a preregistered DPO sweep, and an evaluation
> suite I designed to be hard to fool. The interesting part is what it found:
> the update matrix wasn't low-rank, a preference metric turned out to be
> measuring token count, my DPO sweep returned a null I could quantify, and my
> own LLM judge was a third position-biased. All of that is written down as it
> happened — because the point of the project was learning to tell a measurement
> from an assumption."

---

## 7. Final Recommended CV Version

**AlignLab — LLM Post-Training Pipeline (PyTorch, Hugging Face, Hydra)**

- Implemented **attention (MHA/GQA)**, **RoPE**, **RMSNorm**, **SwiGLU** and
  **KV caching** from first principles, cross-verified against PyTorch's fused
  kernel and two independent re-implementations.
- Fine-tuned `Qwen2.5-1.5B` end to end — **completion-only SFT** (perplexity
  **−16.2%**, stop-token emission **0/6 → 6/6**), then **LoRA** and **4-bit NF4
  QLoRA** at **0.28% trainable parameters** and **~61%** of LoRA's peak VRAM.
- Analysed adapter rank with **SVD/PCA** and **measured** the true
  fine-tuning update — **disproving** the common "updates are low-rank"
  explanation and isolating adapter *target-set reachability* instead.
- Derived and implemented **DPO** from the KL-constrained **RLHF** objective with
  verified initialisation invariants; a **preregistered** β sweep returned a
  **null**, quantified as **~250× under KL budget**.
- Built a **14-metric evaluation suite** with **Wilson intervals**, explicit
  unresolvable verdicts, **order-swapped LLM-as-judge** (**33.3%** position bias
  detected) and full run provenance; **607 passing tests**, reproducible from committed
  manifests.

---

### Evidence basis

Every metric above was read from, and is traceable to, the following repository
documents: `README.md` (§7–§17), `docs/PROJECT_NARRATIVE.md` (findings 1–8),
`docs/EXPERIMENT_REGISTRY.md` (E1–E22 verdicts), `docs/LIMITATIONS.md` (what may
not be claimed), `PROJECT_INSTRUCTIONS.md` (scope boundary and CV framing),
`INTERVIEW_DEFENSE/phase1–phase8*.md`, and `tests/` (610 collected — 607 passed, 3 skipped, verified locally on 2026-08-31).

**Deliberately excluded** because the repository does not support them: any PPO
implementation claim (Phase 5 is conceptual only); GRPO (absent from the project
entirely); any statement that DPO improved quality; any low-rank explanation of
LoRA; confirmation of the E20 reachability hypothesis (**NOT CONFIRMED** — the
decisive MLP-target run is deferred); standard-benchmark or out-of-distribution
results; multi-GPU or scaling claims; run-to-run error bars (single seed
throughout); and any production-quality framing.
