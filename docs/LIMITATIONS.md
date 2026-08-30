# AlignLab — Limitations

What this project does **not** establish. Nothing here is hedging: each item is
a specific reason a specific claim would be unsupported.

---

## Experimental design

### 1. Single seeds in every major experiment
Seed 42 throughout. **No variance estimate exists for any result.** A 3.6%
perplexity gap between LoRA and full SFT has no run-to-run error bar, so we
cannot say whether it exceeds seed noise. The Wilson intervals in the
evaluation quantify *sampling* uncertainty over examples, not *training*
variance across runs — a different thing, and the smaller of the two here.

### 2. Limited training budget
One epoch of `no_robots` (9,499 rows) for SFT; the DPO runs moved the policy
**0.0008 in KL** against SFT's 0.2044 — roughly **250× under-budget**. The
Phase 6 null is a statement about that budget, not about DPO.

### 3. One primary LoRA target set
Everything PEFT-related uses `q_proj, k_proj, v_proj, o_proj` at r=16. Two
learning rates were run (2e-5 matched, 2e-4 conventional); **the target set was
never varied.**

### 4. The MLP-target LoRA experiment was never run
E20's reachability hypothesis — that the PEFT arms failed to stop because
`lm_head` is tied to the embedding and outside the target set — is
**NOT CONFIRMED**. The decisive test is a LoRA variant adapting the MLP or an
untied head, roughly **18 minutes of GPU time**. It is deferred, not skipped,
and until it runs the hypothesis stays a hypothesis.

### 5. No rank sweep
r=16 only. `docs/phase4` computes the break-even rank
`r* = (d_in·d_out)/(d_in+d_out)` and analyses the SVD spectrum, but **no
experiment trained at r ∈ {4, 8, 32, 64}**. Any statement about how quality
varies with rank would be unsupported here.

### 6. DPO was not replicated under the paper's exact training setup
Our DPO uses our own loss and our own loop, verified against the published
formula, but the **training configuration is ours** — batch geometry, LR
schedule, epoch count and data scale differ from the DPO paper's. A null under
our setup is not evidence about theirs.

---

## Evaluation

### 7. Every number is in-distribution
SFT perplexity is measured on `no_robots`' test split — the dataset SFT trained
on. Preference accuracy is on UltraFeedback's `test_prefs` — what DPO trained
on. These are held-out **splits**, so there is no example-level leakage, but the
**distribution-level overlap is total**: the models were optimised toward
exactly the distribution they are scored on.

**"Better" is therefore partly circular.** These metrics measure *fit to the
training distribution*, which is what one epoch of that data should buy. **This
is the single largest gap in the project.**

### 8. No large-scale evaluation benchmark
No MMLU, no HELM, no lm-evaluation-harness, no AlpacaEval. The evaluation suite
is bespoke and narrow: 150 perplexity examples, 184 preference pairs, **6**
generation prompts.

Notably, **AlpacaEval's length-controlled win rate was not inspected or used**,
and it addresses precisely the length bias Phases 5–7 kept measuring.

### 9. Sample sizes are small, and most differences are unresolvable
Of the **24 pairwise comparisons** in the Phase 7 run, **3 were resolvable** —
all stop-token rates. The other 21 are reported as **"NOT resolvable at this
sample size"** rather than as small effects.

### 10. LLM-judge reliability
- **Position sensitivity measured at 33.3%** on the base-vs-SFT comparison — 2
  of 6 pairs flipped when the answers were swapped.
- After excluding flips, only **4 and 2 decided verdicts** remained. Both
  intervals include 0.5.
- The judge is **Qwen2.5-7B-Instruct — the same model family** as the models
  under test.
- Its agreement with human preference was **never validated**; no human study
  was run.
- It is pinned to `main`, **not a commit SHA** — a deviation from this
  project's practice everywhere else, so a future rerun could get a different
  judge.
- The judge prompt instructs it to ignore length. **We did not verify that it
  does.**

### 11. Greedy decoding only
Every comparison uses greedy decoding, deliberately, so results depend on
weights rather than sampling noise. Real deployments sample, and behaviour can
differ.

### 12. `distinct-2` is a crude repetition proxy
It detects degeneration. It is not a fluency, coherence or quality measure.

### 13. No task-specific correctness checks
No unit tests for generated code, no factuality checking, no safety evaluation,
no refusal testing.

### 14. Length was measured and attributed, never controlled
The length decomposition quantifies how much of a preference gap is token
count. **No experiment removed the confound** — there is no length-matched
evaluation set and no length-controlled win rate.

---

## Engineering

### 15. No multi-GPU training
Everything ran on a **single NVIDIA RTX A6000 (47.53 GiB)**. No FSDP, no DeepSpeed,
no tensor/pipeline parallelism, no gradient-accumulation-across-nodes. Nothing
here says how the pipeline behaves at scale.

### 16. One model, one size
`Qwen/Qwen2.5-1.5B` throughout. No cross-architecture or cross-scale
comparison, so no claim generalises to other model families or sizes.

### 17. Cross-machine bitwise reproducibility is not claimed
Bitwise reproducibility is verified **same machine, same seed, same
environment** (Tier A), including across separate interpreter processes.
Bitwise agreement between the local Windows CPU machine and the Linux GPU
server is **not achievable and not claimed**.

### 18. One historical provenance gap remains
The `eval-full-001` dashboard recorded `dataset_fingerprint: null` in all five
per-model records. The code is fixed and a later run is clean, but that
artefact's per-model field is **NOT RECORDED** and is left that way — the
run-level fingerprints are in its protocol block and run log, so the run is
still traceable. It was **not back-filled**, because reconstructing provenance
after the fact would be worse than lacking it.

---

## What this project does not claim

- **Not** that AlignLab produces a production-quality model. It produces a
  1.5B model fine-tuned for one epoch on 9,499 instruction examples.
- **Not** that LoRA works because fine-tuning updates are low-rank. **E17
  disproved that** on the update we measured.
- **Not** that DPO improved the model. The Phase 6 and Phase 7 evidence does
  not establish it.
- **Not** that SFT made a better assistant — only that it substantially changed
  termination behaviour and in-distribution perplexity.
- **Not** that the LLM judge reflects human preference.
- **Not** that any result generalises beyond this model, this data and this
  budget.

---

## What would most change the picture

1. **An out-of-distribution evaluation.** The biggest gap by a wide margin.
2. **The MLP-target LoRA run.** 18 minutes to convert the project's best
   hypothesis into a result — or to kill it.
3. **Multiple seeds** on the headline comparisons, so the 3.6% PEFT gap gets an
   error bar.
4. **A properly-budgeted DPO run**, with KL in the range SFT reached rather than
   250× below it.
5. **A length-controlled evaluation**, which is the only way to settle what the
   preference metrics are measuring.

**Related:** [[project-narrative]] · [[experiment-registry]] · [[phase7-report]]
