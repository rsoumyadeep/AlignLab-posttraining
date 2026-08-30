# Phase 7 — Evaluation: Report

**Date:** 2026-08-30 · **Branch:** `phase-7-evaluation`
**Status:** engineering COMPLETE · USER explain-backs DEFERRED

> Phases 3–6 were not reopened. The Phase 6 null result is preserved exactly as
> reported, the POST-HOC experiment remains labelled POST-HOC, and no Phase 3–6
> document was edited.

---

## 1. Evaluation infrastructure built

| File | Responsibility |
|---|---|
| `src/alignlab/evals/metrics.py` | **pure functions** — perplexity, lengths, termination, repetition, preference stats, Wilson intervals, resolvability |
| `src/alignlab/evals/runners.py` | the layer that touches models — perplexity, preference, generation, provenance |
| `src/alignlab/evals/judge.py` | LLM-as-judge with two-order position randomisation |
| `src/alignlab/evals/report.py` | dashboard + pairwise comparison; **no aggregate score** |
| `src/alignlab/evaluate.py` | Hydra entrypoint |
| `configs/eval.yaml`, `configs/eval/{default,quick}.yaml` | model set, datasets, prompts, judge |
| `tests/test_eval_metrics.py`, `tests/test_eval_judge.py` | 82 tests |

**The pure/impure split is the design.** Metrics are pure so they can be tested
against hand-computed values; runners touch models so they cannot be. A metric
you can only exercise by loading a 1.5B model is a metric nobody checks.

Adding a model to `configs/eval/default.yaml` is the only change needed to
include it. Nothing is hard-coded in the entrypoint.

---

## 2. Metrics implemented

| # | Metric | Code |
|---|---|---|
| 1 | completion-only perplexity | `perplexity_from_totals(..., region="completion")` |
| 2 | full-sequence perplexity | same, `region="full_sequence"` |
| 3 | generation length distribution | `length_stats` (n, mean, median, min, max, p90) |
| 4 | termination rate | `TerminationStats.termination_rate` |
| 5 | stop-token emission rate | `TerminationStats.stop_token_rate` |
| 6 | empty-output count | `TerminationStats.empty` |
| 7 | repetition (`distinct-2`) | `distinct_n` |
| 8 | max n-gram repeat | `max_ngram_repeat` |
| 9 | structural checks | `structural_checks` (descriptive, never scored) |
| 10 | preference accuracy, SUM **and** MEAN | `PreferenceStats` |
| 11 | length attribution | `PreferenceStats.length_attribution()` |
| 12 | Wilson 95% interval | `wilson_interval` |
| 13 | resolvability verdict | `difference_is_resolvable`, `compare_proportions` |
| 14 | judge win rate + position-bias rate | `JudgeResult` |

**There is no aggregate score, and no function that could produce one.** A test
asserts the report module exposes nothing matching `"score"`. The reason is
empirical: in each of Phases 3–6 two metrics disagreed, and the disagreement
*was* the finding.

| phase | metric A said | metric B said |
|---|---|---|
| 3 | loss improved 33× | generation was destroyed (causal leakage) |
| 4 | perplexity within 3.5% | stop behaviour 0/4 vs 4/4 |
| 5 | 47.2% preference | 58.3% preference |
| 6 | accuracy unchanged | reward margin moved, length +30% |

A weighted average would have erased all four.

---

## 3. Verification

| Claim | How verified |
|---|---|
| perplexity arithmetic | hand-computed values; raises on negative NLL |
| region gating | `comparable_to()` is False across regions **even for identical values** |
| Wilson interval | known value 50/100 → **[0.4038, 0.5962]**; stays in [0,1] at 0/10, 10/10, 1/200; width monotone in n |
| resolvability | Phase 4's 0/4 vs 4/4 **is** resolvable; Phase 6's 86/184 vs 86/184 is **not** |
| position-bias handling | scripted always-first judge → **zero** decided verdicts, not a 100% win rate |
| ties/unparsed excluded | asserted not folded into the win rate as half-wins |
| no aggregate score | module introspection test |
| provenance completeness | no field silently null for an enabled family |
| **cache ordering** | source inspection of all three entrypoints; **verified to fail when the old ordering is restored** |

**Test totals:** 82 Phase 7 tests. Suite **585 passed / 3 skipped locally**,
**587 passed / 1 skipped on the server** (the difference is CUDA, SIGUSR1 and
`wandb`, which exist only there).

---

## 4. Experiments and results

Run `eval-full-001`, commit `2e2a51f1`, `git_dirty: false`, NVIDIA RTX A6000.
Full dashboard, run log and every generation are committed in this directory.

| model | ppl[compl] | ppl[full] | pref SUM | pref MEAN | stop | gen len | distinct-2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 8.824 | 14.993 | 45.7% | 61.4% | 0/6 | 212.3 | 0.523 |
| **SFT** | **7.398** | **12.284** | 46.7% | 58.7% | **6/6** | 117.3 | **0.839** |
| LoRA r=16 | 7.665 | 13.600 | 46.7% | 60.9% | 0/6 | 256.0 | 0.449 |
| QLoRA r=16 | 7.733 | 13.732 | 46.7% | 61.4% | 0/6 | 256.0 | 0.572 |
| DPO b=0.1 | 7.398 | 12.287 | 46.7% | 58.7% | **6/6** | 118.2 | 0.831 |

**Of the 24 pairwise comparisons produced, 3 were resolvable — all stop-token
rates.** That is the headline, and the dashboard states it rather than
reporting a number that overstates what the sample supports.

### 4a. SFT is the only stage that moved anything visible
Completion perplexity −16.2%, stop-token emission 0/6 → 6/6 (intervals
disjoint, **RESOLVED**), distinct-2 0.523 → 0.839. Base output is degenerate:
`.DrawString(...)` loops, `-unstyled` to the cap, one English prompt answered
in Chinese.

### 4b. PEFT termination failure reproduced — and the mechanism is now visible
LoRA and QLoRA hit the cap on 6/6 (min = max = 256). Past the answer they fall
back into the chat template:

```
Dear [Name], ... Best regards, [Your Name] комф
You are a helpful assistant.TRGL
You are a helpful assistant.TRGL   [to the cap]
```

Never having learned `<|im_end|>`, the likeliest continuation after a finished
answer is the next ChatML turn. This is Phase 4's structural explanation
(`lm_head` is tied to the embedding, outside LoRA's target set) now visible in
the text rather than only in a rate.

### 4c. DPO changed the weights and no metric
Perplexity moved in the 4th decimal; greedy generation differs on 4 of 6
prompts (2 byte-identical). Preference SUM, preference MEAN and stop rate are
identical to SFT. **An independent replication of the Phase 6 null on a
different evaluation path.** The preregistered Phase 6 conclusion stands.

### 4d. The SUM/MEAN inversion is a property of the metric, not of training

| model | chosen/token | rejected/token | SUM gap | explained by length | residual |
|---|---:|---:|---:|---:|---:|
| base | −1.1470 | −1.1607 | −30.57 | −34.28 | **+3.71** |
| SFT | −1.0713 | −1.0803 | −29.47 | −31.90 | **+2.43** |
| LoRA r=16 | −1.1001 | −1.1098 | −30.13 | −32.78 | **+2.65** |
| QLoRA r=16 | −1.0924 | −1.1005 | −30.30 | −32.50 | **+2.20** |
| DPO b=0.1 | −1.0712 | −1.0804 | −29.43 | −31.91 | **+2.48** |

Every model prefers the chosen response **per token**; every model's SUM
comparison **inverts** that verdict. Positive residual in all five rows,
**including the untrained base model**.

### 4e. The judge did not confirm the one thing everything else agrees on

| | base vs SFT | SFT vs DPO |
|---|---|---|
| a/b/tie | 1/3/0 | 1/1/3 |
| inconsistent (excluded) | **2 of 6** | 1 of 6 |
| position bias rate | **33.3%** | 16.7% |
| win rate (B, decided) | 0.750 [0.301, 0.954] | 0.500 [0.095, 0.905] |
| verdict | NOT resolvable at n=4 | NOT resolvable at n=2 |
| unparsed replies | 0 | 0 |

Base emits `-unstyled` 128 times; SFT writes a correct email; the judge still
could not resolve it. **That is the protocol working** — a one-order judge
would have reported a clean 6-pair win rate while a third of its verdicts were
decided by presentation order. On the cooking prompt (base answered in Chinese)
it replied `A` in **both** orders.

**An accidental positive control passed:** two SFT-vs-DPO pairs were
byte-identical and a third differed by two words; the judge returned `TIE` on
all three, in both orders, unprompted.

---

## 5. Evaluation protocol

```
base model     Qwen/Qwen2.5-1.5B @ 8faed761d45a263340a0528343f099c05c9a4323
perplexity     no_robots test, 150 examples, 27,127 completion tokens
               fingerprint f92fdec349b9c8ed...
preference     ultrafeedback_binarized test_prefs, 200 rows -> 184 usable
               fingerprint b3bc775a6db554cc...
generation     6 fixed prompts, GREEDY, max_new_tokens 256, seed 42
judge          Qwen/Qwen2.5-7B-Instruct, both orders, max_new_tokens 8, seed 42
recorded       git commit + dirty flag, dtype, device, hardware, timestamp
```

- **Greedy by default.** A comparison between two models should depend on their
  weights, not on sampling noise. Sampling parameters are passed **only** when
  `do_sample` is true, because transformers silently ignores `temperature`
  under greedy decoding and a config that looks meaningful would not be.
- **Adapters loaded unmerged.** Phase 4 measured bf16 merging as 12,460× less
  exact than fp32; a merged adapter would measure the adapter *plus* a merge
  artefact.
- **Every judge pair judged twice**, in both orders, verdict mapped back to the
  original labels. Flips are reported as position bias and **excluded**, never
  split as half-wins.

---

## 6. Bugs and corrections

| # | Bug | Resolution |
|---|---|---|
| 1 | `chr()`-obfuscated f-strings in the renderer, written to dodge nested quotes | rewritten with a `fmt()` helper |
| 2 | `load_model` would crash on LoRA/QLoRA adapters | detects `adapter_config.json`, loads via `PeftModel`, unmerged |
| 3 | stray Chinese characters in a `metrics.py` docstring | removed |
| 4 | **`dataset_fingerprint` null in all five provenance records**; one `eval_fingerprint` named the preference rows while sitting beside the perplexity numbers | each metric family now records its own dataset, split, row count, rows dropped, sha256; 4 tests |
| 5 | **`configure_hf_cache` was a no-op in every entrypoint** — see below | ordering fixed in all three entrypoints + live-constant rebinding; 5 tests |
| 6 | latent `KeyError` in `train.py`: logged `applied["HF_HUB_CACHE"]`, absent in exactly the case its `else` branch existed for | reports `effective_hub_cache` |
| 7 | seeding subprocess test depended on the caller exporting `PYTHONPATH` | passes its own env |
| 8 | wrong commentary in my own PyTorch example ("0/4 and 4/4 fail to exclude 0.5") | **caught by executing it** — both exclude 0.5; corrected |

### Bug 5 in detail — the one that cost something

Free space fell 51 → 37 GiB with no training run in between. Qwen2.5-7B-Instruct
(15.24 GB) and Qwen2.5-1.5B (3.10 GB) existed in **both** cache roots.

Verified by execution, not inferred:

```
import transformers                     # imports huggingface_hub
os.environ["HF_HUB_CACHE"] = <configured path>
huggingface_hub.constants.HF_HUB_CACHE  # -> unchanged, user-home default
```

`huggingface_hub` reads `HF_HUB_CACHE` **once, at import**. All three
entrypoints imported transformers at the top of their run function and called
`configure_hf_cache` several lines later, so the call could never have had any
effect — it only appeared to work when the variable happened to be exported in
the shell.

**The function's own docstring said it existed to prevent exactly this.**

Fixed by ordering (the real fix) plus live-constant rebinding across every
namespace holding a copy (the safety net; it never imports `huggingface_hub`
to do it). **Verified end-to-end on the server:** a quick eval run left
`~/.cache/huggingface/hub` byte-identical at 24,398,750,542 B, and no rebind
warning fired — the ordering alone now suffices.

**Two project guards caught me this phase rather than the other way round:**
the hardcoded-path test rejected a literal user-home path in my new test and
docstring, and executing my own example disproved my written commentary.

---

## 7. Resources inspected

Full audit in `LEARNING_RESOURCES/phase7_resources.md`.

| Resource | Status |
|---|---|
| *Judging LLM-as-a-Judge* (arXiv 2306.05685) | **PARTIALLY INSPECTED** — abstract in full, plus the position-bias section and Table 2. NOT read end to end. |
| no_robots, ultrafeedback_binarized | ACTUALLY INSPECTED (carried forward, same fingerprints). Neither dataset **card** was read. |
| Qwen2.5-7B-Instruct | ACTUALLY INSPECTED — downloaded and executed |
| Wilson score interval | **NOT READ FROM A SOURCE** — implemented from standing knowledge and verified numerically instead |

**What the paper changed.** Its "Consistency" metric is exactly the two-order
protocol implemented here, which was written **before** the paper was read —
independent convergence, not a copy. Its numbers recalibrated expectations:
GPT-4 is consistent on only **65.0%** of pairs, Claude-v1 on **23.8%**. A low
consistency rate from a 7B judge is therefore the expected result.

Two of its three named biases apply directly: **verbosity bias** (a judge
favouring our DPO model could be measuring length *again*, after Phases 5 and 6
both found length dominating a preference metric — our prompt instructs the
judge to ignore length, a mitigation we did **not** verify) and
**self-enhancement bias** (our judge is the same family as the models judged).

**Not inspected, and it matters:** AlpacaEval's length-controlled win rate,
which exists precisely for the bias Phases 5–7 kept measuring.

---

## 8. Documentation created / updated

**Created:** `STUDY_WITH_CLAUDE/phase7/01_evaluation_theory.md` ·
`CODE_EXPLANATION/phase7/README.md` · `INTERVIEW_DEFENSE/phase7_evaluation.md` ·
`LEARNING_RESOURCES/phase7_resources.md` ·
`PYTORCH_CONCEPTS/pytorch-evaluation-mechanics.md` + its executed example ·
`docs/phase7/EVAL_RESULTS.md` · this report.

**Updated:** `README.md` (status, layout, evaluation quickstart, test counts).

Every number in the PyTorch note comes from
`PYTORCH_CONCEPTS/examples/evaluation_mechanics_examples.py`, which is
executable and was executed.

**Numbers reconciled, not overwritten.** Phase 4 reported base 8.541 / SFT
7.199 over 38,831 completion tokens; Phase 7 gives 8.824 / 7.398 over 27,127.
Both correct — different token populations, which is *Phase 3's own lesson
recurring a phase later*. The comparison replicates to within 0.5 pp
(−15.7% → −16.2%; +3.5% → +3.6%). No Phase 3–6 document was edited.

---

## 9. Limitations

1. **Everything is in-distribution.** SFT perplexity is measured on no_robots'
   test split — what SFT trained on; preference accuracy on UltraFeedback's
   test split — what DPO trained on. Test *splits*, so no example-level
   leakage, but "better" is partly circular.
2. **Sample sizes are small.** 184 preference pairs, 6 generation prompts, and
   after excluding position-flips only **4 and 2 decided judge verdicts**.
3. **The judge is unvalidated and same-family**, and small by frontier
   standards. No human agreement study was run.
4. **The judge is pinned to `main`, not a commit SHA** — a deviation from this
   project's practice everywhere else. A future rerun could get a different
   judge.
5. **One seed everywhere.** No variance estimate across runs.
6. **Greedy decoding only.** Real deployments sample.
7. **`distinct-2` is a crude repetition proxy**, not a fluency measure.
8. **No task-specific correctness checks** — no unit tests for code answers, no
   factuality checking.
9. **The judge-prompt length-mitigation was not verified.** We instruct the
   judge to ignore length; we did not test whether it does.

---

## 10. What the evaluation system CAN establish

- Whether a model fits a given distribution better, **stated with its token
  region and denominator**, so two perplexities cannot be silently compared
  across populations.
- Whether a model **terminates**, and separately whether it emits the **chat
  template's stop token** — measured, and RESOLVED at n=6.
- Whether generation is degenerate (`distinct-2`, max n-gram repeat, empty).
- Whether a preference gap is **real or length-driven**, by decomposition, on
  any preference set.
- Whether a difference is **larger than its own uncertainty** — and it says
  "NOT resolvable at this sample size" when it is not.
- Whether a judge's verdicts are **preference or position**, by construction.
- Complete provenance: model, revision, checkpoint, both dataset fingerprints,
  decoding settings, git commit **and dirty flag**, hardware, timestamp.

## 11. What it CANNOT establish

- **General assistant quality.** Nothing here is out-of-distribution.
- **That any model is "better" overall.** There is deliberately no aggregate
  score.
- **Ground truth on subjective quality.** The judge is not validated against
  human preference.
- **Small effects.** At these sample sizes, most differences are unresolvable —
  21 of 24 comparisons in the run.
- **Factual correctness, safety, or task success.** Not measured.
- **Run-to-run variance.** One seed.
- **That length was successfully controlled anywhere.** It was *measured* and
  *attributed*, never removed.

---

## 12. Git state

```
branch          phase-7-evaluation
LOCAL  = GITHUB = SERVER   (all at 4843ff3 + this report's commit)
history         preserved; no rewrite, no force-push
```

Phase 7 commits:

| commit | subject |
|---|---|
| `2e2a51f` | evaluation subsystem — metrics, runners, judge, dashboard |
| `76adf5a` | resource audit — the MT-Bench judge paper validates and recalibrates |
| `8fe016b` | the full evaluation pass — five models, and mostly null |
| `4843ff3` | `configure_hf_cache` was a no-op, and it cost 18 GiB |

**Storage.** `/data` has **37 GiB** free. Protected artifacts verified intact:
`checkpoint-295` **8.7 G**, `final/` **2.9 G**. Nothing was deleted.

⚠ **~18.3 GB of duplicate weights remain** in `~/.cache/huggingface/hub`
(Qwen2.5-7B-Instruct 15.24 GB, Qwen2.5-1.5B 3.10 GB), created by bug 5 before
it was fixed. **Not removed — awaiting authorisation**, because that directory
also holds unrelated models belonging to other work (`phi-2` 5.2 G, several
`minari` datasets). Only the two Qwen directories are candidates.

---

## 13. Deferred USER explain-backs

All Phase 7 checkpoints remain **DEFERRED — USER EXPLAIN-BACK REQUIRED**:

- **Checkpoint 23** — Why isn't perplexity enough? (`STUDY_WITH_CLAUDE`)
- **Checkpoint 24** — Design an evaluation for an aligned model.
- **Checkpoint 25** — LLM-as-judge: position bias, the two-order protocol, and
  three reasons this judge is not ground truth.
- **My Understanding** — `PYTORCH_CONCEPTS/pytorch-evaluation-mechanics.md`
- **Own Words** — one entry per resource in `LEARNING_RESOURCES/phase7_resources.md`
- **Explain Back** — `INTERVIEW_DEFENSE/phase7_evaluation.md` Q1, Q2, Q3, Q5,
  and Q8 argued from both sides.

Nothing in these sections was filled in.

---

## 14. Phase 8 prerequisites

**Ready:** evaluation subsystem with provenance on every result; five trained
models; 587 tests green on the server; LOCAL = GITHUB = SERVER; storage bug
fixed and verified.

**Decisions needed before Phase 8:**

1. **The 18.3 GB of duplicate weights** — remove, or keep? Recovering them
   would take `/data` from 37 GiB to ~55 GiB free.
2. **`checkpoint-295` (8.7 G)** — still protected by standing instruction.
   Its `final/` sibling is the DPO starting point and is in use; the
   mid-training checkpoint may no longer be needed.
3. **Pin the judge to a commit SHA** rather than `main`.

**The single biggest gap, and the top Phase 8 candidate: an out-of-distribution
evaluation.** Every number in this project measures fit to a training
distribution. A held-out set from a different source — plus AlpacaEval's
length-controlled protocol, which addresses the bias Phases 5–7 kept measuring
— would be worth more than any additional metric on the current data.

**Phase 8 has NOT been started.**

**Related:** [[phase7-evaluation-theory]] · [[phase7-code-explanation]] ·
[[interview-phase7-evaluation]] · [[phase7-resources]]
