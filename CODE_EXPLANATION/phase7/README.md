# Phase 7 — Code Explanation

The evaluation subsystem. Built on the Phase 1 skeleton rather than replacing
it — `alignlab.evaluation` already had the right shape (results carry the
manifest that produced them; a failing evaluator is recorded `NOT_TESTED`
rather than aborting the pass).

---

## 1. Files

| File | Responsibility |
|---|---|
| `src/alignlab/evals/metrics.py` | **pure functions** — perplexity, lengths, termination, repetition, preference stats, Wilson intervals |
| `src/alignlab/evals/runners.py` | the layer that touches models — perplexity, preference, generation |
| `src/alignlab/evals/judge.py` | LLM-as-judge with two-order position randomisation |
| `src/alignlab/evals/report.py` | dashboard and pairwise comparison; **no aggregate score** |
| `src/alignlab/evaluate.py` | Hydra entrypoint |
| `configs/eval.yaml`, `configs/eval/{default,quick}.yaml` | the model set, datasets, prompts, judge |
| `tests/test_eval_metrics.py`, `tests/test_eval_judge.py` | 78 tests |

**The split is the design.** Metrics are pure so they can be tested against
hand-computed values; runners touch models so they cannot be. A metric you can
only exercise by loading a 1.5B model is a metric nobody checks.

---

## 2. Data flow

```
configs/eval/default.yaml
   │  models[], comparisons[], datasets, generation settings, judge
   ▼
alignlab.evaluate
   │
   ├── load_lm_examples          no_robots test split  -> prompt/completion
   ├── load_preference_examples  ultrafeedback test_prefs -> prompt/chosen/rejected
   │
   └── for each model in config:
         load_model              full checkpoint | ADAPTER (unmerged) | base@revision
         ├── completion_perplexity  -> {completion, full_sequence}, each labelled
         ├── preference_evaluation  -> PreferenceStats (SUM + MEAN + lengths)
         └── generation_evaluation  -> texts, termination, lengths, structure
   ▼
compare_models(baseline, candidate)     Wilson intervals; APPLICABLE / NOT_*
   ▼
run_judge                                both orders, mapped back to a/b
   ▼
Dashboard.write()  +  render_text()      JSON + text, full provenance
```

---

## 3. Metric → code

| Concept | Code |
|---|---|
| `PPL = exp(mean NLL/token)` | `metrics.perplexity` (raises on negative NLL) |
| token-weighted, not example-weighted | `metrics.perplexity_from_totals(total, n_tokens, region)` |
| completion-only vs full-sequence | `runners.completion_perplexity` returns **both**, labelled |
| preference SUM (primary) | `PreferenceStats.sum_accuracy` |
| preference MEAN (diagnostic) | `PreferenceStats.mean_accuracy` |
| Phase 6's length decomposition | `PreferenceStats.length_attribution()` |
| stop-token behaviour | `metrics.termination_stats` |
| repetition | `metrics.distinct_n`, `metrics.max_ngram_repeat` |
| uncertainty | `metrics.wilson_interval` |
| "not resolvable" | `metrics.difference_is_resolvable` |
| position bias | `judge.judge_pairs` (two orders) + `JudgeResult.position_bias_rate` |

---

## 4. The design decisions that carry weight

| Decision | Alternative | Why |
|---|---|---|
| **no aggregate score** | a weighted quality number | Phases 3–6 each found two metrics disagreeing, and the disagreement was the finding. A test asserts `report` exposes nothing matching `"score"`. |
| `PerplexityResult` carries `region` | a bare float | Phase 3 measured prompt 6.3314 vs completion 3.6295 on one model. `comparable_to()` is False across regions **even for identical values**. |
| both perplexity regions always | one | a report quoting "perplexity" without saying which invites exactly that confusion |
| preference SUM **and** MEAN, always | whichever is asked for | Phase 5's 11-point swing; `to_dict()` cannot emit one alone |
| length beside every preference/quality number | on request | Phase 6's metric was length, not quality |
| Wilson intervals | normal approximation | escapes [0,1] and has poor coverage at n=4…184 |
| `difference_is_resolvable` returns no p-value | a significance test | overlapping intervals do not prove no difference; this is a conservative screen, not an inference |
| ties/flips **excluded** from judge win rate | counted as half-wins | they carry no preference information; halves manufacture precision |
| judge every pair **twice** | once | a one-order judge bakes position bias into the win rate invisibly |
| unparsed judge reply → `"unparsed"` | map to tie | guessing invents data |
| adapters loaded **unmerged** | merged | Phase 4 measured bf16 merging 12,460× less exact than fp32 |
| greedy generation by default | sampling | a comparison should depend on weights, not sampling noise |
| sampling params only when `do_sample` | always pass | transformers silently ignores `temperature` when greedy, making a config look meaningful |
| full generation set written out | only shown examples | PROJECT_INSTRUCTIONS forbids cherry-picking; the JSON lets any selection be checked |

---

## 5. Reproducibility record

`EvalProvenance`, attached to every model's results:

```
model_id · model_revision · checkpoint · tokenizer · dtype · device
dataset · dataset_fingerprint · eval_fingerprint
generation settings (incl. seed and decoding mode)
judge_model · git_commit · git_dirty · timestamp · hardware
```

`git_dirty` is recorded, not just the commit — a result produced from a
modified tree is reproducible only up to that modification, and Phase 3's
manifest already caught one such run.

---

## 6. Status vocabulary

Extending PROJECT_INSTRUCTIONS §15 for comparisons:

- **`APPLICABLE`** — meaningful for this model/comparison
- **`NOT_APPLICABLE`** — computable but not informative as a claim
- **`NOT_MEASURED`** — could apply, was not run

A missing metric appears in the comparison list **with a status and a note**,
never as a silent absence. A silently missing metric is the kind of gap that
later gets mistaken for a measured zero.

---

## 7. Bugs and corrections

### 7a. `chr()`-obfuscated f-strings in the renderer
I wrote the dashboard table using `chr(115)+chr(117)+...` to spell dictionary
keys, purely to dodge nested-quote issues inside f-strings. Unreadable, and
inexcusable. Rewritten with a small `fmt()` helper and intermediate variables.

### 7b. Adapters would have crashed the entrypoint
`load_model` initially called `AutoModelForCausalLM.from_pretrained` on every
checkpoint. LoRA and QLoRA checkpoints are **adapters** — that call fails.
Caught before the first real run; now detected via `adapter_config.json` and
loaded through `PeftModel`, unmerged.

### 7c. A stray non-ASCII character in a docstring
Two Chinese characters appeared mid-sentence in `metrics.py`. Removed. (The
remaining non-ASCII — em-dashes, `±`, and the `•` inside the bullet-list regex
— are intentional and safe in UTF-8 source; the Phase 6 lesson was specifically
about `print()` under Windows cp1252.)

---

## 8. Test map

| Claim | Tests |
|---|---|
| perplexity arithmetic; region gating | `TestPerplexity` (6) |
| length distribution | `TestLengthStats` (4) |
| termination vs stop-token; Phase 4 scenario | `TestTermination` (5) |
| repetition detection (Phase 3's degenerate output) | `TestRepetition` (5) |
| structural checks are descriptive, not scored | `TestStructuralChecks` (6) |
| SUM/MEAN disagree; Phase 6 length attribution | `TestPreferenceStats` (7) |
| Wilson intervals incl. a known value | `TestWilsonInterval` (8) |
| resolvability; Phase 6's null is unresolvable | `TestComparison` (5) |
| dashboard refuses an aggregate score | `TestDashboard` (7) |
| judge parsing | `TestParseVerdict` (7) |
| **position bias yields zero decided verdicts** | `TestPositionBias` (6) |
| ties/unparsed not folded into the win rate | `TestAggregation` (5) |
| limitations attached to every result | `TestProvenanceAndLimitations` (5) |

**78 Phase 7 tests. Suite total: 577 passed, 2 skipped.**
