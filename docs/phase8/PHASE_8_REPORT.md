# Phase 8 — Engineering / Finalisation: Report

**Date:** 2026-08-30 · **Branch:** `phase-7-evaluation`
**Status:** COMPLETE · USER explain-backs DEFERRED

> Phases 1–7 were not reopened. No experimental conclusion was changed. All
> negative results are preserved with their original wording, including the
> Phase 6 null and the five disproved hypotheses.

---

## 1. Engineering changes

**No new algorithm.** One new module, one deletion, two documentation fixes in
source, and 22 new tests.

| File | Change |
|---|---|
| `src/alignlab/provenance.py` | **new** — mechanical audit of recorded metadata, plus a CLI |
| `src/alignlab/logprobs.py` | removed `score_sequences` — dead, and a padding trap |
| `src/alignlab/evaluation.py`, `evals/__init__.py` | docstrings disambiguating two similar names |
| `tests/test_provenance.py` | **new** — 20 tests |
| `tests/test_eval_metrics.py` | +2 — the rendered dashboard must be ASCII |
| `README.md` | rewritten to the 20 required sections |

Each refactor, in the required form:

**Removing `score_sequences`**
- **WHY:** the dead-code scan found it was the only unreferenced public symbol
  in the package — written in Phase 5, never called.
- **WHAT CHANGED:** deleted, with a comment recording what it was.
- **WHAT MUST REMAIN IDENTICAL:** everything. Nothing called it.
- It was deleted rather than kept because it built
  `attention_mask = torch.ones_like(input_ids)` — correct only for an unpadded
  batch. On a padded batch it would attend to pad tokens and return quietly
  wrong log-probabilities, which is the exact failure class Phases 3–7 built
  guards against.

**`evaluation` vs `evals` naming**
- **WHY:** two near-identical names for a Phase 1 evaluator *protocol* and a
  Phase 7 metrics *subsystem*.
- **WHAT CHANGED:** docstrings at both ends.
- **WHAT MUST REMAIN IDENTICAL:** all of it — nothing was renamed.
- Renaming was rejected: `alignlab.evaluation` is imported by completed Phase
  3–6 code, and churning working experimental paths for a cosmetic gain is what
  the brief rules out.

---

## 2. Repository audit

| Checked | Result |
|---|---|
| dead code | **1 found, removed** (`score_sequences`) — an AST scan over every public symbol |
| duplicated logic | none — `evaluation.py` and `evals/` are complementary, not duplicates |
| inconsistent naming | **2 found, both documented rather than renamed** (see §1, and the E-number gap in §5) |
| unnecessary dependencies | none identified |
| hard-coded paths | **none** outside `configs/env/` — enforced by a test that caught one of mine |
| hidden environment assumptions | **1 found and fixed in Phase 7** (`configure_hf_cache`); ordering now asserted by test |
| configuration leaks | none |
| fragile imports | the two broad `except` blocks are defensive, documented, and correct |
| missing error handling | `audit_run_directory` now reports an unreadable artefact instead of raising |
| misleading comments | none found |
| **stale documentation** | **2 found** — see below |
| tests testing implementation | the new tests target behaviour; the ordering test asserts a *property*, verified to fail on the old code |
| untested critical paths | ASCII dashboard output was an unenforced convention — now 2 tests |
| platform-specific assumptions | `SIGUSR1` skip is correct and deliberate; cp1252 output now tested |
| reproducibility gaps | **1 historical, reported not repaired** (§4) |
| secrets committed | none |
| broken README links / paths | none — checked mechanically |

### The two stale-documentation findings

**a. The environments have drifted apart.** The README claimed both machines ran
torch 2.6.0. Measured:

| | local | server |
|---|---|---|
| torch | **2.12.1+cpu** | 2.6.0+cu124 |
| transformers | **5.8.1** | 5.16.1 |
| python | **3.13.13** | 3.11.16 |

**Corrected in the README, and recorded as a Tier B weakness.** The drift itself
was **not** fixed: every result came from the server, each manifest pins that
machine's exact versions, and aligning a local machine that only builds docs and
runs CPU tests would be churn without a reproducibility gain.

**b. `LEARNING_RESOURCES/resources.md` was the Phase 1B register**, still
announcing that every external resource was `NOT INSPECTED` — true then, false
now. A consolidated index was added; **the Phase 1B note is preserved as
history** rather than deleted.

---

## 3. Tests

| | local | server |
|---|---|---|
| before Phase 8 | 585 passed / 3 skipped | 587 passed / 1 skipped |
| **after Phase 8** | **607 passed / 3 skipped** | **609 passed / 1 skipped** |

**Differences between environments, explained:** the server runs 2 tests the
local machine skips (CUDA, `SIGUSR1`) and has `wandb` installed. The local
machine runs 1 test the server skips (`test_paths_logging_device.py:206`,
which asserts the **CPU-only failure mode**). Every skip is a genuine
environmental impossibility.

**No skipped test was converted into a passing one.** The three local skips are
CUDA-absent, `SIGUSR1`-absent on Windows, and `wandb`-not-installed.

Also verified: **all 7 `PYTORCH_CONCEPTS` example scripts still execute**, so
every number quoted in those documents remains reproducible.

Status labels are maintained: `VERIFIED` · `MEASURED` · `NOT CONFIRMED`
(E20's reachability hypothesis) · `NOT TESTED` (W&B online, SLURM requeue,
multi-GPU RNG restore) · `DEFERRED` (all USER explain-backs).

---

## 4. Reproducibility audit

Built `alignlab.provenance` — because "we record provenance" is exactly the
claim that rots silently, and Phase 7 proved it by shipping a dashboard with
`dataset_fingerprint: null` while the documentation described provenance as
complete.

```bash
python -m alignlab.provenance          # exit 1 if any applicable field is missing
```

It audits 14 fields per artefact — git commit, dirty flag, model revision,
dataset and eval fingerprints, seed, config, config digest, hardware, software
versions, dtype, training parameters, evaluation parameters, checkpoint
identity — across both artefact shapes the project produces.

**Result over all 18 real run artefacts: 17 complete, 1 gap.**

| gap | run | status |
|---|---|---|
| `dataset_fingerprint` | `eval-full-001` | **NOT RECORDED** |

That is the known Phase 7 bug. The code is fixed, and `eval-cachefix-001` — run
after the fix — is clean, so **the auditor confirms the fix rather than me
asserting it**.

**The gap was not back-filled.** The run remains traceable (its protocol block
and run log carry both fingerprints), but the per-model field is absent and
stays absent. Reconstructing provenance from memory would be worse than lacking
it.

**Two corrections to the auditor itself**, both false positives that would have
made the report useless:

1. It demanded a hub revision from every evaluated model — but a fine-tuned
   checkpoint is a local directory with none; the *base* revision pins the
   lineage.
2. It reported **8 Phase 1 infrastructure runs** as missing model revisions and
   fingerprints. Those runs never loaded a model. Now detected from artefact
   **content**, not from the run's name.

---

## 5. Experiment registry

[`docs/EXPERIMENT_REGISTRY.md`](../EXPERIMENT_REGISTRY.md) — every experiment
from Phases 2–7 with hypothesis **as written before the result**, configuration,
outcome, what was learned, whether the hypothesis held, and the evidence path.

**No hypothesis was rewritten after seeing its result.** The five disproved ones
get their own closing table: E11 (parameter formula off by 57,344 QKV biases),
E12 H5 (prompt region harder, not easier), **E17 H2 (ΔW is not low-rank)**,
E19 H3 (LoRA did not match full fine-tuning), E22 H4 (47.2%, below chance).

**A correction made while writing it:** I had invented experiment IDs "E24" and
"E25" for the Phase 6 post-hoc run and the Phase 7 evaluation pass. **They do
not exist.** Phases 6 and 7 stopped assigning E-numbers and identify work by
report section and run name. The registry now uses the real run names, records
the numbering inconsistency instead of hiding it, and restores E23 to its actual
meaning (the Phase 6 SFT-vs-DPO evaluation). Its numbers are now quoted from
`e23_dpo_evaluation.json` rather than from report prose.

---

## 6. Documentation audit

The five folders are **distinct in kind**, not the same explanation at different
lengths:

| Folder | Kind | Coverage |
|---|---|---|
| `STUDY_WITH_CLAUDE/` | theory and derivation | phases 1–7 |
| `CODE_EXPLANATION/` | what the code actually does | phases 1–**8** |
| `PYTORCH_CONCEPTS/` | mechanics with **executed** examples | 7 notes, 7 scripts, **all verified to run** |
| `LEARNING_RESOURCES/` | honest source audit | phases 2–7 + consolidated index |
| `INTERVIEW_DEFENSE/` | reasoning under challenge | phases 1–7 + **whole-project defence** |

- **All 24 required interview topics** are covered.
- Every `PYTORCH_CONCEPTS` note has a runnable script, and all 7 execute.
- **31 files still carry unfilled USER explain-back markers.** None was filled.
- A mechanical scan for unsupported claims returned **0 genuine hits**; the 9
  matches were all explicit disclaimers or attributed quotations from a paper.

---

## 7. README

Rewritten to the 20 required sections, with every quantitative claim traceable
to an experiment, a test, or a measurement.

**Three claims it deliberately does not make:**

- **never** "LoRA works because fine-tuning updates are low rank" — E17
  disproved that on the update we measured;
- **never** "DPO improved the model" — accuracy was 86/184 for the baseline and
  all three β arms, byte-identical;
- **never** "SFT made a better assistant" — only that it changed termination
  behaviour and in-distribution perplexity.

**A correction:** the README linked a `LICENSE` file that does not exist. Rather
than invent a licence — the owner's decision, not the documentation's — it now
states plainly that none is declared and default copyright applies, and notes
that third-party terms were **not independently verified** because the model and
dataset cards were not read.

---

## 8. Cleanup performed

Authorised removal of exactly two duplicate cache directories left by the Phase
7 `configure_hf_cache` bug. Full record:
[`CLEANUP_RECORD.md`](CLEANUP_RECORD.md).

| | bytes free on `/data` |
|---|---:|
| before | 38,898,704,384 (37 G) |
| after | **57,240,600,576 (54 G)** |
| **recovered** | **18,341,896,192 = 18.34 GB** |

**Verified before deleting.** The 7B copies were identical blob sets at the same
revision. The 1.5B copies were **not** identical — home had 7 blobs, project 8 —
so `comm -23` was run to confirm the home set was a strict subset (the extra
project blob is `LICENSE`). Blobs are content-addressed, so matching names mean
matching bytes.

**Verified after deleting**, under `HF_HUB_OFFLINE=1` so a missing file would
raise rather than silently re-download: tokenizer, base model, LoRA adapter and
SFT checkpoint all loaded, and `~/.cache` was byte-identical afterwards.

**Nothing else was touched.** `phi-2` (5.2 G), a Mistral stub, six `minari`
datasets and both UltraFeedback copies were inventoried before and after.
`checkpoint-295` (**8.7 G**) and `final/` (**2.9 G**) are in a different tree
and remain intact.

---

## 9. Remaining known issues

1. **`eval-full-001` carries one provenance gap** — `dataset_fingerprint`
   NOT RECORDED per model. Fixed in code; the historical artefact is left as
   produced.
2. **The judge is pinned to `main`, not a commit SHA.** A future rerun could get
   a different judge. This is a deviation from the project's practice
   everywhere else.
3. **Local and server environments have drifted** (torch 2.12.1 vs 2.6.0). Tier
   B is weakened; no result is affected.
4. **No `LICENSE` file.** Default copyright applies.
5. **`triton` is installed but unimportable** on the server (missing
   `setuptools`) — recorded in Phase 4, never needed.
6. **W&B online mode, SLURM requeue and multi-GPU RNG restore remain
   `NOT TESTED`** — no API key, no working scheduler, no second GPU allocated.

---

## 10. Project limitations

Full list: [`docs/LIMITATIONS.md`](../LIMITATIONS.md). The required items, all
present:

single seeds in every major experiment · limited training budget (one epoch;
DPO at 250× under SFT's KL) · one primary LoRA target set · **the MLP-target
LoRA confirmation was never run** · no rank sweep · **all evaluation is
in-distribution** · limited DPO training · **DPO not replicated under the
paper's exact setup** · LLM-judge reliability (33.3% position sensitivity,
4 and 2 decided verdicts, unvalidated, same-family) · no multi-GPU training ·
no large-scale evaluation benchmark · **no claim of production-level model
quality**.

---

## 11. Git state

```
branch          phase-7-evaluation
commits         90 total, history fully preserved
force-pushes    0
rewrites        0
```

Phase 8 commits:

| commit | subject |
|---|---|
| `b993688` | remove 18.34 GB of duplicate Qwen weights, with verification |
| `715f2af` | mechanical provenance auditor |
| `24d0055` | experiment registry for Phases 2–7 |
| `7867549` | final README, project narrative, limitations |
| `c368907` | remove one dead function, document one confusing name |
| `8c7aa41` | documentation audit — capstone defence, Phase 8 explanation, resource index |

*(This report's own commit follows and is therefore not listed.)*

**A correction to this section.** The first draft of this table contained six
**invented** short SHAs — plausible-looking hex that I had not read from
`git log`. They were replaced with the real ones before commit. Fabricating an
identifier that a reader would use to verify a claim is the precise failure this
project's evidence rules exist to prevent, and it is recorded here rather than
silently fixed.

---

## 12. Local / GitHub / server synchronisation

```
LOCAL : 8c7aa41   working tree clean
GITHUB: 8c7aa41
SERVER: 8c7aa41   working tree clean
```

**LOCAL = GITHUB = SERVER.** Verified after the final commit. One untracked
`provenance.py` on the server — from an scp during development — was removed so
the tracked version could land; the server now runs only committed code.

Server test suite on that exact commit: **609 passed, 1 skipped.**

---

## 13. Deferred USER explain-backs

**31 files carry unfilled markers. Nothing was filled in.**

| Folder | Sections left for the user |
|---|---|
| `STUDY_WITH_CLAUDE/` | "My Understanding" and "Explain Back" for every phase, Checkpoints 1–25 |
| `PYTORCH_CONCEPTS/` | "My Understanding" in all 7 notes |
| `LEARNING_RESOURCES/` | "Own Words" — one entry per resource, after reading it yourself |
| `INTERVIEW_DEFENSE/` | "Explain Back" in all 8 files, including the new whole-project defence |

The Phase 8 checkpoint (`INTERVIEW_DEFENSE/phase8_project_defence.md`) asks for
Q1, Q3, Q4 and Q9 aloud without notes; then Q5 argued from the opposing side;
then a correction of the claim "AlignLab shows LoRA is nearly as good as full
fine-tuning" using only numbers from this repository.

---

## 14. Recommended future extensions

In order of how much each would change what the project can claim:

1. **An out-of-distribution evaluation set.** Every number here measures fit to
   a training distribution. This is the largest structural weakness by a wide
   margin, and it constrains almost every claim.
2. **The MLP-target LoRA run (~18 minutes).** Converts E20's reachability
   hypothesis from **NOT CONFIRMED** into a result — or kills it. The best
   value-per-minute experiment available.
3. **Multiple seeds** on the headline comparisons, so the 3.6% PEFT perplexity
   gap gets an error bar instead of a point estimate.
4. **A KL-budgeted DPO run.** Target a KL comparable to SFT's 0.2044 rather than
   fixing the epoch count and discovering afterwards that the policy moved 250×
   less.
5. **Length-controlled evaluation** (AlpacaEval's protocol), the only way to
   settle what the preference metrics are measuring.
6. **A rank sweep** r ∈ {4, 8, 32, 64}, to turn the SVD analysis into an
   empirical curve.
7. **A judge from a different family, pinned to a SHA, with a human-agreement
   study** on a subset.

---

## Final assessment

AlignLab is a **rigorous educational and research engineering framework**. It is
**not production-ready**, and this report does not claim it is: it produces a
1.5B model fine-tuned for one epoch on 9,499 examples, evaluated entirely
in-distribution, with single seeds throughout.

What it is good for is the thing it was built for. The pipeline runs end to end;
the measurement apparatus is careful enough to have caught four cases where two
metrics disagreed; five stated hypotheses were disproved and preserved; a widely
repeated explanation of LoRA did not survive contact with the data; and the
evaluation was honest enough to report that it could not confirm the project's
own clearest result.

**Phase 8 is complete. No further implementation phase has been started.**

**Related:** [[project-narrative]] · [[experiment-registry]] · [[limitations]] ·
[[cleanup-record]] · [[phase7-report]]
