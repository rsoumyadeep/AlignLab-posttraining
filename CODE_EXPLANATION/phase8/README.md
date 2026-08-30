# Phase 8 — Code Explanation

Engineering and finalisation. **No new algorithm.** Three code changes, one
new module, and a set of documents that make the existing work legible.

---

## 1. What changed in the source tree

| File | Change |
|---|---|
| `src/alignlab/provenance.py` | **new** — mechanical audit of what each run recorded |
| `src/alignlab/logprobs.py` | removed `score_sequences` (dead, and a padding trap) |
| `src/alignlab/evaluation.py`, `evals/__init__.py` | docstrings disambiguating two similar names |
| `tests/test_provenance.py` | **new** — 20 tests |
| `tests/test_eval_metrics.py` | +2 — the rendered dashboard must be ASCII |

Phase 8 also carries forward the two Phase 7 fixes that were still fresh:
`configure_hf_cache` ordering, and per-family dataset fingerprints.

---

## 2. `alignlab.provenance` — why a module and not a paragraph

The Phase 8 brief asks whether every experiment can be reconstructed from
recorded metadata. Prose in a README claiming "we record provenance" is exactly
the kind of statement that rots silently — and Phase 7 supplied the cautionary
case, shipping a dashboard with `dataset_fingerprint: null` in all five records
while the surrounding documentation described provenance as complete.

So the answer is a program:

```bash
python -m alignlab.provenance          # exit 1 if any applicable field is missing
python -m alignlab.provenance --json   # machine-readable
```

### Design

```
REQUIRED_FIELDS   14 fields: git commit, dirty flag, model revision, dataset
                  and eval fingerprints, seed, config, config digest, hardware,
                  software versions, dtype, training params, eval params,
                  checkpoint identity

audit_manifest    one loaded artefact -> RunProvenance
audit_runs        a directory of run directories -> [RunProvenance]
render_...        an ASCII table
```

Two artefact shapes are understood, because the project produces two:
`run_manifest.json` (training) and `evaluation_dashboard.json` (evaluation).

### The three rules that make the output trustworthy

**Locators never invent a fallback.** A locator returns the recorded value or
`None`. Returning `"unknown"` would report a gap as if it were filled, and
there is a test asserting a nulled field stays `NOT_RECORDED` with no summary.

**A historical gap is reported, never repaired.** Nothing infers a missing
commit from a timestamp or a dtype from a file size. An old run that didn't
record its seed is a run whose seed is unknown.

**`NOT_APPLICABLE` is separated from `NOT_RECORDED`.** An evaluation pass has no
training parameters; a wiring check that never loaded a model has no dataset
fingerprint. Reporting those as gaps would pad the report and teach the reader
to skim past the real one.

### What it found

Run over all 18 real artefacts on the server: **17 complete, 1 gap** —
`eval-full-001`'s `dataset_fingerprint`, the known Phase 7 bug, with
`eval-cachefix-001` (run after the fix) clean.

**That is the auditor confirming the fix rather than me asserting it.**

### Two corrections to the auditor itself

Both were false positives, and both would have made the report useless:

1. It demanded a hub revision from **every** evaluated model. A fine-tuned
   checkpoint is a local directory and has none — the *base* revision is what
   pins the lineage. Fixed; a test covers a checkpoint with `revision: None`.
2. It reported **8 Phase 1 infrastructure runs** as missing model revisions and
   fingerprints. Those runs never loaded a model. Now detected from artefact
   **content** (`notes.model_id` / `notes.dataset` absent) rather than from the
   run's name — a name-based rule would mis-classify the next run somebody calls
   "smoke". A separate test asserts the exemption does not swallow a genuine gap.

---

## 3. Removing `score_sequences`

**WHY.** The dead-code scan found exactly one unreferenced public symbol in the
package. It had been written in Phase 5 as a convenience wrapper and never
called.

**WHAT CHANGED.** Deleted, with a comment in its place recording what it was.

**WHAT MUST REMAIN IDENTICAL.** Everything — nothing called it.

It was removed rather than kept because it was also a trap:

```python
attention_mask = torch.ones_like(input_ids)   # correct ONLY for an unpadded batch
```

Fed a padded batch it would attend to pad tokens and return quietly wrong
log-probabilities — the precise class of silent numerical error Phases 3–7 were
spent building guards against. `dpo_train.py` and `evals/runners.py` already
compose the pieces explicitly with the real mask.

---

## 4. `evaluation` vs `evals` — documented, not renamed

**WHY.** Two modules with near-identical names doing different jobs:

| | |
|---|---|
| `alignlab.evaluation` | the Phase 1 evaluator **protocol** — a registry of named evaluators driven by `train.py`, where a failing evaluator is recorded `NOT_TESTED` rather than aborting the run |
| `alignlab.evals` | the Phase 7 metrics **subsystem** |

**WHAT CHANGED.** Docstrings at both ends spelling out the distinction.

**WHAT MUST REMAIN IDENTICAL.** All of it. Nothing was renamed.

**Why not rename.** `alignlab.evaluation` is imported by completed Phase 3–6
training code. Churning working experimental paths for a cosmetic gain is what
the Phase 8 brief rules out. The ambiguity is real, so it is documented where
each reader will hit it.

---

## 5. The ASCII guarantee

`evaluate.py` ends with `print(text)`. On Windows, `print()` to a cp1252 console
raises `UnicodeEncodeError` on non-ASCII — a lesson Phase 6 learned the hard
way. The dashboard renderer was written ASCII-only in response.

That was an unenforced convention until now. Two tests:

- a rendered `Dashboard` encodes as ASCII;
- **the committed `eval-full-001` dashboard** — the text that was actually
  printed — encodes as ASCII.

The second matters more: it pins the real artefact, not a synthetic one.

---

## 6. What the audit checked and did NOT change

Checked clean, no action taken: bare excepts (both defensive and documented),
mutable default arguments (none), TODO/FIXME markers (none), hard-coded paths in
configs outside `env/` (none), committed secrets (none), broken README links
(none), unreferenced repository paths in the README (none).

**Found and recorded rather than fixed:** the local and server environments have
**drifted apart** — torch 2.12.1 vs 2.6.0, transformers 5.8.1 vs 5.16.1, Python
3.13 vs 3.11 — while the README still claimed both ran torch 2.6.0. The README
is corrected. The drift itself is **not** fixed, because every result in the
project came from the server and each run manifest pins that machine's exact
versions; aligning a local machine that only builds docs and runs CPU tests
would be churn without a reproducibility gain. It is recorded as a **Tier B**
weakness in `README.md` §17.

---

## 7. Test count

| | local | server |
|---|---|---|
| before Phase 8 | 585 passed / 3 skipped | 587 passed / 1 skipped |
| after Phase 8 | **607 passed / 3 skipped** | see `docs/phase8/PHASE_8_REPORT.md` |

The local/server difference is CUDA, `SIGUSR1` and `wandb`, which exist only on
the server.

**Related:** [[phase8-report]] · [[phase7-code-explanation]] ·
[[experiment-registry]]
