# Phase 5 — Code Explanation

Explains the code that exists. Phase 5 built **infrastructure**, not training.

**Scope, stated first:** there is **no PPO implementation** and **no trained
reward model** in AlignLab. The Bradley-Terry functions exist so the concept
has a runnable, tested referent — they are not a reward model.

---

## 1. Files

| File | Status | Responsibility |
|---|---|---|
| `src/alignlab/preference.py` | **IMPLEMENTED** | preference loading, validation, audit, fingerprint, Bradley-Terry |
| `src/alignlab/logprobs.py` | **IMPLEMENTED** | sequence log-probs, log-ratios, exact KL, the `k3` estimator |
| `tests/test_logprobs_and_preference.py` | **IMPLEMENTED** | 55 tests |
| `scripts/experiments/e21_preference_data_readiness.py` | **MEASURED** | is the data ready for DPO? |
| `scripts/experiments/e22_reference_model_and_kl.py` | **MEASURED** | KL scale, reference wiring, pre-DPO baseline |
| PPO training loop | **NOT IMPLEMENTED** | — |
| reward model training | **NOT IMPLEMENTED** | — |
| DPO loss | **NOT IMPLEMENTED** | Phase 6 |

---

## 2. Data flow

```
HuggingFaceH4/ultrafeedback_binarized  (train_prefs / test_prefs)
   │  audit_preferences               ties, identical, empty, length bias
   ▼
to_preference_triple                  {prompt, chosen, rejected}
   │  _strip_boundary_whitespace      reused from Phase 3 - BOTH responses
   │  load_from_cache_file=False      Phase 3's stale-cache lesson
   ▼
is_wellformed_preference              drops identical / empty / blank-prompt
   ▼
fingerprint_dataset                   sha256 over canonical rows
   ▼
expected_labels  [alignlab.masking]   −100 on prompt, ids on completion
   ▼
sequence_logprobs  [alignlab.logprobs]  sum & mean, AFTER the shift
   ▼
logprob_ratio                         log π_θ − log π_ref   ← DPO's implicit reward
   ▼
                                      L_DPO ← PHASE 6, NOT WRITTEN
```

---

## 3. Equation → code

| Maths | Code |
|---|---|
| `P(y_w ≻ y_l) = σ(r_w − r_l)` | `preference.bradley_terry_probability` |
| `−log σ(r_w − r_l)` | `preference.bradley_terry_loss` |
| `log π(y\|x) = Σ_t log p(y_t \| y_<t, x)` | `logprobs.sequence_logprobs(...).sum_logprob` |
| `log π_θ − log π_ref` | `logprobs.logprob_ratio` |
| `KL(p‖q) = Σ_v p(v)[log p(v) − log q(v)]` | `logprobs.token_kl` (exact, full vocab) |
| `k3 = exp(−k1) − 1 + k1` | `logprobs.approximate_kl` |

**The DPO loss is the Bradley-Terry loss with `β·(log-ratio difference)` in
place of `r_w − r_l`.** That is why both live in this codebase already, and it
is the single most useful structural observation for Phase 6.

---

## 4. Key functions

### `logprobs.token_logprobs(logits, labels)`
Applies **the shift exactly once**, centrally, so no caller has to remember
it. Returns `(logprobs, mask)` of shape `[B, T−1]`. Masked positions are
gathered with a clamped index and then zeroed — `gather` needs a valid index
everywhere, including where the label is `-100`.

### `logprobs.sequence_logprobs(...)`
Returns **both** `sum_logprob` and `mean_logprob`, plus `n_tokens` — the count
of tokens that contributed **after** the shift, which differs from the number
of labelled positions whenever position 0 is labelled. That distinction caused
a wrong number in Phase 3's E12 and is now asserted in a test.

### `logprobs.logprob_ratio(policy, reference, length_normalise=False)`
DPO's implicit reward. Defaults to the **summed** form to match the published
objective; length normalisation is opt-in because it is a deviation from the
paper — and §6 shows it changes a headline number by 11 points.

### `logprobs.token_kl(...)` / `sequence_kl(...)`
Exact KL over the full vocabulary. `sequence_kl` applies the same shift as
`sequence_logprobs`, so KL and log-probabilities describe **the same token
positions** and can be discussed together.

### `preference.audit_preferences(dataset)`
Measures ties, inversions, identical pairs, empty responses, prompt
mismatches, multi-turn rows, score margins and length bias — and emits a
`notes` list for anything above threshold.

### `preference.load_preference_dataset(..., drop_ties=False)`
`drop_ties` defaults **False deliberately**. Dropping 11.9% of the data is
defensible but it is a *change to the data*, so it belongs in the manifest as
an explicit decision rather than a silent default.

---

## 5. Design decisions

| Decision | Alternative | Why |
|---|---|---|
| explicit `prompt` split out | pass full conversations | makes the scored/unscored boundary a single index, and lets us CHECK the prompt is shared |
| `drop_ties=False` by default | drop silently | changing the data must be a recorded decision |
| sum as the default ratio | mean | matches the published objective; the deviation is opt-in |
| exact KL for measurement | the `k3` estimator | Phase 5 measures offline where exactness is affordable; `k3` is provided for the conceptual account |
| strip whitespace from BOTH responses | chosen only | the Phase 3 boundary bug applies symmetrically |
| `load_from_cache_file=False` | trust the cache | Phase 3's cache silently served pre-fix rows |
| Bradley-Terry implemented | prose only | gives the concept a runnable, testable referent without pretending to be a reward model |
| **no PPO implementation** | build one | PROJECT_INSTRUCTIONS says Phase 5 is conceptual unless explicitly expanded; a half-built PPO would be worse than none |

---

## 6. Measurements

| Measurement | Value | Source |
|---|---|---|
| raw train rows | 61,135 (`train_prefs`) | E21 |
| score ties | **11.9%** train / 13.6% eval | E21 |
| identical chosen/rejected | 14 in 3,000 | E21 |
| prompt shared | 3000/3000 | E21 |
| token-prefix violations | **0 of 400**, both responses | E21 |
| chosen longer (tokens) | **56.5%** (283.7 vs 241.6) | E21 |
| pairs over `max_length=1024` | 26/400 (6.5%) | E21 |
| KL(SFT ‖ base) | mean 0.6547, **median 0.2044**, max 7.3349 | E22 |
| KL(LoRA ‖ base) / KL(QLoRA ‖ base) | 0.6233 / 0.5564 | E22 |
| implicit reward, policy == reference | **0.000e+00** | E22 |
| pre-DPO baseline, SUM | **47.2%** (51/108) | E22 |
| pre-DPO baseline, MEAN | **58.3%** (63/108) | E22 |

---

## 7. Bugs and corrections

### 7a. A test that mis-indexed the shift — caught on first run
`test_values_match_a_hand_computed_log_softmax` expected `picked[0] == 0` for a
leading `IGNORE_INDEX`. Wrong: a leading ignore is **dropped by the shift, not
masked**, so there are only `T−1` outputs and position 0 scores a real token.
**The module was right; the test was wrong** — which is precisely the
off-by-one `token_logprobs` exists to centralise. Fixed with the indexing
spelled out in the docstring.

### 7b. H1's criterion in E22 could not test what it claimed
The hypothesis said KL would be "small", and the code tested `0 < mean < 1`.
It passed — but the distribution is heavily right-skewed (mean 0.6547, median
0.2044, max 7.3349), so the mean is not the typical value and "small" was doing
unearned work. Now reports mean, median and max, and states the skew.

### 7c. A storage estimate that was right about the wrong thing
Pre-flight estimated **≤620 MiB** for the preference dataset — the *download*
size from the Hub API. Actual disk consumption was **+1.1 GiB**, because the
parquet files expand into an arrow cache. The estimate was not wrong so much as
measuring a different quantity. Recorded so future pre-flights budget for the
decompressed form.

---

## 8. Test map

| Claim | Tests |
|---|---|
| the shift is applied once, correctly | `TestTokenLogprobs` (7) |
| sum/mean, and the contributing-token count | `TestSequenceLogprobs` (5) |
| implicit reward is 0 when π == π_ref | `TestLogprobRatio` (3) |
| KL: zero at equality, non-negative, **asymmetric**, monotone | `TestKL` (8) |
| Bradley-Terry: only differences matter, saturation, loss shape | `TestBradleyTerry` (7) |
| prompt/chosen/rejected extraction, whitespace | `TestPreferenceConversion` (6) |
| identical/empty/blank rejected | `TestWellformed` (5) |
| audit counts ties, inversions, length bias | `TestAudit` (10) |

**55 Phase 5 tests. Suite total: 458 passed, 2 skipped.**
