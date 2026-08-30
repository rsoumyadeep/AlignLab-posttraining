# Phase 6 — Code Explanation

Explains the code that exists. **The DPO loss and the training loop are both
ours**; TRL's `DPOTrainer` is not used.

---

## 1. Files

| File | Responsibility |
|---|---|
| `src/alignlab/dpo.py` | **the algorithm** — implicit rewards, DPO loss, stats, the two safety verifications |
| `src/alignlab/dpo_train.py` | the training loop that uses it |
| `configs/dpo.yaml`, `configs/dpo/{default,smoke}.yaml` | the pre-registered decisions, encoded |
| `tests/test_dpo.py` | 35 tests |
| `scripts/experiments/e23_dpo_evaluation.py` | SFT vs DPO across the beta sweep |
| `docs/phase6/BETA_PREREGISTRATION.md` | committed **before** any DPO code existed |

---

## 2. Equation → code

The loss:

```
L = − log σ( β·[ (log π_θ(y_w|x) − log π_ref(y_w|x))
               − (log π_θ(y_l|x) − log π_ref(y_l|x)) ] )
```

| Maths | Code |
|---|---|
| `log π(y\|x)` | `logprobs.sequence_logprobs(...).sum_logprob` |
| `Δ = log π_θ − log π_ref` | inside `dpo.implicit_rewards` |
| `r = β·Δ` | `dpo.implicit_rewards(...)` → `(chosen, rejected)` |
| `r_w − r_l` | `logits = chosen_rewards - rejected_rewards` |
| `−log σ(·)` | `-F.logsigmoid(logits)` |
| `L` at init | `dpo.LOSS_AT_INIT = log 2` |

**`logsigmoid`, not `log(sigmoid(...))`.** The latter underflows to `-inf` when
the argument is strongly negative — which happens exactly when the policy
confidently prefers the *rejected* response, the case where the gradient
matters most. Asserted by a test that feeds it a −1000 log-probability gap.

---

## 3. Data flow

```
UltraFeedback train_prefs  (Phase 5's audited loader, fingerprinted)
   ▼
to_preference_triple → {prompt, chosen, rejected}, whitespace stripped
   ▼
expected_labels  ×2      −100 on prompt, ids on completion
   │                     over-length pairs DROPPED, not truncated
   ▼
   ├── POLICY    forward ×2   [with grad]     ─┐
   └── REFERENCE forward ×2   [no grad, eval] ─┤
                                               ▼
                          sequence_logprobs ×4  (SUM)
                                               ▼
                          implicit_rewards → β·Δ
                                               ▼
                          −logsigmoid(r_w − r_l)
                                               ▼
                          backward on the POLICY only
```

### Why over-length pairs are dropped, not truncated

Truncation removes tokens from the **tail** — which for a prompt/completion
pair is exactly the completion DPO scores. Worse, chosen and rejected have
different lengths, so truncating both at a fixed limit removes *different
amounts* from each, making their log-probabilities incomparable. Dropping is
the honest choice; the count is recorded in the summary.

### Why one pair per forward

Chosen and rejected differ in length. Batching them needs padding plus a mask,
and the interaction between padding, the completion mask and the shift is the
machinery this phase exists to make legible. Gradient accumulation reaches the
effective batch instead. **Throughput is not the objective here** — this is
recorded as a deliberate trade, not an oversight.

---

## 4. The two verifications that refuse to train

Both run **before the first optimiser step**, and both `raise` rather than warn.

### `verify_reference_is_frozen(reference)`
Asserts zero trainable parameters **and** `eval()` mode. A trainable reference
does not crash — it drifts toward the policy, the implicit rewards shrink
toward zero, and the run optimises progressively less. A reference left in
`train()` mode would apply dropout, making its log-probabilities stochastic and
the implicit rewards noisy.

### `verify_zero_reward_at_init(...)`
Policy and reference start from the **same** checkpoint, so every implicit
reward must be **exactly 0** and the loss exactly `log 2 = 0.6931471805599453`.

This is the check that catches a reference wired to the wrong checkpoint, or a
sum/mean mismatch between the two branches — bugs that otherwise train happily
while optimising the wrong objective. Phase 5's E22 established the property;
this re-verifies it *inside the DPO code path*, which is the thing that
actually matters.

**Measured on the real run:** `max_abs_chosen_reward = 0.0`,
`loss = 0.6931471824645996`, both checks pass.

---

## 5. SUM vs MEAN, in code

`implicit_rewards(..., length_normalise=False)` is the default and uses
`sum_logprob`. That is the **published** objective and what Phase 6 trained.

`length_normalise=True` switches to `mean_logprob`. It exists so the diagnostic
can be computed, and it is never the primary metric. Substituting it silently
would optimise a different objective under the same name — and Phase 5 measured
how different: 47.2% vs 58.3% preference accuracy on identical models and data.

A unit test reproduces that directly: a long chosen response with a **better
per-token mean** still loses on the SUM, purely because it has more tokens.

---

## 6. `reward_accuracy` vs `preference_accuracy` — deliberately different

| function | compares | involves the reference? |
|---|---|---|
| `dpo_loss(...).reward_accuracy` | implicit rewards `β·Δ` | **yes** |
| `preference_accuracy(...)` | raw policy log-probabilities | no |

Both are kept because they can disagree. A policy can improve its
implicit-reward ranking while its raw log-probability ranking barely moves —
`preference_accuracy` is the one comparable to Phase 5's 47.2% baseline, and
conflating them would hide that.

---

## 7. Design decisions

| Decision | Alternative | Why |
|---|---|---|
| own loss + own loop | TRL `DPOTrainer` | the phase requires the core computation be independently understandable and tested |
| SUM default | MEAN | the published objective; pre-registered |
| one pair per forward | padded batching | avoids masking machinery that would obscure the loss |
| drop over-length pairs | truncate | truncation removes exactly the scored tokens, asymmetrically |
| refuse on a non-zero init reward | warn | a mis-wired reference otherwise trains silently |
| final model only, no intermediates | keep checkpoints | 3 runs × 2.9 GiB on a 99%-full shared volume |
| lr 5e-7 | SFT's 2e-5 | DPO starts from a tuned model; convention, **not** a tuned value — no LR sweep was run |
| baseline measured in-process | reuse Phase 5's number | same batches, same code path, no cross-run drift |

---

## 8. Bugs and corrections

### 8a. A test threshold that was a guess
`test_larger_beta_saturates_sooner` asserted `loss < 0.01` at β=1. The true
value is **0.0181** — the log-ratio difference is fixed at 4, so the loss is
exactly `−logsigmoid(4β)`. Replaced the guess with the closed-form values at
each β. **The code was right; the assertion was invented.**

*(Further entries added as the phase proceeds.)*

---

## 9. Test map

| Claim | Tests |
|---|---|
| reward is `β·(log π − log π_ref)`; zero when equal; β scales linearly | `TestImplicitRewards` (4) |
| loss at init is exactly `log 2`; direction; β saturation; stability; gradients | `TestDPOLoss` (10) |
| reward accuracy, margin, token counts recorded | `TestStats` (3) |
| **SUM and MEAN can disagree** | `TestPreferenceAccuracy` (3) |
| reference must be frozen and in eval mode | `TestReferenceFrozen` (3) |
| zero reward at init; detects a mismatched reference | `TestZeroRewardAtInit` (2) |
| the pre-registered decisions are encoded in config | `TestDPOConfigComposition` (10) |

**35 Phase 6 tests. Suite total: 493 passed, 2 skipped.**
