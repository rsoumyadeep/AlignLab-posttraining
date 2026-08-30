# Phase 6 — Beta Sweep Pre-Registration

**Written and committed BEFORE any DPO training code was written or run.**
The git history is the evidence: this file's commit precedes every Phase 6
training commit and every result.

PROJECT_INSTRUCTIONS for Phase 6: *"Do not choose beta after inspecting the
final results. Define a small beta experiment BEFORE training."*

---

## 1. The candidate set — fixed now

```
β ∈ { 0.01, 0.1, 0.5 }
```

**Three values, chosen before seeing any result**, spanning two orders of
magnitude around the community default.

| β | rationale, stated in advance |
|---|---|
| **0.01** | weak KL anchoring. The implicit reward `β·(log π − log π_ref)` is scaled down, so the policy is freer to move far from the reference. Expected: largest behavioural change, highest risk of degeneration. |
| **0.1** | the value most commonly reported for DPO. The default in TRL's `DPOConfig` at the time of writing. Expected: the reference point. |
| **0.5** | strong anchoring. The policy is held close to the reference. Expected: smallest change from the SFT starting point, possibly indistinguishable from it. |

**If the sweep must be reduced** for compute or storage reasons, the reduction
and its reason will be recorded, and β = 0.1 will be the one retained (it is
the reference point).

## 2. What β does, so the prediction is not post-hoc

The DPO loss is

```
L = − log σ( β · [ (log π_θ(y_w|x) − log π_ref(y_w|x))
                 − (log π_θ(y_l|x) − log π_ref(y_l|x)) ] )
```

β multiplies the log-ratio difference *inside* the sigmoid. It therefore
controls **how large a log-ratio gap is needed to saturate the loss**:

- **small β** → the sigmoid argument stays near 0 for a long time, gradients
  remain large, and the policy must move the log-ratios a long way before the
  loss flattens. Weak effective KL constraint.
- **large β** → a small log-ratio gap already saturates the sigmoid, gradients
  vanish early, and the policy stays near π_ref. Strong effective KL constraint.

## 3. Hypotheses — recorded before running

- **H1** All three β values reduce the training loss below its value at
  initialisation (which is exactly `log 2 = 0.693147`, because the implicit
  reward is 0 when π = π_ref).
- **H2** Preference accuracy under the **SUM** objective improves from the
  Phase 5 baseline of **47.2%** for at least one β.
- **H3** β = 0.01 produces the largest KL from the reference; β = 0.5 the
  smallest. Monotone in β.
- **H4** **Response length increases** relative to the SFT baseline for at
  least one β. Phase 5 measured chosen responses at 56.5% longer in tokens,
  and the SUM objective is length-sensitive, so "prefer longer" is a gradient
  direction DPO can exploit. *This is the hypothesis I most expect to be
  uncomfortable.*
- **H5** The **MEAN** diagnostic and the **SUM** primary metric can move in
  different directions. If they do, the disagreement is preserved, not
  resolved in favour of whichever looks better.

## 4. Primary metric — fixed now

**Preference accuracy under SUM log-probability.**

- **Pre-DPO baseline: 47.2%** (51/108), measured in Phase 5's E22.
- The **MEAN** figure of **58.3%** is a **diagnostic only**. It must never be
  presented as the published-objective baseline.

The published DPO objective uses the **SUM** of token log-probabilities, and
Phase 6 trains that. This is recorded here so the choice cannot be revised
after seeing results.

## 5. Evaluation set — fixed now

The Phase 5 preference evaluation split, unchanged:

- `HuggingFaceH4/ultrafeedback_binarized`, split `test_prefs`
- the same filtering as Phase 5 (`drop_ties=False`)
- the same seed (42) and subsample size

**The evaluation set will not be changed after results are seen.** Any change
would be recorded with before/after counts and a new fingerprint.

## 6. Decisions fixed in advance

| decision | value | why |
|---|---|---|
| objective | **SUM** | the published objective; Phase 5 §5 |
| policy start | Phase 3 SFT checkpoint | standard practice |
| reference | the **same** SFT checkpoint, frozen | standard practice |
| ties | **kept** (`drop_ties=False`) | unchanged from Phase 5; changing it would change the fingerprint |
| seed | 42 | unchanged across phases |
| training budget | identical across all β | otherwise β is confounded with budget |
| checkpoint retention | final model only, no intermediates | 3 runs × 2.9 GiB on a 99%-full shared volume |

## 7. What would count as a negative result

Stated in advance so it cannot be reinterpreted later:

- Training loss falls but SUM preference accuracy does not exceed 47.2% →
  **DPO optimised its objective without improving the measured preference**.
- SUM accuracy improves while response length grows sharply →
  **the gain is confounded with length** and must be reported as such, not as
  a quality improvement.
- Qualitative outputs degrade while metrics improve →
  **the disagreement is the finding**, and both are reported.

---

**No DPO code existed when this file was written. No training had been run.**
