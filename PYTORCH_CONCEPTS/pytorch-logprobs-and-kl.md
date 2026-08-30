# PyTorch: sequence log-probabilities, KL, and preference losses

Constructs actually used in Phase 5. Every number below comes from running
`examples/logprob_and_kl_examples.py`; the numbers were **not** written from
expectation. (Phase 3 taught that lesson the hard way — a note there once
carried figures drafted before the script was run.)

---

## 1. `cross_entropy` *is* negative log-probability

For picking out the log-probability of a **specific** token, two routes give
the identical answer:

```python
log_softmax(logits, -1).gather(2, targets.unsqueeze(-1))   # log p(target)
-F.cross_entropy(logits, targets, reduction="none")        # same thing
```

**Executed:**

```
token_logprobs : [-2.107028, -2.337189, -1.501814]
-cross_entropy : [-2.107028, -2.337189, -1.501814]
identical      : True
```

Worth internalising so you never compute one when you meant the other — the
sign error is silent and trains fine.

---

## 2. The shift, and the two different "token counts"

Position `t`'s logits predict token `t+1`, so scoring uses `logits[:, :-1]`
against `labels[:, 1:]`. That creates two counts that are **not** the same:

**Executed:**

```
prompt-masked (leading ignore)
  labels                : [-100, -100, 3, 4, 5, 6]
  labelled positions    : 4
  CONTRIBUTING positions: 4

padding-masked (trailing)
  labels                : [1, 2, 3, 4, -100, -100]
  labelled positions    : 4
  CONTRIBUTING positions: 3      <- position 0 dropped by the shift
```

The asymmetry: a **leading** mask costs nothing (position 0 was ignored
anyway), but a labelled position 0 **is** dropped. Use
`(labels[:, 1:] != -100).sum()` as the denominator.

This exact confusion produced a wrong number in Phase 3's E12, and a Phase 5
*test* repeated it — the test asserted a leading `-100` survived the shift. The
module was right; the test was wrong.

---

## 3. SUM vs MEAN — why DPO's objective is length-sensitive

**Executed** (random model, 5-token vs 20-token completions):

| sequence | tokens | sum logp | mean logp |
|---|---:|---:|---:|
| short | 5 | −24.4548 | −4.8910 |
| long | 20 | −85.0982 | −4.2549 |

The sums differ by 3.48× while the means are close. A longer sequence has a
more negative summed log-probability **largely for being longer**.

**Why this is not academic.** The published DPO objective uses the **sum**, and
UltraFeedback's chosen responses are 56.5% longer in tokens than its rejected
ones. E22 measured what that does: the pre-DPO preference baseline is **47.2%**
by sum and **58.3%** by mean — identical models, identical data, an 11-point
swing that straddles chance.

`alignlab.logprobs.logprob_ratio` defaults to the sum (matching the paper) and
offers `length_normalise=True` as an explicit opt-in.

---

## 4. KL: non-negative, zero at equality, **not symmetric**

**Executed:**

```
KL(p||p) = 0.000e+00
KL(p||q) = 0.214130
KL(q||p) = 0.255932
symmetric: False
```

And a deliberately extreme case — reference confident on one token, policy
flat:

```
KL(pol||ref) =   6.3907
KL(ref||pol) =   1.6074
```

**The direction is a modelling choice, not a formality.**
`KL(policy ‖ reference)` is ~4× larger here, because it heavily penalises the
policy putting mass where the reference puts almost none. That is exactly what
reward hacking looks like in distribution space, and it is why RLHF penalises
this direction rather than the reverse.

---

## 5. The sampled KL estimator PPO actually uses

Exact KL needs the full vocabulary distribution at every position — `[B, T,
151936]` for Qwen2.5-1.5B, too large to hold for a PPO batch. So
implementations estimate it from sampled tokens:

```
k1 = log π(y) − log π_ref(y)          unbiased
k3 = exp(−k1) − 1 + k1                biased low, lower variance
```

**Executed**, 200,000 samples:

```
k1 mean : -0.000031   var 0.010002
k3 mean : +0.005014   var 0.000051
variance ratio k1/k3 : 194x
k1 can be NEGATIVE: True  (min -0.4389)

k3 negatives in float32 : 11 of 200,000  (min -2.980e-08)
k3 negatives in float64 : 0 (min 2.842e-14)
```

`k1` is unbiased but has **194× the variance** and is genuinely negative for
individual samples — awkward for something meant to be a divergence. `k3` is
biased low, far tighter, and non-negative.

### A correction I had to make

I first wrote that `k3` is *always* non-negative. The executed check said
otherwise, so I worked it out: `f(x) = e^{−x} − 1 + x` has `f(0) = 0` and
`f'(x) = 1 − e^{−x}`, so `x = 0` is its **minimum** and `f(x) ≥ 0` for all real
`x`. It **is** non-negative mathematically. The 11 float32 negatives are
catastrophic cancellation when `|k1|` is tiny (magnitude ~3e−08), and they
vanish entirely in float64.

**The practical point:** don't assert `kl >= 0` with a strict comparison in a
training loop. Use a tolerance.

---

## 6. Bradley-Terry: only differences, and saturation

**Executed:**

| r_w | r_l | P(w≻l) | loss |
|---:|---:|---:|---:|
| 0 | 0 | 0.500000 | 0.693147 |
| 1 | 0 | 0.731059 | 0.313262 |
| 2 | 0 | 0.880797 | 0.126928 |
| 5 | 0 | 0.993307 | 0.006715 |
| **101** | **100** | **0.731059** | **0.313262** |
| 0 | 1 | 0.268941 | 1.313262 |

Two things to read off:

- **(1,0) and (101,100) are identical.** Only the difference is identified, so
  preference data can never pin down an absolute reward scale. A reward model's
  raw output is not interpretable on its own, and RM scores are not comparable
  across runs.
- **At margin 5 the loss is 0.0067.** Confidently-ordered pairs stop
  contributing gradient.

Note `loss(0,0) = 0.693147 = log 2` — the maximum-entropy point.

---

## 7. The check to run before any DPO training

**Executed:**

```
policy == reference -> implicit reward: [0.0, 0.0, 0.0]
all exactly zero: True
policy favours these sequences -> [17.437, 18.369, 16.841]
all positive: True
```

DPO's implicit reward `β·(log π_θ − log π_ref)` is **exactly zero** when policy
and reference are the same model. E22 confirmed this on the real SFT
checkpoint: `0.000e+00`.

It costs nothing and catches a reference plumbed to the wrong checkpoint — a
bug that otherwise appears as a run that trains happily while optimising the
wrong objective.

---

## Common mistakes

1. Sign-flipping `cross_entropy` and log-probability.
2. Counting labelled positions instead of contributing ones (§2).
3. Comparing summed log-probabilities across different lengths (§3).
4. Assuming KL is symmetric, or using the wrong direction (§4).
5. Asserting `k3 >= 0` strictly in float32 (§5).
6. Treating a reward model's absolute score as meaningful (§6).
7. Not checking that the reference is the model you think it is (§7).

---

## Connection to AlignLab

| Concept | Where |
|---|---|
| the shift, once and centrally | `logprobs.token_logprobs` |
| sum vs mean, contributing counts | `logprobs.sequence_logprobs` |
| implicit reward | `logprobs.logprob_ratio` |
| exact KL / `k3` | `logprobs.token_kl` / `approximate_kl` |
| Bradley-Terry | `preference.bradley_terry_{probability,loss}` |
| measured on real models | `scripts/experiments/e22` |

---

## My Understanding

> **[USER CHECKPOINT — DEFERRED — USER EXPLAIN-BACK REQUIRED]**
>
> - Write the shift by hand for a 6-token sequence with a trailing pad mask,
>   and give both token counts.
> - Why does the SUM vs MEAN choice interact with a length bias in the data?
>   Which would you optimise, and what would you have to report?
> - Show that `f(x) = e^{−x} − 1 + x ≥ 0`, and explain why float32 still
>   produced negatives.
> - Why can a reward model's absolute output not be compared across runs?

**Related:** [[phase5-rlhf-ppo-and-dpo]] · [[phase5-code-explanation]] ·
[[pytorch-loss-masking-and-shifts]]
