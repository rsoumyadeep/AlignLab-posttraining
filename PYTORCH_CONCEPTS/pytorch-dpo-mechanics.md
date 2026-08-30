# PyTorch: DPO mechanics

Constructs actually used in Phase 6. Every number comes from running
`examples/dpo_mechanics_examples.py`.

---

## 1. `F.logsigmoid` vs `log(sigmoid(x))`

DPO's loss is `−log σ(z)`. The naive spelling fails where it matters most.

**Executed** (float32):

| x | `log(sigmoid(x))` | `F.logsigmoid(x)` |
|---:|---:|---:|
| 0.0 | −0.693147 | −0.693147 |
| −10.0 | −10.000046 | −10.000046 |
| −30.0 | −30.000000 | −30.000000 |
| −80.0 | −80.000000 | −80.000000 |
| **−200.0** | **−inf** | **−200.000000** |
| **−1000.0** | **−inf** | **−1000.000000** |

`sigmoid(x)` underflows to 0 somewhere between −80 and −200; `log(0) = −inf`,
and the gradient is destroyed. `logsigmoid` computes `−softplus(−x)` directly
and stays exact to −1000.

> *A correction.* An earlier draft of this note asserted that float32 underflows
> at −80. Running it showed `log(sigmoid(-80)) = -80.000000` — fine. The
> threshold had been written from expectation. It is −200 where it breaks.

**Why this is DPO's case specifically.** The sigmoid argument is strongly
negative exactly when the policy **confidently prefers the rejected response** —
the pair with the most to learn from. The naive form returns `-inf` there and
propagates `NaN`.

---

## 2. What β does

**Executed**, with the log-ratio difference fixed at 4 nats so the logit is
exactly `4β`:

| β | logit | loss | dL/dlogit |
|---:|---:|---:|---:|
| 0.001 | 0.0040 | 0.691149 | −0.499000 |
| 0.01 | 0.0400 | 0.673347 | −0.490001 |
| 0.1 | 0.4000 | 0.513015 | −0.401312 |
| 0.5 | 2.0000 | 0.126928 | −0.119203 |
| 1.0 | 4.0000 | 0.018150 | −0.017986 |
| 2.0 | 8.0000 | 0.000335 | −0.000335 |

Two readings:

- **Small β keeps the gradient near its maximum** (−0.5) but multiplies the
  *parameter* gradient by β as well, so the policy barely moves. At β = 0.001
  the loss is 0.691 — indistinguishable from `log 2 = 0.693147`, the value at
  initialisation.
- **Large β saturates.** At β = 2 the loss is 0.0003 and the gradient is
  effectively gone.

β trades *how far the policy may move* against *how much signal survives once
it has*.

---

## 3. Gradient accumulation is exactly a larger batch

**Executed**, six examples, one big batch vs one at a time:

```
batched     : [0.51366537, -0.19561927, -2.17878938, 0.75790837, -1.80753724, -2.79719079]
accumulated : [0.51366537, -0.19561927, -2.17878938, 0.75790837, -1.80753724, -2.79719079]
identical   : True
```

Bitwise identical. This is what lets AlignLab's DPO loop process **one pair per
forward** — avoiding all padding machinery, since chosen and rejected differ in
length — and still take an effective-batch-16 step.

**The catch:** equality holds only if *every* microbatch is divided by the
*same* count. A final partial window divided by the full count under-weights
those examples, which is why the loop steps only on a complete window.

---

## 4. What freezing a model actually saves

**Executed**, Qwen2.5-1.5B at 1,543,714,304 parameters, bf16:

| component | bytes/param | GiB |
|---|---:|---:|
| policy weights | 2 | 2.88 |
| policy gradients | 2 | 2.88 |
| AdamW moments (×2) | 4 | 5.75 |
| **reference weights** | 2 | **2.88** |
| reference gradients | **0** | **0.00** |
| reference optimizer state | **0** | **0.00** |

Policy side **11.50 GiB**; reference **2.88 GiB** — a **4.0×** ratio.

A frozen model costs its *weights* and nothing else. That is why DPO's second
model is far cheaper than doubling memory, and why PPO's **four** models are
still expensive: two of them are trained.

**Measured on the real run: peak VRAM 18.56 GiB**, against ~14.4 GiB for the
Phase 3 full SFT of the same model — the reference adds roughly its weight
footprint, as predicted, plus activations for the extra forward passes.

---

## 5. The wiring check that costs nothing

**Executed:**

```
policy == reference
  implicit reward, chosen  : [0.0, 0.0, 0.0]
  implicit reward, rejected: [0.0, 0.0, 0.0]
  loss                     : 0.6931471805599453
  log 2                    : 0.6931471805599453
  exact match              : True

reference off by 0.1 nats on ONE sequence:
  implicit reward, chosen  : [0.01, 0.0, 0.0]
  detected                 : True
```

The cancellation is **structural, not numerical** — it holds for any
log-probabilities, however large. That is what makes it a reliable check rather
than a coincidence, and why a reference off by 0.1 nats on a single sequence is
detected immediately.

**Confirmed on the real 1.5B run:** `max_abs_chosen_reward = 0.0`,
`loss = 0.6931471824645996`.

---

## 6. Common mistakes

1. `log(sigmoid(x))` instead of `F.logsigmoid(x)` — fails on the pairs that
   matter most.
2. Dividing microbatches by inconsistent counts during accumulation.
3. Leaving the reference in `train()` mode — dropout makes its
   log-probabilities stochastic and the implicit rewards noisy.
4. Leaving the reference trainable — it does not crash; it drifts toward the
   policy and the run optimises progressively less.
5. Assuming a second model doubles memory (it costs weights only, 4× less than
   a trained one here).
6. Reading a per-step logged statistic as a trend when it comes from a single
   pair. Phase 6's `history` records the last pair of each accumulation window,
   so a ±0.01 margin there is noise — the eval-set figures over 184 pairs are
   the reliable ones.

---

## Connection to AlignLab

| Concept | Where |
|---|---|
| `logsigmoid` for stability | `alignlab.dpo.dpo_loss` |
| β scaling the logit | `alignlab.dpo.implicit_rewards` |
| gradient accumulation | `alignlab.dpo_train`'s loop |
| frozen reference | `alignlab.dpo.verify_reference_is_frozen` |
| zero reward at init | `alignlab.dpo.verify_zero_reward_at_init` |

---

## My Understanding

> **[USER CHECKPOINT — DEFERRED — USER EXPLAIN-BACK REQUIRED]**
>
> - Why does `log(sigmoid(x))` fail precisely on the examples DPO most needs?
> - Given a fixed log-ratio difference of 4 nats, compute the loss at β = 0.1
>   and β = 1 by hand, and say which gives more gradient signal — and why that
>   is not the same question as which moves the policy more.
> - Show that accumulating N microbatch gradients equals one batched gradient,
>   and state the condition.
> - Why does a frozen reference cost 4× less than the policy here?

**Related:** [[phase6-dpo-theory-and-results]] · [[phase6-code-explanation]] ·
[[pytorch-logprobs-and-kl]]
