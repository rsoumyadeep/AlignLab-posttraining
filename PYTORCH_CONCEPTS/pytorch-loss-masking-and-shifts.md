# PyTorch: `ignore_index`, label shifting, and reduction

Concepts encountered building Phase 3's SFT pipeline. Every example in
`examples/loss_masking_examples.py` was **actually executed**; outputs below are
copied from that run, not predicted.

---

## 1. `ignore_index` — what it does

`torch.nn.functional.cross_entropy(..., ignore_index=-100)` drops every position
whose target equals `ignore_index`. Dropped positions contribute:

- **no loss** to the numerator, and
- **nothing to the denominator** under `reduction="mean"`.

The second half is the part people forget. The mean is over *surviving*
positions only, so masking does not merely remove terms — it changes what the
average is over.

**Why AlignLab uses it:** it is the entire mechanism of SFT loss masking. Prompt
tokens are set to `-100` and thereby excluded from both loss and gradient.

**Why -100 specifically:** it is PyTorch's default `ignore_index`, chosen
because no valid class index is negative.

### Minimal example — expected vs actual

```python
logits = torch.randn(1, 4, 10)
targets_all    = torch.tensor([[1, 2, 3, 4]])
targets_masked = torch.tensor([[-100, -100, 3, 4]])
```

| | expected | **actual** |
|---|---|---|
| `cross_entropy(all)` | mean over 4 | `2.709972` |
| `cross_entropy(masked)` | mean over 2 | `2.436063` |
| masked == mean of last two individually | yes | **True** (`2.436063`) |
| sum-reduction | 4 terms vs 2 | `10.839889` vs `4.872126` |

The masked loss is **not** the unmasked loss minus something — it is a
different average. Note `10.839889 / 4 = 2.709972` and
`4.872126 / 2 = 2.436063`: same numerators, different denominators.

> *Process note.* These figures were first written into this file from
> expectation and were **wrong** (2.706560 / 2.653215). Running the script
> replaced them. The discipline in PROJECT_INSTRUCTIONS §8 — never claim an
> example was executed unless it was — exists for exactly this failure mode,
> and it caught one here.

---

## 2. The shift — position `t` predicts token `t+1`

A causal LM's logits at position `t` are a distribution over the token at
`t+1`. So the loss compares

```python
shift_logits = logits[:, :-1, :]   # drop the LAST position's logits
shift_labels = labels[:, 1:]       # drop the FIRST label
```

**The consequence that bit us.** The number of positions contributing to the
loss is

```python
(labels[:, 1:] != -100).sum()      # correct
(labels     != -100).sum()         # WRONG - counts position 0
```

Position 0 has no predecessor, so its label is discarded by the shift.

**Actual, from the executed example:**

```
prompt masked (leading -100)
  labels                     : [-100, -100, 3, 4, 5]
  labelled positions         : 3
  loss-contributing positions: 3        <- position 0 was already masked

trailing mask (padding-like)
  labels                     : [7, 8, 9, -100, -100]
  labelled positions         : 3
  loss-contributing positions: 2        <- position 0 dropped by the shift
```

Note the asymmetry: leading masks are unaffected, because position 0 was going
to be ignored anyway. The discrepancy appears only when position 0 **is**
labelled — which is exactly the "everything is trained" arm.

In E12 this cost a decomposition check: 26 labelled prompt positions but 25
contributing, producing a 1.8e-02 reconstruction error that vanished (to
1.6e-07) with the right count.

### Common mistakes

1. Shifting in the wrong direction (`logits[:, 1:]` vs `labels[:, :-1]`) — still
   runs, learns nothing useful.
2. Shifting twice, once manually and once inside a model that already does it.
   Modern `transformers` models shift internally when you pass `labels=`; TRL
   passes `shift_labels` explicitly to avoid exactly this.
3. Counting labelled positions instead of contributing positions.

---

## 3. Token-weighted vs example-weighted aggregation

To compare a metric across a dataset you must decide what to average over.

```
mean of per-example means  : each example counts equally
total NLL / total tokens   : each TOKEN counts equally
```

**Actual, from the executed example** (two examples, 2 and 8 tokens, NLL 1.0 and
3.0 per token):

```
example-weighted : 2.000000
token-weighted   : 2.600000
```

They differ by 30%. Neither is wrong, but reporting one while describing the
other is. AlignLab's E13 accumulates total NLL and total tokens — token
weighting — because a 5-token answer should not count as much as a 500-token
one when the question is "how well does the model model these answers".

---

## 4. The decomposition cross-check

A masked loss over disjoint regions must reconstruct the whole, **token
weighted with the contributing counts**. This is a cheap invariant that catches
mis-counting.

**Actual, from the executed example:**

```
prompt region     : 6 counted, CE 4.080406
completion region : 5 counted, CE 4.331021
everything        : 11 counted, CE 4.194323

token-weighted reconstruction : 4.194322153
measured over everything      : 4.194322586
difference                    : 4.33e-07
counts add up (6 + 5 = 11)    : True
```

Had the counts been taken before the shift, they would have been 7 + 5 = 12
against a measured 11, and the reconstruction would have been visibly wrong.
That is precisely what happened in E12's first run.

---

## 5. `reduction` and why perplexity needs care

Perplexity is `exp(mean NLL per token)`. That requires:

- the mean is over **tokens**, not examples, and
- the tokens are the ones you actually care about.

Perplexity over a sequence that includes a masked-out prompt is a different
quantity from perplexity over the completion, and the two are not comparable —
the Phase 3 study note derives why from E12's measurements.

---

## 6. `-100` and padding

Two independent reasons a position may be ignored:

```python
labels[attention_mask == 0]  = -100   # padding
labels[completion_mask == 0] = -100   # prompt
```

Both must be applied. Masking the prompt but forgetting padding trains the model
to predict pad tokens; masking padding but forgetting the prompt is the classic
SFT bug. TRL applies both (`sft_trainer.py:772-774`).

---

## Connection to AlignLab

| Concept | Where it appears |
|---|---|
| `ignore_index=-100` | `alignlab.masking.IGNORE_INDEX`, every SFT run |
| the shift | `e12._masked_mean_ce`, `e13.completion_and_prompt_nll` |
| token-weighted aggregation | `e13` perplexity |
| padding vs prompt masking | TRL collator, verified by `verify_mask_on_real_batch` |

---

## My Understanding

> **[USER CHECKPOINT — DEFERRED — USER EXPLAIN-BACK REQUIRED]**
>
> - Why does `reduction="mean"` with `ignore_index` change the denominator, and
>   why does that make masked and unmasked losses incomparable?
> - Write the shift by hand for a 5-token sequence and say which label is lost.
> - Given per-example NLLs and token counts, compute both aggregations and say
>   which you would report for perplexity, and why.

**Related:** [[phase3-sft-and-loss-masking]] · [[pytorch-tensor-manipulation]] ·
[[phase3-code-explanation]]
