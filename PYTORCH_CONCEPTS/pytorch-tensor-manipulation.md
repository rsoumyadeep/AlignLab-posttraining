# PyTorch: Tensor Manipulation, Masking, and Module State

**All "Actual result" blocks were produced by executing**
`PYTORCH_CONCEPTS/examples/tensor_reshaping_examples.py`
on 2026-08-29 — Windows, Python 3.11.15, **torch 2.6.0+cpu**.
Nothing here is recalled from documentation.

These are the constructs Phase 2 actually used. Concepts about RNG,
`state_dict` and floating point are in [[pytorch-rng-and-state]].

---

## Concept 1 — `view` vs `reshape` vs `transpose` vs `permute`

**What they do.** `view` reinterprets the same memory (no copy, requires
contiguity). `reshape` does the same but *falls back to a copy* when it must.
`transpose` swaps exactly **two** axes; `permute` reorders arbitrarily. Neither
transpose nor permute moves data — they change **strides**.

**Why AlignLab uses them.** The multi-head split and merge are nothing but
these four operations.

**Expected.** `transpose` returns a non-contiguous view.

**Actual.** ✅
```
x                 : (2, 3, 4)
x.view(2,12)      : (2, 12)      - reinterprets, no copy
x.transpose(1,2)  : (2, 4, 3)    - swaps TWO axes
x.permute(2,0,1)  : (4, 2, 3)    - arbitrary reorder
is_contiguous after transpose: False
```

---

## Concept 2 — Why `.contiguous()` is required before `view`

**Expected.** `view` on a transposed tensor raises.

**Actual.** ✅ It raises:
```
t.view(2,12) raised: view size is not compatible with input tensor's size and stride
t.contiguous().view(2,12): (2, 12) OK
t.reshape(2,12)          : (2, 12) OK - reshape copies when it must
```

**Why it matters.** This is *exactly* the bug in `_merge_heads` if
`.contiguous()` is omitted — the most common failure in hand-written attention.

**Common mistake.** Reaching for `reshape` to make the error disappear. It
works, but silently copies; `.contiguous().view(...)` makes the copy explicit.

**Connection.** `MultiHeadAttention._merge_heads`, pinned by
`test_head_split_merge_round_trip`.

---

## Concept 3 — The head split *is* a reshape

**Expected.** `view(B,T,H,d_k).transpose(1,2)` equals stacking explicit column
slices.

**Actual.** ✅ `identical: True`

**Why it matters.** Once you have seen this, `view(B,T,H,d_k).transpose(1,2)`
stops being magic: the head split is a **reinterpretation of the feature axis**,
nothing more. It is why `attention_pure.multi_head_attention` can implement the
same thing by slicing.

---

## Concept 4 — Broadcasting a `[T,T]` mask over `[B,H,T,T]` scores

**Actual.** ✅
```
scores: (2, 4, 5, 5)  mask: (5, 5)  result: (2, 4, 5, 5)
row 0 of head 0: [0.0, -inf, -inf, -inf, -inf]
```

The rule: trailing dimensions align, missing leading dimensions are added. One
`[T,T]` mask therefore applies to every batch and every head automatically — no
`expand`, no loop.

**Common mistake.** Building the mask as `[B,H,T,T]` "to be safe". It wastes
memory and hides the fact that it is the same mask everywhere.

---

## Concept 5 — `masked_fill(-inf)`, and why masking must precede softmax

**The single most instructive measurement in this file.**

**Actual.** ✅
```
mask BEFORE softmax: [0.2689, 0.7311, 0.0]   sum = 1.0
mask AFTER  softmax: [0.09,   0.2447, 0.0]   sum = 0.334759
```

Masking **after** the softmax leaves the row summing to **0.33**, not 1. The
output would be scaled down by a factor that varies with how many positions
were masked — i.e. by *position*, since causal masking hides more of the early
rows than the late ones. That is a spurious positional signal, and no longer a
convex combination of values.

Masking **before** gives `exp(-inf) = 0` and the survivors still sum to 1.

**Connection.** `test_masked_rows_still_sum_to_one`.

---

## Concept 6 — Which axis softmax runs over

**Actual.** ✅
```
softmax(dim=-1) row sums   : [1.0, 1.0, 1.0]
softmax(dim=-2) row sums   : [1.1746, 0.7109, 1.1146]
softmax(dim=-2) COLUMN sums: [1.0, 1.0, 1.0]
```

`dim=-1` gives each **query** a distribution over keys — what attention needs.
`dim=-2` normalises each **key** across queries, which answers a question nobody
asked and yields no convex combination.

**Common mistake.** Writing `softmax(scores, dim=1)` on a `[B,H,T,T]` tensor —
that normalises over the *head* axis. It runs, it trains, and it is wrong.

**Connection.** `test_weights_form_a_distribution_over_keys`.

---

## Concept 7 — `register_buffer` vs `nn.Parameter` vs a plain attribute

**Actual.** ✅
```
parameters      : ['w']
buffers         : ['persistent_mask', 'temp_table']
state_dict keys : ['w', 'persistent_mask']
```

Three distinct categories, and the difference is not cosmetic:

| | trained | moved by `.to(device)` | in `state_dict` |
|---|---|---|---|
| `nn.Parameter` | ✅ | ✅ | ✅ |
| buffer, `persistent=True` | ❌ | ✅ | ✅ |
| buffer, `persistent=False` | ❌ | ✅ | ❌ |
| **plain attribute** | ❌ | **❌** | ❌ |

**The trap.** `self.mask = torch.tril(...)` as a plain attribute is **not moved
by `.to("cuda")`**, so it produces a device-mismatch error that appears *only on
GPU* — invisible in CPU tests.

**Connection.** `SinusoidalPositionalEncoding.pe` is persistent (cheap, worth
saving); RoPE's `cos_cached`/`sin_cached` are **non-persistent** (recomputable
from `head_dim` and `base`, so they should not bloat checkpoints). Pinned by
`test_sinusoidal_is_a_buffer_not_a_parameter` and
`test_rope_tables_are_non_persistent_buffers`.

*Note:* the `.to(device)` half of this was **NOT TESTED locally** — no usable
CUDA. The example prints a placeholder on CPU.

---

## Concept 8 — `torch.multinomial` respects zero probabilities

**Actual.** ✅ 500 draws from `[0, 0.5, 0, 0.5]` returned indices `[1, 3]` only.

**Why it matters.** This is what makes top-k/top-p work: `-inf` logits become
probability exactly 0, and a zero-probability token is never sampled. If it were
merely *small*, filtering would be a suggestion rather than a guarantee.

**Connection.** `test_sampling_never_selects_a_filtered_token` — 200 draws with
`top_k=2`, never anything outside the nucleus.

---

## Concept 9 — `cross_entropy` takes **logits**

**Actual.** ✅
```
cross_entropy(logits)          : 0.41703    CORRECT
cross_entropy(softmax(logits)) : 0.802121   WRONG (double softmax)
```

**Why it matters.** `F.cross_entropy` applies `log_softmax` internally. Passing
softmax output applies it twice: the loss is wrong (0.80 vs 0.42 here), and —
worse — it still *trains*, just badly. Nothing errors.

Doing it internally is also numerically better: `log_softmax` is computed
stably, whereas `log(softmax(x))` can underflow to `log(0) = -inf`.

**Connection.** `DecoderOnlyTransformer.forward` passes raw logits to
`F.cross_entropy`. `test_untrained_loss_is_near_ln_vocab` would catch a double
softmax — the loss would not land near ln(V).

---

## Not covered here

`torch.compile`, DDP, distributed samplers, gradient accumulation, mixed-precision
autocast, quantization. None is used by Phase 2 code, so writing them up would
document an imagined implementation. **DEFERRED** to the phases that use them.

**Related:** [[pytorch-rng-and-state]] · [[phase2-code-explanation]]
