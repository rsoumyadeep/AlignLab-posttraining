# Phase 3 — Supervised Fine-Tuning and Loss Masking

**Status:** engineering complete, USER explain-back DEFERRED.

---

## 1. What SFT actually is

Pretraining teaches next-token prediction over a corpus. SFT continues exactly
the same objective — cross-entropy on next-token prediction — but changes two
things:

1. **the data**: instruction/response pairs instead of raw text
2. **which tokens carry gradient**: only the response

The second is the entire difference between "a model that continues text" and
"a model that answers". It is not a different loss function. It is the same
loss with most of its terms deleted.

```
pretraining :  loss over every token
SFT         :  loss over the answer tokens only
```

That is why loss masking, and not the optimiser or the learning rate, is the
thing Phase 3 verifies before anything else.

---

## 2. The mechanism, concretely

For one example, the sequence handed to the model is

```
<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
<|im_start|>user\nWhat is 2+2?<|im_end|>\n
<|im_start|>assistant\n              <- prompt ends HERE (the boundary)
It is 4.<|im_end|>\n                 <- only these tokens carry loss
```

Labels are built by copying `input_ids` and overwriting the prompt region with
`-100`:

```
labels[i] = -100          for i <  boundary   (ignored)
labels[i] = input_ids[i]  for i >= boundary   (trained)
```

`-100` is `torch.nn.CrossEntropyLoss`'s default `ignore_index`: those positions
contribute no loss and no gradient.

**Measured on the real tokenizer** (E12, arm A): boundary at index 26 of 33
tokens, 7 tokens active (21.2%), active region decodes to exactly
`'It is 4.<|im_end|>\n'`.

### The generation prompt belongs to the prompt

`<|im_start|>assistant\n` is *conditioning*, not a target. The model must be
given it, never asked to predict it. Putting it on the wrong side of the
boundary is the most common chat-SFT bug and moves the boundary by three
tokens.

### The shift

Position `t`'s logits predict token `t+1`. So the loss uses `logits[:, :-1]`
against `labels[:, 1:]`. A consequence that is easy to get wrong, and that
produced a wrong number in E12's first run:

> the number of positions that contribute to the loss is
> `(labels[:, 1:] != -100).sum()`, **not** `(labels != -100).sum()`.

Position 0 has no predecessor, so its label is discarded by the shift. With 26
labelled prompt positions only 25 contribute. Using the wrong count made a loss
decomposition fail to reconstruct by 1.8e-02; with the right count it agrees to
1.6e-07.

---

## 3. Why a broken mask is dangerous rather than merely wrong

Nothing crashes. The model trains. The loss curve looks plausible.

Phase 2's **E2** established the pattern for causal masking: deleting the mask
improved training loss **33×** and destroyed generation. A loss drop is not
evidence of learning.

Phase 3's **E12 arm D** tested the analogous claim for loss masking — and
**disproved the hypothesis we started with**, which turned out to be the more
useful outcome.

**Predicted:** the prompt is *easier* than the answer for a pretrained model,
so an unmasked loss would look better.

**Measured**, base Qwen2.5-1.5B, bf16, on the controlled example:

| region | tokens counted | mean CE |
|---|---:|---:|
| prompt only (masked out) | 25 | **6.3314** |
| completion only (trained) | 7 | **3.6295** |
| everything (broken mask) | 32 | 5.7403 |

The prompt is **harder**, not easier. The reason: this is the **base** model,
which has never seen ChatML. `<|im_start|>system …` is unfamiliar to it, and
the opening tokens have almost no left context to be predicted from.

**The corrected lesson is stronger than the predicted one.** An unmasked loss
is not reliably lower *or* higher. It is a **mean over a different population
of tokens**, so it is not comparable to a masked loss in either direction.
Whether contamination flatters or penalises depends on the model and template.

> Never compare losses computed over different token populations.

The decomposition is exact once counted correctly:

```
(6.3314 × 25 + 3.6295 × 7) / 32 = 5.740331   vs measured 5.740331
```

---

## 4. Tokenization does not respect your boundary

TRL builds its mask as

```python
completion_mask = [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))
```

which is correct **only if** tokenizing the prompt alone yields a genuine
prefix of tokenizing prompt+completion. BPE merges across boundaries, so this
is not guaranteed — and on real data it failed.

**Measured:** 3 of 200 no_robots rows (1.5%). The ChatML prompt ends with
`<|im_start|>assistant\n` — token `198`, a lone newline. When a completion
*starts* with newlines, BPE merges them:

```
prompt alone :  ... 151644, 77091, 198          '<|im_start|>','assistant','\n'
prompt+compl :  ... 151644, 77091, 1406, ...    '<|im_start|>','assistant','\n\n\n'
```

Token `198` becomes token `1406`. The boundary now lands on a token that is
half prompt and half answer.

TRL detects this and warns (`Mismatch between tokenized prompt and the start of
tokenized prompt+completion`) but still uses the length-based boundary.

**Fix:** strip leading/trailing whitespace from completions. A fix, not a
workaround — those newlines are redundant with the newline the template already
emits, and training on them teaches the model to open answers with blank lines.
After the fix: **0 inconsistent rows** out of 300 sampled.

### The fix appeared not to work — and that was a second bug

Rerunning still found the same three indices. The function was correct when
called directly. `datasets.map()` had **cached its output on disk** and did not
invalidate when the mapping function changed.

A silently stale cache is worse than a slow one: it makes a real fix look
ineffective, and in the other direction would let a reverted change appear
still applied. `load_from_cache_file=False` now forces recomputation. The
content fingerprint changed from `b7dff71c…` to `dc6fd746…`, which is how we
know the data actually changed.

---

## 5. Truncation silently deletes the training signal

`max_length` truncates the **tail**. With completion-only masking the tail is
exactly where the trainable tokens live. An example truncated hard enough keeps
its whole prompt, loses its answer, and contributes **zero gradient** while
still occupying a batch slot and diluting the mean.

This is why `audit_truncation` counts examples with zero active tokens rather
than assuming there are none.

**Measured**, full run at `max_length=1024`: 100 of 9,436 examples (1.06%) at
the limit, **0 examples with zero trainable tokens**.

---

## 6. The base model's chat template is a trap

The base (non-Instruct) Qwen2.5-1.5B **ships a ChatML `chat_template`** even
though it was never trained on ChatML. So `apply_chat_template` runs happily
and produces text the model has never seen.

Worse, the terminators disagree:

| | token | id |
|---|---|---|
| `tokenizer.eos_token` | `<\|endoftext\|>` | 151643 |
| template's turn terminator | `<\|im_end\|>` | 151645 |

Both are already in the vocabulary (`additional_special_tokens`), so no resize
is needed. Because the template puts `<|im_end|>` *inside* the trainable
completion, SFT does teach the model to emit it — but **generation must be told
to stop on it**, which is why E13 passes both ids as `eos_token_id`.

### What we could not do

`assistant_only_loss=True` would train every assistant turn in a multi-turn
conversation. It requires a chat template carrying `{% generation %}` markers.
Qwen2.5's stock template does **not** have them, and TRL refuses:

> The chat template is not training-compatible (missing prefix-preservation or
> `{% generation %}` markers)

An earlier check of ours wrongly concluded the template *did* support it, by
substring-matching the word "generation" — which appears only as
`add_generation_prompt`. **A grep is not a parser.**

Recorded as **NOT TESTED**. Phase 3 uses single-turn prompt/completion, where
the boundary is one index and verifiable by hand. Measured cost: ~92% of
no_robots rows are already two-turn, so little is lost.

---

## My Understanding

> **[USER CHECKPOINT — DEFERRED — USER EXPLAIN-BACK REQUIRED]**
>
> **Checkpoint 13 — Which tokens contribute to the SFT loss, and why?**
> Starting from a raw instruction/response pair, narrate every step to the
> labels tensor: template rendering, tokenization, the boundary, `-100`, the
> shift. Say which axis the shift acts on and why position 0 is dropped.
>
> Then, without looking:
> - A colleague reports their SFT loss is 0.8 while yours is 2.1 on the same
>   data and model. What do you check first, and why is "theirs is better" not
>   the default conclusion?
> - Why is training on prompt tokens harmful rather than merely wasteful?
> - You mask the prompt but forget to mask padding. What happens?

## Explain Back

> **[USER CHECKPOINT — DEFERRED — USER EXPLAIN-BACK REQUIRED]**
>
> **Checkpoint 14 — The tokenization boundary.**
> Explain why `tokenize(prompt)` is not guaranteed to be a prefix of
> `tokenize(prompt + completion)`, give the concrete newline example, and say
> what a boundary-based mask does wrong when the property fails. Then propose
> two fixes other than stripping whitespace, and say what each costs.
>
> **Checkpoint 15 — Truncation.**
> Why does truncation interact with completion-only masking in a way it does
> not with plain language-model training? Construct an example that trains on
> nothing at all.

**Related:** [[phase2-attention-from-first-principles]] · [[phase3-code-explanation]] ·
[[phase2-decoder-only]] · [[interview-phase3-sft]] · [[pytorch-loss-masking-and-shifts]]
