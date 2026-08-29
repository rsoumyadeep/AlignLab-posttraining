# Architecture Families, and Why Transformers Replaced RNNs

**WP8.** Terminology and placement — the questions that get asked early in
interviews and are easy to answer *approximately* and hard to answer *precisely*.

---

## 1. Three families, one difference: what may attend to what

| | Encoder-only | Decoder-only | Encoder–decoder |
|---|---|---|---|
| Attention | **bidirectional** | **causal** | bidirectional encoder + causal decoder + **cross-attention** |
| Sees | whole sequence | past only | source fully; target causally |
| Objective | masked LM | next-token | seq2seq |
| Example | BERT | GPT, Llama, **Qwen** | original Transformer, T5 |
| Good at | understanding | generation | transduction |
| Can generate? | **no** | yes | yes |

**The single distinction that generates all the others is the attention mask.**
Same blocks, same attention, same FFN — bidirectional versus causal.

## 2. BERT vs GPT

**BERT** masks ~15% of tokens and predicts them from **both** sides. Rich
representations, and *cannot generate*: it has no notion of "next", and its
training task assumes the future is visible.

**GPT** predicts the next token from the left context only. Every position is a
training signal (versus BERT's 15%), and the model is inherently generative.

**Why the field converged on decoder-only.** Three reasons worth stating:

1. **Every token is supervision.** BERT trains on 15% of positions; GPT trains on
   100%. Far better data efficiency per pass.
2. **Generation is the general interface.** Classification, extraction,
   translation and QA can all be phrased as text continuation. An encoder cannot
   be made to generate.
3. **Simplicity scales.** One stack, one objective, no cross-attention, no
   pretrain/finetune objective mismatch.

**A caveat worth keeping.** Decoder-only is not universally better. For pure
understanding tasks at fixed size, bidirectional context is a genuine advantage —
a decoder-only model reading a sentence never sees the right-hand context when
encoding the left. The field's shift is about **scaling and generality**, not
about decoders winning every benchmark.

## 3. Why Transformers replaced RNN/LSTM

### The parallelism argument — the decisive one

An RNN computes `h_t = f(h_{t−1}, x_t)`. Position *t* **cannot** be computed
before *t−1*. Training on a length-T sequence is **T sequential steps**, and no
amount of hardware removes that dependency.

A Transformer computes all positions **simultaneously** — one big matmul. Causal
masking is what makes this *valid*: it lets the model see all positions at once
during training while still only conditioning on the past.

That is the crux. GPUs are throughput machines; an architecture whose training
is inherently sequential cannot use them. *Attention Is All You Need* states the
motivation directly — "more parallelizable and requiring significantly less time
to train" — and reports **41.8 BLEU** on WMT14 EN→FR after **3.5 days on eight
GPUs**, versus far costlier prior models.

### The path-length argument

To relate tokens *i* and *j*, an RNN must carry information through `|i−j|`
sequential steps — gradients traverse a long chain and vanish or explode. LSTMs
mitigate this with gating; they do not remove it.

In attention, **any two positions are one operation apart**. The gradient path
between them has constant length regardless of distance.

### The bottleneck argument

An RNN compresses all history into a fixed-size hidden state. Attention keeps
every position accessible and *learns* what to retrieve.

### What Transformers gave up

Honest accounting, because "Transformers are better" is not the whole story:

| | RNN | Transformer |
|---|---|---|
| Training over sequence | **sequential** | **parallel** |
| Path between tokens | O(distance) | **O(1)** |
| Memory in T | **O(1)** state | **O(T²)** attention, O(T) cache |
| Inference per token | **O(1)** | O(T) — must attend to all history |
| Context | unbounded in principle | bounded by the window |

**Transformers traded memory for parallelism.** An RNN's constant-size state is
genuinely more efficient *at inference*; the Transformer wins because *training*
is where the scaling bottleneck was, and because O(1) path length makes long-range
learning actually work.

This trade is why the KV cache, GQA and Flash Attention all exist — they are
each an attempt to claw back some of what was given up.

## 4. Terminology, precisely

| Term | Meaning |
|---|---|
| **Attention** | the general mechanism: query, keys, values, weighted average |
| **Self-attention** | Q, K, V all from the **same** sequence |
| **Cross-attention** | Q from one sequence, K/V from **another** (decoder attending to encoder) |
| **Causal / masked self-attention** | self-attention restricted to positions ≤ *i* |
| **Multi-head attention** | several attention operations in parallel subspaces, concatenated |

These are **orthogonal axes**, and conflating them is a common interview stumble.
"Multi-head" describes *how many subspaces*; "causal" describes *what may be
attended to*; "self" versus "cross" describes *where K/V come from*. A layer can
be multi-head causal self-attention (a decoder block) or multi-head
non-causal cross-attention (an encoder–decoder bridge) — the labels stack.

## 5. Where our model sits

`DecoderOnlyTransformer` is **decoder-only, causal, multi-head (GQA-capable)
self-attention** — the same family as Qwen2.5, which WP9 confirms against the
real configuration.

---

## Explain Back

> **[USER CHECKPOINT — DEFERRED — USER EXPLAIN-BACK REQUIRED]**
>
> - What is the *single* architectural difference between BERT and GPT, and what
>   follows from it?
> - Why can an encoder-only model not generate text?
> - Give the three arguments for Transformers over RNNs. Which is decisive, and
>   why?
> - What did Transformers give *up* relative to RNNs? Name the mechanisms that
>   exist to claw it back.
> - Distinguish: attention, self-attention, causal attention, cross-attention,
>   multi-head attention. Which pairs can co-occur?
> - Is decoder-only strictly better than encoder-only? Defend your answer.

**Related:** [[phase2-decoder-only-and-generation]] ·
[[phase2-attention-from-first-principles]] · [[phase2-kv-cache]]
