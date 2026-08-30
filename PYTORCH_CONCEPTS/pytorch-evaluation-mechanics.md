# PyTorch constructs used in evaluation

**Phase 7.** Every number below is produced by
`PYTORCH_CONCEPTS/examples/evaluation_mechanics_examples.py`, run on CPU:

```
PYTHONPATH=src python PYTORCH_CONCEPTS/examples/evaluation_mechanics_examples.py
```

The constructs collected here share one property: **getting them wrong does not
raise.** The evaluation completes, the numbers look plausible, and they are
wrong. That is what makes them worth a note.

---

## 1. `.eval()` and `torch.no_grad()` are two different switches

Measured, same input twice through a module containing `Dropout(p=0.5)`:

| mode | `max|a − b|` | identical |
|---|---:|---|
| `train()` | **0.614458** | False |
| `eval()` | 0.000000 | True |

In `train()` mode dropout resamples on every forward, so **the same input gives
different outputs**. An evaluation run in `train()` mode reports noise.

`torch.no_grad()` is unrelated to this:

| | `requires_grad` | `grad_fn` | values |
|---|---|---|---|
| without | True | `AddmmBackward0` | — |
| with | False | `None` | **identical** |

`no_grad` does not change a single number. It stops the autograd graph being
built, which frees the activation memory backward would have needed. It **does
not** put the model in eval mode.

> Two switches. Remembering only `no_grad` is the quiet bug — you get the
> memory saving and keep the stochasticity.

**Where this is enforced:** `alignlab.dpo.verify_reference_is_frozen()` refuses
to train if the reference model is left in `train()` mode, because its
log-probabilities would be stochastic and every implicit reward noisy.
`load_model()` in `evaluate.py` returns `.eval()` unconditionally.

---

## 2. `eos_token_id` takes a **list**, and it must here

Qwen2.5-1.5B base:

```
tokenizer.eos_token = '<|endoftext|>'   id 151643
ChatML terminator   = '<|im_end|>'      id 151645
```

**The base model's configured EOS is not the chat template's turn terminator.**
Generating with only `tokenizer.eos_token_id` would never stop on `<|im_end|>`,
so a model that *had* learned to emit it would be scored as failing to stop.

```python
eos_ids = [i for i in {stop_id, tokenizer.eos_token_id} if i is not None]
model.generate(..., eos_token_id=eos_ids)
```

Phase 7 measured what this distinguishes, on 6 fixed prompts:

| SFT | DPO b=0.1 | base | LoRA r=16 | QLoRA r=16 |
|---:|---:|---:|---:|---:|
| **6/6** | **6/6** | 0/6 | 0/6 | 0/6 |

---

## 3. "Terminated" ≠ "emitted the stop token"

These are separate fields in `TerminationStats`, and Phase 7 produced the case
that justifies the separation:

| model | terminated | emitted `<\|im_end\|>` | hit cap |
|---|---:|---:|---:|
| base | **2/6** | **0/6** | 4/6 |
| SFT | 6/6 | 6/6 | 0/6 |
| LoRA r=16 | 0/6 | 0/6 | 6/6 |

**The base model stopped twice without ever emitting the ChatML terminator** —
it hit `<|endoftext|>`, the other id in the list. A single "did it finish"
number would score base at 33% and hide that it produced the turn terminator
zero times, which is the thing SFT actually taught it.

`terminated` is computed as `n_generated < max_new_tokens`, not as "a stop token
appeared" — they answer different questions.

---

## 4. The same forward pass yields two perplexities

`completion_perplexity` returns both regions from one pass. Phase 3's measured
numbers on the base model:

| region | tokens | mean NLL | perplexity |
|---|---:|---:|---:|
| completion | 7 | 3.6295 | **37.694** |
| prompt | 25 | 6.3314 | **561.943** |

`comparable_to()` → **False**. Ratio **14.91×**.

Same model, same forward pass, one number "better" than the other by fifteen
times — and the comparison is meaningless, because they are means over
different token populations. This is why `PerplexityResult` carries `region`
and `n_tokens` rather than being a bare float.

**The denominator is the whole story.** Token-weighted (total NLL ÷ total
tokens) is not a mean of per-example means; the latter weights a 5-token answer
like a 500-token one.

---

## 5. What a small sample can and cannot resolve

Wilson 95% intervals on the actual Phase 6 and Phase 7 rates:

| case | rate | 95% interval | excludes 0.5 |
|---|---:|---|---|
| Phase 6 preference 86/184 | 0.467 | [0.397, 0.539] | False |
| Phase 7 stop, PEFT 0/6 | 0.000 | [0.000, 0.390] | **True** |
| Phase 7 stop, SFT 6/6 | 1.000 | [0.610, 1.000] | **True** |
| Phase 7 judge, SFT wins 3/4 | 0.750 | [0.301, 0.954] | False |
| Phase 7 judge, DPO wins 1/2 | 0.500 | [0.095, 0.905] | False |
| a 60/100 result | 0.600 | [0.502, 0.691] | **True** |

**The judge rows are the honest half of the Phase 7 result.** SFT beat base on
3 of 4 decided pairs and the interval still includes 0.5 — the judge did *not*
establish that SFT is better, even though perplexity and stop-token behaviour
both say it is. Reporting "75% win rate" at n=4 would be manufactured precision.

**The stop-token rows behave oppositely.** 0/6 and 6/6 each exclude 0.5, and
their intervals do not overlap, so that comparison **is** resolvable at n=6.

> A tiny sample is not automatically uninformative. It depends on how extreme
> the split is. `difference_is_resolvable()` answers that one narrow question
> and deliberately returns no p-value.

---

## 6. Sequential model loading

The suite evaluates five models in one process. Each is loaded, measured, and
dropped before the next:

```python
model = load_model(path, revision, device, dtype, cfg)   # -> .eval()
...
del model
torch.cuda.empty_cache()
```

Without the `empty_cache()`, the caching allocator holds freed blocks and the
sixth load (the 7B judge, 14.20 GiB) has less headroom than it needs on a
47.53 GiB A6000. This is memory hygiene, not correctness — but a run that dies
at the judge stage after 40 minutes of measurement is a real cost.

**Adapters are loaded unmerged** (`PeftModel.from_pretrained`, no `merge`).
Phase 4 measured bf16 merging as 12,460× less exact than fp32; evaluating a
merged adapter would measure the adapter *plus* a merge artefact and attribute
both to the training run.

---

## My Understanding

> **[USER CHECKPOINT — DEFERRED — USER EXPLAIN-BACK REQUIRED]**
>
> - You run an evaluation and get plausible numbers. Name three things that
>   could be silently wrong that would not raise an exception.
> - Why is `no_grad()` insufficient on its own?
> - The base model "terminated" on 2 of 6 prompts. Why is that not evidence it
>   learned to stop?
> - Two rates are both from n=6. One difference is resolvable and one is not.
>   What decides it?

**Related:** [[phase7-evaluation-theory]] · [[phase7-code-explanation]] ·
[[pytorch-logprob-and-kl]] · [[pytorch-rng-and-state]]
