# Interview Defence — Phase 3, Supervised Fine-Tuning

Questions that test reasoning, not recall. Every number cited is measured and
traceable to a script in this repository.

---

## Q1. "You used `SFTTrainer`. What does it actually do that you'd have to write yourself?"

Four things, and I verified the one that matters rather than trusting it.

1. **Renders the chat template** and tokenizes.
2. **Builds the labels**: copies `input_ids`, then
   `labels[attention_mask == 0] = -100` for padding and
   `labels[completion_mask == 0] = -100` for the prompt.
3. **Decides whether to mask at all.** `completion_only_loss` auto-resolves to
   `True` iff the dataset sample has both `prompt` and `completion` keys
   (`sft_trainer.py:1201`). The dataset's *shape* selects the loss.
4. **Handles the shift**, passing `shift_labels` explicitly rather than relying
   on the model's internal shift.

The dangerous one is (3), because it is implicit. A pre-rendered `text` column
trains on the prompt and nothing tells you.

**What I added:** an independent computation of the expected labels that does
not import TRL, compared position by position. Agreement between two
computations is evidence; inspecting TRL's own output would verify nothing.

---

## Q2. "How do you know your loss mask is right?"

Four ways, in increasing strength.

1. **Decode the active region.** For a controlled example it decodes to exactly
   `'It is 4.<|im_end|>\n'` — 7 of 33 tokens, boundary at index 26.
2. **Compare against an independent computation.** `alignlab.masking` derives
   the boundary from tokenizing the prompt alone. Identical to TRL's, position
   by position.
3. **Measure the broken arm.** With `completion_only_loss=False`, 33/33 tokens
   become active. The difference is measured, not asserted.
4. **Re-verify on every run**, on real data, before the first optimiser step —
   and *raise* if it disagrees.

The distinction in (4) matters: E12 proves the mechanism works; the per-run
check proves *this* wiring is correct for this dataset and config. A
verification that ran once is a claim about the past.

---

## Q3. "A colleague's SFT loss is 0.8, yours is 2.1, same model and data. Who's winning?"

Nobody, until we know what each mean is over. A loss is
`(sum of NLL over active positions) / (number of active positions)`, and a mask
change alters **both**.

I'd check, in order:
1. Is their loss masked at all? Unmasked averages in prompt tokens.
2. Is packing on? Packed sequences change the token population per step.
3. Same `max_length`? Truncation changes which tokens survive.
4. Same aggregation — token-weighted or example-weighted? On a toy case those
   differ by 30%.

**And I have a measurement showing the naive intuition is wrong.** I predicted
an unmasked loss would look *better* because prompts are easy. Measured on base
Qwen2.5-1.5B: prompt CE **6.3314**, completion CE **3.6295**. The prompt is
*harder* — because the base model has never seen ChatML and the opening tokens
have no left context. So a broken mask made the number look **worse**.

The rule that survives is stronger than the one I started with: **never compare
losses computed over different token populations**, in either direction.

---

## Q4. "What's the most subtle bug you hit?"

Tokenization not respecting the prompt/completion boundary.

TRL builds the mask as `[0]*len(prompt_ids) + [1]*(len(full) - len(prompt))`,
which is correct only if the prompt's tokens are a genuine **prefix** of the
full sequence's. BPE merges across boundaries, so that is not guaranteed.

The ChatML prompt ends with `<|im_start|>assistant\n` — token `198`, a lone
newline. When a completion *starts* with newlines, BPE merges them into token
`1406` (`'\n\n\n'`). The boundary then lands on a token that is half prompt and
half answer.

**Measured: 3 of 200 rows, 1.5%.** Found by a check I wrote specifically
because I chose to verify that assumption rather than trust it.

**Follow-up I'd expect: "how would you fix it?"** I strip the completion's
leading whitespace. That is a fix rather than a workaround, because those
newlines are redundant with the newline the template already emits and training
on them teaches the model to open answers with blank lines. Alternatives:
tokenize once and locate the boundary by decoding backwards (robust, slower);
or drop the affected rows (loses data, and hides the cause).

**The second-order bug is the better story.** The fix appeared not to work —
the same three indices came back. The function was correct in isolation;
`datasets.map()` had cached its output and did not invalidate when the mapping
function changed. A silently stale cache makes a real fix look ineffective. The
content fingerprint changing (`b7dff71c…` → `dc6fd746…`) is what proved the data
had finally changed.

---

## Q5. "Your Phase 2 parameter count was wrong. How did you find out?"

By downloading the weights and counting.

Phase 2 computed **1,543,656,960** from `config.json`. The real checkpoint has
**1,543,714,304** — short by **57,344**, which is exactly
`28 × (1536 + 256 + 256)`: the attention QKV biases.

The instructive part: Phase 2 had **already discovered** that Qwen uses QKV
bias and listed it as a wrong prediction. The fact never reached the
arithmetic. Knowing something and having your formula reflect it are different
states, and only contact with the real artefact distinguished them.

The error is **0.0037%** — invisible at the model card's "1.54B", which is why
it survived. A claim agreeing to three significant figures is not the same as a
claim being exact, and the Phase 2 report said "exactly".

**What I'd want an interviewer to take from this:** the same wrong constant had
propagated into `tests/test_storage.py`, and was only caught when the storage
estimates were checked against a real checkpoint file. Errors in derived
constants spread quietly.

---

## Q6. "How much disk does a checkpoint take? Show your working."

**Estimated** (conservative, the guard's default): bf16 params (2 B/param) +
fp32 AdamW moments (8) + fp32 master weights (4) = 14 B/param → **20.13 GiB**.

**Measured**, from the real checkpoint:

```
model.safetensors  3,087,467,144 B  = 2.00 B/param
optimizer.pt       6,175,148,456 B  = 4.00 B/param
total              9,262,615,600 B  = 6.00 B/param = 8.63 GiB
```

The estimate over-predicts by **2.33×**, because this configuration stores the
moments in **bf16** and keeps **no fp32 master copy** — pure bf16 rather than
mixed precision.

**I kept the conservative default deliberately.** A guard that under-predicts
fails halfway through writing a 9 GiB file on a volume shared with eight other
users; one that over-predicts refuses a run that would have fitted. Those costs
are not symmetric.

**Follow-up: "why budget `keep_last + 1`?"** Rotation writes the new checkpoint
*before* deleting the old one — deleting first would risk destroying the only
good checkpoint if the write then failed — so peak usage is one more than the
retention setting.

---

## Q7. "How do you know the model got better?"

I distinguish what was measured from what it means.

**MEASURED:** completion-only perplexity on a held-out split of the same
dataset, prompt-region perplexity as an untrained control, and whether
generation terminates on `<|im_end|>`.

**NOT ESTABLISHED:** that it is a better assistant. Lower perplexity on
no_robots' test split means better fit to no_robots — which is what one epoch
of no_robots should buy, and is not the same claim.

**Why the control matters:** the prompt region carried no gradient. If its
perplexity moved a lot, that is drift, not success. Reporting only the metric
that is supposed to improve is how one fools oneself.

**The most concrete win is not perplexity at all.** The base model has never
emitted `<|im_end|>` and does not stop; the SFT model should. That is binary
and measurable, and it is the thing a user would actually notice.

---

## Q8. "Why is packing off? That's leaving throughput on the table."

Yes, deliberately. Packing concatenates examples to fill the context window,
which raises tokens/second substantially — and makes the loss mask far harder
to verify by hand, because example boundaries stop aligning with sequence
boundaries.

Phase 3's deliverable is a **verified** mask. Trading verifiability for speed in
the phase whose job is verification is the wrong trade. Phase 4 can turn it on,
because by then the mask is trusted.

I'd rather be asked why I left throughput on the table than why my model trained
on its own prompts.

---

## Q9 (challenge). "Isn't a per-run mask check just paranoia? You already have E12."

They test different claims.

E12 tests the **mechanism** against a hand-built example: does TRL mask what we
think it masks? That answer does not change day to day.

The per-run check tests the **wiring**: is *this* dataset, at *this*
`max_length`, with *this* tokenizer, producing the mask we intend? That can
break without E12 noticing — a dataset schema change, a truncation setting, a
new chat template.

Concretely: E12 passed on a hand-built example while real data contained rows
whose tokenization broke the boundary assumption. Only the audit on real data
found them.

The cost is a few seconds per run. The failure it prevents is silent and
expensive.

---

## Q10 (challenge). "What would you do differently with more time?"

1. **A learning-rate sweep.** 2e-5 is convention, not a tuned value, and I have
   no evidence it is right for this model and dataset. Right now it is a
   starting point I have labelled as such.
2. **Multi-turn training.** `assistant_only_loss` needs a chat template with
   `{% generation %}` markers, which Qwen2.5's stock template lacks. ~8% of
   no_robots rows lose their earlier assistant turns as targets.
3. **More than one epoch, and a seed sweep**, so I could say something about
   variance rather than reporting a single run.
4. **A held-out set that is not no_robots**, so "better" is not circular.

---

## Explain Back

> **[USER CHECKPOINT — DEFERRED — USER EXPLAIN-BACK REQUIRED]**
>
> Answer Q2, Q3, Q4 and Q6 aloud, without notes. Then write two questions this
> document does not answer, and answer those too.

**Related:** [[phase3-sft-and-loss-masking]] · [[phase3-code-explanation]] ·
[[interview-phase2-transformers]] · [[pytorch-loss-masking-and-shifts]]
