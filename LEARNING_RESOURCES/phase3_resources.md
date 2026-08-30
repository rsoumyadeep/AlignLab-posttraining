# Phase 3 — Resources Actually Inspected

Honest audit. Statuses follow PROJECT_INSTRUCTIONS §9. Nothing here is claimed
as read unless it was opened. **All inspections dated 2026-08-30.**

A note on what dominates this list: Phase 2's resources were papers, because
Phase 2 was about understanding mechanisms. Phase 3's are **source code and
configuration files**, because Phase 3 is about what a specific library
actually does on a specific day. Three of Phase 3's bugs came from assuming an
API rather than reading it, so reading installed source is the higher-value
activity here — and unlike an arXiv abstract, it is ground truth for the code
that is running.

---

## Resource 1

```
Resource:        TRL SFTTrainer implementation
Type:            source code (installed package)
URL / Identifier: .venv/lib/python3.11/site-packages/trl/trainer/sft_trainer.py
                  trl 1.12.0
Topic:           how completion_only_loss becomes labels
Access Status:   ACTUALLY INSPECTED - specific lines read on the server
What Was Inspected:
                 lines 1200-1203 (auto-resolution), 772-774 (label masking),
                 1551-1553 (completion_mask construction), 794-796 (the shift),
                 1592-1599 (mask column handling)
Claims Obtained:
                 - completion_only_loss resolves True iff the sample has both
                   "prompt" and "completion" keys (l.1201)
                 - labels[attention_mask == 0] = -100  (padding)   (l.772)
                 - labels[completion_mask == 0] = -100 (prompt)    (l.774)
                 - completion_mask = [0]*len(prompt_ids) +
                   [1]*(len(prompt_completion_ids) - len(prompt_ids))  (l.1551)
                 - TRL passes shift_labels itself rather than relying on the
                   model's internal shift
Limitations:     read the masking path only; the optimiser, packing and
                 distributed paths were NOT read.
```

## Resource 2

```
Resource:        Qwen2.5-1.5B model card
Type:            model documentation, at the PINNED revision
URL / Identifier: huggingface.co/Qwen/Qwen2.5-1.5B/raw/
                  8faed761d45a263340a0528343f099c05c9a4323/README.md
Topic:           context length, architecture, intended use
Access Status:   ACTUALLY INSPECTED - full card fetched and read
What Was Inspected: the whole file (3,850 bytes)
Claims Obtained:
                 - line 29: "Context Length: Full 32,768 tokens" - the per-repo
                   spec block for THIS checkpoint
                 - line 18: "Long-context Support up to 128K tokens" - in the
                   SERIES-level introduction, not about this checkpoint
                 - "Architecture: transformers with RoPE, SwiGLU, RMSNorm,
                   Attention QKV bias and tied word embeddings"
                 - "We do NOT recommend using base language models for
                   conversations. Instead, you can apply post-training, e.g.
                   SFT, RLHF..." - the card explicitly endorses what Phase 3 does
Limitations:     a card is a claim by the authors, not an independent
                 measurement. The 32,768 figure is NOT verified by us.
```

**This resolved a Phase 2 open question.** Phase 2 recorded
`max_position_embeddings = 131072` versus a card context of 32,768 as an
UNRESOLVED discrepancy with an unverified hypothesis. Reading the card at the
pinned revision shows the two numbers describe **different scopes** — family
versus checkpoint. Operational decision recorded in the Phase 3 report.

## Resource 3

```
Resource:        Qwen2.5-1.5B config.json and tokenizer_config.json
Type:            configuration, at the PINNED revision
URL / Identifier: same revision as above
Topic:           architecture and tokenizer ground truth
Access Status:   ACTUALLY INSPECTED - downloaded and parsed
What Was Inspected: every key of config.json; tokenizer_config.json's
                 chat_template, eos/pad tokens, additional_special_tokens
Claims Obtained:
                 - hidden 1536, 12 Q heads, 2 KV heads, 28 layers,
                   intermediate 8960 (5.8333x), rope_theta 1e6,
                   vocab 151936, tie_word_embeddings true
                 - the BASE model ships a ChatML chat_template
                 - eos_token is <|endoftext|> (151643), NOT <|im_end|> (151645)
                 - <|im_start|>/<|im_end|> are already in
                   additional_special_tokens - no vocab resize needed
                 - the template does NOT contain {% generation %} markers
Limitations:     configuration describes the architecture, never the behaviour.
```

## Resource 4

```
Resource:        transformers Qwen2Config under transformers 5.16.1
Type:            source behaviour (observed, not read line by line)
URL / Identifier: installed transformers 5.16.1
Topic:           where rope_theta lives after the 5.x migration
Access Status:   PARTIALLY INSPECTED - observed via introspection, source NOT read
What Was Inspected: dir(cfg) and cfg.to_dict() on a loaded Qwen2Config
Claims Obtained:
                 - cfg.rope_theta DOES NOT EXIST in 5.x
                 - the value is at cfg.rope_parameters["rope_theta"], with
                   cfg.rope_scaling as an alias
                 - 5.x normalises the config: sliding_window becomes null and a
                   layer_types list of 28 "full_attention" entries is added
Limitations:     behaviour observed on one config for one model. The migration
                 guide was NOT read, so WHY it moved is not established here.
```

## Resource 5

```
Resource:        HuggingFaceH4/no_robots dataset
Type:            dataset
URL / Identifier: huggingface.co/datasets/HuggingFaceH4/no_robots (revision main)
Topic:           instruction data for SFT
Access Status:   ACTUALLY INSPECTED - schema, splits and distribution measured
What Was Inspected: get_dataset_config_names, get_dataset_split_names, the
                 column names, and the turn-count distribution of 2,000 rows
Claims Obtained:
                 - configs: ["default"]; splits: ["train", "test"]
                   (NOT train_sft/test_sft, despite such filenames in the repo)
                 - 9,500 train rows; columns prompt, prompt_id, messages, category
                 - turn counts in the first 2,000: 1840 x 2-turn, 136 x 7,
                   13 x 9, 6 x 5, 2 x 8, 1 x 6
                 - 2000/2000 sampled conversations end with an assistant turn
                 - 1 of 9,500 rows fails our well-formedness filter
Limitations:     content quality NOT assessed. No licence review beyond noting
                 it is human-authored. We did not read the dataset card prose.
```

## Resource 6

```
Resource:        Hugging Face Hub API - model file tree at a pinned revision
Type:            web API
URL / Identifier: huggingface.co/api/models/Qwen/Qwen2.5-1.5B/tree/<sha>
Topic:           exact download size before downloading
Access Status:   ACTUALLY INSPECTED - queried, parsed, and used as the
                 verification baseline
Claims Obtained:
                 - 8 files, 3,098,966,854 bytes total (2.886 GiB) at the pin
                 - model.safetensors is 3,087,467,144 bytes
                 - revision 8faed761... is currently identical to main
                   (lastModified 2024-10-08)
Limitations:     sizes are the API's claim; independently confirmed after
                 download by comparing every file's on-disk size.
```

## Resource 7

```
Resource:        PROJECT_INSTRUCTIONS.md
Type:            project specification
Access Status:   ACTUALLY INSPECTED - read in full at session start
Topic:           the authoritative spec
Claims Obtained: §3 makes verified loss masking a precondition for PEFT;
                 §15 defines the evidence labels; §16-18 govern experiments,
                 failed experiments and suspicious results; §20 requires
                 before/after; §29 fixes the phase report structure.
Limitations:     none - this is the spec, not a claim about the world.
```

---

## NOT INSPECTED (and therefore not cited anywhere)

| Resource | Why it would matter |
|---|---|
| InstructGPT (Ouyang et al. 2022) | the origin of the SFT→RM→PPO pipeline; Phase 5 |
| Self-Instruct (Wang et al. 2022) | instruction-data generation |
| LIMA (Zhou et al. 2023) | the "1k examples is enough" claim, directly relevant to our 9.5k |
| Qwen2.5 technical report | how the base model was actually trained |
| TRL packing / `assistant_only_loss` implementation | needed before enabling either |
| transformers 5.x migration guide | would explain the API breaks we hit empirically |

These are **NOT INSPECTED**, not "skimmed". No claim in this repository rests
on them.

---

## Own Words

> ## Own Words
> **[USER CHECKPOINT — DEFERRED — USER MUST WRITE THIS]**
> One entry per resource above, after reading it yourself.

**Related:** [[phase3-sft-and-loss-masking]] · [[phase2-learning-resources]] ·
[[phase3-code-explanation]]
