# Learning Resources — Audit

An honest register of external resources. Status vocabulary is fixed:

| Status | Meaning |
|---|---|
| `ACTUALLY INSPECTED` | The resource was opened and read during this project, and what was taken from it is recorded below. |
| `PARTIALLY INSPECTED` | Opened, but only part was read. The part is named. |
| `NOT ACCESSIBLE` | An access attempt was made and failed. |
| `NOT INSPECTED` | No access attempt has been made. |

---

## ⚠ Status of this register as of Phase 1B (2026-08-29)

**Every external resource below is still `NOT INSPECTED`.**

No web page, paper, repository, blog post, video or piece of official
documentation was accessed during Phase 1A **or Phase 1B**. No browsing tool was
used in either phase. The implementation was written from working knowledge and
verified by **executing it** — the passing test suites (104 local / 105 server)
and the example scripts in `PYTORCH_CONCEPTS/` are the evidence base, not any
external source.

**Phase 1B note — two queued questions were answered empirically instead.**
Rather than reading documentation, the server itself was measured:

- *"Does the department server run SLURM?"* — answered by `scontrol ping`, not
  by the SLURM docs. Result: installed (22.05.9) but the controller on
  `csrmaster` is **DOWN**, so no job can be submitted. This is a fact about
  **this machine**, and no document could have supplied it. The queued SLURM
  documentation entry below is consequently **lower priority than before** — it
  would explain a mechanism we currently cannot use.
- *"Is bf16 actually supported?"* — answered by running
  `torch.cuda.is_bf16_supported()` and executing a bf16 matmul, not by consulting
  an architecture table.

Direct measurement is not a substitute for reading the papers in Phases 2–6. It
is the right tool for questions about a specific machine's current state, and
the wrong tool for questions about why an algorithm works.

This is recorded plainly because the alternative — listing plausible-looking
citations for material that was never opened — would be fabrication, and would
poison the one thing this register exists to provide: the ability to trust that
a cited claim was actually checked.

**What this means practically:** any claim in the Phase 1A documentation that is
*not* backed by executed code should be treated as working knowledge subject to
verification, not as a sourced fact. Where behaviour mattered, it was executed
rather than cited — see `PYTORCH_CONCEPTS/pytorch-rng-and-state.md`, in which
three examples were found wrong by running them.

---

## Queue — to inspect during Phase 1B and Phase 2

Listed with the specific question each is expected to answer, so inspection is
targeted rather than aimless. **None has been opened.**

### Engineering / infrastructure

```
Resource:        Hydra documentation
Type:            Official documentation
URL/Identifier:  https://hydra.cc/docs/intro/
Topic:           Config groups, structured configs, override syntax
Access Status:   NOT INSPECTED
Question to answer: Is `hydra/job_logging: none` the documented way to suppress
                 Hydra's file handler, or is there a cleaner mechanism? (Phase
                 1A settled this EMPIRICALLY: `disabled` sets
                 disable_existing_loggers and silently killed our logger;
                 `none` works. Confirm against the docs.)
```

```
Resource:        PyTorch reproducibility notes
Type:            Official documentation
URL/Identifier:  https://pytorch.org/docs/stable/notes/randomness.html
Topic:           Determinism, CUBLAS_WORKSPACE_CONFIG, cuDNN flags
Access Status:   NOT INSPECTED
Question to answer: Is `:4096:8` still the correct CUBLAS_WORKSPACE_CONFIG for
                 CUDA 12.x, and which ops have no deterministic kernel?
```

```
Resource:        PyTorch DDP / distributed docs
Type:            Official documentation
URL/Identifier:  https://pytorch.org/docs/stable/notes/ddp.html
Topic:           DDP, gradient all-reduce, NCCL backend
Access Status:   NOT INSPECTED
Question to answer: What throughput should 2 A6000s WITHOUT NVLink (topology
                 "SYS") be expected to reach, and how is P2P availability
                 checked?
```

```
Resource:        Weights & Biases offline mode
Type:            Official documentation
URL/Identifier:  https://docs.wandb.ai/
Topic:           Offline runs, `wandb sync`
Access Status:   NOT INSPECTED
Question to answer: Exact semantics of syncing an offline run recorded on a
                 machine with no outbound access.
```

```
Resource:        SLURM signal / requeue documentation
Type:            Official documentation
URL/Identifier:  https://slurm.schedmd.com/sbatch.html
Topic:           --signal=USR1@<seconds>, scontrol requeue
Access Status:   NOT INSPECTED
Priority:        LOWERED. Phase 1B VERIFIED that SLURM is installed on csrslave
                 but NON-FUNCTIONAL (Slurmctld at csrmaster is DOWN, slurmctld
                 and slurmd services failed). Jobs run directly under tmux, so
                 the --signal=USR1 mechanism this document describes is not
                 currently reachable. Revisit only if the controller is fixed.
                 NOTE: the SIGUSR1 handling itself is now VERIFIED anyway, via
                 an external "kill -USR1" on a live run - no scheduler needed.
```

### Phase 2+ (modelling)

```
Resource:        Attention Is All You Need (Vaswani et al., 2017)
Type:            Paper
URL/Identifier:  arXiv:1706.03762
Topic:           Transformer, scaled dot-product attention, sinusoidal PE
Access Status:   NOT INSPECTED
Question to answer: The exact sinusoidal formulation, to derive rather than
                 recall it (Tier 1 interview topic #5).
```

```
Resource:        RoFormer / RoPE (Su et al., 2021)
Type:            Paper
URL/Identifier:  arXiv:2104.09864
Topic:           Rotary position embedding
Access Status:   NOT INSPECTED
Question to answer: Why the rotation formulation gives relative-position
                 dependence in the attention inner product.
```

```
Resource:        LoRA (Hu et al., 2021)
Type:            Paper
URL/Identifier:  arXiv:2106.09685
Topic:           Low-rank adaptation, rank choice, scaling alpha
Access Status:   NOT INSPECTED
Question to answer: What the paper actually claims about rank selection, versus
                 the folklore. Tier 1 interview topic #1.
```

```
Resource:        QLoRA (Dettmers et al., 2023)
Type:            Paper
URL/Identifier:  arXiv:2305.14314
Topic:           NF4, double quantization, paged optimizers
Access Status:   NOT INSPECTED
Question to answer: Precisely which component makes QLoRA differ from LoRA.
                 Tier 1 interview topic #4.
```

```
Resource:        DPO (Rafailov et al., 2023)
Type:            Paper
URL/Identifier:  arXiv:2305.18290
Topic:           Direct preference optimization, Bradley-Terry derivation
Access Status:   NOT INSPECTED
Question to answer: The full derivation from the RLHF objective to the DPO
                 loss, to reproduce rather than quote.
```

```
Resource:        Qwen2.5 technical report / model card
Type:            Paper + model card
URL/Identifier:  Qwen/Qwen2.5-1.5B on Hugging Face
Topic:           Architecture, tokenizer, GQA config, context length
Access Status:   NOT INSPECTED
Needed for:      Phase 2 and Phase 3. Also needed to pin an exact model
                 REVISION for reproducibility (instructions section 24).
```

---

## Own Words

> [USER CHECKPOINT — USER MUST EVENTUALLY WRITE THIS]
>
> One entry per resource, written **after** you have actually read it. Claude
> may explain any of this material on request, but must not write this section
> for you — a summary you did not write is not evidence that you understood it.
>
> Suggested shape per resource: what problem it solves · the central idea in
> two sentences · one thing that surprised you · one thing you still find
> unclear.

---

## Register discipline

1. A status changes **only** when the resource is actually opened.
2. Record what was inspected (which section, which figure) — not just "read it".
3. Record limitations: what the resource did *not* answer.
4. If a link is dead, mark `NOT ACCESSIBLE` and say so. Do not substitute a
   different source silently.

**Related:** [[phase1-foundation]] · [[interview-phase1-engineering]]
