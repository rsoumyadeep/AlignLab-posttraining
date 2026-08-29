# AlignLab — Project Instructions

## 1. Project Identity

AlignLab is an end-to-end **LLM post-training framework built from first principles**.

The project starts from the pretrained base model:

```text
Qwen/Qwen2.5-1.5B
```

Specifically, use the **base pretrained model**, not `Qwen/Qwen2.5-1.5B-Instruct`.

AlignLab does **not** pretrain an LLM.

Its scope begins after pretraining:

```text
Large-scale pretraining
        │
        │  OUTSIDE ALIGNLAB
        ▼
Qwen/Qwen2.5-1.5B
        │
        ▼
Supervised Fine-Tuning
        │
        ▼
Parameter-Efficient Fine-Tuning
        │
        ├── LoRA
        └── QLoRA
        │
        ▼
Preference Learning
        │
        ├── Reward Modeling concepts
        └── RLHF concepts
        │
        ▼
Preference Optimization
        │
        └── DPO
        │
        ▼
Evaluation
        │
        ▼
Engineering / Reproducibility
```

The correct project framing for CVs and interviews is:

> "Rather than training an LLM from scratch, AlignLab focuses on what happens after pretraining."

Never describe AlignLab as having "built an LLM from scratch."

---

# 2. Why AlignLab Exists

Most LLM tutorials teach users to fine-tune models by invoking a few high-level library commands.

That can produce a working checkpoint while hiding:

* why the algorithm exists,
* what mathematical objective is being optimized,
* how the implementation works,
* why particular engineering decisions were made,
* what the experiments actually demonstrate.

AlignLab takes the opposite approach.

The goal is:

> **understanding + first-principles implementation + realistic ML engineering + empirical evaluation.**

A working model is not sufficient.

The final project must allow the USER to:

* understand the theory,
* understand the mathematics,
* implement important mechanisms,
* understand the actual code,
* reproduce experiments,
* interpret results,
* explain engineering decisions,
* defend the project in interviews,
* identify limitations,
* and extend the framework independently.

---

# 3. Core Learning Philosophy

Follow this workflow for important concepts:

```text
Understand the Theory
        ↓
Develop Intuition
        ↓
Study the Mathematics
        ↓
Derive Important Equations
        ↓
Implement a Minimal Version
        ↓
Verify the Minimal Version
        ↓
Integrate into AlignLab
        ↓
Experiment
        ↓
Evaluate
        ↓
Document
        ↓
USER Explain-Back
        ↓
Refactor
```

Do not jump directly from a paper or API to a large implementation when a minimal educational implementation would materially improve understanding.

Use high-level libraries where appropriate for realistic engineering, but do not let library abstractions replace understanding.

---

# 4. Eight Project Phases

AlignLab consists of eight major phases.

## Phase 1 — Foundation

Build the engineering foundation:

* repository scaffold,
* configuration management,
* Hydra configuration,
* logging,
* Weights & Biases experiment tracking,
* reproducibility infrastructure,
* checkpoint management,
* SLURM/SIGUSR1 preemption handling where appropriate,
* testing infrastructure.

This phase establishes the engineering foundation for all subsequent experiments.

---

## Phase 2 — Transformer Understanding

Study and implement the core Transformer mechanisms relevant to modern decoder-only LLMs.

Topics include:

* tokenization,
* embeddings,
* self-attention,
* multi-head attention,
* Q/K/V,
* scaled dot-product attention,
* causal masking,
* positional encoding,
* sinusoidal positional encoding,
* RoPE,
* ALiBi,
* MHA,
* MQA,
* GQA,
* RMSNorm,
* LayerNorm,
* SwiGLU,
* decoder-only architecture,
* autoregressive generation,
* KV caching,
* Flash Attention concepts.

Important educational implementations should be built from first principles where practical.

Primary implementation targets may include:

```text
src/models/attention.py
src/models/transformer.py
tests/test_attention.py
```

This is a **high-priority phase** and should receive substantial mathematical and implementation depth.

---

## Phase 3 — Supervised Fine-Tuning

Fine-tune the pretrained Qwen base model on an instruction dataset.

This phase intentionally uses Hugging Face `SFTTrainer` where appropriate because realistic industry engineering is itself an objective.

However, the USER must understand what the trainer is doing.

Particular attention must be given to:

* instruction datasets,
* chat templates,
* tokenization,
* labels,
* prompt/completion boundaries,
* loss masking,
* causal language-model loss,
* training configuration,
* checkpointing,
* evaluation.

**Loss masking must be explicitly verified before PEFT is layered on top.**

The project must be able to explain:

> Which tokens contribute to the training loss and why?

---

## Phase 4 — Parameter-Efficient Fine-Tuning

Implement and study:

* LoRA,
* QLoRA,
* LoRA rank,
* low-rank approximation,
* SVD,
* PCA,
* quantization,
* LoRA-FA where relevant.

The core LoRA mechanism should be implemented manually rather than simply relying on a PEFT library implementation.

A practical library implementation may subsequently be used for comparison.

The project should distinguish:

```text
Educational implementation
        vs
Practical library implementation
```

This phase is one of the **highest-priority interview-defense areas**.

---

## Phase 5 — Preference Learning / RLHF

Study the conceptual RLHF pipeline.

This phase is deliberately **conceptual rather than a full PPO implementation** unless explicitly expanded later.

Understand:

```text
SFT model
   ↓
Preference data
   ↓
Reward model
   ↓
Policy optimization
   ↓
Aligned model
```

Understand:

* preference datasets,
* chosen/rejected responses,
* reward modeling,
* Bradley-Terry preference modeling,
* PPO at a conceptual and mathematical level,
* KL regularization,
* why RLHF is expensive/complex,
* why DPO can be attractive compared with PPO.

The main deliverable is the ability to defend:

> **Why use DPO rather than PPO-based RLHF?**

---

## Phase 6 — Direct Preference Optimization

Implement DPO from first principles.

This phase should include:

* preference dataset,
* chosen/rejected responses,
* reference model,
* policy model,
* DPO objective,
* log-probability computation,
* reference-policy comparison,
* temperature/beta,
* Bradley-Terry interpretation,
* training,
* before/after evaluation,
* LLM-as-judge win-rate evaluation where practical.

The DPO loss should be understood mathematically rather than treated as an opaque library function.

---

## Phase 7 — Evaluation

Evaluation is **not merely a final phase**.

A minimal evaluation pipeline should exist early and be rerunnable throughout the project.

At minimum, investigate:

* perplexity,
* qualitative evaluation,
* instruction-following behavior,
* preference-based evaluation,
* LLM-as-judge evaluation where appropriate,
* before/after comparisons.

Every major post-training phase should have a before/after story where meaningful.

The project must be able to answer:

> **How do you know the model actually got better?**

Avoid relying on a single metric.

Clearly distinguish improvements in measured metrics from subjective qualitative impressions.

---

## Phase 8 — Engineering

Finalize the project as a professional ML engineering framework.

Focus on:

* repository hygiene,
* modularity,
* testing,
* configuration,
* reproducibility,
* documentation,
* experiment tracking,
* README quality,
* design rationale,
* DPO-vs-PPO comparison,
* engineering tradeoffs,
* limitations,
* future extensions.

---

# 5. Five-Folder Documentation System

AlignLab MUST maintain these five top-level documentation folders:

```text
STUDY_WITH_CLAUDE/
CODE_EXPLANATION/
PYTORCH_CONCEPTS/
LEARNING_RESOURCES/
INTERVIEW_DEFENSE/
```

Keep them conceptually distinct.

Do not duplicate identical material across folders.

---

# 6. STUDY_WITH_CLAUDE/

This is the project's **actual learning notebook**.

Document:

* theory,
* intuition,
* mathematics,
* derivations,
* conceptual experiments,
* observations,
* insights,
* misconceptions,
* connections between concepts.

For important concepts, create:

```text
## My Understanding

[USER CHECKPOINT — USER MUST COMPLETE]
```

or:

```text
## Explain Back

[USER CHECKPOINT]
```

Claude MUST NOT write the USER's understanding on the USER's behalf.

At an explain-back checkpoint:

1. explain the concept,
2. ask the USER to explain it,
3. evaluate their explanation,
4. identify gaps,
5. ask follow-up questions,
6. allow the USER to revise their own explanation.

The objective is independent understanding.

Useful conceptual experiment scripts should be stored here when appropriate.

---

# 7. CODE_EXPLANATION/

This folder explains the **actual implementation**.

Never document an imagined implementation.

For every major module/phase document:

### Files

* what exists,
* why it exists,
* responsibilities.

### Classes and Functions

* purpose,
* inputs,
* outputs,
* important internal behavior.

### Data Flow

Explain how data moves through the system.

### Module Interaction

Explain how components interact.

### Mathematics → Code

Explicitly map equations to implementation.

Example:

```text
Mathematical objective
        ↓
Loss function
        ↓
Tensor operations
        ↓
Gradient computation
        ↓
Optimizer update
```

### Design Decisions

Document:

* selected approach,
* alternatives,
* tradeoffs,
* rejected approaches.

### Edge Cases

Document:

* tensor shapes,
* masking,
* padding,
* numerical stability,
* dtype,
* device,
* checkpoint,
* tokenizer,
* generation,
* failure modes.

Whenever code changes substantially, check whether the documentation has become stale.

---

# 8. PYTORCH_CONCEPTS/

Document important PyTorch/programming concepts encountered in AlignLab.

For each important concept document:

1. What it does.
2. Why AlignLab uses it.
3. Minimal isolated example.
4. Expected result.
5. Actual result.
6. Common mistakes.
7. Connection to AlignLab.

Examples include:

* `nn.Module`,
* parameters vs buffers,
* autograd,
* `requires_grad`,
* optimizers,
* gradient accumulation,
* gradient clipping,
* mixed precision,
* tensor broadcasting,
* `view`,
* `reshape`,
* `transpose`,
* `permute`,
* `matmul`,
* masking,
* `Dataset`,
* `DataLoader`,
* distributed samplers,
* DDP,
* checkpointing,
* device management,
* quantization.

Execute examples whenever practical.

Never claim an example was executed unless it was actually executed.

---

# 9. LEARNING_RESOURCES/

Maintain an honest audit of external resources.

Possible resources:

* papers,
* official documentation,
* repositories,
* blogs,
* lectures,
* videos,
* technical articles.

For each resource record:

```text
Resource:
Type:
URL / Identifier:
Topic:
Access Status:
What Was Inspected:
Claims Obtained:
Limitations:
```

Use statuses such as:

```text
ACTUALLY INSPECTED
PARTIALLY INSPECTED
NOT ACCESSIBLE
NOT INSPECTED
```

Never claim to have read a paper, repository, blog, or documentation that was not actually accessed.

Important resources should contain:

```text
## Own Words

[USER CHECKPOINT — USER MUST EVENTUALLY WRITE THIS]
```

Claude may explain the material, but should not fabricate the USER's own summary.

---

# 10. INTERVIEW_DEFENSE/

This folder prepares the USER to defend AlignLab in:

* ML engineering interviews,
* LLM engineering interviews,
* research discussions,
* technical reviews.

Cover:

* project motivation,
* architecture,
* mathematical foundations,
* implementation decisions,
* training methodology,
* evaluation,
* experiments,
* surprising findings,
* bugs,
* corrections,
* rejected approaches,
* limitations,
* ablations,
* comparisons,
* extensions,
* challenging interview questions.

For every important experimental result include:

> What does this result prove?

and:

> What does this result NOT prove?

Do not turn this folder into a collection of memorized definitions.

Questions should test reasoning.

---

# 11. Interview-ROI Prioritization

AlignLab should NOT allocate equal study depth to every topic.

The following are designated **highest-ROI interview topics**.

## Tier 1 — Highest ROI

### 1. LoRA Rank Selection via SVD

Understand:

* why LoRA uses low rank,
* what SVD provides,
* singular-value spectra,
* explained energy,
* approximation error,
* choosing rank,
* limitations of choosing rank using SVD.

The USER must be able to answer:

> "How would you choose the LoRA rank for a new task/model?"

without merely saying "use SVD."

---

### 2. PCA Derivation From Scratch

Understand:

* variance maximization formulation,
* constrained optimization,
* covariance matrix,
* eigenvectors/eigenvalues,
* principal directions,
* explained variance,
* relationship to SVD,
* relationship to low-rank approximation,
* connection to LoRA rank selection.

The USER should eventually be able to derive PCA on a whiteboard.

---

### 3. Multi-Head / Self-Attention From Scratch

Implement and understand attention in:

1. PyTorch,
2. NumPy,
3. pure Python with no numerical libraries where practical.

Be able to track tensor dimensions verbally.

The USER should be able to explain:

```text
X
↓
Q, K, V
↓
QKᵀ
↓
scaling
↓
causal mask
↓
softmax
↓
weighted V
↓
output projection
```

---

### 4. Quantization Mechanics

Understand:

* FP32,
* FP16,
* BF16,
* INT8,
* INT4,
* IEEE 754 basics,
* symmetric quantization,
* asymmetric quantization,
* scale,
* zero-point,
* quantization error,
* dequantization,
* QAT,
* straight-through estimator,
* LoRA,
* QLoRA,
* LoRA-FA,
* training vs inference quantization,
* adapter inference cost.

Be able to explain:

> **What exactly makes QLoRA different from ordinary LoRA?**

---

### 5. Positional Encoding Variants

Understand and compare:

* learned absolute embeddings,
* sinusoidal positional encoding,
* RoPE,
* ALiBi.

Derive sinusoidal positional encoding from scratch.

Explain why relative distance can be represented consistently.

Understand RoPE mathematically and intuitively.

---

### 6. Attention Variant Comparisons

Be able to compare:

* MHA,
* MQA,
* GQA,
* self-attention,
* cross-attention,
* causal attention,
* sliding/rolling-window attention where relevant,
* RMSNorm,
* LayerNorm.

Understand why modern architectures may differ from the original Transformer/GPT implementations.

---

# 12. Tier 2 Interview Topics

Prepare:

### Decoder-only architecture

Be able to draw it live and explain autoregressive generation token-by-token.

Include:

* KV caching,
* Flash Attention concepts,
* training vs inference behavior.

### Systems Engineering

Be able to discuss or implement:

* DDP,
* distributed data loading,
* multi-GPU fine-tuning,
* KV-cache pseudocode,
* memory/latency tradeoffs.

### Transformer Fundamentals

Understand:

* learnable parameters,
* initialization,
* loss functions,
* why identical constant Wq/Wk/Wv initialization is problematic,
* encoder-only vs decoder-only vs encoder-decoder,
* temperature,
* top-k,
* top-p,
* beam search.

### Model Families

Understand:

```text
Transformer
├── BERT-style encoder models
├── GPT-style decoder models
└── encoder-decoder models
```

and why Transformers replaced RNN/LSTM architectures for many large-scale language-modeling applications.

### Terminology

Maintain precise distinctions between:

* attention,
* self-attention,
* multi-head attention,
* causal attention,
* cross-attention.

Also understand why self-attention parallelizes across sequence positions during training whereas RNNs have sequential recurrence.

---

# 13. Mandatory "Drill Five" Explain-Back Checkpoints

The following five checkpoints are mandatory.

## Drill 1 — LoRA via SVD

The USER must explain:

> How does the singular-value spectrum of a weight update help justify a LoRA rank?

---

## Drill 2 — PCA

The USER must derive PCA and explain its relationship to:

* eigenvalues,
* eigenvectors,
* SVD,
* variance explained,
* low-rank approximation,
* LoRA.

---

## Drill 3 — Decoder-only LLM

The USER must:

1. draw the architecture,
2. explain one forward pass,
3. narrate autoregressive generation token-by-token,
4. explain KV caching.

---

## Drill 4 — Attention in Three Implementations

The USER must implement multi-head attention using:

* PyTorch,
* NumPy,
* pure Python.

---

## Drill 5 — Tensor Shape Narration

While implementing masked attention, the USER must verbally explain the shape of every important tensor.

Example:

```text
X      : [B, T, d_model]
Q      : [B, T, d_k]
K      : [B, T, d_k]
V      : [B, T, d_v]
QKᵀ    : [B, T, T]
scores : [B, T, T]
output : [B, T, d_v]
```

The exact shapes should follow the actual implementation.

---

# 14. Additional Structurally Important Topics

Prepare these even if not explicitly confirmed in the interview knowledge base:

* SFT loss masking,
* RLHF/PPO fundamentals,
* DPO derivation,
* Bradley-Terry model,
* DPO vs PPO,
* evaluation methodology,
* ablations,
* model improvement measurement.

Treat these as important gaps to close, not as low-priority topics.

---

# 15. Evidence Status System

Use explicit status labels.

```text
IMPLEMENTED
VERIFIED
MEASURED
RECORDED
UNVERIFIED
NOT TESTED
DEFERRED
```

Definitions:

### IMPLEMENTED

The functionality exists in source code.

### VERIFIED

The functionality was actually tested and behaved as expected.

### MEASURED

A quantitative result was actually obtained.

### RECORDED

The result/observation has been documented.

### UNVERIFIED

The implementation/claim exists but has not been adequately validated.

### NOT TESTED

No relevant test or execution has occurred.

### DEFERRED

The work is intentionally postponed.

Do not use these labels interchangeably.

---

# 16. Experimental Integrity

Every meaningful experiment should record:

```text
Experiment:
Date:
Objective:
Hypothesis:
Configuration:
Baseline:
Variables Changed:
Expected Outcome:
Actual Outcome:
Metrics:
Results:
Interpretation:
Unexpected Findings:
Limitations:
Status:
```

Never fabricate:

* loss values,
* metrics,
* benchmark scores,
* training times,
* GPU memory,
* throughput,
* model quality,
* successful execution,
* resource consumption.

If something wasn't measured, say:

```text
NOT MEASURED
```

If something wasn't tested:

```text
NOT TESTED
```

---

# 17. Preserve Failed Experiments

Failed experiments are valuable.

Never erase a failure merely because the final implementation succeeded.

Preserve:

```text
Initial Hypothesis
       ↓
Implementation
       ↓
Failure / Unexpected Result
       ↓
Diagnosis
       ↓
Correction
       ↓
New Understanding
```

If an initial hypothesis was disproved, preserve the original prediction.

Do not rewrite history to make the project appear more successful than it actually was.

---

# 18. Suspicious Results

If a metric, loss curve, visualization, generated response, benchmark, or qualitative result looks suspicious:

**STOP AND INVESTIGATE.**

Check as appropriate:

* tensor shapes,
* masks,
* labels,
* tokenizer,
* data leakage,
* evaluation code,
* seeds,
* checkpoint loading,
* numerical stability,
* dtype,
* device,
* gradients,
* baseline correctness.

Never assume a surprising result means the model has learned something meaningful.

---

# 19. Testing Philosophy

Important behavioral claims should have tests whenever practical.

Prefer:

```text
Claim
 ↓
Test
 ↓
Observed Behavior
 ↓
Documentation
```

Tests should focus on meaningful invariants and edge cases.

Examples:

* attention shape correctness,
* causal masking,
* attention numerical correctness,
* positional encoding behavior,
* LoRA parameter behavior,
* loss masking,
* quantization behavior,
* checkpoint round-trip,
* tokenizer behavior,
* DPO loss correctness,
* evaluation correctness.

---

# 20. Before/After Evaluation

Whenever a phase modifies model behavior, preserve an appropriate baseline.

The project should make it possible to tell a coherent story:

```text
Base Model
    ↓
Evaluation
    ↓
SFT
    ↓
Evaluation
    ↓
LoRA/QLoRA
    ↓
Evaluation
    ↓
DPO
    ↓
Evaluation
```

Where appropriate, compare:

* quality,
* perplexity,
* preference win-rate,
* qualitative behavior,
* parameter count,
* trainable parameter count,
* memory,
* training cost,
* inference cost.

Do not claim that one model is "better" without defining what "better" means.

---

# 21. First-Principles vs Practical Implementation

For important mechanisms, prefer a two-level approach:

```text
Level 1
Educational / Minimal Implementation
        ↓
Verification
        ↓
Understanding

Level 2
Practical / Industry Implementation
        ↓
Integration
        ↓
Realistic Experiment
```

Examples:

```text
Attention
→ minimal PyTorch/NumPy implementation
→ practical Transformer implementation
```

```text
LoRA
→ minimal LoRA layer
→ practical PEFT-style integration
```

```text
DPO
→ manual loss
→ practical training pipeline
```

Do not implement everything from scratch merely for the sake of saying it is "from scratch."

Use first-principles implementation where it improves understanding.

---

# 22. Engineering Philosophy

AlignLab should resemble a real ML framework rather than a collection of notebooks.

Prioritize:

* modularity,
* readability,
* maintainability,
* extensibility,
* reproducibility,
* testing,
* configuration,
* logging,
* experiment tracking,
* clear interfaces.

Avoid premature abstraction.

Prefer:

```text
Understand
   ↓
Implement
   ↓
Verify
   ↓
Refactor
```

---

# 23. Git Discipline

Preserve clean Git history.

Use clear phase boundaries.

Do not modify previously completed/approved components merely to make later development easier.

If modification is genuinely necessary:

1. explain why,
2. identify the affected completed component,
3. justify the change,
4. obtain USER approval where appropriate,
5. preserve the historical record.

At phase completion, report:

```text
Branch:
HEAD:
Working Tree:
Relevant Commits:
Merged:
Unmerged:
Deferred:
```

Never claim synchronization without actually checking it.

---

# 24. Reproducibility

Where relevant, record:

* model identifier,
* model revision,
* tokenizer,
* dataset/version,
* random seed,
* configuration,
* hyperparameters,
* hardware,
* software environment,
* checkpoint,
* evaluation procedure.

Aim for reproducibility of meaningful experiments.

---

# 25. External Libraries

Libraries such as:

* PyTorch,
* Hugging Face Transformers,
* Hugging Face Datasets,
* PEFT,
* TRL,
* Accelerate,
* bitsandbytes,
* evaluation libraries,
* experiment tracking tools

may be used where appropriate.

However:

> **A library call is not an explanation of the underlying algorithm.**

When a major mechanism is hidden behind a library, explain the conceptual operation and implement a minimal educational version where practical.

---

# 26. Documentation Synchronization

Documentation must correspond to the current repository.

After substantial code changes:

1. inspect affected documentation,
2. identify stale claims,
3. update them,
4. verify them,
5. mark anything not verified.

If uncertain, inspect the source code rather than guessing.

---

# 27. Claude's Role

Claude Code acts as:

* engineering collaborator,
* coding assistant,
* debugging partner,
* research assistant,
* documentation assistant,
* reviewer,
* teacher.

Claude Code is **not a substitute for the USER's understanding**.

Never:

* fabricate execution,
* fabricate results,
* fabricate resource inspection,
* fabricate citations,
* fabricate benchmarks,
* hide failures,
* write the USER's "My Understanding" sections,
* claim an experiment succeeded without evidence.

---

# 28. Explain-Back Protocol

For major concepts, especially high-ROI topics:

```text
Claude explains
      ↓
USER explains back
      ↓
Claude evaluates
      ↓
Identify gaps
      ↓
Follow-up questions
      ↓
USER revises
      ↓
Checkpoint completed
```

Do not mark an explain-back checkpoint complete merely because the topic has been documented.

The USER must demonstrate understanding.

For difficult topics, use increasingly challenging questions:

```text
Definition
   ↓
Intuition
   ↓
Mathematics
   ↓
Implementation
   ↓
Edge case
   ↓
Design tradeoff
   ↓
Interview scenario
```

---

# 29. Phase Completion Report

At the end of every phase, create a concise report containing:

```text
# Phase X Report

## 1. What Was Built

## 2. What Was Verified

## 3. What Was Measured

## 4. Results

## 5. Unexpected Findings

## 6. Corrections / Bugs Discovered

## 7. Limitations

## 8. Resources Actually Inspected

## 9. Git / Synchronization State

## 10. Deferred Explain-Back Checkpoints
```

Clearly distinguish completed work from deferred work.

---

# 30. Definition of Done

A major concept is not considered fully understood merely because its implementation works.

Where practical, completion should approach:

```text
Theory understood
        ↓
Intuition developed
        ↓
Mathematics understood
        ↓
Important equations derived
        ↓
Minimal implementation
        ↓
Minimal implementation verified
        ↓
Integrated implementation
        ↓
Tests passing
        ↓
Experiment completed
        ↓
Results recorded
        ↓
Code explained
        ↓
Resources audited
        ↓
USER explain-back completed
        ↓
Interview defense prepared
```

Not every minor concept requires every step.

Major algorithms should follow this standard whenever practical.

---

# 31. Final Success Criterion

The final AlignLab project should allow the USER to answer:

> "You built this system. Explain exactly why every major component exists, how it works mathematically, how it is implemented, why you designed it this way, what alternatives you considered, what experiments you ran, what the results show, what they do not show, what failed, and what you would do next."

The final artifact should therefore support:

```text
Claude can help build it
        ↓
USER understands it
        ↓
USER can reproduce it
        ↓
USER can explain it
        ↓
USER can defend it
        ↓
USER can extend it
```

The ultimate purpose of AlignLab is **not to produce a working checkpoint**.

It is to develop the knowledge and engineering ability required to build, understand, evaluate, defend, and extend modern LLM post-training systems independently.
