# Phase 4 — Resources Actually Inspected

Honest audit per PROJECT_INSTRUCTIONS §9. **All inspections 2026-08-30.**

Statuses used: `ACTUALLY INSPECTED`, `PARTIALLY INSPECTED`, `NOT INSPECTED`.

---

## Resource 1

```
Resource:        LoRA: Low-Rank Adaptation of Large Language Models
Type:            paper (arXiv)
URL / Identifier: arxiv.org/abs/2106.09685
Topic:           the core method Phase 4 implements
Access Status:   PARTIALLY INSPECTED - abstract page only, full PDF NOT read
What Was Inspected: title and full abstract, fetched programmatically
Claims Obtained (abstract-level, quoted):
  - "freezes the pre-trained model weights and injects trainable rank
     decomposition matrices into each layer"
  - "Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number
     of trainable parameters by 10,000 times and the GPU memory requirement
     by 3 times"
  - "performs on-par or better than fine-tuning in model quality on RoBERTa,
     DeBERTa, GPT-2, and GPT-3, despite having fewer trainable parameters"
  - "unlike adapters, no additional inference latency"
Limitations:     abstract only. The paper's rank ablations, its choice of which
                 matrices to adapt, and its intrinsic-dimension argument were
                 NOT read, and nothing here relies on them.
```

**How this squares with our measurement.** The abstract's "10,000×" is not what
we observe: adapting q/k/v/o at r=16 on Qwen2.5-1.5B gives
1,543,714,304 / 4,358,144 = **354×**, and even q/v-only at r=4 gives **2,834×**.

That is not a contradiction — it is a scale effect, and worth understanding.
Dense update cost grows as `d²` while adapter cost grows as `r·d`, so the ratio
grows roughly linearly with model width. GPT-3 175B is ~100× larger than
Qwen2.5-1.5B. The headline number is a property of *their* model and
configuration, not a constant of the method. **Quoting "10,000×" for a 1.5B
model would be wrong**, and it is exactly the sort of number that gets repeated
without checking.

**"No additional inference latency"** is a claim we independently verified:
E18 confirms `merge()` folds the adapter into the base weight and that merged
output matches adapter output (max diff 2.682e-07, and bitwise-equal weights
against peft's `merge_and_unload()`).

---

## Resource 2

```
Resource:        QLoRA: Efficient Finetuning of Quantized LLMs
Type:            paper (arXiv)
URL / Identifier: arxiv.org/abs/2305.14314
Topic:           4-bit base + LoRA adapters
Access Status:   PARTIALLY INSPECTED - abstract page only, full PDF NOT read
What Was Inspected: title and full abstract, fetched programmatically
Claims Obtained (abstract-level, quoted):
  - "backpropagates gradients through a frozen, 4-bit quantized pretrained
     language model into Low Rank Adapters"
  - "(a) 4-bit NormalFloat (NF4), a new data type that is information
     theoretically optimal for normally distributed weights"
  - "(b) double quantization to reduce the average memory footprint by
     quantizing the quantization constants"
  - "(c) paged optimziers to manage memory spikes"   [sic - typo is theirs]
  - "finetune a 65B parameter model on a single 48GB GPU while preserving
     full 16-bit finetuning task performance"
Limitations:     abstract only. The NF4 derivation, the Guanaco evaluation and
                 the 1,000-model ablation were NOT read.
```

**One claim we measured independently.** The abstract says NF4 is
"information theoretically optimal for normally distributed weights". E15
measured NF4 against FP4 on N(0,1) data: relative error **0.0912 vs 0.1210**,
so NF4 is **24.6% better**. Our measurement is *consistent with* the claim; it
does not verify optimality, which is a stronger statement than a two-way
comparison can support.

**One QLoRA component we did NOT use:** paged optimizers. Not needed — peak
VRAM on the QLoRA smoke run was 2.60 GiB against 47.5 GiB available. Recorded
so the absence is deliberate rather than an oversight.

---

## Resource 3

```
Resource:        bitsandbytes 0.50.2 - installed package behaviour
Type:            software (installed), inspected by execution
URL / Identifier: .venv/lib/python3.11/site-packages/bitsandbytes/
Topic:           whether 4-bit quantization actually works on this machine
Access Status:   ACTUALLY INSPECTED - kernels executed, not merely imported
What Was Inspected:
  - bitsandbytes.cextension.lib -> which .so actually loaded
  - bitsandbytes.functional.quantize_4bit / dequantize_4bit, executed on GPU
  - bitsandbytes.nn.Linear4bit / Params4bit, forward executed
Claims Obtained (all MEASURED by us, not read):
  - the loaded native library is libbitsandbytes_cuda124.so, matching
    torch.version.cuda == 12.4 exactly
  - the wheel ships prebuilt kernels: NO nvcc and NO root were needed
  - NF4 round-trip relative error 0.0912; FP4 0.1210
  - Linear4bit forward vs bf16 nn.Linear: relative error 0.0910
  - packed storage ratio exactly 4.00x for the weights themselves
  - Params4bit.numel() returns BYTES (two 4-bit values per byte)
Limitations:     one GPU, one dtype, synthetic normal data. Real weight
                 distributions were NOT used for the NF4/FP4 comparison.
```

**Why this was inspected by execution rather than by reading docs.** The
question was machine-specific: this server has no `nvcc` and no root, which is
what made `flash-attn` unbuildable in Phase 2. Documentation could not answer
"does the shipped kernel match *this* CUDA"; only running it could.

---

## Resource 4

```
Resource:        peft 0.20.0 - LoraConfig / get_peft_model behaviour
Type:            software (installed), inspected by execution
Topic:           the production LoRA implementation Phase 4 trains with
Access Status:   ACTUALLY INSPECTED - executed and compared against our own
What Was Inspected:
  - LoraConfig(r, lora_alpha, lora_dropout, target_modules, bias, task_type)
  - get_peft_model module wrapping and naming
    (wrapped modules become "<path>.base_layer"; adapters live in
     lora_A["default"] / lora_B["default"])
  - merge_and_unload()
Claims Obtained (MEASURED by E18, on a real tiny Qwen2):
  - peft adapts exactly the same 8 modules our implementation does
  - identical trainable parameter count (7,168)
  - with adapter weights copied across, logits agree EXACTLY (0.000e+00)
  - both are bitwise equal to the base model at initialisation (B=0)
  - merged weights agree exactly (0.000e+00) with our merge()
Limitations:     compared with dropout=0 to remove stochasticity. peft's
                 TRAINING-time behaviour - dropout, dtype casting of norm
                 layers, gradient-checkpointing interaction - was NOT compared.
```

---

## Resource 5

```
Resource:        TRL 1.12.0 SFTTrainer peft_config / quantization_config path
Type:            software (installed), inspected by signature + execution
Access Status:   PARTIALLY INSPECTED
What Was Inspected: SFTTrainer.__init__ signature (peft_config and
                 quantization_config are first-class parameters); both arms
                 executed end-to-end as smoke runs
Claims Obtained: passing peft_config wraps the model; passing a
                 BitsAndBytesConfig via from_pretrained + device_map quantizes
                 on load. Both produce 4,358,144 trainable parameters at r=16.
Limitations:     the internals of TRL's peft integration were NOT read.
```

---

## Resource 6

```
Resource:        Eckart-Young-Mirsky theorem (low-rank approximation)
Type:            mathematical result
Access Status:   NOT READ FROM A SOURCE - stated from standing knowledge and
                 VERIFIED NUMERICALLY instead
What Was Verified: PYTORCH_CONCEPTS/examples/svd_pca_examples.py section 2
                 confirms ||A - A_r||_F equals sqrt(sum of squared discarded
                 singular values) to machine precision at r = 1, 5, 10, 20, 29,
                 and that truncated SVD (error 27.020925) beats the best of 20
                 random rank-5 factorisations (74.302855).
Limitations:     no textbook or paper was consulted. The theorem is asserted
                 here on the strength of the numerical check, not a citation.
                 A proof was NOT reproduced.
```

The same applies to the **PCA ↔ SVD** relationship: verified numerically
(covariance eigenvalues identical to `S²/(n−1)`, principal directions matching
right singular vectors to `|cos| = 1.0000000000`) rather than cited.

---

## NOT INSPECTED

| Resource | Why it would matter |
|---|---|
| LoRA paper, full text | rank ablations; the intrinsic-dimension argument; which matrices they adapted and why |
| QLoRA paper, full text | the NF4 derivation; paged optimizers; the Guanaco evaluation |
| Intrinsic Dimensionality (Aghajanyan et al. 2020) | the empirical basis for LoRA's low-rank premise — directly relevant to E17's disproof of H2 |
| LoRA-FA, DoRA, rsLoRA | variants PROJECT_INSTRUCTIONS mentions; none implemented |
| peft source, in depth | only its observable behaviour was compared |
| GPTQ / AWQ | alternative quantization schemes not used here |

**No claim in this repository rests on any of these.** In particular, E17's
finding that a real ΔW is not low-rank is stated from our own measurement, and
we have **not** read the intrinsic-dimensionality literature that would place
it in context — a gap worth closing before defending the result in an
interview.

---

## Own Words

> ## Own Words
> **[USER CHECKPOINT — DEFERRED — USER MUST WRITE THIS]**
> One entry per resource above, after reading it yourself.

**Related:** [[phase4-lora-svd-and-pca]] · [[phase3-resources]] ·
[[phase4-code-explanation]]
