# PyTorch: quantization, 4-bit storage, and low-rank adapters

Concepts encountered building Phase 4. Every number below is **measured** — on
`csrslave`, 2 × RTX A6000, torch 2.6.0+cu124, bitsandbytes 0.50.2 — by
`scripts/experiments/e15_bitsandbytes_environment.py`, or executed locally by
`examples/svd_pca_examples.py`. Nothing here is quoted from a paper.

---

## 1. What "4-bit" actually means

### It is a STORAGE format, not an arithmetic mode

The single most common misunderstanding. A `Linear4bit` forward does:

```
packed uint8 weights  --dequantize-->  bfloat16 block  --matmul-->  bf16 output
```

The matmul is **bf16**. There is no 4-bit multiplication. What you save is
memory; what you pay is a dequantization on every forward.

**Measured:** a `Linear4bit` (NF4) forward against a bf16 `nn.Linear` holding
the *same* weights differs by relative error **0.0910**. That error is
*quantization error in the stored weights*, not reduced-precision arithmetic.

### Packing: `numel()` lies

bitsandbytes stores two 4-bit values per byte, so a `Params4bit` tensor's
`numel()` returns the number of **bytes**, not logical parameters.

```python
# WRONG - undercounts a 4-bit model's parameters by 2x
total = sum(p.numel() for p in model.parameters())
```

`alignlab.peft_setup.summarise_trainable` corrects for this by doubling
`Params4bit` tensors back. Without the correction the Phase 4 manifest would
have reported a 1.5B model as roughly 0.77B.

### Block-wise quantization

Quantization is per-block (default 64 elements), each with its own `absmax`
scale. So an outlier weight degrades only its own block rather than the whole
tensor — which is why a single large weight does not destroy the whole matrix's
fidelity.

**Consequence for memory accounting:** a 4-bit model is *not* 4× smaller. The
packed weights are exactly 4× smaller (measured: 8,388,608 B → 2,097,152 B,
ratio **4.00×**), but the per-block scales are extra. Double quantization
quantizes those scales too, recovering roughly 0.4 bits/parameter.

---

## 2. NF4 vs FP4 — a measurable difference

Both are 4-bit (16 levels). They differ in *where* the levels sit.

- **FP4** — floating-point-like spacing.
- **NF4** — levels at the **quantiles of a normal distribution**, which is
  approximately how trained neural network weights are distributed.

**Measured**, 4096×4096 bf16 tensor drawn from N(0,1), round-trip error:

| type | mean abs error | max abs error | relative error |
|---|---:|---:|---:|
| **NF4** | 0.07277 | 0.59375 | **0.0912** |
| FP4 | 0.09652 | 0.70312 | 0.1210 |

NF4 is **24.6% better** in relative error on normally-distributed data. This is
a measurement of the QLoRA paper's central claim about NF4 — on *random normal
data*, not on real weights, which is a weaker test than it sounds and is
labelled as such in E15.

This is why `configs/peft/qlora.yaml` sets `quant_type: nf4` and says why.

---

## 3. Verifying a native extension actually works

`import bitsandbytes` succeeds in plenty of broken installations — wrong
kernel, CPU fallback, silently degraded paths. An import is not a verification.

What E15 checks instead:

```python
import bitsandbytes.cextension as cext
lib = getattr(cext, "lib", None)
print(getattr(lib, "_name", None))
# -> .../bitsandbytes/libbitsandbytes_cuda124.so
```

**Measured:** the loaded library is `libbitsandbytes_cuda124.so`, matching
`torch.version.cuda == "12.4"` exactly. That is what makes the missing `nvcc`
irrelevant — the kernel ships prebuilt, so nothing is compiled from source.

**Why this mattered here:** the same machine could not build `flash-attn` in
Phase 2 (no `nvcc`, no root). It was a live question whether QLoRA was possible
at all, and the honest answer required checking rather than assuming.

---

## 4. `requires_grad`, frozen bases, and where memory goes

LoRA's memory saving is **not** mainly the adapter's small size. It is that a
frozen parameter has **no optimizer state**.

For AdamW at 1.54B parameters:

```
full fine-tune :  params + grads + exp_avg + exp_avg_sq   (all 1.54B)
LoRA           :  params (frozen, no grad, no moments)
                  + adapters 4.36M with grads and moments
```

**Measured peak VRAM** (smoke runs, identical settings otherwise):

| arm | peak allocated |
|---|---:|
| LoRA | 4.44 GiB |
| QLoRA | **2.60 GiB** |

QLoRA compounds the saving: 4-bit storage for the frozen base *and* no
optimizer state for 99.7% of the parameters.

### Gradients still flow *through* the frozen base

Freezing means `requires_grad=False`, which removes the parameter from the
optimizer — it does **not** remove it from the computation graph. Activations
still backpropagate through the frozen (and, for QLoRA, dequantized) weights to
reach the adapters. This is why QLoRA still needs activations in memory and why
gradient checkpointing still helps.

---

## 5. Low-rank factorisation: when it actually saves

`r·(d_in + d_out)` versus `d_in·d_out`. Break-even is at

```
r* = (d_in · d_out) / (d_in + d_out)
```

**Executed** (`examples/svd_pca_examples.py` §6):

| shape | dense | r=16 | r=256 | r=730 |
|---|---:|---:|---:|---:|
| 1536×1536 | 2,359,296 | 49,152 (**97.9%** saved) | 786,432 (66.7%) | 2,242,560 (**4.9%**) |
| 256×1536 | 393,216 | 28,672 (92.7%) | 458,752 (**NO SAVING**) | — |
| 8960×1536 | 13,762,560 | 167,936 (98.8%) | 2,686,976 (80.5%) | 7,662,080 (44.3%) |

For a 1536×1536 matrix, break-even is **rank 768**. Above it, a "low-rank"
factorisation costs *more* than the dense update it approximates. For the
narrow GQA projections the crossover comes much sooner — at rank 256 the
factorisation already costs more than dense.

This is what makes E17's measured `r@90% ≈ 730` so pointed: it is just under
break-even, so following the naive SVD recommendation would save 4.9%.

---

## 6. Common mistakes

1. **Calling 4-bit an arithmetic mode.** Storage only; compute is bf16.
2. **`numel()` on `Params4bit`** — returns bytes, undercounts by 2×.
3. **Assuming a 4-bit model is 4× smaller** — packed weights are; the
   per-block scales are not.
4. **Accumulating raw singular values** for "explained energy" — must be
   squared (Eckart–Young).
5. **Forgetting the break-even rank** — "low-rank" is only low-rank below it.
6. **Trusting an import as a verification** of a native CUDA extension.
7. **Believing freezing removes a parameter from the graph** — it removes it
   from the optimizer only.

---

## Connection to AlignLab

| Concept | Where |
|---|---|
| NF4 / FP4, block-wise quantization | `configs/peft/qlora.yaml`, `alignlab.peft_setup` |
| `Params4bit` packing correction | `peft_setup.summarise_trainable` |
| kernel verification | `scripts/experiments/e15` |
| break-even rank | `scripts/experiments/e17`, `examples/svd_pca_examples.py` |
| frozen base, no optimizer state | `alignlab.lora.LoRALinear.__init__` |

---

## My Understanding

> **[USER CHECKPOINT — DEFERRED — USER EXPLAIN-BACK REQUIRED]**
>
> - Explain exactly what happens to a weight between disk and matmul in a
>   QLoRA forward pass. At which step does bf16 first appear?
> - Why is a 4-bit model not 4× smaller on disk?
> - `numel()` on a `Params4bit` returns 8,388,608 for a 4096×4096 matrix.
>   How many logical parameters is that, and why?
> - Derive the break-even rank, and say what it implies for the narrow GQA
>   projections specifically.

**Related:** [[phase4-lora-svd-and-pca]] · [[phase4-code-explanation]] ·
[[pytorch-loss-masking-and-shifts]]
