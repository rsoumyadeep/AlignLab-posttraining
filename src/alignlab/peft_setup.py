"""Translate AlignLab's PEFT config into what the libraries expect.

A thin, deliberate layer. It exists so that:

  * the LoRA/QLoRA decision lives in ONE place and is recorded in the manifest,
    rather than being spread through the training entrypoint;
  * the QLoRA mechanics are written down where they are actually configured,
    not only in a document that can drift from the code;
  * the three comparison arms differ in exactly the settings we intend and
    nothing else.

WHAT QLORA ACTUALLY IS, since "LoRA + 4-bit" is not an explanation.

Three separate things happen, and only the first is about bit-width:

  1. STORAGE. The frozen base weights are stored 4-bit NF4 instead of bf16.
     NF4's 16 quantization levels sit at the quantiles of a normal
     distribution, which is approximately how trained weights are distributed -
     E15 measured NF4 at 24.6% lower relative error than FP4 on normal data.
     Quantization is per-block (default 64 elements) with a per-block absmax
     scale, so an outlier weight degrades only its own block.

  2. DOUBLE QUANTIZATION. Those per-block absmax constants are themselves
     quantized, saving roughly 0.4 bits per parameter. Small, but it is one of
     the three things the QLoRA paper contributes and it is on by default here.

  3. COMPUTE IS NOT 4-BIT. Every forward pass DEQUANTIZES the block it needs
     back to ``compute_dtype`` (bf16) and does an ordinary bf16 matmul. The
     4 bits are a storage format, not an arithmetic mode. This is why QLoRA is
     slower per step than LoRA despite using less memory - it pays a
     dequantization cost on every matmul that LoRA does not.

  4. GRADIENTS NEVER TOUCH THE BASE. The base is frozen, so no optimizer state
     exists for it at all. Gradients flow THROUGH the dequantized base weights
     to the adapters - the base is part of the computation graph as a constant.
     This is what makes the memory saving compound: 4-bit weights AND no
     optimizer moments for 99.7% of the parameters.

WHY THE PEFT LIBRARY IS USED HERE. PROJECT_INSTRUCTIONS section 4 requires the
core mechanism implemented manually - that is ``alignlab.lora``, which is
independently tested by 46 unit tests and compared against this path in E18.
For the actual multi-hour training runs the library implementation is the
right tool: it handles gradient checkpointing interaction, dtype casting of
norm layers, and quantized-base training details that are engineering rather
than understanding.
"""

from __future__ import annotations

from typing import Any

import torch

from alignlab.logging_utils import get_logger

logger = get_logger(__name__)

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def build_quantization_config(cfg) -> Any | None:
    """Return a ``BitsAndBytesConfig`` for QLoRA, or None otherwise.

    Raises rather than silently degrading if bitsandbytes is unavailable: a
    "QLoRA" run that quietly fell back to bf16 would produce memory numbers
    that are wrong in the most misleading possible way.
    """
    if cfg.peft.method != "qlora":
        return None

    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "QLoRA requested but BitsAndBytesConfig is unavailable. Refusing to "
            "fall back to unquantized training, which would silently invalidate "
            "the memory comparison."
        ) from exc

    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "QLoRA requested but bitsandbytes is not installed. Run "
            "scripts/experiments/e15_bitsandbytes_environment.py first - it "
            "verifies the kernel actually matches this machine's CUDA."
        ) from exc

    compute_dtype = _DTYPES.get(cfg.peft.compute_dtype)
    if compute_dtype is None:
        raise ValueError(
            f"unknown compute_dtype {cfg.peft.compute_dtype!r}; "
            f"expected one of {sorted(_DTYPES)}"
        )

    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.peft.quant_type,
        bnb_4bit_use_double_quant=cfg.peft.double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    logger.info(
        "QLoRA quantization: 4-bit %s, double_quant=%s, compute_dtype=%s "
        "(STORAGE is 4-bit; the matmul is %s)",
        cfg.peft.quant_type, cfg.peft.double_quant, cfg.peft.compute_dtype,
        cfg.peft.compute_dtype,
    )
    return config


def build_peft_config(cfg) -> Any | None:
    """Return a ``peft.LoraConfig`` for lora/qlora, or None for full FT."""
    if cfg.peft.method == "none":
        return None

    try:
        from peft import LoraConfig
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            f"peft.method={cfg.peft.method!r} requested but peft is not "
            f"installed. Refusing to silently train all parameters instead."
        ) from exc

    targets = list(cfg.peft.target_modules)
    config = LoraConfig(
        r=cfg.peft.r,
        lora_alpha=cfg.peft.alpha,
        lora_dropout=cfg.peft.dropout,
        target_modules=targets,
        bias=cfg.peft.bias,
        task_type="CAUSAL_LM",
    )
    logger.info(
        "LoRA config: r=%d alpha=%s (scaling %.2f) dropout=%s targets=%s",
        cfg.peft.r, cfg.peft.alpha, cfg.peft.alpha / cfg.peft.r,
        cfg.peft.dropout, targets,
    )
    return config


def describe_peft(cfg) -> dict:
    """A manifest-ready record of the arm being run."""
    record = {
        "method": cfg.peft.method,
        "full_finetune": cfg.peft.method == "none",
    }
    if cfg.peft.method != "none":
        record.update(
            {
                "r": cfg.peft.r,
                "alpha": cfg.peft.alpha,
                "scaling": cfg.peft.alpha / cfg.peft.r,
                "dropout": cfg.peft.dropout,
                "target_modules": list(cfg.peft.target_modules),
                "bias": cfg.peft.bias,
            }
        )
    if cfg.peft.method == "qlora":
        record.update(
            {
                "quant_type": cfg.peft.quant_type,
                "double_quant": cfg.peft.double_quant,
                "compute_dtype": cfg.peft.compute_dtype,
                "storage_bits": 4,
                "compute_is_4bit": False,
            }
        )
    return record


def summarise_trainable(model) -> dict:
    """Count trainable vs frozen parameters on a (possibly wrapped) model.

    Counts by iterating the model rather than trusting peft's own printout, so
    the number in the manifest is one this project measured.

    NOTE ON 4-BIT PARAMETERS. bitsandbytes stores a quantized weight as a
    packed uint8 tensor holding TWO 4-bit values per byte, so ``numel()`` on a
    ``Params4bit`` returns the number of BYTES, not the number of logical
    parameters. The raw count is therefore reported alongside a corrected
    ``logical_total`` that doubles those tensors back.
    """
    total = 0
    trainable = 0
    quantized_elements = 0

    for param in model.parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
        if param.__class__.__name__ == "Params4bit":
            quantized_elements += n

    logical_total = total + quantized_elements  # each packed byte holds 2 values

    return {
        "raw_total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "quantized_packed_elements": quantized_elements,
        "logical_total": logical_total,
        "trainable_fraction_of_logical": (
            trainable / logical_total if logical_total else 0.0
        ),
    }
