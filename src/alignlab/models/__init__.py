"""AlignLab model components, built from first principles (Phase 2).

Two levels, per PROJECT_INSTRUCTIONS section 21:

    Level 1 - educational, minimal, readable. Every step visible.
    Level 2 - practical integration into a working decoder-only model.

The three attention implementations (PyTorch / NumPy / pure Python) express the
SAME algorithm so they can be cross-checked against one another.
"""

from alignlab.models.attention import (
    MultiHeadAttention,
    causal_mask,
    scaled_dot_product_attention,
)
from alignlab.models.feedforward import FeedForward, SwiGLU
from alignlab.models.generation import beam_search, generate
from alignlab.models.kv_cache import KVCache
from alignlab.models.normalization import LayerNorm, RMSNorm
from alignlab.models.positional import (
    ALiBiBias,
    LearnedAbsolutePositionalEmbedding,
    RotaryPositionalEmbedding,
    SinusoidalPositionalEncoding,
)
from alignlab.models.transformer import (
    DecoderBlock,
    DecoderOnlyTransformer,
    TransformerConfig,
)

__all__ = [
    # attention
    "MultiHeadAttention",
    "causal_mask",
    "scaled_dot_product_attention",
    # positional
    "LearnedAbsolutePositionalEmbedding",
    "SinusoidalPositionalEncoding",
    "RotaryPositionalEmbedding",
    "ALiBiBias",
    # norm + ffn
    "LayerNorm",
    "RMSNorm",
    "FeedForward",
    "SwiGLU",
    # model
    "TransformerConfig",
    "DecoderBlock",
    "DecoderOnlyTransformer",
    "KVCache",
    # generation
    "generate",
    "beam_search",
]
