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

__all__ = [
    "MultiHeadAttention",
    "causal_mask",
    "scaled_dot_product_attention",
]
