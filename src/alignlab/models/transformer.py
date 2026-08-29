"""Decoder-only Transformer, assembled from the WP1-WP4 components.

THE FULL PATH, with shapes:

    token IDs        [B, T]              integers in [0, vocab_size)
        │  nn.Embedding
    embeddings       [B, T, d_model]
        │
        │  ┌──────────── repeated n_layers times ────────────┐
        │  │  h = x + Attention(Norm(x))     <- residual      │
        │  │  h = h + FFN(Norm(h))           <- residual      │
        │  └─────────────────────────────────────────────────┘
        │
    final norm       [B, T, d_model]
        │  LM head (Linear d_model -> vocab_size)
    logits           [B, T, vocab_size]

DESIGN: this is a MODERN-LLM-shaped decoder, not the 2017 original.

    pre-norm (not post-norm)   RMSNorm (not LayerNorm)
    RoPE (not sinusoidal)      SwiGLU (not ReLU MLP)
    GQA-capable                no biases on the projections

The 2017 variants all exist as separate modules for comparison; this
configuration is the one Phase 3 will actually fine-tune against, so the
integrated model is faithful to what is being studied rather than to history.

WHY PRE-NORM. Post-norm (2017) puts the normalisation AFTER the residual add:
``x = Norm(x + Sublayer(x))``. That places a normalisation on the residual
path itself, so the identity shortcut is repeatedly rescaled and gradients
degrade with depth - the original needed a learning-rate warmup to train at
all. Pre-norm normalises the sublayer INPUT and leaves the residual path clean:

    post-norm:  x = Norm(x + Sublayer(x))       normalisation ON the highway
    pre-norm:   x = x + Sublayer(Norm(x))       highway untouched

With pre-norm there is an unobstructed additive path from the embedding to the
final norm, so gradients reach early layers directly. That is why every modern
large model uses it.

WHY A FINAL NORM. With pre-norm, nothing normalises the output of the last
block - the residual stream has been accumulating unnormalised additions the
whole way up. A final norm before the LM head fixes the scale the head sees.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from alignlab.models.attention import causal_mask, scaled_dot_product_attention
from alignlab.models.feedforward import SwiGLU
from alignlab.models.kv_cache import KVCache
from alignlab.models.normalization import RMSNorm
from alignlab.models.positional import RotaryPositionalEmbedding
from alignlab.models.shapes import assert_shape


@dataclass
class TransformerConfig:
    """Configuration for the educational decoder-only model.

    Deliberately small defaults: this model is for understanding and for the
    conceptual experiments, NOT a model anyone should train seriously.
    """

    vocab_size: int = 256
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int | None = None  # None -> MHA; 1 -> MQA; else GQA
    d_ff: int | None = None  # None -> SwiGLU's 8/3 rule
    max_seq_len: int = 512
    rope_base: float = 10000.0
    norm_eps: float = 1e-6
    dropout_p: float = 0.0
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by "
                f"n_kv_heads ({self.n_kv_heads})"
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class CausalSelfAttention(nn.Module):
    """Masked multi-head self-attention with RoPE and optional KV cache.

    Separate from models.attention.MultiHeadAttention because it adds two
    things that only make sense inside a full model: rotary embeddings applied
    to Q and K, and cache-aware incremental decoding.
    """

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.dropout_p = cfg.dropout_p

        self.w_q = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.w_k = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.w_o = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)

        self.rope = RotaryPositionalEmbedding(
            self.head_dim, max_seq_len=cfg.max_seq_len, base=cfg.rope_base
        )

    def forward(
        self,
        x: torch.Tensor,
        cache: KVCache | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
            cache: when given, K/V for this layer are appended and the full
                cached history is attended over.
            layer_idx: which layer's cache slot to use.
        """
        B, T, _ = x.shape

        q = self.w_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE must be applied at the ABSOLUTE position. During cached decoding
        # the new token sits at index past_len, not 0 - getting this wrong is
        # the classic KV-cache bug (fluent output, wrong positional structure).
        offset = cache.length(layer_idx) if cache is not None else 0
        q, k = self.rope(q, k, offset=offset)

        # Append AFTER rotating, so the cache stores already-rotated keys and
        # they never need re-rotating on subsequent steps.
        if cache is not None:
            k, v = cache.update(layer_idx, k, v)

        t_k = k.shape[-2]

        if self.n_rep > 1:  # GQA / MQA
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        mask = causal_mask(T, device=x.device, key_len=t_k)

        context, _ = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout_p=self.dropout_p, training=self.training
        )
        assert_shape(context, (B, self.n_heads, T, self.head_dim), "context")

        merged = context.transpose(1, 2).contiguous().view(B, T, -1)
        return self.w_o(merged)


class DecoderBlock(nn.Module):
    """One pre-norm decoder block.

        h = x + Attention(Norm(x))
        y = h + FFN(Norm(h))

    Note the residual adds are OUTSIDE the norms - that is what keeps the
    identity path clean (see the module docstring).
    """

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, d_ff=cfg.d_ff, bias=False, dropout_p=cfg.dropout_p)

    def forward(
        self, x: torch.Tensor, cache: KVCache | None = None, layer_idx: int = 0
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cache=cache, layer_idx=layer_idx)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class DecoderOnlyTransformer(nn.Module):
    """A decoder-only language model, built from first principles.

    EDUCATIONAL. This is not Qwen and shares no weights with it. It exists so
    the architecture can be understood, tested and experimented on end to end.
    """

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(DecoderBlock(cfg) for _ in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            # Weight tying: the input embedding and output projection are the
            # SAME matrix. Saves vocab_size * d_model parameters and reflects
            # that both map between token identity and representation space.
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Small normal init, the GPT-2 convention.

        std=0.02 keeps initial activations small enough that the residual
        stream does not blow up before normalisation has anything to work with.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            input_ids: [B, T] token ids
            targets:   [B, T] next-token labels; when given, the loss is
                returned. Labels equal to -100 are ignored.
            cache: KV cache for incremental decoding.

        Returns:
            (logits [B, T, vocab_size], loss or None)
        """
        B, T = input_ids.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}")

        x = self.token_embedding(input_ids)  # [B, T, d_model]
        assert_shape(x, (B, T, self.cfg.d_model), "embeddings")

        for i, block in enumerate(self.blocks):
            x = block(x, cache=cache, layer_idx=i)

        x = self.final_norm(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]
        assert_shape(logits, (B, T, self.cfg.vocab_size), "logits")

        loss = None
        if targets is not None:
            # Flatten to [B*T, V] and [B*T]; cross_entropy applies log_softmax
            # internally, so raw logits are passed (never pre-softmaxed).
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
            )

        return logits, loss

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def parameter_breakdown(self) -> dict[str, int]:
        """Per-component parameter counts, for the documentation."""
        embed = self.token_embedding.weight.numel()
        per_block = sum(p.numel() for p in self.blocks[0].parameters())
        return {
            "embedding": embed,
            "per_block": per_block,
            "all_blocks": per_block * self.cfg.n_layers,
            "final_norm": self.final_norm.weight.numel(),
            "lm_head": 0 if self.cfg.tie_embeddings else self.lm_head.weight.numel(),
            "total": self.num_parameters(),
        }
