"""Positional encoding: learned absolute, sinusoidal, RoPE, ALiBi.

WHY ANY OF THIS EXISTS. Attention is permutation-equivariant: permute the input
tokens and the output permutes identically (verified in
tests/test_attention.py::test_attention_without_mask_is_permutation_equivariant).
So attention alone cannot distinguish "dog bites man" from "man bites dog".
Position must be injected from outside.

THE ORGANISING AXIS - absolute vs relative:

    ABSOLUTE  the encoding depends on a position index i. "I am token 5."
              Learned embeddings and sinusoidal PE are absolute, added to the
              token embedding once, before the first block.

    RELATIVE  the encoding depends on the DISTANCE i - j between a query and a
              key. "You are 3 tokens behind me." RoPE and ALiBi are relative
              and act INSIDE attention, on every layer.

Why the distinction matters: language regularities are mostly relative (an
adjective modifies a nearby noun regardless of where the sentence starts), and
relative schemes extrapolate to unseen lengths far better - a learned absolute
embedding for position 5000 that was never trained is simply undefined.

    method       type       where applied        learned  extrapolates
    ------------------------------------------------------------------
    learned      absolute   added to embeddings  yes      no (hard limit)
    sinusoidal   absolute   added to embeddings  no       in principle
    RoPE         relative   Q and K, per layer   no       moderately
    ALiBi        relative   score bias, layer    no       well
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ==========================================================================
# 1. Learned absolute positional embeddings
# ==========================================================================


class LearnedAbsolutePositionalEmbedding(nn.Module):
    """A lookup table indexed by position. What GPT-2 uses.

    Simple and effective, with one hard limitation: the table has
    ``max_seq_len`` rows, so position ``max_seq_len`` has NO embedding. The
    model cannot process a longer sequence at all - not "degrades gracefully",
    but "index error". Extrapolation is impossible by construction.
    """

    def __init__(self, max_seq_len: int, d_model: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, T, d_model] -> x + PE[:T], same shape."""
        B, T, _ = x.shape
        if T > self.max_seq_len:
            raise ValueError(
                f"sequence length {T} exceeds max_seq_len {self.max_seq_len}: a "
                "learned absolute embedding has no value for unseen positions"
            )
        positions = torch.arange(T, device=x.device)  # [T]
        return x + self.embedding(positions).unsqueeze(0)  # [1, T, d] broadcast


# ==========================================================================
# 2. Sinusoidal positional encoding
# ==========================================================================


class SinusoidalPositionalEncoding(nn.Module):
    """The original Transformer encoding (Vaswani et al., 2017).

        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    DERIVATION SKETCH (full version in STUDY_WITH_CLAUDE/phase2/03).

    We want a fixed (unlearned) function of position such that:
      1. each position gets a unique code;
      2. the code is bounded, so it does not swamp the token embedding;
      3. relative offsets are LINEARLY recoverable, so the model can learn to
         attend "k positions back".

    A single sine gives uniqueness only up to its period. Using many
    frequencies - a geometric progression from 1 down to 1/10000 - produces
    something like a binary counter written in continuous values: fast
    dimensions flip quickly, slow dimensions distinguish distant positions.

    Property 3 is the elegant part. For a fixed offset k, the pair
    (sin(w·pos), cos(w·pos)) at pos+k is a ROTATION by w·k of the pair at pos:

        [sin(w(pos+k))]   [cos(wk)   sin(wk)] [sin(w·pos)]
        [cos(w(pos+k))] = [-sin(wk)  cos(wk)] [cos(w·pos)]

    The matrix depends on k but NOT on pos. So a linear map can implement
    "shift by k" uniformly across the sequence.

    That rotation observation is exactly the seed RoPE grows from - the
    difference is WHERE it is applied. Sinusoidal ADDS this to the embedding
    (so the relative structure must survive the attention projections); RoPE
    ROTATES Q and K directly (so it is exact and cannot be washed out).

    The encoding is a BUFFER, not a parameter: it is fixed, must move with
    .to(device), and must be saved in the state_dict.
    """

    def __init__(self, max_seq_len: int, d_model: int, base: float = 10000.0) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even for sin/cos pairs, got {d_model}")
        self.d_model = d_model
        self.base = base

        pe = self._build(max_seq_len, d_model, base)  # [max_seq_len, d_model]
        self.register_buffer("pe", pe, persistent=True)

    @staticmethod
    def _build(max_seq_len: int, d_model: int, base: float) -> torch.Tensor:
        position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)  # [T,1]

        # inv_freq[i] = 1 / base^(2i/d_model), computed in log space for
        # stability: base^(-2i/d) = exp(-2i/d * ln base).
        i = torch.arange(0, d_model, 2, dtype=torch.float32)  # [d_model/2]
        inv_freq = torch.exp(-math.log(base) * i / d_model)  # [d_model/2]

        angles = position * inv_freq.unsqueeze(0)  # [T, d_model/2]

        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(angles)  # even dims
        pe[:, 1::2] = torch.cos(angles)  # odd dims
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, T, d_model] -> x + PE[:T]."""
        T = x.shape[1]
        if T > self.pe.shape[0]:
            raise ValueError(f"sequence length {T} exceeds precomputed {self.pe.shape[0]}")
        return x + self.pe[:T].unsqueeze(0).to(dtype=x.dtype)


# ==========================================================================
# 3. RoPE - Rotary Position Embedding
# ==========================================================================


class RotaryPositionalEmbedding(nn.Module):
    """RoPE (Su et al., 2021). Used by Llama, Qwen, Mistral, and most modern LLMs.

    INTUITION. Instead of ADDING a position vector, ROTATE each query and key
    by an angle proportional to its position. Split the head dimension into 2-D
    pairs and rotate pair p of a vector at position m by angle m·theta_p.

    WHY THAT GIVES RELATIVE POSITION. The dot product of two rotated 2-D
    vectors depends only on the DIFFERENCE of their rotation angles:

        <R(m)q, R(n)k> = qᵀ R(m)ᵀ R(n) k = qᵀ R(n - m) k

    because rotation matrices satisfy R(a)ᵀR(b) = R(b - a). Attention consumes
    exactly this inner product, so the score between positions m and n depends
    on (m - n) alone - absolute position CANCELS. That is the whole trick, and
    it is verified numerically in tests/test_positional.py.

    Two consequences worth stating:
      * RoPE preserves vector NORMS (rotations are orthogonal), so it does not
        change the scale of Q or K.
      * It is applied to Q and K INSIDE every attention layer, not once to the
        embeddings - so the relative structure cannot be washed out by the
        projections.

    theta_p = base^(-2p/d) - the same geometric frequency ladder as sinusoidal
    PE. ``base`` (often called rope_theta) is a real tuning knob: raising it
    lengthens the low-frequency wavelengths and is the basis of several
    context-extension methods.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for 2-D rotation pairs, got {head_dim}")
        self.head_dim = head_dim
        self.base = base

        # theta_p for p = 0 .. head_dim/2 - 1
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )  # [head_dim/2]
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        cos, sin = self._build_tables(max_seq_len)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def _build_tables(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, dtype=torch.float32)  # [T]
        angles = torch.outer(t, self.inv_freq)  # [T, head_dim/2]
        # Duplicate so each half-index maps onto the interleave-free "rotate
        # half" layout used below: [T, head_dim]
        emb = torch.cat([angles, angles], dim=-1)
        return emb.cos(), emb.sin()

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """(x1, x2) -> (-x2, x1) on the two halves of the last axis.

        This is the "split in half" convention (Llama/Qwen), not the
        "interleave adjacent" convention of the original paper. They are
        equivalent up to a fixed permutation of the head dimension - what
        matters is that Q and K use the SAME convention. Mixing them is a
        classic silent bug: the model still trains, just worse.
        """
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate q and k by their positions.

        Args:
            q: [B, H, T, head_dim]
            k: [B, H_kv, T, head_dim]
            offset: absolute position of the FIRST token. Non-zero during
                cached generation, where the new token is at position
                past_len rather than 0. Getting this wrong is the most common
                KV-cache bug - the model generates fluent text with subtly
                wrong positional structure.

        Returns:
            (q_rotated, k_rotated), same shapes as the inputs.
        """
        T = q.shape[-2]
        needed = offset + T
        if needed > self.cos_cached.shape[0]:
            cos, sin = self._build_tables(needed)
            self.cos_cached = cos.to(q.device)
            self.sin_cached = sin.to(q.device)

        cos = self.cos_cached[offset : offset + T].to(q.device, q.dtype)  # [T, hd]
        sin = self.sin_cached[offset : offset + T].to(q.device, q.dtype)
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, T, hd] broadcast over B, H
        sin = sin.unsqueeze(0).unsqueeze(0)

        q_out = q * cos + self._rotate_half(q) * sin
        k_out = k * cos + self._rotate_half(k) * sin
        return q_out, k_out


# ==========================================================================
# 4. ALiBi - Attention with Linear Biases
# ==========================================================================


class ALiBiBias(nn.Module):
    """ALiBi (Press et al., 2021): no positional vectors at all.

    Instead of touching the embeddings or Q/K, add a distance-proportional
    PENALTY directly to the attention scores:

        score(i, j) += -m_h * (i - j)

    where ``i - j >= 0`` is how far back key j is, and m_h is a per-head slope
    fixed in advance (a geometric sequence). Nearby tokens are penalised less,
    so each head develops a characteristic "attention range": small slopes look
    far, large slopes look locally.

    CONTRAST WITH RoPE. RoPE modifies the vectors being compared, so position
    affects the score through the geometry of Q and K. ALiBi leaves the vectors
    untouched and edits the score directly. ALiBi is the cruder mechanism -
    a monotone decay, no phase structure - but it extrapolates to longer
    contexts remarkably well, precisely because a linear penalty keeps meaning
    something at lengths never trained on.

    Slopes follow the paper's geometric schedule: for n heads a power of two,
    ratio 2^(-8/n) starting at 2^(-8/n).
    """

    def __init__(self, n_heads: int, max_seq_len: int = 4096) -> None:
        super().__init__()
        self.n_heads = n_heads
        slopes = self._slopes(n_heads)  # [H]
        self.register_buffer("slopes", slopes, persistent=False)
        self.register_buffer("bias", self._build(n_heads, max_seq_len, slopes), persistent=False)

    @staticmethod
    def _slopes(n_heads: int) -> torch.Tensor:
        """Geometric slope sequence, one per head."""

        def powers_of_two_slopes(n: int) -> list[float]:
            start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3)))
            return [start ** (i + 1) for i in range(n)]

        if math.log2(n_heads).is_integer():
            return torch.tensor(powers_of_two_slopes(n_heads), dtype=torch.float32)

        # Non-power-of-two: take the next power of two below, then interleave
        # from the next power above - the paper's own fallback.
        closest = 2 ** math.floor(math.log2(n_heads))
        slopes = powers_of_two_slopes(closest)
        extra = powers_of_two_slopes(2 * closest)[0::2][: n_heads - closest]
        return torch.tensor(slopes + extra, dtype=torch.float32)

    @staticmethod
    def _build(n_heads: int, seq_len: int, slopes: torch.Tensor) -> torch.Tensor:
        """[1, H, T, T] additive bias; 0 on the diagonal, more negative further back."""
        pos = torch.arange(seq_len)
        # distance[i, j] = i - j, positive for keys in the past
        distance = pos.unsqueeze(0) - pos.unsqueeze(1)  # [T, T], = j - i
        distance = distance.to(torch.float32)  # negative for past keys
        return distance.unsqueeze(0).unsqueeze(0) * slopes.view(1, n_heads, 1, 1)

    def forward(self, seq_len: int, device: torch.device | None = None) -> torch.Tensor:
        """Return [1, H, T, T] to ADD to the attention scores before softmax."""
        if seq_len > self.bias.shape[-1]:
            bias = self._build(self.n_heads, seq_len, self.slopes.cpu())
            self.bias = bias.to(self.bias.device)
        out = self.bias[..., :seq_len, :seq_len]
        return out.to(device) if device is not None else out
