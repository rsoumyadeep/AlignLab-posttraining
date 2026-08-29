"""A minimal KV cache, and the argument for why it exists.

THE PROBLEM. Autoregressive generation feeds the model its own output:

    step 1:  [a]              -> predict b
    step 2:  [a, b]           -> predict c
    step 3:  [a, b, c]        -> predict d

Without a cache, step 3 recomputes K and V for tokens a and b - which have not
changed and cannot change, because the model is causal and nothing after a
position can affect it. Generating T tokens therefore costs

    sum over t of O(t)  =  O(T^2)   key/value projections

for work that is provably redundant.

THE FIX. Keep K and V for every position already processed. At each new step
project only the ONE new token, append its K/V, and attend the single new query
over the whole cached history:

    without cache:  Q [B,H,t,d]  K,V [B,H,t,d]   -> recompute everything
    with cache:     Q [B,H,1,d]  K,V [B,H,t,d]   -> only the new token projected

Per-token cost becomes O(1) projections instead of O(t), so total generation
cost drops from O(T^2) to O(T) in the projections. The attention itself is
still O(t) per step (the new query must see all t keys), so total attention
work remains O(T^2) - the cache removes the redundant PROJECTIONS and the
redundant attention rows, not the fundamental cost of attending to history.

WHY THIS IS CORRECT, not an approximation. Causal masking guarantees that the
representation at position i depends only on positions <= i. A past token's K
and V therefore can never change when a new token is appended. The cache is not
a heuristic - it recomputes nothing because there is nothing to recompute. That
is why cached and uncached generation must produce IDENTICAL logits, and why
tests/test_kv_cache.py asserts exactly that.

THE COST. Memory:

    2 (K and V) * B * n_kv_heads * T * head_dim * bytes * n_layers

which is why GQA/MQA exist (WP2): the cache scales with n_kv_heads, and at long
context it dominates inference memory.
"""

from __future__ import annotations

import torch


class KVCache:
    """Per-layer key/value cache for incremental decoding.

    Deliberately simple: a Python list of (K, V) tensors that grows by
    concatenation. Production caches pre-allocate to a maximum length and write
    into slices, avoiding a reallocation per token - but concatenation makes the
    mechanism obvious, and the mechanism is the point here.
    """

    def __init__(self, n_layers: int) -> None:
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self.n_layers = n_layers
        self._keys: list[torch.Tensor | None] = [None] * n_layers
        self._values: list[torch.Tensor | None] = [None] * n_layers

    def length(self, layer_idx: int = 0) -> int:
        """Number of positions cached for a layer. This is the RoPE offset."""
        k = self._keys[layer_idx]
        return 0 if k is None else k.shape[-2]

    def update(
        self, layer_idx: int, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V for a layer and return the full cached history.

        Args:
            layer_idx: which layer.
            key/value: [B, n_kv_heads, T_new, head_dim] for the new token(s).

        Returns:
            (K, V) each [B, n_kv_heads, T_total, head_dim].

        The keys passed in must ALREADY have RoPE applied. Caching pre-rotation
        keys would require re-rotating the whole history every step, which
        defeats the purpose.
        """
        if not 0 <= layer_idx < self.n_layers:
            raise IndexError(f"layer_idx {layer_idx} out of range for {self.n_layers} layers")

        cached_k = self._keys[layer_idx]
        if cached_k is None:
            self._keys[layer_idx] = key
            self._values[layer_idx] = value
        else:
            self._keys[layer_idx] = torch.cat([cached_k, key], dim=-2)
            self._values[layer_idx] = torch.cat([self._values[layer_idx], value], dim=-2)

        return self._keys[layer_idx], self._values[layer_idx]

    def get(self, layer_idx: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self._keys[layer_idx], self._values[layer_idx]

    def reset(self) -> None:
        """Drop all cached state - call between independent generations."""
        self._keys = [None] * self.n_layers
        self._values = [None] * self.n_layers

    def memory_bytes(self) -> int:
        """Actual bytes currently held, summed over layers."""
        total = 0
        for k, v in zip(self._keys, self._values):
            if k is not None:
                total += k.numel() * k.element_size()
            if v is not None:
                total += v.numel() * v.element_size()
        return total

    @staticmethod
    def predicted_bytes(
        batch: int,
        n_kv_heads: int,
        seq_len: int,
        head_dim: int,
        n_layers: int,
        bytes_per_element: int = 2,
    ) -> int:
        """The formula, for comparison against memory_bytes().

        2 * B * n_kv_heads * T * head_dim * bytes * n_layers
        """
        return 2 * batch * n_kv_heads * seq_len * head_dim * bytes_per_element * n_layers

    def __repr__(self) -> str:
        return (
            f"KVCache(n_layers={self.n_layers}, length={self.length(0)}, "
            f"bytes={self.memory_bytes():,})"
        )
