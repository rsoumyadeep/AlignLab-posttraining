"""Scaled dot-product attention in pure Python. No NumPy, no PyTorch.

Reference implementation #3 of 3, and the one that matters most for
understanding. Everything is nested lists and ``math``; the only imported
numerics are ``math.exp`` and ``math.sqrt``.

WHY THIS EXISTS. A framework lets you write attention without ever deciding
what a matrix multiply *is*. Here you must write the loops, so there is nowhere
for a misunderstanding to hide:

  * the inner product runs over d_k - the FEATURE axis
  * the softmax runs over T_k - the KEY axis
  * the value average runs over T_k as well, but contracts a different axis

If you can write this file from memory, you understand attention. That is
Drill 4 in PROJECT_INSTRUCTIONS section 13.

PERFORMANCE. This is O(T_q * T_k * d) in interpreted Python and is roughly
1000x slower than the array versions. It is a CORRECTNESS REFERENCE, never a
benchmark. Keep the dimensions tiny (T <= 8, d <= 8).

PRECISION. Python floats are IEEE-754 doubles, so this implementation is
effectively float64. Compare it against float64 NumPy, not float32 torch - see
tests/test_attention_equivalence.py for the declared tolerances.

TYPES. A 2-D "matrix" is list[list[float]]; a 3-D batch is list of matrices.
Only the unbatched 2-D case is implemented, deliberately: batching adds
bookkeeping and teaches nothing new.
"""

from __future__ import annotations

import math

Matrix = list[list[float]]
Vector = list[float]


def dot(a: Vector, b: Vector) -> float:
    """Inner product over the FEATURE axis. sum_i a_i * b_i."""
    if len(a) != len(b):
        raise ValueError(f"dot: length mismatch {len(a)} vs {len(b)}")
    total = 0.0
    for x, y in zip(a, b):
        total += x * y
    return total


def transpose(m: Matrix) -> Matrix:
    """[rows, cols] -> [cols, rows]."""
    if not m:
        return []
    return [[m[r][c] for r in range(len(m))] for c in range(len(m[0]))]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    """[n, k] @ [k, p] -> [n, p]. The contracted axis is k."""
    if not a or not b:
        raise ValueError("matmul: empty operand")
    n, k, p = len(a), len(b), len(b[0])
    if len(a[0]) != k:
        raise ValueError(f"matmul: inner dims differ, {len(a[0])} vs {k}")

    b_t = transpose(b)  # so each column of b becomes a contiguous row
    return [[dot(a[i], b_t[j]) for j in range(p)] for i in range(n)]


def softmax_row(row: Vector) -> Vector:
    """Numerically stable softmax over ONE row (i.e. over the key axis).

    Subtract the max first: exp(1000) overflows a double, exp(0) does not, and
    the shift cancels in the ratio. Entries of -inf become exactly 0.
    """
    finite = [x for x in row if x != float("-inf")]
    if not finite:
        raise ValueError(
            "softmax_row: every entry is masked; a query attending to nothing "
            "has no defined distribution"
        )
    row_max = max(finite)

    exps = []
    for x in row:
        if x == float("-inf"):
            exps.append(0.0)
        else:
            exps.append(math.exp(x - row_max))

    total = sum(exps)
    return [e / total for e in exps]


def causal_mask(seq_len: int, key_len: int | None = None) -> list[list[bool]]:
    """[T_q, T_k]; True means allowed to attend."""
    key_len = seq_len if key_len is None else key_len
    if key_len < seq_len:
        raise ValueError(f"key_len ({key_len}) must be >= seq_len ({seq_len})")
    offset = key_len - seq_len
    return [[k <= q + offset for k in range(key_len)] for q in range(seq_len)]


def scaled_dot_product_attention(
    query: Matrix,
    key: Matrix,
    value: Matrix,
    mask: list[list[bool]] | None = None,
    scale: float | None = None,
) -> tuple[Matrix, Matrix]:
    """Attention(Q, K, V) = softmax(Q Kᵀ / sqrt(d_k)) V, unbatched.

    Args:
        query: [T_q, d_k]
        key:   [T_k, d_k]
        value: [T_k, d_v]
        mask:  [T_q, T_k]; True means KEEP.
        scale: overrides 1/sqrt(d_k); pass 1.0 to disable.

    Returns:
        (context [T_q, d_v], weights [T_q, T_k])
    """
    t_q, d_k = len(query), len(query[0])
    t_k = len(key)
    if len(key[0]) != d_k:
        raise ValueError(f"query d_k {d_k} != key d_k {len(key[0])}")
    if len(value) != t_k:
        raise ValueError(f"key T_k {t_k} != value T_k {len(value)}")

    if scale is None:
        scale = 1.0 / math.sqrt(d_k)

    # --- 1 & 2. scores = Q Kᵀ / sqrt(d_k)   [T_q, T_k] ----------------------
    # Written as an explicit double loop rather than matmul(), so the shape of
    # the computation is unmistakable: for every (query, key) PAIR, contract
    # over the feature axis.
    scores: Matrix = []
    for i in range(t_q):
        row: Vector = []
        for j in range(t_k):
            row.append(dot(query[i], key[j]) * scale)
        scores.append(row)

    # --- 3. mask BEFORE softmax -------------------------------------------
    if mask is not None:
        for i in range(t_q):
            for j in range(t_k):
                if not mask[i][j]:
                    scores[i][j] = float("-inf")

    # --- 4. softmax over the KEY axis, one distribution per query ----------
    weights: Matrix = [softmax_row(row) for row in scores]

    # --- 5. context = weights @ V   [T_q, T_k] @ [T_k, d_v] -> [T_q, d_v] ---
    d_v = len(value[0])
    context: Matrix = []
    for i in range(t_q):
        out_row = [0.0] * d_v
        for j in range(t_k):
            w = weights[i][j]
            if w == 0.0:
                continue  # masked or vanishing; skip for clarity and speed
            for f in range(d_v):
                out_row[f] += w * value[j][f]
        context.append(out_row)

    return context, weights


def multi_head_attention(
    query: Matrix,
    key: Matrix,
    value: Matrix,
    n_heads: int,
    mask: list[list[bool]] | None = None,
) -> tuple[Matrix, list[Matrix]]:
    """Multi-head attention by SLICING the feature axis, unbatched.

    Takes already-projected Q/K/V of width ``n_heads * d_k`` and splits the
    feature axis into ``n_heads`` contiguous slices. This is exactly what
    ``view(B, T, H, d_k).transpose(1, 2)`` does in the PyTorch version - the
    head split is a reinterpretation of the feature axis, nothing more. Seeing
    it as slicing removes most of the mystery from the reshape.

    Returns:
        (concatenated context [T_q, n_heads*d_v], per-head weights)
    """
    d_total = len(query[0])
    if d_total % n_heads != 0:
        raise ValueError(f"width {d_total} not divisible by n_heads {n_heads}")
    d_head = d_total // n_heads

    contexts: list[Matrix] = []
    all_weights: list[Matrix] = []

    for h in range(n_heads):
        lo, hi = h * d_head, (h + 1) * d_head
        q_h = [row[lo:hi] for row in query]
        k_h = [row[lo:hi] for row in key]
        v_h = [row[lo:hi] for row in value]

        ctx_h, w_h = scaled_dot_product_attention(q_h, k_h, v_h, mask=mask)
        contexts.append(ctx_h)
        all_weights.append(w_h)

    # Concatenate along the feature axis - the inverse of the slicing above.
    merged: Matrix = []
    for i in range(len(query)):
        row: Vector = []
        for h in range(n_heads):
            row.extend(contexts[h][i])
        merged.append(row)

    return merged, all_weights
