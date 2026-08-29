"""Shape assertion helpers.

Phase 2's central discipline is that every tensor's shape is known and stated,
not assumed. These helpers make a shape claim executable, so a wrong claim
fails immediately with a readable message instead of surfacing 40 lines later
as a broadcasting surprise.

The naming convention used throughout AlignLab's model code:

    B       batch size
    T       sequence length (time / positions)
    T_q     query sequence length   (== T for self-attention)
    T_k     key/value sequence length
    d_model model / embedding dimension
    H       number of query heads
    H_kv    number of key-value heads (H for MHA, 1 for MQA, H/G for GQA)
    d_k     per-head key/query dimension  (usually d_model // H)
    d_v     per-head value dimension      (usually d_k)
    V       vocabulary size
"""

from __future__ import annotations

from typing import Any, Sequence

DimSpec = Sequence[int | str | None]


def assert_shape(tensor: Any, expected: DimSpec, name: str = "tensor") -> None:
    """Assert a tensor's shape, allowing named and wildcard dimensions.

    Args:
        tensor: anything with a ``.shape``.
        expected: per-dimension specification. An ``int`` must match exactly,
            ``None`` matches anything, and a ``str`` is a label used only to
            make the error message readable.
        name: label for the tensor in the error message.

    Example:
        assert_shape(q, (B, "H", T, d_k), "q")
    """
    actual = tuple(tensor.shape)

    if len(actual) != len(expected):
        raise AssertionError(
            f"{name}: expected {len(expected)} dimensions {tuple(expected)}, "
            f"got {len(actual)} dimensions {actual}"
        )

    for axis, (got, want) in enumerate(zip(actual, expected)):
        if isinstance(want, int) and got != want:
            raise AssertionError(
                f"{name}: dimension {axis} expected {want}, got {got} "
                f"(full shape {actual}, expected {tuple(expected)})"
            )


def describe(tensor: Any, name: str = "tensor") -> str:
    """Return a one-line shape/dtype description, for logs and docs."""
    shape = tuple(getattr(tensor, "shape", ()))
    dtype = getattr(tensor, "dtype", "?")
    return f"{name:<16} {str(shape):<24} {dtype}"
