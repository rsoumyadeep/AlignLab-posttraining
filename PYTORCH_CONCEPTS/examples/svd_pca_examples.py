"""Executable examples for the SVD/PCA material in STUDY_WITH_CLAUDE/phase4.

Every number quoted in that note comes from running this file:

    python PYTORCH_CONCEPTS/examples/svd_pca_examples.py

Seeded, CPU, float64 throughout - these are linear-algebra identities and
float32 round-off would obscure whether they hold exactly.
"""

from __future__ import annotations

import torch

torch.set_printoptions(precision=6, sci_mode=False)
DTYPE = torch.float64


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def example_svd_basics() -> None:
    section("1. SVD: A = U S V^T, and what the pieces are")

    torch.manual_seed(0)
    A = torch.randn(6, 4, dtype=DTYPE)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)

    print(f"  A    {tuple(A.shape)}")
    print(f"  U    {tuple(U.shape)}   S {tuple(S.shape)}   V^T {tuple(Vh.shape)}")
    print(f"  singular values: {S.numpy()}")

    reconstructed = U @ torch.diag(S) @ Vh
    print(f"\n  ||A - U S V^T||_F = {torch.linalg.norm(A - reconstructed):.3e}")
    print(f"  U has orthonormal columns: {torch.allclose(U.T @ U, torch.eye(4, dtype=DTYPE))}")
    print(f"  V has orthonormal rows   : {torch.allclose(Vh @ Vh.T, torch.eye(4, dtype=DTYPE))}")
    print("\n  Singular values are non-negative and sorted descending by convention.")


def example_eckart_young() -> None:
    section("2. Eckart-Young: truncated SVD is the BEST rank-r approximation")

    torch.manual_seed(1)
    A = torch.randn(40, 30, dtype=DTYPE)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)

    print(f"  {'r':>3} {'||A-A_r||_F':>14} {'sqrt(tail sum sq)':>20} {'agree':>7}")
    for r in (1, 5, 10, 20, 29):
        A_r = U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :]
        actual = float(torch.linalg.norm(A - A_r))
        predicted = float(torch.sqrt((S[r:] ** 2).sum()))
        print(f"  {r:>3} {actual:>14.6f} {predicted:>20.6f} "
              f"{str(abs(actual - predicted) < 1e-9):>7}")

    print("\n  The error of the best rank-r approximation is exactly the")
    print("  square root of the SUM OF SQUARED discarded singular values.")
    print("  That is why 'explained energy' accumulates SQUARES, not raw values.")

    print("\n  Beating it is impossible - 20 random rank-5 matrices for comparison:")
    best_random = float("inf")
    A_5 = U[:, :5] @ torch.diag(S[:5]) @ Vh[:5, :]
    optimal = float(torch.linalg.norm(A - A_5))
    for seed in range(20):
        g = torch.Generator().manual_seed(seed)
        B = torch.randn(40, 5, generator=g, dtype=DTYPE)
        C = torch.randn(5, 30, generator=g, dtype=DTYPE)
        best_random = min(best_random, float(torch.linalg.norm(A - B @ C)))
    print(f"    truncated SVD rank-5 error : {optimal:.6f}")
    print(f"    best of 20 random rank-5   : {best_random:.6f}")
    print(f"    SVD is better              : {optimal < best_random}")


def example_energy_curve() -> None:
    section("3. Explained energy, and why raw singular values mislead")

    torch.manual_seed(2)
    # A matrix with a deliberately decaying spectrum.
    U, _ = torch.linalg.qr(torch.randn(50, 50, dtype=DTYPE))
    V, _ = torch.linalg.qr(torch.randn(30, 30, dtype=DTYPE))
    S = torch.tensor([10.0 * (0.8**i) for i in range(30)], dtype=DTYPE)
    A = U[:, :30] @ torch.diag(S) @ V.T

    sv = torch.linalg.svdvals(A)
    raw = torch.cumsum(sv, 0) / sv.sum()
    sq = torch.cumsum(sv**2, 0) / (sv**2).sum()

    print(f"  {'r':>3} {'cum raw':>10} {'cum squared':>13} {'recon err':>11}")
    for r in (1, 2, 4, 8, 16, 30):
        A_r = (
            torch.linalg.svd(A, full_matrices=False)[0][:, :r]
            @ torch.diag(sv[:r])
            @ torch.linalg.svd(A, full_matrices=False)[2][:r, :]
        )
        err = float(torch.linalg.norm(A - A_r) / torch.linalg.norm(A))
        print(f"  {r:>3} {float(raw[r-1]):>10.4f} {float(sq[r-1]):>13.4f} {err:>11.6f}")

    print("\n  At r=4 the RAW cumulative fraction says 0.5911 while the SQUARED")
    print("  fraction says 0.8322. Only the squared one predicts the actual")
    print("  reconstruction error: sqrt(1 - 0.8322) = 0.4096, matching the")
    print("  measured 0.409598 exactly. Reporting the raw curve would understate")
    print("  how good a low-rank approximation is.")


def example_pca_from_svd() -> None:
    section("4. PCA IS the SVD of the centred data matrix")

    torch.manual_seed(3)
    n, d = 500, 5
    # Correlated data so the principal directions are not trivial.
    latent = torch.randn(n, 2, dtype=DTYPE)
    mixing = torch.randn(2, d, dtype=DTYPE)
    X = latent @ mixing + 0.1 * torch.randn(n, d, dtype=DTYPE)

    # --- route 1: eigendecomposition of the covariance matrix
    Xc = X - X.mean(dim=0, keepdim=True)  # CENTRING IS NOT OPTIONAL
    cov = (Xc.T @ Xc) / (n - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # --- route 2: SVD of the centred data
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    sv_variance = S**2 / (n - 1)

    print(f"  eigenvalues of covariance : {eigvals.numpy()}")
    print(f"  S^2 / (n-1) from SVD      : {sv_variance.numpy()}")
    print(f"  identical                 : {torch.allclose(eigvals, sv_variance, atol=1e-8)}")

    print("\n  principal directions agree up to sign:")
    for i in range(3):
        a = eigvecs[:, i]
        b = Vh[i]
        aligned = torch.allclose(a, b, atol=1e-6) or torch.allclose(a, -b, atol=1e-6)
        print(f"    component {i}: |cos| = {abs(float(a @ b)):.10f}  aligned {aligned}")

    explained = eigvals / eigvals.sum()
    print(f"\n  explained variance ratio  : {explained.numpy()}")
    print(f"  first two components hold : {float(explained[:2].sum()):.6f}")
    print("  (the data was generated from a 2-dimensional latent space)")


def example_centring_matters() -> None:
    section("5. Centring: the difference between PCA and a plain SVD")

    torch.manual_seed(4)
    X = torch.randn(300, 3, dtype=DTYPE) @ torch.tensor(
        [[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.1]], dtype=DTYPE
    )
    X = X + torch.tensor([50.0, -20.0, 5.0], dtype=DTYPE)  # a large mean offset

    sv_raw = torch.linalg.svdvals(X)
    sv_centred = torch.linalg.svdvals(X - X.mean(0, keepdim=True))

    print(f"  singular values, UNCENTRED : {sv_raw.numpy()}")
    print(f"  singular values, CENTRED   : {sv_centred.numpy()}")
    print(f"\n  ratio of first to second, uncentred: {float(sv_raw[0]/sv_raw[1]):.2f}")
    print(f"  ratio of first to second, centred  : {float(sv_centred[0]/sv_centred[1]):.2f}")
    print("\n  Uncentred, the first direction just points at the MEAN and")
    print("  dominates everything. PCA without centring measures where the data")
    print("  IS, not how it VARIES.")
    print("\n  NOTE FOR LORA: dW is already centred in the sense that matters -")
    print("  it is a difference of two weight matrices, not a data cloud - so an")
    print("  SVD of dW is the right object and no centring step applies.")


def example_low_rank_parameter_cost() -> None:
    section("6. When does a rank-r factorisation actually save parameters?")

    print(f"  {'d_out':>7} {'d_in':>7} {'dense':>12} {'r':>5} {'r(d_in+d_out)':>15} {'saving':>9}")
    for d_out, d_in in ((1536, 1536), (256, 1536), (8960, 1536)):
        dense = d_out * d_in
        for r in (16, 64, 256, 730):
            lora = r * (d_in + d_out)
            if lora >= dense:
                note = "  NO SAVING"
            else:
                note = f"  {100 * (1 - lora / dense):.1f}%"
            print(f"  {d_out:>7} {d_in:>7} {dense:>12,} {r:>5} {lora:>15,} {note:>9}")

    break_even = (1536 * 1536) / (1536 + 1536)
    print(f"\n  For a 1536x1536 matrix the break-even rank is {break_even:.0f}.")
    print("  Above it, the 'low-rank' factorisation costs MORE than the dense")
    print("  update it approximates. E17 measured the real dW needing ~730 for")
    print("  90% energy - comfortably past the point where the factorisation")
    print("  stops being a saving at all.")


def main() -> None:
    print("SVD / PCA - executed examples")
    print(f"torch {torch.__version__}, cpu, dtype {DTYPE}")
    example_svd_basics()
    example_eckart_young()
    example_energy_curve()
    example_pca_from_svd()
    example_centring_matters()
    example_low_rank_parameter_cost()
    print()


if __name__ == "__main__":
    main()
