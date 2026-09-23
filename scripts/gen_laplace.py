#!/usr/bin/env python3
"""Generate the multigrid SpGEMM workload of paper 1 (Deveci et al. 2018): Laplace3D.

    A = 7-point 3D Laplacian on an n x n x n grid          (n^3 x n^3, ~7 nnz/row)
    P = piecewise-constant aggregation of 2x2x2 cells      (n^3 x (n/2)^3, one 1.0 per row)
    R = P^T                                                ((n/2)^3 x n^3)

The products R*A (irregular, low locality) and A*P (regular) are the two extremes paper 1 studies.
Files: <out>/laplace3d_n<n>/{A,P,R}.mtx. The printed CSR sizes (FP64, 32-bit indices) help pick n
so that A is 0.5x-4x the fast-memory budget in Phase 4.

Examples:
    python scripts/gen_laplace.py --n 64
    python scripts/gen_laplace.py --n 128 256 --sizes-only
"""
import argparse
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp


def laplace3d(n):
    """7-point stencil: Kronecker sum of three 1D [-1, 2, -1] operators (diagonal = 6)."""
    t = sp.diags([-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1], format="csr")
    eye = sp.identity(n, format="csr")
    return (sp.kron(sp.kron(t, eye), eye) + sp.kron(sp.kron(eye, t), eye) + sp.kron(sp.kron(eye, eye), t)).tocsr()


def aggregation(n):
    """Cell (x, y, z) -> aggregate (x//2, y//2, z//2); index = x*n*n + y*n + z to match kron order."""
    m = n // 2
    x, y, z = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    rows = (x * n * n + y * n + z).ravel()
    cols = ((x // 2) * m * m + (y // 2) * m + (z // 2)).ravel()
    return sp.csr_matrix((np.ones(n ** 3), (rows, cols)), shape=(n ** 3, m ** 3))


def csr_mb(M):
    return (M.nnz * (8 + 4) + (M.shape[0] + 1) * 4) / 1e6


def estimate(n):
    """Sizes without building the matrices (for --sizes-only)."""
    rows, agg = n ** 3, (n // 2) ** 3
    nnz_a = 7 * rows - 6 * n * n  # 7-point stencil minus the missing neighbours on the 6 faces
    return {"A": (rows, rows, nnz_a), "P": (rows, agg, rows), "R": (agg, rows, rows)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, nargs="+", required=True, help="grid size per dimension (even)")
    ap.add_argument("--out", type=Path, default=Path("data/laplace"))
    ap.add_argument("--sizes-only", action="store_true", help="print sizes, write nothing")
    args = ap.parse_args()

    for n in args.n:
        if n < 2 or n % 2:
            ap.error(f"--n must be even and >= 2 (got {n})")
        if args.sizes_only:
            for name, (r, c, nnz) in estimate(n).items():
                print(f"n={n:4d} {name}: {r:>11d} x {c:<11d} nnz {nnz:>12d}  CSR fp64 {(nnz * 12 + (r + 1) * 4) / 1e6:9.1f} MB")
            continue

        A, P = laplace3d(n), aggregation(n)
        R = P.T.tocsr()
        d = args.out / f"laplace3d_n{n}"
        d.mkdir(parents=True, exist_ok=True)
        for name, M in (("A", A), ("P", P), ("R", R)):
            M.sort_indices()
            scipy.io.mmwrite(d / f"{name}.mtx", M, comment=f"Laplace3D n={n} ({name}), RASP gen_laplace.py")
            print(f"n={n:4d} {name}: {M.shape[0]:>11d} x {M.shape[1]:<11d} nnz {M.nnz:>12d}  CSR fp64 {csr_mb(M):9.1f} MB"
                  f"  -> {d / (name + '.mtx')}")


if __name__ == "__main__":
    main()
