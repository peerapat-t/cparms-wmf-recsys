"""Validate ratings and derive binary feedback matrices."""

import numpy as np
from scipy import sparse


LIKE_THRESHOLD = 4.0
DISLIKE_THRESHOLD = 4.0


def to_Y(Y, dtype=np.float64) -> sparse.csr_matrix:
    """Return a validated nonnegative rating matrix in CSR format."""
    if sparse.issparse(Y):
        matrix = Y.tocsr().astype(dtype, copy=True)
    else:
        matrix = sparse.csr_matrix(np.asarray(Y, dtype=dtype))
    if matrix.ndim != 2:
        raise ValueError("Y must be a 2D matrix.")
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if not np.isfinite(matrix.data).all():
        raise ValueError("Y has NaN/Inf.")
    if np.any(matrix.data < 0):
        raise ValueError("Y has negative values.")
    matrix.sort_indices()
    return matrix


def to_B(Y, dtype=np.float64) -> sparse.csr_matrix:
    """Convert observed ratings into a binary interaction matrix."""
    B = to_Y(Y)
    if B.nnz:
        B.data = np.ones(B.nnz, dtype=dtype)
    return B


def to_L(Y, threshold: float = LIKE_THRESHOLD, dtype=np.float64) -> sparse.csr_matrix:
    """Select ratings strictly above the like threshold."""
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")
    L = to_Y(Y)
    if L.nnz:
        L.data = (L.data > float(threshold)).astype(dtype)
        L.eliminate_zeros()
    L.sort_indices()
    return L


def to_D(Y, threshold: float = DISLIKE_THRESHOLD, dtype=np.float64) -> sparse.csr_matrix:
    """Select positive ratings strictly below the dislike threshold."""
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")
    D = to_Y(Y)
    if D.nnz:
        D.data = ((D.data > 0.0) & (D.data < float(threshold))).astype(dtype)
        D.eliminate_zeros()
    D.sort_indices()
    return D
