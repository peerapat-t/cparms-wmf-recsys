# This file centralizes conversion of raw feedback Y into the canonical
# Y / B / L matrices used across the pipeline:
#   Y = raw ratings (values preserved)
#   B = seen / any interaction   (Y != 0) -> 1
#   L = liked / positive         (Y >  LIKE_THRESHOLD) -> 1

import numpy as np
from scipy import sparse


# Rating strictly greater than this threshold counts as a positive "like" (L).
LIKE_THRESHOLD = 3.0


# Inputs:
# - Y: Dense or sparse non-negative user-item interaction matrix.
# - dtype: Output value dtype.
# Output: Canonical non-negative CSR matrix with raw interaction values preserved.
def to_Y(Y, dtype=np.float32) -> sparse.csr_matrix:
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


# Inputs:
# - Y: Dense or sparse non-negative user-item interaction matrix.
# - dtype: Output value dtype.
# Output: Canonical CSR "seen" matrix B where every interaction (any rating) equals one.
def to_B(Y, dtype=np.float32) -> sparse.csr_matrix:
    B = to_Y(Y)
    if B.nnz:
        B.data = np.ones(B.nnz, dtype=dtype)
    return B


# Inputs:
# - Y: Dense or sparse non-negative user-item interaction matrix.
# - threshold: Minimum exclusive rating counted as a positive "like".
# - dtype: Output value dtype.
# Output: Canonical CSR "liked" matrix L where positive entries (Y > threshold) equal one.
def to_L(Y, threshold: float = LIKE_THRESHOLD, dtype=np.float32) -> sparse.csr_matrix:
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")
    L = to_Y(Y)
    if L.nnz:
        L.data = (L.data > float(threshold)).astype(dtype)
        L.eliminate_zeros()
    L.sort_indices()
    return L
