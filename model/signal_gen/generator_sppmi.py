# This file builds item-item SPPMI co-occurrence matrices for CoFactor.

import numpy as np
from scipy import sparse

from util.feedback import LIKE_THRESHOLD, to_L


MODEL_DTYPE = np.float32


# Inputs:
# - Y: Dense or sparse user-item interaction matrix.
# - threshold: Minimum exclusive value counted as an item-history event.
# - negative_samples: Shift value k for SPPMI = max(PMI - log(k), 0).
# Output: Sparse item-item SPPMI matrix used by CoFactor.
def build_item_sppmi_matrix(
    Y,
    threshold: float = LIKE_THRESHOLD,
    negative_samples: float = 1.0,
) -> sparse.csr_matrix:
    negative_samples = float(negative_samples)
    if not np.isfinite(negative_samples) or negative_samples < 1.0:
        raise ValueError("negative_samples must be finite and >= 1.")

    # Step 1: Count how many users positively interacted with each item pair.
    # L.T @ L is symmetric; the diagonal is each item's own count, not a
    # co-occurrence, so it is dropped.
    L = to_L(Y, threshold=threshold)
    item_count = L.shape[1]
    cooccurrence = (L.T @ L).tocsr().astype(
        MODEL_DTYPE,
        copy=False,
    )
    cooccurrence.setdiag(0)
    cooccurrence.eliminate_zeros()
    cooccurrence.sort_indices()

    if cooccurrence.nnz == 0:
        return sparse.csr_matrix(
            (item_count, item_count),
            dtype=MODEL_DTYPE,
        )

    # Step 2: Convert nonzero co-occurrences into PMI values.
    row_count = np.asarray(
        cooccurrence.sum(axis=1)
    ).reshape(-1).astype(np.float64, copy=False)
    pair_count = float(cooccurrence.data.sum())
    if pair_count <= 0.0:
        return sparse.csr_matrix(
            (item_count, item_count),
            dtype=MODEL_DTYPE,
        )

    sppmi = cooccurrence.copy().astype(np.float64, copy=False)
    for item_idx in range(item_count):
        start = sppmi.indptr[item_idx]
        stop = sppmi.indptr[item_idx + 1]
        if start == stop:
            continue
        context_idx = sppmi.indices[start:stop]
        counts = sppmi.data[start:stop]
        denominator = row_count[item_idx] * row_count[context_idx]
        valid = denominator > 0.0
        values = np.zeros(counts.shape, dtype=np.float64)
        values[valid] = np.log(
            counts[valid] * pair_count / denominator[valid]
        )
        sppmi.data[start:stop] = values

    # Step 3: Shift by log(k) and clip at zero, giving SPPMI = max(PMI - log(k), 0).
    # k == 1 makes the shift log(1) = 0, so SPPMI degenerates to plain PPMI.
    if negative_samples > 1.0:
        sppmi.data -= np.log(negative_samples)
    sppmi.data[sppmi.data < 0.0] = 0.0
    sppmi = sppmi.astype(MODEL_DTYPE, copy=False)
    sppmi.eliminate_zeros()
    sppmi.sort_indices()
    return sppmi
