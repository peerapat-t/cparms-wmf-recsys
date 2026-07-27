# This file evaluates ranking quality for recommendation scores.

from numbers import Integral

import numpy as np
from scipy import sparse

from util.feedback import LIKE_THRESHOLD


# Inputs:
# - matrix: Dense or sparse matrix to convert.
# - name: Matrix name used in validation error messages.
# Output: Canonical CSR matrix.
def _as_csr(matrix, name: str) -> sparse.csr_matrix:
    # Step 1: Convert the input to CSR storage.
    if sparse.issparse(matrix):
        matrix = sparse.csr_matrix(matrix)
    else:
        matrix = sparse.csr_matrix(np.asarray(matrix))
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix.")
    # Step 2: Canonicalize duplicate, zero, and unsorted sparse entries.
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


# Inputs:
# - train_mat: Positive-history matrix (L) used to count each user's activity.
#   Pass L, not B: activity bins are defined over positive feedback only.
# Output: Dictionary of user-index arrays grouped by 1, 2, or 3+ positive interactions.
# Users with zero positive history cannot be personalized, so they are out of scope
# and are deliberately not returned as a group.
def build_user_activity_groups(train_mat):
    train_mat = _as_csr(train_mat, "train_mat")
    # Step 1: Count positive training interactions and assign users to activity bins.
    train_counts = np.asarray(
        train_mat.getnnz(axis=1)
    ).reshape(-1)
    interaction_1 = np.flatnonzero(
        train_counts == 1
    ).astype(np.int64, copy=False)
    interaction_2 = np.flatnonzero(
        train_counts == 2
    ).astype(np.int64, copy=False)
    interaction_3_plus = np.flatnonzero(
        train_counts >= 3
    ).astype(np.int64, copy=False)

    return {
        "interaction_1": interaction_1,
        "interaction_2": interaction_2,
        "interaction_3_plus": interaction_3_plus,
    }


# Inputs:
# - user_groups: Mapping from group names to user indices.
# - user_count: Number of users in the fit universe; indices must fall inside it.
# Output: Mapping from each group name to a boolean user-membership mask.
def _prepare_user_groups(user_groups, user_count: int):
    if not user_groups:
        raise ValueError("user_groups must contain at least one group")

    # Step 1: Validate each index collection and convert it to a membership mask.
    prepared = {}
    for group_name, user_idx in user_groups.items():
        if group_name == "all":
            raise ValueError(
                "'all' is reserved and cannot be used as a custom group name"
            )
        raw_idx = np.asarray(user_idx).reshape(-1)
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in raw_idx
        ):
            raise ValueError(
                f"user_groups['{group_name}'] must contain integer indices"
            )
        idx = raw_idx.astype(np.int64, copy=False)
        if np.any(idx < 0) or np.any(idx >= user_count):
            raise ValueError(
                f"user_groups['{group_name}'] contains out-of-range indices"
            )
        mask = np.zeros(user_count, dtype=bool)
        mask[np.unique(idx)] = True
        prepared[group_name] = mask
    return prepared


# Inputs:
# - pred_source: 2D array, sparse matrix, or object exposing a shape attribute.
# Output: Integer (user_count, item_count) dimensions.
def _prediction_shape(pred_source) -> tuple[int, int]:
    shape = getattr(pred_source, "shape", None)
    if shape is None:
        shape = np.asarray(pred_source).shape
    if len(shape) != 2:
        raise ValueError("pred_source must expose a 2D shape.")
    return int(shape[0]), int(shape[1])


# Inputs:
# - pred_source: Model with score_user, sparse matrix, or indexable dense predictions.
# - user_idx: Zero-based user index to score.
# - item_count: Required number of item scores.
# Output: Independent dense float score vector with one value per item.
def _score_user(pred_source, user_idx: int, item_count: int) -> np.ndarray:
    # Step 1: Read one score row using the prediction source's supported interface.
    if hasattr(pred_source, "score_user"):
        scores = pred_source.score_user(user_idx)
    elif sparse.issparse(pred_source):
        scores = pred_source.getrow(user_idx).toarray().reshape(-1)
    else:
        scores = np.asarray(pred_source[user_idx]).reshape(-1)

    # Step 2: Convert to a validated one-dimensional float vector.
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if scores.shape != (item_count,):
        raise ValueError("Prediction source returned an invalid score row.")
    return scores


# Inputs:
# - topk: Ranked item indices for one user.
# - targets: Relevant item indices for that user.
# - ks: Requested ranking cutoffs.
# - discounts: Position discount values through the maximum cutoff.
# - idcg_cum: Cumulative ideal DCG lookup indexed by hit count.
# Output: NDCG values aligned with ks for one user.
def _ndcg_at_ks(topk, targets, ks, discounts, idcg_cum):
    # Step 1: Convert ranked-item relevance into cumulative discounted gain.
    hits = np.isin(topk, targets, assume_unique=False)
    hits_float = hits.astype(float)
    dcg_cum = np.cumsum(hits_float * discounts[: topk.size])

    # Step 2: Normalize DCG by the ideal gain at every requested cutoff.
    ndcg = np.zeros(len(ks), dtype=float)
    for idx, k in enumerate(ks):
        kk = min(k, topk.size)
        if kk <= 0:
            continue
        ideal_hits = min(int(targets.size), kk)
        idcg = float(idcg_cum[ideal_hits])
        ndcg[idx] = (
            float(dcg_cum[kk - 1] / idcg)
            if idcg > 0.0
            else 0.0
        )
    return ndcg


# Inputs:
# - ks: Requested ranking cutoffs.
# - sums_by_group: Accumulated NDCG arrays keyed by user-group name.
# - counts_by_group: Number of evaluated users in each group.
# Output: Nested dictionary containing mean NDCG at every cutoff and group.
def _finalize_ndcg_groups(ks, sums_by_group, counts_by_group):
    # Step 1: Divide each group's accumulated scores by its evaluated-user count.
    results = {}
    for group_name, sums in sums_by_group.items():
        count = counts_by_group[group_name]
        means = (
            np.zeros(len(ks), dtype=float)
            if count == 0
            else sums / count
        )
        results[group_name] = {
            "ndcg": dict(zip(ks, means.astype(float).tolist()))
        }
    return results


# Inputs:
# - pred_source: Model or 2D score matrix used to rank items per user.
# - train_mat: Seen matrix (B) whose interactions are removed from candidate rankings.
#   Pass the full fit-window ratings, not L: an item the user already rated below the
#   like threshold is still seen and must not be recommended back to them.
# - test_mat: User-item evaluation interactions containing relevance targets.
# - user_groups: Named user-index groups defining the in-scope population; users
#   outside every group are skipped entirely.
# - ks: Positive ranking cutoffs at which metrics are reported.
# - threshold: Minimum exclusive test value treated as relevant.
# Output: Dictionary with mean NDCG over all evaluated users ("all"), mean NDCG per
# user group ("user"), the evaluated-user count, and the positive-target count.
def ranking_metrics_at_k(
    pred_source,
    train_mat,
    test_mat,
    user_groups,
    ks=(10, 20, 50, 100, 200),
    threshold: float = LIKE_THRESHOLD,
):
    # Step 1: Validate prediction, training, test, and cutoff inputs.
    train_mat = _as_csr(train_mat, "train_mat")
    test_mat = _as_csr(test_mat, "test_mat")
    user_count, item_count = _prediction_shape(pred_source)
    if train_mat.shape != (user_count, item_count):
        raise ValueError("train_mat shape does not match predictions.")
    if test_mat.shape != (user_count, item_count):
        raise ValueError("test_mat shape does not match predictions.")

    ks = tuple(ks)
    if not ks or any(
        isinstance(k, bool) or not isinstance(k, Integral) or k <= 0
        for k in ks
    ):
        raise ValueError("ks must contain positive integers")
    ks = tuple(int(k) for k in ks)

    # Step 2: Prepare candidate limits and user-group lookup structures.
    k_max = min(max(ks), item_count)
    user_group_masks = _prepare_user_groups(
        user_groups,
        user_count=user_count,
    )
    eligible_mask = np.zeros(user_count, dtype=bool)
    for group_mask in user_group_masks.values():
        eligible_mask |= group_mask

    # Step 3: Precompute logarithmic discounts and ideal cumulative gains.
    discounts = 1.0 / np.log2(
        np.arange(2, k_max + 2, dtype=float)
    )
    idcg_cum = np.zeros(k_max + 1, dtype=float)
    idcg_cum[1:] = np.cumsum(discounts)

    # Step 4: Initialize metric accumulators and evaluation audit counters.
    user_sums = {"all": np.zeros(len(ks), dtype=float)}
    user_counts = {"all": 0}
    for group_name in user_group_masks:
        user_sums[group_name] = np.zeros(len(ks), dtype=float)
        user_counts[group_name] = 0

    positive_target_count = 0

    # Step 5: Evaluate each eligible user with at least one relevant target.
    for user_idx in range(user_count):
        # Step 5.1: Skip users outside the eligible (in-scope) population.
        if not eligible_mask[user_idx]:
            continue

        # Step 5.2: Extract positive test targets.
        start = test_mat.indptr[user_idx]
        stop = test_mat.indptr[user_idx + 1]
        target_items = test_mat.indices[start:stop]
        target_values = test_mat.data[start:stop]
        targets = target_items[target_values > float(threshold)]
        if targets.size == 0:
            continue

        positive_target_count += int(targets.size)

        # Step 5.3: Score candidates and mask previously observed items.
        scores = _score_user(pred_source, user_idx, item_count)
        train_start = train_mat.indptr[user_idx]
        train_stop = train_mat.indptr[user_idx + 1]
        candidate_mask = np.ones(item_count, dtype=bool)
        candidate_mask[train_mat.indices[train_start:train_stop]] = False
        candidate_items = np.flatnonzero(candidate_mask)
        if not np.isfinite(scores[candidate_items]).all():
            raise ValueError(
                "Prediction source returned non-finite candidate scores."
            )

        # Step 5.4: Rank candidates by score with deterministic item-index ties.
        if candidate_items.size:
            k_user = min(k_max, int(candidate_items.size))
            order = np.lexsort((candidate_items, -scores[candidate_items]))
            topk = candidate_items[order[:k_user]]
            user_ndcg = _ndcg_at_ks(
                topk,
                targets,
                ks,
                discounts,
                idcg_cum,
            )
        else:
            user_ndcg = np.zeros(len(ks), dtype=float)

        # Step 5.5: Accumulate NDCG for all applicable user groups.
        user_sums["all"] += user_ndcg
        user_counts["all"] += 1
        for group_name, group_mask in user_group_masks.items():
            if group_mask[user_idx]:
                user_sums[group_name] += user_ndcg
                user_counts[group_name] += 1

    # Step 6: Convert accumulated metrics and counters into the result structure.
    user_results = _finalize_ndcg_groups(ks, user_sums, user_counts)
    return {
        "all": user_results["all"],
        "user": {
            group_name: user_results[group_name]
            for group_name in user_group_masks
        },
        "n_users_eval": int(user_counts["all"]),
        "n_positive_targets": positive_target_count,
    }
