"""Evaluate recommendation scores with grouped NDCG metrics."""

from numbers import Integral

import numpy as np
from scipy import sparse

from util.feedback import LIKE_THRESHOLD


def _as_csr(matrix, name: str) -> sparse.csr_matrix:
    """Normalize a two-dimensional matrix to canonical CSR format."""

    if sparse.issparse(matrix):
        matrix = sparse.csr_matrix(matrix)
    else:
        matrix = sparse.csr_matrix(np.asarray(matrix))
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix.")

    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def build_user_activity_groups(train_mat):
    """Partition users by their number of training interactions."""
    train_mat = _as_csr(train_mat, "train_mat")


    train_counts = np.asarray(
        train_mat.getnnz(axis=1)
    ).reshape(-1)
    interaction_0 = np.flatnonzero(
        train_counts == 0
    ).astype(np.int64, copy=False)
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
        "interaction_0": interaction_0,
        "interaction_1": interaction_1,
        "interaction_2": interaction_2,
        "interaction_3_plus": interaction_3_plus,
    }


def _prepare_user_groups(user_groups, user_count: int):
    """Validate group indices and convert each group to a user mask."""
    if not user_groups:
        raise ValueError("user_groups must contain at least one group")


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


def _prediction_shape(pred_source) -> tuple[int, int]:
    """Read and validate the shape exposed by a prediction source."""
    shape = getattr(pred_source, "shape", None)
    if shape is None:
        shape = np.asarray(pred_source).shape
    if len(shape) != 2:
        raise ValueError("pred_source must expose a 2D shape.")
    return int(shape[0]), int(shape[1])


def _score_user(pred_source, user_idx: int, item_count: int) -> np.ndarray:
    """Read one dense item-score row from any supported prediction source."""

    if hasattr(pred_source, "score_user"):
        scores = pred_source.score_user(user_idx)
    elif sparse.issparse(pred_source):
        scores = pred_source.getrow(user_idx).toarray().reshape(-1)
    else:
        scores = np.asarray(pred_source[user_idx]).reshape(-1)


    scores = np.asarray(scores, dtype=float).reshape(-1)
    if scores.shape != (item_count,):
        raise ValueError("Prediction source returned an invalid score row.")
    return scores


def _ndcg_at_ks(topk, targets, ks, discounts, idcg_cum):
    """Compute one user's NDCG at every requested cutoff."""

    hits = np.isin(topk, targets, assume_unique=False)
    hits_float = hits.astype(float)
    dcg_cum = np.cumsum(hits_float * discounts[: topk.size])


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


def _finalize_ndcg_groups(ks, sums_by_group, counts_by_group):
    """Convert accumulated group totals into mean NDCG dictionaries."""

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


def ranking_metrics_at_k(
    pred_source,
    train_mat,
    test_mat,
    user_groups,
    ks=(10, 20, 50, 100, 200),
    threshold: float = LIKE_THRESHOLD,
    return_per_user: bool = False,
):
    """Evaluate NDCG for eligible users overall and by activity group."""

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


    k_max = min(max(ks), item_count)
    user_group_masks = _prepare_user_groups(
        user_groups,
        user_count=user_count,
    )
    eligible_mask = np.zeros(user_count, dtype=bool)
    for group_mask in user_group_masks.values():
        eligible_mask |= group_mask


    # Precompute rank discounts and ideal DCG values shared by all users.
    discounts = 1.0 / np.log2(
        np.arange(2, k_max + 2, dtype=float)
    )
    idcg_cum = np.zeros(k_max + 1, dtype=float)
    idcg_cum[1:] = np.cumsum(discounts)


    user_sums = {"all": np.zeros(len(ks), dtype=float)}
    user_counts = {"all": 0}
    for group_name in user_group_masks:
        user_sums[group_name] = np.zeros(len(ks), dtype=float)
        user_counts[group_name] = 0

    positive_target_count = 0


    per_user_idx = [] if return_per_user else None
    per_user_ndcg = [] if return_per_user else None


    # Score only eligible users who have at least one positive test target.
    for user_idx in range(user_count):

        if not eligible_mask[user_idx]:
            continue


        start = test_mat.indptr[user_idx]
        stop = test_mat.indptr[user_idx + 1]
        target_items = test_mat.indices[start:stop]
        target_values = test_mat.data[start:stop]
        targets = target_items[target_values > float(threshold)]
        if targets.size == 0:
            continue

        positive_target_count += int(targets.size)


        # Exclude training interactions from the candidate item set.
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


        user_sums["all"] += user_ndcg
        user_counts["all"] += 1
        for group_name, group_mask in user_group_masks.items():
            if group_mask[user_idx]:
                user_sums[group_name] += user_ndcg
                user_counts[group_name] += 1


        if return_per_user:
            per_user_idx.append(user_idx)
            per_user_ndcg.append(user_ndcg)


    user_results = _finalize_ndcg_groups(ks, user_sums, user_counts)
    results = {
        "all": user_results["all"],
        "user": {
            group_name: user_results[group_name]
            for group_name in user_group_masks
        },
        "n_users_eval": int(user_counts["all"]),
        "n_positive_targets": positive_target_count,
    }
    if return_per_user:
        results["per_user"] = {
            "ks": ks,
            "user_idx": np.array(per_user_idx, dtype=np.int64),
            "ndcg": (
                np.array(per_user_ndcg, dtype=float)
                if per_user_ndcg
                else np.zeros((0, len(ks)), dtype=float)
            ),
        }
    return results
