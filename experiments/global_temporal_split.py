"""Create leakage-safe global temporal splits and sparse matrices."""

import numpy as np
import pandas as pd
from scipy import sparse

from util.feedback import DISLIKE_THRESHOLD, LIKE_THRESHOLD


SparseMatrix = sparse.csr_matrix


def _parse_timestamps(timestamps: pd.Series) -> pd.Series:
    """Validate datetime values or parse unambiguous ISO-8601 strings."""
    if pd.api.types.is_datetime64_any_dtype(timestamps):
        return timestamps
    if pd.api.types.is_numeric_dtype(timestamps):
        raise ValueError(
            "timestamp must be datetime-typed or ISO-8601 strings; a numeric "
            "column is ambiguous. Convert it with pd.to_datetime(col, unit=...) "
            "before splitting."
        )

    try:
        parsed = pd.to_datetime(timestamps, format="ISO8601")
    except (ValueError, TypeError) as error:
        raise ValueError(
            "timestamp must be datetime-typed or ISO-8601 strings "
            f"(e.g. '2016-11-07'): {error}"
        ) from error
    if parsed.isna().any():
        raise ValueError("timestamp contains values that are not valid datetimes")
    return parsed


def _dataset_eda_stats(df: pd.DataFrame) -> dict:
    """Summarize users, items, interactions, density, and sparsity."""
    n_users = int(df["userid"].nunique())
    n_items = int(df["itemid"].nunique())
    n_interactions = int(len(df))
    density_pct = 100.0 * n_interactions / (n_users * n_items)
    ratings = df["rating"].to_numpy(dtype=float, copy=False)
    n_liked = int((ratings > LIKE_THRESHOLD).sum())
    n_disliked = int(((ratings > 0.0) & (ratings < DISLIKE_THRESHOLD)).sum())

    return {
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": n_interactions,
        "n_liked": n_liked,
        "n_disliked": n_disliked,
        "n_users_eval": None,
        "n_future_user_interactions_padded": None,
        "n_future_user_positive_targets_padded": None,
        "n_future_item_interactions_dropped": None,
        "n_future_item_positive_targets_dropped": None,
        "interaction_0": None,
        "interaction_1": None,
        "interaction_2": None,
        "interaction_3_plus": None,
        "density_pct": density_pct,
        "sparsity_pct": 100.0 - density_pct,
    }


def _matrix_eda_stats(matrix: SparseMatrix) -> dict:
    """Summarize the active dimensions, density, and L/D split of a matrix."""
    n_users = int(np.count_nonzero(matrix.getnnz(axis=1)))
    n_items = int(np.count_nonzero(matrix.getnnz(axis=0)))
    n_interactions = int(matrix.nnz)
    density_pct = (
        100.0 * n_interactions / (n_users * n_items)
        if n_users and n_items
        else 0.0
    )
    n_liked = int((matrix.data > LIKE_THRESHOLD).sum())
    n_disliked = int(
        ((matrix.data > 0.0) & (matrix.data < DISLIKE_THRESHOLD)).sum()
    )

    return {
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": n_interactions,
        "n_liked": n_liked,
        "n_disliked": n_disliked,
        "density_pct": density_pct,
        "sparsity_pct": 100.0 - density_pct,
    }


def _interaction_group_counts(
    fit_mat: SparseMatrix,
    eval_mat: SparseMatrix,
) -> dict:
    """Count evaluable users by the size of their fitting history."""

    history_counts = np.asarray(fit_mat.getnnz(axis=1)).reshape(-1)


    eval_positive = eval_mat.copy()
    eval_positive.data = (eval_positive.data > LIKE_THRESHOLD).astype(np.float64)
    eval_positive.eliminate_zeros()
    has_relevant_target = np.asarray(eval_positive.getnnz(axis=1)).reshape(-1) > 0
    eligible = has_relevant_target


    group_counts = {
        "interaction_0": int((eligible & (history_counts == 0)).sum()),
        "interaction_1": int((eligible & (history_counts == 1)).sum()),
        "interaction_2": int((eligible & (history_counts == 2)).sum()),
        "interaction_3_plus": int((eligible & (history_counts >= 3)).sum()),
    }
    return group_counts


def _evaluation_eda_stats(
    fit_mat: SparseMatrix,
    eval_mat: SparseMatrix,
) -> dict:
    """Combine evaluation-matrix and user-activity statistics."""
    matrix_stats = _matrix_eda_stats(eval_mat)
    interaction_groups = _interaction_group_counts(fit_mat, eval_mat)

    return {
        "n_users": matrix_stats["n_users"],
        "n_items": matrix_stats["n_items"],
        "n_interactions": matrix_stats["n_interactions"],
        "n_liked": matrix_stats["n_liked"],
        "n_disliked": matrix_stats["n_disliked"],
        "n_users_eval": sum(interaction_groups.values()),
        **interaction_groups,
        "density_pct": matrix_stats["density_pct"],
        "sparsity_pct": matrix_stats["sparsity_pct"],
    }


def _eval_protocol_stats(
    fit_events: pd.DataFrame,
    eval_events: pd.DataFrame,
) -> dict:
    """Count future-user padding and future-item removal events."""
    known_user = eval_events["userid"].isin(
        pd.Index(fit_events["userid"].unique())
    )
    known_item = eval_events["itemid"].isin(
        pd.Index(fit_events["itemid"].unique())
    )
    padded_user = eval_events.loc[~known_user & known_item]
    dropped_item = eval_events.loc[~known_item]

    return {
        "n_future_user_interactions_padded": int(len(padded_user)),
        "n_future_user_positive_targets_padded": int(
            (padded_user["rating"] > LIKE_THRESHOLD).sum()
        ),
        "n_future_item_interactions_dropped": int(len(dropped_item)),
        "n_future_item_positive_targets_dropped": int(
            (dropped_item["rating"] > LIKE_THRESHOLD).sum()
        ),
    }


def _build_sparse_matrix(
    data: pd.DataFrame,
    shape: tuple[int, int],
) -> SparseMatrix:
    """Build a CSR matrix using the maximum duplicate-pair rating."""
    if data.empty:
        return sparse.csr_matrix(shape, dtype=np.float64)


    grouped = (
        data.groupby(["u_idx", "i_idx"], sort=False, as_index=False)["rating"]
        .max()
    )


    matrix = sparse.csr_matrix(
        (
            grouped["rating"].to_numpy(dtype=np.float64, copy=False),
            (
                grouped["u_idx"].to_numpy(dtype=np.int64, copy=False),
                grouped["i_idx"].to_numpy(dtype=np.int64, copy=False),
            ),
        ),
        shape=shape,
        dtype=np.float64,
    )
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def _unique_timestamp_boundaries(
    timestamps: np.ndarray,
    train_ratio: float,
    val_ratio: float,
) -> tuple[int, int]:
    """Choose split boundaries without dividing equal timestamps."""
    n_rows = int(timestamps.size)
    change_positions = np.flatnonzero(timestamps[1:] != timestamps[:-1]) + 1
    if change_positions.size < 2:
        raise ValueError(
            "At least three distinct timestamp groups are required for "
            "train/validation/test splits."
        )

    train_target = int(np.floor(n_rows * train_ratio))
    val_target = int(np.floor(n_rows * (train_ratio + val_ratio)))

    train_candidates = change_positions[:-1]
    train_end = int(
        train_candidates[np.argmin(np.abs(train_candidates - train_target))]
    )
    val_candidates = change_positions[change_positions > train_end]
    val_end = int(
        val_candidates[np.argmin(np.abs(val_candidates - val_target))]
    )
    return train_end, val_end


def _index_and_build(
    fit_events: pd.DataFrame,
    eval_events: pd.DataFrame,
) -> tuple[SparseMatrix, SparseMatrix]:
    """Index fit/evaluation events and drop unknown evaluation items."""


    user_values = pd.concat(
        [fit_events["userid"], eval_events["userid"]],
        ignore_index=True,
    )
    _, user_uniques = pd.factorize(user_values, sort=True)
    _, item_uniques = pd.factorize(fit_events["itemid"], sort=True)
    shape = (int(user_uniques.size), int(item_uniques.size))

    fit_df = fit_events.copy()
    fit_df["u_idx"] = pd.Index(user_uniques).get_indexer(fit_df["userid"])
    fit_df["i_idx"] = pd.Index(item_uniques).get_indexer(fit_df["itemid"])
    fit_mat = _build_sparse_matrix(fit_df, shape)


    eval_user_idx = pd.Index(user_uniques).get_indexer(eval_events["userid"])
    eval_item_idx = pd.Index(item_uniques).get_indexer(eval_events["itemid"])
    known_item = eval_item_idx >= 0
    keep = known_item

    eval_df = eval_events.loc[keep].copy()
    eval_df["u_idx"] = eval_user_idx[keep]
    eval_df["i_idx"] = eval_item_idx[keep]
    eval_mat = _build_sparse_matrix(eval_df, shape)

    return fit_mat, eval_mat


def get_temporal_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> tuple[SparseMatrix, SparseMatrix, SparseMatrix, SparseMatrix, dict]:
    """Build train/validation and train-plus-validation/test matrices."""

    required_cols = {"userid", "itemid", "timestamp", "rating"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=float)
    if np.any(ratios <= 0):
        raise ValueError("train_ratio, val_ratio, test_ratio must all be > 0")
    if not np.isclose(ratios.sum(), 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    df = df.copy()
    if df.empty:
        raise ValueError("df must contain at least one row")
    if len(df) < 3:
        raise ValueError("df must contain at least 3 rows for train/val/test splits")
    if df[["userid", "itemid", "timestamp", "rating"]].isna().any().any():
        raise ValueError("userid, itemid, timestamp, and rating must not contain nulls")
    df["timestamp"] = _parse_timestamps(df["timestamp"])
    if not pd.api.types.is_numeric_dtype(df["rating"]):
        raise ValueError("rating must be numeric")
    ratings = df["rating"].to_numpy(dtype=float, copy=False)
    if not np.isfinite(ratings).all() or np.any(ratings < 0.0):
        raise ValueError("rating must contain finite non-negative values")


    # Preserve input order when two interactions share a timestamp.
    df["_rowid"] = np.arange(df.shape[0], dtype=np.int64)
    df = df.sort_values(
        ["timestamp", "_rowid"],
        ascending=[True, True],
        kind="mergesort",
    ).reset_index(drop=True)


    timestamps = df["timestamp"].to_numpy()
    train_end, val_end = _unique_timestamp_boundaries(
        timestamps, train_ratio, val_ratio
    )

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]


    # Index each evaluation pair against only the history available to it.
    train_mat, val_mat = _index_and_build(train_df, val_df)


    trainval_df = pd.concat([train_df, val_df], ignore_index=True)
    train_val_mat, test_mat = _index_and_build(trainval_df, test_df)

    val_eda = _evaluation_eda_stats(train_mat, val_mat)
    val_eda.update(_eval_protocol_stats(train_df, val_df))
    test_eda = _evaluation_eda_stats(train_val_mat, test_mat)
    test_eda.update(_eval_protocol_stats(trainval_df, test_df))
    eda = {
        "raw": _dataset_eda_stats(df),
        "val": val_eda,
        "test": test_eda,
    }
    return train_mat, val_mat, train_val_mat, test_mat, eda
