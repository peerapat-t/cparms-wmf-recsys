# This file creates globally time-ordered train, validation, and test interaction
# matrices under a fit-window-restricted protocol.
#
# The model universe is fixed by the fit window only:
#   - Tuning stage : fit on train, evaluate on validation.
#   - Final stage  : fit on train+validation, evaluate on test.
# Users and items that appear only in a future (evaluation) window are NOT padded
# into the fit matrix as empty rows/columns. They therefore cannot influence the
# learned parameters (e.g. association-rule support denominators) and cannot appear
# as candidate items. Evaluation interactions on unknown users/items are dropped.

import numpy as np
import pandas as pd
from scipy import sparse

from util.feedback import LIKE_THRESHOLD


SparseMatrix = sparse.csr_matrix


# Inputs:
# - timestamps: Raw timestamp column, already datetime-typed or holding ISO-8601 strings.
# Output: The column as datetime64, so ordering follows calendar time.
# The whole split hangs on this ordering, so the accepted formats are deliberately
# narrow. Strings are parsed as strict ISO-8601 rather than compared directly, because
# string collation only matches chronological order for zero-padded ISO. Ambiguous
# inputs are rejected instead of guessed: a numeric column hides its epoch unit
# (s / ms / us / ns), and a non-ISO layout like "1/5/2020" hides whether it means
# 5 January or 1 May. Guessing either one would silently move every cutoff date.
def _parse_timestamps(timestamps: pd.Series) -> pd.Series:
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


# Inputs:
# - df: All interaction events, before any train/validation/test partitioning.
# Output: Dataset size, density, and sparsity used by the EDA report.
def _dataset_eda_stats(df: pd.DataFrame) -> dict:
    n_users = int(df["userid"].nunique())
    n_items = int(df["itemid"].nunique())
    n_interactions = int(len(df))
    density_pct = 100.0 * n_interactions / (n_users * n_items)

    return {
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": n_interactions,
        "n_users_eval": None,
        "n_future_user_interactions_dropped": None,
        "n_future_user_positive_targets_dropped": None,
        "n_future_item_interactions_dropped": None,
        "n_future_item_positive_targets_dropped": None,
        "interaction_1": None,
        "interaction_2": None,
        "interaction_3_plus": None,
        "density_pct": density_pct,
        "sparsity_pct": 100.0 - density_pct,
    }


# Inputs:
# - matrix: Evaluation interaction matrix after fit-window filtering.
# Output: Active user/item counts, stored interactions, density, and sparsity.
def _matrix_eda_stats(matrix: SparseMatrix) -> dict:
    n_users = int(np.count_nonzero(matrix.getnnz(axis=1)))
    n_items = int(np.count_nonzero(matrix.getnnz(axis=0)))
    n_interactions = int(matrix.nnz)
    density_pct = (
        100.0 * n_interactions / (n_users * n_items)
        if n_users and n_items
        else 0.0
    )

    return {
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": n_interactions,
        "density_pct": density_pct,
        "sparsity_pct": 100.0 - density_pct,
    }


# Inputs:
# - fit_mat: Raw-rating matrix of the stage's fit window.
# - eval_mat: Raw-rating matrix of the stage's evaluation window.
# Output: Evaluation-user counts by positive-history activity group.
# These must reproduce exactly what the experiment scores, so eligibility is defined
# the same way it is at evaluation time: at least one positive (L) target in the eval
# window AND at least one positive (L) interaction in the fit history.
def _activity_group_counts(
    fit_mat: SparseMatrix,
    eval_mat: SparseMatrix,
) -> dict:
    # Step 1: Count each user's positive history inside the fit window.
    fit_positive = fit_mat.copy()
    fit_positive.data = (fit_positive.data > LIKE_THRESHOLD).astype(np.float32)
    fit_positive.eliminate_zeros()
    history_counts = np.asarray(fit_positive.getnnz(axis=1)).reshape(-1)

    # Step 2: Keep users who also have a positive target to be scored against.
    eval_positive = eval_mat.copy()
    eval_positive.data = (eval_positive.data > LIKE_THRESHOLD).astype(np.float32)
    eval_positive.eliminate_zeros()
    has_relevant_target = np.asarray(eval_positive.getnnz(axis=1)).reshape(-1) > 0
    eligible = has_relevant_target & (history_counts >= 1)

    # Step 3: Bin the eligible users by positive-history size.
    group_counts = {
        "interaction_1": int((eligible & (history_counts == 1)).sum()),
        "interaction_2": int((eligible & (history_counts == 2)).sum()),
        "interaction_3_plus": int((eligible & (history_counts >= 3)).sum()),
    }
    return group_counts


# Inputs:
# - fit_mat: Training history used for the evaluation stage.
# - eval_mat: Validation or test interactions after fit-window filtering.
# Output: EDA statistics for the interactions and users actually evaluated.
def _evaluation_eda_stats(
    fit_mat: SparseMatrix,
    eval_mat: SparseMatrix,
) -> dict:
    matrix_stats = _matrix_eda_stats(eval_mat)
    activity_groups = _activity_group_counts(fit_mat, eval_mat)

    return {
        "n_users": matrix_stats["n_users"],
        "n_items": matrix_stats["n_items"],
        "n_interactions": matrix_stats["n_interactions"],
        "n_users_eval": sum(activity_groups.values()),
        **activity_groups,
        "density_pct": matrix_stats["density_pct"],
        "sparsity_pct": matrix_stats["sparsity_pct"],
    }


# Inputs:
# - fit_events: Events whose users and items define the stage's model universe.
# - eval_events: Future events before fit-window filtering.
# Output: Counts of evaluation interactions and positive targets removed because the
# user or the item never appears in the fit window. The two drop reasons are reported
# as disjoint categories: unknown-user events first, then unknown-item events among
# the events whose user is known.
def _eval_drop_stats(
    fit_events: pd.DataFrame,
    eval_events: pd.DataFrame,
) -> dict:
    known_user = eval_events["userid"].isin(
        pd.Index(fit_events["userid"].unique())
    )
    known_item = eval_events["itemid"].isin(
        pd.Index(fit_events["itemid"].unique())
    )
    dropped_user = eval_events.loc[~known_user]
    dropped_item = eval_events.loc[known_user & ~known_item]

    return {
        "n_future_user_interactions_dropped": int(len(dropped_user)),
        "n_future_user_positive_targets_dropped": int(
            (dropped_user["rating"] > LIKE_THRESHOLD).sum()
        ),
        "n_future_item_interactions_dropped": int(len(dropped_item)),
        "n_future_item_positive_targets_dropped": int(
            (dropped_item["rating"] > LIKE_THRESHOLD).sum()
        ),
    }


# Inputs:
# - data: Indexed interaction events with u_idx, i_idx, and rating columns.
# - shape: Target (user_count, item_count) dimensions for the sparse matrix.
# Output: Canonical float32 CSR matrix containing the maximum rating per user-item pair.
def _build_sparse_matrix(
    data: pd.DataFrame,
    shape: tuple[int, int],
) -> SparseMatrix:
    if data.empty:
        return sparse.csr_matrix(shape, dtype=np.float32)

    # Step 1: Collapse duplicate user-item events to their maximum rating.
    grouped = (
        data.groupby(["u_idx", "i_idx"], sort=False, as_index=False)["rating"]
        .max()
    )

    # Step 2: Build and canonicalize the sparse user-item matrix.
    matrix = sparse.csr_matrix(
        (
            grouped["rating"].to_numpy(dtype=np.float32, copy=False),
            (
                grouped["u_idx"].to_numpy(dtype=np.int64, copy=False),
                grouped["i_idx"].to_numpy(dtype=np.int64, copy=False),
            ),
        ),
        shape=shape,
        dtype=np.float32,
    )
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


# Inputs:
# - timestamps: Ascending-sorted timestamp array for all events.
# - train_ratio: Fraction of globally ordered rows targeted for training.
# - val_ratio: Fraction of globally ordered rows targeted for validation.
# Output: (train_end, val_end) row positions snapped to unique-timestamp boundaries
# so that no single timestamp straddles two partitions.
def _unique_timestamp_boundaries(
    timestamps: np.ndarray,
    train_ratio: float,
    val_ratio: float,
) -> tuple[int, int]:
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


# Inputs:
# - fit_events: Events whose users/items define the model universe for this stage.
# - eval_events: Future events used only for evaluation.
# Output: Fit/evaluation matrices indexed over the
# fit window's users and items; eval interactions on unknown users/items are dropped.
def _index_and_build(
    fit_events: pd.DataFrame,
    eval_events: pd.DataFrame,
) -> tuple[SparseMatrix, SparseMatrix]:
    # Step 1: Assign contiguous indices from the fit window only.
    user_codes, user_uniques = pd.factorize(fit_events["userid"], sort=True)
    item_codes, item_uniques = pd.factorize(fit_events["itemid"], sort=True)
    shape = (int(user_uniques.size), int(item_uniques.size))

    fit_df = fit_events.copy()
    fit_df["u_idx"] = user_codes
    fit_df["i_idx"] = item_codes
    fit_mat = _build_sparse_matrix(fit_df, shape)

    # Step 2: Map evaluation events onto the fit index, dropping unknown users/items.
    eval_user_idx = pd.Index(user_uniques).get_indexer(eval_events["userid"])
    eval_item_idx = pd.Index(item_uniques).get_indexer(eval_events["itemid"])
    known_user = eval_user_idx >= 0
    known_item = eval_item_idx >= 0
    keep = known_user & known_item

    eval_df = eval_events.loc[keep].copy()
    eval_df["u_idx"] = eval_user_idx[keep]
    eval_df["i_idx"] = eval_item_idx[keep]
    eval_mat = _build_sparse_matrix(eval_df, shape)

    return fit_mat, eval_mat


# Inputs:
# - df: Interaction events containing userid, itemid, timestamp, and rating.
# - train_ratio: Fraction of globally ordered rows assigned to training.
# - val_ratio: Fraction of globally ordered rows assigned to validation.
# - test_ratio: Fraction of globally ordered rows assigned to testing.
# Output: Fit-window-restricted tuning/final matrices plus raw, validation, and test EDA fields.
def get_temporal_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> tuple[SparseMatrix, SparseMatrix, SparseMatrix, SparseMatrix, dict]:
    # Step 1: Validate required columns, split ratios, row count, and null values.
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

    # Step 2: Sort all events globally by timestamp with stable input-row tie breaking.
    df["_rowid"] = np.arange(df.shape[0], dtype=np.int64)
    df = df.sort_values(
        ["timestamp", "_rowid"],
        ascending=[True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    # Step 3: Choose split points at unique-timestamp boundaries.
    timestamps = df["timestamp"].to_numpy()
    train_end, val_end = _unique_timestamp_boundaries(
        timestamps, train_ratio, val_ratio
    )

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    # Step 4: Tuning stage — fit on train, evaluate on validation (fit-window universe).
    train_mat, val_mat = _index_and_build(train_df, val_df)

    # Step 5: Final stage — fit on train+validation, evaluate on test (fit-window universe).
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)
    train_val_mat, test_mat = _index_and_build(trainval_df, test_df)

    val_eda = _evaluation_eda_stats(train_mat, val_mat)
    val_eda.update(_eval_drop_stats(train_df, val_df))
    test_eda = _evaluation_eda_stats(train_val_mat, test_mat)
    test_eda.update(_eval_drop_stats(trainval_df, test_df))
    eda = {
        "raw": _dataset_eda_stats(df),
        "val": val_eda,
        "test": test_eda,
    }
    return train_mat, val_mat, train_val_mat, test_mat, eda
