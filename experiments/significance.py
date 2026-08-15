"""Compare per-user ranking results with paired significance tests."""

import numpy as np
from scipy import stats


def paired_ttest(
    primary_per_user,
    baseline_per_user,
    k: int,
    group_mask=None,
):
    """Run a paired t-test on two models' per-user NDCG values."""

    primary_idx = primary_per_user["user_idx"]
    baseline_idx = baseline_per_user["user_idx"]
    if not np.array_equal(primary_idx, baseline_idx):
        raise ValueError(
            "primary and baseline were evaluated on different users; "
            "they must share the same train/test split."
        )


    if k not in primary_per_user["ks"] or k not in baseline_per_user["ks"]:
        raise ValueError(f"k={k} was not requested when computing per-user NDCG.")
    primary_col = primary_per_user["ks"].index(k)
    baseline_col = baseline_per_user["ks"].index(k)
    primary_ndcg = primary_per_user["ndcg"][:, primary_col]
    baseline_ndcg = baseline_per_user["ndcg"][:, baseline_col]


    if group_mask is not None:
        keep = group_mask[primary_idx]
        primary_ndcg = primary_ndcg[keep]
        baseline_ndcg = baseline_ndcg[keep]


    n = int(primary_ndcg.size)
    if n < 2:
        return {"n": n, "mean_diff": float("nan"), "t_stat": float("nan"), "p_value": float("nan")}
    diff = primary_ndcg - baseline_ndcg
    if np.allclose(diff, 0.0):
        return {"n": n, "mean_diff": 0.0, "t_stat": 0.0, "p_value": 1.0}


    t_stat, p_value = stats.ttest_rel(primary_ndcg, baseline_ndcg)
    return {
        "n": n,
        "mean_diff": float(np.mean(diff)),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
    }


def significance_table(
    primary_per_user,
    baselines_per_user,
    k: int,
    group_masks=None,
):
    """Build paired-test result rows for each baseline and user group."""

    groups = {"all": None}
    if group_masks:
        groups.update(group_masks)


    rows = []
    for baseline_name, baseline_per_user in baselines_per_user.items():
        for group_name, group_mask in groups.items():
            test = paired_ttest(
                primary_per_user, baseline_per_user, k, group_mask=group_mask,
            )
            rows.append({
                "baseline": baseline_name,
                "group": group_name,
                "k": int(k),
                **test,
            })
    return rows
