#!/usr/bin/env python
# coding: utf-8

# Import experiment dependencies and start the runtime timer.
import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from util.seed_config import configure_reproducibility
from util.feedback import to_B
from util.signal_utils import log_signal_density

from model.baseline.standard import WMF
from model.baseline.cofactor import CoFactor, build_item_sppmi_matrix as build_cofactor_item_sppmi_matrix
from model.baseline.rme import RME
from model.baseline.neumf import NeuMF
from model.baseline.lightgcn import LightGCN
from model.proposed.cparms_all import CPARMS_LD, Generator_CPARMS_Liked as _GL, Generator_CPARMS_Disliked as _GD, _net_signal
from model.baseline.itempop import ItemPop

from experiments.global_temporal_split import get_temporal_split
from experiments.hyperparams_set import generate_hyperparam_samples
from experiments.all_ranker import (
    build_user_activity_groups,
    ranking_metrics_at_k,
)
from experiments.significance import significance_table

PROJECT_ROOT = Path(__file__).resolve().parent
ALL_DATASETS = {
    "01_amz_beauty": "database/csv/dataset_amazon_lux_beauty_5_core.csv",
    # "02_amz_industry": "database/csv/dataset_amazon_industry_5_core.csv",
    # "03_amz_pantry": "database/csv/dataset_amazon_pantry_5_core.csv",
    # "04_amz_music": "database/csv/dataset_amazon_music_5_core.csv",
    # "05_amz_instruments": "database/csv/dataset_amazon_instruments_5_core.csv",
}


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Tune and evaluate all recommender models."
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Unique random-search configurations per tuned model (default: 30).",
    )
    parser.add_argument("--tuning-seed", type=int, default=42)
    parser.add_argument(
        "--sensitivity-seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
    )
    parser.add_argument(
        "--stat-test-seed",
        type=int,
        default=None,
        help="Seed for paired significance tests (default: tuning seed).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(ALL_DATASETS),
        default=None,
        help="Dataset keys to run (default: all).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Output directory, relative to the repository unless absolute.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate imports, paths, and sampling, then exit before training.",
    )
    return parser.parse_args()


def _print_frame(title, frame):
    print(f"\n{title}")
    if frame.empty:
        print("<empty>")
    else:
        print(frame.to_string(index=False))


args = _parse_args()
start_time = time.time()


# Define the reproducible experiment and evaluation configuration.
TUNING_SEED = int(args.tuning_seed)
SENSITIVITY_SEEDS = tuple(args.sensitivity_seeds)
STAT_TEST_SEED = (
    TUNING_SEED if args.stat_test_seed is None else int(args.stat_test_seed)
)

N_ITER_RANDOM_SEARCH = int(args.rounds)
METRIC_KS = (10, 20, 50, 100, 200) 
USER_ACTIVITY_GROUPS = ("interaction_0", "interaction_1", "interaction_2", "interaction_3_plus")

SELECTION_METRIC = "ndcg"
SELECTION_K = 10

MODEL_PARAM_KEY = {
    # Non-personalized baseline
    "01 ItemPop": None,
    # Models sharing the WMF/ALS backbone
    "02 Standard-WMF": "standard_wmf",
    "03 CoFactor": "cofactor_wmf",
    "04 RME": "rme",
    # Models with different backbones
    "05 NeuMF": "neumf",
    "06 LightGCN": "lightgcn",
    # Proposed method
    "07 CPARMS-LD": "cparms_ld",
}

SIGNIFICANCE_PRIMARY_MODEL = "07 CPARMS-LD"

if not SENSITIVITY_SEEDS:
    raise ValueError("Select at least one sensitivity seed.")
if len(set(SENSITIVITY_SEEDS)) != len(SENSITIVITY_SEEDS):
    raise ValueError("SENSITIVITY_SEEDS must contain unique values.")
if STAT_TEST_SEED not in SENSITIVITY_SEEDS:
    raise ValueError("STAT_TEST_SEED must be included in SENSITIVITY_SEEDS.")
if N_ITER_RANDOM_SEARCH <= 0:
    raise ValueError("--rounds must be a positive integer.")

RESULTS_DIR = args.results_dir
if not RESULTS_DIR.is_absolute():
    RESULTS_DIR = PROJECT_ROOT / RESULTS_DIR
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Load each selected dataset and create leakage-safe temporal splits.
selected_dataset_names = tuple(args.datasets or ALL_DATASETS)
SELECTED_DATASETS = {
    name: PROJECT_ROOT / ALL_DATASETS[name]
    for name in selected_dataset_names
}

print("Experiment configuration")
print(f"  project_root={PROJECT_ROOT}")
print(f"  rounds={N_ITER_RANDOM_SEARCH}")
print(f"  tuning_seed={TUNING_SEED}")
print(f"  sensitivity_seeds={SENSITIVITY_SEEDS}")
print(f"  datasets={selected_dataset_names}")
print(f"  results_dir={RESULTS_DIR}")

missing_dataset_paths = [
    str(path) for path in SELECTED_DATASETS.values() if not path.is_file()
]
if missing_dataset_paths:
    raise FileNotFoundError(
        "Missing dataset files:\n" + "\n".join(missing_dataset_paths)
    )
if args.check_only:
    generate_hyperparam_samples(rounds=1, global_seed=TUNING_SEED)
    print("Configuration check passed; training was not started.")
    raise SystemExit(0)

dataset_configs = []
dataset_eda_all = {}
for dataset_name, dataset_path in SELECTED_DATASETS.items():
    dataset_df = pd.read_csv(dataset_path)
    train_mat, val_mat, train_val_mat, test_mat, eda = get_temporal_split(
        dataset_df,
    )

    dataset_configs.append({
        'name': dataset_name,
        'train': train_mat,
        'val': val_mat,
        'train_val': train_val_mat,
        'test': test_mat,
    })
    dataset_eda_all[dataset_name] = eda

if not dataset_configs:
    raise ValueError('Select at least one dataset.')


# Select model hyperparameters once using the dedicated tuning seed.
best_params_by_tuning_seed = {}

for tuning_seed in (TUNING_SEED,):
    configure_reproducibility(tuning_seed)
    print(f"\n#################### TUNING SEED: {tuning_seed} ####################")

    hyperparam_samples = generate_hyperparam_samples(
        rounds=N_ITER_RANDOM_SEARCH,
        global_seed=tuning_seed,
    )

    seed_best = {}
    best_params_by_tuning_seed[int(tuning_seed)] = seed_best

    for dataset_cfg in dataset_configs:
        dataset_name = dataset_cfg["name"]
        train_mat = dataset_cfg["train"]
        val_mat = dataset_cfg["val"]
        user_count, item_count = train_mat.shape
        print(f"\n========== DATASET: {dataset_name} ==========")

        train_B = to_B(train_mat)
        val_user_groups = build_user_activity_groups(train_B)

        for model_name, param_key in MODEL_PARAM_KEY.items():
            if param_key is None:
                continue
            seed_best[(model_name, dataset_name)] = {
                "best_score": float("-inf"), "best_params": None,
            }
        for round_idx, sample in enumerate(hyperparam_samples):
            print(f"\n--- Round {round_idx + 1}/{N_ITER_RANDOM_SEARCH} ---")
            for model_name, param_key in MODEL_PARAM_KEY.items():
                if param_key is None:
                    continue
                params = sample[param_key]
                print(f"\nTraining {model_name}: {params}")

                if param_key == "standard_wmf":
                    model = WMF(
                        user_count=user_count,
                        item_count=item_count,
                        K=params["latent"],
                        lambda_rate=params["lambda_rate"],
                        alpha=params["alpha"],
                        random_state=params["random_state"],
                    )
                    model.fit(Y=train_mat, n_sweeps=params["n_sweeps"])
                    pred_source = model

                elif param_key == "neumf":
                    model = NeuMF(
                        user_count=user_count,
                        item_count=item_count,
                        latent=params["latent"],
                        hidden_layers=params["hidden_layers"],
                        learning_rate=params["learning_rate"],
                        reg_mf=params["reg_mf"],
                        reg_layers=params["reg_layers"],
                        negative_samples=params["negative_samples"],
                        random_state=params["random_state"],
                    )
                    model.fit(
                        Y=train_mat,
                        epochs=params["epochs"],
                        batch_size=params["batch_size"],
                    )
                    pred_source = model

                elif param_key == "lightgcn":
                    model = LightGCN(
                        user_count=user_count,
                        item_count=item_count,
                        latent=params["latent"],
                        n_layers=params["n_layers"],
                        learning_rate=params["learning_rate"],
                        lambda_rate=params["lambda_rate"],
                        negative_samples=params["negative_samples"],
                        random_state=params["random_state"],
                    )
                    model.fit(
                        Y=train_mat,
                        epochs=params["epochs"],
                        batch_size=params["batch_size"],
                    )
                    pred_source = model

                elif param_key == "cofactor_wmf":
                    model = CoFactor(
                        user_count=user_count,
                        item_count=item_count,
                        K=params["latent"],
                        lambda_rate=params["lambda_rate"],
                        alpha=params["alpha"],
                        gamma=params["gamma"],
                        lambda_context_rate=params["lambda_context_rate"],
                        negative_samples=params["negative_samples"],
                        random_state=params["random_state"],
                    )
                    model.fit(Y=train_mat, n_sweeps=params["n_sweeps"])
                    pred_source = model

                elif param_key == "rme":
                    model = RME(
                        user_count=user_count,
                        item_count=item_count,
                        K=params["latent"],
                        lambda_rate=params["lambda_rate"],
                        alpha=params["alpha"],
                        gamma_item_pos=params["gamma_item_pos"],
                        gamma_item_neg=params["gamma_item_neg"],
                        gamma_user_pos=params["gamma_user_pos"],
                        lambda_context_rate=params["lambda_context_rate"],
                        negative_samples=params["negative_samples"],
                        random_state=params["random_state"],
                    )
                    model.fit(Y=train_mat, n_sweeps=params["n_sweeps"])
                    pred_source = model

                elif param_key == "cparms_ld":
                    ld_signal_liked = _GL(
                        k_user=params["k_user"],
                        K_item=params["K_item"],
                        min_support=params["min_support"],
                        min_confidence=params["min_confidence"],
                        min_lift=params["min_lift"],
                        normalize=params["normalize"],
                        random_state=params["random_state"],
                    ).fit_transform(train_mat)
                    log_signal_density("S_liked", ld_signal_liked, user_count, item_count)
                    ld_signal_disliked = _GD(
                        k_user=params["k_user"],
                        K_item=params["K_item"],
                        min_support=params["min_support"],
                        min_confidence=params["min_confidence"],
                        min_lift=params["min_lift"],
                        normalize=params["normalize"],
                        random_state=params["random_state"],
                    ).fit_transform(train_mat)
                    log_signal_density("S_disliked", ld_signal_disliked, user_count, item_count)
                    log_signal_density("S_net (LD)", _net_signal(ld_signal_liked, ld_signal_disliked), user_count, item_count)
                    model = CPARMS_LD(
                        user_count=user_count,
                        item_count=item_count,
                        K=params["latent"],
                        lambda_rate=params["lambda_rate"],
                        alpha=params["alpha"],
                        gamma_like=params["gamma_like"],
                        gamma_dislike=params["gamma_dislike"],
                        random_state=params["random_state"],
                    )
                    model.fit(
                        Y=train_mat,
                        S_liked=ld_signal_liked,
                        S_disliked=ld_signal_disliked,
                        n_sweeps=params["n_sweeps"],
                        fit_user_mask=(train_mat.getnnz(axis=1) > 0),
                    )
                    pred_source = model

                else:
                    raise ValueError(f"Unsupported model parameter key: {param_key}")

                results_ranking = ranking_metrics_at_k(
                    pred_source=pred_source,
                    train_mat=train_mat,
                    test_mat=val_mat,
                    ks=METRIC_KS,
                    user_groups=val_user_groups,
                )
                all_ndcg = results_ranking["all"]["ndcg"]
                ug = results_ranking["user"]
                print(
                    f"Results: NDCG@10 all={all_ndcg[10]:.6f}, "
                    f"i0={ug['interaction_0']['ndcg'][10]:.6f}, "
                    f"i1={ug['interaction_1']['ndcg'][10]:.6f}, "
                    f"i2={ug['interaction_2']['ndcg'][10]:.6f}, "
                    f"i3+={ug['interaction_3_plus']['ndcg'][10]:.6f}"
                )

                score = results_ranking["all"][SELECTION_METRIC][SELECTION_K]
                tracker = seed_best[(model_name, dataset_name)]
                if score > tracker["best_score"]:
                    tracker["best_score"] = float(score)
                    tracker["best_params"] = dict(params)

                del results_ranking


# Retrain the fixed selected configurations across sensitivity seeds.
final_test_rows = []
significance_rows = []
for experiment_seed in SENSITIVITY_SEEDS:
    configure_reproducibility(experiment_seed)
    print(f"\n#################### [TEST] SEED: {experiment_seed} ####################")
    seed_best = best_params_by_tuning_seed[int(TUNING_SEED)]
    run_significance = int(experiment_seed) == int(STAT_TEST_SEED)

    for dataset_cfg in dataset_configs:
        dataset_name = dataset_cfg["name"]
        train_val_mat = dataset_cfg["train_val"]
        test_mat = dataset_cfg["test"]
        user_count, item_count = train_val_mat.shape
        print(f"\n========== [TEST] DATASET: {dataset_name} ==========")

        train_val_B = to_B(train_val_mat)
        test_user_groups = build_user_activity_groups(train_val_B)
        group_bool_masks = {}
        for group_name, user_idx in test_user_groups.items():
            mask = np.zeros(user_count, dtype=bool)
            mask[user_idx] = True
            group_bool_masks[group_name] = mask

        per_user_by_model = {}
        for model_name, param_key in MODEL_PARAM_KEY.items():
            print(f"\n----- [TEST] MODEL: {model_name} -----")
            s_mat_runtime = 0.0
            log_params = {}
            validation_selection_score = float("nan")

            if param_key is None:
                t0 = time.perf_counter()
                model = ItemPop(user_count=user_count, item_count=item_count)
                model.fit(Y=train_val_mat)
                model_runtime = (time.perf_counter() - t0) / 60.0
                pred_source = model

            else:
                tracker = seed_best[(model_name, dataset_name)]
                selected_params = tracker["best_params"]
                if selected_params is None:
                    print(f"[Skip] No valid best parameters for {model_name}")
                    continue
                params = dict(selected_params)
                params["random_state"] = int(experiment_seed)
                print(f"{model_name} hyperparams: {params}")
                validation_selection_score = tracker["best_score"]

                if param_key == "standard_wmf":
                    t0 = time.perf_counter()
                    model = WMF(
                        user_count=user_count,
                        item_count=item_count,
                        K=params["latent"],
                        lambda_rate=params["lambda_rate"],
                        alpha=params["alpha"],
                        random_state=params["random_state"],
                    )
                    model.fit(Y=train_val_mat, n_sweeps=params["n_sweeps"])
                    model_runtime = (time.perf_counter() - t0) / 60.0
                    pred_source = model
                    log_params = dict(params)

                elif param_key == "neumf":
                    t0 = time.perf_counter()
                    model = NeuMF(
                        user_count=user_count,
                        item_count=item_count,
                        latent=params["latent"],
                        hidden_layers=params["hidden_layers"],
                        learning_rate=params["learning_rate"],
                        reg_mf=params["reg_mf"],
                        reg_layers=params["reg_layers"],
                        negative_samples=params["negative_samples"],
                        random_state=params["random_state"],
                    )
                    model.fit(
                        Y=train_val_mat,
                        epochs=params["epochs"],
                        batch_size=params["batch_size"],
                    )
                    model_runtime = (time.perf_counter() - t0) / 60.0
                    pred_source = model
                    log_params = dict(params)

                elif param_key == "lightgcn":
                    t0 = time.perf_counter()
                    model = LightGCN(
                        user_count=user_count,
                        item_count=item_count,
                        latent=params["latent"],
                        n_layers=params["n_layers"],
                        learning_rate=params["learning_rate"],
                        lambda_rate=params["lambda_rate"],
                        negative_samples=params["negative_samples"],
                        random_state=params["random_state"],
                    )
                    model.fit(
                        Y=train_val_mat,
                        epochs=params["epochs"],
                        batch_size=params["batch_size"],
                    )
                    model_runtime = (time.perf_counter() - t0) / 60.0
                    pred_source = model
                    log_params = dict(params)

                elif param_key == "cofactor_wmf":
                    if params["gamma"] > 0.0:
                        t0 = time.perf_counter()
                        sppmi_mat = build_cofactor_item_sppmi_matrix(
                            train_val_mat,
                            negative_samples=params["negative_samples"],
                        )
                        s_mat_runtime = (time.perf_counter() - t0) / 60.0
                    else:
                        sppmi_mat = None
                    t0 = time.perf_counter()
                    model = CoFactor(
                        user_count=user_count,
                        item_count=item_count,
                        K=params["latent"],
                        lambda_rate=params["lambda_rate"],
                        alpha=params["alpha"],
                        gamma=params["gamma"],
                        lambda_context_rate=params["lambda_context_rate"],
                        negative_samples=params["negative_samples"],
                        random_state=params["random_state"],
                    )
                    model.fit(Y=train_val_mat, M=sppmi_mat, n_sweeps=params["n_sweeps"])
                    model_runtime = (time.perf_counter() - t0) / 60.0
                    pred_source = model
                    log_params = dict(params)

                elif param_key == "rme":
                    t0 = time.perf_counter()
                    model = RME(
                        user_count=user_count,
                        item_count=item_count,
                        K=params["latent"],
                        lambda_rate=params["lambda_rate"],
                        alpha=params["alpha"],
                        gamma_item_pos=params["gamma_item_pos"],
                        gamma_item_neg=params["gamma_item_neg"],
                        gamma_user_pos=params["gamma_user_pos"],
                        lambda_context_rate=params["lambda_context_rate"],
                        negative_samples=params["negative_samples"],
                        random_state=params["random_state"],
                    )
                    model.fit(Y=train_val_mat, n_sweeps=params["n_sweeps"])
                    model_runtime = (time.perf_counter() - t0) / 60.0
                    pred_source = model
                    log_params = dict(params)

                elif param_key == "cparms_ld":
                    t0 = time.perf_counter()
                    signal_liked = _GL(
                        k_user=params["k_user"],
                        K_item=params["K_item"],
                        min_support=params["min_support"],
                        min_confidence=params["min_confidence"],
                        min_lift=params["min_lift"],
                        normalize=params["normalize"],
                        random_state=params["random_state"],
                    ).fit_transform(train_val_mat)
                    log_signal_density("S_liked", signal_liked, user_count, item_count)
                    signal_disliked = _GD(
                        k_user=params["k_user"],
                        K_item=params["K_item"],
                        min_support=params["min_support"],
                        min_confidence=params["min_confidence"],
                        min_lift=params["min_lift"],
                        normalize=params["normalize"],
                        random_state=params["random_state"],
                    ).fit_transform(train_val_mat)
                    log_signal_density("S_disliked", signal_disliked, user_count, item_count)
                    log_signal_density("S_net (LD)", _net_signal(signal_liked, signal_disliked), user_count, item_count)
                    s_mat_runtime = (time.perf_counter() - t0) / 60.0
                    t0 = time.perf_counter()
                    model = CPARMS_LD(
                        user_count=user_count,
                        item_count=item_count,
                        K=params["latent"],
                        lambda_rate=params["lambda_rate"],
                        alpha=params["alpha"],
                        gamma_like=params["gamma_like"],
                        gamma_dislike=params["gamma_dislike"],
                        random_state=params["random_state"],
                    )
                    model.fit(
                        Y=train_val_mat,
                        S_liked=signal_liked,
                        S_disliked=signal_disliked,
                        n_sweeps=params["n_sweeps"],
                        fit_user_mask=(train_val_mat.getnnz(axis=1) > 0),
                    )
                    model_runtime = (time.perf_counter() - t0) / 60.0
                    pred_source = model
                    log_params = dict(params)

                else:
                    raise ValueError(f"Unsupported model parameter key: {param_key}")
            results_ranking = ranking_metrics_at_k(
                pred_source=pred_source,
                train_mat=train_val_mat,
                test_mat=test_mat,
                ks=METRIC_KS,
                user_groups=test_user_groups,
                return_per_user=run_significance,
            )
            if run_significance:
                per_user_by_model[model_name] = results_ranking["per_user"]

            all_ndcg = results_ranking["all"]["ndcg"]
            ug = results_ranking["user"]
            print(
                f"Results: NDCG@10 all={all_ndcg[10]:.6f}, "
                f"i0={ug['interaction_0']['ndcg'][10]:.6f}, "
                f"i1={ug['interaction_1']['ndcg'][10]:.6f}, "
                f"i2={ug['interaction_2']['ndcg'][10]:.6f}, "
                f"i3+={ug['interaction_3_plus']['ndcg'][10]:.6f}"
            )

            row = {
                "seed": int(experiment_seed),
                "dataset_name": dataset_name,
                "model": model_name,
                "s_mat_runtime": float(s_mat_runtime),
                "model_runtime": float(model_runtime),
                "total_runtime": float(s_mat_runtime) + float(model_runtime),
                "n_iter_random_search": N_ITER_RANDOM_SEARCH,
                "selection_metric": SELECTION_METRIC,
                "selection_k": SELECTION_K,
                "selection_group": "all",
                "validation_selection_score": validation_selection_score,
                "n_users_eval": int(results_ranking["n_users_eval"]),
                "n_positive_targets": int(results_ranking["n_positive_targets"]),
            }
            row.update(log_params)
            for k, v in results_ranking["all"]["ndcg"].items():
                row[f"ndcg_all@{int(k)}"] = float(v)
            for g in USER_ACTIVITY_GROUPS:
                for k, v in results_ranking["user"][g]["ndcg"].items():
                    row[f"ndcg_user_{g}@{int(k)}"] = float(v)
            final_test_rows.append(row)

            del results_ranking

        if run_significance and SIGNIFICANCE_PRIMARY_MODEL in per_user_by_model:
            baselines_per_user = {
                name: per_user
                for name, per_user in per_user_by_model.items()
                if name != SIGNIFICANCE_PRIMARY_MODEL
            }
            sig_rows = significance_table(
                primary_per_user=per_user_by_model[SIGNIFICANCE_PRIMARY_MODEL],
                baselines_per_user=baselines_per_user,
                k=SELECTION_K,
                group_masks=group_bool_masks,
            )
            for sig_row in sig_rows:
                sig_row["seed"] = int(experiment_seed)
                sig_row["dataset_name"] = dataset_name
                sig_row["primary_model"] = SIGNIFICANCE_PRIMARY_MODEL
            significance_rows.extend(sig_rows)


# Present per-seed results, aggregate sensitivity, and frozen hyperparameters.
df_best_results = pd.DataFrame(final_test_rows)
df_best_results.sort_values(by=["dataset_name", "model", "seed"], inplace=True)

ordered_ndcg_cols = []
for k in METRIC_KS:
    ordered_ndcg_cols.append(f"ndcg_all@{int(k)}")
    ordered_ndcg_cols.extend(
        f"ndcg_user_{g}@{int(k)}" for g in USER_ACTIVITY_GROUPS
    )

summary_value_cols = [
    "s_mat_runtime", "model_runtime", "total_runtime",
    *ordered_ndcg_cols,
]
summary_groups = ["dataset_name", "model"]
df_seed_summary = (
    df_best_results.groupby(summary_groups, sort=True)[summary_value_cols]
    .agg(["mean", "std"])
    .reset_index()
)
df_seed_summary.columns = [
    column[0] if not column[1] else f"{column[0]}_{column[1]}"
    for column in df_seed_summary.columns
]
seed_counts = (
    df_best_results.groupby(summary_groups, sort=True)["seed"]
    .nunique()
    .reset_index(name="n_seeds")
)
df_seed_summary = seed_counts.merge(
    df_seed_summary, on=summary_groups, how="left", validate="one_to_one"
)

frozen_best = best_params_by_tuning_seed[int(TUNING_SEED)]
best_hyperparameter_rows = []
for dataset_cfg in dataset_configs:
    dataset_name = dataset_cfg["name"]
    for model_name, param_key in MODEL_PARAM_KEY.items():
        row = {
            "tuning_seed": int(TUNING_SEED),
            "dataset_name": dataset_name,
            "model": model_name,
        }
        if param_key is None:
            row["validation_selection_score"] = float("nan")
        else:
            tracker = frozen_best[(model_name, dataset_name)]
            row["validation_selection_score"] = tracker["best_score"]
            if tracker["best_params"] is not None:
                row.update(tracker["best_params"])
        best_hyperparameter_rows.append(row)
df_best_hyperparameters = pd.DataFrame(best_hyperparameter_rows)
df_best_hyperparameters.sort_values(
    by=["dataset_name", "model"], inplace=True
)


# Present the single tuning-seed test run (no seed averaging).
tuning_seed_result_metrics = [
    f"ndcg_all@{SELECTION_K}",
    *(f"ndcg_user_{group}@{SELECTION_K}" for group in USER_ACTIVITY_GROUPS),
]
df_tuning_seed_results = df_best_results[
    df_best_results["seed"] == int(TUNING_SEED)
]
_print_frame(
    "Test Results (TUNING_SEED)",
    df_tuning_seed_results[["dataset_name", "model", *tuning_seed_result_metrics]],
)


# Present the frozen best hyperparameters from the tuning seed.
_print_frame("Best Hyperparameters", df_best_hyperparameters)


# Present the per-user paired test from the designated statistical-test seed.
df_significance = pd.DataFrame(significance_rows)
if not df_significance.empty:
    df_significance.sort_values(
        by=["dataset_name", "group", "baseline"], inplace=True
    )
    df_significance["sig_90"] = df_significance["p_value"] < 0.10
    df_significance["sig_95"] = df_significance["p_value"] < 0.05
    df_significance["sig_99"] = df_significance["p_value"] < 0.01

_print_frame("Statistical Tests", df_significance)


# Present the seed-sensitivity summary (mean/std) for the primary metric.
best_result_metrics = [
    f"ndcg_all@{SELECTION_K}",
    *(f"ndcg_user_{group}@{SELECTION_K}" for group in USER_ACTIVITY_GROUPS),
]
best_result_columns = ["dataset_name", "model", "n_seeds"]
for metric in best_result_metrics:
    best_result_columns.extend([f"{metric}_mean", f"{metric}_std"])
_print_frame(
    "Best Results (mean / std across SENSITIVITY_SEEDS)",
    df_seed_summary[best_result_columns],
)


# Export frozen hyperparameters, per-seed metrics, sensitivity, and tests.
timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
file_name = RESULTS_DIR / f"final_results_{timestamp_str}.xlsx"

ordered_ndcg_cols = []
for k in METRIC_KS:
    ordered_ndcg_cols.append(f"ndcg_all@{int(k)}")
    ordered_ndcg_cols.extend(f"ndcg_user_{g}@{int(k)}" for g in USER_ACTIVITY_GROUPS)

other_cols = [c for c in df_best_results.columns if c not in ordered_ndcg_cols]
df_results = df_best_results[other_cols + ordered_ndcg_cols]

df_dataset_eda = pd.DataFrame(
    [
        {"dataset_name": dataset_name, "split": split_name, **metrics}
        for dataset_name, split_eda in dataset_eda_all.items()
        for split_name, metrics in split_eda.items()
    ]
)

with pd.ExcelWriter(file_name) as writer:
    df_results.to_excel(writer, sheet_name='results', index=False)
    df_seed_summary.to_excel(writer, sheet_name='seed_summary', index=False)
    df_best_hyperparameters.to_excel(
        writer, sheet_name='best_hyperparameters', index=False
    )
    df_significance.to_excel(writer, sheet_name='significance', index=False)
    df_dataset_eda.to_excel(writer, sheet_name='dataset_eda', index=False)
print(f"Saved: {file_name}")

end_time = time.time()
print(f"Elapsed: {end_time - start_time:.1f}s")

