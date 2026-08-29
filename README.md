# CPARMS-WMF Recommender-System Experiments

**Author:** Peerapat Tancharoen, KMITL

This repository contains a reproducible study of top-N
recommendation under sparse feedback and new-user cold start. Explicit Amazon
ratings are converted into liked, disliked, and seen interaction matrices, then
used to compare seven non-personalized, matrix-factorization, neural, graph, and
CPARMS-regularized recommenders. `run_experiments.py` (batch/GCE) and
`run_experiments.ipynb` (interactive) run temporal
splitting, hyperparameter selection, final evaluation, significance testing,
dataset EDA, and Excel export.

## Models

The current experiment runners enable seven collaborative-filtering models. Each model has
one primary implementation file; CoFactor, RME, and CPARMS-LD also
keep their auxiliary-signal construction beside the model that consumes it.

| Model key | Model | Implementation | Role |
| --- | --- | --- | --- |
| `01 ItemPop` | ItemPop | `model/baseline/itempop.py` | Deterministic global positive-popularity baseline. |
| `02 Standard-WMF` | Standard-WMF | `model/baseline/standard.py` | ALS-trained implicit-feedback WMF, no auxiliary signal. |
| `03 CoFactor` | CoFactor | `model/baseline/cofactor.py` | WMF with shared item factors regularized by an item-item SPPMI term. |
| `04 RME` | RME | `model/baseline/rme.py` | WMF regularized by three SPPMI co-occurrence terms: item-item positive, item-item negative, and user-user positive. |
| `05 NeuMF` | NeuMF | `model/baseline/neumf.py` | Neural matrix factorization combining GMF and MLP branches. |
| `06 LightGCN` | LightGCN | `model/baseline/lightgcn.py` | Linear user-item graph propagation optimized with BPR loss. |
| `07 CPARMS-LD` | CPARMS-LD | `model/proposed/cparms_all.py` | WMF jointly regularized by separately weighted liked and disliked CPARMS signals. |

`cparms_all.py` keeps both polarity-specific signal generators and the joint
ALS update code in one implementation file.

### Baseline Venues

Five of the six non-proposed models reimplement a published method. ItemPop
is a standard non-personalized heuristic with no single source paper, so it is
omitted. Venue tiers are [CORE2023](http://portal.core.edu.au/conf-ranks/)
ratings, the most recent official CORE conference ranking.

| Model | Paper | Venue | CORE rank |
| --- | --- | --- | --- |
| Standard-WMF | Hu, Koren & Volinsky, "Collaborative Filtering for Implicit Feedback Datasets" | ICDM 2008 | A* |
| CoFactor | Liang, Altosaar, Charlin & Blei, "Factorization Meets the Item Embedding: Regularizing Matrix Factorization with Item Co-occurrence" | RecSys 2016 | A (B in CORE2018; upgraded in CORE2023) |
| RME | Tran, Lee, Liao & Lee, "Regularizing Matrix Factorization with User and Item Embeddings for Recommendation" | CIKM 2018 | A |
| NeuMF | He, Liao, Zhang, Nie, Hu & Chua, "Neural Collaborative Filtering" | WWW 2017 | A* |
| LightGCN | He, Deng, Wang, Li, Zhang & Wang, "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation" | SIGIR 2020 | A* |

All five venues are A or A* under CORE2023, so every baseline is reimplemented
from a top-tier publication in its subfield (data mining, recommender
systems, knowledge management, the web, and information retrieval,
respectively).

## Feedback Matrices

`util/feedback.py` centralizes the conversion from raw ratings to the matrices
used by training and evaluation:

```text
Y = raw non-negative ratings, values preserved
B = seen interactions, where Y != 0 becomes 1
L = liked interactions, where Y > 4.0 becomes 1
D = disliked interactions, where 0 < Y < 4.0 becomes 1
```

Ratings strictly greater than `4.0` are positive feedback (`L`); ratings
strictly less than `4.0` (and greater than `0`) are negative feedback (`D`).
For integer star ratings, this keeps only `5` as a like and `1`, `2`, `3` as
dislikes. A rating of exactly `4.0` is a deliberate neutral buffer: it is
excluded from both `L` and `D` so it never contributes a token to either
CPARMS rule-mining pipeline, but it still counts in `Y` and `B`, so it affects
duplicate collapse, seen-item masking, and fit-window history.

The threshold was moved from `3.0` to `4.0` (both `LIKE_THRESHOLD` and
`DISLIKE_THRESHOLD` in `util/feedback.py`) because Amazon review ratings skew
positive: measured across all five shipped datasets, a `3.0` split left `L:D`
ratios between 9.9:1 and 40.1:1, too imbalanced for the disliked-rule miner to
find workable coverage. At `4.0`, the same datasets measure `L:D` between
2.8:1 and 12.3:1. This changes what counts as a positive target everywhere,
not just for CPARMS: evaluation relevance (`ranking_metrics_at_k`), every
model's `L` fit matrix, and both CPARMS generators. Results produced under the
old `3.0` threshold are not comparable to results produced under `4.0`.

## CPARMS Method

`07 CPARMS-LD` is the proposed method; the other six enabled models
(`01 ItemPop`, `02 Standard-WMF`,
`03 CoFactor`, `04 RME`, `05 NeuMF`, `06 LightGCN`) are the
comparison baselines listed in [Models](#models) above.

### CPARMS Signals

`Generator_CPARMS_Liked` and `Generator_CPARMS_Disliked` in `cparms_all.py`
each build a sparse user-item signal from a training matrix, from one polarity's history only
(`L` for liked, `D` for disliked):

1. Canonicalize raw feedback as `Y`, seen interactions as `B`, and this
   generator's one-sided history (`L` or `D`).
2. Add individual history-item tokens from that one-sided history.
3. Optionally fit KMeans user clusters on active history rows and add
   user-cluster tokens.
4. Optionally fit KMeans item clusters on active history item rows and add
   history-item-cluster presence tokens (E).
5. Combine token blocks in the transaction layout `[L|D | U | E]`.
6. Count directional token pairs over observed fit users only and keep only rules
   whose consequent is an individual item token; cluster tokens provide antecedent
   context only. Padded evaluation users never affect the rule statistics.
7. Filter rules by `min_support`, `min_confidence`, and `min_lift`.
8. Score retained rules by rule `confidence`.
9. Split confidence contributions into generic user-cluster rules and
   personalized item/item-cluster rules, then remove previously seen items
   (any rating, from `B`) from both families.
10. Add the generic and personalized rule-confidence contributions into item
    scores.
11. Optionally normalize with per-row `row_max` or log-damped per-row
    `log_row_max` (`log1p` before row-max scaling).

`CPARMS_LD` (CPARMS-LD) consumes both matrices in the same ALS solve. The liked
term pulls scores toward positive rule confidence with `gamma_like`; the
disliked term pulls them toward negative rule confidence with
`gamma_dislike`. When both signals contain the same user-item pair, both loss
terms remain active and are balanced by their independently sampled weights.
The search value `0.0` disables the corresponding polarity term, so the same
LD implementation can test liked-only, disliked-only, and no-signal settings
without maintaining separate model classes.
The `_net_signal()` helper is used only to log the density of
`S_liked - S_disliked` in the runners; it is not the matrix optimized by
`CPARMS_LD.fit()`.

In CPARMS-LD, shared item factors are learned from users with
observed fit-window ratings. Evaluation-only users are folded in from their
learned CPARMS signals while item factors remain fixed.

## Experiment Workflow

`run_experiments.py` and `run_experiments.ipynb` run the complete experiment:

1. Load selected CSV files from `database/csv/`.
2. Sort all events globally by `timestamp`, using original row order to break
   timestamp ties.
3. Split rows into chronological train, validation, and test partitions targeting
   `70/10/20` ratios. Cut points are snapped to the nearest unique-timestamp
   boundary, so no single timestamp straddles two partitions and the realized
   ratios can drift slightly from the targets.
4. Index tuning matrices over train and validation users, padding validation-only
   users as empty train rows. Keep only train-known items.
5. Run the reproducible random hyperparameter search once under `TUNING_SEED`.
6. Select and freeze one dataset/model configuration by validation overall
   `NDCG@10`.
7. Index final matrices over train, validation, and test users, padding test-only
   users as empty train+validation rows. Keep only train+validation-known items.
8. Retrain each frozen configuration on train + validation under every
   `SENSITIVITY_SEED`, changing only `random_state`.
9. Evaluate on test at `K = 10, 20, 50, 100, 200`, running the per-user paired
   test only for `STAT_TEST_SEED`.
10. Aggregate sensitivity mean/standard deviation and export the five result
    sheets documented below.

The experiment targets **new-user cold start** under sparse history. Every
evaluation user is represented in the stage matrices, including users with no
fit-window interaction. The item catalog remains fit-window-restricted:
validation uses train-known items and test uses train+validation-known items.
Interactions on future-only items are excluded because item cold start is outside
this experiment. Empty padded user rows do not contribute to CPARMS rule fitting
or shared item-factor training; their factors are folded in from the learned
cluster-based CPARMS signal while item factors remain fixed.

The CSV inputs are expected to be deduplicated at the `userid,itemid` level.
`_build_sparse_matrix()` still defensively collapses duplicate user-item events
inside a partition to their maximum rating.

## Current Experiment Configuration

| Setting | Value |
| --- | --- |
| Batch runner | `run_experiments.py` |
| Interactive notebook | `run_experiments.ipynb` |
| Tuning seed | `42` (`TUNING_SEED`; random search runs only once) |
| Sensitivity seeds | `42`, `43`, `44`, `45`, `46` (`SENSITIVITY_SEEDS`) |
| Statistical-test seed | `42` (`STAT_TEST_SEED`; per-user paired test only) |
| Search rounds | `30` unique random configurations per tuned model under the tuning seed, with no forced anchor; the best validation `NDCG@10` configuration is frozen |
| Split ratios | `70/10/20` train/validation/test |
| Ranking cutoffs | `10`, `20`, `50`, `100`, `200` |
| Selection metric | Validation overall `NDCG@10` |
| User activity groups | `interaction_0`, `interaction_1`, `interaction_2`, `interaction_3_plus` |
| Selected datasets | All five prepared CSVs (`01_amz_beauty`, `02_amz_industry`, `03_amz_pantry`, `04_amz_music`, `05_amz_instruments`); toggle them via `SELECTED_DATASETS` |
| Enabled models | `01 ItemPop`, `02 Standard-WMF`, `03 CoFactor`, `04 RME`, `05 NeuMF`, `06 LightGCN`, `07 CPARMS-LD` |
| Output workbook | `results/final_results_<utc_timestamp>.xlsx` |
| Output sheets | `results`, `seed_summary`, `best_hyperparameters`, `significance`, `dataset_eda` |
| Floating-point dtype | `float32` for every model (`FLOAT_DTYPE` in `util/dtype_config.py`) |

Every tuned model owns a complete search-space definition and a separate
reproducible sampling stream. Standard-WMF, CoFactor, RME, and CPARMS-LD use
the same candidate values for WMF-backbone fields
that have the same meaning (`latent`, `lambda_rate`, `n_sweeps`, and
`alpha`), but every model draws those fields independently. Within CPARMS-LD,
its one sampled rule configuration is reused by its internal liked and
disliked generators. Its `gamma_like` and `gamma_dislike` weights are sampled
independently. All model streams
are derived from `TUNING_SEED`, so changing one model's search space does not
shift another model's draws.

The final experiment follows a two-phase seed protocol. First, the runner
runs the full validation search once and freezes the best hyperparameter set for
each dataset/model. It then copies that set for every sensitivity run and
replaces only `random_state` with the current sensitivity seed. This changes
model initialization, stochastic sampling, and CPARMS KMeans without rerunning
hyperparameter selection. The per-user paired significance test is computed
only for `STAT_TEST_SEED`; the five fixed-hyperparameter runs are summarized
separately with their mean and sample standard deviation. ItemPop is
deterministic and therefore has zero seed variation apart from runtime noise.

For an unattended GCE run, install the pinned dependencies and launch the
batch runner from the repository root:

```bash
python -m pip install -r requirements.txt
python run_experiments.py --check-only
python -u run_experiments.py --rounds 30 2>&1 | tee experiment.log
```

Use `python run_experiments.py --help` for dataset, seed, and output-directory
options. Relative input and output paths are resolved from the repository, so
the command can also be invoked from another working directory.

## Hyperparameter Search Space

| Parameter | Values |
| --- | --- |
| WMF-family `latent` candidates | `10`, `20`, `40`, `60`, `80`, `100` |
| WMF-family `lambda_rate` candidates | `0.0001`, `0.001`, `0.01`, `0.1`, `1` |
| WMF-family `n_sweeps` candidates | `5`, `10`, `15`, `20`, `25` |
| WMF-family `alpha` candidates | `1`, `5`, `10`, `20`, `40`, `80` |
| CPARMS-LD `gamma_like`/`gamma_dislike` | `0`, `0.1`, `0.5`, `1`, `2`, `5` |
| `normalize` | `row_max`, `log_row_max` |
| `k_user` | `None`, `1`, `2`, `3`, `5` |
| `K_item` | `None`, `1`, `2`, `3`, `5` |
| `min_support` | `0.0`, `0.0001`, `0.0003`, `0.001`, `0.002` |
| `min_confidence` | `0.0`, `0.0005`, `0.001`, `0.002`, `0.003` |
| `min_lift` | `0.0`, `0.5`, `0.8`, `1.0`, `1.5` |
| CoFactor relative scale `ell` | `0.01`, `0.03`, `0.1`, `1`, `10` |
| CoFactor `gamma` | Derived as `1 / ell` |
| CoFactor normalized context L2 | `0.00001`, `0.0001`, `0.0003`, `0.001`, `0.01`, `0.1`, `1` |
| CoFactor `negative_samples` | Fixed at reference value `1` |
| RME `lambda_context_rate` | Independently drawn from the WMF regularization candidate values |
| RME `gamma_item_pos`, `gamma_item_neg` | Each independently drawn from `0.1`, `1`, `5`, `10`, `20` |
| RME `gamma_user_pos` | `0.01`, `0.1`, `0.5`, `1`, `2` |
| RME `negative_samples` | Fixed at reference value `1` |
| NeuMF predictive factors | `8`, `16`, `32`, `64` |
| NeuMF learning rate | `0.0001`, `0.0005`, `0.001`, `0.005` |
| NeuMF MF/layer L2 | Each independently drawn from `0`, `0.000001`, `0.00001`, `0.0001` |
| NeuMF epochs | `20`, `30`, `50` |
| NeuMF batch size | `128`, `256`, `512`, `1024` |
| NeuMF negatives per positive | Fixed at `4` for a common sampling budget |
| LightGCN `latent` | `32`, `64`, `128` |
| LightGCN layers | `1`, `2`, `3`, `4` |
| LightGCN learning rate | `0.0001`, `0.0005`, `0.001`, `0.005` |
| LightGCN L2 | `0.000001`, `0.00001`, `0.0001`, `0.001`, `0.01` |
| LightGCN epochs | `30`, `50`, `100` |
| LightGCN batch size | `1024`, `2048`, `4096` |
| LightGCN negatives per sampled user | Fixed at `1` by its pairwise training procedure |

There is no fixed first-round configuration. Published/released values are
ordinary candidates in their corresponding search spaces and are subject to
the same random sampling as the other candidates.

The three rule-filter thresholds are sampled independently, so the random search
can explore cross-combinations of rule coverage, confidence, and lift filtering.

## Significance Testing

`experiments/significance.py` runs a paired t-test (`scipy.stats.ttest_rel`)
on per-user NDCG@`SELECTION_K` between `SIGNIFICANCE_PRIMARY_MODEL` (currently
`07 CPARMS-LD`) and every other model evaluated on the same dataset, for the
overall population and for each user activity group. `ranking_metrics_at_k(...,
return_per_user=True)` is what makes this possible: it returns each evaluated
user's own NDCG vector alongside the usual aggregated means, so two models'
scores can be paired user-for-user rather than compared only in aggregate.
The notebook requests these per-user vectors only for `STAT_TEST_SEED` and does
not pool repeated users from the five sensitivity runs.
Results land in the `significance` sheet of the output workbook, with `n`
(evaluated users), `mean_diff` (primary minus baseline), `t_stat`, `p_value`,
and boolean columns at the 90%/95%/99% confidence levels. The notebook reports
the two-sided p-values returned by SciPy without a multiple-comparison
correction.

## Evaluation

`experiments/all_ranker.py` computes NDCG for users with at least one relevant
evaluation item (`rating > 4.0`) in the fit-known item catalog. This includes
evaluation-only users with zero fit-window interactions. All items stored in the
training matrix are removed from each user's candidate list, including ratings
`1`, `2`, `3`, and `4`. Score ties are resolved deterministically by item index.

User activity groups are built from every observed fit-window rating in `B`,
regardless of rating polarity:

| Group | Observed fit-window interactions |
| --- | --- |
| `interaction_0` | 0 |
| `interaction_1` | 1 |
| `interaction_2` | 2 |
| `interaction_3_plus` | 3 or more |

`interaction_0` contains evaluation-only users padded as empty fit rows and is
the protocol's true zero-interaction new-user group.

## Output Workbook Structure

The experiment writes `results/final_results_<utc_timestamp>.xlsx`, where the
timestamp is generated in UTC with `%Y%m%d_%H%M%S`.

### `results`

One row is written for each final test run: `seed` x `dataset_name` x enabled
`model`. Every tuned model uses the same frozen hyperparameters in all five
rows; only `random_state` changes with `seed`.

Non-metric columns for the default model set:

```text
seed
dataset_name
model
s_mat_runtime
model_runtime
total_runtime
n_iter_random_search
selection_metric
selection_k
selection_group
validation_selection_score
n_users_eval
n_positive_targets
latent
lambda_rate
n_sweeps
alpha
random_state
gamma
lambda_context_rate
negative_samples
gamma_item_pos
gamma_item_neg
gamma_user_pos
gamma_like
gamma_dislike
learning_rate
epochs
batch_size
hidden_layers
reg_mf
reg_layers
n_layers
k_user
K_item
min_support
min_confidence
min_lift
normalize
```

Within one CPARMS-LD configuration, its sampled
`k_user`/`K_item`/`min_support`/`min_confidence`/`min_lift`/`normalize` values
are reused by its internal liked and disliked signal generators -- see
[Current Experiment Configuration](#current-experiment-configuration).
`k_user=None` disables user clustering, while `K_item=None` disables item
clustering and its item-cluster-presence tokens.

Parameter columns depend on the enabled models; unused model-specific parameter
columns are blank. Because ItemPop is not tuned, its rows carry no hyperparameter
values and its `validation_selection_score` is `NaN`. `s_mat_runtime` is `0` for
ItemPop, Standard-WMF, NeuMF, and LightGCN. It records explicit SPPMI
construction for CoFactor and rule-signal construction for CPARMS-LD.
RME builds its three SPPMI matrices inside `fit()`, so that work is included in
`model_runtime`. All runtime values are recorded in minutes.

Metric columns are appended in cutoff order for each `K` in
`10, 20, 50, 100, 200`:

```text
ndcg_all@K
ndcg_user_interaction_0@K
ndcg_user_interaction_1@K
ndcg_user_interaction_2@K
ndcg_user_interaction_3_plus@K
```

### `seed_summary`

One row is written for each `dataset_name` x `model`. `n_seeds` records the
number of distinct sensitivity seeds, and every runtime and NDCG column from
`results` is summarized as:

```text
<column>_mean
<column>_std
```

The `_std` fields use pandas' sample standard deviation (`ddof=1`) across the
fixed-hyperparameter sensitivity runs.

### `best_hyperparameters`

One row is written for each enabled `dataset_name` x `model`. It records
`tuning_seed`, `validation_selection_score`, and the exact hyperparameters
selected by the single tuning run. ItemPop has no tuned parameters. The
`random_state` in this sheet is the tuning seed; the per-run `results` sheet
records the sensitivity seed actually used for final retraining.

### `significance`

One row is written for each (`dataset_name`, activity group, baseline model)
combination, comparing `SIGNIFICANCE_PRIMARY_MODEL` against every other model
evaluated under `STAT_TEST_SEED` on that dataset. See
[Significance Testing](#significance-testing) above for the test itself.

```text
seed
dataset_name
group
baseline
primary_model
k
n
mean_diff
t_stat
p_value
sig_90
sig_95
sig_99
```

`group` is `all` or one of the four activity groups. `n` is the number of
users the paired test ran over (all users for `all`, or just that group's
users). `mean_diff` is the primary model's mean NDCG@`k` minus the baseline's;
positive means the primary model scored higher. `sig_90`/`sig_95`/`sig_99` are
boolean flags for `p_value < 0.10`/`0.05`/`0.01` respectively -- `sig_99`
implies `sig_95` implies `sig_90` for the same row.

### `dataset_eda`

Three rows are written for each enabled dataset in the final results workbook,
one for each `raw`, `val`, and `test` split:

```text
dataset_name
split
n_users
n_items
n_interactions
n_liked
n_disliked
n_users_eval
n_future_user_interactions_padded
n_future_user_positive_targets_padded
n_future_item_interactions_dropped
n_future_item_positive_targets_dropped
interaction_0
interaction_1
interaction_2
interaction_3_plus
density_pct
sparsity_pct
```

The `raw` row describes the full input dataset; its evaluation and group fields
are blank. The `val` and `test` rows describe evaluation interactions retained
after future-only items are removed. Activity groups count every observed
fit-window rating; eligibility requires a positive evaluation target.
`interaction_0` contains padded future users. The future-user columns count
known-item events retained for those padded users, while the future-item columns
count out-of-scope item-cold-start events that were dropped. Density is
`interactions / (users * items) * 100`, and sparsity is `100 - density`.
`n_liked` counts ratings strictly above `LIKE_THRESHOLD` and `n_disliked`
counts positive ratings strictly below `DISLIKE_THRESHOLD` (see
[Feedback Matrices](#feedback-matrices)); their sum is `<=` `n_interactions`
because ratings exactly at the threshold count in neither.

## Datasets

The dataset configuration maps a dataset key to a CSV path. The repository
includes five prepared CSVs, and both runners select all five by default:

| Dataset key | File |
| --- | --- |
| `01_amz_beauty` | `database/csv/dataset_amazon_lux_beauty_5_core.csv` |
| `02_amz_industry` | `database/csv/dataset_amazon_industry_5_core.csv` |
| `03_amz_pantry` | `database/csv/dataset_amazon_pantry_5_core.csv` |
| `04_amz_music` | `database/csv/dataset_amazon_music_5_core.csv` |
| `05_amz_instruments` | `database/csv/dataset_amazon_instruments_5_core.csv` |

Each CSV must contain these columns (additional columns are ignored):

```text
userid,itemid,timestamp,rating
```

Input constraints enforced by `get_temporal_split()`:

- No nulls in any of the four columns.
- `timestamp` must be datetime-typed or ISO-8601 strings (e.g. `2016-11-07`).
  A numeric column is rejected rather than guessed, because its epoch unit
  (s / ms / us / ns) is ambiguous and a wrong guess would silently move every
  cutoff date. Convert with `pd.to_datetime(col, unit=...)` beforehand.
- `rating` must be numeric, finite, and non-negative. The shipped datasets use
  integer stars `1`-`5`.
- At least three distinct timestamp groups must exist, since cut points are
  snapped to unique-timestamp boundaries.

For the batch runner, select datasets with `--datasets`; for interactive use,
edit `SELECTED_DATASETS` in `run_experiments.ipynb`.
To change which models run, edit `MODEL_PARAM_KEY`. It lists every model in report
order and maps each one to its key in a hyperparameter sample; ItemPop maps to `None`
because it has nothing to tune, so the random search skips it and it is fitted only at
final-test time.

The supplied CSV files are ready to use. To regenerate them, place the Amazon
5-core JSON Lines dumps expected by `database/preprocess.py` in
`database/json/`, then run the preprocessing script. Raw dumps are not included
in the repository. The preprocessing step keeps the four required fields,
converts Unix timestamps to datetimes, drops incomplete rows, and keeps the
earliest review for each `userid,itemid` pair (breaking timestamp ties by higher
rating), so a later re-review cannot move a user's first observed interaction
into a later temporal split.
No k-core filter is re-applied after deduplication: the `_5_core` suffix refers to the
published 5-core source dumps, and collapsing duplicate `userid,itemid` pairs can leave
some users or items with fewer than five remaining rows.

## Project Structure

```text
cparms-wmf-recsys/
|-- database/
|   |-- csv/                     # Preprocessed experiment CSVs
|   `-- preprocess.py            # Convert optional raw JSON Lines dumps to CSV
|-- experiments/
|   |-- all_ranker.py            # NDCG evaluation and user activity groups
|   |-- global_temporal_split.py # Global temporal split and EDA
|   |-- hyperparams_set.py       # Reproducible random hyperparameter sampling
|   `-- significance.py          # Paired per-user t-test between models
|-- model/
|   |-- baseline/
|   |   |-- itempop.py           # Global-popularity baseline
|   |   |-- standard.py          # Standard implicit WMF
|   |   |-- cofactor.py          # CoFactor: WMF + item-item SPPMI
|   |   |-- rme.py               # RME: WMF + item-pos/item-neg/user-pos SPPMI
|   |   |-- neumf.py             # Neural matrix factorization
|   |   `-- lightgcn.py          # Graph collaborative filtering
|   `-- proposed/
|       `-- cparms_all.py        # CPARMS-LD: jointly regularized WMF
|-- paper/                       # Thesis, paper, presentation, and references
|-- results/                     # Generated Excel experiment outputs
|-- util/
|   |-- dtype_config.py         # Shared float32 dtype for every model
|   |-- feedback.py             # Y/B/L/D feedback conversion helpers
|   |-- seed_config.py          # Reproducibility helpers
|   `-- signal_utils.py         # Signal-density logging helper
|-- .python-version             # Reference Python version
|-- run_experiments.ipynb       # Interactive experiment workflow
|-- run_experiments.py          # Batch/GCE experiment runner
|-- README.md
`-- requirements.txt            # Pinned package versions
```
