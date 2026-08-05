# This file generates CPARMS recommendation signals from association-rule tokens.

import warnings

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

from util.feedback import to_B, to_L
from util.seed_config import resolve_seed


SIGNAL_DTYPE = np.float32
COUNT_DTYPE = np.int64
NORMALIZATION_MODES = frozenset({None, "row_max", "log_row_max"})


# Inputs:
# - labels: Integer cluster label for each entity.
# - cluster_count: Number of columns in the membership matrix.
# Output: CSR one-hot matrix with one cluster membership per entity row.
def _one_hot(labels: np.ndarray, cluster_count: int) -> sparse.csr_matrix:
    entity_count = labels.size
    if entity_count == 0 or cluster_count == 0:
        return sparse.csr_matrix(
            (entity_count, cluster_count),
            dtype=COUNT_DTYPE,
        )
    # Step 1: Place one value at each entity's assigned cluster column.
    return sparse.csr_matrix(
        (
            np.ones(entity_count, dtype=COUNT_DTYPE),
            (np.arange(entity_count, dtype=np.int64), labels),
        ),
        shape=(entity_count, cluster_count),
        dtype=COUNT_DTYPE,
    )


# Generator_CPARMS builds a sparse user-item recommendation signal from positive
# feedback alone: it clusters users and items, mines single-antecedent directional
# association rules over item and cluster tokens from joint token counts, and scores
# each retained rule by its confidence.
class Generator_CPARMS:
    # Inputs:
    # - k_user: Requested number of user clusters, or None to skip them.
    # - K_item: Requested number of item clusters, or None to skip them.
    # - min_support: Minimum association-rule support in [0, 1].
    # - min_confidence: Minimum association-rule confidence in [0, 1].
    # - min_lift: Minimum non-negative association-rule lift.
    # - normalize: None, per-user "row_max", or log-damped per-user
    #   "log_row_max" normalization.
    # - random_state: Optional random seed for KMeans.
    # Output: Initialized CPARMS generator with validated configuration.
    def __init__(
        self,
        k_user: int | None = 1,
        K_item: int | None = 1,
        min_support: float = 0.0,
        min_confidence: float = 0.0,
        min_lift: float = 0.0,
        normalize: str | None = "row_max",
        random_state: int | None = None,
    ):
        # Step 1: Normalize and validate clustering configuration.
        self.k_user = None if k_user is None else int(k_user)
        self.K_item = None if K_item is None else int(K_item)
        if self.k_user is not None and self.k_user <= 0:
            raise ValueError("k_user must be > 0 or None.")
        if self.K_item is not None and self.K_item <= 0:
            raise ValueError("K_item must be > 0 or None.")

        # Step 2: Normalize and validate association-rule metric thresholds.
        self.min_support = float(min_support)
        if not 0.0 <= self.min_support <= 1.0:
            raise ValueError("min_support must be in [0, 1].")
        self.min_confidence = float(min_confidence)
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1].")
        self.min_lift = float(min_lift)
        if not np.isfinite(self.min_lift) or self.min_lift < 0.0:
            raise ValueError("min_lift must be finite and >= 0.")

        # Step 3: Normalize and validate the normalization option.
        self.normalize = None if normalize is None else str(normalize)
        if self.normalize not in NORMALIZATION_MODES:
            raise ValueError(
                "normalize must be None, 'row_max', or 'log_row_max'."
            )

        # Step 4: Store deterministic settings.
        self.random_state = resolve_seed(random_state)

    # Inputs:
    # - features: CSR feature matrix with one row per entity.
    # - active_mask: Boolean mask selecting rows used to fit KMeans.
    # - requested_clusters: Maximum requested number of clusters.
    # Output: Canonical labels for all rows and the used cluster count.
    def _fit_predict_clusters(
        self,
        features: sparse.csr_matrix,
        active_mask: np.ndarray,
        requested_clusters: int,
    ) -> tuple[np.ndarray, int]:
        # Step 1: Handle empty or entirely inactive entity sets.
        entity_count = features.shape[0]
        if entity_count == 0:
            return np.empty(0, dtype=np.int64), 0

        active_count = int(active_mask.sum())
        if active_count == 0:
            return np.zeros(entity_count, dtype=np.int64), 1

        # Step 2: Fit no more clusters than there are active training rows.
        fitted_cluster_count = min(requested_clusters, active_count)
        model = KMeans(
            n_clusters=fitted_cluster_count,
            random_state=self.random_state,
            n_init=10,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(features[active_mask].astype(SIGNAL_DTYPE))
        # Step 3: Predict a cluster for every row, including inactive rows.
        labels = model.predict(features.astype(SIGNAL_DTYPE)).astype(
            np.int64,
            copy=False,
        )

        # Step 4: Remap labels to contiguous identifiers for only the clusters used.
        used_labels = np.unique(labels)
        labels = np.searchsorted(used_labels, labels).astype(
            np.int64,
            copy=False,
        )
        return labels, int(used_labels.size)

    # Inputs:
    # - L: Binary positive-history user-item matrix.
    # - item_cluster_membership: One-hot item-to-cluster membership matrix.
    # Output: Binary CSR user-to-item-cluster presence matrix.
    @staticmethod
    def _cluster_presence(
        L: sparse.csr_matrix,
        item_cluster_membership: sparse.csr_matrix,
    ) -> sparse.csr_matrix:
        # Step 1: Count each user's positive-history items in every item cluster.
        presence = (L @ item_cluster_membership).tocsr()
        if presence.nnz:
            # Step 2: Binarize positive cluster counts to presence indicators.
            presence.data.fill(1)
        presence.eliminate_zeros()
        presence.sort_indices()
        return presence

    # Inputs:
    # - signal: Sparse matrix to convert and canonicalize.
    # Output: Canonical float32 CSR signal matrix.
    @staticmethod
    def _canonicalize_signal(
        signal: sparse.spmatrix,
    ) -> sparse.csr_matrix:
        # Step 1: Convert to float32 CSR and remove duplicate or zero entries.
        signal = signal.tocsr().astype(SIGNAL_DTYPE, copy=False)
        signal.sum_duplicates()
        signal.eliminate_zeros()
        signal.sort_indices()
        return signal

    # Inputs:
    # - transactions: Binary user-token transaction matrix (layout: [L | U | E]).
    # - item_count: Number of individual item tokens (L columns, at offset 0).
    # Output: Retained rule-score matrix from antecedent tokens to item consequents.
    def _build_rule_scores(
        self,
        transactions: sparse.csr_matrix,
        item_count: int,
    ) -> sparse.csr_matrix:
        # Step 1: Determine transaction and consequent dimensions.
        # Transaction layout: [L(0..item_count-1) | U | E]; only L columns may be consequents.
        user_count, token_count = transactions.shape
        if user_count == 0 or token_count == 0 or item_count == 0:
            return sparse.csr_matrix(
                (token_count, item_count),
                dtype=SIGNAL_DTYPE,
            )

        # Step 2: Count token frequency and every token-pair count the rules are mined from.
        token_count_vector = np.asarray(
            transactions.sum(axis=0)
        ).reshape(-1).astype(np.float64, copy=False)
        rule_pair_counts = (transactions.T @ transactions).tocoo()

        # Step 3: Keep directional pairs whose consequent is an individual item.
        # Cluster tokens (col >= item_count) provide antecedent context only.
        rhs_allowed = rule_pair_counts.col < item_count
        keep = rhs_allowed & (rule_pair_counts.row != rule_pair_counts.col)

        antecedent = rule_pair_counts.row[keep].astype(np.int64, copy=False)
        consequent_token = rule_pair_counts.col[keep].astype(
            np.int64,
            copy=False,
        )
        pair_count = rule_pair_counts.data[keep].astype(
            np.float64,
            copy=False,
        )

        # Step 4: Calculate support, confidence, and lift for each candidate rule.
        antecedent_count = token_count_vector[antecedent]
        consequent_count = token_count_vector[consequent_token]
        support = pair_count / float(user_count)
        confidence = pair_count / antecedent_count
        lift = (
            pair_count * float(user_count)
            / (antecedent_count * consequent_count)
        )

        # Step 5: Filter rules against configured metric thresholds.
        metric_keep = (
            (support >= self.min_support)
            & (confidence >= self.min_confidence)
            & (lift >= self.min_lift)
        )
        antecedent = antecedent[metric_keep]
        consequent_token = consequent_token[metric_keep]
        confidence = confidence[metric_keep]

        # Step 6: Use rule confidence as the rule score.
        score = confidence.astype(SIGNAL_DTYPE, copy=False)

        # Step 7: Assemble and canonicalize the antecedent-to-item score matrix.
        rule_scores = sparse.coo_matrix(
            (score, (antecedent, consequent_token)),
            shape=(token_count, item_count),
            dtype=SIGNAL_DTYPE,
        ).tocsr()
        rule_scores.sum_duplicates()
        rule_scores.eliminate_zeros()
        rule_scores.sort_indices()
        return rule_scores

    # Inputs:
    # - signal: Sparse user-item score matrix.
    # Output: CSR matrix whose nonempty rows are divided by their maximum value.
    @staticmethod
    def _row_max_normalize(
        signal: sparse.csr_matrix,
    ) -> sparse.csr_matrix:
        # Step 1: Convert the signal to float32 CSR and handle the empty case.
        signal = signal.tocsr().astype(SIGNAL_DTYPE, copy=False)
        if signal.nnz == 0:
            return signal

        # Step 2: Calculate one maximum score per user row.
        row_max = signal.max(axis=1)
        if sparse.issparse(row_max):
            row_max = row_max.toarray()
        row_max = np.asarray(row_max, dtype=SIGNAL_DTYPE).reshape(-1)
        # Step 3: Divide each stored value by its row's maximum.
        row_divisors = np.repeat(row_max, np.diff(signal.indptr))
        signal.data /= row_divisors
        signal.eliminate_zeros()
        signal.sort_indices()
        return signal

    # Inputs:
    # - signal: Sparse user-item score matrix to normalize.
    # - normalize: None, per-user "row_max", or log-damped per-user
    #   "log_row_max" normalization.
    # Output: Normalized sparse signal matrix. The modes are mutually exclusive;
    # exactly one branch below runs.
    @classmethod
    def _normalize_signal(
        cls,
        signal: sparse.csr_matrix,
        normalize: str | None,
    ) -> sparse.csr_matrix:
        if normalize is None:
            return signal
        if normalize == "row_max":
            return cls._row_max_normalize(signal)
        if normalize == "log_row_max":
            # Compress heavy-tailed rule-score sums with log1p before per-user
            # row-max scaling, so one dominant item does not squash the
            # remaining targets in its row toward zero.
            signal = signal.tocsr().astype(SIGNAL_DTYPE, copy=False)
            signal.data = np.log1p(signal.data)
            return cls._row_max_normalize(signal)
        raise ValueError(
            "normalize must be None, 'row_max', or 'log_row_max'."
        )

    # Inputs:
    # - Y: Dense or sparse non-negative user-item interaction matrix.
    # Output: Normalized sparse CPARMS user-item signal matrix.
    def fit_transform(self, Y) -> sparse.csr_matrix:
        # Step 1: Prepare two binary views of the fit history.
        # L contains positive feedback and is the only view used for clustering,
        # token construction, and rule mining. B contains every observed rating
        # and is used only to prevent already-seen items from being recommended.
        L = to_L(Y, dtype=COUNT_DTYPE)
        B = to_B(Y, dtype=COUNT_DTYPE)
        _, item_count = L.shape
        observed_user_mask = np.asarray(B.getnnz(axis=1)).reshape(-1) > 0
        positive_user_mask = np.asarray(L.getnnz(axis=1)).reshape(-1) > 0

        # Step 2: Add individual liked-item tokens first; layout is [L | U | E].
        transaction_blocks = [L]
        user_membership = None
        item_cluster_presence = None

        # Step 3: Add user-cluster (U) tokens when user clustering is enabled.
        if self.k_user is not None:
            user_labels, user_cluster_count = self._fit_predict_clusters(
                L, positive_user_mask, self.k_user
            )
            user_membership = _one_hot(user_labels, user_cluster_count)
            transaction_blocks.append(user_membership)

        # Step 4: Add liked-item-cluster (E) tokens when item clustering is enabled.
        # These E tokens provide antecedent context only; consequents are always items.
        if self.K_item is not None:
            L_T = L.T.tocsr()
            item_active = np.asarray(L_T.getnnz(axis=1)).reshape(-1) > 0
            item_labels, item_cluster_count = self._fit_predict_clusters(
                L_T, item_active, self.K_item
            )
            item_membership = _one_hot(item_labels, item_cluster_count)
            item_cluster_presence = self._cluster_presence(L, item_membership)
            transaction_blocks.append(item_cluster_presence)

        # Step 5: Combine the enabled token blocks into one transaction matrix.
        transactions = sparse.hstack(
            transaction_blocks,
            format="csr",
            dtype=COUNT_DTYPE,
        )
        transactions.sum_duplicates()
        transactions.eliminate_zeros()
        transactions.sort_indices()

        # Step 6: Mine rules from observed fit users only. Empty rows may represent
        # future evaluation users; they receive the learned rules below but must not
        # affect support, confidence, lift, or token counts during rule fitting.
        rule_scores = self._build_rule_scores(
            transactions[observed_user_mask],
            item_count,
        )

        # Step 7: Apply confidence scores separately for personalized antecedents
        # (liked items and item-cluster presence) and generic antecedents
        # (user-cluster membership).
        token_offset = item_count
        personalized_signal = (
            L.astype(SIGNAL_DTYPE) @ rule_scores[:item_count]
        )
        generic_signal = sparse.csr_matrix(
            L.shape,
            dtype=SIGNAL_DTYPE,
        )

        if user_membership is not None:
            user_token_stop = token_offset + user_membership.shape[1]
            generic_signal = (
                user_membership.astype(SIGNAL_DTYPE)
                @ rule_scores[token_offset:user_token_stop]
            )
            token_offset = user_token_stop

        if item_cluster_presence is not None:
            item_token_stop = token_offset + item_cluster_presence.shape[1]
            personalized_signal = personalized_signal + (
                item_cluster_presence.astype(SIGNAL_DTYPE)
                @ rule_scores[token_offset:item_token_stop]
            )

        personalized_signal = self._canonicalize_signal(personalized_signal)
        generic_signal = self._canonicalize_signal(generic_signal)

        if personalized_signal.nnz or generic_signal.nnz:
            # Step 8: Remove every previously observed item from each rule
            # family. B is used only for this mask; it never contributes tokens,
            # clusters, or rules.
            B_seen = B.astype(SIGNAL_DTYPE)
            personalized_signal = personalized_signal - (
                personalized_signal.multiply(B_seen)
            )
            generic_signal = generic_signal - generic_signal.multiply(B_seen)
            personalized_signal = self._canonicalize_signal(
                personalized_signal
            )
            generic_signal = self._canonicalize_signal(generic_signal)

        # Step 9: Combine personalized and generic rule-confidence contributions.
        signal = self._canonicalize_signal(
            personalized_signal + generic_signal
        )

        return self._normalize_signal(signal, self.normalize)
