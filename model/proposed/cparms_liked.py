"""Generate liked CPARMS signals and fit their weighted MF model."""

import warnings
from numbers import Integral

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

from util.dtype_config import FLOAT_DTYPE
from util.feedback import LIKE_THRESHOLD, to_B, to_L, to_Y
from util.seed_config import resolve_seed


MODEL_DTYPE = FLOAT_DTYPE
SIGNAL_DTYPE = FLOAT_DTYPE
COUNT_DTYPE = np.int64
NORMALIZATION_MODES = frozenset({None, "row_max", "log_row_max"})


def _one_hot(labels: np.ndarray, cluster_count: int) -> sparse.csr_matrix:
    """Encode nonnegative cluster labels as a sparse membership matrix."""
    entity_count = labels.size
    if entity_count == 0 or cluster_count == 0:
        return sparse.csr_matrix(
            (entity_count, cluster_count),
            dtype=COUNT_DTYPE,
        )

    return sparse.csr_matrix(
        (
            np.ones(entity_count, dtype=COUNT_DTYPE),
            (np.arange(entity_count, dtype=np.int64), labels),
        ),
        shape=(entity_count, cluster_count),
        dtype=COUNT_DTYPE,
    )


class Generator_CPARMS_Liked:
    """Generate recommendation signals from liked-item association rules."""


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
        """Store clustering, rule-filtering, and normalization settings."""

        self.k_user = None if k_user is None else int(k_user)
        self.K_item = None if K_item is None else int(K_item)
        if self.k_user is not None and self.k_user <= 0:
            raise ValueError("k_user must be > 0 or None.")
        if self.K_item is not None and self.K_item <= 0:
            raise ValueError("K_item must be > 0 or None.")


        self.min_support = float(min_support)
        if not 0.0 <= self.min_support <= 1.0:
            raise ValueError("min_support must be in [0, 1].")
        self.min_confidence = float(min_confidence)
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1].")
        self.min_lift = float(min_lift)
        if not np.isfinite(self.min_lift) or self.min_lift < 0.0:
            raise ValueError("min_lift must be finite and >= 0.")


        self.normalize = None if normalize is None else str(normalize)
        if self.normalize not in NORMALIZATION_MODES:
            raise ValueError(
                "normalize must be None, 'row_max', or 'log_row_max'."
            )


        self.random_state = resolve_seed(random_state)


    def _fit_predict_clusters(
        self,
        features: sparse.csr_matrix,
        active_mask: np.ndarray,
        requested_clusters: int,
    ) -> tuple[np.ndarray, int]:
        """Cluster active rows and assign a reserved label to inactive rows."""

        entity_count = features.shape[0]
        if entity_count == 0:
            return np.empty(0, dtype=np.int64), 0

        active_count = int(active_mask.sum())
        if active_count == 0:
            return np.zeros(entity_count, dtype=np.int64), 1


        fitted_cluster_count = min(requested_clusters, active_count)
        model = KMeans(
            n_clusters=fitted_cluster_count,
            random_state=self.random_state,
            n_init=10,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(features[active_mask].astype(SIGNAL_DTYPE))

        labels = model.predict(features.astype(SIGNAL_DTYPE)).astype(
            np.int64,
            copy=False,
        )


        used_labels = np.unique(labels)
        labels = np.searchsorted(used_labels, labels).astype(
            np.int64,
            copy=False,
        )
        return labels, int(used_labels.size)


    @staticmethod
    def _cluster_presence(
        L: sparse.csr_matrix,
        item_cluster_membership: sparse.csr_matrix,
    ) -> sparse.csr_matrix:
        """Mark which item clusters appear in each user's feedback."""

        presence = (L @ item_cluster_membership).tocsr()
        if presence.nnz:

            presence.data.fill(1)
        presence.eliminate_zeros()
        presence.sort_indices()
        return presence


    @staticmethod
    def _canonicalize_signal(
        signal: sparse.spmatrix,
    ) -> sparse.csr_matrix:
        """Return a finite, nonnegative, canonical CSR signal matrix."""

        signal = signal.tocsr().astype(SIGNAL_DTYPE, copy=False)
        signal.sum_duplicates()
        signal.eliminate_zeros()
        signal.sort_indices()
        return signal


    def _build_rule_scores(
        self,
        transactions: sparse.csr_matrix,
        item_count: int,
    ) -> sparse.csr_matrix:
        """Mine token-to-item association scores from sparse transactions."""


        user_count, token_count = transactions.shape
        if user_count == 0 or token_count == 0 or item_count == 0:
            return sparse.csr_matrix(
                (token_count, item_count),
                dtype=SIGNAL_DTYPE,
            )


        token_count_vector = np.asarray(
            transactions.sum(axis=0)
        ).reshape(-1).astype(np.float64, copy=False)
        rule_pair_counts = (transactions.T @ transactions).tocoo()


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


        antecedent_count = token_count_vector[antecedent]
        consequent_count = token_count_vector[consequent_token]
        support = pair_count / float(user_count)
        confidence = pair_count / antecedent_count
        lift = (
            pair_count * float(user_count)
            / (antecedent_count * consequent_count)
        )


        metric_keep = (
            (support >= self.min_support)
            & (confidence >= self.min_confidence)
            & (lift >= self.min_lift)
        )
        antecedent = antecedent[metric_keep]
        consequent_token = consequent_token[metric_keep]
        confidence = confidence[metric_keep]


        score = confidence.astype(SIGNAL_DTYPE, copy=False)


        rule_scores = sparse.coo_matrix(
            (score, (antecedent, consequent_token)),
            shape=(token_count, item_count),
            dtype=SIGNAL_DTYPE,
        ).tocsr()
        rule_scores.sum_duplicates()
        rule_scores.eliminate_zeros()
        rule_scores.sort_indices()
        return rule_scores


    @staticmethod
    def _row_max_normalize(
        signal: sparse.csr_matrix,
    ) -> sparse.csr_matrix:
        """Scale each nonempty signal row by its maximum value."""

        signal = signal.tocsr().astype(SIGNAL_DTYPE, copy=False)
        if signal.nnz == 0:
            return signal


        row_max = signal.max(axis=1)
        if sparse.issparse(row_max):
            row_max = row_max.toarray()
        row_max = np.asarray(row_max, dtype=SIGNAL_DTYPE).reshape(-1)

        row_divisors = np.repeat(row_max, np.diff(signal.indptr))
        signal.data /= row_divisors
        signal.eliminate_zeros()
        signal.sort_indices()
        return signal


    @classmethod
    def _normalize_signal(
        cls,
        signal: sparse.csr_matrix,
        normalize: str | None,
    ) -> sparse.csr_matrix:
        """Apply the configured row-wise signal normalization."""
        if normalize is None:
            return signal
        if normalize == "row_max":
            return cls._row_max_normalize(signal)
        if normalize == "log_row_max":


            signal = signal.tocsr().astype(SIGNAL_DTYPE, copy=False)
            signal.data = np.log1p(signal.data)
            return cls._row_max_normalize(signal)
        raise ValueError(
            "normalize must be None, 'row_max', or 'log_row_max'."
        )


    def fit_transform(self, Y) -> sparse.csr_matrix:
        """Generate an unseen-item liked-association signal from feedback."""


        L = to_L(Y, dtype=COUNT_DTYPE)
        B = to_B(Y, dtype=COUNT_DTYPE)
        _, item_count = L.shape
        observed_user_mask = np.asarray(B.getnnz(axis=1)).reshape(-1) > 0
        positive_user_mask = np.asarray(L.getnnz(axis=1)).reshape(-1) > 0


        transaction_blocks = [L]
        user_membership = None
        item_cluster_presence = None


        if self.k_user is not None:
            user_labels, user_cluster_count = self._fit_predict_clusters(
                L, positive_user_mask, self.k_user
            )
            user_membership = _one_hot(user_labels, user_cluster_count)
            transaction_blocks.append(user_membership)


        if self.K_item is not None:
            L_T = L.T.tocsr()
            item_active = np.asarray(L_T.getnnz(axis=1)).reshape(-1) > 0
            item_labels, item_cluster_count = self._fit_predict_clusters(
                L_T, item_active, self.K_item
            )
            item_membership = _one_hot(item_labels, item_cluster_count)
            item_cluster_presence = self._cluster_presence(L, item_membership)
            transaction_blocks.append(item_cluster_presence)


        # Mine rules over item, user-cluster, and item-cluster tokens.
        transactions = sparse.hstack(
            transaction_blocks,
            format="csr",
            dtype=COUNT_DTYPE,
        )
        transactions.sum_duplicates()
        transactions.eliminate_zeros()
        transactions.sort_indices()


        rule_scores = self._build_rule_scores(
            transactions[observed_user_mask],
            item_count,
        )


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


            B_seen = B.astype(SIGNAL_DTYPE)
            personalized_signal = personalized_signal - (
                personalized_signal.multiply(B_seen)
            )
            generic_signal = generic_signal - generic_signal.multiply(B_seen)
            personalized_signal = self._canonicalize_signal(
                personalized_signal
            )
            generic_signal = self._canonicalize_signal(generic_signal)


        signal = self._canonicalize_signal(
            personalized_signal + generic_signal
        )

        return self._normalize_signal(signal, self.normalize)


def _validate_training_schedule(n_sweeps, verbose_every) -> tuple[int, int]:
    """Validate and return the ALS reporting schedule."""
    for name, value in (("n_sweeps", n_sweeps), ("verbose_every", verbose_every)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    return int(n_sweeps), int(verbose_every)


def _should_report_sweep(sweep_idx: int, verbose_every: int) -> bool:
    """Return whether the current sweep should emit diagnostics."""
    return sweep_idx == 0 or (sweep_idx + 1) % verbose_every == 0


def _initialize_factors(
    row_count: int,
    latent_dim: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Draw a deterministic dense latent-factor matrix."""
    return rng.normal(
        0.0,
        0.01,
        size=(int(row_count), int(latent_dim)),
    ).astype(MODEL_DTYPE)


def _wmf_loss(
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    positive_feedback: sparse.csr_matrix,
    alpha: float,
) -> float:
    """Compute the unregularized weighted reconstruction loss."""

    user_gram = user_factors.T @ user_factors
    item_gram = item_factors.T @ item_factors
    loss = float(np.sum(user_gram * item_gram))


    for user_idx in range(positive_feedback.shape[0]):
        start = positive_feedback.indptr[user_idx]
        stop = positive_feedback.indptr[user_idx + 1]
        item_idx = positive_feedback.indices[start:stop]
        if item_idx.size == 0:
            continue
        scores = item_factors[item_idx] @ user_factors[user_idx]
        loss += float(
            np.sum(
                (1.0 + alpha) * (1.0 - scores) ** 2
                - scores**2
            )
        )
    return loss


def _als_factor_update(
    L: sparse.csr_matrix,
    signal: sparse.csr_matrix,
    fixed_factors: np.ndarray,
    lambda_rate: float,
    alpha: float,
    gamma: float,
) -> np.ndarray:
    """Solve one ALS update using feedback and liked-signal confidence."""

    row_count = L.shape[0]
    latent_dim = fixed_factors.shape[1]
    identity = np.eye(latent_dim, dtype=MODEL_DTYPE)
    base = (
        fixed_factors.T @ fixed_factors
        + MODEL_DTYPE(lambda_rate) * identity
    )
    updated = np.zeros((row_count, latent_dim), dtype=MODEL_DTYPE)


    for row_idx in range(row_count):
        pref_start = L.indptr[row_idx]
        pref_stop = L.indptr[row_idx + 1]
        positive_idx = L.indices[pref_start:pref_stop]

        signal_start = signal.indptr[row_idx]
        signal_stop = signal.indptr[row_idx + 1]
        signal_idx = signal.indices[signal_start:signal_stop]
        signal_values = signal.data[signal_start:signal_stop]

        system_matrix = base.copy()
        right_hand_side = np.zeros(latent_dim, dtype=MODEL_DTYPE)

        if positive_idx.size:

            positive_factors = fixed_factors[positive_idx]
            system_matrix += MODEL_DTYPE(alpha) * (
                positive_factors.T @ positive_factors
            )
            right_hand_side += MODEL_DTYPE(1.0 + alpha) * (
                positive_factors.sum(axis=0)
            )

        if gamma > 0.0 and signal_idx.size:

            signal_factors = fixed_factors[signal_idx]
            system_matrix += MODEL_DTYPE(gamma) * (
                signal_factors.T @ signal_factors
            )
            right_hand_side += MODEL_DTYPE(gamma) * (
                signal_values @ signal_factors
            )


        updated[row_idx] = np.linalg.solve(
            system_matrix,
            right_hand_side,
        )

    return updated


def _signal_loss(
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    signal: sparse.csr_matrix,
    gamma: float,
) -> float:
    """Compute the weighted liked-signal reconstruction loss."""
    if gamma <= 0.0 or signal.nnz == 0:
        return 0.0

    loss = 0.0
    for user_idx in range(signal.shape[0]):
        start = signal.indptr[user_idx]
        stop = signal.indptr[user_idx + 1]
        item_idx = signal.indices[start:stop]
        if item_idx.size == 0:
            continue
        values = signal.data[start:stop]
        scores = item_factors[item_idx] @ user_factors[user_idx]
        loss += float(np.sum((values - scores) ** 2))

    return float(gamma) * loss


class CPARMS_L:
    """Factorize positive feedback jointly with a liked CPARMS signal."""


    def __init__(
        self,
        user_count,
        item_count,
        K,
        lambda_rate,
        alpha,
        gamma,
        threshold: float = LIKE_THRESHOLD,
        random_state=None,
    ):
        """Validate hyperparameters and initialize latent factors."""
        self.user_count = int(user_count)
        self.item_count = int(item_count)
        self.K = int(K)
        self.lambda_rate = float(lambda_rate)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.threshold = float(threshold)
        self.random_state = resolve_seed(random_state)


        if self.K <= 0:
            raise ValueError("K must be > 0.")
        if not np.isfinite(self.lambda_rate) or self.lambda_rate <= 0.0:
            raise ValueError("lambda_rate must be finite and > 0.")
        if not np.isfinite(self.alpha) or self.alpha < 0.0:
            raise ValueError("alpha must be finite and >= 0.")
        if not np.isfinite(self.gamma) or self.gamma < 0.0:
            raise ValueError("gamma must be finite and >= 0.")


        user_rng = np.random.RandomState(self.random_state)
        item_rng = np.random.RandomState((self.random_state + 104729) % (2**32))
        self.P = _initialize_factors(self.user_count, self.K, user_rng)
        self.Q = _initialize_factors(self.item_count, self.K, item_rng)


    @property
    def shape(self) -> tuple[int, int]:
        """Return the expected user-item matrix shape."""
        return self.user_count, self.item_count


    def fit(
        self,
        Y,
        S,
        n_sweeps: int,
        verbose_every: int = 5,
        fit_user_mask=None,
    ):
        """Fit liked-signal factors and infer held-out user factors."""
        n_sweeps, verbose_every = _validate_training_schedule(
            n_sweeps, verbose_every
        )

        signal = to_Y(S)
        if signal.shape != self.shape:
            raise ValueError(f"S must be shape {self.shape}.")
        L = to_L(Y, threshold=self.threshold)
        if L.shape != self.shape:
            raise ValueError(f"Y must be shape {self.shape}.")


        if fit_user_mask is None:
            fit_user_mask = np.ones(self.user_count, dtype=bool)
        else:
            fit_user_mask = np.asarray(fit_user_mask)
            if fit_user_mask.shape != (self.user_count,):
                raise ValueError(
                    f"fit_user_mask must be shape ({self.user_count},)."
                )
            if fit_user_mask.dtype != np.bool_:
                raise ValueError("fit_user_mask must contain boolean values.")
            fit_user_mask = fit_user_mask.copy()
        if not np.any(fit_user_mask):
            raise ValueError("fit_user_mask must select at least one user.")

        fit_L = L[fit_user_mask].tocsr()
        fit_signal = signal[fit_user_mask].tocsr()
        fit_L_T = fit_L.T.tocsr()
        fit_signal_t = fit_signal.T.tocsr()


        # Alternate user and item solves over the selected fitting users.
        for sweep_idx in range(n_sweeps):

            fit_user_factors = _als_factor_update(
                fit_L,
                fit_signal,
                self.Q,
                self.lambda_rate,
                self.alpha,
                self.gamma,
            )
            self.P[fit_user_mask] = fit_user_factors


            self.Q = _als_factor_update(
                fit_L_T,
                fit_signal_t,
                fit_user_factors,
                self.lambda_rate,
                self.alpha,
                self.gamma,
            )

            if _should_report_sweep(sweep_idx, verbose_every):

                wmf_loss = _wmf_loss(
                    fit_user_factors,
                    self.Q,
                    fit_L,
                    self.alpha,
                )
                signal_loss = _signal_loss(
                    fit_user_factors,
                    self.Q,
                    fit_signal,
                    self.gamma,
                )
                l2_sum = float(
                    np.sum(fit_user_factors * fit_user_factors)
                    + np.sum(self.Q * self.Q)
                )
                regularization = self.lambda_rate * l2_sum
                total_loss = wmf_loss + signal_loss + regularization
                print(
                    f"[Sweep {sweep_idx + 1}/{n_sweeps}] "
                    f"WMF_LOSS: {wmf_loss:.6f} "
                    f"LIKE_LOSS: {signal_loss:.6f} "
                    f"L2_SUM: {l2_sum:.6f} "
                    f"REG: {regularization:.6f} "
                    f"TOTAL: {total_loss:.6f}"
                )


        cold_user_mask = ~fit_user_mask
        if np.any(cold_user_mask):
            self.P[cold_user_mask] = _als_factor_update(
                L[cold_user_mask].tocsr(),
                signal[cold_user_mask].tocsr(),
                self.Q,
                self.lambda_rate,
                self.alpha,
                self.gamma,
            )
        return self


    def score_user(self, user_idx: int) -> np.ndarray:
        """Score every item for one user."""
        user_idx = int(user_idx)
        if user_idx < 0 or user_idx >= self.user_count:
            raise IndexError("user_idx is out of range.")
        return self.Q @ self.P[user_idx]
