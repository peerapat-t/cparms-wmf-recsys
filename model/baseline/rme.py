"""Train the regularized multi-embedding recommender."""

from numbers import Integral

import numpy as np
from scipy import sparse

from util.dtype_config import FLOAT_DTYPE
from util.feedback import DISLIKE_THRESHOLD, LIKE_THRESHOLD, to_D, to_L, to_Y
from util.seed_config import resolve_seed


MODEL_DTYPE = FLOAT_DTYPE


def build_sppmi_matrix(
    B: sparse.spmatrix,
    negative_samples: float = 1.0,
) -> sparse.csr_matrix:
    """Build a shifted positive-PMI co-occurrence matrix."""
    negative_samples = float(negative_samples)
    if not np.isfinite(negative_samples) or negative_samples < 1.0:
        raise ValueError("negative_samples must be finite and >= 1.")


    B = B.tocsr().astype(MODEL_DTYPE, copy=False)
    entity_count = B.shape[1]
    cooccurrence = (B.T @ B).tocsr().astype(
        MODEL_DTYPE,
        copy=False,
    )
    cooccurrence.setdiag(0)
    cooccurrence.eliminate_zeros()
    cooccurrence.sort_indices()

    if cooccurrence.nnz == 0:
        return sparse.csr_matrix(
            (entity_count, entity_count),
            dtype=MODEL_DTYPE,
        )


    row_count = np.asarray(
        cooccurrence.sum(axis=1)
    ).reshape(-1).astype(np.float64, copy=False)
    pair_count = float(cooccurrence.data.sum())
    if pair_count <= 0.0:
        return sparse.csr_matrix(
            (entity_count, entity_count),
            dtype=MODEL_DTYPE,
        )

    sppmi = cooccurrence.copy().astype(np.float64, copy=False)
    for entity_idx in range(entity_count):
        start = sppmi.indptr[entity_idx]
        stop = sppmi.indptr[entity_idx + 1]
        if start == stop:
            continue
        context_idx = sppmi.indices[start:stop]
        counts = sppmi.data[start:stop]
        denominator = row_count[entity_idx] * row_count[context_idx]
        valid = denominator > 0.0
        values = np.zeros(counts.shape, dtype=np.float64)
        values[valid] = np.log(
            counts[valid] * pair_count / denominator[valid]
        )
        sppmi.data[start:stop] = values


    if negative_samples > 1.0:
        sppmi.data -= np.log(negative_samples)
    sppmi.data[sppmi.data < 0.0] = 0.0
    sppmi = sppmi.astype(MODEL_DTYPE, copy=False)
    sppmi.eliminate_zeros()
    sppmi.sort_indices()
    return sppmi


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


class _CofactorTerm:
    """Bundle one weighted co-occurrence objective and its parameters."""


    def __init__(
        self,
        context_factors,
        primary_biases,
        context_biases,
        global_bias,
        cooccurrence,
        cooccurrence_t,
        lambda_context_rate,
        gamma,
    ):
        """Store factors, biases, matrices, and objective weights."""
        self.context_factors = context_factors
        self.primary_biases = primary_biases
        self.context_biases = context_biases
        self.global_bias = global_bias
        self.cooccurrence = cooccurrence
        self.cooccurrence_t = cooccurrence_t
        self.lambda_context_rate = lambda_context_rate
        self.gamma = gamma


    def is_active(self) -> bool:
        """Return whether the term contributes to optimization."""
        return self.gamma > 0.0 and self.cooccurrence.nnz > 0


def _primary_factor_update(
    primary_feedback: sparse.csr_matrix,
    fixed_factors: np.ndarray,
    lambda_rate: float,
    alpha: float,
    terms: list,
) -> np.ndarray:
    """Update primary factors from feedback and active context terms."""

    row_count = primary_feedback.shape[0]
    latent_dim = fixed_factors.shape[1]
    identity = np.eye(latent_dim, dtype=MODEL_DTYPE)
    base = (
        fixed_factors.T @ fixed_factors
        + MODEL_DTYPE(lambda_rate) * identity
    )
    updated = np.zeros((row_count, latent_dim), dtype=MODEL_DTYPE)


    for row_idx in range(row_count):
        start = primary_feedback.indptr[row_idx]
        stop = primary_feedback.indptr[row_idx + 1]
        positive_idx = primary_feedback.indices[start:stop]

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


        for term in terms:
            if not term.is_active():
                continue
            t_start = term.cooccurrence.indptr[row_idx]
            t_stop = term.cooccurrence.indptr[row_idx + 1]
            context_idx = term.cooccurrence.indices[t_start:t_stop]
            if context_idx.size == 0:
                continue
            targets = (
                term.cooccurrence.data[t_start:t_stop]
                - term.primary_biases[row_idx]
                - term.context_biases[context_idx]
                - term.global_bias
            )
            context = term.context_factors[context_idx]
            weight = MODEL_DTYPE(term.gamma)
            system_matrix += weight * (context.T @ context)
            right_hand_side += weight * (targets @ context)


        updated[row_idx] = np.linalg.solve(system_matrix, right_hand_side)
    return updated


def _context_factor_update(
    primary_factors: np.ndarray,
    term: _CofactorTerm,
) -> np.ndarray:
    """Update context factors for one co-occurrence term."""

    context_count = term.cooccurrence_t.shape[0]
    latent_dim = primary_factors.shape[1]
    identity = np.eye(latent_dim, dtype=MODEL_DTYPE)
    regularization = MODEL_DTYPE(term.lambda_context_rate) * identity
    weight = MODEL_DTYPE(term.gamma)
    updated = np.zeros((context_count, latent_dim), dtype=MODEL_DTYPE)


    for context_idx in range(context_count):
        start = term.cooccurrence_t.indptr[context_idx]
        stop = term.cooccurrence_t.indptr[context_idx + 1]
        primary_idx = term.cooccurrence_t.indices[start:stop]

        system_matrix = regularization.copy()
        right_hand_side = np.zeros(latent_dim, dtype=MODEL_DTYPE)
        if primary_idx.size:
            targets = (
                term.cooccurrence_t.data[start:stop]
                - term.primary_biases[primary_idx]
                - term.context_biases[context_idx]
                - term.global_bias
            )
            primary = primary_factors[primary_idx]
            system_matrix += weight * (primary.T @ primary)
            right_hand_side += weight * (targets @ primary)

        updated[context_idx] = np.linalg.solve(system_matrix, right_hand_side)
    return updated


def _primary_bias_update(
    primary_factors: np.ndarray,
    term: _CofactorTerm,
) -> np.ndarray:
    """Update primary-entity biases for one co-occurrence term."""
    primary_count = term.cooccurrence.shape[0]
    biases = np.zeros(primary_count, dtype=MODEL_DTYPE)
    if term.cooccurrence.nnz == 0:
        return biases

    for row_idx in range(primary_count):
        start = term.cooccurrence.indptr[row_idx]
        stop = term.cooccurrence.indptr[row_idx + 1]
        context_idx = term.cooccurrence.indices[start:stop]
        if context_idx.size == 0:
            continue
        targets = term.cooccurrence.data[start:stop]
        scores = term.context_factors[context_idx] @ primary_factors[row_idx]
        residual = (
            targets
            - scores
            - term.context_biases[context_idx]
            - term.global_bias
        )
        biases[row_idx] = residual.mean()
    return biases


def _context_bias_update(
    primary_factors: np.ndarray,
    term: _CofactorTerm,
) -> np.ndarray:
    """Update context-entity biases for one co-occurrence term."""
    context_count = term.cooccurrence_t.shape[0]
    biases = np.zeros(context_count, dtype=MODEL_DTYPE)
    if term.cooccurrence_t.nnz == 0:
        return biases

    for context_idx in range(context_count):
        start = term.cooccurrence_t.indptr[context_idx]
        stop = term.cooccurrence_t.indptr[context_idx + 1]
        primary_idx = term.cooccurrence_t.indices[start:stop]
        if primary_idx.size == 0:
            continue
        targets = term.cooccurrence_t.data[start:stop]
        scores = primary_factors[primary_idx] @ term.context_factors[context_idx]
        residual = targets - scores - term.primary_biases[primary_idx] - term.global_bias
        biases[context_idx] = residual.mean()
    return biases


def _global_bias_update(
    primary_factors: np.ndarray,
    term: _CofactorTerm,
) -> np.float64:
    """Update the global intercept for one co-occurrence term."""
    if term.cooccurrence.nnz == 0:
        return MODEL_DTYPE(0.0)
    residual_sum = 0.0
    for row_idx in range(term.cooccurrence.shape[0]):
        start = term.cooccurrence.indptr[row_idx]
        stop = term.cooccurrence.indptr[row_idx + 1]
        context_idx = term.cooccurrence.indices[start:stop]
        if context_idx.size == 0:
            continue
        targets = term.cooccurrence.data[start:stop]
        estimates = (
            term.context_factors[context_idx] @ primary_factors[row_idx]
            + term.primary_biases[row_idx]
            + term.context_biases[context_idx]
        )
        residual_sum += float(np.sum(targets - estimates))
    return MODEL_DTYPE(residual_sum / term.cooccurrence.nnz)


def _cooccurrence_loss(
    primary_factors: np.ndarray,
    term: _CofactorTerm,
) -> float:
    """Compute one weighted co-occurrence reconstruction loss."""
    if not term.is_active():
        return 0.0
    loss = 0.0
    for row_idx in range(term.cooccurrence.shape[0]):
        start = term.cooccurrence.indptr[row_idx]
        stop = term.cooccurrence.indptr[row_idx + 1]
        context_idx = term.cooccurrence.indices[start:stop]
        if context_idx.size == 0:
            continue
        targets = term.cooccurrence.data[start:stop]
        scores = term.context_factors[context_idx] @ primary_factors[row_idx]
        scores = (
            scores
            + term.primary_biases[row_idx]
            + term.context_biases[context_idx]
            + term.global_bias
        )
        loss += float(np.sum((targets - scores) ** 2))
    return float(term.gamma) * loss


def _refit_term(primary_factors: np.ndarray, term: _CofactorTerm) -> None:
    """Refit every parameter associated with an active context term."""
    if not term.is_active():
        return
    term.context_factors = _context_factor_update(primary_factors, term)
    term.primary_biases = _primary_bias_update(primary_factors, term)
    term.context_biases = _context_bias_update(primary_factors, term)
    term.global_bias = _global_bias_update(primary_factors, term)


class RME:
    """Combine feedback with positive and negative multi-embedding terms."""


    def __init__(
        self,
        user_count,
        item_count,
        K,
        lambda_rate,
        alpha,
        gamma_item_pos,
        gamma_item_neg,
        gamma_user_pos,
        lambda_context_rate=None,
        negative_samples: float = 1.0,
        like_threshold: float = LIKE_THRESHOLD,
        dislike_threshold: float = DISLIKE_THRESHOLD,
        random_state=None,
    ):
        """Validate hyperparameters and initialize model state."""
        self.user_count = int(user_count)
        self.item_count = int(item_count)
        self.K = int(K)
        self.lambda_rate = float(lambda_rate)
        self.alpha = float(alpha)
        self.gamma_item_pos = float(gamma_item_pos)
        self.gamma_item_neg = float(gamma_item_neg)
        self.gamma_user_pos = float(gamma_user_pos)
        self.lambda_context_rate = (
            float(lambda_context_rate)
            if lambda_context_rate is not None
            else self.lambda_rate
        )
        self.negative_samples = float(negative_samples)
        self.like_threshold = float(like_threshold)
        self.dislike_threshold = float(dislike_threshold)
        self.random_state = resolve_seed(random_state)


        if self.K <= 0:
            raise ValueError("K must be > 0.")
        if not np.isfinite(self.lambda_rate) or self.lambda_rate <= 0.0:
            raise ValueError("lambda_rate must be finite and > 0.")
        if not np.isfinite(self.lambda_context_rate) or self.lambda_context_rate <= 0.0:
            raise ValueError("lambda_context_rate must be finite and > 0.")
        if not np.isfinite(self.alpha) or self.alpha < 0.0:
            raise ValueError("alpha must be finite and >= 0.")
        for name, value in (
            ("gamma_item_pos", self.gamma_item_pos),
            ("gamma_item_neg", self.gamma_item_neg),
            ("gamma_user_pos", self.gamma_user_pos),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0.")
        if not np.isfinite(self.negative_samples) or self.negative_samples < 1.0:
            raise ValueError("negative_samples must be finite and >= 1.")


        seed = self.random_state
        self.P = _initialize_factors(
            self.user_count, self.K, np.random.RandomState(seed)
        )
        self.Q = _initialize_factors(
            self.item_count, self.K, np.random.RandomState((seed + 104729) % (2**32))
        )
        self.W = _initialize_factors(
            self.user_count, self.K, np.random.RandomState((seed + 209759) % (2**32))
        )
        self.Zp = _initialize_factors(
            self.item_count, self.K, np.random.RandomState((seed + 314773) % (2**32))
        )
        self.Zn = _initialize_factors(
            self.item_count, self.K, np.random.RandomState((seed + 419713) % (2**32))
        )
        self.user_pos_primary_bias = np.zeros(self.user_count, dtype=MODEL_DTYPE)
        self.user_pos_context_bias = np.zeros(self.user_count, dtype=MODEL_DTYPE)
        self.user_pos_global_bias = MODEL_DTYPE(0.0)
        self.item_pos_primary_bias = np.zeros(self.item_count, dtype=MODEL_DTYPE)
        self.item_pos_context_bias = np.zeros(self.item_count, dtype=MODEL_DTYPE)
        self.item_pos_global_bias = MODEL_DTYPE(0.0)
        self.item_neg_primary_bias = np.zeros(self.item_count, dtype=MODEL_DTYPE)
        self.item_neg_context_bias = np.zeros(self.item_count, dtype=MODEL_DTYPE)
        self.item_neg_global_bias = MODEL_DTYPE(0.0)


    @property
    def shape(self) -> tuple[int, int]:
        """Return the expected user-item matrix shape."""
        return self.user_count, self.item_count


    def fit(
        self,
        Y,
        n_sweeps: int,
        X_item_pos=None,
        X_item_neg=None,
        Y_user_pos=None,
        verbose_every: int = 5,
    ):
        """Fit primary and context factors with alternating updates."""
        n_sweeps, verbose_every = _validate_training_schedule(
            n_sweeps, verbose_every
        )
        L = to_L(Y, threshold=self.like_threshold)
        if L.shape != self.shape:
            raise ValueError(f"Y must be shape {self.shape}.")
        L_T = L.T.tocsr()


        if X_item_pos is None:
            X_item_pos = build_sppmi_matrix(L, negative_samples=self.negative_samples)
        else:
            X_item_pos = to_Y(X_item_pos)
        if X_item_neg is None:
            D = to_D(Y, threshold=self.dislike_threshold)
            X_item_neg = build_sppmi_matrix(D, negative_samples=self.negative_samples)
        else:
            X_item_neg = to_Y(X_item_neg)
        if Y_user_pos is None:
            Y_user_pos = build_sppmi_matrix(L_T, negative_samples=self.negative_samples)
        else:
            Y_user_pos = to_Y(Y_user_pos)

        for name, matrix, expected in (
            ("X_item_pos", X_item_pos, (self.item_count, self.item_count)),
            ("X_item_neg", X_item_neg, (self.item_count, self.item_count)),
            ("Y_user_pos", Y_user_pos, (self.user_count, self.user_count)),
        ):
            if matrix.shape != expected:
                raise ValueError(f"{name} must be shape {expected}.")

        for matrix in (X_item_pos, X_item_neg, Y_user_pos):
            matrix.setdiag(0)
            matrix.eliminate_zeros()
            matrix.sort_indices()


        item_pos_term = _CofactorTerm(
            self.Zp, self.item_pos_primary_bias, self.item_pos_context_bias,
            self.item_pos_global_bias, X_item_pos, X_item_pos.T.tocsr(),
            self.lambda_context_rate,
            self.gamma_item_pos,
        )
        item_neg_term = _CofactorTerm(
            self.Zn, self.item_neg_primary_bias, self.item_neg_context_bias,
            self.item_neg_global_bias, X_item_neg, X_item_neg.T.tocsr(),
            self.lambda_context_rate,
            self.gamma_item_neg,
        )
        user_pos_term = _CofactorTerm(
            self.W, self.user_pos_primary_bias, self.user_pos_context_bias,
            self.user_pos_global_bias, Y_user_pos, Y_user_pos.T.tocsr(),
            self.lambda_context_rate,
            self.gamma_user_pos,
        )


        # Alternate feedback-factor solves with all active context refits.
        for sweep_idx in range(n_sweeps):
            self.P = _primary_factor_update(
                L, self.Q, self.lambda_rate, self.alpha, [user_pos_term],
            )
            self.Q = _primary_factor_update(
                L_T, self.P, self.lambda_rate, self.alpha,
                [item_pos_term, item_neg_term],
            )

            _refit_term(self.P, user_pos_term)
            _refit_term(self.Q, item_pos_term)
            _refit_term(self.Q, item_neg_term)

            if _should_report_sweep(sweep_idx, verbose_every):


                wmf_loss = self._wmf_loss(L)
                item_pos_loss = _cooccurrence_loss(self.Q, item_pos_term)
                item_neg_loss = _cooccurrence_loss(self.Q, item_neg_term)
                user_pos_loss = _cooccurrence_loss(self.P, user_pos_term)
                l2_sum = float(np.sum(self.P**2) + np.sum(self.Q**2))
                active_terms = tuple(
                    term
                    for term in (item_pos_term, item_neg_term, user_pos_term)
                    if term.is_active()
                )
                context_l2_sum = float(sum(
                    np.sum(term.context_factors**2)
                    for term in active_terms
                ))
                context_regularization = float(sum(
                    term.lambda_context_rate
                    * np.sum(term.context_factors**2)
                    for term in active_terms
                ))
                regularization = self.lambda_rate * l2_sum + context_regularization
                total_loss = (
                    wmf_loss + item_pos_loss + item_neg_loss + user_pos_loss
                    + regularization
                )
                print(
                    f"[Sweep {sweep_idx + 1}/{n_sweeps}] "
                    f"WMF_LOSS: {wmf_loss:.6f} "
                    f"ITEM_POS_LOSS: {item_pos_loss:.6f} "
                    f"ITEM_NEG_LOSS: {item_neg_loss:.6f} "
                    f"USER_POS_LOSS: {user_pos_loss:.6f} "
                    f"L2_SUM: {l2_sum:.6f} "
                    f"CONTEXT_L2_SUM: {context_l2_sum:.6f} "
                    f"REG: {regularization:.6f} "
                    f"TOTAL: {total_loss:.6f}"
                )


        self.Zp = item_pos_term.context_factors
        self.item_pos_primary_bias = item_pos_term.primary_biases
        self.item_pos_context_bias = item_pos_term.context_biases
        self.item_pos_global_bias = item_pos_term.global_bias
        self.Zn = item_neg_term.context_factors
        self.item_neg_primary_bias = item_neg_term.primary_biases
        self.item_neg_context_bias = item_neg_term.context_biases
        self.item_neg_global_bias = item_neg_term.global_bias
        self.W = user_pos_term.context_factors
        self.user_pos_primary_bias = user_pos_term.primary_biases
        self.user_pos_context_bias = user_pos_term.context_biases
        self.user_pos_global_bias = user_pos_term.global_bias
        return self


    def _wmf_loss(self, L: sparse.csr_matrix) -> float:
        """Compute the model's unregularized weighted feedback loss."""
        user_factors = np.asarray(self.P, dtype=np.float64)
        item_factors = np.asarray(self.Q, dtype=np.float64)

        user_gram = user_factors.T @ user_factors
        item_gram = item_factors.T @ item_factors
        loss = float(np.sum(user_gram * item_gram))
        for user_idx in range(L.shape[0]):
            start = L.indptr[user_idx]
            stop = L.indptr[user_idx + 1]
            item_idx = L.indices[start:stop]
            if item_idx.size == 0:
                continue
            scores = item_factors[item_idx] @ user_factors[user_idx]
            loss += float(
                np.sum((1.0 + self.alpha) * (1.0 - scores) ** 2 - scores**2)
            )
        return max(loss, 0.0)


    def score_user(self, user_idx: int) -> np.ndarray:
        """Score every item for one user."""
        user_idx = int(user_idx)
        if user_idx < 0 or user_idx >= self.user_count:
            raise IndexError("user_idx is out of range.")
        return self.Q @ self.P[user_idx]
