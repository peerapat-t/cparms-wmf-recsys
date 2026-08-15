"""Train the CoFactor recommender on feedback and item co-occurrence."""

from numbers import Integral

import numpy as np
from scipy import sparse

from util.feedback import LIKE_THRESHOLD, to_L, to_Y
from util.seed_config import resolve_seed


MODEL_DTYPE = np.float64


def build_item_sppmi_matrix(
    Y,
    threshold: float = LIKE_THRESHOLD,
    negative_samples: float = 1.0,
) -> sparse.csr_matrix:
    """Build a symmetric shifted positive-PMI item co-occurrence matrix."""
    negative_samples = float(negative_samples)
    if not np.isfinite(negative_samples) or negative_samples < 1.0:
        raise ValueError("negative_samples must be finite and >= 1.")


    L = to_L(Y, threshold=threshold)
    item_count = L.shape[1]
    cooccurrence = (L.T @ L).tocsr().astype(
        MODEL_DTYPE,
        copy=False,
    )
    cooccurrence.setdiag(0)
    cooccurrence.eliminate_zeros()
    cooccurrence.sort_indices()

    if cooccurrence.nnz == 0:
        return sparse.csr_matrix(
            (item_count, item_count),
            dtype=MODEL_DTYPE,
        )


    row_count = np.asarray(
        cooccurrence.sum(axis=1)
    ).reshape(-1).astype(np.float64, copy=False)
    pair_count = float(cooccurrence.data.sum())
    if pair_count <= 0.0:
        return sparse.csr_matrix(
            (item_count, item_count),
            dtype=MODEL_DTYPE,
        )

    sppmi = cooccurrence.copy().astype(np.float64, copy=False)
    for item_idx in range(item_count):
        start = sppmi.indptr[item_idx]
        stop = sppmi.indptr[item_idx + 1]
        if start == stop:
            continue
        context_idx = sppmi.indices[start:stop]
        counts = sppmi.data[start:stop]
        denominator = row_count[item_idx] * row_count[context_idx]
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


def _wmf_factor_update(
    positive_feedback: sparse.csr_matrix,
    fixed_factors: np.ndarray,
    lambda_rate: float,
    alpha: float,
) -> np.ndarray:
    """Solve one implicit-feedback ALS update for every matrix row."""

    row_count = positive_feedback.shape[0]
    latent_dim = fixed_factors.shape[1]
    identity = np.eye(latent_dim, dtype=MODEL_DTYPE)
    base = (
        fixed_factors.T @ fixed_factors
        + MODEL_DTYPE(lambda_rate) * identity
    )
    updated = np.zeros((row_count, latent_dim), dtype=MODEL_DTYPE)


    for row_idx in range(row_count):
        start = positive_feedback.indptr[row_idx]
        stop = positive_feedback.indptr[row_idx + 1]
        positive_idx = positive_feedback.indices[start:stop]

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


        updated[row_idx] = np.linalg.solve(
            system_matrix,
            right_hand_side,
        )
    return updated


def _wmf_loss(
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    positive_feedback: sparse.csr_matrix,
    alpha: float,
) -> float:
    """Compute the unregularized weighted reconstruction loss."""
    user_factors_64 = np.asarray(user_factors, dtype=np.float64)
    item_factors_64 = np.asarray(item_factors, dtype=np.float64)


    user_gram = user_factors_64.T @ user_factors_64
    item_gram = item_factors_64.T @ item_factors_64
    loss = float(np.sum(user_gram * item_gram))


    for user_idx in range(positive_feedback.shape[0]):
        start = positive_feedback.indptr[user_idx]
        stop = positive_feedback.indptr[user_idx + 1]
        item_idx = positive_feedback.indices[start:stop]
        if item_idx.size == 0:
            continue
        scores = item_factors_64[item_idx] @ user_factors_64[user_idx]
        loss += float(
            np.sum(
                (1.0 + alpha) * (1.0 - scores) ** 2
                - scores**2
            )
        )
    return max(loss, 0.0)


def _item_factor_update(
    L_T: sparse.csr_matrix,
    user_factors: np.ndarray,
    context_factors: np.ndarray,
    item_biases: np.ndarray,
    context_biases: np.ndarray,
    global_bias: float,
    cooccurrence: sparse.csr_matrix,
    lambda_rate: float,
    alpha: float,
    gamma: float,
) -> np.ndarray:
    """Update item factors from feedback and co-occurrence objectives."""

    item_count = L_T.shape[0]
    latent_dim = user_factors.shape[1]
    identity = np.eye(latent_dim, dtype=MODEL_DTYPE)
    base = (
        user_factors.T @ user_factors
        + MODEL_DTYPE(lambda_rate) * identity
    )
    confidence_delta = MODEL_DTYPE(alpha)
    observed_confidence = MODEL_DTYPE(1.0 + alpha)
    cofactor_weight = MODEL_DTYPE(gamma)
    updated = np.zeros((item_count, latent_dim), dtype=MODEL_DTYPE)


    for item_idx in range(item_count):
        start = L_T.indptr[item_idx]
        stop = L_T.indptr[item_idx + 1]
        user_idx = L_T.indices[start:stop]
        system_matrix = base.copy()
        right_hand_side = np.zeros(latent_dim, dtype=MODEL_DTYPE)

        if user_idx.size:

            positive_factors = user_factors[user_idx]
            system_matrix += confidence_delta * (
                positive_factors.T @ positive_factors
            )
            right_hand_side += observed_confidence * positive_factors.sum(axis=0)

        if gamma > 0.0:

            start = cooccurrence.indptr[item_idx]
            stop = cooccurrence.indptr[item_idx + 1]
            context_idx = cooccurrence.indices[start:stop]
            if context_idx.size:
                targets = (
                    cooccurrence.data[start:stop]
                    - item_biases[item_idx]
                    - context_biases[context_idx]
                    - global_bias
                )
                context = context_factors[context_idx]
                system_matrix += cofactor_weight * (context.T @ context)
                right_hand_side += cofactor_weight * (targets @ context)


        updated[item_idx] = np.linalg.solve(
            system_matrix,
            right_hand_side,
        )
    return updated


def _context_factor_update(
    item_factors: np.ndarray,
    item_biases: np.ndarray,
    context_biases: np.ndarray,
    global_bias: float,
    cooccurrence_t: sparse.csr_matrix,
    lambda_context_rate: float,
    gamma: float,
) -> np.ndarray:
    """Update context factors for the co-occurrence objective."""

    item_count, latent_dim = item_factors.shape
    identity = np.eye(latent_dim, dtype=MODEL_DTYPE)
    regularization = MODEL_DTYPE(lambda_context_rate) * identity
    cofactor_weight = MODEL_DTYPE(gamma)
    updated = np.zeros((item_count, latent_dim), dtype=MODEL_DTYPE)


    for context_idx in range(item_count):
        start = cooccurrence_t.indptr[context_idx]
        stop = cooccurrence_t.indptr[context_idx + 1]
        item_idx = cooccurrence_t.indices[start:stop]

        system_matrix = regularization.copy()
        right_hand_side = np.zeros(latent_dim, dtype=MODEL_DTYPE)

        if item_idx.size:
            targets = (
                cooccurrence_t.data[start:stop]
                - item_biases[item_idx]
                - context_biases[context_idx]
                - global_bias
            )
            item = item_factors[item_idx]
            system_matrix += cofactor_weight * (item.T @ item)
            right_hand_side += cofactor_weight * (targets @ item)

        updated[context_idx] = np.linalg.solve(
            system_matrix,
            right_hand_side,
        )
    return updated


def _item_bias_update(
    item_factors: np.ndarray,
    context_factors: np.ndarray,
    cooccurrence: sparse.csr_matrix,
    context_biases: np.ndarray,
    global_bias: float,
) -> np.ndarray:
    """Update per-item biases for observed co-occurrences."""
    item_count = cooccurrence.shape[0]
    biases = np.zeros(item_count, dtype=MODEL_DTYPE)
    if cooccurrence.nnz == 0:
        return biases

    for item_idx in range(item_count):
        start = cooccurrence.indptr[item_idx]
        stop = cooccurrence.indptr[item_idx + 1]
        context_idx = cooccurrence.indices[start:stop]
        if context_idx.size == 0:
            continue

        targets = cooccurrence.data[start:stop]
        scores = context_factors[context_idx] @ item_factors[item_idx]
        residual = (
            targets
            - scores
            - context_biases[context_idx]
            - global_bias
        )
        biases[item_idx] = residual.mean()
    return biases


def _context_bias_update(
    item_factors: np.ndarray,
    context_factors: np.ndarray,
    cooccurrence_t: sparse.csr_matrix,
    item_biases: np.ndarray,
    global_bias: float,
) -> np.ndarray:
    """Update per-context biases for observed co-occurrences."""
    context_count = cooccurrence_t.shape[0]
    biases = np.zeros(context_count, dtype=MODEL_DTYPE)
    if cooccurrence_t.nnz == 0:
        return biases

    for context_idx in range(context_count):
        start = cooccurrence_t.indptr[context_idx]
        stop = cooccurrence_t.indptr[context_idx + 1]
        item_idx = cooccurrence_t.indices[start:stop]
        if item_idx.size == 0:
            continue

        targets = cooccurrence_t.data[start:stop]
        scores = item_factors[item_idx] @ context_factors[context_idx]
        residual = targets - scores - item_biases[item_idx] - global_bias
        biases[context_idx] = residual.mean()
    return biases


def _global_bias_update(
    item_factors: np.ndarray,
    context_factors: np.ndarray,
    item_biases: np.ndarray,
    context_biases: np.ndarray,
    cooccurrence: sparse.csr_matrix,
) -> np.float64:
    """Update the global co-occurrence intercept."""
    if cooccurrence.nnz == 0:
        return MODEL_DTYPE(0.0)
    residual_sum = 0.0
    for item_idx in range(cooccurrence.shape[0]):
        start = cooccurrence.indptr[item_idx]
        stop = cooccurrence.indptr[item_idx + 1]
        context_idx = cooccurrence.indices[start:stop]
        if context_idx.size == 0:
            continue
        targets = cooccurrence.data[start:stop]
        estimates = (
            context_factors[context_idx] @ item_factors[item_idx]
            + item_biases[item_idx]
            + context_biases[context_idx]
        )
        residual_sum += float(np.sum(targets - estimates))
    return MODEL_DTYPE(residual_sum / cooccurrence.nnz)


def _cooccurrence_loss(
    item_factors: np.ndarray,
    context_factors: np.ndarray,
    item_biases: np.ndarray,
    context_biases: np.ndarray,
    global_bias: float,
    cooccurrence: sparse.csr_matrix,
    gamma: float,
) -> float:
    """Compute squared error on observed item-context values."""
    if gamma <= 0.0 or cooccurrence.nnz == 0:
        return 0.0

    loss = 0.0
    for item_idx in range(cooccurrence.shape[0]):
        start = cooccurrence.indptr[item_idx]
        stop = cooccurrence.indptr[item_idx + 1]
        context_idx = cooccurrence.indices[start:stop]
        if context_idx.size == 0:
            continue
        targets = cooccurrence.data[start:stop]
        scores = context_factors[context_idx] @ item_factors[item_idx]
        scores = (
            scores
            + item_biases[item_idx]
            + context_biases[context_idx]
            + global_bias
        )
        loss += float(np.sum((targets - scores) ** 2))
    return float(gamma) * loss


class CoFactor:
    """Jointly factorize implicit feedback and shifted-PMI co-occurrence."""


    def __init__(
        self,
        user_count,
        item_count,
        K,
        lambda_rate,
        alpha,
        gamma,
        lambda_context_rate=None,
        negative_samples: float = 1.0,
        threshold: float = LIKE_THRESHOLD,
        random_state=None,
    ):
        """Validate hyperparameters and initialize model state."""
        self.user_count = int(user_count)
        self.item_count = int(item_count)
        self.K = int(K)
        self.lambda_rate = float(lambda_rate)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.lambda_context_rate = (
            float(lambda_context_rate)
            if lambda_context_rate is not None
            else (
                self.gamma * self.lambda_rate
                if self.gamma > 0.0
                else self.lambda_rate
            )
        )
        self.negative_samples = float(negative_samples)
        self.threshold = float(threshold)
        self.random_state = resolve_seed(random_state)


        if self.K <= 0:
            raise ValueError("K must be > 0.")
        if not np.isfinite(self.lambda_rate) or self.lambda_rate <= 0.0:
            raise ValueError("lambda_rate must be finite and > 0.")
        if (
            not np.isfinite(self.lambda_context_rate)
            or self.lambda_context_rate <= 0.0
        ):
            raise ValueError("lambda_context_rate must be finite and > 0.")
        if not np.isfinite(self.alpha) or self.alpha < 0.0:
            raise ValueError("alpha must be finite and >= 0.")
        if not np.isfinite(self.gamma) or self.gamma < 0.0:
            raise ValueError("gamma must be finite and >= 0.")
        if not np.isfinite(self.negative_samples) or self.negative_samples < 1.0:
            raise ValueError("negative_samples must be finite and >= 1.")


        user_rng = np.random.RandomState(self.random_state)
        item_rng = np.random.RandomState((self.random_state + 104729) % (2**32))
        context_rng = np.random.RandomState(
            (self.random_state + 209759) % (2**32)
        )
        self.P = _initialize_factors(
            self.user_count,
            self.K,
            user_rng,
        )
        self.Q = _initialize_factors(
            self.item_count,
            self.K,
            item_rng,
        )
        self.Z = _initialize_factors(
            self.item_count,
            self.K,
            context_rng,
        )
        self.item_biases = np.zeros(self.item_count, dtype=MODEL_DTYPE)
        self.context_biases = np.zeros(self.item_count, dtype=MODEL_DTYPE)
        self.global_bias = MODEL_DTYPE(0.0)


    @property
    def shape(self) -> tuple[int, int]:
        """Return the expected user-item matrix shape."""
        return self.user_count, self.item_count


    def fit(
        self,
        Y,
        M=None,
        n_sweeps: int = 10,
        verbose_every: int = 5,
    ):
        """Fit feedback and context factors with alternating updates."""
        n_sweeps, verbose_every = _validate_training_schedule(
            n_sweeps, verbose_every
        )
        L = to_L(Y, threshold=self.threshold)
        if L.shape != self.shape:
            raise ValueError(f"Y must be shape {self.shape}.")
        L_T = L.T.tocsr()


        if self.gamma == 0.0:
            cooccurrence = sparse.csr_matrix(
                (self.item_count, self.item_count),
                dtype=MODEL_DTYPE,
            )
        elif M is None:
            cooccurrence = build_item_sppmi_matrix(
                Y,
                threshold=self.threshold,
                negative_samples=self.negative_samples,
            )
        else:
            cooccurrence = to_Y(M)
            expected_shape = (self.item_count, self.item_count)
            if cooccurrence.shape != expected_shape:
                raise ValueError(f"M must be shape {expected_shape}.")
        cooccurrence.setdiag(0)
        cooccurrence.eliminate_zeros()
        cooccurrence.sort_indices()
        cooccurrence_t = cooccurrence.T.tocsr()
        has_cofactor = (
            self.gamma > 0.0
            and cooccurrence.nnz > 0
        )
        if not has_cofactor:


            self.item_biases.fill(0.0)
            self.context_biases.fill(0.0)
            self.global_bias = MODEL_DTYPE(0.0)


        # Alternate feedback, co-occurrence factor, and bias updates.
        for sweep_idx in range(n_sweeps):
            self.P = _wmf_factor_update(
                L,
                self.Q,
                self.lambda_rate,
                self.alpha,
            )
            self.Q = _item_factor_update(
                L_T,
                self.P,
                self.Z,
                self.item_biases,
                self.context_biases,
                self.global_bias,
                cooccurrence,
                self.lambda_rate,
                self.alpha,
                self.gamma,
            )

            if has_cofactor:

                self.Z = _context_factor_update(
                    self.Q,
                    self.item_biases,
                    self.context_biases,
                    self.global_bias,
                    cooccurrence_t,
                    self.lambda_context_rate,
                    self.gamma,
                )
                self.item_biases = _item_bias_update(
                    self.Q,
                    self.Z,
                    cooccurrence,
                    self.context_biases,
                    self.global_bias,
                )
                self.context_biases = _context_bias_update(
                    self.Q,
                    self.Z,
                    cooccurrence_t,
                    self.item_biases,
                    self.global_bias,
                )
                self.global_bias = _global_bias_update(
                    self.Q,
                    self.Z,
                    self.item_biases,
                    self.context_biases,
                    cooccurrence,
                )

            if _should_report_sweep(sweep_idx, verbose_every):

                wmf_loss = _wmf_loss(
                    self.P,
                    self.Q,
                    L,
                    self.alpha,
                )
                cofactor_loss = _cooccurrence_loss(
                    self.Q,
                    self.Z,
                    self.item_biases,
                    self.context_biases,
                    self.global_bias,
                    cooccurrence,
                    self.gamma,
                )
                l2_sum = float(
                    np.sum(self.P * self.P)
                    + np.sum(self.Q * self.Q)
                )
                context_l2_sum = (
                    float(np.sum(self.Z * self.Z))
                    if has_cofactor
                    else 0.0
                )
                regularization = (
                    self.lambda_rate * l2_sum
                    + self.lambda_context_rate * context_l2_sum
                )
                total_loss = wmf_loss + cofactor_loss + regularization
                print(
                    f"[Sweep {sweep_idx + 1}/{n_sweeps}] "
                    f"WMF_LOSS: {wmf_loss:.6f} "
                    f"COFACTOR_LOSS: {cofactor_loss:.6f} "
                    f"L2_SUM: {l2_sum:.6f} "
                    f"CONTEXT_L2_SUM: {context_l2_sum:.6f} "
                    f"REG: {regularization:.6f} "
                    f"TOTAL: {total_loss:.6f}"
                )
        return self


    def score_user(self, user_idx: int) -> np.ndarray:
        """Score every item for one user."""
        user_idx = int(user_idx)
        if user_idx < 0 or user_idx >= self.user_count:
            raise IndexError("user_idx is out of range.")
        return self.Q @ self.P[user_idx]
