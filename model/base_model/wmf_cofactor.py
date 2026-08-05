# This file implements CoFactor-style WMF regularized by item co-occurrence.

from numbers import Integral

import numpy as np
from scipy import sparse

from model.signal_gen.generator_sppmi import build_item_sppmi_matrix
from util.feedback import LIKE_THRESHOLD, to_L, to_Y
from util.seed_config import resolve_seed


MODEL_DTYPE = np.float32


# Inputs:
# - n_sweeps: Number of full ALS update sweeps.
# - verbose_every: Interval controlling loss reporting.
# Output: The validated (n_sweeps, verbose_every) pair as integers.
def _validate_training_schedule(n_sweeps, verbose_every) -> tuple[int, int]:
    for name, value in (("n_sweeps", n_sweeps), ("verbose_every", verbose_every)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    return int(n_sweeps), int(verbose_every)


# Inputs:
# - sweep_idx: Zero-based index of the current sweep.
# - verbose_every: Positive interval controlling loss reporting.
# Output: True when this sweep should print its losses.
def _should_report_sweep(sweep_idx: int, verbose_every: int) -> bool:
    return sweep_idx == 0 or (sweep_idx + 1) % verbose_every == 0


# Inputs:
# - row_count: Number of latent-factor rows to create.
# - latent_dim: Latent factor dimension.
# - rng: Seeded random generator for reproducible initialization.
# Output: Small-random-normal latent factor matrix.
def _initialize_factors(
    row_count: int,
    latent_dim: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    return rng.normal(
        0.0,
        0.01,
        size=(int(row_count), int(latent_dim)),
    ).astype(MODEL_DTYPE)


# Inputs:
# - positive_feedback: Binary CSR positive preferences for the rows being updated.
# - fixed_factors: Opposite-side latent factors held fixed.
# - lambda_rate: L2 regularization coefficient.
# - alpha: Extra confidence assigned to positive L entries.
# Output: Updated latent factor matrix for all rows.
def _wmf_factor_update(
    positive_feedback: sparse.csr_matrix,
    fixed_factors: np.ndarray,
    lambda_rate: float,
    alpha: float,
) -> np.ndarray:
    # Step 1: Build the shared unobserved-feedback and regularization matrix.
    row_count = positive_feedback.shape[0]
    latent_dim = fixed_factors.shape[1]
    identity = np.eye(latent_dim, dtype=MODEL_DTYPE)
    base = (
        fixed_factors.T @ fixed_factors
        + MODEL_DTYPE(lambda_rate) * identity
    )
    updated = np.zeros((row_count, latent_dim), dtype=MODEL_DTYPE)

    # Step 2: Construct and solve one weighted normal equation per row.
    for row_idx in range(row_count):
        start = positive_feedback.indptr[row_idx]
        stop = positive_feedback.indptr[row_idx + 1]
        positive_idx = positive_feedback.indices[start:stop]

        system_matrix = base.copy()
        right_hand_side = np.zeros(latent_dim, dtype=MODEL_DTYPE)
        if positive_idx.size:
            # Step 2.1: Add confidence contributions from positive preferences.
            positive_factors = fixed_factors[positive_idx]
            system_matrix += MODEL_DTYPE(alpha) * (
                positive_factors.T @ positive_factors
            )
            right_hand_side += MODEL_DTYPE(1.0 + alpha) * (
                positive_factors.sum(axis=0)
            )

        # Step 2.2: Solve the row-specific linear system.
        updated[row_idx] = np.linalg.solve(
            system_matrix,
            right_hand_side,
        )
    return updated


# Inputs:
# - user_factors: Current user latent-factor matrix.
# - item_factors: Current item latent-factor matrix.
# - positive_feedback: Binary CSR user-item positive preferences.
# - alpha: Extra confidence assigned to positive L entries.
# Output: Confidence-weighted squared WMF loss over all user-item pairs.
def _wmf_loss(
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    positive_feedback: sparse.csr_matrix,
    alpha: float,
) -> float:
    # Step 1: Score every unobserved pair at once through the factor Gram matrices.
    user_gram = user_factors.T @ user_factors
    item_gram = item_factors.T @ item_factors
    loss = float(np.sum(user_gram * item_gram))

    # Step 2: Correct each observed pair to its higher confidence weight.
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


# Inputs:
# - L_T: Binary CSR item-user positive preferences.
# - user_factors: Current user latent factors.
# - context_factors: Current item-context latent factors.
# - item_biases: Current item-side co-occurrence biases.
# - context_biases: Current context-side co-occurrence biases.
# - global_bias: Current global co-occurrence bias.
# - cooccurrence: CSR item-item SPPMI target matrix.
# - lambda_rate: Item-factor L2 regularization coefficient.
# - alpha: Extra confidence assigned to positive L entries.
# - gamma: Weight assigned to the co-occurrence term.
# Output: Updated item factor matrix.
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
    # Step 1: Build the shared WMF part of the normal equation.
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

    # Step 2: Solve one combined WMF/co-occurrence system per item.
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


# Inputs:
# - item_factors: Current item latent factors.
# - item_biases: Current item-side co-occurrence biases.
# - context_biases: Current context-side co-occurrence biases.
# - global_bias: Current global co-occurrence bias.
# - cooccurrence_t: CSR transposed SPPMI target matrix.
# - lambda_context_rate: Context-factor L2 regularization coefficient.
# - gamma: Weight assigned to the co-occurrence term.
# Output: Updated context factor matrix.
def _context_factor_update(
    item_factors: np.ndarray,
    item_biases: np.ndarray,
    context_biases: np.ndarray,
    global_bias: float,
    cooccurrence_t: sparse.csr_matrix,
    lambda_context_rate: float,
    gamma: float,
) -> np.ndarray:
    item_count, latent_dim = item_factors.shape
    identity = np.eye(latent_dim, dtype=MODEL_DTYPE)
    regularization = MODEL_DTYPE(lambda_context_rate) * identity
    cofactor_weight = MODEL_DTYPE(gamma)
    updated = np.zeros((item_count, latent_dim), dtype=MODEL_DTYPE)

    # Step 1: Solve one weighted co-occurrence system per context item.
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


# Inputs:
# - item_factors: Current item latent factors.
# - context_factors: Current context latent factors.
# - cooccurrence: CSR SPPMI target matrix.
# - context_biases: Current context-side co-occurrence biases.
# - global_bias: Current global co-occurrence bias.
# Output: Updated unregularized item-side co-occurrence biases.
def _item_bias_update(
    item_factors: np.ndarray,
    context_factors: np.ndarray,
    cooccurrence: sparse.csr_matrix,
    context_biases: np.ndarray,
    global_bias: float,
) -> np.ndarray:
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


# Inputs:
# - item_factors: Current item latent factors.
# - context_factors: Current context latent factors.
# - cooccurrence_t: CSR transposed SPPMI target matrix.
# - item_biases: Current item-side co-occurrence biases.
# - global_bias: Current global co-occurrence bias.
# Output: Updated unregularized context-side co-occurrence biases.
def _context_bias_update(
    item_factors: np.ndarray,
    context_factors: np.ndarray,
    cooccurrence_t: sparse.csr_matrix,
    item_biases: np.ndarray,
    global_bias: float,
) -> np.ndarray:
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
) -> np.float32:
    """Update the unregularized global bias from all SPPMI residuals."""
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


# Inputs:
# - item_factors: Current item latent factors.
# - context_factors: Current context latent factors.
# - item_biases: Current item-side co-occurrence biases.
# - context_biases: Current context-side co-occurrence biases.
# - global_bias: Current global co-occurrence bias.
# - cooccurrence: CSR SPPMI matrix.
# - gamma: Weight assigned to the co-occurrence term.
# Output: Gamma-weighted item co-occurrence loss.
def _cooccurrence_loss(
    item_factors: np.ndarray,
    context_factors: np.ndarray,
    item_biases: np.ndarray,
    context_biases: np.ndarray,
    global_bias: float,
    cooccurrence: sparse.csr_matrix,
    gamma: float,
) -> float:
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


class WMF_Cofactor:
    # Inputs:
    # - user_count: Number of users represented by the model.
    # - item_count: Number of items represented by the model.
    # - K: Latent factor dimension.
    # - lambda_rate: L2 regularization coefficient for user and item factors.
    # - alpha: Extra confidence assigned to positive L entries.
    # - gamma: Item co-occurrence weight, equal to 1 / relative WMF scale.
    # - lambda_context_rate: Optional context L2 penalty. Defaults to
    #   gamma * lambda_rate.
    # - negative_samples: Shift k for SPPMI = max(PMI - log(k), 0).
    # - threshold: Minimum exclusive rating interpreted as positive feedback.
    # - random_state: Optional seed for factor initialization.
    # Output: Initialized CoFactor-regularized WMF model.
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

        # Step 1: Validate model hyperparameters.
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
        # Step 2: Use independent deterministic streams so padding extra users
        # cannot change the initial item or context factors.
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

    # Inputs: None; reads model dimensions from this instance.
    # Output: A (user_count, item_count) tuple.
    @property
    def shape(self) -> tuple[int, int]:
        return self.user_count, self.item_count

    # Inputs:
    # - Y: Dense or sparse user-item interaction matrix.
    # - M: Optional item-item SPPMI/co-occurrence matrix. If omitted, built from Y.
    # - n_sweeps: Number of full ALS update sweeps.
    # - verbose_every: Positive interval controlling loss reporting.
    # Output: This fitted WMF_Cofactor instance.
    def fit(
        self,
        Y,
        M=None,
        n_sweeps: int = 10,
        verbose_every: int = 5,
    ):
        n_sweeps, verbose_every = _validate_training_schedule(
            n_sweeps, verbose_every
        )
        # Step 1: Validate interactions and prepare the user-item WMF target.
        L = to_L(Y, threshold=self.threshold)
        if L.shape != self.shape:
            raise ValueError(f"Y must be shape {self.shape}.")
        L_T = L.T.tocsr()

        # Step 2: Build or validate the item-item co-occurrence target.
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
        # Without a usable co-occurrence target the model degenerates to plain WMF,
        # so the context factors and biases stay frozen at zero and contribute
        # nothing to the item update.
        has_cofactor = (
            self.gamma > 0.0
            and cooccurrence.nnz > 0
        )
        if not has_cofactor:
            self.item_biases.fill(0.0)
            self.context_biases.fill(0.0)
            self.global_bias = MODEL_DTYPE(0.0)

        # Step 3: Alternate user, item, and context updates.
        for sweep_idx in range(n_sweeps):
            # Step 3.1: Update all user factors while holding item factors fixed.
            self.P = _wmf_factor_update(
                L,
                self.Q,
                self.lambda_rate,
                self.alpha,
            )
            # Step 3.2: Update item factors from both interactions and co-occurrence.
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
                # Step 3.3: Refit the context factors and biases to the SPPMI targets.
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
                # Step 3.4: Calculate WMF, CoFactor, and regularization losses.
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

    # Inputs:
    # - user_idx: Zero-based user index to score.
    # Output: Dense item-score vector from the user's factors and all item factors.
    def score_user(self, user_idx: int) -> np.ndarray:
        user_idx = int(user_idx)
        if user_idx < 0 or user_idx >= self.user_count:
            raise IndexError("user_idx is out of range.")
        return self.Q @ self.P[user_idx]
