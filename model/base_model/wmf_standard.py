# This file implements standard implicit-feedback weighted matrix factorization with ALS.

from numbers import Integral

import numpy as np
from scipy import sparse

from util.feedback import LIKE_THRESHOLD, to_L
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
def _als_factor_update(
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


class WMF_Standard:
    # Inputs:
    # - user_count: Number of users represented by the model.
    # - item_count: Number of items represented by the model.
    # - K: Latent factor dimension.
    # - lambda_rate: L2 regularization coefficient.
    # - alpha: Extra confidence assigned to positive L entries.
    # - threshold: Minimum exclusive rating interpreted as positive feedback.
    # - random_state: Optional seed for factor initialization.
    # Output: Initialized WMF model with random user and item factor matrices.
    def __init__(
        self,
        user_count,
        item_count,
        K,
        lambda_rate,
        alpha,
        threshold: float = LIKE_THRESHOLD,
        random_state=None,
    ):
        self.user_count = int(user_count)
        self.item_count = int(item_count)
        self.K = int(K)
        self.lambda_rate = float(lambda_rate)
        self.alpha = float(alpha)
        self.threshold = float(threshold)
        self.random_state = resolve_seed(random_state)

        # Step 1: Validate model hyperparameters.
        if self.K <= 0:
            raise ValueError("K must be > 0.")
        if not np.isfinite(self.lambda_rate) or self.lambda_rate <= 0.0:
            raise ValueError("lambda_rate must be finite and > 0.")
        if not np.isfinite(self.alpha) or self.alpha < 0.0:
            raise ValueError("alpha must be finite and >= 0.")

        # Step 2: Use independent deterministic streams so padding extra users
        # cannot change the initial item factors.
        user_rng = np.random.RandomState(self.random_state)
        item_rng = np.random.RandomState((self.random_state + 104729) % (2**32))
        self.P = _initialize_factors(self.user_count, self.K, user_rng)
        self.Q = _initialize_factors(self.item_count, self.K, item_rng)

    # Inputs: None; reads model dimensions from this instance.
    # Output: A (user_count, item_count) tuple.
    @property
    def shape(self) -> tuple[int, int]:
        return self.user_count, self.item_count

    # Inputs:
    # - Y: Dense or sparse user-item interaction matrix.
    # - n_sweeps: Number of full user/item ALS update sweeps.
    # - verbose_every: Positive interval controlling loss reporting.
    # Output: This fitted WMF_Standard instance.
    def fit(
        self,
        Y,
        n_sweeps: int,
        verbose_every: int = 5,
    ):
        n_sweeps, verbose_every = _validate_training_schedule(
            n_sweeps, verbose_every
        )
        # Step 1: Validate interactions and binarize positive preferences.
        L = to_L(Y, threshold=self.threshold)
        if L.shape != self.shape:
            raise ValueError(f"Y must be shape {self.shape}.")
        L_T = L.T.tocsr()

        # Step 2: Alternate user-factor and item-factor least-squares updates.
        for sweep_idx in range(n_sweeps):
            # Step 2.1: Update all user factors while holding item factors fixed.
            self.P = _als_factor_update(
                L,
                self.Q,
                self.lambda_rate,
                self.alpha,
            )
            # Step 2.2: Update all item factors while holding user factors fixed.
            self.Q = _als_factor_update(
                L_T,
                self.P,
                self.lambda_rate,
                self.alpha,
            )

            if _should_report_sweep(sweep_idx, verbose_every):
                # Step 2.3: Calculate WMF, regularization, and total losses.
                wmf_loss = _wmf_loss(
                    self.P,
                    self.Q,
                    L,
                    self.alpha,
                )
                l2_sum = float(
                    np.sum(self.P * self.P)
                    + np.sum(self.Q * self.Q)
                )
                regularization = self.lambda_rate * l2_sum
                total_loss = wmf_loss + regularization
                print(
                    f"[Sweep {sweep_idx + 1}/{n_sweeps}] "
                    f"WMF_LOSS: {wmf_loss:.6f} "
                    f"L2_SUM: {l2_sum:.6f} "
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
