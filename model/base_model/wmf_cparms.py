# This file implements signal-regularized weighted matrix factorization using CPARMS scores.

from numbers import Integral

import numpy as np
from scipy import sparse

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
# - user_factors: Current user latent-factor matrix.
# - item_factors: Current item latent-factor matrix.
# - positive_feedback: Binary CSR user-item positive preferences.
# - alpha: Extra confidence assigned to positive L entries.
# Output: Confidence-weighted squared reconstruction loss over all user-item pairs.
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
# - L: Binary CSR positive preferences for the rows being updated.
# - signal: CSR CPARMS target signal for the same rows.
# - fixed_factors: Opposite-side latent factors held fixed.
# - lambda_rate: L2 regularization coefficient.
# - alpha: Extra confidence assigned to positive L entries.
# - gamma: Weight assigned to signal reconstruction.
# Output: Updated latent factor matrix for all rows.
def _als_factor_update(
    L: sparse.csr_matrix,
    signal: sparse.csr_matrix,
    fixed_factors: np.ndarray,
    lambda_rate: float,
    alpha: float,
    gamma: float,
) -> np.ndarray:
    # Step 1: Build the shared unobserved-feedback and regularization matrix.
    row_count = L.shape[0]
    latent_dim = fixed_factors.shape[1]
    identity = np.eye(latent_dim, dtype=MODEL_DTYPE)
    base = (
        fixed_factors.T @ fixed_factors
        + MODEL_DTYPE(lambda_rate) * identity
    )
    updated = np.zeros((row_count, latent_dim), dtype=MODEL_DTYPE)

    # Step 2: Construct and solve one weighted normal equation per row.
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
            # Step 2.1: Add confidence contributions from positive preferences.
            positive_factors = fixed_factors[positive_idx]
            system_matrix += MODEL_DTYPE(alpha) * (
                positive_factors.T @ positive_factors
            )
            right_hand_side += MODEL_DTYPE(1.0 + alpha) * (
                positive_factors.sum(axis=0)
            )

        if gamma > 0.0 and signal_idx.size:
            # Step 2.2: Add regularization toward the CPARMS signal values.
            signal_factors = fixed_factors[signal_idx]
            system_matrix += MODEL_DTYPE(gamma) * (
                signal_factors.T @ signal_factors
            )
            right_hand_side += MODEL_DTYPE(gamma) * (
                signal_values @ signal_factors
            )

        # Step 2.3: Solve the row-specific linear system.
        updated[row_idx] = np.linalg.solve(
            system_matrix,
            right_hand_side,
        )

    return updated


# Inputs:
# - user_factors: Current user latent-factor matrix.
# - item_factors: Current item latent-factor matrix.
# - signal: Sparse user-item CPARMS signal matrix.
# - gamma: Signal reconstruction weight.
# Output: Gamma-weighted squared reconstruction loss over nonzero signal entries.
def _signal_loss(
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    signal: sparse.csr_matrix,
    gamma: float,
) -> float:
    # Step 1: Sum squared signal errors for each user's nonzero targets.
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
    # Step 2: Scale the accumulated signal loss by gamma.
    return float(gamma) * loss


class WMF_CPARMS:
    # Inputs:
    # - user_count: Number of users represented by the model.
    # - item_count: Number of items represented by the model.
    # - K: Latent factor dimension.
    # - lambda_rate: L2 regularization coefficient.
    # - alpha: Extra confidence assigned to positive L entries.
    # - gamma: Weight assigned to CPARMS-signal reconstruction.
    # - threshold: Minimum exclusive rating interpreted as positive feedback.
    # - random_state: Optional seed for factor initialization.
    # Output: Initialized CPARMS-regularized WMF model.
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
        self.user_count = int(user_count)
        self.item_count = int(item_count)
        self.K = int(K)
        self.lambda_rate = float(lambda_rate)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.threshold = float(threshold)
        self.random_state = resolve_seed(random_state)

        # Step 1: Validate model hyperparameters.
        if self.K <= 0:
            raise ValueError("K must be > 0.")
        if not np.isfinite(self.lambda_rate) or self.lambda_rate <= 0.0:
            raise ValueError("lambda_rate must be finite and > 0.")
        if not np.isfinite(self.alpha) or self.alpha < 0.0:
            raise ValueError("alpha must be finite and >= 0.")
        if not np.isfinite(self.gamma) or self.gamma < 0.0:
            raise ValueError("gamma must be finite and >= 0.")

        # Step 2: Initialize reproducible user and item latent factors.
        rng = np.random.RandomState(self.random_state)
        self.P = _initialize_factors(self.user_count, self.K, rng)
        self.Q = _initialize_factors(self.item_count, self.K, rng)

    # Inputs: None; reads model dimensions from this instance.
    # Output: A (user_count, item_count) tuple.
    @property
    def shape(self) -> tuple[int, int]:
        return self.user_count, self.item_count

    # Inputs:
    # - Y: Dense or sparse user-item interaction matrix.
    # - S: Dense or sparse CPARMS signal matrix with the model shape.
    # - n_sweeps: Number of full user/item ALS update sweeps.
    # - verbose_every: Positive interval controlling loss reporting.
    # Output: This fitted WMF_CPARMS instance.
    def fit(
        self,
        Y,
        S,
        n_sweeps: int,
        verbose_every: int = 5,
    ):
        n_sweeps, verbose_every = _validate_training_schedule(
            n_sweeps, verbose_every
        )
        # Step 1: Validate the signal and binarize positive preferences.
        signal = to_Y(S)
        if signal.shape != self.shape:
            raise ValueError(f"S must be shape {self.shape}.")
        L = to_L(Y, threshold=self.threshold)
        if L.shape != self.shape:
            raise ValueError(f"Y must be shape {self.shape}.")

        # Step 2: Transpose L and signal matrices for item updates.
        L_T = L.T.tocsr()
        signal_t = signal.T.tocsr()

        # Step 3: Alternate user-factor and item-factor least-squares updates.
        for sweep_idx in range(n_sweeps):
            # Step 3.1: Update users from interactions, signal, and fixed item factors.
            self.P = _als_factor_update(
                L,
                signal,
                self.Q,
                self.lambda_rate,
                self.alpha,
                self.gamma,
            )
            # Step 3.2: Update items from transposed inputs and fixed user factors.
            self.Q = _als_factor_update(
                L_T,
                signal_t,
                self.P,
                self.lambda_rate,
                self.alpha,
                self.gamma,
            )

            if _should_report_sweep(sweep_idx, verbose_every):
                # Step 3.3: Calculate reconstruction, signal, regularization, and total losses.
                wmf_loss = _wmf_loss(
                    self.P,
                    self.Q,
                    L,
                    self.alpha,
                )
                signal_loss = _signal_loss(
                    self.P,
                    self.Q,
                    signal,
                    self.gamma,
                )
                l2_sum = float(
                    np.sum(self.P * self.P)
                    + np.sum(self.Q * self.Q)
                )
                regularization = self.lambda_rate * l2_sum
                total_loss = wmf_loss + signal_loss + regularization
                print(
                    f"[Sweep {sweep_idx + 1}/{n_sweeps}] "
                    f"WMF_LOSS: {wmf_loss:.6f} "
                    f"SIGNAL_LOSS: {signal_loss:.6f} "
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
