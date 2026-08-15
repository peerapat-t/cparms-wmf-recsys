"""Train standard weighted matrix factorization with implicit ALS."""

from numbers import Integral

import numpy as np
from scipy import sparse

from util.feedback import LIKE_THRESHOLD, to_L
from util.seed_config import resolve_seed


MODEL_DTYPE = np.float64


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


def _als_factor_update(
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


class WMF:
    """Fit Hu-style weighted matrix factorization with alternating solves."""


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
        """Validate hyperparameters and initialize latent factors."""
        self.user_count = int(user_count)
        self.item_count = int(item_count)
        self.K = int(K)
        self.lambda_rate = float(lambda_rate)
        self.alpha = float(alpha)
        self.threshold = float(threshold)
        self.random_state = resolve_seed(random_state)

        if self.K <= 0:
            raise ValueError("K must be > 0.")
        if not np.isfinite(self.lambda_rate) or self.lambda_rate <= 0.0:
            raise ValueError("lambda_rate must be finite and > 0.")
        if not np.isfinite(self.alpha) or self.alpha < 0.0:
            raise ValueError("alpha must be finite and >= 0.")


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
        n_sweeps: int,
        verbose_every: int = 5,
    ):
        """Alternate user and item factor updates over positive feedback."""
        n_sweeps, verbose_every = _validate_training_schedule(
            n_sweeps, verbose_every
        )
        L = to_L(Y, threshold=self.threshold)
        if L.shape != self.shape:
            raise ValueError(f"Y must be shape {self.shape}.")
        L_T = L.T.tocsr()

        # Alternate closed-form user and item updates until the schedule ends.
        for sweep_idx in range(n_sweeps):
            self.P = _als_factor_update(
                L,
                self.Q,
                self.lambda_rate,
                self.alpha,
            )
            self.Q = _als_factor_update(
                L_T,
                self.P,
                self.lambda_rate,
                self.alpha,
            )

            if _should_report_sweep(sweep_idx, verbose_every):
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


    def score_user(self, user_idx: int) -> np.ndarray:
        """Score every item for one user."""
        user_idx = int(user_idx)
        if user_idx < 0 or user_idx >= self.user_count:
            raise IndexError("user_idx is out of range.")
        return self.Q @ self.P[user_idx]
