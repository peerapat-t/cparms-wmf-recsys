"""Train a Bayesian Personalized Ranking matrix-factorization model."""

from numbers import Integral

import numpy as np
from scipy import sparse

from util.feedback import to_L
from util.seed_config import resolve_seed


def validate_positive_int(name: str, value) -> int:
    """Validate and return a positive integer hyperparameter."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def validate_nonnegative_float(name: str, value) -> float:
    """Validate and return a finite nonnegative hyperparameter."""
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return value


def prepare_feedback(Y, shape) -> sparse.csr_matrix:
    """Validate the feedback shape and retain positive interactions."""
    liked = to_L(Y)
    if liked.shape != shape:
        raise ValueError(f"Y must be shape {shape}.")
    return liked


def build_row_item_arrays(
    matrix: sparse.csr_matrix,
) -> tuple[np.ndarray, ...]:
    """Return each CSR row's item indices as an independent array."""
    return tuple(
        matrix.indices[
            matrix.indptr[user_idx] : matrix.indptr[user_idx + 1]
        ].astype(np.int64, copy=True)
        for user_idx in range(matrix.shape[0])
    )


def build_row_item_sets(matrix: sparse.csr_matrix) -> tuple[frozenset[int], ...]:
    """Return each CSR row's item indices as a membership set."""
    return tuple(frozenset(items.tolist()) for items in build_row_item_arrays(matrix))


def build_positive_pairs(
    liked: sparse.csr_matrix,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract positive pairs for users with at least one negative item."""
    positive_counts = np.diff(liked.indptr)
    users = np.repeat(
        np.arange(liked.shape[0], dtype=np.int64),
        positive_counts,
    )
    items = liked.indices.astype(np.int64, copy=True)
    trainable = positive_counts[users] < liked.shape[1]
    return users[trainable], items[trainable]


def sample_negative_item(
    user_idx: int,
    positive_sets: tuple[frozenset[int], ...],
    item_count: int,
    rng: np.random.Generator,
) -> int:
    """Sample an item outside the selected user's positive set."""
    positive_set = positive_sets[user_idx]
    while True:
        item_idx = int(rng.integers(item_count))
        if item_idx not in positive_set:
            return item_idx


class BPRMF:
    """Learn pairwise user-item preferences with stochastic BPR updates."""

    def __init__(
        self,
        user_count,
        item_count,
        latent=16,
        learning_rate=0.05,
        lambda_user=0.0025,
        lambda_positive=0.0025,
        lambda_negative=0.00025,
        lambda_bias=0.0,
        random_state=None,
    ):
        """Validate hyperparameters and initialize latent parameters."""
        self.user_count = validate_positive_int("user_count", user_count)
        self.item_count = validate_positive_int("item_count", item_count)
        self.latent = validate_positive_int("latent", latent)
        self.learning_rate = float(learning_rate)
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and > 0")
        self.lambda_user = validate_nonnegative_float(
            "lambda_user", lambda_user
        )
        self.lambda_positive = validate_nonnegative_float(
            "lambda_positive", lambda_positive
        )
        self.lambda_negative = validate_nonnegative_float(
            "lambda_negative", lambda_negative
        )
        self.lambda_bias = validate_nonnegative_float(
            "lambda_bias", lambda_bias
        )
        self.random_state = resolve_seed(random_state)

        self._initialize_parameters()

    def _initialize_parameters(self):
        """Reset user factors, item factors, and item biases."""
        rng = np.random.default_rng(self.random_state)
        self.user_factors = rng.normal(
            0.0,
            0.1,
            size=(self.user_count, self.latent),
        ).astype(np.float64)
        self.item_factors = rng.normal(
            0.0,
            0.1,
            size=(self.item_count, self.latent),
        ).astype(np.float64)
        self.item_bias = np.zeros(self.item_count, dtype=np.float64)

    @property
    def shape(self) -> tuple[int, int]:
        """Return the expected user-item matrix shape."""
        return self.user_count, self.item_count

    def fit(self, Y, epochs=100, verbose_every=10):
        """Fit the model with shuffled positive-negative item pairs."""
        epochs = validate_positive_int("epochs", epochs)
        verbose_every = validate_positive_int("verbose_every", verbose_every)
        liked = prepare_feedback(Y, self.shape)
        self._initialize_parameters()
        positive_users, positive_items = build_positive_pairs(liked)
        if positive_users.size == 0:
            return self

        positive_sets = build_row_item_sets(liked)
        rng = np.random.default_rng(self.random_state)
        pair_order = np.arange(positive_users.size, dtype=np.int64)

        # Update one sampled pair at a time using the BPR objective.
        for epoch_idx in range(epochs):
            epoch_loss = 0.0
            rng.shuffle(pair_order)
            for pair_idx in pair_order:
                user_idx = int(positive_users[pair_idx])
                positive_idx = int(positive_items[pair_idx])
                negative_idx = sample_negative_item(
                    user_idx,
                    positive_sets,
                    self.item_count,
                    rng,
                )

                user = self.user_factors[user_idx].copy()
                positive = self.item_factors[positive_idx].copy()
                negative = self.item_factors[negative_idx].copy()
                positive_bias = float(self.item_bias[positive_idx])
                negative_bias = float(self.item_bias[negative_idx])
                score_difference = (
                    positive_bias
                    - negative_bias
                    + float(user @ (positive - negative))
                )
                gradient_scale = float(np.exp(-np.logaddexp(0.0, score_difference)))

                self.item_bias[positive_idx] += self.learning_rate * (
                    gradient_scale - self.lambda_bias * positive_bias
                )
                self.item_bias[negative_idx] += self.learning_rate * (
                    -gradient_scale - self.lambda_bias * negative_bias
                )

                self.user_factors[user_idx] += self.learning_rate * (
                    gradient_scale * (positive - negative)
                    - self.lambda_user * user
                )
                self.item_factors[positive_idx] += self.learning_rate * (
                    gradient_scale * user
                    - self.lambda_positive * positive
                )
                self.item_factors[negative_idx] += self.learning_rate * (
                    -gradient_scale * user
                    - self.lambda_negative * negative
                )

                epoch_loss += float(np.logaddexp(0.0, -score_difference))
                epoch_loss += 0.5 * (
                    self.lambda_user * float(user @ user)
                    + self.lambda_positive * float(positive @ positive)
                    + self.lambda_negative * float(negative @ negative)
                    + self.lambda_bias * (
                        positive_bias**2 + negative_bias**2
                    )
                )

            if epoch_idx == 0 or (epoch_idx + 1) % verbose_every == 0:
                print(
                    f"[Epoch {epoch_idx + 1}/{epochs}] "
                    f"BPR_LOSS: {epoch_loss / positive_users.size:.6f}"
                )
        return self

    def score_user(self, user_idx: int) -> np.ndarray:
        """Score every item for one user."""
        user_idx = int(user_idx)
        if user_idx < 0 or user_idx >= self.user_count:
            raise IndexError("user_idx is out of range.")
        return (
            self.item_bias
            + self.item_factors @ self.user_factors[user_idx]
        )
