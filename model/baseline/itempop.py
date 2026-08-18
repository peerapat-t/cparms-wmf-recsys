"""Provide an item-popularity recommendation baseline."""

import numpy as np

from util.dtype_config import FLOAT_DTYPE
from util.feedback import LIKE_THRESHOLD, to_L


MODEL_DTYPE = FLOAT_DTYPE


class ItemPop:
    """Rank items by the number of observed positive interactions."""


    def __init__(
        self,
        user_count,
        item_count,
        threshold: float = LIKE_THRESHOLD,
    ):
        """Initialize the baseline for a fixed user-item matrix shape."""
        self.user_count = int(user_count)
        self.item_count = int(item_count)
        self.threshold = float(threshold)
        self.item_popularity = np.zeros(self.item_count, dtype=MODEL_DTYPE)


    @property
    def shape(self) -> tuple[int, int]:
        """Return the expected user-item matrix shape."""
        return self.user_count, self.item_count


    def fit(self, Y):
        """Count positive interactions for each item."""
        L = to_L(Y, threshold=self.threshold)
        if L.shape != self.shape:
            raise ValueError(f"Y must be shape {self.shape}.")
        self.item_popularity = (
            np.asarray(L.sum(axis=0)).reshape(-1).astype(MODEL_DTYPE)
        )
        return self


    def score_user(self, user_idx: int) -> np.ndarray:
        """Return the global popularity scores for a valid user."""
        user_idx = int(user_idx)
        if user_idx < 0 or user_idx >= self.user_count:
            raise IndexError("user_idx is out of range.")
        return self.item_popularity.copy()
