# This file implements the MostPop non-personalized recommendation baseline.

import numpy as np

from util.feedback import LIKE_THRESHOLD, to_L


# MostPop ranks every user by global item positive-popularity: the number of
# users who positively interacted (rating > threshold) with each item in the fit
# window. It is deterministic and learns no per-user information, so it is the
# reference "shared prior" every collaborative model must beat.
class MostPop:
    # Inputs:
    # - user_count: Number of users represented by the model.
    # - item_count: Number of items represented by the model.
    # - threshold: Minimum exclusive rating counted as a positive interaction.
    # Output: Initialized MostPop baseline with an empty popularity vector.
    def __init__(
        self,
        user_count,
        item_count,
        threshold: float = LIKE_THRESHOLD,
    ):
        self.user_count = int(user_count)
        self.item_count = int(item_count)
        self.threshold = float(threshold)
        self.item_popularity = np.zeros(self.item_count, dtype=np.float64)

    # Inputs: None; reads model dimensions from this instance.
    # Output: A (user_count, item_count) tuple.
    @property
    def shape(self) -> tuple[int, int]:
        return self.user_count, self.item_count

    # Inputs:
    # - Y: Dense or sparse user-item interaction matrix over the fit window.
    # Output: This fitted MostPop instance.
    def fit(self, Y):
        # Step 1: Binarize positive feedback and count each item's supporters.
        L = to_L(Y, threshold=self.threshold)
        if L.shape != self.shape:
            raise ValueError(f"Y must be shape {self.shape}.")
        self.item_popularity = (
            np.asarray(L.sum(axis=0)).reshape(-1).astype(np.float64)
        )
        return self

    # Inputs:
    # - user_idx: Zero-based user index to score.
    # Output: The shared global popularity vector (identical for every user).
    def score_user(self, user_idx: int) -> np.ndarray:
        user_idx = int(user_idx)
        if user_idx < 0 or user_idx >= self.user_count:
            raise IndexError("user_idx is out of range.")
        return self.item_popularity.copy()
