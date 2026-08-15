"""Centralize deterministic random-seed configuration."""

import random

import numpy as np

GLOBAL_SEED = 42


def resolve_seed(seed: int | None = None) -> int:
    """Resolve an optional seed against the repository-wide default."""
    return GLOBAL_SEED if seed is None else int(seed)


def configure_reproducibility(seed: int | None = None) -> int:
    """Seed Python and NumPy, then return the resolved seed."""
    resolved_seed = resolve_seed(seed)
    random.seed(resolved_seed)
    np.random.seed(resolved_seed)
    return resolved_seed
