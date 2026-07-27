# This file centralizes random-seed handling for reproducible experiments.

import random

import numpy as np

GLOBAL_SEED = 42


# Inputs:
# - seed: Optional caller-provided seed; None selects GLOBAL_SEED.
# Output: The integer seed that all random-number generators should use.
def resolve_seed(seed: int | None = None) -> int:
    return GLOBAL_SEED if seed is None else int(seed)


# Inputs:
# - seed: Optional seed for Python and NumPy random generators.
# Output: The resolved integer seed after reproducibility settings are applied.
def configure_reproducibility(seed: int | None = None) -> int:
    resolved_seed = resolve_seed(seed)
    random.seed(resolved_seed)
    np.random.seed(resolved_seed)
    return resolved_seed
