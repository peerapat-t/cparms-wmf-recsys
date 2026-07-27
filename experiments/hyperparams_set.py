# This file generates reproducible random hyperparameter samples shared across datasets.

import random
from numbers import Integral

from util.seed_config import resolve_seed


# Inputs:
# - rounds: Number of aligned hyperparameter configurations to sample.
# - global_seed: Optional seed used to make sampling reproducible.
# Output: List of model-parameter dictionaries, one dictionary per sampling round.
def generate_hyperparam_samples(
    rounds,
    global_seed=None,
):
    # Step 1: Validate the requested number of sampling rounds.
    if isinstance(rounds, bool) or not isinstance(rounds, Integral) or rounds <= 0:
        raise ValueError("rounds must be a positive integer")
    rounds = int(rounds)

    # Step 2: Define shared and model-specific hyperparameter spaces.
    param_space_common = {
        # Size of user/item latent vectors.
        "latent": [10, 20, 30, 40],
        # User/item L2 penalty.
        "lambda_rate": [0.0001, 0.001, 0.01, 0.1],
        # Number of full ALS updates.
        "n_sweeps": [15, 20, 25, 30],
        # Confidence added to positive feedback.
        "alpha": [10.0, 20.0, 40.0, 80.0],
    }

    # Weight of CPARMS signal reconstruction.
    gamma_space = [0.5, 1.0, 2.0, 4.0]
    # Scaling method for the generated CPARMS signal.
    normalize_space = ["row_max", "log_row_max"]

    # Relative scale ell of the WMF term; the co-occurrence weight is 1 / ell.
    cofactor_relative_scale_space = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]

    # Shift k in SPPMI = max(PMI - log(k), 0).
    negative_samples_space = [1, 2, 5, 10, 50]

    param_space_cparms_wmf = {
        # Number of user clusters sampled for generic rule antecedents.
        "k_user": (1, 2, 3, 5),
        # Number of item clusters sampled for personalized rule antecedents.
        "K_item": (1, 2, 3, 5),
        # Minimum support required to keep an association rule.
        "min_support": (0.0, 0.00005, 0.0001, 0.0003),
        # Minimum confidence required to keep an association rule.
        "min_confidence": (0.0, 0.0005, 0.001, 0.002),
        # Minimum lift required to keep an association rule.
        "min_lift": (0.0, 0.2, 0.5, 0.8),
    }

    # Step 3: Create deterministic random streams for aligned model samples.
    resolved_seed = resolve_seed(global_seed)
    rng = random.Random(resolved_seed)
    cofactor_rng = random.Random(resolved_seed + 104729)
    samples = []

    # Step 4: Sample one parameter dictionary per model in each round.
    for _ in range(rounds):
        standard_wmf_params = {
            key: rng.choice(values)
            for key, values in param_space_common.items()
        }
        # Seed used to initialize model factors reproducibly.
        standard_wmf_params["random_state"] = resolved_seed
        normalize = rng.choice(normalize_space)
        gamma = rng.choice(gamma_space)

        cparms_wmf_params = {
            **standard_wmf_params,
            "gamma": gamma,
            "k_user": rng.choice(param_space_cparms_wmf["k_user"]),
            "K_item": rng.choice(param_space_cparms_wmf["K_item"]),
            "min_support": rng.choice(
                param_space_cparms_wmf["min_support"]
            ),
            "min_confidence": rng.choice(
                param_space_cparms_wmf["min_confidence"]
            ),
            "min_lift": rng.choice(param_space_cparms_wmf["min_lift"]),
            "normalize": normalize,
        }

        cofactor_relative_scale = cofactor_rng.choice(
            cofactor_relative_scale_space
        )
        cofactor_gamma = 1.0 / cofactor_relative_scale
        cofactor_wmf_params = {
            **standard_wmf_params,
            "gamma": cofactor_gamma,
            "lambda_context_rate": (
                cofactor_gamma * standard_wmf_params["lambda_rate"]
            ),
            "negative_samples": cofactor_rng.choice(negative_samples_space),
        }

        samples.append({
            "standard_wmf": standard_wmf_params,
            "cparms_wmf": cparms_wmf_params,
            "cofactor_wmf": cofactor_wmf_params,
        })

    return samples
