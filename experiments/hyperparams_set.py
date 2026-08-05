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
        "latent": [10, 20, 30, 40, 50],
        # User/item L2 penalty.
        "lambda_rate": [0.00001, 0.0001, 0.001, 0.01, 0.1],
        # Number of full ALS updates.
        "n_sweeps": [10, 15, 20, 25, 30],
        # Confidence added to positive feedback.
        "alpha": [5.0, 10.0, 20.0, 30.0, 40.0],
    }

    # Weight of the CPARMS signal term.
    gamma_space = [0.1, 0.25, 0.5, 1.0, 2.0]
    # Scaling method for the generated CPARMS signal.
    normalize_space = ["row_max", "log_row_max"]

    # Relative scale ell of the WMF term; the co-occurrence weight is 1 / ell.
    cofactor_relative_scale_space = [
        0.01,
        0.03,
        0.05,
        0.1,
        0.5,
        1.0,
        5.0,
        10.0,
    ]
    # The upstream ML20M run uses 100 factors, 20 iterations, and c1/c0=10,
    # which is alpha=9 after normalizing the unobserved confidence to one.
    cofactor_latent_space = (50, 100)
    cofactor_sweeps_space = (10, 15, 20)
    cofactor_alpha_space = (4.0, 9.0, 19.0, 39.0)
    # CoFacto exposes context regularization independently from user/item
    # regularization. Values are in this implementation's c0-normalized scale.
    cofactor_context_regularization_space = (
        0.0001,
        0.0003,
        0.001,
        0.01,
        0.1,
        1.0,
        10.0,
        100.0,
    )

    # Shift k in SPPMI = max(PMI - log(k), 0).
    negative_samples_space = [1, 2, 5, 10, 50]

    # NeuMF follows the canonical tower: for f predictive factors the MLP
    # concatenation and successive widths are 8f -> 4f -> 2f -> f.
    neumf_predictive_factors = (8, 16, 32)
    param_space_neumf = {
        "learning_rate": (0.0001, 0.0005, 0.001, 0.005),
        # The official NeuMF defaults do not regularize either branch.
        "reg_mf": (0.0,),
        # Runtime-conscious budgets for repeated validation sweeps.
        "epochs": (20, 30, 50),
        "negative_samples": (4,),
        "batch_size": (256,),
    }
    param_space_lightgcn = {
        "latent": (32, 64),
        "n_layers": (1, 2, 3, 4),
        "learning_rate": (0.0005, 0.001),
        "lambda_rate": (0.00001, 0.0001),
        # repeated validation sweeps in this experiment pipeline.
        "epochs": (30, 50, 100),
        "negative_samples": (1,),
        "batch_size": (2048,),
    }

    param_space_cparms_wmf = {
        # Candidate user-cluster counts, from a global segment to finer groups.
        "k_user": (1, 2, 3, 4, 5),
        # Candidate item-cluster counts, from a global segment to finer groups.
        "k_item": (1, 2, 3, 4, 5),
        # Minimum rule support. These values span light through strong filtering
        "min_support": (0.0, 0.0001, 0.0003, 0.001, 0.002),
        # Minimum confidence required to keep an association rule.
        "min_confidence": (0.0, 0.0005, 0.001, 0.002, 0.003),
        # Minimum lift required to keep an association rule.
        "min_lift": (0.0, 0.5, 0.8, 1.0, 1.5),
    }

    # Step 3: Create deterministic random streams for aligned model samples.
    resolved_seed = resolve_seed(global_seed)
    rng = random.Random(resolved_seed)
    cofactor_rng = random.Random(resolved_seed + 104729)
    neumf_rng = random.Random(resolved_seed + 424243)
    lightgcn_rng = random.Random(resolved_seed + 524287)
    samples = []

    # Step 4: Sample one parameter dictionary per model in each round.
    for round_idx in range(rounds):
        standard_wmf_params = {
            key: rng.choice(values)
            for key, values in param_space_common.items()
        }
        # Seed used to initialize model factors reproducibly.
        standard_wmf_params["random_state"] = resolved_seed
        normalize = rng.choice(normalize_space)
        gamma = rng.choice(gamma_space)
        k_user = rng.choice(param_space_cparms_wmf["k_user"])
        k_item = rng.choice(param_space_cparms_wmf["k_item"])

        cparms_signal_params = {
            "k_user": k_user,
            # Generator_CPARMS currently exposes this argument as ``K_item``.
            "K_item": k_item,
            "min_support": rng.choice(
                param_space_cparms_wmf["min_support"]
            ),
            "min_confidence": rng.choice(
                param_space_cparms_wmf["min_confidence"]
            ),
            "min_lift": rng.choice(param_space_cparms_wmf["min_lift"]),
            "normalize": normalize,
            "random_state": resolved_seed,
        }
        cparms_wmf_params = {
            **standard_wmf_params,
            "gamma": gamma,
            **cparms_signal_params,
        }
        cofactor_relative_scale = cofactor_rng.choice(
            cofactor_relative_scale_space
        )
        cofactor_gamma = 1.0 / cofactor_relative_scale
        cofactor_wmf_params = {
            **standard_wmf_params,
            "latent": cofactor_rng.choice(cofactor_latent_space),
            "n_sweeps": cofactor_rng.choice(cofactor_sweeps_space),
            "alpha": cofactor_rng.choice(cofactor_alpha_space),
            "gamma": cofactor_gamma,
            "lambda_context_rate": cofactor_rng.choice(
                cofactor_context_regularization_space
            ),
            "negative_samples": cofactor_rng.choice(negative_samples_space),
        }
        neumf_params = {
            key: neumf_rng.choice(values)
            for key, values in param_space_neumf.items()
        }
        neumf_factor = neumf_rng.choice(neumf_predictive_factors)
        neumf_params.update({
            "latent": neumf_factor,
            "hidden_layers": tuple(
                neumf_factor * scale for scale in (8, 4, 2, 1)
            ),
            "reg_layers": (0.0, 0.0, 0.0, 0.0),
        })
        neumf_params["random_state"] = resolved_seed
        lightgcn_params = {
            key: lightgcn_rng.choice(values)
            for key, values in param_space_lightgcn.items()
        }
        lightgcn_params["random_state"] = resolved_seed

        # Round zero anchors the upstream architecture and optimizer settings,
        # but caps neural training budgets so baseline sweeps remain practical.
        if round_idx == 0:
            neumf_params.update({
                "latent": 8,
                "hidden_layers": (64, 32, 16, 8),
                "learning_rate": 0.001,
                "reg_mf": 0.0,
                "reg_layers": (0.0, 0.0, 0.0, 0.0),
                "epochs": 50,
                "negative_samples": 4,
                "batch_size": 256,
            })
            lightgcn_params.update({
                "latent": 64,
                "n_layers": 3,
                "learning_rate": 0.001,
                "lambda_rate": 0.0001,
                "epochs": 100,
                "negative_samples": 1,
                "batch_size": 2048,
            })
            cofactor_wmf_params.update({
                "latent": 100,
                "lambda_rate": 0.00001,
                "n_sweeps": 20,
                "alpha": 9.0,
                "gamma": 1.0 / 0.03,
                "lambda_context_rate": 0.00001 / 0.03,
                "negative_samples": 1,
            })

        samples.append({
            "standard_wmf": standard_wmf_params,
            "cofactor_wmf": cofactor_wmf_params,
            "neumf": neumf_params,
            "lightgcn": lightgcn_params,
            "cparms_wmf": cparms_wmf_params,
        })

    return samples
