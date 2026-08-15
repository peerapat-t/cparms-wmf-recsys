"""Sample reproducible, model-independent hyperparameter configurations."""

import random
from numbers import Integral

from util.seed_config import resolve_seed


# Standard-WMF search space.
PARAM_SPACE_STANDARD_WMF = {
    "latent": (10, 20, 40, 60, 80, 100),
    "lambda_rate": (0.0001, 0.001, 0.01, 0.1, 1.0),
    "n_sweeps": (10, 15, 20, 25, 30),
    "alpha": (1.0, 5.0, 10.0, 20.0, 40.0, 80.0),
}


# CoFactor search space.
PARAM_SPACE_COFACTOR_WMF = {
    "latent": (10, 20, 40, 60, 80, 100),
    "lambda_rate": (0.0001, 0.001, 0.01, 0.1, 1.0),
    "n_sweeps": (10, 15, 20, 25, 30),
    "alpha": (1.0, 5.0, 10.0, 20.0, 40.0, 80.0),
    "relative_scale": (0.01, 0.03, 0.1, 1.0, 10.0),
    "lambda_context_rate": (
        0.00001,
        0.0001,
        0.0003,
        0.001,
        0.01,
        0.1,
        1.0,
    ),
    "negative_samples": (1,),
}


# RME search space.
PARAM_SPACE_RME = {
    "latent": (10, 20, 40, 60, 80, 100),
    "lambda_rate": (0.0001, 0.001, 0.01, 0.1, 1.0),
    "n_sweeps": (10, 15, 20, 25, 30),
    "alpha": (1.0, 5.0, 10.0, 20.0, 40.0, 80.0),
    "gamma_item_pos": (0.1, 1.0, 5.0, 10.0, 20.0),
    "gamma_item_neg": (0.1, 1.0, 5.0, 10.0, 20.0),
    "gamma_user_pos": (0.01, 0.1, 0.5, 1.0, 2.0),
    "lambda_context_rate": (0.0001, 0.001, 0.01, 0.1, 1.0),
    "negative_samples": (1,),
}


# BPR-MF-only search space.
PARAM_SPACE_BPR = {
    "latent": (8, 16, 32, 64, 128),
    "learning_rate": (0.01, 0.025, 0.05),
    "lambda_user": (0.00025, 0.0025, 0.025),
    "lambda_positive": (0.00025, 0.0025, 0.025),
    "lambda_negative": (0.000025, 0.00025, 0.0025),
    "lambda_bias": (0.0, 0.0001, 0.001, 0.01),
    "epochs": (10, 25, 50, 100),
}


# NeuMF-only search space. ``latent`` is the predictive-factor count;
# ``hidden_layers`` is derived from it after sampling.
PARAM_SPACE_NEUMF = {
    "latent": (8, 16, 32, 64),
    "learning_rate": (0.0001, 0.0005, 0.001, 0.005),
    "reg_mf": (0.0, 0.000001, 0.00001, 0.0001),
    "reg_layers": (0.0, 0.000001, 0.00001, 0.0001),
    "epochs": (20, 30, 50),
    "negative_samples": (4,),
    "batch_size": (128, 256, 512, 1024),
}


# LightGCN-only search space.
PARAM_SPACE_LIGHTGCN = {
    "latent": (32, 64, 128),
    "n_layers": (1, 2, 3, 4),
    "learning_rate": (0.0001, 0.0005, 0.001, 0.005),
    "lambda_rate": (0.000001, 0.00001, 0.0001, 0.001, 0.01),
    "epochs": (30, 50, 100),
    "negative_samples": (1,),
    "batch_size": (1024, 2048, 4096),
}


# CPARMS-L-only search space.
PARAM_SPACE_CPARMS_L = {
    "latent": (10, 20, 40, 60, 80, 100),
    "lambda_rate": (0.0001, 0.001, 0.01, 0.1, 1.0),
    "n_sweeps": (10, 15, 20, 25, 30),
    "alpha": (1.0, 5.0, 10.0, 20.0, 40.0, 80.0),
    "gamma": (0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
    "k_user": (1, 2, 3, 4, 5),
    "K_item": (1, 2, 3, 4, 5),
    "min_support": (0.0, 0.0001, 0.0003, 0.001, 0.002),
    "min_confidence": (0.0, 0.0005, 0.001, 0.002, 0.003),
    "min_lift": (0.0, 0.5, 0.8, 1.0, 1.5),
    "normalize": ("row_max", "log_row_max"),
}


# CPARMS-D-only search space.
PARAM_SPACE_CPARMS_D = {
    "latent": (10, 20, 40, 60, 80, 100),
    "lambda_rate": (0.0001, 0.001, 0.01, 0.1, 1.0),
    "n_sweeps": (10, 15, 20, 25, 30),
    "alpha": (1.0, 5.0, 10.0, 20.0, 40.0, 80.0),
    "gamma": (0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
    "k_user": (1, 2, 3, 4, 5),
    "K_item": (1, 2, 3, 4, 5),
    "min_support": (0.0, 0.0001, 0.0003, 0.001, 0.002),
    "min_confidence": (0.0, 0.0005, 0.001, 0.002, 0.003),
    "min_lift": (0.0, 0.5, 0.8, 1.0, 1.5),
    "normalize": ("row_max", "log_row_max"),
}


# CPARMS-LD-only search space.
PARAM_SPACE_CPARMS_LD = {
    "latent": (10, 20, 40, 60, 80, 100),
    "lambda_rate": (0.0001, 0.001, 0.01, 0.1, 1.0),
    "n_sweeps": (10, 15, 20, 25, 30),
    "alpha": (1.0, 5.0, 10.0, 20.0, 40.0, 80.0),
    "gamma_like": (0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
    "gamma_dislike": (0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
    "k_user": (1, 2, 3, 4, 5),
    "K_item": (1, 2, 3, 4, 5),
    "min_support": (0.0, 0.0001, 0.0003, 0.001, 0.002),
    "min_confidence": (0.0, 0.0005, 0.001, 0.002, 0.003),
    "min_lift": (0.0, 0.5, 0.8, 1.0, 1.5),
    "normalize": ("row_max", "log_row_max"),
}


MODEL_PARAM_SPACES = {
    "standard_wmf": PARAM_SPACE_STANDARD_WMF,
    "cofactor_wmf": PARAM_SPACE_COFACTOR_WMF,
    "rme": PARAM_SPACE_RME,
    "bpr": PARAM_SPACE_BPR,
    "neumf": PARAM_SPACE_NEUMF,
    "lightgcn": PARAM_SPACE_LIGHTGCN,
    "cparms_l": PARAM_SPACE_CPARMS_L,
    "cparms_d": PARAM_SPACE_CPARMS_D,
    "cparms_ld": PARAM_SPACE_CPARMS_LD,
}


def _sample_space(space, rng):
    """Draw one value independently for every field in a model space."""
    return {key: rng.choice(values) for key, values in space.items()}


def _finalize_sample(model_key, params, resolved_seed):
    """Convert search-only fields into the schema consumed by the notebook."""
    params = dict(params)
    if model_key == "cofactor_wmf":
        relative_scale = params.pop("relative_scale")
        params["gamma"] = 1.0 / relative_scale
    elif model_key == "neumf":
        predictive_factors = params["latent"]
        layer_regularization = params["reg_layers"]
        params["hidden_layers"] = tuple(
            predictive_factors * scale for scale in (8, 4, 2, 1)
        )
        params["reg_layers"] = (layer_regularization,) * 4
    params["random_state"] = resolved_seed
    return params


def generate_hyperparam_samples(
    rounds,
    global_seed=None,
):
    """Generate unique, reproducible, independently drawn model configs."""
    if (
        isinstance(rounds, bool)
        or not isinstance(rounds, Integral)
        or rounds <= 0
    ):
        raise ValueError("rounds must be a positive integer")
    rounds = int(rounds)
    resolved_seed = resolve_seed(global_seed)
    model_rngs = {
        model_key: random.Random(f"{resolved_seed}:{model_key}")
        for model_key in MODEL_PARAM_SPACES
    }

    samples = []
    seen_configurations = {}
    attempts = 0
    max_attempts = max(1000, rounds * 100)

    # Reject a round if it duplicates any model configuration already used.
    while len(samples) < rounds:
        attempts += 1
        if attempts > max_attempts:
            raise ValueError(
                "rounds exceeds the number of unique configurations that "
                "can be sampled reliably"
            )

        sampled_models = {
            model_key: _finalize_sample(
                model_key,
                _sample_space(space, model_rngs[model_key]),
                resolved_seed,
            )
            for model_key, space in MODEL_PARAM_SPACES.items()
        }
        fingerprints = {
            model_key: tuple(sorted(params.items()))
            for model_key, params in sampled_models.items()
        }
        if any(
            fingerprint in seen_configurations.get(model_key, set())
            for model_key, fingerprint in fingerprints.items()
        ):
            continue
        for model_key, fingerprint in fingerprints.items():
            seen_configurations.setdefault(model_key, set()).add(fingerprint)
        samples.append(sampled_models)

    return samples
