"""Vanilla Neural Matrix Factorization (NeuMF) implicit-feedback baseline."""

from numbers import Integral

import numpy as np
import torch
from scipy import sparse
from torch import nn

from util.feedback import to_B, to_L
from util.seed_config import resolve_seed


def validate_positive_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def prepare_feedback(Y, shape) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    liked = to_L(Y)
    seen = to_B(Y)
    if liked.shape != shape or seen.shape != shape:
        raise ValueError(f"Y must be shape {shape}.")
    return liked, seen


def trainable_positive_pairs(
    liked: sparse.csr_matrix,
    negative_exclusions: sparse.csr_matrix,
) -> tuple[np.ndarray, np.ndarray]:
    """Return positive pairs for users with an item outside the exclusion set."""
    positive = liked.tocoo()
    if negative_exclusions.shape != liked.shape:
        raise ValueError("negative_exclusions must have the same shape as liked")
    can_sample = (
        np.asarray(negative_exclusions.getnnz(axis=1)).reshape(-1)
        < negative_exclusions.shape[1]
    )
    keep = can_sample[positive.row]
    return (
        positive.row[keep].astype(np.int64, copy=False),
        positive.col[keep].astype(np.int64, copy=False),
    )


def build_row_item_sets(matrix: sparse.csr_matrix) -> tuple[frozenset[int], ...]:
    return tuple(
        frozenset(
            matrix.indices[
                matrix.indptr[user_idx] : matrix.indptr[user_idx + 1]
            ].tolist()
        )
        for user_idx in range(matrix.shape[0])
    )


def sample_complement_items(
    user_idx: np.ndarray,
    excluded_sets: tuple[frozenset[int], ...],
    item_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one item outside each user's positive-item set."""
    negatives = rng.integers(0, item_count, size=user_idx.size, dtype=np.int64)
    invalid = np.fromiter(
        (
            int(item_idx) in excluded_sets[int(user)]
            for user, item_idx in zip(user_idx, negatives)
        ),
        dtype=bool,
        count=user_idx.size,
    )
    while np.any(invalid):
        positions = np.flatnonzero(invalid)
        negatives[positions] = rng.integers(
            0,
            item_count,
            size=positions.size,
            dtype=np.int64,
        )
        invalid[positions] = np.fromiter(
            (
                int(negatives[position])
                in excluded_sets[int(user_idx[position])]
                for position in positions
            ),
            dtype=bool,
            count=positions.size,
        )
    return negatives


def resolve_torch_device(device: str | None) -> torch.device:
    resolved = torch.device("cpu" if device is None else device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    return resolved


def torch_generator(seed: int, offset: int = 0) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed((int(seed) + int(offset)) % (2**63 - 1))
    return generator


class _NeuMFNetwork(nn.Module):
    def __init__(
        self,
        user_count,
        item_count,
        latent,
        hidden_layers,
        random_state,
    ):
        super().__init__()
        self.gmf_user = nn.Embedding(user_count, latent)
        self.gmf_item = nn.Embedding(item_count, latent)
        mlp_embedding_dim = hidden_layers[0] // 2
        self.mlp_user = nn.Embedding(user_count, mlp_embedding_dim)
        self.mlp_item = nn.Embedding(item_count, mlp_embedding_dim)

        layers = []
        input_size = hidden_layers[0]
        for hidden_size in hidden_layers[1:]:
            layers.extend([nn.Linear(input_size, hidden_size), nn.ReLU()])
            input_size = hidden_size
        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(latent + hidden_layers[-1], 1)
        self._reset_parameters(random_state)

    def _reset_parameters(self, random_state):
        embeddings = (
            self.gmf_user,
            self.gmf_item,
            self.mlp_user,
            self.mlp_item,
        )
        for idx, embedding in enumerate(embeddings):
            nn.init.normal_(
                embedding.weight,
                mean=0.0,
                std=0.01,
                generator=torch_generator(random_state, idx * 104729),
            )
        linear_idx = 0
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(
                    module.weight,
                    generator=torch_generator(
                        random_state,
                        500009 + linear_idx * 104729,
                    ),
                )
                nn.init.zeros_(module.bias)
                linear_idx += 1
        # Keras' ``lecun_uniform`` samples from
        # U(-sqrt(3 / fan_in), sqrt(3 / fan_in)).
        prediction_bound = np.sqrt(3.0 / self.output.in_features)
        nn.init.uniform_(
            self.output.weight,
            -prediction_bound,
            prediction_bound,
            generator=torch_generator(random_state, 900001),
        )
        nn.init.zeros_(self.output.bias)

    def forward(self, user_idx, item_idx):
        gmf = self.gmf_user(user_idx) * self.gmf_item(item_idx)
        mlp_input = torch.cat(
            [self.mlp_user(user_idx), self.mlp_item(item_idx)],
            dim=1,
        )
        mlp_output = self.mlp(mlp_input)
        return self.output(torch.cat([gmf, mlp_output], dim=1)).reshape(-1)


class NeuMF:
    # Inputs:
    # - user_count: Number of users represented by the model.
    # - item_count: Number of fit-window candidate items.
    # - latent: GMF embedding dimension and final predictive-factor width.
    # - hidden_layers: Canonical MLP tower widths including the concatenated
    #   embedding input; the first width must be even.
    # - learning_rate: Adam learning rate.
    # - reg_mf: Keras-style L2 coefficient for the two GMF embeddings.
    # - reg_layers: Keras-style L2 coefficients for the MLP embedding and
    #   successive dense layers. Must align with hidden_layers.
    # - negative_samples: Non-positive items sampled per positive pair.
    # - random_state: Seed for initialization, shuffling, and negative sampling.
    # - device: Torch device string; None selects CPU.
    # Output: Initialized vanilla NeuMF model.
    def __init__(
        self,
        user_count,
        item_count,
        latent=8,
        hidden_layers=(64, 32, 16, 8),
        learning_rate=0.001,
        reg_mf=0.0,
        reg_layers=None,
        negative_samples=4,
        random_state=None,
        device=None,
    ):
        self.user_count = validate_positive_int("user_count", user_count)
        self.item_count = validate_positive_int("item_count", item_count)
        self.latent = validate_positive_int("latent", latent)
        if not isinstance(hidden_layers, (tuple, list)) or not hidden_layers:
            raise ValueError("hidden_layers must contain positive integers")
        if any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or value <= 0
            for value in hidden_layers
        ):
            raise ValueError("hidden_layers must contain positive integers")
        self.hidden_layers = tuple(int(value) for value in hidden_layers)
        if self.hidden_layers[0] % 2 != 0:
            raise ValueError("hidden_layers[0] must be even")
        self.negative_samples = validate_positive_int(
            "negative_samples", negative_samples
        )
        self.learning_rate = float(learning_rate)
        self.reg_mf = float(reg_mf)
        if reg_layers is None:
            reg_layers = (0.0,) * len(self.hidden_layers)
        if len(reg_layers) != len(self.hidden_layers):
            raise ValueError("reg_layers must have the same length as hidden_layers")
        self.reg_layers = tuple(float(value) for value in reg_layers)
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and > 0")
        if not np.isfinite(self.reg_mf) or self.reg_mf < 0.0:
            raise ValueError("reg_mf must be finite and >= 0")
        if any(not np.isfinite(value) or value < 0.0 for value in self.reg_layers):
            raise ValueError("reg_layers must contain finite values >= 0")
        self.random_state = resolve_seed(random_state)
        self.device = resolve_torch_device(device)
        self.network = _NeuMFNetwork(
            self.user_count,
            self.item_count,
            self.latent,
            self.hidden_layers,
            self.random_state,
        ).to(self.device)

    @property
    def shape(self) -> tuple[int, int]:
        return self.user_count, self.item_count

    # Inputs:
    # - Y: Dense or sparse raw fit-window feedback matrix.
    # - epochs: Number of full passes over positive pairs.
    # - batch_size: Positive pairs processed per optimizer step.
    # - verbose_every: Epoch interval controlling loss reporting.
    # Output: This fitted NeuMF instance.
    def fit(self, Y, epochs=100, batch_size=256, verbose_every=10):
        epochs = validate_positive_int("epochs", epochs)
        batch_size = validate_positive_int("batch_size", batch_size)
        verbose_every = validate_positive_int("verbose_every", verbose_every)
        liked, _ = prepare_feedback(Y, self.shape)
        positive_users, positive_items = trainable_positive_pairs(liked, liked)
        if positive_users.size == 0:
            return self

        positive_sets = build_row_item_sets(liked)
        rng = np.random.default_rng(self.random_state)
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
        )
        criterion = nn.BCEWithLogitsLoss()
        dense_layers = [
            module
            for module in self.network.mlp
            if isinstance(module, nn.Linear)
        ]
        has_regularization = self.reg_mf > 0.0 or any(
            coefficient > 0.0 for coefficient in self.reg_layers
        )
        self.network.train()

        for epoch_idx in range(epochs):
            negative_users = np.repeat(
                positive_users,
                self.negative_samples,
            )
            negative_items = sample_complement_items(
                negative_users,
                positive_sets,
                self.item_count,
                rng,
            )
            users = np.concatenate([positive_users, negative_users])
            items = np.concatenate([positive_items, negative_items])
            labels = np.concatenate(
                [
                    np.ones(positive_users.size, dtype=np.float32),
                    np.zeros(negative_users.size, dtype=np.float32),
                ]
            )
            permutation = rng.permutation(users.size)
            epoch_loss = 0.0
            example_count = 0
            for start in range(0, permutation.size, batch_size):
                selection = permutation[start : start + batch_size]
                user_tensor = torch.as_tensor(
                    users[selection],
                    device=self.device,
                )
                item_tensor = torch.as_tensor(
                    items[selection],
                    device=self.device,
                )
                label_tensor = torch.as_tensor(
                    labels[selection],
                    device=self.device,
                )

                optimizer.zero_grad(set_to_none=True)
                logits = self.network(user_tensor, item_tensor)
                loss = criterion(logits, label_tensor)
                if has_regularization:
                    regularization = self.reg_mf * (
                        self.network.gmf_user.weight.square().sum()
                        + self.network.gmf_item.weight.square().sum()
                    )
                    regularization = regularization + self.reg_layers[0] * (
                        self.network.mlp_user.weight.square().sum()
                        + self.network.mlp_item.weight.square().sum()
                    )
                    for coefficient, layer in zip(
                        self.reg_layers[1:],
                        dense_layers,
                    ):
                        regularization = (
                            regularization
                            + coefficient * layer.weight.square().sum()
                        )
                    loss = loss + regularization
                loss.backward()
                optimizer.step()

                current_size = int(selection.size)
                epoch_loss += float(loss.detach().cpu()) * current_size
                example_count += current_size
            if epoch_idx == 0 or (epoch_idx + 1) % verbose_every == 0:
                print(
                    f"[Epoch {epoch_idx + 1}/{epochs}] "
                    f"BCE_LOSS: {epoch_loss / example_count:.6f}"
                )
        return self

    # Inputs:
    # - user_idx: Zero-based user index to score.
    # - batch_size: Number of candidate items scored per forward pass.
    # Output: Dense sigmoid probabilities for every candidate item.
    def score_user(self, user_idx: int, batch_size=8192) -> np.ndarray:
        user_idx = int(user_idx)
        batch_size = validate_positive_int("batch_size", batch_size)
        if user_idx < 0 or user_idx >= self.user_count:
            raise IndexError("user_idx is out of range.")
        result = np.empty(self.item_count, dtype=np.float32)
        self.network.eval()
        with torch.no_grad():
            for start in range(0, self.item_count, batch_size):
                stop = min(start + batch_size, self.item_count)
                users = torch.full(
                    (stop - start,),
                    user_idx,
                    dtype=torch.long,
                    device=self.device,
                )
                items = torch.arange(start, stop, device=self.device)
                result[start:stop] = (
                    torch.sigmoid(self.network(users, items))
                    .detach()
                    .cpu()
                    .numpy()
                )
        return result
