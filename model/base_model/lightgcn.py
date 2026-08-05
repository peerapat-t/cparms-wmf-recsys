"""Vanilla LightGCN recommender trained with pairwise BPR loss."""

from numbers import Integral

import numpy as np
import torch
from scipy import sparse
from scipy.special import expit
from torch import nn
from torch.nn import functional as functional

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


def sample_uniform_user_triples(
    liked: sparse.csr_matrix,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce LightGCN's Python fallback BPR sampler for one epoch.

    The upstream repository first tries an optional C++ extension and otherwise
    uses this Python path. The Python path draws ``liked.nnz`` users uniformly,
    skips users without a train positive, then draws one positive and one
    non-positive item for every retained user.
    """
    user_count, item_count = liked.shape
    row_counts = np.diff(liked.indptr)
    sampled_users = rng.integers(
        0,
        user_count,
        size=liked.nnz,
        dtype=np.int64,
    )
    usable = (row_counts[sampled_users] > 0) & (
        row_counts[sampled_users] < item_count
    )
    sampled_users = sampled_users[usable]
    positives = np.empty(sampled_users.size, dtype=np.int64)
    for idx, user_idx in enumerate(sampled_users):
        start = liked.indptr[user_idx]
        stop = liked.indptr[user_idx + 1]
        positives[idx] = liked.indices[rng.integers(start, stop)]

    positive_sets = build_row_item_sets(liked)
    negatives = sample_complement_items(
        sampled_users,
        positive_sets,
        item_count,
        rng,
    )
    return sampled_users, positives, negatives


def resolve_torch_device(device: str | None) -> torch.device:
    resolved = torch.device("cpu" if device is None else device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    return resolved


def torch_generator(seed: int, offset: int = 0) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed((int(seed) + int(offset)) % (2**63 - 1))
    return generator


# Inputs:
# - liked: Binary positive user-item interaction matrix.
# - device: Torch device that stores the sparse normalized graph.
# Output: Symmetric D^-1/2 A D^-1/2 sparse adjacency over user and item nodes.
def _build_normalized_adjacency(
    liked: sparse.csr_matrix,
    device: torch.device,
) -> torch.Tensor:
    positive = liked.tocoo()
    user_count, item_count = liked.shape
    node_count = user_count + item_count
    user_nodes = positive.row.astype(np.int64, copy=False)
    item_nodes = user_count + positive.col.astype(np.int64, copy=False)
    source = np.concatenate([user_nodes, item_nodes])
    target = np.concatenate([item_nodes, user_nodes])
    degree = np.bincount(source, minlength=node_count).astype(np.float64)
    values = np.zeros(source.size, dtype=np.float32)
    if source.size:
        values = (
            1.0 / np.sqrt(degree[source] * degree[target])
        ).astype(np.float32, copy=False)
    indices = torch.as_tensor(
        np.vstack([source, target]),
        dtype=torch.long,
        device=device,
    )
    graph_values = torch.as_tensor(values, device=device)
    return torch.sparse_coo_tensor(
        indices,
        graph_values,
        size=(node_count, node_count),
        device=device,
    ).coalesce()


class _LightGCNNetwork(nn.Module):
    # Inputs:
    # - user_count: Number of user-node embeddings.
    # - item_count: Number of item-node embeddings.
    # - latent: Embedding dimension.
    # - n_layers: Number of graph propagation layers.
    # - random_state: Initialization seed.
    # Output: Initialized LightGCN embedding network.
    def __init__(self, user_count, item_count, latent, n_layers, random_state):
        super().__init__()
        self.user_count = int(user_count)
        self.n_layers = int(n_layers)
        self.user_embedding = nn.Embedding(user_count, latent)
        self.item_embedding = nn.Embedding(item_count, latent)
        nn.init.normal_(
            self.user_embedding.weight,
            mean=0.0,
            std=0.1,
            generator=torch_generator(random_state),
        )
        nn.init.normal_(
            self.item_embedding.weight,
            mean=0.0,
            std=0.1,
            generator=torch_generator(random_state, 104729),
        )

    # Inputs:
    # - adjacency: Normalized sparse adjacency over all user and item nodes.
    # Output: Layer-mean user and item embeddings used for ranking.
    def propagate(self, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight],
            dim=0,
        )
        layer_embeddings = [embeddings]
        for _ in range(self.n_layers):
            embeddings = torch.sparse.mm(adjacency, embeddings)
            layer_embeddings.append(embeddings)
        combined = torch.stack(layer_embeddings, dim=0).mean(dim=0)
        return combined[: self.user_count], combined[self.user_count :]


# Inputs:
# - network: LightGCN embedding network containing the trainable ego factors.
# - adjacency: Symmetric normalized user-item graph.
# - users/positives/negatives: One sampled training triple per row.
# - lambda_rate: Upstream ``decay`` coefficient for ego-embedding L2.
# Output: The upstream mean pairwise ranking loss and half-scaled L2 penalty.
def _bpr_batch_objective(
    network: _LightGCNNetwork,
    adjacency: torch.Tensor,
    users: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    lambda_rate: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    user_factors, item_factors = network.propagate(adjacency)
    sampled_users = user_factors[users]
    positive_scores = torch.sum(
        sampled_users * item_factors[positives],
        dim=1,
    )
    negative_scores = torch.sum(
        sampled_users * item_factors[negatives],
        dim=1,
    )
    # -log(sigmoid(pos-neg)) is algebraically identical to the upstream
    # softplus(neg-pos) expression.
    ranking_loss = -functional.logsigmoid(
        positive_scores - negative_scores
    ).mean()
    # The reference regularizes only sampled layer-0 (ego) embeddings and
    # divides their summed squared norms by twice the batch size.
    regularization = float(lambda_rate) * (
        network.user_embedding(users).square().sum(dim=1)
        + network.item_embedding(positives).square().sum(dim=1)
        + network.item_embedding(negatives).square().sum(dim=1)
    ).mean() * 0.5
    return ranking_loss, regularization


class LightGCN:
    # Inputs:
    # - user_count: Number of users represented by the model.
    # - item_count: Number of fit-window candidate items.
    # - latent: Initial node-embedding dimension.
    # - n_layers: Number of parameter-free graph propagation layers.
    # - learning_rate: Adam learning rate.
    # - lambda_rate: L2 coefficient on sampled initial embeddings.
    # - negative_samples: Non-positive items sampled per positive pair.
    # - random_state: Seed for initialization, shuffling, and negative sampling.
    # - device: Torch device string; None selects CPU.
    # Output: Initialized vanilla LightGCN model.
    def __init__(
        self,
        user_count,
        item_count,
        latent=64,
        n_layers=3,
        learning_rate=0.001,
        lambda_rate=0.0001,
        negative_samples=1,
        random_state=None,
        device=None,
    ):
        self.user_count = validate_positive_int("user_count", user_count)
        self.item_count = validate_positive_int("item_count", item_count)
        self.latent = validate_positive_int("latent", latent)
        self.n_layers = validate_positive_int("n_layers", n_layers)
        self.negative_samples = validate_positive_int(
            "negative_samples", negative_samples
        )
        if self.negative_samples != 1:
            raise ValueError("LightGCN reference training uses one negative per user")
        self.learning_rate = float(learning_rate)
        self.lambda_rate = float(lambda_rate)
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and > 0")
        if not np.isfinite(self.lambda_rate) or self.lambda_rate < 0.0:
            raise ValueError("lambda_rate must be finite and >= 0")
        self.random_state = resolve_seed(random_state)
        self.device = resolve_torch_device(device)
        self.network = _LightGCNNetwork(
            self.user_count,
            self.item_count,
            self.latent,
            self.n_layers,
            self.random_state,
        ).to(self.device)
        self.adjacency = None
        self.user_factors = np.zeros(
            (self.user_count, self.latent),
            dtype=np.float32,
        )
        self.item_factors = np.zeros(
            (self.item_count, self.latent),
            dtype=np.float32,
        )

    # Inputs: None; reads model dimensions from this instance.
    # Output: A (user_count, item_count) tuple.
    @property
    def shape(self) -> tuple[int, int]:
        return self.user_count, self.item_count

    # Inputs: None; propagates the fitted graph and reads learned embeddings.
    # Output: None; refreshes cached CPU factors used during ranking.
    def _refresh_factor_cache(self) -> None:
        self.network.eval()
        with torch.no_grad():
            user_factors, item_factors = self.network.propagate(self.adjacency)
        self.user_factors = user_factors.detach().cpu().numpy().astype(
            np.float32,
            copy=True,
        )
        self.item_factors = item_factors.detach().cpu().numpy().astype(
            np.float32,
            copy=True,
        )

    # Inputs:
    # - Y: Dense or sparse raw fit-window feedback matrix.
    # - epochs: Number of full passes over positive pairs.
    # - batch_size: Positive pairs processed per optimizer step.
    # - verbose_every: Epoch interval controlling loss reporting.
    # Output: This fitted LightGCN instance.
    def fit(self, Y, epochs=1000, batch_size=2048, verbose_every=10):
        epochs = validate_positive_int("epochs", epochs)
        batch_size = validate_positive_int("batch_size", batch_size)
        verbose_every = validate_positive_int("verbose_every", verbose_every)
        liked, _ = prepare_feedback(Y, self.shape)
        self.adjacency = _build_normalized_adjacency(liked, self.device)
        if liked.nnz == 0:
            self._refresh_factor_cache()
            return self

        rng = np.random.default_rng(self.random_state)
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
        )
        self.network.train()

        for epoch_idx in range(epochs):
            users, positives, negatives = sample_uniform_user_triples(
                liked,
                rng,
            )
            if users.size == 0:
                break
            permutation = rng.permutation(users.size)
            users = users[permutation]
            positives = positives[permutation]
            negatives = negatives[permutation]
            epoch_loss = 0.0
            sample_count = 0
            for start in range(0, permutation.size, batch_size):
                stop = start + batch_size
                user_tensor = torch.as_tensor(
                    users[start:stop],
                    device=self.device,
                )
                positive_tensor = torch.as_tensor(
                    positives[start:stop],
                    device=self.device,
                )
                negative_tensor = torch.as_tensor(
                    negatives[start:stop],
                    device=self.device,
                )

                optimizer.zero_grad(set_to_none=True)
                ranking_loss, regularization = _bpr_batch_objective(
                    self.network,
                    self.adjacency,
                    user_tensor,
                    positive_tensor,
                    negative_tensor,
                    self.lambda_rate,
                )
                loss = ranking_loss + regularization
                loss.backward()
                optimizer.step()

                current_size = int(user_tensor.numel())
                epoch_loss += float(loss.detach().cpu()) * current_size
                sample_count += current_size
            if epoch_idx == 0 or (epoch_idx + 1) % verbose_every == 0:
                print(
                    f"[Epoch {epoch_idx + 1}/{epochs}] "
                    f"LIGHTGCN_LOSS: {epoch_loss / sample_count:.6f}"
                )

        self._refresh_factor_cache()
        return self

    # Inputs:
    # - user_idx: Zero-based user index to score.
    # Output: Dense propagated-factor scores for every candidate item.
    def score_user(self, user_idx: int) -> np.ndarray:
        user_idx = int(user_idx)
        if user_idx < 0 or user_idx >= self.user_count:
            raise IndexError("user_idx is out of range.")
        logits = self.item_factors @ self.user_factors[user_idx]
        return expit(logits).astype(np.float32, copy=False)
