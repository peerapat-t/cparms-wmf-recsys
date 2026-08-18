"""Train a LightGCN recommender with pairwise ranking loss."""

from numbers import Integral

import numpy as np
import torch
from scipy import sparse
from scipy.special import expit
from torch import nn
from torch.nn import functional as functional

from util.dtype_config import FLOAT_DTYPE
from util.feedback import to_B, to_L
from util.seed_config import resolve_seed


MODEL_DTYPE = FLOAT_DTYPE


def validate_positive_int(name: str, value) -> int:
    """Validate and return a positive integer hyperparameter."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def prepare_feedback(Y, shape) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Validate feedback and return positive and observed interactions."""
    liked = to_L(Y)
    seen = to_B(Y)
    if liked.shape != shape or seen.shape != shape:
        raise ValueError(f"Y must be shape {shape}.")
    return liked, seen


def build_row_item_sets(matrix: sparse.csr_matrix) -> tuple[frozenset[int], ...]:
    """Return each CSR row's item indices as a membership set."""
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
    """Sample one nonexcluded item for every supplied user index."""
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
    """Sample user-positive-negative triples with users drawn uniformly."""
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
    """Resolve a PyTorch device and reject unavailable CUDA requests."""
    resolved = torch.device("cpu" if device is None else device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    return resolved


def torch_generator(seed: int, offset: int = 0) -> torch.Generator:
    """Create a deterministic CPU generator from a seed and offset."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed((int(seed) + int(offset)) % (2**63 - 1))
    return generator


def _build_normalized_adjacency(
    liked: sparse.csr_matrix,
    device: torch.device,
) -> torch.Tensor:
    """Build the symmetric degree-normalized user-item graph."""
    positive = liked.tocoo()
    user_count, item_count = liked.shape
    node_count = user_count + item_count
    user_nodes = positive.row.astype(np.int64, copy=False)
    item_nodes = user_count + positive.col.astype(np.int64, copy=False)
    source = np.concatenate([user_nodes, item_nodes])
    target = np.concatenate([item_nodes, user_nodes])
    degree = np.bincount(source, minlength=node_count).astype(np.float64)
    values = np.zeros(source.size, dtype=MODEL_DTYPE)
    if source.size:
        values = (
            1.0 / np.sqrt(degree[source] * degree[target])
        ).astype(MODEL_DTYPE, copy=False)
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
    """Hold trainable node embeddings and propagate them over the graph."""


    def __init__(self, user_count, item_count, latent, n_layers, random_state):
        """Initialize user and item embeddings deterministically."""
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


    def propagate(self, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Average initial and propagated embeddings across graph layers."""
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


def _bpr_batch_objective(
    network: _LightGCNNetwork,
    adjacency: torch.Tensor,
    users: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    lambda_rate: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute pairwise ranking and embedding regularization losses."""
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


    ranking_loss = -functional.logsigmoid(
        positive_scores - negative_scores
    ).mean()


    regularization = float(lambda_rate) * (
        network.user_embedding(users).square().sum(dim=1)
        + network.item_embedding(positives).square().sum(dim=1)
        + network.item_embedding(negatives).square().sum(dim=1)
    ).mean() * 0.5
    return ranking_loss, regularization


class LightGCN:
    """Expose LightGCN training and per-user scoring for sparse feedback."""


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
        """Validate hyperparameters and initialize the graph network."""
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
            dtype=MODEL_DTYPE,
        )
        self.item_factors = np.zeros(
            (self.item_count, self.latent),
            dtype=MODEL_DTYPE,
        )


    @property
    def shape(self) -> tuple[int, int]:
        """Return the expected user-item matrix shape."""
        return self.user_count, self.item_count


    def _refresh_factor_cache(self) -> None:
        """Cache fully propagated factors as NumPy arrays for scoring."""
        self.network.eval()
        with torch.no_grad():
            user_factors, item_factors = self.network.propagate(self.adjacency)
        self.user_factors = user_factors.detach().cpu().numpy().astype(
            MODEL_DTYPE,
            copy=True,
        )
        self.item_factors = item_factors.detach().cpu().numpy().astype(
            MODEL_DTYPE,
            copy=True,
        )


    def fit(self, Y, epochs=1000, batch_size=2048, verbose_every=10):
        """Fit graph embeddings with mini-batch BPR optimization."""
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

        # Resample training triples each epoch before mini-batch updates.
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


    def score_user(self, user_idx: int) -> np.ndarray:
        """Return sigmoid-transformed item scores for one user."""
        user_idx = int(user_idx)
        if user_idx < 0 or user_idx >= self.user_count:
            raise IndexError("user_idx is out of range.")
        logits = self.item_factors @ self.user_factors[user_idx]
        return expit(logits).astype(MODEL_DTYPE, copy=False)
