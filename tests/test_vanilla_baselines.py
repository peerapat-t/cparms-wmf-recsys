import contextlib
import io
import unittest

import numpy as np
import torch
from scipy import sparse

from experiments.hyperparams_set import generate_hyperparam_samples
from model.base_model.lightgcn import (
    LightGCN,
    _bpr_batch_objective,
    _build_normalized_adjacency,
    build_row_item_sets as build_lightgcn_row_item_sets,
    prepare_feedback as prepare_lightgcn_feedback,
    sample_uniform_user_triples,
)
from model.base_model.neumf import (
    NeuMF,
    build_row_item_sets as build_neumf_row_item_sets,
    prepare_feedback as prepare_neumf_feedback,
    sample_complement_items,
)
from model.base_model.wmf_cofactor import (
    _cooccurrence_loss,
    _context_bias_update,
    _context_factor_update,
    _global_bias_update,
    _item_bias_update,
    _item_factor_update,
)
from model.base_model.wmf_standard import _als_factor_update, _wmf_loss
from model.signal_gen.generator_sppmi import build_item_sppmi_matrix


def _tiny_feedback():
    return sparse.csr_matrix(
        np.array(
            [
                [5, 5, 0, 0],
                [5, 0, 5, 0],
                [0, 5, 5, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
    )


class VanillaBaselineTests(unittest.TestCase):
    def test_standard_wmf_update_matches_hu_normal_equation(self):
        positive = sparse.csr_matrix(
            np.array([[1.0, 0.0]], dtype=np.float32)
        )
        fixed = np.array(
            [[1.0, 0.0], [0.0, 2.0]],
            dtype=np.float32,
        )
        actual = _als_factor_update(
            positive,
            fixed,
            lambda_rate=0.5,
            alpha=3.0,
        )
        expected = np.linalg.solve(
            np.diag([4.5, 4.5]).astype(np.float32),
            np.array([4.0, 0.0], dtype=np.float32),
        )
        np.testing.assert_allclose(actual[0], expected, rtol=1e-6)

    def test_standard_wmf_loss_matches_dense_confidence_objective(self):
        positive = sparse.csr_matrix(
            np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
        )
        users = np.array([[0.2, -0.1], [0.4, 0.3]], dtype=np.float32)
        items = np.array(
            [[0.5, 0.2], [-0.2, 0.6], [0.1, 0.7]],
            dtype=np.float32,
        )
        alpha = 2.5
        target = positive.toarray()
        confidence = 1.0 + alpha * target
        expected = np.sum(confidence * (target - users @ items.T) ** 2)
        actual = _wmf_loss(users, items, positive, alpha)
        self.assertAlmostEqual(actual, float(expected), places=5)

    def test_cofactor_item_and_context_updates_match_upstream_equations(self):
        # Upstream ML20M scaling: c0=.03, c1=.30, ell=.03. Dividing
        # the full objective by c0 gives local alpha=9 and gamma=1/c0.
        c0, c1 = 0.03, 0.30
        gamma = 1.0 / c0
        alpha = c1 / c0 - 1.0
        lambda_beta_upstream = 1e-5 * c0
        lambda_beta = lambda_beta_upstream / c0
        lambda_gamma_upstream = 1e-5
        lambda_context = lambda_gamma_upstream / c0

        L_t = sparse.csr_matrix(
            np.array([[1, 0], [1, 1], [0, 1]], dtype=np.float32)
        )
        theta = np.array([[0.2, 0.7], [0.6, -0.1]], dtype=np.float32)
        context = np.array(
            [[0.3, -0.2], [0.5, 0.4], [-0.1, 0.8]],
            dtype=np.float32,
        )
        cooccurrence = sparse.csr_matrix(
            np.array(
                [[0.0, 1.2, 0.4], [1.2, 0.0, 0.7], [0.4, 0.7, 0.0]],
                dtype=np.float32,
            )
        )
        item_bias = np.array([0.1, -0.2, 0.05], dtype=np.float32)
        context_bias = np.array([-0.1, 0.2, 0.03], dtype=np.float32)
        global_bias = 0.15

        actual_items = _item_factor_update(
            L_t,
            theta,
            context,
            item_bias,
            context_bias,
            global_bias,
            cooccurrence,
            lambda_beta,
            alpha,
            gamma,
        )
        expected_items = np.empty_like(actual_items)
        for item_idx in range(L_t.shape[0]):
            user_idx = L_t.indices[L_t.indptr[item_idx] : L_t.indptr[item_idx + 1]]
            context_idx = cooccurrence.indices[
                cooccurrence.indptr[item_idx] : cooccurrence.indptr[item_idx + 1]
            ]
            target = cooccurrence.data[
                cooccurrence.indptr[item_idx] : cooccurrence.indptr[item_idx + 1]
            ]
            T = theta[user_idx]
            G = context[context_idx]
            residual = (
                target
                - item_bias[item_idx]
                - context_bias[context_idx]
                - global_bias
            )
            system = (
                c0 * (theta.T @ theta)
                + lambda_beta_upstream * np.eye(theta.shape[1])
                + (c1 - c0) * (T.T @ T)
                + G.T @ G
            )
            rhs = c1 * T.sum(axis=0) + residual @ G
            expected_items[item_idx] = np.linalg.solve(system, rhs)
        np.testing.assert_allclose(actual_items, expected_items, rtol=2e-4, atol=2e-4)

        cooccurrence_t = cooccurrence.T.tocsr()
        actual_context = _context_factor_update(
            actual_items,
            item_bias,
            context_bias,
            global_bias,
            cooccurrence_t,
            lambda_context,
            gamma,
        )
        expected_context = np.empty_like(actual_context)
        for context_idx in range(cooccurrence_t.shape[0]):
            item_idx = cooccurrence_t.indices[
                cooccurrence_t.indptr[context_idx] : cooccurrence_t.indptr[context_idx + 1]
            ]
            target = cooccurrence_t.data[
                cooccurrence_t.indptr[context_idx] : cooccurrence_t.indptr[context_idx + 1]
            ]
            B = actual_items[item_idx]
            residual = (
                target
                - item_bias[item_idx]
                - context_bias[context_idx]
                - global_bias
            )
            system = B.T @ B + lambda_gamma_upstream * np.eye(B.shape[1])
            expected_context[context_idx] = np.linalg.solve(
                system,
                residual @ B,
            )
        np.testing.assert_allclose(
            actual_context,
            expected_context,
            rtol=2e-4,
            atol=2e-4,
        )

    def test_cofactor_sppmi_matches_upstream_directed_pair_counts(self):
        feedback = sparse.csr_matrix(
            np.array(
                [[5, 5, 5], [5, 5, 0], [0, 5, 5]],
                dtype=np.float32,
            )
        )
        actual = build_item_sppmi_matrix(feedback, negative_samples=1.2)
        counts = np.array(
            [[0, 2, 1], [2, 0, 2], [1, 2, 0]],
            dtype=np.float64,
        )
        marginals = counts.sum(axis=1)
        expected = np.zeros_like(counts)
        nonzero = counts > 0
        rows, cols = np.nonzero(nonzero)
        expected[rows, cols] = np.maximum(
            np.log(
                counts[rows, cols]
                * counts.sum()
                / (marginals[rows] * marginals[cols])
            )
            - np.log(1.2),
            0.0,
        )
        np.testing.assert_allclose(actual.toarray(), expected, rtol=1e-6, atol=1e-7)

    def test_cofactor_global_bias_matches_upstream_residual_mean(self):
        item_factors = np.zeros((2, 2), dtype=np.float32)
        context_factors = np.zeros((2, 2), dtype=np.float32)
        item_biases = np.array([1.0, 2.0], dtype=np.float32)
        context_biases = np.array([0.5, -0.5], dtype=np.float32)
        cooccurrence = sparse.csr_matrix(
            np.array([[0.0, 3.0], [5.0, 0.0]], dtype=np.float32)
        )
        global_bias = _global_bias_update(
            item_factors,
            context_factors,
            item_biases,
            context_biases,
            cooccurrence,
        )
        self.assertAlmostEqual(float(global_bias), 2.5)
        self.assertAlmostEqual(
            _cooccurrence_loss(
                item_factors,
                context_factors,
                item_biases,
                context_biases,
                global_bias,
                cooccurrence,
                gamma=1.0,
            ),
            0.0,
        )

    def test_cofactor_bias_updates_match_unregularized_residual_means(self):
        item_factors = np.array([[0.2, 0.4], [0.5, -0.1]], dtype=np.float32)
        context_factors = np.array([[0.3, 0.2], [-0.2, 0.6]], dtype=np.float32)
        cooccurrence = sparse.csr_matrix(
            np.array([[0.0, 1.4], [0.8, 0.0]], dtype=np.float32)
        )
        context_biases = np.array([0.1, -0.2], dtype=np.float32)
        global_bias = 0.05
        item_biases = _item_bias_update(
            item_factors,
            context_factors,
            cooccurrence,
            context_biases,
            global_bias,
        )
        expected_item = np.array(
            [
                1.4 - item_factors[0] @ context_factors[1] - context_biases[1] - global_bias,
                0.8 - item_factors[1] @ context_factors[0] - context_biases[0] - global_bias,
            ]
        )
        np.testing.assert_allclose(item_biases, expected_item, rtol=1e-6)

        actual_context = _context_bias_update(
            item_factors,
            context_factors,
            cooccurrence.T.tocsr(),
            item_biases,
            global_bias,
        )
        expected_context = np.array(
            [
                0.8 - item_factors[1] @ context_factors[0] - item_biases[1] - global_bias,
                1.4 - item_factors[0] @ context_factors[1] - item_biases[0] - global_bias,
            ]
        )
        np.testing.assert_allclose(actual_context, expected_context, atol=1e-6)

    def test_negative_sampler_can_use_seen_nonpositive_items(self):
        feedback = sparse.csr_matrix(
            np.array([[5, 2]], dtype=np.float32)
        )
        liked, seen = prepare_neumf_feedback(feedback, feedback.shape)
        self.assertEqual(liked.nnz, 1)
        self.assertEqual(seen.nnz, 2)

        users = np.zeros(8, dtype=np.int64)
        sampled = sample_complement_items(
            users,
            build_neumf_row_item_sets(liked),
            item_count=2,
            rng=np.random.default_rng(11),
        )
        np.testing.assert_array_equal(sampled, np.ones(8, dtype=np.int64))

    def test_neumf_is_reproducible_and_scores_are_finite(self):
        feedback = _tiny_feedback()
        first = NeuMF(
            4,
            4,
            latent=4,
            hidden_layers=(8, 4),
            negative_samples=1,
            random_state=23,
        )
        second = NeuMF(
            4,
            4,
            latent=4,
            hidden_layers=(8, 4),
            negative_samples=1,
            random_state=23,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            first.fit(feedback, epochs=2, batch_size=2, verbose_every=1)
            second.fit(feedback, epochs=2, batch_size=2, verbose_every=1)

        np.testing.assert_allclose(first.score_user(0), second.score_user(0))
        self.assertTrue(np.isfinite(first.score_user(0)).all())
        self.assertTrue(np.isfinite(first.score_user(3)).all())
        self.assertTrue(np.any(first.score_user(3) != 0.0))

    def test_neumf_uses_canonical_separate_mlp_embedding_width(self):
        model = NeuMF(
            4,
            5,
            latent=8,
            hidden_layers=(64, 32, 16, 8),
            random_state=23,
        )
        self.assertEqual(model.network.gmf_user.embedding_dim, 8)
        self.assertEqual(model.network.mlp_user.embedding_dim, 32)
        self.assertEqual(model.network.mlp[0].in_features, 64)
        self.assertEqual(model.network.mlp[0].out_features, 32)
        self.assertEqual(model.network.output.in_features, 16)

        with self.assertRaisesRegex(ValueError, "must be even"):
            NeuMF(4, 5, hidden_layers=(63, 32, 16, 8))

        prediction_bound = np.sqrt(3.0 / model.network.output.in_features)
        output_weight = model.network.output.weight.detach().cpu().numpy()
        self.assertTrue(np.all(np.abs(output_weight) <= prediction_bound))

        users = torch.tensor([0, 2], dtype=torch.long)
        items = torch.tensor([1, 4], dtype=torch.long)
        with torch.no_grad():
            gmf = model.network.gmf_user(users) * model.network.gmf_item(items)
            mlp_input = torch.cat(
                [model.network.mlp_user(users), model.network.mlp_item(items)],
                dim=1,
            )
            mlp = model.network.mlp(mlp_input)
            expected = model.network.output(torch.cat([gmf, mlp], dim=1)).reshape(-1)
            actual = model.network(users, items)
        torch.testing.assert_close(actual, expected)

    def test_lightgcn_is_reproducible_and_scores_are_finite(self):
        feedback = _tiny_feedback()
        first = LightGCN(4, 4, latent=4, n_layers=2, random_state=29)
        second = LightGCN(4, 4, latent=4, n_layers=2, random_state=29)
        with contextlib.redirect_stdout(io.StringIO()):
            first.fit(feedback, epochs=2, batch_size=2, verbose_every=1)
            second.fit(feedback, epochs=2, batch_size=2, verbose_every=1)

        np.testing.assert_allclose(first.score_user(0), second.score_user(0))
        self.assertTrue(np.isfinite(first.score_user(0)).all())
        self.assertTrue(np.isfinite(first.score_user(3)).all())
        self.assertTrue(np.any(first.score_user(3) != 0.0))
        self.assertEqual(first.adjacency._nnz(), 2 * feedback.nnz)

    def test_lightgcn_reference_sampler_and_initialization(self):
        feedback = _tiny_feedback()
        liked, _ = prepare_lightgcn_feedback(feedback, feedback.shape)
        users, positives, negatives = sample_uniform_user_triples(
            liked,
            np.random.default_rng(31),
        )
        positive_sets = build_lightgcn_row_item_sets(liked)
        self.assertGreater(users.size, 0)
        for user_idx, positive_idx, negative_idx in zip(
            users,
            positives,
            negatives,
        ):
            self.assertIn(int(positive_idx), positive_sets[int(user_idx)])
            self.assertNotIn(int(negative_idx), positive_sets[int(user_idx)])

        model = LightGCN(1000, 1000, latent=16, random_state=29)
        user_std = float(
            model.network.user_embedding.weight.detach().cpu().std()
        )
        self.assertAlmostEqual(user_std, 0.1, delta=0.005)

    def test_lightgcn_graph_propagation_and_objective_match_upstream(self):
        liked = sparse.csr_matrix(
            np.array(
                [[1, 1, 0], [0, 1, 1]],
                dtype=np.float32,
            )
        )
        graph = _build_normalized_adjacency(liked, torch.device("cpu"))
        dense_interactions = liked.toarray()
        adjacency = np.block(
            [
                [np.zeros((2, 2), dtype=np.float32), dense_interactions],
                [dense_interactions.T, np.zeros((3, 3), dtype=np.float32)],
            ]
        )
        degree = adjacency.sum(axis=1)
        inverse_sqrt = np.zeros_like(degree)
        inverse_sqrt[degree > 0] = degree[degree > 0] ** -0.5
        expected_graph = (
            inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]
        )
        np.testing.assert_allclose(
            graph.to_dense().numpy(),
            expected_graph,
            rtol=1e-6,
            atol=1e-7,
        )

        model = LightGCN(2, 3, latent=4, n_layers=2, random_state=37)
        with torch.no_grad():
            propagated_users, propagated_items = model.network.propagate(graph)
            embedding = torch.cat(
                [model.network.user_embedding.weight, model.network.item_embedding.weight]
            )
            layers = [embedding]
            for _ in range(2):
                embedding = graph.to_dense() @ embedding
                layers.append(embedding)
            expected_embedding = torch.stack(layers).mean(dim=0)
        torch.testing.assert_close(
            torch.cat([propagated_users, propagated_items]),
            expected_embedding,
        )

        users = torch.tensor([0, 1], dtype=torch.long)
        positives = torch.tensor([0, 2], dtype=torch.long)
        negatives = torch.tensor([2, 0], dtype=torch.long)
        ranking, regularization = _bpr_batch_objective(
            model.network,
            graph,
            users,
            positives,
            negatives,
            lambda_rate=1e-4,
        )
        user_factors, item_factors = model.network.propagate(graph)
        positive_scores = (user_factors[users] * item_factors[positives]).sum(1)
        negative_scores = (user_factors[users] * item_factors[negatives]).sum(1)
        expected_ranking = torch.nn.functional.softplus(
            negative_scores - positive_scores
        ).mean()
        expected_regularization = 0.5e-4 * (
            model.network.user_embedding(users).norm().square()
            + model.network.item_embedding(positives).norm().square()
            + model.network.item_embedding(negatives).norm().square()
        ) / users.numel()
        torch.testing.assert_close(ranking, expected_ranking)
        torch.testing.assert_close(regularization, expected_regularization)

    def test_hyperparameter_samples_include_all_vanilla_models(self):
        samples = generate_hyperparam_samples(20, global_seed=42)
        sample = samples[0]
        self.assertIn("neumf", sample)
        self.assertIn("lightgcn", sample)
        self.assertEqual(sample["neumf"]["random_state"], 42)
        self.assertEqual(sample["lightgcn"]["random_state"], 42)
        for sampled in samples:
            neumf = sampled["neumf"]
            factor = neumf["latent"]
            self.assertEqual(
                neumf["hidden_layers"],
                tuple(factor * scale for scale in (8, 4, 2, 1)),
            )
            self.assertIn(neumf["epochs"], (20, 30, 50))
            self.assertEqual(neumf["negative_samples"], 4)
            self.assertEqual(neumf["batch_size"], 256)
            self.assertEqual(neumf["reg_mf"], 0.0)
            self.assertEqual(neumf["reg_layers"], (0.0, 0.0, 0.0, 0.0))

            lightgcn = sampled["lightgcn"]
            self.assertIn(lightgcn["epochs"], (30, 50, 100))
            self.assertEqual(lightgcn["batch_size"], 2048)
            self.assertIn(lightgcn["learning_rate"], (0.0005, 0.001))

            cofactor = sampled["cofactor_wmf"]
            self.assertIn(cofactor["latent"], (50, 100))
            self.assertIn(cofactor["n_sweeps"], (10, 15, 20))
            self.assertIn(cofactor["alpha"], (4.0, 9.0, 19.0, 39.0))

            cparms_wmf = sampled["cparms_wmf"]
            for key in (
                "k_user",
                "K_item",
                "min_support",
                "min_confidence",
                "min_lift",
                "normalize",
                "random_state",
            ):
                self.assertIn(key, cparms_wmf)

        reference = samples[0]
        self.assertEqual(reference["neumf"]["hidden_layers"], (64, 32, 16, 8))
        self.assertEqual(reference["neumf"]["epochs"], 50)
        self.assertEqual(reference["neumf"]["batch_size"], 256)
        self.assertEqual(reference["lightgcn"]["epochs"], 100)
        self.assertEqual(reference["lightgcn"]["batch_size"], 2048)
        self.assertEqual(reference["cofactor_wmf"]["latent"], 100)
        self.assertEqual(reference["cofactor_wmf"]["alpha"], 9.0)


if __name__ == "__main__":
    unittest.main()
