import unittest

import numpy as np
import pandas as pd
from scipy import sparse

from experiments.all_ranker import (
    build_user_activity_groups,
    ranking_metrics_at_k,
)
from experiments.global_temporal_split import (
    _activity_group_counts,
    _future_item_drop_stats,
    _index_and_build,
)
from model.base_model.wmf_cparms import WMF_CPARMS
from model.signal_gen.generator_cparms import Generator_CPARMS


class ColdStartSplitTests(unittest.TestCase):
    def test_evaluation_users_are_padded_but_future_items_are_removed(self):
        fit_events = pd.DataFrame(
            {
                "userid": ["u1", "u1", "u2"],
                "itemid": ["i1", "i2", "i1"],
                "rating": [5.0, 2.0, 4.0],
            }
        )
        eval_events = pd.DataFrame(
            {
                "userid": ["u2", "u3", "u4"],
                "itemid": ["i2", "i1", "i_future"],
                "rating": [5.0, 5.0, 5.0],
            }
        )

        fit_mat, eval_mat = _index_and_build(fit_events, eval_events)

        self.assertEqual(fit_mat.shape, (4, 2))
        self.assertEqual(eval_mat.shape, (4, 2))
        np.testing.assert_array_equal(fit_mat.getnnz(axis=1), [2, 1, 0, 0])
        np.testing.assert_array_equal(eval_mat.getnnz(axis=1), [0, 1, 1, 0])

        counts = _activity_group_counts(fit_mat, eval_mat)
        self.assertEqual(
            counts,
            {
                "interaction_0": 1,
                "interaction_1": 1,
                "interaction_2": 0,
                "interaction_3_plus": 0,
            },
        )
        self.assertEqual(
            _future_item_drop_stats(fit_events, eval_events),
            {
                "n_future_item_interactions_dropped": 1,
                "n_future_item_positive_targets_dropped": 1,
            },
        )


class ActivityGroupTests(unittest.TestCase):
    def test_groups_cover_zero_one_two_and_three_plus_observed_interactions(self):
        train_mat = sparse.csr_matrix(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                    [1, 2, 0, 0, 0, 0],
                    [1, 2, 3, 0, 0, 0],
                ],
                dtype=np.float32,
            )
        )
        test_mat = sparse.csr_matrix(
            np.array(
                [
                    [0, 0, 0, 0, 0, 5],
                    [0, 0, 0, 0, 0, 5],
                    [0, 0, 0, 0, 0, 5],
                    [0, 0, 0, 0, 0, 5],
                ],
                dtype=np.float32,
            )
        )
        scores = np.tile(np.arange(6, dtype=float), (4, 1))

        groups = build_user_activity_groups(train_mat)
        self.assertEqual({name: idx.tolist() for name, idx in groups.items()}, {
            "interaction_0": [0],
            "interaction_1": [1],
            "interaction_2": [2],
            "interaction_3_plus": [3],
        })

        metrics = ranking_metrics_at_k(
            pred_source=scores,
            train_mat=train_mat,
            test_mat=test_mat,
            user_groups=groups,
            ks=(1,),
        )
        self.assertEqual(metrics["n_users_eval"], 4)
        self.assertEqual(metrics["all"]["ndcg"][1], 1.0)
        for group_name in groups:
            self.assertEqual(metrics["user"][group_name]["ndcg"][1], 1.0)


class ColdStartLearningTests(unittest.TestCase):
    def test_padded_users_do_not_change_fitted_rule_statistics(self):
        active = sparse.csr_matrix(
            np.array(
                [
                    [5, 5, 0],
                    [5, 0, 5],
                    [5, 5, 0],
                ],
                dtype=np.float32,
            )
        )
        padded = sparse.vstack(
            [active, sparse.csr_matrix((2, 3), dtype=np.float32)],
            format="csr",
        )
        generator = Generator_CPARMS(
            k_user=1,
            K_item=None,
            normalize=None,
            random_state=7,
        )

        active_signal = generator.fit_transform(active)
        padded_signal = generator.fit_transform(padded)

        np.testing.assert_allclose(
            padded_signal[: active.shape[0]].toarray(),
            active_signal.toarray(),
        )
        self.assertGreater(padded_signal[active.shape[0] :].nnz, 0)

    def test_cold_signal_is_folded_in_without_updating_item_factors(self):
        interactions = sparse.csr_matrix(
            np.array(
                [
                    [5, 0],
                    [0, 5],
                    [0, 0],
                ],
                dtype=np.float32,
            )
        )
        cold_signal = sparse.csr_matrix(
            np.array(
                [
                    [0, 0],
                    [0, 0],
                    [1, 0],
                ],
                dtype=np.float32,
            )
        )
        active_interactions = interactions[:2].tocsr()
        active_signal = sparse.csr_matrix(active_interactions.shape, dtype=np.float32)
        fit_user_mask = np.array([True, True, False])

        with_signal = WMF_CPARMS(3, 2, 2, 0.1, 5.0, 1.0, random_state=11)
        fit_only = WMF_CPARMS(2, 2, 2, 0.1, 5.0, 1.0, random_state=11)
        with_signal.fit(
            interactions,
            cold_signal,
            n_sweeps=1,
            fit_user_mask=fit_user_mask,
        )
        fit_only.fit(
            active_interactions,
            active_signal,
            n_sweeps=1,
        )

        np.testing.assert_allclose(with_signal.Q, fit_only.Q)
        self.assertGreater(np.linalg.norm(with_signal.P[2]), 0.0)


if __name__ == "__main__":
    unittest.main()
