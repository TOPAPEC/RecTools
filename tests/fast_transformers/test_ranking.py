"""Tests for rectools.fast_transformers.ranking.rank_topk."""

import numpy as np
import pytest
import torch
from scipy import sparse

from rectools.fast_transformers.ranking import rank_topk


class TestRankTopk:
    """Tests for rank_topk function."""

    def _make_embeddings(self) -> tuple:
        """Create deterministic user/item embeddings for testing.

        3 users, 5 items, dimension 2.
        Scores matrix (user_embs @ item_embs.T):
            user0: [2, 5, 1, 4, 3]
            user1: [3, 1, 5, 2, 4]
            user2: [4, 3, 2, 5, 1]
        """
        # Construct embeddings so the dot-product scores are easy to reason about.
        # We use a trick: set item_embs to one-hot-ish vectors so each column
        # of the score matrix is directly controlled.
        item_embs = torch.eye(5, dtype=torch.float32)
        # user_embs rows are just the desired score rows
        user_embs = torch.tensor(
            [
                [2.0, 5.0, 1.0, 4.0, 3.0],
                [3.0, 1.0, 5.0, 2.0, 4.0],
                [4.0, 3.0, 2.0, 5.0, 1.0],
            ],
            dtype=torch.float32,
        )
        return user_embs, item_embs

    def test_basic_topk(self):
        """Top-k returns the correct items and scores for each user."""
        user_embs, item_embs = self._make_embeddings()
        k = 3
        user_ids, item_ids, scores = rank_topk(user_embs, item_embs, k)

        # user0 top-3: item1(5), item3(4), item4(3)
        # user1 top-3: item2(5), item4(4), item0(3)
        # user2 top-3: item3(5), item0(4), item1(3)
        expected_items = {
            0: [1, 3, 4],
            1: [2, 4, 0],
            2: [3, 0, 1],
        }
        expected_scores = {
            0: [5.0, 4.0, 3.0],
            1: [5.0, 4.0, 3.0],
            2: [5.0, 4.0, 3.0],
        }

        for uid in range(3):
            mask = user_ids == uid
            assert mask.sum() == k
            np.testing.assert_array_equal(item_ids[mask], expected_items[uid])
            np.testing.assert_array_almost_equal(scores[mask], expected_scores[uid])

    def test_output_shapes(self):
        """Output arrays all have length n_users * k."""
        user_embs, item_embs = self._make_embeddings()
        k = 2
        user_ids, item_ids, scores = rank_topk(user_embs, item_embs, k)

        n_users = user_embs.shape[0]
        expected_len = n_users * k
        assert len(user_ids) == expected_len
        assert len(item_ids) == expected_len
        assert len(scores) == expected_len

    def test_scores_sorted_descending_per_user(self):
        """Scores within each user block are in descending order."""
        user_embs, item_embs = self._make_embeddings()
        k = 4
        user_ids, item_ids, scores = rank_topk(user_embs, item_embs, k)

        for uid in range(user_embs.shape[0]):
            mask = user_ids == uid
            user_scores = scores[mask]
            assert np.all(user_scores[:-1] >= user_scores[1:]), (
                f"Scores for user {uid} are not in descending order: {user_scores}"
            )

    def test_filter_csr_excludes_viewed_items(self):
        """Items present in filter_csr are excluded from recommendations."""
        user_embs, item_embs = self._make_embeddings()
        k = 3

        # user0 has viewed item1 (their top item with score 5)
        # user1 has viewed item2 (their top item with score 5)
        filter_csr = sparse.csr_matrix(
            ([1, 1], ([0, 1], [1, 2])),
            shape=(3, 5),
        )

        user_ids, item_ids, scores = rank_topk(user_embs, item_embs, k, filter_csr=filter_csr)

        # user0: item1 excluded -> top-3: item3(4), item4(3), item0(2)
        mask0 = user_ids == 0
        np.testing.assert_array_equal(item_ids[mask0], [3, 4, 0])
        np.testing.assert_array_almost_equal(scores[mask0], [4.0, 3.0, 2.0])

        # user1: item2 excluded -> top-3: item4(4), item0(3), item3(2)
        mask1 = user_ids == 1
        np.testing.assert_array_equal(item_ids[mask1], [4, 0, 3])
        np.testing.assert_array_almost_equal(scores[mask1], [4.0, 3.0, 2.0])

        # user2: nothing excluded -> top-3: item3(5), item0(4), item1(3)
        mask2 = user_ids == 2
        np.testing.assert_array_equal(item_ids[mask2], [3, 0, 1])
        np.testing.assert_array_almost_equal(scores[mask2], [5.0, 4.0, 3.0])

    def test_whitelist_restricts_items(self):
        """Only whitelisted items appear in results, but with original indices."""
        user_embs, item_embs = self._make_embeddings()
        k = 2

        # Only consider items 0, 2, 4
        whitelist = np.array([0, 2, 4])
        user_ids, item_ids, scores = rank_topk(user_embs, item_embs, k, whitelist=whitelist)

        for uid in range(3):
            mask = user_ids == uid
            # All returned items must be in the whitelist
            assert set(item_ids[mask]).issubset(set(whitelist))

        # user0 scores on [0,2,4]: [2,1,3] -> top-2: item4(3), item0(2)
        mask0 = user_ids == 0
        np.testing.assert_array_equal(item_ids[mask0], [4, 0])
        np.testing.assert_array_almost_equal(scores[mask0], [3.0, 2.0])

        # user1 scores on [0,2,4]: [3,5,4] -> top-2: item2(5), item4(4)
        mask1 = user_ids == 1
        np.testing.assert_array_equal(item_ids[mask1], [2, 4])
        np.testing.assert_array_almost_equal(scores[mask1], [5.0, 4.0])

    def test_filter_csr_and_whitelist_combined(self):
        """filter_csr and whitelist work correctly together."""
        user_embs, item_embs = self._make_embeddings()
        k = 2

        # Whitelist: items 0, 1, 3
        whitelist = np.array([0, 1, 3])

        # user0 viewed item1 (top item in whitelist)
        filter_csr = sparse.csr_matrix(
            ([1], ([0], [1])),
            shape=(3, 5),
        )

        user_ids, item_ids, scores = rank_topk(
            user_embs, item_embs, k, filter_csr=filter_csr, whitelist=whitelist
        )

        # user0 whitelist scores: item0(2), item1(5), item3(4)
        # After filter (item1 excluded): item0(2), item3(4)
        # top-2: item3(4), item0(2)
        mask0 = user_ids == 0
        np.testing.assert_array_equal(item_ids[mask0], [3, 0])
        np.testing.assert_array_almost_equal(scores[mask0], [4.0, 2.0])

        # user1 no items filtered, whitelist scores: item0(3), item1(1), item3(2)
        # top-2: item0(3), item3(2)
        mask1 = user_ids == 1
        np.testing.assert_array_equal(item_ids[mask1], [0, 3])
        np.testing.assert_array_almost_equal(scores[mask1], [3.0, 2.0])

    def test_k_greater_than_n_items(self):
        """When k > n_items, returns all items per user."""
        user_embs, item_embs = self._make_embeddings()
        n_items = item_embs.shape[0]
        k = n_items + 10  # Much larger than n_items

        user_ids, item_ids, scores = rank_topk(user_embs, item_embs, k)

        # Should return n_items results per user, not k
        n_users = user_embs.shape[0]
        assert len(user_ids) == n_users * n_items
        assert len(item_ids) == n_users * n_items
        assert len(scores) == n_users * n_items

        # Check that all items appear for each user
        for uid in range(n_users):
            mask = user_ids == uid
            assert sorted(item_ids[mask]) == list(range(n_items))

    def test_k_greater_than_n_items_with_whitelist(self):
        """When k > len(whitelist), returns len(whitelist) items per user."""
        user_embs, item_embs = self._make_embeddings()
        whitelist = np.array([1, 3])
        k = 10

        user_ids, item_ids, scores = rank_topk(user_embs, item_embs, k, whitelist=whitelist)

        n_users = user_embs.shape[0]
        assert len(user_ids) == n_users * len(whitelist)

        for uid in range(n_users):
            mask = user_ids == uid
            assert set(item_ids[mask]) == set(whitelist)

    def test_batch_size_does_not_affect_results(self):
        """Different batch sizes produce identical results."""
        user_embs, item_embs = self._make_embeddings()
        k = 3

        uid_full, iid_full, sc_full = rank_topk(user_embs, item_embs, k, batch_size=256)
        uid_bs1, iid_bs1, sc_bs1 = rank_topk(user_embs, item_embs, k, batch_size=1)
        uid_bs2, iid_bs2, sc_bs2 = rank_topk(user_embs, item_embs, k, batch_size=2)

        np.testing.assert_array_equal(uid_full, uid_bs1)
        np.testing.assert_array_equal(iid_full, iid_bs1)
        np.testing.assert_array_almost_equal(sc_full, sc_bs1)

        np.testing.assert_array_equal(uid_full, uid_bs2)
        np.testing.assert_array_equal(iid_full, iid_bs2)
        np.testing.assert_array_almost_equal(sc_full, sc_bs2)

    def test_batch_size_with_filter_and_whitelist(self):
        """Batch processing gives same results with filter_csr and whitelist."""
        user_embs, item_embs = self._make_embeddings()
        k = 2
        whitelist = np.array([0, 2, 4])
        filter_csr = sparse.csr_matrix(
            ([1, 1], ([0, 2], [0, 4])),
            shape=(3, 5),
        )

        uid_full, iid_full, sc_full = rank_topk(
            user_embs, item_embs, k, filter_csr=filter_csr, whitelist=whitelist, batch_size=256
        )
        uid_bs1, iid_bs1, sc_bs1 = rank_topk(
            user_embs, item_embs, k, filter_csr=filter_csr, whitelist=whitelist, batch_size=1
        )

        np.testing.assert_array_equal(uid_full, uid_bs1)
        np.testing.assert_array_equal(iid_full, iid_bs1)
        np.testing.assert_array_almost_equal(sc_full, sc_bs1)

    def test_multiple_users_independent_topk(self):
        """Each user gets their own independent top-k based on their embeddings."""
        user_embs, item_embs = self._make_embeddings()
        k = 1

        user_ids, item_ids, scores = rank_topk(user_embs, item_embs, k)

        # Each user should get exactly 1 result
        assert len(user_ids) == 3
        np.testing.assert_array_equal(user_ids, [0, 1, 2])

        # Best items: user0->item1(5), user1->item2(5), user2->item3(5)
        np.testing.assert_array_equal(item_ids, [1, 2, 3])
        np.testing.assert_array_almost_equal(scores, [5.0, 5.0, 5.0])

    def test_single_user(self):
        """Works correctly with a single user."""
        user_embs = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
        item_embs = torch.tensor(
            [[3.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        k = 2

        user_ids, item_ids, scores = rank_topk(user_embs, item_embs, k)

        np.testing.assert_array_equal(user_ids, [0, 0])
        np.testing.assert_array_equal(item_ids, [0, 2])
        np.testing.assert_array_almost_equal(scores, [3.0, 2.0])

    def test_single_item(self):
        """Works correctly with a single item."""
        user_embs = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        item_embs = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        k = 5  # k > n_items

        user_ids, item_ids, scores = rank_topk(user_embs, item_embs, k)

        # Only 1 item, so each user gets 1 result
        assert len(user_ids) == 2
        np.testing.assert_array_equal(user_ids, [0, 1])
        np.testing.assert_array_equal(item_ids, [0, 0])
        np.testing.assert_array_almost_equal(scores, [3.0, 7.0])

    def test_user_ids_are_sequential_indices(self):
        """Returned user_ids are sequential integer indices starting from 0."""
        user_embs, item_embs = self._make_embeddings()
        k = 2

        user_ids, _, _ = rank_topk(user_embs, item_embs, k)

        # user_ids should be [0,0, 1,1, 2,2]
        expected = np.repeat(np.arange(3), k)
        np.testing.assert_array_equal(user_ids, expected)

    def test_return_types_are_numpy(self):
        """All returned arrays are numpy ndarrays."""
        user_embs, item_embs = self._make_embeddings()
        k = 2

        user_ids, item_ids, scores = rank_topk(user_embs, item_embs, k)

        assert isinstance(user_ids, np.ndarray)
        assert isinstance(item_ids, np.ndarray)
        assert isinstance(scores, np.ndarray)

    def test_filter_all_items_for_user(self):
        """When all items are filtered for a user, scores are -inf."""
        user_embs = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        item_embs = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        k = 1

        # Filter all items for user 0
        filter_csr = sparse.csr_matrix(
            ([1, 1], ([0, 0], [0, 1])),
            shape=(2, 2),
        )

        user_ids, item_ids, scores = rank_topk(user_embs, item_embs, k, filter_csr=filter_csr)

        # user0: all filtered -> score is -inf
        mask0 = user_ids == 0
        assert np.all(np.isneginf(scores[mask0]))

        # user1: nothing filtered -> normal result
        mask1 = user_ids == 1
        assert scores[mask1][0] == pytest.approx(1.0)
