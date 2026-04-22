"""Tests for FlatSASRecModel."""

import pickle

import numpy as np
import pandas as pd
import pytest

from rectools import Columns
from rectools.dataset import Dataset
from rectools.fast_transformers import FlatSASRecConfig, FlatSASRecModel


def _make_model(**kwargs) -> FlatSASRecModel:
    defaults = dict(
        n_factors=16, n_blocks=1, n_heads=2, session_max_len=8,
        epochs=1, batch_size=16, lr=1e-3, verbose=0,
    )
    defaults.update(kwargs)
    return FlatSASRecModel(**defaults)


class TestFitRecommend:
    def test_recommend_columns(self, tiny_dataset: Dataset) -> None:
        model = _make_model()
        model.fit(tiny_dataset)
        users = list(range(5))
        reco = model.recommend(users=users, dataset=tiny_dataset, k=3, filter_viewed=False)
        assert set(reco.columns) == {Columns.User, Columns.Item, Columns.Score, Columns.Rank}
        assert reco[Columns.User].nunique() == 5

    def test_filter_viewed(self, tiny_dataset: Dataset) -> None:
        model = _make_model()
        model.fit(tiny_dataset)
        users = list(range(5))
        reco = model.recommend(users=users, dataset=tiny_dataset, k=5, filter_viewed=True)
        interactions = tiny_dataset.get_raw_interactions()
        for uid in users:
            viewed = set(interactions[interactions[Columns.User] == uid][Columns.Item])
            recommended = set(reco[reco[Columns.User] == uid][Columns.Item])
            assert viewed.isdisjoint(recommended), f"User {uid} got viewed items in recommendations"

    def test_i2i(self, tiny_dataset: Dataset) -> None:
        model = _make_model()
        model.fit(tiny_dataset)
        items = list(range(5))
        reco = model.recommend_to_items(target_items=items, dataset=tiny_dataset, k=3)
        assert set(reco.columns) == {Columns.TargetItem, Columns.Item, Columns.Score, Columns.Rank}
        assert reco[Columns.TargetItem].nunique() == 5

    def test_metrics_positive(self, tiny_dataset: Dataset) -> None:
        model = _make_model(epochs=3)
        model.fit(tiny_dataset)
        users = list(range(tiny_dataset.user_id_map.size))
        reco = model.recommend(users=users, dataset=tiny_dataset, k=5, filter_viewed=False)
        assert len(reco) > 0
        assert reco[Columns.Score].notna().all()


class TestConfig:
    def test_config_roundtrip(self) -> None:
        model = _make_model(n_factors=32, n_blocks=3)
        config = model.get_config(mode="pydantic")
        model2 = FlatSASRecModel.from_config(config)
        assert model2.n_factors == 32
        assert model2.n_blocks == 3

    def test_pickle_roundtrip(self, tiny_dataset: Dataset) -> None:
        model = _make_model()
        model.fit(tiny_dataset)
        data = pickle.dumps(model)
        model2 = pickle.loads(data)
        assert model2.is_fitted
        users = list(range(3))
        reco = model2.recommend(users=users, dataset=tiny_dataset, k=3, filter_viewed=False)
        assert len(reco) > 0


class TestLosses:
    def test_bce_training(self, tiny_dataset: Dataset) -> None:
        model = _make_model(loss="BCE", n_negatives=2)
        model.fit(tiny_dataset)
        users = list(range(5))
        reco = model.recommend(users=users, dataset=tiny_dataset, k=3, filter_viewed=False)
        assert len(reco) > 0

    def test_invalid_loss(self) -> None:
        with pytest.raises(ValueError, match="Unsupported loss"):
            _make_model(loss="invalid_loss_name")
