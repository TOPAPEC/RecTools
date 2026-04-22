"""Tests for UniSRecModel."""

import numpy as np
import pandas as pd
import pytest
import torch

from rectools import Columns
from rectools.dataset import Dataset
from rectools.fast_transformers import UniSRecConfig, UniSRecModel


def _make_dataset(n_users: int = 20, n_items: int = 25, seed: int = 42) -> Dataset:
    rng = np.random.RandomState(seed)
    rows = []
    for u in range(n_users):
        n_inter = rng.randint(3, 8)
        items = rng.choice(n_items, size=n_inter, replace=False)
        for rank, item in enumerate(items):
            rows.append({
                Columns.User: u,
                Columns.Item: item,
                Columns.Weight: 1.0,
                Columns.Datetime: pd.Timestamp("2024-01-01") + pd.Timedelta(hours=rank),
            })
    return Dataset.construct(pd.DataFrame(rows))


def _make_embeddings(n_items: int = 25, dim: int = 64) -> torch.Tensor:
    torch.manual_seed(0)
    emb = torch.randn(n_items, dim)
    emb[0] = 0.0
    return emb


def _make_model(**kwargs) -> UniSRecModel:
    defaults = dict(
        pretrained_item_embeddings=_make_embeddings(),
        n_factors=16,
        projection_hidden=32,
        n_blocks=1,
        n_heads=2,
        session_max_len=8,
        phase1_epochs=1,
        phase2_epochs=1,
        phase3_epochs=1,
        batch_size=16,
        verbose=0,
    )
    defaults.update(kwargs)
    return UniSRecModel(**defaults)


class TestFitRecommend:
    def test_recommend_columns(self) -> None:
        ds = _make_dataset()
        model = _make_model()
        model.fit(ds)
        users = list(range(5))
        reco = model.recommend(users=users, dataset=ds, k=3, filter_viewed=False)
        assert set(reco.columns) == {Columns.User, Columns.Item, Columns.Score, Columns.Rank}
        assert reco[Columns.User].nunique() == 5

    def test_filter_viewed(self) -> None:
        ds = _make_dataset()
        model = _make_model()
        model.fit(ds)
        users = list(range(5))
        reco = model.recommend(users=users, dataset=ds, k=5, filter_viewed=True)
        interactions = ds.get_raw_interactions()
        for uid in users:
            viewed = set(interactions[interactions[Columns.User] == uid][Columns.Item])
            recommended = set(reco[reco[Columns.User] == uid][Columns.Item])
            assert viewed.isdisjoint(recommended), f"User {uid} got viewed items"

    def test_i2i(self) -> None:
        ds = _make_dataset()
        model = _make_model()
        model.fit(ds)
        items = list(range(5))
        reco = model.recommend_to_items(target_items=items, dataset=ds, k=3)
        assert set(reco.columns) == {Columns.TargetItem, Columns.Item, Columns.Score, Columns.Rank}
        assert reco[Columns.TargetItem].nunique() == 5

    def test_scores_not_nan(self) -> None:
        ds = _make_dataset()
        model = _make_model(phase1_epochs=2, phase3_epochs=2)
        model.fit(ds)
        users = list(range(ds.user_id_map.size))
        reco = model.recommend(users=users, dataset=ds, k=5, filter_viewed=False)
        assert len(reco) > 0
        assert reco[Columns.Score].notna().all()


class TestPhaseSkipping:
    def test_skip_phase1(self) -> None:
        ds = _make_dataset()
        model = _make_model(phase1_epochs=0)
        model.fit(ds)
        reco = model.recommend(users=[0, 1], dataset=ds, k=3, filter_viewed=False)
        assert len(reco) > 0

    def test_skip_phase2(self) -> None:
        ds = _make_dataset()
        model = _make_model(phase2_epochs=0)
        model.fit(ds)
        reco = model.recommend(users=[0, 1], dataset=ds, k=3, filter_viewed=False)
        assert len(reco) > 0

    def test_only_phase3(self) -> None:
        ds = _make_dataset()
        model = _make_model(phase1_epochs=0, phase2_epochs=0, phase3_epochs=2)
        model.fit(ds)
        reco = model.recommend(users=[0, 1], dataset=ds, k=3, filter_viewed=False)
        assert len(reco) > 0


class TestWithNegatives:
    def test_sampled_loss(self) -> None:
        ds = _make_dataset()
        model = _make_model(n_negatives=4)
        model.fit(ds)
        reco = model.recommend(users=[0, 1, 2], dataset=ds, k=3, filter_viewed=False)
        assert len(reco) > 0


class TestConfig:
    def test_get_config(self) -> None:
        model = _make_model()
        config = model.get_config(mode="pydantic")
        assert config.model.n_factors == 16
        assert config.model.n_blocks == 1

    def test_from_config_raises(self) -> None:
        model = _make_model()
        config = model.get_config(mode="pydantic")
        with pytest.raises(NotImplementedError, match="pretrained_item_embeddings"):
            UniSRecModel.from_config(config)
