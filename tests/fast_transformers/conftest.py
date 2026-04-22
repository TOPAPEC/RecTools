"""Fixtures for fast_transformers tests."""

import numpy as np
import pandas as pd
import pytest

from rectools import Columns
from rectools.dataset import Dataset


@pytest.fixture()
def tiny_dataset() -> Dataset:
    """20 users x 25 items, each user has 3-8 interactions."""
    rng = np.random.RandomState(42)
    n_users, n_items = 20, 25

    rows = []
    for u in range(n_users):
        n_inter = rng.randint(3, 9)
        items = rng.choice(n_items, size=n_inter, replace=False)
        for rank, item in enumerate(items):
            rows.append(
                {
                    Columns.User: u,
                    Columns.Item: item,
                    Columns.Weight: 1.0,
                    Columns.Datetime: pd.Timestamp("2023-01-01") + pd.Timedelta(days=rank),
                }
            )
    df = pd.DataFrame(rows)
    return Dataset.construct(df)
