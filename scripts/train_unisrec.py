"""End-to-end smoke test for UniSRecModel with synthetic data and fake embeddings."""

import numpy as np
import pandas as pd
import torch

from rectools import Columns
from rectools.dataset import Dataset
from rectools.fast_transformers import UniSRecModel


def main() -> None:
    # --- Synthetic dataset: 80 users x 60 items ---
    rng = np.random.RandomState(123)
    n_users, n_items = 80, 60

    rows = []
    for u in range(n_users):
        n_inter = rng.randint(4, 15)
        items = rng.choice(n_items, size=n_inter, replace=False)
        for rank, item in enumerate(items):
            rows.append({
                Columns.User: u,
                Columns.Item: item,
                Columns.Weight: 1.0,
                Columns.Datetime: pd.Timestamp("2024-01-01") + pd.Timedelta(hours=rank),
            })
    df = pd.DataFrame(rows)
    dataset = Dataset.construct(df)
    print(f"Dataset: {n_users} users, {n_items} items, {len(df)} interactions")

    # --- Fake pretrained embeddings (random, shape [n_items, 64]) ---
    torch.manual_seed(42)
    pretrained = torch.randn(n_items, 64)

    # --- Train ---
    model = UniSRecModel(
        pretrained_item_embeddings=pretrained,
        n_factors=32,
        projection_hidden=64,
        n_blocks=2,
        n_heads=2,
        session_max_len=16,
        phase1_epochs=2,
        phase2_epochs=2,
        phase3_epochs=2,
        phase1_lr=1e-3,
        phase2_lr=3e-4,
        phase3_lr=1e-4,
        batch_size=32,
        verbose=1,
    )
    model.fit(dataset)
    print("Training done (3 phases).")

    # --- Recommend ---
    users = list(range(n_users))
    reco = model.recommend(users=users, dataset=dataset, k=5, filter_viewed=True)
    print(f"\nTop-5 recommendations (first 3 users):")
    print(reco[reco[Columns.User].isin(range(3))].to_string(index=False))

    # --- Simple metrics ---
    interactions = dataset.get_raw_interactions()
    hits = 0
    total = 0
    ap_sum = 0.0
    for u in users:
        viewed = set(interactions[interactions[Columns.User] == u][Columns.Item])
        rec_items = reco[reco[Columns.User] == u][Columns.Item].tolist()
        rel = [1 if i in viewed else 0 for i in rec_items]
        hits += sum(rel)
        total += len(rec_items)
        if sum(rel) > 0:
            precision_at = np.cumsum(rel) / np.arange(1, len(rel) + 1)
            ap_sum += np.sum(precision_at * rel) / sum(rel)
    recall = hits / max(total, 1)
    map_at_k = ap_sum / len(users)
    print(f"\nRecall@5 (train overlap): {recall:.4f}")
    print(f"MAP@5 (train overlap): {map_at_k:.4f}")

    # --- NaN check ---
    nan_count = reco[Columns.Score].isna().sum()
    print(f"NaN scores: {nan_count} / {len(reco)}")
    assert nan_count == 0, "Found NaN scores!"

    # --- I2I ---
    target_items = list(range(10))
    i2i = model.recommend_to_items(target_items=target_items, dataset=dataset, k=5)
    print(f"\nI2I recommendations (first 3 target items):")
    print(i2i[i2i[Columns.TargetItem].isin(range(3))].to_string(index=False))

    print("\nSmoke test passed!")


if __name__ == "__main__":
    main()
