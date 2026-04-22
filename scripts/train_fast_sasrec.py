"""End-to-end smoke test: synthetic dataset, train, recommend, metrics, i2i."""

import numpy as np
import pandas as pd

from rectools import Columns
from rectools.dataset import Dataset
from rectools.fast_transformers import FlatSASRecModel


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

    # --- Train ---
    model = FlatSASRecModel(
        n_factors=32, n_blocks=2, n_heads=2, session_max_len=16,
        loss="softmax", epochs=2, batch_size=32, lr=1e-3, verbose=1,
    )
    model.fit(dataset)
    print("Training done.")

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
        # For this smoke test, "relevance" = items the user actually interacted with
        # (training set overlap is expected since we don't do train/test split here)
        rel = [1 if i in viewed else 0 for i in rec_items]
        hits += sum(rel)
        total += len(rec_items)
        # AP
        if sum(rel) > 0:
            precision_at = np.cumsum(rel) / np.arange(1, len(rel) + 1)
            ap_sum += np.sum(precision_at * rel) / sum(rel)
    recall = hits / max(total, 1)
    map_at_k = ap_sum / len(users)
    print(f"\nRecall@5 (train overlap): {recall:.4f}")
    print(f"MAP@5 (train overlap): {map_at_k:.4f}")

    # --- I2I ---
    target_items = list(range(10))
    i2i = model.recommend_to_items(target_items=target_items, dataset=dataset, k=5)
    print(f"\nI2I recommendations (first 3 target items):")
    print(i2i[i2i[Columns.TargetItem].isin(range(3))].to_string(index=False))

    print("\nSmoke test passed!")


if __name__ == "__main__":
    main()
