"""Quick 1-epoch smoke test of the full pipeline."""

import time
from pathlib import Path

import pandas as pd
import torch

from rectools.fast_transformers import UniSRecModel

DATA_DIR = Path("data/ml-20m")
MIN_RATING = 4.0
MIN_ITEM_INTERACTIONS = 50
MIN_USER_INTERACTIONS = 5


def load_data():
    ratings = pd.read_csv(DATA_DIR / "ml-20m" / "ratings.csv")
    ratings.columns = ["user_id", "item_id", "rating", "timestamp"]
    ratings = ratings[ratings["rating"] >= MIN_RATING]
    item_counts = ratings.groupby("item_id").size()
    popular = item_counts[item_counts >= MIN_ITEM_INTERACTIONS].index
    ratings = ratings[ratings["item_id"].isin(popular)]
    user_counts = ratings.groupby("user_id").size()
    valid = user_counts[user_counts >= MIN_USER_INTERACTIONS].index
    ratings = ratings[ratings["user_id"].isin(valid)]
    return ratings


def main():
    print("Loading data...")
    ratings = load_data()
    print(f"  {len(ratings):,} interactions, {ratings['user_id'].nunique():,} users, {ratings['item_id'].nunique():,} items")

    pretrained = torch.load(DATA_DIR / "qwen_embeddings.pt", weights_only=True)
    print(f"  Pretrained embeddings: {pretrained.shape}")

    user_ids = torch.tensor(ratings["user_id"].values, dtype=torch.long)
    item_ids = torch.tensor(ratings["item_id"].values, dtype=torch.long)
    timestamps = torch.tensor(ratings["timestamp"].values, dtype=torch.long)

    model = UniSRecModel(
        pretrained_item_embeddings=pretrained,
        n_factors=512,
        projection_hidden=512,
        n_blocks=2,
        n_heads=1,
        session_max_len=200,
        dropout=0.1,
        adaptor_dropout=0.2,
        adaptor_type="pca",
        use_adaptor_ffn=True,
        phase1_epochs=0,
        phase2_epochs=0,
        phase3_epochs=1,
        phase3_lr=1e-4,
        lr_head=0.3,
        lr_wp=0.1,
        lr_transformer=3.0,
        optimizer="adamw",
        scheduler="cosine_warmup",
        warmup_ratio=0.05,
        min_lr_ratio=1.0,
        grad_clip=1.0,
        weight_decay=0.01,
        loss="softmax",
        batch_size=128,
        dataloader_num_workers=0,
        train_min_user_interactions=2,
        verbose=1,
    )

    print("\nStarting 1-epoch training...")
    t0 = time.time()
    model.fit(user_ids, item_ids, timestamps)
    elapsed = time.time() - t0
    print(f"\n1-epoch training complete in {elapsed:.1f}s")

    # Verify item_id_mapping contains original IDs
    unique_items = model.item_id_mapping
    print(f"unique_items range: [{unique_items.min().item()}, {unique_items.max().item()}]")
    print(f"Original item_id range: [{ratings['item_id'].min()}, {ratings['item_id'].max()}]")
    assert unique_items.max().item() > 100, "IDs should be original MovieLens IDs, not 0-based reindexed"
    print("ID mapping verified — original external IDs preserved!")


if __name__ == "__main__":
    main()
