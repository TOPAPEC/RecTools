"""Train UniSRec on ML-20M with Qwen embeddings."""

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from rectools.fast_transformers import UniSRecModel

DESCRIPTIONS_PATH = "training_folder/uniSRec/item_descriptions_compact.json"
QWEN_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
QWEN_DIM = 1024
DATA_DIR = Path("data/ml-20m")
CACHE_EMB_PATH = DATA_DIR / "qwen_embeddings.pt"
ML20M_URL = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"

MIN_RATING = 4.0
MIN_ITEM_INTERACTIONS = 50
MIN_USER_INTERACTIONS = 5
PHASE3_EPOCHS = 30


def download_ml20m():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ratings_path = DATA_DIR / "ml-20m" / "ratings.csv"
    if ratings_path.exists():
        return
    zip_path = DATA_DIR / "ml-20m.zip"
    if not zip_path.exists():
        print(f"Downloading ML-20M...")
        import urllib.request
        urllib.request.urlretrieve(ML20M_URL, zip_path)
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)


def load_and_preprocess():
    download_ml20m()
    ratings = pd.read_csv(DATA_DIR / "ml-20m" / "ratings.csv")
    ratings.columns = ["user_id", "item_id", "rating", "timestamp"]

    if MIN_RATING > 0:
        ratings = ratings[ratings["rating"] >= MIN_RATING]
        print(f"After rating filter (>={MIN_RATING}): {len(ratings):,} interactions")

    if MIN_ITEM_INTERACTIONS > 0:
        item_counts = ratings.groupby("item_id").size()
        popular = item_counts[item_counts >= MIN_ITEM_INTERACTIONS].index
        ratings = ratings[ratings["item_id"].isin(popular)]
        print(f"After item filter (>={MIN_ITEM_INTERACTIONS}): {ratings['item_id'].nunique():,} items")

    user_counts = ratings.groupby("user_id").size()
    valid = user_counts[user_counts >= MIN_USER_INTERACTIONS].index
    ratings = ratings[ratings["user_id"].isin(valid)]
    print(f"Final: {len(ratings):,} interactions, {ratings['user_id'].nunique():,} users, {ratings['item_id'].nunique():,} items")

    movies = pd.read_csv(DATA_DIR / "ml-20m" / "movies.csv")
    movies.columns = ["movieId", "title", "genres"]
    return ratings, movies


def _last_token_pool(hidden_states, attention_mask):
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return hidden_states[:, -1]
    seq_lengths = attention_mask.sum(dim=1) - 1
    return hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), seq_lengths]


@torch.no_grad()
def encode_qwen(texts, device="cuda", batch_size=1024):
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME, padding_side="left")
    model = AutoModel.from_pretrained(QWEN_MODEL_NAME, torch_dtype=torch.float16).to(device).eval()

    embeddings = torch.zeros(len(texts), QWEN_DIM)
    for start in tqdm(range(0, len(texts), batch_size), desc="Qwen encode"):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        hidden = model(**inputs).last_hidden_state
        out = _last_token_pool(hidden, inputs["attention_mask"])
        embeddings[start:start + len(batch)] = out.float().cpu()

    del model, tokenizer
    torch.cuda.empty_cache()
    return embeddings


def build_pretrained_embeddings(movies, descriptions):
    all_movie_ids = sorted(movies["movieId"].unique())
    max_id = max(all_movie_ids)
    texts_by_id = {}

    for mid in all_movie_ids:
        key = str(mid)
        if key in descriptions:
            val = descriptions[key]
            texts_by_id[mid] = val[0] if isinstance(val, list) else val
        else:
            row = movies[movies["movieId"] == mid]
            if len(row) > 0:
                texts_by_id[mid] = f"{row.iloc[0]['title']} {row.iloc[0]['genres']}"
            else:
                texts_by_id[mid] = f"movie {mid}"

    ordered_ids = sorted(texts_by_id.keys())
    ordered_texts = [texts_by_id[mid] for mid in ordered_ids]

    if CACHE_EMB_PATH.exists():
        print(f"Loading cached embeddings from {CACHE_EMB_PATH}")
        return torch.load(CACHE_EMB_PATH, weights_only=True)

    raw_embs = encode_qwen(ordered_texts, batch_size=512)

    embeddings = torch.zeros(max_id + 1, QWEN_DIM)
    for i, mid in enumerate(ordered_ids):
        embeddings[mid] = raw_embs[i]

    torch.save(embeddings, CACHE_EMB_PATH)
    print(f"Saved embeddings to {CACHE_EMB_PATH}, shape={embeddings.shape}")
    return embeddings


def split_eval(ratings):
    ratings = ratings.sort_values(["user_id", "timestamp"])
    grouped = ratings.groupby("user_id")
    test_idx = grouped.tail(1).index
    remaining = ratings.drop(test_idx)
    val_idx = remaining.groupby("user_id").tail(1).index
    train_idx = remaining.drop(val_idx).index

    train = ratings.loc[train_idx]
    val = ratings.loc[val_idx]
    test = ratings.loc[test_idx]
    return train, val, test


def to_tensors(df):
    """Convert a ratings DataFrame to (user_ids, item_ids, timestamps) tensors."""
    return (
        torch.tensor(df["user_id"].values, dtype=torch.long),
        torch.tensor(df["item_id"].values, dtype=torch.long),
        torch.tensor(df["timestamp"].values, dtype=torch.long),
    )


@torch.no_grad()
def evaluate_fast(model, train_ratings_df, test_df, k=10, batch_size=256):
    net = model.net
    net.cuda().eval()
    device = torch.device("cuda")
    maxlen = net.session_max_len

    item_embs = net.project_all()
    unique_items = model.item_id_mapping

    ext_to_int = {}
    for i in range(len(unique_items)):
        ext_to_int[int(unique_items[i].item())] = i + 1

    train_grouped = train_ratings_df.sort_values("timestamp").groupby("user_id")["item_id"].agg(list).to_dict()
    test_grouped = test_df.groupby("user_id")["item_id"].first().to_dict()
    test_users = list(test_grouped.keys())

    hits, ndcg_sum, mrr_sum, total = 0, 0.0, 0.0, 0

    for start in tqdm(range(0, len(test_users), batch_size), desc="Evaluating"):
        batch_users = test_users[start:start + batch_size]
        seqs, targets = [], []
        for uid in batch_users:
            history = train_grouped.get(uid, [])
            mapped = [ext_to_int[iid] for iid in history if iid in ext_to_int]
            if not mapped:
                continue
            seq = mapped[-maxlen:]
            seqs.append([0] * (maxlen - len(seq)) + seq)
            targets.append(ext_to_int.get(test_grouped[uid]))

        if not seqs:
            continue

        x = torch.tensor(seqs, dtype=torch.long, device=device)
        h = net.encode_last(x, use_id=False)
        scores = h @ item_embs.T
        scores[:, 0] = float("-inf")

        for i, target_int in enumerate(targets):
            if target_int is None:
                continue
            _, topk_idx = scores[i].topk(k)
            topk = topk_idx.cpu().tolist()
            if target_int in topk:
                rank = topk.index(target_int)
                hits += 1
                ndcg_sum += 1.0 / np.log2(rank + 2)
                mrr_sum += 1.0 / (rank + 1)
            total += 1

    return {
        f"HR@{k}": hits / total if total else 0,
        f"NDCG@{k}": ndcg_sum / total if total else 0,
        f"MRR@{k}": mrr_sum / total if total else 0,
        "n_users": total,
    }


def main():
    print("=" * 60)
    print("UniSRec Training on ML-20M")
    print("=" * 60)

    ratings, movies = load_and_preprocess()
    descriptions = json.loads(Path(DESCRIPTIONS_PATH).read_text())
    print(f"Loaded {len(descriptions)} descriptions")

    pretrained = build_pretrained_embeddings(movies, descriptions)
    print(f"Pretrained embeddings: {pretrained.shape}")

    train_ratings, val_ratings, test_ratings = split_eval(ratings)
    print(f"Split: train={len(train_ratings):,}, val={len(val_ratings):,}, test={len(test_ratings):,}")

    train_with_val = pd.concat([train_ratings, val_ratings])

    checkpoint_path = DATA_DIR / "unisrec_v3.pt"

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
        phase3_epochs=PHASE3_EPOCHS,
        phase1_lr=1e-3,
        phase2_lr=3e-4,
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
        patience=10,
        batch_size=128,
        dataloader_num_workers=0,
        train_min_user_interactions=2,
        verbose=1,
    )

    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        model.load_checkpoint(checkpoint_path)
    else:
        print("\nStarting training...")
        user_ids, item_ids, timestamps = to_tensors(train_with_val)
        model.fit(user_ids, item_ids, timestamps)
        model.save_checkpoint(checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    print("Training complete!")

    print("\n--- Validation Metrics ---")
    val_results = evaluate_fast(model, train_ratings, val_ratings)
    for m, v in val_results.items():
        print(f"  {m}: {v}")

    print("\n--- Test Metrics ---")
    test_results = evaluate_fast(model, train_with_val, test_ratings)
    for m, v in test_results.items():
        print(f"  {m}: {v}")

    print("\n--- Expected Metrics ---")
    print("  val:  HR@10=0.2431  NDCG@10=0.1335")
    print("  test: HR@10=0.2218  NDCG@10=0.1251  MRR@10=0.0957")


if __name__ == "__main__":
    main()
