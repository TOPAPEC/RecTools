# UniSRec Training Demo: KION Dataset

This guide demonstrates training a UniSRec sequential recommender on the KION movie dataset using real text embeddings from movie descriptions.

## Overview

UniSRec jointly trains a PCA-based adaptor and a SASRec transformer encoder on frozen pretrained text embeddings. This allows the model to leverage semantic item representations without requiring collaborative item IDs.

## Prerequisites

```bash
pip install torch pytorch-lightning sentence-transformers
```

## 1. Prepare Data

### Download the KION dataset

```bash
git clone https://github.com/irsafilo/KION_DATASET kion_data
```

### Load and filter interactions

```python
import pandas as pd
import torch

# Load interactions
interactions = pd.read_csv("kion_data/interactions.csv")
interactions = interactions.rename(columns={"last_watch_dt": "timestamp"})
interactions["timestamp"] = pd.to_datetime(interactions["timestamp"]).astype(int) // 10**9

# Filter: min 5 interactions per item, min 2 per user
item_counts = interactions.groupby("item_id").size()
interactions = interactions[interactions["item_id"].isin(item_counts[item_counts >= 5].index)]
user_counts = interactions.groupby("user_id").size()
interactions = interactions[interactions["user_id"].isin(user_counts[user_counts >= 2].index)]

print(f"Interactions: {len(interactions):,}")
print(f"Users: {interactions['user_id'].nunique():,}")
print(f"Items: {interactions['item_id'].nunique():,}")
# Interactions: 643,786
# Users: 201,851
# Items: 6,228
```

### Leave-last-out split

```python
interactions = interactions.sort_values(["user_id", "timestamp"])
test = interactions.groupby("user_id").tail(1)
train_val = interactions.drop(test.index)

print(f"Train+Val: {len(train_val):,}, Test: {len(test):,}")
# Train+Val: 441,935, Test: 201,851
```

## 2. Generate Text Embeddings

Use English movie descriptions from the dataset with Qwen3-Embedding-0.6B:

```bash
pip install transformers
```

```python
from transformers import AutoTokenizer, AutoModel

# Load item metadata (English descriptions)
items = pd.read_csv("kion_data/data_en/items_en.csv")
items = items.set_index("item_id")

# Build description text
texts = {}
for item_id, row in items.iterrows():
    parts = [str(row.get("title", ""))]
    if pd.notna(row.get("description")):
        parts.append(str(row["description"]))
    if pd.notna(row.get("genres")):
        parts.append(f"Genres: {row['genres']}")
    texts[item_id] = " ".join(parts)

# Encode with Qwen3-Embedding-0.6B
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
encoder = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B", dtype=torch.float16)
encoder.cuda().eval()

max_item_id = items.index.max()
embeddings = torch.zeros(max_item_id + 1, 1024)

item_ids_list = list(texts.keys())
text_list = list(texts.values())

with torch.no_grad():
    for start in range(0, len(text_list), 32):
        batch_texts = text_list[start:start + 32]
        batch_ids = item_ids_list[start:start + 32]
        encoded = tokenizer(batch_texts, return_tensors="pt", padding=True,
                            truncation=True, max_length=512).to("cuda")
        outputs = encoder(**encoded)
        mask = encoded["attention_mask"].unsqueeze(-1).half()
        pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        for i, item_id in enumerate(batch_ids):
            embeddings[item_id] = pooled[i].cpu().float()

torch.save(embeddings, "item_embeddings.pt")
print(f"Embeddings: {embeddings.shape}")
# Embeddings: torch.Size([16519, 1024])
```

## 3. Train UniSRec

```python
from rectools.fast_transformers import UniSRecModel

embeddings = torch.load("item_embeddings.pt", weights_only=True)

user_ids = torch.tensor(train_val["user_id"].values, dtype=torch.long)
item_ids = torch.tensor(train_val["item_id"].values, dtype=torch.long)
timestamps = torch.tensor(train_val["timestamp"].values, dtype=torch.long)

model = UniSRecModel(
    pretrained_item_embeddings=embeddings,
    # Architecture
    n_factors=256,
    projection_hidden=512,
    n_blocks=2,
    n_heads=2,
    session_max_len=50,
    dropout=0.1,
    adaptor_dropout=0.2,
    adaptor_type="pca",
    use_adaptor_ffn=True,
    ffn_type="conv1d",
    ffn_expansion=1,
    # Training
    epochs=10,
    lr=1e-4,
    lr_head=0.3,
    lr_wp=0.1,
    lr_transformer=3.0,
    optimizer="adamw",
    scheduler="cosine_warmup",
    warmup_ratio=0.05,
    min_lr_ratio=0.1,
    grad_clip=1.0,
    weight_decay=0.01,
    loss="softmax",
    batch_size=128,
    train_min_user_interactions=2,
    verbose=1,
)

model.fit(user_ids, item_ids, timestamps)
# Training: ~194s on RTX 3090 (10 epochs)
```

### Save / load checkpoint

```python
model.save_checkpoint("unisrec_kion.pt")

# Later:
model2 = UniSRecModel(pretrained_item_embeddings=embeddings, n_factors=256, ...)
model2.load_checkpoint("unisrec_kion.pt", device="cuda")
```

## 4. Evaluate

Leave-last-out evaluation with HR@K and NDCG@K:

```python
import numpy as np

net = model.net
net.eval().cuda()
device = torch.device("cuda")

# Get projected item embeddings
item_embs = net.project_all()
unique_items = model.item_id_mapping
ext_to_int = {int(unique_items[i].item()): i + 1 for i in range(len(unique_items))}

# Build user histories
train_grouped = train_val.sort_values("timestamp").groupby("user_id")["item_id"].agg(list).to_dict()
test_grouped = test.groupby("user_id")["item_id"].first().to_dict()
test_users = list(test_grouped.keys())

hits10, ndcg10, total = 0, 0.0, 0
maxlen = model.session_max_len

with torch.no_grad():
    for start in range(0, len(test_users), 256):
        batch_users = test_users[start:start + 256]
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
        h = net.encode_last(x)
        scores = h @ item_embs.T
        scores[:, 0] = float("-inf")
        for i, target_int in enumerate(targets):
            if target_int is None:
                continue
            _, topk = scores[i].topk(10)
            topk = topk.cpu().tolist()
            if target_int in topk:
                rank = topk.index(target_int)
                hits10 += 1
                ndcg10 += 1.0 / np.log2(rank + 2)
            total += 1

print(f"HR@10  = {hits10/total:.4f}")
print(f"NDCG@10 = {ndcg10/total:.4f}")
```

## 5. Results

Trained on NVIDIA RTX 3090, 10 epochs, same architecture (256d, 2 blocks, 2 heads, max_len=50):

| Model | Embedder | HR@5 | NDCG@5 | HR@10 | NDCG@10 | Train Time |
|-------|----------|------|--------|-------|---------|------------|
| **UniSRec** | all-MiniLM-L6-v2 (384d) | 0.1421 | 0.0988 | 0.1896 | 0.1145 | ~194s |
| **UniSRec** | Qwen3-Embedding-0.6B (1024d) | 0.1529 | 0.1012 | 0.2018 | 0.1171 | ~178s |
| **SASRec** (RecTools) | ID embeddings | 0.1606 | 0.1081 | 0.2175 | 0.1265 | ~166s |

Qwen3-Embedding-0.6B closes most of the gap to SASRec (HR@10 delta: 1.6pp vs 2.8pp with MiniLM). SASRec with learned ID embeddings is stronger when sufficient interaction data is available. UniSRec's advantage is in cold-start and transfer scenarios where text embeddings provide semantic signal for items with no interaction history.

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_factors` | Hidden dimension of the transformer | 256 |
| `adaptor_type` | Adaptor type: `"pca"` or `"bn"` | `"pca"` |
| `session_max_len` | Maximum sequence length | 200 |
| `epochs` | Number of training epochs | 10 |
| `lr` | Base learning rate (adaptor layernorms) | 1e-4 |
| `lr_wp` | Multiplier for PCA whitening projection | 0.1 |
| `lr_transformer` | Multiplier for transformer layers | 3.0 |
| `lr_head` | Multiplier for head layer | 0.3 |
| `loss` | Loss function: `"softmax"`, `"BCE"`, `"gBCE"`, `"sampled_softmax"` | `"softmax"` |
| `patience` | Early stopping patience (None = disabled) | None |
| `scheduler` | LR scheduler: `None` or `"cosine_warmup"` | None |

## ONNX Export

```python
model.export_to_onnx(
    encoder_path="unisrec_encoder.onnx",
    items_path="unisrec_items.onnx",
)
```
