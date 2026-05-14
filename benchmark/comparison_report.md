# SASRec vs UniSRec-ID Comparison

**Date:** 2026-04-24 19:59  
**GPU:** NVIDIA GeForce RTX 4090  
**Dataset:** ML-20M (min_rating=-1, min_item=5, min_user=2)

## Data

| | Count |
|---|---:|
| Interactions | 19,984,024 |
| Users | 138,493 |
| Items | 18,345 |
| Train | 19,707,038 |
| Val | 138,493 |
| Test | 138,493 |

## Config

| Parameter | Value |
|---|---|
| n_factors | 256 |
| n_blocks | 2 |
| n_heads | 1 |
| session_max_len | 200 |
| batch_size | 128 |
| lr | 0.001 |
| loss | softmax |
| optimizer | Adam |
| epochs | 10 |
| patience | None |
| dropout | 0.1 |

## Timing

| Stage | SASRec | UniSRec ID |
|---|---:|---:|
| Data load & split | 0.0s | 0.0s |
| Preprocessing | 14.6s | 0.5s |
| Model init | 0.0s | 0.0s |
| Training (10 epochs) | 911.8s | 639.5s |
| Evaluation | 175.6s | 28.0s |
| **Total** | **1102.1s** | **668.0s** |

| | SASRec | UniSRec ID |
|---|---:|---:|
| Epochs completed | 11 | 10 |
| Time per epoch | 82.9s | 63.9s |
| Preprocessing speedup | — | 29x |

## Quality (test set, 138,493 users)

| Model | HR@10 | NDCG@10 | MRR@10 |
|---|---:|---:|---:|
| SASRec | 0.2417 | 0.1410 | 0.1103 |
| UniSRec ID | 0.2528 | 0.1495 | 0.1179 |

UniSRec vs SASRec: HR@10 +4.6%, NDCG@10 +6.0%
