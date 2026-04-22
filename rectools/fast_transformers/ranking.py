"""Batch top-k ranking with optional viewed-item filtering."""

import typing as tp

import numpy as np
import torch
from scipy import sparse


def rank_topk(
    user_embs: torch.Tensor,
    item_embs: torch.Tensor,
    k: int,
    filter_csr: tp.Optional[sparse.csr_matrix] = None,
    whitelist: tp.Optional[np.ndarray] = None,
    batch_size: int = 256,
) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Batch-wise top-k ranking: user_embs @ item_embs.T with optional filtering.

    Parameters
    ----------
    user_embs : Tensor (N, D)
        User embeddings.
    item_embs : Tensor (M, D)
        Item embeddings.
    k : int
        Number of items to recommend per user.
    filter_csr : csr_matrix (N, M), optional
        Binary matrix of viewed items to mask out.
    whitelist : ndarray, optional
        Sorted array of item indices to consider.
    batch_size : int
        Batch size for processing users.

    Returns
    -------
    all_user_ids, all_item_ids, all_scores : ndarray, ndarray, ndarray
        Flattened arrays of recommendations.
    """
    device = user_embs.device
    n_users = user_embs.shape[0]

    if whitelist is not None:
        item_embs = item_embs[whitelist]

    all_user_ids = []
    all_item_ids = []
    all_scores = []

    for start in range(0, n_users, batch_size):
        end = min(start + batch_size, n_users)
        scores = user_embs[start:end] @ item_embs.T  # (batch, M)

        if filter_csr is not None:
            batch_csr = filter_csr[start:end]
            if whitelist is not None:
                batch_csr = batch_csr[:, whitelist]
            viewed_mask = torch.tensor(batch_csr.toarray(), dtype=torch.bool, device=device)
            scores[viewed_mask] = -float("inf")

        actual_k = min(k, scores.shape[1])
        topk_scores, topk_idx = torch.topk(scores, actual_k, dim=1)  # (batch, k)

        if whitelist is not None:
            topk_idx_np = topk_idx.cpu().numpy()
            topk_idx_mapped = whitelist[topk_idx_np]
        else:
            topk_idx_mapped = topk_idx.cpu().numpy()

        batch_users = np.arange(start, end)
        user_ids = np.repeat(batch_users, actual_k)
        item_ids = topk_idx_mapped.ravel()
        s = topk_scores.cpu().numpy().ravel()

        all_user_ids.append(user_ids)
        all_item_ids.append(item_ids)
        all_scores.append(s)

    return np.concatenate(all_user_ids), np.concatenate(all_item_ids), np.concatenate(all_scores)
