"""GPU-native sequence building for transformer training. Pure torch, no pandas/numpy."""

import typing as tp

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset


def _splitmix64(x: torch.Tensor) -> torch.Tensor:
    """Vectorized splitmix64 bit-mixer: element-wise int64 hash over a torch tensor.

    Standard library hashes (``hash()``, ``hashlib``) operate on scalar Python objects
    and cannot be vectorized across GPU tensors.  Splitmix64 is pure int64 arithmetic,
    so it maps naturally to ``torch.Tensor`` ops and runs on any device.

    Reference: https://xorshift.di.unimi.it/splitmix64.c (Vigna, 2015).
    """
    x = x.long()
    x = (x ^ (x >> 30)) * (-4658895280553007687)  # 0xbf58476d1ce4e5b9 as signed int64
    x = (x ^ (x >> 27)) * (-7723592293110705685)  # 0x94d049bb133111eb as signed int64
    return x ^ (x >> 31)


def hash_item_ids(item_ids: torch.Tensor, dict_size: int) -> torch.Tensor:
    """Map arbitrary integer item IDs to [1, dict_size] via splitmix64 hash."""
    return _splitmix64(item_ids) % dict_size + 1


def build_sequences(
    user_ids: torch.Tensor,
    item_ids: torch.Tensor,
    timestamps: torch.Tensor,
    max_len: int,
    min_interactions: int = 2,
    device: str = "cuda",
    id_mapping: str = "dense",
) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    user_ids = user_ids.to(device)
    item_ids = item_ids.to(device)
    timestamps = timestamps.to(device)

    unique_items = torch.unique(item_ids)
    n_unique = len(unique_items)

    if id_mapping == "dense":
        _, item_inv = torch.unique(item_ids, return_inverse=True)
        internal_items = item_inv + 1
    elif id_mapping == "hash":
        internal_items = hash_item_ids(item_ids, n_unique)
    else:
        raise ValueError(f"Unknown id_mapping: {id_mapping}. Use 'dense' or 'hash'")

    unique_users, user_inv = torch.unique(user_ids, return_inverse=True)

    order1 = torch.argsort(timestamps, stable=True)
    order2 = torch.argsort(user_inv[order1], stable=True)
    order = order1[order2]

    sorted_user_inv = user_inv[order]
    sorted_items = internal_items[order]

    changes = torch.where(sorted_user_inv[1:] != sorted_user_inv[:-1])[0] + 1
    starts = torch.cat([torch.tensor([0], device=device), changes])
    ends = torch.cat([changes, torch.tensor([len(sorted_user_inv)], device=device)])
    lengths = ends - starts

    mask = lengths >= min_interactions
    starts = starts[mask]
    ends = ends[mask]
    lengths = lengths[mask]
    n_users = len(starts)

    capped_lens = torch.clamp(lengths, max=max_len + 1)

    effective_lens = torch.clamp(capped_lens - 1, min=0)
    total_elements = effective_lens.sum().item()

    x = torch.zeros(n_users, max_len, dtype=torch.long, device=device)
    y = torch.zeros(n_users, max_len, dtype=torch.long, device=device)

    if total_elements > 0:
        user_indices = torch.repeat_interleave(torch.arange(n_users, device=device), effective_lens)
        cumsum = effective_lens.cumsum(0)
        offsets = torch.arange(total_elements, device=device) - torch.repeat_interleave(
            cumsum - effective_lens, effective_lens
        )

        x_src = torch.repeat_interleave(ends - capped_lens, effective_lens) + offsets
        y_src = x_src + 1
        col_indices = max_len - torch.repeat_interleave(effective_lens, effective_lens) + offsets

        x[user_indices, col_indices] = sorted_items[x_src]
        y[user_indices, col_indices] = sorted_items[y_src]

    valid_user_indices = torch.where(mask)[0]
    result_users = unique_users[valid_user_indices] if len(valid_user_indices) < len(unique_users) else unique_users

    return x, y, unique_items, result_users


def align_embeddings(
    pretrained: torch.Tensor,
    unique_items: torch.Tensor,
    n_items: int,
    id_mapping: str = "dense",
) -> torch.Tensor:
    idx = unique_items.long().cpu()
    valid = (idx >= 0) & (idx < pretrained.shape[0])

    if pretrained.ndim == 2:
        aligned = torch.zeros(n_items + 1, pretrained.shape[1])
    else:
        aligned = torch.zeros(n_items + 1, pretrained.shape[1], pretrained.shape[2])

    if id_mapping == "dense":
        aligned[1:][valid] = pretrained[idx[valid]]
    elif id_mapping == "hash":
        positions = hash_item_ids(idx, n_items)
        aligned[positions[valid]] = pretrained[idx[valid]]
    else:
        raise ValueError(f"Unknown id_mapping: {id_mapping}. Use 'dense' or 'hash'")

    return aligned


class GPUBatchDataset(TorchDataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, transform: tp.Optional[tp.Callable] = None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tp.Dict[str, torch.Tensor]:
        batch = {"x": self.x[idx], "y": self.y[idx]}
        if self.transform:
            batch = self.transform(batch)
        return batch


def make_dataloader(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    shuffle: bool = True,
    transform: tp.Optional[tp.Callable] = None,
) -> DataLoader:
    ds = GPUBatchDataset(x, y, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
