"""Profile build_sequences on synthetic data matching ML-20M scale."""

import time
import torch

def build_sequences_profiled(
    user_ids, item_ids, timestamps, max_len, min_interactions=2, device="cuda",
):
    t0 = time.time()
    user_ids = user_ids.to(device)
    item_ids = item_ids.to(device)
    timestamps = timestamps.to(device)
    torch.cuda.synchronize()
    t_transfer = time.time() - t0

    t0 = time.time()
    unique_items, item_inv = torch.unique(item_ids, return_inverse=True)
    internal_items = item_inv + 1
    unique_users, user_inv = torch.unique(user_ids, return_inverse=True)
    torch.cuda.synchronize()
    t_unique = time.time() - t0

    t0 = time.time()
    order1 = torch.argsort(timestamps, stable=True)
    order2 = torch.argsort(user_inv[order1], stable=True)
    order = order1[order2]
    sorted_user_inv = user_inv[order]
    sorted_items = internal_items[order]
    torch.cuda.synchronize()
    t_sort = time.time() - t0

    t0 = time.time()
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
    torch.cuda.synchronize()
    t_boundaries = time.time() - t0

    t0 = time.time()
    effective_lens = torch.clamp(capped_lens - 1, min=0)
    total_elements = effective_lens.sum().item()
    x = torch.zeros(n_users, max_len, dtype=torch.long, device=device)
    y = torch.zeros(n_users, max_len, dtype=torch.long, device=device)

    if total_elements > 0:
        user_indices = torch.repeat_interleave(torch.arange(n_users, device=device), effective_lens)
        cumsum = effective_lens.cumsum(0)
        offsets = torch.arange(total_elements, device=device) - torch.repeat_interleave(cumsum - effective_lens, effective_lens)
        x_src = torch.repeat_interleave(ends - capped_lens, effective_lens) + offsets
        y_src = x_src + 1
        col_indices = max_len - torch.repeat_interleave(effective_lens, effective_lens) + offsets
        x[user_indices, col_indices] = sorted_items[x_src]
        y[user_indices, col_indices] = sorted_items[y_src]
    torch.cuda.synchronize()
    t_scatter = time.time() - t0

    valid_user_indices = torch.where(mask)[0]
    result_users = unique_users[valid_user_indices] if len(valid_user_indices) < len(unique_users) else unique_users

    print(f"  transfer to GPU:   {t_transfer:.3f}s")
    print(f"  unique:            {t_unique:.3f}s")
    print(f"  sort (2x argsort): {t_sort:.3f}s")
    print(f"  boundaries:        {t_boundaries:.3f}s")
    print(f"  scatter (vectorized): {t_scatter:.3f}s")
    print(f"  TOTAL:             {t_transfer + t_unique + t_sort + t_boundaries + t_scatter:.3f}s")
    print(f"  n_users={n_users}, total_elements={total_elements}")

    return x, y, unique_items, result_users


def verify_correctness():
    """Small test to verify vectorized scatter produces correct results."""
    torch.manual_seed(42)
    n = 50
    user_ids = torch.tensor([0,0,0,0,0, 1,1,1, 2,2,2,2])
    item_ids = torch.tensor([10,20,30,40,50, 60,70,80, 90,100,110,120])
    timestamps = torch.arange(n := len(user_ids))

    from rectools.fast_transformers.gpu_data import build_sequences
    x, y, ui, uu = build_sequences(user_ids, item_ids, timestamps, max_len=4, min_interactions=2, device="cuda")

    x_cpu = x.cpu()
    y_cpu = y.cpu()

    print("\n=== Correctness check ===")
    print(f"x:\n{x_cpu}")
    print(f"y:\n{y_cpu}")

    # User 0: items [1,2,3,4,5], capped to 5 (max_len+1=5), effective=4
    #   x row: [2, 3, 4, 5] wait, max_len=4 so x[0] should be [1,2,3,4], y[0]=[2,3,4,5]
    # Actually: capped = min(5, 4+1=5) = 5, effective = 4
    #   seq = items[-5:] = [1,2,3,4,5]
    #   x: seq[:-1] = [1,2,3,4] placed at cols 0..3
    #   y: seq[1:]  = [2,3,4,5] placed at cols 0..3
    assert x_cpu[0].tolist() == [1,2,3,4], f"Got {x_cpu[0].tolist()}"
    assert y_cpu[0].tolist() == [2,3,4,5], f"Got {y_cpu[0].tolist()}"

    # User 1: items [6,7,8], capped=3, effective=2
    #   seq = [6,7,8], x: [6,7] at cols 2..3, y: [7,8] at cols 2..3
    assert x_cpu[1].tolist() == [0,0,6,7], f"Got {x_cpu[1].tolist()}"
    assert y_cpu[1].tolist() == [0,0,7,8], f"Got {y_cpu[1].tolist()}"

    # User 2: items [9,10,11,12], capped=4, effective=3
    #   seq = [9,10,11,12], x: [9,10,11] at cols 1..3, y: [10,11,12] at cols 1..3
    assert x_cpu[2].tolist() == [0,9,10,11], f"Got {x_cpu[2].tolist()}"
    assert y_cpu[2].tolist() == [0,10,11,12], f"Got {y_cpu[2].tolist()}"

    print("All assertions passed!")


def profile_ml20m_scale():
    """Generate data at ML-20M scale and profile."""
    print("\n=== ML-20M scale profile ===")
    torch.manual_seed(0)
    N = 5_000_000
    n_users_approx = 136_000
    n_items_approx = 7_000

    user_ids = torch.randint(0, n_users_approx, (N,))
    item_ids = torch.randint(0, n_items_approx, (N,))
    timestamps = torch.randint(0, 10**9, (N,), dtype=torch.long)

    # warmup
    print("Warmup...")
    _ = build_sequences_profiled(user_ids[:1000], item_ids[:1000], timestamps[:1000], max_len=200, device="cuda")

    print("\nFull run:")
    x, y, ui, uu = build_sequences_profiled(user_ids, item_ids, timestamps, max_len=200, device="cuda")
    print(f"Output shape: x={x.shape}, y={y.shape}")
    print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    verify_correctness()
    profile_ml20m_scale()
