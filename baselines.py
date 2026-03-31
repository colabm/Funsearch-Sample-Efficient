# baselines.py
"""基线算法和评估工具."""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bin_packing_utils


def online_first_fit(items, capacity):
    """Online First Fit: 放入第一个能放下的箱子."""
    n = len(items)
    bins = np.array([capacity] * n, dtype=float)
    for item in items:
        valid = np.where(bins >= item)[0]
        bins[valid[0]] -= item
    return int((bins != capacity).sum())


def online_best_fit(items, capacity):
    """Online Best Fit: 放入剩余空间最小但够放的箱子."""
    n = len(items)
    bins = np.array([capacity] * n, dtype=float)
    for item in items:
        valid = np.where(bins >= item)[0]
        remaining = bins[valid] - item
        bins[valid[np.argmin(remaining)]] -= item
    return int((bins != capacity).sum())


def offline_ffd(items, capacity):
    """Offline First Fit Decreasing."""
    sorted_items = sorted(items, reverse=True)
    bins_list = []
    for item in sorted_items:
        placed = False
        for i, rem in enumerate(bins_list):
            if rem >= item:
                bins_list[i] -= item
                placed = True
                break
        if not placed:
            bins_list.append(capacity - item)
    return len(bins_list)


def offline_bfd(items, capacity):
    """Offline Best Fit Decreasing."""
    sorted_items = sorted(items, reverse=True)
    bins_list = []
    for item in sorted_items:
        best_idx, best_rem = -1, capacity + 1
        for i, rem in enumerate(bins_list):
            if rem >= item and rem - item < best_rem:
                best_rem = rem - item
                best_idx = i
        if best_idx >= 0:
            bins_list[best_idx] -= item
        else:
            bins_list.append(capacity - item)
    return len(bins_list)


def evaluate_on_instances(algo_func, instances: dict) -> float:
    """在一组实例上评估算法, 返回 -avg_bins (与 FunSearch 评分一致)."""
    num_bins = []
    for name in instances:
        inst = instances[name]
        nb = algo_func(inst['items'], inst['capacity'])
        num_bins.append(nb)
    return -np.mean(num_bins)


def evaluate_priority_function(priority_code: str, instances: dict) -> float:
    """用给定的 priority 函数代码评估装箱性能.
    
    注意：使用与 FunSearch specification 完全一致的评估逻辑：
    - bins 数组不指定 dtype（保持 capacity 的原始类型，通常为 int）
    - 使用 np.nonzero((bins - item) >= 0) 而非 np.where(bins >= item)
    """
    namespace = {'np': np}
    exec(priority_code, namespace)
    priority_fn = namespace['priority']

    num_bins = []
    for name in instances:
        inst = instances[name]
        capacity = inst['capacity']
        items = inst['items']
        # 与 FunSearch specification 一致：不指定 dtype
        bins_arr = np.array([capacity for _ in range(inst['num_items'])])
        for item in items:
            valid = np.nonzero((bins_arr - item) >= 0)[0]
            priorities = priority_fn(item, bins_arr[valid])
            best = valid[np.argmax(priorities)]
            bins_arr[best] -= item
        num_bins.append((bins_arr != capacity).sum())
    return -np.mean(num_bins)


def compute_all_baselines():
    """计算所有数据集划分上的所有基线."""
    or3 = bin_packing_utils.datasets['OR3']
    weibull = bin_packing_utils.datasets['Weibull 5k']

    search_keys = [f'u500_{i:02d}' for i in range(14)]
    test_keys = [f'u500_{i:02d}' for i in range(14, 20)]

    search_set = {k: or3[k] for k in search_keys}
    test_set = {k: or3[k] for k in test_keys}

    results = {}
    for dataset_name, dataset in [
        ('OR3_search', search_set),
        ('OR3_test', test_set),
        ('OR3_full', or3),
        ('Weibull_5k', weibull),
    ]:
        results[dataset_name] = {}
        for algo_name, algo_func in [
            ('Online_FF', online_first_fit),
            ('Online_BF', online_best_fit),
            ('Offline_FFD', offline_ffd),
            ('Offline_BFD', offline_bfd),
        ]:
            score = evaluate_on_instances(algo_func, dataset)
            results[dataset_name][algo_name] = score

    return results


if __name__ == '__main__':
    results = compute_all_baselines()
    print("=== 基线结果 ===")
    for ds_name, algos in results.items():
        print(f"\n{ds_name}:")
        for algo_name, score in algos.items():
            print(f"  {algo_name}: score={score:.2f} (avg_bins={-score:.2f})")
