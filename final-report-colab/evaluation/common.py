from __future__ import annotations

import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def to_project_relative_path(path: str | Path) -> str:
    target = Path(path).resolve()
    try:
        return target.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(target)


def resolve_samples_dir(log_dir: str | Path) -> Path:
    base = Path(log_dir)
    direct_samples = base / 'samples'
    if direct_samples.exists():
        return direct_samples

    wrapped_samples = base / 'funsearch_output' / 'samples'
    if wrapped_samples.exists():
        return wrapped_samples

    return direct_samples


def load_sample_rows(samples_dir: str | Path) -> list[dict]:
    rows = []
    for path in sorted(Path(samples_dir).glob('samples_*.json')):
        with open(path) as f:
            rows.append(json.load(f))
    rows.sort(key=lambda row: row.get('sample_order', -1))
    return rows


def select_best_function(samples_dir: str | Path, cutoff: int) -> dict | None:
    best_row = None
    best_score = float('-inf')
    for row in load_sample_rows(samples_dir):
        sample_order = row.get('sample_order', -1)
        score = row.get('score')
        if score is None or sample_order > cutoff:
            continue
        if float(score) > best_score:
            best_score = float(score)
            best_row = row
    return best_row


def evaluate_function_on_dataset(func_code: str, dataset: dict[str, dict]) -> dict:
    namespace = {'np': np}
    exec(func_code, namespace)
    priority_fn = namespace['priority']

    instance_bins: dict[str, int] = {}
    for name, inst in dataset.items():
        capacity = inst['capacity']
        items = inst['items']
        bins = np.array([capacity for _ in range(inst['num_items'])], dtype=float)
        for item in items:
            valid_bin_indices = np.nonzero((bins - item) >= 0)[0]
            priorities = priority_fn(item, bins[valid_bin_indices])
            best_bin = valid_bin_indices[np.argmax(priorities)]
            bins[best_bin] -= item
        instance_bins[name] = int((bins != capacity).sum())

    score = -float(np.mean(list(instance_bins.values()))) if instance_bins else None
    return {
        'status': 'ok',
        'score': score,
        'instance_bins': instance_bins,
    }
