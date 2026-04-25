from __future__ import annotations

import argparse
import json
import multiprocessing
import queue
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import bin_packing_utils

from evaluation.common import resolve_samples_dir, select_best_function, to_project_relative_path


def _eval_single_instance(func_code: str, inst_name: str, inst_data: dict, result_queue) -> None:
    import numpy as np

    try:
        namespace = {'np': np}
        exec(func_code, namespace)
        priority_fn = namespace['priority']
        capacity = inst_data['capacity']
        items = inst_data['items']
        num_items = inst_data['num_items']
        bins = np.array([capacity for _ in range(num_items)], dtype=float)
        for item in items:
            valid_bin_indices = np.nonzero((bins - item) >= 0)[0]
            priorities = priority_fn(item, bins[valid_bin_indices])
            best_bin = valid_bin_indices[np.argmax(priorities)]
            bins[best_bin] -= item
        num_bins = int((bins != capacity).sum())
        result_queue.put({'status': 'ok', 'instance': inst_name, 'bins': num_bins})
    except Exception as exc:
        result_queue.put({'status': 'error', 'instance': inst_name, 'message': str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate best function on Weibull 5k dataset')
    parser.add_argument('--log-dir', required=True, help='Experiment log directory containing samples/')
    parser.add_argument('--cutoff', type=int, required=True, help='Cutoff sample order for best-function selection')
    parser.add_argument('--output-dir', required=True, help='Directory to write weibull_test.json')
    parser.add_argument('--timeout-per-instance', type=int, default=300, help='Per-instance timeout in seconds')
    args = parser.parse_args()

    multiprocessing.set_start_method('spawn', force=True)

    samples_dir = resolve_samples_dir(args.log_dir)
    best_row = select_best_function(samples_dir, cutoff=args.cutoff)
    if best_row is None:
        raise SystemExit('No valid function found within cutoff')

    weibull = bin_packing_utils.datasets['Weibull 5k']
    inst_names = sorted(weibull.keys())

    results = {}
    timed_out = False
    for inst_name in inst_names:
        q = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=_eval_single_instance,
            args=(best_row['function'], inst_name, weibull[inst_name], q),
        )
        inst_start = time.time()
        p.start()
        p.join(timeout=args.timeout_per_instance)
        if p.is_alive():
            p.terminate()
            p.join()
            results[inst_name] = {'status': 'timeout', 'elapsed_s': round(time.time() - inst_start, 1)}
            timed_out = True
            continue
        try:
            payload = q.get_nowait()
        except queue.Empty:
            payload = {'status': 'error', 'instance': inst_name, 'message': 'NO RESULT'}
        payload['elapsed_s'] = round(time.time() - inst_start, 1)
        results[inst_name] = payload

    ok_bins = [item['bins'] for item in results.values() if item.get('status') == 'ok']
    score = -sum(ok_bins) / len(ok_bins) if ok_bins and not timed_out else None
    output = {
        'dataset': 'Weibull 5k',
        'source_log_dir': to_project_relative_path(args.log_dir),
        'cutoff': args.cutoff,
        'best_sample_order': best_row['sample_order'],
        'best_search_score': best_row['score'],
        'score': score,
        'instance_results': results,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'weibull_test.json', 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
