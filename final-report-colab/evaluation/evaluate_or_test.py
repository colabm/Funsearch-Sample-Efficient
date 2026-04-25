from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import bin_packing_utils

from evaluation.common import (
    evaluate_function_on_dataset,
    resolve_samples_dir,
    select_best_function,
    to_project_relative_path,
)


TEST_INSTANCES = [f'u500_{i:02d}' for i in range(14, 20)]


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate best function on OR3 test set')
    parser.add_argument('--log-dir', required=True, help='Experiment log directory containing samples/')
    parser.add_argument('--cutoff', type=int, required=True, help='Cutoff sample order for best-function selection')
    parser.add_argument('--output-dir', required=True, help='Directory to write or_test.json')
    args = parser.parse_args()

    samples_dir = resolve_samples_dir(args.log_dir)
    best_row = select_best_function(samples_dir, cutoff=args.cutoff)
    if best_row is None:
        raise SystemExit('No valid function found within cutoff')

    or3 = bin_packing_utils.datasets['OR3']
    dataset = {name: or3[name] for name in TEST_INSTANCES}
    result = evaluate_function_on_dataset(best_row['function'], dataset)
    result.update({
        'dataset': 'OR3 test',
        'instances': TEST_INSTANCES,
        'source_log_dir': to_project_relative_path(args.log_dir),
        'cutoff': args.cutoff,
        'best_sample_order': best_row['sample_order'],
        'best_search_score': best_row['score'],
    })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'or_test.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
