"""Experiment manager for report presets and consistent log naming.

Supported report presets:
1. `temp03_80`:
   - temperature = 0.3
   - samples = 80
   - 4 groups: no_dedup / ast_only / behavior_only / full_dedup
2. `no_temp_200`:
   - do not pass the temperature parameter
   - samples = 200
   - 2 groups: no_dedup / full_dedup

Log directory naming rules:
    <dedup_mode>-temp=<temperature>-<samples>samples
    <dedup_mode>-no_temp-<samples>samples

Examples:
    python run_experiments.py --preset temp03_80
    python run_experiments.py --preset no_temp_200
    python run_experiments.py --preset no_temp_200 --experiment full_dedup
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime


PYTHON = sys.executable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_ROOT = os.path.join(PROJECT_ROOT, 'logs')

DEFAULT_RANDOM_SEED = 42
DEFAULT_MODEL = 'gpt-3.5-turbo'
DEFAULT_SAMPLES_PER_PROMPT = 4

GROUPS = {
    'no_dedup': {
        'dedup_ast': False,
        'dedup_behavior': False,
        'description': 'Baseline group: no deduplication',
    },
    'ast_only': {
        'dedup_ast': True,
        'dedup_behavior': False,
        'description': 'AST deduplication only',
    },
    'behavior_only': {
        'dedup_ast': False,
        'dedup_behavior': True,
        'description': 'Behavioral-fingerprint deduplication only',
    },
    'full_dedup': {
        'dedup_ast': True,
        'dedup_behavior': True,
        'description': 'Two-layer deduplication: AST + behavioral fingerprint',
    },
}

PRESETS = {
    'temp03_80': {
        'groups': ['no_dedup', 'ast_only', 'behavior_only', 'full_dedup'],
        'samples': 80,
        'temperature': 0.3,
        'samples_per_prompt': 4,
        'model': DEFAULT_MODEL,
        'description': 'Experiment 1: temp=0.3, 80 samples, 4 groups',
    },
    'no_temp_200': {
        'groups': ['no_dedup', 'full_dedup'],
        'samples': 200,
        'temperature': None,
        'samples_per_prompt': 4,
        'model': DEFAULT_MODEL,
        'description': 'Stage 2 experiment: default temperature, 200 samples, 2 groups',
    },
}


def _temperature_label(temperature: float | None) -> str:
    if temperature is None:
        return 'no_temp'
    return f'temp={temperature:g}'


def _format_log_dir_name(group_name: str, temperature: float | None,
                         max_samples: int) -> str:
    return f'{group_name}-{_temperature_label(temperature)}-{max_samples}samples'


def _copy_tree_contents(src_dir: str, dst_dir: str) -> None:
    for item in os.listdir(src_dir):
        src = os.path.join(src_dir, item)
        dst = os.path.join(dst_dir, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)


def _resolve_temperature(args: argparse.Namespace,
                         default_temperature: float | None) -> float | None:
    if args.no_temperature:
        return None
    if args.temperature is not None:
        return args.temperature
    return default_temperature


def _prepare_env(group_config: dict, max_samples: int, temperature: float | None,
                 seed: int, model: str, samples_per_prompt: int,
                 temp_log_suffix: str) -> dict[str, str]:
    env = os.environ.copy()
    env['FUNSEARCH_MAX_SAMPLES'] = str(max_samples)
    env['FUNSEARCH_DEDUP_AST'] = '1' if group_config['dedup_ast'] else '0'
    env['FUNSEARCH_DEDUP_BEHAVIOR'] = '1' if group_config['dedup_behavior'] else '0'
    env['FUNSEARCH_SEED'] = str(seed)
    env['FUNSEARCH_TEMPERATURE'] = 'none' if temperature is None else str(temperature)
    env['FUNSEARCH_MODEL'] = model
    env['FUNSEARCH_SAMPLES_PER_PROMPT'] = str(samples_per_prompt)
    env['FUNSEARCH_LOG_SUFFIX'] = temp_log_suffix

    if 'FUNSEARCH_API_KEY' in os.environ:
        env['FUNSEARCH_API_KEY'] = os.environ['FUNSEARCH_API_KEY']
    if 'OPENAI_API_KEY' in os.environ:
        env['OPENAI_API_KEY'] = os.environ['OPENAI_API_KEY']
    if 'FUNSEARCH_BASE_URL' in os.environ:
        env['FUNSEARCH_BASE_URL'] = os.environ['FUNSEARCH_BASE_URL']
    if 'OPENAI_BASE_URL' in os.environ:
        env['OPENAI_BASE_URL'] = os.environ['OPENAI_BASE_URL']
    return env


def run_experiment(group_name: str, group_config: dict,
                   max_samples: int, temperature: float | None,
                   seed: int, model: str,
                   samples_per_prompt: int,
                   overwrite: bool = False) -> int:
    """Run one experiment and copy run_funsearch outputs into the target log directory."""
    os.makedirs(LOG_ROOT, exist_ok=True)

    target_dir_name = _format_log_dir_name(group_name, temperature, max_samples)
    target_dir = os.path.join(LOG_ROOT, target_dir_name)
    if os.path.exists(target_dir):
        if overwrite:
            shutil.rmtree(target_dir)
        else:
            raise FileExistsError(
                f'Target log directory already exists: {target_dir}. Use --overwrite to replace it.'
            )
    os.makedirs(target_dir, exist_ok=True)

    temp_log_suffix = f'tmp_{group_name}_{int(time.time())}'
    actual_log_dir = os.path.join(LOG_ROOT, f'funsearch_{temp_log_suffix}')
    env = _prepare_env(
        group_config=group_config,
        max_samples=max_samples,
        temperature=temperature,
        seed=seed,
        model=model,
        samples_per_prompt=samples_per_prompt,
        temp_log_suffix=temp_log_suffix,
    )

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n{'=' * 60}")
    print(f'[{timestamp}] Starting experiment: {group_name}')
    print(f'  Target directory: {target_dir}')
    print(
        f'  Parameters: samples={max_samples}, temperature={temperature}, '
        f'samples_per_prompt={samples_per_prompt}, seed={seed}, model={model}'
    )
    print(
        f"  Dedup: AST={group_config['dedup_ast']}, "
        f"Behavior={group_config['dedup_behavior']}"
    )
    print(f'  Python: {PYTHON}')
    print(f"{'=' * 60}\n", flush=True)

    start = time.time()
    exp_log_file = os.path.join(target_dir, 'experiment_output.log')
    with open(exp_log_file, 'w') as f:
        result = subprocess.run(
            [PYTHON, 'run_funsearch.py'],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - start

    if os.path.isdir(actual_log_dir):
        _copy_tree_contents(actual_log_dir, target_dir)
        shutil.rmtree(actual_log_dir, ignore_errors=True)

    meta = {
        'group_name': group_name,
        'log_dir_name': target_dir_name,
        'description': group_config['description'],
        'dedup_ast': group_config['dedup_ast'],
        'dedup_behavior': group_config['dedup_behavior'],
        'max_samples': max_samples,
        'samples_per_prompt': samples_per_prompt,
        'temperature': 'no_temp' if temperature is None else temperature,
        'seed': seed,
        'model': model,
        'python': PYTHON,
        'start_time': timestamp,
        'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_seconds': round(elapsed, 1),
        'return_code': result.returncode,
    }
    with open(os.path.join(target_dir, 'experiment_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    status = 'success' if result.returncode == 0 else f'failed(code={result.returncode})'
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {group_name} {status}')
    print(f'  Log directory: {target_dir}')
    return result.returncode


def _resolve_run_plan(args: argparse.Namespace) -> tuple[list[str], int, float | None, int, str]:
    if args.preset:
        preset = PRESETS[args.preset]
        groups = preset['groups']
        if args.experiment:
            if args.experiment not in groups:
                raise ValueError(
                    f'Experiment group {args.experiment} is not in preset {args.preset}: {groups}'
                )
            groups = [args.experiment]
        max_samples = args.samples if args.samples is not None else preset['samples']
        temperature = _resolve_temperature(args, preset['temperature'])
        samples_per_prompt = (
            args.samples_per_prompt
            if args.samples_per_prompt is not None
            else preset['samples_per_prompt']
        )
        model = args.model if args.model is not None else preset['model']
        return groups, max_samples, temperature, samples_per_prompt, model

    if args.custom_dedup_mode:
        max_samples = args.samples if args.samples is not None else 80
        temperature = _resolve_temperature(args, 0.3)
        samples_per_prompt = args.samples_per_prompt if args.samples_per_prompt is not None else DEFAULT_SAMPLES_PER_PROMPT
        model = args.model if args.model is not None else DEFAULT_MODEL
        return [args.name], max_samples, temperature, samples_per_prompt, model

    if args.experiment:
        max_samples = args.samples if args.samples is not None else 80
        temperature = _resolve_temperature(args, 0.3)
        samples_per_prompt = args.samples_per_prompt if args.samples_per_prompt is not None else DEFAULT_SAMPLES_PER_PROMPT
        model = args.model if args.model is not None else DEFAULT_MODEL
        return [args.experiment], max_samples, temperature, samples_per_prompt, model

    raise ValueError('Provide --preset, --experiment, or custom dedup parameters')


def main() -> None:
    parser = argparse.ArgumentParser(description='FunSearch experiment manager for stages 1 and 2')
    parser.add_argument('--preset', choices=sorted(PRESETS.keys()),
                        help='Run a preset experiment group: temp03_80 or no_temp_200')
    parser.add_argument('--experiment', choices=sorted(GROUPS.keys()),
                        help='Run only the selected experiment group')
    parser.add_argument('--name', type=str, default='custom',
                        help='Custom experiment name for custom dedup mode')
    parser.add_argument('--dedup-ast', type=int, choices=[0, 1],
                        help='Custom mode: enable AST deduplication')
    parser.add_argument('--dedup-behavior', type=int, choices=[0, 1],
                        help='Custom mode: enable behavioral-fingerprint deduplication')
    parser.add_argument('--samples', type=int,
                        help='Number of samples; preset default is used when omitted')
    parser.add_argument('--samples-per-prompt', type=int,
                        help='Samples generated per API call; preset default is used when omitted')
    parser.add_argument('--temperature', type=float,
                        help='Explicit temperature value')
    parser.add_argument('--no-temperature', action='store_true',
                        help='Do not pass the temperature parameter to the API')
    parser.add_argument('--seed', type=int, default=DEFAULT_RANDOM_SEED,
                        help=f'Random seed (default: {DEFAULT_RANDOM_SEED})')
    parser.add_argument('--model', type=str,
                        help=f'Model name (default: {DEFAULT_MODEL})')
    parser.add_argument('--overwrite', action='store_true',
                        help='Delete an existing target directory before rerunning')
    args = parser.parse_args()

    args.custom_dedup_mode = (
        args.dedup_ast is not None and args.dedup_behavior is not None
    )

    try:
        groups, max_samples, temperature, samples_per_prompt, model = _resolve_run_plan(args)
    except ValueError as exc:
        parser.error(str(exc))

    if args.custom_dedup_mode:
        custom_group = {
            'dedup_ast': bool(args.dedup_ast),
            'dedup_behavior': bool(args.dedup_behavior),
            'description': 'Custom-parameter experiment group',
        }
        group_map = {args.name: custom_group}
    else:
        group_map = GROUPS

    overall_start = time.time()
    results: dict[str, int] = {}
    for name in groups:
        rc = run_experiment(
            group_name=name,
            group_config=group_map[name],
            max_samples=max_samples,
            temperature=temperature,
            seed=args.seed,
            model=model,
            samples_per_prompt=samples_per_prompt,
            overwrite=args.overwrite,
        )
        results[name] = rc
        if rc != 0:
            print(f'Warning: experiment {name} returned non-zero status {rc}')

    overall_elapsed = time.time() - overall_start
    print(f"\n{'=' * 60}")
    print(f'All experiments completed. Total time: {overall_elapsed:.1f}s ({overall_elapsed / 60:.1f}min)')
    for name, rc in results.items():
        status = 'success' if rc == 0 else f'failed(code={rc})'
        print(f'  {name}: {status}')
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
