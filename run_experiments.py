# run_experiments.py
"""统一实验管理脚本.

用法:
    # 顺序运行4组正式实验（前台）
    python run_experiments.py --formal

    # 后台运行（推荐，避免终端超时）
    nohup python run_experiments.py --formal > logs/all_experiments.log 2>&1 &

    # 运行指定实验
    python run_experiments.py --experiment exp_no_dedup

    # 快速验证
    python run_experiments.py --quick
"""
import subprocess
import sys
import json
import os
import time
from datetime import datetime

PYTHON = '/Users/zs/A/Coding/cityu_homework/cs5491/project_test/venv/bin/python'
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_ROOT = os.path.join(PROJECT_ROOT, 'logs')

# 每组实验使用相同的随机种子，保证可复现
RANDOM_SEED = 42

# 第二轮实验日期前缀
ROUND2_PREFIX = '2026-03-31-1'
# 第三轮实验日期前缀（修复去重算法后重跑）
ROUND3_PREFIX = '2026-03-31-2'

# 实验配置
EXPERIMENTS = {
    # 实验 0: 快速验证
    'quick_baseline': {
        'max_samples': 20,
        'dedup_ast': False,
        'dedup_behavior': False,
        'log_suffix': 'quick_baseline',
        'temperature': 0.8,
    },
    'quick_dedup': {
        'max_samples': 20,
        'dedup_ast': True,
        'dedup_behavior': True,
        'log_suffix': 'quick_dedup',
        'temperature': 0.8,
    },
    # ---- 第一轮 (temp=0.8) — 已完成, 保留旧日志 ----
    # 'exp_no_dedup', 'exp_ast_only', 'exp_behavior_only', 'exp_full_dedup'
    # 日志在 logs/funsearch_exp_*

    # ---- 第二轮 (temp=0.3) — 新实验 ----
    'r2_no_dedup': {
        'max_samples': 200,
        'dedup_ast': False,
        'dedup_behavior': False,
        'log_suffix': f'{ROUND2_PREFIX}-funsearch_exp_no_dedup',
        'temperature': 0.3,
    },
    'r2_ast_only': {
        'max_samples': 200,
        'dedup_ast': True,
        'dedup_behavior': False,
        'log_suffix': f'{ROUND2_PREFIX}-funsearch_exp_ast_only',
        'temperature': 0.3,
    },
    'r2_behavior_only': {
        'max_samples': 200,
        'dedup_ast': False,
        'dedup_behavior': True,
        'log_suffix': f'{ROUND2_PREFIX}-funsearch_exp_behavior_only',
        'temperature': 0.3,
    },
    'r2_full_dedup': {
        'max_samples': 200,
        'dedup_ast': True,
        'dedup_behavior': True,
        'log_suffix': f'{ROUND2_PREFIX}-funsearch_exp_full_dedup',
        'temperature': 0.3,
    },

    # ---- 第三轮 (temp=0.3, 修复去重算法后) — 80 samples ----
    'r3_no_dedup': {
        'max_samples': 80,
        'dedup_ast': False,
        'dedup_behavior': False,
        'log_suffix': f'{ROUND3_PREFIX}-funsearch_exp_no_dedup',
        'temperature': 0.3,
    },
    'r3_ast_only': {
        'max_samples': 80,
        'dedup_ast': True,
        'dedup_behavior': False,
        'log_suffix': f'{ROUND3_PREFIX}-funsearch_exp_ast_only',
        'temperature': 0.3,
    },
    'r3_behavior_only': {
        'max_samples': 80,
        'dedup_ast': False,
        'dedup_behavior': True,
        'log_suffix': f'{ROUND3_PREFIX}-funsearch_exp_behavior_only',
        'temperature': 0.3,
    },
    'r3_full_dedup': {
        'max_samples': 80,
        'dedup_ast': True,
        'dedup_behavior': True,
        'log_suffix': f'{ROUND3_PREFIX}-funsearch_exp_full_dedup',
        'temperature': 0.3,
    },
}

# 正式实验顺序 (第二轮)
FORMAL_ORDER = ['r2_no_dedup', 'r2_ast_only', 'r2_behavior_only', 'r2_full_dedup']
# 第三轮实验顺序 (修复后重跑)
ROUND3_ORDER = ['r3_no_dedup', 'r3_ast_only', 'r3_behavior_only', 'r3_full_dedup']


def run_experiment(name: str, exp_config: dict, seed: int = RANDOM_SEED):
    """运行单个实验, 输出同时写到 stdout 和日志文件."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n{'='*60}")
    print(f"[{timestamp}] 开始实验: {name}")
    print(f"配置: {exp_config}")
    print(f"随机种子: {seed}")
    print(f"{'='*60}\n", flush=True)

    env = os.environ.copy()
    env['FUNSEARCH_MAX_SAMPLES'] = str(exp_config['max_samples'])
    env['FUNSEARCH_DEDUP_AST'] = '1' if exp_config['dedup_ast'] else '0'
    env['FUNSEARCH_DEDUP_BEHAVIOR'] = '1' if exp_config['dedup_behavior'] else '0'
    env['FUNSEARCH_LOG_SUFFIX'] = exp_config['log_suffix']
    env['FUNSEARCH_SEED'] = str(seed)
    env['FUNSEARCH_TEMPERATURE'] = str(exp_config.get('temperature', 0.8))

    # 实验专用日志文件
    log_suffix = exp_config['log_suffix']
    if log_suffix.startswith('20') and '-funsearch_' in log_suffix:
        exp_log_dir = os.path.join(LOG_ROOT, log_suffix)
    else:
        exp_log_dir = os.path.join(LOG_ROOT, f"funsearch_{log_suffix}")
    os.makedirs(exp_log_dir, exist_ok=True)
    exp_log_file = os.path.join(exp_log_dir, 'experiment_output.log')

    start = time.time()

    # 同时输出到 stdout 和日志文件
    with open(exp_log_file, 'w') as f:
        result = subprocess.run(
            [PYTHON, 'run_funsearch.py'],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )

    elapsed = time.time() - start

    status = '成功' if result.returncode == 0 else f'失败(code={result.returncode})'
    timestamp_end = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp_end}] 实验 {name} {status}, 耗时 {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # 保存实验元信息
    meta = {
        'name': name,
        'config': exp_config,
        'seed': seed,
        'start_time': timestamp,
        'end_time': timestamp_end,
        'elapsed_seconds': round(elapsed, 1),
        'return_code': result.returncode,
    }
    meta_file = os.path.join(exp_log_dir, 'experiment_meta.json')
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"日志: {exp_log_file}", flush=True)
    print(f"元信息: {meta_file}", flush=True)
    return result.returncode


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='FunSearch 实验管理脚本')
    parser.add_argument('--quick', action='store_true', help='只运行快速验证实验')
    parser.add_argument('--formal', action='store_true', help='运行4组正式对比实验 (第二轮)')
    parser.add_argument('--round3', action='store_true', help='运行第三轮实验 (修复去重算法后, 4×80 samples)')
    parser.add_argument('--experiment', type=str, help='运行指定实验')
    parser.add_argument('--skip', type=str, nargs='+', default=[], help='跳过指定实验 (配合 --formal/--round3 使用)')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help=f'随机种子 (默认: {RANDOM_SEED})')
    args = parser.parse_args()

    os.makedirs(LOG_ROOT, exist_ok=True)

    overall_start = time.time()
    results = {}

    if args.experiment:
        if args.experiment in EXPERIMENTS:
            rc = run_experiment(args.experiment, EXPERIMENTS[args.experiment], seed=args.seed)
            results[args.experiment] = rc
        else:
            print(f"未知实验: {args.experiment}")
            print(f"可用实验: {list(EXPERIMENTS.keys())}")
            sys.exit(1)
    elif args.quick:
        for name in ['quick_baseline', 'quick_dedup']:
            rc = run_experiment(name, EXPERIMENTS[name], seed=args.seed)
            results[name] = rc
    elif args.formal:
        for name in FORMAL_ORDER:
            if name in args.skip:
                print(f"跳过实验: {name}")
                continue
            rc = run_experiment(name, EXPERIMENTS[name], seed=args.seed)
            results[name] = rc
            if rc != 0:
                print(f"警告: 实验 {name} 返回非零状态码 {rc}, 继续下一组...")
    elif args.round3:
        for name in ROUND3_ORDER:
            if name in args.skip:
                print(f"跳过实验: {name}")
                continue
            rc = run_experiment(name, EXPERIMENTS[name], seed=args.seed)
            results[name] = rc
            if rc != 0:
                print(f"警告: 实验 {name} 返回非零状态码 {rc}, 继续下一组...")
    else:
        # 默认运行所有实验
        for name, exp_config in EXPERIMENTS.items():
            rc = run_experiment(name, exp_config, seed=args.seed)
            results[name] = rc

    overall_elapsed = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"全部实验完成! 总耗时: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f}min)")
    print(f"结果汇总:")
    for name, rc in results.items():
        status = '✓ 成功' if rc == 0 else f'✗ 失败(code={rc})'
        print(f"  {name}: {status}")
    print(f"{'='*60}")
