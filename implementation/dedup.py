# implementation/dedup.py
"""代码去重过滤器：AST 标准化 + 行为指纹."""
from __future__ import annotations

import ast
import hashlib
import copy
import os
import time
import numpy as np
from typing import Optional

_DEBUG = os.environ.get('FUNSEARCH_DEBUG', '0') == '1'


class _AlphaRenamer(ast.NodeTransformer):
    """将所有局部变量名替换为 v0, v1, v2...（按出现顺序）.

    两阶段处理：先扫描所有 Store（赋值）节点建立映射，
    再处理 Load（读取）节点，避免因读取顺序导致的误判。
    """

    def __init__(self):
        self._name_map: dict[str, str] = {}
        self._counter = 0
        # 保留的名称（参数名、内置函数、numpy 等）
        self._reserved = {'item', 'bins', 'np', 'numpy', 'range', 'len', 'max',
                          'min', 'abs', 'float', 'int', 'True', 'False', 'None',
                          'print', 'isinstance', 'type', 'sum', 'sorted',
                          'enumerate', 'zip', 'map', 'filter', 'list', 'tuple',
                          'dict', 'set', 'str', 'bool', 'math'}

    def _get_canonical_name(self, name: str) -> str:
        if name in self._reserved:
            return name
        if name not in self._name_map:
            self._name_map[name] = f'v{self._counter}'
            self._counter += 1
        return self._name_map[name]

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            # Store: 注册并替换
            node.id = self._get_canonical_name(node.id)
        elif isinstance(node.ctx, ast.Load):
            # Load: 只替换已注册的名称，保留未知名称原样
            if node.id in self._name_map:
                node.id = self._name_map[node.id]
            elif node.id not in self._reserved:
                # 未见过的 Load 名称（闭包变量等），仍注册以保持一致性
                node.id = self._get_canonical_name(node.id)
        else:
            # Del 等其他上下文
            node.id = self._get_canonical_name(node.id)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        # 重命名函数名但保留参数名
        node.name = self._get_canonical_name(node.name)
        self.generic_visit(node)
        return node

    def visit_arg(self, node):
        # 保留参数名不变（item, bins 等）
        return node


class _DocstringRemover(ast.NodeTransformer):
    """移除 docstring 和字符串常量表达式."""

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None  # 删除 docstring
        return self.generic_visit(node)


def normalize_code_ast(code: str) -> Optional[str]:
    """将代码 AST 标准化：统一变量名、去 docstring、序列化.

    Returns:
        标准化后的 AST dump 字符串，解析失败返回 None.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    # 移除 docstring
    tree = _DocstringRemover().visit(tree)
    ast.fix_missing_locations(tree)

    # Alpha 重命名
    tree = _AlphaRenamer().visit(tree)
    ast.fix_missing_locations(tree)

    return ast.dump(tree, annotate_fields=False)


def code_hash(code: str) -> Optional[str]:
    """返回代码 AST 标准化后的 SHA256 hash."""
    normalized = normalize_code_ast(code)
    if normalized is None:
        return None
    return hashlib.sha256(normalized.encode()).hexdigest()


def is_empty_body(code: str) -> bool:
    """检测函数体是否为空（仅有 docstring 或空白）."""
    lines = code.strip().splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # 跳过 docstring 行
        if stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        if stripped.startswith('"') or stripped.startswith("'"):
            continue
        # 有实际代码行
        return False
    return True


# ============================================================================
# 行为指纹
# ============================================================================

# Probe 输入：覆盖边界/典型/特殊情况
PROBE_INPUTS = [
    (50.0,  np.array([150.0, 100.0, 50.0, 30.0])),
    (100.0, np.array([100.0, 100.0, 100.0])),
    (20.0,  np.array([150.0, 140.0, 130.0, 120.0])),
    (100.0, np.array([50.0, 30.0, 20.0])),
    (75.0,  np.array([150.0, 75.0, 80.0, 60.0, 40.0])),
    (1.0,   np.array([150.0])),
    (149.0, np.array([150.0, 100.0])),
    (50.0,  np.array([50.0, 51.0, 49.0, 100.0, 25.0])),
    (80.0,  np.array([150.0, 80.0, 85.0, 120.0])),
    (60.0,  np.array([90.0, 60.0, 120.0, 45.0, 150.0, 30.0, 75.0, 100.0])),
]


def compute_behavior_fingerprint(func_code: str, timeout: float = 2.0) -> Optional[str]:
    """运行 priority 函数在 probe 输入上，收集输出指纹.

    Args:
        func_code: 完整可执行的程序代码（包含 priority 函数定义）
        timeout: 单次 probe 超时秒数

    Returns:
        行为指纹的 SHA256 hash，失败返回 None.
    """
    try:
        namespace = {'np': np, 'numpy': np}
        exec(func_code, namespace)

        if 'priority' not in namespace:
            return None

        priority_fn = namespace['priority']
        outputs = []

        for item, bins in PROBE_INPUTS:
            try:
                bins_copy = bins.copy()  # 防止被修改
                result = priority_fn(item, bins_copy)
                if result is None:
                    outputs.append(None)
                elif isinstance(result, np.ndarray):
                    # 只保留排名顺序：priority 输出经 argmax 决策，
                    # 仅排名影响装箱行为，数值本身不重要
                    rank = tuple(np.argsort(result).tolist())
                    outputs.append(rank)
                else:
                    outputs.append(round(float(result), 4))
            except Exception:
                outputs.append('ERROR')

        fingerprint_str = str(tuple(outputs))
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()

    except Exception:
        return None


# ============================================================================
# DedupFilter 主类
# ============================================================================

class DedupFilter:
    """前置双层去重过滤器.

    Layer 1: AST 标准化去重 (hash 比较)
    Layer 2: 行为指纹去重 (probe 运行)
    """

    def __init__(self, enable_ast: bool = True, enable_behavior: bool = True):
        self._enable_ast = enable_ast
        self._enable_behavior = enable_behavior

        # 已见过的 hash 集合
        self._ast_hashes: set[str] = set()
        self._behavior_hashes: set[str] = set()

        # 统计
        self.stats = {
            'total': 0,
            'empty_filtered': 0,
            'ast_filtered': 0,
            'behavior_filtered': 0,
            'passed': 0,
        }

    def should_evaluate(self, function_body: str, full_program: str = '') -> bool:
        """判断该函数是否应该进入完整评估.

        Args:
            function_body: 函数体代码（不含 def 行）
            full_program: 完整可执行的程序代码（用于行为指纹）

        Returns:
            True 表示应评估（非重复），False 表示应跳过（重复）
        """
        self.stats['total'] += 1

        # 空函数（仅含 docstring 或空白）直接拦截。
        # 空函数 return None → Sandbox 得到 None → scores_per_test 为空 →
        # 不会写入 ProgramsDatabase，只浪费沙箱评估时间。
        if is_empty_body(function_body):
            self.stats['empty_filtered'] += 1
            if _DEBUG:
                print(f"[DEDUP] 空函数拦截 (total={self.stats['total']})")
            return False

        # Layer 1: AST 去重
        if self._enable_ast:
            # 包装为完整函数以便 AST 解析
            wrapped = f"def priority(item, bins):\n{function_body}"
            h = code_hash(wrapped)
            if h is not None:
                if h in self._ast_hashes:
                    self.stats['ast_filtered'] += 1
                    if _DEBUG:
                        print(f"[DEDUP] AST 去重拦截 (total={self.stats['total']})")
                    return False
                self._ast_hashes.add(h)

        # Layer 2: 行为指纹去重
        if self._enable_behavior and full_program:
            fp = compute_behavior_fingerprint(full_program)
            if fp is not None:
                if fp in self._behavior_hashes:
                    self.stats['behavior_filtered'] += 1
                    if _DEBUG:
                        print(f"[DEDUP] 行为指纹拦截 (total={self.stats['total']})")
                    return False
                self._behavior_hashes.add(fp)

        self.stats['passed'] += 1
        return True

    def get_stats_summary(self) -> str:
        """返回统计摘要字符串."""
        s = self.stats
        total = s['total'] or 1  # 避免除零
        return (
            f"[DEDUP] 统计:\n"
            f"  LLM 生成总数: {s['total']}\n"
            f"  空函数(已放行): {s['empty_filtered']} ({s['empty_filtered']/total*100:.1f}%)\n"
            f"  AST 去重拦截: {s['ast_filtered']} ({s['ast_filtered']/total*100:.1f}%)\n"
            f"  行为指纹拦截: {s['behavior_filtered']} ({s['behavior_filtered']/total*100:.1f}%)\n"
            f"  进入评估: {s['passed']} ({s['passed']/total*100:.1f}%)\n"
            f"  去重节省率: {(s['ast_filtered']+s['behavior_filtered'])/total*100:.1f}%"
        )
