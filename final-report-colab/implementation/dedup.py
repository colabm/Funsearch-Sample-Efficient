# implementation/dedup.py
"""Code deduplication filter: AST normalization plus behavioral fingerprints."""
from __future__ import annotations

import ast
import hashlib
import copy
import os
import signal
import time
import numpy as np
from typing import Optional

_DEBUG = os.environ.get('FUNSEARCH_DEBUG', '0') == '1'


class _AlphaRenamer(ast.NodeTransformer):
    """Replace local variable names with v0, v1, v2... in encounter order.

    The transformer first registers assigned names and then rewrites loaded
    names, which avoids false matches caused by read order.
    """

    def __init__(self):
        self._name_map: dict[str, str] = {}
        self._counter = 0
        # Names that should remain stable: arguments, builtins, numpy aliases.
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
            # Store: register the local name and replace it with a canonical name.
            node.id = self._get_canonical_name(node.id)
        elif isinstance(node.ctx, ast.Load):
            # Load: only replace known locals; keep helper/global names intact.
            if node.id in self._name_map:
                node.id = self._name_map[node.id]
            elif node.id in self._reserved:
                node.id = node.id
            else:
                # Unknown symbols are external references, not alpha-renamable locals.
                node.id = node.id
        else:
            # Other contexts such as Del still refer to local symbols.
            node.id = self._get_canonical_name(node.id)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        # Rename the function object itself while preserving its argument names.
        node.name = self._get_canonical_name(node.name)
        self.generic_visit(node)
        return node

    def visit_arg(self, node):
        # Preserve argument names such as item and bins for semantic clarity.
        return node


class _ConstantNormalizer(ast.NodeTransformer):
    """Replace numeric constants with 0 and string constants with an empty string.

    Note: this transformer is currently disabled. Experiments showed that
    constant normalization can block the LLM's coefficient-tuning path and
    over-filter useful candidates. The course-required AST normalization only
    covers variable-name unification, comment/docstring removal, and formatting
    normalization. The class is kept as a reference for future ablations.
    """

    def visit_Constant(self, node):
        if isinstance(node.value, bool) or node.value is None:
            return node
        if isinstance(node.value, (int, float, complex)):
            node.value = 0
        elif isinstance(node.value, str):
            node.value = ''
        return node


class _DocstringRemover(ast.NodeTransformer):
    """Remove docstrings and standalone string-expression statements."""

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None  # Drop docstrings and standalone string comments.
        return self.generic_visit(node)


def normalize_code_ast(code: str) -> Optional[str]:
    """Normalize code through AST parsing, alpha-renaming, and serialization.

    Normalization steps, matching the course requirements:
    1. Remove docstrings and standalone string expressions.
    2. Alpha-rename local variables to v0, v1, v2...
    3. Serialize with ast.dump(), which removes formatting differences.

    Returns:
        Normalized AST dump string, or None when parsing fails.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    # Remove docstrings before structural hashing.
    tree = _DocstringRemover().visit(tree)
    ast.fix_missing_locations(tree)

    # Alpha-rename local variables for structure-level equivalence.
    tree = _AlphaRenamer().visit(tree)
    ast.fix_missing_locations(tree)

    return ast.dump(tree, annotate_fields=False)


def code_hash(code: str) -> Optional[str]:
    """Return the SHA256 hash of the normalized AST representation."""
    normalized = normalize_code_ast(code)
    if normalized is None:
        return None
    return hashlib.sha256(normalized.encode()).hexdigest()


def is_empty_body(code: str) -> bool:
    """Return whether a function body is empty except for docstrings or blanks."""
    lines = code.strip().splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Ignore docstring lines when detecting empty generated bodies.
        if stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        if stripped.startswith('"') or stripped.startswith("'"):
            continue
        # Any other line is executable code.
        return False
    return True


# ============================================================================
# Behavioral fingerprinting: legacy rank-order scheme kept for comparison.
# ============================================================================

# Legacy probe inputs are static (item, bins) pairs for one priority() call.
_LEGACY_PROBE_INPUTS = [
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

# Keep backward compatibility for older code that imports PROBE_INPUTS.
PROBE_INPUTS = _LEGACY_PROBE_INPUTS


def _legacy_compute_behavior_fingerprint(func_code: str, timeout: float = 2.0) -> Optional[str]:
    """Legacy rank-order behavioral fingerprint kept for comparison and tests.

    Limitations: static bin states only, roughly 40 dimensions, and no timeout
    protection.
    """
    try:
        namespace = {'np': np, 'numpy': np}
        exec(func_code, namespace)

        if 'priority' not in namespace:
            return None

        priority_fn = namespace['priority']
        outputs = []

        for item, bins in _LEGACY_PROBE_INPUTS:
            try:
                bins_copy = bins.copy()
                result = priority_fn(item, bins_copy)
                if result is None:
                    outputs.append(None)
                elif isinstance(result, np.ndarray):
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
# Behavioral fingerprinting: decision-sequence scheme (v2).
# ============================================================================
#
# Design:
#   - Run a full online bin-packing simulation for every probe instance.
#   - Record the chosen bin index at each step as the behavioral signature.
#   - Dynamic bin states expose policy differences at different fill levels.
#   - 10 probes with 35-40 items each give a 375-dimensional integer signature.
#   - Mix five OR3 subsets and five synthetic distributions for coverage.
#   - Use signal.SIGALRM to guard against infinite loops.

# --- Ten probe instances ---
# Five real OR3 subsets use the first 35-40 items from u500_00 to u500_04.
# Five synthetic distributions cover uniform, bimodal, monotone, and sawtooth cases.

PROBING_INSTANCES = [
        # === Real OR3 data ===
    {
        # First 40 items from OR3 u500_00, closest to the evaluation setting.
        'name': 'or3_u500_00',
        'capacity': 150,
        'items': [42, 69, 67, 57, 93, 90, 38, 36, 45, 42,
                  33, 79, 27, 57, 44, 84, 86, 92, 46, 38,
                  85, 33, 82, 73, 49, 70, 59, 23, 57, 72,
                  74, 69, 33, 42, 28, 46, 30, 64, 29, 74],  # 40 items
    },
    {
        # First 38 items from OR3 u500_01.
        'name': 'or3_u500_01',
        'capacity': 150,
        'items': [81, 39, 75, 66, 85, 36, 60, 56, 50, 75,
                  75, 37, 87, 95, 21, 99, 42, 57, 31, 37,
                  42, 40, 69, 91, 45, 97, 84, 90, 52, 43,
                  68, 53, 37, 65, 79, 73, 92, 87],  # 38 items
    },
    {
        # First 37 items from OR3 u500_02.
        'name': 'or3_u500_02',
        'capacity': 150,
        'items': [73, 39, 49, 79, 54, 57, 98, 69, 67, 49,
                  38, 34, 96, 27, 92, 82, 69, 45, 69, 20,
                  75, 97, 51, 70, 29, 91, 98, 77, 48, 45,
                  43, 61, 36, 82, 89, 94, 26],  # 37 items
    },
    {
        # First 36 items from OR3 u500_03.
        'name': 'or3_u500_03',
        'capacity': 150,
        'items': [64, 42, 86, 65, 47, 68, 20, 45, 69, 78,
                  44, 96, 50, 27, 58, 55, 81, 87, 76, 38,
                  79, 71, 60, 76, 91, 69, 77, 57, 33, 22,
                  76, 51, 66, 90, 34, 46],  # 36 items
    },
    {
        # First 39 items from OR3 u500_04.
        'name': 'or3_u500_04',
        'capacity': 150,
        'items': [68, 90, 38, 98, 44, 66, 76, 67, 65, 81,
                  95, 62, 34, 33, 56, 75, 40, 72, 49, 95,
                  59, 40, 53, 27, 70, 27, 72, 92, 79, 66,
                  92, 47, 87, 32, 51, 94, 22, 79, 75],  # 39 items
    },

    # === Synthetic distributions ===
    {
        # Medium uniform sizes create many half-filled bins for fine distinctions.
        'name': 'medium_uniform',
        'capacity': 150,
        'items': list(range(30, 68)),  # [30,31,...,67], 38 items
    },
    {
        # Bimodal sizes expose strategy differences when small items fill gaps.
        'name': 'bimodal',
        'capacity': 150,
        'items': [100 + i % 5 for i in range(18)]
              + [25 + i % 8 for i in range(17)],  # 35 items
    },
    {
        # Ascending order builds bin state early and stresses later large items.
        'name': 'ascending',
        'capacity': 150,
        'items': [25 + i * 3 for i in range(37)],  # [25,28,...,133], 37 items
    },
    {
        # Descending wide-range sequence tests FFD-like policy differences.
        'name': 'descending',
        'capacity': 200,
        'items': [140 - i * 3 for i in range(37)],  # [140,137,...,32], 37 items
    },
    {
        # Sawtooth sizes create repeated choice points under capacity 100.
        'name': 'sawtooth',
        'capacity': 100,
        'items': [(i % 6) * 8 + 25 for i in range(38)],  # 6-cycle: [25,33,41,49,57,65]×6+2, 38 items
    },
]

# Total fingerprint dimension = 40+38+37+36+39+38+35+37+37+38 = 375.
TOTAL_FINGERPRINT_DIM = sum(len(p['items']) for p in PROBING_INSTANCES)
assert TOTAL_FINGERPRINT_DIM == 375, (
    f"Fingerprint dimension should be 375, got {TOTAL_FINGERPRINT_DIM}"
)


def _run_single_probe(priority_fn, capacity: int, items: list[int]) -> tuple[int, ...]:
    """Run one online bin-packing probe and return the chosen-bin sequence.

    This lightweight simulation is independent of the sandbox and does not use
    multiprocessing.

    Args:
        priority_fn: priority(item, bins) -> np.ndarray function.
        capacity: Bin capacity.
        items: Item-size list.

    Returns:
        Decision sequence tuple (bin_index_0, bin_index_1, ..., bin_index_n).
    """
    bins = np.array([float(capacity)], dtype=np.float64)
    decisions = []
    for item in items:
        # Find bins whose remaining capacity can fit the current item.
        valid = np.where(bins - item >= 0)[0]
        if len(valid) == 0:
            # Open a new bin if no existing bin can fit the item.
            bins = np.append(bins, float(capacity))
            valid = np.array([len(bins) - 1])
        # Score each feasible bin with the candidate priority function.
        priorities = priority_fn(float(item), bins[valid].copy())
        if priorities is None:
            # Match evaluator semantics: no output falls back to the first feasible bin.
            best = valid[0]
        else:
            try:
                best_idx = int(np.argmax(priorities))
                if 0 <= best_idx < len(valid):
                    best = valid[best_idx]
                else:
                    best = valid[0]
            except Exception:
                # Bad return types should not crash the probe simulation.
                best = valid[0]
        bins[best] -= item
        decisions.append(int(best))
    return tuple(decisions)


class _ProbeTimeoutError(Exception):
    """Raised when probe execution exceeds the timeout."""
    pass


def _timeout_handler(signum, frame):
    raise _ProbeTimeoutError("Probe execution timed out")


def compute_behavior_fingerprint(func_code: str, timeout: float = 5.0) -> Optional[str]:
    """Run priority() on 10 probes and hash the decision-sequence fingerprint.

    Decision-sequence scheme (v2):
    - Run a full bin-packing simulation for each probe.
    - Record the chosen bin index for every item.
    - 10 probes with 35-40 items produce a 375-dimensional integer signature.
    - Use signal.SIGALRM for timeout protection on macOS/Linux.

    Args:
        func_code: Full executable program code containing priority().
        timeout: Total timeout in seconds for all probes.

    Returns:
        SHA256 hash of the behavioral fingerprint, or None on failure.
    """
    try:
        namespace = {'np': np, 'numpy': np}
        exec(func_code, namespace)

        if 'priority' not in namespace:
            return None

        priority_fn = namespace['priority']

        # Enable timeout protection where SIGALRM is available.
        use_alarm = hasattr(signal, 'SIGALRM')
        if use_alarm:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(int(timeout))

        try:
            all_decisions: list[int] = []
            for probe in PROBING_INSTANCES:
                seq = _run_single_probe(
                    priority_fn, probe['capacity'], probe['items']
                )
                all_decisions.extend(seq)

            fingerprint = tuple(all_decisions)
            fingerprint_str = str(fingerprint)
            return hashlib.sha256(fingerprint_str.encode()).hexdigest()

        finally:
            if use_alarm:
                signal.alarm(0)  # Cancel the active timer.
                signal.signal(signal.SIGALRM, old_handler)

    except Exception:
        # Any parse, timeout, or runtime error disables only the behavior layer.
        # The caller can still proceed to normal sandbox evaluation.
        return None


# ============================================================================
# DedupFilter main class.
# ============================================================================

class DedupFilter:
    """Two-layer pre-evaluation deduplication filter.

    Layer 1: AST-normalization deduplication through hash comparison.
    Layer 2: Behavioral-fingerprint deduplication through probe execution.
    """

    def __init__(self, enable_ast: bool = True, enable_behavior: bool = True):
        self._enable_ast = enable_ast
        self._enable_behavior = enable_behavior

        # Hash sets for candidates that have already been admitted.
        self._ast_hashes: set[str] = set()
        self._behavior_hashes: set[str] = set()

        # Aggregate counters used in experiment reports.
        self.stats = {
            'total': 0,
            'empty_filtered': 0,
            'ast_filtered': 0,
            'behavior_filtered': 0,
            'passed': 0,
            'dedup_total_time_ms': 0.0,
        }

        # Details for the most recent should_evaluate() call.
        self._last_check_time_ms: float = 0.0
        self._last_check_level: str = ''

    def should_evaluate(self, function_body: str, full_program: str = '') -> bool:
        """Return whether the generated function should enter full evaluation.

        Args:
            function_body: Function body code without the def line.
            full_program: Full executable program used for behavioral probes.

        Returns:
            True for a non-duplicate candidate that should be evaluated; False
            for a duplicate that should be skipped.

        Side effects:
            Updates self._last_check_time_ms and self._last_check_level.
        """
        t_start = time.perf_counter()
        self.stats['total'] += 1

        # Filter empty bodies before computing more expensive fingerprints.
        if is_empty_body(function_body):
            self.stats['empty_filtered'] += 1
            self._record_check_time(t_start, 'empty')
            if _DEBUG:
                print(f"[DEDUP] Empty body filtered (total={self.stats['total']}, "
                      f"time={self._last_check_time_ms:.2f}ms)")
            return False

        # Layer 1: AST hash deduplication.
        if self._enable_ast:
            wrapped = f"def priority(item, bins):\n{function_body}"
            h = code_hash(wrapped)
            if h is not None:
                if h in self._ast_hashes:
                    self.stats['ast_filtered'] += 1
                    self._record_check_time(t_start, 'ast')
                    if _DEBUG:
                        print(f"[DEDUP] AST duplicate filtered (total={self.stats['total']}, "
                              f"time={self._last_check_time_ms:.2f}ms)")
                    return False
                self._ast_hashes.add(h)

        # Layer 2: behavioral-fingerprint deduplication.
        if self._enable_behavior and full_program:
            fp = compute_behavior_fingerprint(full_program)
            if fp is not None:
                if fp in self._behavior_hashes:
                    self.stats['behavior_filtered'] += 1
                    self._record_check_time(t_start, 'behavior')
                    if _DEBUG:
                        print(f"[DEDUP] Behavior duplicate filtered (total={self.stats['total']}, "
                              f"time={self._last_check_time_ms:.2f}ms)")
                    return False
                self._behavior_hashes.add(fp)

        self.stats['passed'] += 1
        self._record_check_time(t_start, 'passed')
        return True

    def _record_check_time(self, t_start: float, level: str) -> None:
        """Record elapsed time for the current deduplication check."""
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self._last_check_time_ms = elapsed_ms
        self._last_check_level = level
        self.stats['dedup_total_time_ms'] += elapsed_ms

    def get_stats_summary(self) -> str:
        """Return a human-readable statistics summary."""
        s = self.stats
        total = s['total'] or 1  # Avoid division by zero.
        avg_time = s['dedup_total_time_ms'] / total
        return (
            f"[DEDUP] Statistics:\n"
            f"  LLM-generated total: {s['total']}\n"
            f"  Empty bodies filtered: {s['empty_filtered']} ({s['empty_filtered']/total*100:.1f}%)\n"
            f"  AST duplicates filtered: {s['ast_filtered']} ({s['ast_filtered']/total*100:.1f}%)\n"
            f"  Behavior duplicates filtered: {s['behavior_filtered']} ({s['behavior_filtered']/total*100:.1f}%)\n"
            f"  Entered evaluation: {s['passed']} ({s['passed']/total*100:.1f}%)\n"
            f"  Dedup saving rate: {(s['ast_filtered']+s['behavior_filtered'])/total*100:.1f}%\n"
            f"  Total dedup time: {s['dedup_total_time_ms']:.1f}ms (avg {avg_time:.2f}ms/sample)"
        )
