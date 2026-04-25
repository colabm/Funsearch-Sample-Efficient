"""
FunSearch Bin Packing - Standalone runner using custom OpenAI-compatible API.

Usage:
    python run_funsearch.py

This script:
1. Uses the bltcy.ai API with gpt-5-nano model
2. Properly handles markdown code blocks in LLM responses
3. Includes debug logging to track why programs might be rejected
4. Supports local multiprocessing (not Jupyter)
"""
import sys
import os
import time
import re
import json
import multiprocessing
import traceback
from typing import Collection, Any

import numpy as np
import openai

# Ensure the project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Enable debug logging in evaluator.py (set before imports)
# Will be synced with DEBUG flag below after config section
os.environ.setdefault('FUNSEARCH_DEBUG', '0')

from implementation import funsearch
from implementation import config
from implementation import sampler
from implementation import evaluator
from implementation import evaluator_accelerate
from implementation import code_manipulation
import bin_packing_utils

# ============================================================================
# Configuration
# ============================================================================
API_KEY = os.environ.get('FUNSEARCH_API_KEY') or os.environ.get('OPENAI_API_KEY', '')
HOST_URL = os.environ.get('FUNSEARCH_BASE_URL') or os.environ.get('OPENAI_BASE_URL', 'https://api.bltcy.ai')
MODEL_NAME = os.environ.get('FUNSEARCH_MODEL', 'gpt-3.5-turbo')
MAX_TOKENS = 4096
SAMPLES_PER_PROMPT = int(os.environ.get('FUNSEARCH_SAMPLES_PER_PROMPT', '4'))
EVALUATE_TIMEOUT = 30
NUMBA_ACCELERATE = False  # Set to False to avoid numba issues with complex generated code
DEBUG = False or os.environ.get('FUNSEARCH_DEBUG', '0') == '1'  # Set True for verbose debug logging (or set FUNSEARCH_DEBUG=1 env var)

# Allow experiment scripts to override runner settings through environment variables.
MAX_SAMPLE_NUM = int(os.environ.get('FUNSEARCH_MAX_SAMPLES', '60'))
# FUNSEARCH_TEMPERATURE='none' means do not send temperature to the API.
_temp_env = os.environ.get('FUNSEARCH_TEMPERATURE', '0.8')
TEMPERATURE: float | None = None if _temp_env.lower() == 'none' else float(_temp_env)
ENABLE_DEDUP_AST = os.environ.get('FUNSEARCH_DEDUP_AST', '1') == '1'
ENABLE_DEDUP_BEHAVIOR = os.environ.get('FUNSEARCH_DEDUP_BEHAVIOR', '1') == '1'
LOG_SUFFIX = os.environ.get('FUNSEARCH_LOG_SUFFIX', 'default')
# Full timestamped log suffixes are used as-is; short suffixes get a funsearch_ prefix.
if LOG_SUFFIX.startswith('20') and '-funsearch_' in LOG_SUFFIX:
    LOG_DIR = os.path.join(PROJECT_ROOT, 'logs', LOG_SUFFIX)
else:
    LOG_DIR = os.path.join(PROJECT_ROOT, 'logs', f'funsearch_{LOG_SUFFIX}')

# Random seed for reproducibility.
RANDOM_SEED = int(os.environ.get('FUNSEARCH_SEED', '42'))


def _normalize_base_url(host_url: str) -> str:
    """Normalize OpenAI-compatible base URL to include '/v1' exactly once."""
    base = host_url.strip().rstrip('/')
    if base.endswith('/v1'):
        return base
    return f"{base}/v1"

# Dataset split: search instances for FunSearch and held-back OR3 test instances.
SEARCH_INSTANCES = [f'u500_{i:02d}' for i in range(14)]   # u500_00 ~ u500_13
TEST_INSTANCES = [f'u500_{i:02d}' for i in range(14, 20)]  # u500_14 ~ u500_19

# Sync env var so evaluator.py and sampler.py also get the debug flag
if DEBUG:
    os.environ['FUNSEARCH_DEBUG'] = '1'


# ============================================================================
# Robust trim function that handles markdown code blocks and various LLM output formats
# ============================================================================
def _trim_preface_of_body(sample: str) -> str:
    """Trim the LLM response to extract only the function body.

    Handles:
    - Markdown code blocks (```python ... ```)
    - Descriptive text before/after the function
    - Multiple def statements (picks the relevant one)
    - Code with or without the def line
    """
    if not sample or not sample.strip():
        return sample

    # Step 1: Remove markdown code fences if present
    # Handle ```python ... ``` and ``` ... ```
    cleaned = sample
    # Remove opening fence: ```python or ```
    cleaned = re.sub(r'^```(?:python|py)?\s*\n', '', cleaned, flags=re.MULTILINE)
    # Remove closing fence: ```
    cleaned = re.sub(r'\n```\s*$', '', cleaned, flags=re.MULTILINE)
    # Also handle fences that might be in the middle
    cleaned = re.sub(r'```(?:python|py)?\s*\n', '', cleaned)
    cleaned = re.sub(r'\n```', '\n', cleaned)

    lines = cleaned.splitlines()
    func_body_lineno = -1

    # Step 2: Find the first 'def' statement (handle indentation and whitespace)
    for lineno, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith('def '):
            func_body_lineno = lineno
            break

    # Step 3: Extract function body (everything after the def line)
    if func_body_lineno >= 0:
        body_lines = lines[func_body_lineno + 1:]
        # Find the end of the function body by looking for lines with proper indentation
        # The body should be indented; stop at the first non-indented, non-empty line
        # that isn't a comment or continuation
        result_lines = []
        for line in body_lines:
            # Empty lines are ok within a function body
            if line.strip() == '':
                result_lines.append(line)
                continue
            # Lines that are indented are part of the body
            if line[0] in (' ', '\t'):
                result_lines.append(line)
                continue
            # Non-indented non-empty line: likely description after the function
            # But could also be a decorator or another def — stop here
            break

        code = '\n'.join(result_lines)
        if code.strip():
            return code + '\n'

    # If no def found, try to return the cleaned text as-is
    # (it might already be just the function body)
    if cleaned.strip():
        return cleaned
    return sample


# ============================================================================
# LLM class using OpenAI-compatible API
# ============================================================================
class LLMAPI(sampler.LLM):
    """Language model using OpenAI-compatible API."""

    # Class-level counters aggregate token usage across LLMAPI instances.
    _cls_prompt_tokens = 0
    _cls_completion_tokens = 0
    _cls_api_calls = 0

    @classmethod
    def reset_token_usage(cls):
        cls._cls_prompt_tokens = 0
        cls._cls_completion_tokens = 0
        cls._cls_api_calls = 0

    @classmethod
    def get_token_usage(cls) -> dict:
        return {
            'api_calls': cls._cls_api_calls,
            'prompt_tokens': cls._cls_prompt_tokens,
            'completion_tokens': cls._cls_completion_tokens,
            'total_tokens': cls._cls_prompt_tokens + cls._cls_completion_tokens,
        }

    def __init__(self, samples_per_prompt: int, trim=True):
        super().__init__(samples_per_prompt)
        self._additional_prompt = (
            'Complete a different and more complex Python function. '
            'Be creative and you can insert multiple if-else and for-loop in the code logic. '
            'Only output the Python code, no descriptions. Do not use markdown code blocks. '
            'Do NOT include any comments or docstrings in the code. Output pure code only.'
        )
        self._trim = trim
        if not API_KEY:
            raise ValueError(
                'API key missing. Please set FUNSEARCH_API_KEY (preferred) '
                'or OPENAI_API_KEY before running.'
            )
        self._client = openai.OpenAI(
            base_url=_normalize_base_url(HOST_URL),
            api_key=API_KEY,
            timeout=120,
        )

    def draw_samples(self, prompt: str) -> Collection[str]:
        """Returns multiple predicted continuations of `prompt`."""
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]

    def _draw_sample(self, content: str) -> str:
        prompt = '\n'.join([content, self._additional_prompt])
        attempt = 0
        while True:
            attempt += 1
            try:
                # Reasoning models such as o1/o3/nano do not accept temperature.
                _is_reasoning = any(k in MODEL_NAME.lower() for k in ('nano', 'o1', 'o3'))
                api_kwargs: dict[str, Any] = dict(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=MAX_TOKENS,
                )
                if not _is_reasoning and TEMPERATURE is not None:
                    api_kwargs['temperature'] = TEMPERATURE

                response = self._client.chat.completions.create(**api_kwargs)
                raw_response = response.choices[0].message.content

                # Accumulate token usage for experiment reporting.
                if response.usage:
                    LLMAPI._cls_prompt_tokens += response.usage.prompt_tokens
                    LLMAPI._cls_completion_tokens += response.usage.completion_tokens
                    LLMAPI._cls_api_calls += 1

                if DEBUG:
                    print(f"\n{'='*60}")
                    print(f"[LLM RAW RESPONSE] (attempt {attempt}, {len(raw_response) if raw_response else 0} chars):")
                    if raw_response:
                        print(raw_response[:600])
                    else:
                        print("<EMPTY>")
                    print(f"{'='*60}")

                # Trim function
                if self._trim:
                    trimmed = _trim_preface_of_body(raw_response)
                    if DEBUG:
                        print(f"[TRIMMED RESPONSE] ({len(trimmed) if trimmed else 0} chars):")
                        if trimmed:
                            print(trimmed[:400])
                        else:
                            print("<EMPTY>")
                        print(f"{'='*60}\n")
                    return trimmed
                return raw_response

            except Exception as e:
                if DEBUG:
                    print(f"[LLM ERROR] Attempt {attempt}: {e}")
                if attempt >= 5:
                    print(f"[LLM ERROR] Failed after {attempt} attempts, returning empty string")
                    return "    pass\n"
                time.sleep(2)
                continue


# ============================================================================
# Sandbox class
# ============================================================================
class Sandbox(evaluator.Sandbox):
    """Sandbox for executing generated code with multiprocessing isolation."""

    def __init__(self, verbose=False, numba_accelerate=NUMBA_ACCELERATE):
        self._verbose = verbose
        self._numba_accelerate = numba_accelerate

    def run(
            self,
            program: str,
            function_to_run: str,
            function_to_evolve: str,
            inputs: Any,
            test_input: str,
            timeout_seconds: int,
            **kwargs
    ) -> tuple[Any, bool]:
        """Returns `function_to_run(test_input)` and whether execution succeeded."""
        dataset = inputs[test_input]
        try:
            result_queue = multiprocessing.Queue()
            process = multiprocessing.Process(
                target=self._compile_and_run_function,
                args=(program, function_to_run, function_to_evolve, dataset,
                      self._numba_accelerate, result_queue)
            )
            process.start()
            process.join(timeout=timeout_seconds)
            if process.is_alive():
                process.terminate()
                process.join()
                if DEBUG:
                    print(f"[SANDBOX] Timeout after {timeout_seconds}s")
                return None, False
            else:
                if not result_queue.empty():
                    results = result_queue.get_nowait()
                    if DEBUG and results[1]:
                        print(f"[SANDBOX] Execution succeeded, score: {results[0]}")
                    elif DEBUG:
                        print(f"[SANDBOX] Execution failed (returned False)")
                    return results
                else:
                    if DEBUG:
                        print(f"[SANDBOX] No result in queue (process likely crashed)")
                    return None, False
        except Exception as e:
            if DEBUG:
                print(f"[SANDBOX] Exception: {e}")
            return None, False

    def _compile_and_run_function(self, program, function_to_run, function_to_evolve,
                                  dataset, numba_accelerate, result_queue):
        try:
            if numba_accelerate:
                program = evaluator_accelerate.add_numba_decorator(
                    program=program,
                    function_to_evolve=function_to_evolve
                )
            all_globals_namespace = {}
            exec(program, all_globals_namespace)
            function_to_run = all_globals_namespace[function_to_run]
            results = function_to_run(dataset)
            if not isinstance(results, (int, float)):
                result_queue.put((None, False))
                return
            result_queue.put((results, True))
        except Exception as e:
            if DEBUG:
                # Print to stderr so it shows in the subprocess output
                import traceback
                traceback.print_exc()
            result_queue.put((None, False))


# ============================================================================
# Specification (template)
# ============================================================================
specification = r'''
import numpy as np


def get_valid_bin_indices(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns indices of bins in which item can fit."""
    return np.nonzero((bins - item) >= 0)[0]


def online_binpack(
        items: tuple[float, ...], bins: np.ndarray
) -> tuple[list[list[float, ...], ...], np.ndarray]:
    """Performs online binpacking of `items` into `bins`."""
    # Track which items are added to each bin.
    packing = [[] for _ in bins]
    # Add items to bins.
    for item in items:
        # Extract bins that have sufficient space to fit item.
        valid_bin_indices = get_valid_bin_indices(item, bins)
        # Score each bin based on heuristic.
        priorities = priority(item, bins[valid_bin_indices])
        # Add item to bin with highest priority.
        best_bin = valid_bin_indices[np.argmax(priorities)]
        bins[best_bin] -= item
        packing[best_bin].append(item)
    # Remove unused bins from packing.
    packing = [bin_items for bin_items in packing if bin_items]
    return packing, bins


@funsearch.run
def evaluate(instances: dict) -> float:
    """Evaluate heuristic function on a set of online binpacking instances."""
    # List storing number of bins used for each instance.
    num_bins = []
    # Perform online binpacking for each instance.
    for name in instances:
        instance = instances[name]
        capacity = instance['capacity']
        items = instance['items']
        # Create num_items bins so there will always be space for all items,
        # regardless of packing order. Array has shape (num_items,).
        bins = np.array([capacity for _ in range(instance['num_items'])])
        # Pack items into bins and return remaining capacity in bins_packed, which
        # has shape (num_items,).
        _, bins_packed = online_binpack(items, bins)
        # If remaining capacity in a bin is equal to initial capacity, then it is
        # unused. Count number of used bins.
        num_bins.append((bins_packed != capacity).sum())
    # Score of heuristic function is negative of average number of bins used
    # across instances (as we want to minimize number of bins).
    return -np.mean(num_bins)


@funsearch.evolve
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.

    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.

    Return:
        Array of same size as bins with priority score of each bin.
    """
    ratios = item / bins
    log_ratios = np.log(ratios)
    priorities = -log_ratios
    return priorities
'''


# ============================================================================
# Main entry point
# ============================================================================
if __name__ == '__main__':
    # Required for multiprocessing on macOS
    multiprocessing.set_start_method('spawn', force=True)

    # Set random seeds before sampling starts.
    import random
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    print(f"Random seed: {RANDOM_SEED}")

    # Build the search split from OR3 instances.
    or3_full = bin_packing_utils.datasets['OR3']
    search_data = {'OR3': {k: or3_full[k] for k in SEARCH_INSTANCES}}

    print(f"Starting FunSearch Bin Packing")
    print(f"API: {HOST_URL}, Model: {MODEL_NAME}")
    print(f"Max samples: {MAX_SAMPLE_NUM}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Search set: {len(SEARCH_INSTANCES)} instances ({SEARCH_INSTANCES[0]}~{SEARCH_INSTANCES[-1]})")
    print(f"Test set: {len(TEST_INSTANCES)} instances ({TEST_INSTANCES[0]}~{TEST_INSTANCES[-1]})")
    print(f"Dedup AST: {ENABLE_DEDUP_AST}, Behavior: {ENABLE_DEDUP_BEHAVIOR}")
    print(f"Numba acceleration: {NUMBA_ACCELERATE}")
    print(f"Log directory: {LOG_DIR}")
    print(f"Debug mode: {DEBUG}")
    print()

    # Create the optional deduplication filter according to environment flags.
    from implementation.dedup import DedupFilter
    dedup = DedupFilter(
        enable_ast=ENABLE_DEDUP_AST,
        enable_behavior=ENABLE_DEDUP_BEHAVIOR,
    ) if (ENABLE_DEDUP_AST or ENABLE_DEDUP_BEHAVIOR) else None

    class_config = config.ClassConfig(llm_class=LLMAPI, sandbox_class=Sandbox)
    cfg = config.Config(
        samples_per_prompt=SAMPLES_PER_PROMPT,
        evaluate_timeout_seconds=EVALUATE_TIMEOUT
    )

    funsearch.main(
        specification=specification,
        inputs=search_data,  # Search split only.
        config=cfg,
        max_sample_nums=MAX_SAMPLE_NUM,
        class_config=class_config,
        log_dir=LOG_DIR,
        dedup_filter=dedup,
    )

    # Save deduplication statistics for later notebook/report analysis.
    os.makedirs(LOG_DIR, exist_ok=True)
    if dedup:
        print(dedup.get_stats_summary())
        dedup_stats_file = os.path.join(LOG_DIR, 'dedup_stats.json')
        with open(dedup_stats_file, 'w') as f:
            json.dump(dedup.stats, f, indent=2, ensure_ascii=False)
        print(f"Dedup statistics saved: {dedup_stats_file}")
    else:
        # Save an empty statistics file even when deduplication is disabled.
        dedup_stats_file = os.path.join(LOG_DIR, 'dedup_stats.json')
        with open(dedup_stats_file, 'w') as f:
            json.dump({'total': 0, 'empty_filtered': 0, 'ast_filtered': 0,
                       'behavior_filtered': 0, 'passed': 0, 'note': 'dedup disabled'}, f, indent=2)

    # Save token usage summary for cost and efficiency analysis.
    token_usage = LLMAPI.get_token_usage()
    print(f"\n[TOKEN] API calls: {token_usage['api_calls']}")
    print(f"[TOKEN] Prompt tokens: {token_usage['prompt_tokens']}")
    print(f"[TOKEN] Completion tokens: {token_usage['completion_tokens']}")
    print(f"[TOKEN] Total tokens: {token_usage['total_tokens']}")
    token_file = os.path.join(LOG_DIR, 'token_usage.json')
    with open(token_file, 'w') as f:
        json.dump(token_usage, f, indent=2)
    print(f"Token statistics saved: {token_file}")

    print("\nFunSearch completed!")
