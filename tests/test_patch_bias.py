"""
tests/test_patch_bias.py

Tests for _select_biased_index and _selftest_patch_bias in
tools/nnunet_wandb_variant.py.

These are pure-function tests — no nnU-Net imports. The trainer
file's nnunetv2 imports would break this test on systems without
nnU-Net installed, so we extract the function source via ast and
exec it in an isolated namespace with only numpy as a dependency.

Run:
    /opt/conda/bin/python3 -m pytest tests/test_patch_bias.py -v
or:
    /opt/conda/bin/python3 tests/test_patch_bias.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_TOOLS_DIR = _THIS_DIR.parent / "tools"


def _extract_pure_functions():
    """Parse the trainer source and exec just the bits we need.

    Pulled bits:
      - module-level constants (label IDs, subtype names)
      - _select_biased_index function definition
      - _selftest_patch_bias function definition

    Standard library (random, sys) and numpy go into the namespace
    so the function bodies can call them.

    Compatibility tiers (newest → oldest Python):
      - 3.9+: ast.unparse (not used; we use the next two for fidelity)
      - 3.8+: ast.get_source_segment, OR slice via node.end_lineno
      - 3.7:  end_lineno not present on AST nodes — fall back to
              computing end from the next top-level node's start line.
              `tree.body` is a flat ordered list of module-level nodes,
              so for any node at index i, the next sibling at i+1
              starts after this node ends. Take end = sibling.lineno - 1
              (or len(src_lines) for the last node).
    """
    src = (_TOOLS_DIR / "nnunet_wandb_variant.py").read_text()
    tree = ast.parse(src)

    src_lines = src.splitlines(keepends=True)

    # Pre-compute end-line for each top-level node by looking at the
    # next sibling. This is the only fallback that works on Python 3.7,
    # which lacks node.end_lineno.
    sibling_end_line = {}
    body = tree.body
    for i, node in enumerate(body):
        if i + 1 < len(body):
            sibling_end_line[id(node)] = body[i + 1].lineno - 1
        else:
            sibling_end_line[id(node)] = len(src_lines)

    def _node_source(node):
        # Tier 1: ast.get_source_segment (Python 3.8+)
        try:
            seg = ast.get_source_segment(src, node)
            if seg is not None:
                return seg
        except AttributeError:
            pass
        # Tier 2: slice using node.end_lineno (also Python 3.8+)
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        # Tier 3 (Python 3.7): use next-sibling start line as our end
        if end is None:
            end = sibling_end_line.get(id(node))
        if start is None or end is None:
            raise RuntimeError(
                "AST node lacks lineno; cannot extract source. "
                "This test requires Python 3.7+ with the standard `ast` module.")
        # Python lineno is 1-indexed.
        return "".join(src_lines[start - 1:end])

    wanted_funcs = {"_select_biased_index", "_selftest_patch_bias"}
    wanted_consts = {
        "_L6_LABEL_ID", "_SACRUM_LABEL_ID",
        "_SUBTYPE_LUMB", "_SUBTYPE_SACR_COUNT", "_SUBTYPE_NORMAL",
        "_SUBTYPE_SEMI", "_SUBTYPE_SACR", "_SUBTYPE_AMBIG",
    }
    func_src = {}
    const_src = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted_funcs:
            func_src[node.name] = _node_source(node)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted_consts:
                    const_src[target.id] = _node_source(node)

    missing_funcs = wanted_funcs - set(func_src)
    missing_consts = wanted_consts - set(const_src)
    if missing_funcs:
        raise AssertionError(f"could not extract functions: {missing_funcs}")
    if missing_consts:
        raise AssertionError(f"could not extract constants: {missing_consts}")

    from typing import Callable, Dict, Optional, Sequence, Tuple
    import random as _random
    import sys as _sys
    ns = {
        "np": np,
        "Sequence": Sequence,
        "Dict": Dict,
        "Tuple": Tuple,
        "Callable": Callable,
        "Optional": Optional,
        "random": _random,
        "sys": _sys,
    }
    for code in const_src.values():
        exec(code, ns)
    for code in func_src.values():
        exec(code, ns)
    return ns


_NS = _extract_pure_functions()
_select_biased_index = _NS["_select_biased_index"]
_selftest_patch_bias = _NS["_selftest_patch_bias"]
_L6 = _NS["_L6_LABEL_ID"]
_SACRUM = _NS["_SACRUM_LABEL_ID"]
_LUMB = _NS["_SUBTYPE_LUMB"]
_SACR_COUNT = _NS["_SUBTYPE_SACR_COUNT"]


# ── Helpers ─────────────────────────────────────────────────────────────────

def _always_zero(): return 0.0
def _always_one():  return 1.0


# ── _select_biased_index: deterministic behavior ────────────────────────────

def test_lumb_with_l6_in_eligible_returns_l6_index():
    eligible = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    rules = {_LUMB: (_L6, "ENV", 1.0)}
    idx = _select_biased_index(eligible, rules, _always_zero)
    assert idx == 5, f"expected idx=5 (L6 position), got {idx}"
    assert eligible[idx] == _L6


def test_lumb_with_frac_zero_does_not_bias():
    eligible = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    rules = {_LUMB: (_L6, "ENV", 0.0)}
    idx = _select_biased_index(eligible, rules, _always_zero)
    assert idx is None


def test_lumb_with_frac_one_always_biases():
    eligible = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    rules = {_LUMB: (_L6, "ENV", 1.0)}
    for _ in range(10):
        idx = _select_biased_index(eligible, rules, lambda: 0.5)
        assert idx == 5


def test_sacr_count_classic_pattern():
    """L4 in fg, L5 NOT in fg, sacrum in fg -> sacr_count rule fires."""
    eligible = [1, 2, 3, 4, 7, 8, 9]
    rules = {_SACR_COUNT: (_SACRUM, "ENV", 1.0)}
    idx = _select_biased_index(eligible, rules, _always_zero)
    assert idx == 4
    assert eligible[idx] == _SACRUM


def test_sacr_count_does_not_fire_without_l4():
    """Pelvic-only normal case: sacrum in fg, no lumbars at all -> don't fire."""
    eligible = [7, 8, 9]
    rules = {_SACR_COUNT: (_SACRUM, "ENV", 1.0)}
    idx = _select_biased_index(eligible, rules, _always_zero)
    assert idx is None


def test_sacr_count_does_not_fire_with_l5_present():
    """L5 in fg means it's a normal anatomy case, not sacr_count."""
    eligible = [1, 2, 3, 4, 5, 7, 8, 9]
    rules = {_SACR_COUNT: (_SACRUM, "ENV", 1.0)}
    idx = _select_biased_index(eligible, rules, _always_zero)
    assert idx is None


def test_l6_present_takes_precedence_over_sacr_count():
    """If L6 is in the eligible list, it's a lumb case — skip the
    sacr_count rule entirely (don't fall through). Otherwise we'd
    have both rules potentially firing on the same case, which
    breaks the per-subtype emphasis design."""
    eligible = [1, 2, 3, 4, 6, 7, 8, 9]   # has L6, no L5
    rules = {
        _LUMB:       (_L6,     "L6_ENV",   0.0),  # lumb rule disabled
        _SACR_COUNT: (_SACRUM, "SACR_ENV", 1.0),  # sacr_count would otherwise fire
    }
    idx = _select_biased_index(eligible, rules, _always_zero)
    assert idx is None, \
        f"L6 presence should prevent sacr_count fall-through; got idx={idx}"


def test_no_rules_means_no_bias():
    eligible = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    idx = _select_biased_index(eligible, {}, _always_zero)
    assert idx is None


def test_empty_eligible_returns_none():
    idx = _select_biased_index([], {_LUMB: (_L6, "ENV", 1.0)}, _always_zero)
    assert idx is None


def test_eligible_with_tuple_keys_skips_them():
    """Region-based sampling adds tuple keys for "any foreground"
    regions. _select_biased_index must skip those (only int classes
    can be biased)."""
    eligible = [(1, 2, 3), 6, (4, 5)]  # L6 at index 1
    rules = {_LUMB: (_L6, "ENV", 1.0)}
    idx = _select_biased_index(eligible, rules, _always_zero)
    assert idx == 1
    assert eligible[idx] == _L6


def test_numpy_int_in_eligible_works():
    """Eligible may contain numpy ints (from class_locations.keys())."""
    eligible = [np.int64(1), np.int64(6), np.int64(7)]
    rules = {_LUMB: (_L6, "ENV", 1.0)}
    idx = _select_biased_index(eligible, rules, _always_zero)
    assert idx == 1


def test_frac_clipped_high_still_works():
    eligible = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    rules = {_LUMB: (_L6, "ENV", 1.5)}
    idx = _select_biased_index(eligible, rules, lambda: 0.99)
    assert idx == 5


# ── Statistical: bias frequency matches frac ────────────────────────────────

def test_bias_frequency_matches_frac_with_uniform_rand():
    """Use real random.random; bias frequency over many trials should
    converge near frac."""
    import random
    random.seed(42)
    eligible = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    rules = {_LUMB: (_L6, "ENV", 0.6)}
    n_trials = 5000
    n_biased = 0
    for _ in range(n_trials):
        idx = _select_biased_index(eligible, rules, random.random)
        if idx is not None:
            n_biased += 1
    observed_frac = n_biased / n_trials
    assert 0.55 < observed_frac < 0.65, \
        f"observed frac {observed_frac:.3f} far from expected 0.6"


# ── Self-test of the np.random.choice intercept mechanism ───────────────────

def test_selftest_runs_and_intercepts_every_call():
    """The end-to-end self-test should fire the intercept on every
    fake generate_train_batch call (frame inspection works)."""
    rules = {_LUMB: (_L6, "L6_ENV", 0.6)}
    n_trials, n_calls, n_l6 = _selftest_patch_bias(rules, n_trials=200)
    assert n_trials == 200
    assert n_calls == 200, \
        f"intercept should fire every call; got {n_calls}/200"


def test_selftest_with_zero_frac_gives_zero_l6_returns():
    """frac=0 -> rule never fires -> n_l6 == 0."""
    rules = {_LUMB: (_L6, "L6_ENV", 0.0)}
    n_trials, n_calls, n_l6 = _selftest_patch_bias(rules, n_trials=100)
    assert n_calls == 100
    assert n_l6 == 0


def test_selftest_with_full_frac_gives_full_l6_returns():
    """frac=1 -> rule always fires -> n_l6 == n_trials."""
    rules = {_LUMB: (_L6, "L6_ENV", 1.0)}
    n_trials, n_calls, n_l6 = _selftest_patch_bias(rules, n_trials=100)
    assert n_calls == 100
    assert n_l6 == 100


def test_selftest_restores_np_random_choice():
    """After self-test, np.random.choice must be back to original."""
    original = np.random.choice
    rules = {_LUMB: (_L6, "L6_ENV", 0.6)}
    _selftest_patch_bias(rules, n_trials=50)
    assert np.random.choice is original, \
        "np.random.choice was not restored after self-test"


# ── Standalone runner ───────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed.append(name)
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            failed.append(name)
    print()
    print(f"  ran {len(tests)} tests, {len(failed)} failed")
    if failed:
        sys.exit(1)
