"""
tests/test_lstv_dedicated_val.py

Tests for v17.1 dedicated LSTV validation pass.

The dedicated pass forward-runs every LSTV case in a fold's val set
every epoch, so per-subgroup metrics like
val/lstv_subgroup/lumb/dice/L6 are deterministic instead of
subsampling-noisy. Tests here cover the pure-Python logic — key
filtering, buffer-slot clearing, env-var resolution, no-op
short-circuits — without instantiating an nnU-Net trainer or
needing a network.

The full forward+record path (which requires a network and CUDA)
is exercised end-to-end at runtime by the SLURM training job; the
trainer-side log line "LSTV dedicated val (epoch N): forward-passed
M/M LSTV val cases" confirms it.

Run:
    /opt/conda/bin/python3 -m pytest tests/test_lstv_dedicated_val.py -v

Or, without pytest (stdlib):
    /opt/conda/bin/python3 tests/test_lstv_dedicated_val.py
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# We extract the relevant logic from the trainer source via AST. This
# avoids needing nnU-Net / torch / wandb installed in the test
# environment. Same approach as test_patch_bias.py.

_THIS_DIR = Path(__file__).resolve().parent
_TOOLS_DIR = _THIS_DIR.parent / "tools"


def _extract_constants_from_trainer():
    """Read the trainer source, eval the small constants we need."""
    src = (_TOOLS_DIR / "nnunet_wandb_variant.py").read_text()
    tree = ast.parse(src)
    src_lines = src.splitlines(keepends=True)

    # Pre-compute end-line for each top-level node (Python 3.7 compat).
    sibling_end_line = {}
    body = tree.body
    for i, node in enumerate(body):
        if i + 1 < len(body):
            sibling_end_line[id(node)] = body[i + 1].lineno - 1
        else:
            sibling_end_line[id(node)] = len(src_lines)

    def _node_source(node):
        try:
            seg = ast.get_source_segment(src, node)
            if seg is not None:
                return seg
        except AttributeError:
            pass
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        if end is None:
            end = sibling_end_line.get(id(node))
        if start is None or end is None:
            raise RuntimeError("AST node lacks lineno; need Python 3.7+")
        return "".join(src_lines[start - 1:end])

    wanted_consts = {
        "_SUBTYPE_NORMAL", "_SUBTYPE_LUMB", "_SUBTYPE_SACR_COUNT",
        "_SUBTYPE_SEMI", "_SUBTYPE_SACR", "_SUBTYPE_AMBIG",
        "_KNOWN_SUBTYPES", "_OVERSAMPLE_SUBTYPES",
    }
    const_src = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted_consts:
                    const_src[target.id] = _node_source(node)

    missing = wanted_consts - set(const_src)
    if missing:
        raise AssertionError(f"could not extract constants: {missing}")

    ns = {}
    for code in const_src.values():
        exec(code, ns)
    return ns


_NS = _extract_constants_from_trainer()
_SUBTYPE_NORMAL = _NS["_SUBTYPE_NORMAL"]
_SUBTYPE_LUMB = _NS["_SUBTYPE_LUMB"]
_SUBTYPE_SACR_COUNT = _NS["_SUBTYPE_SACR_COUNT"]
_SUBTYPE_SEMI = _NS["_SUBTYPE_SEMI"]
_SUBTYPE_SACR = _NS["_SUBTYPE_SACR"]
_SUBTYPE_AMBIG = _NS["_SUBTYPE_AMBIG"]
_KNOWN_SUBTYPES = _NS["_KNOWN_SUBTYPES"]
_OVERSAMPLE_SUBTYPES = _NS["_OVERSAMPLE_SUBTYPES"]


# ─── Pure-logic helpers (copies of the trainer-side logic) ──────────────────
# These are intentionally separate from the trainer source so the tests
# can exercise them deterministically without instantiating the mixin.

def _filter_lstv_val_keys(
    val_keys: List[str],
    case_to_subtype: Dict[str, str],
    lstv_subtypes: Set[str],
) -> List[str]:
    """Mirror of the filter inside _validate_lstv_dedicated."""
    return [
        k for k in val_keys
        if case_to_subtype.get(k, _SUBTYPE_NORMAL) in lstv_subtypes
    ]


def _clear_lstv_slots(
    buffer: Dict[str, list],
    lstv_subtypes: Set[str],
) -> Dict[str, list]:
    """Mirror of the slot-clearing inside _validate_lstv_dedicated.
    Returns the buffer for chaining (in practice mutates in place)."""
    for sub in lstv_subtypes:
        buffer[sub] = []
    return buffer


def _resolve_settings_pure(
    enabled_env: Optional[str],
    every_env: Optional[str],
) -> Tuple[bool, int]:
    """Mirror of _resolve_lstv_dedicated_val_settings; takes env values
    directly so tests don't need to monkey-patch os.environ."""
    if enabled_env is None or enabled_env.strip() == "":
        enabled = True  # default
    else:
        enabled = enabled_env.strip().lower() in ("1", "true", "yes", "on")
    try:
        every = max(1, int(every_env)) if every_env else 1
    except (TypeError, ValueError):
        every = 1
    return enabled, every


# ─── Constants sanity ────────────────────────────────────────────────────────

def test_oversample_subtypes_includes_all_lstv():
    expected = {_SUBTYPE_LUMB, _SUBTYPE_SACR_COUNT, _SUBTYPE_SEMI,
                _SUBTYPE_SACR, _SUBTYPE_AMBIG}
    assert set(_OVERSAMPLE_SUBTYPES) == expected, (
        f"expected all 5 LSTV subtypes in oversample pool, "
        f"got {set(_OVERSAMPLE_SUBTYPES)}"
    )


def test_oversample_excludes_normal():
    assert _SUBTYPE_NORMAL not in _OVERSAMPLE_SUBTYPES


def test_known_subtypes_is_six():
    assert len(_KNOWN_SUBTYPES) == 6


# ─── _filter_lstv_val_keys ──────────────────────────────────────────────────

def test_filter_only_lstv_keys_returned():
    val_keys = ["A", "B", "C", "D", "E"]
    case_to_subtype = {
        "A": _SUBTYPE_NORMAL,
        "B": _SUBTYPE_LUMB,
        "C": _SUBTYPE_NORMAL,
        "D": _SUBTYPE_SACR,
        "E": _SUBTYPE_AMBIG,
    }
    out = _filter_lstv_val_keys(val_keys, case_to_subtype, set(_OVERSAMPLE_SUBTYPES))
    assert set(out) == {"B", "D", "E"}


def test_filter_preserves_order():
    val_keys = ["x", "y", "z", "w"]
    case_to_subtype = {
        "x": _SUBTYPE_NORMAL,
        "y": _SUBTYPE_LUMB,
        "z": _SUBTYPE_SACR_COUNT,
        "w": _SUBTYPE_NORMAL,
    }
    out = _filter_lstv_val_keys(val_keys, case_to_subtype, set(_OVERSAMPLE_SUBTYPES))
    assert out == ["y", "z"]


def test_filter_empty_when_no_lstv_in_val():
    """If a fold's val set has no LSTV cases, the filter returns
    empty and the dedicated pass should no-op."""
    val_keys = ["A", "B", "C"]
    case_to_subtype = {k: _SUBTYPE_NORMAL for k in val_keys}
    out = _filter_lstv_val_keys(val_keys, case_to_subtype, set(_OVERSAMPLE_SUBTYPES))
    assert out == []


def test_filter_unknown_keys_treated_as_normal():
    """Keys not in the subtype map default to 'normal' → filtered out."""
    val_keys = ["unknown1", "lumb_case", "unknown2"]
    case_to_subtype = {"lumb_case": _SUBTYPE_LUMB}
    out = _filter_lstv_val_keys(val_keys, case_to_subtype, set(_OVERSAMPLE_SUBTYPES))
    assert out == ["lumb_case"]


def test_filter_handles_all_5_lstv_subtypes():
    """Each LSTV subtype should appear in the filtered output."""
    case_to_subtype = {
        "lumb_case":   _SUBTYPE_LUMB,
        "sc_case":     _SUBTYPE_SACR_COUNT,
        "semi_case":   _SUBTYPE_SEMI,
        "sacr_case":   _SUBTYPE_SACR,
        "ambig_case":  _SUBTYPE_AMBIG,
        "normal_case": _SUBTYPE_NORMAL,
    }
    val_keys = list(case_to_subtype.keys())
    out = _filter_lstv_val_keys(val_keys, case_to_subtype, set(_OVERSAMPLE_SUBTYPES))
    assert "normal_case" not in out
    assert len(out) == 5
    assert set(out) == set(case_to_subtype.keys()) - {"normal_case"}


# ─── _clear_lstv_slots ──────────────────────────────────────────────────────

def test_clear_replaces_only_lstv_slots():
    buffer = {sub: [f"old_{sub}"] for sub in _KNOWN_SUBTYPES}
    _clear_lstv_slots(buffer, set(_OVERSAMPLE_SUBTYPES))
    # All LSTV slots cleared
    for sub in _OVERSAMPLE_SUBTYPES:
        assert buffer[sub] == [], f"{sub} should be empty after clear"
    # Normal slot preserved (regular val sampler's contribution)
    assert buffer[_SUBTYPE_NORMAL] == [f"old_{_SUBTYPE_NORMAL}"]


def test_clear_idempotent():
    buffer = {sub: [] for sub in _KNOWN_SUBTYPES}
    _clear_lstv_slots(buffer, set(_OVERSAMPLE_SUBTYPES))
    _clear_lstv_slots(buffer, set(_OVERSAMPLE_SUBTYPES))
    for sub in _OVERSAMPLE_SUBTYPES:
        assert buffer[sub] == []


def test_clear_does_not_remove_keys():
    """Clearing a slot empties it but doesn't delete the key — the
    aggregator iterates over fixed keys and a missing slot would
    crash."""
    buffer = {sub: [["fake_dice"]] for sub in _KNOWN_SUBTYPES}
    _clear_lstv_slots(buffer, set(_OVERSAMPLE_SUBTYPES))
    for sub in _KNOWN_SUBTYPES:
        assert sub in buffer, f"slot {sub} should still exist as a key"


# ─── Settings resolution ────────────────────────────────────────────────────

def test_default_settings_enabled_every_1():
    enabled, every = _resolve_settings_pure(None, None)
    assert enabled is True
    assert every == 1


def test_disable_via_env():
    for v in ("0", "false", "no", "off"):
        enabled, _ = _resolve_settings_pure(v, None)
        assert enabled is False, f"value {v!r} should disable"


def test_explicit_enable():
    for v in ("1", "true", "yes", "on", "TRUE", "Yes"):
        enabled, _ = _resolve_settings_pure(v, None)
        assert enabled is True, f"value {v!r} should enable"


def test_every_n_epochs():
    _, every = _resolve_settings_pure(None, "5")
    assert every == 5


def test_every_clamped_to_minimum_1():
    _, every = _resolve_settings_pure(None, "0")
    assert every == 1, "every<=0 should be clamped to 1 to avoid div by zero"
    _, every = _resolve_settings_pure(None, "-3")
    assert every == 1


def test_every_invalid_falls_back_to_1():
    _, every = _resolve_settings_pure(None, "abc")
    assert every == 1
    _, every = _resolve_settings_pure(None, "")
    assert every == 1


# ─── Epoch-skip logic (mod) ─────────────────────────────────────────────────

def test_every_1_runs_every_epoch():
    every = 1
    for epoch in range(20):
        assert (epoch % every) == 0, "every=1 should fire on every epoch"


def test_every_5_runs_at_correct_epochs():
    every = 5
    fires = [e for e in range(50) if (e % every) == 0]
    assert fires == [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]


# ─── Trainer source contains the expected hook structure ────────────────────

def test_trainer_has_dedicated_val_hook_in_on_epoch_end():
    """Confirm _validate_lstv_dedicated is called from on_epoch_end
    BEFORE _aggregate_subgroup_dice. Order matters: dedicated pass
    must populate buffer before aggregation reads it."""
    src = (_TOOLS_DIR / "nnunet_wandb_variant.py").read_text()
    # Find on_epoch_end body
    tree = ast.parse(src)
    on_epoch_end_body_src = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "on_epoch_end":
            try:
                on_epoch_end_body_src = ast.get_source_segment(src, node)
            except AttributeError:
                # 3.7 fallback
                lines = src.splitlines(keepends=True)
                end = getattr(node, "end_lineno", None) or len(lines)
                on_epoch_end_body_src = "".join(lines[node.lineno - 1:end])
            break
    assert on_epoch_end_body_src is not None, "on_epoch_end not found"

    # Both calls present — match `self.METHOD(` to skip docstring/comment
    # mentions of the method name.
    call_validate = "self._validate_lstv_dedicated("
    call_aggregate = "self._aggregate_subgroup_dice("
    assert call_validate in on_epoch_end_body_src, (
        "on_epoch_end must call self._validate_lstv_dedicated()"
    )
    assert call_aggregate in on_epoch_end_body_src, (
        "on_epoch_end must call self._aggregate_subgroup_dice()"
    )
    # Dedicated must come BEFORE aggregation in the body
    pos_validate = on_epoch_end_body_src.index(call_validate)
    pos_aggregate = on_epoch_end_body_src.index(call_aggregate)
    assert pos_validate < pos_aggregate, (
        "self._validate_lstv_dedicated() must run BEFORE "
        "self._aggregate_subgroup_dice() so the buffer reflects "
        "dedicated-pass results when aggregation reads it"
    )


def test_trainer_dedicated_val_method_signature():
    """Confirm _validate_lstv_dedicated takes self only (no required
    args) and is callable from on_epoch_end without extra plumbing."""
    src = (_TOOLS_DIR / "nnunet_wandb_variant.py").read_text()
    tree = ast.parse(src)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_validate_lstv_dedicated":
            args = node.args.args
            assert len(args) == 1, (
                f"_validate_lstv_dedicated should take only self, got "
                f"{[a.arg for a in args]}"
            )
            assert args[0].arg == "self"
            found = True
            break
    assert found, "_validate_lstv_dedicated method not found in trainer"


def test_trainer_clears_lstv_slots_before_recording():
    """Verify the implementation clears LSTV slots in the buffer
    before the forward-pass loop. Otherwise the dedicated pass would
    APPEND to whatever the regular sampler caught, producing
    duplicate per-case dice for cases the regular sampler hit."""
    src = (_TOOLS_DIR / "nnunet_wandb_variant.py").read_text()
    tree = ast.parse(src)
    method_src = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_validate_lstv_dedicated":
            try:
                method_src = ast.get_source_segment(src, node)
            except AttributeError:
                lines = src.splitlines(keepends=True)
                end = getattr(node, "end_lineno", None) or len(lines)
                method_src = "".join(lines[node.lineno - 1:end])
            break
    assert method_src is not None
    # Look for the slot-clearing pattern: `self._val_subgroup_dice[sub] = []`
    # within an _OVERSAMPLE_SUBTYPES loop.
    assert "_OVERSAMPLE_SUBTYPES" in method_src, (
        "_validate_lstv_dedicated should iterate _OVERSAMPLE_SUBTYPES "
        "to clear LSTV slots"
    )
    assert "self._val_subgroup_dice[sub] = []" in method_src, (
        "_validate_lstv_dedicated should clear LSTV slots before "
        "the forward-pass loop"
    )


def test_trainer_restores_keys_in_finally():
    """The dedicated pass mutates loader._data.identifiers temporarily.
    A try/finally must restore the full keys list even if a forward
    pass raises — otherwise the regular val sampler would only see
    the last single-case keys list on the next epoch."""
    src = (_TOOLS_DIR / "nnunet_wandb_variant.py").read_text()
    tree = ast.parse(src)
    method_src = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_validate_lstv_dedicated":
            try:
                method_src = ast.get_source_segment(src, node)
            except AttributeError:
                lines = src.splitlines(keepends=True)
                end = getattr(node, "end_lineno", None) or len(lines)
                method_src = "".join(lines[node.lineno - 1:end])
            break
    assert method_src is not None
    assert "finally:" in method_src, (
        "_validate_lstv_dedicated must use try/finally to restore the "
        "loader's keys list even on exception"
    )
    assert "setattr(keys_owner, keys_attr, full_keys)" in method_src, (
        "_validate_lstv_dedicated must restore the original keys list "
        "via setattr(keys_owner, keys_attr, full_keys) in finally"
    )


# ─── End-to-end logic test (deterministic, no nnU-Net) ──────────────────────

def test_v172_find_underlying_returns_4tuple():
    """v17.2 changed _find_underlying_val_loader to return a 4-tuple
    (underlying, keys_owner, keys_attr, is_callable) instead of a
    3-tuple. The is_callable flag tells the consumer whether to
    invoke the attribute or use it directly."""
    src = (_TOOLS_DIR / "nnunet_wandb_variant.py").read_text()
    tree = ast.parse(src)
    method_src = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_find_underlying_val_loader":
            try:
                method_src = ast.get_source_segment(src, node)
            except AttributeError:
                lines = src.splitlines(keepends=True)
                end = getattr(node, "end_lineno", None) or len(lines)
                method_src = "".join(lines[node.lineno - 1:end])
            break
    assert method_src is not None, "_find_underlying_val_loader not found"
    # Returns must include is_callable boolean
    assert "is_callable" in method_src, (
        "_find_underlying_val_loader must return is_callable flag (v17.2)"
    )
    # All return statements should produce 4-tuples (or use a 4-element
    # tuple-typed expression).
    assert "return None, None, None, False" in method_src, (
        "Empty-result return should be 4-tuple (None, None, None, False)"
    )


def test_v172_consumer_uses_is_callable_flag():
    """The consumer of _find_underlying_val_loader inside
    _validate_lstv_dedicated must unpack the 4-tuple and respect
    the is_callable flag — calling the attribute if it's a method,
    using it directly otherwise."""
    src = (_TOOLS_DIR / "nnunet_wandb_variant.py").read_text()
    tree = ast.parse(src)
    method_src = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_validate_lstv_dedicated":
            try:
                method_src = ast.get_source_segment(src, node)
            except AttributeError:
                lines = src.splitlines(keepends=True)
                end = getattr(node, "end_lineno", None) or len(lines)
                method_src = "".join(lines[node.lineno - 1:end])
            break
    assert method_src is not None
    # Must unpack 4 elements including is_callable
    assert "is_callable" in method_src, (
        "_validate_lstv_dedicated must unpack is_callable from "
        "_find_underlying_val_loader (v17.2)"
    )
    # Must call raw() when is_callable is True
    assert "raw()" in method_src, (
        "_validate_lstv_dedicated must call the attribute when "
        "is_callable is True"
    )


def test_v172_skips_setattr_for_method_attrs():
    """When is_callable=True, attempting setattr(keys_owner, 'keys',
    [single_case]) would replace the method with a list — breaking
    subsequent callers. v17.2 must detect this case and skip the
    forced single-case override (passing through to the regular
    val sampler instead)."""
    src = (_TOOLS_DIR / "nnunet_wandb_variant.py").read_text()
    tree = ast.parse(src)
    method_src = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_validate_lstv_dedicated":
            try:
                method_src = ast.get_source_segment(src, node)
            except AttributeError:
                lines = src.splitlines(keepends=True)
                end = getattr(node, "end_lineno", None) or len(lines)
                method_src = "".join(lines[node.lineno - 1:end])
            break
    assert method_src is not None
    # There should be an early-return when is_callable is True, BEFORE
    # the for-loop that does setattr.
    setattr_pos = method_src.find("setattr(keys_owner, keys_attr, [cid])")
    assert setattr_pos > 0, (
        "expected setattr(keys_owner, keys_attr, [cid]) in dedicated "
        "pass body"
    )
    # The early-return for callable case must come before the setattr
    if_callable_pos = method_src.find("if is_callable:")
    assert 0 < if_callable_pos < setattr_pos, (
        "v17.2 must check `if is_callable:` and return early BEFORE "
        "attempting setattr(keys_owner, keys_attr, [cid]) which would "
        "replace a method with a list"
    )


# ─── Module-level bias-hook state tests (v17.2) ──────────────────────────────

def test_v172_module_level_bias_state_present():
    """v17.2 introduces module-level state for the bias hook that
    survives worker fork. Verify the module exports the expected
    names."""
    src = (_TOOLS_DIR / "nnunet_wandb_variant.py").read_text()
    expected_names = [
        "_BIAS_RULES_FOR_WORKERS",
        "_BIAS_INSTALLED",
        "_BIAS_CALL_COUNT",
        "_BIAS_L6_RETURNS",
        "_BIAS_SACRUM_RETURNS",
        "_BIAS_ORIGINAL_NP_CHOICE",
    ]
    for name in expected_names:
        assert f"\n{name}" in src or src.startswith(name), (
            f"v17.2 module-level state missing: {name}"
        )


def test_v172_module_level_helpers_present():
    """The module-level helpers must exist and be top-level
    functions (not class methods), so workers can call them after
    fork without needing a trainer instance."""
    src = (_TOOLS_DIR / "nnunet_wandb_variant.py").read_text()
    tree = ast.parse(src)
    top_level_funcs = {
        node.name for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    expected_helpers = [
        "_global_biased_np_choice",
        "_install_global_bias_hook",
        "_ensure_bias_counters",
        "_read_bias_counters",
        "_reset_bias_counters_for_new_fold",
    ]
    for name in expected_helpers:
        assert name in top_level_funcs, (
            f"v17.2 module-level helper missing or not top-level: {name}. "
            f"Found top-level: {sorted(top_level_funcs)[:10]}..."
        )


def test_v172_install_uses_module_state_not_closure():
    """v17 wrapped generate_train_batch with a closure over
    trainer-instance state. v17.2 must replace that with the
    module-level _install_global_bias_hook call."""
    src = (_TOOLS_DIR / "nnunet_wandb_variant.py").read_text()
    tree = ast.parse(src)
    install_method_src = None
    # The method is named _maybe_apply_l6_patch_bias (which calls
    # _install_global_bias_hook internally in v17.2).
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_maybe_apply_l6_patch_bias":
            try:
                install_method_src = ast.get_source_segment(src, node)
            except AttributeError:
                lines = src.splitlines(keepends=True)
                end = getattr(node, "end_lineno", None) or len(lines)
                install_method_src = "".join(lines[node.lineno - 1:end])
            break
    assert install_method_src is not None, (
        "Could not find _maybe_apply_l6_patch_bias method"
    )
    # v17.2 must call the module-level installer
    assert "_install_global_bias_hook(rules)" in install_method_src, (
        "v17.2 _maybe_apply_l6_patch_bias must delegate to "
        "_install_global_bias_hook(rules) for cross-process patch"
    )
    # The old closure-based approach must be GONE
    assert "patched_generate_train_batch" not in install_method_src, (
        "v17.2 must not define patched_generate_train_batch closure "
        "(replaced by module-level np.random.choice patch)"
    )
    assert "underlying.generate_train_batch = patched_generate_train_batch" not in install_method_src, (
        "v17.2 must not assign instance-level generate_train_batch "
        "(closure didn't survive worker fork)"
    )


def test_v172_payload_refreshes_shared_counters():
    """The W&B payload generator reads self._bias_call_count etc.,
    which used to be incremented by the closure. In v17.2 the real
    counters live in multiprocessing.Value — the payload generator
    must call _refresh_bias_counters_from_shared() before reading
    instance attrs, otherwise it'll show 0 forever."""
    src = (_TOOLS_DIR / "nnunet_wandb_variant.py").read_text()
    # Find _aggregate_subgroup_dice — it's where the payload is built
    tree = ast.parse(src)
    method_src = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_aggregate_subgroup_dice":
            try:
                method_src = ast.get_source_segment(src, node)
            except AttributeError:
                lines = src.splitlines(keepends=True)
                end = getattr(node, "end_lineno", None) or len(lines)
                method_src = "".join(lines[node.lineno - 1:end])
            break
    assert method_src is not None
    # Refresh must happen BEFORE reading self._bias_call_count
    refresh_pos = method_src.find("_refresh_bias_counters_from_shared")
    read_pos = method_src.find("self._bias_call_count")
    assert refresh_pos > 0, (
        "v17.2 _aggregate_subgroup_dice must call "
        "_refresh_bias_counters_from_shared() to read worker increments"
    )
    assert refresh_pos < read_pos, (
        "_refresh_bias_counters_from_shared must run BEFORE reading "
        "self._bias_call_count (otherwise shared->instance copy hasn't "
        "happened yet)"
    )


# ─── Final end-to-end logic test (deterministic, no nnU-Net) ────────────────

def test_end_to_end_filter_clear_dedicated_pass():
    """Simulate the full dedicated-pass logic on a mock buffer and
    val-keys list. Verify:
      1. LSTV cases are correctly identified
      2. LSTV buffer slots are cleared
      3. Normal slot is preserved
      4. After the simulated forward passes, all LSTV cases have
         exactly one set of per-case dice in their slot.
    """
    val_keys = ["normal_a", "normal_b", "lumb_1", "sacr_1", "ambig_1",
                 "normal_c", "lumb_2", "normal_d"]
    case_to_subtype = {
        "normal_a": _SUBTYPE_NORMAL, "normal_b": _SUBTYPE_NORMAL,
        "lumb_1":   _SUBTYPE_LUMB,   "sacr_1":   _SUBTYPE_SACR,
        "ambig_1":  _SUBTYPE_AMBIG,  "normal_c": _SUBTYPE_NORMAL,
        "lumb_2":   _SUBTYPE_LUMB,   "normal_d": _SUBTYPE_NORMAL,
    }

    # Initialize buffer as if regular val sampler caught some normals
    # and got lucky and caught lumb_1. The dedicated pass should
    # clear LSTV slots and re-record both lumb cases.
    buffer = {sub: [] for sub in _KNOWN_SUBTYPES}
    buffer[_SUBTYPE_NORMAL] = [["fake_dice_normal_b"]]
    buffer[_SUBTYPE_LUMB]   = [["regular_sampler_caught_lumb_1"]]

    # Step 1: filter
    lstv_in_val = _filter_lstv_val_keys(
        val_keys, case_to_subtype, set(_OVERSAMPLE_SUBTYPES))
    assert set(lstv_in_val) == {"lumb_1", "sacr_1", "ambig_1", "lumb_2"}

    # Step 2: clear LSTV slots
    _clear_lstv_slots(buffer, set(_OVERSAMPLE_SUBTYPES))
    # Normal preserved
    assert buffer[_SUBTYPE_NORMAL] == [["fake_dice_normal_b"]]
    # LSTV cleared (regular sampler's lumb_1 contribution dropped)
    assert buffer[_SUBTYPE_LUMB] == []

    # Step 3: simulate dedicated pass — append fake dice for each LSTV case
    for cid in lstv_in_val:
        sub = case_to_subtype[cid]
        buffer[sub].append([f"dedicated_pass:{cid}"])

    # Step 4: verify
    assert len(buffer[_SUBTYPE_LUMB]) == 2  # lumb_1 and lumb_2
    assert len(buffer[_SUBTYPE_SACR]) == 1
    assert len(buffer[_SUBTYPE_AMBIG]) == 1
    assert len(buffer[_SUBTYPE_SACR_COUNT]) == 0  # no sacr_count cases
    assert len(buffer[_SUBTYPE_SEMI]) == 0
    assert len(buffer[_SUBTYPE_NORMAL]) == 1  # untouched


# ─── Standalone runner ──────────────────────────────────────────────────────

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
