"""
tests/integration/test_real_nnunet_integration.py

Integration tests that exercise the v17.2 fixes against ACTUAL nnU-Net
data structures, not mocked AST extractions. These tests catch bugs
that the host-side unit tests cannot — bugs that live in the boundary
between our trainer code and nnU-Net's internals.

These tests MUST run inside the container (need nnunetv2, numpy, etc.).
The host-side run_all_tests.sh skips this file when nnunetv2 is not
importable.

Run inside container:
    singularity exec --nv $CONTAINER /opt/conda/bin/python3 -m pytest \\
        tests/integration/test_real_nnunet_integration.py -v

Or:
    singularity exec --nv $CONTAINER /opt/conda/bin/python3 \\
        tests/integration/test_real_nnunet_integration.py
"""
from __future__ import annotations

import importlib
import multiprocessing as mp
import os
import sys
from pathlib import Path

# ─── Skip cleanly if nnunetv2 not available ──────────────────────────────────
try:
    import numpy as np
    import nnunetv2  # noqa: F401
    NNUNETV2_AVAILABLE = True
except ImportError:
    NNUNETV2_AVAILABLE = False

# ─── Import the trainer module under test ────────────────────────────────────
TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if NNUNETV2_AVAILABLE:
    if str(TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(TOOLS_DIR))
    # Force a fresh import in case the module name is cached.
    if "nnunet_wandb_variant" in sys.modules:
        del sys.modules["nnunet_wandb_variant"]
    nwv = importlib.import_module("nnunet_wandb_variant")
else:
    nwv = None


# ─── Test 1: Module-level state initializes correctly ────────────────────────

def test_module_state_starts_uninstalled():
    """Before _install_global_bias_hook is called, the module-level
    flag should be False and counters should be None or all zero."""
    if not NNUNETV2_AVAILABLE:
        print("  SKIP module not loadable")
        return
    # Re-import fresh to reset state
    if "nnunet_wandb_variant" in sys.modules:
        del sys.modules["nnunet_wandb_variant"]
    fresh = importlib.import_module("nnunet_wandb_variant")
    assert fresh._BIAS_INSTALLED is False, (
        "_BIAS_INSTALLED should start False"
    )
    assert fresh._BIAS_RULES_FOR_WORKERS is None, (
        "_BIAS_RULES_FOR_WORKERS should start None"
    )
    print("  PASS module-level state starts uninstalled")


def test_install_global_bias_hook_idempotent():
    """Calling _install_global_bias_hook twice should not raise and
    should keep np.random.choice patched after the second call."""
    if not NNUNETV2_AVAILABLE:
        print("  SKIP module not loadable")
        return
    if "nnunet_wandb_variant" in sys.modules:
        del sys.modules["nnunet_wandb_variant"]
    fresh = importlib.import_module("nnunet_wandb_variant")
    rules = {
        "lumb": (6, "SPINESURG_L6_PATCH_BIAS_FRAC", 0.6),
        "sacr_count": (7, "SPINESURG_SACRUM_PATCH_BIAS_FRAC", 0.5),
    }
    original_np_choice = np.random.choice
    try:
        ok1 = fresh._install_global_bias_hook(rules)
        assert ok1 is True
        patched = np.random.choice
        assert patched is fresh._global_biased_np_choice, (
            "np.random.choice should be replaced after install"
        )
        ok2 = fresh._install_global_bias_hook(rules)  # idempotent
        assert ok2 is True
        assert np.random.choice is fresh._global_biased_np_choice, (
            "np.random.choice should remain replaced after second install"
        )
        # Original was captured for pass-through
        assert fresh._BIAS_ORIGINAL_NP_CHOICE is original_np_choice
        print("  PASS install idempotent, original captured")
    finally:
        # Restore np.random.choice for downstream tests
        np.random.choice = original_np_choice
        fresh._BIAS_INSTALLED = False
        fresh._BIAS_ORIGINAL_NP_CHOICE = None


# ─── Test 2: Bias hook actually fires when called from generate_train_batch ──

def test_bias_hook_fires_in_simulated_call_site():
    """Construct a function literally named ``generate_train_batch``,
    populate its locals with ``eligible_classes_or_regions``, call
    ``np.random.choice(len(eligible))`` — verify the patched function
    intercepts and the shared counter increments."""
    if not NNUNETV2_AVAILABLE:
        print("  SKIP module not loadable")
        return
    if "nnunet_wandb_variant" in sys.modules:
        del sys.modules["nnunet_wandb_variant"]
    fresh = importlib.import_module("nnunet_wandb_variant")
    rules = {
        "lumb": (6, "SPINESURG_L6_PATCH_BIAS_FRAC", 1.0),  # always bias
    }
    original_np_choice = np.random.choice
    try:
        fresh._install_global_bias_hook(rules)
        fresh._reset_bias_counters_for_new_fold()

        def generate_train_batch():
            eligible_classes_or_regions = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            return np.random.choice(len(eligible_classes_or_regions))

        results = [generate_train_batch() for _ in range(50)]
        calls, l6, sacr = fresh._read_bias_counters()
        assert calls == 50, f"expected 50 intercepts, got {calls}"
        # frac=1.0 + lumb rule + 6 in eligible => every call returns
        # the index of L6 (which is 5, since eligible[5]==6).
        assert l6 == 50, f"expected 50 L6 returns, got {l6}"
        assert all(r == 5 for r in results), (
            f"expected all returns to be index 5 (L6), got {results[:5]}..."
        )
        print(f"  PASS bias hook fires correctly: {calls} calls, "
              f"{l6} L6 returns")
    finally:
        np.random.choice = original_np_choice
        fresh._BIAS_INSTALLED = False
        fresh._BIAS_ORIGINAL_NP_CHOICE = None


def test_bias_hook_passes_through_unrelated_calls():
    """np.random.choice called from a function NOT named
    generate_train_batch should pass through to the original numpy
    implementation unchanged."""
    if not NNUNETV2_AVAILABLE:
        print("  SKIP module not loadable")
        return
    if "nnunet_wandb_variant" in sys.modules:
        del sys.modules["nnunet_wandb_variant"]
    fresh = importlib.import_module("nnunet_wandb_variant")
    rules = {
        "lumb": (6, "SPINESURG_L6_PATCH_BIAS_FRAC", 1.0),
    }
    original_np_choice = np.random.choice
    try:
        fresh._install_global_bias_hook(rules)
        fresh._reset_bias_counters_for_new_fold()

        # Unrelated callsite — different function name.
        def some_other_function():
            return np.random.choice(10)

        # Multi-arg form (size kwarg) — also pass-through.
        def another_callsite():
            return np.random.choice(10, size=3)

        results = [some_other_function() for _ in range(30)]
        results2 = [another_callsite() for _ in range(10)]
        calls, _, _ = fresh._read_bias_counters()
        assert calls == 0, (
            f"unrelated calls should not increment counter, got {calls}"
        )
        # Returns should be normal random ints in [0,10)
        assert all(0 <= r < 10 for r in results)
        # size=3 returns should be arrays of length 3
        assert all(hasattr(r, "shape") and r.shape == (3,) for r in results2)
        print("  PASS unrelated calls pass through correctly")
    finally:
        np.random.choice = original_np_choice
        fresh._BIAS_INSTALLED = False
        fresh._BIAS_ORIGINAL_NP_CHOICE = None


# ─── Test 3: Counters increment in worker process (the v17.2 main fix) ───────

def _worker_call_generate_train_batch(n: int):
    """Run in a child process. Imports the module fresh, then calls
    ``np.random.choice`` from a function literally named
    ``generate_train_batch`` ``n`` times. Returns the call count
    observed by the worker (for sanity).

    This simulates exactly what NonDetMultiThreadedAugmenter does:
    the parent installs the patch, then forks/spawns workers which
    must inherit it.
    """
    import importlib as _imp
    import numpy as _np
    if "nnunet_wandb_variant" in sys.modules:
        del sys.modules["nnunet_wandb_variant"]
    sys.path.insert(0, str(TOOLS_DIR))
    fresh = _imp.import_module("nnunet_wandb_variant")

    def generate_train_batch():
        eligible_classes_or_regions = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        return _np.random.choice(len(eligible_classes_or_regions))

    for _ in range(n):
        generate_train_batch()
    # Read the shared counter from this worker's perspective
    return fresh._read_bias_counters()


def test_bias_hook_survives_fork():
    """The crucial v17.2 test: install the global hook in the parent,
    fork a worker process, have the worker call np.random.choice from
    inside generate_train_batch, verify the parent's shared counter
    sees the worker's increments.

    On Linux (default fork start method), child processes inherit
    parent memory including module state and the patched
    np.random.choice. The shared multiprocessing.Value counters allow
    the worker's increments to be visible in the parent.

    If this test passes inside the container, the actual SLURM
    training run will see real bias-hook activity (calls > 0 in the
    epoch logs)."""
    if not NNUNETV2_AVAILABLE:
        print("  SKIP module not loadable")
        return
    if "nnunet_wandb_variant" in sys.modules:
        del sys.modules["nnunet_wandb_variant"]
    fresh = importlib.import_module("nnunet_wandb_variant")
    rules = {
        "lumb": (6, "SPINESURG_L6_PATCH_BIAS_FRAC", 1.0),
    }
    original_np_choice = np.random.choice
    try:
        # Install in parent BEFORE spawning worker (matches real flow)
        fresh._install_global_bias_hook(rules)
        fresh._reset_bias_counters_for_new_fold()
        parent_calls_before, _, _ = fresh._read_bias_counters()
        assert parent_calls_before == 0

        # Force fork mode so worker inherits the patch via memory copy.
        # On some Linux configs the default may be 'spawn' (Python 3.8+
        # macOS); fork is what nnU-Net's data loaders use under SLURM.
        ctx = mp.get_context("fork")
        n_calls = 30
        with ctx.Pool(processes=2) as pool:
            results = pool.map(_worker_call_generate_train_batch,
                                [n_calls, n_calls])

        # Parent reads the SHARED counter — should see worker increments
        parent_calls_after, parent_l6, _ = fresh._read_bias_counters()
        expected = 2 * n_calls
        assert parent_calls_after == expected, (
            f"parent should see {expected} calls from 2 workers × "
            f"{n_calls} calls each, got {parent_calls_after}. "
            f"Worker self-reports: {results}"
        )
        assert parent_l6 == expected, (
            f"frac=1.0 + lumb rule => all calls should produce L6 returns, "
            f"got {parent_l6}/{expected}"
        )
        print(f"  PASS bias hook survives fork: parent saw "
              f"{parent_calls_after} calls from workers ({parent_l6} L6 returns)")
    finally:
        np.random.choice = original_np_choice
        fresh._BIAS_INSTALLED = False
        fresh._BIAS_ORIGINAL_NP_CHOICE = None


# ─── Test 4: Val loader keys probe handles real nnU-Net structures ───────────

def test_val_loader_keys_with_dict_data():
    """Simulate the failure mode from the production log: a loader
    whose ``_data`` is a dict (so ``_data.keys`` is a bound method,
    not a list). The v17.1 consumer raised
    "'method' object is not iterable" — v17.2's
    ``_find_underlying_val_loader`` should detect this and return
    ``is_callable=True`` so the consumer can handle it without
    crashing."""
    if not NNUNETV2_AVAILABLE:
        print("  SKIP module not loadable")
        return
    if "nnunet_wandb_variant" in sys.modules:
        del sys.modules["nnunet_wandb_variant"]
    fresh = importlib.import_module("nnunet_wandb_variant")

    class FakeLoader:
        # Mimics nnUNetDataLoader3D's exposed surface
        def __init__(self):
            self._data = {"COLONOG_001": object(), "COLONOG_002": object()}
        def generate_train_batch(self):
            return None

    class FakeAugmenter:
        def __init__(self, gen):
            self.generator = gen

    fake_aug = FakeAugmenter(FakeLoader())

    class FakeTrainer:
        dataloader_val = fake_aug
    # Bind the v17.2 method to the fake trainer instance
    fake = FakeTrainer()
    result = fresh._WandBMixin._find_underlying_val_loader(fake)
    assert isinstance(result, tuple), f"expected tuple, got {type(result)}"
    assert len(result) == 4, f"expected 4-tuple, got {len(result)}"
    underlying, keys_owner, keys_attr, is_callable = result
    assert underlying is fake_aug.generator
    assert keys_owner is fake_aug.generator._data
    assert keys_attr == "keys"
    assert is_callable is True, (
        "dict._data.keys is a method; is_callable should be True"
    )
    # Verify the consumer pattern works
    raw = getattr(keys_owner, keys_attr)
    if is_callable:
        materialized = list(raw())
    else:
        materialized = list(raw)
    assert set(materialized) == {"COLONOG_001", "COLONOG_002"}
    print("  PASS dict-data loader probe returns is_callable=True; "
          "consumer materializes correctly")


def test_val_loader_keys_with_list_attr():
    """The other case: loader with `.indices` as a list attribute
    (which is what nnUNetDataLoader3D actually uses for sampling).
    Should return is_callable=False; keys_attr='indices'."""
    if not NNUNETV2_AVAILABLE:
        print("  SKIP module not loadable")
        return
    if "nnunet_wandb_variant" in sys.modules:
        del sys.modules["nnunet_wandb_variant"]
    fresh = importlib.import_module("nnunet_wandb_variant")

    class FakeLoader:
        def __init__(self):
            self.indices = ["COLONOG_001", "COLONOG_002", "COLONOG_003"]
            self._data = {"COLONOG_001": object()}  # also has dict
        def generate_train_batch(self):
            return None

    class FakeAugmenter:
        def __init__(self, gen):
            self.generator = gen

    class FakeTrainer:
        dataloader_val = FakeAugmenter(FakeLoader())

    fake = FakeTrainer()
    result = fresh._WandBMixin._find_underlying_val_loader(fake)
    underlying, keys_owner, keys_attr, is_callable = result
    # Should pick `indices` (loader-level, highest priority) not `keys`
    assert keys_attr == "indices", (
        f"should prefer loader.indices over _data.keys, got {keys_attr!r}"
    )
    assert is_callable is False, "list attr should be is_callable=False"
    assert keys_owner is FakeTrainer.dataloader_val.generator
    raw = getattr(keys_owner, keys_attr)
    assert list(raw) == ["COLONOG_001", "COLONOG_002", "COLONOG_003"]
    print("  PASS list-indices loader probe returns is_callable=False; "
          "loader.indices preferred over _data.keys")


def test_val_loader_no_loader():
    """Trainer without a dataloader_val attribute should return
    all-None tuple cleanly (not crash)."""
    if not NNUNETV2_AVAILABLE:
        print("  SKIP module not loadable")
        return
    if "nnunet_wandb_variant" in sys.modules:
        del sys.modules["nnunet_wandb_variant"]
    fresh = importlib.import_module("nnunet_wandb_variant")

    class FakeTrainer:
        dataloader_val = None

    fake = FakeTrainer()
    result = fresh._WandBMixin._find_underlying_val_loader(fake)
    assert result == (None, None, None, False)
    print("  PASS no-loader case returns (None, None, None, False)")


# ─── Test 5: Real nnUNetDataLoader3D structure inspection ────────────────────

def test_real_nnunet_data_loader_attributes_exist():
    """Direct check on the actual nnUNetDataLoader3D class: confirm
    it has either ``indices`` or ``list_of_keys`` so our search paths
    will find something. This protects against nnU-Net upstream
    refactoring its sampling internals."""
    if not NNUNETV2_AVAILABLE:
        print("  SKIP nnunetv2 not importable")
        return

    try:
        from nnunetv2.training.dataloading.data_loader_3d import (
            nnUNetDataLoader3D,
        )
    except ImportError:
        try:
            from nnunetv2.training.dataloading.nnunet_dataloader_3d import (
                nnUNetDataLoader3D,
            )
        except ImportError:
            print("  SKIP cannot import nnUNetDataLoader3D")
            return

    # Inspect the class — these are class-level or instance-level
    # attributes that should be present in any reasonable
    # nnUNetDataLoader3D version.
    has_indices = (
        hasattr(nnUNetDataLoader3D, "indices")
        or "indices" in nnUNetDataLoader3D.__init__.__code__.co_names
    )
    has_list_of_keys = (
        hasattr(nnUNetDataLoader3D, "list_of_keys")
        or "list_of_keys" in nnUNetDataLoader3D.__init__.__code__.co_names
    )
    has_data = (
        hasattr(nnUNetDataLoader3D, "_data")
        or "_data" in nnUNetDataLoader3D.__init__.__code__.co_names
    )
    print(f"  nnUNetDataLoader3D detected: "
          f"indices={has_indices} "
          f"list_of_keys={has_list_of_keys} "
          f"_data={has_data}")
    # At least one of the sampling-list attrs should be present.
    assert has_indices or has_list_of_keys or has_data, (
        "Cannot find any of indices/list_of_keys/_data on "
        "nnUNetDataLoader3D — search paths need updating"
    )
    print("  PASS real nnUNetDataLoader3D has at least one expected "
          "sampling attribute")


# ─── Test 6: v17.3 — Production fork timing (raw mp.Process) ─────────────────

def _worker_v173_raw_process(n_calls, result_queue):
    """Worker function for raw mp.Process timing test.

    Each worker:
      1. Reports whether np.random.choice is patched in its memory.
      2. Calls np.random.choice from a function literally named
         generate_train_batch n_calls times.
      3. Reports the number of L6 returns it observed.

    The PARENT counter (multiprocessing.Value) should see the
    increments via the shared memory backing.
    """
    import os as _os
    import sys as _sys
    if "nnunet_wandb_variant" in _sys.modules:
        m = _sys.modules["nnunet_wandb_variant"]
    else:
        m = importlib.import_module("nnunet_wandb_variant")
    info = {
        "pid": _os.getpid(),
        "ppid": _os.getppid(),
        "np_choice_is_patched": np.random.choice is m._global_biased_np_choice,
        "bias_installed_in_worker": m._BIAS_INSTALLED,
        "rules_present_in_worker": m._BIAS_RULES_FOR_WORKERS is not None,
        "l6_returns_local": 0,
    }

    def generate_train_batch():
        eligible_classes_or_regions = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        n_l6 = 0
        for _ in range(n_calls):
            idx = np.random.choice(len(eligible_classes_or_regions))
            if int(idx) == 6:
                n_l6 += 1
        return n_l6

    info["l6_returns_local"] = generate_train_batch()
    result_queue.put(info)


def test_v173_pre_fork_install_with_raw_mp_process():
    """v17.3 critical test: mirrors PRODUCTION timing exactly.

    Production architecture:
      1. Trainer calls _install_global_bias_hook(rules) in parent
      2. nnUNetTrainer.on_train_start calls get_dataloaders()
      3. get_dataloaders constructs NonDetMultiThreadedAugmenter
      4. Augmenter calls mp.Process(target=producer, ...) for each
         worker (NOT mp.Pool — raw Process)
      5. Each Process forks; child inherits parent memory image

    v17.2's test used mp.Pool which uses a different fork pathway
    (Pool spawns a manager subprocess that forks workers via os.fork
    after Pool initialization). Production uses raw Process(), which
    forks IMMEDIATELY at .start(). This test reproduces that exact
    pathway.

    If this passes, production fold-0 will see calls > 0 in epoch
    logs after the v17.3 fix is deployed.
    """
    if not NNUNETV2_AVAILABLE:
        print("  SKIP module not loadable")
        return
    if "nnunet_wandb_variant" in sys.modules:
        del sys.modules["nnunet_wandb_variant"]
    fresh = importlib.import_module("nnunet_wandb_variant")
    rules = {
        "lumb": (6, "SPINESURG_L6_PATCH_BIAS_FRAC", 1.0),
    }
    original_np_choice = np.random.choice
    try:
        # Step 1: install global patch in parent BEFORE forking
        fresh._install_global_bias_hook(rules)
        fresh._reset_bias_counters_for_new_fold()
        assert fresh._BIAS_INSTALLED
        assert fresh._BIAS_RULES_FOR_WORKERS is not None
        assert np.random.choice is fresh._global_biased_np_choice, (
            "patch did not take effect in parent"
        )

        # Step 2: spawn raw mp.Process workers (matches batchgenerators
        # NonDetMultiThreadedAugmenter exactly)
        ctx = mp.get_context("fork")
        n_calls_per_worker = 25
        n_workers = 3
        result_queue = ctx.Queue()
        procs = []
        for _ in range(n_workers):
            p = ctx.Process(
                target=_worker_v173_raw_process,
                args=(n_calls_per_worker, result_queue),
            )
            procs.append(p)
            p.start()  # ← FORKS HERE; child inherits patched np.random.choice

        # Step 3: collect worker reports
        worker_infos = []
        for _ in range(n_workers):
            worker_infos.append(result_queue.get(timeout=30))
        for p in procs:
            p.join(timeout=10)
            assert p.exitcode == 0, (
                f"worker exited with code {p.exitcode}; "
                f"check stderr for traceback"
            )

        # Step 4: every worker should have inherited the patch
        for i, info in enumerate(worker_infos):
            assert info["np_choice_is_patched"], (
                f"Worker {i} (pid={info['pid']}): "
                f"np.random.choice was NOT the patched function. "
                f"Pre-fork install did not propagate via fork. "
                f"This is the production bug we're fixing."
            )
            assert info["bias_installed_in_worker"], (
                f"Worker {i}: _BIAS_INSTALLED was False"
            )
            assert info["rules_present_in_worker"], (
                f"Worker {i}: _BIAS_RULES_FOR_WORKERS was None"
            )

        # Step 5: parent's shared counter should see all increments
        parent_calls, parent_l6, _ = fresh._read_bias_counters()
        expected = n_workers * n_calls_per_worker
        assert parent_calls == expected, (
            f"parent counter should be {expected} ({n_workers} workers × "
            f"{n_calls_per_worker} calls each), got {parent_calls}. "
            f"Worker local L6 counts: {[i['l6_returns_local'] for i in worker_infos]}"
        )
        assert parent_l6 == expected, (
            f"frac=1.0 should drive all calls to L6, got {parent_l6}/{expected}"
        )
        local_l6_total = sum(info["l6_returns_local"] for info in worker_infos)
        assert local_l6_total == expected, (
            f"workers self-reported {local_l6_total} L6 returns locally, "
            f"expected {expected}. The patch ran but did not return L6."
        )

        print(f"  PASS v17.3 pre-fork install with raw mp.Process: "
              f"{n_workers} workers forked, all saw patched np.random.choice, "
              f"parent counter = {parent_calls} ({parent_l6} L6 returns)")
    finally:
        np.random.choice = original_np_choice
        fresh._BIAS_INSTALLED = False
        fresh._BIAS_ORIGINAL_NP_CHOICE = None


# ─── Test 7: v17.3 — POST-fork install fails (negative control) ─────────────

def _worker_v173_post_fork_check(n_calls, result_queue):
    """Worker that records whether the patch reached it. Used by the
    negative-control test to prove that POST-fork install does NOT
    work — establishing that pre-fork install is the only way."""
    import os as _os, sys as _sys
    if "nnunet_wandb_variant" in _sys.modules:
        m = _sys.modules["nnunet_wandb_variant"]
    else:
        m = importlib.import_module("nnunet_wandb_variant")

    def generate_train_batch():
        eligible = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        n_l6 = 0
        for _ in range(n_calls):
            idx = np.random.choice(len(eligible))
            if int(idx) == 6:
                n_l6 += 1
        return n_l6

    n_l6_local = generate_train_batch()
    result_queue.put({
        "pid": _os.getpid(),
        "np_choice_is_patched": np.random.choice is m._global_biased_np_choice,
        "l6_local": n_l6_local,
    })


def test_v173_post_fork_install_FAILS_as_expected():
    """Negative control: prove that the v17.2 timing (install AFTER
    fork) does NOT propagate to workers. This test asserts the BUG —
    it should pass IFF the bug is reproducible in this environment.

    If this test fails (i.e., post-fork install somehow works), then
    our diagnosis is wrong and v17.3 is solving the wrong problem.

    Test sequence:
      1. Spawn workers FIRST (before any install)
      2. Workers immediately call np.random.choice
      3. Workers report np_choice_is_patched = False (the bug)

    This mirrors v17.2 production: trainer __init__ → super().on_train_start
    (forks workers) → _install_global_bias_hook (too late).
    """
    if not NNUNETV2_AVAILABLE:
        print("  SKIP module not loadable")
        return
    if "nnunet_wandb_variant" in sys.modules:
        del sys.modules["nnunet_wandb_variant"]
    fresh = importlib.import_module("nnunet_wandb_variant")
    original_np_choice = np.random.choice

    # CRITICAL: ensure no patch is active before fork
    fresh._BIAS_INSTALLED = False
    fresh._BIAS_RULES_FOR_WORKERS = None
    if hasattr(np.random, "choice"):
        # restore the truly-original numpy function in case a previous
        # test left things patched
        try:
            import numpy.random as _nr
            np.random.choice = _nr.mtrand.RandomState().choice.__func__ if False else _nr._rand.choice
        except Exception:
            pass

    try:
        ctx = mp.get_context("fork")
        result_queue = ctx.Queue()
        procs = []
        for _ in range(2):
            p = ctx.Process(
                target=_worker_v173_post_fork_check,
                args=(20, result_queue),
            )
            procs.append(p)
            p.start()  # ← workers fork BEFORE any install

        # NOW install in parent (too late for these already-forked workers)
        rules = {"lumb": (6, "SPINESURG_L6_PATCH_BIAS_FRAC", 1.0)}
        fresh._install_global_bias_hook(rules)

        worker_infos = []
        for _ in range(2):
            worker_infos.append(result_queue.get(timeout=30))
        for p in procs:
            p.join(timeout=10)

        # The bug: workers should NOT see the patch
        for i, info in enumerate(worker_infos):
            assert not info["np_choice_is_patched"], (
                f"NEGATIVE CONTROL FAILED: Worker {i} unexpectedly sees "
                f"the patch despite post-fork install. Either fork "
                f"behavior is unusual on this system, or our diagnosis "
                f"of v17.2's bug is wrong. Investigate before deploying."
            )
        print(f"  PASS v17.3 negative control: post-fork install correctly "
              f"FAILS to reach already-forked workers (this is the v17.2 bug "
              f"we're fixing in v17.3)")
    finally:
        np.random.choice = original_np_choice
        fresh._BIAS_INSTALLED = False
        fresh._BIAS_ORIGINAL_NP_CHOICE = None


# ─── Standalone runner ───────────────────────────────────────────────────────

if __name__ == "__main__":
    if not NNUNETV2_AVAILABLE:
        print("SKIPPED: nnunetv2 not importable in this environment.")
        print("Run inside the container:")
        print("  singularity exec --nv $CONTAINER /opt/conda/bin/python3 \\")
        print("    tests/integration/test_real_nnunet_integration.py")
        sys.exit(0)

    tests = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj)]
    failed = []
    for name, fn in tests:
        try:
            fn()
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed.append(name)
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed.append(name)
    print()
    print(f"  ran {len(tests)} integration tests, {len(failed)} failed")
    if failed:
        sys.exit(1)
