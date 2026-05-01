"""
tests/test_nnunet_call_site.py

Smoke test: confirm the assumptions our np.random.choice intercept
relies on actually hold in the installed nnU-Net version. This test
SHOULD be run inside the singularity container where nnU-Net is
available.

If this test fails after an nnU-Net upgrade, the patch-bias hook may
silently stop working — re-read base_data_loader.py and update the
intercept accordingly.

Run inside container:
    /opt/conda/bin/python3 tests/test_nnunet_call_site.py
"""
from __future__ import annotations

import inspect
import sys


def test_base_data_loader_uses_np_random_choice():
    """Verify the call site we're patching still exists."""
    try:
        from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoader
    except ImportError:
        print("  SKIP  nnU-Net not installed in this Python env.")
        print("        Run this test inside the singularity container:")
        print("        singularity exec --nv $CONTAINER /opt/conda/bin/python3 tests/test_nnunet_call_site.py")
        return

    src = inspect.getsource(nnUNetDataLoader)
    assert "eligible_classes_or_regions" in src, (
        "base_data_loader.py no longer mentions 'eligible_classes_or_regions'. "
        "The intercept relies on this local variable name; update intercept code.")
    assert "np.random.choice" in src, (
        "base_data_loader.py no longer uses np.random.choice. "
        "The intercept patches the wrong target; check current call.")

    # Verify the specific pattern: np.random.choice(len(eligible_classes_or_regions))
    found = False
    for line in src.splitlines():
        if "np.random.choice" in line and "eligible_classes_or_regions" in line:
            found = True
            break
    assert found, (
        "Expected pattern 'np.random.choice(... eligible_classes_or_regions ...)' "
        "not found in nnUNetDataLoader source. Intercept may misfire.")
    print("  PASS  base_data_loader.py uses np.random.choice with eligible_classes_or_regions")


def test_generate_train_batch_method_exists():
    """The intercept hooks generate_train_batch — verify it's there."""
    try:
        from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
    except ImportError:
        try:
            from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D as nnUNetDataLoader3D
        except ImportError:
            print("  SKIP  nnU-Net not installed.")
            return

    assert hasattr(nnUNetDataLoader3D, "generate_train_batch") or \
           any(hasattr(b, "generate_train_batch")
                for b in nnUNetDataLoader3D.__mro__), (
        "Could not find generate_train_batch on nnUNetDataLoader3D or its bases. "
        "The intercept's installation path must be updated.")
    print("  PASS  generate_train_batch method present")


def test_class_locations_includes_per_class_keys():
    """Confirm per-class int keys exist in class_locations (i.e., the
    sampler is in single-class mode, not pure region mode). This is
    what lets the bias hook succeed.

    Reads one preprocessed pkl file. Uses the same path-resolution
    fallbacks as the trainer."""
    import os, pickle, json
    from pathlib import Path

    candidates = []
    pre = os.environ.get("nnUNet_preprocessed", "")
    if pre:
        for ds in os.listdir(pre):
            if ds.startswith("Dataset"):
                candidates.append(Path(pre) / ds)
    candidates.extend([
        Path("/wsu/home/go/go24/go2432/spinesurg-ct-nnunet/nnunet/preprocessed/Dataset802_SpineSurgCTFull"),
    ])

    prep_ds = None
    for c in candidates:
        if c.exists() and (c / "lstv_cases.json").exists():
            prep_ds = c
            break
    if prep_ds is None:
        print("  SKIP  no preprocessed dataset found; can't check class_locations.")
        return

    # Find a plans subdir with .pkl files
    plans_subdirs = [d for d in prep_ds.iterdir()
                      if d.is_dir() and any(d.glob("*.pkl"))]
    if not plans_subdirs:
        print(f"  SKIP  no plans subdir with .pkl files under {prep_ds}")
        return

    pkl_files = list(plans_subdirs[0].glob("*.pkl"))[:5]
    if not pkl_files:
        print("  SKIP  no .pkl files in plans subdir")
        return

    has_int_keys = 0
    for pkl in pkl_files:
        with open(pkl, "rb") as f:
            props = pickle.load(f)
        cl = props.get("class_locations", {})
        n_int = sum(1 for k in cl.keys()
                     if isinstance(k, int) or (hasattr(k, "ndim") and k.ndim == 0))
        if n_int > 0:
            has_int_keys += 1

    assert has_int_keys > 0, (
        "No int keys found in class_locations across sampled cases. "
        "Sampler is in pure region mode; bias hook needs different logic.")
    print(f"  PASS  {has_int_keys}/{len(pkl_files)} sampled cases have int keys "
          f"in class_locations (single-class sampling mode)")


if __name__ == "__main__":
    tests = [
        ("test_base_data_loader_uses_np_random_choice",
          test_base_data_loader_uses_np_random_choice),
        ("test_generate_train_batch_method_exists",
          test_generate_train_batch_method_exists),
        ("test_class_locations_includes_per_class_keys",
          test_class_locations_includes_per_class_keys),
    ]
    failed = []
    for name, fn in tests:
        try:
            fn()
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
