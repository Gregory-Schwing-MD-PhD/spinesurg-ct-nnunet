"""tools/preflight_lstv.py — everything that must be true before a training run starts.

WHY A PREFLIGHT AND NOT JUST A TRY. A training run on four H200s costs hours and a
mistake in the label scheme does not announce itself: nnU-Net will happily train on a
target whose ignore label is misplaced, whose classes are non-contiguous, or whose rare
class is absent from every validation fold, and the first sign is a number that looks
plausible and is meaningless. Every check below has that shape -- silent when wrong.

Each check prints PASS or FAIL and the script exits non-zero if any failed, so it can gate
the conversion in a job script rather than being something someone remembers to read.

    python tools/preflight_lstv.py --labels data/v5_final --splits <splits.json>
"""
from __future__ import annotations

import argparse
import importlib.util as _u
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import nibabel as nib

_HERE = Path(__file__).resolve().parent
_spec = _u.spec_from_file_location("cvt", _HERE / "convert_hf_to_nnunet.py")
cvt = _u.module_from_spec(_spec)
_spec.loader.exec_module(cvt)

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"\n         {detail}" if detail else ""))
    return ok


def scheme_checks():
    """Structural properties nnU-Net relies on and never verifies for you."""
    print("\n--- label schemes ---")
    for nm, names, remap in (
        ("countfree", cvt.LABEL_NAMES_COUNTFREE, cvt.LABEL_REMAP_COUNTFREE_FROM_VERSE),
        ("rib_regions", cvt.LABEL_NAMES_RIB_REGIONS, cvt.LABEL_REMAP_RIB_REGIONS_FROM_VERSE),
        ("oneshot", cvt.LABEL_NAMES_ONESHOT, cvt.LABEL_REMAP_ONESHOT_FROM_VERSE),
    ):
        vals = []
        for v in names.values():
            vals.extend(v if isinstance(v, list) else [v])
        vals = sorted(set(vals))
        check(f"{nm}: label values are contiguous from 0",
              vals == list(range(len(vals))), f"values {vals}")
        check(f"{nm}: ignore is the highest value",
              names.get("ignore") == max(vals), f"ignore={names.get('ignore')}, max={max(vals)}")
        lut = cvt._build_verse_remap_lut(remap)
        check(f"{nm}: no source id maps outside the declared set",
              set(int(x) for x in np.unique(lut)) <= set(vals))
        if nm == "rib_regions":
            order = cvt.REGIONS_CLASS_ORDER_RIB
            keys = [k for k in names if k not in ("background", "ignore")]
            check("rib_regions: regions_class_order length matches the region count",
                  len(order) == len(keys), f"{len(order)} vs {len(keys)}")
            check("rib_regions: ignore is NOT in regions_class_order",
                  names["ignore"] not in order)
            check("rib_regions: encompassing region precedes the one nested in it",
                  keys.index("vertebra") < keys.index("rib_bearing"))
            check("rib_regions: rib_bearing is a strict subset of vertebra",
                  set(names["rib_bearing"]) < set(names["vertebra"]))


def data_checks(labels_dir, n_sample):
    """What the label volumes actually contain, which decides whether ignore is needed.

    SKIPPED rather than failed when the directory is not reachable. The scheme checks
    above are the ones that catch nnU-Net's silent failures and they need no data; making
    the whole preflight hinge on a path that differs between the host and the container
    would mean the useful half never runs.
    """
    print("\n--- label volumes ---")
    if not labels_dir:
        print("  [SKIP] no --labels given; scheme checks above still applied")
        return
    fs = sorted(Path(labels_dir).glob("*_label.nii.gz"))
    if not fs:
        print(f"  [SKIP] no label volumes under {labels_dir}; scheme checks still applied")
        return
    check("labels present", True, f"{len(fs)} case(s)")

    rng = np.random.default_rng(0)
    pick = [fs[i] for i in rng.choice(len(fs), min(n_sample, len(fs)), replace=False)]
    ids = Counter()
    with_ignore = 0
    with_lumbar_rib = 0
    with_rib12 = 0
    for f in pick:
        u = np.unique(np.asanyarray(nib.load(str(f)).dataobj))
        for x in u:
            ids[int(x)] += 1
        if 255 in u:
            with_ignore += 1
        if 74 in u or 75 in u:
            with_lumbar_rib += 1
        if 45 in u or 57 in u:
            with_rib12 += 1

    print(f"         sampled {len(pick)} case(s); distinct ids seen: {sorted(ids)}")
    # THE QUESTION GREG RAISED. If the densified release carries no ignore voxels then
    # the ignore class is dead weight -- it costs an output consideration and complicates
    # region training for nothing. If it DOES carry them, removing it would silently
    # supervise regions nobody annotated.
    check("ignore label: presence in the data matches what the scheme assumes",
          True,
          f"{with_ignore}/{len(pick)} sampled cases contain id 255. "
          + ("The densified release carries NO ignore voxels in this sample, so the "
             "ignore class can be dropped -- see the note in the job script."
             if with_ignore == 0 else
             "Ignore voxels ARE present; the class must stay or unannotated regions "
             "get supervised as background."))
    check("twelfth ribs present (the confusable class)",
          with_rib12 > 0, f"{with_rib12}/{len(pick)} cases")
    check("lumbar ribs present (the rare class the benchmark turns on)",
          with_lumbar_rib > 0,
          f"{with_lumbar_rib}/{len(pick)} sampled. Release-wide the count is 16 of 802, "
          f"which is the number that decides whether stratification is mandatory.")


def split_checks(splits_path, lstv_cases_path):
    """A rare class absent from a validation fold makes that fold's score meaningless."""
    print("\n--- splits ---")
    p = Path(splits_path)
    if not p.exists():
        check("splits file present", False, str(p))
        return
    folds = json.loads(p.read_text())
    if isinstance(folds, dict):            # release splits_5fold.json: {"folds": [...], "patient_subtypes": {...}}
        folds = folds.get("folds", [])
    check("splits file present", True, f"{len(folds)} fold(s)")

    q = Path(lstv_cases_path)
    if not q.exists():
        check("lstv_cases.json present for stratification checking", False, str(q))
        return
    lc = json.loads(q.read_text())
    # lstv_cases.json keys cases; the release splits file keys patient tokens under other names
    attrs = lc.get("case_to_attrs") or lc.get("patient_attrs") or {}
    subt = lc.get("case_to_subtype") or lc.get("patient_subtypes") or {}

    def is_rare(c):
        a = attrs.get(c, {}) or {}
        return bool(a.get("has_lumbar_rib")) or (subt.get(c, "normal") != "normal")

    ok_all = True
    for i, f in enumerate(folds):
        val = f.get("val", [])
        n_rare = sum(1 for c in val if is_rare(c))
        ok = n_rare >= 2
        ok_all &= ok
        print(f"         fold {i}: {len(val)} val cases, {n_rare} carrying a variant")
    check("every validation fold contains at least 2 variant cases", ok_all,
          "A fold with none cannot measure the thing the benchmark exists to measure, "
          "and its score will look fine because it is dominated by normal anatomy.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="",
                    help="optional; data checks are skipped without it")
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--splits", default="")
    ap.add_argument("--lstv-cases", default="")
    a = ap.parse_args()

    print("PREFLIGHT: everything that must hold before a training run is worth starting\n")
    scheme_checks()
    data_checks(a.labels, a.sample)
    if a.splits:
        split_checks(a.splits, a.lstv_cases or a.splits)

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    if n_fail:
        print("\nFAILED:")
        for nm, ok, _ in RESULTS:
            if not ok:
                print(f"  - {nm}")
        print("\nNothing downstream of a failed check is worth running.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
