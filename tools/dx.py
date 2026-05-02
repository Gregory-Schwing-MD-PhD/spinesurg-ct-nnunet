#!/usr/bin/env python3
"""
diagnose_missing_lumb_cases.py

Answers the question: "Why are 3 of my 14 lumb cases missing from
preprocessed/?"

Hypothesis: they're in the test set (manifest_test.json -> raw/imagesTs/),
not the train set, so nnUNetv2_plan_and_preprocess never touches them.
This script verifies that hypothesis by reading case_to_attrs["split"]
from lstv_cases.json (which is recorded for every case during
convert_hf_to_nnunet.py's build) and cross-referencing the actual
files in raw/imagesTr, raw/imagesTs, raw/labelsTr, raw/labelsTs.

Output is a per-lumb-case table showing:
  - split (train / validation / test)
  - config (fused / spine_only / pelvic_native)
  - file presence in each raw/ subdirectory
  - whether the case has a preprocessed .npz

If --rescue-test-lumb is passed, the script will move test-set lumb
cases from raw/imagesTs|labelsTs to raw/imagesTr|labelsTr so they can
be picked up by the next preprocessing run. (Doesn't actually re-run
preprocessing — that's a separate command. Just stages the files.)

Usage:
    # diagnose only (default)
    python diagnose_missing_lumb_cases.py

    # diagnose + rescue test-set lumb cases into train
    python diagnose_missing_lumb_cases.py --rescue-test-lumb

Author: Claude (for Greg Schwing), May 2026
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

DEFAULT_PREP = Path(
    "/wsu/home/go/go24/go2432/spinesurg-ct-nnunet/nnunet/preprocessed/"
    "Dataset802_SpineSurgCTFull"
)
DEFAULT_RAW = Path(
    "/wsu/home/go/go24/go2432/spinesurg-ct-nnunet/nnunet/raw/"
    "Dataset802_SpineSurgCTFull"
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prep-dir", type=Path, default=DEFAULT_PREP)
    p.add_argument("--raw-dir",  type=Path, default=DEFAULT_RAW)
    p.add_argument("--plans",    default="nnUNetPlans_3d_fullres")
    p.add_argument("--subtype", default="lumb",
                    help="which subtype to diagnose (default: lumb). "
                         "Use 'all_lstv' to check all non-normal subtypes.")
    p.add_argument("--rescue-test-lumb", action="store_true",
                    help="MOVE lumb cases from imagesTs/labelsTs to "
                         "imagesTr/labelsTr (and update lstv_cases.json's "
                         "split attribute). Files are MOVED not copied. "
                         "Make sure you have a backup or version control.")
    return p.parse_args()


def diagnose(args):
    cases_json_path = args.prep_dir / "lstv_cases.json"
    if not cases_json_path.exists():
        print(f"ERROR: {cases_json_path} not found", file=sys.stderr)
        return 1

    data = json.loads(cases_json_path.read_text())
    cts   = data.get("case_to_subtype", {})
    attrs = data.get("case_to_attrs", {})

    if args.subtype == "all_lstv":
        wanted = {cid: st for cid, st in cts.items() if st != "normal"}
    else:
        wanted = {cid: st for cid, st in cts.items() if st == args.subtype}

    if not wanted:
        print(f"No cases of subtype '{args.subtype}' in lstv_cases.json")
        return 0

    plans_dir = args.prep_dir / args.plans
    raw_imagesTr = args.raw_dir / "imagesTr"
    raw_labelsTr = args.raw_dir / "labelsTr"
    raw_imagesTs = args.raw_dir / "imagesTs"
    raw_labelsTs = args.raw_dir / "labelsTs"

    print(f"Diagnostic: subtype = {args.subtype!r}, n = {len(wanted)}")
    print(f"prep dir : {args.prep_dir}")
    print(f"raw dir  : {args.raw_dir}")
    print()

    # Header
    cols = ("case_id",         "split",   "config",        "ImgTr",
            "LblTr",           "ImgTs",   "LblTs",         "preprocessed")
    print(f"{cols[0]:<22s} {cols[1]:<10s} {cols[2]:<14s} "
           f"{cols[3]:>5s} {cols[4]:>5s} {cols[5]:>5s} {cols[6]:>5s} "
           f"{cols[7]:>13s}")
    print("─" * 100)

    by_split = {"train": 0, "validation": 0, "test": 0, "unknown": 0}
    by_split_preprocessed = {"train": 0, "validation": 0, "test": 0, "unknown": 0}
    rescue_targets = []  # cases we'd move from test -> train

    for cid in sorted(wanted):
        a = attrs.get(cid, {})
        split = a.get("split", "unknown")
        config = a.get("config", "")[:13]

        img_tr_exists = (raw_imagesTr / f"{cid}_0000.nii.gz").exists()
        lbl_tr_exists = (raw_labelsTr / f"{cid}.nii.gz").exists()
        img_ts_exists = (raw_imagesTs / f"{cid}_0000.nii.gz").exists()
        lbl_ts_exists = (raw_labelsTs / f"{cid}.nii.gz").exists()
        npz_exists    = (plans_dir    / f"{cid}.npz").exists()

        def m(b): return "✓" if b else " "
        npz_str = "✓ has .npz" if npz_exists else "✗ MISSING"

        print(f"{cid:<22s} {split:<10s} {config:<14s} "
               f"{m(img_tr_exists):>5s} {m(lbl_tr_exists):>5s} "
               f"{m(img_ts_exists):>5s} {m(lbl_ts_exists):>5s} "
               f"{npz_str:>13s}")

        bucket = split if split in by_split else "unknown"
        by_split[bucket] += 1
        if npz_exists:
            by_split_preprocessed[bucket] += 1

        if (split == "test" and not npz_exists
                and img_ts_exists and lbl_ts_exists):
            rescue_targets.append(cid)

    print()
    print("─" * 60)
    print(f"  Counts by split (subtype={args.subtype!r}):")
    print(f"    {'split':<12s} {'total':>8s} {'preprocessed':>15s}")
    for sp in ("train", "validation", "test", "unknown"):
        n = by_split[sp]
        n_pp = by_split_preprocessed[sp]
        if n == 0 and sp == "unknown": continue
        print(f"    {sp:<12s} {n:>8d} {n_pp:>15d}")

    print()
    if rescue_targets:
        print(f"  {len(rescue_targets)} case(s) eligible for rescue "
               f"(in test set, files exist in imagesTs/labelsTs, no .npz):")
        for cid in rescue_targets:
            print(f"    {cid}")
        if args.rescue_test_lumb:
            return rescue(args, rescue_targets)
        else:
            print()
            print("  -> Re-run with --rescue-test-lumb to move these to "
                   "imagesTr/labelsTr.")
            print("  -> Then run nnUNetv2_plan_and_preprocess to generate "
                   ".npz files.")
    else:
        if args.rescue_test_lumb:
            print("  No rescue targets (no test cases of this subtype have "
                   "files in imagesTs/labelsTs without preprocessed .npz).")
        else:
            cases_missing_no_files = [
                cid for cid in wanted
                if not (plans_dir / f"{cid}.npz").exists()
                and not (raw_imagesTs / f"{cid}_0000.nii.gz").exists()
                and not (raw_imagesTr / f"{cid}_0000.nii.gz").exists()
            ]
            if cases_missing_no_files:
                print(f"  {len(cases_missing_no_files)} case(s) missing from "
                       f"BOTH preprocessed AND raw — these need investigation:")
                for cid in cases_missing_no_files:
                    a = attrs.get(cid, {})
                    print(f"    {cid:<22s} split={a.get('split','?')} "
                           f"config={a.get('config','?')} "
                           f"token={a.get('token', cid.replace('COLONOG_', '').rstrip('_r0123456789'))}")
    return 0


def rescue(args, rescue_targets):
    """Move test-set lumb cases from imagesTs/labelsTs -> imagesTr/labelsTr.

    Updates lstv_cases.json case_to_attrs[cid]["split"] -> "train" so
    the next conversion reflects the change. Does NOT re-run
    nnUNetv2_plan_and_preprocess — caller does that.
    """
    raw_imagesTr = args.raw_dir / "imagesTr"
    raw_labelsTr = args.raw_dir / "labelsTr"
    raw_imagesTs = args.raw_dir / "imagesTs"
    raw_labelsTs = args.raw_dir / "labelsTs"
    cases_json_path = args.prep_dir / "lstv_cases.json"

    print()
    print("─" * 60)
    print(f"RESCUE MODE: moving {len(rescue_targets)} cases test -> train")
    print("─" * 60)

    moved = 0
    failed = []
    for cid in rescue_targets:
        src_img = raw_imagesTs / f"{cid}_0000.nii.gz"
        dst_img = raw_imagesTr / f"{cid}_0000.nii.gz"
        src_lbl = raw_labelsTs / f"{cid}.nii.gz"
        dst_lbl = raw_labelsTr / f"{cid}.nii.gz"

        if dst_img.exists() or dst_lbl.exists():
            failed.append((cid, "destination already exists"))
            continue
        if not src_img.exists() or not src_lbl.exists():
            failed.append((cid, "source missing"))
            continue

        try:
            # shutil.move handles symlinks correctly (preserves them)
            shutil.move(str(src_img), str(dst_img))
            shutil.move(str(src_lbl), str(dst_lbl))
            moved += 1
            print(f"  moved {cid}")
        except Exception as exc:
            failed.append((cid, str(exc)))

    print()
    if failed:
        print(f"  WARNING: {len(failed)} case(s) failed:")
        for cid, why in failed:
            print(f"    {cid}: {why}")

    # Update lstv_cases.json
    if moved > 0:
        print(f"  Updating {cases_json_path}...")
        backup = cases_json_path.with_suffix(".json.bak_rescue")
        shutil.copy2(cases_json_path, backup)
        data = json.loads(cases_json_path.read_text())
        for cid in rescue_targets:
            if cid in data.get("case_to_attrs", {}):
                data["case_to_attrs"][cid]["split"] = "train"
        # Note: numTraining / numTest counts in dataset.json are NOT
        # auto-updated. nnU-Net doesn't strictly enforce them but you
        # may want to bump dataset.json["numTraining"] by `moved`.
        cases_json_path.write_text(json.dumps(data, indent=2, default=str))
        print(f"  backed up old to {backup}")

    print()
    print(f"DONE. {moved} case(s) moved. Next steps:")
    print(f"  1. Bump numTraining in dataset.json by {moved} "
           f"(or just regenerate it).")
    print(f"  2. Re-run nnUNetv2_plan_and_preprocess -d 802 "
           f"--verify_dataset_integrity")
    print(f"     (it will pick up the newly-moved files in imagesTr).")
    print(f"  3. Regenerate splits_final.json so the new cases land in "
           f"some fold:")
    print(f"     python tools/sync_splits_to_nnunet.py   "
           f"# or whatever your splits regeneration command is")
    print(f"  4. Verify with the diagnostic step-1 / step-2 commands "
           f"from the previous conversation.")
    return 0


if __name__ == "__main__":
    sys.exit(diagnose(parse_args()))
