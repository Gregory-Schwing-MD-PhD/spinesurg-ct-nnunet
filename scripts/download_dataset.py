#!/usr/bin/env python3
"""
Download CTSpinoPelvic1K from HuggingFace Hub -> data/hf_export/.

The dataset is kept in a separate repo so it can be distributed without
cloning the pipeline code. This script downloads (or refreshes) a local
copy into the layout this pipeline expects.

Authentication
--------------
If the dataset is private, set HF_TOKEN in your environment OR authenticate
via `huggingface-cli login` first.

Usage
-----
    python scripts/download_dataset.py
    python scripts/download_dataset.py --repo_id user/MyDataset
    python scripts/download_dataset.py --dest /scratch/mydata
    python scripts/download_dataset.py --revision v1.0
    python scripts/download_dataset.py --allow_patterns 'ct/*.nii.gz' 'labels/*.nii.gz'

Expected layout after download
-----------------------------
    data/hf_export/
        ct/             <token>.nii.gz ...
        labels/         <token>.nii.gz ...
        placed_manifest.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--repo_id",
        default=os.environ.get("HF_DATASET_REPO", "gschwing/CTSpinoPelvic1K"),
        help="HuggingFace dataset repo ID (env: HF_DATASET_REPO)",
    )
    ap.add_argument(
        "--dest",
        type=Path,
        default=Path("data/hf_export"),
        help="Local destination directory (default: data/hf_export)",
    )
    ap.add_argument(
        "--revision",
        default="main",
        help="Git revision, branch, or tag (default: main)",
    )
    ap.add_argument(
        "--token",
        default=None,
        help="HF access token (env: HF_TOKEN). Needed for private datasets.",
    )
    ap.add_argument(
        "--allow_patterns",
        nargs="*",
        default=None,
        help="Only download files matching these glob patterns "
             "(e.g. 'ct/*.nii.gz' 'labels/*.nii.gz'). Default: all files.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume partial downloads (default: on).",
    )
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed.", file=sys.stderr)
        print("  pip install huggingface_hub", file=sys.stderr)
        return 1

    args.dest.mkdir(parents=True, exist_ok=True)
    token = args.token or os.environ.get("HF_TOKEN")

    print(f"Downloading {args.repo_id}@{args.revision}")
    print(f"  -> {args.dest.resolve()}")
    if args.allow_patterns:
        print(f"  patterns: {args.allow_patterns}")
    if token:
        print(f"  auth: HF token ({'*' * (len(token) - 4) + token[-4:]})")
    else:
        print("  auth: none (public dataset, or relying on cached CLI login)")

    try:
        path = snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=str(args.dest),
            revision=args.revision,
            token=token,
            allow_patterns=args.allow_patterns,
            resume_download=args.resume,
        )
    except Exception as e:
        print(f"\nERROR: download failed: {e}", file=sys.stderr)
        if "401" in str(e) or "403" in str(e):
            print("Hint: for a private dataset, set HF_TOKEN or run "
                  "`huggingface-cli login`.", file=sys.stderr)
        return 2

    print(f"\nDownload complete: {path}")

    # Sanity check expected layout
    ct_dir = args.dest / "ct"
    labels_dir = args.dest / "labels"
    manifest = args.dest / "placed_manifest.json"

    issues = []
    if not ct_dir.exists():
        issues.append(f"  missing: {ct_dir}")
    else:
        n = len(list(ct_dir.glob("*.nii.gz")))
        print(f"  ct/:     {n} .nii.gz files")

    if not labels_dir.exists():
        issues.append(f"  missing: {labels_dir}")
    else:
        n = len(list(labels_dir.glob("*.nii.gz")))
        print(f"  labels/: {n} .nii.gz files")

    if not manifest.exists():
        issues.append(f"  missing: {manifest} "
                      f"(pipeline scripts expect this for stratification)")
    else:
        print(f"  placed_manifest.json: present")

    if issues:
        print("\nWARN: expected files are missing; the repo layout may differ:",
              file=sys.stderr)
        for i in issues:
            print(i, file=sys.stderr)
        print("\nYou may need to adjust the repo structure or skip certain "
              "stages of the pipeline.", file=sys.stderr)

    print("\nNext steps:")
    print("  1. Generate splits:       make splits")
    print("  2. Plan at multi-targets: make plans")
    print("  3. Visualize patches:     make viz CT=data/hf_export/ct/<case>.nii.gz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
