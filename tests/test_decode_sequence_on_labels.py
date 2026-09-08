"""End-to-end check of tools/decode_sequence.py on ground-truth labels made to look like predictions.

A released v10 label volume is remapped to the one-shot classes, written as a "prediction",
and (unless --onehot) a one-hot probability .npz is written beside it. Decoding one-hot
ground truth must reproduce the label's own rib-free count exactly, with a posterior of 1.0
on it. That proves the instance separation, the axis handling and the monotone decode before
any network output exists.

--collapse renames every L6 voxel L5 first. That is the network's known failure mode and the
output of every competitor without an L6 class: two touching bodies with one name. The count
must STILL come out one higher than the number of distinct lumbar names, which is what the
tall-component split in decode_sequence.py exists for.

    python tests/test_decode_sequence_on_labels.py --labels <v10 labels dir> --cases 0007 0376 1053 --out /tmp/dec_test
    python tests/test_decode_sequence_on_labels.py --labels ... --cases 0376 1053 --out /tmp/dec_collapse --collapse --onehot
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "tools"))
import convert_hf_to_nnunet as cvt  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--collapse", action="store_true", help="rename L6 -> L5 in the 'prediction'")
    ap.add_argument("--onehot", action="store_true", help="write no .npz; decode with --onehot")
    a = ap.parse_args()
    out = Path(a.out); pred = out / "pred"; gt = out / "gt"
    pred.mkdir(parents=True, exist_ok=True); gt.mkdir(parents=True, exist_ok=True)
    names = cvt.LABEL_NAMES_ONESHOT
    lut = cvt._build_verse_remap_lut(cvt.LABEL_REMAP_ONESHOT_FROM_VERSE)
    n_cls = max(names.values()) + 1
    (out / "dataset.json").write_text(json.dumps({"labels": names, "channel_names": {"0": "CT"},
                                                  "numTraining": len(a.cases), "file_ending": ".nii.gz"}))
    for c in a.cases:
        img = nib.load(str(Path(a.labels) / f"{c}_label.nii.gz"))
        lab = np.asanyarray(img.dataobj).astype(np.int64)
        one = lut[np.clip(lab, 0, 255)].astype(np.int16)
        one[one == names["ignore"]] = 0                      # ignore never appears in a prediction
        nib.save(nib.Nifti1Image(one, img.affine, img.header), str(gt / f"COLONOG_{c}.nii.gz"))
        if a.collapse:
            one[one == names["L6"]] = names["L5"]
        nib.save(nib.Nifti1Image(one, img.affine, img.header), str(pred / f"COLONOG_{c}.nii.gz"))
        if not a.onehot:
            probs = np.zeros((n_cls,) + one.shape, dtype=np.float16)
            for k in range(n_cls):
                probs[k][one == k] = 1.0
            np.savez_compressed(pred / f"COLONOG_{c}.npz", probabilities=probs)
        print("made", c, "classes present:", sorted(int(x) for x in np.unique(one)))
    cmd = [sys.executable, str(HERE / "tools/decode_sequence.py"), "--predictions_dir", str(pred),
           "--dataset_json", str(out / "dataset.json"), "--labels_dir", str(gt), "--out", str(out / "decoded.csv")]
    if a.onehot:
        cmd.append("--onehot")
    print(" ".join(cmd))
    rc = subprocess.call(cmd)
    if rc:
        return rc
    bad = 0
    with open(out / "decoded.csv") as fh:
        for r in csv.DictReader(fh):
            ok = r.get("count_error") == "0"
            bad += not ok
            print(f"{r['case']}: decoded rib-free {r.get('rib_free_map')} vs label {r.get('gt_rib_free')} "
                  f"{'OK' if ok else 'MISMATCH'}")
    print("PASS" if not bad else f"FAIL ({bad})")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
