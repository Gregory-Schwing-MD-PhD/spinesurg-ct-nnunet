"""End-to-end check of tools/decode_sequence.py on ground-truth labels made to look like predictions.

A released v10 label volume is remapped to the one-shot classes, written as a "prediction",
and a one-hot probability .npz is written beside it. Decoding one-hot ground truth must
reproduce the label's own rib-free count exactly, with a posterior of 1.0 on it. That
proves the instance separation, the axis handling and the monotone decode before any
network output exists.

    python tests/test_decode_sequence_on_labels.py --labels <v10 labels dir> --cases 0007 0376 1053 --out /tmp/dec_test
"""
from __future__ import annotations

import argparse
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
        nib.save(nib.Nifti1Image(one, img.affine, img.header), str(pred / f"COLONOG_{c}.nii.gz"))
        nib.save(nib.Nifti1Image(one, img.affine, img.header), str(gt / f"COLONOG_{c}.nii.gz"))
        probs = np.zeros((n_cls,) + one.shape, dtype=np.float16)
        for k in range(n_cls):
            probs[k][one == k] = 1.0
        np.savez_compressed(pred / f"COLONOG_{c}.npz", probabilities=probs)
        print("made", c, "classes present:", sorted(int(x) for x in np.unique(one)))
    cmd = [sys.executable, str(HERE / "tools/decode_sequence.py"), "--predictions_dir", str(pred),
           "--dataset_json", str(out / "dataset.json"), "--labels_dir", str(gt), "--out", str(out / "decoded.csv")]
    print(" ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
