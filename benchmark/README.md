# Competitor benchmark on CTSpinoPelvic1K

Every system is scored in the **same space with the same two scorers** as our own folds: the
Dataset810 one-shot labels (T10–T13, L1–L6, sacrum, ribs 1–11 pooled per side, rib 12, rib 13
and the lumbar rib per side, hips, femur; 22 foreground classes + ignore) through
`tools/eval_oneshot.py` (per-class Dice split typical/transitional, rib-free count accuracy by
transitional status, where the twelfth-rib / thirteenth-rib / lumbar-rib voxels went,
rare-class emission) and `tools/decode_sequence.py` (instances → posterior over the rib-free
count and the A/B/C readings; `--onehot` for hard-label systems).

Predictions stay **native on disk** (`$BENCH/<system>/native/`); the map into the one-shot
space is a re-runnable step (`to_oneshot.py`, `mappings/<system>.json`), so a re-score never
needs a re-run. `$BENCH` defaults to `~/bench` on the grid.

## Case table

`make_case_table.py` → `benchmark/cases.csv`: all 802 Dataset810 records with host CT path,
one-shot label, fold, subtype, L6 / lumbar-rib flags. Our model is scored on its validation
fold (5 folds = all 802 once); every competitor runs on all 802 (none was trained on them;
VISTA3D's training list should be checked for CTSpine1K/CTPelvic1K overlap before its number
is quoted).

## Systems

| system | what it gives | how it runs | status |
|---|---|---|---|
| ours (Dataset810 one-shot) | 22 classes + softmax | `slurm/eval_fold_checkpoint.sh` per fold | folds training (9/2026) |
| TotalSegmentator v2 (`total`) | vertebrae C1–L5 + S1, sacrum, ribs 1–12/side, hips, femora | `run_totalsegmentator.sh` (4-shard GPU array; container + weights from the dataset build) | ready |
| SPINEPS-CT + VERIDAH | vertebra instances in VerSe ids incl. **L6 (25) and T13 (28)**; no ribs/pelvis | `~/CTSpinoPelvic1K/slurm/spineps_bench.sh` (existing; container `ctspinopelvic1k-spineps.sif` bakes the CT weights) then `score.sh` with `PATTERN='{case}_seg-vert_msk.nii.gz'` | container to pull; rib assignment map to write |
| Möller rib nnU-Net + rib-segmentation | binary ribs, rib→vertebra assignment, rib length, stump features | same SPINEPS job (`RIB_MODEL=…`) | with SPINEPS |
| RibSeg v2 (PointNet++) | ribs 1–12 per side from a binary rib mask | `~/CTSpinoPelvic1K/slurm/ribseg.sh` (needs `third_party/RibSeg` + weights; not on the grid yet) | to set up |
| VISTA3D (NVIDIA/MONAI) | 127 classes incl. vertebrae C1–L5, S1, sacrum, ribs 1–12/side, hips, femora | MONAI bundle in a new conda env (`vista3d`), runner to write | to set up |

None of the competitors has a lumbar-rib class; TotalSegmentator, VISTA3D and RibSeg have
no L6 and no T13. Those are scored as what they are: the lumbar rib lands in their rib-12
or rib-11 call (visible in the rib confusion), and the sixth body lands in L5 (the
tall-component split in the decoder still counts it, so the **count** is compared fairly
while the **name** is a miss).

## Run order

```bash
# once
python benchmark/make_case_table.py --raw nnunet/raw/Dataset810_SpineSurgLSTVOneShot \
    --preprocessed nnunet/preprocessed/Dataset810_SpineSurgLSTVOneShot \
    --hf_export ~/data/CTSpinoPelvic1K --out benchmark/cases.csv
# sanity: the scorers on ground truth itself must give Dice 1 and exact counts
SYSTEM=gt PRED_DIR=nnunet/raw/Dataset810_SpineSurgLSTVOneShot/labelsTr NAME_MAP=identity sbatch benchmark/score.sh
# TotalSegmentator
LIMIT=2 sbatch --array=0 --time=00:40:00 benchmark/run_totalsegmentator.sh   # smoke
sbatch benchmark/run_totalsegmentator.sh
SYSTEM=totalsegmentator sbatch benchmark/score.sh
# ours (after slurm/eval_fold_checkpoint.sh)
SYSTEM=ours_f2 PRED_DIR=<fold_2/eval_checkpoint_best_*/pred> NAME_MAP=identity PROBS=1 sbatch benchmark/score.sh
```

`collect.py` (to write) joins `$BENCH/*/score/summary.json` into the comparison table.
