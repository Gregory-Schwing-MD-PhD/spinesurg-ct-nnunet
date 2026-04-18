# Dataset

This pipeline expects the CTSpinoPelvic1K dataset at `data/hf_export/`.
The dataset lives in a separate repo ([gschwing/CTSpinoPelvic1K on
HuggingFace](https://huggingface.co/datasets/gschwing/CTSpinoPelvic1K))
so you have three options for getting it here.

## Option A — Download from HuggingFace (easiest)

```bash
pip install huggingface_hub

# If the dataset is private, authenticate first:
huggingface-cli login          # or: export HF_TOKEN=hf_xxx

# Then download:
make download-data
# OR directly:
python scripts/download_dataset.py --repo_id gschwing/CTSpinoPelvic1K
```

To download only CT + label NIfTIs and skip auxiliary files:

```bash
python scripts/download_dataset.py \
    --allow_patterns 'ct/*.nii.gz' 'labels/*.nii.gz' 'placed_manifest.json'
```

## Option B — Symlink an existing local copy (recommended on HPC)

If the dataset is already on a shared filesystem (e.g. `/shared/datasets/`),
symlink into place — no file duplication:

```bash
ln -s /shared/datasets/CTSpinoPelvic1K data/hf_export
```

## Option C — Bind mount

If you want a read-only mount (e.g. for multi-user clusters):

```bash
mkdir -p data/hf_export
sudo mount --bind -o ro /shared/datasets/CTSpinoPelvic1K data/hf_export
```

## Expected layout

After any of the above, `data/hf_export/` must look like:

```
data/hf_export/
├── ct/
│   ├── <token_0001>.nii.gz
│   ├── <token_0002>.nii.gz
│   └── ...
├── labels/
│   ├── <token_0001>.nii.gz
│   ├── <token_0002>.nii.gz
│   └── ...
└── placed_manifest.json          # per-token metadata (match_type, LSTV, ...)
```

**Label convention** (10-class segmentation + ignore):

| value | structure       |
|------:|-----------------|
|   0   | background      |
|   1   | L1              |
|   2   | L2              |
|   3   | L3              |
|   4   | L4              |
|   5   | L5              |
|   6   | L6 (LSTV)       |
|   7   | sacrum          |
|   8   | left hip        |
|   9   | right hip       |
|  10   | ignore          |

**placed_manifest.json schema** (per-token):

```json
{
  "<token>": {
    "match_type": "fused | separate | spine_only | pelvic_only | partial",
    "has_lstv": true,
    "patient_id": "COLONOG_0123",
    "series_uid": "1.2.840...",
    "source": "CTSpine1K | CTPelvic1K | COLONOG | ..."
  },
  ...
}
```

## Directory is gitignored

`data/hf_export/` is listed in `.gitignore` — it will never be committed to
this repo. Keep dataset versioning in the dataset repo.
