# The LSTV benchmark: what to run, in what order, and what would make it fail

Design notes for the single-pass transitional-anatomy experiment. Written before the first
run, so the failure modes are predictions rather than excuses.

---

## The claim under test

A single network, given a target that contains the right classes, can name a rib by the
vertebra it sits on — distinguishing a **hypoplastic twelfth rib** from a **lumbar rib** —
and can therefore get the rib-free count right in transitional anatomy.

The complementary claim, which should also be tested and is expected to fail: that the same
network can settle **L5 versus L6**. It cannot, because they are the same bone under two
counts and no morphology distinguishes them. Demonstrating exactly where the boundary
between those two claims falls is a sharper result than either "one shot works" or "one
shot fails".

**The premise has already been tested without a GPU and it holds.** On this corpus, plain
logistic regression on eleven shape features — each normalised by the case's own median, so
patient size is removed — separates thoracic from lumbar at **AUC 0.998**, and T12 from L1
specifically at **AUC 0.990, accuracy 97.3%**. The dominant feature is transverse-process
span. That is a floor, not a ceiling: costal facets are not in the feature set at all and a
network sees them directly. See `scripts/test_morphometric_separability.py` in the
CTSpinoPelvic1K repo.

---

## The single largest risk, and it is not the architecture

**nnU-Net's oversampling will not save the rare class, and it looks like it should.**

nnU-Net guarantees that 33% of patches contain a foreground class, choosing that class
uniformly among those present *in the already-selected case*. **Case selection itself is
uniform across the dataset.** So for a class present in 16 of 802 cases:

    P(a patch contains a lumbar rib) ≈ (16/802) × 0.33 × (1 / classes present in that case)

which is a fraction of a percent. The network will see lumbar ribs a handful of times per
epoch and will learn, correctly by its loss, to never predict one.

This is the documented VerSe failure mode arriving by a different route, and it is a
property of the *sampler*, not the backbone. No amount of ResEnc or Primus fixes it.

**The fix has two halves and this repo currently has one of them.**
`tools/lstv_biased_dataloader.py` overrides `get_bbox` to force the patch onto a chosen
class via nnU-Net's own `overwrite_class` hook — that solves the *within-case* half
properly, and it does so as a subclass rather than by monkey-patching, which is why it
survives multiprocessing forks. What it does not do is change *which cases are drawn*, and
it keys on L6 and sacralisation signatures rather than on lumbar ribs.

**Both need extending before the one-shot arm is worth running:**

1. **Case-level oversampling.** Draw cases carrying a variant more often than their 2%
   share. A reasonable target is that variant cases occupy 15–25% of batch slots — enough
   exposure to learn from, not so much that normal anatomy is distorted.
2. **A lumbar-rib signature.** Under `--oneshot` the rare classes are
   `lumbar_rib_left` / `lumbar_rib_right`, which the current subtype detection does not
   look for. Their presence in `class_locations` is the signature and it is unambiguous.

Both are small changes to an existing, working class. Neither is optional.

---

## Arms, in the order worth running

Each is a change to exactly one thing, so a difference can be attributed.

| # | arm | what it isolates |
|---|---|---|
| 1 | `SCHEME=oneshot`, ResEnc-L, default planning | the honest baseline |
| 2 | arm 1 + case oversampling + lumbar-rib signature | whether the rare class is learnable at all |
| 3 | arm 2 with a tall narrow patch (112 × 112 × 320) | whether column-spanning context helps the count |
| 4 | arm 2 as `3d_cascade_fullres` | global context *and* native detail, the principled resolution |
| 5 | `SCHEME=rib_regions` on the best of the above | whether nested regions beat exclusive classes |
| 6 | `SCHEME=countfree` + the deterministic counting stage | the staged comparator that already exists |

Arms 1 and 2 are the pair that matters most: run them on identical folds and the delta is
the sampling contribution, cleanly separated from everything else. If arm 2 does not move
the transitional numbers, nothing later will.

**Do not start with Primus.** *nnU-Net Revisited* found CNNs beating Transformers under
controlled comparison, and although that benchmark was on organ segmentation — where
long-range ordering is not the task, so the result transfers imperfectly here — it is still
the case that a transformer arm before a working sampler would confound two changes at
once. Primus/PrimusV2 (Wald, Roy, Isensee et al., TMLR 2025) belongs after arm 4, as a
backbone swap on a pipeline that already works.

---

## Patch geometry, measured on this corpus

T11 to the S1 endplate is about **265 mm** of column. Bi-iliac width is **278 mm** at the
median and **309 mm** at the 95th percentile.

| patch (vox) | spacing | covers (mm) | Mvox | |
|---|---|---|---:|---|
| 192³ (nnU-Net default) | 1.0 | 192³ | 7.08 | sees neither |
| **112 × 112 × 320** | 1.0 | 112 × 112 × 320 | **4.01** | whole column, no pelvis |
| 160 × 160 × 320 | 2.0/2.0/1.0 | 320³ | 8.19 | both, but 2 mm blurs the costotransverse joint |

The tall narrow patch is **cheaper than the isotropic default** — memory scales with voxel
count and a thin tall box has fewer voxels than a fat cube. The context is not bought, it
is saved. nnU-Net will not choose this shape on its own because its heuristic aims at
isotropic context; overriding means editing `patch_size` in the generated plans and giving
the configuration a new name.

The 2 mm in-plane option is the one that covers the pelvis too, and it costs exactly the
resolution the rib discrimination needs — the costotransverse joint space is 2–4 mm. That
is the argument for the cascade rather than a single compromise patch.

---

## What to measure

Dice is not the endpoint and reporting it alone would hide the result.

1. **Rib-free count accuracy, reported separately for typical and transitional anatomy.**
   A model 97% accurate overall and 0% on transitional cases is the null result in
   disguise, and only the split reveals it. This is the headline number.
2. **Hypoplastic-twelfth versus lumbar-rib confusion**, as a 2×2. This is the specific
   claim. Note that among ribs 50 mm or shorter this corpus holds **152 twelfth ribs
   against 10 lumbar ribs**, so a model that always says "twelfth" scores 94% on short
   ribs — the confusion matrix must be read by class, never by accuracy.
3. **L5 versus L6 accuracy**, expected to be poor, reported anyway. It is the boundary of
   what morphology can do and it is half the result.
4. **Identification rate** in the VerSe sense, for comparability with published work.
5. **The failure mode when it fails** — off-by-one traceable to a specific vertebra, or a
   silent misnumbering.

Per-fold, with the variant subgroup counted in each fold before any of it is believed. The
preflight refuses to proceed if a validation fold contains fewer than two variant cases,
because such a fold cannot measure the thing the benchmark exists to measure and its score
will look fine regardless.

---

## Running it

```bash
# preflight only (fast, no GPU) -- also runs automatically inside spine_prep.sh
python tools/preflight_lstv.py --labels data/v5_final --splits <splits_5fold.json>

# arm 1: convert + preprocess
SCHEME=oneshot DATASET_ID=810 DATASET_NAME=SpineSurgLSTVOneShot \
  PLANNER=nnUNetPlannerResEncL sbatch slurm/spine_prep.sh

# then five folds
for f in 0 1 2 3 4; do
  FOLD=$f DATASET_ID=810 sbatch slurm/spine_train_fold.sh
done
```

Dataset ids are kept apart deliberately — **810 oneshot, 811 rib_regions, 812 countfree**.
Two arms sharing an id overwrite each other's preprocessed data, and the failure presents
as an unexplained change in accuracy rather than as an error. `spine_prep.sh` refuses to
run an LSTV scheme under the default id 803 for exactly that reason.

---

## Prior art the design rests on

- **VerSe** (Sekuboyina et al., *Med Image Anal* 2021) — the benchmark, and the source of
  the finding that transitional anatomy is the dominant labelling failure. Two-stage
  locate-then-label beat direct multi-class segmentation.
- **SpatialConfiguration-Net** (Payer et al.) — won VerSe-2020: local appearance multiplied
  by the global joint configuration of all vertebrae, after a U-Net centreline heatmap.
  The strongest baseline in this domain and the cleanest statement that identity needs
  global configuration.
- **Btrfly Net with an adversarial spine prior** (Sekuboyina et al.) — learns the
  anatomical constraint instead of coding it. The direct comparator for a learned prior.
- **nnU-Net Revisited** (Isensee et al., MICCAI 2024) — CNNs beat Transformers and Mamba
  under controlled comparison; many published claims did not survive scrutiny. Use
  ResEnc-L as the baseline.
- **Primus / PrimusV2** (Wald, Roy, Isensee et al., TMLR 2025) — most 3D "Transformers"
  over-rely on convolutional blocks so heavily that removing the Transformer changes
  nothing. Transformer-centric with improved positional embeddings, implemented inside
  nnU-Net. The attention arm, once the pipeline works.
- **CoordConv** (Liu et al. 2018) and **Kayhan & van Gemert** (CVPR 2020) — CNNs already
  leak absolute position through zero padding, so the invariance is a leaky abstraction and
  a network may be counting by a cue nobody can audit. An argument for supplying position
  explicitly, e.g. a sacrum-relative distance channel.
- **Castellvi** (*Spine* 1984) and **Konin & Walz** (*AJNR* 2010) — the clinical
  classification and why misidentification matters.

Fuller treatment in the CTSpinoPelvic1K repo: `docs/COUNTING_PRIOR_ART.md` and
`docs/ANCHOR_ORDINAL_DESIGN.md`.
