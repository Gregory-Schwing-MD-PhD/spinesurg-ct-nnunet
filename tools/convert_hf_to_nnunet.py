"""
convert_hf_to_nnunet.py — Convert HF NIfTI-pair export to nnU-Net v2 raw layout.

Reads
-----
  HF_DIR/
    manifest_train.json
    manifest_validation.json
    manifest_test.json
    ct/{token}_ct.nii.gz
    labels/{token}_label.nii.gz
  splits_5fold.json    (from generate_5fold_splits.py v6+)

Writes
------
  NNUNET_RAW/Dataset{ID}_{NAME}/
    imagesTr/{caseID}_0000.nii.gz   (symlinks or copies of CT)
    labelsTr/{caseID}.nii.gz        (REMAPPED copies — see below)
    imagesTs/{caseID}_0000.nii.gz   (test set)
    labelsTs/{caseID}.nii.gz
    dataset.json                    (label scheme + ignore)
    lstv_cases.json                 (schema v3 — 6-way LSTV subtypes)
    splits_final.json               (5-fold CV splits, mirrored from input)

Modes
-----
Default: full build (link/copy + dataset.json + splits + lstv_cases).
--regen_splits_only: only re-write splits_final.json + lstv_cases.json.
    Skips the per-record link/copy loop and skips dataset.json. Use this
    when the dataset already exists on disk and you only need to refresh
    the fold assignments / subtype mapping (e.g. after re-running
    generate_5fold_splits.py).

Source label scheme (VerSe-native — the dataset's single source of truth)
-------------------------------------------------------------------------
As of the VerSe migration, the HF export label NIfTIs carry VerSe-native
label ids (frozen in label_scheme.py in the CTSpinoPelvic1K repo; that
file is NOT importable here, so the relevant ids are hardcoded below as
named constants):
  vertebrae:   C1-C7 = 1-7,  T1-T12 = 8-19,  L1-L6 = 20-25,
               sacrum = 26,  coccyx = 27,  T13 = 28,  S1 = 29
  pelvis/limb: left_hip = 30, right_hip = 31, femur_left = 32,
               femur_right = 33
  ribs = 34-46 left, 47-59 right (thirteen per side; 46/59 are the
               thirteenth rib, empty in this cohort),  lumbar ribs = 60/61,
               hardware = 62-68
  ignore = 255
(v10 scheme, September 2026. Earlier releases put ribs at 34-57,
soft tissue at 58-73 and lumbar ribs at 74/75; those ids are gone.)
Only the lumbar bodies (20-25), sacrum (26), hips (30/31) and ignore
(255) are training classes; EVERY OTHER nonzero VerSe id (thoracic /
cervical vertebrae, coccyx, T13, S1, femurs, ribs, hardware) is
dropped to background at convert time. See LABEL_REMAP_MERGED_FROM_VERSE
/ LABEL_REMAP_UNMERGED_FROM_VERSE.

Target label scheme (May 2026 — Dataset803 / "merged" variant)
--------------------------------------------------------------
  0 background
  1 L1           (from VerSe L1 = 20)
  2 L2           (from VerSe L2 = 21)
  3 L3           (from VerSe L3 = 22)
  4 L4           (from VerSe L4 = 23)
  5 last_lumbar  (VerSe L5 = 24 AND L6 = 25 both merged here — see
                  LABEL_REMAP_MERGED_FROM_VERSE)
  6 sacrum       (from VerSe sacrum = 26)
  7 left_hip     (from VerSe left_hip = 30)
  8 right_hip    (from VerSe right_hip = 31)
  9 ignore       (from VerSe ignore = 255)

The unmerged Dataset802 target keeps L5 (24) and L6 (25) distinct:
  0 bg, 1-4 L1-L4, 5 L5, 6 L6, 7 sacrum, 8 left_hip, 9 right_hip,
  10 ignore.  Selected by --no_remap (VerSe-native -> unmerged 10-class).

The COUNT-FREE target (--countfree) asks for no vertebral identity at all:
  0 bg, 1 vertebra, 2 disc_space, 3 rib_left, 4 rib_right, 5 sacrum,
  6 left_hip, 7 right_hip, 8 femur, 9 ignore.
Both schemes above hand the network a naming problem it cannot solve
locally -- L5 and L6 are the same bone under two counts, and which one
you are looking at depends on how many vertebrae lie above, possibly
outside the field of view. The count-free target moves that decision out
of the loss and into code: the network says what is bone and where one
vertebra ends, and counting, naming and transitional flagging happen
afterwards, deterministically. Class 2 is DERIVED at convert time and
appears in no source file; see _derive_disc_space().

Why contiguous renumbering matters
==================================
nnU-Net v2 builds the network's output channel count from the SORTED
LIST of label values (excluding the ignore label). The trainer's
per-class dice / confusion code then compares network argmax outputs
(channel indices) against ground-truth label values, treating them
as equivalent. This equivalence holds only when label values are
CONTIGUOUS — a gap (e.g., labels 0,1,...,5,7,8,9 with no 6) breaks
the channel-index ↔ label-value mapping and corrupts every per-class
metric. So when we drop L6, we shift sacrum/hips/ignore down by one
to maintain contiguity.

Why labels are merged
=====================
Direct multi-class semantic segmentation of L5 vs L6 fails on
transitional anatomy because the disambiguating signal is non-local:
"the bottom-most lumbar vertebra" looks anatomically near-identical
whether it is the L5 of a 5-lumbar normal spine or the L6 of a
6-lumbar lumbarized spine. Voxel-wise classification cannot solve
this — see Möller et al. 2026 (VERIDAH), Section 2.2:

    "we replace the T13 and L6 labels in the reference annotation
     with T12 and L5, respectively. Thus, the existence of a T13
     and L6 is indicated by two consecutive T12 or L5 labels."

We follow the same pattern for L6 only (we do not have T13 cases in
this dataset). Individual L5/L6 labels are recovered downstream via
instance-level post-processing (connected components on the
last_lumbar mask, ordered superior-to-inferior, with count
determining whether the bottom is L5 or L6).

Why "ignore" is here
====================
Some HF export records are *partial annotations* — separate-mode
patients have one CT scan annotated only for spine (L1-L6) and a
different CT scan annotated only for pelvis (sacrum + hips). The
patched export_hf.py (May 2026) writes those label files with value
10 in voxels outside the present annotator's domain.

nnU-Net v2 honors `"ignore"` in `labels` by:
  - setting `has_ignore_label = True` on the LabelManager
  - using `ignore_label = 10` as `ignore_index` in CE loss
  - masking ignore voxels out of the Dice loss via the DC_and_CE_loss
    wrapper when `ignore_label` is configured
  - NOT allocating an output head for class 10

Without this entry, partial-annotation labels would falsely supervise
the network with "background" wherever the partial annotator did not
trace, poisoning the loss for last_lumbar / sacrum / hips on roughly
two-thirds of the training set. See export_hf.py
"PARTIAL ANNOTATION CONTRACT".

Per-record subtype refinement (May 2026, schema v3)
===================================================
Unchanged from prior version. A patient with LSTV morphology may
contribute multiple records with different `config` values (fused,
spine_only, pelvic_native). The patient-level subtype is correct only
for records whose config actually shows the relevant anatomy:

  pelvic_native  ->  demote {lumb, sacr_count} to "normal"
                     (lumbar region masked as ignore=10)
  spine_only     ->  demote {sacralization, semisacralization} to
                     "normal" (pelvis masked as ignore=10)
  fused          ->  preserve all subtypes
  ambiguous      ->  preserved on every config

lstv_cases.json schema v3 (Apr 2026)
====================================
6-way LSTV subtype taxonomy:
  normal, lumb, sacr_count, semisacralization, sacralization, ambiguous

  {
    "schema_version": 3,
    "subtypes": ["normal", "lumb", ...],
    "subtype_counts": {...},
    "case_to_subtype":   {caseID: subtype, ...},
    "case_to_token":     {caseID: patient_token, ...},
    "case_to_attrs":     {caseID: {n_lumb_max, has_l6_any, lstv_label, ...}},
    "lstv_oversample_pool": [caseID, ...]   (all non-normal cases for sampler)
  }
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("spinesurg.convert_hf_to_nnunet")

LSTV_CASES_SCHEMA = 3

SUBTYPES = (
    "normal",
    "lumb",
    "sacr_count",
    "semisacralization",
    "sacralization",
    "ambiguous",
)

# Subtypes whose defining anatomy lives in the LUMBAR vertebral region.
# pelvic_native config masks the lumbar region as ignore=10, so a
# pelvic_native record does NOT carry these subtypes' signal.
_LUMBAR_REGION_SUBTYPES = frozenset({
    "lumb",
    "sacr_count",
})

# Subtypes whose defining anatomy lives in the SACRAL/PELVIC region.
# spine_only config masks the pelvis as ignore=10, so a spine_only
# record does NOT carry these subtypes' signal.
_PELVIS_REGION_SUBTYPES = frozenset({
    "sacralization",
    "semisacralization",
})

_VALID_CONFIGS = ("fused", "spine_only", "pelvic_native")
_PELVIC_ONLY_CONFIG = "pelvic_native"
_SPINE_ONLY_CONFIG  = "spine_only"

# ── Label scheme (Dataset803 — merged last_lumbar) ─────────────────────
#
# nnU-Net v2 reads the "labels" dict to determine the network's output
# head channel count. Each foreground name -> int value pair becomes
# an output channel. The "ignore" entry is special: it is NOT a
# channel; it is used as the ignore_index in the CE loss.
#
# Label 6 (VerSe L6 = 25) is intentionally absent from LABEL_NAMES — the
# L6 voxels from source NIfTIs are merged into label 5 (last_lumbar) at
# convert time via LABEL_REMAP_MERGED_FROM_VERSE. The network has no
# channel for L6 because there are no L6 voxels in the produced data.
LABEL_NAMES = {
    "background":  0,
    "L1":          1,
    "L2":          2,
    "L3":          3,
    "L4":          4,
    "last_lumbar": 5,   # merged VerSe L5 (24) + L6 (25) (see module docstring)
    "sacrum":      6,   # from VerSe sacrum = 26
    "left_hip":    7,   # from VerSe left_hip = 30
    "right_hip":   8,   # from VerSe right_hip = 31
    "ignore":      9,   # from VerSe ignore = 255
}

# 10-key UNMERGED scheme (Dataset802) — L5 and L6 kept distinct.
# Selected by --no_remap. Produced by LABEL_REMAP_UNMERGED_FROM_VERSE.
LABEL_NAMES_UNMERGED = {
    "background":  0,
    "L1":          1,
    "L2":          2,
    "L3":          3,
    "L4":          4,
    "L5":          5,   # from VerSe L5 = 24
    "L6":          6,   # from VerSe L6 = 25
    "sacrum":      7,   # from VerSe sacrum = 26
    "left_hip":    8,   # from VerSe left_hip = 30
    "right_hip":   9,   # from VerSe right_hip = 31
    "ignore":      10,  # from VerSe ignore = 255
}

# ── VerSe-native SOURCE label ids (single source of truth) ─────────────
#
# These ids are FROZEN in label_scheme.py in the CTSpinoPelvic1K repo.
# That module is NOT importable here (it lives in a different repo that is
# not present on the cluster), so the training-relevant ids are hardcoded
# below. If label_scheme.py ever changes, update these to match.
#
#   vertebrae:   C1-C7 = 1-7,  T1-T12 = 8-19,  L1-L6 = 20-25,
#                sacrum = 26,  coccyx = 27,  T13 = 28,  S1 = 29
#   pelvis/limb: left_hip = 30, right_hip = 31, femur_left = 32,
#                femur_right = 33
#   ribs = 34-46 left, 47-59 right (13 per side; 46/59 = rib 13),
#   lumbar ribs = 60/61,  hardware = 62-68
#   ignore = 255
VERSE_RIB_L = tuple(range(34, 47))      # rib_left_1 .. rib_left_13
VERSE_RIB_R = tuple(range(47, 60))      # rib_right_1 .. rib_right_13
VERSE_LUMBAR_RIB_L, VERSE_LUMBAR_RIB_R = 60, 61
VERSE_L1        = 20
VERSE_L2        = 21
VERSE_L3        = 22
VERSE_L4        = 23
VERSE_L5        = 24
VERSE_L6        = 25
VERSE_SACRUM    = 26
VERSE_LEFT_HIP  = 30
VERSE_RIGHT_HIP = 31
VERSE_IGNORE    = 255

# ── COUNT-FREE scheme (--countfree) ───────────────────────────────────
#
# NO VERTEBRAL IDENTITY IS ASKED OF THE NETWORK. Every vertebra is one
# class regardless of level. The level name is not a property of the bone
# — L5 and L6 are the same object under two different counts — so a
# per-voxel loss cannot express what makes the answer decidable, and a
# network trained on it learns whichever naming convention was commoner
# among the annotators. Counting happens afterwards, in code, where it
# has an audit trail.
#
# Ribs keep a side but not a number, for the same reason: side is locally
# decidable from the image, number is not. A rib on a lumbar body folds
# into the rib class rather than being forced to be a twelfth rib — that
# is the same object under two counts, and choosing here would bake in
# the answer the counting stage exists to derive.
#
# Class 2 is DERIVED, not remapped: see _derive_disc_space().
LABEL_NAMES_COUNTFREE = {
    "background":  0,
    "vertebra":    1,
    "disc_space":  2,   # DERIVED at convert time, not present in any source
    "rib_left":    3,
    "rib_right":   4,
    "sacrum":      5,
    "left_hip":    6,
    "right_hip":   7,
    "femur":       8,
    "ignore":      9,
}

_CF_VERTEBRA, _CF_DISC = 1, 2
_CF_RIB_L, _CF_RIB_R = 3, 4
_CF_SACRUM, _CF_HIP_L, _CF_HIP_R, _CF_FEMUR, _CF_IGNORE = 5, 6, 7, 8, 9

# Every vertebra 1..25 plus T13 (28) collapses to one class. Sacrum (26)
# and the carved S1 (29) are one sacrum. Coccyx (27) and hardware
# (62-68) drop to background.
LABEL_REMAP_COUNTFREE_FROM_VERSE: Dict[int, int] = {
    **{i: _CF_VERTEBRA for i in range(1, 26)},      # C1..L6
    28: _CF_VERTEBRA,                               # T13
    VERSE_SACRUM: _CF_SACRUM,
    29: _CF_SACRUM,                                 # S1, carved from sacrum top
    VERSE_LEFT_HIP: _CF_HIP_L,
    VERSE_RIGHT_HIP: _CF_HIP_R,
    32: _CF_FEMUR, 33: _CF_FEMUR,
    **{i: _CF_RIB_L for i in VERSE_RIB_L},          # ribs 1-13 left
    **{i: _CF_RIB_R for i in VERSE_RIB_R},          # ribs 1-13 right
    VERSE_LUMBAR_RIB_L: _CF_RIB_L,                  # lumbar ribs: still ribs
    VERSE_LUMBAR_RIB_R: _CF_RIB_R,
    VERSE_IGNORE: _CF_IGNORE,
}

# Adjacent-vertebra id pairs whose seam becomes a disc. Consecutive VerSe
# ids within the vertebral column only.
_CF_VERTEBRA_IDS = tuple(range(1, 26))


def _derive_disc_space(arr, zooms, reach_mm: float = 10.0,
                       max_gap_mm: float = 14.0):
    """Return a boolean mask of the intervertebral spaces.

    WHY THIS CLASS EXISTS AT ALL. nnU-Net is semantic, not instance. With
    one "vertebra" class the column comes back as a single connected blob
    wherever two bodies touch — and where they touch is exactly where the
    counting stage needs a boundary. Predicting the space between them as
    its own class turns instance separation into a local segmentation
    problem the network can actually do, and it degrades gracefully: a
    missed disc merges two vertebrae, which shows up as an outlier in
    component height rather than as a silent off-by-one.

    HOW THE SEAM IS FOUND. For each pair of consecutive vertebra ids that
    are BOTH present, the disc is the background lying within reach of
    both, with the sum of the two distances bounded so that a vertebra
    missing from the field of view is not papered over with one enormous
    disc. Distances are in millimetres via the voxel spacing, so the rule
    means the same thing at every slice thickness in the corpus.

    A STRICTER RULE WAS TRIED AND IS WRONG. Requiring each voxel to lie
    geometrically BETWEEN its two nearest surface points removes a collar
    of voxels that wrap around the sides of the bodies -- but real
    endplates are concave, so the two nearest points are seldom collinear
    with the voxel between them, and on real anatomy the test rejected
    almost the whole seam. It left 2262 fragments instead of a disc on
    case 0001, and the vertebrae it was supposed to separate stayed
    merged into one component. The lateral collar the strict rule removes
    is roughly one slice deep and does no harm; failing to cut the column
    does.

    WHAT IS DELIBERATELY NOT DERIVED: the lumbosacral disc. Whether a
    disc exists between the lowest mobile vertebra and the sacrum is the
    transitional finding itself, and a geometric rule that manufactures
    one whenever there is a gap would answer the question the counting
    stage is supposed to ask. Sacrum keeps its own class and needs no
    separator from the vertebra above it, so nothing is lost by leaving
    it alone.

    Operates in the array's own frame. Nothing is reoriented; the caller
    writes back with the source affine.
    """
    from scipy import ndimage

    out = np.zeros(arr.shape, dtype=bool)
    present = {v for v in _CF_VERTEBRA_IDS if np.any(arr == v)}
    pad = [int(np.ceil(reach_mm / max(z, 1e-6))) + 1 for z in zooms]

    for v in sorted(present):
        w = v + 1
        if w not in present:
            continue
        both = (arr == v) | (arr == w)
        idx = np.argwhere(both)
        lo = np.maximum(idx.min(0) - pad, 0)
        hi = np.minimum(idx.max(0) + 1 + pad, arr.shape)
        sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        sub = arr[sl]

        d_v = ndimage.distance_transform_edt(sub != v, sampling=zooms)
        d_w = ndimage.distance_transform_edt(sub != w, sampling=zooms)
        disc = ((sub == 0) & (d_v <= reach_mm) & (d_w <= reach_mm)
                & (d_v + d_w <= max_gap_mm))

        # AND A CUT WHERE THE TWO BODIES ABUT, WHICH THE GAP RULE CANNOT REACH.
        # Adjacent vertebrae meet in three places -- the disc and the two facet joints
        # -- and in these labels they frequently touch with NO background voxel between
        # them, because the tools that produced the source segment bone and not joints.
        # Where there is no gap there is nothing for a gap rule to fill, so the column
        # stayed connected through the facets and came back as 2 components for 10
        # vertebrae: the separator did not separate.
        #
        # The fix carves a one-voxel sheet from each body along the surface where they
        # touch. That means the class is not purely background -- it takes back a thin
        # shell the source assigned to bone -- and that is the honest reading: a real
        # joint occupies space these labels give to one side or the other, and the
        # boundary has to exist somewhere. 26-connectivity so that a diagonal contact is
        # cut too; a diagonal leak reconnects the column just as thoroughly as a face one.
        # THE SHELL COMES OFF THE LARGER OF THE TWO, NOT OFF BOTH. Removing a layer
        # from both sides separates them twice over and costs nothing on two full
        # vertebrae -- but the topmost and bottom-most vertebrae in a field of view are
        # truncated slabs a few voxels thick, and taking a layer off one of those breaks
        # it into pieces. On case 0001 that turned a 4064-voxel partial T8 into 3673 +
        # 391 and made the component count 11 for 10 vertebrae: an off-by-one, which is
        # the exact failure this whole design exists to prevent.
        #
        # One side is enough. Under 26-connectivity, deleting every voxel of one body
        # that touches the other leaves nothing of it adjacent to the other, so the two
        # are separated whichever side the layer came from. Taking it from the larger
        # body means a truncated sliver is never the one cut into.
        nb = ndimage.generate_binary_structure(3, 3)
        mv, mw = (sub == v), (sub == w)
        if int(mv.sum()) >= int(mw.sum()):
            disc |= mv & ndimage.binary_dilation(mw, structure=nb)
        else:
            disc |= mw & ndimage.binary_dilation(mv, structure=nb)
        out[sl] |= disc
    return out



# Source (VerSe-native) -> trained-label remaps. Applied by
# _remap_and_write_label() during the link/copy loop. These are consumed
# by _build_verse_remap_lut(), which defaults EVERY unlisted nonzero
# source id to 0 (background) — thoracic/cervical vertebrae, coccyx, T13,
# S1, femurs, ribs and soft-tissue are NOT training classes and are
# dropped. Background (0) stays 0.
#
# MERGED (Dataset803, default): collapse VerSe L5+L6 into last_lumbar (5).
LABEL_REMAP_MERGED_FROM_VERSE: Dict[int, int] = {
    VERSE_L1:        1,
    VERSE_L2:        2,
    VERSE_L3:        3,
    VERSE_L4:        4,
    VERSE_L5:        5,   # L5 -> last_lumbar
    VERSE_L6:        5,   # L6 -> last_lumbar (the merge)
    VERSE_SACRUM:    6,
    VERSE_LEFT_HIP:  7,
    VERSE_RIGHT_HIP: 8,
    VERSE_IGNORE:    9,   # partial-annotation contract
}

# UNMERGED (Dataset802, --no_remap): keep VerSe L5 and L6 distinct.
LABEL_REMAP_UNMERGED_FROM_VERSE: Dict[int, int] = {
    VERSE_L1:        1,
    VERSE_L2:        2,
    VERSE_L3:        3,
    VERSE_L4:        4,
    VERSE_L5:        5,
    VERSE_L6:        6,
    VERSE_SACRUM:    7,
    VERSE_LEFT_HIP:  8,
    VERSE_RIGHT_HIP: 9,
    VERSE_IGNORE:    10,
}

# No-ignore variant (fused-only ablation, May 2026).
# ==================================================
# Reviewer ask (camera-ready): a "fused-only versus all-cases" ablation,
# evaluated on the fused test split. Fused records are FULLY annotated —
# export_hf.py guarantees they carry NO ignore (10) voxels (it logs an
# error if a fused label ever contains one). So when we restrict the
# training/test pool to fused records, the ignore label becomes vestigial
# and we drop it entirely: there is no partial-annotation machinery, no
# ignore_index in the loss. This is the "don't use the ignore protocol"
# arm of the ablation.
#
# The L6 -> last_lumbar merge is KEPT (it is orthogonal to the ignore
# protocol) so the label semantics stay identical to the main Dataset803
# model and the two arms differ only in (training pool, ignore protocol).
#
# VerSe ignore (255) -> background (0) defensively. On fused data this
# never fires; it only matters if a stray ignore voxel slips through, in
# which case treating it as background is the safe, fully-supervised
# choice (there is no ignore class to route it to). Same VerSe->merged
# mapping as LABEL_REMAP_MERGED_FROM_VERSE, but 255 -> 0 instead of 9.
LABEL_REMAP_MERGED_NO_IGNORE_FROM_VERSE: Dict[int, int] = {
    VERSE_L1:        1,
    VERSE_L2:        2,
    VERSE_L3:        3,
    VERSE_L4:        4,
    VERSE_L5:        5,   # L5 -> last_lumbar
    VERSE_L6:        5,   # L6 -> last_lumbar (the merge)
    VERSE_SACRUM:    6,
    VERSE_LEFT_HIP:  7,
    VERSE_RIGHT_HIP: 8,
    VERSE_IGNORE:    0,   # ignore -> background (defensive; fused carries none)
}

# 9-key scheme (8 foreground + background), no ignore class.
LABEL_NAMES_NO_IGNORE = {
    "background":  0,
    "L1":          1,
    "L2":          2,
    "L3":          3,
    "L4":          4,
    "last_lumbar": 5,
    "sacrum":      6,
    "left_hip":    7,
    "right_hip":   8,
}

# The maximum label value the VerSe-native source NIfTIs are expected to
# contain (ignore = 255). The LUT spans the full 0..255 range so every
# VerSe id — including ribs (34-59), lumbar ribs (60/61) and hardware
# (62-68) that are NOT training classes — has an explicit destination.
_MAX_SOURCE_LABEL = 255


# ── RIB-BEARING REGIONS (--rib_regions) ───────────────────────────────
#
# WHY SPLIT THE VERTEBRA CLASS THIS WAY, AND WHY AS A REGION.
#
# The count-free scheme above refuses to ask the network for vertebral
# identity because identity is not a local property. Rib-bearing status
# IS one: the rib is in the image, articulating with the vertebra, and a
# per-voxel loss can see it. So it is the one piece of the counting
# problem that can honestly be moved INTO the network, and it happens to
# be the piece the count is anchored on -- the lowest rib-bearing
# vertebra is the top bracket of the interval that defines the phenotype.
#
# Regions rather than two more mutually exclusive classes, because the
# two facts are NESTED: rib-bearing implies vertebra. As exclusive
# classes a softmax must spend probability deciding between them, so a
# vertebra the network is sure about but whose rib status is genuinely
# ambiguous gets a diluted answer on BOTH. As nested regions with
# independent sigmoids it can say "certainly a vertebra, 0.5 that it
# bears a rib", which is the true state of belief at a transitional
# level and is exactly what the downstream counting stage wants to
# consume.
#
# WHAT THIS DOES NOT SOLVE, stated plainly. The hard case is a lumbar
# rib against a large transverse process, and that IS the Castellvi
# question -- the ambiguity does not disappear because the label was
# reorganised. What changes is that the model can now EXPRESS the
# ambiguity as a calibrated probability instead of being forced to
# resolve it, and a count assembled from probabilities can carry its own
# uncertainty. That is the argument for expecting it to beat the
# exclusive-class formulation, and it is a claim to be measured, not
# assumed.
#
# Underlying (mutually exclusive) values stored in the label files:
_RR_VERT_PLAIN, _RR_VERT_RIB = 1, 2
_RR_DISC, _RR_RIB_L, _RR_RIB_R = 3, 4, 5
_RR_SACRUM, _RR_HIP_L, _RR_HIP_R, _RR_FEMUR, _RR_IGNORE = 6, 7, 8, 9, 10

# dataset.json "labels": encompassing regions FIRST, substructures after,
# because regions_class_order paints them in order and later wins.
# Insertion order matters and must survive json.dump -- see sort_keys.
LABEL_NAMES_RIB_REGIONS = {
    "background":  0,
    "vertebra":    [_RR_VERT_PLAIN, _RR_VERT_RIB],   # every vertebra
    "rib_bearing": [_RR_VERT_RIB],                   # nested inside it
    "disc_space":  [_RR_DISC],
    "rib_left":    [_RR_RIB_L],
    "rib_right":   [_RR_RIB_R],
    "sacrum":      [_RR_SACRUM],
    "left_hip":    [_RR_HIP_L],
    "right_hip":   [_RR_HIP_R],
    "femur":       [_RR_FEMUR],
    "ignore":      _RR_IGNORE,                       # NOT a region
}
# ignore is deliberately absent: it is not predicted
REGIONS_CLASS_ORDER_RIB = [_RR_VERT_PLAIN, _RR_VERT_RIB, _RR_DISC, _RR_RIB_L,
                           _RR_RIB_R, _RR_SACRUM, _RR_HIP_L, _RR_HIP_R, _RR_FEMUR]

# Base remap: every vertebra starts as NON-rib-bearing and is promoted by
# geometry, because whether a rib touches it is not a lookup.
LABEL_REMAP_RIB_REGIONS_FROM_VERSE: Dict[int, int] = {
    **{i: _RR_VERT_PLAIN for i in range(1, 26)},
    28: _RR_VERT_PLAIN,
    VERSE_SACRUM: _RR_SACRUM,
    29: _RR_SACRUM,
    VERSE_LEFT_HIP: _RR_HIP_L,
    VERSE_RIGHT_HIP: _RR_HIP_R,
    32: _RR_FEMUR, 33: _RR_FEMUR,
    **{i: _RR_RIB_L for i in VERSE_RIB_L},
    **{i: _RR_RIB_R for i in VERSE_RIB_R},
    VERSE_LUMBAR_RIB_L: _RR_RIB_L, VERSE_LUMBAR_RIB_R: _RR_RIB_R,
    VERSE_IGNORE: _RR_IGNORE,
}

_RIB_IDS = VERSE_RIB_L + VERSE_RIB_R + (VERSE_LUMBAR_RIB_L, VERSE_LUMBAR_RIB_R)


def _rib_bearing_vertebrae(arr, zooms, reach_mm=4.0):
    """Which source vertebra ids have a rib articulating with them.

    A LUMBAR RIB COUNTS. Ids 60 and 61 are ribs borne on a lumbar body,
    and treating them as anything else would be deciding the very
    question this scheme exists to leave open: a rib on the first
    lumbar-type vertebra makes that vertebra rib-bearing, which shortens
    the rib-free interval to four, which IS the transitional phenotype.
    Excluding them would hide the phenotype inside the label.

    Proximity rather than true articulation, at a few millimetres. The
    costovertebral joint is a joint, so the two are near but need not
    touch, and a rib head sits within a few millimetres of the body it
    articulates with while the next vertebra down is a whole body height
    away. The margin between those two distances is what makes a
    threshold safe here where it would not be elsewhere.
    """
    from scipy import ndimage

    ribs = np.isin(arr, _RIB_IDS)
    if not ribs.any():
        return set()
    # distance from every voxel to the nearest rib, in millimetres
    d = ndimage.distance_transform_edt(~ribs, sampling=zooms)
    near = d <= reach_mm
    hit = set()
    for v in range(1, 26):
        m = arr == v
        if m.any() and bool((m & near).any()):
            hit.add(v)
    return hit


# ── ONE-SHOT AT THE JUNCTION (--oneshot) ──────────────────────────────
#
# THE EXPERIMENT THIS SCHEME EXISTS TO MAKE POSSIBLE. The claim under
# test is that a sufficiently powerful network can tell a HYPOPLASTIC
# TWELFTH RIB from a LUMBAR RIB in one pass, from morphology alone. The
# existing one-shot target (--no_remap) cannot be used to test it,
# because it has no thoracic classes and no rib classes: the two things
# to be distinguished are both absent, so the model can be neither right
# nor wrong about them.
#
# WHY THE CLAIM IS REASONABLE, which the count-free argument understated.
# Vertebral IDENTITY is not local -- L5 and L6 are the same bone under
# two counts, and no patch can settle which. But THORACIC versus LUMBAR
# is not that kind of question. They are morphologically different bones
# and every discriminating feature is patch-local:
#
#   costal facets on the BODY   thoracic only; the decisive feature. A
#                               T12 body carries one, an L1 body does not,
#                               so the vertebra tells you what the rib is.
#   costotransverse joint       a rib meets the process across a JOINT; a
#                               transverse process is continuous with the
#                               pedicle. Joint versus continuity is visible.
#   process morphology          thoracic processes are short, thick and
#                               club-shaped, pointing posterolaterally;
#                               lumbar are long, flat and blade-like, with
#                               mammillary and accessory processes.
#
# So the discrimination is available to a network with the receptive
# field to see a vertebra together with what is attached to it. Whether
# it LEARNS it at 16 lumbar-rib cases is the empirical question, and the
# point of running it.
#
# THE TWELFTH RIB GETS ITS OWN CLASS, separately from ribs 1-11, because
# the confusable pair is specifically {hypoplastic twelfth rib, lumbar
# rib}. Collapsing all thoracic ribs together would hide the comparison
# inside a class that is easy for other reasons.
#
# WHAT REMAINS OUT OF REACH, and should be said in the talk rather than
# discovered in the results: L5 versus L6 is still a count, and no
# morphology settles it. A one-shot model can be expected to get the
# thoracolumbar junction right and still be undecidable at the
# lumbosacral one. That is a sharper and more interesting claim than
# either "one shot works" or "one shot fails".
_OS = {}
_OS_NAMES = {"background": 0}
_i = 0


def _os_add(name, src_ids):
    global _i
    _i += 1
    _OS_NAMES[name] = _i
    for sid in src_ids:
        _OS[sid] = _i


# thoracic levels around the junction, kept distinct: the costal facet is
# what names the rib, so the vertebra has to be nameable
_os_add("T10", [17])
_os_add("T11", [18])
_os_add("T12", [19])
_os_add("T13", [28])
for _n, _v in (("L1", 20), ("L2", 21), ("L3", 22), ("L4", 23), ("L5", 24), ("L6", 25)):
    _os_add(_n, [_v])
_os_add("sacrum", [26, 29])
# ribs 1-11 pooled; the TWELFTH kept apart, because it is the confusable one;
# the THIRTEENTH (a rib on T13, not a lumbar rib) kept apart for the same reason
_os_add("rib_left", list(VERSE_RIB_L[:11]))
_os_add("rib_right", list(VERSE_RIB_R[:11]))
_os_add("rib12_left", [VERSE_RIB_L[11]])
_os_add("rib12_right", [VERSE_RIB_R[11]])
_os_add("rib13_left", [VERSE_RIB_L[12]])
_os_add("rib13_right", [VERSE_RIB_R[12]])
_os_add("lumbar_rib_left", [VERSE_LUMBAR_RIB_L])
_os_add("lumbar_rib_right", [VERSE_LUMBAR_RIB_R])
_os_add("left_hip", [30])
_os_add("right_hip", [31])
_os_add("femur", [32, 33])
_i += 1
_OS_NAMES["ignore"] = _i
_OS[VERSE_IGNORE] = _i

LABEL_NAMES_ONESHOT = dict(_OS_NAMES)
LABEL_REMAP_ONESHOT_FROM_VERSE: Dict[int, int] = dict(_OS)
# thoracic vertebrae above T10 are outside the question and drop to
# background; they are also outside most of these fields of view
del _OS, _OS_NAMES, _i


def _build_verse_remap_lut(remap: Dict[int, int],
                           max_label: int = _MAX_SOURCE_LABEL) -> np.ndarray:
    """Build a vectorized lookup table for VerSe-native label remapping.

    Returns an int32 array `lut` of length `max_label + 1` where
    `lut[old_label] == new_label`. UNLISTED source ids default to 0
    (BACKGROUND) — NOT to themselves. This is the critical difference
    from a passthrough LUT: VerSe carries many non-training ids
    (thoracic/cervical vertebrae, coccyx, T13, S1, femurs, ribs,
    soft-tissue) that must be dropped to background rather than leaked
    through and rejected by nnU-Net's --verify_dataset_integrity.

    Background (0) stays 0 (it is 0 in the zero-initialized LUT and not
    overridden unless the remap dict says so).
    """
    lut = np.zeros(max_label + 1, dtype=np.int32)  # unlisted -> 0 (background)
    for src, dst in remap.items():
        if not (0 <= src <= max_label):
            raise ValueError(
                f"label remap source {src} outside expected range "
                f"[0, {max_label}]; widen _MAX_SOURCE_LABEL or fix remap dict")
        lut[src] = dst
    return lut


def _remap_and_write_label(src: Path, dst: Path, lut: np.ndarray,
                           derive_disc: bool = False,
                           rib_regions: bool = False) -> None:
    """Read source label NIfTI, apply the remap LUT, write to dst.

    Used when the dataset.json label scheme differs from the source
    files (e.g., merging L6 into last_lumbar to avoid class collision
    in transitional vertebra training). Preserves the source affine
    and header. Output dtype is uint8 (label IDs are small).

    The LUT must accommodate the maximum value in the source file.
    Values exceeding the LUT length are clipped to the LUT max — this
    is defensive against unexpected source labels and mirrors the
    behavior of np.clip + indexing.
    """
    import nibabel as nib

    if dst.exists() or dst.is_symlink():
        dst.unlink()
    img = nib.load(src)
    arr = np.asarray(img.get_fdata(), dtype=np.int32)
    src_max = int(arr.max()) if arr.size else 0
    if src_max >= len(lut):
        log.warning("source label %s contains value %d above LUT max %d; clipping",
                    src.name, src_max, len(lut) - 1)
        arr = np.clip(arr, 0, len(lut) - 1)
    arr_remapped = lut[arr].astype(np.uint8)

    if rib_regions:
        # promote every vertebra that a rib reaches. Done from the SOURCE ids, because
        # after the remap all vertebrae are one value and the association is gone.
        zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
        bearing = _rib_bearing_vertebrae(arr, zooms)
        if bearing:
            arr_remapped[np.isin(arr, list(bearing))] = _RR_VERT_RIB

    if derive_disc:
        # derived from the SOURCE ids, which still carry per-level identity --
        # after the remap every vertebra is class 1 and the seams are gone
        zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
        disc = _derive_disc_space(arr, zooms)
        # The disc deliberately includes a one-voxel shell taken from the two bodies
        # where they abut, so it is NOT gated on background -- gating there would
        # restore exactly the leak it exists to close. It IS gated away from the
        # ignore region: the ignore contract is what makes partial annotation safe,
        # and nothing may supervise a region that was never annotated.
        disc_v = _RR_DISC if rib_regions else _CF_DISC
        ign_v = _RR_IGNORE if rib_regions else _CF_IGNORE
        arr_remapped[disc & (arr_remapped != ign_v)] = disc_v

    out = nib.Nifti1Image(arr_remapped, img.affine, img.header)
    out.set_data_dtype(np.uint8)
    nib.save(out, dst)


def _refine_subtype_for_record_config(patient_subtype: str,
                                       record_config: str) -> str:
    """Adjust the patient-level subtype based on this record's
    annotation config (symmetric anatomical-region demotion).

    Unchanged from prior version. See module docstring for full
    semantics.
    """
    rc = (record_config or "").strip().lower()
    if rc == _PELVIC_ONLY_CONFIG:
        if patient_subtype in _LUMBAR_REGION_SUBTYPES:
            return "normal"
    elif rc == _SPINE_ONLY_CONFIG:
        if patient_subtype in _PELVIS_REGION_SUBTYPES:
            return "normal"
    return patient_subtype


def _load_manifest_records(hf_dir: Path) -> Dict[str, List[Dict]]:
    """Load manifests grouped by split name (train/validation/test).

    A RELEASE TREE (Zenodo/Hugging Face v7 onward) ships ONE manifest.json and no held-out
    test split: every record is training material and the folds in splits_5fold.json carry
    the validation assignment. When none of the three split manifests exist but
    manifest.json does, every record is loaded as "train"."""
    out: Dict[str, List[Dict]] = {}
    single = hf_dir / "manifest.json"
    if single.exists() and not any((hf_dir / fn).exists() for fn in
                                    ("manifest_train.json", "manifest_validation.json",
                                     "manifest_test.json")):
        data = json.loads(single.read_text())
        if isinstance(data, dict):
            data = data.get("records", data.get("cases", list(data.values())))
        recs = [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []
        log.info("release tree: loaded %d records from manifest.json as train; "
                 "validation comes from the folds, no held-out test", len(recs))
        return {"train": recs, "validation": [], "test": []}
    for split, fn in (
        ("train",      "manifest_train.json"),
        ("validation", "manifest_validation.json"),
        ("test",       "manifest_test.json"),
    ):
        p = hf_dir / fn
        if not p.exists():
            log.warning("manifest not found: %s", p)
            out[split] = []
            continue
        data = json.loads(p.read_text())
        if isinstance(data, dict):
            data = data.get("records", data.get("cases", list(data.values())))
        if isinstance(data, list):
            out[split] = [r for r in data if isinstance(r, dict)]
        else:
            out[split] = []
        log.info("loaded %d records from %s", len(out[split]), fn)
    return out


def _link_or_copy(src: Path, dst: Path, use_symlinks: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if use_symlinks:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def _case_id(token: str, record_idx: int = 0) -> str:
    """Generate canonical caseID for nnU-Net. Multiple records per patient
    get suffixed _r1, _r2 etc; first record uses bare token."""
    if record_idx == 0:
        return f"COLONOG_{token}"
    return f"COLONOG_{token}_r{record_idx}"


def _resolve_subtype_for_case(record: Dict,
                                splits_subtypes: Dict[str, str]) -> str:
    """Determine the canonical 6-way subtype for a single record.

    Steps:
      1. Resolve patient-level subtype via splits-derived map (preferred)
         or per-record manifest fields (fallback).
      2. Refine based on this record's `config` to demote lumbar-region
         subtypes to 'normal' for pelvic-only views.
    """
    tok = str(record.get("token") or record.get("patient_token") or "")
    record_config = str(record.get("config", ""))

    # Step 1: patient-level subtype
    if tok in splits_subtypes:
        patient_subtype = splits_subtypes[tok]
    else:
        label = str(record.get("lstv_label", "")).upper()
        pel   = str(record.get("lstv_pelvic", "")).upper()
        vert  = str(record.get("lstv_vertebral", "")).upper()
        n_lumb = int(record.get("n_lumbar_labels", 0) or 0)
        has_l6 = bool(record.get("has_l6", False))

        if label == "AMBIGUOUS":
            patient_subtype = "ambiguous"
        elif (vert == "LUMBARIZATION" and pel == "SACRALIZATION") or \
             (vert == "SACRALIZATION" and pel == "LUMBARIZATION"):
            patient_subtype = "ambiguous"
        elif n_lumb == 6 or has_l6 or label == "LUMBARIZATION":
            patient_subtype = "lumb"
        elif label in ("SEMI_SACRAL", "SEMI_SACRALIZATION") or \
             pel in ("SEMI_SACRAL", "SEMI_SACRALIZATION"):
            patient_subtype = "semisacralization"
        elif vert == "SACRALIZATION" and n_lumb == 4:
            patient_subtype = "sacr_count"
        elif label == "SACRALIZATION" or pel == "SACRALIZATION":
            patient_subtype = "sacralization"
        else:
            patient_subtype = "normal"

    return _refine_subtype_for_record_config(patient_subtype, record_config)


def _build_case_indices(
    manifests: Dict[str, List[Dict]],
    splits_subtypes: Dict[str, str],
    hf_dir: Path,
    images_tr: Optional[Path],
    labels_tr: Optional[Path],
    images_ts: Optional[Path],
    labels_ts: Optional[Path],
    use_symlinks: bool,
    do_link: bool,
    label_remap_lut: Optional[np.ndarray],
    include_configs: Optional[frozenset] = None,
    derive_disc: bool = False,
    rib_regions: bool = False,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Dict],
           List[str], List[str], int, Dict[str, int],
           Dict[str, int], int, Dict[str, int]]:
    """
    Iterate manifests, optionally write CT (link/copy) and label
    (remap-and-write) files into the nnU-Net layout, build indices.

    include_configs: when not None, only records whose `config` (lower-cased)
        is in this set are kept; all others are dropped from EVERY split
        (train/val/test). Used by the fused-only ablation (include_configs
        = {"fused"}). None means keep all records.

    Returns: (case_to_subtype, case_to_token, case_to_attrs,
              train_case_ids, test_case_ids, n_skipped,
              refinement_stats, remap_stats, n_filtered, filtered_by_config)

    refinement_stats: how many cases were demoted by config.
    remap_stats: how many label files were physically rewritten via
                 the remap LUT (vs symlinked when no remap configured).
    n_filtered: how many records were dropped by include_configs.
    filtered_by_config: per-config breakdown of the dropped records.
    """
    case_to_subtype: Dict[str, str] = {}
    case_to_token:   Dict[str, str] = {}
    case_to_attrs:   Dict[str, Dict] = {}
    train_case_ids: List[str] = []
    test_case_ids:  List[str] = []
    n_skipped = 0

    refinement_stats: Dict[str, int] = defaultdict(int)
    remap_stats: Dict[str, int] = defaultdict(int)
    n_filtered = 0
    filtered_by_config: Dict[str, int] = defaultdict(int)

    record_idx_by_token: Dict[str, int] = defaultdict(int)

    for split, recs in manifests.items():
        for rec in recs:
            tok = str(rec.get("token") or rec.get("patient_token") or "")
            if not tok:
                n_skipped += 1
                continue

            # Config filter (fused-only ablation). Drop records whose
            # `config` is not in the allow-list from every split. Done
            # BEFORE the per-token record index is consumed so the kept
            # records keep contiguous _r suffixes.
            if include_configs is not None:
                rec_config = str(rec.get("config", "")).strip().lower()
                if rec_config not in include_configs:
                    n_filtered += 1
                    filtered_by_config[rec_config or "(none)"] += 1
                    continue

            ct_rel    = rec.get("ct_file") or rec.get("ct") or ""
            label_rel = rec.get("label_file") or rec.get("label") or ""
            if not ct_rel or not label_rel:
                log.warning("missing ct/label paths token=%s split=%s; skipping",
                            tok, split)
                n_skipped += 1
                continue

            ct_src    = hf_dir / ct_rel
            label_src = hf_dir / label_rel

            if do_link:
                if not ct_src.exists() or not label_src.exists():
                    log.warning("file not found token=%s ct=%s label=%s; skipping",
                                tok, ct_src.exists(), label_src.exists())
                    n_skipped += 1
                    continue

            r_idx = record_idx_by_token[tok]
            record_idx_by_token[tok] += 1
            case_id = _case_id(tok, r_idx)

            if split == "test":
                test_case_ids.append(case_id)
                if do_link:
                    ct_dst    = images_ts / f"{case_id}_0000.nii.gz"
                    label_dst = labels_ts / f"{case_id}.nii.gz"
            else:
                train_case_ids.append(case_id)
                if do_link:
                    ct_dst    = images_tr / f"{case_id}_0000.nii.gz"
                    label_dst = labels_tr / f"{case_id}.nii.gz"

            if do_link:
                try:
                    # CT image: symlink/copy unchanged.
                    _link_or_copy(ct_src, ct_dst, use_symlinks)
                    # Label: remap-and-write if a remap LUT is
                    # configured; otherwise symlink/copy through.
                    if label_remap_lut is not None:
                        _remap_and_write_label(label_src, label_dst,
                                                label_remap_lut,
                                                derive_disc=derive_disc,
                                                rib_regions=rib_regions)
                        remap_stats["n_remapped"] += 1
                    else:
                        _link_or_copy(label_src, label_dst, use_symlinks)
                        remap_stats["n_passthrough"] += 1
                except Exception as e:
                    log.warning("link/copy/remap failed token=%s: %s", tok, e)
                    n_skipped += 1
                    if split == "test":
                        test_case_ids.pop()
                    else:
                        train_case_ids.pop()
                    continue

            patient_subtype_raw = splits_subtypes.get(tok)
            if patient_subtype_raw is None:
                subtype = _resolve_subtype_for_case(rec, splits_subtypes)
            else:
                rec_config = str(rec.get("config", ""))
                subtype = _refine_subtype_for_record_config(
                    patient_subtype_raw, rec_config)
                if subtype != patient_subtype_raw:
                    key = f"{patient_subtype_raw}->normal (config={rec_config})"
                    refinement_stats[key] += 1

            case_to_subtype[case_id] = subtype
            case_to_token[case_id]   = tok
            case_to_attrs[case_id]   = {
                "n_lumbar_labels": int(rec.get("n_lumbar_labels", 0) or 0),
                "has_l6":          bool(rec.get("has_l6", False)),
                "lstv_label":      str(rec.get("lstv_label", "")),
                "lstv_pelvic":     str(rec.get("lstv_pelvic", "")),
                "lstv_vertebral":  str(rec.get("lstv_vertebral", "")),
                "lstv_class":      int(rec.get("lstv_class", 0) or 0),
                "match_type":      str(rec.get("match_type", "")),
                "config":          str(rec.get("config", "")),
                "split":           split,
                "partial_annotation": bool(rec.get("partial_annotation", False)),
                "patient_subtype_raw": patient_subtype_raw or "(none)",
            }

    return (case_to_subtype, case_to_token, case_to_attrs,
            train_case_ids, test_case_ids, n_skipped,
            dict(refinement_stats), dict(remap_stats),
            n_filtered, dict(filtered_by_config))


def _write_splits_final(
    folds: List[Dict],
    case_to_token: Dict[str, str],
    train_case_ids: List[str],
    out_path: Path,
) -> List[Dict]:
    """Mirror splits_5fold.json folds onto nnU-Net case_ids."""
    token_to_caseids: Dict[str, List[str]] = defaultdict(list)
    train_set = set(train_case_ids)
    for case_id, tok in case_to_token.items():
        if case_id in train_set:
            token_to_caseids[tok].append(case_id)

    nnunet_splits = []
    for fold in folds:
        train_toks = fold.get("train", [])
        val_toks   = fold.get("val", [])
        train_caseids = []
        val_caseids   = []
        for t in train_toks:
            train_caseids.extend(token_to_caseids.get(t, []))
        for t in val_toks:
            val_caseids.extend(token_to_caseids.get(t, []))
        nnunet_splits.append({
            "train": sorted(train_caseids),
            "val":   sorted(val_caseids),
        })

    out_path.write_text(json.dumps(nnunet_splits, indent=2))
    log.info("Wrote splits_final.json: %d folds -> %s", len(nnunet_splits), out_path)
    for i, sf in enumerate(nnunet_splits):
        log.info("  fold %d: train=%d  val=%d", i, len(sf["train"]), len(sf["val"]))
    return nnunet_splits


def _write_lstv_cases(
    case_to_subtype: Dict[str, str],
    case_to_token:   Dict[str, str],
    case_to_attrs:   Dict[str, Dict],
    train_case_ids:  List[str],
    test_case_ids:   List[str],
    splits_schema:   int,
    out_path:        Path,
) -> Dict:
    """Write lstv_cases.json schema v3 (6-way subtypes)."""
    train_set = set(train_case_ids)
    subtype_counts = Counter(case_to_subtype.values())
    lstv_oversample_pool = sorted(
        cid for cid, st in case_to_subtype.items()
        if st != "normal" and cid in train_set
    )

    lstv_cases = {
        "schema_version":      LSTV_CASES_SCHEMA,
        "splits_schema":       splits_schema,
        "subtypes":            list(SUBTYPES),
        "subtype_counts":      dict(subtype_counts),
        "n_train_cases":       len(train_case_ids),
        "n_test_cases":        len(test_case_ids),
        "case_to_subtype":     case_to_subtype,
        "case_to_token":       case_to_token,
        "case_to_attrs":       case_to_attrs,
        "lstv_oversample_pool": lstv_oversample_pool,
    }
    out_path.write_text(json.dumps(lstv_cases, indent=2, default=str))
    log.info("Wrote lstv_cases.json (schema v%d) -> %s", LSTV_CASES_SCHEMA, out_path)
    log.info("  subtype counts:")
    for st in SUBTYPES:
        log.info("    %-20s %d", st, subtype_counts.get(st, 0))
    log.info("  oversample pool size: %d (non-normal training cases)",
             len(lstv_oversample_pool))
    return lstv_cases


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf_dir",   required=True, type=Path)
    p.add_argument("--splits",   required=True, type=Path)
    p.add_argument("--nnunet_raw", required=True, type=Path)
    p.add_argument("--dataset_id",   type=int, default=803,
                    help="Dataset ID. 803 = merged-label variant (default); "
                         "802 = unmerged baseline.")
    p.add_argument("--dataset_name", default="SpineSurgCTFullMerged",
                    help="Dataset name. SpineSurgCTFullMerged (default) for "
                         "the merged-label variant; SpineSurgCTFull for the "
                         "unmerged baseline.")
    p.add_argument("--symlinks", action="store_true",
                    help="Symlink CT images instead of copying. Note: label "
                         "files are ALWAYS physically rewritten (the source is "
                         "VerSe-native and is always remapped to the training "
                         "scheme), so --symlinks applies to CT images only.")
    p.add_argument("--no_remap", action="store_true",
                    help="Select the UNMERGED remap: VerSe-native -> unmerged "
                         "10-class (Dataset802), keeping L5 and L6 as distinct "
                         "classes. Default (flag absent): the MERGED remap, "
                         "VerSe-native -> merged 9-class (Dataset803), collapsing "
                         "L5+L6 into last_lumbar. (Both remaps drop every "
                         "non-training VerSe id to background.)")
    p.add_argument("--include_configs", default="",
                    help="Comma-separated list of record `config` values to "
                         "KEEP (e.g. 'fused'). Records whose config is not "
                         "listed are dropped from ALL splits (train/val/test). "
                         "Empty (default) keeps every record. Use 'fused' for "
                         "the fused-only ablation.")
    p.add_argument("--drop_ignore_label", action="store_true",
                    help="Build the label scheme WITHOUT the ignore class and "
                         "remap any source ignore voxels (10) to background "
                         "(0). For the fused-only ablation, where records are "
                         "fully annotated and carry no ignore voxels. Keeps the "
                         "L6->last_lumbar merge. Incompatible with --no_remap.")
    p.add_argument("--countfree", action="store_true",
                    help="Select the COUNT-FREE scheme: no vertebral identity is "
                         "asked of the network at all. Every vertebra is one "
                         "class, ribs carry a side but no number, and an "
                         "intervertebral-space class is DERIVED at convert time "
                         "so connected components of vertebra-minus-space are "
                         "individual vertebrae. Level naming and transitional "
                         "flagging then happen in code, downstream, where they "
                         "are auditable. Incompatible with --no_remap and with "
                         "--drop_ignore_label.")
    p.add_argument("--oneshot", action="store_true",
                    help="One-shot at the thoracolumbar junction: T10-T12 and L1-L6 as "
                         "distinct classes, ribs 1-11 pooled, the TWELFTH rib kept apart "
                         "from the lumbar rib. This is the target that makes the "
                         "hypoplastic-twelfth-rib versus lumbar-rib discrimination "
                         "measurable at all -- the existing one-shot scheme has neither "
                         "thoracic nor rib classes, so it can be neither right nor wrong "
                         "about them.")
    p.add_argument("--rib_regions", action="store_true",
                    help="The count-free scheme, with the vertebra class split into "
                         "rib-bearing and not, expressed as nnU-Net REGIONS rather than "
                         "as exclusive classes. Rib-bearing status is a local property, "
                         "unlike vertebral identity, so it is the one part of the "
                         "counting problem that can honestly be learned -- and it is the "
                         "part the count is anchored on. Nested regions let the network "
                         "be certain a thing is a vertebra while uncertain whether it "
                         "bears a rib, which exclusive classes cannot express.")
    p.add_argument("--regen_splits_only", action="store_true")
    args = p.parse_args()

    if args.oneshot and (args.no_remap or args.drop_ignore_label
                         or args.countfree or args.rib_regions):
        log.error("--oneshot defines its own label scheme and is incompatible with the "
                  "other scheme flags.")
        raise SystemExit(2)

    if args.rib_regions and (args.no_remap or args.drop_ignore_label
                             or args.countfree):
        log.error("--rib_regions defines its own label scheme and is incompatible with "
                  "--countfree, --no_remap and --drop_ignore_label.")
        raise SystemExit(2)

    if args.countfree and (args.no_remap or args.drop_ignore_label):
        log.error("--countfree is incompatible with --no_remap and "
                  "--drop_ignore_label (it defines its own label scheme).")
        raise SystemExit(2)

    if args.drop_ignore_label and args.no_remap:
        log.error("--drop_ignore_label is incompatible with --no_remap "
                  "(no-ignore is defined only on the merged contiguous scheme).")
        raise SystemExit(2)

    include_configs: Optional[frozenset] = None
    if args.include_configs.strip():
        include_configs = frozenset(
            c.strip().lower() for c in args.include_configs.split(",") if c.strip()
        )
        unknown = include_configs - set(_VALID_CONFIGS)
        if unknown:
            log.warning("--include_configs lists unrecognized config(s) %s; "
                        "valid configs are %s. Records with those configs "
                        "(if any) will be kept; typos silently keep nothing.",
                        sorted(unknown), list(_VALID_CONFIGS))
        log.info("Config filter ACTIVE: keeping only records with config in %s",
                 sorted(include_configs))

    if not args.hf_dir.exists():
        log.error("HF dir not found: %s", args.hf_dir)
        raise SystemExit(1)
    if not args.splits.exists():
        log.error("splits file not found: %s", args.splits)
        raise SystemExit(1)

    splits = json.loads(args.splits.read_text())
    splits_schema = splits.get("schema_version", 0)
    if splits_schema < 6:
        log.warning("splits_5fold.json schema_version=%d (expected >= 6); "
                    "lstv_cases.json subtype assignments may be inconsistent.",
                    splits_schema)

    splits_subtypes: Dict[str, str] = splits.get("patient_subtypes", {})
    log.info("Splits: %d patients, %d with subtype assigned",
             splits.get("n_patients", 0), len(splits_subtypes))

    manifests = _load_manifest_records(args.hf_dir)
    n_total_records = sum(len(v) for v in manifests.values())
    log.info("Manifests: %d total records across train/val/test", n_total_records)

    # Resolve label remap config and build LUT. The source is ALWAYS
    # VerSe-native, so a remap LUT is ALWAYS built (label files are always
    # physically rewritten). Unlisted VerSe ids drop to background — see
    # _build_verse_remap_lut.
    if args.oneshot:
        label_remap_lut = _build_verse_remap_lut(LABEL_REMAP_ONESHOT_FROM_VERSE)
        active_label_names = LABEL_NAMES_ONESHOT
        log.info("Label scheme: ONE-SHOT AT THE JUNCTION, %d classes.",
                 len(LABEL_NAMES_ONESHOT))
        log.info("  -> the twelfth rib is a class of its own, separate from the lumbar "
                 "rib, because that pair IS the discrimination under test. Thoracic "
                 "levels are kept distinct because the costal facet on the body is what "
                 "names the rib attached to it.")
        log.info("  -> L5 versus L6 remains a COUNT and no morphology settles it. Expect "
                 "the junction to be learnable and the lumbosacral one not.")
    elif args.rib_regions:
        label_remap_lut = _build_verse_remap_lut(LABEL_REMAP_RIB_REGIONS_FROM_VERSE)
        active_label_names = LABEL_NAMES_RIB_REGIONS
        log.info("Label scheme: RIB-BEARING REGIONS. Vertebrae split into rib-bearing "
                 "and not, as nested regions: 'vertebra' covers both, 'rib_bearing' "
                 "sits inside it.")
        log.info("  -> a lumbar rib (74/75) makes its vertebra rib-bearing, which is "
                 "the transitional phenotype and must not be hidden inside the label. "
                 "Rib association is computed geometrically from the SOURCE ids, before "
                 "the remap collapses them.")
    elif args.countfree:
        label_remap_lut = _build_verse_remap_lut(LABEL_REMAP_COUNTFREE_FROM_VERSE)
        active_label_names = LABEL_NAMES_COUNTFREE
        log.info("Label remap: VerSe-native -> COUNT-FREE 10-key: %s",
                 dict(LABEL_REMAP_COUNTFREE_FROM_VERSE))
        log.info("  -> every vertebra collapses to one class; ribs keep a side "
                 "but no number; a lumbar rib is a rib. Class 2 (disc_space) is "
                 "DERIVED from the source per-level ids at write time and "
                 "appears in no source file -- it is what makes connected "
                 "components of the vertebra class separable, which is what the "
                 "counting stage needs. The lumbosacral disc is deliberately "
                 "NOT derived: whether one exists is the transitional finding, "
                 "and manufacturing it would answer the question downstream "
                 "code is meant to ask.")
    elif args.no_remap:
        label_remap_lut = _build_verse_remap_lut(LABEL_REMAP_UNMERGED_FROM_VERSE)
        active_label_names = LABEL_NAMES_UNMERGED
        log.info("Label remap: VerSe-native -> unmerged 10-class (Dataset802; "
                 "--no_remap): %s", dict(LABEL_REMAP_UNMERGED_FROM_VERSE))
        log.info("  -> L5 (VerSe 24) and L6 (VerSe 25) kept as distinct "
                 "classes; output has non-contiguous training semantics only "
                 "in the sense that L6 is a real class. Every non-training "
                 "VerSe id drops to background; source labels are PHYSICALLY "
                 "REWRITTEN (--symlinks applies to CT images only).")
    elif args.drop_ignore_label:
        label_remap_lut = _build_verse_remap_lut(LABEL_REMAP_MERGED_NO_IGNORE_FROM_VERSE)
        active_label_names = LABEL_NAMES_NO_IGNORE
        log.info("Label remap: VerSe-native -> merged 9-key NO-IGNORE: %s",
                 dict(LABEL_REMAP_MERGED_NO_IGNORE_FROM_VERSE))
        log.info("  -> 9-key scheme (8 fg + bg), NO ignore class. VerSe ignore "
                 "(255) -> background (0) defensively. Every non-training VerSe "
                 "id drops to background. Intended for the fused-only ablation "
                 "(fused records carry no ignore voxels).")
        if include_configs is None:
            log.warning("--drop_ignore_label set WITHOUT --include_configs: "
                        "any partial-annotation (spine_only/pelvic_native) "
                        "records present will have their ignore voxels folded "
                        "into BACKGROUND, which falsely supervises unannotated "
                        "regions. Pass --include_configs fused for the ablation.")
    else:
        label_remap_lut = _build_verse_remap_lut(LABEL_REMAP_MERGED_FROM_VERSE)
        active_label_names = LABEL_NAMES
        log.info("Label remap: VerSe-native -> merged 9-class (Dataset803): %s",
                 dict(LABEL_REMAP_MERGED_FROM_VERSE))
        log.info("  -> VerSe L5+L6 collapsed into last_lumbar (5); every "
                 "non-training VerSe id drops to background; source labels are "
                 "PHYSICALLY REWRITTEN (--symlinks applies to CT images only).")

    ds_dir = args.nnunet_raw / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"

    do_link = not args.regen_splits_only
    if do_link:
        images_tr = ds_dir / "imagesTr"
        labels_tr = ds_dir / "labelsTr"
        images_ts = ds_dir / "imagesTs"
        labels_ts = ds_dir / "labelsTs"
        for d in (images_tr, labels_tr, images_ts, labels_ts):
            d.mkdir(parents=True, exist_ok=True)
        log.info("Mode: full build (link/copy + dataset.json + splits + lstv_cases)")
        log.info("Output: %s", ds_dir)
    else:
        if not ds_dir.exists():
            log.error("--regen_splits_only requires existing dataset dir: %s", ds_dir)
            log.error("       Run without --regen_splits_only first to build the dataset.")
            raise SystemExit(1)
        images_tr = labels_tr = images_ts = labels_ts = None
        log.info("Mode: regen_splits_only (refresh splits_final.json + lstv_cases.json only)")
        log.info("Dataset dir: %s (existing, not modified)", ds_dir)

    (case_to_subtype, case_to_token, case_to_attrs,
     train_case_ids, test_case_ids, n_skipped,
     refinement_stats, remap_stats,
     n_filtered, filtered_by_config) = _build_case_indices(
        manifests, splits_subtypes, args.hf_dir,
        images_tr, labels_tr, images_ts, labels_ts,
        use_symlinks=args.symlinks, do_link=do_link,
        label_remap_lut=label_remap_lut,
        derive_disc=args.countfree or args.rib_regions,
        rib_regions=args.rib_regions,
        include_configs=include_configs,
    )

    if include_configs is not None:
        log.info("Config filter dropped %d record(s): %s",
                 n_filtered, filtered_by_config or "{}")
        log.info("  kept configs: %s", sorted(include_configs))

    if do_link:
        log.info("Linked/copied: train=%d  test=%d  skipped=%d",
                 len(train_case_ids), len(test_case_ids), n_skipped)
        if remap_stats:
            log.info("Label file disposition:")
            for k, n in sorted(remap_stats.items()):
                log.info("  %-20s %d", k, n)
    else:
        log.info("Indexed: train=%d  test=%d  skipped=%d  (no link/copy in regen mode)",
                 len(train_case_ids), len(test_case_ids), n_skipped)

    if refinement_stats:
        log.info("Per-record subtype refinements (anatomical-region demotions):")
        for transition, n in sorted(refinement_stats.items()):
            log.info("  %-50s %d", transition, n)
    else:
        log.info("Per-record subtype refinements: none "
                 "(no spine_only / pelvic_native records of LSTV patients, "
                 "or `config` field missing)")

    # ── dataset.json (skipped in regen mode) ───────────────────────────────
    if do_link:
        if args.drop_ignore_label:
            description = (
                "CTSpinoPelvic1K — FUSED-ONLY ablation, 9-key spine + pelvis "
                "CT (8 fg + bg) with merged last_lumbar (L5+L6) and NO ignore "
                "label. Training/test pool restricted to fully-annotated "
                "fused records; the partial-annotation ignore protocol is "
                "disabled. L6 source labels remapped to last_lumbar (5); "
                "sacrum/hips shifted down for contiguous label values. "
                "Reviewer ablation: fused-only vs all-cases, evaluated on the "
                "fused test split. Recover individual L5/L6 via instance "
                "post-processing or VERIDAH (Möller 2026)."
            )
            release = (
                "May 2026 — fused-only ablation (no ignore protocol) / "
                "schema v6 splits / v3 lstv_cases / L6 merged into "
                "last_lumbar following Möller 2026 (VERIDAH §2.2) / "
                "contiguous label IDs"
            )
        elif not args.no_remap:
            description = (
                "CTSpinoPelvic1K — fused 9-class spine + pelvis CT (8 fg + bg) "
                "with merged last_lumbar (L5+L6) and ignore label (9) for "
                "partial annotations. VerSe-native source labels remapped to "
                "the merged 9-class scheme at convert time (VerSe L5/L6 -> "
                "last_lumbar 5, sacrum 26 -> 6, hips 30/31 -> 7/8, ignore "
                "255 -> 9); every non-training VerSe id -> background. "
                "Recover individual L5/L6 via instance post-processing or "
                "VERIDAH (Möller 2026)."
            )
            release = (
                "May 2026 — schema v6 splits / v3 lstv_cases / "
                "partial-annotation ignore label / per-record subtype "
                "refinement / VerSe-native source remapped to merged "
                "9-class following Möller 2026 (VERIDAH §2.2) / "
                "contiguous label IDs"
            )
        else:
            description = (
                "CTSpinoPelvic1K — fused 10-class spine + pelvis CT with "
                "ignore label (10) for partial annotations (UNMERGED VARIANT; "
                "L5 and L6 distinct). VerSe-native source labels remapped to "
                "the unmerged 10-class scheme at convert time (VerSe L5 -> 5, "
                "L6 -> 6, sacrum 26 -> 7, hips 30/31 -> 8/9, ignore 255 -> 10); "
                "every non-training VerSe id -> background."
            )
            release = (
                "May 2026 — schema v6 splits / v3 lstv_cases / "
                "partial-annotation ignore label / per-record subtype "
                "refinement / VerSe-native source remapped to unmerged "
                "10-class"
            )
        dataset_json = {
            "channel_names": {"0": "CT"},
            "labels":        active_label_names,
            "numTraining":   len(train_case_ids),
            "file_ending":   ".nii.gz",
            "name":          args.dataset_name,
            "description":   description,
            "reference":     "CTSpine1K + CTPelvic1K, COLONOG cohort",
            "release":       release,
        }
        if args.rib_regions:
            # regions_class_order paints the regions back into an integer map IN ORDER,
            # later entries overwriting earlier, so the encompassing 'vertebra' must
            # come before the 'rib_bearing' nested inside it. The ignore label is
            # deliberately NOT listed: it is not predicted.
            dataset_json["regions_class_order"] = REGIONS_CLASS_ORDER_RIB
        # sort_keys=False is load-bearing, not style. nnU-Net reads the labels dict IN
        # ORDER for region-based training, and alphabetical sorting would put
        # 'rib_bearing' after 'sacrum' and before 'vertebra' -- silently painting the
        # encompassing region over the substructure it contains and erasing it.
        (ds_dir / "dataset.json").write_text(
            json.dumps(dataset_json, indent=2, sort_keys=False))
        n_fg = sum(1 for k in active_label_names
                   if k not in ("background", "ignore"))
        has_ignore = "ignore" in active_label_names
        log.info("Wrote dataset.json with %d label keys (%d foreground + bg%s)",
                 len(active_label_names), n_fg,
                 " + ignore" if has_ignore else "; NO ignore class")

    # ── splits_final.json + lstv_cases.json (always written) ───────────────
    _write_splits_final(
        splits.get("folds", []),
        case_to_token, train_case_ids,
        ds_dir / "splits_final.json",
    )
    _write_lstv_cases(
        case_to_subtype, case_to_token, case_to_attrs,
        train_case_ids, test_case_ids, splits_schema,
        ds_dir / "lstv_cases.json",
    )

    log.info("=" * 70)
    log.info("Done. Dataset at: %s", ds_dir)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
