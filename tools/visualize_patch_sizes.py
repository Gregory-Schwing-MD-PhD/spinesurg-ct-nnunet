#!/usr/bin/env python3
"""
visualize_patch_sizes.py — overlay candidate nnU-Net patches on a CT.

Anatomical orientation
----------------------
Volumes are reoriented to PIR voxel space on load (axis 0 = P, A->P;
axis 1 = I, S->I; axis 2 = R, L->R). Sagittal slices are transposed
for head-up display. Anatomic-orientation labels (S/I/A/P/L/R) drop
in each panel's corners.

Coordinate system
-----------------
Everything inside the figure — image, label overlays, rectangles,
axis ticks — lives in VOXEL-PIXEL coordinates. Mm labels are placed
on the axes by relabeling tick positions; we don't use imshow's
`extent=`. This keeps patch overlays geometrically consistent with
the underlying anatomy regardless of imshow origin/extent
interactions.

Slice selection
---------------
Slices pass through the sacrum centroid drawn from a paired label
file. Priority:
  1. Explicit --label path
  2. Sibling labels/<stem>_label.nii.gz next to the CT
  3. Bone-density (HU > 200) centroid in the lower SI half
  4. Volume center

Sacrum (label 7) is preferred; falls back to L5 (5), then any
foreground voxel.

Mask rendering
--------------
When a label NIfTI is found, segmentation masks are rendered as
semi-transparent color overlays on each panel using the
SpineSurg-CT 10-class color scheme (matches export_hf.py and
visualize_qc.py). Mask classes present in the slice get a separate
legend below the figure.

Overlapping patches
-------------------
nnU-Net plans often produce identical patch sizes once the planner
saturates a memory budget. To keep all candidates visible:
  - line widths grade thin → thick from smallest to largest patch
  - identical-extent boxes get a small ~3 px diagonal offset so
    edges fan out instead of overlapping pixel-perfect
All candidates use solid lines (no linestyle distinction).

USAGE
-----
  python visualize_patch_sizes.py \\
      --ct    data/hf_export/ct/0001_unknown_pelvic_ct.nii.gz \\
      --plans nnunet/preprocessed/Dataset802.../nnUNetResEncUNetPlans_60G.json \\
      --plans nnunet/preprocessed/Dataset802.../nnUNetResEncUNetPlans_100G.json \\
      --out   patch_size_comparison.png

  # Manual patches (no planner output needed)
  python visualize_patch_sizes.py \\
      --ct case.nii.gz \\
      --patch_spec "small:96,128,128" \\
      --patch_spec "large:160,192,192" \\
      --out manual_patches.png

  # Skip mask overlays even when a label file exists
  python visualize_patch_sizes.py \\
      --ct case.nii.gz --no_mask_overlay --plans plans.json --out fig.png

Patch spec format: --patch_spec "<label>:<patch_z>,<patch_y>,<patch_x>"
where (z, y, x) are voxel counts at the case's NATIVE spacing.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("patch_viz")

_PALETTE = [
    "#e41a1c", "#377eb8", "#4daf4a",
    "#ff7f00", "#984ea3", "#a65628",
]
_LINEWIDTHS = [1.6, 2.0, 2.4, 2.8, 3.2]

_HU_MIN, _HU_MAX = -200, 800
_BONE_HU = 200.0
_SACRUM_LABEL = 7
_L5_LABEL     = 5

# 10-class label scheme — matches export_hf.py and visualize_qc.py.
# Format: class_id -> (display_name, RGBA tuple in [0, 1])
CLASS_NAMES: Dict[int, str] = {
    1: "L1", 2: "L2", 3: "L3", 4: "L4", 5: "L5", 6: "L6",
    7: "sacrum", 8: "left_hip", 9: "right_hip",
}

# Use the same colormap as visualize_qc.py / export_hf.py for consistency.
# Alpha is bumped slightly higher than QC since this figure is publication-grade.
_SEG_COLORS: Dict[int, Tuple[float, float, float, float]] = {
    1: (0.15, 0.40, 0.80, 0.55),
    2: (0.25, 0.55, 0.85, 0.55),
    3: (0.35, 0.65, 0.90, 0.55),
    4: (0.45, 0.75, 0.92, 0.55),
    5: (0.10, 0.80, 0.85, 0.55),
    6: (0.75, 0.85, 0.20, 0.65),   # L6 — slightly opaque since it's the headline class
    7: (0.85, 0.15, 0.15, 0.55),
    8: (0.95, 0.50, 0.10, 0.55),
    9: (0.95, 0.80, 0.05, 0.55),
}


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class PatchSpec:
    label:       str
    patch_voxel: Tuple[int, int, int]   # (z, y, x) at the plan's target spacing
    spacing_mm:  Tuple[float, float, float]
    color:       str
    linewidth:   float = 2.0

    @property
    def patch_mm(self) -> Tuple[float, float, float]:
        return tuple(self.patch_voxel[i] * self.spacing_mm[i] for i in range(3))

    def __str__(self) -> str:
        pmm = self.patch_mm
        return (f"{self.label}  patch={list(self.patch_voxel)} vox  "
                f"= {pmm[0]:.0f}x{pmm[1]:.0f}x{pmm[2]:.0f} mm")


# =============================================================================
# Orientation: PIR canonicalization
# =============================================================================

def _load_pir(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI and reorient to PIR voxel space; return (data, affine)."""
    import nibabel as nib
    from nibabel.orientations import (
        axcodes2ornt, ornt_transform, apply_orientation, inv_ornt_aff,
    )
    img      = nib.load(str(path))
    src_ornt = nib.io_orientation(img.affine)
    dst_ornt = axcodes2ornt(("P", "I", "R"))
    xfm      = ornt_transform(src_ornt, dst_ornt)
    data     = apply_orientation(img.get_fdata(dtype=np.float32), xfm).squeeze()
    new_aff  = img.affine @ inv_ornt_aff(xfm, img.shape[:3])
    return data, new_aff


def _hu_window(arr: np.ndarray) -> np.ndarray:
    return np.clip((arr - _HU_MIN) / (_HU_MAX - _HU_MIN), 0.0, 1.0)


# =============================================================================
# Slice picker — uses sibling label NIfTI when available
# =============================================================================

def _resolve_label_path(ct_path: Path, explicit: Optional[Path]) -> Optional[Path]:
    if explicit is not None:
        if explicit.exists():
            return explicit
        log.warning("Explicit --label not found: %s (falling back to auto)", explicit)
    name = ct_path.name
    if name.endswith("_ct.nii.gz"):
        stem = name[: -len("_ct.nii.gz")]
        candidate = ct_path.parent.parent / "labels" / f"{stem}_label.nii.gz"
        if candidate.exists():
            return candidate
    if name.endswith(".nii.gz"):
        stem = name[: -len(".nii.gz")]
        if stem.endswith("_ct"):
            stem = stem[: -len("_ct")]
            candidate = ct_path.parent.parent / "labels" / f"{stem}_label.nii.gz"
            if candidate.exists():
                return candidate
    return None


def _centroid_from_label(label: np.ndarray) -> Optional[Tuple[int, int, int]]:
    for cls in (_SACRUM_LABEL, _L5_LABEL):
        vox = np.argwhere(label == cls)
        if len(vox) >= 100:
            c = vox.mean(axis=0)
            return (int(round(c[0])), int(round(c[1])), int(round(c[2])))
    fg = np.argwhere(label > 0)
    if len(fg) >= 100:
        c = fg.mean(axis=0)
        return (int(round(c[0])), int(round(c[1])), int(round(c[2])))
    return None


def _centroid_from_bone_hu(ct: np.ndarray) -> Optional[Tuple[int, int, int]]:
    si_axis = 1
    si_mid  = ct.shape[si_axis] // 2
    slab = [slice(None)] * 3
    slab[si_axis] = slice(si_mid, None)
    bone = (ct[tuple(slab)] > _BONE_HU)
    if bone.sum() < 1000:
        return None
    vox = np.argwhere(bone)
    vox[:, si_axis] += si_mid
    c = vox.mean(axis=0)
    return (int(round(c[0])), int(round(c[1])), int(round(c[2])))


def load_label_and_pick_centroid(
        ct: np.ndarray, ct_path: Path, explicit_label: Optional[Path]
        ) -> Tuple[Optional[np.ndarray], Tuple[int, int, int], str, Optional[Path]]:
    """
    Returns (label_array_or_None, centroid_pir, source_description, label_path).

    The label array is returned alongside the centroid so render_figure can
    overlay it without re-loading. Returned label is in PIR voxel space,
    int16.
    """
    label_path = _resolve_label_path(ct_path, explicit_label)
    if label_path is not None:
        try:
            log.info("Loading label: %s", label_path)
            label_pir, _ = _load_pir(label_path)
            label_pir = label_pir.astype(np.int16)
            cen = _centroid_from_label(label_pir)
            if cen is not None:
                src = f"sacrum/L5 ({label_path.name})"
                log.info("  PIR voxel centroid (P, I, R) = %s  (from label)", cen)
                return label_pir, cen, src, label_path
            log.warning("  label has no sacrum/L5/fg voxels; falling through")
        except Exception as e:
            log.warning("  label load failed: %s", e)

    cen = _centroid_from_bone_hu(ct)
    if cen is not None:
        log.info("  PIR voxel centroid (P, I, R) = %s  (bone HU heuristic)", cen)
        return None, cen, "bone HU heuristic", None

    cen = (ct.shape[0] // 2, ct.shape[1] // 2, ct.shape[2] // 2)
    log.info("  PIR voxel centroid (P, I, R) = %s  (volume center fallback)", cen)
    return None, cen, "volume center (fallback)", None


# =============================================================================
# Plans / patch parsing
# =============================================================================

def parse_plans_file(plans_path: Path, color: str,
                      config: str = "3d_fullres") -> Optional[PatchSpec]:
    data = json.loads(plans_path.read_text())
    cfg  = data.get("configurations", {}).get(config, {})
    patch   = cfg.get("patch_size")
    spacing = cfg.get("spacing") or data.get("original_median_spacing_after_transp")
    if patch is None or spacing is None:
        log.warning("Skipping %s: no patch_size/spacing in %s", plans_path.name, config)
        return None
    stem = plans_path.stem
    label = stem
    for marker in ("Plans_", "plans_"):
        if marker in stem:
            label = stem.split(marker, 1)[1]
            break
    return PatchSpec(
        label=label,
        patch_voxel=tuple(int(v) for v in patch),
        spacing_mm=tuple(float(v) for v in spacing),
        color=color,
    )


def parse_patch_spec(spec: str, color: str,
                      native_spacing: Tuple[float, float, float]) -> PatchSpec:
    if ":" not in spec:
        raise ValueError(f"--patch_spec must be 'label:z,y,x', got: {spec}")
    label, dims = spec.split(":", 1)
    parts = [p.strip() for p in dims.split(",")]
    if len(parts) != 3:
        raise ValueError(f"--patch_spec dims must be 3 ints, got: {spec}")
    return PatchSpec(
        label=label.strip(),
        patch_voxel=tuple(int(p) for p in parts),
        spacing_mm=native_spacing,
        color=color,
    )


def assign_widths_and_offsets(patch_specs: List[PatchSpec]) -> List[Tuple[float, float]]:
    """
    Assign linewidth by ascending physical volume so smaller patches are
    thinnest. Detect identical-extent groups and return a small diagonal
    offset for each so identical boxes fan out a few pixels.

    Returns list of (offset_x, offset_y) in PIXEL units.
    """
    order = sorted(
        range(len(patch_specs)),
        key=lambda i: (
            patch_specs[i].patch_mm[0] *
            patch_specs[i].patch_mm[1] *
            patch_specs[i].patch_mm[2],
            i,
        )
    )
    for rank, idx in enumerate(order):
        patch_specs[idx].linewidth = _LINEWIDTHS[rank % len(_LINEWIDTHS)]

    offsets = [(0.0, 0.0)] * len(patch_specs)
    seen_groups: Dict[tuple, List[int]] = {}
    for i, ps in enumerate(patch_specs):
        key = tuple(round(x, 3) for x in ps.patch_mm)
        seen_groups.setdefault(key, []).append(i)
    for key, indices in seen_groups.items():
        if len(indices) <= 1:
            continue
        for k, i in enumerate(indices):
            offsets[i] = (k * 4.0, k * 4.0)
    return offsets


# =============================================================================
# Plotting helpers
# =============================================================================

def _slice_to_display(arr3d: np.ndarray, view_axis: int, idx: int) -> np.ndarray:
    """Pull a 2D slice for a given view in display-pixel layout."""
    s = [slice(None)] * 3
    s[view_axis] = idx
    arr2d = arr3d[tuple(s)]
    if view_axis == 2:
        arr2d = arr2d.T
    return arr2d


def _display_axes_for_view(view_axis: int) -> Tuple[int, int]:
    """Return (vertical_voxel_axis, horizontal_voxel_axis) for the displayed image."""
    if view_axis == 1:   # axial
        return 0, 2
    if view_axis == 0:   # coronal
        return 1, 2
    if view_axis == 2:   # sagittal (transposed)
        return 1, 0
    raise ValueError(view_axis)


def _label_to_rgba(label_2d: np.ndarray) -> np.ndarray:
    """Convert a 2D label slice to an RGBA image suitable for imshow overlay."""
    h, w = label_2d.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    for cls, (r, g, b, a) in _SEG_COLORS.items():
        mask = (label_2d == cls)
        if not mask.any():
            continue
        rgba[mask, 0] = r
        rgba[mask, 1] = g
        rgba[mask, 2] = b
        rgba[mask, 3] = a
    return rgba


def _set_mm_ticks(ax, axis_pixel_count: int, spacing_along_axis: float,
                  which: str) -> None:
    """Place ticks at integer voxel positions and label them in mm."""
    extent_mm = axis_pixel_count * spacing_along_axis
    if   extent_mm >= 800: step_mm = 200
    elif extent_mm >= 400: step_mm = 100
    elif extent_mm >= 200: step_mm = 50
    else:                  step_mm = 25
    tick_mm     = np.arange(0, extent_mm + 1, step_mm)
    tick_pixels = tick_mm / spacing_along_axis
    valid = (tick_pixels >= 0) & (tick_pixels <= axis_pixel_count - 1)
    tick_pixels = tick_pixels[valid]
    tick_mm     = tick_mm[valid]
    if which == "x":
        ax.set_xticks(tick_pixels)
        ax.set_xticklabels([f"{v:.0f}" for v in tick_mm])
    else:
        ax.set_yticks(tick_pixels)
        ax.set_yticklabels([f"{v:.0f}" for v in tick_mm])


# =============================================================================
# Main figure
# =============================================================================

def render_figure(ct: np.ndarray, affine: np.ndarray,
                   patch_specs: List[PatchSpec],
                   centroid_pir: Tuple[int, int, int],
                   out_path: Path,
                   label_pir: Optional[np.ndarray] = None,
                   ct_filename: str = "",
                   centroid_source: str = "") -> None:
    """
    Three orthogonal panels in PIR voxel space. Each panel:
      - shows the CT slice in HU window
      - overlays segmentation labels (if provided) as semi-transparent colors
      - overlays candidate patch rectangles (solid lines, varying thickness)
      - draws a yellow crosshair at the slicing centroid
      - labels the axes in mm via tick relabeling
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle, Patch

    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    pir_to_patch = {1: 0, 0: 1, 2: 2}

    pixel_offsets = assign_widths_and_offsets(patch_specs)

    p_idx, i_idx, r_idx = centroid_pir
    views = [
        ("Axial",    1, i_idx),
        ("Coronal",  0, p_idx),
        ("Sagittal", 2, r_idx),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 7), constrained_layout=False)

    # Track which classes are visible somewhere in the figure for the legend
    visible_classes: set = set()

    for col, (view_name, view_axis, idx) in enumerate(views):
        ax = axes[col]
        idx = int(np.clip(idx, 0, ct.shape[view_axis] - 1))

        # Background CT
        ct_disp = _slice_to_display(ct, view_axis, idx)
        ax.imshow(_hu_window(ct_disp), cmap="gray",
                  origin="upper", aspect="equal", interpolation="nearest")

        # Mask overlay (if available)
        if label_pir is not None:
            lbl_disp = _slice_to_display(label_pir, view_axis, idx)
            visible_classes |= {int(c) for c in np.unique(lbl_disp) if c > 0}
            rgba = _label_to_rgba(lbl_disp)
            if rgba[..., 3].max() > 0:
                ax.imshow(rgba, origin="upper", aspect="equal", interpolation="nearest")

        # Determine which PIR voxel axes correspond to display rows/cols
        v_pir, h_pir = _display_axes_for_view(view_axis)
        cx_px = centroid_pir[h_pir]
        cy_px = centroid_pir[v_pir]

        # Patch rectangles (solid lines, varying widths, small offsets for duplicates)
        for ps, off_px in zip(patch_specs, pixel_offsets):
            patch_h_mm = ps.patch_mm[pir_to_patch[h_pir]]
            patch_v_mm = ps.patch_mm[pir_to_patch[v_pir]]
            patch_h_px = patch_h_mm / spacing[h_pir]
            patch_v_px = patch_v_mm / spacing[v_pir]
            x0 = cx_px - patch_h_px / 2 + off_px[0]
            y0 = cy_px - patch_v_px / 2 + off_px[1]
            ax.add_patch(Rectangle(
                (x0, y0), patch_h_px, patch_v_px,
                linewidth=ps.linewidth, edgecolor=ps.color,
                linestyle="solid", facecolor="none",
            ))

        # Crosshair at centroid
        crosshair_len_px = max(15, int(min(ct_disp.shape) * 0.04))
        ax.plot([cx_px - crosshair_len_px, cx_px + crosshair_len_px],
                [cy_px, cy_px], color="yellow", linewidth=1.0, alpha=0.85)
        ax.plot([cx_px, cx_px],
                [cy_px - crosshair_len_px, cy_px + crosshair_len_px],
                color="yellow", linewidth=1.0, alpha=0.85)

        _set_mm_ticks(ax, ct_disp.shape[1], spacing[h_pir], which="x")
        _set_mm_ticks(ax, ct_disp.shape[0], spacing[v_pir], which="y")

        # Anatomic-orientation labels
        def _orient_label(x, y, text, ha, va):
            ax.text(x, y, text, transform=ax.transAxes, color="white",
                    fontsize=12, fontweight="bold", va=va, ha=ha,
                    bbox=dict(boxstyle="round,pad=0.18", fc="black", alpha=0.65))
        if view_name == "Axial":
            _orient_label(0.50, 0.98, "A", "center", "top")
            _orient_label(0.50, 0.02, "P", "center", "bottom")
            _orient_label(0.98, 0.50, "R", "right",  "center")
            _orient_label(0.02, 0.50, "L", "left",   "center")
        elif view_name == "Coronal":
            _orient_label(0.50, 0.98, "S", "center", "top")
            _orient_label(0.50, 0.02, "I", "center", "bottom")
            _orient_label(0.98, 0.50, "R", "right",  "center")
            _orient_label(0.02, 0.50, "L", "left",   "center")
        else:
            _orient_label(0.50, 0.98, "S", "center", "top")
            _orient_label(0.50, 0.02, "I", "center", "bottom")
            _orient_label(0.98, 0.50, "P", "right",  "center")
            _orient_label(0.02, 0.50, "A", "left",   "center")

        ax.set_title(view_name, fontsize=14, fontweight="bold")
        ax.set_xlabel("mm", fontsize=11)
        if col == 0:
            ax.set_ylabel("mm", fontsize=11)
        ax.tick_params(labelsize=10)

    # ── Legends ─────────────────────────────────────────────────────────────
    # Two stacked legends below the panels:
    #   1) patch boxes (color-coded line samples)
    #   2) mask classes (color-coded patches), only if any are visible
    patch_legend_handles = [
        Line2D([0], [0], color=ps.color, linewidth=ps.linewidth, linestyle="solid",
               label=f"{ps.label}  patch=[{ps.patch_voxel[0]},{ps.patch_voxel[1]},{ps.patch_voxel[2]}] vox  "
                     f"= {ps.patch_mm[0]:.0f}×{ps.patch_mm[1]:.0f}×{ps.patch_mm[2]:.0f} mm  "
                     f"(target {ps.spacing_mm[0]:.2f}×{ps.spacing_mm[1]:.2f}×{ps.spacing_mm[2]:.2f} mm)")
        for ps in patch_specs
    ]
    leg_patches = fig.legend(
        handles=patch_legend_handles, loc="lower center",
        ncol=1, fontsize=11, frameon=True, title="Candidate patches",
        title_fontsize=11, bbox_to_anchor=(0.5, 0.02),
    )
    leg_patches.get_frame().set_linewidth(0.8)
    leg_patches._legend_box.align = "left"

    if visible_classes:
        mask_legend_handles = [
            Patch(facecolor=_SEG_COLORS[c][:3], edgecolor="none",
                  alpha=_SEG_COLORS[c][3], label=CLASS_NAMES[c])
            for c in sorted(visible_classes) if c in CLASS_NAMES
        ]
        # Add the class legend separately so it doesn't merge with patches
        leg_classes = fig.legend(
            handles=mask_legend_handles, loc="lower center",
            ncol=min(9, len(mask_legend_handles)), fontsize=10, frameon=True,
            title="Segmentation classes (slice)", title_fontsize=10,
            bbox_to_anchor=(0.5, -0.06),
        )
        leg_classes.get_frame().set_linewidth(0.8)
        # matplotlib drops earlier legends when you add a new one via fig.legend;
        # re-add the patch legend explicitly so both render.
        fig.add_artist(leg_patches)

    # ── Title ───────────────────────────────────────────────────────────────
    spacing_str = "x".join(f"{s:.2f}" for s in spacing)
    fov_mm = tuple(ct.shape[i] * spacing[i] for i in range(3))
    fov_str = "x".join(f"{m:.0f}" for m in fov_mm)
    title = "Patch-size comparison"
    if ct_filename:
        title += f" on {ct_filename}"
    subtitle = (f"Volume: {ct.shape[0]}×{ct.shape[1]}×{ct.shape[2]} voxels "
                f"@ {spacing_str} mm = {fov_str} mm FOV (PIR)")
    if centroid_source:
        subtitle += f"   |   slice centroid: {centroid_source}"
    fig.suptitle(f"{title}\n{subtitle}", fontsize=15, fontweight="bold", y=0.99)

    # Reserve enough room at the bottom for both legends if both present
    bottom_margin = 0.30 if visible_classes else 0.22
    fig.subplots_adjust(top=0.86, bottom=bottom_margin,
                         left=0.04, right=0.99, wspace=0.18)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Overlay candidate nnU-Net patches on a CT (PIR-oriented).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--ct", required=True, type=Path,
                    help="CT NIfTI (any orientation; reoriented to PIR).")
    ap.add_argument("--label", default=None, type=Path,
                    help="Explicit label NIfTI for slice-centroid + mask overlay. "
                         "If omitted, auto-detects sibling labels/<stem>_label.nii.gz.")
    ap.add_argument("--no_mask_overlay", action="store_true",
                    help="Skip rendering label masks even when a label file is found.")
    ap.add_argument("--plans", action="append", default=[], type=Path,
                    help="Plans.json file (repeatable).")
    ap.add_argument("--patch_spec", action="append", default=[], type=str,
                    help="Manual patch as 'label:z,y,x' (voxels at native spacing). Repeatable.")
    ap.add_argument("--config", default="3d_fullres", type=str,
                    help="Configuration key in plans.json (default: 3d_fullres).")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    if not args.ct.exists():
        log.error("CT not found: %s", args.ct)
        return 1
    if not args.plans and not args.patch_spec:
        log.error("Provide at least one --plans or --patch_spec")
        return 1

    log.info("Loading CT (PIR-canonicalized): %s", args.ct)
    ct, affine = _load_pir(args.ct)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    log.info("  shape=%s  spacing=%s mm", ct.shape, spacing.tolist())

    label_pir, centroid_pir, centroid_source, _ = load_label_and_pick_centroid(
        ct, args.ct, args.label
    )
    if args.no_mask_overlay:
        log.info("Mask overlay disabled by --no_mask_overlay")
        label_pir = None

    patch_specs: List[PatchSpec] = []
    color_iter = iter(_PALETTE)
    next_color = lambda: next(color_iter, _PALETTE[-1])

    for plans_path in args.plans:
        ps = parse_plans_file(plans_path, color=next_color(), config=args.config)
        if ps is not None:
            patch_specs.append(ps)

    native_spacing = tuple(float(s) for s in spacing)
    for spec in args.patch_spec:
        try:
            ps = parse_patch_spec(spec, color=next_color(),
                                   native_spacing=native_spacing)
            patch_specs.append(ps)
        except ValueError as e:
            log.error("%s", e)
            return 1

    if not patch_specs:
        log.error("No usable patch specs after parsing.")
        return 1

    log.info("Rendering %d patch spec(s) on %s:", len(patch_specs), args.ct.name)
    for ps in patch_specs:
        log.info("  %s", str(ps))

    render_figure(
        ct, affine, patch_specs, centroid_pir, args.out,
        label_pir=label_pir,
        ct_filename=args.ct.name,
        centroid_source=centroid_source,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
