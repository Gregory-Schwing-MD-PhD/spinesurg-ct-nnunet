#!/usr/bin/env python3
"""
SpineSurg-CT -- patch-size visualizer
tools/visualize_patch_sizes.py

Overlays one or more candidate patch sizes on mid-slices of a real CT to
help you logically pick a GPU memory target. Patch size is the main thing
that changes with -gpu_memory_target: larger target -> larger patch ->
more anatomic context per training sample, at the cost of VRAM.

Two input modes
---------------
1. Read nnU-Net plans.json files (the outputs of `plan_experiment` /
   `plan_and_preprocess`). Patch size and target spacing are pulled from
   each plans file's 3d_fullres configuration, so you see EXACTLY what
   the planner chose for each GPU memory target:

       python tools/visualize_patch_sizes.py \\
           --ct      data/hf_export/ct/CASE_001.nii.gz \\
           --plans   nnunet/preprocessed/Dataset802.../nnUNetResEncUNetPlans_80G.json \\
           --plans   nnunet/preprocessed/Dataset802.../nnUNetResEncUNetPlans_100G.json \\
           --plans   nnunet/preprocessed/Dataset802.../nnUNetResEncUNetPlans_140G.json \\
           --out     patch_size_comparison.png

2. Explicit patch sizes (skip the plans files; useful for what-if
   exploration without running the planner):

       python tools/visualize_patch_sizes.py \\
           --ct         data/hf_export/ct/CASE_001.nii.gz \\
           --patch_spec "80G:112,128,128" \\
           --patch_spec "100G:128,160,160" \\
           --patch_spec "140G:160,192,192" \\
           --out        patch_size_comparison.png

Either way the output is a PNG (or multi-page PDF if you pass .pdf) with
three mid-slices of your CT, one per axis, with colored boxes showing the
spatial extent each candidate patch covers, centered on the volume.

Author: Gregory Schwing, MD-PhD  |  Wayne State University / DMC
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# =============================================================================
# Plans-file parsing
# =============================================================================

def load_plans_file(path: Path, config: str = "3d_fullres") -> Dict:
    """
    Returns {label, patch_size (voxels), target_spacing (mm)}.
    Label is derived from filename (e.g. nnUNetResEncUNetPlans_100G.json -> "100G").
    """
    data = json.loads(path.read_text())
    if "configurations" not in data or config not in data["configurations"]:
        raise ValueError(
            f"{path} has no '{config}' configuration. "
            f"Available: {list(data.get('configurations', {}).keys())}")
    cfg = data["configurations"][config]

    patch = cfg.get("patch_size")
    if patch is None:
        raise ValueError(f"{path}/{config} has no patch_size")
    spacing = cfg.get("spacing") or data.get("original_median_spacing_after_transp")

    # Pull GPU memory target label out of filename:
    #   nnUNetResEncUNetPlans_100G.json -> 100G
    #   nnUNetPlans.json                 -> nnUNetPlans (no mem info)
    stem = path.stem
    label = stem
    for marker in ("Plans_", "plans_"):
        if marker in stem:
            label = stem.split(marker, 1)[1]
            break

    return {
        "label":          label,
        "source":         str(path),
        "patch_size":     [int(x) for x in patch],
        "target_spacing": [float(x) for x in spacing] if spacing else None,
    }


def parse_patch_spec(spec: str) -> Dict:
    """
    Parse "LABEL:x,y,z" into a dict. If ':' missing, label = patch_str.
    """
    if ":" in spec:
        label, dims = spec.split(":", 1)
    else:
        label, dims = spec, spec
    patch = [int(x.strip()) for x in dims.split(",")]
    if len(patch) != 3:
        raise ValueError(f"patch_spec '{spec}' must have 3 comma-separated ints")
    return {
        "label":          label.strip(),
        "source":         f"cli:{spec}",
        "patch_size":     patch,
        "target_spacing": None,   # unknown, use CT's native spacing for mm display
    }


# =============================================================================
# Visualization
# =============================================================================

_PALETTE = [
    "#e41a1c", "#377eb8", "#4daf4a", "#ff7f00",
    "#984ea3", "#a65628", "#f781bf", "#999999",
]


def render(
    ct_path: Path,
    patch_specs: List[Dict],
    out_path: Path,
    title: Optional[str] = None,
) -> None:
    img = nib.load(str(ct_path))
    data = np.asarray(img.get_fdata())
    ct_zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    ct_shape = tuple(int(s) for s in data.shape[:3])
    ct_mm = tuple(ct_shape[i] * ct_zooms[i] for i in range(3))

    # Clip for display (soft-tissue-ish window)
    vmin, vmax = -200, 400

    fig, axes = plt.subplots(1, 3, figsize=(18, 8.5), constrained_layout=False)

    # For each of 3 axis-views, pick the axes indices visible in the slice
    # view[i] = (slice_axis, (disp_axis_horiz, disp_axis_vert))
    views = [
        ("axis 0 mid-slice", 0, (2, 1)),
        ("axis 1 mid-slice", 1, (2, 0)),
        ("axis 2 mid-slice", 2, (1, 0)),
    ]

    for ax_idx, (view_name, slice_axis, (hax, vax)) in enumerate(views):
        ax = axes[ax_idx]
        mid = ct_shape[slice_axis] // 2
        slicer = [slice(None)] * 3
        slicer[slice_axis] = mid
        plane = data[tuple(slicer)]

        # Put horizontal axis on x, vertical on y; need to transpose if slice_axis
        # indexing leaves us with (axis a, axis b) but we want (hax shown on x, vax on y).
        # After slicing axis `slice_axis`, plane has shape of the other two axes
        # in their original order (i.e. for slice_axis=0, plane is (shape[1], shape[2])).
        remaining_axes = [i for i in range(3) if i != slice_axis]
        # remaining_axes is in natural order; e.g. slice_axis=0 -> [1,2]
        # plane[i, j] is (axis remaining_axes[0]=1, axis remaining_axes[1]=2)
        # We want x to show hax, y to show vax. So figure out the mapping.
        if remaining_axes[0] == hax:
            # plane's first axis is horizontal. imshow convention: first axis is Y.
            # So transpose to put horizontal on X.
            plane_disp = plane.T
        else:
            # plane's first axis is vertical already -- matches imshow Y
            plane_disp = plane

        extent = (0.0, ct_mm[hax], 0.0, ct_mm[vax])
        ax.imshow(plane_disp, cmap="gray", vmin=vmin, vmax=vmax,
                  extent=extent, origin="lower", aspect="equal")

        # Draw patch rectangles, centered on volume center in the hax/vax plane
        cx = ct_mm[hax] / 2.0
        cy = ct_mm[vax] / 2.0
        for i, spec in enumerate(patch_specs):
            color = _PALETTE[i % len(_PALETTE)]
            # Use target_spacing if provided (patch in target-space voxels),
            # otherwise interpret patch as voxels at the CT's native spacing.
            sp_mm = spec.get("target_spacing") or list(ct_zooms)
            patch_mm = [spec["patch_size"][k] * sp_mm[k] for k in range(3)]
            w = patch_mm[hax]
            h = patch_mm[vax]
            rect = Rectangle(
                (cx - w / 2, cy - h / 2), w, h,
                linewidth=2.2, edgecolor=color, facecolor="none",
                label=f"{spec['label']}  {patch_mm[hax]:.0f}x{patch_mm[vax]:.0f} mm",
            )
            ax.add_patch(rect)

        ax.set_title(f"{view_name}  (slice {mid}/{ct_shape[slice_axis]}, "
                     f"axis {slice_axis} = {ct_mm[slice_axis]:.0f} mm)")
        ax.set_xlabel(f"axis {hax}  ({ct_mm[hax]:.0f} mm)")
        ax.set_ylabel(f"axis {vax}  ({ct_mm[vax]:.0f} mm)")

    # Shared legend + title
    # Build descriptive labels (patch dims in vox + mm + spacing note)
    handles, _ = axes[0].get_legend_handles_labels()
    full_labels = []
    for i, spec in enumerate(patch_specs):
        sp_mm = spec.get("target_spacing") or list(ct_zooms)
        patch_mm = [spec["patch_size"][k] * sp_mm[k] for k in range(3)]
        sp_note = (f"target {sp_mm[0]:.2f}x{sp_mm[1]:.2f}x{sp_mm[2]:.2f} mm"
                   if spec.get("target_spacing")
                   else f"native {ct_zooms[0]:.2f}x{ct_zooms[1]:.2f}x{ct_zooms[2]:.2f} mm")
        full_labels.append(
            f"{spec['label']}   patch={spec['patch_size']} vox  "
            f"= {patch_mm[0]:.0f}x{patch_mm[1]:.0f}x{patch_mm[2]:.0f} mm  ({sp_note})"
        )

    if title is None:
        title = (f"Patch-size comparison on {ct_path.name}\n"
                 f"Volume: {ct_shape[0]}x{ct_shape[1]}x{ct_shape[2]} voxels "
                 f"@ {ct_zooms[0]:.2f}x{ct_zooms[1]:.2f}x{ct_zooms[2]:.2f} mm "
                 f"= {ct_mm[0]:.0f}x{ct_mm[1]:.0f}x{ct_mm[2]:.0f} mm FOV")

    # Explicit margins: top for title, bottom for legend + xlabels
    fig.subplots_adjust(top=0.88, bottom=0.20, left=0.04, right=0.98, wspace=0.22)
    fig.suptitle(title, fontsize=12, y=0.96)
    fig.legend(
        handles, full_labels,
        loc="lower center", ncol=1, frameon=True,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=10,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {out_path}")


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Overlay candidate patch sizes on a CT's mid-slices.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--ct", required=True, type=Path,
                    help="Path to a CT NIfTI (.nii.gz) for the background.")
    ap.add_argument("--plans", action="append", type=Path, default=[],
                    help="Path to a plans.json (repeatable). Patch size + "
                         "target spacing taken from 3d_fullres configuration.")
    ap.add_argument("--patch_spec", action="append", default=[],
                    help="Explicit 'LABEL:x,y,z' spec in voxels (repeatable). "
                         "Interpreted at the CT's native voxel spacing unless "
                         "a plans file says otherwise.")
    ap.add_argument("--config", default="3d_fullres",
                    help="Which configuration to read from each plans.json.")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output image path (.png or .pdf).")
    ap.add_argument("--title", default=None,
                    help="Custom figure title (default: auto-generated).")
    args = ap.parse_args()

    if not args.ct.exists():
        print(f"ERROR: CT not found: {args.ct}", file=sys.stderr)
        return 1
    if not args.plans and not args.patch_spec:
        print("ERROR: give at least one --plans or --patch_spec.", file=sys.stderr)
        return 1

    specs: List[Dict] = []
    for p in args.plans:
        if not p.exists():
            print(f"ERROR: plans file not found: {p}", file=sys.stderr)
            return 1
        specs.append(load_plans_file(p, args.config))
    for s in args.patch_spec:
        specs.append(parse_patch_spec(s))

    # De-dup on label (later wins) + preserve insertion order
    seen = {}
    for s in specs:
        seen[s["label"]] = s
    specs = list(seen.values())

    print(f"Rendering {len(specs)} patch spec(s) on {args.ct.name}:")
    for s in specs:
        sp = s.get("target_spacing")
        sp_str = f"{sp[0]:.2f}x{sp[1]:.2f}x{sp[2]:.2f}" if sp else "native"
        print(f"  {s['label']:>8s}  patch={s['patch_size']}  spacing={sp_str}")

    render(args.ct, specs, args.out, title=args.title)
    return 0


if __name__ == "__main__":
    sys.exit(main())
