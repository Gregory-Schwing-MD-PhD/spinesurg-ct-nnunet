"""
spinesurg-ct-nnunet: host-side utilities for the pipeline.

Modules:
    visualize_patch_sizes    overlay candidate patch sizes on CT+label slices
    convert_hf_to_nnunet     build nnU-Net raw layout from HF export
    compare_ablations        aggregate ablation metrics into md/tex/csv tables
    nnunet_wandb_variant     W&B + LSTV-oversampling trainers (loaded in container)
"""

__version__ = "0.1.0"
