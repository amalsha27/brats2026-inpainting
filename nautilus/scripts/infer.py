#!/usr/bin/env python3
"""
BraTS 2026 Inpainting Challenge — Submission Inference Script
==============================================================
Generates prediction NIfTI files for Synapse upload.

Output format: BraTS-GLI-XXXXX-YYY-t1n-inference.nii.gz
Output folder: ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Validation-Results/

Usage:
    python infer.py --model improved --output_dir /data/experiments/improved
    python infer.py --model improved --output_dir /data/experiments/improved-aug --tta
"""

import argparse
import os
import sys
import time

import numpy as np
import nibabel as nib
import torch
from tqdm import tqdm

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

from train import MODEL_REGISTRY, MASK_KEYS, normalize
from evaluate import sliding_window_inference, EvalCfg, load_checkpoint

RESULTS_FOLDER = "ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Validation-Results"


def load_nifti_obj(val_dir, sample, key):
    """Load NIfTI object (keeps affine/header) for a given key."""
    for ext in (".nii.gz", ".nii"):
        p = os.path.join(val_dir, sample, f"{sample}-{key}{ext}")
        if os.path.exists(p):
            return nib.load(p)
    return None


def load_model(model_name, output_dir, device):
    """Load a trained model from output_dir/model_name/best.pt (or latest.pt)."""
    cfg = EvalCfg()
    run_dir = os.path.join(output_dir, model_name)
    ckpt = os.path.join(run_dir, "best.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(run_dir, "latest.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"No checkpoint in {run_dir}")
    model = MODEL_REGISTRY[model_name](cfg).to(device)
    load_checkpoint(ckpt, model, device)
    model.eval()
    return model


def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Primary model
    model = load_model(args.model, args.output_dir, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} | Params: {n_params/1e6:.2f}M")

    # Optional ensemble model
    model2 = None
    if args.model2 and args.output_dir2:
        model2 = load_model(args.model2, args.output_dir2, device)
        n2 = sum(p.numel() for p in model2.parameters())
        print(f"Ensemble model: {args.model2} | Params: {n2/1e6:.2f}M")

    # Output folder
    out_dir = os.path.join(args.output_dir, RESULTS_FOLDER)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output → {out_dir}")

    # Discover validation cases
    if not os.path.isdir(args.val_dir):
        raise FileNotFoundError(f"Val dir not found: {args.val_dir}")
    samples = sorted([
        d for d in os.listdir(args.val_dir)
        if os.path.isdir(os.path.join(args.val_dir, d))
        and os.path.exists(os.path.join(args.val_dir, d, f"{d}-t1n-voided.nii.gz"))
    ])
    print(f"Found {len(samples)} validation cases\n")

    patch_size = tuple(args.patch_size)

    for sample in tqdm(samples, desc="Inference"):
        out_path = os.path.join(out_dir, f"{sample}-t1n-inference.nii.gz")
        if os.path.exists(out_path) and not args.overwrite:
            tqdm.write(f"  SKIP {sample} (already exists)")
            continue

        # Load NIfTI objects (for affine/header)
        voided_nib = load_nifti_obj(args.val_dir, sample, "t1n-voided")
        mask_nib   = load_nifti_obj(args.val_dir, sample, "mask")
        if voided_nib is None or mask_nib is None:
            tqdm.write(f"  SKIP {sample}: missing voided or mask")
            continue

        # Original intensity data (for output reconstruction)
        voided_orig = np.asarray(voided_nib.dataobj, dtype=np.float32)
        mask_arr    = np.asarray(mask_nib.dataobj,   dtype=np.float32)
        mask_bin    = (mask_arr > 0).astype(np.float32)

        # Compute normalization params from non-zero, non-masked brain voxels
        brain_voxels = voided_orig[(voided_orig > 0) & (mask_bin < 0.5)]
        if brain_voxels.size < 100:
            # Fallback: use all non-zero voxels
            brain_voxels = voided_orig[voided_orig > 0]
        p1  = float(np.percentile(brain_voxels, 1))
        p99 = float(np.percentile(brain_voxels, 99))

        # Normalized volume for model input
        voided_norm = normalize(voided_orig, is_mask=False)
        vol = {
            "voided": voided_norm,
            "mask":   mask_bin,
        }

        t0 = time.time()

        def run_with_tta(m, v):
            p = sliding_window_inference(m, v, patch_size, overlap=args.overlap, device=str(device))
            if args.tta:
                preds = [p]
                for axes in [(2,), (1,), (0,), (1,2), (0,2), (0,1), (0,1,2)]:
                    v_aug = {k: np.flip(arr, axis=axes).copy() for k, arr in v.items()}
                    p_aug = sliding_window_inference(m, v_aug, patch_size, overlap=args.overlap, device=str(device))
                    preds.append(np.flip(p_aug, axis=axes).copy())
                return np.mean(preds, axis=0)
            return p

        pred_norm = run_with_tta(model, vol)

        # Ensemble: average with second model if provided
        if model2 is not None:
            pred2 = run_with_tta(model2, vol)
            pred_norm = (pred_norm + pred2) * 0.5

        elapsed = time.time() - t0

        # Denormalize prediction back to original MRI intensity range
        pred_orig = pred_norm * (p99 - p1) + p1
        pred_orig = np.clip(pred_orig, 0.0, None)  # no negative intensities in T1

        # Composite: prediction in masked region, original elsewhere
        result = pred_orig * mask_bin + voided_orig * (1.0 - mask_bin)

        # Save as NIfTI using the original voided image's affine/header
        out_nib = nib.Nifti1Image(result, voided_nib.affine, voided_nib.header)
        nib.save(out_nib, out_path)

        tqdm.write(f"  {sample}: saved ({elapsed:.1f}s)")

    print(f"\nDone! {len(samples)} predictions saved to:")
    print(f"  {out_dir}")
    print(f"\nNext step: upload that folder to Synapse validation submission.")


def parse_args():
    p = argparse.ArgumentParser(description="BraTS 2026 — Generate Synapse submission files")
    p.add_argument("--model",      required=True, choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--val_dir",    default="/data/brats2023/validation")
    p.add_argument("--output_dir", default="/data/experiments/improved",
                   help="Directory containing the model/<model>/ checkpoint folder")
    p.add_argument("--patch_size", type=int, nargs=3, default=[96, 96, 96])
    p.add_argument("--overlap",    type=float, default=0.5)
    p.add_argument("--tta",        action="store_true",
                   help="8-way test-time augmentation (all 3-axis flip combinations)")
    p.add_argument("--overwrite",  action="store_true",
                   help="Re-generate files that already exist")
    p.add_argument("--model2",      type=str, default="",
                   help="Second model name for ensemble (optional)")
    p.add_argument("--output_dir2", type=str, default="",
                   help="output_dir for the second ensemble model")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    infer(args)
