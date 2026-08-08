#!/usr/bin/env python3
"""
SwinUNETR Inpainting — Inference on BraTS validation set
Outputs predictions to ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Validation-Results/

Usage:
    python infer_swinunetr.py \
        --checkpoint /data/experiments/swinunetr/best.pt \
        --val_dir    /data/brats2023/validation \
        --output_dir /data/experiments/swinunetr \
        --tta
"""

import os, sys, math, argparse
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference as monai_swi
from pathlib import Path

# ── args ──────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--checkpoint",  required=True,  help="Path to best.pt")
p.add_argument("--val_dir",     required=True,  help="/data/brats2023/validation")
p.add_argument("--output_dir",  required=True,  help="base output dir for swinunetr")
p.add_argument("--patch_size",  type=int, nargs=3, default=[96,96,96])
p.add_argument("--overlap",     type=float, default=0.5)
p.add_argument("--tta",         action="store_true", help="8-way flip TTA")
p.add_argument("--sw_batch",    type=int, default=2, help="sw_batch_size for sliding window")
args = p.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

OUT_FOLDER = "ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Validation-Results"
out_dir = os.path.join(args.output_dir, OUT_FOLDER)
os.makedirs(out_dir, exist_ok=True)
print(f"Predictions → {out_dir}")

# ── model ─────────────────────────────────────────────────────────────────────
model = SwinUNETR(
    img_size=tuple(args.patch_size),
    in_channels=4,
    out_channels=1,
    feature_size=48,
    use_checkpoint=False,   # not needed for inference
    spatial_dims=3,
).to(device)

ckpt = torch.load(args.checkpoint, map_location=device)
# Support multiple checkpoint key formats
if "model_state" in ckpt:
    state = ckpt["model_state"]
elif "model_state_dict" in ckpt:
    state = ckpt["model_state_dict"]
elif "state_dict" in ckpt:
    state = ckpt["state_dict"]
else:
    state = ckpt
model.load_state_dict(state, strict=True)
model.eval()
print(f"Loaded checkpoint from {args.checkpoint}")

# ── helpers ───────────────────────────────────────────────────────────────────
def normalise(v):
    lo, hi = np.percentile(v, 1), np.percentile(v, 99)
    return np.clip((v - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)

def run_model(inp_tensor):
    """inp_tensor: (1,4,D,H,W) on device"""
    def _pred(x):
        return model(x)
    return monai_swi(
        inputs=inp_tensor,
        roi_size=tuple(args.patch_size),
        sw_batch_size=args.sw_batch,
        predictor=_pred,
        overlap=args.overlap,
    )

def run_tta(inp_tensor):
    """8-way flip TTA on a (1,4,D,H,W) tensor."""
    preds = []
    for flip_dims in [
        [],
        [2], [3], [4],
        [2,3], [2,4], [3,4],
        [2,3,4],
    ]:
        x = inp_tensor.flip(dims=flip_dims) if flip_dims else inp_tensor
        with torch.no_grad():
            pred = run_model(x)
        pred = pred.flip(dims=flip_dims) if flip_dims else pred
        preds.append(pred)
    return torch.stack(preds, dim=0).mean(dim=0)

# ── discover validation cases ─────────────────────────────────────────────────
# Val data structure: {val_dir}/{case}/{case}-t1n-voided.nii.gz + {case}-mask.nii.gz
cases = []
for s in sorted(os.listdir(args.val_dir)):
    d = os.path.join(args.val_dir, s)
    if not os.path.isdir(d): continue
    voided_f = os.path.join(d, f"{s}-t1n-voided.nii.gz")
    mask_f   = os.path.join(d, f"{s}-mask.nii.gz")
    if os.path.exists(voided_f) and os.path.exists(mask_f):
        cases.append({"name": s, "voided": voided_f, "mask": mask_f})

print(f"Found {len(cases)} validation cases")

# ── run inference ─────────────────────────────────────────────────────────────
for i, c in enumerate(cases):
    name = c["name"]
    out_path = os.path.join(out_dir, f"{name}-t1n.nii.gz")

    if os.path.exists(out_path):
        print(f"[{i+1}/{len(cases)}] {name}  — already exists, skipping")
        continue

    print(f"[{i+1}/{len(cases)}] {name}")

    # Load voided image (already has void applied) and mask
    voided_img = nib.load(c["voided"])
    voided_raw = voided_img.get_fdata(dtype=np.float32)
    affine     = voided_img.affine
    header     = voided_img.header
    mask_raw   = nib.load(c["mask"]).get_fdata(dtype=np.float32)

    mask_b = (mask_raw > 0.5).astype(np.float32)

    # Normalise the voided image for model input
    # Use non-zero voxels for percentile (avoid the zeroed-out void region)
    nonzero = voided_raw[voided_raw > 0]
    if len(nonzero) > 100:
        lo, hi = np.percentile(nonzero, 1), np.percentile(nonzero, 99)
    else:
        lo, hi = voided_raw.min(), voided_raw.max()
    voided_norm = np.clip((voided_raw - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)

    # Build 4-channel input: [voided, mask, voided, mask]
    inp = np.stack([voided_norm, mask_b, voided_norm, mask_b], axis=0)  # (4,D,H,W)
    inp_tensor = torch.from_numpy(inp).float().unsqueeze(0).to(device)  # (1,4,D,H,W)

    with torch.no_grad():
        if args.tta:
            pred_tensor = run_tta(inp_tensor)
        else:
            pred_tensor = run_model(inp_tensor)

    pred_norm = pred_tensor[0, 0].cpu().numpy().astype(np.float32)
    pred_norm = np.clip(pred_norm, 0.0, 1.0)

    # Denormalise back to original intensity scale
    pred_denorm = pred_norm * (hi - lo + 1e-8) + lo

    # Output: original voided image outside mask, prediction inside mask
    result = voided_raw.copy()
    result[mask_b > 0.5] = pred_denorm[mask_b > 0.5]

    # Save as NIfTI with original affine/header
    out_img = nib.Nifti1Image(result, affine=affine, header=header)
    nib.save(out_img, out_path)

    print(f"    → saved {out_path}")

total = len([f for f in os.listdir(out_dir) if f.endswith(".nii.gz")])
print(f"\nDone. {total}/{len(cases)} predictions saved to {out_dir}/")
