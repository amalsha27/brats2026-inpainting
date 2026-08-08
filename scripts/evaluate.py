#!/usr/bin/env python3
"""
BraTS Inpainting — Evaluation Script
======================================
Runs full-volume sliding window inference on the validation split,
then computes SSIM / PSNR / MSE on the masked (inpainted) region only,
matching the official BraTS challenge evaluation protocol.

Usage:
    python evaluate.py --model zhang2025 --output_dir /data/experiments
    python evaluate.py --model unet      --output_dir /data/experiments --overlap 0.5
"""

import argparse
import csv
import json
import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

import nibabel as nib
from skimage.metrics import structural_similarity

# Import shared utilities and all model definitions from train.py
from train import (
    MODEL_REGISTRY,
    load_nifti, normalize, discover_samples,
    FILE_KEYS, MASK_KEYS,
)


# ---------------------------------------------------------------------------
# Sliding window inference
# ---------------------------------------------------------------------------

def gaussian_kernel_1d(size, sigma=None):
    if sigma is None:
        sigma = size / 6.0
    coords = np.arange(size) - size // 2
    g = np.exp(-(coords**2) / (2 * sigma**2))
    return g / g.sum()


def gaussian_patch_3d(patch_size):
    """3D Gaussian weight window for smooth overlap-add blending."""
    pd, ph, pw = patch_size
    gd = gaussian_kernel_1d(pd)
    gh = gaussian_kernel_1d(ph)
    gw = gaussian_kernel_1d(pw)
    return (gd[:, None, None] * gh[None, :, None] * gw[None, None, :]).astype(np.float32)


def sliding_window_inference(model, vol_data, patch_size, overlap=0.5, device="cuda"):
    """Full-volume inference via overlapping patches with Gaussian blending.

    Args:
        model:      trained model (deterministic, expects batch["input"])
        vol_data:   dict of {key: np.ndarray (D,H,W)}
        patch_size: (pd, ph, pw)
        overlap:    fraction of patch overlap (0–1); 0.5 = 50% overlap
        device:     torch device string

    Returns:
        pred_vol: np.ndarray (D,H,W), values in [0,1]
    """
    pd, ph, pw = patch_size
    D, H, W = vol_data["voided"].shape

    stride_d = max(1, int(pd * (1 - overlap)))
    stride_h = max(1, int(ph * (1 - overlap)))
    stride_w = max(1, int(pw * (1 - overlap)))

    pred_sum = np.zeros((D, H, W), dtype=np.float32)
    weight   = np.zeros((D, H, W), dtype=np.float32)
    gw       = gaussian_patch_3d(patch_size)

    # Generate start positions, always include the last valid position
    def positions(total, patch, stride):
        pts = list(range(0, total - patch + 1, stride))
        if not pts or pts[-1] + patch < total:
            pts.append(max(0, total - patch))
        return pts

    ds = positions(D, pd, stride_d)
    hs = positions(H, ph, stride_h)
    ws = positions(W, pw, stride_w)

    model.eval()
    with torch.no_grad():
        for d0 in ds:
            for h0 in hs:
                for w0 in ws:
                    # Crop patch from each modality
                    patch = {}
                    for k, v in vol_data.items():
                        sl = v[d0:d0+pd, h0:h0+ph, w0:w0+pw]
                        # Pad if volume is smaller than patch_size
                        pad = [(0, max(0, s - sl.shape[i]))
                               for i, s in enumerate([pd, ph, pw])]
                        if any(p[1] > 0 for p in pad):
                            sl = np.pad(sl, pad, mode="reflect")
                        patch[k] = torch.from_numpy(sl).float()

                    voided = patch["voided"].unsqueeze(0).unsqueeze(0).to(device)  # (1,1,D,H,W)
                    mask   = patch["mask"].unsqueeze(0).unsqueeze(0).to(device)
                    batch  = {"input": torch.cat([voided, mask], dim=1)}

                    out = model(batch)[0, 0].cpu().numpy()  # (pd,ph,pw)

                    # Crop output back to actual size if volume was smaller
                    actual_d = min(pd, D - d0)
                    actual_h = min(ph, H - h0)
                    actual_w = min(pw, W - w0)
                    out = out[:actual_d, :actual_h, :actual_w]
                    gw_crop = gw[:actual_d, :actual_h, :actual_w]

                    pred_sum[d0:d0+actual_d, h0:h0+actual_h, w0:w0+actual_w] += out * gw_crop
                    weight  [d0:d0+actual_d, h0:h0+actual_h, w0:w0+actual_w] += gw_crop

    weight = np.maximum(weight, 1e-8)
    return pred_sum / weight


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def best_slice(mask_vol):
    """Return axial slice index with the most mask voxels."""
    counts = (mask_vol > 0).sum(axis=(1, 2))   # sum over H,W per D-slice
    return int(np.argmax(counts))


def save_sample_figure(out_path, voided, target, pred, mask, sample_name,
                       ssim, psnr, mse):
    """Save a 7-panel figure for one sample:
       Voided | Ground Truth | Prediction | Difference | Error Heatmap |
       Mask Overlay | SSIM Map
    All panels show the axial slice with most mask coverage.
    """
    sl = best_slice(mask)

    # Extract 2D slices (H, W)
    v  = voided[sl]
    gt = target[sl]
    pr = pred[sl]
    mk = mask[sl]

    diff      = np.abs(pr - gt)
    err_mask  = diff * mk                          # error only in masked region
    ssim_map  = 1.0 - np.abs(pr - gt)             # crude per-pixel "ssim proxy"

    # Overlay: prediction + mask boundary
    from skimage.segmentation import find_boundaries
    boundary = find_boundaries(mk > 0.5, mode="outer").astype(np.float32)

    fig = plt.figure(figsize=(22, 4))
    fig.suptitle(
        f"{sample_name}   SSIM={ssim:.4f}  PSNR={psnr:.2f} dB  MSE={mse:.6f}",
        fontsize=11, fontweight="bold"
    )
    gs = gridspec.GridSpec(1, 7, figure=fig, wspace=0.05)

    panels = [
        ("Voided Input",    v,         "gray",    None),
        ("Ground Truth",    gt,        "gray",    None),
        ("Prediction",      pr,        "gray",    None),
        ("|GT − Pred|",     diff,      "hot",     None),
        ("Error (mask)",    err_mask,  "inferno", None),
        ("Overlay",         None,      None,      None),   # handled separately
        ("SSIM proxy",      ssim_map,  "RdYlGn",  (0, 1)),
    ]

    for i, (title, img, cmap, vrange) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        ax.set_title(title, fontsize=9)
        ax.axis("off")

        if title == "Overlay":
            # Show prediction in grey, highlight mask region in colour, draw boundary
            rgb = plt.cm.gray(Normalize()(pr))[:, :, :3]
            # Tint the masked region with a transparent orange
            tint = np.zeros_like(rgb)
            tint[mk > 0.5] = [1.0, 0.5, 0.0]
            rgb = np.clip(rgb * 0.7 + tint * 0.3, 0, 1)
            # Draw boundary in green
            rgb[boundary > 0] = [0.0, 1.0, 0.0]
            ax.imshow(rgb)
        else:
            vmin, vmax = vrange if vrange else (img.min(), img.max())
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            # Colourbar only for heatmaps
            if cmap in ("hot", "inferno", "RdYlGn"):
                div = make_axes_locatable(ax)
                cax = div.append_axes("right", size="5%", pad=0.03)
                plt.colorbar(im, cax=cax)

    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def save_comparison_grid(out_path, cases, model_name):
    """Save a grid of best / median / worst cases (3 rows × 5 panels).
    Matches the qualitative figure style in the Zhang2025 paper (Fig. 2).
    """
    labels = ["Best", "Median", "Worst"]
    col_titles = ["Voided Input", "Ground Truth", "Prediction",
                  "|GT − Pred|", "Mask Overlay"]

    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle(f"Qualitative Results — {model_name}", fontsize=14, fontweight="bold")

    for row, (label, data) in enumerate(zip(labels, cases)):
        voided, target, pred, mask, metrics, name = data
        sl = best_slice(mask)
        v, gt, pr, mk = voided[sl], target[sl], pred[sl], mask[sl]
        diff = np.abs(pr - gt)

        from skimage.segmentation import find_boundaries
        boundary = find_boundaries(mk > 0.5, mode="outer").astype(np.float32)
        rgb = plt.cm.gray(Normalize()(pr))[:, :, :3]
        tint = np.zeros_like(rgb); tint[mk > 0.5] = [1.0, 0.5, 0.0]
        rgb = np.clip(rgb * 0.7 + tint * 0.3, 0, 1)
        rgb[boundary > 0] = [0.0, 1.0, 0.0]

        imgs   = [v, gt, pr, diff, None]
        cmaps  = ["gray","gray","gray","hot", None]

        for col in range(5):
            ax = axes[row, col]
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(
                    f"{label}\n{name}\nSSIM={metrics['ssim']:.3f}",
                    fontsize=8, rotation=0, labelpad=80, va="center"
                )
            if col == 4:
                ax.imshow(rgb)
            else:
                ax.imshow(imgs[col], cmap=cmaps[col])
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Metrics (mask-region only, matching challenge protocol)
# ---------------------------------------------------------------------------

def compute_metrics(pred, target, mask):
    """SSIM / PSNR / MSE computed only on the masked (inpainted) region.
    SSIM is computed on the tight bounding box around the mask (not full volume)
    to match the challenge evaluation protocol.
    """
    m = mask > 0
    if not m.any():
        return {"ssim": float("nan"), "psnr": float("nan"), "mse": float("nan")}

    mse  = float(np.mean((pred[m] - target[m]) ** 2))
    psnr = float(10 * math.log10(1.0 / (mse + 1e-10)))

    # Crop to mask bounding box for SSIM — avoids penalising the whole volume
    coords = np.argwhere(m)
    d0, h0, w0 = coords.min(axis=0)
    d1, h1, w1 = coords.max(axis=0) + 1
    # Add small padding so SSIM window has context
    pad = 8
    D, H, W = pred.shape
    d0, h0, w0 = max(0,d0-pad), max(0,h0-pad), max(0,w0-pad)
    d1, h1, w1 = min(D,d1+pad), min(H,h1+pad), min(W,w1+pad)
    pred_crop   = pred  [d0:d1, h0:h1, w0:w1]
    target_crop = target[d0:d1, h0:h1, w0:w1]
    ssim = float(structural_similarity(target_crop, pred_crop, data_range=1.0))

    return {"ssim": ssim, "psnr": psnr, "mse": mse}


# ---------------------------------------------------------------------------
# Load a checkpoint into model
# ---------------------------------------------------------------------------

def load_checkpoint(path, model, device):
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    model.load_state_dict(state, strict=True)
    iteration = ckpt.get("iteration", 0)
    print(f"  Loaded checkpoint: {path}  (iter {iteration})")
    return iteration


# ---------------------------------------------------------------------------
# Minimal cfg substitute for MODEL_REGISTRY lambdas
# ---------------------------------------------------------------------------

class EvalCfg:
    base_ch   = 32
    ch_mult   = (1, 2, 4, 8)
    n_blocks  = 2
    d_state   = 16
    patch_size = (96, 96, 96)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_dir  = os.path.join(args.output_dir, args.model)
    ckpt     = os.path.join(run_dir, "best.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(run_dir, "latest.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"No checkpoint found in {run_dir}. "
                                f"Train the model first.")

    # Build and load model
    cfg   = EvalCfg()
    model = MODEL_REGISTRY[args.model](cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} | Params: {n_params/1e6:.2f}M")
    load_checkpoint(ckpt, model, device)
    model.eval()

    # Load the val split saved during training
    split_path = os.path.join(run_dir, "val_samples.json")
    if os.path.exists(split_path):
        with open(split_path) as f:
            split = json.load(f)
        val_samples = split["val_samples"]
        val_root    = split["val_root"]
        print(f"  Val split: {len(val_samples)} samples from {val_root}")
    else:
        # Fallback: discover from train dir with ground truth
        print("  WARNING: val_samples.json not found — discovering from train_dir")
        val_root    = args.train_dir
        all_samples = discover_samples(val_root, require_target=True)
        np.random.seed(42)
        np.random.shuffle(all_samples)
        n_val       = max(1, int(len(all_samples) * 0.1))
        val_samples = all_samples[:n_val]

    print(f"\nEvaluating {len(val_samples)} samples  "
          f"(patch={args.patch_size}, overlap={args.overlap})")

    results    = []
    vis_data   = []   # (voided, target, pred, mask, metrics, name) for grid
    patch_size = tuple(args.patch_size)

    img_dir = os.path.join(run_dir, "eval", "images")
    os.makedirs(img_dir, exist_ok=True)

    for name in tqdm(val_samples, desc="Samples"):
        # Load full volume
        vol = {}
        for key, fk in FILE_KEYS.items():
            v = load_nifti(val_root, name, fk)
            if v is not None:
                vol[key] = normalize(v, is_mask=(key in MASK_KEYS))

        if "voided" not in vol or "target" not in vol or "mask" not in vol:
            print(f"  SKIP {name}: missing required files (keys={list(vol.keys())})")
            continue

        t0   = time.time()
        pred = sliding_window_inference(model, vol, patch_size,
                                        overlap=args.overlap, device=str(device))

        # Test-time augmentation: left-right flip (axis=2) and average
        if args.tta:
            vol_lr   = {k: np.flip(v, axis=2).copy() for k, v in vol.items()}
            pred_lr  = sliding_window_inference(model, vol_lr, patch_size,
                                                overlap=args.overlap, device=str(device))
            pred     = (pred + np.flip(pred_lr, axis=2).copy()) * 0.5

        elapsed = time.time() - t0

        # Composite: keep model output only in the mask region,
        # restore original voided values everywhere else.
        # This is what the challenge does — only the mask region is evaluated.
        pred = pred * vol["mask"] + vol["voided"] * (1.0 - vol["mask"])

        m = compute_metrics(pred, vol["target"], vol["mask"])
        m["sample"]  = name
        m["time_s"]  = round(elapsed, 2)
        results.append(m)
        # pred is already composited — store for visualisation
        vis_data.append((vol["voided"], vol["target"], pred, vol["mask"], m, name))

        # Per-sample figure
        fig_path = os.path.join(img_dir, f"{name}.png")
        save_sample_figure(fig_path, vol["voided"], vol["target"], pred,
                           vol["mask"], name, m["ssim"], m["psnr"], m["mse"])
        # Also save the raw model prediction (before compositing) mask crop as npy
        # — omitted to save space; full nifti export can be added later

        tqdm.write(f"  {name}: SSIM={m['ssim']:.4f}  PSNR={m['psnr']:.2f}  "
                   f"MSE={m['mse']:.6f}  ({elapsed:.1f}s)")

    if not results:
        print("No results — check that val samples exist and have ground truth.")
        return

    # Summary statistics
    ssims = [r["ssim"] for r in results if not math.isnan(r["ssim"])]
    psnrs = [r["psnr"] for r in results if not math.isnan(r["psnr"])]
    mses  = [r["mse"]  for r in results if not math.isnan(r["mse"])]

    summary = {
        "model":   args.model,
        "n":       len(results),
        "ssim":  {"mean": float(np.mean(ssims)), "std": float(np.std(ssims)),
                  "median": float(np.median(ssims)),
                  "q25": float(np.percentile(ssims, 25)),
                  "q75": float(np.percentile(ssims, 75))},
        "psnr":  {"mean": float(np.mean(psnrs)), "std": float(np.std(psnrs)),
                  "median": float(np.median(psnrs))},
        "mse":   {"mean": float(np.mean(mses)),  "std": float(np.std(mses)),
                  "median": float(np.median(mses))},
    }

    print("\n" + "="*60)
    print(f"  Model  : {args.model}")
    print(f"  Samples: {summary['n']}")
    print(f"  SSIM   : {summary['ssim']['mean']:.4f} ± {summary['ssim']['std']:.4f}  "
          f"(median {summary['ssim']['median']:.4f})")
    print(f"  PSNR   : {summary['psnr']['mean']:.2f} ± {summary['psnr']['std']:.2f} dB")
    print(f"  MSE    : {summary['mse']['mean']:.6f} ± {summary['mse']['std']:.6f}")
    print("="*60)

    # Save outputs
    out_dir = os.path.join(run_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    csv_path = os.path.join(out_dir, "per_sample.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample", "ssim", "psnr", "mse", "time_s"])
        writer.writeheader()
        writer.writerows(results)

    # Best / median / worst comparison grid (matches paper Fig. 2 style)
    if len(vis_data) >= 3:
        sorted_by_ssim = sorted(vis_data, key=lambda x: x[4]["ssim"], reverse=True)
        best   = sorted_by_ssim[0]
        worst  = sorted_by_ssim[-1]
        median = sorted_by_ssim[len(sorted_by_ssim) // 2]
        grid_path = os.path.join(out_dir, "best_median_worst.png")
        save_comparison_grid(grid_path, [best, median, worst], args.model)
        print(f"  best_median_worst.png — qualitative comparison grid")

    print(f"\nResults saved to {out_dir}/")
    print(f"  summary.json       — aggregate stats")
    print(f"  per_sample.csv     — per-sample breakdown")
    print(f"  images/            — {len(vis_data)} per-sample figures")
    print(f"                       (voided | GT | pred | diff | error | overlay | ssim-map)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="BraTS Inpainting Evaluation")
    p.add_argument("--model",       required=True,
                   choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--train_dir",   default="/data/brats2023/training")
    p.add_argument("--output_dir",  default="/data/experiments")
    p.add_argument("--patch_size",  type=int, nargs=3, default=[96, 96, 96])
    p.add_argument("--overlap",     type=float, default=0.5,
                   help="Patch overlap fraction for sliding window (default 0.5)")
    p.add_argument("--tta",         action="store_true",
                   help="Test-time augmentation: average LR-flip prediction")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.patch_size = tuple(args.patch_size)
    evaluate(args)
