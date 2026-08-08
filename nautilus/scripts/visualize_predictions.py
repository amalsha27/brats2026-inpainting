#!/usr/bin/env python3
"""
Visualise inpainting results: per-sample 7-panel figures + best/median/worst grid.
Reuses all helpers from evaluate.py.

Usage:
    python visualize_predictions.py --model improved --output_dir /data/experiments/improved-aug
    python visualize_predictions.py --model symmetry --output_dir /data/experiments/symmetry --n_samples 9
"""
import argparse, json, math, os, random, sys
import numpy as np
import torch

CODE_DIR = "/data/code"
sys.path.insert(0, CODE_DIR)

from train import (
    MODEL_REGISTRY, load_nifti, normalize, discover_samples,
    FILE_KEYS, MASK_KEYS,
)
from evaluate import (
    sliding_window_inference, best_slice,
    save_sample_figure, save_comparison_grid,
    compute_metrics, load_checkpoint, EvalCfg,
)

# ── args ──────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--model",      required=True,  help="improved | symmetry | zhang_boundary")
p.add_argument("--output_dir", required=True,  help="e.g. /data/experiments/improved-aug")
p.add_argument("--train_dir",  default="/data/brats2023/training")
p.add_argument("--n_samples",  type=int, default=9,  help="how many samples to visualise")
p.add_argument("--patch_size", nargs=3, type=int, default=[96, 96, 96])
p.add_argument("--overlap",    type=float, default=0.5)
p.add_argument("--seed",       type=int, default=42)
args = p.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_dir  = os.path.join(args.output_dir, args.model)
vis_dir  = os.path.join(args.output_dir, "visualizations")
os.makedirs(vis_dir, exist_ok=True)
print(f"Device: {device} | Saving figures → {vis_dir}")

# ── load model ────────────────────────────────────────────────────────────────
cfg   = EvalCfg()
model = MODEL_REGISTRY[args.model](cfg).to(device)
ckpt  = os.path.join(run_dir, "best.pt")
load_checkpoint(ckpt, model, device)
model.eval()

# ── discover samples ──────────────────────────────────────────────────────────
split_path = os.path.join(run_dir, "val_samples.json")
if os.path.exists(split_path):
    with open(split_path) as f:
        split = json.load(f)
    all_cases  = split["val_samples"]
    data_root  = split["val_root"]
    print(f"Using val split: {len(all_cases)} samples from {data_root}")
else:
    print("val_samples.json not found — using train_dir directly")
    data_root = args.train_dir
    all_cases = [
        s for s in sorted(os.listdir(data_root))
        if os.path.isdir(os.path.join(data_root, s))
        and os.path.exists(os.path.join(data_root, s, f"{s}-t1n.nii.gz"))
        and os.path.exists(os.path.join(data_root, s, f"{s}-mask-unhealthy.nii.gz"))
    ]

selected = random.sample(all_cases, min(args.n_samples, len(all_cases)))
print(f"Visualising {len(selected)} samples")

# ── run inference + save per-sample figures ───────────────────────────────────
patch_size = tuple(args.patch_size)
results    = []   # (ssim, name, voided, target, pred, mask)

for i, case_name in enumerate(selected):
    base = os.path.join(data_root, case_name)
    print(f"[{i+1}/{len(selected)}] {case_name}")

    # load & normalise
    t1    = load_nifti(os.path.join(base, f"{case_name}-t1n.nii.gz"))
    mask  = load_nifti(os.path.join(base, f"{case_name}-mask-unhealthy.nii.gz"))
    t1    = normalize(t1)
    mask  = (mask > 0.5).astype(np.float32)

    voided = t1.copy(); voided[mask > 0.5] = 0.0

    vol_data = {"voided": voided, "mask": mask}
    pred = sliding_window_inference(model, vol_data, patch_size,
                                    overlap=args.overlap, device=device)
    pred = np.clip(pred, 0, 1)

    metrics = compute_metrics(pred, t1, mask)
    ssim    = metrics["ssim"] if not math.isnan(metrics["ssim"]) else 0.0
    psnr    = metrics["psnr"] if not math.isnan(metrics["psnr"]) else 0.0
    mse     = metrics["mse"]  if not math.isnan(metrics["mse"])  else 1.0

    print(f"    SSIM={ssim:.4f}  PSNR={psnr:.2f}  MSE={mse:.6f}")

    # per-sample figure
    out_path = os.path.join(vis_dir, f"{case_name}.png")
    save_sample_figure(out_path, voided, t1, pred, mask, case_name, ssim, psnr, mse)
    print(f"    → {out_path}")

    results.append((ssim, case_name, voided, t1, pred, mask, metrics))

# ── best / median / worst comparison grid ────────────────────────────────────
results.sort(key=lambda x: x[0])   # sort by SSIM ascending (worst first)
n = len(results)
picks = {
    "Worst":  results[0],
    "Median": results[n // 2],
    "Best":   results[-1],
}

cases_for_grid = []
for label, (ssim, name, voided, target, pred, mask, metrics) in picks.items():
    cases_for_grid.append((voided, target, pred, mask, metrics, f"{label}: {name}"))

grid_path = os.path.join(vis_dir, f"comparison_grid_{args.model}.png")
save_comparison_grid(grid_path, cases_for_grid, model_name=args.model)
print(f"\nComparison grid → {grid_path}")

print(f"\nAll done. {len(results)} per-sample figures + 1 grid saved to {vis_dir}/")
