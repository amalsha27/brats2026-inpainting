#!/usr/bin/env python3
"""
SwinUNETR Inpainting — BraTS 2026
Uses MONAI's SSL-pretrained SwinUNETR encoder (pretrained on 5000 epochs of
unlabeled medical data) as backbone. Only the decoder is trained from scratch.

Strategy:
  - Phase 1 (0–20k):  freeze encoder, train decoder only  (fast feature adaptation)
  - Phase 2 (20k+):   unfreeze all, end-to-end fine-tuning with lower LR

Checkpoint format matches existing evaluate.py / infer.py expectations:
  best.pt / latest.pt  →  {"model_state": ..., "iteration": ..., "best_ssim": ...}
"""

import os, sys, json, math, time, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path

# ── monai ─────────────────────────────────────────────────────────────────────
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference as monai_swi

# ── nibabel / skimage ─────────────────────────────────────────────────────────
import nibabel as nib
from skimage.metrics import structural_similarity

# ── args ──────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--train_dir",    default="/data/brats2023/training")
p.add_argument("--output_dir",   default="/data/experiments/swinunetr")
p.add_argument("--pretrained",   default="/data/code/swin_pretrained.pt",
               help="Path to SSL-pretrained SwinUNETR weights")
p.add_argument("--n_iters",      type=int,   default=200_000)
p.add_argument("--val_interval", type=int,   default=5_000)
p.add_argument("--batch_size",   type=int,   default=1)
p.add_argument("--lr",           type=float, default=2e-4)
p.add_argument("--patch_size",   type=int,   nargs=3, default=[96,96,96])
p.add_argument("--val_frac",     type=float, default=0.1)
p.add_argument("--val_cases",    type=int,   default=20,
               help="Max validation cases per eval (cap to save time/memory)")
p.add_argument("--seed",         type=int,   default=42)
p.add_argument("--unfreeze_iter",type=int,   default=20_000,
               help="Iteration at which to unfreeze the encoder")
args = p.parse_args()

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

os.makedirs(args.output_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── discover samples ──────────────────────────────────────────────────────────
def find_cases(root):
    cases = []
    for s in sorted(os.listdir(root)):
        d = os.path.join(root, s)
        if not os.path.isdir(d): continue
        t1   = os.path.join(d, f"{s}-t1n.nii.gz")
        mask = os.path.join(d, f"{s}-mask-unhealthy.nii.gz")
        if os.path.exists(t1) and os.path.exists(mask):
            cases.append({"name": s, "t1": t1, "mask": mask})
    return cases

all_cases = find_cases(args.train_dir)
random.shuffle(all_cases)
n_val  = max(1, int(len(all_cases) * args.val_frac))
val_cases   = all_cases[:n_val]
train_cases = all_cases[n_val:]
print(f"Train: {len(train_cases)} | Val: {len(val_cases)}")

split_path = os.path.join(args.output_dir, "val_samples.json")
with open(split_path, "w") as f:
    json.dump({"val_samples": [c["name"] for c in val_cases],
               "val_root": args.train_dir}, f, indent=2)

# ── normalise ─────────────────────────────────────────────────────────────────
def normalise(v):
    lo, hi = np.percentile(v, 1), np.percentile(v, 99)
    return np.clip((v - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)

def load_case(c):
    t1   = nib.load(c["t1"]).get_fdata(dtype=np.float32)
    mask = nib.load(c["mask"]).get_fdata(dtype=np.float32)
    t1   = normalise(t1)
    mask = (mask > 0.5).astype(np.float32)
    return t1, mask

# ── dataset ───────────────────────────────────────────────────────────────────
class InpaintDataset(Dataset):
    def __init__(self, cases, patch_size, augment=True):
        self.cases      = cases
        self.patch_size = patch_size
        self.augment    = augment
        # No in-memory cache — avoids 30GB RAM usage with 1126 cases

    def __len__(self):  return len(self.cases)

    def __getitem__(self, idx):
        t1, mask = load_case(self.cases[idx])
        pd, ph, pw = self.patch_size
        D, H, W = t1.shape

        # Random crop
        d0 = random.randint(0, max(0, D-pd))
        h0 = random.randint(0, max(0, H-ph))
        w0 = random.randint(0, max(0, W-pw))
        t1_p   = t1  [d0:d0+pd, h0:h0+ph, w0:w0+pw]
        mask_p = mask[d0:d0+pd, h0:h0+ph, w0:w0+pw]

        # Pad if needed
        def pad(x, target):
            pad_w = [(0, max(0, t-s)) for s,t in zip(x.shape, target)]
            return np.pad(x, pad_w, mode="reflect") if any(p[1]>0 for p in pad_w) else x
        t1_p   = pad(t1_p,   (pd,ph,pw))
        mask_p = pad(mask_p, (pd,ph,pw))

        voided = t1_p.copy(); voided[mask_p > 0.5] = 0.0

        if self.augment:
            # Random flips
            for ax in range(3):
                if random.random() < 0.5:
                    t1_p   = np.flip(t1_p,   axis=ax).copy()
                    mask_p = np.flip(mask_p, axis=ax).copy()
                    voided = np.flip(voided, axis=ax).copy()

        # Build 4-channel input: [voided, mask, voided, mask]
        # (matches pretrained 4-channel SwinUNETR patch embedding)
        inp = np.stack([voided, mask_p, voided, mask_p], axis=0)

        return {
            "input":  torch.from_numpy(inp),
            "target": torch.from_numpy(t1_p[None]),
            "mask":   torch.from_numpy(mask_p[None]),
        }

train_ds = InpaintDataset(train_cases, args.patch_size, augment=True)
train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=2, pin_memory=True,
                          prefetch_factor=2, persistent_workers=True)

# ── model ─────────────────────────────────────────────────────────────────────
model = SwinUNETR(
    img_size=tuple(args.patch_size),
    in_channels=4,       # [voided, mask, voided, mask]
    out_channels=1,
    feature_size=48,
    use_checkpoint=True,
    spatial_dims=3,
).to(device)

# Load SSL-pretrained encoder backbone
if os.path.exists(args.pretrained):
    weights = torch.load(args.pretrained, map_location="cpu")
    model.load_from(weights)
    print(f"Loaded pretrained encoder from {args.pretrained}")
else:
    print(f"WARNING: pretrained weights not found at {args.pretrained}, training from scratch")

n_params = sum(p.numel() for p in model.parameters())
print(f"SwinUNETR params: {n_params/1e6:.1f}M")

# Phase 1: freeze encoder, train decoder only
def set_encoder_grad(requires_grad: bool):
    for name, param in model.named_parameters():
        if "swinViT" in name:
            param.requires_grad = requires_grad

set_encoder_grad(False)
print("Phase 1: encoder frozen — training decoder only")

# ── optimiser & scheduler ─────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=args.lr, weight_decay=1e-5
)
# Cosine decay over full training
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.n_iters, eta_min=args.lr * 0.01
)
# AMP scaler — halves VRAM usage (essential for 11GB GPUs)
scaler = GradScaler()

# ── loss ──────────────────────────────────────────────────────────────────────
def ssim_loss(pred, target, mask=None):
    """Masked SSIM loss (1 - SSIM), patch-level."""
    # Use F.avg_pool3d to compute local means, then SSIM approximation
    C1, C2 = 0.01**2, 0.03**2
    mu_p = F.avg_pool3d(pred,   kernel_size=11, stride=1, padding=5)
    mu_t = F.avg_pool3d(target, kernel_size=11, stride=1, padding=5)
    mu_pp = F.avg_pool3d(pred**2,   kernel_size=11, stride=1, padding=5) - mu_p**2
    mu_tt = F.avg_pool3d(target**2, kernel_size=11, stride=1, padding=5) - mu_t**2
    mu_pt = F.avg_pool3d(pred*target, kernel_size=11, stride=1, padding=5) - mu_p*mu_t
    ssim_map = ((2*mu_p*mu_t + C1) * (2*mu_pt + C2)) / \
               ((mu_p**2 + mu_t**2 + C1) * (mu_pp + mu_tt + C2) + 1e-8)
    if mask is not None:
        return 1.0 - (ssim_map * mask).sum() / (mask.sum() + 1e-8)
    return 1.0 - ssim_map.mean()

def compute_loss(pred, target, mask):
    l1   = F.l1_loss(pred, target)
    ssl  = ssim_loss(pred, target, mask)
    # Extra penalty inside masked region
    l1m  = (F.l1_loss(pred * mask, target * mask, reduction="sum")
            / (mask.sum() + 1e-8))
    return 0.5 * l1 + 0.3 * ssl + 0.2 * l1m

# ── validation ────────────────────────────────────────────────────────────────
def validate():
    model.eval()
    results = []
    # Cap to args.val_cases to avoid OOM on 11GB GPUs
    subset = val_cases[:args.val_cases]
    with torch.no_grad():
        for c in subset:
            t1, mask = load_case(c)
            voided = t1.copy(); voided[mask > 0.5] = 0.0

            def _infer(x):  # x: (1,4,d,h,w)
                with autocast():
                    return model(x)

            inp_vol = torch.from_numpy(
                np.stack([voided, mask, voided, mask], axis=0)
            ).float().unsqueeze(0).to(device)  # (1,4,D,H,W)

            pred_vol = monai_swi(
                inputs=inp_vol,
                roi_size=tuple(args.patch_size),
                sw_batch_size=1,
                predictor=_infer,
                overlap=0.25,   # lower overlap = less memory during val
            )[0, 0].cpu().numpy()

            pred_vol = np.clip(pred_vol, 0, 1)
            m = mask > 0.5
            if not m.any(): continue
            mse  = float(np.mean((pred_vol[m] - t1[m])**2))
            coords = np.argwhere(m)
            d0,h0,w0 = coords.min(axis=0); d1,h1,w1 = coords.max(axis=0)+1
            pad=8; D,H,W=t1.shape
            d0,h0,w0=max(0,d0-pad),max(0,h0-pad),max(0,w0-pad)
            d1,h1,w1=min(D,d1+pad),min(H,h1+pad),min(W,w1+pad)
            ssim_val = float(structural_similarity(
                t1[d0:d1,h0:h1,w0:w1],
                pred_vol[d0:d1,h0:h1,w0:w1],
                data_range=1.0))
            psnr_val = float(10*math.log10(1/(mse+1e-10)))
            results.append({"ssim": ssim_val, "psnr": psnr_val, "mse": mse})

    model.train()
    mean_ssim = float(np.mean([r["ssim"] for r in results]))
    mean_psnr = float(np.mean([r["psnr"] for r in results]))
    mean_mse  = float(np.mean([r["mse"]  for r in results]))
    return mean_ssim, mean_psnr, mean_mse

# ── checkpoint helpers ────────────────────────────────────────────────────────
def save_ckpt(path, iteration, best_ssim):
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration,
        "best_ssim": best_ssim,
    }, path)

def load_ckpt(path):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt.get("iteration", 0), ckpt.get("best_ssim", 0.0)

# ── resume ────────────────────────────────────────────────────────────────────
start_iter = 0
best_ssim  = 0.0
latest_pt  = os.path.join(args.output_dir, "latest.pt")
best_pt    = os.path.join(args.output_dir, "best.pt")
history    = {"iter": [], "val_ssim": [], "val_psnr": [], "val_mse": []}
hist_path  = os.path.join(args.output_dir, "history.json")

if os.path.exists(latest_pt):
    try:
        start_iter, best_ssim = load_ckpt(latest_pt)
        print(f"Resumed from iter {start_iter}, best SSIM={best_ssim:.4f}")
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                history = json.load(f)
    except Exception as e:
        print(f"WARNING: could not resume: {e}")

# ── training loop ─────────────────────────────────────────────────────────────
model.train()
data_iter   = iter(train_loader)
iter_count  = start_iter
encoder_unfrozen = (iter_count >= args.unfreeze_iter)
t0 = time.time()

print(f"\nStarting from iter {start_iter}, target {args.n_iters}")

while iter_count < args.n_iters:

    # Phase 2 switch: unfreeze encoder
    if not encoder_unfrozen and iter_count >= args.unfreeze_iter:
        set_encoder_grad(True)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr * 0.1, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.n_iters - iter_count, eta_min=args.lr * 0.001
        )
        scaler = GradScaler()
        encoder_unfrozen = True
        print(f"\n[{iter_count}] Phase 2: encoder unfrozen, LR={args.lr*0.1:.2e}")

    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        batch = next(data_iter)

    inp    = batch["input"].to(device)
    target = batch["target"].to(device)
    mask   = batch["mask"].to(device)

    optimizer.zero_grad()
    with autocast():
        pred = model(inp)
        loss = compute_loss(pred, target, mask)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    iter_count += 1

    # logging
    if iter_count % 100 == 0:
        elapsed = time.time() - t0
        its     = (iter_count - start_iter) / elapsed
        eta_h   = (args.n_iters - iter_count) / (its * 3600 + 1e-8)
        lr_now  = optimizer.param_groups[0]["lr"]
        print(f"[{iter_count:>7}/{args.n_iters}] loss={loss.item():.4f} | "
              f"{its:.1f}it/s | ETA {eta_h:.1f}h | lr={lr_now:.2e}")

    # validation
    if iter_count % args.val_interval == 0:
        print(f"\n  Validating at iter {iter_count}...")
        ssim_v, psnr_v, mse_v = validate()
        print(f"  Val [{iter_count}] SSIM={ssim_v:.4f} PSNR={psnr_v:.2f} MSE={mse_v:.6f}")

        history["iter"].append(iter_count)
        history["val_ssim"].append(ssim_v)
        history["val_psnr"].append(psnr_v)
        history["val_mse"].append(mse_v)

        try:
            with open(hist_path, "w") as f:
                json.dump(history, f)
        except Exception:
            pass

        save_ckpt(latest_pt, iter_count, best_ssim)

        if ssim_v > best_ssim:
            best_ssim = ssim_v
            save_ckpt(best_pt, iter_count, best_ssim)
            print(f"  ** New best SSIM: {best_ssim:.4f} — saved best.pt")

print(f"\nTraining complete. Best SSIM={best_ssim:.4f}")
