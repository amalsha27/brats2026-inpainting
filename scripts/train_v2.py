#!/usr/bin/env python3
"""
train_v2.py — Ablation training script for void-shape augmentation study.

Extends train_synth_aug.py with:
  - --aug_strategy  : none | ellipsoid | sphere | cuboid | morphed | mixed
  - --hcp_dir       : optional HCP healthy-brain T1w directory
  - Masked loss     : gradient flows only inside the void region (compositing trick)

Checkpoint format identical to train_synth_aug.py (model key = 'improved').
"""

import os, sys, json, math, time, random, argparse, glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import nibabel as nib
from skimage.metrics import structural_similarity

CODE_DIR = "/data/code"
sys.path.insert(0, CODE_DIR)

from train    import MODEL_REGISTRY
from evaluate import EvalCfg

# Import new augmentation module
AUG_DIR = os.path.join(CODE_DIR, "scripts")
sys.path.insert(0, AUG_DIR)
from synth_aug_v2 import get_void_mask, STRATEGIES

# ── args ───────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--train_dir",       default="/data/brats2023/training")
p.add_argument("--hcp_dir",         default="",
               help="Optional: /data/hcp_preprocessed  (adds healthy-brain training samples)")
p.add_argument("--output_dir",      default="/data/experiments/ablation-ellipsoid")
p.add_argument("--aug_strategy",    default="ellipsoid", choices=STRATEGIES,
               help="Void shape strategy for synthetic augmentation")
p.add_argument("--aug_prob",        type=float, default=0.5,
               help="Probability of applying synthetic void (ignored for 'none'/'morphed')")
p.add_argument("--model",           default="improved")
p.add_argument("--n_iters",         type=int,   default=300_000)
p.add_argument("--val_interval",    type=int,   default=5_000)
p.add_argument("--batch_size",      type=int,   default=1)
p.add_argument("--lr",              type=float, default=1e-4)
p.add_argument("--patch_size",      type=int,   nargs=3, default=[96, 96, 96])
p.add_argument("--val_frac",        type=float, default=0.1)
p.add_argument("--masked_loss_w",   type=float, default=0.5,
               help="Weight of masked-only loss term (0 = full-image loss only)")
p.add_argument("--seed",            type=int,   default=42)
args = p.parse_args()

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)

os.makedirs(args.output_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device:        {device}")
print(f"Strategy:      {args.aug_strategy}  (aug_prob={args.aug_prob})")
print(f"Masked loss w: {args.masked_loss_w}")

# ── discover BraTS cases ───────────────────────────────────────────────────
def find_brats_cases(root):
    cases = []
    for s in sorted(os.listdir(root)):
        d = os.path.join(root, s)
        if not os.path.isdir(d):
            continue
        t1   = os.path.join(d, f"{s}-t1n.nii.gz")
        mask = os.path.join(d, f"{s}-mask-unhealthy.nii.gz")
        if os.path.exists(t1) and os.path.exists(mask):
            cases.append({"name": s, "t1": t1, "mask": mask, "source": "brats"})
    return cases

all_brats = find_brats_cases(args.train_dir)
random.shuffle(all_brats)
n_val       = max(1, int(len(all_brats) * args.val_frac))
val_cases   = all_brats[:n_val]
brats_train = all_brats[n_val:]
print(f"BraTS  — train: {len(brats_train)}  val: {len(val_cases)}")

# ── discover HCP cases ─────────────────────────────────────────────────────
hcp_train = []
if args.hcp_dir and os.path.isdir(args.hcp_dir):
    hcp_files = sorted(glob.glob(os.path.join(args.hcp_dir, "*_preprocessed.nii.gz")))
    hcp_train = [{"name": os.path.basename(f).replace("_preprocessed.nii.gz",""),
                  "t1":   f,
                  "mask": None,       # no real mask for healthy brains
                  "source": "hcp"}
                 for f in hcp_files]
    print(f"HCP    — train: {len(hcp_train)}")
else:
    print("HCP    — not used")

train_cases = brats_train + hcp_train
random.shuffle(train_cases)

# ── collect BraTS seg masks for 'morphed' / 'mixed' ───────────────────────
brats_seg_paths = []
if args.aug_strategy in ("morphed", "mixed"):
    for c in brats_train:
        seg = c["mask"]   # unhealthy mask  (close enough for shape donor)
        if seg and os.path.exists(seg):
            brats_seg_paths.append(seg)
    print(f"Morphed donor masks: {len(brats_seg_paths)}")

# Save val split
with open(os.path.join(args.output_dir, "val_samples.json"), "w") as f:
    json.dump({"val_samples": [c["name"] for c in val_cases],
               "val_root": args.train_dir}, f, indent=2)

# ── helpers ────────────────────────────────────────────────────────────────
def normalise(v):
    lo, hi = np.percentile(v, 1), np.percentile(v, 99)
    return np.clip((v - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)

def load_case(c):
    t1 = nib.load(c["t1"]).get_fdata(dtype=np.float32)
    t1 = normalise(t1)
    if c["mask"]:
        mask = nib.load(c["mask"]).get_fdata(dtype=np.float32)
        mask = (mask > 0.5).astype(np.float32)
    else:
        mask = None
    return t1, mask

# ── dataset ────────────────────────────────────────────────────────────────
class AblationDataset(Dataset):
    def __init__(self, cases, patch_size, strategy, aug_prob,
                 brats_seg_paths=None, augment=True):
        self.cases           = cases
        self.patch_size      = patch_size
        self.strategy        = strategy
        self.aug_prob        = aug_prob
        self.brats_seg_paths = brats_seg_paths or []
        self.augment         = augment
        self.rng             = np.random.default_rng()

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        c = self.cases[idx]
        t1, real_mask = load_case(c)
        pd, ph, pw    = self.patch_size
        D, H, W       = t1.shape

        # ── void selection ─────────────────────────────────────────────────
        if self.strategy == "none":
            # Use real BraTS mask (HCP samples get a zero mask → skip void)
            if real_mask is not None:
                mask = real_mask
            else:
                # HCP healthy brain: fall back to a small ellipsoid so
                # the sample still contributes signal
                mask = get_void_mask("ellipsoid", (D, H, W),
                                     aug_prob=1.0, rng=self.rng).astype(np.float32)
        else:
            mask_synth = get_void_mask(
                self.strategy, (D, H, W),
                aug_prob=self.aug_prob,
                brats_mask_paths=self.brats_seg_paths if self.brats_seg_paths else None,
                rng=self.rng,
            ).astype(np.float32)

            if mask_synth.sum() < 10 and real_mask is not None:
                # aug_prob gate fired → use real mask
                mask = real_mask
            elif mask_synth.sum() < 10:
                # HCP + gate fired → tiny random ellipsoid as fallback
                mask = get_void_mask("ellipsoid", (D, H, W),
                                     aug_prob=1.0, rng=self.rng).astype(np.float32)
            else:
                mask = mask_synth

        # ── random patch ───────────────────────────────────────────────────
        d0 = random.randint(0, max(0, D - pd))
        h0 = random.randint(0, max(0, H - ph))
        w0 = random.randint(0, max(0, W - pw))
        t1_p   = t1  [d0:d0+pd, h0:h0+ph, w0:w0+pw]
        mask_p = mask[d0:d0+pd, h0:h0+ph, w0:w0+pw]

        def pad3(x, tgt):
            pw_ = [(0, max(0, t - s)) for s, t in zip(x.shape, tgt)]
            return np.pad(x, pw_, mode="reflect") if any(p[1] > 0 for p in pw_) else x

        t1_p   = pad3(t1_p,   (pd, ph, pw))
        mask_p = pad3(mask_p, (pd, ph, pw))

        voided = t1_p.copy()
        voided[mask_p > 0.5] = 0.0

        # ── spatial / intensity augmentation ──────────────────────────────
        if self.augment:
            for ax in range(3):
                if random.random() < 0.5:
                    t1_p   = np.flip(t1_p,   axis=ax).copy()
                    mask_p = np.flip(mask_p, axis=ax).copy()
                    voided = np.flip(voided, axis=ax).copy()
            if random.random() < 0.3:
                alpha = random.uniform(0.8, 1.2)
                beta  = random.uniform(-0.1, 0.1)
                t1_p   = np.clip(t1_p   * alpha + beta, 0, 1)
                voided = np.clip(voided * alpha + beta, 0, 1)
                voided[mask_p > 0.5] = 0.0

        inp = np.stack([voided, mask_p], axis=0)   # (2, D, H, W)
        return {
            "input":  torch.from_numpy(inp),
            "target": torch.from_numpy(t1_p[None]),
            "mask":   torch.from_numpy(mask_p[None]),
        }

train_ds = AblationDataset(
    train_cases, args.patch_size,
    strategy=args.aug_strategy,
    aug_prob=args.aug_prob,
    brats_seg_paths=brats_seg_paths,
    augment=True,
)
train_loader = DataLoader(
    train_ds, batch_size=args.batch_size,
    shuffle=True, num_workers=2, pin_memory=True,
    prefetch_factor=2, persistent_workers=True,
)

# ── model ──────────────────────────────────────────────────────────────────
cfg   = EvalCfg()
model = MODEL_REGISTRY[args.model](cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model '{args.model}': {n_params/1e6:.1f}M params")

# ── optimiser + AMP ────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.n_iters, eta_min=args.lr * 0.01
)
scaler = GradScaler()

# ── loss ───────────────────────────────────────────────────────────────────
def ssim_loss(pred, target, mask=None):
    C1, C2 = 0.01**2, 0.03**2
    mu_p  = F.avg_pool3d(pred,        kernel_size=11, stride=1, padding=5)
    mu_t  = F.avg_pool3d(target,      kernel_size=11, stride=1, padding=5)
    mu_pp = F.avg_pool3d(pred**2,     kernel_size=11, stride=1, padding=5) - mu_p**2
    mu_tt = F.avg_pool3d(target**2,   kernel_size=11, stride=1, padding=5) - mu_t**2
    mu_pt = F.avg_pool3d(pred*target, kernel_size=11, stride=1, padding=5) - mu_p*mu_t
    ssim_map = ((2*mu_p*mu_t + C1) * (2*mu_pt + C2)) / \
               ((mu_p**2 + mu_t**2 + C1) * (mu_pp + mu_tt + C2) + 1e-8)
    if mask is not None:
        return 1.0 - (ssim_map * mask).sum() / (mask.sum() + 1e-8)
    return 1.0 - ssim_map.mean()


def masked_mse(pred, target, mask):
    """MSE computed only inside the void region (compositing trick)."""
    n = mask.sum().clamp(min=1)
    return ((pred - target)**2 * mask).sum() / n


def compute_loss(pred, target, mask, masked_w=0.5):
    """
    Combined loss:
      - Full-image L1 + SSIM   (standard reconstruction signal everywhere)
      - Masked MSE             (focused gradient inside void, winner-paper trick)

    masked_w=0   → pure full-image loss (original behaviour)
    masked_w=0.5 → balanced
    masked_w=1   → masked loss only
    """
    full_w = 1.0 - masked_w

    loss = 0.0
    if full_w > 0:
        loss = loss + full_w * (
            0.6 * F.l1_loss(pred, target) +
            0.4 * ssim_loss(pred, target, mask)
        )
    if masked_w > 0:
        loss = loss + masked_w * masked_mse(pred, target, mask)
    return loss

# ── validation ─────────────────────────────────────────────────────────────
def sliding_window(model, t1, mask, patch_size, overlap=0.5):
    pd, ph, pw = patch_size
    D, H, W    = t1.shape
    voided = t1.copy(); voided[mask > 0.5] = 0.0

    count = np.zeros_like(t1)
    accum = np.zeros_like(t1)
    stride = [max(1, int(p * (1 - overlap))) for p in patch_size]

    with torch.no_grad():
        for d in range(0, max(1, D - pd + 1), stride[0]):
            for h in range(0, max(1, H - ph + 1), stride[1]):
                for w in range(0, max(1, W - pw + 1), stride[2]):
                    d1 = min(d + pd, D); h1 = min(h + ph, H); w1 = min(w + pw, W)
                    d0 = d1 - pd;        h0 = h1 - ph;        w0 = w1 - pw
                    inp = torch.from_numpy(
                        np.stack([voided[d0:d1,h0:h1,w0:w1],
                                  mask  [d0:d1,h0:h1,w0:w1]], axis=0)
                    ).float().unsqueeze(0).to(device)
                    with autocast():
                        pred = model({"input": inp})[0, 0].cpu().numpy()
                    accum[d0:d1,h0:h1,w0:w1] += pred
                    count[d0:d1,h0:h1,w0:w1] += 1
    return np.clip(accum / (count + 1e-8), 0, 1)


def validate(n_cases=20):
    model.eval()
    results = []
    for c in val_cases[:n_cases]:
        t1, mask = load_case(c)
        if mask is None or not (mask > 0.5).any():
            continue
        pred = sliding_window(model, t1, mask, args.patch_size, overlap=0.25)
        m    = mask > 0.5
        mse  = float(np.mean((pred[m] - t1[m])**2))
        coords = np.argwhere(m)
        d0,h0,w0 = coords.min(axis=0); d1,h1,w1 = coords.max(axis=0) + 1
        pad = 8; D, H, W = t1.shape
        d0,h0,w0 = max(0,d0-pad), max(0,h0-pad), max(0,w0-pad)
        d1,h1,w1 = min(D,d1+pad), min(H,h1+pad), min(W,w1+pad)
        ssim_v = float(structural_similarity(
            t1[d0:d1,h0:h1,w0:w1], pred[d0:d1,h0:h1,w0:w1], data_range=1.0))
        results.append({"ssim": ssim_v, "psnr": 10*math.log10(1/(mse+1e-10)), "mse": mse})
    model.train()
    return (float(np.mean([r["ssim"] for r in results])),
            float(np.mean([r["psnr"] for r in results])),
            float(np.mean([r["mse"]  for r in results])))

# ── checkpoint helpers ─────────────────────────────────────────────────────
def save_ckpt(path, iteration, best_ssim):
    torch.save({
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration":       iteration,
        "best_ssim":       best_ssim,
        "aug_strategy":    args.aug_strategy,
    }, path)

def load_ckpt(path):
    ckpt = torch.load(path, map_location=device)
    key  = next((k for k in ["model_state","model_state_dict","state_dict"] if k in ckpt), None)
    model.load_state_dict(ckpt[key] if key else ckpt, strict=False)
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt.get("iteration", 0), ckpt.get("best_ssim", 0.0)

# ── resume ─────────────────────────────────────────────────────────────────
start_iter = 0; best_ssim = 0.0
latest_pt  = os.path.join(args.output_dir, "latest.pt")
best_pt    = os.path.join(args.output_dir, "best.pt")
hist_path  = os.path.join(args.output_dir, "history.json")
history    = {"iter": [], "val_ssim": [], "val_psnr": [], "val_mse": []}

if os.path.exists(latest_pt):
    try:
        start_iter, best_ssim = load_ckpt(latest_pt)
        if os.path.exists(hist_path):
            history = json.load(open(hist_path))
        print(f"Resumed from iter {start_iter}, best SSIM={best_ssim:.4f}")
    except Exception as e:
        print(f"Could not resume: {e}")

# ── training loop ──────────────────────────────────────────────────────────
model.train()
data_iter  = iter(train_loader)
iter_count = start_iter
t0 = time.time()

print(f"\nStarting from iter {start_iter}, target {args.n_iters}\n")

while iter_count < args.n_iters:
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
        pred = model({"input": inp})
        loss = compute_loss(pred, target, mask, masked_w=args.masked_loss_w)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    iter_count += 1

    if iter_count % 100 == 0:
        elapsed = time.time() - t0
        its   = (iter_count - start_iter) / elapsed
        eta_h = (args.n_iters - iter_count) / (its * 3600 + 1e-8)
        print(f"[{iter_count:>7}/{args.n_iters}] loss={loss.item():.4f} | "
              f"{its:.1f}it/s | ETA {eta_h:.1f}h | lr={optimizer.param_groups[0]['lr']:.2e}")

    if iter_count % args.val_interval == 0:
        print(f"\n  Validating at iter {iter_count}...")
        ssim_v, psnr_v, mse_v = validate()
        print(f"  Val [{iter_count}] SSIM={ssim_v:.4f}  PSNR={psnr_v:.2f}  MSE={mse_v:.6f}")

        history["iter"].append(iter_count)
        history["val_ssim"].append(ssim_v)
        history["val_psnr"].append(psnr_v)
        history["val_mse"].append(mse_v)
        json.dump(history, open(hist_path, "w"))

        save_ckpt(latest_pt, iter_count, best_ssim)
        if ssim_v > best_ssim:
            best_ssim = ssim_v
            save_ckpt(best_pt, iter_count, best_ssim)
            print(f"  ** New best SSIM: {best_ssim:.4f} — saved best.pt")

print(f"\nTraining complete. Best SSIM={best_ssim:.4f}")
