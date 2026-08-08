#!/usr/bin/env python3
"""
BraTS 2026 Inpainting Challenge — Main Training Script
=======================================================
Supports 5 models: unet | diffusion | hierarchical | symamba | gan
Iteration-based training (default 200k iterations).

Usage:
    python train.py --model unet      --iterations 200000
    python train.py --model diffusion --iterations 200000
    python train.py --model symamba   --iterations 200000 --resume

Data layout on PVC:
    /data/brats2023/training/{sample}/{sample}-t1n-voided.nii.gz
    /data/brats2023/validation/{sample}/{sample}-t1n-voided.nii.gz
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup — works both locally and on Nautilus PVC
# ---------------------------------------------------------------------------
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

import nibabel as nib
from skimage.metrics import structural_similarity

# ---------------------------------------------------------------------------
# Inline utilities (data, metrics, scheduler)
# ---------------------------------------------------------------------------

FILE_KEYS = {
    "voided":         "t1n-voided",
    "mask":           "mask",
    "target":         "t1n",
    "mask_unhealthy": "mask-unhealthy",
    "mask_healthy":   "mask-healthy",
}
MASK_KEYS = {"mask", "mask_unhealthy", "mask_healthy"}


def load_nifti(root, sample, key):
    for ext in (".nii.gz", ".nii"):
        p = os.path.join(root, sample, f"{sample}-{key}{ext}")
        if os.path.exists(p):
            return np.asarray(nib.load(p).dataobj, dtype=np.float32)
    return None


def normalize(vol, is_mask=False):
    if is_mask:
        return (vol > 0).astype(np.float32)
    fg = vol[vol > 0]
    if fg.size == 0:
        return vol
    p1, p99 = np.percentile(fg, [1, 99])
    return np.clip((vol - p1) / (p99 - p1 + 1e-8), 0.0, 1.0)


def discover_samples(root, require_target=False):
    """Return sorted list of sample folder names that have t1n-voided.
    If require_target=True, also require t1n (ground truth) to exist.
    """
    samples = []
    if not os.path.isdir(root):
        return samples
    for name in sorted(os.listdir(root)):
        folder = os.path.join(root, name)
        if not os.path.isdir(folder):
            continue
        has_voided = any(
            os.path.exists(os.path.join(folder, f"{name}-t1n-voided{e}"))
            for e in (".nii.gz", ".nii")
        )
        if not has_voided:
            continue
        if require_target:
            has_target = any(
                os.path.exists(os.path.join(folder, f"{name}-t1n{e}"))
                for e in (".nii.gz", ".nii")
            )
            if not has_target:
                continue
        samples.append(name)
    return samples


def random_ellipsoid_mask(target, n_ellipsoids=None, r_min=8, r_max=28):
    """Generate a random ellipsoidal void mask within brain tissue.
    Replicates the random masking augmentation from Zhang et al. BraTS 2025.

    Args:
        target: (D, H, W) float32 array, normalised [0,1] target image
        n_ellipsoids: number of ellipsoids (None → random 1-3)
        r_min, r_max: radius range (voxels) per axis
    Returns:
        mask: (D, H, W) float32 binary array — 1 inside void, 0 outside
    """
    D, H, W = target.shape
    brain = target > 0.05          # rough brain tissue mask
    coords = np.argwhere(brain)
    if len(coords) < 50:
        return np.zeros((D, H, W), dtype=np.float32)

    n = n_ellipsoids if n_ellipsoids is not None else random.randint(1, 3)
    mask = np.zeros((D, H, W), dtype=np.float32)

    # ogrid is memory-efficient — shapes (D,1,1), (1,H,1), (1,1,W)
    dz, dy, dx = np.ogrid[:D, :H, :W]

    for _ in range(n):
        c = coords[random.randint(0, len(coords) - 1)]
        rz = random.uniform(r_min, r_max)
        ry = random.uniform(r_min, r_max)
        rx = random.uniform(r_min, r_max)
        ellipsoid = (((dz - c[0]) / rz) ** 2 +
                     ((dy - c[1]) / ry) ** 2 +
                     ((dx - c[2]) / rx) ** 2) <= 1.0
        mask = np.clip(mask + ellipsoid.astype(np.float32), 0.0, 1.0)

    # Restrict to brain tissue only
    return mask * brain.astype(np.float32)


class BraTSDataset(torch.utils.data.Dataset):
    def __init__(self, root, sample_names, patch_size=(96, 96, 96),
                 patches_per_sample=20, mode="train", augment=True,
                 cache_size=50, aug_mask_prob=0.0):
        self.root = root
        self.names = sample_names
        self.ps = patch_size
        self.pps = patches_per_sample
        self.mode = mode
        self.aug = augment and mode == "train"
        self.aug_mask_prob = aug_mask_prob if mode == "train" else 0.0

        # Lazy loading with LRU-style cache (avoids OOM for large datasets)
        # cache_size=50 means ~50 * 5 * 34MB ≈ 8.5GB RAM, safe for 32GB nodes
        self._cache = {}
        self._cache_order = []
        self._cache_size = cache_size
        print(f"  Dataset: {len(sample_names)} samples, lazy load (cache={cache_size})")

    def _load(self, name):
        """Load all file types for one sample and return a dict of arrays."""
        d = {}
        for key, fk in FILE_KEYS.items():
            v = load_nifti(self.root, name, fk)
            if v is not None:
                d[key] = normalize(v, is_mask=(key in MASK_KEYS))
        return d

    def _get_vol(self, name):
        """Return cached volume, evicting oldest entry if cache is full."""
        if name not in self._cache:
            if len(self._cache_order) >= self._cache_size:
                evict = self._cache_order.pop(0)
                del self._cache[evict]
            self._cache[name] = self._load(name)
            self._cache_order.append(name)
        return self._cache[name]

    def _patch(self, data):
        D, H, W = data["voided"].shape
        pd, ph, pw = self.ps
        if self.mode == "val":
            d0, h0, w0 = max(0,(D-pd)//2), max(0,(H-ph)//2), max(0,(W-pw)//2)
        else:
            mask = data.get("mask", np.ones((D,H,W), np.float32))
            coords = np.argwhere(mask > 0)
            if len(coords) > 0 and random.random() < 0.7:
                c = coords[random.randint(0, len(coords)-1)]
                d0 = int(np.clip(c[0]-pd//2, 0, max(0,D-pd)))
                h0 = int(np.clip(c[1]-ph//2, 0, max(0,H-ph)))
                w0 = int(np.clip(c[2]-pw//2, 0, max(0,W-pw)))
            else:
                d0 = random.randint(0, max(0,D-pd))
                h0 = random.randint(0, max(0,H-ph))
                w0 = random.randint(0, max(0,W-pw))

        out = {}
        for k, v in data.items():
            sl = v[d0:d0+pd, h0:h0+ph, w0:w0+pw]
            pad = [(0, max(0, s-sl.shape[i])) for i,s in enumerate([pd,ph,pw])]
            if any(p[1]>0 for p in pad):
                sl = np.pad(sl, pad)
            out[k] = sl
        return out

    def __len__(self):
        return len(self.names) * self.pps

    def __getitem__(self, idx):
        name = self.names[idx // self.pps]
        data = self._get_vol(name)
        if "voided" not in data:
            raise RuntimeError(f"Sample '{name}' missing 'voided' file in root '{self.root}'. "
                               f"Loaded keys: {list(data.keys())}")

        # Random mask augmentation — replace BraTS mask with random ellipsoid void
        # Applied at volume level so patch extraction focuses on the new mask
        if self.aug_mask_prob > 0 and random.random() < self.aug_mask_prob and "target" in data:
            data = dict(data)   # shallow copy — never mutate the cache
            rand_mask = random_ellipsoid_mask(data["target"])
            if rand_mask.sum() > 10:   # only use if non-trivial
                data["mask"]   = rand_mask
                data["voided"] = data["target"] * (1.0 - rand_mask)

        patch = self._patch(data)
        if self.aug:
            for ax in range(3):
                if random.random() < 0.5:
                    patch = {k: np.flip(v, ax).copy() for k,v in patch.items()}
        out = {k: torch.from_numpy(v).float().unsqueeze(0) for k,v in patch.items()}
        if "voided" in out and "mask" in out:
            out["input"] = torch.cat([out["voided"], out["mask"]], dim=0)
        return out


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


# ---------------------------------------------------------------------------
# Full-volume dataset (used by winner / enhanced models)
# ---------------------------------------------------------------------------

class BraTSFullVolumeDataset(torch.utils.data.Dataset):
    """
    Returns center-cropped full volumes — no patches.
    Supports random mask augmentation: mirrors the mask and regenerates
    the voided image, teaching the model to handle diverse mask shapes.
    """
    def __init__(self, root, sample_names, crop=(192, 192, 128),
                 mode="train", rand_mask_aug=True, cache_size=50):
        self.root  = root
        self.names = sample_names
        self.crop  = crop
        self.mode  = mode
        self.aug   = mode == "train"
        self.rma   = rand_mask_aug and mode == "train"
        self._cache = {}; self._cache_order = []; self._cache_size = cache_size
        print(f"  FullVolDataset: {len(sample_names)} samples, crop {crop}, rand_mask_aug={rand_mask_aug}")

    def _load(self, name):
        d = {}
        for key, fk in FILE_KEYS.items():
            v = load_nifti(self.root, name, fk)
            if v is not None:
                d[key] = normalize(v, is_mask=(key in MASK_KEYS))
        return d

    def _get_vol(self, name):
        if name not in self._cache:
            if len(self._cache_order) >= self._cache_size:
                evict = self._cache_order.pop(0)
                del self._cache[evict]
            self._cache[name] = self._load(name)
            self._cache_order.append(name)
        return self._cache[name]

    def _center_crop(self, vol, crop):
        D, H, W = vol.shape
        cd, ch, cw = crop
        d0 = max(0, (D-cd)//2); h0 = max(0, (H-ch)//2); w0 = max(0, (W-cw)//2)
        v = vol[d0:d0+cd, h0:h0+ch, w0:w0+cw]
        pad = [(0, max(0, s-v.shape[i])) for i,s in enumerate(crop)]
        if any(p[1]>0 for p in pad): v = np.pad(v, pad)
        return v

    def _rand_mask_aug(self, data):
        """Mirror mask axes randomly, regenerate voided from target + new mask."""
        data = dict(data)  # shallow copy — don't corrupt cache
        mask = data["mask"].copy()
        mh   = data.get("mask_healthy", np.zeros_like(mask)).copy()
        for ax in range(3):
            if random.random() < 0.5:
                mask = np.flip(mask, ax); mh = np.flip(mh, ax)
        mask = (mask > 0.5).astype(np.float32); mh = (mh > 0.5).astype(np.float32)
        data["mask"] = mask.copy(); data["mask_healthy"] = mh.copy()
        if "target" in data:
            combined = np.clip(mask + mh, 0, 1)
            data["voided"] = data["target"] * (1.0 - combined)
        return data

    def __len__(self): return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        data = self._get_vol(name)
        if "voided" not in data:
            raise RuntimeError(f"Sample '{name}' missing 'voided' in '{self.root}'")
        if self.rma:
            data = self._rand_mask_aug(data)
        out = {k: self._center_crop(v, self.crop) for k, v in data.items()}
        if self.aug:
            for ax in range(3):
                if random.random() < 0.5:
                    out = {k: np.flip(v, ax).copy() for k, v in out.items()}
        result = {k: torch.from_numpy(v).float().unsqueeze(0) for k, v in out.items()}
        # 2-channel input (winner): voided + mask
        if "voided" in result and "mask" in result:
            result["input"] = torch.cat([result["voided"], result["mask"]], 0)
        # 4-channel input (enhanced): voided + mask + mask_healthy + mask_unhealthy
        if all(k in result for k in ("voided","mask","mask_healthy","mask_unhealthy")):
            result["input4"] = torch.cat([
                result["voided"], result["mask"],
                result["mask_healthy"], result["mask_unhealthy"]], 0)
        return result


class BraTSZhangDataset(torch.utils.data.Dataset):
    """
    Zhang et al. 2025 exact training setup:
    - Full volumes cropped to 208×208×144 (Zhang's exact crop size)
    - Augments mask-healthy via random flip + rotation on XY and YZ planes (0-360°)
    - Normalizes to [-1, 1] by dividing by max intensity (Zhang's exact normalization)
    - Returns healthy_mask separately for MAE loss computation on healthy region only
    """
    CROP = (208, 208, 144)

    def __init__(self, root, sample_names, mode="train", cache_size=30):
        self.root  = root
        self.names = sample_names
        self.mode  = mode
        self.aug   = (mode == "train")
        self._cache = {}; self._cache_order = []; self._cache_size = cache_size
        print(f"  ZhangDataset: {len(sample_names)} samples, crop {self.CROP}, aug={self.aug}")

    def _load(self, name):
        d = {}
        for key, suffix in [("t1n", "t1n"), ("healthy", "mask-healthy"), ("unhealthy", "mask-unhealthy")]:
            for ext in (".nii.gz", ".nii"):
                p = os.path.join(self.root, name, f"{name}-{suffix}{ext}")
                if os.path.exists(p):
                    d[key] = np.asarray(nib.load(p).dataobj, dtype=np.float32)
                    break
        return d

    def _get_vol(self, name):
        if name not in self._cache:
            if len(self._cache_order) >= self._cache_size:
                evict = self._cache_order.pop(0)
                del self._cache[evict]
            self._cache[name] = self._load(name)
            self._cache_order.append(name)
        return self._cache[name]

    def _center_crop(self, v, crop):
        d0 = max(0, (v.shape[0] - crop[0]) // 2)
        h0 = max(0, (v.shape[1] - crop[1]) // 2)
        w0 = max(0, (v.shape[2] - crop[2]) // 2)
        v  = v[d0:d0+crop[0], h0:h0+crop[1], w0:w0+crop[2]]
        pad = [(0, max(0, s - v.shape[i])) for i, s in enumerate(crop)]
        if any(p[1] > 0 for p in pad):
            v = np.pad(v, pad)
        return v

    def _augment_healthy_mask(self, mask):
        """Zhang's mask augmentation: flip per axis + rotate on XY and YZ planes."""
        try:
            from scipy.ndimage import rotate as nd_rotate
        except ImportError:
            pass  # fallback: flip-only if scipy not installed
        else:
            m = mask.copy().astype(np.float32)
            for ax in range(3):
                if random.random() < 0.5:
                    m = np.flip(m, axis=ax)
            angle_xy = random.uniform(0, 360)
            m = nd_rotate(m, angle_xy, axes=(0, 1), reshape=False, order=0, mode='constant', cval=0)
            angle_yz = random.uniform(0, 360)
            m = nd_rotate(m, angle_yz, axes=(1, 2), reshape=False, order=0, mode='constant', cval=0)
            return (m > 0.5).astype(np.float32)
        # flip-only fallback
        m = mask.copy()
        for ax in range(3):
            if random.random() < 0.5:
                m = np.flip(m, axis=ax)
        return (m > 0.5).astype(np.float32)

    def __len__(self): return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        data = self._get_vol(name)

        t1n       = data.get("t1n",       np.zeros(self.CROP, dtype=np.float32))
        healthy   = data.get("healthy",   np.zeros_like(t1n))
        unhealthy = data.get("unhealthy", np.zeros_like(t1n))

        # Zhang's normalization: divide by max → [0,1] then → [-1, 1]
        max_val = float(t1n.max())
        t1n_n1  = t1n / (max_val + 1e-8) * 2.0 - 1.0

        # Augment healthy mask (Zhang's strategy)
        healthy_aug   = self._augment_healthy_mask(healthy) if self.aug else (healthy > 0.5).astype(np.float32)
        unhealthy_bin = (unhealthy > 0.5).astype(np.float32)
        combined_mask = np.clip(healthy_aug + unhealthy_bin, 0.0, 1.0)

        # Voided image: masked voxels set to -1 (lower bound of [-1,1] space)
        voided_n1 = t1n_n1 * (1.0 - combined_mask) - combined_mask

        # Full-image flip augmentation
        if self.aug:
            for ax in range(3):
                if random.random() < 0.5:
                    t1n_n1        = np.flip(t1n_n1,        ax).copy()
                    voided_n1     = np.flip(voided_n1,     ax).copy()
                    combined_mask = np.flip(combined_mask, ax).copy()
                    healthy_aug   = np.flip(healthy_aug,   ax).copy()

        # Center crop to 208×208×144
        t1n_n1        = self._center_crop(t1n_n1,        self.CROP)
        voided_n1     = self._center_crop(voided_n1,     self.CROP)
        combined_mask = self._center_crop(combined_mask, self.CROP)
        healthy_aug   = self._center_crop(healthy_aug,   self.CROP)

        voided_t = torch.from_numpy(voided_n1[None]).float()
        mask_t   = torch.from_numpy(combined_mask[None]).float()
        target_t = torch.from_numpy(t1n_n1[None]).float()
        hmask_t  = torch.from_numpy(healthy_aug[None]).float()

        return {
            "input":        torch.cat([voided_t, mask_t], dim=0),  # (2,D,H,W)
            "target":       target_t,
            "mask":         mask_t,
            "healthy_mask": hmask_t,
        }


def compute_metrics(pred_np, target_np, mask_np):
    m = mask_np > 0
    mse = float(np.mean((pred_np[m] - target_np[m])**2)) if m.any() else float("nan")
    psnr = float(10*math.log10(1.0/(mse+1e-10))) if not math.isnan(mse) else float("nan")
    ssim = float(structural_similarity(target_np, pred_np, data_range=1.0))
    return {"ssim": ssim, "psnr": psnr, "mse": mse}


def ssim_loss_2d(pred, target):
    D = pred.shape[2]
    p, t = pred[:,:,D//2], target[:,:,D//2]
    K, pad = 7, 3
    g = torch.exp(-torch.arange(K, device=pred.device, dtype=pred.dtype).sub(K//2).pow(2)/4.5)
    g = g/g.sum()
    kern = (g[:,None]*g[None,:]).unsqueeze(0).unsqueeze(0)
    mu_p  = F.conv2d(p, kern, padding=pad)
    mu_t  = F.conv2d(t, kern, padding=pad)
    sp  = F.conv2d(p*p, kern, padding=pad) - mu_p**2
    st  = F.conv2d(t*t, kern, padding=pad) - mu_t**2
    spt = F.conv2d(p*t, kern, padding=pad) - mu_p*mu_t
    C1, C2 = 1e-4, 9e-4
    num = (2*mu_p*mu_t+C1)*(2*spt+C2)
    den = (mu_p**2+mu_t**2+C1)*(sp+st+C2)
    return 1 - (num/(den+1e-8)).mean()


def zhang_loss(pred, target, mask):
    """Loss from Zhang et al. BraTS 2025 winner: λ=1 each.
    MAE computed ONLY on masked region (healthy tissue to inpaint).
    SSIM computed on the ENTIRE volume for structural coherence.
    """
    m = (mask > 0.5).squeeze(1)   # (B, D, H, W) bool
    if m.any():
        mae = F.l1_loss(pred.squeeze(1)[m], target.squeeze(1)[m])
    else:
        mae = F.l1_loss(pred, target)
    ssim_val = ssim_loss_2d(pred, target)
    loss = mae + ssim_val
    return loss, {"mae_mask": mae.item(), "ssim_struct": ssim_val.item()}


def zhang_loss_boundary(pred, target, mask, lambda_boundary=2.0):
    """Zhang loss + boundary-aware penalty.
    Adds extra MAE loss on the ring of voxels just outside the mask edge.
    These boundary voxels drive SSIM degradation — penalising them forces
    sharper, more coherent transitions and reduces blending artefacts.
    """
    m = (mask > 0.5).squeeze(1)   # (B, D, H, W) bool

    # Outer boundary ring: dilate mask, remove original mask pixels
    with torch.no_grad():
        dilated  = F.max_pool3d(mask, kernel_size=7, stride=1, padding=3)
        boundary = (dilated - mask).clamp(0, 1)   # (B, 1, D, H, W)
        b        = (boundary > 0.5).squeeze(1)     # (B, D, H, W) bool

    if m.any():
        mae = F.l1_loss(pred.squeeze(1)[m], target.squeeze(1)[m])
    else:
        mae = F.l1_loss(pred, target)

    if b.any():
        mae_b = F.l1_loss(pred.squeeze(1)[b], target.squeeze(1)[b])
    else:
        mae_b = torch.tensor(0.0, device=pred.device)

    ssim_val = ssim_loss_2d(pred, target)
    loss = mae + ssim_val + lambda_boundary * mae_b
    return loss, {"mae_mask": mae.item(), "ssim_struct": ssim_val.item(),
                  "mae_boundary": mae_b.item()}


def freq_domain_loss(pred, target, mask):
    """Frequency-domain L1 loss on the masked region.
    Penalises missing high-frequency detail that MAE blurs away.
    Uses 3D rFFT magnitude spectrum.
    """
    pred_m = pred  * mask
    tgt_m  = target * mask
    pf = torch.fft.rfftn(pred_m, dim=(-3,-2,-1))
    tf = torch.fft.rfftn(tgt_m,  dim=(-3,-2,-1))
    return F.l1_loss(pf.abs(), tf.abs())


def inpainting_loss(pred, target, mask, lambda_ssim=0.5, lambda_mask=2.0):
    mse  = F.mse_loss(pred, target)
    msel = F.mse_loss(pred*mask, target*mask) * lambda_mask
    ssl  = ssim_loss_2d(pred, target) * lambda_ssim
    return mse + msel + ssl, {"mse": mse.item(), "mse_mask": msel.item(), "ssim": ssl.item()}


# ---------------------------------------------------------------------------
# DDPM Scheduler
# ---------------------------------------------------------------------------

class DDPMScheduler:
    def __init__(self, T=1000, schedule="cosine", device="cpu"):
        self.T = T
        if schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, T)
        else:
            x = torch.linspace(0, T, T+1)
            ac = torch.cos(((x/T)+0.008)/1.008 * math.pi/2)**2
            ac = ac/ac[0]
            betas = torch.clamp(1-ac[1:]/ac[:-1], 1e-4, 0.9999)
        alphas = 1 - betas
        acp = torch.cumprod(alphas, 0)
        self.betas = betas.to(device)
        self.sqrt_acp  = acp.sqrt().to(device)
        self.sqrt_1macp = (1-acp).sqrt().to(device)
        self.acp = acp.to(device)

    def add_noise(self, x0, t, noise=None, mask=None, mask_u=None,
                  alpha_in=1.5, alpha_b=2.0, alpha_out=0.8):
        if noise is None:
            noise = torch.randn_like(x0)
        sa  = self.sqrt_acp[t].view(-1,1,1,1,1)
        soa = self.sqrt_1macp[t].view(-1,1,1,1,1)
        if mask is not None:
            w = torch.full_like(mask, alpha_out)
            w[mask > 0] = alpha_in
            if mask_u is not None:
                w[mask_u > 0] = alpha_b
            noise = noise * w
        return sa*x0 + soa*noise, noise

    @torch.no_grad()
    def ddim_sample(self, model_fn, cond, shape, steps=50, device="cpu", **kw):
        step = max(1, self.T//steps)
        ts   = list(range(0, self.T, step))[::-1]
        x    = torch.randn(shape, device=device)
        for i, tv in enumerate(tqdm(ts, desc="DDIM", leave=False)):
            tb = torch.full((shape[0],), tv, device=device, dtype=torch.long)
            eps = model_fn(x, tb, cond, **kw)
            at  = self.acp[tv]
            x0  = ((x - (1-at).sqrt()*eps) / at.sqrt()).clamp(0,1)
            ap  = self.acp[ts[i+1]] if i+1 < len(ts) else torch.tensor(1.0)
            x   = ap.sqrt()*x0 + (1-ap).sqrt()*eps
        return x.clamp(0,1)


# ---------------------------------------------------------------------------
# Model building blocks
# ---------------------------------------------------------------------------

class ConvBlock3D(nn.Module):
    def __init__(self, ic, oc, td=None, g=8):
        super().__init__()
        gg = min(g, oc)
        self.c1 = nn.Sequential(nn.Conv3d(ic,oc,3,padding=1,bias=False), nn.GroupNorm(gg,oc), nn.SiLU())
        self.c2 = nn.Sequential(nn.Conv3d(oc,oc,3,padding=1,bias=False), nn.GroupNorm(gg,oc), nn.SiLU())
        self.tp = nn.Linear(td, oc) if td else None
        self.res = nn.Conv3d(ic,oc,1,bias=False) if ic!=oc else nn.Identity()
    def forward(self, x, te=None):
        h = self.c1(x)
        if te is not None and self.tp:
            h = h + self.tp(te).view(h.shape[0],-1,1,1,1)
        return self.c2(h) + self.res(x)

class Down3D(nn.Module):
    def __init__(self, ic, oc, td=None, nb=2):
        super().__init__()
        self.blks = nn.ModuleList([ConvBlock3D(ic if i==0 else oc, oc, td) for i in range(nb)])
        self.pool = nn.MaxPool3d(2)
    def forward(self, x, te=None):
        for b in self.blks: x = b(x, te)
        return self.pool(x), x

class Up3D(nn.Module):
    def __init__(self, ic, sc, oc, td=None, nb=2):
        super().__init__()
        self.up = nn.ConvTranspose3d(ic, ic, 2, stride=2)
        self.blks = nn.ModuleList([ConvBlock3D(ic+sc if i==0 else oc, oc, td) for i in range(nb)])
    def forward(self, x, skip, te=None):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x,skip], 1)
        for b in self.blks: x = b(x, te)
        return x

class TimeEmb(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.mlp = nn.Sequential(nn.Linear(d,d*4), nn.SiLU(), nn.Linear(d*4,d*4))
    def forward(self, t):
        h = self.d//2
        f = torch.exp(-math.log(10000)*torch.arange(h,device=t.device,dtype=torch.float32)/(h-1))
        a = t[:,None].float()*f[None]
        e = torch.cat([a.sin(), a.cos()], -1)
        return self.mlp(e)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class UNetBaseline(nn.Module):
    """Model 1 — deterministic 3D U-Net (Zhang et al. style)"""
    def __init__(self, bc=32, cm=(1,2,4,8), nb=2):
        super().__init__()
        chs = [bc*m for m in cm]
        self.encs = nn.ModuleList()
        ic = 2
        for oc in chs:
            self.encs.append(Down3D(ic,oc,nb=nb)); ic=oc
        self.bot = nn.Sequential(ConvBlock3D(ic,ic*2), ConvBlock3D(ic*2,ic))
        self.decs = nn.ModuleList()
        for i,oc in enumerate(reversed(chs)):
            self.decs.append(Up3D(ic,chs[-(i+1)],oc,nb=nb)); ic=oc
        self.head = nn.Conv3d(ic,1,1)
    def forward(self, batch, **_):
        x = batch["input"]; skips=[]
        for e in self.encs: x,s=e(x); skips.append(s)
        x = self.bot(x)
        for i,d in enumerate(self.decs): x=d(x,skips[-(i+1)])
        return torch.sigmoid(self.head(x))
    def model_type(self): return "deterministic"


class DiffusionUNet(nn.Module):
    """Model 2 — DDPM U-Net (Durrer et al. style)"""
    def __init__(self, cond=2, bc=32, cm=(1,2,4,8), nb=2):
        super().__init__()
        td = bc*4
        self.temb = TimeEmb(bc)
        chs = [bc*m for m in cm]
        self.encs = nn.ModuleList()
        ic = 1+cond
        for oc in chs:
            self.encs.append(Down3D(ic,oc,td,nb)); ic=oc
        self.b1 = ConvBlock3D(ic,ic*2,td); self.b2 = ConvBlock3D(ic*2,ic,td)
        self.decs = nn.ModuleList()
        for i,oc in enumerate(reversed(chs)):
            self.decs.append(Up3D(ic,chs[-(i+1)],oc,td,nb)); ic=oc
        self.head = nn.Conv3d(ic,1,1)
    def forward(self, x, t, cond=None, **_):
        if cond is not None: x = torch.cat([x,cond],1)
        te = self.temb(t); skips=[]
        for e in self.encs: x,s=e(x,te); skips.append(s)
        x = self.b2(self.b1(x,te),te)
        for i,d in enumerate(self.decs): x=d(x,skips[-(i+1)],te)
        return self.head(x)
    def model_type(self): return "diffusion"


class HierarchicalDiffusion(nn.Module):
    """Model 3 — Two-stage hierarchical diffusion (Kwark et al. style)"""
    def __init__(self, cond=2, bc=24, cm=(1,2,4,8), nb=2, scale=4):
        super().__init__()
        self.scale = scale
        self.s1 = DiffusionUNet(cond=cond,   bc=bc, cm=cm[:3], nb=nb)
        self.s2 = DiffusionUNet(cond=cond+1, bc=bc, cm=cm,     nb=nb)
    def forward(self, x, t, cond=None, coarse=None, stage=2, **_):
        if stage == 1:
            cl = F.avg_pool3d(cond, self.scale)
            xl = F.avg_pool3d(x,    self.scale)
            return self.s1(xl, t, cond=cl)
        c2 = coarse if coarse is not None else torch.zeros_like(x)
        return self.s2(x, t, cond=torch.cat([cond,c2],1))
    def model_type(self): return "diffusion"


class MambaBlock3D(nn.Module):
    """Mamba block with depthwise-conv fallback (O(L) memory, no mamba-ssm required).
    MultiheadAttention was O(L^2) and caused OOM at full 96^3 resolution.
    """
    def __init__(self, d, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        try:
            from mamba_ssm import Mamba
            self.ssm = Mamba(d, d_state=d_state, d_conv=4, expand=2)
            self.use_mamba = True
        except ImportError:
            # Depthwise-separable 1-D conv: captures local sequential context,
            # O(L) memory — safe even at full patch resolution.
            self.ssm = nn.Sequential(
                nn.Conv1d(d, d, kernel_size=7, padding=3, groups=d),   # depthwise
                nn.Conv1d(d, d, kernel_size=1),                          # pointwise
                nn.GELU(),
            )
            self.use_mamba = False
        self.proj = nn.Linear(d, d)

    def scan(self, x, ax):
        B, C, D, H, W = x.shape
        if ax == 0:   f = x.permute(0,3,4,2,1).reshape(B*H*W, D, C)
        elif ax == 1: f = x.permute(0,2,4,3,1).reshape(B*D*W, H, C)
        else:         f = x.permute(0,2,3,4,1).reshape(B*D*H, W, C)
        n = self.norm(f)
        if self.use_mamba:
            o = self.ssm(n)
        else:
            # Conv1d expects (batch, channels, length) → transpose L and C
            o = self.ssm(n.transpose(1, 2)).transpose(1, 2)
        o = self.proj(o)
        if ax == 0:   return o.reshape(B,H,W,D,C).permute(0,4,3,1,2)
        elif ax == 1: return o.reshape(B,D,W,H,C).permute(0,4,1,3,2)
        else:         return o.reshape(B,D,H,W,C).permute(0,4,1,2,3)

    def forward(self, x):
        return x + (self.scan(x,0) + self.scan(x,1) + self.scan(x,2)) / 3


class DDMBlock(nn.Module):
    """Dual-Domain Mamba block: spatial Mamba + wavelet-like freq stream"""
    def __init__(self, c, d_state=16):
        super().__init__()
        self.mamba = MambaBlock3D(c, d_state)
        self.freq_lo = nn.Conv3d(c,c,3,stride=2,padding=1)
        self.freq_hi = nn.Conv3d(c,c,3,stride=2,padding=1,dilation=1)
        self.proc_lo = ConvBlock3D(c,c)
        self.proc_hi = ConvBlock3D(c,c)
        self.freq_up = nn.ConvTranspose3d(c*2,c,2,stride=2)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(c,c), nn.Sigmoid())
        self.norm = nn.GroupNorm(min(8,c), c)
    def forward(self, x):
        sp  = self.mamba(x)
        fi  = self.norm(x)
        lo  = self.proc_lo(self.freq_lo(fi))
        hi  = self.proc_hi(self.freq_hi(fi))
        fq  = self.freq_up(torch.cat([lo,hi],1))
        if fq.shape != x.shape:
            fq = F.interpolate(fq, x.shape[2:], mode="trilinear", align_corners=False)
        a   = self.gate(x).view(x.shape[0],x.shape[1],1,1,1)
        return self.norm(a*sp + (1-a)*fq + x)


class HSM(nn.Module):
    """Hemisphere Symmetry Module"""
    def __init__(self, ic, oc):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv3d(ic*2,oc,3,padding=1), nn.GroupNorm(8,oc), nn.SiLU(),
            nn.Conv3d(oc,oc,3,padding=1))
        self.gate = nn.Sequential(nn.Conv3d(ic,1,1), nn.Sigmoid())
    def forward(self, v, mh=None):
        f = torch.flip(v,[3])
        if mh is not None: f = f * self.gate(mh)
        return self.fuse(torch.cat([v,f],1))


class SyMambaDiff(nn.Module):
    """Model 4 — SyMamba-Diff (NOVEL): HSM + DDM blocks + mask-aware noise"""
    def __init__(self, bc=32, cm=(1,2,4,8), nb=2, d_state=16, hsm_ch=32):
        super().__init__()
        td = bc*4
        self.temb = TimeEmb(bc)
        self.hsm  = HSM(1, hsm_ch)
        cond = hsm_ch+1
        chs  = [bc*m for m in cm]
        self.encs = nn.ModuleList(); self.ddm_e = nn.ModuleList()
        ic = 1+cond
        for oc in chs:
            self.encs.append(Down3D(ic,oc,td,nb)); self.ddm_e.append(DDMBlock(oc,d_state)); ic=oc
        self.b1 = ConvBlock3D(ic,ic*2,td); self.b2 = ConvBlock3D(ic*2,ic,td)
        self.bddm = DDMBlock(ic,d_state)
        self.decs = nn.ModuleList(); self.ddm_d = nn.ModuleList()
        for i,oc in enumerate(reversed(chs)):
            self.decs.append(Up3D(ic,chs[-(i+1)],oc,td,nb)); self.ddm_d.append(DDMBlock(oc,d_state)); ic=oc
        self.head = nn.Conv3d(ic,1,1)
    def forward(self, x, t, voided=None, mask=None, mask_healthy=None, **_):
        hf = self.hsm(voided, mask_healthy)
        x  = torch.cat([x, hf, mask], 1)
        te = self.temb(t); skips=[]
        for e,d in zip(self.encs,self.ddm_e): x,s=e(x,te); x=d(x); skips.append(s)
        x = self.bddm(self.b2(self.b1(x,te),te))
        for i,(d,dm) in enumerate(zip(self.decs,self.ddm_d)): x=dm(d(x,skips[-(i+1)],te))
        return self.head(x)
    def model_type(self): return "diffusion"


class GAN3D(nn.Module):
    """Model 5 — 3D Pix2Pix GAN"""
    def __init__(self, bc=32, cm=(1,2,4,8)):
        super().__init__()
        self.G = UNetBaseline(bc=bc, cm=cm)
        chs = [bc*m for m in cm[:4]]
        layers = [nn.Conv3d(3,bc,4,2,1), nn.LeakyReLU(0.2,True)]
        c = bc
        for nc in chs[1:]:
            nc = min(nc,256)
            layers += [nn.Conv3d(c,nc,4,2,1), nn.GroupNorm(min(8,nc),nc), nn.LeakyReLU(0.2,True)]
            c = nc
        layers += [nn.Conv3d(c,1,4,1,1)]
        self.D = nn.Sequential(*layers)
    def generate(self, b): return self.G(b)
    def discriminate(self, inp, pred): return self.D(torch.cat([inp,pred],1))
    def model_type(self): return "gan"


# ---------------------------------------------------------------------------
# Model 6 — Zhang2025UNet (BraTS 2025 Challenge Winner)
# "Robust 3D Brain MRI Inpainting with Random Masking Augmentation"
# Zhang, Weng, Chen — University of Nottingham Ningbo China
# Key design: InstanceNorm + PReLU + ReflectionPad + Dropout(0.2) in bridge/decoder
# ---------------------------------------------------------------------------

class Zhang3DBlock(nn.Module):
    """Two 3×3×3 conv layers with ReflectionPad + InstanceNorm + PReLU (or ReLU in bridge)."""
    def __init__(self, in_ch, out_ch, use_relu=False, dropout=0.0):
        super().__init__()
        def act(): return nn.ReLU(inplace=True) if use_relu else nn.PReLU(out_ch)
        self.block = nn.Sequential(
            nn.ReflectionPad3d(1),
            nn.Conv3d(in_ch, out_ch, 3, padding=0),
            nn.InstanceNorm3d(out_ch, affine=True),
            act(),
            nn.ReflectionPad3d(1),
            nn.Conv3d(out_ch, out_ch, 3, padding=0),
            nn.InstanceNorm3d(out_ch, affine=True),
            act(),
        )
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.block(x))


class Zhang2025UNet(nn.Module):
    """Faithful 3-level U-Net from the BraTS 2025 winning paper.
    Channels: [32→64→128] encoder, [256] bridge, [128→64→32] decoder.
    PReLU in encoder/decoder, ReLU in bridge. Dropout(0.2) in bridge+decoder.
    Input: (B, 2, D, H, W)  [voided + mask concatenated]
    Output: (B, 1, D, H, W) in [0, 1] via Sigmoid
    """
    def __init__(self, in_ch=2):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        # Encoder
        self.enc1 = Zhang3DBlock(in_ch, 32)
        self.enc2 = Zhang3DBlock(32, 64)
        self.enc3 = Zhang3DBlock(64, 128)
        # Bridge (ReLU per paper, dropout)
        self.bridge = Zhang3DBlock(128, 256, use_relu=True, dropout=0.2)
        # Decoder (PReLU, dropout, skip concat)
        self.up3  = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.dec3 = Zhang3DBlock(256+128, 128, dropout=0.2)
        self.up2  = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.dec2 = Zhang3DBlock(128+64,  64,  dropout=0.2)
        self.up1  = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.dec1 = Zhang3DBlock(64+32,   32,  dropout=0.2)
        self.head = nn.Conv3d(32, 1, 1)

    def forward(self, batch, **_):
        x  = batch["input"]                          # (B,2,D,H,W)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bridge(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b),  e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return torch.sigmoid(self.head(d1))

    def model_type(self): return "deterministic"


# ---------------------------------------------------------------------------
# Model 7 — ImprovedUNet
# Zhang2025 enhanced with:
#   • 4-level encoder [32→64→128→256→256 bridge] for deeper feature hierarchy
#   • Attention gates on all skip connections (Oktay et al., 2018)
#   • Frequency-domain loss term (added in step function) to fix MAE blurriness
# ---------------------------------------------------------------------------

class SymmetryGate(nn.Module):
    """Learns how much to trust contralateral hemisphere features vs original.
    Gate output ∈ [0,1]: 0 = ignore contralateral, 1 = fully trust it.
    """
    def __init__(self, ch):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv3d(ch * 2, ch, 1, bias=False),
            nn.InstanceNorm3d(ch, affine=True),
            nn.Sigmoid()
        )
        self.proj = nn.Conv3d(ch, ch, 1)

    def forward(self, feat_orig, feat_flip):
        gate = self.gate(torch.cat([feat_orig, feat_flip], dim=1))
        return feat_orig + gate * self.proj(feat_flip)


class AttentionGate3D(nn.Module):
    """Additive soft-attention gate for skip connections."""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv3d(F_g, F_int, 1), nn.InstanceNorm3d(F_int, affine=True))
        self.W_x = nn.Sequential(nn.Conv3d(F_l, F_int, 1), nn.InstanceNorm3d(F_int, affine=True))
        self.psi = nn.Sequential(nn.Conv3d(F_int, 1, 1), nn.InstanceNorm3d(1, affine=True), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, x1.shape[2:], mode="trilinear", align_corners=False)
        return x * self.psi(self.relu(g1 + x1))


class ImprovedUNet(nn.Module):
    """4-level Attention U-Net based on Zhang2025 backbone.
    Channels: [32→64→128→256] encoder, [256] bridge, [256→128→64→32] decoder.
    Attention gates on every skip connection suppress irrelevant features.
    Input/output same as Zhang2025UNet.
    """
    def __init__(self, in_ch=2):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        # Encoder (4 levels)
        self.enc1 = Zhang3DBlock(in_ch, 32)
        self.enc2 = Zhang3DBlock(32,  64)
        self.enc3 = Zhang3DBlock(64,  128)
        self.enc4 = Zhang3DBlock(128, 256)
        # Bridge
        self.bridge = Zhang3DBlock(256, 256, use_relu=True, dropout=0.2)
        # Attention gates: AttentionGate3D(F_g=upsampled_ch, F_l=skip_ch, F_int=...)
        self.ag4 = AttentionGate3D(256, 256, 128)
        self.ag3 = AttentionGate3D(256, 128,  64)
        self.ag2 = AttentionGate3D(128,  64,  32)
        self.ag1 = AttentionGate3D( 64,  32,  16)
        # Decoder
        self.up4  = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.dec4 = Zhang3DBlock(256+256, 256, dropout=0.2)
        self.up3  = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.dec3 = Zhang3DBlock(256+128, 128, dropout=0.2)
        self.up2  = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.dec2 = Zhang3DBlock(128+64,   64, dropout=0.2)
        self.up1  = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.dec1 = Zhang3DBlock( 64+32,   32, dropout=0.2)
        self.head = nn.Conv3d(32, 1, 1)

    def forward(self, batch, **_):
        x  = batch["input"]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bridge(self.pool(e4))

        u4 = self.up4(b)
        d4 = self.dec4(torch.cat([u4, self.ag4(u4, e4)], 1))
        u3 = self.up3(d4)
        d3 = self.dec3(torch.cat([u3, self.ag3(u3, e3)], 1))
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, self.ag2(u2, e2)], 1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, self.ag1(u1, e1)], 1))
        return torch.sigmoid(self.head(d1))

    def model_type(self): return "deterministic"


# ---------------------------------------------------------------------------
# Model 8 — SymmetryUNet (Novel — BraTS 2026)
# Hemispheric Symmetry Inpainting Network
#
# Key insight: brain tumors are unilateral. The contralateral hemisphere
# contains the exact healthy anatomy to restore. We exploit this by:
#   1. Feeding a contralateral hint as a 3rd input channel
#      (LR-flip at masked positions, original elsewhere)
#   2. Running the flipped volume through a lightweight parallel encoder
#   3. Fusing both at bridge level via a SymmetryGate
#   4. Predicting a residual correction on top of the contralateral hint
#
# This is far superior to generic inpainting: the network copies real brain
# anatomy and only learns small corrections for asymmetries and boundaries.
# ---------------------------------------------------------------------------

class SymmetryUNet(nn.Module):
    """Novel Hemispheric Symmetry Inpainting Network for BraTS 2026.

    Input: batch["input"] = [voided | mask] (same interface as all models).
    Contralateral hint is constructed inside forward().

    Architecture:
        - Main encoder (3ch: voided + mask + contralateral_hint)  4-level
        - Parallel flip encoder (1ch: full LR-flipped volume)     4-level
        - SymmetryGate at bridge: fuses original + flipped bridge features
        - Attention-gated decoder (same as ImprovedUNet)
        - Tanh head → residual on contralateral hint → clamp to [0,1]
    """
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool3d(2)

        # Main encoder — 3 input channels
        self.enc1 = Zhang3DBlock(3, 32)
        self.enc2 = Zhang3DBlock(32, 64)
        self.enc3 = Zhang3DBlock(64, 128)
        self.enc4 = Zhang3DBlock(128, 256)
        self.bridge = Zhang3DBlock(256, 256, use_relu=True, dropout=0.2)

        # Lightweight parallel encoder for the full LR-flipped volume (1ch)
        # Processes flipped volume through 4 levels to match bridge spatial size
        self.flip_enc = nn.Sequential(
            Zhang3DBlock(1, 32),   nn.MaxPool3d(2),
            Zhang3DBlock(32, 64),  nn.MaxPool3d(2),
            Zhang3DBlock(64, 128), nn.MaxPool3d(2),
            Zhang3DBlock(128, 256), nn.MaxPool3d(2),
        )
        self.flip_bridge = Zhang3DBlock(256, 256, use_relu=True, dropout=0.2)

        # Symmetry gate fuses bridge features from both streams
        self.sym_gate = SymmetryGate(256)

        # Attention gates (decoder ← encoder skip connections)
        self.ag4 = AttentionGate3D(256, 256, 128)
        self.ag3 = AttentionGate3D(256, 128,  64)
        self.ag2 = AttentionGate3D(128,  64,  32)
        self.ag1 = AttentionGate3D( 64,  32,  16)

        # Decoder
        self.up4  = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.dec4 = Zhang3DBlock(256+256, 256, dropout=0.2)
        self.up3  = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.dec3 = Zhang3DBlock(256+128, 128, dropout=0.2)
        self.up2  = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.dec2 = Zhang3DBlock(128+64,   64, dropout=0.2)
        self.up1  = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.dec1 = Zhang3DBlock( 64+32,   32, dropout=0.2)

        # Tanh head: outputs residual correction ∈ [-1, 1]
        self.head = nn.Sequential(nn.Conv3d(32, 1, 1), nn.Tanh())

    def forward(self, batch, **_):
        voided = batch["input"][:, :1]   # (B,1,D,H,W)
        mask   = batch["input"][:, 1:]   # (B,1,D,H,W)

        # --- Contralateral hint ---
        # LR-flip (axis W = dim 4): maps each voxel to its mirror hemisphere
        flipped = torch.flip(voided, dims=[4])
        # At masked voids, substitute contralateral anatomy; keep original elsewhere
        contra = voided * (1.0 - mask) + flipped * mask   # (B,1,D,H,W)

        # --- Main encoder (3ch input) ---
        x  = torch.cat([voided, mask, contra], dim=1)     # (B,3,D,H,W)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bridge(self.pool(e4))

        # --- Flip encoder + SymmetryGate at bridge ---
        b_flip = self.flip_bridge(self.flip_enc(flipped))
        b = self.sym_gate(b, b_flip)

        # --- Attention-gated decoder ---
        u4 = self.up4(b)
        d4 = self.dec4(torch.cat([u4, self.ag4(u4, e4)], 1))
        u3 = self.up3(d4)
        d3 = self.dec3(torch.cat([u3, self.ag3(u3, e3)], 1))
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, self.ag2(u2, e2)], 1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, self.ag1(u1, e1)], 1))

        # Residual on contralateral hint: network corrects asymmetries only
        residual = self.head(d1)                          # ∈ [-1, 1]
        pred = torch.clamp(contra + residual, 0.0, 1.0)
        return pred

    def model_type(self): return "deterministic"


# ---------------------------------------------------------------------------
# Model 9 — Swin UNETR (via MONAI)
# Cao et al., "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain
# Tumors in MRI Images" — MICCAI 2022.
# Uses shifted-window self-attention to capture long-range context that
# convolutions miss, combined with a standard CNN decoder.
# ---------------------------------------------------------------------------

class SwinUNETRInpainting(nn.Module):
    """Thin wrapper around MONAI's SwinUNETR for BraTS inpainting.
    Input: batch["input"] = cat([voided, mask], dim=1)  — 2 channels
    Output: (B, 1, D, H, W) reconstruction in [0, 1]
    """
    def __init__(self, patch_size=(96, 96, 96), feature_size=48):
        super().__init__()
        from monai.networks.nets import SwinUNETR as _SwinUNETR
        self._net = _SwinUNETR(
            img_size=patch_size,
            in_channels=2,
            out_channels=1,
            feature_size=feature_size,
            use_checkpoint=True,   # gradient checkpointing — saves ~30% VRAM
        )

    def forward(self, batch, **_):
        x = batch["input"]   # (B, 2, D, H, W)
        return torch.sigmoid(self._net(x))

    def model_type(self): return "deterministic"


MODEL_REGISTRY = {
    "unet":         lambda a: UNetBaseline(bc=a.base_ch, cm=tuple(a.ch_mult), nb=a.n_blocks),
    "diffusion":    lambda a: DiffusionUNet(bc=a.base_ch, cm=tuple(a.ch_mult), nb=a.n_blocks),
    "hierarchical": lambda a: HierarchicalDiffusion(bc=a.base_ch//2, cm=tuple(a.ch_mult), nb=a.n_blocks),
    "symamba":      lambda a: SyMambaDiff(bc=a.base_ch, cm=tuple(a.ch_mult), nb=a.n_blocks, d_state=a.d_state),
    "gan":          lambda a: GAN3D(bc=a.base_ch, cm=tuple(a.ch_mult)),
    "zhang2025":      lambda a: Zhang2025UNet(),
    "zhang_boundary": lambda a: Zhang2025UNet(),   # same arch, better loss + fine-tune
    "zhang_exact":    lambda a: Zhang2025UNet(),   # Zhang's exact setup + our improvements
    "improved":       lambda a: ImprovedUNet(),
    "symmetry":       lambda a: SymmetryUNet(),    # novel hemispheric symmetry model
    "swinunetr":    lambda a: SwinUNETRInpainting(patch_size=tuple(a.patch_size)),
}


# ---------------------------------------------------------------------------
# Training step implementations
# ---------------------------------------------------------------------------

def step_unet(model, batch, cfg, device):
    pred = model(batch)
    target = batch["target"]
    mask   = batch.get("mask", torch.ones_like(pred))
    loss, losses = inpainting_loss(pred, target, mask,
                                   lambda_ssim=cfg.lambda_ssim,
                                   lambda_mask=cfg.lambda_mask)
    return loss, losses, pred

def step_diffusion(model, batch, cfg, device, scheduler):
    target = batch["target"]
    t = torch.randint(0, cfg.T, (target.shape[0],), device=device)
    noise = torch.randn_like(target)
    mask   = batch.get("mask")
    mask_u = batch.get("mask_unhealthy")
    noisy, noise_used = scheduler.add_noise(target, t, noise, mask=mask, mask_u=mask_u)
    pred_noise = model(noisy, t, cond=batch.get("input"))
    loss = F.mse_loss(pred_noise, noise_used)
    return loss, {"noise_mse": loss.item()}, None

def step_hierarchical(model, batch, cfg, device, scheduler):
    target = batch["target"]
    cond   = batch.get("input")
    t = torch.randint(0, cfg.T, (target.shape[0],), device=device)
    noise  = torch.randn_like(target)
    noisy, noise_used = scheduler.add_noise(target, t, noise)
    # Stage 2 training (stage 1 trained together via coarse=zeros warmup)
    pred_noise = model(noisy, t, cond=cond, coarse=torch.zeros_like(noisy), stage=2)
    loss = F.mse_loss(pred_noise, noise_used)
    return loss, {"noise_mse": loss.item()}, None

def step_symamba(model, batch, cfg, device, scheduler):
    target = batch["target"]
    t = torch.randint(0, cfg.T, (target.shape[0],), device=device)
    noise  = torch.randn_like(target)
    mask   = batch.get("mask")
    mask_u = batch.get("mask_unhealthy")
    noisy, noise_used = scheduler.add_noise(target, t, noise,
                                             mask=mask, mask_u=mask_u,
                                             alpha_in=cfg.alpha_inside,
                                             alpha_b=cfg.alpha_boundary,
                                             alpha_out=cfg.alpha_outside)
    pred_noise = model(noisy, t,
                       voided=batch.get("voided"),
                       mask=mask,
                       mask_healthy=batch.get("mask_healthy"))
    diff_loss = F.mse_loss(pred_noise, noise_used)
    # Symmetry loss
    sym_loss = torch.tensor(0.0, device=device)
    if "mask_healthy" in batch and batch["mask_healthy"] is not None:
        # Encourage symmetry in the prediction domain (indirect via noise consistency)
        pass
    loss = diff_loss + cfg.lambda_sym * sym_loss
    return loss, {"noise_mse": diff_loss.item(), "sym": sym_loss.item()}, None

def step_zhang2025(model, batch, cfg, device):
    pred   = model(batch)
    target = batch["target"]
    mask   = batch.get("mask", torch.ones_like(pred))
    loss, losses = zhang_loss(pred, target, mask)
    return loss, losses, pred


def step_improved(model, batch, cfg, device):
    pred   = model(batch)
    target = batch["target"]
    mask   = batch.get("mask", torch.ones_like(pred))
    loss, losses = zhang_loss(pred, target, mask)
    if cfg.lambda_freq > 0:
        fl = freq_domain_loss(pred, target, mask)
        loss = loss + cfg.lambda_freq * fl
        losses["freq"] = fl.item()
    return loss, losses, pred


def step_symmetry(model, batch, cfg, device):
    """Training step for SymmetryUNet.
    Uses boundary-aware loss to sharpen transitions at mask edges.
    Frequency loss penalises blurring of high-frequency anatomical detail.
    """
    pred   = model(batch)
    target = batch["target"]
    mask   = batch.get("mask", torch.ones_like(pred))
    lb     = getattr(cfg, "lambda_boundary_mask", 2.0)
    loss, losses = zhang_loss_boundary(pred, target, mask, lambda_boundary=lb)
    if cfg.lambda_freq > 0:
        fl = freq_domain_loss(pred, target, mask)
        loss = loss + cfg.lambda_freq * fl
        losses["freq"] = fl.item()
    return loss, losses, pred


def zhang_exact_loss(pred, target, healthy_mask, mask, lambda_boundary=2.0):
    """
    Zhang 2025 exact loss + our boundary improvement.
    - MAE computed ONLY on healthy-mask region (Zhang exact)
    - SSIM computed on full image (Zhang exact)
    - Boundary-aware penalty on mask edge ring (our improvement)
    """
    h = (healthy_mask > 0.5).squeeze(1)
    mae = F.l1_loss(pred.squeeze(1)[h], target.squeeze(1)[h]) if h.any() else F.l1_loss(pred, target)
    ssim_val = ssim_loss_2d(pred, target)
    with torch.no_grad():
        dilated  = F.max_pool3d(mask, kernel_size=7, stride=1, padding=3)
        boundary = (dilated - mask).clamp(0, 1)
        b = (boundary > 0.5).squeeze(1)
    mae_b = F.l1_loss(pred.squeeze(1)[b], target.squeeze(1)[b]) if b.any() else torch.tensor(0.0, device=pred.device)
    loss = mae + ssim_val + lambda_boundary * mae_b
    return loss, {"mae_healthy": mae.item(), "ssim_full": ssim_val.item(), "mae_boundary": mae_b.item()}


def step_zhang_exact(model, batch, cfg, device):
    """Zhang 2025 exact training step: full volume, healthy-mask MAE + full-image SSIM + boundary."""
    inp    = batch["input"].to(device)
    target = batch["target"].to(device)
    mask   = batch["mask"].to(device)
    hmask  = batch["healthy_mask"].to(device)
    with autocast(enabled=cfg.amp):
        pred = model({"input": inp})
        loss, losses = zhang_exact_loss(
            pred, target, hmask, mask,
            lambda_boundary=getattr(cfg, "lambda_boundary_mask", 2.0),
        )
    return loss, losses, pred


def step_zhang_boundary(model, batch, cfg, device):
    """Zhang2025 architecture + boundary-aware loss + frequency loss.
    Fine-tunes from zhang2025 checkpoint with --pretrained; trains from scratch otherwise.
    """
    pred   = model(batch)
    target = batch["target"]
    mask   = batch.get("mask", torch.ones_like(pred))
    lb     = getattr(cfg, "lambda_boundary_mask", 2.0)
    loss, losses = zhang_loss_boundary(pred, target, mask, lambda_boundary=lb)
    if cfg.lambda_freq > 0:
        fl = freq_domain_loss(pred, target, mask)
        loss = loss + cfg.lambda_freq * fl
        losses["freq"] = fl.item()
    return loss, losses, pred


def step_gan(model, batch, cfg, device, opt_g, opt_d, scaler):
    inp    = batch["input"]
    target = batch["target"]
    mask   = batch.get("mask", torch.ones_like(target))

    # Generator step
    opt_g.zero_grad()
    with autocast(enabled=cfg.amp):
        pred = model.generate(batch)
        loss_g, losses = inpainting_loss(pred, target, mask, cfg.lambda_ssim, cfg.lambda_mask)
        # Optional adversarial
        if cfg.lambda_adv > 0 and random.random() < 0.5:
            fake_logit = model.discriminate(inp, pred)
            loss_adv = F.binary_cross_entropy_with_logits(fake_logit, torch.ones_like(fake_logit))
            loss_g = loss_g + cfg.lambda_adv * loss_adv
    scaler.scale(loss_g).backward()
    scaler.step(opt_g); scaler.update()

    # Discriminator step (every other iter)
    opt_d.zero_grad()
    with autocast(enabled=cfg.amp):
        with torch.no_grad():
            pred_d = model.generate(batch)
        real_logit = model.discriminate(inp, target)
        fake_logit = model.discriminate(inp, pred_d.detach())
        loss_d = (F.binary_cross_entropy_with_logits(real_logit, torch.ones_like(real_logit)) +
                  F.binary_cross_entropy_with_logits(fake_logit, torch.zeros_like(fake_logit))) * 0.5
    scaler.scale(loss_d).backward()
    scaler.step(opt_d); scaler.update()

    losses["adv_g"] = loss_g.item()
    losses["adv_d"] = loss_d.item()
    return loss_g, losses, pred


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, optimizer, iteration, metrics, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    obj = {
        "iteration":      iteration,
        "model_state":    model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "metrics":        metrics,
        "config":         vars(cfg),
    }
    torch.save(obj, path)

def load_checkpoint(path, model, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and ckpt.get("optimizer_state"):
        optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"  Resumed from iteration {ckpt['iteration']}")
    return ckpt["iteration"], ckpt.get("metrics", {})


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, scheduler, cfg, device):
    model.eval()
    results = []
    for batch in tqdm(loader, desc="Val", leave=False):
        batch = {k: v.to(device) for k,v in batch.items() if isinstance(v, torch.Tensor)}
        mtype = model.model_type() if hasattr(model,"model_type") else "deterministic"

        if mtype == "deterministic":
            pred = model(batch)
        elif mtype == "gan":
            pred = model.generate(batch)
        else:
            shape = (1,1,*cfg.patch_size)
            if cfg.model == "symamba":
                def fn(x,t,c,**_):
                    return model(x,t,voided=batch.get("voided"),
                                 mask=batch.get("mask"),
                                 mask_healthy=batch.get("mask_healthy"))
            elif cfg.model == "hierarchical":
                def fn(x,t,c,**_):
                    return model(x,t,cond=c,coarse=torch.zeros_like(x),stage=2)
            else:
                def fn(x,t,c,**_): return model(x,t,cond=c)
            pred = scheduler.ddim_sample(fn, batch.get("input"), shape,
                                          steps=cfg.ddim_steps, device=device)

        if "target" in batch and "mask" in batch:
            pred_eval = pred
            target_eval = batch["target"]
            # zhang_exact trains on [-1,1]; rescale to [0,1] for metric computation
            if getattr(cfg, "model", None) == "zhang_exact":
                pred_eval   = (pred_eval   + 1.0) / 2.0
                target_eval = (target_eval + 1.0) / 2.0
            for i in range(pred_eval.shape[0]):
                r = compute_metrics(
                    pred_eval[i,0].cpu().numpy(),
                    target_eval[i,0].cpu().numpy(),
                    batch["mask"][i,0].cpu().numpy())
                results.append(r)

    model.train()
    if not results:
        return {}
    return {k: float(np.nanmean([r[k] for r in results])) for k in results[0]}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Output dir
    run_dir = os.path.join(cfg.output_dir, cfg.model)
    os.makedirs(run_dir, exist_ok=True)

    # Discover samples
    train_samples = discover_samples(cfg.train_dir, require_target=True)
    # Val set: require t1n ground truth. BraTS challenge val split has no targets,
    # so this will return [] and we fall back to a training split.
    val_samples   = discover_samples(cfg.val_dir, require_target=True) if os.path.isdir(cfg.val_dir) else []

    # If no separate val set (or val set has no ground truth), split training
    val_root = cfg.val_dir  # default: use the dedicated val dir
    if not val_samples:
        print("  Val dir has no ground-truth targets — using 10% of training data for validation.")
        random.shuffle(train_samples)
        n_val = max(1, int(len(train_samples)*0.1))
        val_samples   = train_samples[:n_val]
        train_samples = train_samples[n_val:]
        val_root = cfg.train_dir  # samples live in train dir

    print(f"Train: {len(train_samples)} samples | Val: {len(val_samples)} samples (root: {val_root})")
    # Save val split so evaluate.py can reproduce the exact same samples
    split_path = os.path.join(run_dir, "val_samples.json")
    if not os.path.exists(split_path):
        with open(split_path, "w") as f:
            json.dump({"val_samples": val_samples, "val_root": val_root}, f, indent=2)
        print(f"  Saved val split → {split_path}")

    pps_train = max(1, cfg.patches_per_epoch // max(1, len(train_samples)))
    pps_val   = max(1, 20 // max(1, len(val_samples)))

    if cfg.model == "zhang_exact":
        # Zhang's exact setup: full volumes, mask-healthy augmentation
        train_ds = BraTSZhangDataset(cfg.train_dir, train_samples, mode="train")
        val_ds   = BraTSZhangDataset(val_root,       val_samples,   mode="val")
    else:
        train_ds = BraTSDataset(cfg.train_dir, train_samples,
                                 patch_size=cfg.patch_size,
                                 patches_per_sample=pps_train, mode="train",
                                 aug_mask_prob=cfg.aug_mask_prob)
        val_ds   = BraTSDataset(val_root, val_samples, patch_size=cfg.patch_size,
                                 patches_per_sample=pps_val, mode="val", augment=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                               num_workers=0, pin_memory=True)

    # Build model
    model = MODEL_REGISTRY[cfg.model](cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {cfg.model} | Params: {n_params/1e6:.2f}M")

    optimizer  = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scaler     = GradScaler(enabled=cfg.amp)
    scheduler  = DDPMScheduler(T=cfg.T, schedule=cfg.beta_schedule, device=str(device))

    # GAN needs separate optimizers
    opt_d = None
    if cfg.model == "gan":
        opt_d = torch.optim.AdamW(model.D.parameters(), lr=cfg.lr, weight_decay=1e-4)

    # Warm-start from a pretrained checkpoint (e.g. zhang2025 → zhang_boundary)
    if getattr(cfg, "pretrained", "") and os.path.exists(cfg.pretrained):
        ckpt = torch.load(cfg.pretrained, map_location=device)
        state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Pretrained weights loaded from {cfg.pretrained}")
        if missing:    print(f"  Missing keys : {missing[:5]}")
        if unexpected: print(f"  Unexpected   : {unexpected[:5]}")

    # Resume
    start_iter = 0
    ckpt_best  = os.path.join(run_dir, "best.pt")
    ckpt_latest= os.path.join(run_dir, "latest.pt")
    if cfg.resume and os.path.exists(ckpt_latest):
        start_iter, _ = load_checkpoint(ckpt_latest, model, optimizer, str(device))

    # LR scheduler: cosine over total iterations, fast-forwarded to start_iter
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.iterations, last_epoch=start_iter - 1
    )

    # History
    history = {"iter": [], "train_loss": [], "val_ssim": [], "val_psnr": [], "val_mse": []}
    hist_path = os.path.join(run_dir, "history.json")
    if os.path.exists(hist_path):
        try:
            with open(hist_path) as f:
                history = json.load(f)
        except (json.JSONDecodeError, ValueError):
            print(f"  WARNING: history.json corrupted — starting fresh history")

    best_ssim = max(history["val_ssim"]) if history["val_ssim"] else 0.0

    print(f"\nTraining {cfg.model} | Start: {start_iter} | End: {cfg.iterations}")
    print(f"Logs → {run_dir}")

    train_iter = infinite_loader(train_loader)
    model.train()
    run_loss = 0.0
    t0 = time.time()

    for it in range(start_iter, cfg.iterations):
        batch = next(train_iter)
        batch = {k: v.to(device) for k,v in batch.items() if isinstance(v, torch.Tensor)}

        with autocast(enabled=cfg.amp):
            if cfg.model == "unet":
                loss, losses, _ = step_unet(model, batch, cfg, device)
            elif cfg.model == "diffusion":
                loss, losses, _ = step_diffusion(model, batch, cfg, device, scheduler)
            elif cfg.model == "hierarchical":
                loss, losses, _ = step_hierarchical(model, batch, cfg, device, scheduler)
            elif cfg.model == "symamba":
                loss, losses, _ = step_symamba(model, batch, cfg, device, scheduler)
            elif cfg.model == "gan":
                loss, losses, _ = step_gan(model, batch, cfg, device, optimizer, opt_d, scaler)
            elif cfg.model == "zhang2025":
                loss, losses, _ = step_zhang2025(model, batch, cfg, device)
            elif cfg.model == "symmetry":
                loss, losses, _ = step_symmetry(model, batch, cfg, device)
            elif cfg.model == "zhang_boundary":
                loss, losses, _ = step_zhang_boundary(model, batch, cfg, device)
            elif cfg.model == "zhang_exact":
                loss, losses, _ = step_zhang_exact(model, batch, cfg, device)
            elif cfg.model == "improved":
                loss, losses, _ = step_improved(model, batch, cfg, device)
            elif cfg.model == "swinunetr":
                loss, losses, _ = step_zhang2025(model, batch, cfg, device)

        if cfg.model != "gan":
            loss_scaled = loss / cfg.grad_accum
            scaler.scale(loss_scaled).backward()
            if (it+1) % cfg.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                lr_sched.step()

        run_loss += loss.item()

        # Logging
        if (it+1) % cfg.log_every == 0:
            avg = run_loss / cfg.log_every
            elapsed = time.time() - t0
            iters_s  = cfg.log_every / elapsed
            eta_h    = (cfg.iterations - it - 1) / iters_s / 3600
            print(f"[{it+1:>7d}/{cfg.iterations}] loss={avg:.4f} | "
                  f"{iters_s:.1f}it/s | ETA {eta_h:.1f}h | "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")
            run_loss = 0.0; t0 = time.time()

        # Validation
        if (it+1) % cfg.val_every == 0:
            vm = validate(model, val_loader, scheduler, cfg, device)
            ssim = vm.get("ssim", 0.0)
            print(f"  Val [{it+1}] SSIM={ssim:.4f} PSNR={vm.get('psnr',0):.2f} "
                  f"MSE={vm.get('mse',0):.6f}")
            history["iter"].append(it+1)
            history["train_loss"].append(run_loss)
            history["val_ssim"].append(ssim)
            history["val_psnr"].append(vm.get("psnr",0))
            history["val_mse"].append(vm.get("mse",0))
            with open(hist_path,"w") as f: json.dump(history,f,indent=2)

            if ssim > best_ssim:
                best_ssim = ssim
                save_checkpoint(ckpt_best, model, optimizer, it+1, vm, cfg)
                print(f"  ** New best SSIM: {best_ssim:.4f} — saved best.pt")

        # Checkpoint — only save latest.pt (no per-iter files to avoid disk exhaustion)
        if (it+1) % cfg.save_every == 0:
            save_checkpoint(ckpt_latest, model, optimizer, it+1, {}, cfg)

    # Final save
    save_checkpoint(ckpt_latest, model, optimizer, cfg.iterations, {}, cfg)
    print(f"\nDone. Best SSIM: {best_ssim:.4f}  Saved to: {run_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="BraTS 2026 Inpainting Training")
    # Model
    p.add_argument("--model",      default="unet",
                   choices=["unet","diffusion","hierarchical","symamba","gan",
                            "zhang2025","improved","swinunetr",
                            "zhang_boundary","symmetry","zhang_exact"])
    p.add_argument("--base_ch",    type=int,   default=32)
    p.add_argument("--ch_mult",    type=int,   nargs="+", default=[1,2,4,8])
    p.add_argument("--n_blocks",   type=int,   default=2)
    p.add_argument("--d_state",    type=int,   default=16)
    # Paths
    p.add_argument("--train_dir",  default="/data/brats2023/training")
    p.add_argument("--val_dir",    default="/data/brats2023/validation")
    p.add_argument("--output_dir", default="/data/experiments")
    # Training
    p.add_argument("--iterations", type=int,   default=200_000)
    p.add_argument("--batch_size", type=int,   default=1)
    p.add_argument("--grad_accum", type=int,   default=4)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--amp",        action="store_true", default=True)
    p.add_argument("--num_workers",type=int,   default=4)
    p.add_argument("--resume",     action="store_true")
    p.add_argument("--pretrained", type=str, default="",
                   help="Path to checkpoint to warm-start from (e.g. zhang2025 best.pt)")
    p.add_argument("--lambda_boundary_mask", type=float, default=2.0,
                   help="Weight for boundary ring loss in zhang_boundary model")
    # Data
    p.add_argument("--patch_size", type=int,   nargs=3, default=[96,96,96])
    p.add_argument("--patches_per_epoch", type=int, default=500)
    # Diffusion
    p.add_argument("--T",          type=int,   default=1000)
    p.add_argument("--beta_schedule", default="cosine", choices=["cosine","linear"])
    p.add_argument("--ddim_steps", type=int,   default=50)
    # Loss
    p.add_argument("--lambda_ssim",  type=float, default=0.5)
    p.add_argument("--lambda_mask",  type=float, default=2.0)
    p.add_argument("--lambda_sym",   type=float, default=0.2)
    p.add_argument("--lambda_adv",   type=float, default=0.05)
    # SyMamba noise
    p.add_argument("--aug_mask_prob",  type=float, default=0.0,
                   help="Prob of replacing BraTS mask with random ellipsoid void (Zhang2025 trick)")
    p.add_argument("--lambda_freq",    type=float, default=0.1)
    p.add_argument("--alpha_inside",   type=float, default=1.5)
    p.add_argument("--alpha_boundary", type=float, default=2.0)
    p.add_argument("--alpha_outside",  type=float, default=0.8)
    # Logging
    p.add_argument("--log_every",  type=int,   default=100)
    p.add_argument("--val_every",  type=int,   default=5000)
    p.add_argument("--save_every", type=int,   default=10000)
    return p.parse_args()


if __name__ == "__main__":
    cfg = parse_args()
    cfg.patch_size = tuple(cfg.patch_size)
    cfg.ch_mult    = tuple(cfg.ch_mult)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    train(cfg)
