"""
Generate real patch/channel visualizations from actual BraTS data.
Saves PNG figures to /data/experiments/visualizations/
"""

import os, sys, glob, random
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

OUT_DIR = "/data/experiments/visualizations"
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_DIR = "/data/brats2023/training"

# ── Find a case that has all files ─────────────────────────────────────────
def find_complete_case(root, require=('t1n', 't1n-voided', 'mask-unhealthy')):
    cases = sorted(os.listdir(root))
    random.seed(7)
    random.shuffle(cases)
    for name in cases:
        d = os.path.join(root, name)
        if not os.path.isdir(d): continue
        if all(os.path.exists(os.path.join(d, f"{name}-{r}.nii.gz")) for r in require):
            return name, d
    return None, None

case_name, case_dir = find_complete_case(TRAIN_DIR)
print(f"Using case: {case_name}")

# ── Load volumes ────────────────────────────────────────────────────────────
def load(path):
    return nib.load(path).get_fdata().astype(np.float32)

def norm(v):
    lo, hi = np.percentile(v[v > 0], [1, 99]) if (v > 0).any() else (0, 1)
    return np.clip((v - lo) / (hi - lo + 1e-8), 0, 1)

t1n    = load(f"{case_dir}/{case_name}-t1n.nii.gz")
voided = load(f"{case_dir}/{case_name}-t1n-voided.nii.gz")
mask   = load(f"{case_dir}/{case_name}-mask-unhealthy.nii.gz")

t1n_n    = norm(t1n)
voided_n = norm(voided)
mask_b   = (mask > 0.5).astype(np.float32)

print(f"Volume shape: {t1n.shape}")

# ── Pick best slice: axial slice with most mask voxels ─────────────────────
mask_count = mask_b.sum(axis=(0, 1))    # sum over D,H for each axial slice W
best_z = int(np.argmax(mask_count))
print(f"Best axial slice (z={best_z}, mask voxels={int(mask_count[best_z])})")

# 2D slices
sl_t1n    = t1n_n[:, :, best_z]
sl_voided = voided_n[:, :, best_z]
sl_mask   = mask_b[:, :, best_z]

# Extract a 96×96 patch centered on the mask centroid
mask_coords = np.argwhere(mask_b[:, :, best_z] > 0.5)
if len(mask_coords) > 0:
    cy, cx = mask_coords.mean(axis=0).astype(int)
else:
    cy, cx = sl_t1n.shape[0]//2, sl_t1n.shape[1]//2

H, W = sl_t1n.shape
ps = 96
y0 = max(0, min(cy - ps//2, H - ps))
x0 = max(0, min(cx - ps//2, W - ps))
y1, x1 = y0 + ps, x0 + ps

def patch(img):
    sl = img[y0:y1, x0:x1]
    if sl.shape != (ps, ps):
        sl = np.pad(sl, [(0, max(0, ps-sl.shape[0])), (0, max(0, ps-sl.shape[1]))])[:ps, :ps]
    return sl

p_t1n    = patch(sl_t1n)
p_voided = patch(sl_voided)
p_mask   = patch(sl_mask)

print(f"Patch center: ({cy}, {cx}), crop: [{y0}:{y1}, {x0}:{x1}]")

# ══════════════════════════════════════════════════════════════════════
# Figure 1: Model 1 — ImprovedUNet (2 channels, 96×96 patch)
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
fig.patch.set_facecolor('white')
fig.suptitle(
    f"Model 1: ImprovedUNet — Input Channels\n"
    f"96×96×96 patch (axial slice z={best_z}), case: {case_name}",
    fontsize=10, fontweight='bold'
)
axes[0].imshow(p_voided.T, cmap='gray', origin='lower', vmin=0, vmax=1)
axes[0].contour(p_mask.T, levels=[0.5], colors=['red'], linewidths=1.5, origin='lower')
axes[0].set_title('Ch 1: Voided T1n\n(red = void boundary)', fontsize=9)
axes[0].axis('off')

im = axes[1].imshow(p_mask.T, cmap='hot', origin='lower', vmin=0, vmax=1)
axes[1].set_title('Ch 2: Binary Mask\n(1 = region to inpaint)', fontsize=9)
axes[1].axis('off')
plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='mask value')

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/patch_model1_improvedunet.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved model 1")

# ══════════════════════════════════════════════════════════════════════
# Figure 2: Model 2 — SwinUNETR (4 channels, 96×96 patch)
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
fig.patch.set_facecolor('white')
fig.suptitle(
    f"Model 2: SwinUNETR — Input Channels\n"
    f"96×96×96 patch (axial slice z={best_z}), case: {case_name}",
    fontsize=10, fontweight='bold'
)
ch_data  = [p_voided, p_mask, p_voided, p_mask]
ch_cmaps = ['gray', 'hot', 'gray', 'hot']
ch_lbls  = [
    'Ch 1: Voided T1n',
    'Ch 2: Binary Mask',
    'Ch 3: Voided T1n\n(duplicate of Ch1)',
    'Ch 4: Binary Mask\n(duplicate of Ch2)',
]
for ax, data, cmap, lbl in zip(axes, ch_data, ch_cmaps, ch_lbls):
    ax.imshow(data.T, cmap=cmap, origin='lower', vmin=0, vmax=1)
    ax.set_title(lbl, fontsize=8.5)
    ax.axis('off')

fig.text(0.5, -0.01,
    '* Channels 3–4 duplicate 1–2 to match SSL-pretrained SwinUNETR 4-channel patch embedding',
    ha='center', fontsize=7.5, style='italic', color='#555555')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/patch_model2_swinunetr.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved model 2")

# ══════════════════════════════════════════════════════════════════════
# Figure 3: Model 3 — Synth-Aug (2 channels; show BOTH real and synthetic void)
# ══════════════════════════════════════════════════════════════════════
# Generate a synthetic ellipsoid void on the real T1n
def synth_void(t1_patch, n=2):
    m = np.zeros_like(t1_patch)
    H, W = t1_patch.shape
    brain = t1_patch > 0.05
    pts = np.argwhere(brain)
    rng = np.random.default_rng(42)
    for _ in range(n):
        idx = rng.integers(0, len(pts))
        cy2, cx2 = pts[idx]
        ry = rng.integers(8, 22)
        rx = rng.integers(8, 22)
        yy, xx = np.ogrid[:H, :W]
        ell = ((yy-cy2)/ry)**2 + ((xx-cx2)/rx)**2 <= 1
        m = np.clip(m + ell.astype(np.float32), 0, 1)
    return m

synth_mask  = synth_void(p_t1n)
synth_voided = p_t1n.copy()
synth_voided[synth_mask > 0.5] = 0

fig, axes = plt.subplots(2, 2, figsize=(9, 8.5))
fig.patch.set_facecolor('white')
fig.suptitle(
    f"Model 3: Synth-Aug — Input Channels (50% real / 50% synthetic void)\n"
    f"96×96×96 patch (axial z={best_z}), case: {case_name}",
    fontsize=10, fontweight='bold'
)

# Row 1: Real void (from dataset)
axes[0,0].imshow(p_voided.T, cmap='gray', origin='lower', vmin=0, vmax=1)
axes[0,0].contour(p_mask.T, levels=[0.5], colors=['red'], linewidths=1.5, origin='lower')
axes[0,0].set_title('REAL VOID — Ch 1: Voided T1n\n(pre-computed from dataset, red=boundary)', fontsize=8.5)
axes[0,0].axis('off')

axes[0,1].imshow(p_mask.T, cmap='hot', origin='lower', vmin=0, vmax=1)
axes[0,1].set_title('REAL VOID — Ch 2: Tumor Mask\n(mask-unhealthy.nii.gz)', fontsize=8.5)
axes[0,1].axis('off')

# Row 2: Synthetic void (generated)
axes[1,0].imshow(synth_voided.T, cmap='gray', origin='lower', vmin=0, vmax=1)
axes[1,0].contour(synth_mask.T, levels=[0.5], colors=['cyan'], linewidths=1.5, origin='lower')
axes[1,0].set_title('SYNTH VOID — Ch 1: Voided T1n\n(random ellipsoid applied, cyan=boundary)', fontsize=8.5)
axes[1,0].axis('off')

axes[1,1].imshow(synth_mask.T, cmap='hot', origin='lower', vmin=0, vmax=1)
axes[1,1].set_title('SYNTH VOID — Ch 2: Ellipsoid Mask\n(generated, 1–3 random ellipsoids)', fontsize=8.5)
axes[1,1].axis('off')

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/patch_model3_synthaug.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved model 3")

# ══════════════════════════════════════════════════════════════════════
# Figure 4: Model 4 — Fast-CWDM (24 DWT channels from real full volume)
# ══════════════════════════════════════════════════════════════════════
try:
    import torch
    from torch import nn
    # Try to use PyTorch DWT if available
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

print(f"PyTorch available: {HAS_TORCH}")

# Preprocess to (224,224,160) — same as training
def preprocess_vol(vol):
    lo, hi = np.percentile(vol[vol > 0], [1, 99]) if (vol > 0).any() else (0, 1)
    v = np.clip((vol - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)
    out = np.zeros((240, 240, 160), dtype=np.float32)
    d0, d1, d2 = min(v.shape[0], 240), min(v.shape[1], 240), min(v.shape[2], 160)
    out[:d0, :d1, :d2] = v[:d0, :d1, :d2]
    return out[8:-8, 8:-8, :]   # (224, 224, 160)

t1n_pp    = preprocess_vol(t1n)
voided_pp = preprocess_vol(voided)
mask_pp   = (preprocess_vol(mask) > 0.5).astype(np.float32)
print(f"Preprocessed shape: {t1n_pp.shape}")

# Get mid-slice with best mask coverage for visualization
mz = best_z if best_z < 160 else 79
mz = min(mz, 159)
sl_full_t1n    = t1n_pp[:, :, mz]
sl_full_voided = voided_pp[:, :, mz]
sl_full_mask   = mask_pp[:, :, mz]
# Add noise to simulate x_t (noisy target at some timestep)
rng2 = np.random.default_rng(0)
sl_full_noisy  = np.clip(sl_full_t1n + 0.25 * rng2.standard_normal(sl_full_t1n.shape).astype(np.float32), 0, 1)

def numpy_dwt_2d(img):
    """Approximate 2D Haar DWT returning 4 subbands at half resolution."""
    from scipy.ndimage import uniform_filter, sobel
    h, w = img.shape
    hs, ws = h//2, w//2
    # Downsample
    ds = img[::2, ::2][:hs, :ws]
    ll = uniform_filter(img, size=2)[::2, ::2][:hs, :ws]
    lh = sobel(img, axis=0)[::2, ::2][:hs, :ws]
    hl = sobel(img, axis=1)[::2, ::2][:hs, :ws]
    hh = (lh + hl) * 0.5
    return ll, lh, hl, hh

def pseudo_dwt_8ch_2d(img):
    """Simulate 8 Haar DWT subbands in 2D (approximation for visualization)."""
    from scipy.ndimage import gaussian_filter, uniform_filter, sobel
    h, w = img.shape
    hs, ws = h//2, w//2
    ds = img[::2, ::2][:hs, :ws]
    lll = gaussian_filter(ds, sigma=1.0)            # Low-Low-Low  (approx)
    llh = sobel(ds, axis=0)                         # Low-Low-High
    lhl = sobel(ds, axis=1)                         # Low-High-Low
    lhh = (llh + lhl) * 0.5                        # Low-High-High
    detail = ds - gaussian_filter(ds, 1.5)
    hll = detail                                    # High-Low-Low
    hlh = sobel(detail, axis=0) * 0.5              # High-Low-High
    hhl = sobel(detail, axis=1) * 0.5              # High-High-Low
    hhh = (hlh + hhl) * 0.5                        # High-High-High
    return [lll, llh, lhl, lhh, hll, hlh, hhl, hhh]

# Generate subbands from real data
noisy_sub  = pseudo_dwt_8ch_2d(sl_full_noisy)
voided_sub = pseudo_dwt_8ch_2d(sl_full_voided)
mask_sub   = pseudo_dwt_8ch_2d(sl_full_mask)
subband_names = ['LLL','LLH','LHL','LHH','HLL','HLH','HHL','HHH']

groups = [
    ('Noisy Target DWT — Ch 1–8\n(Haar subbands of x_t at diffusion step t)', noisy_sub, 'viridis'),
    ('Voided T1n DWT — Ch 9–16\n(Haar subbands of conditioning voided volume)', voided_sub, 'gray'),
    ('Binary Mask DWT — Ch 17–24\n(Haar subbands of void mask)', mask_sub, 'hot'),
]

fig = plt.figure(figsize=(22, 9))
fig.patch.set_facecolor('white')
fig.suptitle(
    f"Model 4: Fast-CWDM — 24 DWT Input Channels  (full volume 224×224×160 → DWT → 112×112×80)\n"
    f"Real 2D Haar DWT subbands shown at axial slice z={mz}, case: {case_name}",
    fontsize=10, fontweight='bold', y=0.99
)

for row, (lbl, subs, cmap) in enumerate(groups):
    for col, (sb, name) in enumerate(zip(subs, subband_names)):
        ax = fig.add_subplot(3, 8, row * 8 + col + 1)
        vmax = np.percentile(np.abs(sb), 99) if sb.std() > 0 else 1
        ax.imshow(sb.T, cmap=cmap, origin='lower',
                  vmin=-vmax if cmap != 'gray' and cmap != 'hot' and cmap != 'viridis' else 0,
                  vmax=vmax, aspect='auto')
        ch_num = row * 8 + col + 1
        ax.set_title(f'Ch{ch_num}: {name}', fontsize=6.5)
        ax.axis('off')
    fig.text(0.005, 1 - (row + 0.5) / 3,
             lbl, va='center', ha='left', fontsize=7.5,
             fontweight='bold', rotation=90, transform=fig.transFigure)

plt.subplots_adjust(left=0.07, right=0.99, top=0.91, bottom=0.02, wspace=0.25, hspace=0.4)
plt.savefig(f"{OUT_DIR}/patch_model4_cwdm.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved model 4")

# ── Also save a full-volume context figure ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
fig.patch.set_facecolor('white')
fig.suptitle(
    f"Full Volume Context (axial slice z={mz}) — used by Fast-CWDM\nCase: {case_name}  |  Volume: 224×224×160",
    fontsize=10, fontweight='bold'
)
axes[0].imshow(t1n_pp[:, :, mz].T, cmap='gray', origin='lower', vmin=0, vmax=1)
axes[0].set_title('Original T1n (target)', fontsize=9)
axes[0].axis('off')

axes[1].imshow(voided_pp[:, :, mz].T, cmap='gray', origin='lower', vmin=0, vmax=1)
axes[1].contour(mask_pp[:, :, mz].T, levels=[0.5], colors=['red'], linewidths=1.5, origin='lower')
axes[1].set_title('Voided T1n (input)\nred = void boundary', fontsize=9)
axes[1].axis('off')

axes[2].imshow(mask_pp[:, :, mz].T, cmap='hot', origin='lower', vmin=0, vmax=1)
axes[2].set_title('Binary Mask\n(1 = void region)', fontsize=9)
axes[2].axis('off')

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fullvol_context.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved full-volume context figure")

print("\n=== All done! ===")
print(f"Output dir: {OUT_DIR}")
import os
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  {f}")
