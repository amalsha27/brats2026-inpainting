#!/usr/bin/env python3
"""
synth_aug_v2.py - Void shape augmentation strategies for 3D brain MRI inpainting.

Available strategies (pass as --aug_strategy to train.py):
  none      - no synthetic voids; use real BraTS mask as-is
  ellipsoid - random ellipsoidal voids (Synth-Aug baseline)
  sphere    - isotropic sphere voids
  cuboid    - randomly oriented box voids
  morphed   - elastically deformed real BraTS tumor masks
  mixed     - random choice from ellipsoid / sphere / cuboid / morphed

Unified entry point:
  mask = get_void_mask(strategy, shape, ...)
"""

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


# ── Ellipsoid ──────────────────────────────────────────────────────────────

def generate_ellipsoid_mask(
    shape,
    n_voids=(1, 3),
    semi_axes_range=(5, 45),
    max_rotation_deg=180,
    rng=None,
):
    """Random ellipsoidal voids with in-plane (z-axis) rotation."""
    if rng is None:
        rng = np.random.default_rng()
    D, H, W = shape
    mask = np.zeros(shape, dtype=np.uint8)
    n = rng.integers(n_voids[0], n_voids[1] + 1)

    for _ in range(n):
        a, b, c = [rng.uniform(*semi_axes_range) for _ in range(3)]
        pad = int(max(a, b, c)) + 2
        cx = int(rng.integers(pad, max(pad + 1, D - pad)))
        cy = int(rng.integers(pad, max(pad + 1, H - pad)))
        cz = int(rng.integers(pad, max(pad + 1, W - pad)))

        theta = np.radians(rng.uniform(0, max_rotation_deg))
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        x0, x1 = max(0, cx - pad), min(D, cx + pad)
        y0, y1 = max(0, cy - pad), min(H, cy + pad)
        z0, z1 = max(0, cz - pad), min(W, cz + pad)

        xx, yy, zz = np.meshgrid(
            np.arange(x0, x1) - cx,
            np.arange(y0, y1) - cy,
            np.arange(z0, z1) - cz,
            indexing='ij',
        )
        xx_r = cos_t * xx - sin_t * yy
        yy_r = sin_t * xx + cos_t * yy

        mask[x0:x1, y0:y1, z0:z1] |= ((xx_r / a)**2 + (yy_r / b)**2 + (zz / c)**2 <= 1)
    return mask


# ── Sphere ─────────────────────────────────────────────────────────────────

def generate_sphere_mask(
    shape,
    n_voids=(1, 3),
    radius_range=(8, 40),
    rng=None,
):
    """Isotropic sphere voids."""
    if rng is None:
        rng = np.random.default_rng()
    D, H, W = shape
    mask = np.zeros(shape, dtype=np.uint8)
    n = rng.integers(n_voids[0], n_voids[1] + 1)

    for _ in range(n):
        r = rng.uniform(*radius_range)
        pad = int(r) + 2
        cx = int(rng.integers(pad, max(pad + 1, D - pad)))
        cy = int(rng.integers(pad, max(pad + 1, H - pad)))
        cz = int(rng.integers(pad, max(pad + 1, W - pad)))

        x0, x1 = max(0, cx - pad), min(D, cx + pad)
        y0, y1 = max(0, cy - pad), min(H, cy + pad)
        z0, z1 = max(0, cz - pad), min(W, cz + pad)

        xx, yy, zz = np.meshgrid(
            np.arange(x0, x1) - cx,
            np.arange(y0, y1) - cy,
            np.arange(z0, z1) - cz,
            indexing='ij',
        )
        mask[x0:x1, y0:y1, z0:z1] |= (xx**2 + yy**2 + zz**2 <= r**2)
    return mask


# ── Cuboid ─────────────────────────────────────────────────────────────────

def generate_cuboid_mask(
    shape,
    n_voids=(1, 3),
    side_range=(10, 60),
    max_rotation_deg=30,
    rng=None,
):
    """Randomly oriented box voids."""
    if rng is None:
        rng = np.random.default_rng()
    D, H, W = shape
    mask = np.zeros(shape, dtype=np.uint8)
    n = rng.integers(n_voids[0], n_voids[1] + 1)

    for _ in range(n):
        hx = rng.uniform(side_range[0] / 2, side_range[1] / 2)
        hy = rng.uniform(side_range[0] / 2, side_range[1] / 2)
        hz = rng.uniform(side_range[0] / 2, side_range[1] / 2)
        pad = int(max(hx, hy, hz) * 1.5) + 2

        cx = int(rng.integers(pad, max(pad + 1, D - pad)))
        cy = int(rng.integers(pad, max(pad + 1, H - pad)))
        cz = int(rng.integers(pad, max(pad + 1, W - pad)))

        theta = np.radians(rng.uniform(-max_rotation_deg, max_rotation_deg))
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        x0, x1 = max(0, cx - pad), min(D, cx + pad)
        y0, y1 = max(0, cy - pad), min(H, cy + pad)
        z0, z1 = max(0, cz - pad), min(W, cz + pad)

        xx, yy, zz = np.meshgrid(
            np.arange(x0, x1) - cx,
            np.arange(y0, y1) - cy,
            np.arange(z0, z1) - cz,
            indexing='ij',
        )
        xx_r = cos_t * xx - sin_t * yy
        yy_r = sin_t * xx + cos_t * yy

        mask[x0:x1, y0:y1, z0:z1] |= (
            (np.abs(xx_r) <= hx) & (np.abs(yy_r) <= hy) & (np.abs(zz) <= hz)
        )
    return mask


# ── Morphed Tumor ──────────────────────────────────────────────────────────

def _elastic_deform_3d(mask, alpha=200.0, sigma=10.0, rng=None):
    """Random elastic deformation of a binary mask."""
    if rng is None:
        rng = np.random.default_rng()
    shape = mask.shape
    dx = gaussian_filter(rng.standard_normal(shape).astype(np.float32), sigma) * alpha
    dy = gaussian_filter(rng.standard_normal(shape).astype(np.float32), sigma) * alpha
    dz = gaussian_filter(rng.standard_normal(shape).astype(np.float32), sigma) * alpha

    x, y, z = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij'
    )
    coords = [
        np.clip(x + dx, 0, shape[0] - 1).ravel(),
        np.clip(y + dy, 0, shape[1] - 1).ravel(),
        np.clip(z + dz, 0, shape[2] - 1).ravel(),
    ]
    deformed = map_coordinates(mask.astype(np.float32), coords, order=1)
    return (deformed.reshape(shape) > 0.5).astype(np.uint8)


def generate_morphed_tumor_mask(
    shape,
    brats_mask_paths,
    alpha_range=(100, 400),
    sigma_range=(8, 15),
    n_voids=(1, 2),
    rng=None,
):
    """
    Load a random BraTS segmentation, elastically deform it, and place it
    at a random location in the target volume.

    Parameters
    ----------
    brats_mask_paths : list[str]
        Paths to BraTS *_seg.nii.gz files (any label > 0 = tumor).
    """
    import nibabel as nib
    from scipy.ndimage import zoom as ndzoom

    if rng is None:
        rng = np.random.default_rng()
    D, H, W = shape
    mask_out = np.zeros(shape, dtype=np.uint8)
    n = rng.integers(n_voids[0], n_voids[1] + 1)

    for _ in range(n):
        seg_path = brats_mask_paths[rng.integers(len(brats_mask_paths))]
        seg = nib.load(seg_path).get_fdata()
        tumor = (seg > 0).astype(np.uint8)

        coords = np.argwhere(tumor)
        if len(coords) == 0:
            continue
        mn, mx = coords.min(axis=0), coords.max(axis=0)
        tumor_crop = tumor[mn[0]:mx[0]+1, mn[1]:mx[1]+1, mn[2]:mx[2]+1]

        # Elastic deformation
        alpha = float(rng.uniform(*alpha_range))
        sigma = float(rng.uniform(*sigma_range))
        tumor_def = _elastic_deform_3d(tumor_crop, alpha=alpha, sigma=sigma, rng=rng)

        # Scale down if too large for target volume
        td, th, tw = tumor_def.shape
        if td >= D or th >= H or tw >= W:
            scale = min(D / td, H / th, W / tw) * 0.8
            tumor_def = (ndzoom(tumor_def.astype(np.float32), scale, order=0) > 0.5).astype(np.uint8)
            td, th, tw = tumor_def.shape

        # Random placement
        ox = int(rng.integers(0, max(1, D - td)))
        oy = int(rng.integers(0, max(1, H - th)))
        oz = int(rng.integers(0, max(1, W - tw)))
        mask_out[ox:ox+td, oy:oy+th, oz:oz+tw] |= tumor_def

    return mask_out


# ── Mixed ──────────────────────────────────────────────────────────────────

def generate_mixed_mask(shape, brats_mask_paths=None, rng=None):
    """Randomly sample one strategy per call."""
    if rng is None:
        rng = np.random.default_rng()

    choices = ["ellipsoid", "sphere", "cuboid"]
    if brats_mask_paths:
        choices.append("morphed")

    choice = rng.choice(choices)
    if choice == "ellipsoid":
        return generate_ellipsoid_mask(shape, rng=rng)
    elif choice == "sphere":
        return generate_sphere_mask(shape, rng=rng)
    elif choice == "cuboid":
        return generate_cuboid_mask(shape, rng=rng)
    else:
        return generate_morphed_tumor_mask(shape, brats_mask_paths, rng=rng)


# ── Unified API ────────────────────────────────────────────────────────────

STRATEGIES = ["none", "ellipsoid", "sphere", "cuboid", "morphed", "mixed"]


def get_void_mask(
    strategy: str,
    shape: tuple,
    aug_prob: float = 0.5,
    brats_mask_paths=None,
    existing_mask=None,
    rng=None,
) -> np.ndarray:
    """
    Generate a binary void mask.

    Parameters
    ----------
    strategy : str
        One of STRATEGIES.
    shape : tuple
        3D volume shape (D, H, W).
    aug_prob : float
        Probability of applying augmentation (not used for 'none' or 'morphed').
    brats_mask_paths : list[str] or None
        Required for 'morphed' and 'mixed'.
    existing_mask : np.ndarray or None
        For 'none': pass real tumor segmentation mask.
    rng : np.random.Generator or None

    Returns
    -------
    mask : np.ndarray, shape=shape, dtype=uint8
        1 = void (region to reconstruct), 0 = observed (keep as-is)
    """
    if rng is None:
        rng = np.random.default_rng()

    if strategy == "none":
        if existing_mask is not None:
            return (existing_mask > 0).astype(np.uint8)
        return np.zeros(shape, dtype=np.uint8)

    # Probabilistic gate for synthetic strategies
    if strategy in ("ellipsoid", "sphere", "cuboid", "mixed"):
        if rng.random() > aug_prob:
            return np.zeros(shape, dtype=np.uint8)

    if strategy == "ellipsoid":
        return generate_ellipsoid_mask(shape, rng=rng)
    elif strategy == "sphere":
        return generate_sphere_mask(shape, rng=rng)
    elif strategy == "cuboid":
        return generate_cuboid_mask(shape, rng=rng)
    elif strategy == "morphed":
        if not brats_mask_paths:
            raise ValueError("brats_mask_paths required for 'morphed' strategy")
        return generate_morphed_tumor_mask(shape, brats_mask_paths, rng=rng)
    elif strategy == "mixed":
        return generate_mixed_mask(shape, brats_mask_paths=brats_mask_paths, rng=rng)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {STRATEGIES}")


# ── Quick sanity check ─────────────────────────────────────────────────────

if __name__ == "__main__":
    shape = (240, 240, 155)
    rng = np.random.default_rng(42)

    for strat in ["ellipsoid", "sphere", "cuboid", "mixed"]:
        mask = get_void_mask(strat, shape, aug_prob=1.0, rng=rng)
        vox = mask.sum()
        pct = 100 * vox / np.prod(shape)
        print(f"{strat:12s}  voxels={vox:7d}  ({pct:.1f}% of volume)")

    print("\nAll generators OK.")
