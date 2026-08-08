"""
Synthetic Void Augmentation (Synth-Aug)
========================================
BraTS 2026 Task 4 — T1n MRI Inpainting

This module implements the Synth-Aug data augmentation strategy proposed in:
  "Synthetic Void Augmentation for MRI Inpainting: A BraTS 2026 Challenge Entry"

Core idea
---------
The BraTS inpainting challenge uses real tumor masks as voids during training,
but evaluates on arbitrary void shapes. This domain gap hurts generalisation.
Synth-Aug addresses it by randomly replacing the real tumor mask with a union
of 1–3 synthetic ellipsoidal voids (p=0.5), so the model learns to inpaint
under diverse void geometries.

Usage
-----
Standalone (NumPy only):

    from synth_aug import synth_aug_transform
    import nibabel as nib
    import numpy as np

    t1n  = nib.load("BraTS-GLI-00001-000-t1n.nii.gz").get_fdata()
    mask = nib.load("BraTS-GLI-00001-000-mask.nii.gz").get_fdata()
    brain = t1n > 0

    corrupted, mask_used = synth_aug_transform(t1n, mask, brain)

PyTorch Dataset integration — see SynthAugDataset below.

Requirements
------------
    numpy >= 1.21
    nibabel (optional, for the example)
    torch (optional, for SynthAugDataset)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Core geometry
# ---------------------------------------------------------------------------

def generate_ellipsoid(center, brain_mask, semi_axes, angle=0.0):
    """
    Generate a 3D binary ellipsoid mask clipped to the brain region.

    Parameters
    ----------
    center : (cx, cy, cz)  —  voxel coordinates of ellipsoid centre
    brain_mask : ndarray (H, W, D) bool  —  valid brain voxels
    semi_axes  : (rx, ry, rz)  —  semi-axis lengths in voxels
    angle : float  —  in-plane (x-y) rotation in radians

    Returns
    -------
    ndarray (H, W, D) float32  —  1 inside ellipsoid ∩ brain, 0 elsewhere
    """
    H, W, D = brain_mask.shape
    cx, cy, cz = int(center[0]), int(center[1]), int(center[2])
    rx, ry, rz = int(semi_axes[0]), int(semi_axes[1]), int(semi_axes[2])

    # Bounding box (add 5-voxel margin to avoid boundary artefacts)
    pad = 5
    xs = np.arange(max(0, cx - rx - pad), min(H, cx + rx + pad))
    ys = np.arange(max(0, cy - ry - pad), min(W, cy + ry + pad))
    zs = np.arange(max(0, cz - rz - pad), min(D, cz + rz + pad))

    if len(xs) == 0 or len(ys) == 0 or len(zs) == 0:
        return np.zeros(brain_mask.shape, dtype=np.float32)

    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")

    # Rotate in the x-y plane
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    dx = xx - cx;  dy = yy - cy;  dz = zz - cz
    dx_r = cos_a * dx - sin_a * dy
    dy_r = sin_a * dx + cos_a * dy

    inside = (dx_r / rx) ** 2 + (dy_r / ry) ** 2 + (dz / rz) ** 2 <= 1.0

    # IMPORTANT: fancy-index assignment requires filling a sub-array first
    # (A[fancy][bool] = v modifies a copy — use A[fancy] = sub instead)
    sub = np.zeros((len(xs), len(ys), len(zs)), dtype=np.float32)
    sub[inside] = 1.0
    ellipsoid = np.zeros(brain_mask.shape, dtype=np.float32)
    ellipsoid[xs[:, None, None], ys[None, :, None], zs[None, None, :]] = sub

    return ellipsoid * brain_mask.astype(np.float32)


def generate_synthetic_mask(brain_mask, rng=None,
                             n_ellipsoids=None,
                             semi_axis_range=(5, 45),
                             max_ellipsoids=3):
    """
    Generate a synthetic void mask as a union of random ellipsoids.

    Parameters
    ----------
    brain_mask     : ndarray (H, W, D) bool  —  valid brain voxels
    rng            : numpy.random.Generator (creates one if None)
    n_ellipsoids   : int or None — if None, sampled uniformly from
                     [1, max_ellipsoids]
    semi_axis_range: (a_min, a_max) in voxels, applied to all three axes
    max_ellipsoids : upper bound when n_ellipsoids is None

    Returns
    -------
    ndarray (H, W, D) float32  —  binary synthetic mask
    """
    if rng is None:
        rng = np.random.default_rng()

    brain_coords = np.argwhere(brain_mask)
    if len(brain_coords) == 0:
        return np.zeros(brain_mask.shape, dtype=np.float32)

    if n_ellipsoids is None:
        n_ellipsoids = int(rng.integers(1, max_ellipsoids + 1))

    a_min, a_max = semi_axis_range
    result = np.zeros(brain_mask.shape, dtype=np.float32)

    for _ in range(n_ellipsoids):
        # Sample centre uniformly from valid brain voxels
        idx    = int(rng.integers(0, len(brain_coords)))
        center = brain_coords[idx]

        rx    = int(rng.integers(a_min, a_max + 1))
        ry    = int(rng.integers(a_min, a_max + 1))
        rz    = int(rng.integers(a_min, a_max + 1))
        angle = float(rng.uniform(-np.pi / 4, np.pi / 4))

        e = generate_ellipsoid(center, brain_mask, (rx, ry, rz), angle)
        result = np.clip(result + e, 0.0, 1.0)

    return result


# ---------------------------------------------------------------------------
# Main augmentation transform
# ---------------------------------------------------------------------------

def synth_aug_transform(t1n_volume, real_mask, brain_mask,
                        p=0.5, rng=None,
                        semi_axis_range=(5, 45),
                        max_ellipsoids=3):
    """
    Apply Synthetic Void Augmentation to one training sample.

    With probability *p* the real tumor mask is replaced by a synthetic
    ellipsoidal mask before corrupting the volume.  The model therefore
    learns to recover signal under diverse void geometries, not just
    tumor shapes.

    Parameters
    ----------
    t1n_volume     : ndarray (H, W, D) float  —  T1n MRI volume
    real_mask      : ndarray (H, W, D)        —  real tumor mask (1 = void)
    brain_mask     : ndarray (H, W, D) bool   —  non-zero brain region
    p              : float  —  probability of synthetic substitution
    rng            : numpy.random.Generator
    semi_axis_range: (a_min, a_max) semi-axis range in voxels
    max_ellipsoids : maximum number of ellipsoids per sample

    Returns
    -------
    corrupted : ndarray (H, W, D) — t1n with void region zeroed
    mask_used : ndarray (H, W, D) — mask actually applied
    synthetic : bool              — True if synthetic mask was used
    """
    if rng is None:
        rng = np.random.default_rng()

    use_synthetic = rng.random() < p
    if use_synthetic:
        mask_used = generate_synthetic_mask(
            brain_mask, rng=rng,
            semi_axis_range=semi_axis_range,
            max_ellipsoids=max_ellipsoids,
        )
    else:
        mask_used = real_mask.astype(np.float32)

    corrupted = t1n_volume * (1.0 - mask_used)
    return corrupted, mask_used, use_synthetic


# ---------------------------------------------------------------------------
# PyTorch Dataset wrapper (optional — requires torch)
# ---------------------------------------------------------------------------

class SynthAugDataset:
    """
    Minimal PyTorch-compatible dataset that applies Synth-Aug on the fly.

    Each item is a dict with keys:
        "input"  : torch.Tensor (2, H, W, D)  — [corrupted_t1n, mask]
        "target" : torch.Tensor (1, H, W, D)  — original t1n (normalised)
        "mask"   : torch.Tensor (1, H, W, D)  — mask used (real or synth)

    Parameters
    ----------
    samples : list of dict, each with keys "t1n_path" and "mask_path"
    p_synth : float  —  Synth-Aug probability (default 0.5)
    patch_size : int  —  cubic crop size for training (default 96)
    norm_percentiles : (lo, hi) — percentile clipping for normalisation
    """

    def __init__(self, samples, p_synth=0.5, patch_size=96,
                 norm_percentiles=(1, 99)):
        try:
            import torch          # noqa: F401
            import nibabel as nib # noqa: F401
        except ImportError as e:
            raise ImportError("SynthAugDataset requires torch and nibabel") from e

        self.samples = samples
        self.p_synth = p_synth
        self.patch_size = patch_size
        self.norm_lo, self.norm_hi = norm_percentiles

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import torch
        import nibabel as nib

        s = self.samples[idx]
        rng = np.random.default_rng()   # fresh per sample for reproducibility

        t1n  = nib.load(s["t1n_path"]).get_fdata(dtype=np.float32)
        mask = nib.load(s["mask_path"]).get_fdata(dtype=np.float32)
        brain = t1n > 0

        # Normalise
        brain_vox = t1n[brain]
        lo = np.percentile(brain_vox, self.norm_lo)
        hi = np.percentile(brain_vox, self.norm_hi)
        t1n_norm = np.clip((t1n - lo) / (hi - lo + 1e-8), 0.0, 1.0)

        # Synth-Aug
        corrupted, mask_used, _ = synth_aug_transform(
            t1n_norm, mask, brain, p=self.p_synth, rng=rng
        )

        # Random cubic crop
        H, W, D = t1n_norm.shape
        ps = self.patch_size
        x0 = int(rng.integers(0, max(1, H - ps)))
        y0 = int(rng.integers(0, max(1, W - ps)))
        z0 = int(rng.integers(0, max(1, D - ps)))

        def crop(vol):
            return vol[x0:x0+ps, y0:y0+ps, z0:z0+ps]

        return {
            "input":  torch.tensor(
                np.stack([crop(corrupted), crop(mask_used)], axis=0)),
            "target": torch.tensor(crop(t1n_norm)[None]),
            "mask":   torch.tensor(crop(mask_used)[None]),
        }


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Synth-Aug demo")
    parser.add_argument("--t1n",  required=True, help="T1n NIfTI path")
    parser.add_argument("--mask", required=True, help="Mask NIfTI path")
    parser.add_argument("--out",  default="synth_mask.nii.gz",
                        help="Output synthetic mask path")
    parser.add_argument("--n", type=int, default=None,
                        help="Number of ellipsoids (default: random 1-3)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    try:
        import nibabel as nib
    except ImportError:
        sys.exit("nibabel required for demo: pip install nibabel")

    rng = np.random.default_rng(args.seed)

    t1n_img  = nib.load(args.t1n)
    t1n      = t1n_img.get_fdata(dtype=np.float32)
    mask     = nib.load(args.mask).get_fdata(dtype=np.float32)
    brain    = t1n > 0

    synth = generate_synthetic_mask(brain, rng=rng, n_ellipsoids=args.n)
    nib.save(nib.Nifti1Image(synth, t1n_img.affine), args.out)

    print(f"Synthetic mask: {int(synth.sum()):,} voxels  →  {args.out}")
    print(f"Real mask:      {int(mask.sum()):,} voxels")
