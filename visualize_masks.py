"""
Visualize real tumor masks vs synthetic ellipsoidal voids (Synth-Aug).

Produces a 3×4 grid figure:
  Rows    — small / medium / large void cases
  Columns — T1n brain slice | Real mask | Synth 1 ellipsoid | Synth 1-3 ellipsoids

Usage
-----
    python visualize_masks.py \
        --data_dir /path/to/brats2023/training \
        --cases BraTS-GLI-00001-000 BraTS-GLI-00002-000 BraTS-GLI-00003-000 \
        --out augmentation_comparison.png

    # Or auto-pick small/medium/large from a directory of cases:
    python visualize_masks.py --data_dir /path/to/brats2023/training --auto_pick --out fig.png
"""

import argparse
import os
import sys
import numpy as np

try:
    import nibabel as nib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError as e:
    sys.exit(f"Missing dependency: {e}\n  pip install nibabel matplotlib")

from synth_aug import generate_synth_mask_seeded


def generate_synth_mask_seeded(real_mask, brain_mask, n_ellipsoids, seed):
    """Wrapper with fixed seed for reproducible visualisation."""
    from synth_aug import generate_synthetic_mask
    rng = np.random.default_rng(seed)
    return generate_synthetic_mask(brain_mask, rng=rng, n_ellipsoids=n_ellipsoids)


def load_case(data_dir, case):
    t1n_path  = os.path.join(data_dir, case, f"{case}-t1n.nii.gz")
    mask_path = os.path.join(data_dir, case, f"{case}-mask.nii.gz")
    if not os.path.exists(t1n_path) or not os.path.exists(mask_path):
        raise FileNotFoundError(f"Missing files for {case} in {data_dir}")
    t1n  = nib.load(t1n_path).get_fdata(dtype=np.float32)
    mask = nib.load(mask_path).get_fdata(dtype=np.float32)
    return t1n, mask


def auto_pick_cases(data_dir, n_small=1, n_med=1, n_large=1):
    """Scan data_dir and return one small, one medium, one large void case."""
    cases_info = []
    for case in sorted(os.listdir(data_dir)):
        mask_path = os.path.join(data_dir, case, f"{case}-mask.nii.gz")
        t1n_path  = os.path.join(data_dir, case, f"{case}-t1n.nii.gz")
        if not os.path.exists(mask_path) or not os.path.exists(t1n_path):
            continue
        mask = nib.load(mask_path).get_fdata(dtype=np.float32)
        cases_info.append((case, int(mask.sum())))
    cases_info.sort(key=lambda x: x[1])
    N = len(cases_info)
    if N < 3:
        raise ValueError(f"Need at least 3 cases in {data_dir}, found {N}")
    return [
        cases_info[N // 6][0],
        cases_info[N // 2][0],
        cases_info[5 * N // 6][0],
    ]


def make_figure(data_dir, cases, out_path, dpi=200):
    col_labels = [
        "T1n Brain Slice",
        "Real Tumor Mask",
        "Synth-Aug Void\n(1 ellipsoid)",
        "Synth-Aug Void\n(1–3 ellipsoids)",
    ]
    row_labels = ["Small void", "Medium void", "Large void"]

    from synth_aug import generate_synthetic_mask

    fig, axes = plt.subplots(3, 4, figsize=(12.0, 9.5), dpi=dpi)
    fig.patch.set_facecolor("white")

    for col_idx, label in enumerate(col_labels):
        axes[0, col_idx].set_title(label, fontsize=10, fontweight="bold", pad=7)

    for row_idx, case in enumerate(cases):
        t1n, mask = load_case(data_dir, case)
        void_size = int(mask.sum())

        brain = (t1n > 0) | (mask > 0.5)
        brain_vox = t1n[t1n > 0]
        lo, hi = np.percentile(brain_vox, 1), np.percentile(brain_vox, 99)
        t1n_norm = np.clip((t1n - lo) / (hi - lo + 1e-8), 0.0, 1.0)

        # Best axial slice = most real mask voxels
        sl = int(np.argmax(mask.sum(axis=(0, 1))))

        rng1 = np.random.default_rng(7)
        rng2 = np.random.default_rng(42)
        synth1 = generate_synthetic_mask(brain, rng=rng1, n_ellipsoids=1)
        n_ell  = int(rng2.integers(2, 4))
        synth3 = generate_synthetic_mask(brain, rng=rng2, n_ellipsoids=n_ell)

        def best_slice(vol, preferred):
            if vol[:, :, preferred].sum() > 0:
                return preferred
            return int(np.argmax(vol.sum(axis=(0, 1))))

        def show_overlay(ax, brain_img, mask_vol, color, force_sl=None):
            use_sl = force_sl if force_sl is not None else best_slice(mask_vol, sl)
            brain_sl = np.rot90(brain_img[:, :, use_sl])
            mask_sl  = np.rot90(mask_vol[:, :, use_sl])
            ax.imshow(brain_sl, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
            if mask_sl.sum() > 0:
                overlay = np.zeros((*brain_sl.shape, 4))
                overlay[mask_sl > 0.5] = color
                ax.imshow(overlay, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)

        axes[row_idx, 0].imshow(
            np.rot90(t1n_norm[:, :, sl]), cmap="gray", vmin=0, vmax=1,
            interpolation="bilinear",
        )
        axes[row_idx, 0].set_xticks([]); axes[row_idx, 0].set_yticks([])
        for sp in axes[row_idx, 0].spines.values():
            sp.set_visible(False)

        show_overlay(axes[row_idx, 1], t1n_norm, mask,   [1.0, 0.15, 0.15, 0.65], force_sl=sl)
        show_overlay(axes[row_idx, 2], t1n_norm, synth1, [0.1, 0.50, 1.00, 0.65])
        show_overlay(axes[row_idx, 3], t1n_norm, synth3, [0.1, 0.50, 1.00, 0.65])

        axes[row_idx, 0].set_ylabel(
            f"{row_labels[row_idx]}\n({void_size:,} vox)", fontsize=9, labelpad=5
        )

    real_p  = mpatches.Patch(color=(1.0, 0.15, 0.15, 0.8), label="Real tumor mask (irregular)")
    synth_p = mpatches.Patch(color=(0.1, 0.50, 1.00, 0.8), label="Synthetic ellipsoid void (Synth-Aug)")
    fig.legend(handles=[real_p, synth_p], loc="lower center", ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, -0.01), frameon=True, edgecolor="gray")

    plt.suptitle(
        "Synthetic Void Augmentation: Real Tumor Masks vs Synthetic Ellipsoidal Voids",
        fontsize=11, fontweight="bold", y=1.01,
    )
    plt.tight_layout(pad=0.6, h_pad=0.5, w_pad=0.4)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualise Synth-Aug masks")
    parser.add_argument("--data_dir", required=True,
                        help="Root BraTS training directory (contains case subdirs)")
    parser.add_argument("--cases", nargs=3, metavar="CASE",
                        help="Exactly 3 case IDs (small, medium, large)")
    parser.add_argument("--auto_pick", action="store_true",
                        help="Auto-select small/medium/large cases from data_dir")
    parser.add_argument("--out", default="augmentation_comparison.png")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    if args.auto_pick:
        cases = auto_pick_cases(args.data_dir)
        print(f"Auto-selected cases: {cases}")
    elif args.cases:
        cases = args.cases
    else:
        parser.error("Provide --cases or --auto_pick")

    make_figure(args.data_dir, cases, args.out, dpi=args.dpi)


if __name__ == "__main__":
    main()
