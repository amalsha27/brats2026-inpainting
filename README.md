# BraTS 2026 Inpainting — Synthetic Void Augmentation

Code for **"Synthetic Void Augmentation for MRI Inpainting"**, submitted to the
BraTS 2026 Task 4 Inpainting Challenge (MICCAI Workshop).

## Method

The BraTS inpainting challenge trains models on real tumor void masks but
evaluates on arbitrary void geometries. This domain gap hurts generalisation.

**Synth-Aug** addresses it by randomly replacing the real tumor mask with a
union of 1–3 synthetic ellipsoidal voids (probability p = 0.5) during training.
The model therefore learns to inpaint under diverse void shapes, not just
tumor-derived ones.

```
With prob p:   real mask  ──→  union of 1–3 random ellipsoids ∩ brain
Otherwise:     real mask  ──→  real mask (unchanged)
```

Each ellipsoid has:
- Centre sampled uniformly from valid brain voxels
- Semi-axes drawn independently from U(5, 45) voxels
- Optional in-plane (x-y) rotation sampled from U(−π/4, π/4)

## Files

| File | Description |
|---|---|
| `synth_aug.py` | Standalone augmentation implementation (NumPy + optional PyTorch) |
| `visualize_masks.py` | Generate the mask comparison figure |
| `requirements.txt` | Python dependencies |
| `nautilus/` | Kubernetes job YAMLs for training and evaluation on NRP Nautilus |

## Quick start

```bash
pip install numpy nibabel matplotlib

# Generate a synthetic void mask for one case
python synth_aug.py \
    --t1n  /data/BraTS-GLI-00001-000/BraTS-GLI-00001-000-t1n.nii.gz \
    --mask /data/BraTS-GLI-00001-000/BraTS-GLI-00001-000-mask.nii.gz \
    --out  synth_mask.nii.gz \
    --n 2 --seed 42
```

```python
# Python API
from synth_aug import synth_aug_transform
import nibabel as nib
import numpy as np

t1n  = nib.load("...-t1n.nii.gz").get_fdata()
mask = nib.load("...-mask.nii.gz").get_fdata()
brain = t1n > 0

corrupted, mask_used, was_synthetic = synth_aug_transform(
    t1n, mask, brain, p=0.5
)
```

```python
# PyTorch Dataset
from synth_aug import SynthAugDataset
from torch.utils.data import DataLoader

samples = [
    {"t1n_path": ".../case1-t1n.nii.gz", "mask_path": ".../case1-mask.nii.gz"},
    ...
]
dataset = SynthAugDataset(samples, p_synth=0.5, patch_size=96)
loader  = DataLoader(dataset, batch_size=2, num_workers=4)
```

## Visualisation

```bash
# Produce the 3×4 comparison figure (auto-picks small/medium/large cases)
python visualize_masks.py \
    --data_dir /path/to/brats2023/training \
    --auto_pick \
    --out augmentation_comparison.png
```

## Results (internal validation, 125 cases)

| Model | SSIM ↑ | PSNR ↑ | MSE ↓ |
|---|---|---|---|
| Improved Attention U-Net | 0.9113 | 16.643 dB | 0.02382 |
| + Synth-Aug (this work) | **0.9175** | **16.753 dB** | 0.02462 |

Synth-Aug also improves consistency across void sizes (see paper for stratification by quartile).

## Citation

```bibtex
@inproceedings{brats2026inpainting,
  title     = {Synthetic Void Augmentation for {MRI} Inpainting},
  booktitle = {BraTS 2026 Challenge Proceedings, MICCAI Workshop},
  year      = {2026},
}
```

## Data

Training data: [BraTS 2023 / 2026 GLI dataset](https://www.synapse.org/#!Synapse:syn51156910/wiki/).
Download requires Synapse registration and challenge sign-up.
