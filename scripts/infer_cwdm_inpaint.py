"""
infer_cwdm_inpaint.py
Fast-cWDM Inpainting — Inference on BraTS 2026 validation set

Loads validation cases ({case}-t1n-voided.nii.gz + {case}-mask.nii.gz),
runs DDIM sampling in wavelet space, blends output with original.

Output: ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Validation-Results/
        {case}-t1n-inference.nii.gz

Usage:
  python infer_cwdm_inpaint.py \
    --checkpoint /data/experiments/cwdm-inpaint/best_model_cwdm_inpaint.pt \
    --val_dir    /data/brats2023/validation \
    --output_dir /data/experiments/cwdm-inpaint \
    --repo_dir   /tmp/fast-cwdm \
    --ddim_steps 50
"""

import argparse, os, sys, glob
import numpy as np
import nibabel
import torch
from pathlib import Path

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint',   required=True)
parser.add_argument('--val_dir',      required=True)
parser.add_argument('--output_dir',   required=True)
parser.add_argument('--repo_dir',     default='/tmp/fast-cwdm')
parser.add_argument('--ddim_steps',   type=int, default=50,
                    help='DDIM denoising steps. 50 is fast, 100 is better quality.')
parser.add_argument('--num_channels', type=int, default=32)
args = parser.parse_args()

# ── Setup ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, args.repo_dir)
os.makedirs(args.output_dir, exist_ok=True)

OUT_FOLDER = 'ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Validation-Results'
out_dir = os.path.join(args.output_dir, OUT_FOLDER)
os.makedirs(out_dir, exist_ok=True)
print(f'Predictions → {out_dir}')

from guided_diffusion.bratsloader import clip_and_normalize
from guided_diffusion.script_util import create_model_and_diffusion
from guided_diffusion import dist_util
from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D

dist_util.setup_dist(devices=[0])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if device.type == 'cuda':
    print(f'GPU   : {torch.cuda.get_device_name(0)}')
    free, total = torch.cuda.mem_get_info()
    print(f'VRAM  : {free/1e9:.1f}/{total/1e9:.1f} GB free/total')

# ── DWT helpers ───────────────────────────────────────────────────────────────
_dwt  = DWT_3D('haar').to(device)
_idwt = IDWT_3D('haar').to(device)

def dwt_8ch(vol):
    """(B,1,H,W,D) float32 → (B,8,H/2,W/2,D/2)"""
    if vol.abs().sum() == 0:
        B, _, H, W, D = vol.shape
        return torch.zeros(B, 8, H//2, W//2, D//2, device=device)
    LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = _dwt(vol.float().to(device))
    return torch.cat([LLL/3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)

def idwt_8ch(pred):
    """(B,8,H,W,D) float32 → (B,1,H*2,W*2,D*2)"""
    LLL = pred[:, 0:1] * 3.0  # undo /3 from dwt_8ch
    return _idwt(
        LLL.float(), pred[:,1:2].float(), pred[:,2:3].float(), pred[:,3:4].float(),
        pred[:,4:5].float(), pred[:,5:6].float(), pred[:,6:7].float(), pred[:,7:8].float()
    )

# ── Model ─────────────────────────────────────────────────────────────────────
ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

# Infer num_channels from checkpoint config if available
saved_cfg = ckpt.get('config', {})
num_channels = saved_cfg.get('num_channels', args.num_channels)

MODEL_CONFIG = dict(
    image_size            = 224,
    num_channels          = num_channels,
    num_res_blocks        = 2,
    channel_mult          = '1,2,2,4,4',
    learn_sigma           = False,
    class_cond            = False,
    use_checkpoint        = False,   # not needed at inference
    attention_resolutions = '',
    num_heads             = 1,
    num_head_channels     = -1,
    num_heads_upsample    = -1,
    use_scale_shift_norm  = False,
    dropout               = 0.0,
    resblock_updown       = True,
    use_fp16              = False,
    use_new_attention_order = False,
    dims                  = 3,
    num_groups            = 32,
    in_channels           = 24,   # 8 target + 8 voided + 8 mask
    out_channels          = 8,
    bottleneck_attention  = False,
    resample_2d           = False,
    additive_skips        = False,
    use_freq              = False,
    diffusion_steps       = 1000,
    noise_schedule        = 'linear',
    timestep_respacing    = '',
    use_kl                = False,
    predict_xstart        = True,
    rescale_timesteps     = False,
    rescale_learned_sigmas = False,
    dataset               = 'brats',
    mode                  = 'i2i',
    sample_schedule       = 'direct',
)

model, diffusion = create_model_and_diffusion(**MODEL_CONFIG)
model.load_state_dict(ckpt['model'])
model.to(device)
model.eval()
print(f'Loaded checkpoint from {args.checkpoint}  (iter {ckpt.get("iteration", "?")})')

# ── DDIM sampling ─────────────────────────────────────────────────────────────
def ddim_sample(cond, num_steps=50):
    """
    cond : (1, 16, H, W, D) — [voided_dwt | mask_dwt]
    Returns (1, 8, H, W, D) — predicted clean T1n in wavelet space
    """
    T    = diffusion.num_timesteps
    step = max(1, T // num_steps)
    ts   = list(range(0, T, step))[::-1]
    acp  = torch.from_numpy(diffusion.alphas_cumprod).float().to(device)

    H, W, D = cond.shape[2], cond.shape[3], cond.shape[4]
    x = torch.randn(1, 8, H, W, D, device=device)

    with torch.no_grad():
        for i, t_val in enumerate(ts):
            t_tensor = torch.tensor([t_val], device=device)
            x_input  = torch.cat([x, cond], dim=1)   # (1, 24, ...)
            pred_x0  = model(x_input, t_tensor).float()

            if i == len(ts) - 1:
                x = pred_x0
            else:
                t_prev  = ts[i + 1]
                a_t     = acp[t_val].sqrt()
                a_prev  = acp[t_prev].sqrt()
                s_t     = (1 - acp[t_val]).sqrt()
                s_prev  = (1 - acp[t_prev]).sqrt()
                eps     = (x - a_t * pred_x0) / s_t
                x       = a_prev * pred_x0 + s_prev * eps
    return x

# ── Preprocessing / postprocessing helpers ────────────────────────────────────
def pad_crop(vol_np):
    """
    vol_np : (H, W, D) numpy   (typically 240×240×155)
    Returns: (1, 1, 224, 224, 160) tensor on device, plus (orig_H, orig_W, orig_D)
             and the pad/crop parameters for inversion.
    """
    H, W, D = vol_np.shape

    # Normalize on non-zero voxels to avoid void zeros skewing stats
    nz = vol_np[vol_np > 0]
    if len(nz) > 100:
        lo, hi = np.percentile(nz, 1), np.percentile(nz, 99)
    else:
        lo, hi = vol_np.min(), vol_np.max()
    normed = np.clip((vol_np - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)

    # Pad to (240, 240, 160)
    t = np.zeros((240, 240, 160), dtype=np.float32)
    t[:min(H,240), :min(W,240), :min(D,160)] = normed[:min(H,240), :min(W,240), :min(D,160)]

    # Crop xy: 240 → 224 (remove 8 each side)
    t = t[8:-8, 8:-8, :]   # (224, 224, 160)

    return (
        torch.tensor(t).float().unsqueeze(0).unsqueeze(0).to(device),  # (1,1,224,224,160)
        (lo, hi, H, W, D)
    )

def unpad_uncrop(pred_224, orig_params):
    """
    pred_224   : (224, 224, 160) numpy, normalized [0,1]
    orig_params: (lo, hi, H, W, D)
    Returns    : (H, W, D) numpy, in original intensity scale
    """
    lo, hi, H, W, D = orig_params

    # Uncrop: embed 224×224 back into 240×240
    full = np.zeros((240, 240, 160), dtype=np.float32)
    full[8:-8, 8:-8, :] = pred_224

    # Crop to original spatial dims
    out = full[:H, :W, :D]

    # Denormalize
    return out * (hi - lo + 1e-8) + lo


# ── Discover validation cases ─────────────────────────────────────────────────
cases = []
for s in sorted(os.listdir(args.val_dir)):
    d = os.path.join(args.val_dir, s)
    if not os.path.isdir(d):
        continue
    voided_f = os.path.join(d, f'{s}-t1n-voided.nii.gz')
    mask_f   = os.path.join(d, f'{s}-mask.nii.gz')
    if os.path.exists(voided_f) and os.path.exists(mask_f):
        cases.append({'name': s, 'voided': voided_f, 'mask': mask_f})

print(f'Found {len(cases)} validation cases')

# ── Inference loop ────────────────────────────────────────────────────────────
for i, c in enumerate(cases):
    name     = c['name']
    out_path = os.path.join(out_dir, f'{name}-t1n-inference.nii.gz')

    if os.path.exists(out_path):
        print(f'[{i+1}/{len(cases)}] {name}  already exists, skipping')
        continue

    print(f'[{i+1}/{len(cases)}] {name}', flush=True)

    # Load originals
    voided_img = nibabel.load(c['voided'])
    voided_np  = voided_img.get_fdata(dtype=np.float32)   # (H, W, D)
    mask_np    = nibabel.load(c['mask']).get_fdata(dtype=np.float32)
    affine     = voided_img.affine
    header     = voided_img.header

    # Preprocess
    voided_t, vparams  = pad_crop(voided_np)   # (1,1,224,224,160)
    # Mask: same pad/crop but no intensity normalization
    lo_v, hi_v, H, W, D = vparams
    mask_pad = np.zeros((240, 240, 160), dtype=np.float32)
    mask_pad[:min(H,240), :min(W,240), :min(D,160)] = \
        mask_np[:min(H,240), :min(W,240), :min(D,160)]
    mask_pad = mask_pad[8:-8, 8:-8, :]   # (224, 224, 160)
    mask_t   = torch.tensor(mask_pad).float().unsqueeze(0).unsqueeze(0).to(device)  # (1,1,224,224,160)

    # DWT conditioning
    voided_dwt = dwt_8ch(voided_t)   # (1,8,112,112,80)
    mask_dwt   = dwt_8ch(mask_t)     # (1,8,112,112,80)
    cond       = torch.cat([voided_dwt, mask_dwt], dim=1)  # (1,16,112,112,80)

    # DDIM sampling
    pred_dwt = ddim_sample(cond, num_steps=args.ddim_steps)   # (1,8,112,112,80)

    # IDWT → image space
    pred_vol = idwt_8ch(pred_dwt)      # (1,1,224,224,160)
    pred_224 = pred_vol[0, 0].cpu().numpy()
    pred_224 = np.clip(pred_224, 0.0, 1.0)

    # Denormalize to original scale
    pred_orig = unpad_uncrop(pred_224, vparams)  # (H, W, D) in original scale

    # Blend: keep voided image outside mask, use prediction inside mask
    mask_b   = (mask_np > 0.5).astype(np.float32)
    result   = voided_np.copy()
    result[mask_b > 0.5] = pred_orig[mask_b > 0.5]

    # Save
    out_img = nibabel.Nifti1Image(result, affine=affine, header=header)
    nibabel.save(out_img, out_path)
    print(f'   → saved', flush=True)

total = len([f for f in os.listdir(out_dir) if f.endswith('.nii.gz')])
print(f'\nDone. {total}/{len(cases)} predictions in {out_dir}/')
