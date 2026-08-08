"""
train_cwdm_inpaint.py
Fast-cWDM adapted for BraTS 2026 Inpainting Challenge
Based on DSP Project train_option2.py (Dr. Lina Chato, USD)

Architecture:
  in_channels = 24 = 8 (target DWT) + 8 (voided DWT) + 8 (mask DWT)
  Target     : original T1n ({case}-t1n.nii.gz)
  Condition  : voided T1n ({case}-t1n-voided.nii.gz) + binary mask ({case}-mask-unhealthy.nii.gz)

Training augmentation:
  (1-synth_prob) real void  (pre-voided data: t1n-voided + mask-unhealthy)
  synth_prob     synthetic  (apply random ellipsoid void to original t1n)

Usage:
  python train_cwdm_inpaint.py \
    --train_dir /data/brats2023/training \
    --save_dir  /data/experiments/cwdm-inpaint \
    --repo_dir  /tmp/fast-cwdm \
    --iters 300000
"""

import argparse, os, sys, glob, random
import numpy as np
import nibabel
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--train_dir',    default='/data/brats2023/training')
parser.add_argument('--save_dir',     default='/data/experiments/cwdm-inpaint')
parser.add_argument('--repo_dir',     default='/tmp/fast-cwdm')
parser.add_argument('--iters',        type=int,   default=300_000)
parser.add_argument('--lr',           type=float, default=1e-5)
parser.add_argument('--grad_accum',   type=int,   default=4)
parser.add_argument('--num_channels', type=int,   default=32,
                    help='UNet base channels. Use 16 for 11GB GPU, 32 for 32GB+')
parser.add_argument('--synth_prob',   type=float, default=0.5,
                    help='Fraction of iterations using synthetic (ellipsoid) void')
parser.add_argument('--resume',       default='')
parser.add_argument('--log_every',    type=int,   default=100)
parser.add_argument('--val_every',    type=int,   default=5000)
parser.add_argument('--save_every',   type=int,   default=10000)
parser.add_argument('--num_workers',  type=int,   default=4)
args = parser.parse_args()

# ── Setup ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, args.repo_dir)
os.makedirs(args.save_dir, exist_ok=True)

from guided_diffusion.bratsloader import clip_and_normalize
from guided_diffusion.script_util import create_model_and_diffusion
from guided_diffusion import dist_util
from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D

dist_util.setup_dist(devices=[0])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device : {device}')
if device.type == 'cuda':
    print(f'GPU    : {torch.cuda.get_device_name(0)}')
    print(f'PyTorch: {torch.__version__}')
    free, total = torch.cuda.mem_get_info()
    print(f'VRAM   : {free/1e9:.1f} GB free / {total/1e9:.1f} GB total')

# ── DWT helpers — always float32 to avoid checkpoint/autocast conflicts ────────
_dwt  = DWT_3D('haar').to(device)
_idwt = IDWT_3D('haar').to(device)

def dwt(x):
    return _dwt(x.float())

def idwt(LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH):
    return _idwt(
        LLL.float(), LLH.float(), LHL.float(), LHH.float(),
        HLL.float(), HLH.float(), HHL.float(), HHH.float()
    )

def dwt_8ch(vol):
    """(B,1,H,W,D) → (B,8,H/2,W/2,D/2). Zero tensor stays zero."""
    if vol.abs().sum() == 0:
        B, _, H, W, D = vol.shape
        return torch.zeros(B, 8, H//2, W//2, D//2, device=device)
    LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(vol.to(device))
    return torch.cat([LLL/3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)

# ── Synthetic void generation ─────────────────────────────────────────────────
def make_brain_mask(vol_np):
    """Simple brain mask: non-zero voxels in the normalized volume."""
    return vol_np > 0.02

def generate_synthetic_void(vol_shape, brain_mask_np=None):
    """
    Create a random ellipsoidal void mask.
    Returns float32 numpy array with 1s inside void, 0s outside.
    """
    D, H, W = vol_shape
    void = np.zeros((D, H, W), dtype=np.float32)

    if brain_mask_np is not None:
        positions = np.argwhere(brain_mask_np)
    else:
        positions = None

    n_ellipsoids = random.randint(1, 3)
    for _ in range(n_ellipsoids):
        if positions is not None and len(positions) > 0:
            idx = random.randint(0, len(positions) - 1)
            cd, ch, cw = positions[idx]
        else:
            cd = random.randint(D//4, 3*D//4)
            ch = random.randint(H//4, 3*H//4)
            cw = random.randint(W//4, 3*W//4)

        rd = random.randint(5, 45)
        rh = random.randint(5, 45)
        rw = random.randint(5, 45)

        d0 = max(0, cd - rd); d1 = min(D, cd + rd + 1)
        h0 = max(0, ch - rh); h1 = min(H, ch + rh + 1)
        w0 = max(0, cw - rw); w1 = min(W, cw + rw + 1)

        dd = np.arange(d0, d1) - cd
        hh = np.arange(h0, h1) - ch
        ww = np.arange(w0, w1) - cw
        DD, HH, WW = np.meshgrid(dd, hh, ww, indexing='ij')
        ellipsoid = (DD/rd)**2 + (HH/rh)**2 + (WW/rw)**2 <= 1.0
        void[d0:d1, h0:h1, w0:w1] = np.maximum(
            void[d0:d1, h0:h1, w0:w1], ellipsoid.astype(np.float32)
        )

    return void

# ── Dataset ───────────────────────────────────────────────────────────────────
VOL_SHAPE = (1, 224, 224, 160)

def preprocess_vol(vol_np):
    """Clip+normalize, pad z 155→160, crop xy 240→224. Returns (1,224,224,160) float32."""
    vol = clip_and_normalize(vol_np)
    t = torch.zeros(1, 240, 240, 160)
    d = min(vol.shape[2] if vol.ndim == 3 else vol.shape[0], 155)
    if vol.ndim == 3:
        t[:, :vol.shape[0], :vol.shape[1], :d] = torch.tensor(vol[:, :, :d] if vol.ndim == 3 else vol).float()
    else:
        t[:, :, :, :d] = torch.tensor(vol).float().unsqueeze(0)[:, :240, :240, :d]
    return t[:, 8:-8, 8:-8, :].float()   # (1, 224, 224, 160)

def load_nifti(path):
    """Load NIfTI and return numpy array (H,W,D)."""
    try:
        return nibabel.load(path).get_fdata().astype(np.float32)
    except Exception as e:
        print(f'WARNING: failed to load {path}: {e}', flush=True)
        return None

class InpaintDataset(torch.utils.data.Dataset):
    """
    Loads pre-computed (t1n, t1n-voided, mask-unhealthy) triplets.
    50% of samples replace the real void with a synthetic ellipsoid void.
    """
    def __init__(self, sample_dirs, synth_prob=0.5):
        self.dirs = sample_dirs
        self.synth_prob = synth_prob

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        d    = Path(self.dirs[idx])
        name = d.name

        t1n_np    = load_nifti(str(d / f'{name}-t1n.nii.gz'))
        voided_np = load_nifti(str(d / f'{name}-t1n-voided.nii.gz'))
        mask_np   = load_nifti(str(d / f'{name}-mask-unhealthy.nii.gz'))

        if t1n_np is None or voided_np is None or mask_np is None:
            return self._zeros()

        # Preprocess all to (1, 224, 224, 160)
        t1n    = preprocess_vol(t1n_np)
        t1n_arr = t1n.squeeze().numpy()

        use_synth = random.random() < self.synth_prob
        if use_synth:
            # Synthetic void: generate random ellipsoid, apply to original t1n
            brain_m   = make_brain_mask(t1n_arr)
            void_mask = generate_synthetic_void(t1n_arr.shape, brain_m)
            void_t    = torch.tensor(void_mask).float()
            voided    = t1n.squeeze() * (1.0 - void_t)
        else:
            # Real void: use pre-computed voided image + mask
            voided = preprocess_vol(voided_np).squeeze()
            # Mask: pad/crop same as images but keep binary (no clip_and_normalize)
            m = np.zeros((240, 240, 160), dtype=np.float32)
            dz = min(mask_np.shape[2], 155)
            m[:min(mask_np.shape[0],240), :min(mask_np.shape[1],240), :dz] = \
                mask_np[:min(mask_np.shape[0],240), :min(mask_np.shape[1],240), :dz]
            void_t = torch.tensor(m[8:-8, 8:-8, :]).float()  # (224,224,160)

        return {
            't1n':    t1n,                   # (1,224,224,160) — TARGET
            'voided': voided.unsqueeze(0),   # (1,224,224,160) — condition
            'mask':   void_t.unsqueeze(0),   # (1,224,224,160) — condition
        }

    def _zeros(self):
        z = torch.zeros(*VOL_SHAPE)
        return {'t1n': z, 'voided': z, 'mask': z}


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch

# ── Model config ──────────────────────────────────────────────────────────────
MODEL_CONFIG = dict(
    image_size            = 224,
    num_channels          = args.num_channels,
    num_res_blocks        = 2,
    channel_mult          = '1,2,2,4,4',
    learn_sigma           = False,
    class_cond            = False,
    use_checkpoint        = True,
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
    in_channels           = 24,   # 8 target + 8 voided + 8 mask  ← KEY CHANGE
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

# ── Loss ──────────────────────────────────────────────────────────────────────
def compute_loss(model, batch, diffusion):
    """
    Forward diffusion loss.
    Input to UNet: [noisy_target_8ch | voided_8ch | mask_8ch] = 24ch total
    """
    t1n    = batch['t1n']    # (B,1,224,224,160)
    voided = batch['voided'] # (B,1,224,224,160)
    mask   = batch['mask']   # (B,1,224,224,160)

    if t1n.abs().sum() == 0:
        return None

    B = t1n.shape[0]

    # DWT everything — always float32
    x0         = dwt_8ch(t1n)                                    # (B,8,112,112,80)
    voided_dwt = dwt_8ch(voided)                                  # (B,8,112,112,80)
    mask_dwt   = dwt_8ch(mask)                                    # (B,8,112,112,80)
    cond       = torch.cat([voided_dwt, mask_dwt], dim=1)         # (B,16,112,112,80)

    # Forward diffusion
    t        = torch.randint(0, diffusion.num_timesteps, (B,), device=device)
    noise    = torch.randn_like(x0)
    sqrt_acp = torch.from_numpy(
        diffusion.sqrt_alphas_cumprod).float().to(device)[t][:,None,None,None,None]
    sqrt_1ma = torch.from_numpy(
        diffusion.sqrt_one_minus_alphas_cumprod).float().to(device)[t][:,None,None,None,None]
    x_t      = sqrt_acp * x0 + sqrt_1ma * noise                  # (B,8,...)

    # UNet: 24ch input
    x_t_cond = torch.cat([x_t, cond], dim=1)                     # (B,24,...)
    pred     = model(x_t_cond, t)
    return F.mse_loss(pred.float(), x0)


# ── Discover training cases ───────────────────────────────────────────────────
def find_cases(data_dir):
    """Require t1n.nii.gz + t1n-voided.nii.gz + mask-unhealthy.nii.gz"""
    cases = []
    for d in sorted(Path(data_dir).iterdir()):
        if not d.is_dir():
            continue
        n = d.name
        if (d / f'{n}-t1n.nii.gz').exists() and \
           (d / f'{n}-t1n-voided.nii.gz').exists() and \
           (d / f'{n}-mask-unhealthy.nii.gz').exists():
            cases.append(str(d))
    return cases


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    cases = find_cases(args.train_dir)
    print(f'Total cases found: {len(cases)}')

    train_c, valtest = train_test_split(cases, test_size=0.15, random_state=42)
    val_c, _         = train_test_split(valtest, test_size=0.5, random_state=42)
    print(f'Train={len(train_c)}  Val={len(val_c)}')

    train_loader = torch.utils.data.DataLoader(
        InpaintDataset(train_c, synth_prob=args.synth_prob),
        batch_size=1, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        InpaintDataset(val_c, synth_prob=0.0),  # real masks only for val
        batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    # Build model
    model, diffusion = create_model_and_diffusion(**MODEL_CONFIG)
    model.to(device)
    diffusion.mode = 'i2i'
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Parameters  : {total_params/1e6:.2f}M')
    print(f'in_channels : {MODEL_CONFIG["in_channels"]}  (8 target + 8 voided + 8 mask)')

    num_opt_steps = args.iters // args.grad_accum
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_opt_steps, eta_min=1e-7)

    history    = {'train_iters': [], 'train_losses': [], 'val_iters': [], 'val_losses': []}
    best_val   = float('inf')
    start_iter = 1

    if args.resume and os.path.exists(args.resume):
        ckpt       = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        history    = ckpt.get('history', history)
        best_val   = ckpt.get('val_loss', best_val)
        start_iter = ckpt['iteration'] + 1
        steps_done = (start_iter - 1) // args.grad_accum
        for _ in range(steps_done):
            scheduler.step()
        print(f'Resumed from iteration {ckpt["iteration"]}')

    print(f'\nStarting CWDM inpainting training')
    print(f'  iters={args.iters:,}  lr={args.lr}  grad_accum={args.grad_accum}')
    print(f'  synth_prob={args.synth_prob}  num_channels={args.num_channels}')
    print('-' * 65)

    train_iter     = infinite_loader(train_loader)
    running_losses = []
    optimizer.zero_grad()

    for iteration in range(start_iter, args.iters + 1):
        model.train()
        batch = next(train_iter)

        # float32 throughout — avoids checkpoint+autocast dtype conflict
        loss = compute_loss(model, batch, diffusion)
        if loss is None:
            continue

        (loss / args.grad_accum).backward()
        running_losses.append(loss.item())

        if iteration % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        if iteration % args.log_every == 0:
            avg = np.mean(running_losses[-args.log_every:])
            history['train_iters'].append(iteration)
            history['train_losses'].append(avg)
            print(f'Iter {iteration:7d}/{args.iters} | loss={avg:.4f}'
                  f' | lr={scheduler.get_last_lr()[0]:.2e}', flush=True)

        if iteration % args.val_every == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vb in val_loader:
                    l = compute_loss(model, vb, diffusion)
                    if l is not None:
                        val_losses.append(l.item())
            val_avg = np.mean(val_losses) if val_losses else 0.0
            history['val_iters'].append(iteration)
            history['val_losses'].append(val_avg)

            if val_avg < best_val:
                best_val = val_avg
                torch.save({
                    'iteration': iteration,
                    'model':     model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_loss':  best_val,
                    'config':    MODEL_CONFIG,
                }, f'{args.save_dir}/best_model_cwdm_inpaint.pt')
                print(f'         VAL | val={val_avg:.4f} | best={best_val:.4f}  *** NEW BEST ***', flush=True)
            else:
                print(f'         VAL | val={val_avg:.4f} | best={best_val:.4f}', flush=True)

        if iteration % args.save_every == 0:
            torch.save({
                'iteration': iteration,
                'model':     model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'history':   history,
                'config':    MODEL_CONFIG,
            }, f'{args.save_dir}/checkpoint_cwdm_inpaint_iter{iteration:07d}.pt')
            print(f'  Checkpoint saved: iter {iteration}', flush=True)

    print(f'\nDone. best_val={best_val:.4f}')
    print(f'Best model: {args.save_dir}/best_model_cwdm_inpaint.pt')
