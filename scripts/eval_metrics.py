#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Strict evaluation of PSNR/SSIM/LPIPS/NIQE/PI/CLIP-IQA/MUSIQ over a directory of generated images
against a reference directory with matching filenames.

Dependencies (optional but recommended):
- scikit-image (for SSIM/NIQE)
- piq (for LPIPS, PI, CLIP-IQA, MUSIQ) -- if unavailable, will skip

Usage (programmatic):
from scripts.eval_metrics import evaluate_directory
print(evaluate_directory(gen_dir, ref_dir))
"""
import os
import math
from typing import Dict

import torch
import torchvision
import torch.nn.functional as F
from PIL import Image


def _to_tensor(p: str) -> torch.Tensor:
    im = Image.open(p).convert('RGB')
    return torchvision.transforms.ToTensor()(im)


def _mean_safe(lst):
    lst = [float(x) for x in lst]
    lst = [x for x in lst if not (math.isnan(x) or math.isinf(x))]
    return sum(lst) / max(1, len(lst)) if lst else float('nan')


def evaluate_directory(gen_dir: str, ref_dir: str) -> str:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    names = sorted([n for n in os.listdir(gen_dir) if n.lower().endswith(('.png','.jpg','.jpeg'))])

    psnr_l, ssim_l, lpips_l, niqe_l, pi_l, clip_l, musiq_l = [], [], [], [], [], [], []

    # Strict SSIM (scikit-image)
    try:
        from skimage.metrics import structural_similarity
        def _ssim_strict(x: torch.Tensor, y: torch.Tensor) -> float:
            # x,y: 3xHxW [0,1]
            x_np = (x.clamp(0,1).permute(1,2,0).cpu().numpy())
            y_np = (y.clamp(0,1).permute(1,2,0).cpu().numpy())
            s = structural_similarity(x_np, y_np, channel_axis=2, win_size=11, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
            return float(s)
    except Exception:
        def _ssim_strict(x, y):
            return float('nan')

    # LPIPS (piq or latent_lpips fallback)
    lpips_fn = None
    try:
        import piq
        lpips_fn = lambda a,b: piq.LPIPS(reduction='none').to(device)(a, b)
    except Exception:
        try:
            from latent_lpips.lpips import LPIPS as _LPIPS
            lp_model = _LPIPS(pretrained=False, net='vgg', lpips=True, spatial=False,
                              pnet_rand=False, pnet_tune=True, use_dropout=True, eval_mode=True,
                              latent=False, in_chans=3, verbose=False).to(device).eval()
            def to_m1_p1(x):
                return x.clamp(0,1) * 2.0 - 1.0
            lpips_fn = lambda a,b: lp_model(to_m1_p1(a), to_m1_p1(b))
        except Exception:
            lpips_fn = None

    # NIQE (scikit-image)
    try:
        from skimage.metrics import niqe as _niqe
        def _niqe_score(x: torch.Tensor) -> float:
            x_np = (x.clamp(0,1).permute(1,2,0).cpu().numpy())
            # skimage expects grayscale; convert by luminance
            import numpy as np
            g = 0.299*x_np[...,0] + 0.587*x_np[...,1] + 0.114*x_np[...,2]
            return float(_niqe(g))
    except Exception:
        def _niqe_score(x):
            return float('nan')

    # PI, CLIP-IQA, MUSIQ via piq (if available)
    piq_modules = {}
    try:
        import piq
        piq_modules['pi'] = piq.PerceptualIndex()
        piq_modules['clipiqa'] = piq.CLIP_IQA()
        piq_modules['musiq'] = piq.MUSIQ()
        for k,v in piq_modules.items():
            piq_modules[k] = v.to(device).eval()
    except Exception:
        piq_modules = {}

    for n in names:
        gp = os.path.join(gen_dir, n)
        rp = os.path.join(ref_dir, n)
        if not os.path.exists(rp):
            continue
        g = _to_tensor(gp).to(device)
        r = _to_tensor(rp).to(device)
        # align size
        if g.shape != r.shape:
            h = min(g.shape[1], r.shape[1]); w = min(g.shape[2], r.shape[2])
            g = g[:, :h, :w]; r = r[:, :h, :w]
        # PSNR
        mse = F.mse_loss(g, r).item()
        psnr = (20.0 * math.log10(1.0) - 10.0 * math.log10(mse)) if mse > 0 else float('inf')
        psnr_l.append(psnr)
        # SSIM strict
        ssim_l.append(_ssim_strict(g, r))
        # LPIPS
        if lpips_fn is not None:
            lpv = float(lpips_fn(g.unsqueeze(0), r.unsqueeze(0)).mean().item())
            lpips_l.append(lpv)
        # NIQE
        niqe_l.append(_niqe_score(g))
        # PI/CLIP-IQA/MUSIQ
        if 'pi' in piq_modules:
            try:
                pi_l.append(float(piq_modules['pi'](g.unsqueeze(0), r.unsqueeze(0)).item()))
            except Exception:
                pi_l.append(float('nan'))
        else:
            pi_l.append(float('nan'))
        if 'clipiqa' in piq_modules:
            try:
                clip_l.append(float(piq_modules['clipiqa'](g.unsqueeze(0)).item()))
            except Exception:
                clip_l.append(float('nan'))
        else:
            clip_l.append(float('nan'))
        if 'musiq' in piq_modules:
            try:
                musiq_l.append(float(piq_modules['musiq'](g.unsqueeze(0)).item()))
            except Exception:
                musiq_l.append(float('nan'))
        else:
            musiq_l.append(float('nan'))

    summary = (
        f"Overall means (strict): PSNR={_mean_safe(psnr_l):.4f} SSIM={_mean_safe(ssim_l):.4f} "
        f"LPIPS={_mean_safe(lpips_l):.4f} NIQE={_mean_safe(niqe_l):.4f} PI={_mean_safe(pi_l):.4f} "
        f"CLIPIQA={_mean_safe(clip_l):.4f} MUSIQ={_mean_safe(musiq_l):.4f}"
    )
    return summary
