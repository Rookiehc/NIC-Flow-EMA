#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2023-03-11 17:17:41

import warnings
warnings.filterwarnings("ignore")

# Ensure local 'src' package is importable before site-packages (for bundled diffusers, etc.)
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.exists():
    # Prepend to sys.path so 'import diffusers' resolves to ./src/diffusers first
    sys.path.insert(0, str(_SRC))

import argparse
import numpy as np
from omegaconf import OmegaConf
from sampler_invsr import InvSamplerSR
from utils.fid_utils import FIDOnline

from utils import util_common
from utils.util_opts import str2bool
from basicsr.utils.download_util import load_file_from_url
import torch

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-i", "--in_path", type=str, default="", help="Input path")
    parser.add_argument("-o", "--out_path", type=str, default="", help="Output path")
    parser.add_argument("--bs", type=int, default=1, help="Batchsize for loading image")
    parser.add_argument("--chopping_bs", type=int, default=8, help="Batchsize for chopped patch")
    parser.add_argument("-t", "--timesteps", type=int, nargs="+", help="The inversed timesteps")
    parser.add_argument("-n", "--num_steps", type=int, default=1, help="Number of inference steps")
    parser.add_argument(
        "--cfg_path", type=str, default="./configs/sample-sd-turbo.yaml", help="Configuration path.",
    )
    parser.add_argument(
        "--sd_path", type=str, default="", help="Path for Stable Diffusion Model",
    )
    parser.add_argument(
        "--started_ckpt_path", type=str, default="", help="Checkpoint path for noise predictor"
    )
    parser.add_argument(
        "--imm_ckpt_path", type=str, default="", help="IMM训练得到的UNet/NoisePredictor权重（.pth）路径"
    )
    parser.add_argument(
        "--ref_dir", type=str, default="", help="(optional) directory with ground-truth images for reference metrics"
    )
    parser.add_argument(
        "--tiled_vae", type=str2bool, default='true', help="Enabled tiled VAE.",
    )
    parser.add_argument(
        "--color_fix", type=str, default='', choices=['wavelet', 'ycbcr'], help="Fix the color shift",
    )
    parser.add_argument(
        "--chopping_size", type=int, default=128, help="Chopping size when dealing large images"
    )
    parser.add_argument(
        "--fid_ref", type=str, default="", help="Optional path to ref_stats_fid.npz for FID computation",
    )
    parser.add_argument(
        "--strict_metrics", type=str2bool, default='false', help="启用严格评估(SSIM/NIQE/PI/CLIP-IQA/MUSIQ)"
    )
    args = parser.parse_args()

    return args

def get_configs(args):
    configs = OmegaConf.load(args.cfg_path)

    if args.timesteps is not None:
        assert len(args.timesteps) == args.num_steps
        configs.timesteps = sorted(args.timesteps, reverse=True)
    else:
        if args.num_steps == 1:
            configs.timesteps = [200,]
        elif args.num_steps == 2:
            configs.timesteps = [200, 100]
        elif args.num_steps == 3:
            configs.timesteps = [200, 100, 50]
        elif args.num_steps == 4:
            configs.timesteps = [200, 150, 100, 50]
        elif args.num_steps == 5:
            configs.timesteps = [250, 200, 150, 100, 50]
        else:
            assert args.num_steps <= 250
            configs.timesteps = np.linspace(
                start=args.started_step, stop=0, num=args.num_steps, endpoint=False, dtype=np.int64()
            ).tolist()
    print(f'Setting timesteps for inference: {configs.timesteps}')

    # path to save Stable Diffusion
    sd_path = args.sd_path if args.sd_path else "./weights"
    util_common.mkdir(sd_path, delete=False, parents=True)
    configs.sd_pipe.params.cache_dir = sd_path

    # Ensure start noise predictor block exists; create a minimal default if missing
    if getattr(configs, 'model_start', None) is None:
        # Minimal block compatible with sampler_invsr.py expectations
        configs.model_start = OmegaConf.create({
            'target': 'diffusers.models.autoencoders.NoisePredictor',
            'ckpt_path': '',
            'params': {
                'in_channels': 3,
                'down_block_types': ['DownBlock2D', 'AttnDownBlock2D'],
                'up_block_types': ['AttnUpBlock2D', 'UpBlock2D'],
                'block_out_channels': [256, 512],
                'layers_per_block': [3, 3],
                'act_fn': 'silu',
                'latent_channels': 4,
                'norm_num_groups': 32,
                'sample_size': 64,
                'mid_block_add_attention': True,
                'resnet_time_scale_shift': 'default',
                'temb_channels': 512,
                'attention_head_dim': 64,
                'freq_shift': 0,
                'flip_sin_to_cos': True,
                'double_z': True,
            },
        })

    # path to save noise predictor
    if args.started_ckpt_path:
        started_ckpt_path = args.started_ckpt_path
    else:
        started_ckpt_name = "noise_predictor_sd_turbo_v5.pth"
        started_ckpt_dir = "./weights"
        util_common.mkdir(started_ckpt_dir, delete=False, parents=True)
        started_ckpt_path = Path(started_ckpt_dir) / started_ckpt_name
        if not started_ckpt_path.exists():
            load_file_from_url(
                url="https://huggingface.co/OAOA/InvSR/resolve/main/noise_predictor_sd_turbo_v5.pth",
                model_dir=started_ckpt_dir,
                progress=True,
                file_name=started_ckpt_name,
            )
    configs.model_start.ckpt_path = str(started_ckpt_path)

    configs.bs = args.bs
    configs.tiled_vae = args.tiled_vae
    configs.color_fix = args.color_fix
    configs.basesr.chopping.pch_size = args.chopping_size
    if args.bs > 1:
        configs.basesr.chopping.extra_bs = 1
    else:
        configs.basesr.chopping.extra_bs = args.chopping_bs

    # FID reference stats path (optional). If provided, InvSamplerSR can use
    # it to compute FID over generated images.
    if args.fid_ref:
        configs.fid_ref_stats = args.fid_ref

    return configs

def main():
    args = get_parser()

    configs = get_configs(args)
    sampler = InvSamplerSR(configs)

    # 加载 IMM 训练生成的 UNet 与噪声预测器权重（trainer_imm 保存格式：{'unet': ..., 'noise_predictor': ...}）
    if args.imm_ckpt_path:
        try:
            ckpt = torch.load(args.imm_ckpt_path, map_location="cpu")
            state = ckpt.get('unet', ckpt)
            sampler.unet.load_state_dict(state, strict=False)
            print(f"[INFO] Loaded IMM UNet weights from {args.imm_ckpt_path}")
            np_state = ckpt.get('noise_predictor', None)
            # model_start / noise_predictor 可能位于 sampler 或 sd_pipe 中
            noise_pred = None
            if hasattr(sampler, 'model_start'):
                noise_pred = getattr(sampler, 'model_start')
            elif hasattr(sampler, 'noise_predictor'):
                noise_pred = getattr(sampler, 'noise_predictor')
            if noise_pred is None and hasattr(sampler, 'sd_pipe'):
                noise_pred = getattr(sampler.sd_pipe, 'noise_predictor', None)
            if np_state is not None and noise_pred is not None:
                noise_pred.load_state_dict(np_state, strict=False)
                print(f"[INFO] Loaded NoisePredictor weights from {args.imm_ckpt_path}")
        except Exception as e:
            print(f"[WARN] Failed to load IMM UNet weights: {e}")

    # If FID reference stats are provided, create an online FID accumulator
    # and pass it through configs so that sampler_invsr can optionally use it
    # in its inference loop.
    if getattr(configs, "fid_ref_stats", None):
        try:
            fid_acc = FIDOnline(configs.fid_ref_stats, device="cuda" if torch.cuda.is_available() else "cpu")
            # Do NOT store non-primitive objects inside OmegaConf (configs).
            # Attach the FID accumulator to the runtime sampler instance instead.
            sampler.fid_online = fid_acc
            print(f"[INFO] FIDOnline initialized and attached to sampler")
        except Exception as e:
            print(f"[WARN] Failed to initialize FIDOnline: {e}")

    # 运行推理
    sampler.inference(args.in_path, out_path=args.out_path, bs=args.bs, ref_dir=args.ref_dir)

    # 推理结束后，如提供 ref_dir，则优先使用 scripts.eval_metrics.evaluate_directory（更严格、依赖处理更稳健）
    if args.ref_dir:
        try:
            from scripts.eval_metrics import evaluate_directory
            gen_dir = args.out_path
            ref_dir = args.ref_dir
            try:
                # Also print per-image metrics before strict summary by doing a light inline pass
                import os, math
                import torchvision
                import torch.nn.functional as F
                from PIL import Image
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                names = sorted([n for n in os.listdir(gen_dir) if n.lower().endswith(('.png','.jpg','.jpeg'))])
                # Prepare optional modules
                lpips_fn = None; _niqe_fn = None; piq_modules = {}
                try:
                    import piq
                    lpips_fn = lambda a, b: piq.LPIPS(reduction='none').to(device)(a, b)
                    try:
                        piq_modules['clipiqa'] = piq.CLIP_IQA().to(device).eval()
                    except Exception:
                        pass
                    try:
                        piq_modules['musiq'] = piq.MUSIQ().to(device).eval()
                    except Exception:
                        pass
                except Exception:
                    lpips_fn = None
                try:
                    from skimage.metrics import niqe as _niqe
                    def _niqe_score(x):
                        x_np = (x.clamp(0,1).permute(1,2,0).cpu().numpy())
                        import numpy as _np
                        g = 0.299*x_np[...,0] + 0.587*x_np[...,1] + 0.114*x_np[...,2]
                        return float(_niqe(g))
                    _niqe_fn = _niqe_score
                except Exception:
                    _niqe_fn = None
                def _to_tensor(p: str) -> torch.Tensor:
                    im = Image.open(p).convert('RGB')
                    return torchvision.transforms.ToTensor()(im)
                # accumulators
                psnr_l, ssim_l, niqe_l, clip_l, musiq_l = [], [], [], [], []
                for n in names:
                    gp = os.path.join(gen_dir, n)
                    rp = os.path.join(ref_dir, n)
                    if not os.path.exists(rp):
                        continue
                    g = _to_tensor(gp).to(device)
                    r = _to_tensor(rp).to(device)
                    if g.shape != r.shape:
                        h = min(g.shape[1], r.shape[1]); w = min(g.shape[2], r.shape[2])
                        g = g[:, :h, :w]; r = r[:, :h, :w]
                    mse = F.mse_loss(g, r).item()
                    psnr = (20.0 * math.log10(1.0) - 10.0 * math.log10(mse)) if mse > 0 else float('inf')
                    # simple ssim
                    def _ssim_simple(x, y, C1=0.01**2, C2=0.03**2):
                        mu_x = x.mean(); mu_y = y.mean()
                        sigma_x = x.var(); sigma_y = y.var()
                        sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
                        num = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2)
                        den = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
                        return float((num / den).item())
                    ssim = _ssim_simple(g, r)
                    # no-reference metrics
                    niqe = float('nan'); clip = float('nan'); musiq = float('nan')
                    if _niqe_fn is not None:
                        try: niqe = _niqe_fn(g)
                        except Exception: pass
                    if 'clipiqa' in piq_modules:
                        try: clip = float(piq_modules['clipiqa'](g.unsqueeze(0)).item())
                        except Exception: pass
                    if 'musiq' in piq_modules:
                        try: musiq = float(piq_modules['musiq'](g.unsqueeze(0)).item())
                        except Exception: pass
                    print(f"{n}: PSNR={psnr:.4f} SSIM={ssim:.4f} NIQE={niqe if isinstance(niqe,float) else niqe:.4f} CLIPIQA={clip if isinstance(clip,float) else clip:.4f} MUSIQ={musiq if isinstance(musiq,float) else musiq:.4f}")
                    psnr_l.append(psnr); ssim_l.append(ssim); niqe_l.append(niqe); clip_l.append(clip); musiq_l.append(musiq)
                def _mean_safe(l):
                    import math as _m
                    l2 = [x for x in l if isinstance(x, (int,float)) and not (_m.isnan(x) or _m.isinf(x))]
                    return sum(l2)/max(1, len(l2)) if l2 else float('nan')
                print(f"Means (quick): PSNR={_mean_safe(psnr_l):.4f} SSIM={_mean_safe(ssim_l):.4f} NIQE={_mean_safe(niqe_l):.4f} CLIPIQA={_mean_safe(clip_l):.4f} MUSIQ={_mean_safe(musiq_l):.4f}")
                # then strict summary
                summary = evaluate_directory(gen_dir, ref_dir)
                print(summary)
            except Exception as e:
                print(f"[WARN] evaluate_directory failed: {e}. Falling back to inline summarizer.")
                raise e
        except Exception:
            # Fallback: existing inline summarization (keeps current behavior if eval_metrics not available)
            try:
                import os
                import math
                import torchvision
                import torch.nn.functional as F
                from PIL import Image

                def _load_img(p):
                    im = Image.open(p).convert('RGB')
                    t = torchvision.transforms.ToTensor()(im)
                    return t

                # 收集生成与GT文件对
                gen_dir = args.out_path
                ref_dir = args.ref_dir
                names = sorted([n for n in os.listdir(gen_dir) if n.lower().endswith(('.png','.jpg','.jpeg'))])
                psnr_l = []
                ssim_l = []
                lpips_l = []
                niqe_l = []
                pi_l = []
                clip_l = []
                musiq_l = []

                # 准备可用的逐图像参考/非参考指标函数（优先使用 piq / scikit-image）
                lpips_fn = None
                piq_modules = {}
                _niqe_fn = None
                try:
                    import piq
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    # LPIPS via piq
                    lpips_fn = lambda a, b: piq.LPIPS(reduction='none').to(device)(a, b)
                    # PI / CLIP-IQA / MUSIQ
                    try:
                        piq_modules['pi'] = piq.PerceptualIndex().to(device).eval()
                    except Exception:
                        pass
                    try:
                        piq_modules['clipiqa'] = piq.CLIP_IQA().to(device).eval()
                    except Exception:
                        pass
                    try:
                        piq_modules['musiq'] = piq.MUSIQ().to(device).eval()
                    except Exception:
                        pass
                except Exception:
                    # Fallback: try latent_lpips for LPIPS (image mode)
                    try:
                        from latent_lpips.lpips import LPIPS as _LPIPS
                        lp_model = _LPIPS(pretrained=False, net='vgg', lpips=True, spatial=False,
                                          pnet_rand=False, pnet_tune=True, use_dropout=True, eval_mode=True,
                                          latent=False, in_chans=3, verbose=False).to('cuda' if torch.cuda.is_available() else 'cpu').eval()
                        def _to_lpips(a, b):
                            def to_m1_p1(x):
                                return x.clamp(0,1) * 2.0 - 1.0
                            return lp_model(to_m1_p1(a), to_m1_p1(b))
                        lpips_fn = _to_lpips
                    except Exception:
                        lpips_fn = None

                # NIQE via scikit-image if available
                try:
                    from skimage.metrics import niqe as _niqe
                    def _niqe_score(x: torch.Tensor) -> float:
                        x_np = (x.clamp(0,1).permute(1,2,0).cpu().numpy())
                        import numpy as _np
                        g = 0.299*x_np[...,0] + 0.587*x_np[...,1] + 0.114*x_np[...,2]
                        return float(_niqe(g))
                    _niqe_fn = _niqe_score
                except Exception:
                    _niqe_fn = None

                # 简化版 SSIM（窗口法较复杂，这里给出一个快速近似，可替换为更严格实现）
                def _ssim_simple(x, y, C1=0.01**2, C2=0.03**2):
                    mu_x = x.mean()
                    mu_y = y.mean()
                    sigma_x = x.var()
                    sigma_y = y.var()
                    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
                    num = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2)
                    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
                    return float((num / den).item())

                device = 'cuda' if torch.cuda.is_available() else 'cpu'

                for n in names:
                    gp = os.path.join(gen_dir, n)
                    rp = os.path.join(ref_dir, n)
                    if not os.path.exists(rp):
                        continue
                    g = _load_img(gp).to(device)
                    r = _load_img(rp).to(device)
                    # 对齐尺寸
                    if g.shape != r.shape:
                        h = min(g.shape[1], r.shape[1]); w = min(g.shape[2], r.shape[2])
                        g = g[:, :h, :w]; r = r[:, :h, :w]
                    # PSNR
                    mse = F.mse_loss(g, r).item()
                    psnr = (20.0 * math.log10(1.0) - 10.0 * math.log10(mse)) if mse > 0 else float('inf')
                    psnr_l.append(psnr)
                    # SSIM（简化）
                    ssim_l.append(_ssim_simple(g, r))
                    # LPIPS (use lpips_fn if available)
                    if lpips_fn is not None:
                        try:
                            # lpips_fn expects batch tensors [B,C,H,W]
                            lp_in_g = g.unsqueeze(0)
                            lp_in_r = r.unsqueeze(0)
                            lpv = float(lpips_fn(lp_in_g, lp_in_r).mean().item())
                            lpips_l.append(lpv)
                        except Exception:
                            lpips_l.append(float('nan'))
                    else:
                        lpips_l.append(float('nan'))

                    # NIQE (no-reference) if available
                    if _niqe_fn is not None:
                        try:
                            niqe_l.append(_niqe_fn(g))
                        except Exception:
                            niqe_l.append(float('nan'))
                    else:
                        niqe_l.append(float('nan'))

                    # PI/CLIP-IQA/MUSIQ via piq (if available)
                    if piq_modules:
                        try:
                            if 'pi' in piq_modules:
                                pi_l.append(float(piq_modules['pi'](g.unsqueeze(0), r.unsqueeze(0)).item()))
                            else:
                                pi_l.append(float('nan'))
                        except Exception:
                            pi_l.append(float('nan'))
                        try:
                            if 'clipiqa' in piq_modules:
                                clip_l.append(float(piq_modules['clipiqa'](g.unsqueeze(0)).item()))
                            else:
                                clip_l.append(float('nan'))
                        except Exception:
                            clip_l.append(float('nan'))
                        try:
                            if 'musiq' in piq_modules:
                                musiq_l.append(float(piq_modules['musiq'](g.unsqueeze(0)).item()))
                            else:
                                musiq_l.append(float('nan'))
                        except Exception:
                            musiq_l.append(float('nan'))
                    else:
                        pi_l.append(float('nan'))
                        clip_l.append(float('nan'))
                        musiq_l.append(float('nan'))

                def _mean_safe(l):
                    l2 = [x for x in l if not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))]
                    return sum(l2)/max(1, len(l2)) if l2 else float('nan')

                summary = (
                    f"Overall means: PSNR={_mean_safe(psnr_l):.4f} SSIM={_mean_safe(ssim_l):.4f} "
                    f"LPIPS={_mean_safe(lpips_l):.4f} NIQE={_mean_safe(niqe_l):.4f} PI={_mean_safe(pi_l):.4f} "
                    f"CLIPIQA={_mean_safe(clip_l):.4f} MUSIQ={_mean_safe(musiq_l):.4f}"
                )
                print(summary)
            except Exception as e:
                print(f"[WARN] Failed to summarize metrics: {e}")

if __name__ == '__main__':
    main()
