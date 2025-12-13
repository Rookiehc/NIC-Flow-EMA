#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-07-13 16:59:27

import os, sys, math, random

import cv2
import numpy as np
from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf

from utils import util_net
from utils import util_image
from utils import util_common
from utils import util_color_fix

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from datapipe.datasets import create_dataset
from diffusers import StableDiffusionInvEnhancePipeline, AutoencoderKL
try:
    # Prefer PyTorch 2.0 scaled-dot product attention to reduce memory
    from diffusers.models.attention_processor import AttnProcessor2_0 as _SDPAProc
except Exception:
    _SDPAProc = None
import time
try:
    import pyiqa
except Exception:
    pyiqa = None

_positive= 'Cinematic, high-contrast, photo-realistic, 8k, ultra HD, ' +\
           'meticulous detailing, hyper sharpness, perfect without deformations'
_negative= 'Low quality, blurring, jpeg artifacts, deformed, over-smooth, cartoon, noisy,' +\
           'painting, drawing, sketch, oil painting'

class BaseSampler:
    def __init__(self, configs):
        '''
        Input:
            configs: config, see the yaml file in folder ./configs/
                configs.sampler_config.{start_timesteps, padding_mod, seed, sf, num_sample_steps}
            seed: int, random seed
        '''
        self.configs = configs

        self.setup_seed()

        self.build_model()

    def setup_seed(self, seed=None):
        seed = self.configs.seed if seed is None else seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def write_log(self, log_str):
        print(log_str, flush=True)

    def build_model(self):
        # Build Stable diffusion
        params = dict(self.configs.sd_pipe.params)
        torch_dtype = params.pop('torch_dtype')
        params['torch_dtype'] = get_torch_dtype(torch_dtype)
        base_pipe = util_common.get_obj_from_str(self.configs.sd_pipe.target).from_pretrained(**params)
        if self.configs.get('scheduler', None) is not None:
            pipe_id = self.configs.scheduler.target.split('.')[-1]
            self.write_log(f'Loading scheduler of {pipe_id}...')
            base_pipe.scheduler = util_common.get_obj_from_str(self.configs.scheduler.target).from_config(
                base_pipe.scheduler.config
            )
            self.write_log('Loaded Done')
        if self.configs.get('vae_fp16', None) is not None:
            params_vae = dict(self.configs.vae_fp16.params)
            torch_dtype = params_vae.pop('torch_dtype')
            params_vae['torch_dtype'] = get_torch_dtype(torch_dtype)
            pipe_id = self.configs.vae_fp16.params.pretrained_model_name_or_path
            self.write_log(f'Loading improved vae from {pipe_id}...')
            # If the specified VAE path is not available locally and network
            # access is unreliable, attempting to download will raise an
            # exception and abort. Prefer a safe fallback: only call
            # from_pretrained when a local model dir with config.json exists,
            # otherwise skip and continue with the pipeline's default VAE.
            try:
                local_ok = False
                try:
                    # If pipe_id is a path to a directory containing config.json
                    if isinstance(pipe_id, str) and os.path.isdir(pipe_id):
                        if os.path.exists(os.path.join(pipe_id, 'config.json')):
                            local_ok = True
                except Exception:
                    local_ok = False

                if not local_ok:
                    self.write_log(
                        f"Improved VAE path '{pipe_id}' not found locally or missing config.json. "
                        "Skipping improved VAE load to avoid attempting a remote download."
                    )
                else:
                    base_pipe.vae = util_common.get_obj_from_str(self.configs.vae_fp16.target).from_pretrained(
                        **params_vae,
                    )
                    self.write_log('Loaded Done')
            except Exception as e:
                # Log and continue with default VAE rather than failing hard.
                self.write_log(f'Failed to load improved VAE from {pipe_id}: {e}. Continuing with default VAE.')
        if self.configs.base_model in ['sd-turbo', 'sd2base'] :
            sd_pipe = StableDiffusionInvEnhancePipeline.from_pipe(base_pipe)
        else:
            raise ValueError(f"Unsupported base model: {self.configs.base_model}!")
        # Move pipeline to the correct local device for DDP to avoid piling all ranks onto cuda:0
        try:
            local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        except Exception:
            local_rank = 0
        sd_pipe.to(f"cuda:{local_rank}")
        # Memory optimizations: attention and checkpointing
        try:
            # Enable gradient checkpointing on UNet to reduce activation memory
            enable_gc = bool(self.configs.sd_pipe.get('enable_grad_checkpoint', False)) or bool(self.configs.get('enable_unet_gradient_checkpointing', False))
        except Exception:
            enable_gc = False
        if enable_gc and hasattr(sd_pipe, 'unet'):
            try:
                sd_pipe.unet.enable_gradient_checkpointing()
                self.write_log('Enabled UNet gradient checkpointing')
            except Exception:
                pass
        # Prefer SDPA attention when available (PyTorch>=2.0), else keep default
        if _SDPAProc is not None and hasattr(sd_pipe, 'unet'):
            try:
                sd_pipe.unet.set_attn_processor(_SDPAProc())
                self.write_log('Using SDPA attention processor on UNet')
            except Exception:
                pass
        # Optional: enable xformers memory-efficient attention if installed and requested
        try:
            use_xformers = bool(self.configs.get('use_xformers', False))
        except Exception:
            use_xformers = False
        if use_xformers and hasattr(sd_pipe.unet, 'enable_xformers_memory_efficient_attention'):
            try:
                sd_pipe.unet.enable_xformers_memory_efficient_attention()
                self.write_log('Enabled xformers memory-efficient attention')
            except Exception:
                pass
        # Attention slicing on pipeline (reduces memory at small perf cost)
        if bool(self.configs.get('enable_attention_slicing', False)) and hasattr(sd_pipe, 'enable_attention_slicing'):
            try:
                sd_pipe.enable_attention_slicing("auto")
                self.write_log('Enabled attention slicing (pipeline)')
            except Exception:
                pass
        # Ensure expected attributes for downstream IMM trainer
        # Some SD variants use 'transformer' instead of 'unet'. Provide a compatibility alias.
        if not hasattr(sd_pipe, 'unet') and hasattr(sd_pipe, 'transformer'):
            setattr(sd_pipe, 'unet', sd_pipe.transformer)
        # Ensure scheduler exists (should be carried from base_pipe)
        if not hasattr(sd_pipe, 'scheduler') and hasattr(base_pipe, 'scheduler'):
            sd_pipe.scheduler = base_pipe.scheduler
        # VAE memory optimizations: slicing/tiling
        # Support flags under both top-level and sd_pipe config blocks.
        try:
            sliced_vae_flag = bool(self.configs.get('sliced_vae', False)) or bool(self.configs.sd_pipe.get('vae_split', 0))
        except Exception:
            sliced_vae_flag = bool(self.configs.get('sliced_vae', False))
        if sliced_vae_flag:
            try:
                sd_pipe.vae.enable_slicing()
                self.write_log('Enabled VAE slicing')
            except Exception:
                pass
        if bool(self.configs.get('tiled_vae', False)):
            sd_pipe.vae.enable_tiling()
            # set optional tile sizes if present
            try:
                if 'latent_tiled_size' in self.configs:
                    sd_pipe.vae.tile_latent_min_size = int(self.configs.latent_tiled_size)
                if 'sample_tiled_size' in self.configs:
                    sd_pipe.vae.tile_sample_min_size = int(self.configs.sample_tiled_size)
            except Exception:
                pass
        if bool(self.configs.get('gradient_checkpointing_vae', False)):
            self.write_log(f"Activating gradient checkpoing for vae...")
            sd_pipe.vae._set_gradient_checkpointing(sd_pipe.vae.encoder, True)
            sd_pipe.vae._set_gradient_checkpointing(sd_pipe.vae.decoder, True)

        # Support both legacy 'model_start' (InvSR original) and new 'model' (IMM config)
        model_configs = self.configs.get('model_start', None)
        if model_configs is None:
            model_configs = self.configs.get('model', None)
            if model_configs is None:
                raise ValueError("Missing model_start/model in configs for start noise predictor.")
        params = model_configs.get('params', dict)
        model_start = util_common.get_obj_from_str(model_configs.target)(**params)
        model_start.to(f"cuda:{local_rank}")
        ckpt_path = model_configs.get('ckpt_path')
        if ckpt_path:
            self.write_log(f"Loading started model from {ckpt_path}...")
            state = torch.load(ckpt_path, map_location=f"cuda:{local_rank}")
            if 'state_dict' in state:
                state = state['state_dict']
            # Try to load state_dict robustly. util_net.reload_model expects an exact
            # mapping of parameter names -> tensors; checkpoints from different
            # training wrappers may have 'module.' prefixes or compiled names. We try a few fallbacks.
            try:
                util_net.reload_model(model_start, state)
                self.write_log("Loaded checkpoint with exact key match.")
            except AssertionError:
                # 1) try removing 'module.' from checkpoint keys
                def _strip_module(d):
                    new = {}
                    for k, v in d.items():
                        nk = k[len('module.'):] if k.startswith('module.') else k
                        new[nk] = v
                    return new

                # 2) try adding 'module.' prefix to keys
                def _add_module(d):
                    new = {}
                    for k, v in d.items():
                        nk = 'module.' + k if not k.startswith('module.') else k
                        new[nk] = v
                    return new

                # 3) try suffix-based matching (match ckpt keys that endwith model key)
                def _suffix_match(model_sd, ckpt_sd):
                    mapped = {}
                    ck_keys = list(ckpt_sd.keys())
                    for mkey in model_sd.keys():
                        if mkey in ckpt_sd:
                            mapped[mkey] = ckpt_sd[mkey]
                            continue
                        matches = [k for k in ck_keys if k.endswith(mkey)]
                        if len(matches) == 1:
                            mapped[mkey] = ckpt_sd[matches[0]]
                        else:
                            return None
                    return mapped

                loaded = False
                # try strip module
                try_state = _strip_module(state)
                try:
                    util_net.reload_model(model_start, try_state)
                    loaded = True
                    self.write_log("Loaded checkpoint after stripping 'module.' prefix from keys.")
                except AssertionError:
                    pass

                if not loaded:
                    try_state = _add_module(state)
                    try:
                        util_net.reload_model(model_start, try_state)
                        loaded = True
                        self.write_log("Loaded checkpoint after adding 'module.' prefix to keys.")
                    except AssertionError:
                        pass

                if not loaded:
                    mapped = _suffix_match(model_start.state_dict(), state)
                    if mapped is not None:
                        try:
                            util_net.reload_model(model_start, mapped)
                            loaded = True
                            self.write_log("Loaded checkpoint using suffix-based key matching.")
                        except AssertionError:
                            loaded = False

                if not loaded:
                    ck_keys_preview = list(state.keys())[:10]
                    self.write_log(
                        "Failed to match checkpoint keys to model. Example checkpoint keys: "
                        f"{ck_keys_preview}...\n"
                        f"Continuing with freshly initialized start model."
                    )
        else:
            self.write_log("No start checkpoint provided. Using freshly initialized start model.")
        self.write_log(f"Loading Done")
        model_start.eval()
        setattr(sd_pipe, 'start_noise_predictor', model_start)

        self.sd_pipe = sd_pipe
        # Expose core components directly for downstream trainers expecting attributes
        self.unet = getattr(sd_pipe, 'unet', None)
        if self.unet is None and hasattr(sd_pipe, 'transformer'):
            self.unet = sd_pipe.transformer
        self.scheduler = getattr(sd_pipe, 'scheduler', None)
        self.vae = getattr(sd_pipe, 'vae', None)

class InvSamplerSR(BaseSampler):
    @torch.no_grad()
    def sample_func(self, im_cond):
        '''
        Input:
            im_cond: b x c x h x w, torch tensor, [0,1], RGB
        Output:
            xt: h x w x c, numpy array, [0,1], RGB
        '''
        if self.configs.cfg_scale > 1.0:
            negative_prompt = [_negative,]*im_cond.shape[0]
        else:
            negative_prompt = None

        ori_h_lq, ori_w_lq = im_cond.shape[-2:]
        ori_w_hq = ori_w_lq * self.configs.basesr.sf
        ori_h_hq = ori_h_lq * self.configs.basesr.sf
        vae_sf = (2 ** (len(self.sd_pipe.vae.config.block_out_channels) - 1))
        if hasattr(self.sd_pipe, 'unet'):
            diffusion_sf = (2 ** (len(self.sd_pipe.unet.config.block_out_channels) - 1))
        else:
            diffusion_sf = self.sd_pipe.transformer.patch_size
        mod_lq = vae_sf // self.configs.basesr.sf * diffusion_sf
        idle_pch_size = self.configs.basesr.chopping.pch_size

        total_pad_h_up = total_pad_w_left = 0
        if min(im_cond.shape[-2:]) < idle_pch_size:
            while min(im_cond.shape[-2:]) < idle_pch_size:
                pad_h_up = max(min((idle_pch_size - im_cond.shape[-2]) // 2, im_cond.shape[-2]-1), 0)
                pad_h_down = max(min(idle_pch_size - im_cond.shape[-2] - pad_h_up, im_cond.shape[-2]-1), 0)
                pad_w_left = max(min((idle_pch_size - im_cond.shape[-1]) // 2, im_cond.shape[-1]-1), 0)
                pad_w_right = max(min(idle_pch_size - im_cond.shape[-1] - pad_w_left, im_cond.shape[-1]-1), 0)
                im_cond = F.pad(im_cond, pad=(pad_w_left, pad_w_right, pad_h_up, pad_h_down), mode='reflect')
                total_pad_h_up += pad_h_up
                total_pad_w_left += pad_w_left

        if im_cond.shape[-2] == idle_pch_size and im_cond.shape[-1] == idle_pch_size:
            target_size = (
                im_cond.shape[-2] * self.configs.basesr.sf,
                im_cond.shape[-1] * self.configs.basesr.sf
            )
            res_sr = self.sd_pipe(
                image=im_cond.type(torch.float16),
                prompt=[_positive, ]*im_cond.shape[0],
                negative_prompt=negative_prompt,
                target_size=target_size,
                timesteps=self.configs.timesteps,
                guidance_scale=self.configs.cfg_scale,
                output_type="pt",    # torch tensor, b x c x h x w, [0, 1]
            ).images
        else:
            if not (im_cond.shape[-2] % mod_lq == 0 and im_cond.shape[-1] % mod_lq == 0):
                target_h_lq = math.ceil(im_cond.shape[-2] / mod_lq) * mod_lq
                target_w_lq = math.ceil(im_cond.shape[-1] / mod_lq) * mod_lq
                pad_h = target_h_lq - im_cond.shape[-2]
                pad_w = target_w_lq - im_cond.shape[-1]
                im_cond= F.pad(im_cond, pad=(0, pad_w, 0, pad_h), mode='reflect')

            im_spliter = util_image.ImageSpliterTh(
                im_cond,
                pch_size=idle_pch_size,
                stride= int(idle_pch_size * 0.50),
                sf=self.configs.basesr.sf,
                weight_type=self.configs.basesr.chopping.weight_type,
                extra_bs=self.configs.basesr.chopping.extra_bs,
            )
            for im_lq_pch, index_infos in im_spliter:
                target_size = (
                    im_lq_pch.shape[-2] * self.configs.basesr.sf,
                    im_lq_pch.shape[-1] * self.configs.basesr.sf,
                )

                # start = torch.cuda.Event(enable_timing=True)
                # end = torch.cuda.Event(enable_timing=True)
                # start.record()

                res_sr_pch = self.sd_pipe(
                    image=im_lq_pch.type(torch.float16),
                    prompt=[_positive, ]*im_lq_pch.shape[0],
                    negative_prompt=negative_prompt,
                    target_size=target_size,
                    timesteps=self.configs.timesteps,
                    guidance_scale=self.configs.cfg_scale,
                    output_type="pt",    # torch tensor, b x c x h x w, [0, 1]
                ).images

                # end.record()
                # torch.cuda.synchronize()
                # print(f"Time: {start.elapsed_time(end):.6f}")

                im_spliter.update(res_sr_pch, index_infos)
            res_sr = im_spliter.gather()

        total_pad_h_up *= self.configs.basesr.sf
        total_pad_w_left *= self.configs.basesr.sf
        res_sr = res_sr[:, :, total_pad_h_up:ori_h_hq+total_pad_h_up, total_pad_w_left:ori_w_hq+total_pad_w_left]

        if self.configs.color_fix:
            im_cond_up = F.interpolate(
                im_cond, size=res_sr.shape[-2:], mode='bicubic', align_corners=False, antialias=True
            )
            if self.configs.color_fix == 'ycbcr':
                res_sr = util_color_fix.ycbcr_color_replace(res_sr, im_cond_up)
            elif self.configs.color_fix == 'wavelet':
                res_sr = util_color_fix.wavelet_reconstruction(res_sr, im_cond_up)
            else:
                raise ValueError(f"Unsupported color fixing type: {self.configs.color_fix}")

        res_sr = res_sr.clamp(0.0, 1.0).cpu().permute(0,2,3,1).float().numpy()

        return res_sr

    def inference(self, in_path, out_path, bs=1, ref_dir: str = ""):
        '''
        Inference demo with optional metric computation.
        Input:
            in_path: str, folder or image path for LQ image
            out_path: str, folder save the results
            bs: int, default bs=1, bs % num_gpus == 0
            ref_dir: optional path to ground-truth images (matching stems)
        '''

        in_path = Path(in_path) if not isinstance(in_path, Path) else in_path
        out_path = Path(out_path) if not isinstance(out_path, Path) else out_path

        if not out_path.exists():
            out_path.mkdir(parents=True)

        # Set local cache dir for IQA model weights so pyiqa/piq can find/download them
        try:
            cache_dir_env = os.environ.get('PYIQA_CACHE_DIR', None)
            if cache_dir_env is None:
                default_cache = Path(self.configs.get('weights_dir', 'weights')) / 'metrics'
                os.environ['PYIQA_CACHE_DIR'] = str(default_cache)
                os.environ.setdefault('HF_HOME', str(default_cache))
                os.environ.setdefault('TORCH_HOME', str(default_cache))
                default_cache.mkdir(parents=True, exist_ok=True)
                self.write_log(f"[IQA] PYIQA_CACHE_DIR not set. Using default: {default_cache}")
            else:
                Path(cache_dir_env).mkdir(parents=True, exist_ok=True)
                self.write_log(f"[IQA] Using PYIQA_CACHE_DIR={cache_dir_env}")
        except Exception as e:
            self.write_log(f"[IQA][WARN] Failed to set IQA cache dir: {e}")

        # prepare metrics with robust fallbacks (pyiqa -> piq -> custom)
        metrics_enabled = False
        # Primary: pyiqa
        if pyiqa is not None:
            metrics_enabled = True
            # reference metrics (when GT provided)
            try:
                psnr_metric = pyiqa.create_metric(
                    'psnr',
                    test_y_channel=self.configs.train.get('val_y_channel', True),
                    color_space='ycbcr',
                    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
                )
            except Exception:
                psnr_metric = None
            try:
                ssim_metric = pyiqa.create_metric(
                    'ssim',
                    test_y_channel=self.configs.train.get('val_y_channel', True),
                    color_space='ycbcr',
                    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
                )
            except Exception:
                ssim_metric = None
            try:
                lpips_metric = pyiqa.create_metric('lpips-vgg', device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
            except Exception:
                lpips_metric = None
            # non-reference metrics
            try:
                niqe_metric = pyiqa.create_metric('niqe')
            except Exception:
                niqe_metric = None
            try:
                pi_metric = pyiqa.create_metric('pi')
            except Exception:
                pi_metric = None
            try:
                clipiqa_metric = pyiqa.create_metric('clipiqa')
            except Exception:
                clipiqa_metric = None
            try:
                musiq_metric = pyiqa.create_metric('musiq')
            except Exception:
                musiq_metric = None
        else:
            psnr_metric = ssim_metric = lpips_metric = niqe_metric = pi_metric = clipiqa_metric = musiq_metric = None

        # Fallbacks via piq and skimage, if some pyiqa metrics are unavailable
        lpips_fn = None
        piq_modules = {}
        _niqe_fn = None
        try:
            import piq  # type: ignore
            # LPIPS
            if lpips_metric is None:
                try:
                    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                    def _lpips_piq(a, b):
                        return piq.LPIPS(reduction='none').to(device)(a, b)
                    lpips_fn = _lpips_piq
                except Exception:
                    lpips_fn = None
            # No-reference: PI/CLIP-IQA/MUSIQ
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            if pi_metric is None:
                try:
                    piq_modules['pi'] = piq.PerceptualIndex().to(device).eval()
                except Exception:
                    pass
            if clipiqa_metric is None:
                try:
                    piq_modules['clipiqa'] = piq.CLIP_IQA().to(device).eval()
                except Exception:
                    pass
            if musiq_metric is None:
                try:
                    piq_modules['musiq'] = piq.MUSIQ().to(device).eval()
                except Exception:
                    pass
        except Exception:
            lpips_fn = None
            piq_modules = {}
        # NIQE via scikit-image (if pyiqa missing)
        if niqe_metric is None:
            try:
                from skimage.metrics import niqe as _niqe  # type: ignore
                def _niqe_score(x: torch.Tensor) -> float:
                    x_np = (x.clamp(0,1).permute(0,2,3,1).cpu().numpy())  # N H W C
                    # convert each image to grayscale and average score
                    scores = []
                    for i in range(x_np.shape[0]):
                        g = 0.299*x_np[i,...,0] + 0.587*x_np[i,...,1] + 0.114*x_np[i,...,2]
                        scores.append(float(_niqe(g)))
                    return float(np.mean(scores)) if scores else float('nan')
                _niqe_fn = _niqe_score
            except Exception:
                _niqe_fn = None

        # Print backend status and helpful hints for missing weights
        try:
            def _bk(flag, alt1=None, alt2=None):
                return 'pyiqa' if flag else (alt1 if alt1 else (alt2 if alt2 else 'none'))
            bk_lpips = _bk(lpips_metric is not None, 'piq' if lpips_fn is not None else None)
            bk_niqe = _bk(niqe_metric is not None, 'skimage' if _niqe_fn is not None else None)
            bk_pi = _bk(pi_metric is not None, 'piq' if ('pi' in piq_modules) else None)
            bk_clip = _bk(clipiqa_metric is not None, 'piq' if ('clipiqa' in piq_modules) else None)
            bk_musiq = _bk(musiq_metric is not None, 'piq' if ('musiq' in piq_modules) else None)
            self.write_log(f"[IQA] Backends: LPIPS={bk_lpips} NIQE={bk_niqe} PI={bk_pi} CLIPIQA={bk_clip} MUSIQ={bk_musiq}")
            cache_dir_show = os.environ.get('PYIQA_CACHE_DIR', '')
            for name, flag in [('LPIPS', lpips_metric is not None or lpips_fn is not None),
                               ('NIQE', (niqe_metric is not None) or (_niqe_fn is not None)),
                               ('PI', (pi_metric is not None) or ('pi' in piq_modules)),
                               ('CLIPIQA', (clipiqa_metric is not None) or ('clipiqa' in piq_modules)),
                               ('MUSIQ', (musiq_metric is not None) or ('musiq' in piq_modules))]:
                if not flag:
                    self.write_log(
                        f"[IQA][HINT] {name} unavailable. Ensure pretrained weights are present under {cache_dir_show}. "
                        f"You can pre-download by running once with internet: \n"
                        f"  python -c \"import pyiqa; pyiqa.create_metric('{name.lower()}')\" \n"
                        f"or manually placing the weights referenced by pyiqa into {cache_dir_show}."
                    )
        except Exception:
            pass

        # model param count
        def _count_module_params(m):
            return sum(p.numel() for p in m.parameters()) if isinstance(m, torch.nn.Module) else 0
        def _count_pipeline_params(pipe):
            names = ['unet', 'transformer', 'vae', 'text_encoder', 'text_encoder_2', 'image_encoder', 'safety_checker', 'controlnet', 'prior', 'refiner']
            visited = set()
            total = 0
            for n in names:
                m = getattr(pipe, n, None)
                if isinstance(m, torch.nn.Module):
                    for p in m.parameters():
                        if id(p) not in visited:
                            visited.add(id(p))
                            total += p.numel()
            return total
        try:
            pipe_params = _count_pipeline_params(self.sd_pipe)
            start_params = _count_module_params(getattr(self.sd_pipe, 'start_noise_predictor', None))
            total_params = pipe_params + start_params
            self.write_log(
                f"Model params (M): total={total_params/1e6:.2f} | pipe={pipe_params/1e6:.2f} | start={start_params/1e6:.2f}"
            )
        except Exception:
            self.write_log("Inference model params: Unknown")

        # build ref map if recursive ref_dir is provided (support nested subfolders)
        ref_map = {}
        if ref_dir:
            try:
                ref_path = Path(ref_dir)
                if ref_path.exists() and ref_path.is_dir():
                    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
                    for p in ref_path.rglob('*'):
                        if p.suffix.lower() in exts:
                            stem = p.stem
                            # keep the first occurrence for a stem (avoid overwriting)
                            if stem not in ref_map:
                                ref_map[stem] = p
                    self.write_log(f'Found {len(ref_map)} reference images under {ref_dir}')
                else:
                    self.write_log(f'Ref dir {ref_dir} not found or not a directory')
            except Exception as e:
                self.write_log(f'Failed to scan ref_dir {ref_dir}: {e}')

        if in_path.is_dir():
            data_config = {'type': 'base',
                           'params': {'dir_path': str(in_path),
                                      'transform_type': 'default',
                                      'transform_kwargs': {
                                          'mean': 0.0,
                                          'std': 1.0,
                                          },
                                      'need_path': True,
                                      'recursive': False,
                                      'length': None,
                                      }
                           }
            dataset = create_dataset(data_config)
            self.write_log(f'Find {len(dataset)} images in {in_path}')
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=bs, shuffle=False, drop_last=False,
            )

            # aggregate metrics
            debug_metrics = bool(self.configs.get('debug_metrics', False))
            agg = { 'count': 0 }
            metric_sums = {k: 0.0 for k in ['PSNR','SSIM','LPIPS','NIQE','PI','CLIPIQA','MUSIQ']}
            metric_counts = {k: 0 for k in ['PSNR','SSIM','LPIPS','NIQE','PI','CLIPIQA','MUSIQ']}

            # Optional FID accumulator: inference_invsr should attach a
            # runtime FIDOnline instance to the sampler (sampler.fid_online).
            # For backward compatibility we fall back to checking configs.
            fid_acc = getattr(self, "fid_online", None)
            if fid_acc is None:
                fid_acc = getattr(self.configs, "fid_online", None)
            # helper: safe PSNR/SSIM computation with fallbacks
            def safe_psnr_ssim(pred_255, gt_255, ycbcr_flag=False):
                """Compute PSNR and SSIM robustly. Return (psnr_value_or_none, ssim_value_or_none)."""
                try:
                    # ensure numpy arrays, contiguous, uint8
                    pred = np.ascontiguousarray(pred_255.astype(np.uint8))
                    gt = np.ascontiguousarray(gt_255.astype(np.uint8))
                except Exception:
                    return None, None
                psnr_v = None
                ssim_v = None
                # PSNR: prefer OpenCV, then util_image, then manual
                try:
                    psnr_v = cv2.PSNR(pred, gt)
                    if debug_metrics:
                        self.write_log(f"[DEBUG] cv2.PSNR OK: {psnr_v}")
                except Exception as e0:
                    if debug_metrics:
                        self.write_log(f"[DEBUG] cv2.PSNR failed: {e0}")
                if psnr_v is None or (isinstance(psnr_v, float) and np.isnan(psnr_v)):
                    try:
                        psnr_v = util_image.calculate_psnr(pred, gt, border=0, ycbcr=ycbcr_flag)
                        if debug_metrics:
                            self.write_log(f"[DEBUG] util_image.calculate_psnr OK: {psnr_v}")
                    except Exception as e:
                        if debug_metrics:
                            self.write_log(f"[DEBUG] util_image.calculate_psnr failed: {e}")
                        try:
                            im1 = pred.astype(np.float64)
                            im2 = gt.astype(np.float64)
                            mse = np.mean((im1 - im2) ** 2)
                            if mse == 0:
                                psnr_v = float('inf')
                            else:
                                psnr_v = 20 * math.log10(255.0 / math.sqrt(mse))
                            if debug_metrics:
                                self.write_log(f"[DEBUG] manual PSNR OK: {psnr_v}")
                        except Exception as e3:
                            if debug_metrics:
                                self.write_log(f"[DEBUG] manual PSNR failed: {e3}")
                            psnr_v = None
                # SSIM: try util_image.calculate_ssim, fallback to per-channel util_image.ssim
                try:
                    ssim_v = util_image.calculate_ssim(pred, gt, border=0, ycbcr=ycbcr_flag)
                    if debug_metrics:
                        self.write_log(f"[DEBUG] util_image.calculate_ssim OK: {ssim_v}")
                except Exception as e:
                    if debug_metrics:
                        self.write_log(f"[DEBUG] util_image.calculate_ssim failed: {e}")
                    try:
                        # fallback: per-channel ssim on float images
                        im1 = pred.astype(np.float64)
                        im2 = gt.astype(np.float64)
                        if im1.ndim == 3 and im1.shape[2] == 3:
                            ssims = [util_image.ssim(im1[:,:,i], im2[:,:,i]) for i in range(3)]
                            ssim_v = float(np.mean(ssims))
                        elif im1.ndim == 2:
                            ssim_v = float(util_image.ssim(im1, im2))
                        else:
                            ssim_v = None
                        if debug_metrics:
                            self.write_log(f"[DEBUG] per-channel SSIM OK: {ssim_v}")
                    except Exception as e2:
                        if debug_metrics:
                            self.write_log(f"[DEBUG] per-channel SSIM failed: {e2}")
                        # try skimage as last resort
                        try:
                            from skimage.metrics import structural_similarity as sk_ssim
                            if pred.ndim == 3 and pred.shape[2] == 3:
                                ssim_acc = 0.0
                                for ch in range(3):
                                    ssim_acc += sk_ssim(pred[:,:,ch], gt[:,:,ch], data_range=255)
                                ssim_v = float(ssim_acc / 3.0)
                            else:
                                ssim_v = float(sk_ssim(pred, gt, data_range=255))
                            if debug_metrics:
                                self.write_log(f"[DEBUG] skimage SSIM OK: {ssim_v}")
                        except Exception as e3:
                            if debug_metrics:
                                self.write_log(f"[DEBUG] skimage SSIM failed: {e3}")
                            ssim_v = None
                return psnr_v, ssim_v

            for data in dataloader:
                lq = data['lq'].cuda()
                start_t = time.time()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                res = self.sample_func(lq)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.time() - start_t

                # res: b x h x w x c (numpy)
                # If FID is enabled, update online accumulator with current
                # batch of predictions (converted to torch NCHW [0,1]).
                if fid_acc is not None:
                    # res is numpy float32 in [0,1], BxHxWxC
                    res_t = torch.from_numpy(res.astype('float32')).permute(0,3,1,2)
                    try:
                        fid_acc.update(res_t)
                    except Exception as e:
                        self.write_log(f"[WARN] FID update failed for batch: {e}")
                for jj in range(res.shape[0]):
                    im_name = Path(data['path'][jj]).stem
                    save_path = str(out_path / f"{im_name}.png")
                    util_image.imwrite(res[jj], save_path, dtype_in='float32')

                    # prepare tensors for metrics (CHW, [0,1])
                    im_pred = torch.from_numpy(res[jj].astype('float32')).permute(2,0,1).unsqueeze(0)
                    if torch.cuda.is_available():
                        im_pred = im_pred.cuda()

                    # per-image last metrics
                    last_psnr = None
                    last_ssim = None
                    last_lpips = None
                    last_niqe = None
                    last_pi = None
                    last_clipiqa = None
                    last_musiq = None

                    # find GT if provided
                    has_ref = False
                    im_gt = None
                    if ref_dir:
                        # Prefer pre-built recursive map (supports nested subfolders)
                        stem = im_name
                        found = None
                        if ref_map and stem in ref_map:
                            found = ref_map[stem]
                        else:
                            # fallback: try common extensions directly under ref_dir
                            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                                candidate = Path(ref_dir) / f"{stem}{ext}"
                                if candidate.exists():
                                    found = candidate
                                    break
                        if found:
                            try:
                                gt_np = util_image.imread(found, chn='rgb', dtype='float32')
                                im_gt = torch.from_numpy(gt_np.transpose(2,0,1)).unsqueeze(0)
                                if torch.cuda.is_available():
                                    im_gt = im_gt.cuda()
                                has_ref = True
                            except Exception:
                                has_ref = False

                    # compute reference metrics when GT exists
                    if has_ref:
                        try:
                            # convert both to numpy HWC [0,255] uint8 for stable PSNR/SSIM
                            # im_pred: torch Tensor 1xC x H x W in [0,1]
                            if isinstance(im_pred, torch.Tensor):
                                pred_np = im_pred.detach().cpu().squeeze(0).permute(1,2,0).numpy()
                            else:
                                pred_np = im_pred.squeeze(0).permute(1,2,0).numpy()
                            if isinstance(im_gt, torch.Tensor):
                                gt_np = im_gt.detach().cpu().squeeze(0).permute(1,2,0).numpy()
                            else:
                                gt_np = im_gt.squeeze(0).permute(1,2,0).numpy()
                            # clip and convert to uint8 0-255
                            pred_255 = (np.clip(pred_np, 0.0, 1.0) * 255.0).round().astype('uint8')
                            gt_255 = (np.clip(gt_np, 0.0, 1.0) * 255.0).round().astype('uint8')
                            # optional detailed debug and saving
                            try:
                                if debug_metrics:
                                    self.write_log(
                                        f"[DEBUG] pred_255: shape={pred_255.shape} dtype={pred_255.dtype} min={pred_255.min()} max={pred_255.max()} | "
                                        f"gt_255: shape={gt_255.shape} dtype={gt_255.dtype} min={gt_255.min()} max={gt_255.max()}"
                                    )
                                    dbg_dir = out_path / 'debug'
                                    dbg_dir.mkdir(parents=True, exist_ok=True)
                                    cv2.imwrite(str(dbg_dir / f"pred_{im_name}.png"), cv2.cvtColor(pred_255, cv2.COLOR_RGB2BGR))
                                    cv2.imwrite(str(dbg_dir / f"gt_{im_name}.png"), cv2.cvtColor(gt_255, cv2.COLOR_RGB2BGR))
                            except Exception:
                                pass

                            # PSNR / SSIM using robust helper (默认不使用Y通道，避免某些实现差异)
                            ycbcr_flag = False
                            try:
                                cur_psnr, cur_ssim = safe_psnr_ssim(pred_255, gt_255, ycbcr_flag)
                                if cur_psnr is not None:
                                    metric_sums['PSNR'] += float(cur_psnr)
                                    metric_counts['PSNR'] += 1
                                    last_psnr = float(cur_psnr)
                                else:
                                    self.write_log(f"[DEBUG] PSNR returned None for {im_name}")
                                    # save debug images if not already saved
                                    try:
                                        if not debug_metrics:
                                            dbg_dir = out_path / 'debug'
                                            dbg_dir.mkdir(parents=True, exist_ok=True)
                                            cv2.imwrite(str(dbg_dir / f"pred_{im_name}.png"), cv2.cvtColor(pred_255, cv2.COLOR_RGB2BGR))
                                            cv2.imwrite(str(dbg_dir / f"gt_{im_name}.png"), cv2.cvtColor(gt_255, cv2.COLOR_RGB2BGR))
                                    except Exception:
                                        pass
                                if cur_ssim is not None:
                                    metric_sums['SSIM'] += float(cur_ssim)
                                    metric_counts['SSIM'] += 1
                                    last_ssim = float(cur_ssim)
                                else:
                                    self.write_log(f"[DEBUG] SSIM returned None for {im_name}")
                                    try:
                                        if not debug_metrics:
                                            dbg_dir = out_path / 'debug'
                                            dbg_dir.mkdir(parents=True, exist_ok=True)
                                            cv2.imwrite(str(dbg_dir / f"pred_{im_name}.png"), cv2.cvtColor(pred_255, cv2.COLOR_RGB2BGR))
                                            cv2.imwrite(str(dbg_dir / f"gt_{im_name}.png"), cv2.cvtColor(gt_255, cv2.COLOR_RGB2BGR))
                                    except Exception:
                                        pass
                            except Exception:
                                try:
                                    self.write_log(f"[DEBUG] safe_psnr_ssim failed for {im_name}")
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    # Log which GT path was used (or missing) for traceability
                    if ref_dir:
                        stem = im_name
                        found = None
                        if ref_map and stem in ref_map:
                            found = ref_map[stem]
                        else:
                            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                                candidate = Path(ref_dir) / f"{stem}{ext}"
                                if candidate.exists():
                                    found = candidate
                                    break
                        if found:
                            self.write_log(f"GT resolved for {im_name}: {str(found)}")
                        else:
                            self.write_log(f"No GT found for {im_name} under {ref_dir}")

                    if has_ref and (lpips_metric is not None or lpips_fn is not None):
                        try:
                            if lpips_metric is not None:
                                cur_lpips = lpips_metric(im_pred, im_gt).sum().item()
                            else:
                                cur_lpips = lpips_fn(im_pred, im_gt).mean().item()
                            metric_sums['LPIPS'] += cur_lpips
                            metric_counts['LPIPS'] += 1
                            last_lpips = float(cur_lpips)
                        except Exception:
                            pass

                    # non-reference metrics
                    if (niqe_metric is not None) or (_niqe_fn is not None):
                        try:
                            if niqe_metric is not None:
                                cur_niqe = niqe_metric(im_pred).sum().item()
                            else:
                                cur_niqe = _niqe_fn(im_pred)
                            metric_sums['NIQE'] += cur_niqe
                            metric_counts['NIQE'] += 1
                            last_niqe = float(cur_niqe)
                        except Exception:
                            pass
                    if (pi_metric is not None) or ('pi' in piq_modules):
                        try:
                            if pi_metric is not None:
                                cur_pi = pi_metric(im_pred).sum().item()
                            else:
                                cur_pi = piq_modules['pi'](im_pred).sum().item()
                            metric_sums['PI'] += cur_pi
                            metric_counts['PI'] += 1
                            last_pi = float(cur_pi)
                        except Exception:
                            pass
                    if (clipiqa_metric is not None) or ('clipiqa' in piq_modules):
                        try:
                            # CLIP-IQA often expects 224x224; resize if needed
                            import torchvision.transforms.functional as TF
                            inp = im_pred
                            # prefer CPU for heavy backbones when CUDA not available
                            if not torch.cuda.is_available():
                                inp = inp.cpu()
                            # resize to 224x224 while preserving batch/channel
                            h, w = inp.shape[-2:]
                            if h != 224 or w != 224:
                                inp_resized = TF.resize(inp, size=[224, 224], interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
                            else:
                                inp_resized = inp
                            if clipiqa_metric is not None:
                                cur_clipiqa = clipiqa_metric(inp_resized).sum().item()
                            else:
                                cur_clipiqa = piq_modules['clipiqa'](inp_resized).sum().item()
                            metric_sums['CLIPIQA'] += cur_clipiqa
                            metric_counts['CLIPIQA'] += 1
                            last_clipiqa = float(cur_clipiqa)
                        except Exception:
                            pass
                    if (musiq_metric is not None) or ('musiq' in piq_modules):
                        try:
                            import torchvision.transforms.functional as TF
                            inp = im_pred
                            if not torch.cuda.is_available():
                                inp = inp.cpu()
                            h, w = inp.shape[-2:]
                            # MUSIQ expects 224x224 (ViT), resize to standard size
                            if h != 224 or w != 224:
                                inp_resized = TF.resize(inp, size=[224, 224], interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
                            else:
                                inp_resized = inp
                            if musiq_metric is not None:
                                cur_musiq = musiq_metric(inp_resized).sum().item()
                            else:
                                cur_musiq = piq_modules['musiq'](inp_resized).sum().item()
                            metric_sums['MUSIQ'] += cur_musiq
                            metric_counts['MUSIQ'] += 1
                            last_musiq = float(cur_musiq)
                        except Exception:
                            pass

                    agg['count'] += 1
                    # print per-image summary with fixed formatting and full metric set
                    def _fmt_val(v, fmt):
                        import math as _m
                        if v is None:
                            return 'nan'
                        if isinstance(v, float) and (_m.isnan(v) or _m.isinf(v)):
                            return 'nan' if _m.isnan(v) else 'inf'
                        return (f"{v:{fmt}}")

                    psnr_str = _fmt_val(last_psnr, '.4f') if has_ref else 'nan'
                    ssim_str = _fmt_val(last_ssim, '.4f') if has_ref else 'nan'
                    lpips_str = _fmt_val(last_lpips, '.4f') if has_ref else 'nan'
                    niqe_str = _fmt_val(last_niqe, '.6f')
                    pi_str = _fmt_val(last_pi, '.6f')
                    clip_str = _fmt_val(last_clipiqa, '.6f')
                    musiq_str = _fmt_val(last_musiq, '.6f')
                    msg = (
                        f"{im_name}: time={elapsed:.3f}s | "
                        f"PSNR={psnr_str} SSIM={ssim_str} LPIPS={lpips_str} | "
                        f"NIQE={niqe_str} PI={pi_str} CLIPIQA={clip_str} MUSIQ={musiq_str}"
                    )
                    self.write_log(msg)

                    # print running averages across all seen images (uniform .4f for readability)
                    parts = []
                    for k in ['PSNR','SSIM','LPIPS','NIQE','PI','CLIPIQA','MUSIQ']:
                        if metric_counts.get(k, 0) > 0:
                            parts.append(f"{k}={metric_sums[k]/metric_counts[k]:.4f}")
                    if parts:
                        self.write_log("Running means so far: " + " ".join(parts))

            # final averages
            if agg['count'] > 0:
                self.write_log("\n=== Inference summary ===")
                if metric_counts['PSNR'] > 0:
                    self.write_log(f"Mean PSNR: {metric_sums['PSNR']/metric_counts['PSNR']:.4f}")
                if metric_counts['SSIM'] > 0:
                    self.write_log(f"Mean SSIM: {metric_sums['SSIM']/metric_counts['SSIM']:.4f}")
                if metric_counts['LPIPS'] > 0:
                    self.write_log(f"Mean LPIPS: {metric_sums['LPIPS']/metric_counts['LPIPS']:.4f}")
                if metric_counts['NIQE'] > 0:
                    self.write_log(f"Mean NIQE: {metric_sums['NIQE']/metric_counts['NIQE']:.4f}")
                if metric_counts['PI'] > 0:
                    self.write_log(f"Mean PI: {metric_sums['PI']/metric_counts['PI']:.4f}")
                if metric_counts['CLIPIQA'] > 0:
                    self.write_log(f"Mean CLIPIQA: {metric_sums['CLIPIQA']/metric_counts['CLIPIQA']:.4f}")
                if metric_counts['MUSIQ'] > 0:
                    self.write_log(f"Mean MUSIQ: {metric_sums['MUSIQ']/metric_counts['MUSIQ']:.4f}")
                # consolidated one-line summary for quick copy/paste
                means = {}
                for k in ['PSNR','SSIM','LPIPS','NIQE','PI','CLIPIQA','MUSIQ']:
                    if metric_counts.get(k, 0) > 0:
                        means[k] = metric_sums[k] / metric_counts[k]
                if means:
                    parts = [f"{k}={means[k]:.4f}" for k in ['PSNR','SSIM','LPIPS','NIQE','PI','CLIPIQA','MUSIQ'] if k in means]
                    self.write_log("Overall means: " + " ".join(parts))

                # If FID accumulator is present, compute and report FID at
                # the very end (over all generated images).
                if fid_acc is not None:
                    try:
                        fid_value = fid_acc.compute()
                        self.write_log(f"FID vs ref stats: {fid_value:.4f}")
                    except Exception as e:
                        self.write_log(f"[WARN] Failed to compute FID: {e}")
        else:
            im_cond = util_image.imread(in_path, chn='rgb', dtype='float32')  # h x w x c
            im_cond = util_image.img2tensor(im_cond).cuda()                   # 1 x c x h x w

            # single image inference: measure time and optionally metrics
            start_t = time.time()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            image = self.sample_func(im_cond).squeeze(0)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - start_t

            save_path = str(out_path / f"{in_path.stem}.png")
            util_image.imwrite(image, save_path, dtype_in='float32')

            # compute metrics if possible
            if pyiqa is not None:
                im_pred = torch.from_numpy(image.astype('float32')).permute(2,0,1).unsqueeze(0)
                if torch.cuda.is_available():
                    im_pred = im_pred.cuda()
                has_ref = False
                im_gt = None
                if ref_dir:
                    # Prefer pre-built recursive map (supports nested subfolders)
                    stem = in_path.stem
                    found = None
                    if ref_map and stem in ref_map:
                        found = ref_map[stem]
                    else:
                        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                            cand = Path(ref_dir) / f"{in_path.stem}{ext}"
                            if cand.exists():
                                found = cand
                                break
                    if found:
                        try:
                            gt_np = util_image.imread(found, chn='rgb', dtype='float32')
                            im_gt = torch.from_numpy(gt_np.transpose(2,0,1)).unsqueeze(0)
                            if torch.cuda.is_available():
                                im_gt = im_gt.cuda()
                            has_ref = True
                        except Exception:
                            has_ref = False

                out_msg = f"{in_path.stem}: time={elapsed:.3f}s"
                if has_ref and psnr_metric is not None:
                    try:
                        v = psnr_metric(im_pred, im_gt).sum().item(); out_msg += f" PSNR={v:.4f}"
                    except Exception:
                        pass
                if has_ref and ssim_metric is not None:
                    try:
                        v = ssim_metric(im_pred, im_gt).sum().item(); out_msg += f" SSIM={v:.4f}"
                    except Exception:
                        pass
                if has_ref and lpips_metric is not None:
                    try:
                        v = lpips_metric(im_pred, im_gt).sum().item(); out_msg += f" LPIPS={v:.4f}"
                    except Exception:
                        pass
                if niqe_metric is not None:
                    try:
                        v = niqe_metric(im_pred).sum().item(); out_msg += f" NIQE={v:.4f}"
                    except Exception:
                        pass
                if pi_metric is not None:
                    try:
                        v = pi_metric(im_pred).sum().item(); out_msg += f" PI={v:.4f}"
                    except Exception:
                        pass
                if clipiqa_metric is not None:
                    try:
                        inp = im_pred.cpu() if not torch.cuda.is_available() else im_pred
                        v = clipiqa_metric(inp).sum().item(); out_msg += f" CLIPIQA={v:.4f}"
                    except Exception:
                        pass
                if musiq_metric is not None:
                    try:
                        inp = im_pred.cpu() if not torch.cuda.is_available() else im_pred
                        v = musiq_metric(inp).sum().item(); out_msg += f" MUSIQ={v:.4f}"
                    except Exception:
                        pass
                self.write_log(out_msg)

        self.write_log(f"Processing done, enjoy the results in {str(out_path)}")

def get_torch_dtype(torch_dtype: str):
    if torch_dtype == 'torch.float16':
        return torch.float16
    elif torch_dtype == 'torch.bfloat16':
        return torch.bfloat16
    elif torch_dtype == 'torch.float32':
        return torch.float32
    else:
        raise ValueError(f'Unexpected torch dtype:{torch_dtype}')

if __name__ == '__main__':
    pass

