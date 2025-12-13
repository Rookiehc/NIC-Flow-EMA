#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import math
import time
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from omegaconf import OmegaConf
from utils import util_common
import contextlib

from sampler_invsr import InvSamplerSR
from invsr.imm.adapters.sd_adapter import SDAdapter
from invsr.imm.modules.time_embed import TimeFusion
from invsr.losses.imm_loss import IMMLoss


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


class TrainerIMM:
    """
    Minimal IMM trainer for InvSR:
    - Uses InvSamplerSR to build UNet/VAEs/scheduler and dataloader/optimizer when available.
    - Wraps UNet with SDAdapter (student + EMA teacher).
    - Trains with IMMLoss (MMD-like) over dual-time branches.
    - Saves checkpoints via sampler or direct torch.save.
    """

    def __init__(self, configs):
        self.cfg = configs
        # DDP setup
        self.distributed = False
        self.local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0] if 'CUDA_VISIBLE_DEVICES' in os.environ else 0)) if torch.cuda.is_available() else -1
        self.rank = int(os.environ.get('RANK', 0)) if torch.cuda.is_available() else 0
        self.world_size = int(os.environ.get('WORLD_SIZE', 1)) if torch.cuda.is_available() else 1
        if torch.cuda.is_available() and self.world_size > 1:
            # Initialize process group if not already inited (torch.distributed.run sets envs)
            if not (dist.is_available() and dist.is_initialized()):
                dist.init_process_group(backend='nccl', init_method='env://')
            torch.cuda.set_device(self.local_rank)
            self.distributed = True
        self.device = torch.device(f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu')

        # Ensure top-level seed exists for legacy code paths that expect configs.seed
        if not hasattr(self.cfg, 'seed'):
            # try common locations
            seed_val = None
            if isinstance(self.cfg.get('train', None), dict):
                seed_val = self.cfg.get('train', {}).get('seed', None)
            if seed_val is None and isinstance(self.cfg.get('training', None), dict):
                seed_val = self.cfg.get('training', {}).get('seed', None)
            if seed_val is None:
                seed_val = 123456
            try:
                self.cfg.seed = int(seed_val)
            except Exception:
                self.cfg.seed = seed_val

        # Build sampler to reuse model and data wiring.
        # Ensure configs contain `sd_pipe` required by InvSamplerSR.build_model()
        if not hasattr(self.cfg, 'sd_pipe') or self.cfg.sd_pipe is None:
            print("configs.sd_pipe missing — injecting minimal sd_pipe defaults for IMM training")
            self.cfg.sd_pipe = OmegaConf.create({
                'target': 'diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline',
                'num_train_steps': 1000,
                'enable_grad_checkpoint': True,
                'compile': False,
                'vae_split': 8,
                'offload_vae_to_cpu': True,
                'params': {
                    'pretrained_model_name_or_path': 'stabilityai/sd-turbo',
                    'cache_dir': 'weights',
                    'use_safetensors': True,
                    'torch_dtype': 'torch.float16',
                }
            })
        self.sampler = InvSamplerSR(self.cfg)

        # Expect sampler to expose core components.
        pipe = getattr(self.sampler, 'sd_pipe', None)
        has_unet = hasattr(pipe, 'unet') or hasattr(pipe, 'transformer')
        has_scheduler = hasattr(pipe, 'scheduler')
        if not has_unet or not has_scheduler:
            raise RuntimeError('InvSamplerSR must expose `unet/transformer` and `scheduler` for IMM training.')
        self.unet = self.sampler.unet
        self.vae = getattr(self.sampler, 'vae', None)
        self.scheduler = self.sampler.scheduler

        # Time fusion for t+s embedding.
        self.time_fusion = TimeFusion(hidden_size=256)

        # Student & EMA teacher adapters.
        # Expose tokenizer/text_encoder from sd_pipe when available for cross-attn text conditioning
        tokenizer = getattr(self.sampler.sd_pipe, 'tokenizer', None) if hasattr(self.sampler, 'sd_pipe') else None
        text_encoder = getattr(self.sampler.sd_pipe, 'text_encoder', None) if hasattr(self.sampler, 'sd_pipe') else None

        self.adapter_student = SDAdapter(
            unet=self.unet, vae=self.vae, scheduler=self.scheduler, time_embed=self.time_fusion,
            tokenizer=tokenizer, text_encoder=text_encoder,
        )
        self.adapter_teacher = SDAdapter(
            unet=deepcopy(self.unet).eval().requires_grad_(False), vae=self.vae, scheduler=self.scheduler, time_embed=self.time_fusion,
            tokenizer=tokenizer, text_encoder=text_encoder,
        )

        # Move models to correct device
        try:
            self.adapter_student.unet.to(self.device)
            self.adapter_teacher.unet.to(self.device)
        except Exception:
            pass

        # Wrap student with DDP if distributed
        if self.distributed:
            # It is important to wrap before creating optimizer
            self.unet = DDP(self.adapter_student.unet, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=False)
            # Ensure adapter_student references the wrapped module
            self.adapter_student.unet = self.unet

        # Optional: force training params to FP32 for stability when weights are fp16
        train_cfg = self.cfg.get('training', None) or self.cfg.get('train', {}) or {}
        force_fp32 = bool(train_cfg.get('force_fp32', True))
        if force_fp32:
            try:
                self.adapter_student.unet.to(torch.float32)
                self.adapter_teacher.unet.to(torch.float32)
            except Exception:
                pass

        # Loss.
        loss_cfg = self.cfg.get('loss', {})
        self.loss_fn = IMMLoss(**loss_cfg)

        # Optional LPIPS for HR重建质量评估/训练
        self.lpips_fn = None
        try:
            llp_cfg = self.cfg.get('llpips', None)
            if llp_cfg is not None:
                from latent_lpips.lpips import LPIPS as _LPIPS
                self.lpips_fn = _LPIPS(
                    pretrained=bool(llp_cfg.params.get('pretrained', False)),
                    net=str(llp_cfg.params.get('net', 'vgg')),
                    lpips=bool(llp_cfg.params.get('lpips', True)),
                    spatial=bool(llp_cfg.params.get('spatial', False)),
                    pnet_rand=bool(llp_cfg.params.get('pnet_rand', False)),
                    pnet_tune=bool(llp_cfg.params.get('pnet_tune', True)),
                    use_dropout=bool(llp_cfg.params.get('use_dropout', True)),
                    eval_mode=bool(llp_cfg.params.get('eval_mode', True)),
                    # 对图像空间 LPIPS：latent=False 且 in_chans=3（RGB）
                    latent=False,
                    in_chans=3,
                    verbose=bool(llp_cfg.params.get('verbose', False)),
                ).to(self.device).eval()
        except Exception:
            self.lpips_fn = None

        # Optimizer: prefer sampler's optimizer if provided.
        if hasattr(self.sampler, 'get_optimizer'):
            # If DDP wrapped, pass underlying module for param extraction when needed
            base_model_for_opt = self.unet.module if hasattr(self.unet, 'module') else self.unet
            self.optim = self.sampler.get_optimizer(base_model_for_opt)
        else:
            opt_cfg = self.cfg.get('optimizer', {})
            lr = float(opt_cfg.get('lr', 1e-4))
            betas = tuple(opt_cfg.get('betas', [0.9, 0.99]))
            wd = float(opt_cfg.get('weight_decay', 0.01))
            params = (self.unet.module.parameters() if hasattr(self.unet, 'module') else self.unet.parameters())
            self.optim = torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=wd)

        # 可选：实例化判别器（来自配置 discriminator），用于替代之前的 ldis 简化方案
        self.dis = None
        self.optim_dis = None
        try:
            dis_cfg = self.cfg.get('discriminator', None)
            if dis_cfg and dis_cfg.get('target', None):
                # 通过 util_common.instantiate_from_config 创建判别器
                self.dis = util_common.instantiate_from_config(dis_cfg)
                self.dis.to(self.device)
                # 判别器优化器
                tr_cfg = self.cfg.get('train', {})
                lr_dis = float(tr_cfg.get('lr_dis', 5e-5))
                wd_dis = float(tr_cfg.get('weight_decay_dis', 1e-3))
                self.optim_dis = torch.optim.AdamW(self.dis.parameters(), lr=lr_dis, weight_decay=wd_dis)
        except Exception as e:
            print(f"[IMM][WARN] Discriminator init failed: {e}")

        # Dataloader: prefer sampler's train loader; else build from configs.data.train
        if hasattr(self.sampler, 'get_train_loader'):
            self.loader = self.sampler.get_train_loader()
        else:
            # Build minimal train dataloader using existing dataset factory if available
            try:
                from datapipe.datasets import create_dataset
                data_cfg_root = self.cfg.get('data', {})
                train_data_cfg = data_cfg_root.get('train', {})
                # Factory expects a single config object; pass the train block directly
                dataset = create_dataset(train_data_cfg)
                # Simple sanity check: ensure non-empty dataset and log target roots for debugging
                try:
                    ds_len = len(dataset)
                except Exception:
                    ds_len = None
                if not ds_len:
                    hint = ''
                    try:
                        params = train_data_cfg.get('params', {})
                        if 'data_source' in params:
                            srcs = []
                            for k, v in params['data_source'].items():
                                root = getattr(v, 'root_path', None) or v.get('root_path')
                                imgp = getattr(v, 'image_path', None) or v.get('image_path')
                                ext = getattr(v, 'im_ext', None) or v.get('im_ext')
                                srcs.append(f"{k}: {root}/{imgp} (*.{ext})")
                            hint = ' | sources: ' + ' ; '.join(srcs)
                        rec_flag = params.get('recursive', None)
                        if rec_flag is not None:
                            hint += f" | recursive={rec_flag}"
                    except Exception:
                        pass
                    raise RuntimeError(f"Constructed empty training dataset (len=0){hint}. If your images are inside nested subfolders (e.g., /data/.../train_hr/0046000/*.png), set data.train.params.recursive: True and confirm im_ext matches.")
                train_cfg = self.cfg.get('training', None) or self.cfg.get('train', {}) or {}
                batch_size = int(train_cfg.get('batch', 4))
                num_workers = int(train_cfg.get('num_workers', 4))
                prefetch_factor = int(train_cfg.get('prefetch_factor', 2))
                self.loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    prefetch_factor=prefetch_factor,
                    pin_memory=True,
                    drop_last=True,
                )
            except Exception as e:
                raise RuntimeError(f'Failed to construct train dataloader from configs.data.train: {e}')
        # If distributed, ensure DataLoader uses DistributedSampler
        self.train_sampler = None
        if self.distributed and self.loader is not None:
            try:
                from torch.utils.data.distributed import DistributedSampler
                if not isinstance(getattr(self.loader, 'sampler', None), DistributedSampler):
                    # Rebuild DataLoader with DistributedSampler
                    ds = self.loader.dataset
                    bs = self.loader.batch_size if hasattr(self.loader, 'batch_size') else int(self.cfg.get('train', {}).get('batch', 4))
                    nw = self.loader.num_workers if hasattr(self.loader, 'num_workers') else int(self.cfg.get('train', {}).get('num_workers', 4))
                    pf = getattr(self.loader, 'prefetch_factor', 2)
                    collate = getattr(self.loader, 'collate_fn', None)
                    persistent = getattr(self.loader, 'persistent_workers', False)
                    dl = DataLoader(
                        ds,
                        batch_size=bs,
                        shuffle=False,
                        num_workers=nw,
                        prefetch_factor=pf,
                        pin_memory=True,
                        drop_last=True,
                        sampler=DistributedSampler(ds, shuffle=True),
                        collate_fn=collate,
                        persistent_workers=persistent,
                    )
                    self.loader = dl
                self.train_sampler = self.loader.sampler
            except Exception as e:
                print(f"[IMM][DDP][WARN] Failed to attach DistributedSampler: {e}")

        # EMA decay. accept either 'training' or 'train' section in config
        train_cfg = self.cfg.get('training', None) or self.cfg.get('train', {}) or {}
        self.ema_decay = float(train_cfg.get('ema_beta', train_cfg.get('ema_rate', 0.999)))

        # Save dir.
        self.save_dir = self.cfg.get('save_dir', './save_dir')
        _ensure_dir(self.save_dir)
        # 训练日志文件
        try:
            self.log_file_path = os.path.join(self.save_dir, 'train.log')
            # Only main process writes to log file
            if (not self.distributed) or (self.rank == 0):
                self._log_fp = open(self.log_file_path, 'a', buffering=1)
            else:
                self._log_fp = None
        except Exception:
            self._log_fp = None
        # AMP scaler for mixed precision stability
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    def _ema_update(self):
        with torch.no_grad():
            for p_t, p_s in zip(self.adapter_teacher.unet.parameters(), self.adapter_student.unet.parameters()):
                p_t.data.mul_(self.ema_decay).add_(p_s.data, alpha=1 - self.ema_decay)

    def _save(self, tag='imm_last'):
        if self.distributed and self.rank != 0:
            return
        if hasattr(self.sampler, 'save_checkpoint'):
            # Pass base module if wrapped
            model_to_save = self.unet.module if hasattr(self.unet, 'module') else self.unet
            self.sampler.save_checkpoint(model_to_save, tag=tag)
            return
        # Fallback: save UNet state_dict
        path = os.path.join(self.save_dir, f'{tag}.pth')
        model_to_save = self.unet.module if hasattr(self.unet, 'module') else self.unet
        payload = {'unet': model_to_save.state_dict()}
        # 额外保存噪声预测器（NoisePredictor）权重，如果存在
        try:
            # 优先从 sampler 中查找 model_start 或 noise_predictor
            noise_pred = None
            if hasattr(self.sampler, 'model_start'):
                noise_pred = getattr(self.sampler, 'model_start')
            elif hasattr(self.sampler, 'noise_predictor'):
                noise_pred = getattr(self.sampler, 'noise_predictor')
            # 也可能在 sd_pipe 里（按项目约定）
            if noise_pred is None and hasattr(self.sampler, 'sd_pipe'):
                noise_pred = getattr(self.sampler.sd_pipe, 'noise_predictor', None)
            if noise_pred is not None and hasattr(noise_pred, 'state_dict'):
                payload['noise_predictor'] = noise_pred.state_dict()
        except Exception:
            pass
        torch.save(payload, path)

    def train(self):
        import time
        train_cfg = self.cfg.get('training', None) or self.cfg.get('train', {}) or {}
        # 以“步”为单位控制总迭代数与保存/日志频率
        total_steps = int(train_cfg.get('iterations', train_cfg.get('total_ticks', 100)))
        save_every = int(train_cfg.get('save_freq', train_cfg.get('snapshot_ticks', 5000)))
        log_freq_cfg = train_cfg.get('log_freq', 200)
        # OmegaConf may return a ListConfig; convert to plain Python container first
        try:
            if OmegaConf.is_config(log_freq_cfg):
                log_freq_val = OmegaConf.to_container(log_freq_cfg, resolve=True)
            else:
                log_freq_val = log_freq_cfg
        except Exception:
            log_freq_val = log_freq_cfg

        if isinstance(log_freq_val, (list, tuple)) and len(log_freq_val) > 0:
            log_every = int(log_freq_val[0])
        else:
            log_every = int(log_freq_val)
        local_logging = bool(train_cfg.get('local_logging', True))

        # 统计信息
        try:
            steps_per_epoch = len(self.loader)
        except Exception:
            steps_per_epoch = None
        is_main = (not self.distributed) or (self.rank == 0)
        if local_logging and is_main:
            print(f"[IMM] Training start: total_steps={total_steps}, save_every={save_every}, log_every={log_every}, steps_per_epoch={steps_per_epoch}")
            if self._log_fp:
                self._log_fp.write(f"[IMM] Training start: total_steps={total_steps}, save_every={save_every}, log_every={log_every}, steps_per_epoch={steps_per_epoch}\n")

        # Resume support: if configs.resume is a checkpoint path, load weights and set starting step
        it = 0
        resume_path = str(self.cfg.get('resume', '') or '')
        if resume_path:
            try:
                if os.path.isfile(resume_path):
                    payload = torch.load(resume_path, map_location='cpu')
                    if isinstance(payload, dict):
                        # Load UNet
                        if 'unet' in payload:
                            target_model = self.unet.module if hasattr(self.unet, 'module') else self.unet
                            target_model.load_state_dict(payload['unet'], strict=False)
                        # Load NoisePredictor if present
                        try:
                            noise_pred = None
                            if hasattr(self.sampler, 'model_start'):
                                noise_pred = getattr(self.sampler, 'model_start')
                            elif hasattr(self.sampler, 'noise_predictor'):
                                noise_pred = getattr(self.sampler, 'noise_predictor')
                            elif hasattr(self.sampler, 'sd_pipe'):
                                noise_pred = getattr(self.sampler.sd_pipe, 'noise_predictor', None)
                            if noise_pred is not None and ('noise_predictor' in payload):
                                noise_pred.load_state_dict(payload['noise_predictor'], strict=False)
                        except Exception:
                            pass
                        # Infer start step from filename imm_step_XXXXX.pth
                        try:
                            import re
                            m = re.search(r"imm_step_(\d+)\.pth", os.path.basename(resume_path))
                            if m:
                                it = int(m.group(1))
                                print(f"[IMM][resume] Starting from step {it} based on filename.")
                        except Exception:
                            pass
                        print(f"[IMM][resume] Loaded checkpoint from {resume_path}")
                        # Sync among ranks to ensure all have loaded
                        if self.distributed and dist.is_initialized():
                            dist.barrier()
                else:
                    print(f"[IMM][resume][WARN] Path not found: {resume_path}")
            except Exception as e:
                print(f"[IMM][resume][WARN] Failed to load {resume_path}: {e}")
        t0 = time.time()
        epoch = 0
        printed_data_stats = False
        # 验证集（可选）：读取配置 data.val 并构建 loader
        val_cfg_root = self.cfg.get('data', {})
        val_data_cfg = val_cfg_root.get('val', None)
        val_loader = None
        val_every = int(train_cfg.get('val_freq', 0))
        val_max_batches = int(train_cfg.get('val_max_batches', 4))
        val_out_dir = os.path.join(self.save_dir, 'val_samples')
        _ensure_dir(val_out_dir)
        if val_data_cfg and val_every > 0 and is_main:
            try:
                from datapipe.datasets import create_dataset
                val_dataset = create_dataset(val_data_cfg)
                val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
                if local_logging and is_main:
                    print(f"[IMM] Val enabled: freq={val_every}, max_batches={val_max_batches}, len={len(val_dataset)}")
                    if self._log_fp:
                        self._log_fp.write(f"[IMM] Val enabled: freq={val_every}, max_batches={val_max_batches}, len={len(val_dataset)}\n")
            except Exception as e:
                print(f"[IMM][WARN] Failed to build val loader: {e}")
                val_loader = None

        def _tensor_stats(x: torch.Tensor):
            try:
                x_det = x.detach()
                numel = int(x_det.numel())
                bsz = int(x_det.shape[0]) if x_det.dim() > 0 else 0
                x_min = float(x_det.min().item()) if numel > 0 else float('nan')
                x_max = float(x_det.max().item()) if numel > 0 else float('nan')
                x_mean = float(x_det.mean().item()) if numel > 0 else float('nan')
                x_std = float(x_det.std().item()) if numel > 1 else float('nan')
                zero_frac = float((x_det == 0).float().mean().item()) if numel > 0 else float('nan')
                return bsz, numel, x_min, x_max, x_mean, x_std, zero_frac
            except Exception:
                return 0, 0, float('nan'), float('nan'), float('nan'), float('nan'), float('nan')

        def _grad_global_norm(module: nn.Module) -> float:
            total = 0.0
            for p in module.parameters():
                if p.grad is not None:
                    g = p.grad.detach()
                    total += float(g.pow(2).sum().item())
            return float(total) ** 0.5
        def _optimizer_has_fp16_params(opt: torch.optim.Optimizer) -> bool:
            try:
                for g in opt.param_groups:
                    for p in g.get('params', []):
                        if p is None:
                            continue
                        if getattr(p, 'dtype', None) is torch.float16:
                            return True
            except Exception:
                return False
            return False
        while it < total_steps:
            epoch += 1
            # Set epoch for distributed sampler
            if self.train_sampler is not None:
                try:
                    self.train_sampler.set_epoch(epoch)
                except Exception:
                    pass
            for batch in self.loader:
                it += 1
                images = batch.get('target', None)
                if images is None:
                    # fallback: try 'hr' or 'gt'
                    images = batch.get('hr', batch.get('gt'))
                cond = batch.get('cond', None)
                if images is None:
                    raise RuntimeError('Training batch must provide target/hr/gt images for IMM loss.')
                images = images.to(self.device)
                # 构造 cond_drop 以匹配 loss_fn 的 label dropout 逻辑
                cond_drop = self.loss_fn._build_cond(cond or {}, drop_prob=self.loss_fn.label_dropout)

                # Data sanity check (打印一次即可)
                if local_logging and not printed_data_stats:
                    bsz, numel, x_min, x_max, x_mean, x_std, zero_frac = _tensor_stats(images)
                    print(
                        f"[IMM][data] batch_size={bsz}, numel={numel}, min={x_min:.4f}, max={x_max:.4f}, "
                        f"mean={x_mean:.4f}, std={x_std:.4f}, zero_frac={zero_frac:.4f}",
                        flush=True,
                    )
                    printed_data_stats = True

                # Mixed precision autocast for numerical stability/perf
                # If model/optimizer params are already float16, GradScaler cannot unscale FP16 grads.
                # Detect that case and disable scaler (fall back to normal backward/step).
                optim_has_fp16 = _optimizer_has_fp16_params(self.optim)
                use_scaler = (self.scaler is not None) and (not optim_has_fp16)
                autocast_ctx = torch.cuda.amp.autocast if use_scaler else contextlib.nullcontext
                with autocast_ctx():
                    loss_mmd, logs = self.loss_fn(self.adapter_student, self.adapter_teacher, images, cond=cond, device=self.device)
                # 组合其它分项损失（HR L1 / LPIPS 等），使用配置中的系数
                coef = self.cfg.get('train', {}).get('loss_coef', {})
                ldif_w = float(coef.get('ldif', 0.0))
                ldis_w = float(coef.get('ldis', 0.0))
                llpips_w = float(coef.get('llpips', 0.0))
                lhr_w = float(coef.get('lhr', 0.0))

                # 计算 HR 重建：将 clean latent 解码为图像，与原图像进行 L1（作为重建占位）
                try:
                    y_r_img = self.adapter_student._decode_from_latents(
                        self.adapter_student._encode_to_latents(images)  # clean latent from images
                    )
                except Exception:
                    y_r_img = images
                lhr = torch.abs(y_r_img - images).mean()

                # 计算 LPIPS（若可用），输入期望 [-1,1]
                if self.lpips_fn is not None:
                    def to_m1_p1(x):
                        return x.clamp(0, 1) * 2.0 - 1.0
                    llp = self.lpips_fn(to_m1_p1(y_r_img), to_m1_p1(images)).mean()
                else:
                    llp = torch.zeros((), device=self.device)

                # 差分项 ldif：学生/教师特征差的 L1（使用与 MMD 相同的时间采样）
                try:
                    B = images.shape[0]
                    t_s, r_s, s_s = self.loss_fn.sample_trs(B, self.adapter_student, device=self.device)
                    f_st2 = self.adapter_student.forward_features(images, t_s, s_s, cond=cond_drop, force_fp32=False, cond_drop=True)
                    f_sr2 = self.adapter_teacher.forward_features(images, r_s, s_s, cond=cond, force_fp32=False, cond_drop=False)
                    ldif = torch.abs(f_st2 - f_sr2).mean()
                    # 判别器项 ldis：使用余弦相似的 hinge 形式（margin=0.8）
                    def _cos_sim(a, b):
                        a_f = a.flatten(1)
                        b_f = b.flatten(1)
                        a_n = torch.nn.functional.normalize(a_f, dim=1)
                        b_n = torch.nn.functional.normalize(b_f, dim=1)
                        return (a_n * b_n).sum(1)
                    margin = 0.8
                    sim = _cos_sim(f_st2, f_sr2)
                    ldis = torch.clamp(margin - sim, min=0.0).mean()
                except Exception:
                    ldif = torch.zeros((), device=self.device)
                    ldis = torch.zeros((), device=self.device)

                # 若配置了真实判别器，则使用对抗损失替代简化 ldis（按你的需求：学生输出 vs HR 图像进行对抗）
                adv_g = torch.zeros((), device=self.device)
                adv_d = torch.zeros((), device=self.device)
                try:
                    if self.dis is not None:
                        # 学生输出（重建）与 HR 图像作为 fake/real
                        # 将图像转为 4 通道以匹配判别器 in_channels=4（RGB + zeros）
                        def _img_to_c4(x: torch.Tensor) -> torch.Tensor:
                            B, C, H, W = x.shape
                            if C == 4:
                                return x
                            if C >= 3:
                                pad = torch.zeros((B, 1, H, W), device=x.device, dtype=x.dtype)
                                return torch.cat([x[:, :3], pad], dim=1)
                            return x.repeat(1, 3, 1, 1)[:, :3].contiguous()

                        y_fake = _img_to_c4(y_r_img)
                        y_real = _img_to_c4(images)

                        # 构造判别器所需的时间步与文本条件
                        Bf = y_real.shape[0]
                        t_for_d = s_s.detach().flatten().to(self.device)
                        try:
                            enc_hid = cond.get('encoder_hidden_states', None) if isinstance(cond, dict) else None
                        except Exception:
                            enc_hid = None
                        if enc_hid is None:
                            try:
                                hidden = getattr(self.sampler.sd_pipe.text_encoder.config, 'hidden_size', 1024)
                            except Exception:
                                hidden = 1024
                            token_len = 77
                            enc_hid = torch.zeros((Bf, token_len, hidden), device=self.device, dtype=y_real.dtype)

                        bce = torch.nn.functional.binary_cross_entropy_with_logits
                        def _as_tensor_list(x):
                            if isinstance(x, (list, tuple)):
                                return [t for t in x if isinstance(t, torch.Tensor)]
                            return [x]

                        # 1) 训练判别器：对 D 打开梯度，仅让样本梯度不回传到生成器（y_fake.detach()）
                        dis_update_freq = int(self.cfg.get('train', {}).get('dis_update_freq', 1))
                        if (it % dis_update_freq) == 0:
                            for p in self.dis.parameters():
                                p.requires_grad_(True)
                            pred_real_d = self.dis(y_real.detach(), timestep=t_for_d, encoder_hidden_states=enc_hid)
                            pred_fake_d = self.dis(y_fake.detach(), timestep=t_for_d, encoder_hidden_states=enc_hid)
                            real_list_d = _as_tensor_list(pred_real_d)
                            fake_list_d = _as_tensor_list(pred_fake_d)
                            d_terms = []
                            for pr in real_list_d:
                                d_terms.append(bce(pr, torch.ones_like(pr)))
                            for pf in fake_list_d:
                                d_terms.append(bce(pf, torch.zeros_like(pf)))
                            if len(d_terms) > 0:
                                adv_d = 0.5 * torch.stack([t.mean() for t in d_terms]).mean()
                            else:
                                adv_d = torch.zeros((), device=self.device)
                            self.optim_dis.zero_grad(set_to_none=True)
                            adv_d.backward()
                            self.optim_dis.step()

                        # 2) 训练生成器：冻结 D 权重，重新前向一次，让梯度只流向 y_fake（进而更新生成器）
                        for p in self.dis.parameters():
                            p.requires_grad_(False)
                        pred_fake_g = self.dis(y_fake, timestep=t_for_d, encoder_hidden_states=enc_hid)
                        fake_list_g = _as_tensor_list(pred_fake_g)
                        g_terms = [bce(pf, torch.ones_like(pf)) for pf in fake_list_g]
                        if len(g_terms) > 0:
                            adv_g = torch.stack([t.mean() for t in g_terms]).mean()
                        else:
                            adv_g = torch.zeros((), device=self.device)
                except Exception as e:
                    print(f"[IMM][WARN] Discriminator step failed: {e}")

                # 将生成器对抗损失按 ldis_w 合并（作为替代/增强项）
                loss = loss_mmd + ldif_w * ldif + ldis_w * (ldis + adv_g) + llpips_w * llp + lhr_w * lhr
                # Finite check and early clamp to avoid propagating NaN/Inf
                if not torch.isfinite(loss).all():
                    # Replace non-finite with zero to allow anomaly detection path later
                    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
                # 记录分项到日志（含权重系数），便于打印与分析
                # 追加训练阶段的 PSNR/SSIM（快速近似）
                try:
                    import torch.nn.functional as F
                    mse_tr = F.mse_loss(y_r_img, images).item()
                    psnr_tr = (20.0 * math.log10(1.0) - 10.0 * math.log10(mse_tr)) if mse_tr > 0 else float('inf')
                    def _ssim_simple_t(x, y, C1=0.01**2, C2=0.03**2):
                        try:
                            mu_x = float(x.mean().item()); mu_y = float(y.mean().item())
                            sigma_x = float(x.var().item()); sigma_y = float(y.var().item())
                            sigma_xy = float(((x - mu_x) * (y - mu_y)).mean().item())
                            num = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2)
                            den = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
                            return float(num / den) if den != 0 else float('nan')
                        except Exception:
                            return float('nan')
                    ssim_tr = _ssim_simple_t(y_r_img, images)
                except Exception:
                    psnr_tr = float('nan'); ssim_tr = float('nan')
                logs.update({
                    'loss_mmd': loss_mmd.detach(),
                    'ldif': ldif.detach(),
                    'ldis': ldis.detach(),
                    'llpips': llp.detach(),
                    'lhr': lhr.detach(),
                    'psnr_train': psnr_tr,
                    'ssim_train': ssim_tr,
                    'adv_g': adv_g.detach() if adv_g.numel() else torch.tensor(0.0, device=self.device),
                    'adv_d': adv_d.detach() if adv_d.numel() else torch.tensor(0.0, device=self.device),
                    'coef_ldif': ldif_w,
                    'coef_ldis': ldis_w,
                    'coef_llpips': llpips_w,
                    'coef_lhr': lhr_w,
                })
                self.optim.zero_grad(set_to_none=True)
                if use_scaler:
                    # Safe scaled backward/step when optimizer params are FP32
                    # If loss becomes non-finite, run anomaly detection to locate source
                    if not torch.isfinite(loss).all():
                        with torch.autograd.detect_anomaly():
                            self.scaler.scale(loss).backward()
                    else:
                        self.scaler.scale(loss).backward()
                    try:
                        self.scaler.step(self.optim)
                        self.scaler.update()
                    except torch.cuda.OutOfMemoryError as e:
                        print(f"[IMM][WARN] OOM during scaler.step: {e}. Skipping optimizer step for this iteration.")
                        # Free cache and skip the rest of this iteration to avoid double unscale/update
                        torch.cuda.empty_cache()
                        self.optim.zero_grad(set_to_none=True)
                        # Skip logging/EMA with invalid step to keep state consistent
                        continue
                    except Exception as e:
                        # Non-OOM failure: skip this step without calling unscale_ twice
                        print(f"[IMM][WARN] Optimizer step failed: {e}. Skipping this iteration.")
                        self.optim.zero_grad(set_to_none=True)
                        continue
                else:
                    # Fallback: no grad scaler (e.g., model already in fp16). Use standard backward/step.
                    if not torch.isfinite(loss).all():
                        with torch.autograd.detect_anomaly():
                            loss.backward()
                    else:
                        loss.backward()
                    self.optim.step()
                # 计算梯度范数，辅助判断是否梯度为 0
                # Guard grad norm computation with try/except
                try:
                    grad_norm = _grad_global_norm(self.adapter_student.unet)
                    if not math.isfinite(grad_norm):
                        grad_norm = float('nan')
                except Exception:
                    grad_norm = float('nan')
                # 执行优化器 step
                self._ema_update()

                if local_logging and is_main and (it == 1 or (it % log_every == 0)):
                    dt = time.time() - t0
                    steps_per_sec = it / dt if dt > 0 else 0.0
                    remain = total_steps - it
                    eta_sec = remain / steps_per_sec if steps_per_sec > 0 else float('inf')
                    eta_min = eta_sec / 60.0
                    loss_val = float(getattr(loss, 'item', lambda: loss)()) if hasattr(loss, 'item') else float(loss)
                    # 读取学习率
                    try:
                        lr = float(self.optim.param_groups[0].get('lr', 0.0))
                    except Exception:
                        lr = 0.0
                    # 打印各分项损失（取均值）以及权重系数
                    extra_msg = ''
                    try:
                        if isinstance(logs, dict) and len(logs) > 0:
                            def _val(x):
                                if isinstance(x, torch.Tensor):
                                    return float(x.mean().item())
                                return float(x)
                            mmd_v = _val(logs.get('loss_mmd', 0.0))
                            ldif_v = _val(logs.get('ldif', 0.0))
                            ldis_v = _val(logs.get('ldis', 0.0))
                            llp_v = _val(logs.get('llpips', 0.0))
                            lhr_v = _val(logs.get('lhr', 0.0))
                            c_ldif = _val(logs.get('coef_ldif', 0.0))
                            c_ldis = _val(logs.get('coef_ldis', 0.0))
                            c_llp = _val(logs.get('coef_llpips', 0.0))
                            c_lhr = _val(logs.get('coef_lhr', 0.0))
                            # 总合成项表达式
                            total_expr = mmd_v + c_ldif*ldif_v + c_ldis*ldis_v + c_llp*llp_v + c_lhr*lhr_v
                            psnr_v = _val(logs.get('psnr_train', float('nan')))
                            ssim_v = _val(logs.get('ssim_train', float('nan')))
                            extra_msg = (
                                f" | loss_mmd={mmd_v:.4f}, ldif={ldif_v:.4f}*{c_ldif:.2f}, "
                                f"ldis={ldis_v:.4f}*{c_ldis:.2f}, llpips={llp_v:.4f}*{c_llp:.2f}, "
                                f"lhr={lhr_v:.4f}*{c_lhr:.2f} => total={total_expr:.4f}, "
                                f"PSNR={psnr_v:.2f}dB, SSIM={ssim_v:.4f}"
                            )
                    except Exception:
                        pass

                    msg = (
                        f"[IMM][it {it}/{total_steps}] loss={loss_val:.4f} | grad_norm={grad_norm:.4e} | lr={lr:.2e} | "
                        f"speed={steps_per_sec:.2f} it/s | ETA~{eta_min:.1f} min{extra_msg}"
                    )
                    if steps_per_epoch:
                        cur_ep = (it - 1) // steps_per_epoch + 1
                        msg += f" | epoch~{cur_ep}"
                    print(msg, flush=True)
                    if self._log_fp:
                        self._log_fp.write(msg + "\n")

                if is_main and save_every > 0 and (it % save_every == 0):
                    self._save(tag=f'imm_step_{it}')

                # 验证：定期对少量样本做重建并打印/保存结果
                if is_main and val_loader is not None and val_every > 0 and (it % val_every == 0):
                    try:
                        import torchvision
                        import torch.nn.functional as F
                        val_logs = []
                        n_done = 0
                        for vb in val_loader:
                            if n_done >= val_max_batches:
                                break
                            vimg = vb.get('target', vb.get('hr', vb.get('gt')))
                            if vimg is None:
                                continue
                            vimg = vimg.to(self.device)
                            # 重建图像
                            try:
                                v_lat = self.adapter_student._encode_to_latents(vimg)
                                v_rec = self.adapter_student._decode_from_latents(v_lat)
                            except Exception:
                                v_rec = vimg
                            # 指标：L1、PSNR（对[0,1]）、SSIM（简版）以及LPIPS（若可用）
                            l1_v = torch.abs(v_rec - vimg).mean().item()
                            mse = F.mse_loss(v_rec, vimg).item()
                            psnr = (20.0 * math.log10(1.0) - 10.0 * math.log10(mse)) if mse > 0 else float('inf')
                            # 简版 SSIM
                            def _ssim_simple_val(x, y, C1=0.01**2, C2=0.03**2):
                                try:
                                    mu_x = float(x.mean().item()); mu_y = float(y.mean().item())
                                    sigma_x = float(x.var().item()); sigma_y = float(y.var().item())
                                    sigma_xy = float(((x - mu_x) * (y - mu_y)).mean().item())
                                    num = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2)
                                    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
                                    return float(num / den) if den != 0 else float('nan')
                                except Exception:
                                    return float('nan')
                            ssim_v = _ssim_simple_val(v_rec, vimg)
                            if self.lpips_fn is not None:
                                llp_v = float(self.lpips_fn(v_rec * 2 - 1, vimg * 2 - 1).mean().item())
                            else:
                                llp_v = float('nan')
                            val_logs.append((l1_v, psnr, ssim_v, llp_v))
                            # 保存对比图
                            grid = torchvision.utils.make_grid(torch.stack([vimg.clamp(0,1), v_rec.clamp(0,1)], dim=0), nrow=2)
                            out_p = os.path.join(val_out_dir, f"step_{it:06d}_idx_{n_done}.png")
                            torchvision.utils.save_image(grid, out_p)
                            n_done += 1
                        if val_logs:
                            l1_m = sum(x[0] for x in val_logs) / len(val_logs)
                            psnr_m = sum(x[1] for x in val_logs) / len(val_logs)
                            ssim_m = sum(x[2] for x in val_logs if not math.isnan(x[2])) / max(1, sum(1 for x in val_logs if not math.isnan(x[2])))
                            llp_m = sum(x[3] for x in val_logs if not math.isnan(x[3])) / max(1, sum(1 for x in val_logs if not math.isnan(x[3])))
                            vmsg = f"[IMM][val step {it}] L1={l1_m:.4f}, PSNR={psnr_m:.2f}dB, SSIM={ssim_m:.4f}, LPIPS={llp_m:.4f} (over {len(val_logs)} samples)"
                            print(vmsg)
                            if self._log_fp:
                                self._log_fp.write(vmsg + "\n")
                    except Exception as e:
                        print(f"[IMM][WARN] Val failed at step {it}: {e}")

                if it >= total_steps:
                    break

        # 最终保存
        if is_main:
            self._save(tag='imm_last')
        if self._log_fp:
            try:
                self._log_fp.close()
            except Exception:
                pass
