import math
from typing import Tuple, Optional

import torch
import torch.nn as nn


class SDAdapter(nn.Module):
    """
    Minimal adapter that exposes IMM-required interfaces over a diffusers-style UNet+VAE+scheduler.

    Contract:
    - add_noise(x0, t): returns (x_t, noise)
    - ddim(x_t, x0, t, r, noise): single deterministic step from t -> r
    - sample_eta_t(bs, ...): sample nt in sigma-space (or log-sigma)
    - nt_to_t(nt): map sigma (or log-sigma) to scheduler timesteps
    - get_log_nt(t): map scheduler timestep to log-sigma
    - get_kernel_weight(t, s, a, b): IMM kernel weighting
    - forward_features(x, t, s, cond, ...): produce feature for MMD (default: epsilon prediction)
    - cfg_forward(x, t_cur, t_next, cond, cfg_scale): classifier-free guidance forward (optional for sampling)
    """

    # Reuse InvSR's default prompts for CFG when no explicit prompts are given
    _POS_PROMPT = (
        "Cinematic, high-contrast, photo-realistic, 8k, ultra HD, "
        "meticulous detailing, hyper sharpness, perfect without deformations"
    )
    _NEG_PROMPT = (
        "Low quality, blurring, jpeg artifacts, deformed, over-smooth, cartoon, noisy,"
        "painting, drawing, sketch, oil painting"
    )

    def __init__(self, unet, vae=None, scheduler=None, time_embed=None, tokenizer=None, text_encoder=None):
        super().__init__()
        self.unet = unet
        self.vae = vae
        self.scheduler = scheduler
        self.time_embed = time_embed  # module with forward(t) and forward_s(s) to fuse
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder

        # derive sigma range from scheduler if available
        # For DDPM-like schedulers, beta schedule => alphas_cumprod => sigmas
        self._init_sigma_bounds()

    def _init_sigma_bounds(self):
        try:
            # Use scheduler to estimate sigma bounds
            alphas_cumprod = self.scheduler.alphas_cumprod
            sigmas = torch.sqrt(1 - torch.tensor(alphas_cumprod))
            self.nt_low = float(sigmas.min().item())
            self.nt_high = float(sigmas.max().item())
        except Exception:
            # Fallback bounds
            self.nt_low = 0.0
            self.nt_high = 1.0
        # time scalar range (scheduler timesteps typically [0, num_train_timesteps])
        self.T = float(self.scheduler.config.num_train_timesteps) if hasattr(self.scheduler, "config") else 1000.0
        self.eps = 0.0

    # --- helpers: image <-> latent ---
    def _get_vae_scaling(self) -> float:
        sf = getattr(getattr(self.vae, 'config', object()), 'scaling_factor', None)
        if sf is None:
            # common default for SD 1.x/2.x
            return 0.18215
        try:
            return float(sf)
        except Exception:
            return 0.18215

    @torch.no_grad()
    def _encode_to_latents(self, x: torch.Tensor) -> torch.Tensor:
        """Encode RGB images in [0,1] to latent space using VAE; pass latents through if already 4-ch."""
        # check if already latent
        if x.shape[1] == 4:
            return x
        
        # encode
        dtype = x.dtype
        if self.vae:
            self.vae.to(device=x.device) # ensure device
            # vae expects [-1, 1] range if already normalized; but if we assume standard [0,1] or simple tensor?
            # Standard diffusers pipeline expects inputs in [-1, 1].
            # Assume caller normalizes.
            
            # Use deterministic encoding for teacher generation in IMM? Or sample?
            # Typically sample() for variation, mode() for determinism.
            # SD-Turbo is trained with standard VAE sampling.
            
            # Using autocast if available
            with torch.cuda.amp.autocast(enabled=False):
                # VAE is often strict on fp32
                dist = self.vae.encode(x.float()).latent_dist
                latents = dist.mode() # use mode for stability in distillation targets
            
            latents = latents * self.vae.config.scaling_factor
            return latents.to(dtype)
        else:
            raise ValueError("VAE required for encoding images to latents")

    @torch.no_grad()
    def _decode_from_latents(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents to RGB images in [0,1] using VAE; pass through if already 3-ch image.
        Returns BCHW float32 tensor in [0,1].
        """
        z = z / self.vae.config.scaling_factor
        with torch.cuda.amp.autocast(enabled=False):
            image = self.vae.decode(z.float()).sample
        image = torch.clamp(image, -1.0, 1.0) # Assume [-1,1] range
        return image

    # --- sigma<->t mappings ---
    def nt_to_t(self, nt: torch.Tensor) -> torch.Tensor:
        """Map sigma to scheduler timestep via nearest match in sigma schedule."""
        if not hasattr(self.scheduler, "alphas_cumprod"):
            return nt.clone()  # best-effort fallback
        device = nt.device
        alphas_cumprod = torch.tensor(self.scheduler.alphas_cumprod, device=device, dtype=torch.float64)
        sigmas = torch.sqrt(1.0 - alphas_cumprod)
        # find nearest index
        idx = torch.bucketize(nt.to(torch.float64), sigmas, right=False).clamp(min=0, max=len(sigmas)-1)
        return idx.to(torch.float32)

    def get_log_nt(self, t: torch.Tensor) -> torch.Tensor:
        """Map timestep to log-sigma."""
        device = t.device
        if not hasattr(self.scheduler, "alphas_cumprod"):
            return torch.log(torch.clamp(t.to(torch.float64), min=1e-8))
        alphas_cumprod = torch.tensor(self.scheduler.alphas_cumprod, device=device, dtype=torch.float64)
        # clamp t to valid range
        t_idx = t.to(torch.long).clamp(min=0, max=len(alphas_cumprod)-1)
        sigmas = torch.sqrt(1.0 - alphas_cumprod[t_idx])
        return torch.log(torch.clamp(sigmas, min=1e-8))

    # --- sampling nt ---
    def sample_eta_t(self, bs: int, device: torch.device, log_low: float, log_high: float,
                      low: float, high: float, sample_mode: str = "lognormal",
                      P_mean: float = -1.1, P_std: float = 2.0) -> Tuple[torch.Tensor, torch.Tensor]:
        if sample_mode == "lognormal":
            log_nt = torch.randn(bs, device=device, dtype=torch.float64) * P_std + P_mean
            # clamp bounds: support scalar or tensor per-sample bounds
            if torch.is_tensor(log_low):
                min_t = log_low.to(device=device, dtype=log_nt.dtype)
            else:
                min_t = torch.full((bs,), float(log_low), device=device, dtype=log_nt.dtype)
            if torch.is_tensor(log_high):
                # If log_high has more than 1 element, clamp per-sample using tensor max bound
                max_t = log_high.to(device=device, dtype=log_nt.dtype)
            else:
                max_t = torch.full((bs,), float(log_high), device=device, dtype=log_nt.dtype)
            log_nt = torch.clamp(log_nt, min=min_t, max=max_t)
            nt = torch.exp(log_nt)
        elif sample_mode == "uniform":
            nt = torch.rand(bs, device=device, dtype=torch.float64) * (high - low) + low
            log_nt = torch.log(torch.clamp(nt, min=1e-8))
        else:
            # default to uniform
            nt = torch.rand(bs, device=device, dtype=torch.float64) * (high - low) + low
            log_nt = torch.log(torch.clamp(nt, min=1e-8))
        return nt.to(torch.float64), log_nt.to(torch.float64)

    # --- noise ops ---
    def add_noise(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise at timestep t using scheduler alphas in latent space."""
        # ensure latent space
        x0_latent = self._encode_to_latents(x0)
        device = x0_latent.device
        # convert t to indices
        t_idx = t.to(torch.long).clamp(min=0, max=self.scheduler.config.num_train_timesteps-1)
        alphas_cumprod = torch.tensor(self.scheduler.alphas_cumprod, device=device, dtype=x0_latent.dtype)
        alpha_t = alphas_cumprod[t_idx].view(-1, 1, 1, 1)
        sigma_t = torch.sqrt(1.0 - alpha_t)
        noise = torch.randn_like(x0_latent)
        x_t = torch.sqrt(alpha_t) * x0_latent + sigma_t * noise
        return x_t, noise

    @torch.no_grad()
    def ddim(self, x_t: torch.Tensor, x0: torch.Tensor, t: torch.Tensor, r: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Single deterministic DDIM step from t -> r using clean latent x0.
        Accepts x0 as image (B,3,*) or latent (B,4,*); converts to latent if needed.
        """
        # ensure clean x0 latent
        x0_latent = self._encode_to_latents(x0)
        device = x_t.device
        t_idx = t.to(torch.long).clamp(min=0, max=self.scheduler.config.num_train_timesteps-1)
        r_idx = r.to(torch.long).clamp(min=0, max=self.scheduler.config.num_train_timesteps-1)
        alphas = torch.tensor(self.scheduler.alphas_cumprod, device=device, dtype=x_t.dtype)
        alpha_t = alphas[t_idx].view(-1, 1, 1, 1)
        alpha_r = alphas[r_idx].view(-1, 1, 1, 1)
        sigma_t = torch.sqrt(1.0 - alpha_t)
        sigma_r = torch.sqrt(1.0 - alpha_r)
        # Deterministic DDIM update: keep noise direction, adjust magnitude
        eps = (x_t - torch.sqrt(alpha_t) * x0_latent) / (sigma_t + 1e-8)
        x_r = torch.sqrt(alpha_r) * x0_latent + sigma_r * eps
        return x_r

    # --- kernel weights ---
    def get_kernel_weight(self, t: torch.Tensor, s: torch.Tensor, a: float, b: float):
        """Weighting per IMM; simple power-law of ratios as placeholder."""
        # Convert to float64 for stability
        t = t.to(torch.float64)
        s = s.to(torch.float64)
        ratio = torch.clamp(s / torch.clamp(t, min=1e-8), min=1e-8, max=1.0)
        w = (ratio ** a)
        wout = (1.0 + b * (1.0 - ratio))
        return w.to(torch.float32), wout.to(torch.float32)

    # --- features for MMD ---
    def forward_features(self, x: torch.Tensor, t: torch.Tensor, s: Optional[torch.Tensor], cond: Optional[dict] = None,
                         force_fp32: bool = False, cond_drop: bool = False) -> torch.Tensor:
        # Choose input dtype compatible with UNet weights; avoid forcing FP32 when UNet is FP16
        unet_dtype = getattr(self.unet, 'dtype', torch.float16)
        # Prefer稳定性：训练阶段强制使用 FP32 以避免 NaN
        dtype = torch.float32 if force_fp32 else (unet_dtype if unet_dtype != torch.float32 else torch.float32)
        # ensure latent input to UNet
        x_latent = self._encode_to_latents(x).to(dtype)
        # build UNet input: typically (latents, timestep, encoder_hidden_states)
        # time embedding fusion (t+s)
        timesteps = t.to(torch.float32)
        if s is not None:
            timesteps = timesteps  # we rely on external time_embed to fuse if provided
        # Encode prompts for CFG using InvSR-style positive/negative prompts by default
        bs = x_latent.shape[0]
        device = x_latent.device
        pos_prompts = None
        neg_prompts = None
        if isinstance(cond, dict):
            pos_prompts = cond.get('prompt', None)
            neg_prompts = cond.get('negative_prompt', None)
        # Fallback to InvSR defaults if not provided
        if pos_prompts is None:
            pos_prompts = [self._POS_PROMPT] * bs
        elif isinstance(pos_prompts, str):
            pos_prompts = [pos_prompts] * bs
        if neg_prompts is None:
            neg_prompts = [self._NEG_PROMPT] * bs
        elif isinstance(neg_prompts, str):
            neg_prompts = [neg_prompts] * bs

        def _encode_text(prompts_list):
            if self.tokenizer is None or self.text_encoder is None:
                # no text encoder available: return zeros as placeholder
                expected_dim = getattr(getattr(self.unet, 'config', object()), 'cross_attention_dim', 1024)
                seq_len = getattr(self.tokenizer, 'model_max_length', 77) if self.tokenizer is not None else 77
                return torch.zeros((bs, seq_len, expected_dim), device=device, dtype=dtype)
            with torch.no_grad():
                tokens = self.tokenizer(
                    prompts_list,
                    padding='max_length',
                    max_length=getattr(self.tokenizer, 'model_max_length', 77),
                    truncation=True,
                    return_tensors='pt',
                )
                tokens = {k: v.to(device) for k, v in tokens.items()}
                te_out = self.text_encoder(**tokens)
                if hasattr(te_out, 'last_hidden_state'):
                    return te_out.last_hidden_state.to(dtype)
                else:
                    return te_out[0].to(dtype)

        # Build conditional and unconditional (negative) embeddings
        eh_pos = _encode_text(pos_prompts)
        eh_neg = _encode_text(neg_prompts)

        # Optional dropout on positive embeddings to simulate label drop
        if cond_drop and isinstance(eh_pos, torch.Tensor):
            drop_mask = torch.rand(eh_pos.shape[0], device=eh_pos.device) < 0.1
            eh_pos[drop_mask] = 0.0

        # Guidance scale: if provided in cond use it; otherwise default 1.0
        cfg_scale = 1.0
        if isinstance(cond, dict) and 'cfg_scale' in cond and cond['cfg_scale'] is not None:
            try:
                cfg_scale = float(cond['cfg_scale'])
            except Exception:
                cfg_scale = 1.0

        # epsilon prediction as feature proxy with CFG (neg + scale*(pos - neg))
        # Optimize: if cfg_scale == 1.0, only one forward is needed (equals eps_pos)
        if abs(cfg_scale - 1.0) < 1e-6:
            eps_pred = self.unet(x_latent, timestep=timesteps.to(dtype), encoder_hidden_states=eh_pos).sample
        else:
            eps_u = self.unet(x_latent, timestep=timesteps.to(dtype), encoder_hidden_states=eh_neg).sample
            eps_c = self.unet(x_latent, timestep=timesteps.to(dtype), encoder_hidden_states=eh_pos).sample
            eps_pred = eps_u + cfg_scale * (eps_c - eps_u)
        return eps_pred

    @torch.no_grad()
    def cfg_forward(self, x: torch.Tensor, t_cur: torch.Tensor, t_next: torch.Tensor, cond: Optional[dict], cfg_scale: Optional[float]):
        """Classifier-free guidance forward step: x_next from x at t_cur to t_next."""
        # Unconditional
        eps_u = self.unet(x, timestep=t_cur.to(torch.float32), encoder_hidden_states=None).sample
        # Conditional
        eps_c = eps_u  # keep unconditional (no text states)
        scale = cfg_scale or 1.0
        eps = eps_u + scale * (eps_c - eps_u)
        # DDIM-like deterministic update to t_next
        device = x.device
        t_idx = t_cur.to(torch.long).clamp(min=0, max=self.scheduler.config.num_train_timesteps-1)
        n_idx = t_next.to(torch.long).clamp(min=0, max=self.scheduler.config.num_train_timesteps-1)
        alphas = torch.tensor(self.scheduler.alphas_cumprod, device=device, dtype=x.dtype)
        alpha_t = alphas[t_idx].view(-1, 1, 1, 1)
        alpha_n = alphas[n_idx].view(-1, 1, 1, 1)
        sigma_t = torch.sqrt(1.0 - alpha_t)
        sigma_n = torch.sqrt(1.0 - alpha_n)
        x0 = (x - sigma_t * eps) / (torch.sqrt(alpha_t) + 1e-8)
        x_next = torch.sqrt(alpha_n) * x0 + sigma_n * eps
        return x_next

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, model_output: torch.Tensor) -> torch.Tensor:
        """
        Predict x0 from x_t and model output (assumed epsilon by default for SD-Turbo unless v-pred).
        """
        device = x_t.device
        t_idx = t.to(torch.long).clamp(min=0, max=self.scheduler.config.num_train_timesteps-1)
        alphas_cumprod = torch.tensor(self.scheduler.alphas_cumprod, device=device, dtype=model_output.dtype)
        alpha_t = alphas_cumprod[t_idx].view(-1, 1, 1, 1)
        sigma_t = torch.sqrt(1.0 - alpha_t)
        
        # SD-Turbo typically uses epsilon prediction. x0 = (x_t - sigma * eps) / sqrt(alpha)
        # However, check scheduler config if using v-prediction.
        # Assuming epsilon for now as standard SD backbone.
        
        pred_original_sample = (x_t - sigma_t * model_output) / torch.sqrt(alpha_t)
        return pred_original_sample

    @torch.no_grad()
    def get_text_embeddings(self, prompts_list: list) -> torch.Tensor:
        """Helper to encode text prompts to embeddings."""
        dtype = getattr(self.unet, 'dtype', torch.float16)
        if dtype != torch.float32:
            dtype = torch.float32  # Prefer fp32 for text encoder outputs if checking stability, but match adapter logic
            # Actually adapter.forward_features tries to match unet dtype or force fp32.
            # Let's default to fp32 or unet dtype.
            dtype = getattr(self.unet, 'dtype', torch.float16)

        bs = len(prompts_list)
        device = self.unet.device if hasattr(self.unet, 'device') else torch.device('cuda')

        if self.tokenizer is None or self.text_encoder is None:
             # Placeholder
             expected_dim = getattr(getattr(self.unet, 'config', object()), 'cross_attention_dim', 1024)
             seq_len = getattr(self.tokenizer, 'model_max_length', 77) if self.tokenizer is not None else 77
             return torch.zeros((bs, seq_len, expected_dim), device=device, dtype=dtype)
        
        tokens = self.tokenizer(
            prompts_list,
            padding='max_length',
            max_length=getattr(self.tokenizer, 'model_max_length', 77),
            truncation=True,
            return_tensors='pt',
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        te_out = self.text_encoder(**tokens)
        if hasattr(te_out, 'last_hidden_state'):
            out = te_out.last_hidden_state
        else:
            out = te_out[0]
        return out.to(dtype)
