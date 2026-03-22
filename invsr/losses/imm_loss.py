import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


class IMMLoss(nn.Module):
    def __init__(self, sigma=1.0, sample_t_mode="lognormal", P_mean=-1.1, P_std=2.0,
                 matrix_size=4, sample_repeat=1, label_dropout=0.1, k=12, a=2, b=4, min_tr_gap=None):
        super().__init__()
        self.P_mean = P_mean
        self.P_std = P_std
        self.min_tr_gap = min_tr_gap
        self.sigma = sigma
        self.sample_t_mode = sample_t_mode
        self.matrix_size = matrix_size
        self.sample_repeat = sample_repeat
        self.label_dropout = label_dropout
        self.k = k
        self.a = a
        self.b = b

    def nt_to_nr(self, nt: torch.Tensor, adapter) -> torch.Tensor:
        u = (adapter.nt_high - adapter.nt_low) * (0.5 ** self.k)
        nr = torch.clamp(nt - u, min=adapter.nt_low, max=adapter.nt_high)
        return nr

    def kernel_fn(self, x, y, flatten_dim, w):
        # compute in float32 to avoid fp16 underflow/overflow
        x32 = x.to(torch.float32)
        y32 = y.to(torch.float32)
        denom = float(np.prod(y32.shape[flatten_dim:]))
        denom = denom if denom > 0 else 1.0
        sigma = float(self.sigma) if self.sigma is not None else 1.0
        sigma = max(sigma, 1e-6)
        dist = torch.clamp_min(((x32 - y32) ** 2).flatten(flatten_dim).sum(-1), 1e-12).sqrt() / denom / sigma
        expo = -dist * w
        expo = torch.clamp(expo, min=-1e6, max=0.0)
        ret = torch.exp(expo)
        # 数值稳定处理：替换 NaN/Inf
        ret = torch.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)
        return ret

    def kernel(self, x, y, w=None):
        x = x.unsqueeze(2)
        y = y.unsqueeze(1)
        if w is None:
            w = 1
        else:
            w = w[:, None, None]
        ret = self.kernel_fn(x, y, flatten_dim=3, w=w)
        return ret

    def sample_trs(self, t_bs: int, adapter, device):
        high = adapter.nt_high
        low = adapter.nt_low

        nt, log_nt = adapter.sample_eta_t(t_bs, device, log_low=np.log(low), log_high=np.log(high), low=low, high=high,
                                          sample_mode=self.sample_t_mode, P_mean=self.P_mean, P_std=self.P_std)
        ns_upper = nt; logns_upper = log_nt
        ns, log_ns = adapter.sample_eta_t(t_bs, device, log_low=np.log(low), log_high=logns_upper,
                                          low=low, high=ns_upper, sample_mode=self.sample_t_mode,
                                          P_mean=self.P_mean, P_std=self.P_std)
        ns = torch.minimum(ns, nt).clamp(min=low)
        nr = self.nt_to_nr(nt, adapter)
        t = adapter.nt_to_t(nt)
        r = adapter.nt_to_t(nr)
        s = adapter.nt_to_t(ns)
        if self.min_tr_gap is not None:
            max_r = torch.clamp(t - self.min_tr_gap, min=adapter.nt_low)
            r = torch.minimum(r, max_r)
        r = torch.maximum(r, s).clamp(min=adapter.eps)
        return t, r, s

    def _build_cond(self, cond_raw, drop_prob=0.1):
        cond = cond_raw.copy() if isinstance(cond_raw, dict) else {}
        if cond.get("encoder_hidden_states", None) is not None:
            eh = cond["encoder_hidden_states"]
            mask = (torch.rand(eh.shape[0], device=eh.device) < drop_prob)
            eh[mask] = 0.0
            cond["encoder_hidden_states"] = eh
        return cond

    def compute_loss(self, f_st, f_sr, t, s, adapter, return_logs=False):
        """
        Computes IMM loss given features and time steps.
        Args:
            f_st: Student features at time s (from input t). Shape (B, M, ...) or (B, ...).
            f_sr: Teacher features at time s (from input r). Shape (B, M, ...) or (B, ...).
            t: Time steps t. sorted/arranged matching features. Shape (B,).
            s: Time steps s. sorted/arranged matching features. Shape (B,).
            adapter: The student adapter (for weight calculation).
        """
        # Ensure features are (B, M, ...)
        if f_st.ndim == 4: # (B, C, H, W) -> (B, 1, C, H, W)
            f_st = f_st.unsqueeze(1)
        if f_sr.ndim == 4:
            f_sr = f_sr.unsqueeze(1)
        
        # M is second dim
        if f_st.shape[1] != self.matrix_size and self.matrix_size > 1:
             # This happens if we manually pass non-matricized batches.
             # We treat them as independent samples (matrix_size=1 effectively)
             pass 

        wt, wtout = adapter.get_kernel_weight(t, s, a=self.a, b=self.b)
        
        inter_sample = torch.nan_to_num(self.kernel(f_st, f_st, w=wt).mean((1, 2)), nan=0.0)
        inter_gt = torch.nan_to_num(self.kernel(f_sr, f_sr, w=wt).mean((1, 2)), nan=0.0)
        cross = torch.nan_to_num(self.kernel(f_st, f_sr, w=wt).mean((1, 2)), nan=0.0)
        
        loss_raw = torch.nan_to_num(inter_sample + inter_gt - 2 * cross, nan=0.0)
        loss = loss_raw
        if wtout is not None:
            loss = wtout * loss
        loss = torch.nan_to_num(loss, nan=0.0)
        
        if return_logs:
            logs = {
                "inter_sample_sim": inter_sample.detach().mean(),
                "inter_gt_sim": inter_gt.detach().mean(),
                "cross_sim": cross.detach().mean(),
                "loss": loss.detach().mean(),
                "wt_mean": wt.mean().detach(),
                "wtout_mean": wtout.mean().detach() if wtout is not None else torch.tensor(float('nan'), device=wt.device),
                "loss_raw": loss_raw.detach().mean(),
            }
            return loss.mean(), logs
            
        return loss.mean()

    def forward(self, arg1, arg2, arg3=None, arg4=None, arg5=None, **kwargs):
        """
        Flexible forward that dispatches to compute_loss (if inputs are tensors) 
        or _forward_pipeline (if inputs are adapters/images).
        Arguments are generic to support both signatures:
        1. compute_loss(f_st, f_sr, t, s, adapter, ...)
        2. _forward_pipeline(adapter_student, adapter_teacher, images, cond, device, ...)
        """
        # Heuristic: if arg1 and arg2 are tensors, we are likely doing compute_loss
        if isinstance(arg1, torch.Tensor) and isinstance(arg2, torch.Tensor):
            return self.compute_loss(arg1, arg2, arg3, arg4, arg5, **kwargs)
        
        # Otherwise assume we are in pipeline mode
        return self._forward_pipeline(arg1, arg2, arg3, arg4, arg5, **kwargs)

    def _forward_pipeline(self, adapter_student, adapter_teacher, images, cond=None, device=None):
        if images is None:
             raise ValueError("IMMLoss pipeline forward called but 'images' is None.")
        
        device = device or images.device
        # 保证每个 batch 的大小是 matrix_size 的整数倍，否则会导致时间步 t 的长度与图像 batch 不匹配
        batch_size = int(images.shape[0])
        if self.matrix_size <= 0:
            raise ValueError(f"matrix_size must be >= 1, got {self.matrix_size}")
        if batch_size < self.matrix_size or (batch_size % self.matrix_size != 0):
            raise RuntimeError(
                f"当前 batch_size={batch_size} 与 matrix_size={self.matrix_size} 不兼容。"
                f"请将 loss.matrix_size 设为不大于 batch_size 的值，且 batch_size % matrix_size == 0；"
                f"或在训练配置中将 batch 提高到 matrix_size 的整数倍。"
            )
        B = batch_size // self.matrix_size
        t, r, s = self.sample_trs(B, adapter_student, device=device)
        t = t.repeat_interleave(self.matrix_size, dim=0)
        r = r.repeat_interleave(self.matrix_size, dim=0)
        s = s.repeat_interleave(self.matrix_size, dim=0)

        y = images
        y_t, noise_t = adapter_student.add_noise(y, t)
        y_r = adapter_student.ddim(y_t, y, t, r, noise_t)

        cond_drop = self._build_cond(cond or {}, drop_prob=self.label_dropout)
        # 使用与 UNet 相同的 dtype 以避免输入/权重 dtype 不匹配
        f_st = adapter_student.forward_features(y_t, t, s, cond=cond_drop, force_fp32=False, cond_drop=True)
        
        # Move teacher to GPU for forward pass
        # Teacher is already on GPU
        with torch.no_grad():
            # Use teacher in its native dtype to avoid FP32/F16 dtype mismatch inside UNet time embedding
            f_sr = adapter_teacher.forward_features(y_r, r, s, cond=cond, force_fp32=False, cond_drop=False)

        f_st = rearrange(f_st, "(b m) ... -> b m ...", m=self.matrix_size)
        f_sr = rearrange(f_sr, "(b m) ... -> b m ...", m=self.matrix_size)
        t_b = rearrange(t, "(b m) ... -> b m ...", m=self.matrix_size)[:, 0].flatten()
        s_b = rearrange(s, "(b m) ... -> b m ...", m=self.matrix_size)[:, 0].flatten()

        return self.compute_loss(f_st, f_sr, t_b, s_b, adapter_student, return_logs=True)
