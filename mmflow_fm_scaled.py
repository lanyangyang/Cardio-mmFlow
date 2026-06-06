import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Union, List


from flow_transformer_no_cross import DirectFlowTransformer
from base_solver import Solver, ConditionTypes

class mmflowWrapper(nn.Module):
    def __init__(self, dit_backbone):
        super().__init__()
        self.dit = dit_backbone

    def forward(self, x, t=None, log_snr=None, null_indicator=None, meta=None, **kwargs):
        if null_indicator is not None and null_indicator.all():
            actual_meta = None
        else:
            actual_meta = meta
        
        if t is not None:
            velocity = self.dit(x, t, meta=actual_meta)
            return (None, velocity)
        
        if t is None and log_snr is not None:
            t = (4.0 - log_snr) / 8.0
            t = torch.clamp(t, 0.0, 1.0)
            velocity = self.dit(x, t, meta=actual_meta)
            return (None, velocity)
            
        raise ValueError("mmflowWrapper: input must contain either 't' or 'log_snr'")


class mmflowFMScaled(Solver, nn.Module):
    def __init__(
        self,
        dit_backbone: DirectFlowTransformer,
        guidance_scale: float = 1.0, 
        cfg_rate: float = 0.1,
        meta_mask_rate: float = 0.1,       
        sigma_min: float = 1e-5,
        sigma_max: float = 1.0,
        loss_type: str = 'mse',
        **kwargs
    ):
        nn.Module.__init__(self)
        
        self.model_wrapper = mmflowWrapper(dit_backbone)
        
        Solver.__init__(
            self,
            model_fn=self.model_wrapper, 
            guidance_scale=guidance_scale,
            conditioning_types=["caption"], 
            **kwargs
        )

        self.dit = dit_backbone
        self.cfg_rate = cfg_rate
        self.meta_mask_rate = meta_mask_rate
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.loss_type = loss_type.lower()
        
        if self.loss_type not in ['mse', 'l1']:
            raise ValueError(f"loss_type must be 'mse' or 'l1', got '{loss_type}'")
        
        self.scale_coeff = (sigma_min / sigma_max) - 1.0
        print(f"[mmflowFMScaled] sigma_min={sigma_min}, sigma_max={sigma_max}")
        print(f"[mmflowFMScaled] scale_coeff={(self.scale_coeff):.6f}")
        
        self.use_metadata = dit_backbone.use_metadata

    @property
    def device(self):
        return next(self.parameters()).device

    def psi(self, t, x0, x1):
        original_dtype = x0.dtype
        device_type = x0.device.type
        
        with torch.autocast(device_type=device_type, enabled=False):
            t = t.float().view(-1, 1, 1)
            x0_fp32 = x0.float()
            x1_fp32 = x1.float()
            x_t = (1 - t) * x0_fp32 + t * x1_fp32
        
        return x_t.to(original_dtype)

    def forward(self, x0, x1, meta=None, t=None):
        B = x0.shape[0]
        device = x0.device

        x0_train = x0
        meta_train = meta.clone() if meta is not None else None
        
        cfg_drop_mask = None
        if self.cfg_rate > 0 and self.training and meta is not None:
            cfg_drop_mask = torch.rand(B, device=device) < self.cfg_rate
            drop_indices = cfg_drop_mask.nonzero(as_tuple=True)[0]
            if len(drop_indices) > 0:
                meta_train[drop_indices] = float('nan')
        
        if self.meta_mask_rate > 0 and self.training and meta_train is not None:
            meta_mask = torch.rand(B, 5, device=device) < self.meta_mask_rate
            if cfg_drop_mask is not None:
                non_cfg_mask = ~cfg_drop_mask.unsqueeze(-1)
                meta_mask = meta_mask & non_cfg_mask
            meta_train = meta_train.masked_fill(meta_mask, 0.0)

        if t is None:
            t = torch.rand(B, device=device)
        
        x_t = self.psi(t, x0, x1)
        
        v_target = self.scale_coeff * x0 + x1
        
        _, v_pred = self.model(x_t, t=t, meta=meta_train)
        
        if torch.isnan(v_pred).any() or torch.isinf(v_pred).any():
            print("[WARNING] NaN/Inf detected in v_pred, clipping...")
            v_pred = torch.nan_to_num(v_pred, nan=0.0, posinf=10.0, neginf=-10.0)
        
        v_pred = torch.clamp(v_pred, -100.0, 100.0)
        
        if self.loss_type == 'mse':
            fm_loss = F.mse_loss(v_pred, v_target)
        elif self.loss_type == 'l1':
            fm_loss = F.l1_loss(v_pred, v_target)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        if torch.isnan(fm_loss) or torch.isinf(fm_loss):
            print(f"[ERROR] Invalid loss detected! fm_loss={fm_loss.item()}")
            print(f"v_pred stats: min={v_pred.min()}, max={v_pred.max()}, mean={v_pred.mean()}")
            print(f"v_target stats: min={v_target.min()}, max={v_target.max()}, mean={v_target.mean()}")
            fm_loss = torch.tensor(1.0, device=device, requires_grad=True)
        
        x1_pred = v_pred - self.scale_coeff * x0
        
        return fm_loss, x1_pred

    def sample_euler(
        self,
        x_T,
        num_timesteps,
        unconditional_guidance_scale,
        has_null_indicator,
        **kwargs
    ):
        if num_timesteps is None:
            if hasattr(self, 'num_inf_timesteps') and self.num_inf_timesteps is not None:
                num_timesteps = self.num_inf_timesteps
            else:
                num_timesteps = 50

        B = x_T.shape[0]
        device = x_T.device
        x = x_T
        
        meta = kwargs.get('meta', None)
        use_random_t = kwargs.get('use_random_t', False)
        
        if use_random_t:
            t_sequence = torch.sort(torch.rand(num_timesteps, device=device))[0]
            t_sequence = torch.cat([
                torch.tensor([0.0], device=device), 
                t_sequence[1:-1], 
                torch.tensor([1.0], device=device)
            ])
        else:
            t_sequence = torch.linspace(0, 1, num_timesteps + 1, device=device)[:-1]
        
        for i in range(num_timesteps):
            t_float = t_sequence[i].item()
            t_tensor = torch.full((B,), t_float, device=device, dtype=x.dtype)
            
            if i < num_timesteps - 1:
                dt = t_sequence[i+1] - t_sequence[i]
            else:
                dt = 1.0 - t_sequence[i]
            
            velocity = self.get_model_output_dimr(
                x, 
                t_continuous=t_tensor, 
                unconditional_guidance_scale=unconditional_guidance_scale,
                has_null_indicator=has_null_indicator,
                meta=meta
            )
            
            if torch.isnan(velocity).any() or torch.isinf(velocity).any():
                print(f"[WARNING] Invalid velocity at step {i}, using zero velocity")
                velocity = torch.zeros_like(velocity)
            
            velocity = torch.clamp(velocity, -100.0, 100.0)
            
            x = x + velocity * dt
            
            x = torch.clamp(x, -50.0, 50.0)
            
            if torch.isnan(x).any():
                print(f"[ERROR] NaN detected in latent at step {i}")
                x = torch.nan_to_num(x, nan=0.0)
            
        return x, None

    def inference(self, x0_source, steps=50, meta=None, use_random_t=False):
        B = x0_source.shape[0]
        self.unconditional_guidance_scale = 1.0
        self.num_inf_timesteps = steps

        samples, _ = self.sample(
            sample_steps=steps,
            batch_size=B,
            sampling_method=self.sample_euler,
            unconditional_guidance_scale=1.0,
            has_null_indicator=False,
            x_T=x0_source,
            t_schedule="time_uniform",
            num_timesteps=steps,
            meta=meta,
            use_random_t=use_random_t
        )
        return samples
