from __future__ import annotations

import argparse
import os
import random
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from mmflow_dataset import build_mmflow_dataloaders
from mmflow_fm_scaled import mmflowFMScaled
from flow_transformer_no_cross import DirectFlowTransformer
from resnet_vae import SignalVAE



class LatentProjector(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(hidden_dim, in_channels, kernel_size=1),
        )
        
    def forward(self, x):
        residual = self.proj(x)
        x = x + residual
        return x 


def compute_wasserstein_distance(source: torch.Tensor, target: torch.Tensor, eps: float = 0.1, max_iter: int = 50) -> torch.Tensor:

    B, C, L = source.shape
    if B < 2:
        return torch.tensor(0.0, device=source.device)

    x = source.reshape(B, -1)
    y = target.reshape(B, -1)
    
    x_col = x.unsqueeze(1)  # [B, 1, D]
    y_lin = y.unsqueeze(0)  # [1, B, D]
    M = torch.sum((x_col - y_lin) ** 2, dim=-1)  # [B, B]

    M = M / (M.max() + 1e-8)
    mu = torch.empty(B, device=source.device).fill_(1.0 / B)
    nu = torch.empty(B, device=source.device).fill_(1.0 / B)

    K = torch.exp(-M / eps)
    
    u = torch.ones_like(mu)
    for _ in range(max_iter):
        v = nu / (torch.matmul(K.t(), u) + 1e-8)
        u = mu / (torch.matmul(K, v) + 1e-8)

    wasserstein_loss = torch.sum(u * torch.matmul(K * M, v))
    
    return wasserstein_loss


def compute_kl_divergence(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    
    B, C, L = source.shape
    if B < 2:
        return torch.tensor(0.0, device=source.device)
    
    source_flat = source.reshape(B, -1)
    target_flat = target.reshape(B, -1)
    
    source_mean = source_flat.mean(dim=0, keepdim=True)
    source_var = source_flat.var(dim=0, keepdim=True, unbiased=False) + 1e-8
    
    target_mean = target_flat.mean(dim=0, keepdim=True)
    target_var = target_flat.var(dim=0, keepdim=True, unbiased=False) + 1e-8
    
    kl = 0.5 * (
        torch.log(target_var / source_var) + 
        (source_var + (source_mean - target_mean) ** 2) / target_var - 
        1.0
    )

    kl_loss = kl.sum() / (C * L)
    
    return kl_loss

class LogitNormalSampler:

    def __init__(self, mean: float = 0.0, std: float = 1.0):

        self.mean = mean
        self.std = std
        print(f"[LogitNormalSampler] mean={mean}, std={std}")
    
    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:

        z = torch.randn(batch_size, device=device) * self.std + self.mean
        t = torch.sigmoid(z)
        
        return t


def spectral_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

    original_dtype = pred.dtype
    pred_fp32 = pred.float()
    target_fp32 = target.float()
    
    pred_fft = torch.fft.rfft(pred_fp32, dim=-1)
    target_fft = torch.fft.rfft(target_fp32, dim=-1)
    
    pred_mag = torch.abs(pred_fft)
    target_mag = torch.abs(target_fft)

    scale_factor = target_mag.max() + 1e-8
    pred_mag_norm = pred_mag / scale_factor
    target_mag_norm = target_mag / scale_factor

    mag_loss = F.mse_loss(pred_mag_norm, target_mag_norm)

    pred_phase = torch.angle(pred_fft)
    target_phase = torch.angle(target_fft)
    phase_loss = 1.0 - F.cosine_similarity(pred_phase.flatten(1), target_phase.flatten(1), dim=1).mean()
    
    return (mag_loss + 0.5 * phase_loss).to(original_dtype)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_vae(
    checkpoint_path: str,
    device: torch.device,
    in_channels: int = 1,
    base_channels: int = 64,
    ch_mult: tuple = (1, 2, 4, 4),
    latent_dim: int = 4,
    name: str = "VAE",
) -> SignalVAE:
    """Load pre-trained VAE from checkpoint."""
    vae = SignalVAE(
        in_channels=in_channels,
        base_channels=base_channels,
        ch_mult=ch_mult,
        latent_dim=latent_dim,
        dropout=0.0,
    ).to(device)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            vae.load_state_dict(checkpoint['model_state_dict'])
        else:
            vae.load_state_dict(checkpoint)
        print(f"✓ Loaded {name} from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"{name} checkpoint not found: {checkpoint_path}")
    
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    
    return vae


def encode_batch(vae: SignalVAE, signals: torch.Tensor, device: torch.device, use_sampling: bool = False, ns: float = 1.0) -> torch.Tensor:
    signals = signals.to(device)
    
    with torch.no_grad():
        mean, logvar = vae.encode(signals)

    if use_sampling and vae.training and ns > 0:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + ns * eps * std
    else:
        return mean

@torch.no_grad()
def decode_batch(vae: SignalVAE, latents: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Decode latents to signal space using VAE decoder."""
    latents = latents.to(device)
    return vae.decode(latents)


def train_epoch(
    model: mmflowFMScaled,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,  
    ecg_vae: SignalVAE,
    radar_vae: SignalVAE,
    device: torch.device,
    epoch: int,
    use_metadata: bool,
    amp_enabled: bool,
    grad_clip: float,
    log_interval: int,
    ns: float = 1.0,
    freq_ode_steps: int = 10,  
    t_sampler: Optional[LogitNormalSampler] = None,  
    ecg_projector: Optional[LatentProjector] = None, 
    wasserstein_weight: float = 0.01,  
    kl_weight: float = 0.01,  
    mmd_weight: float = 0.01,  
    mmd_kernel: str = 'simple',  
) -> float:
    """Train for one epoch."""
    model.train()
    if ecg_projector is not None:
        ecg_projector.train()
    total_loss = 0.0
    total_fm_loss = 0.0
    total_recon_loss = 0.0
    total_freq_loss = 0.0
    total_wass_loss = 0.0
    total_kl_loss = 0.0
    total_mmd_loss = 0.0
    batch_count = 0
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=True)
    
    for batch_idx, batch_data in enumerate(pbar):
        if use_metadata:
            radar, ecg, meta = batch_data
            meta = meta.to(device, non_blocking=True)
        else:
            radar, ecg, _ = batch_data
            meta = None
        
        radar = radar.to(device, non_blocking=True)
        ecg = ecg.to(device, non_blocking=True)

        x0_latent = encode_batch(ecg_vae, ecg, device, use_sampling=True, ns=ns)
        x1_latent = encode_batch(radar_vae, radar, device, use_sampling=False, ns=0.0)

        if ecg_projector is not None:
            x0_latent_proj = ecg_projector(x0_latent)
            x1_latent_target = x1_latent
        else:
            x0_latent_proj = x0_latent
            x1_latent_target = x1_latent
        
        B = x0_latent.shape[0]
        optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast(device_type=autocast_device, enabled=amp_enabled):

            if ecg_projector is not None:
                if wasserstein_weight > 0:
                    wass_loss = compute_wasserstein_distance(x0_latent_proj, x1_latent_target, eps=0.5)
                else:
                    wass_loss = torch.tensor(0.0, device=device)
                
                if kl_weight > 0:
                    kl_loss = compute_kl_divergence(x0_latent_proj, x1_latent_target)
                else:
                    kl_loss = torch.tensor(0.0, device=device)
            else:
                wass_loss = torch.tensor(0.0, device=device)
                kl_loss = torch.tensor(0.0, device=device)

            if t_sampler is not None:
                t = t_sampler.sample(B, device)
            else:
                t = None  
            
            fm_loss, x1_pred = model(x0_latent_proj, x1_latent_target, meta=meta, t=t)

            if freq_ode_steps > 0:
                x1_pred_ode = model.inference(x0_latent_proj, steps=freq_ode_steps, meta=meta)
                radar_recon = radar_vae.decode(x1_pred_ode)

                recon_loss = F.mse_loss(radar_recon, radar)

                freq_loss = spectral_loss(radar_recon, radar)
                loss = fm_loss + 0.5 * recon_loss + 0.1 * freq_loss + wasserstein_weight * wass_loss + kl_weight * kl_loss
            else:
                radar_recon = radar_vae.decode(x1_pred)
                recon_loss = F.l1_loss(radar_recon, radar)
                freq_loss = spectral_loss(radar_recon, radar)
                loss = fm_loss + recon_loss + wasserstein_weight * wass_loss + kl_weight * kl_loss

        
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        total_loss += loss.item()
        total_fm_loss += fm_loss.item()
        total_recon_loss += recon_loss.item()
        total_freq_loss += freq_loss.item()
        total_wass_loss += wass_loss.item()
        total_kl_loss += kl_loss.item()
        batch_count += 1
        
        pbar.set_postfix({
            "loss": f"{loss.item():.5f}",
            "fm": f"{fm_loss.item():.4f}",
            "recon": f"{recon_loss.item():.5f}",
            "freq": f"{freq_loss.item():.5f}",
            "wass": f"{wass_loss.item():.5f}",
            "kl": f"{kl_loss.item():.5f}"
        })
        
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / batch_count
            print(f"  [{batch_idx + 1}/{len(train_loader)}] Loss: {loss.item():.5f} (FM: {fm_loss.item():.4f}, Recon: {recon_loss.item():.5f}, Freq: {freq_loss.item():.5f}, Wass: {wass_loss.item():.5f}, KL: {kl_loss.item():.5f}, avg: {avg_loss:.5f})")
    
    if batch_count > 0:
        return (
            total_loss / batch_count,
            total_fm_loss / batch_count,
            total_recon_loss / batch_count,
            total_freq_loss / batch_count,
            total_wass_loss / batch_count,
            total_kl_loss / batch_count
        )
    else:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')


@torch.no_grad()
def validate(
    model: mmflowFMScaled,
    val_loader: Optional[DataLoader],
    ecg_vae: SignalVAE,
    radar_vae: SignalVAE,
    device: torch.device,
    use_metadata: bool,
    amp_enabled: bool,
    num_inference_steps: int = 50,
    ecg_projector: Optional[LatentProjector] = None,
    wasserstein_weight: float = 0.01,
    kl_weight: float = 0.01,
) -> tuple[float, float, float, float, float, float]:
    if val_loader is None:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
    
    model.eval()
    if ecg_projector is not None:
        ecg_projector.eval()
    total_loss = 0.0
    total_fm_loss = 0.0
    total_recon_loss = 0.0
    total_freq_loss = 0.0
    total_wass_loss = 0.0
    total_kl_loss = 0.0
    batch_count = 0
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    ecg_vae.eval()
    radar_vae.eval()
   
    for batch_idx, batch_data in enumerate(val_loader):
        if use_metadata:
            radar, ecg, meta = batch_data
            meta = meta.to(device, non_blocking=True)
        else:
            radar, ecg, _ = batch_data
            meta = None
        
        radar = radar.to(device, non_blocking=True)
        ecg = ecg.to(device, non_blocking=True)

        x0_latent = encode_batch(ecg_vae, ecg, device, use_sampling=False)
        x1_latent = encode_batch(radar_vae, radar, device, use_sampling=False)

        if ecg_projector is not None:
            x0_latent_proj = ecg_projector(x0_latent)
            x1_latent_target = x1_latent
        else:
            x0_latent_proj = x0_latent
            x1_latent_target = x1_latent

        with torch.autocast(device_type=autocast_device, enabled=amp_enabled):

            if ecg_projector is not None:
                if wasserstein_weight > 0:
                    wass_loss = compute_wasserstein_distance(x0_latent_proj, x1_latent_target, eps=0.5)
                else:
                    wass_loss = torch.tensor(0.0, device=device)
                
                if kl_weight > 0:
                    kl_loss = compute_kl_divergence(x0_latent_proj, x1_latent_target)
                else:
                    kl_loss = torch.tensor(0.0, device=device)
            else:
                wass_loss = torch.tensor(0.0, device=device)
                kl_loss = torch.tensor(0.0, device=device)
            
            fm_loss, _ = model(x0_latent_proj, x1_latent_target, meta=meta)

            x1_pred_infer = model.inference(x0_latent_proj, steps=num_inference_steps, meta=meta)
            
            radar_recon = radar_vae.decode(x1_pred_infer)
            
            recon_loss = F.mse_loss(radar_recon, radar)
            freq_loss = spectral_loss(radar_recon, radar)
            
            loss = fm_loss + 0.5 * recon_loss + 0.1 * freq_loss + wasserstein_weight * wass_loss + kl_weight * kl_loss
        
        total_loss += loss.item()
        total_fm_loss += fm_loss.item()
        total_recon_loss += recon_loss.item()
        total_freq_loss += freq_loss.item()
        total_wass_loss += wass_loss.item()
        total_kl_loss += kl_loss.item()
        batch_count += 1
        if batch_count >= 5:
            break
    
    if batch_count > 0:
        return (
            total_loss / batch_count,
            total_fm_loss / batch_count,
            total_recon_loss / batch_count,
            total_freq_loss / batch_count,
            total_wass_loss / batch_count,
            total_kl_loss / batch_count
        )
    else:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')


def save_checkpoint(
    save_dir: str,
    model: mmflowFMScaled,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_val: float,
    tag: str,
) -> None:
    path = Path(save_dir) / f"mmflow_scaled_{tag}.pth"

    save_dict = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val": best_val,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    
    torch.save(save_dict, path)
    print(f"✓ Saved checkpoint to {path}")


def train(args: argparse.Namespace) -> None:
    """Main training loop."""
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("=" * 60)
    print("mmflow Scaled Training Configuration")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Use metadata: {args.use_metadata}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"CFG rate: {args.cfg_rate}")    
    print(f"Meta mask rate: {args.meta_mask_rate}")
    print(f"Loss type: {args.loss_type}")
    print(f"Sigma min: {args.sigma_min}")
    print(f"Sigma max: {args.sigma_max}")
    print(f"Use LogitNormal sampler: {args.use_logit_normal}")
    if args.use_logit_normal:
        print(f"  LogitNormal mean: {args.logit_normal_mean}")
        print(f"  LogitNormal std: {args.logit_normal_std}")
    print("=" * 60)

    t_sampler = None
    if args.use_logit_normal:
        t_sampler = LogitNormalSampler(mean=args.logit_normal_mean, std=args.logit_normal_std)
    else:
        print("[Time Sampling] Using uniform distribution U(0, 1)")
    
    train_loader, val_loader, dataset = build_mmflow_dataloaders(
        train_npz=args.train_npz,
        train_meta_csv=args.train_meta_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_npz=args.val_npz,
        val_meta_csv=args.val_meta_csv,
        normalize=args.normalize,
        limit=args.limit_samples,
        pin_memory=device.type == "cuda",
    )
    
    print("\nDataset Info:")
    print(dataset.describe())
    if val_loader:
        print(f"Validation batches: {len(val_loader)}")
    print("\nLoading VAEs...")
    ecg_vae = load_vae(
        checkpoint_path=args.ecg_vae_checkpoint,
        device=device,
        in_channels=dataset.num_channels,
        base_channels=args.vae_base_channels,
        ch_mult=tuple(args.vae_ch_mult),
        latent_dim=args.latent_dim,
        name="ECG VAE",
    )
    
    radar_vae = load_vae(
        checkpoint_path=args.radar_vae_checkpoint,
        device=device,
        in_channels=dataset.num_channels,
        base_channels=args.vae_base_channels,
        ch_mult=tuple(args.vae_ch_mult),
        latent_dim=args.latent_dim,
        name="Radar VAE",
    )

    ecg_vae.eval()
    radar_vae.eval()
    for param in ecg_vae.parameters():
        param.requires_grad = False
    for param in radar_vae.parameters():
        param.requires_grad = False

    downsample_factor = 2 ** (len(args.vae_ch_mult) - 1)
    latent_seq_len = dataset.seq_len // downsample_factor
    print(f"Latent space: channels={args.latent_dim}, seq_len={latent_seq_len}")

    print("\nBuilding mmflow Scaled model...")
    
    if args.use_flim:
        print("Using FLiM for metadata conditioning")
    else:
        print("Using pure AdaLN (no metadata conditioning)")
    
    dit = DirectFlowTransformer(
        in_channels=args.latent_dim,
        seq_len=latent_seq_len,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        use_flim=args.use_flim,
        use_metadata=args.use_metadata,
        dropout=args.dropout,
    ).to(device)
    
    model = mmflowFMScaled(
        dit_backbone=dit,
        guidance_scale=1.0,
        cfg_rate=args.cfg_rate,
        meta_mask_rate=args.meta_mask_rate,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        loss_type=args.loss_type,
    )
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"mmflow Scaled trainable parameters: {total_params:,}")
    

    ecg_projector = None
    if args.use_projector:
        print(f"\nBuilding ECG Latent Projector...")
        ecg_projector = LatentProjector(
            in_channels=args.latent_dim,
            hidden_dim=args.projector_hidden_dim
        ).to(device)
        proj_params = sum(p.numel() for p in ecg_projector.parameters() if p.requires_grad)
        print(f"Projector trainable parameters: {proj_params:,}")
        print(f"Wasserstein loss weight: {args.wasserstein_weight}")
        print(f"KL divergence loss weight: {args.kl_weight}")
    else:
        print(f"\nNo projector enabled.")
    
    print("="*60)

    if ecg_projector is not None:
        params_to_optimize = list(model.parameters()) + list(ecg_projector.parameters())
    else:
        params_to_optimize = model.parameters()
    
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.lr_warmup_steps
    min_lr_ratio = 1e-7 / args.lr  
    
    def cosine_schedule_with_warmup(step):
        """Cosine annealing with linear warmup and minimum lr."""
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        else:

            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    scheduler = None
    if args.lr_warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=cosine_schedule_with_warmup
        )
    
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    
    print("\nStarting training...")
    global_step = 0
    best_val = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        train_loss, train_fm_loss, train_recon_loss, train_freq_loss, train_wass_loss, train_kl_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,  
            ecg_vae=ecg_vae,
            radar_vae=radar_vae,
            device=device,
            epoch=epoch,
            use_metadata=args.use_metadata,
            amp_enabled=args.amp,
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
            ns=args.ns,
            freq_ode_steps=args.freq_ode_steps,
            t_sampler=t_sampler,
            ecg_projector=ecg_projector,
            wasserstein_weight=args.wasserstein_weight,
            kl_weight=args.kl_weight,
        )
        global_step += len(train_loader)
        if scheduler is not None:
            for _ in range(len(train_loader)):
                scheduler.step()
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.5f} (FM: {train_fm_loss:.5f}, Recon: {train_recon_loss:.5f}, Freq: {train_freq_loss:.5f}, Wass: {train_wass_loss:.5f}, KL: {train_kl_loss:.5f})")

        if epoch % 10 == 0:
            val_loss, val_fm_loss, val_recon_loss, val_freq_loss, val_wass_loss, val_kl_loss = validate(
                model=model,
                val_loader=val_loader,
                ecg_vae=ecg_vae,
                radar_vae=radar_vae,
                device=device,
                use_metadata=args.use_metadata,
                amp_enabled=args.amp,
                num_inference_steps=args.val_num_steps,
                ecg_projector=ecg_projector,
                wasserstein_weight=args.wasserstein_weight,
                kl_weight=args.kl_weight,
            )
            
            if not np.isnan(val_loss) and not np.isinf(val_loss):
                print(f"  Val Loss: {val_loss:.5f} (FM: {val_fm_loss:.5f}, Recon: {val_recon_loss:.5f}, Freq: {val_freq_loss:.5f}, Wass: {val_wass_loss:.5f}, KL: {val_kl_loss:.5f})")
                if val_loss < best_val:
                    best_val = val_loss

                    if epoch % 50 == 0:
                        save_checkpoint(
                            args.save_dir,
                            model,
                            optimizer,
                            epoch,
                            global_step,
                            best_val,
                            tag="best",
                        )
                  
                        if ecg_projector is not None:
                            proj_path = Path(args.save_dir) / f"ecg_projector_best.pth"
                            torch.save(ecg_projector.state_dict(), proj_path)
                            print(f"✓ Saved projector to {proj_path}")
        else:
            print(f"  (Skipping validation this epoch)")
        
        if epoch % args.save_every == 0:
            save_checkpoint(
                args.save_dir,
                model,
                optimizer,
                epoch,
                global_step,
                best_val,
                tag=f"ep{epoch}",
            )

            if ecg_projector is not None:
                proj_path = Path(args.save_dir) / f"ecg_projector_ep{epoch}.pth"
                torch.save(ecg_projector.state_dict(), proj_path)
    
    print(f"Best validation loss: {best_val:.5f}")


def parse_ch_mult(value: str) -> list:
    parts = [v.strip() for v in value.split(",") if v.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("ch_mult must contain at least one integer")
    return [int(p) for p in parts]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train_npz", type=str, default=None)
    parser.add_argument("--train_meta_csv", type=str, default=None)
    parser.add_argument("--val_npz", type=str, default=None)
    parser.add_argument("--val_meta_csv", type=str, default=None)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--limit_samples", type=int, default=None)
    parser.add_argument("--use_flim", action="store_true")
    
    parser.add_argument("--ecg_vae_checkpoint", type=str, default=None)
    parser.add_argument("--radar_vae_checkpoint", type=str, default=None)
    parser.add_argument("--vae_base_channels", type=int, default=32)
    parser.add_argument("--vae_ch_mult", type=parse_ch_mult, default=parse_ch_mult("1,2,2,4"))
    parser.add_argument("--latent_dim", type=int, default=16)
    
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--use_metadata", action="store_true")
    parser.add_argument("--cfg_rate", type=float, default=0.15)
    parser.add_argument("--meta_mask_rate", type=float, default=0.15)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "l1"])
    parser.add_argument("--sigma_min", type=float, default=1e-5)
    parser.add_argument("--sigma_max", type=float, default=1.0)
    
    parser.add_argument("--use_projector", action="store_true")
    parser.add_argument("--projector_hidden_dim", type=int, default=256)
    parser.add_argument("--wasserstein_weight", type=float, default=0.01)
    parser.add_argument("--kl_weight", type=float, default=0)
    
    parser.add_argument("--use_logit_normal", action="store_true")
    parser.add_argument("--logit_normal_mean", type=float, default=0.0)
    parser.add_argument("--logit_normal_std", type=float, default=1.0)
    
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_steps", type=int, default=1000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ns", type=float, default=0.05)
    parser.add_argument("--val_num_steps", type=int, default=50)
    parser.add_argument("--freq_ode_steps", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default=None)
    
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
