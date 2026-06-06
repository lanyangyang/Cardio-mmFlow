#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  

from mmflow_dataset import build_mmflow_dataloaders
from mmflow_fm_scaled import mmflowFMScaled
from flow_transformer_no_cross import DirectFlowTransformer
from resnet_vae import SignalVAE


def encode_batch(vae: SignalVAE, signals: torch.Tensor, device: torch.device, use_sampling: bool = False) -> torch.Tensor:
    signals = signals.to(device)
    
    with torch.no_grad():
        mean, logvar = vae.encode(signals)
    
    if use_sampling and vae.training:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    else:
        return mean


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


def load_vae(
    checkpoint_path: str,
    device: torch.device,
    in_channels: int = 1,
    base_channels: int = 64,
    ch_mult: tuple = (1, 2, 4, 4),
    latent_dim: int = 4,
    name: str = "VAE",
) -> SignalVAE:
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


def load_projector(
    checkpoint_path: str,
    device: torch.device,
    in_channels: int,
    hidden_dim: int = 256,
) -> Optional[LatentProjector]:
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print("⚠ Projector checkpoint not found, will not use projector")
        return None
    
    projector = LatentProjector(in_channels=in_channels, hidden_dim=hidden_dim).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        projector.load_state_dict(checkpoint['model_state_dict'])
    else:
        projector.load_state_dict(checkpoint)
    
    projector.eval()
    for param in projector.parameters():
        param.requires_grad = False
    
    print(f"✓ Loaded projector from {checkpoint_path}")
    return projector


@torch.no_grad()
def run_inference(
    model: mmflowFMScaled,
    test_loader: DataLoader,
    ecg_vae: SignalVAE,
    radar_vae: SignalVAE,
    device: torch.device,
    use_metadata: bool,
    num_inference_steps: int,
    save_dir: str,
    ecg_projector: Optional[LatentProjector] = None,
    guidance_scale: float = 1.0,
) -> None:
    model.eval()
    ecg_vae.eval()
    radar_vae.eval()
    if ecg_projector is not None:
        ecg_projector.eval()
    
    all_ecg_signals = []
    all_radar_real_signals = []
    all_radar_gen_signals = []
    all_ecg_latents = []
    all_radar_real_latents = []
    all_radar_gen_latents = []
    
    for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Inference")):
        if use_metadata:
            radar, ecg, meta = batch_data
            meta = meta.to(device)
        else:
            radar, ecg, _ = batch_data
            meta = None
        
        radar = radar.to(device)
        ecg = ecg.to(device)
        
        ecg_latent = encode_batch(ecg_vae, ecg, device, use_sampling=False)
        radar_latent = encode_batch(radar_vae, radar, device, use_sampling=False)
        
        if ecg_projector is not None:
            x0_latent = ecg_projector(ecg_latent)
        else:
            x0_latent = ecg_latent
        
        if guidance_scale > 1.0:
            B = x0_latent.shape[0]
            x1_pred_latent, _ = model.sample(
                sample_steps=num_inference_steps,
                batch_size=B,
                sampling_method=model.sample_euler,
                unconditional_guidance_scale=guidance_scale,
                has_null_indicator=True,
                x_T=x0_latent,
                meta=meta,
            )
        else:
            x1_pred_latent = model.inference(x0_latent, steps=num_inference_steps, meta=meta)
        
        radar_gen = radar_vae.decode(x1_pred_latent)
        
        all_ecg_signals.append(ecg.cpu())
        all_radar_real_signals.append(radar.cpu())
        all_radar_gen_signals.append(radar_gen.cpu())
        all_ecg_latents.append(ecg_latent.cpu())
        all_radar_real_latents.append(radar_latent.cpu())
        all_radar_gen_latents.append(x1_pred_latent.cpu())
    
    all_ecg_signals = torch.cat(all_ecg_signals, dim=0).numpy()
    all_radar_real_signals = torch.cat(all_radar_real_signals, dim=0).numpy()
    all_radar_gen_signals = torch.cat(all_radar_gen_signals, dim=0).numpy()
    all_ecg_latents = torch.cat(all_ecg_latents, dim=0).numpy()
    all_radar_real_latents = torch.cat(all_radar_real_latents, dim=0).numpy()
    all_radar_gen_latents = torch.cat(all_radar_gen_latents, dim=0).numpy()
    
    os.makedirs(save_dir, exist_ok=True)
    
    signals_path = os.path.join(save_dir, 'signals.npz')
    latents_path = os.path.join(save_dir, 'latents.npz')
    
    np.savez(
        signals_path,
        ecg=all_ecg_signals,
        radar_real=all_radar_real_signals,
        radar_gen=all_radar_gen_signals,
    )
    
    np.savez(
        latents_path,
        ecg=all_ecg_latents,
        radar_real=all_radar_real_latents,
        radar_gen=all_radar_gen_latents,
    )


def plot_random_samples(
    ecg_signals: np.ndarray,
    radar_real_signals: np.ndarray,
    radar_gen_signals: np.ndarray,
    save_dir: str,
    num_samples: int = 5,
    seed: int = 42,
) -> None:
    np.random.seed(seed)
    num_total = ecg_signals.shape[0]
    indices = np.random.choice(num_total, size=min(num_samples, num_total), replace=False)
    
    os.makedirs(save_dir, exist_ok=True)
    
    for idx in indices:
        ecg = ecg_signals[idx, 0, :]
        radar_real = radar_real_signals[idx, 0, :]
        radar_gen = radar_gen_signals[idx, 0, :]
        
        radar_real_fft = np.fft.rfft(radar_real)
        radar_gen_fft = np.fft.rfft(radar_gen)
        radar_real_mag = np.abs(radar_real_fft)
        radar_gen_mag = np.abs(radar_gen_fft)
        
        seq_len = len(radar_real)
        freqs = np.fft.rfftfreq(seq_len, d=1.0)
        

        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(f'mmflow: ECG→Radar Generation & Analysis (Sample {idx})', 
                     fontsize=16, fontweight='bold')
        
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(ecg, 'b-', linewidth=1.5)
        ax1.set_title('Input ECG (Sample {})'.format(idx), fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        
        ax2 = plt.subplot(3, 2, 3)
        ax2.plot(radar_real, 'g-', linewidth=1.5)
        ax2.set_title('Ground Truth Radar - Time Domain', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True, alpha=0.3)
        
        ax3 = plt.subplot(3, 2, 5)
        ax3.plot(radar_gen, 'r-', linewidth=1.5)
        ax3.set_title('Generated Radar - Time Domain', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('Amplitude')
        ax3.grid(True, alpha=0.3)
        
        ax4 = plt.subplot(3, 2, 4)
        ax4.plot(freqs, radar_real_mag, 'g-', linewidth=1.5, label='Ground Truth')
        ax4.set_title('Ground Truth - Frequency Domain', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Frequency (Hz)')
        ax4.set_ylabel('Magnitude')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        ax5 = plt.subplot(3, 2, 6)
        ax5.plot(freqs, radar_real_mag, 'g-', linewidth=1.5, label='Ground Truth', alpha=0.7)
        ax5.plot(freqs, radar_gen_mag, 'r-', linewidth=1.5, label='Generated', alpha=0.7)
        ax5.set_title('Frequency Domain Comparison', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Frequency (Hz)')
        ax5.set_ylabel('Magnitude')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        ax6 = plt.subplot(3, 2, 2)
        ax6.plot(radar_real, 'g-', linewidth=1.5, label='Ground Truth', alpha=0.7)
        ax6.plot(radar_gen, 'r-', linewidth=1.5, label='Generated', alpha=0.7)
        ax6.set_title('Time Domain Comparison', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Time Steps')
        ax6.set_ylabel('Amplitude')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        save_path = os.path.join(save_dir, f'sample_{idx:04d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def parse_ch_mult(value: str) -> list:
    parts = [v.strip() for v in value.split(",") if v.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("ch_mult must contain at least one integer")
    return [int(p) for p in parts]


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--test_npz", type=str, required=True)
    parser.add_argument("--test_meta_csv", type=str, default=None)
    parser.add_argument("--normalize", action="store_true")
    
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--ecg_vae_checkpoint", type=str, required=True)
    parser.add_argument("--radar_vae_checkpoint", type=str, required=True)
    parser.add_argument("--projector_checkpoint", type=str, default=None)
    
    parser.add_argument("--vae_base_channels", type=int, default=32)
    parser.add_argument("--vae_ch_mult", type=parse_ch_mult, default=parse_ch_mult("1,2,2,4"))
    parser.add_argument("--latent_dim", type=int, default=4)
    
    parser.add_argument("--projector_hidden_dim", type=int, default=128)
    
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--use_flim", action="store_true")
    parser.add_argument("--use_metadata", action="store_true")
    parser.add_argument("--cfg_rate", type=float, default=0.15)
    parser.add_argument("--meta_mask_rate", type=float, default=0.15)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--sigma_min", type=float, default=1e-5)
    parser.add_argument("--sigma_max", type=float, default=1.0)
    
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--save_dir", type=str, default="./inference_results")
    parser.add_argument("--num_visualize", type=int, default=2)
    
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    print("="*60)
    print("mmflow Pure Inference")
    print("="*60)
    print(f"Device: {device}")
    print(f"Test data: {args.test_npz}")
    print(f"Model checkpoint: {args.model_checkpoint}")
    print(f"Projector: {args.projector_checkpoint if args.projector_checkpoint else 'None'}")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Save directory: {args.save_dir}")
    print("="*60)
    
    test_loader, _, test_dataset = build_mmflow_dataloaders(
        train_npz=args.test_npz,
        train_meta_csv=args.test_meta_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_npz=None,
        val_meta_csv=None,
        normalize=args.normalize,
        limit=None,
        pin_memory=device.type == "cuda",
    )
    print(f"Test dataset: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    ecg_vae = load_vae(
        args.ecg_vae_checkpoint, device,
        in_channels=test_dataset.num_channels,
        base_channels=args.vae_base_channels,
        ch_mult=tuple(args.vae_ch_mult),
        latent_dim=args.latent_dim,
        name="ECG VAE"
    )
    
    radar_vae = load_vae(
        args.radar_vae_checkpoint, device,
        in_channels=test_dataset.num_channels,
        base_channels=args.vae_base_channels,
        ch_mult=tuple(args.vae_ch_mult),
        latent_dim=args.latent_dim,
        name="Radar VAE"
    )
    
    ecg_projector = None
    if args.projector_checkpoint:
        ecg_projector = load_projector(
            args.projector_checkpoint,
            device,
            in_channels=args.latent_dim,
            hidden_dim=args.projector_hidden_dim,
        )
    
    checkpoint = torch.load(args.model_checkpoint, map_location=device)
    
    downsample_factor = 2 ** (len(args.vae_ch_mult) - 1)
    latent_seq_len = test_dataset.seq_len // downsample_factor
    
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
        loss_type='mse',
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    run_inference(
        model=model,
        test_loader=test_loader,
        ecg_vae=ecg_vae,
        radar_vae=radar_vae,
        device=device,
        use_metadata=args.use_metadata,
        num_inference_steps=args.num_inference_steps,
        save_dir=args.save_dir,
        ecg_projector=ecg_projector,
        guidance_scale=args.guidance_scale,
    )
    
    signals_data = np.load(os.path.join(args.save_dir, 'signals.npz'))
    plot_random_samples(
        ecg_signals=signals_data['ecg'],
        radar_real_signals=signals_data['radar_real'],
        radar_gen_signals=signals_data['radar_gen'],
        save_dir=os.path.join(args.save_dir, 'visualizations'),
        num_samples=args.num_visualize,
        seed=42,
    )


if __name__ == "__main__":
    main()
