#!/usr/bin/env python3

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class mmflowDataset(Dataset):

    def __init__(
        self,
        npz_path: str | os.PathLike[str],
        meta_csv_path: str | os.PathLike[str],
        normalize: bool = False,
        limit: Optional[int] = None,
        dtype: np.dtype = np.float32,
        use_single_meta: bool = False,
    ) -> None:
        self.npz_path = Path(npz_path)
        self.meta_csv_path = Path(meta_csv_path)
        self.normalize = normalize
        self.dtype = dtype
        self.use_single_meta = use_single_meta

        self._load_signals()
        
        self._load_metadata()
        
        if limit is not None and limit > 0:
            limit = min(limit, len(self.radar))
            self.radar = self.radar[:limit]
            self.ecg = self.ecg[:limit]
            self.person_ids = self.person_ids[:limit]

    def _load_signals(self) -> None:
        with np.load(self.npz_path) as data:
            radar = data['radar']
            ecg = data['ecg']
            person_ids = data['person_labels']
            
            if radar.ndim == 2:
                radar = radar[:, None, :]
            if ecg.ndim == 2:
                ecg = ecg[:, None, :]
            
            self.radar = radar.astype(self.dtype)
            self.ecg = ecg.astype(self.dtype)
            self.person_ids = person_ids
            
            if self.normalize:
                radar_max = np.abs(self.radar).max()
                ecg_max = np.abs(self.ecg).max()
                if radar_max > 0:
                    self.radar = self.radar / radar_max
                if ecg_max > 0:
                    self.ecg = self.ecg / ecg_max
            
            self.seq_len = self.radar.shape[-1]
            self.num_channels = self.radar.shape[1]

    def _load_metadata(self) -> None:
        df = pd.read_csv(self.meta_csv_path)
        
        if self.use_single_meta:
            if len(df) == 0:
                raise ValueError("CSV file is empty, cannot use single meta mode")
            
            row = df.iloc[0]
            gender = 1 if str(row['Sex']).strip().upper() == 'M' else 0
            age = float(row['Age'])
            height = float(row['Height (cm)'])
            weight = float(row['Weight (kg)'])
            bmi = float(row['BMI'])
            single_meta = np.array([gender, age, height, weight, bmi], dtype=np.float32)
            
            self.metadata = np.tile(single_meta, (len(self.person_ids), 1))
            print(f"[Single Meta Mode] Using first CSV row for all {len(self.person_ids)} samples:")
            print(f"  Gender={gender}, Age={age}, Height={height}, Weight={weight}, BMI={bmi}")
            
        else:
            self.meta_dict = {}
            
            for _, row in df.iterrows():
                person_id = str(row['ID']).strip()
                
                gender = 1 if str(row['Sex']).strip().upper() == 'M' else 0
                age = float(row['Age'])
                height = float(row['Height (cm)'])
                weight = float(row['Weight (kg)'])
                bmi = float(row['BMI'])
                
                self.meta_dict[person_id] = np.array(
                    [gender, age, height, weight, bmi], 
                    dtype=np.float32
                )
            
            self.metadata = np.zeros((len(self.person_ids), 5), dtype=np.float32)
            
            for i, pid in enumerate(self.person_ids):
                pid_str = pid.decode('utf-8') if isinstance(pid, bytes) else str(pid)
                
                base_id = pid_str.replace('_data', '').strip()
                
                if base_id in self.meta_dict:
                    self.metadata[i] = self.meta_dict[base_id]
                else:
                    print(f"Warning: Metadata not found for '{base_id}' (original: '{pid_str}'), using defaults")
                    self.metadata[i] = np.array([0, 30, 170, 70, 24.2], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.radar)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        radar = torch.from_numpy(self.radar[idx])
        ecg = torch.from_numpy(self.ecg[idx])
        meta = torch.from_numpy(self.metadata[idx])
        
        return radar, ecg, meta

    @property
    def num_samples(self) -> int:
        return len(self)

    def describe(self) -> str:
        stats = {
            "samples": self.num_samples,
            "channels": self.num_channels,
            "seq_len": self.seq_len,
            "radar_min": float(self.radar.min()),
            "radar_max": float(self.radar.max()),
            "radar_mean": float(self.radar.mean()),
            "radar_std": float(self.radar.std()),
            "ecg_min": float(self.ecg.min()),
            "ecg_max": float(self.ecg.max()),
            "ecg_mean": float(self.ecg.mean()),
            "ecg_std": float(self.ecg.std()),
            "unique_persons": len(self.meta_dict),
        }
        
        meta_stats = {
            "age_mean": float(self.metadata[:, 1].mean()),
            "age_std": float(self.metadata[:, 1].std()),
            "height_mean": float(self.metadata[:, 2].mean()),
            "weight_mean": float(self.metadata[:, 3].mean()),
            "bmi_mean": float(self.metadata[:, 4].mean()),
        }
        
        desc = (
            f"mmflow Dataset:\n"
            f"  Samples: {stats['samples']} from {stats['unique_persons']} persons\n"
            f"  Signal: channels={stats['channels']}, seq_len={stats['seq_len']}\n"
            f"  Radar: min={stats['radar_min']:.4f}, max={stats['radar_max']:.4f}, "
            f"mean={stats['radar_mean']:.4f}, std={stats['radar_std']:.4f}\n"
            f"  ECG: min={stats['ecg_min']:.4f}, max={stats['ecg_max']:.4f}, "
            f"mean={stats['ecg_mean']:.4f}, std={stats['ecg_std']:.4f}\n"
            f"  Metadata: age={meta_stats['age_mean']:.1f}±{meta_stats['age_std']:.1f}, "
            f"height={meta_stats['height_mean']:.1f}cm, weight={meta_stats['weight_mean']:.1f}kg, "
            f"BMI={meta_stats['bmi_mean']:.1f}"
        )
        
        return desc


def build_mmflow_dataloaders(
    train_npz: str,
    train_meta_csv: str,
    batch_size: int,
    num_workers: int = 4,
    val_npz: Optional[str] = None,
    val_meta_csv: Optional[str] = None,
    normalize: bool = False,
    limit: Optional[int] = None,
    pin_memory: bool = True,
    use_single_meta: bool = False,
) -> Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader], mmflowDataset]:
    from torch.utils.data import DataLoader
    
    train_dataset = mmflowDataset(
        npz_path=train_npz,
        meta_csv_path=train_meta_csv,
        normalize=normalize,
        limit=limit,
        use_single_meta=use_single_meta,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    val_loader = None
    if val_npz is not None and val_meta_csv is not None:
        val_dataset = mmflowDataset(
            npz_path=val_npz,
            meta_csv_path=val_meta_csv,
            normalize=normalize,
            use_single_meta=use_single_meta,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=max(1, num_workers // 2),
            pin_memory=pin_memory,
            drop_last=False,
        )
    
    return train_loader, val_loader, train_dataset


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python mmflow_dataset.py <npz_path> <meta_csv_path>")
        sys.exit(1)
    
    dataset = mmflowDataset(
        npz_path=sys.argv[1],
        meta_csv_path=sys.argv[2],
        normalize=False,
    )
    
    print(dataset.describe())
    print("\nSample data:")
    radar, ecg, meta = dataset[0]
    print(f"  Radar shape: {radar.shape}")
    print(f"  ECG shape: {ecg.shape}")
    print(f"  Meta shape: {meta.shape}")
    print(f"  Meta values: {meta.numpy()}")
    print(f"  Meta [Gender, Age, Height, Weight, BMI]")
