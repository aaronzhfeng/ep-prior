"""
PTB-XL Dataset for EP-Prior

Wraps the PTB-XL dataset for self-supervised learning with:
- Standard preprocessing (normalization, optional resampling)
- Train/val/test splits based on PTB-XL folds
- Optional augmentations for contrastive learning
- Few-shot sampling utilities

Dataset Info:
- 21,837 12-lead ECG records
- 10 seconds at 500Hz (or 100Hz downsampled)
- 71 SCP diagnostic statements
- 10-fold stratified splits
"""

import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Optional, Dict, List, Tuple, Callable
import wfdb


class PTBXLDataset(Dataset):
    """
    PTB-XL Dataset for self-supervised learning.
    
    Args:
        data_path: Path to PTB-XL root directory
        split: One of "train", "val", "test" or list of fold numbers
        sampling_rate: 100 or 500 Hz
        normalize: Whether to z-score normalize per-lead
        transform: Optional transform/augmentation function
        return_labels: Whether to return diagnostic labels
        label_type: "all", "superclass", or "subclass"
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        sampling_rate: int = 100,
        normalize: bool = True,
        transform: Optional[Callable] = None,
        return_labels: bool = False,
        label_type: str = "superclass",
    ):
        self.data_path = data_path
        self.sampling_rate = sampling_rate
        self.normalize = normalize
        self.transform = transform
        self.return_labels = return_labels
        self.label_type = label_type
        
        # Load metadata
        self.df = pd.read_csv(os.path.join(data_path, "ptbxl_database.csv"), index_col="ecg_id")
        self.df.scp_codes = self.df.scp_codes.apply(ast.literal_eval)
        
        # Load SCP statements for label mapping
        self.scp_df = pd.read_csv(os.path.join(data_path, "scp_statements.csv"), index_col=0)
        
        # Get superclass aggregation
        self.superclasses = self._get_superclass_mapping()
        
        # Filter by split
        self.df = self._filter_split(split)
        
        # Record folder based on sampling rate
        self.record_folder = f"records{sampling_rate}"
        
    def _get_superclass_mapping(self) -> Dict[str, str]:
        """Map SCP codes to their superclass."""
        mapping = {}
        for code in self.scp_df.index:
            if pd.notna(self.scp_df.loc[code, "diagnostic_class"]):
                mapping[code] = self.scp_df.loc[code, "diagnostic_class"]
        return mapping
    
    def _filter_split(self, split: str) -> pd.DataFrame:
        """Filter dataframe by split."""
        if split == "train":
            # Folds 1-8 for training
            return self.df[self.df.strat_fold.isin(range(1, 9))]
        elif split == "val":
            # Fold 9 for validation
            return self.df[self.df.strat_fold == 9]
        elif split == "test":
            # Fold 10 for test
            return self.df[self.df.strat_fold == 10]
        elif isinstance(split, list):
            return self.df[self.df.strat_fold.isin(split)]
        else:
            return self.df
    
    def _load_record(self, idx: int) -> np.ndarray:
        """Load ECG record from file."""
        row = self.df.iloc[idx]
        
        # Build path
        if self.sampling_rate == 100:
            path = os.path.join(self.data_path, row.filename_lr)
        else:
            path = os.path.join(self.data_path, row.filename_hr)
        
        # Remove .hea/.dat extension if present
        path = path.replace(".hea", "").replace(".dat", "")
        
        # Load with wfdb
        record = wfdb.rdrecord(path)
        signal = record.p_signal  # (T, 12)
        
        return signal.T  # (12, T)
    
    def _get_labels(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get diagnostic labels for a record."""
        row = self.df.iloc[idx]
        scp_codes = row.scp_codes
        
        labels = {}
        
        if self.label_type in ["all", "superclass"]:
            # Binary labels for 5 superclasses
            superclass_names = ["NORM", "MI", "STTC", "CD", "HYP"]
            superclass_labels = torch.zeros(5)
            
            for code, prob in scp_codes.items():
                if code in self.superclasses:
                    sc = self.superclasses[code]
                    if sc in superclass_names:
                        superclass_labels[superclass_names.index(sc)] = 1.0
            
            labels["superclass"] = superclass_labels
            labels["superclass_names"] = superclass_names
        
        return labels
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dict with:
            - "x": ECG signal (12, T)
            - "x_view1", "x_view2": Augmented views (if transform returns two)
            - "labels": Diagnostic labels (if return_labels=True)
        """
        # Load signal
        signal = self._load_record(idx)  # (12, T)
        
        # Normalize per-lead
        if self.normalize:
            mean = signal.mean(axis=1, keepdims=True)
            std = signal.std(axis=1, keepdims=True) + 1e-8
            signal = (signal - mean) / std
        
        signal = torch.from_numpy(signal).float()
        
        result = {"x": signal}
        
        # Apply transform (may return augmented views)
        if self.transform is not None:
            transformed = self.transform(signal)
            if isinstance(transformed, tuple) and len(transformed) == 2:
                result["x_view1"] = transformed[0]
                result["x_view2"] = transformed[1]
            else:
                result["x"] = transformed
        
        # Add labels
        if self.return_labels:
            labels = self._get_labels(idx)
            result.update(labels)
        
        return result


class PTBXLDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for PTB-XL.
    
    Args:
        data_path: Path to PTB-XL root
        sampling_rate: 100 or 500 Hz
        batch_size: Batch size
        num_workers: DataLoader workers
        normalize: Whether to normalize
        train_transform: Transform for training
        return_labels: Whether to return labels
    """
    
    def __init__(
        self,
        data_path: str = "/root/ep-prior/data/ptb-xl",
        sampling_rate: int = 100,
        batch_size: int = 64,
        num_workers: int = 4,
        normalize: bool = True,
        train_transform: Optional[Callable] = None,
        return_labels: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.data_path = data_path
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        self.train_transform = train_transform
        self.return_labels = return_labels
        
    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        if stage == "fit" or stage is None:
            self.train_dataset = PTBXLDataset(
                self.data_path,
                split="train",
                sampling_rate=self.sampling_rate,
                normalize=self.normalize,
                transform=self.train_transform,
                return_labels=self.return_labels,
            )
            self.val_dataset = PTBXLDataset(
                self.data_path,
                split="val",
                sampling_rate=self.sampling_rate,
                normalize=self.normalize,
                transform=None,
                return_labels=self.return_labels,
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = PTBXLDataset(
                self.data_path,
                split="test",
                sampling_rate=self.sampling_rate,
                normalize=self.normalize,
                transform=None,
                return_labels=self.return_labels,
            )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# =============================================================================
# Augmentations
# =============================================================================

class ECGAugmentation:
    """
    ECG-specific augmentations for contrastive learning.
    
    Augmentations are designed to be label-preserving:
    - Small time shifts
    - Amplitude scaling
    - Additive noise
    - Lead dropout
    """
    
    def __init__(
        self,
        time_shift_max: int = 50,
        amplitude_scale_range: Tuple[float, float] = (0.8, 1.2),
        noise_std: float = 0.05,
        lead_dropout_prob: float = 0.1,
    ):
        self.time_shift_max = time_shift_max
        self.amplitude_scale_range = amplitude_scale_range
        self.noise_std = noise_std
        self.lead_dropout_prob = lead_dropout_prob
    
    def _augment_single(self, x: torch.Tensor) -> torch.Tensor:
        """Apply augmentations to a single sample."""
        x = x.clone()
        C, T = x.shape
        
        # Time shift
        if self.time_shift_max > 0:
            shift = torch.randint(-self.time_shift_max, self.time_shift_max + 1, (1,)).item()
            x = torch.roll(x, shifts=shift, dims=-1)
        
        # Amplitude scaling (per-lead)
        if self.amplitude_scale_range[0] < self.amplitude_scale_range[1]:
            scale = torch.empty(C, 1).uniform_(*self.amplitude_scale_range)
            x = x * scale
        
        # Additive noise
        if self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        
        # Lead dropout
        if self.lead_dropout_prob > 0:
            mask = torch.rand(C) > self.lead_dropout_prob
            x = x * mask.unsqueeze(-1).float()
        
        return x
    
    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate two augmented views."""
        view1 = self._augment_single(x)
        view2 = self._augment_single(x)
        return view1, view2


# =============================================================================
# Few-shot sampling
# =============================================================================

class FewShotSampler:
    """
    Sample k examples per class for few-shot evaluation.
    
    For multi-label classification, ensures at least k positives per label.
    """
    
    def __init__(
        self,
        dataset: PTBXLDataset,
        k_shot: int = 10,
        label_key: str = "superclass",
        seed: int = 42,
    ):
        self.dataset = dataset
        self.k_shot = k_shot
        self.label_key = label_key
        self.seed = seed
        
        # Build label index
        self._build_index()
    
    def _build_index(self):
        """Build index of samples per label."""
        self.label_to_indices = {}
        
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            if self.label_key in sample:
                labels = sample[self.label_key]
                for label_idx, val in enumerate(labels):
                    if val > 0:
                        if label_idx not in self.label_to_indices:
                            self.label_to_indices[label_idx] = []
                        self.label_to_indices[label_idx].append(idx)
    
    def sample(self) -> List[int]:
        """Sample k examples per label, returning unique indices."""
        np.random.seed(self.seed)
        
        selected = set()
        
        for label_idx, indices in self.label_to_indices.items():
            # Sample up to k
            n_sample = min(self.k_shot, len(indices))
            sampled = np.random.choice(indices, size=n_sample, replace=False)
            selected.update(sampled)
        
        return list(selected)


# =============================================================================
# Unit tests
# =============================================================================

def _test_dataset():
    """Test PTB-XL dataset loading."""
    print("Testing PTBXLDataset...")
    
    data_path = "/root/ep-prior/data/ptb-xl"
    
    if not os.path.exists(os.path.join(data_path, "ptbxl_database.csv")):
        print("  PTB-XL not found, skipping test")
        return
    
    # Test basic loading
    dataset = PTBXLDataset(
        data_path,
        split="train",
        sampling_rate=100,
        normalize=True,
        return_labels=True,
    )
    
    print(f"  Train samples: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"  Signal shape: {sample['x'].shape}")
    print(f"  Superclass labels: {sample.get('superclass', 'N/A')}")
    
    # Test augmentation
    aug = ECGAugmentation()
    view1, view2 = aug(sample["x"])
    print(f"  Augmented view shape: {view1.shape}")
    
    # Test DataModule
    print("\nTesting PTBXLDataModule...")
    dm = PTBXLDataModule(data_path, batch_size=4, num_workers=0)
    dm.setup("fit")
    
    batch = next(iter(dm.train_dataloader()))
    print(f"  Batch 'x' shape: {batch['x'].shape}")
    
    print("✓ Dataset tests passed!")


if __name__ == "__main__":
    _test_dataset()

