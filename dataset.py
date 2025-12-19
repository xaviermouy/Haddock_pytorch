"""
Dataset loader for Ketos HDF5 format spectrograms (structured array format)
"""
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np


class KetosSpectrogramDataset(Dataset):
    """
    PyTorch Dataset for loading spectrograms from Ketos HDF5 database
    Handles structured array format with compound dtype

    Args:
        h5_path: Path to the HDF5 file
        split: 'train' or 'test' - which split to load from the file
        transform: Optional transforms to apply to spectrograms
    """
    def __init__(self, h5_path, split='train', transform=None):
        self.h5_path = h5_path
        self.split = split
        self.transform = transform

        # Open HDF5 file and load metadata
        with h5py.File(h5_path, 'r') as f:
            # Get the data for the specified split
            if split in f.keys():
                # Load structured array
                structured_data = f[split]['data'][:]

                # Extract spectrograms and labels from structured array
                self.data = structured_data['data']  # Shape: (N, H, W)
                self.labels = structured_data['label']  # Shape: (N,)
            else:
                raise ValueError(f"Split '{split}' not found in HDF5 file. Available keys: {list(f.keys())}")

        print(f"Loaded {split} split: {len(self.data)} samples")
        print(f"Data shape: {self.data.shape}")
        print(f"Labels shape: {self.labels.shape}")
        print(f"Unique labels: {np.unique(self.labels)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get spectrogram and label
        spectrogram = self.data[idx]
        label = self.labels[idx]

        # Ensure spectrogram has shape (1, H, W) for single channel
        if spectrogram.ndim == 2:
            spectrogram = np.expand_dims(spectrogram, axis=0)

        # Convert to torch tensors
        spectrogram = torch.from_numpy(spectrogram).float()
        label = torch.tensor(label, dtype=torch.long)

        # Apply transforms if any
        if self.transform:
            spectrogram = self.transform(spectrogram)

        return spectrogram, label


def get_dataloaders(h5_path, batch_size=32, num_workers=4):
    """
    Create train and validation dataloaders

    Args:
        h5_path: Path to the HDF5 file
        batch_size: Batch size for training
        num_workers: Number of workers for data loading

    Returns:
        train_loader, val_loader (or test_loader if val doesn't exist)
    """
    # Check which splits are available
    with h5py.File(h5_path, 'r') as f:
        available_splits = list(f.keys())
        print(f"Available splits in HDF5: {available_splits}")

    # Create datasets
    train_dataset = KetosSpectrogramDataset(h5_path, split='train')

    # Use 'val' if available, otherwise 'test'
    val_split = 'val' if 'val' in available_splits else 'test'
    val_dataset = KetosSpectrogramDataset(h5_path, split=val_split)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader