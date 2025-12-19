"""
Training script for haddock vs noise classification with ResNet18
"""
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import json
import h5py
import numpy as np

from model import SpectrogramClassifier
from dataset import get_dataloaders


def load_spec_config(config_path):
    """Load spectrogram configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config['spectrogram']


def get_data_shape(h5_path, split='train'):
    """Get the shape of a single spectrogram sample"""
    with h5py.File(h5_path, 'r') as f:
        # Handle structured array format (compound dtype with 'data' field)
        sample_shape = f[split]['data'][0]['data'].shape
        print(f"Single sample shape: {sample_shape}")
    
    return sample_shape


def compute_normalization_stats(h5_path, split='train'):
    """
    Compute mean and std from training data for normalization
    Works with structured array format (compound dtype)

    Args:
        h5_path: Path to HDF5 file
        split: Which split to use (default: 'train')

    Returns:
        mean, std: Normalization statistics
    """
    print(f"Computing normalization statistics from {split} split...")
    with h5py.File(h5_path, 'r') as f:
        # Load all spectrogram data from structured array
        # Access the 'data' field which contains the spectrograms
        data = f[split]['data']['data']
        print(f"Loaded data shape: {data.shape}")

    # Compute statistics over ALL data
    mean = np.mean(data)
    std = np.std(data)

    print(f"Normalization stats - Mean: {mean:.4f}, Std: {std:.4f}")
    return float(mean), float(std)


def train_one_epoch(model, dataloader, criterion, optimizer, device, normalize_stats=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc='Training')
    for inputs, labels in pbar:
        # Apply normalization if stats are provided
        if normalize_stats is not None:
            mean, std = normalize_stats
            inputs = (inputs - mean) / (std + 1e-8)

        inputs, labels = inputs.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, normalize_stats=None):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for inputs, labels in pbar:
            # Apply normalization if stats are provided
            if normalize_stats is not None:
                mean, std = normalize_stats
                inputs = (inputs - mean) / (std + 1e-8)

            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot and save training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot losses
    ax1.plot(train_losses, label='Train Loss', marker='o')
    ax1.plot(val_losses, label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot accuracies
    ax2.plot(train_accs, label='Train Acc', marker='o')
    ax2.plot(val_accs, label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Training plot saved to {save_path}")
    plt.close()


def create_checkpoint_metadata(args, spec_config, data_shape, normalize_stats, num_classes=2):
    """
    Create comprehensive metadata dictionary for model checkpoint

    Args:
        args: Command line arguments
        spec_config: Spectrogram configuration dictionary
        data_shape: Shape of input spectrograms (H, W)
        normalize_stats: Tuple of (mean, std) or None
        num_classes: Number of output classes

    Returns:
        metadata: Dictionary containing all configuration
    """
    metadata = {
        'model_config': {
            'architecture': 'ResNet18',
            'num_classes': num_classes,
            'input_channels': 1,
            'input_shape': [data_shape[1], data_shape[0]],  # [freq_bins, time_bins] - swap because HDF5 stores as [time_bins, freq_bins]
        },
        'spec_config': spec_config,
        'training_config': {
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'num_epochs': args.epochs,
            'optimizer': 'Adam',
            'loss_function': 'CrossEntropyLoss',
        },
    }

    # Add normalization stats if enabled
    if normalize_stats is not None:
        metadata['normalization'] = {
            'enabled': True,
            'mean': normalize_stats[0],
            'std': normalize_stats[1]
        }
    else:
        metadata['normalization'] = {'enabled': False}

    return metadata


def train(args):
    """Main training function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load spectrogram configuration
    print(f"Loading spectrogram config from {args.spec_config}")
    spec_config = load_spec_config(args.spec_config)
    print(f"Spec config: {json.dumps(spec_config, indent=2)}")

    # Get data shape from HDF5
    print(f"Reading data shape from {args.data_path}")
    data_shape = get_data_shape(args.data_path, split='train')
    print(f"Input spectrogram shape: {data_shape}")

    # Compute normalization statistics if requested
    normalize_stats = None
    if args.normalize:
        mean, std = compute_normalization_stats(args.data_path, split='train')
        normalize_stats = (mean, std)
        print(f"Data normalization enabled: mean={mean:.4f}, std={std:.4f}")
    else:
        print("Data normalization disabled")

    # Create comprehensive metadata
    metadata = create_checkpoint_metadata(args, spec_config, data_shape, normalize_stats)

    # Save metadata to JSON for reference
    with open(output_dir / 'training_config.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Training configuration saved to {output_dir / 'training_config.json'}")

    # Load data
    print(f"\nLoading data from {args.data_path}")
    train_loader, val_loader = get_dataloaders(
        args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Create model
    print("\nCreating ResNet18 model for single-channel spectrograms")
    model = SpectrogramClassifier(num_classes=2)
    model = model.to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Learning rate scheduler (optional, reduces LR when validation loss plateaus)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_loss = float('inf')

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, normalize_stats
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, normalize_stats)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = output_dir / 'best_model.pth'
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'metadata': metadata,  # Include all configuration
            }
            torch.save(checkpoint, best_model_path)
            print(f"Saved best model to {best_model_path}")

        # Plot training history after each epoch
        plot_path = output_dir / 'training_history.png'
        plot_training_history(train_losses, val_losses, train_accs, val_accs, plot_path)

    # Save final model
    final_model_path = output_dir / 'final_model.pth'
    final_checkpoint = {
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'metadata': metadata,  # Include all configuration
    }
    torch.save(final_checkpoint, final_model_path)
    print(f"\nFinal model saved to {final_model_path}")

    # Save training history to JSON
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'metadata': metadata,
    }
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ResNet18 for haddock vs noise classification')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to HDF5 database file')
    parser.add_argument('--spec_config', type=str, required=True,
                        help='Path to spectrogram config JSON file')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Directory to save outputs (default: ./outputs)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--normalize', action='store_true',
                        help='Apply normalization to data (compute mean/std from training set)')

    args = parser.parse_args()
    train(args)