# Haddock vs Noise Classification with ResNet18

PyTorch implementation for classifying haddock sounds vs noise using ResNet18 on single-channel spectrogram images.

## Project Structure

```
.
├── dataset.py                    # HDF5 Dataset loader for Ketos format
├── model.py                      # ResNet18 adapted for single-channel input
├── train.py                      # Training script with config embedding
├── inference.py                  # Inference script (reads config from model)
├── spec_config_custom.json       # Spectrogram configuration parameters
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

1. Install PyTorch with CUDA support:
```bash
pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128
```

2. Install other dependencies:
```bash
pip install -r requirements.txt
```

3. Install ecosound (for spectrogram processing):
```bash
pip install ecosound
```

## Usage

### 1. Training

Train the model on your HDF5 database with spectrogram configuration:

```bash
python train.py \
    --data_path path/to/your/database.h5 \
    --spec_config spec_config_custom.json \
    --output_dir ./outputs \
    --epochs 50 \
    --batch_size 32 \
    --normalize
```

**Training Arguments:**
- `--data_path`: Path to your HDF5 database file (required)
- `--spec_config`: Path to spectrogram config JSON file (required)
- `--output_dir`: Directory to save outputs (default: `./outputs`)
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 0.001)
- `--num_workers`: Number of data loading workers (default: 4)
- `--normalize`: Enable data normalization (recommended)

**Training Outputs:**
- `best_model.pth`: Model checkpoint with **all configuration embedded**
- `final_model.pth`: Final model with configuration
- `training_config.json`: Human-readable training configuration
- `training_history.png`: Plot of training/validation loss and accuracy
- `training_history.json`: Raw training metrics

**What's Saved in Model Checkpoint:**
- Model weights
- Normalization statistics (mean, std)
- **Spectrogram configuration** (rate, window, step, frequencies, denoiser)
- **Model architecture info** (ResNet18, input shape, num_classes)
- Training configuration (batch_size, learning_rate, optimizer)

### 2. Inference on Continuous Recordings

Process audio files and detect haddock calls. 

```bash
python inference.py \
    --model_path ./outputs/best_model.pth \
    --audio_dir ./audio_files \
    --output_dir ./detections \
    --threshold 0.7
```

**Inference Arguments:**
- `--model_path`: Path to trained model checkpoint (.pth file) - required
- `--audio_dir`: Directory containing audio files - required
- `--output_dir`: Output directory for results (default: `./detections`)
- `--temp_dir`: Temporary directory for decimated files (default: `./temp`)
- `--extension`: Audio file extension (default: `.wav`)
- `--channel`: Audio channel to process, 0-indexed (default: 0)
- `--threshold`: Confidence threshold for detections (default: 0.5)
- `--step_duration`: Step between consecutive windows in seconds (default: 0.01)
- `--min_duration`: Minimum detection duration in seconds (default: 0.1)
- `--max_duration`: Maximum detection duration in seconds (default: 1.0)
- `--merge_gap`: Maximum gap to merge detections in seconds (default: 0.2)
- `--batch_size`: Batch size for inference (default: 32)
- `--save_plots`: Save time series plots of predictions and confidences for each file (optional)
- `--save_csv`: Save detections in CSV format - one file per audio file (optional)
- `--save_raven`: Save detections in Raven format (.txt) - one file per audio file (optional)
- `--save_netcdf`: Save detections in NetCDF format (.nc) - one file per audio file (optional)

**Output:**
- `model_config.json`: Complete model configuration for reference
- **Per-file outputs** (optional, based on flags):
  - `audiofile.csv`: CSV with columns `start_time`, `end_time`, `confidence` (if `--save_csv`)
  - `audiofile.txt`: Raven selection table format (if `--save_raven`)
  - `audiofile.nc`: NetCDF annotation format (if `--save_netcdf`)
  - `audiofile_predictions.png`: Time series plot of predictions and confidences (if `--save_plots`)

## Model Architecture

- **Base**: ResNet18
- **Input**: Single-channel spectrograms (1 x H x W)
- **Output**: Binary classification (haddock vs noise)
- **Modifications**:
  - First conv layer: 3 channels → 1 channel
  - Final FC layer: adapted for 2 classes
  - Trained from scratch (no ImageNet pretrained weights)

## Spectrogram Configuration

The `spec_config_custom.json` file defines spectrogram parameters:

```json
{
    "spectrogram": {
        "rate": "4000 Hz",
        "window": "0.064 s",
        "step": "0.01 s",
        "freq_min": "0 Hz",
        "freq_max": "1200 Hz",
        "window_func": "hann",
        "duration": "0.2 s",
        "denoiser": [
            {"name": "median_equalizer", "window_duration_sec": 3}
        ]
    }
}
```

This config is:
- Required during **training** (embedded in checkpoint)
- **NOT needed** during inference (loaded from checkpoint)

## HDF5 Database Format

The code expects a Ketos-format HDF5 file:

```
database.h5
├── train
│   ├── data      # Shape: (N_train, H, W) - spectrograms
│   └── labels    # Shape: (N_train,) - binary labels (0 or 1)
└── val
    ├── data      # Shape: (N_val, H, W) - spectrograms
    └── labels    # Shape: (N_val,) - binary labels (0 or 1)
```

## Example Workflow

1. **Create training database** (using your existing `create_ketos_database_v2-db.py`)

2. **Train model with configuration embedding**:
   ```bash
   python train.py \
       --data_path database.h5 \
       --spec_config spec_config_custom.json \
       --normalize \
       --epochs 100
   ```

3. **Run inference** (no config needed!):
   ```bash
   python inference.py \
       --model_path outputs/best_model.pth \
       --audio_dir ./test_audio \
       --threshold 0.7 \
       --save_csv \
       --save_raven \
       --save_netcdf \
       --save_plots
   ```

4. **Review detections** in output directory (one file per audio file processed)

## Sanity Checks

The inference script performs automatic sanity checks:
- Verifies spectrogram frequency bins match training data
- Ensures time bins are sufficient for windowing
- **Compares against saved input_shape** from training
- Warns and skips files if dimensions don't match
- Prevents errors from configuration mismatches

**If you get dimension mismatch errors**, you likely used different spectrogram parameters than training. The model checkpoint will show you the expected parameters.

## Notes

- Spectrograms should be denoised before creating HDF5 database
- All configuration is saved in model checkpoints automatically
- Inference loads config from checkpoint - no manual config needed
- Normalization is automatic if enabled during training
- Training plot updates after each epoch
- Best model saved based on validation loss

## Switching to ResNet50

To use ResNet50 instead of ResNet18, simply modify `model.py:10`:

```python
from torchvision.models import resnet50  # Change from resnet18
model = resnet50(pretrained=pretrained)  # Change from resnet18
```