"""
Inference script for continuous audio recordings
Detects haddock calls in audio files using trained ResNet18 model
All configuration parameters are loaded from the model checkpoint
"""
import torch
import numpy as np
import argparse
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
import librosa
import soundfile as sf
import tempfile
import os

from ecosound.core.audiotools import Sound
from ecosound.core.spectrogram import Spectrogram
from ecosound.core.annotation import Annotation
from model import SpectrogramClassifier
import matplotlib.pyplot as plt
from datetime import datetime
import uuid
import platform

def parse_spec_config(spec_config):
    """Parse spectrogram config from JSON format to numerical values"""
    params = {}
    params['rate'] = int(spec_config['rate'].replace(' Hz', ''))
    params['window'] = float(spec_config['window'].replace(' s', ''))
    params['step'] = float(spec_config['step'].replace(' s', ''))
    params['freq_min'] = float(spec_config['freq_min'].replace(' Hz', ''))
    params['freq_max'] = float(spec_config['freq_max'].replace(' Hz', ''))
    params['window_func'] = spec_config['window_func']
    params['duration'] = float(spec_config['duration'].replace(' s', ''))
    params['denoiser'] = spec_config.get('denoiser', [])
    params['dB'] = True
    params['use_dask'] = False
    params['dask_chunks'] = (1000, 1000)
    return params


def load_model_and_config(model_path, device):
    """
    Load trained model and all configuration from checkpoint

    Args:
        model_path: Path to model checkpoint
        device: torch device

    Returns:
        model: Loaded model
        metadata: Complete configuration metadata
    """
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # Check if metadata exists in checkpoint
    if 'metadata' not in checkpoint:
        raise ValueError(
            "Model checkpoint does not contain metadata! "
            "Please retrain the model with the updated train.py script."
        )

    metadata = checkpoint['metadata']

    # Print model configuration
    print("\n" + "="*80)
    print("MODEL CONFIGURATION")
    print("="*80)
    print(json.dumps(metadata, indent=2))
    print("="*80 + "\n")

    # Create and load model
    num_classes = metadata['model_config']['num_classes']
    model = SpectrogramClassifier(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully: {metadata['model_config']['architecture']}")
    print(f"Expected input shape: {metadata['model_config']['input_shape']}")

    return model, metadata


def decimate_audio(audio_path, target_sr, channel=0, output_dir=None):
    """
    Resample audio to target sampling rate using librosa

    Args:
        audio_path: Path to audio file
        target_sr: Target sampling rate in Hz
        channel: Channel to read (0-indexed)
        output_dir: Optional output directory for decimated file

    Returns:
        decimated_audio_path: Path to decimated audio file
        duration: Duration of audio in seconds
    """
    print(f"Decimating audio: {audio_path}")

    # Read audio
    sound = Sound(audio_path)
    sound.read(channel=channel, detrend=True)

    fs_orig = sound.file_sampling_frequency
    waveform = np.array(sound.waveform).astype(np.float32).flatten()

    # Resample if needed
    if fs_orig != target_sr:
        waveform = librosa.resample(y=waveform, orig_sr=fs_orig, target_sr=target_sr)

    # Normalize
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)

    # Save decimated audio
    if output_dir is None:
        output_dir = tempfile.gettempdir()

    output_path = os.path.join(output_dir, f"decimated_{Path(audio_path).name}")
    sf.write(output_path, waveform, target_sr)

    duration = len(waveform) / target_sr
    print(f"Decimated: {fs_orig} Hz -> {target_sr} Hz | Duration: {duration:.2f}s")

    return output_path, duration


def calculate_spectrogram(audio_path, spec_params):
    """
    Calculate spectrogram using ecosound library

    Args:
        audio_path: Path to audio file
        spec_params: Spectrogram parameters dictionary

    Returns:
        spectro: Spectrogram object with denoising applied
    """
    print("Calculating spectrogram...")

    # Load audio
    sound = Sound(audio_path)
    sound.read(channel=0)

    # Create spectrogram
    spectro = Spectrogram(
        frame=spec_params['window'],
        window_type=spec_params['window_func'],
        fft=spec_params['window'],
        step=spec_params['step'],
        sampling_frequency=spec_params['rate'],
        unit='sec',
        verbose=False
    )

    # Compute spectrogram
    spectro.compute(sound,
                   dB=spec_params['dB'],
                   use_dask=spec_params['use_dask'],
                   dask_chunks=spec_params['dask_chunks'])

    print(f"Spectrogram shape before crop: {spectro.spectrogram.shape}")

    # Crop frequencies
    spectro.crop(frequency_min=spec_params['freq_min'],
                frequency_max=spec_params['freq_max'],
                inplace=True)

    print(f"Spectrogram shape after crop: {spectro.spectrogram.shape}")

    # Apply denoising
    for denoiser in spec_params['denoiser']:
        print(f"Applying denoiser: {denoiser['name']}")
        spectro.denoise(
            denoiser['name'],
            window_duration=denoiser['window_duration_sec'],
            use_dask=False,
            dask_chunks=(1000, 1000),
            inplace=True
        )

    return spectro


def check_spectrogram_dimensions(spectro, expected_shape):
    """
    Sanity check: verify spectrogram dimensions match training data

    Args:
        spectro: Spectrogram object
        expected_shape: Expected (height, width) of spectrogram from training

    Returns:
        is_valid: Boolean indicating if dimensions are valid
        message: Description of check result
    """
    freq_bins, time_bins = spectro.spectrogram.shape
    expected_freq_bins, expected_time_bins = expected_shape

    if freq_bins != expected_freq_bins:
        return False, (f"Frequency bins mismatch: got {freq_bins}, "
                      f"expected {expected_freq_bins}")

    if time_bins < expected_time_bins:
        return False, (f"Not enough time bins: got {time_bins}, "
                      f"need at least {expected_time_bins}")

    return True, f"Dimensions OK: freq_bins={freq_bins}, time_bins={time_bins}"


def extract_windows(spectro, window_time_bins, step_duration):
    """
    Extract sliding windows from spectrogram

    Args:
        spectro: Spectrogram object
        window_time_bins: Number of time bins per window
        step_duration: Step between windows in seconds

    Returns:
        windows: List of spectrogram windows (freq_bins, time_bins_per_window)
        times: List of start times for each window
    """
    freq_bins, total_time_bins = spectro.spectrogram.shape
    time_resolution = spectro.time_resolution

    # Calculate step in bins
    step_bins = int(step_duration / time_resolution)

    windows = []
    times = []

    # Slide window across spectrogram
    for start_bin in range(0, total_time_bins - window_time_bins + 1, step_bins):
        end_bin = start_bin + window_time_bins
        window = spectro.spectrogram[:, start_bin:end_bin]

        # Only keep if window has correct size
        if window.shape[1] == window_time_bins:
            windows.append(window)
            times.append(spectro.axis_times[start_bin])

    return windows, times


def run_inference(model, windows, normalize_stats, device, batch_size=32):
    """
    Run model inference on spectrogram windows

    Args:
        model: Trained PyTorch model
        windows: List of spectrogram windows
        normalize_stats: Dict with normalization info or None
        device: torch device
        batch_size: Batch size for inference

    Returns:
        predictions: Array of class predictions (0 or 1)
        confidences: Array of confidence scores for positive class
    """
    model.eval()
    predictions = []
    confidences = []

    # Check if normalization is enabled
    if normalize_stats and normalize_stats.get('enabled', False):
        mean = normalize_stats['mean']
        std = normalize_stats['std']
        apply_norm = True
    else:
        apply_norm = False

    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = windows[i:i+batch_size]

            # Convert to tensor (batch, 1, H, W)
            batch_tensor = torch.stack([torch.from_numpy(w.T).unsqueeze(0).float()
                                       for w in batch])

            # Apply normalization if enabled
            if apply_norm:
                batch_tensor = (batch_tensor - mean) / (std + 1e-8)

            # Move to device
            batch_tensor = batch_tensor.to(device)

            # Forward pass
            outputs = model(batch_tensor)
            probs = torch.softmax(outputs, dim=1)

            # Get predictions and confidences for positive class (class 1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            confs = probs[:, 1].cpu().numpy()  # Confidence for haddock class

            predictions.extend(preds)
            confidences.extend(confs)

    return np.array(predictions), np.array(confidences)


def plot_predictions_timeseries(times, predictions, confidences, threshold,
                                audio_filename, output_dir):
    """
    Plot time series of predictions and confidences

    Args:
        times: Array of window start times
        predictions: Array of predictions (0 or 1)
        confidences: Array of confidence scores
        threshold: Confidence threshold used
        audio_filename: Name of the audio file being processed
        output_dir: Directory to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Plot confidences
    ax1.plot(times, confidences, 'b-', linewidth=0.5, alpha=0.7, label='Confidence')
    ax1.axhline(y=threshold, color='r', linestyle='--', linewidth=1.5, label=f'Threshold ({threshold})')
    ax1.fill_between(times, 0, confidences, where=(confidences >= threshold),
                     alpha=0.3, color='green', label='Above threshold')
    ax1.set_ylabel('Confidence Score', fontsize=11)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set_title(f'Haddock Call Detection - {audio_filename}', fontsize=12, fontweight='bold')

    # Plot predictions (binary)
    ax2.fill_between(times, 0, predictions, step='post', alpha=0.6, color='orange')
    ax2.set_ylabel('Prediction (0=Noise, 1=Haddock)', fontsize=11)
    ax2.set_xlabel('Time (seconds)', fontsize=11)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Noise', 'Haddock'])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_filename = Path(audio_filename).stem + '_predictions.png'
    plot_path = Path(output_dir) / plot_filename
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved prediction plot to: {plot_path}")


def detections_to_annotation(detections_df, audio_path, spec_params, channel=0):
    """
    Convert detections DataFrame to ecosound Annotation object

    Args:
        detections_df: DataFrame with columns [start_time, end_time, confidence]
        audio_path: Path to the audio file
        spec_params: Spectrogram parameters dictionary
        channel: Audio channel used

    Returns:
        annot: Annotation object
    """
    if len(detections_df) == 0:
        return Annotation()

    annot = Annotation()
    annot_data = annot.data

    # Detection times and frequencies
    annot_data['time_min_offset'] = detections_df['start_time'].tolist()
    annot_data['time_max_offset'] = detections_df['end_time'].tolist()
    annot_data['frequency_min'] = spec_params['freq_min']
    annot_data['frequency_max'] = spec_params['freq_max']

    # Audio file info
    annot_data['audio_file_dir'] = os.path.dirname(audio_path)
    annot_data['audio_file_name'] = os.path.splitext(os.path.basename(audio_path))[0]
    annot_data['audio_file_extension'] = Path(audio_path).suffix
    annot_data['audio_channel'] = channel

    # Audio properties
    try:
        sf_info = sf.info(audio_path)
        annot_data['audio_sampling_frequency'] = int(sf_info.samplerate)

        # Extract bit depth
        bit_depth = 16  # default
        try:
            parts = sf_info.subtype_info.split()
            for p in parts:
                if p.isdigit():
                    bit_depth = int(p)
                    break
        except Exception:
            pass
        annot_data['audio_bit_depth'] = bit_depth
    except Exception as e:
        print(f"Warning: Could not read audio file info: {e}")
        annot_data['audio_sampling_frequency'] = spec_params['rate']
        annot_data['audio_bit_depth'] = 16

    # Detection metadata
    annot_data['label_class'] = "Haddock"
    annot_data['confidence'] = detections_df['confidence'].tolist()
    annot_data['software_name'] = "Haddock_pytorch"
    annot_data['entry_date'] = datetime.now()
    annot_data['duration'] = (detections_df['end_time'] - detections_df['start_time']).tolist()
    annot_data['uuid'] = [str(uuid.uuid4()) for _ in range(len(detections_df))]
    annot_data['operator_name'] = platform.uname().node

    annot.data = annot_data
    return annot


def merge_detections(times, predictions, confidences, threshold,
                    min_duration, max_duration, merge_gap):
    """
    Merge consecutive detections and filter by duration

    Args:
        times: Array of window start times
        predictions: Array of predictions (0 or 1)
        confidences: Array of confidence scores
        threshold: Minimum confidence threshold
        min_duration: Minimum detection duration in seconds
        max_duration: Maximum detection duration in seconds
        merge_gap: Maximum gap to merge detections in seconds

    Returns:
        detections: DataFrame with columns [start_time, end_time, confidence]
    """
    # Filter by threshold
    mask = confidences >= threshold

    if not np.any(mask):
        return pd.DataFrame(columns=['start_time', 'end_time', 'confidence'])

    det_times = times[mask]
    det_confs = confidences[mask]

    # Merge consecutive detections
    detections = []
    start_time = det_times[0]
    end_time = det_times[0]
    conf_sum = det_confs[0]
    conf_count = 1

    for i in range(1, len(det_times)):
        gap = det_times[i] - end_time

        if gap <= merge_gap:
            # Extend current detection
            end_time = det_times[i]
            conf_sum += det_confs[i]
            conf_count += 1
        else:
            # Save current detection if it meets duration criteria
            duration = end_time - start_time
            if min_duration <= duration <= max_duration:
                avg_conf = conf_sum / conf_count
                detections.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'confidence': avg_conf
                })

            # Start new detection
            start_time = det_times[i]
            end_time = det_times[i]
            conf_sum = det_confs[i]
            conf_count = 1

    # Don't forget the last detection
    duration = end_time - start_time
    if min_duration <= duration <= max_duration:
        avg_conf = conf_sum / conf_count
        detections.append({
            'start_time': start_time,
            'end_time': end_time,
            'confidence': avg_conf
        })

    return pd.DataFrame(detections)


def process_audio_file(audio_path, model, metadata, args, device):
    """
    Process a single audio file and detect haddock calls

    Args:
        audio_path: Path to audio file
        model: Trained model
        metadata: Configuration metadata from checkpoint
        args: Command line arguments
        device: torch device

    Returns:
        detections: DataFrame with detection results
    """
    print(f"\n{'='*80}")
    print(f"Processing: {audio_path}")
    print(f"{'='*80}")

    # Parse spectrogram config
    spec_config = metadata['spec_config']
    spec_params = parse_spec_config(spec_config)

    # Get expected input shape from metadata
    expected_shape = tuple(metadata['model_config']['input_shape'])

    # Decimate audio to target sampling rate
    decimated_path, duration = decimate_audio(audio_path, spec_params['rate'],
                                              channel=args.channel,
                                              output_dir=args.temp_dir)

    # Calculate spectrogram
    spectro = calculate_spectrogram(decimated_path, spec_params)

    # Sanity check: verify spectrogram dimensions
    is_valid, message = check_spectrogram_dimensions(spectro, expected_shape)
    print(f"Sanity check: {message}")

    if not is_valid:
        print("ERROR: Spectrogram dimensions do not match training data!")
        print(f"Expected shape: {expected_shape}")
        print(f"Got shape: {spectro.spectrogram.shape}")
        return pd.DataFrame()

    # Extract sliding windows
    window_time_bins = expected_shape[1]
    print(f"Extracting windows (time_bins={window_time_bins}, step={args.step_duration}s)...")
    windows, times = extract_windows(spectro, window_time_bins, args.step_duration)
    print(f"Extracted {len(windows)} windows")

    if len(windows) == 0:
        print("No windows extracted!")
        return pd.DataFrame()

    # Run inference
    print(f"Running model inference on {len(windows)} windows...")
    normalize_stats = metadata.get('normalization', {'enabled': False})
    raw_predictions, confidences = run_inference(model, windows, normalize_stats,
                                                 device, args.batch_size)

    # Merge detections
    print(f"Applying threshold ({args.threshold}) and merging detections...")
    detections = merge_detections(
        np.array(times), raw_predictions, confidences,
        threshold=args.threshold,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        merge_gap=args.merge_gap
    )

    print(f"Found {len(detections)} detections")

    # Save prediction time series plot if requested (after threshold is applied)
    if args.save_plots:
        # Create thresholded predictions for accurate visualization
        thresholded_predictions = (confidences >= args.threshold).astype(int)
        plot_predictions_timeseries(
            np.array(times), thresholded_predictions, confidences,
            args.threshold, Path(audio_path).name, args.output_dir
        )

    # Add classification window duration to detection end times
    if len(detections) > 0:
        window_duration = spec_params['duration']
        detections['end_time'] = detections['end_time'] + window_duration
        print(f"Added classification window duration ({window_duration}s) to detection end times")

    # Save in CSV, Raven and NetCDF formats if requested
    if args.save_csv or args.save_raven or args.save_netcdf:
        if len(detections) > 0:
            # Save CSV format
            if args.save_csv:
                csv_filename = Path(audio_path).stem + '.csv'
                csv_path = Path(args.output_dir) / csv_filename
                print(f"Saving CSV format detections...")
                detections.to_csv(csv_path, index=False)
                print(f"Saved CSV detections to: {csv_path}")

            # Save Raven and NetCDF formats
            annot = detections_to_annotation(detections, audio_path, spec_params, args.channel)

            if args.save_raven:
                print(f"Saving Raven format annotations...")
                annot.to_raven(args.output_dir, single_file=False)
                print(f"Saved Raven annotations to: {args.output_dir}")

            if args.save_netcdf:
                nc_filename = Path(audio_path).stem + '.nc'
                nc_path = Path(args.output_dir) / nc_filename
                print(f"Saving NetCDF format annotations...")
                annot.to_netcdf(str(nc_path))
                print(f"Saved NetCDF annotations to: {nc_path}")
        else:
            # Create empty files even if no detections
            if args.save_csv:
                csv_filename = Path(audio_path).stem + '.csv'
                csv_path = Path(args.output_dir) / csv_filename
                pd.DataFrame(columns=['start_time', 'end_time', 'confidence']).to_csv(csv_path, index=False)

            annot = Annotation()
            if args.save_raven:
                annot.to_raven(args.output_dir, single_file=False)
            if args.save_netcdf:
                nc_filename = Path(audio_path).stem + '.nc'
                nc_path = Path(args.output_dir) / nc_filename
                annot.to_netcdf(str(nc_path))

    # Clean up decimated file
    if os.path.exists(decimated_path):
        os.remove(decimated_path)

    return detections


def main(args):
    """Main inference function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model and configuration from checkpoint
    model, metadata = load_model_and_config(args.model_path, device)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model configuration for reference
    config_output = output_dir / 'model_config.json'
    with open(config_output, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Model configuration saved to {config_output}")

    # Create temp directory
    temp_dir = Path(args.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Find audio files
    audio_dir = Path(args.audio_dir)
    audio_files = list(audio_dir.glob(f"*{args.extension}"))
    print(f"\nFound {len(audio_files)} audio files")

    # Process each file
    for audio_file in tqdm(audio_files, desc="Processing files"):
        detections = process_audio_file(
            str(audio_file), model, metadata, args, device
        )

        if len(detections) > 0:
            detections['filename'] = audio_file.name

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Detect haddock calls in continuous audio recordings. '
                   'All configuration is loaded from the model checkpoint.'
    )
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--audio_dir', type=str, required=True,
                        help='Directory containing audio files')
    parser.add_argument('--output_dir', type=str, default='./detections',
                        help='Output directory for results')
    parser.add_argument('--temp_dir', type=str, default='./temp',
                        help='Temporary directory for decimated files')
    parser.add_argument('--extension', type=str, default='.wav',
                        help='Audio file extension')
    parser.add_argument('--channel', type=int, default=0,
                        help='Audio channel to process (0-indexed)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold for detections')
    parser.add_argument('--step_duration', type=float, default=0.01,
                        help='Step between consecutive windows in seconds')
    parser.add_argument('--min_duration', type=float, default=0.1,
                        help='Minimum detection duration in seconds')
    parser.add_argument('--max_duration', type=float, default=1.0,
                        help='Maximum detection duration in seconds')
    parser.add_argument('--merge_gap', type=float, default=0.2,
                        help='Maximum gap to merge detections in seconds')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--save_plots', action='store_true',
                        help='Save time series plots of predictions and confidences for each file')
    parser.add_argument('--save_csv', action='store_true',
                        help='Save detections in CSV format (.csv) - one file per audio file')
    parser.add_argument('--save_raven', action='store_true',
                        help='Save detections in Raven format (.txt) - one file per audio file')
    parser.add_argument('--save_netcdf', action='store_true',
                        help='Save detections in NetCDF format (.nc) - one file per audio file')

    args = parser.parse_args()
    main(args)