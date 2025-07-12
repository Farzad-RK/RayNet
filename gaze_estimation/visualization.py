"""
ARGaze Training Visualization Utilities
-------------------------------------

This module provides visualization tools specifically for ARGaze dataset training logs.
It generates training loss and validation MAE plots with per-subject curves and mean values.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, Union


def load_argaze_logs(log_path: Union[str, Path]) -> pd.DataFrame:
    """Load ARGaze training logs from CSV file.
    
    Args:
        log_path: Path to the training log CSV file
        
    Returns:
        DataFrame containing the training logs
    """
    return pd.read_csv(log_path)


def annotate_mean_cusps(ax, epochs, mean_y, color, n_cusps: int = 2):
    """Annotate the largest improvements (cusps) on the mean curve.
    
    Args:
        ax: Matplotlib axis object
        epochs: Array of epoch numbers
        mean_y: Array of mean values (loss or MAE)
        color: Color for annotations
        n_cusps: Number of largest improvements to annotate
    """
    diffs = np.diff(mean_y)
    cusp_indices = np.argsort(diffs)[:n_cusps]
    for idx in cusp_indices:
        if diffs[idx] < 0:  # Only actual drops
            epoch = epochs[idx+1]
            val = mean_y[idx+1]
            ax.annotate(f'Epoch {epoch}', (epoch, val),
                       textcoords="offset points", xytext=(8, -18),
                       ha='center', color=color, fontsize=11,
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.5))


def plot_argaze_training_curves(
    log_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    epoch_min: int = 1,
    epoch_max: int = 30,
    log_scale: bool = False,
    dpi: int = 100
) -> None:
    """Generate training curves for ARGaze dataset.
    
    Creates two plots: training loss and validation MAE, both showing per-subject
    curves and the mean across all subjects.
    
    Args:
        log_path: Path to the training log CSV file
        output_dir: Directory to save plots (default: same as log file's directory)
        epoch_min: First epoch to plot
        epoch_max: Last epoch to plot
        log_scale: Whether to use log scale for the y-axis
        dpi: Resolution for saved figures
    """
    # Load and prepare data
    df = load_argaze_logs(log_path)
    folds = df['fold'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(folds)))
    
    # Set up output directory
    log_path = Path(log_path)
    if output_dir is None:
        output_dir = log_path.parent.parent / "plots"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ======== TRAINING LOSS PLOT =========
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    # Plot individual subjects
    for i, fold in enumerate(folds):
        subj_data = df[df['fold'] == fold]
        ax.plot(subj_data['epoch'], subj_data['train_loss'], 
                label=fold, color=colors[i], alpha=0.5)
    
    # Calculate and plot mean
    mean_loss = df.pivot(index='epoch', columns='fold', values='train_loss').mean(axis=1)
    epochs = mean_loss.index.values
    ax.plot(epochs, mean_loss, 'k-', linewidth=3, label='Mean', zorder=10)
    
    # Add annotations and style
    annotate_mean_cusps(ax, epochs, mean_loss, 'black')
    ax.set(xlabel='Epoch', ylabel='Training Loss',
           title=f'ARGaze - Training Loss (Epochs {epoch_min}-{epoch_max})',
           xlim=(epoch_min, epoch_max))
    ax.grid(True, alpha=0.3)
    if log_scale:
        ax.set_yscale('log')
    ax.legend(title='Subject', bbox_to_anchor=(1.02, 1), 
              loc='upper left', borderaxespad=0)
    
    # Save and clear
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_dir / 'training_loss.png', dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # ======== VALIDATION MAE PLOT =========
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    # Plot individual subjects
    for i, fold in enumerate(folds):
        subj_data = df[df['fold'] == fold]
        ax.plot(subj_data['epoch'], subj_data['val_mae'], 
                label=fold, color=colors[i], alpha=0.5)
    
    # Calculate and plot mean
    mean_mae = df.pivot(index='epoch', columns='fold', values='val_mae').mean(axis=1)
    ax.plot(epochs, mean_mae, 'k-', linewidth=3, label='Mean', zorder=10)
    
    # Add annotations and style
    annotate_mean_cusps(ax, epochs, mean_mae, 'black')
    ax.set(xlabel='Epoch', ylabel='Validation MAE (degrees)',
           title=f'ARGaze - Validation MAE (Epochs {epoch_min}-{epoch_max})',
           xlim=(epoch_min, epoch_max))
    ax.grid(True, alpha=0.3)
    ax.legend(title='Subject', bbox_to_anchor=(1.02, 1), 
              loc='upper left', borderaxespad=0)
    
    # Save and clear
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_dir / 'validation_mae.png', dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # Find subject closest to average for both metrics
    pivot_loss = df.pivot(index='epoch', columns='fold', values='train_loss')
    mean_loss = pivot_loss.mean(axis=1)
    loss_distances = ((pivot_loss - mean_loss.values.reshape(-1,1))**2).sum(axis=0)
    closest_loss_subject = loss_distances.idxmin()
    
    pivot_mae = df.pivot(index='epoch', columns='fold', values='val_mae')
    mean_mae = pivot_mae.mean(axis=1)
    mae_distances = ((pivot_mae - mean_mae.values.reshape(-1,1))**2).sum(axis=0)
    closest_mae_subject = mae_distances.idxmin()
    
    print(f"\nSubject closest to average training loss: {closest_loss_subject}")
    print(f"Subject closest to average validation MAE: {closest_mae_subject}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate ARGaze training curves')
    parser.add_argument('log_path', help='Path to training logs CSV')
    parser.add_argument('--output_dir', '-o', default=None,
                       help='Directory to save plots (default: log_dir/../plots)')
    parser.add_argument('--epoch_min', type=int, default=1,
                       help='First epoch to plot')
    parser.add_argument('--epoch_max', type=int, default=30,
                       help='Last epoch to plot')
    parser.add_argument('--log_scale', action='store_true',
                       help='Use log scale for y-axis')
    parser.add_argument('--dpi', type=int, default=100,
                       help='Resolution for saved figures')
    
    args = parser.parse_args()
    plot_argaze_training_curves(
        log_path=args.log_path,
        output_dir=args.output_dir,
        epoch_min=args.epoch_min,
        epoch_max=args.epoch_max,
        log_scale=args.log_scale,
        dpi=args.dpi
    )
