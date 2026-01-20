"""
Common styles, colors, and utilities for EP-Prior paper figures.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = Path("/root/workspace/ep-prior")
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "paper" / "figures"

# ============================================================================
# COLOR PALETTE
# ============================================================================
COLORS = {
    'ep_prior': '#2E86AB',      # Blue
    'baseline': '#A23B72',       # Magenta
    'ablation_no_ep': '#7CB518', # Green (for ablation - no EP loss)
    'P_wave': '#E8871E',         # Orange
    'QRS': '#1E88E5',            # Blue
    'T_wave': '#43A047',         # Green
    'HRV': '#9E9E9E',            # Gray
    'full_ecg': '#212121',       # Dark gray
}

# Diagnostic superclass colors
SUPERCLASS_COLORS = {
    'NORM': '#1f77b4',
    'MI': '#ff7f0e',
    'STTC': '#2ca02c',
    'CD': '#d62728',
    'HYP': '#9467bd',
}

# ============================================================================
# PLOT STYLE CONFIGURATION (LARGER TEXT)
# ============================================================================
def setup_style():
    """Configure matplotlib style with larger, more readable text."""
    plt.rcParams.update({
        # Font settings
        'font.family': 'serif',
        'font.size': 14,              # Base font size (was 11)
        
        # Axes
        'axes.labelsize': 16,         # Axis labels (was 12)
        'axes.titlesize': 18,         # Subplot titles (was 13)
        'axes.titleweight': 'bold',
        'axes.linewidth': 1.2,
        
        # Ticks
        'xtick.labelsize': 13,        # X tick labels (was 10)
        'ytick.labelsize': 13,        # Y tick labels (was 10)
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        
        # Legend
        'legend.fontsize': 13,        # Legend text (was 10)
        'legend.title_fontsize': 14,
        
        # Figure
        'figure.dpi': 150,
        'figure.titlesize': 20,       # Suptitle (was 14)
        'figure.titleweight': 'bold',
        
        # Saving
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })

# ============================================================================
# ANNOTATION SIZES
# ============================================================================
ANNOTATION_SIZE = 13      # For data point annotations (was 9)
CELL_TEXT_SIZE = 12       # For heatmap cell values (was 10)
COLORBAR_LABEL_SIZE = 14  # For colorbar labels

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def save_figure(fig, name: str, output_dir: Path = None):
    """Save figure in both PDF and PNG formats."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_dir / f'{name}.pdf')
    fig.savefig(output_dir / f'{name}.png')
    print(f"  Saved: {name}.pdf and {name}.png")


def load_model_and_data(checkpoint_path: str = None, device: str = 'cuda'):
    """Load EP-Prior model and test dataloader."""
    import torch
    from torch.utils.data import DataLoader
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    
    from ep_prior.models import EPPriorSSL
    from ep_prior.data import PTBXLDataset
    
    if checkpoint_path is None:
        checkpoint_path = PROJECT_ROOT / "runs" / "ep_prior_v4_contrastive_fixed" / "checkpoints" / "last.ckpt"
    
    model = EPPriorSSL.load_from_checkpoint(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.eval()
    model.to(device)
    
    test_dataset = PTBXLDataset(
        PROJECT_ROOT / "data" / "ptb-xl", 
        split='test', 
        return_labels=True
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    return model, test_loader

