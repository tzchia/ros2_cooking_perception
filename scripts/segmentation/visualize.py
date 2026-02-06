"""Visualization helpers with Multi-Class Coloring."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from segmentation.args import parse_args, parse_method_list
from segmentation.data import load_pair
from segmentation.io import prepare_io
from segmentation.methods import generate_mask_by_method


def overlay_multiclass_mask(
    image: np.ndarray, mask: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """Draws a semi-transparent colored mask for each class ID."""
    if mask.sum() == 0:
        return image
        
    overlay = image.copy()
    unique_ids = np.unique(mask)
    
    # Use matplotlib colormap for distinct colors (skip background)
    cmap = plt.get_cmap("tab10")
    
    for uid in unique_ids:
        if uid == 0: continue
        
        # Determine color (uid-1 to align with 0-indexed colormap)
        color_rgb = cmap((uid - 1) % 10)[:3] 
        color_bgr = tuple(int(c * 255) for c in color_rgb) # CV2 uses BGR if needed, but plt uses RGB. 
        # Assuming input image is RGB for matplotlib
        color_int = tuple(int(c * 255) for c in color_rgb)

        class_mask = (mask == uid)
        
        # Color overlay
        overlay[class_mask] = color_int
        
        # Contours
        contours, _ = cv2.findContours(class_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)

    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def create_comparison_grid(
    df: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
    methods: list[str],
    num_samples: int = 5,
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(df), size=min(num_samples, len(df)), replace=False)
    
    n_cols = 2 + len(methods)
    n_rows = len(indices)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1: axes = axes[np.newaxis, :]
        
    for row_idx, data_idx in enumerate(indices):
        data_row = df.iloc[data_idx]
        rgb, thermal = load_pair(output_dir, data_row)
        
        # Col 0: RGB
        axes[row_idx, 0].imshow(rgb)
        axes[row_idx, 0].set_title(f"Sample {data_idx}\nRGB")
        axes[row_idx, 0].axis("off")
        
        # Col 1: Thermal
        t_norm = (thermal - thermal.min()) / (thermal.max() - thermal.min() + 1e-6)
        axes[row_idx, 1].imshow(t_norm, cmap="inferno")
        axes[row_idx, 1].set_title("Thermal")
        axes[row_idx, 1].axis("off")
        
        # Col 2+: Methods
        for m_idx, method in enumerate(methods):
            try:
                mask = generate_mask_by_method(rgb, thermal, method, args)
                viz = overlay_multiclass_mask(rgb, mask, alpha=0.5)
            except Exception as e:
                print(f"Error {method} sample {data_idx}: {e}")
                viz = np.zeros_like(rgb)
            
            axes[row_idx, m_idx + 2].imshow(viz)
            axes[row_idx, m_idx + 2].set_title(method)
            axes[row_idx, m_idx + 2].axis("off")
            
    plt.tight_layout()
    save_path = args.output_dir / "comparison_grid.jpg"
    plt.savefig(save_path, dpi=150)
    print(f"Comparison grid saved to: {save_path}")


if __name__ == "__main__":
    args = parse_args()
    output_dir, index_csv = prepare_io(args)
    df = pd.read_csv(index_csv)
    methods = parse_method_list(args.methods)
    print(f"Visualizing methods: {methods}")
    create_comparison_grid(df, output_dir, args, methods, num_samples=5)