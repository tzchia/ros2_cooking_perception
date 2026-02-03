"""Comparison and metrics helpers."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from segmentation.args import parse_method_list
from segmentation.data import load_pair
from segmentation.methods import generate_mask_by_method


def overlay_mask(
    rgb_img: np.ndarray, mask_img: np.ndarray, alpha: float = 0.4, color=(255, 64, 64)
) -> np.ndarray:
    overlay = rgb_img.astype(np.float32).copy()
    color_arr = np.array(color, dtype=np.float32)
    mask_bool = mask_img.astype(bool)
    overlay[mask_bool] = (1 - alpha) * overlay[mask_bool] + alpha * color_arr
    return overlay.astype(np.uint8)


def compare_masks(df: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    methods = parse_method_list(args.compare_methods)
    row = df.iloc[args.compare_row_idx]
    rgb_cmp, thermal_cmp = load_pair(output_dir, row)

    fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 4))
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        mask_img = generate_mask_by_method(rgb_cmp, thermal_cmp, method, args)
        if args.compare_overlay:
            view = overlay_mask(rgb_cmp, mask_img, alpha=args.compare_alpha)
            ax.imshow(view)
        else:
            ax.imshow(mask_img, cmap="gray")
        ax.set_title(method)
        ax.axis("off")

    if args.compare_out:
        fig.savefig(args.compare_out, bbox_inches="tight", dpi=200)
        print(f"Saved comparison to {args.compare_out}")
    else:
        plt.show()


def mask_metrics(mask_img: np.ndarray) -> dict[str, float]:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required for metrics. Install opencv-python.") from exc

    area_ratio = mask_img.mean().item()
    if mask_img.dtype != np.uint8:
        mask_img = mask_img.astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(mask_img)
    return {"area_ratio": area_ratio, "components": max(num_labels - 1, 0)}


def evaluate_metrics(
    df: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
    methods: List[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    methods = methods or (
        parse_method_list(args.eval_methods)
        if args.eval_methods
        else parse_method_list(args.compare_methods)
    )
    rng = np.random.default_rng(args.eval_seed)
    sample_count = min(args.eval_max_samples, len(df))
    sample_indices = rng.choice(len(df), size=sample_count, replace=False)
    rows = [df.iloc[idx] for idx in sample_indices]

    records: List[dict[str, float]] = []
    for row in rows:
        rgb_img, thermal_img = load_pair(output_dir, row)
        for method in methods:
            mask_img = generate_mask_by_method(rgb_img, thermal_img, method, args)
            metrics = mask_metrics(mask_img)
            metrics["method"] = method
            records.append(metrics)

    metrics_df = pd.DataFrame(records)
    summary_df = metrics_df.groupby("method").agg(["mean", "std"]).round(4)
    print(summary_df)

    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(args.metrics_out)
        print(f"Saved metrics summary to {args.metrics_out}")

    if args.metrics_raw_out:
        args.metrics_raw_out.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(args.metrics_raw_out, index=False)
        print(f"Saved metrics raw table to {args.metrics_raw_out}")

    return summary_df, metrics_df
