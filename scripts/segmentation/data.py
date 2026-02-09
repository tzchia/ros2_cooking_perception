"""Data loading utilities."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def load_pair(output_dir: Path, row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    rgb = Image.open(output_dir / row["rgb_path"])
    thermal = Image.open(output_dir / row["thermal_raw_path"])
    return np.array(rgb), np.array(thermal)


def select_split_df(
    df: pd.DataFrame, args: argparse.Namespace, split: str
) -> pd.DataFrame:
    if split == "all":
        return df
    from segmentation.split import get_train_val_split

    train_df, val_df = get_train_val_split(df, args)
    return train_df if split == "train" else val_df
