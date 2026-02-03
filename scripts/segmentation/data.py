"""Data loading utilities."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def load_pair(output_dir: Path, row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    rgb = Image.open(output_dir / row["rgb_path"])
    thermal = Image.open(output_dir / row["thermal_raw_path"])
    return np.array(rgb), np.array(thermal)
