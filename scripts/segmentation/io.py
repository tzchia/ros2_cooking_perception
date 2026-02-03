"""Shared IO helpers for segmentation scripts."""
from __future__ import annotations

import argparse
from pathlib import Path


def prepare_io(args: argparse.Namespace) -> tuple[Path, Path]:
    output_dir = args.output_dir or (args.root / "output")
    index_csv = args.index_csv or (output_dir / "index.csv")
    if not index_csv.exists():
        raise FileNotFoundError(f"Missing {index_csv}. Run extract_rgbt_bag.py first.")
    args.sam_low = args.sam_low if args.sam_low is not None else args.thermal_low
    args.sam_checkpoint = args.sam_checkpoint or (args.root / "weights" / "sam_vit_b_01ec64.pth")
    args.split_file = args.split_file or (output_dir / "split_train_val.csv")
    return output_dir, index_csv
