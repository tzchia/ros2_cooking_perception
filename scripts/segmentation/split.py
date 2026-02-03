"""Train/val split helpers with persistence."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def split_train_val(
    df: pd.DataFrame, val_ratio: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if val_ratio <= 0 or df.empty:
        return df, df.iloc[0:0]
    df_sorted = df.sort_values("rgb_path").reset_index(drop=True)
    if len(df_sorted) < 2:
        return df_sorted, df_sorted.iloc[0:0]
    val_count = int(math.ceil(len(df_sorted) * val_ratio))
    val_count = min(val_count, len(df_sorted) - 1)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(df_sorted))
    rng.shuffle(indices)
    val_indices = np.sort(indices[:val_count])
    train_indices = np.sort(indices[val_count:])
    return df_sorted.iloc[train_indices], df_sorted.iloc[val_indices]


def save_split_file(
    split_file: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    val_ratio: float,
    split_seed: int,
) -> None:
    split_df = pd.concat(
        [
            train_df[["rgb_path"]].assign(split="train"),
            val_df[["rgb_path"]].assign(split="val"),
        ],
        ignore_index=True,
    )
    split_df["val_ratio"] = val_ratio
    split_df["split_seed"] = split_seed
    split_file.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(split_file, index=False)
    print(f"Saved train/val split to {split_file}")


def load_split_file(
    split_file: Path, df: pd.DataFrame, val_ratio: float, split_seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_df = pd.read_csv(split_file)
    if "rgb_path" not in split_df.columns or "split" not in split_df.columns:
        raise ValueError(f"Split file is missing required columns: {split_file}")
    if "val_ratio" in split_df.columns:
        stored_ratio = split_df["val_ratio"].iloc[0]
        if not np.isclose(float(stored_ratio), float(val_ratio)):
            print(
                f"Warning: split file ratio {stored_ratio} differs from val_ratio {val_ratio}."
            )
    if "split_seed" in split_df.columns:
        stored_seed = split_df["split_seed"].iloc[0]
        if int(stored_seed) != int(split_seed):
            print(
                f"Warning: split file seed {stored_seed} differs from split_seed {split_seed}."
            )
    split_df = split_df[["rgb_path", "split"]]
    merged = df.merge(split_df, on="rgb_path", how="left")
    missing = merged["split"].isna().sum()
    if missing:
        raise ValueError(
            f"Split file {split_file} is missing {missing} rows from index.csv."
        )
    if not set(merged["split"].unique()).issubset({"train", "val"}):
        raise ValueError(f"Split file {split_file} contains invalid split values.")
    train_df = merged[merged["split"] == "train"].drop(columns=["split"])
    val_df = merged[merged["split"] == "val"].drop(columns=["split"])
    if val_df.empty:
        raise ValueError(f"Split file {split_file} has no validation rows.")
    return train_df, val_df


def get_train_val_split(
    df: pd.DataFrame, args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cached = getattr(args, "_split_cache", None)
    if cached is not None:
        return cached
    split_file = getattr(args, "split_file", None)
    if split_file is not None:
        split_file = Path(split_file)
    if split_file and split_file.exists() and not args.refresh_split:
        train_df, val_df = load_split_file(
            split_file, df, args.val_ratio, args.split_seed
        )
        args._split_cache = (train_df, val_df)
        return train_df, val_df
    train_df, val_df = split_train_val(df, args.val_ratio, args.split_seed)
    if split_file:
        save_split_file(split_file, train_df, val_df, args.val_ratio, args.split_seed)
    args._split_cache = (train_df, val_df)
    return train_df, val_df
