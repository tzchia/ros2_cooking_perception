"""YOLO dataset export helpers."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image

from segmentation.args import resolve_dataset_dir
from segmentation.data import load_pair
from segmentation.methods import generate_mask_by_method
from segmentation.split import get_train_val_split


def mask_to_polygons(mask_img: np.ndarray) -> list[np.ndarray]:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required for polygon conversion. Install opencv-python.") from exc

    mask = (mask_img > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for contour in contours:
        if len(contour) < 3:
            continue
        poly = contour.squeeze(1)
        polys.append(poly)
    return polys


def write_label(polys: Iterable[np.ndarray], w: int, h: int, label_path: Path, class_id: int = 0) -> None:
    lines = []
    for poly in polys:
        coords = []
        for x, y in poly:
            coords.extend([x / w, y / h])
        if coords:
            lines.append(" ".join([str(class_id)] + [f"{v:.6f}" for v in coords]))
    label_path.write_text("\n".join(lines))


def _prepare_image_dir(
    dataset_dir: Path,
    shared_images_dir: Path,
    split_name: str,
) -> tuple[Path, bool]:
    images_root = dataset_dir / "images"
    split_dir = images_root / split_name
    shared_split_dir = shared_images_dir / split_name
    shared_split_dir.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)

    if split_dir.is_symlink():
        if split_dir.exists():
            return shared_split_dir, True
        split_dir.unlink()

    if split_dir.exists():
        if split_dir.is_dir() and not any(split_dir.iterdir()):
            split_dir.rmdir()
        else:
            return split_dir, False

    try:
        split_dir.symlink_to(shared_split_dir, target_is_directory=True)
        return shared_split_dir, True
    except OSError:
        split_dir.mkdir(parents=True, exist_ok=True)
        return split_dir, False


def export_yolo(
    df: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
    method: str | None = None,
    dataset_dir: Path | None = None,
) -> Path:
    method = method or args.mask_method
    dataset_dir = dataset_dir or resolve_dataset_dir(args, method, multi=False)
    train_df, val_df = get_train_val_split(df, args)
    split_sets = [("train", train_df), ("val", val_df)]

    share_images = bool(args.share_images)
    shared_images_dir = args.shared_images_dir or (args.root / "dataset_yolo" / "images")
    if not shared_images_dir.is_absolute():
        shared_images_dir = args.root / shared_images_dir

    for split_name, split_df in split_sets:
        img_dir = dataset_dir / "images" / split_name
        lbl_dir = dataset_dir / "labels" / split_name
        if share_images:
            img_write_dir, _ = _prepare_image_dir(dataset_dir, shared_images_dir, split_name)
        else:
            img_dir.mkdir(parents=True, exist_ok=True)
            img_write_dir = img_dir
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for _, row in split_df.iterrows():
            rgb_img, thermal_img = load_pair(output_dir, row)
            mask_img = generate_mask_by_method(rgb_img, thermal_img, method, args)
            polys = mask_to_polygons(mask_img)

            img_name = Path(row["rgb_path"]).name
            Image.fromarray(rgb_img).save(img_write_dir / img_name)
            label_path = lbl_dir / f"{Path(img_name).stem}.txt"
            write_label(polys, rgb_img.shape[1], rgb_img.shape[0], label_path)

    dataset_yaml = dataset_dir / "dataset.yaml"
    dataset_yaml.write_text(
        "path: "
        + str(dataset_dir)
        + "\ntrain: images/train\nval: images/val\nnames:\n  0: cookware\n"
    )
    print(f"Export complete: {dataset_dir}")
    return dataset_dir


def dataset_ready(dataset_dir: Path) -> bool:
    return (dataset_dir / "dataset.yaml").exists()
