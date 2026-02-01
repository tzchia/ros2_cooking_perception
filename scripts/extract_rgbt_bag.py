#!/usr/bin/env python3
"""Extract RGBT rosbag into RGB + thermal images.

Assumes /rgbt/rgbt/compressed contains PNG-compressed BGRA images,
where the alpha channel stores thermal intensity.
"""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

import numpy as np
from PIL import Image
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore


def normalize_gray(gray: np.ndarray) -> np.ndarray:
    """Normalize grayscale image to 0-255 per frame."""
    gmin = int(gray.min())
    gmax = int(gray.max())
    if gmax == gmin:
        return np.zeros_like(gray, dtype=np.uint8)
    scaled = (gray.astype(np.float32) - gmin) * (255.0 / (gmax - gmin))
    return scaled.clip(0, 255).astype(np.uint8)


def jet_colormap(gray_8u: np.ndarray) -> np.ndarray:
    """Apply a lightweight Jet-like colormap to 8-bit grayscale."""
    x = gray_8u.astype(np.float32) / 255.0
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1) * 255.0
    return rgb.astype(np.uint8)


def extract_bag(
    bag_path: Path,
    output_dir: Path,
    topic: str,
    max_frames: int | None,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = output_dir / "rgb"
    thermal_raw_dir = output_dir / "thermal_raw"
    thermal_color_dir = output_dir / "thermal_color"
    for d in (rgb_dir, thermal_raw_dir, thermal_color_dir):
        d.mkdir(parents=True, exist_ok=True)

    store = get_typestore(Stores.ROS2_FOXY)
    index_path = output_dir / "index.csv"

    count = 0
    with Reader(bag_path) as reader, index_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["timestamp_ns", "rgb_path", "thermal_raw_path", "thermal_color_path"]
        )

        for conn, timestamp, rawdata in reader.messages():
            if conn.topic != topic:
                continue

            msg = store.deserialize_cdr(rawdata, conn.msgtype)
            img = Image.open(io.BytesIO(msg.data))
            arr = np.array(img)

            if arr.ndim != 3 or arr.shape[2] < 4:
                raise ValueError(f"Expected 4-channel image, got shape {arr.shape}")

            # Use format (if available) to interpret channel order; default to RGBA.
            fmt = getattr(msg, "format", "").lower()
            if "bgra" in fmt or ("bgr" in fmt and "rgb" not in fmt):
                rgb = arr[..., [2, 1, 0]]
            else:
                rgb = arr[..., :3]
            thermal_raw = arr[..., 3].astype(np.uint8)
            thermal_norm = normalize_gray(thermal_raw)
            thermal_color = jet_colormap(thermal_norm)

            base = f"{count:06d}_{timestamp}"
            rgb_path = rgb_dir / f"{base}.png"
            thermal_raw_path = thermal_raw_dir / f"{base}.png"
            thermal_color_path = thermal_color_dir / f"{base}.png"

            Image.fromarray(rgb).save(rgb_path)
            Image.fromarray(thermal_raw).save(thermal_raw_path)
            Image.fromarray(thermal_color).save(thermal_color_path)

            writer.writerow(
                [
                    timestamp,
                    rgb_path.relative_to(output_dir),
                    thermal_raw_path.relative_to(output_dir),
                    thermal_color_path.relative_to(output_dir),
                ]
            )

            count += 1
            if max_frames is not None and count >= max_frames:
                break

    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract RGB + thermal images from a compressed RGBT rosbag."
    )
    parser.add_argument(
        "--bag",
        type=Path,
        default=Path("/data3/TC/ros2_cooking_perception/dataset"),
        help="Path to rosbag2 directory (containing .db3 and metadata.yaml).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data3/TC/ros2_cooking_perception/output"),
        help="Output directory for extracted images.",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="/rgbt/rgbt/compressed",
        help="Topic name containing the compressed RGBT images.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit on number of frames to extract.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = extract_bag(args.bag, args.output, args.topic, args.max_frames)
    print(f"Extracted {count} frames to {args.output}")


if __name__ == "__main__":
    main()
