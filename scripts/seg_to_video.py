#!/usr/bin/env python3
"""
Reassemble train/val frames into a video with YOLO-seg inference overlay.

Usage:
    python3 scripts/seg_to_video.py \
        --images-dir dataset_yolo/images \
        --model runs/segment/seg_manual/v2/weights/best.pt \
        --output output/seg_inference.mp4 \
        --imgsz 320 --conf 0.25 --device 0 --fps 30
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


# ── helpers ──────────────────────────────────────────────────────────────────

def overlay_multiclass_mask(
    image: np.ndarray, mask: np.ndarray, alpha: float = 0.45
) -> np.ndarray:
    """Semi-transparent coloured mask + contours for each class ID."""
    if mask is None or mask.sum() == 0:
        return image
    overlay = image.copy()
    cmap = plt.get_cmap("tab10")
    for uid in np.unique(mask):
        if uid == 0:
            continue
        color_rgb = cmap((uid - 1) % 10)[:3]
        color_bgr = tuple(int(c * 255) for c in color_rgb[::-1])  # BGR for cv2
        class_mask = (mask == uid).astype(np.uint8)
        overlay[class_mask > 0] = color_bgr
        contours, _ = cv2.findContours(
            class_mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def draw_labels(
    image: np.ndarray, result, class_names: dict[int, str]
) -> np.ndarray:
    """Draw bounding-box labels (class + confidence) on the image."""
    if result.boxes is None:
        return image
    boxes = result.boxes
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        conf = boxes.conf[i].item()
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
        label = f"{class_names.get(cls_id, str(cls_id))} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, max(y1 - th - 6, 0)), (x1 + tw, y1), (0, 0, 0), -1)
        cv2.putText(
            image, label, (x1, max(y1 - 4, th + 2)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return image


def yolo_seg_mask(result, h: int, w: int) -> np.ndarray:
    """Convert a single YOLO result to an (H, W) int32 multi-class mask."""
    if result.masks is None:
        return np.zeros((h, w), dtype=np.int32)
    masks = result.masks.data.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    order = np.argsort(-confs)
    final = np.zeros((h, w), dtype=np.int32)
    for idx in order:
        m = cv2.resize(masks[idx], (w, h), interpolation=cv2.INTER_NEAREST)
        final[(final == 0) & (m > 0.5)] = int(cls_ids[idx]) + 1
    return final


def collect_frames(images_dir: Path) -> list[Path]:
    """Gather all frames from train/ and val/, sorted by frame index."""
    all_files: list[Path] = []
    for split in ("train", "val"):
        split_dir = images_dir / split
        if split_dir.is_dir():
            all_files.extend(split_dir.glob("*.png"))
    # Sort by the leading frame index in the filename
    all_files.sort(key=lambda p: int(p.stem.split("_")[0]))
    return all_files


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Reassemble dataset frames into a video with YOLO-seg overlay."
    )
    parser.add_argument(
        "--images-dir", type=Path,
        default=Path("dataset_yolo/images"),
        help="Directory containing train/ and val/ image folders",
    )
    parser.add_argument(
        "--model", type=Path,
        default=Path("runs/segment/seg_manual/v2/weights/best.pt"),
        help="Path to YOLO segmentation model weights",
    )
    parser.add_argument("--output", type=Path, default=Path("output/seg_inference.mp4"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay transparency")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames (for testing)")
    args = parser.parse_args()

    # resolve relative paths against CWD
    images_dir = args.images_dir.resolve()
    model_path = args.model.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # collect & sort frames
    frames = collect_frames(images_dir)
    if not frames:
        raise FileNotFoundError(f"No .png frames found under {images_dir}/train or val")
    if args.max_frames:
        frames = frames[: args.max_frames]
    print(f"Found {len(frames)} frames, writing to {output_path}")

    # load model
    model = YOLO(str(model_path))
    class_names: dict[int, str] = model.names  # type: ignore[assignment]
    print(f"Classes: {class_names}")

    # probe first frame for resolution
    sample = cv2.imread(str(frames[0]))
    h, w = sample.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (w, h))

    # track inference times
    inference_times: list[float] = []

    for i, fpath in enumerate(frames):
        bgr = cv2.imread(str(fpath))
        if bgr is None:
            print(f"[WARN] cannot read {fpath}, skipping")
            continue

        # inference (YOLO expects BGR or RGB — ultralytics handles it)
        start = time.perf_counter()
        results = model.predict(
            bgr, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
            device=args.device, verbose=False,
        )
        elapsed = time.perf_counter() - start
        inference_times.append(elapsed)
        result = results[0]

        # build mask & overlay
        mask = yolo_seg_mask(result, h, w)
        vis = overlay_multiclass_mask(bgr, mask, alpha=args.alpha)
        vis = draw_labels(vis, result, class_names)

        # frame counter
        cv2.putText(
            vis, f"#{int(fpath.stem.split('_')[0]):05d}", (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
        )

        writer.write(vis)
        if (i + 1) % 200 == 0 or i == len(frames) - 1:
            print(f"  [{i+1}/{len(frames)}]")

    writer.release()
    avg_time = sum(inference_times) / len(inference_times) if inference_times else 0
    print("\nInference timing:")
    print(f"  Frames processed: {len(inference_times)}")
    print(f"  Total inference time: {sum(inference_times):.3f}s")
    print(f"  Average per frame: {avg_time*1000:.2f} ms ({1/avg_time:.2f} FPS)")
    print(f"Done — saved {output_path}  ({len(frames)} frames, {args.fps} fps)")


if __name__ == "__main__":
    main()
