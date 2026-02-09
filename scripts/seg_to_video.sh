#!/bin/bash
set -e

# ==============================================================================
# 把 train/val frames 串回影片，並用 YOLO-seg model 做 inference overlay
# ==============================================================================

ROOT="/data3/TC/ros2_cooking_perception"

python3 scripts/seg_to_video.py \
  --images-dir dataset_yolo/images \
  --model runs/segment/seg_manual/11n_v2/weights/best.pt \
  --output output/11n_v2_inference.mp4 \
  --imgsz 320 \
  --conf 0.25 \
  --iou 0.7 \
  --device 0 \
  --fps 30 \
  --alpha 0.45

