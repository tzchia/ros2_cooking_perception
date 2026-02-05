#!/usr/bin/env bash
set -euo pipefail

# Step-by-step workflow for copy/paste usage.
# Copy each block into the terminal as needed.

# 0) Common variables (run once)
SPLIT_SEED=${SPLIT_SEED:-0}
METHODS=${METHODS:-thermal,thermal_cluster,sam,sam_v2,sam_v3,groundingdino}
EXTRA_ARGS=${EXTRA_ARGS:-}

# Example for custom args:
# METHOD=groundingdino
# EXTRA_ARGS="--dino-checkpoint ${ROOT_DIR}/weights/groundingdino_swint_ogc.pth"

# 1) Export labels for a single method
python3 scripts/seg_export.py \
  --mask-method thermal \
  $EXTRA_ARGS

# 2) Export labels for all methods
python3 scripts/seg_export.py \
  --export-all \
  --methods "$METHODS" \
  --val-ratio "$VAL_RATIO" \
  --split-seed "$SPLIT_SEED" \
  $EXTRA_ARGS

# Florence-2
python3 scripts/seg_export.py --mask-method florence2

# YOLO-World + SAM
python3 scripts/seg_export.py --mask-method yolo_world_sam \
  --yolo-world-classes "black wok,cooking pot" \
  --yolo-world-conf 0.15

# All methods including new ones
python3 scripts/seg_export.py --export-all

# 3) Refresh train/val split then export single method
python3 scripts/seg_export.py \
  --mask-method "$METHOD" \
  --val-ratio "$VAL_RATIO" \
  --split-seed "$SPLIT_SEED" \
  --refresh-split \
  $EXTRA_ARGS

# 4) Visual comparison
python3 scripts/seg_benchmark.py --compare $EXTRA_ARGS

# 5) Metrics summary
python3 scripts/seg_benchmark.py --metrics $EXTRA_ARGS

# 6) Export all + metrics
python3 scripts/seg_benchmark.py \
  --label-all \
  --methods "$METHODS" \
  --val-ratio "$VAL_RATIO" \
  --split-seed "$SPLIT_SEED" \
  $EXTRA_ARGS

# 7) Train single method
python3 scripts/seg_train.py \
  --train \
  --mask-method "$METHOD" \
  $EXTRA_ARGS

# 8) Train all methods
python3 scripts/seg_train.py \
  --train-all \
  --methods "$METHODS" \
  $EXTRA_ARGS

# 9) Legacy all-in-one pipeline
python3 scripts/seg_pipeline.py $EXTRA_ARGS
