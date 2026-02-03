#!/usr/bin/env bash
set -euo pipefail

# Step-by-step workflow for copy/paste usage.
# Copy each block into the terminal as needed.

# 0) Common variables (run once)
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VAL_RATIO=${VAL_RATIO:-0.1}
SPLIT_SEED=${SPLIT_SEED:-0}
METHOD=${METHOD:-thermal_cluster}
METHODS=${METHODS:-thermal,thermal_cluster,sam,sam_v2,sam_v3,sam_v4}
EXTRA_ARGS=${EXTRA_ARGS:-}

# Example for custom args:
# METHOD=sam_v4
# EXTRA_ARGS="--dino-checkpoint ${ROOT_DIR}/weights/groundingdino_swint_ogc.pth"

# 1) Export labels for a single method
python3 "$ROOT_DIR/scripts/seg_export.py" \
  --mask-method "$METHOD" \
  --val-ratio "$VAL_RATIO" \
  --split-seed "$SPLIT_SEED" \
  $EXTRA_ARGS

# 2) Export labels for all methods
python3 "$ROOT_DIR/scripts/seg_export.py" \
  --export-all \
  --methods "$METHODS" \
  --val-ratio "$VAL_RATIO" \
  --split-seed "$SPLIT_SEED" \
  $EXTRA_ARGS

# 3) Refresh train/val split then export single method
python3 "$ROOT_DIR/scripts/seg_export.py" \
  --mask-method "$METHOD" \
  --val-ratio "$VAL_RATIO" \
  --split-seed "$SPLIT_SEED" \
  --refresh-split \
  $EXTRA_ARGS

# 4) Visual comparison
python3 "$ROOT_DIR/scripts/seg_benchmark.py" --compare $EXTRA_ARGS

# 5) Metrics summary
python3 "$ROOT_DIR/scripts/seg_benchmark.py" --metrics $EXTRA_ARGS

# 6) Export all + metrics
python3 "$ROOT_DIR/scripts/seg_benchmark.py" \
  --label-all \
  --methods "$METHODS" \
  --val-ratio "$VAL_RATIO" \
  --split-seed "$SPLIT_SEED" \
  $EXTRA_ARGS

# 7) Train single method
python3 "$ROOT_DIR/scripts/seg_train.py" \
  --train \
  --mask-method "$METHOD" \
  $EXTRA_ARGS

# 8) Train all methods
python3 "$ROOT_DIR/scripts/seg_train.py" \
  --train-all \
  --methods "$METHODS" \
  $EXTRA_ARGS

# 9) Legacy all-in-one pipeline
python3 "$ROOT_DIR/scripts/seg_pipeline.py" $EXTRA_ARGS
