#!/usr/bin/env bash
# Download SAM, HQ-SAM, and SAM 2 checkpoints

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/data3/TC/ros2_cooking_perception}"
WEIGHTS_DIR="${ROOT_DIR}/weights"
mkdir -p "$WEIGHTS_DIR"

echo "Downloading SAM family checkpoints to $WEIGHTS_DIR..."

# SAM (ViT-B)
SAM_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_FILE="$WEIGHTS_DIR/sam_vit_b_01ec64.pth"
if [ -f "$SAM_FILE" ]; then
    echo "✓ SAM ViT-B already exists"
else
    echo "→ Downloading SAM ViT-B..."
    wget -O "$SAM_FILE" "$SAM_URL" || curl -L -o "$SAM_FILE" "$SAM_URL"
    echo "✓ SAM ViT-B downloaded"
fi

# HQ-SAM
HQ_SAM_URL="https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth"
HQ_SAM_FILE="$WEIGHTS_DIR/sam_hq_vit_b.pth"
if [ -f "$HQ_SAM_FILE" ]; then
    echo "✓ HQ-SAM already exists"
else
    echo "→ Downloading HQ-SAM..."
    wget -O "$HQ_SAM_FILE" "$HQ_SAM_URL" || curl -L -o "$HQ_SAM_FILE" "$HQ_SAM_URL"
    echo "✓ HQ-SAM downloaded"
fi

# SAM 2 Large
SAM2_URL="https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
SAM2_FILE="$WEIGHTS_DIR/sam2_hiera_large.pt"
if [ -f "$SAM2_FILE" ]; then
    echo "✓ SAM 2 Large already exists"
else
    echo "→ Downloading SAM 2 Large..."
    wget -O "$SAM2_FILE" "$SAM2_URL" || curl -L -o "$SAM2_FILE" "$SAM2_URL"
    echo "✓ SAM 2 Large downloaded"
fi

echo ""
echo "All checkpoints ready in $WEIGHTS_DIR:"
ls -lh "$WEIGHTS_DIR"
