#!/bin/bash
set -e

# ==============================================================================
# 手標資料 YOLO11n-seg 訓練 + 推論流程
# 注意：scripts/seg_visualize.py 需要 output/index.csv (extract_rgbt_bag.py 產生)
# ==============================================================================

ROOT="/data3/TC/ros2_cooking_perception"
OUTPUT_DIR="$ROOT/output"
DATASET_YAML="$ROOT/dataset_yolo/manual/dataset.yaml"

# 類別 (dataset.yaml 中的順序)
CLASSES="black wok,wooden handle spatula,fried egg,stainless steel bowl,thermometer gun"

# ==============================================================================
# 1. 用手標資料訓練 YOLO11n-seg
# ==============================================================================
echo "Training YOLO11n-seg on manual labels..."
python3 scripts/seg_train.py \
  --root /data3/TC/ros2_cooking_perception \
  --train \
  --train-data dataset_yolo/manual/dataset.yaml \
  --mask-method manual \
  --yolo-model yolo11n-seg.pt \
  --epochs 100 \
  --imgsz 320 \
  --batch 8 \
  --device 0 \
  --train-name-prefix seg \
  --train-exist-ok

# 如果想用自己的 run name，可以改 --train-name-prefix 或 --mask-method
# 訓練輸出預設在: $ROOT/runs/segment/seg_manual/weights/best.pt

# ==============================================================================
# SAM 2 + Qwen2.5-VL (最強 Reasoning 組合)
# 注意：第一次執行會下載 ~15GB 模型，請確保網路與磁碟空間
# ==============================================================================
echo "Running SAM 2 + Qwen2.5-VL..."
python3 scripts/seg_export.py \
  --mask-method qwen_sam \
  --sam2-config sam2_hiera_l.yaml \
  --sam2-checkpoint weights/sam2_hiera_large.pt \
  --qwen-model Qwen/Qwen2.5-VL-7B-Instruct \
  --yolo-world-classes "black wok,cooking pot,fried egg,raw egg yolk,transparent egg white,wooden handle spatula,stainless steel bowl,container" \
  --thermal-low 0.6 \
  --device 0

# ==============================================================================
# 2. 用訓練後模型在 val split 推論並和 SAM2 系列比較
# ==============================================================================
echo "Visualizing val split (YOLO-seg vs SAM2 methods)..."
python3 scripts/seg_visualize.py \
  --root /data3/TC/ros2_cooking_perception \
  --output-dir output \
  --methods yolo_seg,sam2_sahi \
  --viz-split val \
  --viz-samples 20 \
  --viz-out output/compare_val.jpg \
  --yolo-seg-model runs/segment/seg_manual/weights/best.pt \
  --yolo-seg-imgsz 320 \
  --yolo-world-classes "black wok,wooden handle spatula,fried egg,stainless steel bowl,thermometer gun" \
  --sam2-config sam2_hiera_l.yaml \
  --sam2-checkpoint weights/sam2_hiera_large.pt \
  --dino-config third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --dino-checkpoint weights/groundingdino_swint_ogc.pth \
  --dino-box-threshold 0.30 \
  --dino-text-threshold 0.25 \
  --sahi-slice-size 320 \
  --sahi-overlap 0.2 \
  --thermal-low 0.6

# 只看 YOLO-seg (不比較其他方法)：
# python3 scripts/seg_visualize.py \
#   --root "$ROOT" \
#   --output-dir "$OUTPUT_DIR" \
#   --methods yolo_seg \
#   --viz-split val \
#   --viz-samples 20 \
#   --yolo-seg-model "$BEST_PT" \
#   --yolo-seg-imgsz 320