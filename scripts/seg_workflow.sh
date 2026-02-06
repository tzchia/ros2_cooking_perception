# 1. 純 Thermal (閾值切分)
python3 scripts/seg_export.py --mask-method thermal --thermal-low 0.6

# 2. Thermal Clustering (K-Means 聚類，形狀更完整)
python3 scripts/seg_export.py --mask-method thermal_cluster --cluster-k 3 --cluster-iters 10

# 3. SAM v2 (使用標準 SAM ViT-B)
# 邏輯：Thermal Points/Box -> SAM -> Thermal IoU 篩選
python3 scripts/seg_export.py \
  --mask-method sam_v2 \
  --sam-model-type vit_b \
  --sam-checkpoint weights/sam_vit_b_01ec64.pth \
  --sam-low 0.6 \
  --sam-topk 20

# 4. HQ-SAM (推薦：修復內部孔洞最強)
# 邏輯：同上，但使用 HQ Token 修復複雜拓樸
python3 scripts/seg_export.py \
  --mask-method hq_sam \
  --sam-model-type vit_b \
  --hq-sam-checkpoint weights/sam_hq_vit_b.pth \
  --sam-low 0.6 \
  --sam-topk 20

# 5. SAM 2 (推薦：魯棒性最強，不易受反光干擾)
# 邏輯：Hiera Encoder + Thermal Verification
python3 scripts/seg_export.py \
  --mask-method sam2 \
  --sam2-config sam2_hiera_l.yaml \
  --sam2-checkpoint weights/sam2_hiera_large.pt \
  --sam-low 0.6 \
  --sam-topk 20

# 6. GroundingDINO (優化版)
# 邏輯：Text -> Box -> Thermal IoU Filter -> SAM -> Mask
python3 scripts/seg_export.py \
  --mask-method groundingdino \
  --dino-config third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --dino-checkpoint weights/groundingdino_swint_ogc.pth \
  --dino-text-prompt "black wok" \
  --dino-box-threshold 0.35

# 7. Florence-2 (優化版)
# 邏輯：Text -> Polygon -> Thermal Overlap & Area Check
python3 scripts/seg_export.py \
  --mask-method florence2 \
  --florence2-model-id microsoft/Florence-2-large \
  --florence2-text-prompt "black wok"

# 8. YOLO-World (解決紅海現象版)
# 邏輯：Detect -> Area Ratio Filter (< 2.5x Thermal Area) -> SAM
# 建議：改用 yolov8l-worldv2.pt (Large) 並且 Prompt 寫詳細一點
python3 scripts/seg_export.py \
  --mask-method yolo_world_sam \
  --yolo-world-model yolov8l-worldv2.pt \
  --yolo-world-sam-model mobile_sam.pt \
  --yolo-world-classes "round black metal wok" \
  --yolo-world-conf 0.15 \
  --device 0

# All methods including new ones
python3 scripts/seg_export.py --export-all

# Visualize all methods
python3 scripts/seg_visualize.py \
  --methods thermal,thermal_cluster,sam_v2,sam2,hq_sam,yolo_world_sam \
  --output-dir output


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
