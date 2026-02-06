#!/bin/bash

# ==============================================================================
# 設定偵測目標 (Classes)
# 邏輯說明：
# 1. Hot Keywords (wok, pot, pan): 程式會強制檢查 Thermal 重疊，避免抓到背景冷鍋。
# 2. Cold Keywords (egg, spatula...): 程式會直接信任 YOLO 的位置，跳過熱驗證。
# ==============================================================================
CLASSES=""black wok,cooking pot,egg,spatula,bowl,container""

# 設定 YOLO 模型 (用於產生冷物件的 Prompt)
# 建議使用 'yolov8s-world.pt' (速度快) 或 'yolov8l-worldv2.pt' (精度高)
YOLO_MODEL="yolov8s-world.pt"
YOLO_CONF="0.10"  # 設低一點，讓 SAM 負責修整邊緣，避免漏抓小蛋液

# ==============================================================================
# 1. 純 Thermal (Baseline)
# 僅能抓到高溫區域，無法偵測蛋或鏟子
# ==============================================================================
echo "Running Thermal Baseline..."
python3 scripts/seg_export.py --mask-method thermal --thermal-low 0.6

# ==============================================================================
# 2. Thermal Clustering (Baseline)
# 形狀比純閾值好一點，但依然無法偵測冷物件
# ==============================================================================
echo "Running Thermal Cluster..."
python3 scripts/seg_export.py --mask-method thermal_cluster --cluster-k 3 --cluster-iters 10

# ==============================================================================
# 3. YOLO-World + MobileSAM (純偵測流)
# 邏輯：YOLO Detect -> Filter Hot Boxes -> MobileSAM Segment
# 優點：速度最快，但分割精細度不如下面的 Hybrid Prompting
# ==============================================================================
echo "Running YOLO-World Direct..."
python3 scripts/seg_export.py \
  --mask-method yolo_world_sam \
  --yolo-world-model yolov8l-worldv2.pt \
  --sam2-checkpoint weights/sam2_hiera_tiny.pt \
  --sam2-config sam2_hiera_t.yaml \
  --yolo-world-classes "black iron wok,cooking pot,fried egg,yellow egg yolk,food,wooden handle spatula,small metal bowl,container" \
  --yolo-world-conf 0.05

# ==============================================================================
# 4. Naive SAM (原 sam_v2, 改良版)
# 邏輯：[Thermal Prompts (Hot) + YOLO Prompts (Cold)] -> Standard SAM (ViT-B)
# 優點：速度尚可，標準 SAM 權重容易取得
# ==============================================================================
echo "Running Naive SAM (Hybrid)..."
python3 scripts/seg_export.py \
  --mask-method naive_sam \
  --sam-model-type vit_b \
  --sam-checkpoint weights/sam_vit_b_01ec64.pth \
  --sam-low 0.6 \
  --sam-topk 20 \
  --yolo-world-model yolov8s-world.pt\
  --yolo-world-classes "black wok,cooking pot,egg,spatula,bowl,container" \
  --yolo-world-conf 0.10

# ==============================================================================
# 5. HQ-SAM (推薦：內部結構修復最強)
# 邏輯：[Thermal Prompts (Hot) + YOLO Prompts (Cold)] -> HQ-SAM
# 優點：對於鍋內的蛋液、鍋鏟邊緣，HQ-Token 能切得更細緻，少破洞
# ==============================================================================
echo "Running HQ-SAM (Hybrid)..."
python3 scripts/seg_export.py \
  --mask-method hq_sam \
  --sam-model-type vit_b \
  --hq-sam-checkpoint weights/sam_hq_vit_b.pth \
  --sam-low 0.6 \
  --sam-topk 20 \
  --yolo-world-model yolov8s-world.pt\
  --yolo-world-classes "black wok,cooking pot,egg,spatula,bowl,container" \
  --yolo-world-conf 0.10

# ==============================================================================
# 6. SAM 2 (推薦：魯棒性最強)
# 邏輯：[Thermal Prompts (Hot) + YOLO Prompts (Cold)] -> SAM 2
# 優點：Hiera Encoder 對反光和複雜背景的抗性最好
# ==============================================================================
echo "Running SAM 2 (Hybrid)..."
python3 scripts/seg_export.py \
  --mask-method sam2 \
  --sam2-config sam2_hiera_l.yaml \
  --sam2-checkpoint weights/sam2_hiera_large.pt \
  --sam-low 0.6 \
  --sam-topk 20 \
  --yolo-world-model yolov8l-worldv2.pt \
  --yolo-world-classes "black iron wok,cooking pot,fried egg,yellow egg yolk,food,wooden handle spatula,small metal bowl,container" \
  --yolo-world-conf 0.05

# ==============================================================================
# 7. 視覺化比較 (Visualization)
# 比較所有方法在同一張圖上的表現
# ==============================================================================
echo "Generating Comparison Grid..."
python scripts/seg_visualize.py --methods yolo_world_sam,sam2 --output-dir output

# ==============================================================================
# 8. 匯出 YOLO 訓練格式 (Export)
# 如果你決定用 HQ-SAM 的結果來訓練最終模型
# ==============================================================================
# echo "Exporting Dataset for Training..."
# python3 scripts/seg_export.py \
#   --mask-method hq_sam \
#   --yolo-world-classes "black wok,cooking pot,egg,spatula,bowl,container" \
#   --val-ratio 0.1 \
#   --split-seed 42 \
#   --refresh-split