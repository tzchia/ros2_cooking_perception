# ROS2 Cooking Perception

## Project Progression (Milestones)

1. **Data extraction from RGBT bag** ✅
   - `scripts/extract_rgbt_bag.py` produces RGB + thermal frames and `index.csv`.
2. **Pseudo-label exploration** ✅
   - Thermal threshold, thermal + clustering, and SAM promptable masks (see notebook).
3. **Dataset export for training** ✅
   - YOLO segmentation polygons generated per method.
4. **Model training** 🟡
   - Ultralytics YOLOv8-seg baseline provided in notebook.
5. **Real-time inference node** ⏳
6. **Analysis & documentation** ⏳

## Repo Layout (Key Paths)

- `dataset/` ROS2 bag + metadata
- `output/` extracted frames and `index.csv`
- `scripts/` data extraction scripts
- `notebooks/train_segmentation.ipynb` full segmentation workflow

## RGBT Compression Assumption

`scripts/extract_rgbt_bag.py` reads **PNG-compressed** RGBT images from
`/rgbt/rgbt/compressed` (ROS2 `sensor_msgs/CompressedImage`). It **does not
perform compression**; it only decodes the PNG bytes, then splits RGB and the
thermal alpha channel. The channel order is determined by `msg.format` when
available (BGRA vs RGBA).

## Dataset Extraction

```bash
python3 scripts/extract_rgbt_bag.py \
  --bag /data3/TC/ros2_cooking_perception/dataset \
  --output /data3/TC/ros2_cooking_perception/output \
  --topic /rgbt/rgbt/compressed
```

Outputs:
- `output/rgb/` RGB frames
- `output/thermal_raw/` 8-bit thermal channel
- `output/thermal_color/` thermal visualization
- `output/index.csv` timestamp-to-file mapping

## Segmentation Training Notebook

Notebook: `notebooks/train_segmentation.ipynb` (run locally with Jupyter).

### Workflow Overview

1. Load RGB + thermal pairs from `output/`.
2. Select a mask generation method (`thermal`, `thermal_cluster`, `sam`, `sam_v2`, `sam_v3`, `groundingdino`).
3. Preview the mask quality.
4. Export YOLO segmentation labels to `dataset_yolo_<method>/`.
5. Train YOLOv8-seg with the exported labels.

### CLI Workflow (split across scripts)

We now persist a **shared train/val split** so that different methods reuse the
same images (labels differ by method). The split file is saved at:

- `output/split_train_val.csv`

Use `--refresh-split` to regenerate the split, or keep it to ensure
repeatability across methods.

**Export labels (single method):**

```bash
python3 scripts/seg_export.py \
  --mask-method groundingdino \
  --val-ratio 0.1 \
  --split-seed 0
```

**Export labels (all methods):**

```bash
python3 scripts/seg_export.py \
  --export-all \
  --val-ratio 0.1 \
  --split-seed 0
```

**Benchmark (compare / metrics / label-all):**

```bash
python3 scripts/seg_benchmark.py --compare
```

```bash
python3 scripts/seg_benchmark.py --metrics
```

```bash
python3 scripts/seg_benchmark.py --label-all --val-ratio 0.1 --split-seed 0
```

**Training:**

```bash
python3 scripts/seg_train.py --train --mask-method thermal_cluster
```

```bash
python3 scripts/seg_train.py --train-all
```

### Mask Methods

- **thermal**: fixed threshold on normalized thermal intensity.
- **thermal_cluster**: 1D k-means on thermal intensity, pick hottest cluster.
- **sam (v1)**: SAM automatic mask generator (no prompts) with optional thresholds.
- **sam_v2**: thermal-guided SAM (centroid + rough box prompt).
- **sam_v3**: tuned SAM auto masks + center/area filtering for noise suppression.
- **groundingdino**: Grounding DINO text prompt → box → SAM mask.

### Label Metrics (Heuristic)

We evaluate pseudo-label quality with two quick heuristics:

- **area_ratio**: fraction of pixels labeled (higher ⇒ larger masks).
- **components**: number of connected components per mask (lower ⇒ cleaner, less fragmented).

Latest run (2026-02-01):

| Method | area_ratio (mean) | components (mean) | Notes |
| --- | --- | --- | --- |
| sam | 0.3155 | 134.66 | High fragmentation; likely noisy without tuning |
| thermal | 0.2364 | 3.70 | Stable, but slightly more components |
| thermal_cluster | 0.2236 | 2.94 | **Cleanest** (lowest components) |

**Current best (heuristic):** `thermal_cluster` shows the cleanest masks with the lowest
fragmentation while keeping a similar area_ratio to `thermal`. `sam` produces larger
masks but is highly fragmented; it likely needs better prompts or post-processing.

> Note: These are **proxy metrics** without ground truth. The final choice should be
> validated by visual inspection and downstream training mAP.

### Parameters (set in the notebook)

<!-- NOTEBOOK_PARAMS_START -->
| Parameter | Applies To | Meaning | Notes |
| --- | --- | --- | --- |
| `MASK_METHOD` | all | Select mask generation method | `thermal`, `thermal_cluster`, `sam` |
| `THERMAL_LOW` | thermal, sam | Normalized threshold (0–1) | higher ⇒ smaller hot region |
| `CLUSTER_K` | thermal_cluster | Number of k-means clusters | typical 2–4 |
| `CLUSTER_ITERS` | thermal_cluster | K-means iterations | more ⇒ stable clusters |
| `CLUSTER_MIN_RATIO` | thermal_cluster | Minimum cluster size fraction | fallback to `thermal` if too small |
| `SAM_MODEL_TYPE` | sam | SAM backbone (`vit_b`, `vit_l`, `vit_h`) | must match checkpoint |
| `SAM_TOPK` | sam | Number of hottest points as prompts | larger ⇒ more guidance |
| `SAM_LOW` | sam | Thermal threshold for SAM prompts | usually = `THERMAL_LOW` |
| `DATASET_DIR` | all | Output dataset folder | default `dataset_yolo_<method>` |
<!-- NOTEBOOK_PARAMS_END -->

### SAM Setup

1. Install SAM: https://github.com/facebookresearch/segment-anything
2. Download a checkpoint (e.g., `sam_vit_b_01ec64.pth`) to `ROOT/weights/`.
3. Ensure `torch` is installed and CUDA is available for speed.

### Grounding DINO Setup (for `groundingdino`)

Run the helper script to clone + install Grounding DINO dependencies:

```bash
bash scripts/install_grounding_dino.sh
```

Then download a checkpoint (e.g. `groundingdino_swint_ogc.pth`) to `ROOT/weights/`
and pass the config/ckpt paths when running `groundingdino`.

## Library Survey

- **rclpy**: ROS2 Python client.
- **OpenCV**: Image processing & polygon conversion.
- **Ultralytics YOLO**: Instance segmentation training (YOLOv8-seg in notebook).
- **Segment Anything (SAM)**: Promptable segmentation for auto-labeling.
