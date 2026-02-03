"""README parameter table updater."""
from __future__ import annotations

import argparse


def update_readme(args: argparse.Namespace) -> None:
    readme_path = args.root / "README.md"
    start_marker = "<!-- NOTEBOOK_PARAMS_START -->"
    end_marker = "<!-- NOTEBOOK_PARAMS_END -->"

    param_rows = [
        {
            "Parameter": "MASK_METHOD",
            "Applies To": "all",
            "Meaning": "Select mask generation method",
            "Notes": "`thermal`, `thermal_cluster`, `sam`, `sam_v2`, `sam_v3`, `sam_v4`",
        },
        {
            "Parameter": "THERMAL_LOW",
            "Applies To": "thermal, sam_v2",
            "Meaning": "Normalized threshold (0–1)",
            "Notes": "higher ⇒ smaller hot region",
        },
        {
            "Parameter": "CLUSTER_K",
            "Applies To": "thermal_cluster",
            "Meaning": "Number of k-means clusters",
            "Notes": "typical 2–4",
        },
        {
            "Parameter": "CLUSTER_ITERS",
            "Applies To": "thermal_cluster",
            "Meaning": "K-means iterations",
            "Notes": "more ⇒ stable clusters",
        },
        {
            "Parameter": "CLUSTER_MIN_RATIO",
            "Applies To": "thermal_cluster",
            "Meaning": "Minimum cluster size fraction",
            "Notes": "fallback to `thermal` if too small",
        },
        {
            "Parameter": "SAM_MODEL_TYPE",
            "Applies To": "sam, sam_v2, sam_v3, sam_v4",
            "Meaning": "SAM backbone (`vit_b`, `vit_l`, `vit_h`)",
            "Notes": "must match checkpoint",
        },
        {
            "Parameter": "SAM_TOPK",
            "Applies To": "sam_v2",
            "Meaning": "Number of hottest points as prompts",
            "Notes": "larger ⇒ more guidance",
        },
        {
            "Parameter": "SAM_LOW",
            "Applies To": "sam_v2",
            "Meaning": "Thermal threshold for SAM prompts",
            "Notes": "usually = `THERMAL_LOW`",
        },
        {
            "Parameter": "SAM_AUTO_MIN_AREA",
            "Applies To": "sam",
            "Meaning": "SAM auto-mask min region area",
            "Notes": "None uses SAM default",
        },
        {
            "Parameter": "SAM_AUTO_PRED_IOU",
            "Applies To": "sam",
            "Meaning": "SAM auto-mask pred_iou_thresh",
            "Notes": "None uses SAM default",
        },
        {
            "Parameter": "SAM_AUTO_STABILITY",
            "Applies To": "sam",
            "Meaning": "SAM auto-mask stability_score_thresh",
            "Notes": "None uses SAM default",
        },
        {
            "Parameter": "SAM_V3_MIN_AREA",
            "Applies To": "sam_v3",
            "Meaning": "SAM v3 min_mask_region_area",
            "Notes": "higher filters noise",
        },
        {
            "Parameter": "SAM_V3_PRED_IOU",
            "Applies To": "sam_v3",
            "Meaning": "SAM v3 pred_iou_thresh",
            "Notes": "higher = stricter",
        },
        {
            "Parameter": "SAM_V3_STABILITY",
            "Applies To": "sam_v3",
            "Meaning": "SAM v3 stability_score_thresh",
            "Notes": "higher = stricter",
        },
        {
            "Parameter": "SAM_V3_CENTER_FRAC",
            "Applies To": "sam_v3",
            "Meaning": "Center filter box fraction",
            "Notes": "0–1 of image size",
        },
        {
            "Parameter": "DINO_CONFIG",
            "Applies To": "sam_v4",
            "Meaning": "Grounding DINO config path",
            "Notes": "required",
        },
        {
            "Parameter": "DINO_CHECKPOINT",
            "Applies To": "sam_v4",
            "Meaning": "Grounding DINO checkpoint path",
            "Notes": "required",
        },
        {
            "Parameter": "DINO_TEXT_PROMPT",
            "Applies To": "sam_v4",
            "Meaning": "Grounding DINO text prompt",
            "Notes": "e.g. pan/wok",
        },
        {
            "Parameter": "DINO_BOX_THRESHOLD",
            "Applies To": "sam_v4",
            "Meaning": "Grounding DINO box score threshold",
            "Notes": "higher = stricter",
        },
        {
            "Parameter": "DINO_TEXT_THRESHOLD",
            "Applies To": "sam_v4",
            "Meaning": "Grounding DINO text score threshold",
            "Notes": "higher = stricter",
        },
        {
            "Parameter": "DATASET_DIR",
            "Applies To": "all",
            "Meaning": "Output dataset folder",
            "Notes": "default `dataset_yolo_<method>`",
        },
        {
            "Parameter": "VAL_RATIO",
            "Applies To": "export/train",
            "Meaning": "Validation split ratio",
            "Notes": "0.1=1/10, 0.05=1/20",
        },
        {
            "Parameter": "SPLIT_SEED",
            "Applies To": "export/train",
            "Meaning": "Random seed for train/val split",
            "Notes": "keeps same split across methods",
        },
        {
            "Parameter": "SPLIT_FILE",
            "Applies To": "export/train",
            "Meaning": "Saved train/val split CSV",
            "Notes": "defaults to output/split_train_val.csv",
        },
        {
            "Parameter": "REFRESH_SPLIT",
            "Applies To": "export/train",
            "Meaning": "Regenerate train/val split",
            "Notes": "overwrites saved split",
        },
        {
            "Parameter": "METHODS",
            "Applies To": "multi-run",
            "Meaning": "Methods used for export/train-all",
            "Notes": "comma-separated list",
        },
        {
            "Parameter": "COMPARE_METHODS",
            "Applies To": "comparison",
            "Meaning": "Methods to show in side-by-side plot",
            "Notes": "list of method names",
        },
        {
            "Parameter": "COMPARE_ROW_IDX",
            "Applies To": "comparison",
            "Meaning": "Row index from index.csv to preview",
            "Notes": "default 0",
        },
        {
            "Parameter": "COMPARE_OVERLAY",
            "Applies To": "comparison",
            "Meaning": "Overlay mask on RGB instead of raw mask",
            "Notes": "`True`/`False`",
        },
        {
            "Parameter": "COMPARE_ALPHA",
            "Applies To": "comparison",
            "Meaning": "Mask overlay alpha",
            "Notes": "0–1",
        },
        {
            "Parameter": "EVAL_METHODS",
            "Applies To": "metrics",
            "Meaning": "Methods to evaluate",
            "Notes": "defaults to `COMPARE_METHODS`",
        },
        {
            "Parameter": "EVAL_MAX_SAMPLES",
            "Applies To": "metrics",
            "Meaning": "Max samples to evaluate",
            "Notes": "subset for speed",
        },
        {
            "Parameter": "EVAL_SEED",
            "Applies To": "metrics",
            "Meaning": "Random seed for sampling",
            "Notes": "reproducible metrics",
        },
        {
            "Parameter": "TRAIN_ALL",
            "Applies To": "training",
            "Meaning": "Train YOLO for all methods",
            "Notes": "creates per-method runs",
        },
        {
            "Parameter": "TRAIN_SKIP_EXISTING",
            "Applies To": "training",
            "Meaning": "Skip training if results.csv exists",
            "Notes": "keeps prior runs, still reports metrics",
        },
        {
            "Parameter": "TRAINING_OUT",
            "Applies To": "training",
            "Meaning": "Save training summary CSV",
            "Notes": "optional output path",
        },
    ]

    header = "| Parameter | Applies To | Meaning | Notes |"
    sep = "| --- | --- | --- | --- |"
    lines = [header, sep]
    for row in param_rows:
        lines.append("| {Parameter} | {Applies To} | {Meaning} | {Notes} |".format(**row))
    table_md = "\n".join(lines)

    readme_text = readme_path.read_text()
    if start_marker not in readme_text or end_marker not in readme_text:
        raise ValueError("README markers not found. Insert NOTEBOOK_PARAMS_START/END first.")

    new_block = f"{start_marker}\n{table_md}\n{end_marker}"
    start_idx = readme_text.index(start_marker)
    end_idx = readme_text.index(end_marker) + len(end_marker)
    updated_readme = readme_text[:start_idx] + new_block + readme_text[end_idx:]
    readme_path.write_text(updated_readme)
    print("README parameter summary updated.")
