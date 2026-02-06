"""README parameter table updater."""
from __future__ import annotations

import argparse


def update_readme(args: argparse.Namespace) -> None:
    readme_path = args.root / "README.md"
    start_marker = ""
    end_marker = ""

    param_rows = [
        {
            "Parameter": "MASK_METHOD",
            "Applies To": "all",
            "Meaning": "Select mask generation method",
            "Notes": "`thermal`, `thermal_cluster`, `sam`, `sam_v2`, `sam2`, `hq_sam`, `groundingdino`",
        },
        {
            "Parameter": "THERMAL_LOW",
            "Applies To": "thermal, groundingdino",
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
            "Parameter": "SAM_MODEL_TYPE",
            "Applies To": "sam, sam_v2, sam_v3, hq_sam, groundingdino",
            "Meaning": "SAM backbone (`vit_b`, `vit_l`, `vit_h`)",
            "Notes": "must match checkpoint",
        },
        {
            "Parameter": "SAM_TOPK",
            "Applies To": "sam_v2, sam2, hq_sam",
            "Meaning": "Number of hottest points as prompts",
            "Notes": "larger ⇒ more guidance",
        },
        {
            "Parameter": "SAM_LOW",
            "Applies To": "sam_v2, sam2, hq_sam",
            "Meaning": "Thermal threshold for SAM prompts",
            "Notes": "usually = `THERMAL_LOW`",
        },
        {
            "Parameter": "HQ_SAM_CHECKPOINT",
            "Applies To": "hq_sam",
            "Meaning": "HQ-SAM weights path",
            "Notes": "default `weights/sam_hq_vit_b.pth`",
        },
        {
            "Parameter": "SAM2_CONFIG",
            "Applies To": "sam2",
            "Meaning": "SAM 2 config YAML",
            "Notes": "e.g., `sam2_hiera_l.yaml`",
        },
        {
            "Parameter": "SAM2_CHECKPOINT",
            "Applies To": "sam2",
            "Meaning": "SAM 2 weights path",
            "Notes": "default `weights/sam2_hiera_large.pt`",
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
            "Parameter": "DINO_CONFIG",
            "Applies To": "groundingdino",
            "Meaning": "Grounding DINO config path",
            "Notes": "required",
        },
        {
            "Parameter": "DINO_CHECKPOINT",
            "Applies To": "groundingdino",
            "Meaning": "Grounding DINO checkpoint path",
            "Notes": "required",
        },
        {
            "Parameter": "DINO_TEXT_PROMPT",
            "Applies To": "groundingdino",
            "Meaning": "Grounding DINO text prompt",
            "Notes": "e.g. pan/wok",
        },
        {
            "Parameter": "DINO_BOX_THRESHOLD",
            "Applies To": "groundingdino",
            "Meaning": "Grounding DINO box score threshold",
            "Notes": "higher = stricter",
        },
        {
            "Parameter": "DINO_TEXT_THRESHOLD",
            "Applies To": "groundingdino",
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
            "Parameter": "TRAIN_ALL",
            "Applies To": "training",
            "Meaning": "Train YOLO for all methods",
            "Notes": "creates per-method runs",
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