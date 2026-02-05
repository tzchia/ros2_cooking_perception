"""Argument parsing and shared path helpers."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RGB+Thermal segmentation utilities")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/data3/TC/ros2_cooking_perception"),
        help="Project root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory containing index.csv (default: <root>/output)",
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=None,
        help="Path to index.csv (default: <output-dir>/index.csv)",
    )
    parser.add_argument(
        "--mask-method",
        type=str,
        default="thermal",
        choices=["thermal", "thermal_cluster", "sam", "sam_v2", "sam_v3", "groundingdino", "florence2", "yolo_world_sam"],
        help="Mask generation method",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="thermal,thermal_cluster,sam,sam_v2,sam_v3,groundingdino,florence2,yolo_world_sam",
        help="Comma-separated method list for multi-run actions",
    )
    parser.add_argument("--thermal-low", type=float, default=0.6)
    parser.add_argument("--cluster-k", type=int, default=3)
    parser.add_argument("--cluster-iters", type=int, default=10)
    parser.add_argument("--cluster-min-ratio", type=float, default=0.002)
    parser.add_argument("--sam-model-type", type=str, default="vit_b")
    parser.add_argument("--sam-topk", type=int, default=20)
    parser.add_argument("--sam-low", type=float, default=None)
    parser.add_argument(
        "--sam-auto-min-area",
        type=int,
        default=None,
        help="SAM v1 auto-mask min_mask_region_area (None = SAM default)",
    )
    parser.add_argument(
        "--sam-auto-pred-iou",
        type=float,
        default=None,
        help="SAM v1 auto-mask pred_iou_thresh (None = SAM default)",
    )
    parser.add_argument(
        "--sam-auto-stability",
        type=float,
        default=None,
        help="SAM v1 auto-mask stability_score_thresh (None = SAM default)",
    )
    parser.add_argument(
        "--sam-v3-min-area",
        type=int,
        default=3000,
        help="SAM v3 auto-mask min_mask_region_area",
    )
    parser.add_argument(
        "--sam-v3-pred-iou",
        type=float,
        default=0.92,
        help="SAM v3 auto-mask pred_iou_thresh",
    )
    parser.add_argument(
        "--sam-v3-stability",
        type=float,
        default=0.95,
        help="SAM v3 auto-mask stability_score_thresh",
    )
    parser.add_argument(
        "--sam-v3-center-frac",
        type=float,
        default=0.6,
        help="SAM v3 post-filter center box fraction (0–1)",
    )
    parser.add_argument(
        "--dino-config",
        type=Path,
        default="third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        help="Grounding DINO config path (required for groundingdino)",
    )
    parser.add_argument(
        "--dino-checkpoint",
        type=Path,
        default="weights/groundingdino_swint_ogc.pth",
        help="Grounding DINO checkpoint path (required for groundingdino)",
    )
    parser.add_argument(
        "--dino-text-prompt",
        type=str,
        default="black wok, cooking pot",
        help="Grounding DINO text prompt, e.g. 'pan' or 'wok'",
    )
    parser.add_argument("--dino-box-threshold", type=float, default=0.35) # original 0.35
    parser.add_argument("--dino-text-threshold", type=float, default=0.25)
    # Florence-2 arguments
    parser.add_argument(
        "--florence2-model-id",
        type=str,
        default="microsoft/Florence-2-large",
        help="Florence-2 model ID (required for florence2)",
    )
    parser.add_argument(
        "--florence2-text-prompt",
        type=str,
        default="black wok,cooking pot",
        help="Florence-2 text prompt for referring expression segmentation",
    )
    # YOLO-World + SAM arguments
    parser.add_argument(
        "--yolo-world-model",
        type=str,
        default="yolov8s-world.pt",
        help="YOLO-World model checkpoint (required for yolo_world_sam)",
    )
    parser.add_argument(
        "--yolo-world-sam-model",
        type=str,
        default="mobile_sam.pt",
        help="SAM model for YOLO-World segmentation (mobile_sam.pt or fastsam.pt)",
    )
    parser.add_argument(
        "--yolo-world-classes",
        type=str,
        default="black wok,cooking pot",
        help="Comma-separated classes for YOLO-World (e.g., 'black wok,cooking pot')",
    )
    parser.add_argument(
        "--yolo-world-conf",
        type=float,
        default=0.15,
        help="YOLO-World confidence threshold (lower = more detections)",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        default=None,
        help="SAM checkpoint path (default: <root>/weights/sam_vit_b_01ec64.pth)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Output dataset folder (default: <root>/dataset_yolo/<method>)",
    )
    parser.add_argument(
        "--share-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Share images in <root>/dataset_yolo/images and symlink per-method images/",
    )
    parser.add_argument(
        "--shared-images-dir",
        type=Path,
        default=Path("dataset_yolo/images"),
        help="Override shared images directory (default: <root>/dataset_yolo/images)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (e.g. 0.1=1/10, 0.05=1/20)",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="Random seed for train/val split",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="Path to saved train/val split CSV (default: <output-dir>/split_train_val.csv)",
    )
    parser.add_argument(
        "--refresh-split",
        action="store_true",
        help="Regenerate train/val split even if split file exists",
    )
    parser.add_argument("--compare", action="store_true", help="Run side-by-side plot")
    parser.add_argument(
        "--compare-methods",
        type=str,
        default="thermal,thermal_cluster,sam,sam_v2,sam_v3,groundingdino,florence2,yolo_world_sam",
        help="Comma-separated methods for comparison",
    )
    parser.add_argument("--compare-row-idx", type=int, default=0)
    parser.add_argument("--compare-overlay", action="store_true", default=True)
    parser.add_argument("--compare-alpha", type=float, default=0.4)
    parser.add_argument(
        "--compare-out",
        type=Path,
        default=None,
        help="Save comparison image to file (if omitted, shows a window)",
    )
    parser.add_argument("--metrics", action="store_true", help="Run metrics summary")
    parser.add_argument(
        "--eval-methods",
        type=str,
        default=None,
        help="Comma-separated methods for metrics (default: compare-methods)",
    )
    parser.add_argument("--eval-max-samples", type=int, default=50)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Save metrics summary CSV",
    )
    parser.add_argument(
        "--metrics-raw-out",
        type=Path,
        default=None,
        help="Save per-sample metrics CSV",
    )
    parser.add_argument("--export", action="store_true", help="Export YOLO dataset")
    parser.add_argument("--export-all", action="store_true", help="Export datasets for all methods")
    parser.add_argument(
        "--label-all",
        action="store_true",
        help="Export datasets for all methods and evaluate label metrics",
    )
    parser.add_argument("--train", action="store_true", help="Train YOLO for single method")
    parser.add_argument(
        "--train-all",
        action="store_true",
        help="Train YOLO for all methods and compare metrics",
    )
    parser.add_argument("--yolo-model", type=str, default="yolov8s-seg.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument(
        "--train-project",
        type=Path,
        default=None,
        help="Training project directory (default: <root>/runs/segment)",
    )
    parser.add_argument(
        "--train-name-prefix",
        type=str,
        default="seg",
        help="Run name prefix for training outputs",
    )
    parser.add_argument("--train-exist-ok", action="store_true")
    parser.add_argument(
        "--train-skip-existing",
        action="store_true",
        help="Skip training if results.csv exists for the method",
    )
    parser.add_argument(
        "--training-out",
        type=Path,
        default=None,
        help="Save training summary CSV",
    )
    parser.add_argument("--update-readme", action="store_true", help="Update README params table")
    return parser.parse_args()


def parse_method_list(methods_str: str) -> List[str]:
    return [m.strip() for m in methods_str.split(",") if m.strip()]


def resolve_dataset_dir(args: argparse.Namespace, method: str, multi: bool) -> Path:
    if args.dataset_dir:
        return args.dataset_dir / method if multi else args.dataset_dir
    return args.root / "dataset_yolo" / method
