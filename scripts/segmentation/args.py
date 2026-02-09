"""Argument parsing and shared path helpers (Added Qwen Support)."""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--root", type=Path, default=Path("/data3/TC/ros2_cooking_perception"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--index-csv", type=Path, default=None)
    parser.add_argument("--mask-method", type=str, default="thermal")
    parser.add_argument("--methods", type=str, default="sam2_sahi,qwen_sam") 
    
    # Thermal
    parser.add_argument("--thermal-low", type=float, default=0.6)
    parser.add_argument("--cluster-k", type=int, default=3)
    parser.add_argument("--cluster-iters", type=int, default=10)
    parser.add_argument("--cluster-min-ratio", type=float, default=0.002)
    
    # SAM Base
    parser.add_argument("--sam-model-type", type=str, default="vit_b")
    parser.add_argument("--sam-checkpoint", type=Path, default=None)
    parser.add_argument("--sam-topk", type=int, default=20)
    parser.add_argument("--sam-low", type=float, default=None)
    parser.add_argument("--sam-auto-min-area", type=int, default=None)
    parser.add_argument("--sam-auto-pred-iou", type=float, default=None)
    parser.add_argument("--sam-auto-stability", type=float, default=None)
    
    # HQ-SAM
    parser.add_argument("--hq-sam-checkpoint", type=Path, default=None)
    
    # SAM 2
    parser.add_argument("--sam2-config", type=str, default="sam2_hiera_l.yaml")
    parser.add_argument("--sam2-checkpoint", type=Path, default=None)
    
    # YOLO-World
    parser.add_argument("--yolo-world-model", type=str, default="yolov8l-worldv2.pt")
    parser.add_argument("--yolo-world-classes", type=str, default="black wok,cooking pot,raw egg yolk,transparent egg white,spatula,bowl")
    parser.add_argument("--yolo-world-conf", type=float, default=0.05)

    # YOLO-Seg
    parser.add_argument("--yolo-seg-model", type=Path, default=Path("yolo11n-seg.pt"))
    parser.add_argument("--yolo-seg-conf", type=float, default=0.25)
    parser.add_argument("--yolo-seg-iou", type=float, default=0.7)
    parser.add_argument("--yolo-seg-imgsz", type=int, default=None)
    parser.add_argument("--yolo-seg-max-det", type=int, default=100)
    
    # SAHI Params
    parser.add_argument("--sahi-slice-size", type=int, default=320)
    parser.add_argument("--sahi-overlap", type=float, default=0.2)
    
    # Grounding DINO Params
    parser.add_argument("--dino-config", type=Path, default="weights/groundingdino/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--dino-checkpoint", type=Path, default="weights/groundingdino/groundingdino_swint_ogc.pth")
    parser.add_argument("--dino-box-threshold", type=float, default=0.30)
    parser.add_argument("--dino-text-threshold", type=float, default=0.25)

    # --- NEW: Qwen-VL Params ---
    parser.add_argument("--qwen-model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="HuggingFace model ID for Qwen")
    parser.add_argument("--qwen-text-prompt", type=str, default="Detect the black wok, fried egg, spatula, and bowls.", help="Natural language prompt")
    
    # Data & Export
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--share-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shared-images-dir", type=Path, default=Path("dataset_yolo/images"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--refresh-split", action="store_true")
    
    # Actions
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--compare-methods", type=str, default="sam2_sahi,qwen_sam")
    parser.add_argument("--compare-row-idx", type=int, default=0)
    parser.add_argument("--compare-overlay", action="store_true", default=True)
    parser.add_argument("--compare-alpha", type=float, default=0.4)
    parser.add_argument("--compare-out", type=Path, default=None)
    
    # Metrics/Train/Export flags
    parser.add_argument("--metrics", action="store_true")
    parser.add_argument("--eval-methods", type=str, default=None)
    parser.add_argument("--eval-max-samples", type=int, default=50)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--metrics-out", type=Path, default=None)
    parser.add_argument("--metrics-raw-out", type=Path, default=None)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--export-all", action="store_true")
    parser.add_argument("--label-all", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--train-all", action="store_true")
    parser.add_argument("--yolo-model", type=str, default="yolov8s-seg.pt")
    parser.add_argument("--train-data", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--train-project", type=Path, default=None)
    parser.add_argument("--train-name-prefix", type=str, default="seg")
    parser.add_argument("--train-exist-ok", action="store_true")
    parser.add_argument("--train-skip-existing", action="store_true")
    parser.add_argument("--training-out", type=Path, default=None)
    parser.add_argument("--update-readme", action="store_true")

    # Visualization
    parser.add_argument("--viz-split", type=str, choices=["all", "train", "val"], default="all")
    parser.add_argument("--viz-samples", type=int, default=5)
    parser.add_argument("--viz-seed", type=int, default=42)
    parser.add_argument("--viz-out", type=Path, default=None)
    
    return parser.parse_args()

def parse_method_list(methods_str: str) -> List[str]:
    return [m.strip() for m in methods_str.split(",") if m.strip()]

def resolve_dataset_dir(args: argparse.Namespace, method: str, multi: bool) -> Path:
    if args.dataset_dir:
        return args.dataset_dir / method if multi else args.dataset_dir
    return args.root / "dataset_yolo" / method