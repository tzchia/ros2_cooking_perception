#!/usr/bin/env python3
"""Segmentation dataset prep for RGB + thermal data.

This script consolidates the notebook workflow into a runnable pipeline:
- mask generation (thermal / thermal + clustering / SAM)
- side-by-side mask comparison
- quick mask metrics (area ratio, connected components)
- YOLO segmentation export
- multi-method labeling + metrics comparison
- multi-method training + metrics comparison
- README parameter table update
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


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
        choices=["thermal", "thermal_cluster", "sam"],
        help="Mask generation method",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="thermal,thermal_cluster,sam",
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
        "--sam-checkpoint",
        type=Path,
        default=None,
        help="SAM checkpoint path (default: <root>/weights/sam_vit_b_01ec64.pth)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Output dataset folder (default: <root>/dataset_yolo_<method>)",
    )
    parser.add_argument("--compare", action="store_true", help="Run side-by-side plot")
    parser.add_argument(
        "--compare-methods",
        type=str,
        default="thermal,thermal_cluster,sam",
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
    return args.root / f"dataset_yolo_{method}"


def normalize_thermal(thermal_img: np.ndarray) -> np.ndarray:
    tmin = float(thermal_img.min())
    tmax = float(thermal_img.max())
    return (thermal_img - tmin) / (tmax - tmin + 1e-6)


def thermal_mask(thermal_img: np.ndarray, low: float = 0.6) -> np.ndarray:
    norm = normalize_thermal(thermal_img)
    return (norm >= low).astype(np.uint8)


def kmeans_1d(values: np.ndarray, k: int = 3, iters: int = 10) -> tuple[np.ndarray, np.ndarray]:
    centers = np.linspace(values.min(), values.max(), k)
    labels = np.zeros(len(values), dtype=np.int32)
    for _ in range(iters):
        dists = np.abs(values[:, None] - centers[None, :])
        labels = dists.argmin(axis=1)
        for idx in range(k):
            mask = labels == idx
            if np.any(mask):
                centers[idx] = values[mask].mean()
    return centers, labels


def thermal_cluster_mask(
    thermal_img: np.ndarray,
    k: int = 3,
    iters: int = 10,
    min_ratio: float = 0.002,
    low_fallback: float = 0.6,
) -> np.ndarray:
    norm = normalize_thermal(thermal_img)
    values = norm.flatten()
    centers, labels = kmeans_1d(values, k=k, iters=iters)
    hottest = np.argsort(centers)[-1]
    mask = labels.reshape(norm.shape) == hottest
    if mask.mean() < min_ratio:
        return thermal_mask(thermal_img, low=low_fallback)
    return mask.astype(np.uint8)


_SAM_PREDICTOR = None


def get_sam_predictor(root: Path, model_type: str, checkpoint: Path):
    global _SAM_PREDICTOR
    if _SAM_PREDICTOR is not None:
        return _SAM_PREDICTOR

    try:
        import torch
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError as exc:
        raise ImportError(
            "segment-anything is required. Install from https://github.com/facebookresearch/segment-anything"
        ) from exc

    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing SAM checkpoint: {checkpoint}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device)
    _SAM_PREDICTOR = SamPredictor(sam)
    return _SAM_PREDICTOR


def thermal_prompts(
    thermal_img: np.ndarray, low: float = 0.6, topk: int = 20
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    norm = normalize_thermal(thermal_img)
    mask = norm >= low
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None, None, None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    coords = np.column_stack([ys, xs])
    vals = norm[mask]
    top_idx = np.argsort(vals)[-min(topk, len(vals)) :]
    points = coords[top_idx][:, ::-1]
    labels = np.ones(len(points), dtype=np.int32)
    box = np.array([x1, y1, x2, y2])
    return points, labels, box


def sam_mask(
    rgb_img: np.ndarray,
    thermal_img: np.ndarray,
    root: Path,
    low: float = 0.6,
    topk: int = 20,
    model_type: str = "vit_b",
    checkpoint: Path | None = None,
) -> np.ndarray:
    points, labels, box = thermal_prompts(thermal_img, low=low, topk=topk)
    if points is None:
        return np.zeros(thermal_img.shape, dtype=np.uint8)
    predictor = get_sam_predictor(root, model_type=model_type, checkpoint=checkpoint)
    predictor.set_image(rgb_img)
    masks, _, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        box=box[None, :],
        multimask_output=False,
    )
    return masks[0].astype(np.uint8)


def load_pair(output_dir: Path, row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    rgb = Image.open(output_dir / row["rgb_path"])
    thermal = Image.open(output_dir / row["thermal_raw_path"])
    return np.array(rgb), np.array(thermal)


def generate_mask_by_method(
    rgb_img: np.ndarray,
    thermal_img: np.ndarray,
    method: str,
    args: argparse.Namespace,
) -> np.ndarray:
    if method == "thermal":
        return thermal_mask(thermal_img, low=args.thermal_low)
    if method == "thermal_cluster":
        return thermal_cluster_mask(
            thermal_img,
            k=args.cluster_k,
            iters=args.cluster_iters,
            min_ratio=args.cluster_min_ratio,
            low_fallback=args.thermal_low,
        )
    if method == "sam":
        sam_low = args.sam_low if args.sam_low is not None else args.thermal_low
        return sam_mask(
            rgb_img,
            thermal_img,
            root=args.root,
            low=sam_low,
            topk=args.sam_topk,
            model_type=args.sam_model_type,
            checkpoint=args.sam_checkpoint,
        )
    raise ValueError(f"Unknown method: {method}")


def overlay_mask(
    rgb_img: np.ndarray, mask_img: np.ndarray, alpha: float = 0.4, color=(255, 64, 64)
) -> np.ndarray:
    overlay = rgb_img.astype(np.float32).copy()
    color_arr = np.array(color, dtype=np.float32)
    mask_bool = mask_img.astype(bool)
    overlay[mask_bool] = (1 - alpha) * overlay[mask_bool] + alpha * color_arr
    return overlay.astype(np.uint8)


def compare_masks(df: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    methods = parse_method_list(args.compare_methods)
    row = df.iloc[args.compare_row_idx]
    rgb_cmp, thermal_cmp = load_pair(output_dir, row)

    fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 4))
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        mask_img = generate_mask_by_method(rgb_cmp, thermal_cmp, method, args)
        if args.compare_overlay:
            view = overlay_mask(rgb_cmp, mask_img, alpha=args.compare_alpha)
            ax.imshow(view)
        else:
            ax.imshow(mask_img, cmap="gray")
        ax.set_title(method)
        ax.axis("off")

    if args.compare_out:
        fig.savefig(args.compare_out, bbox_inches="tight", dpi=200)
        print(f"Saved comparison to {args.compare_out}")
    else:
        plt.show()


def mask_metrics(mask_img: np.ndarray) -> dict[str, float]:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required for metrics. Install opencv-python.") from exc

    area_ratio = mask_img.mean().item()
    if mask_img.dtype != np.uint8:
        mask_img = mask_img.astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(mask_img)
    return {"area_ratio": area_ratio, "components": max(num_labels - 1, 0)}


def evaluate_metrics(
    df: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
    methods: List[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    methods = methods or (
        parse_method_list(args.eval_methods)
        if args.eval_methods
        else parse_method_list(args.compare_methods)
    )
    rng = np.random.default_rng(args.eval_seed)
    sample_count = min(args.eval_max_samples, len(df))
    sample_indices = rng.choice(len(df), size=sample_count, replace=False)
    rows = [df.iloc[idx] for idx in sample_indices]

    records: List[dict[str, float]] = []
    for row in rows:
        rgb_img, thermal_img = load_pair(output_dir, row)
        for method in methods:
            mask_img = generate_mask_by_method(rgb_img, thermal_img, method, args)
            metrics = mask_metrics(mask_img)
            metrics["method"] = method
            records.append(metrics)

    metrics_df = pd.DataFrame(records)
    summary_df = metrics_df.groupby("method").agg(["mean", "std"]).round(4)
    print(summary_df)

    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(args.metrics_out)
        print(f"Saved metrics summary to {args.metrics_out}")

    if args.metrics_raw_out:
        args.metrics_raw_out.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(args.metrics_raw_out, index=False)
        print(f"Saved metrics raw table to {args.metrics_raw_out}")

    return summary_df, metrics_df


def mask_to_polygons(mask_img: np.ndarray) -> list[np.ndarray]:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required for polygon conversion. Install opencv-python.") from exc

    mask = (mask_img > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for contour in contours:
        if len(contour) < 3:
            continue
        poly = contour.squeeze(1)
        polys.append(poly)
    return polys


def write_label(polys: Iterable[np.ndarray], w: int, h: int, label_path: Path, class_id: int = 0) -> None:
    lines = []
    for poly in polys:
        coords = []
        for x, y in poly:
            coords.extend([x / w, y / h])
        if coords:
            lines.append(" ".join([str(class_id)] + [f"{v:.6f}" for v in coords]))
    label_path.write_text("\n".join(lines))


def export_yolo(
    df: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
    method: str | None = None,
    dataset_dir: Path | None = None,
) -> Path:
    method = method or args.mask_method
    dataset_dir = dataset_dir or resolve_dataset_dir(args, method, multi=False)
    img_dir = dataset_dir / "images" / "train"
    lbl_dir = dataset_dir / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        rgb_img, thermal_img = load_pair(output_dir, row)
        mask_img = generate_mask_by_method(rgb_img, thermal_img, method, args)
        polys = mask_to_polygons(mask_img)

        img_name = Path(row["rgb_path"]).name
        Image.fromarray(rgb_img).save(img_dir / img_name)
        label_path = lbl_dir / f"{Path(img_name).stem}.txt"
        write_label(polys, rgb_img.shape[1], rgb_img.shape[0], label_path)

    dataset_yaml = dataset_dir / "dataset.yaml"
    dataset_yaml.write_text(
        "path: "
        + str(dataset_dir)
        + "\ntrain: images/train\nval: images/train\nnames:\n  0: cookware\n"
    )
    print(f"Export complete: {dataset_dir}")
    return dataset_dir


def dataset_ready(dataset_dir: Path) -> bool:
    return (dataset_dir / "dataset.yaml").exists()


def train_yolo(dataset_dir: Path, args: argparse.Namespace, method: str) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Ultralytics is required. Install with: pip install ultralytics") from exc

    project_dir = args.train_project or (args.root / "runs" / "segment")
    name = f"{args.train_name_prefix}_{method}"
    model = YOLO(args.yolo_model)
    model.train(
        data=str(dataset_dir / "dataset.yaml"),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=str(project_dir),
        name=name,
        exist_ok=args.train_exist_ok,
    )

    if hasattr(model, "trainer") and getattr(model.trainer, "save_dir", None) is not None:
        return Path(model.trainer.save_dir)
    return project_dir / name


def collect_train_metrics(run_dir: Path, method: str) -> dict:
    record = {"method": method, "run_dir": str(run_dir)}
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return record

    results_df = pd.read_csv(results_csv)
    if results_df.empty:
        return record

    last_row = results_df.iloc[-1].to_dict()
    for key, value in last_row.items():
        record[f"last_{key}"] = value

    map_key = "metrics/mAP50-95(B)"
    if map_key in results_df.columns:
        best_idx = results_df[map_key].idxmax()
        best_row = results_df.loc[best_idx].to_dict()
        for key, value in best_row.items():
            record[f"best_{key}"] = value
    return record


def train_all_methods(
    df: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
    methods: List[str],
) -> pd.DataFrame:
    records: List[dict] = []
    for method in methods:
        dataset_dir = resolve_dataset_dir(args, method, multi=True)
        if not dataset_ready(dataset_dir):
            export_yolo(df, output_dir, args, method=method, dataset_dir=dataset_dir)
        run_dir = train_yolo(dataset_dir, args, method)
        records.append(collect_train_metrics(run_dir, method))

    summary_df = pd.DataFrame(records)
    print(summary_df)

    if args.training_out:
        args.training_out.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(args.training_out, index=False)
        print(f"Saved training summary to {args.training_out}")
    return summary_df


def update_readme(args: argparse.Namespace) -> None:
    readme_path = args.root / "README.md"
    start_marker = "<!-- NOTEBOOK_PARAMS_START -->"
    end_marker = "<!-- NOTEBOOK_PARAMS_END -->"

    param_rows = [
        {
            "Parameter": "MASK_METHOD",
            "Applies To": "all",
            "Meaning": "Select mask generation method",
            "Notes": "`thermal`, `thermal_cluster`, `sam`",
        },
        {
            "Parameter": "THERMAL_LOW",
            "Applies To": "thermal, sam",
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
            "Applies To": "sam",
            "Meaning": "SAM backbone (`vit_b`, `vit_l`, `vit_h`)",
            "Notes": "must match checkpoint",
        },
        {
            "Parameter": "SAM_TOPK",
            "Applies To": "sam",
            "Meaning": "Number of hottest points as prompts",
            "Notes": "larger ⇒ more guidance",
        },
        {
            "Parameter": "SAM_LOW",
            "Applies To": "sam",
            "Meaning": "Thermal threshold for SAM prompts",
            "Notes": "usually = `THERMAL_LOW`",
        },
        {
            "Parameter": "DATASET_DIR",
            "Applies To": "all",
            "Meaning": "Output dataset folder",
            "Notes": "default `dataset_yolo_<method>`",
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
        lines.append(
            "| {Parameter} | {Applies To} | {Meaning} | {Notes} |".format(**row)
        )
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


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir or (args.root / "output")
    index_csv = args.index_csv or (output_dir / "index.csv")
    if not index_csv.exists():
        raise FileNotFoundError(f"Missing {index_csv}. Run extract_rgbt_bag.py first.")

    args.sam_low = args.sam_low if args.sam_low is not None else args.thermal_low
    args.sam_checkpoint = args.sam_checkpoint or (args.root / "weights" / "sam_vit_b_01ec64.pth")

    methods = parse_method_list(args.methods)
    if args.metrics_out is None and args.label_all:
        args.metrics_out = args.root / "label_metrics_summary.csv"
    if args.metrics_raw_out is None and args.label_all:
        args.metrics_raw_out = args.root / "label_metrics_raw.csv"
    if args.training_out is None and args.train_all:
        args.training_out = args.root / "training_summary.csv"

    if not any([args.export_all, args.label_all, args.train_all]):
        args.dataset_dir = args.dataset_dir or (args.root / f"dataset_yolo_{args.mask_method}")

    df = pd.read_csv(index_csv)

    if args.compare:
        compare_masks(df, output_dir, args)

    if args.metrics:
        evaluate_metrics(df, output_dir, args)

    if args.export:
        export_yolo(df, output_dir, args)

    if args.export_all:
        for method in methods:
            dataset_dir = resolve_dataset_dir(args, method, multi=True)
            export_yolo(df, output_dir, args, method=method, dataset_dir=dataset_dir)

    if args.label_all:
        for method in methods:
            dataset_dir = resolve_dataset_dir(args, method, multi=True)
            export_yolo(df, output_dir, args, method=method, dataset_dir=dataset_dir)
        evaluate_metrics(df, output_dir, args, methods=methods)

    if args.train:
        dataset_dir = resolve_dataset_dir(args, args.mask_method, multi=False)
        if not dataset_ready(dataset_dir):
            export_yolo(df, output_dir, args, method=args.mask_method, dataset_dir=dataset_dir)
        run_dir = train_yolo(dataset_dir, args, args.mask_method)
        summary_df = pd.DataFrame([collect_train_metrics(run_dir, args.mask_method)])
        print(summary_df)
        if args.training_out:
            args.training_out.parent.mkdir(parents=True, exist_ok=True)
            summary_df.to_csv(args.training_out, index=False)
            print(f"Saved training summary to {args.training_out}")

    if args.train_all:
        train_all_methods(df, output_dir, args, methods)

    if args.update_readme:
        update_readme(args)

    if not any(
        [
            args.compare,
            args.metrics,
            args.export,
            args.export_all,
            args.label_all,
            args.train,
            args.train_all,
            args.update_readme,
        ]
    ):
        print("No action specified. Use --export, --compare, --metrics, or --update-readme.")


if __name__ == "__main__":
    main()
