"""YOLO training helpers."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from segmentation.export import dataset_ready, export_yolo
from segmentation.args import resolve_dataset_dir


def train_yolo(
    dataset_dir: Path | None,
    args: argparse.Namespace,
    method: str,
    dataset_yaml: Path | None = None,
) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Ultralytics is required. Install with: pip install ultralytics") from exc

    project_dir = args.train_project or (args.root / "runs" / "segment")
    name = f"{args.train_name_prefix}_{method}"
    model = YOLO(args.yolo_model)
    if dataset_yaml is None:
        if dataset_dir is None:
            raise ValueError("dataset_dir or dataset_yaml must be provided for training")
        dataset_yaml = dataset_dir / "dataset.yaml"

    model.train(
        data=str(dataset_yaml),
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


def resolve_train_run_dir(args: argparse.Namespace, method: str) -> Path:
    project_dir = args.train_project or (args.root / "runs" / "segment")
    name = f"{args.train_name_prefix}_{method}"
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
        run_dir = resolve_train_run_dir(args, method)
        results_csv = run_dir / "results.csv"
        if args.train_skip_existing and results_csv.exists():
            records.append(collect_train_metrics(run_dir, method))
            continue
        run_dir = train_yolo(dataset_dir, args, method)
        records.append(collect_train_metrics(run_dir, method))

    summary_df = pd.DataFrame(records)
    print(summary_df)

    if args.training_out:
        args.training_out.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(args.training_out, index=False)
        print(f"Saved training summary to {args.training_out}")
    return summary_df
