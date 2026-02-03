#!/usr/bin/env python3
"""Train YOLOv8-seg with exported datasets."""
from __future__ import annotations

import pandas as pd

from segmentation.args import parse_args, parse_method_list, resolve_dataset_dir
from segmentation.export import dataset_ready, export_yolo
from segmentation.io import prepare_io
from segmentation.training import collect_train_metrics, train_all_methods, train_yolo


def main() -> None:
    args = parse_args()
    output_dir, index_csv = prepare_io(args)
    df = pd.read_csv(index_csv)

    if args.training_out is None and args.train_all:
        args.training_out = args.root / "training_summary.csv"

    if args.train_all:
        methods = parse_method_list(args.methods)
        train_all_methods(df, output_dir, args, methods)
        return

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
        return

    print("No action specified. Use --train or --train-all.")


if __name__ == "__main__":
    main()
