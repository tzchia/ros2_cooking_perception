#!/usr/bin/env python3
"""Legacy all-in-one segmentation pipeline entrypoint."""
from __future__ import annotations

import pandas as pd

from segmentation.args import parse_args, parse_method_list, resolve_dataset_dir
from segmentation.export import dataset_ready, export_yolo
from segmentation.io import prepare_io
from segmentation.metrics import compare_masks, evaluate_metrics
from segmentation.readme import update_readme
from segmentation.training import collect_train_metrics, train_all_methods, train_yolo


def main() -> None:
    args = parse_args()
    output_dir, index_csv = prepare_io(args)

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
