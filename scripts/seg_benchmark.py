#!/usr/bin/env python3
"""Benchmark segmentation labels (comparison + metrics)."""
from __future__ import annotations

import pandas as pd

from segmentation.args import parse_args, parse_method_list, resolve_dataset_dir
from segmentation.export import export_yolo
from segmentation.io import prepare_io
from segmentation.metrics import compare_masks, evaluate_metrics


def main() -> None:
    args = parse_args()
    output_dir, index_csv = prepare_io(args)
    df = pd.read_csv(index_csv)

    if args.compare:
        compare_masks(df, output_dir, args)

    if args.metrics:
        evaluate_metrics(df, output_dir, args)

    if args.label_all:
        methods = parse_method_list(args.methods)
        for method in methods:
            dataset_dir = resolve_dataset_dir(args, method, multi=True)
            export_yolo(df, output_dir, args, method=method, dataset_dir=dataset_dir)
        evaluate_metrics(df, output_dir, args, methods=methods)

    if not any([args.compare, args.metrics, args.label_all]):
        print("No action specified. Use --compare, --metrics, or --label-all.")


if __name__ == "__main__":
    main()
