#!/usr/bin/env python3
"""Export YOLO segmentation datasets with a shared train/val split."""
from __future__ import annotations

import pandas as pd

from segmentation.args import parse_args, parse_method_list, resolve_dataset_dir
from segmentation.export import export_yolo
from segmentation.io import prepare_io


def main() -> None:
    args = parse_args()
    output_dir, index_csv = prepare_io(args)
    df = pd.read_csv(index_csv)

    if args.export_all:
        methods = parse_method_list(args.methods)
        for method in methods:
            dataset_dir = resolve_dataset_dir(args, method, multi=True)
            export_yolo(df, output_dir, args, method=method, dataset_dir=dataset_dir)
        return

    dataset_dir = resolve_dataset_dir(args, args.mask_method, multi=False)
    export_yolo(df, output_dir, args, method=args.mask_method, dataset_dir=dataset_dir)


if __name__ == "__main__":
    main()
