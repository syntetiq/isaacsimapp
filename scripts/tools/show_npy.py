#!/usr/bin/env python
"""
Inspect NumPy .npy files emitted by Replicator writers.

Example:
    python scripts/show_npy.py _out_basic_writer/bounding_box_2d_tight_0000.npy --limit 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def positive_int(value: str) -> int:
    try:
        ivalue = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from exc
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return ivalue


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Display basic stats and a preview of a .npy array.")
    parser.add_argument("file", type=Path, help="Path to the .npy file to inspect.")
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=10,
        help="Number of leading rows to print (default: 10).",
    )
    parser.add_argument(
        "--flatten",
        action="store_true",
        help="Flatten the array before printing values (useful for 1-D previews).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    path = args.file.resolve()

    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    try:
        array = np.load(path)
    except Exception as exc:  # pragma: no cover - informative
        print(f"Failed to load {path}: {exc}", file=sys.stderr)
        return 2

    print(f"File      : {path}")
    print(f"Shape     : {array.shape}")
    print(f"Dtype     : {array.dtype}")

    if array.dtype.names:
        print("Fields:")
        for name in array.dtype.names:
            field = array[name]
            stats = []
            if field.size:
                if np.issubdtype(field.dtype, np.number):
                    stats.append(f"min={field.min()}")
                    stats.append(f"max={field.max()}")
                    stats.append(f"mean={field.mean():g}")
                else:
                    unique = np.unique(field)
                    stats.append(f"unique={len(unique)}")
            else:
                stats.append("empty")
            print(f"  - {name}: dtype={field.dtype} {' | '.join(stats)}")
        print()
    else:
        try:
            print(f"Min / Max : {array.min(initial=None)} / {array.max(initial=None)}")
        except Exception as exc:
            print(f"Min / Max : n/a ({exc})")
        try:
            mean_value = array.mean() if array.size else "n/a"
        except Exception as exc:
            mean_value = f"n/a ({exc})"
        print(f"Mean      : {mean_value}")
        print()

    if array.dtype.names:
        preview = array[: args.limit]
        print(f"First {len(preview)} entries:")

        def format_struct_row(row):
            formatted = {name: row[name].item() if np.isscalar(row[name]) else row[name].tolist() for name in array.dtype.names}
            return formatted

        for idx, row in enumerate(preview):
            print(f"[{idx:04d}] {format_struct_row(row)}")
    else:
        preview = array.flatten() if args.flatten else array
        limit = min(args.limit, preview.shape[0] if preview.ndim >= 1 else 1)
        print(f"First {limit} entries:")
        if preview.ndim == 1:
            print(preview[:limit])
        else:
            for idx in range(limit):
                print(f"[{idx:04d}] {preview[idx]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
