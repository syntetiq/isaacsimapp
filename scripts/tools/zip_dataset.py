#!/usr/bin/env python
"""
Zip a Pascal VOC dataset subset (objects/ and data_set_info.xml).

Usage:
  python scripts/zip_dataset.py \
      --voc-dir path/to/voc \
      --dataset-name asset.usd [--output-dir path/to/save]

Creates ZIP named: <dataset-name>_<hash>.zip
The hash is computed from the contents of the relevant files for the specified format:
- For Pascal VOC: data_set_info.xml and all files under objects/.
- For YOLO: data.yaml, and all files under train/, val/, test/, and labels.txt.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Iterable, Optional
from zipfile import ZipFile, ZIP_DEFLATED


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zip dataset subset")
    p.add_argument("--input-dir", "--voc-dir", dest="input_dir", type=Path, required=True, help="Path to output root (e.g. voc or yolo dir)")
    p.add_argument("--format", type=str, default="yolo", choices=["pascal_voc", "yolo"], help="Dataset format")
    p.add_argument("--dataset-name", type=str, required=True, help="Dataset base name used for archive name")
    p.add_argument("--output-dir", type=Path, default=None, help="Directory to place the zip (default: parent of input-dir)")
    return p.parse_args(argv)


def _iter_files_for_hash(input_dir: Path, dataset_format: str) -> Iterable[Path]:
    if dataset_format == "pascal_voc":
        info = input_dir / "data_set_info.xml"
        if info.is_file():
            yield info
        objects_dir = input_dir / "objects"
        if objects_dir.is_dir():
            for p in sorted(objects_dir.rglob("*")):
                if p.is_file():
                    yield p
    elif dataset_format == "yolo":
        yaml = input_dir / "data.yaml"
        if yaml.is_file():
            yield yaml
        labels_txt = input_dir / "labels.txt"
        if labels_txt.is_file():
            yield labels_txt
        train_dir = input_dir / "train"
        if train_dir.is_dir():
            for p in sorted(train_dir.rglob("*")):
                if p.is_file():
                    yield p
        val_dir = input_dir / "val"
        if val_dir.is_dir():
            for p in sorted(val_dir.rglob("*")):
                if p.is_file():
                    yield p
        test_dir = input_dir / "test"
        if test_dir.is_dir():
            for p in sorted(test_dir.rglob("*")):
                if p.is_file():
                    yield p


def compute_hash(input_dir: Path, dataset_format: str) -> str:
    h = hashlib.sha1()
    for path in _iter_files_for_hash(input_dir, dataset_format):
        h.update(str(path.relative_to(input_dir)).encode("utf-8"))
        try:
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
        except Exception:
            continue
    return h.hexdigest()[:10]


def make_zip_name(dataset_name: str, digest: str) -> str:
    safe = dataset_name.replace(" ", "_")
    return f"{safe}_{digest}.zip"


def build_archive(input_dir: Path, out_path: Path, dataset_format: str) -> None:
    with ZipFile(out_path, mode="w", compression=ZIP_DEFLATED) as zf:
        if dataset_format == "pascal_voc":
            info = input_dir / "data_set_info.xml"
            if info.is_file():
                zf.write(info, arcname="data_set_info.xml")
            objects_dir = input_dir / "objects"
            if objects_dir.is_dir():
                for path in sorted(objects_dir.rglob("*")):
                    if path.is_file():
                        zf.write(path, arcname=str(Path("objects") / path.relative_to(objects_dir)))
        elif dataset_format == "yolo":
            yaml = input_dir / "data.yaml"
            if yaml.is_file():
                zf.write(yaml, arcname="data.yaml")
            labels_txt = input_dir / "labels.txt"
            if labels_txt.is_file():
                zf.write(labels_txt, arcname="labels.txt")
            train_dir = input_dir / "train"
            if train_dir.is_dir():
                for path in sorted(train_dir.rglob("*")):
                    if path.is_file():
                        zf.write(path, arcname=str(Path("train") / path.relative_to(train_dir)))
            val_dir = input_dir / "val"
            if val_dir.is_dir():
                for path in sorted(val_dir.rglob("*")):
                    if path.is_file():
                        zf.write(path, arcname=str(Path("val") / path.relative_to(val_dir)))
            test_dir = input_dir / "test"
            if test_dir.is_dir():
                for path in sorted(test_dir.rglob("*")):
                    if path.is_file():
                        zf.write(path, arcname=str(Path("test") / path.relative_to(test_dir)))


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    input_dir = args.input_dir.resolve()
    if not input_dir.exists():
        print(f"Directory not found: {input_dir}", file=sys.stderr)
        return 2
    digest = compute_hash(input_dir, args.format)
    out_dir = args.output_dir.resolve() if args.output_dir else input_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_name = make_zip_name(args.dataset_name, digest)
    out_path = out_dir / zip_name
    build_archive(input_dir, out_path, args.format)
    print(f"[ZIP] Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

