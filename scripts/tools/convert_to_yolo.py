#!/usr/bin/env python
"""
Convert Replicator BasicWriter outputs into YOLO format.

The script expects the usual BasicWriter artifacts:
  * RGB images named like ``rgb_0000.png`` (either in the run root or in ``rgb/``)
  * Bounding boxes stored as ``bounding_box_2d_tight_XXXX.npy`` structured arrays
  * Optional class metadata in ``bounding_box_2d_tight_labels_XXXX.json``

Usage:
    python scripts/convert_to_yolo.py \
        --run-dir _out_basic_writer \
        --output-dir _out_yolo
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


@dataclass
class YoloObject:
    name: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    occluded: Optional[float] = None
    semantic_id: Optional[int] = None


@dataclass
class YoloAnnotation:
    filename: str
    folder: str
    path: str
    width: int
    height: int
    depth: int
    objects: List[YoloObject]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert BasicWriter outputs into YOLO format.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory containing BasicWriter outputs (images + NPY files).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory for YOLO dataset. Defaults to <run-dir>/yolo.",
    )
    parser.add_argument(
        "--image-subdir",
        type=Path,
        default=None,
        help="Optional relative path to the image directory (defaults to auto-detect).",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images into YOLO images folder (default behaviour).",
    )
    parser.add_argument(
        "--symlink-images",
        action="store_true",
        help="Create symlinks instead of copying images (Windows requires developer mode).",
    )
    parser.add_argument(
        "--convert-images-to-jpeg",
        action="store_true",
        help="Convert source images to JPEG in the output instead of copying PNGs.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality when --convert-images-to-jpeg is used (default: 95).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of frames to convert (useful for quick tests).",
    )
    # Filtering options
    parser.add_argument(
        "--include-label",
        action="append",
        default=None,
        help="Include only objects with this label; can be repeated.",
    )
    # Optional dataset metadata
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="dataset",
        help="Dataset name.",
    )
    parser.set_defaults(copy_images=True)
    return parser.parse_args(argv)


def ensure_output_dirs(root: Path) -> Tuple[Path, Path]:
    labels_dir = root / "train" / "labels"
    images_dir = root / "train" / "images"
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    return labels_dir, images_dir


def find_npy_files(run_dir: Path) -> List[Path]:
    patterns = [
        "*bounding_box_2d_tight_*.npy",
        "**/*bounding_box_2d_tight_*.npy",
    ]
    results: List[Path] = []
    seen = set()
    for pattern in patterns:
        for path in sorted(run_dir.glob(pattern)):
            if path.name not in seen:
                results.append(path)
                seen.add(path.name)
    return results


def extract_index(stem: str) -> Optional[int]:
    for part in reversed(stem.split("_")):
        if part.isdigit():
            try:
                return int(part)
            except Exception:
                return None
    return None


def extract_prefix_from_bbox(stem: str) -> str:
    marker = "bounding_box_2d_tight"
    if stem.startswith(marker):
        return ""
    if marker in stem:
        prefix = stem.split(marker)[0].rstrip("_")
        return prefix
    return ""


def load_labels_for_index(run_dir: Path, index: Optional[int], prefix: str = "") -> Dict[int, str]:
    if index is None:
        return {}
    stem_prefix = f"{prefix}_" if prefix else ""
    suffixes = [
        f"{stem_prefix}bounding_box_2d_tight_labels_{index:04d}.json",
        f"*{stem_prefix}bounding_box_2d_tight_labels_{index:04d}.json",
    ]
    candidates: List[Path] = []
    for suffix in suffixes:
        candidates.extend(run_dir.glob(suffix))
        candidates.extend(run_dir.glob(f"**/{suffix}"))
    for path in candidates:
        if path.is_file():
            try:
                payload = json.loads(path.read_text())
            except Exception:
                continue
            labels: Dict[int, str] = {}
            if isinstance(payload, dict):
                for key, meta in payload.items():
                    try:
                        idx = int(key)
                    except Exception:
                        continue
                    label = None
                    if isinstance(meta, dict):
                        label = meta.get("class") or meta.get("label") or meta.get("name")
                    elif isinstance(meta, str):
                        label = meta
                    if label:
                        labels[idx] = str(label)
            return labels
    return {}


def resolve_image_path(run_dir: Path, index: Optional[int], image_subdir: Optional[Path], prefix: str = "") -> Optional[Path]:
    if index is None:
        return None
    stem_prefix = f"{prefix}_" if prefix else ""
    candidates = []
    if image_subdir:
        base = run_dir / image_subdir
        candidates.extend(
            [
                base / f"{stem_prefix}rgb_{index:04d}.png",
                base / f"{stem_prefix}rgb_{index:04d}.jpg",
                base / f"rgb_{index:04d}.png",
                base / f"rgb_{index:04d}.jpg",
                *base.glob(f"*{stem_prefix}rgb_{index:04d}.png"),
                *base.glob(f"*{stem_prefix}rgb_{index:04d}.jpg"),
                *base.glob(f"*rgb_{index:04d}.png"),
                *base.glob(f"*rgb_{index:04d}.jpg"),
            ]
        )
    candidates.extend(
        [
            run_dir / f"{stem_prefix}rgb_{index:04d}.png",
            run_dir / f"{stem_prefix}rgb_{index:04d}.jpg",
            run_dir / f"rgb_{index:04d}.png",
            run_dir / f"rgb_{index:04d}.jpg",
            run_dir / "rgb" / f"{stem_prefix}rgb_{index:04d}.png",
            run_dir / "rgb" / f"{stem_prefix}rgb_{index:04d}.jpg",
            run_dir / "rgb" / f"rgb_{index:04d}.png",
            run_dir / "rgb" / f"rgb_{index:04d}.jpg",
            *run_dir.glob(f"*{stem_prefix}rgb_{index:04d}.png"),
            *run_dir.glob(f"*{stem_prefix}rgb_{index:04d}.jpg"),
            *run_dir.glob(f"*rgb_{index:04d}.png"),
            *run_dir.glob(f"*rgb_{index:04d}.jpg"),
            *run_dir.glob(f"**/*{stem_prefix}rgb_{index:04d}.png"),
            *run_dir.glob(f"**/*{stem_prefix}rgb_{index:04d}.jpg"),
            *run_dir.glob(f"**/*rgb_{index:04d}.png"),
            *run_dir.glob(f"**/*rgb_{index:04d}.jpg"),
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def iter_boxes(array: np.ndarray) -> Iterator[YoloObject]:
    if array is None or array.size == 0:
        return
    structured = array.dtype.names is not None

    if structured:
        names = array.dtype.names
        x_keys = {"x_min", "xmin", "left"}
        y_keys = {"y_min", "ymin", "top"}
        xmax_keys = {"x_max", "xmax", "right"}
        ymax_keys = {"y_max", "ymax", "bottom"}
        occ_keys = {"occlusionRatio", "occlusion_ratio", "occluded"}
        sid_keys = {"semanticId", "semantic_id", "classId", "class_id"}

        def pick(row, keys):
            for key in keys:
                if key in names:
                    return row[key]
            raise KeyError(keys)

        for row in array:
            try:
                x_min = int(round(float(pick(row, x_keys))))
                y_min = int(round(float(pick(row, y_keys))))
                x_max = int(round(float(pick(row, xmax_keys))))
                y_max = int(round(float(pick(row, ymax_keys))))
            except Exception:
                continue
            if x_max <= x_min or y_max <= y_min:
                continue
            occlusion = None
            for key in occ_keys:
                if key in names:
                    try:
                        occlusion = float(row[key])
                    except Exception:
                        occlusion = None
                    break
            sem_id = None
            for key in sid_keys:
                if key in names:
                    try:
                        sem_id = int(row[key])
                    except Exception:
                        sem_id = None
                    break
            yield YoloObject(name="", xmin=x_min, ymin=y_min, xmax=x_max, ymax=y_max, occluded=occlusion, semantic_id=sem_id)
    else:
        for row in array.reshape((-1, array.shape[-1])):
            if len(row) < 4:
                continue
            x0, y0, x1, y1 = map(float, row[:4])
            normalized = all(0.0 <= c <= 1.0 for c in (x0, y0, x1, y1))
            if normalized:
                # Normalized coordinates need actual resolution to scale later; skip here skip
                continue
            x_min = int(round(min(x0, x1)))
            y_min = int(round(min(y0, y1)))
            x_max = int(round(max(x0, x1)))
            y_max = int(round(max(y0, y1)))
            if x_max <= x_min or y_max <= y_min:
                continue
            yield YoloObject(name="", xmin=x_min, ymin=y_min, xmax=x_max, ymax=y_max)


def build_annotation_yolo(annotation: YoloAnnotation, class_to_id: Dict[str, int]) -> str:
    lines = []
    width = float(annotation.width)
    height = float(annotation.height)

    # YOLO format: class_id x_center y_center width height
    for obj in annotation.objects:
        class_id = class_to_id.get(obj.name, -1)
        if class_id == -1:
            continue
        
        # Calculate centers and dims
        x_c = (obj.xmin + obj.xmax) / 2.0
        y_c = (obj.ymin + obj.ymax) / 2.0
        w = float(obj.xmax - obj.xmin)
        h = float(obj.ymax - obj.ymin)

        # Normalize to [0, 1]
        x_c /= width
        y_c /= height
        w /= width
        h /= height

        lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

    return "\n".join(lines)


def determine_depth(image: Image.Image) -> int:
    mode_to_depth = {
        "1": 1,
        "L": 1,
        "P": 1,
        "RGB": 3,
        "RGBA": 4,
        "CMYK": 4,
        "F": 1,
        "I": 1,
    }
    return mode_to_depth.get(image.mode, len(image.getbands()))


def write_classes_file(output_dir: Path, class_names: List[str]) -> None:
    classes_path = output_dir / "classes.txt"
    classes_path.write_text("\n".join(class_names))


def write_dataset_yaml(output_root: Path, name: str, class_names: List[str]) -> None:
    """Write YOLO data.yaml file."""
    yaml_lines = [
        "train: ../train/images",
        "val: ../val/images",
        "test: ../test/images",
        "names:"
    ]
    for cls_name in class_names:
        yaml_lines.append(f"    - {cls_name}")
    yaml_lines.append("")

    yaml_path = output_root / "data.yaml"
    yaml_path.write_text("\n".join(yaml_lines))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 1

    output_dir = args.output_dir.resolve() if args.output_dir else run_dir / "yolo"
    labels_dir, images_dir = ensure_output_dirs(output_dir)

    npy_files = find_npy_files(run_dir)
    if not npy_files:
        print("No bounding_box_2d_tight_*.npy files found.", file=sys.stderr)
        return 2

    class_names: List[str] = []
    class_to_id: Dict[str, int] = {}

    def ensure_class(name: str) -> None:
        if name not in class_to_id:
            class_to_id[name] = len(class_names)
            class_names.append(name)

    processed = 0

    for npy_path in npy_files:
        if args.limit is not None and processed >= args.limit:
            break

        prefix = extract_prefix_from_bbox(npy_path.stem)
        index = extract_index(npy_path.stem)
        image_path = resolve_image_path(run_dir, index, args.image_subdir, prefix=prefix)

        if not image_path:
            print(f"[WARN] No image found for {npy_path.name}, skipping.")
            continue

        try:
            boxes_raw = np.load(npy_path)
        except Exception as exc:
            print(f"[WARN] Failed to load {npy_path}: {exc}")
            continue

        boxes = list(iter_boxes(boxes_raw))
        if not boxes:
            print(f"[INFO] No valid boxes in {npy_path.name}, skipping annotation.")
            continue

        labels_map = load_labels_for_index(run_dir, index, prefix=prefix)
        include_labels = set((args.include_label or []))

        filtered_boxes: List[YoloObject] = []
        for idx, obj in enumerate(boxes):
            label = None
            if hasattr(obj, "semantic_id") and obj.semantic_id is not None:
                label = labels_map.get(int(obj.semantic_id))
            if label is None:
                # Fallback to positional index if semantic_id not available
                label = labels_map.get(idx)
            if not label:
                label = "object"
            obj.name = label

            # Decide inclusion: only label-based filtering is supported.
            keep = (not include_labels) or (label in include_labels)
            if keep:
                ensure_class(label)
                filtered_boxes.append(obj)
        boxes = filtered_boxes
        if not boxes:
            print(f"[INFO] All boxes filtered out for {npy_path.name}; skipping annotation.")
            continue

        # Load image metadata
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                depth = determine_depth(img)
        except Exception as exc:
            print(f"[WARN] Failed to open image {image_path}: {exc}")
            continue

        # Decide destination filename and write/copy/symlink image
        if args.convert_images_to_jpeg:
            dest_image_path = images_dir / f"{image_path.stem}.jpg"
            try:
                with Image.open(image_path) as src_img:
                    # Ensure an RGB image for JPEG; handle alpha by compositing on black
                    img_mode = src_img.mode
                    if img_mode in ("RGBA", "LA") or (img_mode == "P" and "transparency" in src_img.info):
                        base = Image.new("RGB", src_img.size, (0, 0, 0))
                        if img_mode != "RGBA":
                            src_img = src_img.convert("RGBA")
                        base.paste(src_img, mask=src_img.split()[-1])
                        out_img = base
                    else:
                        out_img = src_img.convert("RGB")
                    out_img.save(dest_image_path, format="JPEG", quality=max(1, min(100, int(args.jpeg_quality))), optimize=True)
                depth = 3  # JPEG is RGB
            except Exception as exc:
                print(f"[WARN] Failed to convert {image_path} to JPEG: {exc}. Falling back to copy.")
                dest_image_path = images_dir / image_path.name
                try:
                    shutil.copy2(image_path, dest_image_path)
                except Exception as exc2:
                    print(f"[WARN] Fallback copy failed for {image_path}: {exc2}")
                    continue
        else:
            dest_image_path = images_dir / image_path.name
            if args.symlink_images:
                if dest_image_path.exists():
                    dest_image_path.unlink()
                try:
                    dest_image_path.symlink_to(image_path)
                except Exception as exc:
                    print(f"[WARN] Failed to symlink {image_path} -> {dest_image_path}: {exc}. Copying instead.")
                    shutil.copy2(image_path, dest_image_path)
            elif args.copy_images:
                shutil.copy2(image_path, dest_image_path)

        annotation = YoloAnnotation(
            filename=dest_image_path.name,
            folder=images_dir.name,
            path=str(dest_image_path if dest_image_path.exists() else image_path),
            width=width,
            height=height,
            depth=depth,
            objects=boxes,
        )

        yolo_str = build_annotation_yolo(annotation, class_to_id)
        if yolo_str:
            txt_path = labels_dir / f"{image_path.stem}.txt"
            txt_path.write_text(yolo_str, encoding="utf-8")

        processed += 1
        print(f"[YOLO] Wrote {image_path.stem}.txt with {len(boxes)} objects.")

    write_classes_file(labels_dir, class_names)
    try:
        write_dataset_yaml(output_dir, args.dataset_name, class_names)
    except Exception as exc:
        print(f"[WARN] Failed to write data.yaml: {exc}")
    print(f"[YOLO] Completed. Processed {processed} frames. Classes: {class_names}")
    print(f"[YOLO] Output directories:\n  Images: {images_dir}\n  Labels: {labels_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
