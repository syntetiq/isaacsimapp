#!/usr/bin/env python
"""
Convert Replicator BasicWriter outputs into Pascal VOC XML annotations.

The script expects the usual BasicWriter artifacts:
  * RGB images named like ``rgb_0000.png`` (either in the run root or in ``rgb/``)
  * Bounding boxes stored as ``bounding_box_2d_tight_XXXX.npy`` structured arrays
  * Optional class metadata in ``bounding_box_2d_tight_labels_XXXX.json``

Usage:
    python scripts/convert_to_pascal_voc.py \
        --run-dir _out_basic_writer \
        --output-dir _out_pascal_voc
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
from xml.etree.ElementTree import Element, SubElement, ElementTree


@dataclass
class PascalObject:
    name: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    occluded: Optional[float] = None
    semantic_id: Optional[int] = None


@dataclass
class PascalAnnotation:
    filename: str
    folder: str
    path: str
    width: int
    height: int
    depth: int
    objects: List[PascalObject]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert BasicWriter outputs into Pascal VOC format.")
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
        help="Destination directory for Pascal VOC dataset. Defaults to <run-dir>/voc.",
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
        help="Copy images into Pascal VOC JPEGImages folder (default behaviour).",
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
    # Optional dataset metadata for writing a summary XML alongside the converted dataset
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Optional dataset name to embed into data_set_info.xml (e.g., last part of usd_path).",
    )
    parser.add_argument(
        "--dataset-image-width",
        type=int,
        default=None,
        help="Optional image width to record in data_set_info.xml (e.g., --width from load_stage).",
    )
    parser.add_argument(
        "--dataset-image-height",
        type=int,
        default=None,
        help="Optional image height to record in data_set_info.xml (e.g., --height from load_stage).",
    )
    parser.set_defaults(copy_images=True)
    return parser.parse_args(argv)


def ensure_output_dirs(root: Path) -> Tuple[Path, Path]:
    annotations_dir = root / "objects"
    images_dir = root / "objects"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    return annotations_dir, images_dir


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


## Note: prim path loading removed. Labels are sourced by semanticId.


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


def iter_boxes(array: np.ndarray) -> Iterator[PascalObject]:
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
            yield PascalObject(name="", xmin=x_min, ymin=y_min, xmax=x_max, ymax=y_max, occluded=occlusion, semantic_id=sem_id)
    else:
        for row in array.reshape((-1, array.shape[-1])):
            if len(row) < 4:
                continue
            x0, y0, x1, y1 = map(float, row[:4])
            normalized = all(0.0 <= c <= 1.0 for c in (x0, y0, x1, y1))
            if normalized:
                # Normalized coordinates need actual resolution to scale later; skip here
                continue
            x_min = int(round(min(x0, x1)))
            y_min = int(round(min(y0, y1)))
            x_max = int(round(max(x0, x1)))
            y_max = int(round(max(y0, y1)))
            if x_max <= x_min or y_max <= y_min:
                continue
            yield PascalObject(name="", xmin=x_min, ymin=y_min, xmax=x_max, ymax=y_max)


def build_annotation_xml(annotation: PascalAnnotation) -> ElementTree:
    root = Element("annotation")
    SubElement(root, "folder").text = annotation.folder
    SubElement(root, "filename").text = annotation.filename
    SubElement(root, "path").text = annotation.path

    source = SubElement(root, "source")
    SubElement(source, "database").text = "Omniverse Replicator"

    size = SubElement(root, "size")
    SubElement(size, "width").text = str(annotation.width)
    SubElement(size, "height").text = str(annotation.height)
    SubElement(size, "depth").text = str(annotation.depth)

    SubElement(root, "segmented").text = "0"

    for obj in annotation.objects:
        obj_el = SubElement(root, "object")
        SubElement(obj_el, "name").text = obj.name
        SubElement(obj_el, "pose").text = "Unspecified"
        SubElement(obj_el, "truncated").text = "0"
        if obj.occluded is not None:
            SubElement(obj_el, "occluded").text = f"{obj.occluded:.6f}"
        else:
            SubElement(obj_el, "occluded").text = "0"
        SubElement(obj_el, "difficult").text = "0"
        bbox = SubElement(obj_el, "bndbox")
        SubElement(bbox, "xmin").text = str(obj.xmin)
        SubElement(bbox, "ymin").text = str(obj.ymin)
        SubElement(bbox, "xmax").text = str(obj.xmax)
        SubElement(bbox, "ymax").text = str(obj.ymax)

    return ElementTree(root)


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


def write_dataset_info_file(output_dir: Path, name: Optional[str], width: Optional[int], height: Optional[int]) -> None:
    """Write minimal dataset metadata into data_set_info.xml at the output root.

    When any of the parameters are missing, the file is not written.
    """
    if not name or width is None or height is None:
        return
    root = Element("annotation")
    SubElement(root, "name").text = str(name)
    # Encode as HEIGHT_WIDTH per request (e.g., 480_640)
    SubElement(root, "imageSize").text = f"{int(height)}_{int(width)}"
    tree = ElementTree(root)
    tree.write(output_dir / "data_set_info.xml", encoding="utf-8", xml_declaration=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 1

    output_dir = args.output_dir.resolve() if args.output_dir else run_dir / "voc"
    annotations_dir, images_dir = ensure_output_dirs(output_dir)

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

        filtered_boxes: List[PascalObject] = []
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

        annotation = PascalAnnotation(
            filename=dest_image_path.name,
            folder=images_dir.name,
            path=str(dest_image_path if dest_image_path.exists() else image_path),
            width=width,
            height=height,
            depth=depth,
            objects=boxes,
        )

        xml_tree = build_annotation_xml(annotation)
        xml_path = annotations_dir / f"{image_path.stem}.xml"
        xml_tree.write(xml_path, encoding="utf-8", xml_declaration=True)

        processed += 1
        print(f"[VOC] Wrote {xml_path.name} with {len(boxes)} objects.")

    write_classes_file(output_dir, class_names)
    # Optionally write dataset summary info
    try:
        write_dataset_info_file(
            output_dir=output_dir,
            name=args.dataset_name,
            width=args.dataset_image_width,
            height=args.dataset_image_height,
        )
    except Exception as exc:
        print(f"[WARN] Failed to write data_set_info.xml: {exc}")
    print(f"[VOC] Completed. Processed {processed} frames. Classes: {class_names}")
    print(f"[VOC] Output directories:\n  Images: {images_dir}\n  Annotations: {annotations_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
