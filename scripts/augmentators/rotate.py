from csv import writer
from pathlib import Path
from typing import Tuple

import math
import numpy as np
import omni.replicator.core as rep

from .base import AugmentorBase, _move_with_prefix


class RotateAugmentor(AugmentorBase):
    name = "rotate"
    prefix = "rotate"

    def __init__(self, args, camera_prim_path: str, angle: float, frequency: int = 1, prefix: str = None):
        super().__init__(args, camera_prim_path, frequency=frequency, prefix=prefix)
        self.angle = float(angle)
        self._resolution: Tuple[int, int] = (0, 0)

    def attach(self, output_dir: Path, resolution: Tuple[int, int]):
        # Override so we can keep bbox outputs and rotate them post-process.
        self._resolution = resolution
        rp = self.create_render_product(resolution)
        tmp_dir = output_dir / f"_tmp_{self.prefix}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(output_dir=str(tmp_dir), rgb=True, bounding_box_2d_tight=True)
        self.apply(writer)
        writer.attach(rp, trigger=None)
        return writer, rp, tmp_dir

    def apply(self, writer):
        # We use both parameter names to support different versions of Replicator
        writer.augment_annotator("rgb", "omni.replicator.core.AugRotateExp", rotateDegrees=self.angle, rotation=self.angle)
        print(
            f"[AUG] Writer '{self.name}': attached rotate "
            f"( rotateDegrees={self.angle}, frequency={self.frequency})."
        )


    @staticmethod
    def _rotate_point(x: float, y: float, cx: float, cy: float, cos_t: float, sin_t: float) -> Tuple[float, float]:
        """Rotate a point around (cx, cy) using standard 2D rotation (counterclockwise in math coords)."""
        dx, dy = x - cx, y - cy
        x_new = dx * cos_t - dy * sin_t + cx
        y_new = dx * sin_t + dy * cos_t + cy
        return x_new, y_new

    def _rotate_boxes(self, boxes: np.ndarray) -> np.ndarray:
        if boxes is None or boxes.size == 0:
            return boxes
        width, height = self._resolution
        # Rotate around the pixel-center midpoint.
        cx, cy = width * 0.5, height * 0.5
        # Image is y-down; rotate bboxes with -angle using standard rotation.
        theta = math.radians(-self.angle)
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        x_keys = ("x_min", "xmin", "left")
        y_keys = ("y_min", "ymin", "top")
        xmax_keys = ("x_max", "xmax", "right")
        ymax_keys = ("y_max", "ymax", "bottom")

        def pick_key(names, keys):
            for k in keys:
                if k in names:
                    return k
            return None

        names = boxes.dtype.names
        structured = names is not None
        out = boxes.copy()

        if structured:
            xk = pick_key(names, x_keys)
            yk = pick_key(names, y_keys)
            xxk = pick_key(names, xmax_keys)
            yyk = pick_key(names, ymax_keys)
            if not all((xk, yk, xxk, yyk)):
                return out
            for i, row in enumerate(out):
                try:
                    x0 = float(row[xk]); y0 = float(row[yk]); x1 = float(row[xxk]); y1 = float(row[yyk])
                except Exception:
                    continue
                normalized = 0.0 <= x0 <= 1.0 and 0.0 <= y0 <= 1.0 and 0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0
                if normalized:
                    x0 *= width; x1 *= width; y0 *= height; y1 *= height
                corners = [
                    self._rotate_point(x0, y0, cx, cy, cos_t, sin_t),
                    self._rotate_point(x0, y1, cx, cy, cos_t, sin_t),
                    self._rotate_point(x1, y0, cx, cy, cos_t, sin_t),
                    self._rotate_point(x1, y1, cx, cy, cos_t, sin_t),
                ]
                xs = [p[0] for p in corners]; ys = [p[1] for p in corners]
                xmin = max(0.0, min(xs)); ymin = max(0.0, min(ys))
                xmax = min(width - 1.0, max(xs)); ymax = min(height - 1.0, max(ys))
                if normalized:
                    row[xk] = xmin / width; row[yk] = ymin / height
                    row[xxk] = xmax / width; row[yyk] = ymax / height
                else:
                    row[xk] = xmin; row[yk] = ymin
                    row[xxk] = xmax; row[yyk] = ymax
        else:
            flat = out.reshape((-1, out.shape[-1]))
            for row in flat:
                if len(row) < 4:
                    continue
                x0, y0, x1, y1 = map(float, row[:4])
                normalized = 0.0 <= x0 <= 1.0 and 0.0 <= y0 <= 1.0 and 0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0
                if normalized:
                    x0 *= width; x1 *= width; y0 *= height; y1 *= height
                corners = [
                    self._rotate_point(x0, y0, cx, cy, cos_t, sin_t),
                    self._rotate_point(x0, y1, cx, cy, cos_t, sin_t),
                    self._rotate_point(x1, y0, cx, cy, cos_t, sin_t),
                    self._rotate_point(x1, y1, cx, cy, cos_t, sin_t),
                ]
                xs = [p[0] for p in corners]; ys = [p[1] for p in corners]
                xmin = max(0.0, min(xs)); ymin = max(0.0, min(ys))
                xmax = min(width - 1.0, max(xs)); ymax = min(height - 1.0, max(ys))
                if normalized:
                    row[0] = xmin / width; row[1] = ymin / height
                    row[2] = xmax / width; row[3] = ymax / height
                else:
                    row[0] = xmin; row[1] = ymin
                    row[2] = xmax; row[3] = ymax
        return out

    @staticmethod
    def _preview_box(arr: np.ndarray):
        if arr is None or arr.size == 0:
            return None
        names = arr.dtype.names
        if names:
            xk = next((k for k in ("x_min", "xmin", "left") if k in names), None)
            yk = next((k for k in ("y_min", "ymin", "top") if k in names), None)
            xxk = next((k for k in ("x_max", "xmax", "right") if k in names), None)
            yyk = next((k for k in ("y_max", "ymax", "bottom") if k in names), None)
            if all((xk, yk, xxk, yyk)):
                first = arr[0]
                return (float(first[xk]), float(first[yk]), float(first[xxk]), float(first[yyk]))
        else:
            flat = arr.reshape((-1, arr.shape[-1]))
            first = flat[0]
            if first.size >= 4:
                return tuple(map(float, first[:4]))
        return None

    def finalize(self, tmp_dir: Path, dest_dir: Path):
        # Rotate bbox files before moving.
        width, height = self._resolution
        if width and height:
            for npy_path in tmp_dir.glob("bounding_box_2d_tight_*.npy"):
                try:
                    data = np.load(npy_path)
                    before = self._preview_box(data)
                    rotated = self._rotate_boxes(data)
                    np.save(npy_path, rotated)
                    after = self._preview_box(rotated)
                    print(
                        f"[AUG] Rotated bboxes in {npy_path.name} for '{self.prefix}' by {self.angle} degrees "
                        f"(res={width}x{height}, before={before}, after={after})."
                    )
                except Exception as exc:
                    print(f"[AUG] Failed to rotate bboxes in {npy_path.name}: {exc}")
        return _move_with_prefix(tmp_dir, dest_dir, self.prefix, self.frequency)
