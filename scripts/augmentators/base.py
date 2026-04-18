from pathlib import Path
from typing import Optional, Tuple

import omni.replicator.core as rep


def _extract_frame_index(stem: str) -> Optional[int]:
    """Return the first numeric chunk found in a filename stem."""
    for part in stem.split("_"):
        if part.isdigit():
            try:
                return int(part)
            except Exception:
                return None
    return None


def _move_with_prefix(src_dir: Path, dest_dir: Path, prefix: str, frequency: int = 1) -> int:
    """Move writer outputs from src_dir into dest_dir with a filename prefix.

    When frequency > 1, only files whose numeric index is divisible by frequency are moved.
    """
    count = 0
    dest_dir.mkdir(parents=True, exist_ok=True)
    patterns = [
        "rgb_*.png",
        "rgb_*.jpg",
        "bounding_box_2d_tight_*.npy",
        "bounding_box_2d_tight_labels_*.json",
    ]
    for pattern in patterns:
        for path in src_dir.glob(pattern):
            if frequency > 1:
                idx = _extract_frame_index(path.stem)
                if idx is not None and idx % frequency != 0:
                    continue
            target = dest_dir / f"{prefix}_{path.name}"
            try:
                path.rename(target)
                count += 1
            except Exception as exc:
                print(f"[AUG] Failed to move {path.name} -> {target.name}: {exc}")
    try:
        if src_dir.exists() and not any(src_dir.iterdir()) and src_dir != dest_dir:
            src_dir.rmdir()
    except Exception:
        pass
    return count


class AugmentorBase:
    """Base class for per-augmentation writers."""

    name: str = "augmentor"
    prefix: str = "aug"

    def __init__(self, args, camera_prim_path: str, frequency: int = 1, prefix: Optional[str] = None):
        self.args = args
        self.camera_prim_path = camera_prim_path
        self.frequency = max(1, int(frequency))
        if prefix:
            self.prefix = prefix

    def create_render_product(self, resolution: Tuple[int, int]):
        return rep.create.render_product(self.camera_prim_path, resolution)

    def attach(self, output_dir: Path, resolution: Tuple[int, int]):
        rp = self.create_render_product(resolution)
        tmp_dir = output_dir / f"_tmp_{self.prefix}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(output_dir=str(tmp_dir), rgb=True, bounding_box_2d_tight=True)
        self.apply(writer)
        writer.attach(rp, trigger=None)
        return writer, rp, tmp_dir

    def apply(self, writer):
        raise NotImplementedError

    def finalize(self, tmp_dir: Path, dest_dir: Path):
        print(f"[DEBUG] Finalizing {self.name}: tmp_dir={tmp_dir}, exists={tmp_dir.exists()}")
        if tmp_dir.exists():
             print(f"[DEBUG] Files in tmp_dir: {list(tmp_dir.iterdir())}")
        # Pass frequency=1 because load_stage.py already handles frequency by scheduling writes.
        # BasicWriter outputs sequential indices (0, 1, 2...) for the writes it performs,
        # so filtering by frequency again (e.g. checking if 1 % 4 == 0) would incorrectly drop frames.
        return _move_with_prefix(tmp_dir, dest_dir, self.prefix, self.frequency)
