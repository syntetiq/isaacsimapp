import math
from typing import Iterable, List

from .base import AugmentorBase


class Conv2dAugmentor(AugmentorBase):
    name = "conv2d"
    prefix = "conv2d"

    def __init__(
        self,
        args,
        camera_prim_path: str,
        kernel: Iterable[float],
        frequency: int = 1,
        prefix: str = None,
    ):
        super().__init__(args, camera_prim_path, frequency=frequency, prefix=prefix)
        self.kernel = self._normalize_kernel(kernel)

    @staticmethod
    def _normalize_kernel(kernel: Iterable[float]) -> List[float]:
        try:
            flat = [float(v) for v in kernel]
        except Exception:
            return []
        if not flat:
            return flat
        length = len(flat)
        size = int(math.sqrt(length))
        if size * size != length:
            print(f"[AUG] Conv2d kernel length {length} is not a perfect square; Replicator expects N*N entries.")
        return flat

    def apply(self, writer):
        writer.augment_annotator(
            "rgb",
            "Conv2d",
            kernel=self.kernel,
        )
        print(
            f"[AUG] Writer '{self.name}': attached Conv2d "
            f"(kernel_len={len(self.kernel)}, frequency={self.frequency})."
        )
