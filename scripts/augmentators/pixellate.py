from .base import AugmentorBase


class PixellateAugmentor(AugmentorBase):
    name = "pixellate"
    prefix = "pixellate"

    def __init__(self, args, camera_prim_path: str, kernel: int, frequency: int = 1, prefix: str = None):
        super().__init__(args, camera_prim_path, frequency=frequency, prefix=prefix)
        self.kernel = max(1, int(kernel))

    def apply(self, writer):
        writer.augment_annotator("rgb", "omni.replicator.core.AugPixellateExp", kernelSize=self.kernel)
        print(
            f"[AUG] Writer '{self.name}': attached AugPixellateExp "
            f"(kernel={self.kernel}, frequency={self.frequency})."
        )
