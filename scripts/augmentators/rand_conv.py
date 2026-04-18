import omni.replicator.core as rep

from .base import AugmentorBase


class RandConvAugmentor(AugmentorBase):
    name = "randconv"
    prefix = "randconv"

    def __init__(
        self,
        args,
        camera_prim_path: str,
        kernel_width: int = 3,
        alpha: float = 0.7,
        frequency: int = 1,
        prefix: str = None,
    ):
        super().__init__(args, camera_prim_path, frequency=frequency, prefix=prefix)
        self.kernel_width = kernel_width
        self.alpha = alpha

    def apply(self, writer):
        writer.augment_annotator(
            "rgb",
            "omni.replicator.core.AugConv2dExp",
            alpha=self.alpha,
            kernelWidth=self.kernel_width
        )
        print(
            f"[AUG] Writer '{self.name}': attached AugConv2dExp "
            f"(kernel_width={self.kernel_width}, alpha={self.alpha}, frequency={self.frequency})."
        )
