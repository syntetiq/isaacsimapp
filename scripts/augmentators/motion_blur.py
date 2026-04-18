from .base import AugmentorBase


class MotionBlurAugmentor(AugmentorBase):
    name = "motionblur"
    prefix = "motionblur"

    def __init__(
        self,
        args,
        camera_prim_path: str,
        angle: float,
        strength: float,
        kernel: int,
        frequency: int = 1,
        prefix: str = None,
    ):
        super().__init__(args, camera_prim_path, frequency=frequency, prefix=prefix)
        self.kernel = max(1, int(kernel))
        self.angle = float(angle)
        self.strength = float(strength)

    def apply(self, writer):
        writer.augment_annotator(
            "rgb",
            "omni.replicator.core.AugMotionBlurExp",
            motionAngle=self.angle,
            strength=self.strength,
            kernelSize=self.kernel,
        )
        print(
            f"[AUG] Writer '{self.name}': attached AugMotionBlurExp "
            f"(angle={self.angle}, strength={self.strength}, kernel={self.kernel}, frequency={self.frequency})."
        )
