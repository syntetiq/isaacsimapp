from .base import AugmentorBase


class BrightnessAugmentor(AugmentorBase):
    name = "brightness"
    prefix = "brightness"

    def __init__(
        self,
        args,
        camera_prim_path: str,
        brightness_factor: float = 0.0,
        frequency: int = 1,
        prefix: str = None,
    ):
        super().__init__(args, camera_prim_path, frequency=frequency, prefix=prefix)
        # Clamp to Replicator's expected range [-100, 100].
        self.brightness_factor = max(-100.0, min(float(brightness_factor), 100.0))

    def apply(self, writer):
        writer.augment_annotator(
            "rgb",
            "Brightness",
            brightnessFactor=self.brightness_factor,
            brightness_factor=self.brightness_factor,
        )
        print(
            f"[AUG] Writer '{self.name}': attached Brightness "
            f"(brightness_factor={self.brightness_factor}, frequency={self.frequency})."
        )
