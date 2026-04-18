from .base import AugmentorBase


class ContrastAugmentor(AugmentorBase):
    name = "contrast"
    prefix = "contrast"

    def __init__(
        self,
        args,
        camera_prim_path: str,
        contrast_factor: float = 1.0,
        frequency: int = 1,
        prefix: str = None,
    ):
        super().__init__(args, camera_prim_path, frequency=frequency, prefix=prefix)
        # Replicator expects positive values; 1.0 = identity.
        self.contrast_factor = max(0.0, float(contrast_factor))

    def apply(self, writer):
        writer.augment_annotator(
            "rgb",
            "Contrast",
            contrastFactor=self.contrast_factor,
            contrast_factor=self.contrast_factor,
        )
        print(
            f"[AUG] Writer '{self.name}': attached Contrast "
            f"(contrast_factor={self.contrast_factor}, frequency={self.frequency})."
        )
