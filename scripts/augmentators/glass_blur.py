import omni.replicator.core as rep

from .base import AugmentorBase

class GlassBlurAugmentor(AugmentorBase):
    name = "glassblur"
    prefix = "glassblur"

    def __init__(
        self,
        args,
        camera_prim_path: str,
        delta: int = 4,
        seed: int = 0,
        frequency: int = 1,
        prefix: str = None,
    ):
        super().__init__(args, camera_prim_path, frequency=frequency, prefix=prefix)
        self.delta = max(1, int(delta))
        self.seed = int(seed)

    def apply(self, writer):
        try:
            # Apply GlassBlur directly via augment_annotator (matches other augmentors and avoids passing an Annotator).
            writer.augment_annotator(
                "rgb",
                "GlassBlur",
                seed=self.seed,
                delta=self.delta,
            )

            print(
                f"[AUG] Writer '{self.name}': attached GlassBlur "
                f"(delta={self.delta}, seed={self.seed}, frequency={self.frequency})."
            )
        except Exception as exc:
            print(f"[AUG] Failed to attach glass blur augmentor: {exc}")
