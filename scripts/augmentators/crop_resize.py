import omni.replicator.core as rep

from .base import AugmentorBase


class CropResizeAugmentor(AugmentorBase):
    name = "cropresize"
    prefix = "cropresize"

    def __init__(
        self,
        args,
        camera_prim_path: str,
        crop_factor: float = 1.0,
        offset_factor=None,
        seed: int = 0,
        frequency: int = 1,
        prefix: str = None,
    ):
        super().__init__(args, camera_prim_path, frequency=frequency, prefix=prefix)
        cf = float(crop_factor)
        # Clamp to Replicator's valid range (>0, <=1.0)
        self.crop_factor = min(max(cf, 1e-6), 1.0)
        off = offset_factor or (0.0, 0.0)
        try:
            v, h = off
        except Exception:
            v = h = 0.0
        # Clamp offsets to [-1, 1] to keep the crop region on-screen.
        self.offset_factor = (max(-1.0, min(float(v), 1.0)), max(-1.0, min(float(h), 1.0)))
        self.seed = max(0, int(seed))

    def apply(self, writer):
        writer.augment_annotator(
            "rgb",
            "CropResize",
            cropFactor=self.crop_factor,
            offsetFactor=self.offset_factor,
            crop_factor=self.crop_factor,
            offset_factor=self.offset_factor,
            seed=self.seed
        )
        print(
            f"[AUG] Writer '{self.name}': attached CropResize "
            f"(cropFactor={self.crop_factor}, offsetFactor={self.offset_factor}, seed={self.seed}, frequency={self.frequency})."
        )
