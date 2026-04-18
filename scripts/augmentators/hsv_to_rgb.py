import omni.replicator.core as rep

from .base import AugmentorBase


class HsvToRgbAugmentor(AugmentorBase):
    name = "hsv_to_rgb"
    prefix = "hsv2rgb"

    def apply(self, writer):
        try:
            writer.augment_annotator("rgb", "HsvToRgb")
            print(f"[AUG] Writer '{self.name}': attached HsvToRgb (frequency={self.frequency}).")
        except Exception as exc:
            print(f"[AUG] Failed to attach HSV->RGB augmentor: {exc}")
