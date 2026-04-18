import omni.replicator.core as rep

from .base import AugmentorBase


class RgbToHsvAugmentor(AugmentorBase):
    name = "rgb_to_hsv"
    prefix = "rgb2hsv"

    def apply(self, writer):
        try:
            writer.augment_annotator("rgb", "RgbToHsv")
            print(f"[AUG] Writer '{self.name}': attached RgbToHsv (frequency={self.frequency}).")
        except Exception as exc:
            print(f"[AUG] Failed to attach RGB->HSV augmentor: {exc}")
