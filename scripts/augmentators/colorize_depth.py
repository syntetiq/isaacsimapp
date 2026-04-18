from .base import AugmentorBase


class ColorizeDepthAugmentor(AugmentorBase):
    name = "colorizedepth"
    prefix = "colorizedepth"

    def apply(self, writer):
        writer.augment_annotator("rgb", "ColorizeDepth")
        print(f"[AUG] Writer '{self.name}': attached ColorizeDepth (frequency={self.frequency}).")
