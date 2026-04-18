from .base import AugmentorBase


class ColorizeNormalsAugmentor(AugmentorBase):
    name = "colorizenormals"
    prefix = "colorizenormals"

    def apply(self, writer):
        writer.augment_annotator("rgb", "ColorizeNormals")
        print(f"[AUG] Writer '{self.name}': attached ColorizeNormals (frequency={self.frequency}).")
