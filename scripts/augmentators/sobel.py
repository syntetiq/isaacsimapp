from .base import AugmentorBase


class SobelAugmentor(AugmentorBase):
    name = "sobel"
    prefix = "sobel"

    def apply(self, writer):
        writer.augment_annotator("rgb", "Sobel")
        print(f"[AUG] Writer '{self.name}': attached Sobel (frequency={self.frequency}).")
