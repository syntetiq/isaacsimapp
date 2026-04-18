from .base import AugmentorBase


class CannyAugmentor(AugmentorBase):
    name = "canny"
    prefix = "canny"

    def __init__(
        self,
        args,
        camera_prim_path: str,
        threshold_low: float = 50.0,
        threshold_high: float = 150.0,
        frequency: int = 1,
        prefix: str = None,
    ):
        super().__init__(args, camera_prim_path, frequency=frequency, prefix=prefix)
        self.threshold_low = float(threshold_low)
        self.threshold_high = float(threshold_high)

    def apply(self, writer):
        writer.augment_annotator(
            "rgb",
            "Canny",
            thresholdLow=self.threshold_low,
            thresholdHigh=self.threshold_high,
        )
        print(
            f"[AUG] Writer '{self.name}': attached Canny "
            f"(thresholdLow={self.threshold_low}, thresholdHigh={self.threshold_high}, frequency={self.frequency})."
        )
