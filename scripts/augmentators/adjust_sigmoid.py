from .base import AugmentorBase


class AdjustSigmoidAugmentor(AugmentorBase):
    name = "adjustsigmoid"
    prefix = "adjustsigmoid"

    def __init__(
        self,
        args,
        camera_prim_path: str,
        cutoff: float = 0.5,
        gain: float = 1.0,
        frequency: int = 1,
        prefix: str = None,
    ):
        super().__init__(args, camera_prim_path, frequency=frequency, prefix=prefix)
        self.cutoff = float(cutoff)
        self.gain = float(gain)

    def apply(self, writer):
        writer.augment_annotator(
            "rgb",
            "AdjustSigmoid",
            cutoff=self.cutoff,
            gain=self.gain,
        )
        print(
            f"[AUG] Writer '{self.name}': attached AdjustSigmoid "
            f"(cutoff={self.cutoff}, gain={self.gain}, frequency={self.frequency})."
        )
