from .base import AugmentorBase


class SpeckleNoiseAugmentor(AugmentorBase):
    name = "specklenoise"
    prefix = "specklenoise"

    def __init__(
        self,
        args,
        camera_prim_path: str,
        sigma: float = 0.1,
        seed: int = 0,
        frequency: int = 1,
        prefix: str = None,
    ):
        super().__init__(args, camera_prim_path, frequency=frequency, prefix=prefix)
        self.sigma = max(0.0, float(sigma))
        self.seed = max(0, int(seed))

    def apply(self, writer):
        try:
            writer.augment_annotator(
                "rgb",
                "SpeckleNoise",
                sigma=self.sigma,
                seed=self.seed,
            )
            print(
                f"[AUG] Writer '{self.name}': attached SpeckleNoise "
                f"(sigma={self.sigma}, seed={self.seed}, frequency={self.frequency})."
            )
        except Exception as exc:
            print(f"[AUG] Failed to attach speckle noise augmentor: {exc}")
