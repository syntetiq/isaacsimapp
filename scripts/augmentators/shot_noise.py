from .base import AugmentorBase


class ShotNoiseAugmentor(AugmentorBase):
    name = "shotnoise"
    prefix = "shotnoise"

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
        writer.augment_annotator(
            "rgb",
            "ShotNoise",
            sigma=self.sigma,
            seed=self.seed,
        )
        print(
            f"[AUG] Writer '{self.name}': attached ShotNoise "
            f"(sigma={self.sigma}, seed={self.seed}, frequency={self.frequency})."
        )
