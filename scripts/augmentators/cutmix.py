import omni.replicator.core as rep

from .base import AugmentorBase


class CutMixAugmentor(AugmentorBase):
    name = "cutmix"
    prefix = "cutmix"

    def __init__(self, args, camera_prim_path: str, folderpath: str, frequency: int = 1, prefix: str = None):
        super().__init__(args, camera_prim_path, frequency=frequency, prefix=prefix)
        self.folderpath = str(folderpath)
        # Use a unique name so multiple augmentors can coexist
        self._registered_name = f"{self.prefix}.AugCutMixExp"

    def apply(self, writer):
        try:
            writer.augment_annotator(
                "rgb",
                "omni.replicator.core.AugCutMixExp",
                folderpath=self.folderpath
            )
            print(
                f"[AUG] Writer '{self.name}': attached AugCutMixExp "
                f"(folderpath={self.folderpath}, frequency={self.frequency})."
            )
        except Exception as exc:
            print(f"[AUG] Failed to attach CutMix augmentor: {exc}")
