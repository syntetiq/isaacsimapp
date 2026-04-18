import omni.replicator.core as rep

from .base import AugmentorBase


class ImgBlendAugmentor(AugmentorBase):
    name = "imgblend"
    prefix = "imgblend"

    def __init__(self, args, camera_prim_path: str, folderpath: str, frequency: int = 1, prefix: str = None):
        super().__init__(args, camera_prim_path, frequency=frequency, prefix=prefix)
        self.folderpath = str(folderpath)

    def apply(self, writer):
        try:
            writer.augment_annotator(
                "rgb",
                "omni.replicator.core.AugImgBlendExp",
                folderpath=self.folderpath,
            )
            print(
                f"[AUG] Writer '{self.name}': attached AugImgBlendExp "
                f"(folderpath={self.folderpath}, frequency={self.frequency})."
            )
        except Exception as exc:
            print(f"[AUG] Failed to attach image blend augmentor: {exc}")
