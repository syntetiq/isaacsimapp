
import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from isaacsim import SimulationApp

CAMERA_PRIM_PATH = "/Replicator/CaptureCamera"

# This sample loads a usd stage, positions a camera, and captures RGB + bbox data.
CONFIG = {"width": 1280, "height": 720, "sync_loads": True, "headless": False, "renderer": "RaytracedLighting"}


# Set up command line arguments
parser = argparse.ArgumentParser("USD load + capture sample")
parser.add_argument(
    "--usd_path", type=str, help="Path to usd file, should be relative to your default assets folder", required=True
)
parser.add_argument("--headless", default=False, action="store_true", help="Run stage headless")
parser.add_argument("--frames", type=int, default=5, help="Number of frames to capture (default: 5)")
parser.add_argument(
    "--warmup-frames",
    type=int,
    default=1,
    help="Number of frames to run before starting captures (helps avoid first-frame blur).",
)
parser.add_argument("--width", type=int, default=CONFIG["width"], help="Render product width")
parser.add_argument("--height", type=int, default=CONFIG["height"], help="Render product height")
parser.add_argument(
    "--output-dir",
    type=Path,
    default=None,
    help="Directory to store captures (default: ./data/_out_loaded_stage)",
)
parser.add_argument(
    "--camera-pos",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=(0.0, 300.0, 400.0),
    help="Camera position in world coordinates",
)
parser.add_argument(
    "--camera-pos-end",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=None,
    help="Optional end position for the camera. When set, the camera moves linearly from --camera-pos to this point across frames.",
)
parser.add_argument(
    "--camera-look-at",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=(0.0, 0.0, 0.0),
    help="Camera look-at point",
)
parser.add_argument(
    "--camera-rotation",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=None,
    help="Camera rotation in degrees (XYZ order). Overrides look-at targeting if provided.",
)
parser.add_argument(
    "--focal-length",
    type=float,
    default=35.0,
    help="Camera focal length",
)
parser.add_argument(
    "--target-prim",
    type=str,
    default=None,
    help="USD prim path to focus camera on (e.g. /Root/table_low_327/table_low).",
)
parser.add_argument(
    "--label-name",
    type=str,
    default=None,
    help="Optional semantic label to assign to the target prim for bbox output.",
)
parser.add_argument(
    "--distance-scale",
    type=float,
    default=2.0,
    help="Multiplier for auto camera distance when focusing a target prim.",
)
parser.add_argument(
    "--keep-open",
    action="store_true",
    help="Keep the SimulationApp window open after captures complete.",
)
parser.add_argument(
    "--spawn-cube",
    action="store_true",
    help="Spawn a synthetic cube into the scene (helpful when a target asset is missing).",
)
parser.add_argument(
    "--cube-path",
    type=str,
    default="/Replicator/TargetCube",
    help="Prim path for the spawned cube (default: /Replicator/TargetCube).",
)
parser.add_argument(
    "--cube-translate",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=(0.0, 0.0, 0.0),
    help="Translation applied to the spawned cube.",
)
parser.add_argument(
    "--cube-scale",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=(1.0, 1.0, 1.0),
    help="Scale applied to the spawned cube.",
)
parser.add_argument(
    "--cube-size",
    type=float,
    default=100.0,
    help="Size attribute of the spawned cube in stage units (default: 100).",
)
parser.add_argument(
    "--aug-pixellate",
    type=str,
    action="append",
    help="JSON object describing one pixellate augmentor (e.g., {\"kernel\":12,\"frequency\":3}). Can be repeated.",
)
parser.add_argument(
    "--aug-motionblur",
    type=str,
    action="append",
    help=(
        "JSON object describing one motion blur augmentor "
        '(e.g., {"angle":45, "strength":0.7, "kernel":11, "frequency":2}). Can be repeated.'
    ),
)
parser.add_argument(
    "--aug-glassblur",
    type=str,
    action="append",
    help=(
        "JSON object describing one glass blur augmentor "
        '(e.g., {"frequency":2,"delta":6,"seed":123}). Can be repeated.'
    ),
)
parser.add_argument(
    "--aug-brightness",
    type=str,
    action="append",
    help='JSON object describing one brightness augmentor (e.g., {"brightness_factor":10,"frequency":2}). Can be repeated.',
)
parser.add_argument(
    "--aug-colorizedepth",
    type=str,
    action="append",
    help='JSON object describing one colorize depth augmentor (e.g., {"frequency":2}). Can be repeated.',
)
parser.add_argument(
    "--aug-colorizenormals",
    type=str,
    action="append",
    help='JSON object describing one colorize normals augmentor (e.g., {"frequency":2}). Can be repeated.',
)
parser.add_argument(
    "--aug-sobel",
    type=str,
    action="append",
    help='JSON object describing one sobel augmentor (e.g., {"frequency":2}). Can be repeated.',
)
parser.add_argument(
    "--aug-adjustsigmoid",
    type=str,
    action="append",
    help='JSON object describing one adjust sigmoid augmentor (e.g., {"cutoff":0.5,"gain":1.0,"frequency":2}). Can be repeated.',
)
parser.add_argument(
    "--aug-contrast",
    type=str,
    action="append",
    help='JSON object describing one contrast augmentor (e.g., {"contrastFactor":1.5,"frequency":2}). Can be repeated.',
)
parser.add_argument(
    "--aug-conv2d",
    type=str,
    action="append",
    help='JSON object describing one conv2d augmentor (e.g., {"kernel":[0,1,0,1,4,1,0,1,0],"frequency":2}). Can be repeated.',
)
parser.add_argument(
    "--aug-canny",
    type=str,
    action="append",
    help='JSON object describing one canny augmentor (e.g., {"thresholdLow":50,"thresholdHigh":150,"frequency":2}). Can be repeated.',
)
parser.add_argument(
    "--aug-shotnoise",
    type=str,
    action="append",
    help='JSON object describing one shot noise augmentor (e.g., {"sigma":0.1,"seed":0,"frequency":2}). Can be repeated.',
)
parser.add_argument(
    "--aug-specklenoise",
    type=str,
    action="append",
    help=(
        "JSON object describing one speckle noise augmentor "
        '(e.g., {"sigma":0.1,"seed":0,"frequency":2}). Can be repeated.'
    ),
)
parser.add_argument(
    "--aug-cropresize",
    type=str,
    action="append",
    help=(
        "JSON object describing one crop-resize augmentor "
        '(e.g., {"cropFactor":0.5,"offsetFactor":[0.0,0.0],"seed":123,"frequency":2}). Can be repeated.'
    ),
)
parser.add_argument(
    "--aug-rotate",
    type=str,
    action="append",
    help='JSON object describing one rotate augmentor (e.g., {"angle":15, "frequency":2}). Can be repeated.',
)
parser.add_argument(
    "--aug-randconv",
    type=str,
    action="append",
    help='JSON object describing one rand_conv augmentor (e.g., {"kernel_width":3, "alpha":0.7, "frequency":2}). Can be repeated.',
)
parser.add_argument(
    "--aug-cutmix",
    type=str,
    action="append",
    help=(
        "JSON object describing one CutMix augmentor "
        '(e.g., {"folderpath":"/abs/path/to/images","frequency":2}). Can be repeated.'
    ),
)
parser.add_argument(
    "--aug-rgb2hsv",
    type=str,
    action="append",
    help='JSON object describing one rgb->hsv augmentor (e.g., {"frequency":2}). Can be repeated.',
)
parser.add_argument(
    "--aug-hsv2rgb",
    type=str,
    action="append",
    help='JSON object describing one hsv->rgb augmentor (e.g., {"frequency":2}). Can be repeated.',
)
parser.add_argument(
    "--aug-imgblend",
    type=str,
    action="append",
    help=(
        "JSON object describing one image blend augmentor "
        '(e.g., {"folderpath":"/abs/path/to/images","frequency":2}). Can be repeated.'
    ),
)
parser.add_argument("--test", default=False, action="store_true", help="Retained for compatibility (no effect)")
parser.add_argument(
    "--rt-subframes",
    type=int,
    default=4,
    help="Number of subframes to render per capture. Useful for forcing synchronous renderer completion (default: 4)."
)
parser.add_argument(
    "--disable-async-rendering",
    action="store_true",
    default=True,
    help="Disable async rendering to prevent random black frames. Trades speed for reliability.",
)


args, unknown = parser.parse_known_args()

CONFIG["headless"] = args.headless
CONFIG["width"] = args.width
CONFIG["height"] = args.height
kit = SimulationApp(launch_config=CONFIG)
# Auto-approve OmniGraph script nodes so augmentor graphs load without prompting.
kit.set_setting("/app/omni.graph.scriptnode/opt_in", True)
if args.disable_async_rendering:
    print("[CONFIG] Disabling async rendering to prevent black frames")
    kit.set_setting("/exts/isaacsim.core.throttling/enable_async", False)
    kit.set_setting("/app/asyncRendering", False)
    kit.set_setting("/app/asyncRenderingLowLatency", False)

import carb
import omni
import omni.replicator.core as rep

# Locate Isaac Sim assets folder to load sample
from isaacsim.storage.native import get_assets_root_path, is_file
from isaacsim.core.utils.stage import is_stage_loading
from isaacsim.core.utils.semantics import add_labels

from pxr import Usd, UsdGeom, Gf, Sdf
import numpy as np
import shutil
from augmentators import (
    AugmentorBase,
    PixellateAugmentor,
    MotionBlurAugmentor,
    GlassBlurAugmentor,
    SpeckleNoiseAugmentor,
    BrightnessAugmentor,
    ColorizeDepthAugmentor,
    ColorizeNormalsAugmentor,
    SobelAugmentor,
    AdjustSigmoidAugmentor,
    ContrastAugmentor,
    Conv2dAugmentor,
    CannyAugmentor,
    ShotNoiseAugmentor,
    RotateAugmentor,
    RandConvAugmentor,
    CutMixAugmentor,
    RgbToHsvAugmentor,
    HsvToRgbAugmentor,
    ImgBlendAugmentor,
    CropResizeAugmentor,
)


def _label_prim_subtree(root_prim: Usd.Prim, label: str) -> bool:
    """Attach the given semantic label to a prim and all its descendants."""
    success = False
    stack = [root_prim]
    while stack:
        prim = stack.pop()
        if not prim or not prim.IsValid():
            continue
        try:
            add_labels(prim, labels=[label], instance_name="class")
            success = True
        except Exception:
            pass
        stack.extend(list(prim.GetChildren()))
    return success


def _parse_aug_list(raw_list, kind: str):
    configs = []
    for raw in raw_list or []:
        try:
            cfg = json.loads(raw)
            if not isinstance(cfg, dict):
                raise ValueError("Expected JSON object")
            configs.append(cfg)
        except Exception as exc:
            print(f"[AUG] Skipping {kind} augmentation entry '{raw}': {exc}")
    return configs


def _normalize_pixellate_configs(args):
    configs = _parse_aug_list(args.aug_pixellate, "pixellate")
    normalized = []
    for cfg in configs:
        kernel = max(1, int(cfg.get("kernel", 8)))
        frequency = max(1, int(cfg.get("frequency", 1)))
        normalized.append({"kernel": kernel, "frequency": frequency})
    return normalized


def _normalize_motionblur_configs(args):
    configs = _parse_aug_list(args.aug_motionblur, "motionblur")
    normalized = []
    for cfg in configs:
        angle = float(cfg.get("angle", 45.0))
        strength = float(cfg.get("strength", 0.7))
        frequency = max(1, int(cfg.get("frequency", 1)))
        kernel = max(1, int(cfg.get("kernel", 11)))
        normalized.append(
            {
                "angle": angle,
                "strength": strength,
                "kernel": kernel,
                "frequency": frequency,
            }
        )
    return normalized


def _normalize_glassblur_configs(args):
    configs = _parse_aug_list(args.aug_glassblur, "glassblur")
    normalized = []
    for cfg in configs:
        frequency = max(1, int(cfg.get("frequency", 1)))
        delta = int(cfg.get("delta", 4))
        seed = int(cfg.get("seed", 0))
        normalized.append(
            {
                "frequency": frequency,
                "delta": delta,
                "seed": seed,
            }
        )
    return normalized


def _normalize_brightness_configs(args):
    configs = _parse_aug_list(args.aug_brightness, "brightness")
    normalized = []
    for cfg in configs:
        frequency = max(1, int(cfg.get("frequency", 1)))
        bf = float(cfg.get("brightness_factor", cfg.get("brightnessFactor", 0.0)))
        bf = max(-100.0, min(bf, 100.0))
        normalized.append({"brightness_factor": bf, "frequency": frequency})
    return normalized


def _normalize_colorizedepth_configs(args):
    configs = _parse_aug_list(args.aug_colorizedepth, "colorizedepth")
    normalized = []
    for cfg in configs:
        frequency = max(1, int(cfg.get("frequency", 1)))
        normalized.append({"frequency": frequency})
    return normalized


def _normalize_colorizenormals_configs(args):
    configs = _parse_aug_list(args.aug_colorizenormals, "colorizenormals")
    normalized = []
    for cfg in configs:
        frequency = max(1, int(cfg.get("frequency", 1)))
        normalized.append({"frequency": frequency})
    return normalized


def _normalize_sobel_configs(args):
    configs = _parse_aug_list(args.aug_sobel, "sobel")
    normalized = []
    for cfg in configs:
        frequency = max(1, int(cfg.get("frequency", 1)))
        normalized.append({"frequency": frequency})
    return normalized


def _normalize_adjustsigmoid_configs(args):
    configs = _parse_aug_list(args.aug_adjustsigmoid, "adjustsigmoid")
    normalized = []
    for cfg in configs:
        frequency = max(1, int(cfg.get("frequency", 1)))
        cutoff = float(cfg.get("cutoff", 0.5))
        gain = float(cfg.get("gain", 1.0))
        normalized.append({"cutoff": cutoff, "gain": gain, "frequency": frequency})
    return normalized


def _normalize_contrast_configs(args):
    configs = _parse_aug_list(args.aug_contrast, "contrast")
    normalized = []
    for cfg in configs:
        frequency = max(1, int(cfg.get("frequency", 1)))
        factor = max(0.0, float(cfg.get("contrastFactor", cfg.get("contrast_factor", 1.0))))
        normalized.append({"contrast_factor": factor, "frequency": frequency})
    return normalized


def _normalize_conv2d_configs(args):
    configs = _parse_aug_list(args.aug_conv2d, "conv2d")
    normalized = []
    for cfg in configs:
        frequency = max(1, int(cfg.get("frequency", 1)))
        kernel = cfg.get("kernel", [])
        normalized.append({"kernel": kernel, "frequency": frequency})
    return normalized


def _normalize_canny_configs(args):
    configs = _parse_aug_list(args.aug_canny, "canny")
    normalized = []
    for cfg in configs:
        frequency = max(1, int(cfg.get("frequency", 1)))
        th_low = float(cfg.get("thresholdLow", cfg.get("threshold_low", 50.0)))
        th_high = float(cfg.get("thresholdHigh", cfg.get("threshold_high", 150.0)))
        normalized.append({"threshold_low": th_low, "threshold_high": th_high, "frequency": frequency})
    return normalized


def _normalize_shotnoise_configs(args):
    configs = _parse_aug_list(args.aug_shotnoise, "shotnoise")
    normalized = []
    for cfg in configs:
        frequency = max(1, int(cfg.get("frequency", 1)))
        sigma = max(0.0, float(cfg.get("sigma", 0.1)))
        seed = max(0, int(cfg.get("seed", 0)))
        normalized.append({"sigma": sigma, "seed": seed, "frequency": frequency})
    return normalized


def _normalize_specklenoise_configs(args):
    configs = _parse_aug_list(args.aug_specklenoise, "specklenoise")
    normalized = []
    for cfg in configs:
        sigma = max(0.0, float(cfg.get("sigma", 0.1)))
        seed = max(0, int(cfg.get("seed", 0)))
        frequency = max(1, int(cfg.get("frequency", 1)))
        normalized.append(
            {
                "sigma": sigma,
                "seed": seed,
                "frequency": frequency,
            }
        )
    return normalized


def _normalize_cropresize_configs(args):
    configs = _parse_aug_list(args.aug_cropresize, "cropresize")
    normalized = []
    for cfg in configs:
        crop_factor = float(cfg.get("cropFactor", 1.0))
        # Clamp to (0, 1].
        crop_factor = min(max(crop_factor, 1e-6), 1.0)
        offset = cfg.get("offsetFactor", [0.0, 0.0])
        try:
            v, h = offset
        except Exception:
            v = h = 0.0
        offset_factor = (max(-1.0, min(float(v), 1.0)), max(-1.0, min(float(h), 1.0)))
        seed = max(0, int(cfg.get("seed", 0)))
        frequency = max(1, int(cfg.get("frequency", 1)))
        normalized.append(
            {
                "crop_factor": crop_factor,
                "offset_factor": offset_factor,
                "seed": seed,
                "frequency": frequency,
            }
        )
    return normalized


def _normalize_rotate_configs(args):
    configs = _parse_aug_list(args.aug_rotate, "rotate")
    normalized = []
    for cfg in configs:
        angle = float(cfg.get("angle", 15.0))
        frequency = max(1, int(cfg.get("frequency", 1)))
        normalized.append({"angle": angle, "frequency": frequency})
    return normalized


def _normalize_randconv_configs(args):
    configs = _parse_aug_list(args.aug_randconv, "randconv")
    normalized = []
    for cfg in configs:
        kernel_width = int(cfg.get("kernel_width", 3))
        alpha = float(cfg.get("alpha", 0.7))
        frequency = max(1, int(cfg.get("frequency", 1)))
        normalized.append({"kernel_width": kernel_width, "alpha": alpha, "frequency": frequency})
    return normalized


def _normalize_cutmix_configs(args):
    configs = _parse_aug_list(args.aug_cutmix, "cutmix")
    normalized = []
    for cfg in configs:
        folderpath = cfg.get("folderpath") or cfg.get("folder_path") or cfg.get("folder")
        if not folderpath:
            print(f"[AUG] Skipping cutmix entry without folderpath: {cfg}")
            continue
        frequency = max(1, int(cfg.get("frequency", 1)))
        normalized.append({"folderpath": str(folderpath), "frequency": frequency})
    return normalized


def _normalize_rgb2hsv_configs(args):
    configs = _parse_aug_list(args.aug_rgb2hsv, "rgb2hsv")
    normalized = []
    for cfg in configs:
        frequency = max(1, int(cfg.get("frequency", 1)))
        normalized.append({"frequency": frequency})
    return normalized


def _normalize_hsv2rgb_configs(args):
    configs = _parse_aug_list(args.aug_hsv2rgb, "hsv2rgb")
    normalized = []
    for cfg in configs:
        frequency = max(1, int(cfg.get("frequency", 1)))
        normalized.append({"frequency": frequency})
    return normalized


def _normalize_imgblend_configs(args):
    configs = _parse_aug_list(args.aug_imgblend, "imgblend")
    normalized = []
    for cfg in configs:
        folderpath = cfg.get("folderpath") or cfg.get("folder_path") or cfg.get("folder")
        if not folderpath:
            print(f"[AUG] Skipping imgblend entry without folderpath: {cfg}")
            continue
        frequency = max(1, int(cfg.get("frequency", 1)))
        normalized.append({"folderpath": str(folderpath), "frequency": frequency})
    return normalized




assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    kit.close()
    sys.exit()
usd_path = assets_root_path + args.usd_path

# make sure the file exists before we try to open it
try:
    result = is_file(usd_path)
except Exception:
    result = False

if result:
    omni.usd.get_context().open_stage(usd_path)
else:
    carb.log_error(
        f"the usd path {usd_path} could not be opened, please make sure that {args.usd_path} is a valid usd file in {assets_root_path}"
    )
    kit.close()
    sys.exit()

# Wait two frames so that stage starts loading
kit.update()
kit.update()

print("Loading stage...")
while is_stage_loading():
    kit.update()
print("Loading Complete")

rep.orchestrator.set_capture_on_play(False)

stage = omni.usd.get_context().get_stage()

spawned_cube_prim = None
if args.spawn_cube:
    cube_root_path = Sdf.Path(args.cube_path)
    cube_xform = UsdGeom.Xform.Define(stage, str(cube_root_path))
    cube_api = UsdGeom.XformCommonAPI(cube_xform.GetPrim())
    cube_api.SetTranslate(Gf.Vec3d(*args.cube_translate))
    cube_api.SetScale(Gf.Vec3f(*args.cube_scale))
    cube_geom_path = cube_root_path.AppendChild("Cube")
    cube_geom = UsdGeom.Cube.Define(stage, str(cube_geom_path))
    cube_geom.CreateSizeAttr(args.cube_size)
    spawned_cube_prim = cube_geom.GetPrim()
    cube_label = args.label_name or "cube"
    try:
        add_labels(spawned_cube_prim, labels=[cube_label], instance_name="class")
        if not args.label_name:
            args.label_name = cube_label
            print(f"No label supplied; using default semantic label '{cube_label}' for the spawned cube.")
    except Exception as exc:
        print(f"[WARN] Unable to assign semantic label '{cube_label}' to spawned cube: {exc}")
    if not args.target_prim:
        args.target_prim = spawned_cube_prim.GetPath().pathString
    print(
        f"Spawned cube at {spawned_cube_prim.GetPath()} "
        f"(translate={args.cube_translate}, scale={args.cube_scale}, size={args.cube_size})"
    )

camera_position = tuple(args.camera_pos)
camera_position_end = tuple(args.camera_pos_end) if args.camera_pos_end else camera_position
camera_look_at = tuple(args.camera_look_at)
camera_rotation = tuple(args.camera_rotation) if args.camera_rotation else None

target_prim = stage.GetPrimAtPath(args.target_prim) if args.target_prim else None
if target_prim and target_prim.IsValid():
    if args.label_name:
        if _label_prim_subtree(target_prim, args.label_name):
            print(f"Assigned semantic label '{args.label_name}' to {args.target_prim}")
        else:
            print(f"[WARN] Failed to assign semantic label to {args.target_prim}")

    if camera_rotation is None:
        try:
            bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
                useExtentsHint=True,
            )
            world_box = bbox_cache.ComputeWorldBound(target_prim)
            bbox_range = world_box.GetRange()
            bbox_min: Optional[Gf.Vec3d] = bbox_range.GetMin()
            bbox_max: Optional[Gf.Vec3d] = bbox_range.GetMax()
        except Exception as exc:
            bbox_min = bbox_max = None
            print(f"[WARN] Failed to compute bounding box for {args.target_prim}: {exc}")
        if bbox_min and bbox_max:
            center = tuple((bbox_min[i] + bbox_max[i]) * 0.5 for i in range(3))
            extent = tuple(bbox_max[i] - bbox_min[i] for i in range(3))
            diag = max((sum(e * e for e in extent)) ** 0.5, 1.0)
            distance = max(diag * args.distance_scale, 10.0)
            offset = (0.0, -distance, distance * 0.5)
            camera_position = tuple(center[i] + offset[i] for i in range(3))
            camera_look_at = center
            print(f"Auto camera targeting {args.target_prim}: position={camera_position}, look_at={camera_look_at}")
        else:
            print(f"[WARN] Bounding box unavailable for {args.target_prim}, using provided camera position.")
else:
    if args.target_prim:
        print(f"[WARN] Target prim {args.target_prim} not found. Using default camera settings.")

camera_prim = stage.GetPrimAtPath(CAMERA_PRIM_PATH)
if camera_prim:
    stage.RemovePrim(CAMERA_PRIM_PATH)
camera_prim = UsdGeom.Camera.Define(stage, CAMERA_PRIM_PATH)
camera_prim.CreateFocalLengthAttr(args.focal_length)

xformable = UsdGeom.Xformable(camera_prim)
xformable.ClearXformOpOrder()

camera_transform_op = None
camera_xform_api: Optional[UsdGeom.XformCommonAPI] = None

if camera_rotation is not None:
    camera_xform_api = UsdGeom.XformCommonAPI(camera_prim)
    camera_xform_api.SetTranslate(Gf.Vec3d(*camera_position))
    camera_xform_api.SetRotate(
        Gf.Vec3f(*camera_rotation), UsdGeom.XformCommonAPI.RotationOrderXYZ
    )
else:
    camera_transform_op = xformable.AddTransformOp()
    camera_transform_op.Set(
        Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(*camera_position),
            Gf.Vec3d(*camera_look_at),
            Gf.Vec3d(0.0, 0.0, 1.0),
        )
    )

render_product = rep.create.render_product(CAMERA_PRIM_PATH, (args.width, args.height))

output_dir = (args.output_dir.resolve() if args.output_dir else Path.cwd() / "data/_out_loaded_stage")
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {output_dir}")

print("Waiting for stage and materials to load...")
while is_stage_loading():
    kit.update()
for _ in range(15):
    kit.update()

# Run a small warmup without writers attached to avoid first-frame artifacts/blur.
warmup_frames = max(0, args.warmup_frames)
if warmup_frames:
    print(f"Running {warmup_frames} warmup frame(s) with no captures...")
    for i in range(warmup_frames):
        if args.rt_subframes > 0:
            rep.orchestrator.step(rt_subframes=args.rt_subframes)
        else:
            rep.orchestrator.step()

class _BaseAug:
    prefix = None

    @staticmethod
    def finalize(tmp_dir: Path, dest_dir: Path) -> int:
        return 0

base_writer = rep.WriterRegistry.get("BasicWriter")
base_writer.initialize(output_dir=str(output_dir), rgb=True, bounding_box_2d_tight=True)
base_writer.attach(render_product)

augmentors: List[AugmentorBase] = []

pixellate_configs = _normalize_pixellate_configs(args)
motionblur_configs = _normalize_motionblur_configs(args)
glassblur_configs = _normalize_glassblur_configs(args)
brightness_configs = _normalize_brightness_configs(args)
colorizedepth_configs = _normalize_colorizedepth_configs(args)
colorizenormals_configs = _normalize_colorizenormals_configs(args)
sobel_configs = _normalize_sobel_configs(args)
adjustsigmoid_configs = _normalize_adjustsigmoid_configs(args)
contrast_configs = _normalize_contrast_configs(args)
conv2d_configs = _normalize_conv2d_configs(args)
canny_configs = _normalize_canny_configs(args)
shotnoise_configs = _normalize_shotnoise_configs(args)
specklenoise_configs = _normalize_specklenoise_configs(args)
cropresize_configs = _normalize_cropresize_configs(args)
rotate_configs = _normalize_rotate_configs(args)
randconv_configs = _normalize_randconv_configs(args)
cutmix_configs = _normalize_cutmix_configs(args)
rgb2hsv_configs = _normalize_rgb2hsv_configs(args)
hsv2rgb_configs = _normalize_hsv2rgb_configs(args)
imgblend_configs = _normalize_imgblend_configs(args)

for idx, cfg in enumerate(pixellate_configs):
    prefix = "pixellate" if len(pixellate_configs) == 1 else f"pixellate{idx + 1}"
    try:
        augmentors.append(
            PixellateAugmentor(
                args,
                CAMERA_PRIM_PATH,
                kernel=cfg["kernel"],
                frequency=cfg["frequency"],
                prefix=prefix,
            )
        )
    except Exception as exc:
        print(f"[AUG] Failed to configure pixellate augmentor #{idx + 1}: {exc}")

for idx, cfg in enumerate(motionblur_configs):
    prefix = "motionblur" if len(motionblur_configs) == 1 else f"motionblur{idx + 1}"
    try:
        augmentors.append(
            MotionBlurAugmentor(
                args,
                CAMERA_PRIM_PATH,
                angle=cfg["angle"],
                strength=cfg["strength"],
                kernel=cfg["kernel"],
                frequency=cfg["frequency"],
                prefix=prefix,
            )
        )
    except Exception as exc:
        print(f"[AUG] Failed to configure motion blur augmentor #{idx + 1}: {exc}")

for idx, cfg in enumerate(glassblur_configs):
    prefix = "glassblur" if len(glassblur_configs) == 1 else f"glassblur{idx + 1}"
    try:
        augmentors.append(
            GlassBlurAugmentor(
                args,
                CAMERA_PRIM_PATH,
                delta=cfg["delta"],
                seed=cfg["seed"],
                frequency=cfg["frequency"],
                prefix=prefix,
            )
        )
    except Exception as exc:
        print(f"[AUG] Failed to configure glass blur augmentor #{idx + 1}: {exc}")

for idx, cfg in enumerate(brightness_configs):
    prefix = "brightness" if len(brightness_configs) == 1 else f"brightness{idx + 1}"
    try:
        augmentors.append(
            BrightnessAugmentor(
                args,
                CAMERA_PRIM_PATH,
                brightness_factor=cfg["brightness_factor"],
                frequency=cfg["frequency"],
                prefix=prefix,
            )
        )
    except Exception as exc:
        print(f"[AUG] Failed to configure brightness augmentor #{idx + 1}: {exc}")

for idx, cfg in enumerate(colorizedepth_configs):
    prefix = "colorizedepth" if len(colorizedepth_configs) == 1 else f"colorizedepth{idx + 1}"
    try:
        augmentors.append(
            ColorizeDepthAugmentor(
                args,
                CAMERA_PRIM_PATH,
                frequency=cfg["frequency"],
                prefix=prefix,
            )
        )
    except Exception as exc:
        print(f"[AUG] Failed to configure colorizedepth augmentor #{idx + 1}: {exc}")

for idx, cfg in enumerate(colorizenormals_configs):
    prefix = "colorizenormals" if len(colorizenormals_configs) == 1 else f"colorizenormals{idx + 1}"
    try:
        augmentors.append(
            ColorizeNormalsAugmentor(
                args,
                CAMERA_PRIM_PATH,
                frequency=cfg["frequency"],
                prefix=prefix,
            )
        )
    except Exception as exc:
        print(f"[AUG] Failed to configure colorizenormals augmentor #{idx + 1}: {exc}")

for idx, cfg in enumerate(sobel_configs):
    prefix = "sobel" if len(sobel_configs) == 1 else f"sobel{idx + 1}"
    try:
        augmentors.append(
            SobelAugmentor(
                args,
                CAMERA_PRIM_PATH,
                frequency=cfg["frequency"],
                prefix=prefix,
            )
        )
    except Exception as exc:
        print(f"[AUG] Failed to configure sobel augmentor #{idx + 1}: {exc}")

for idx, cfg in enumerate(adjustsigmoid_configs):
    prefix = "adjustsigmoid" if len(adjustsigmoid_configs) == 1 else f"adjustsigmoid{idx + 1}"
    try:
        augmentors.append(
            AdjustSigmoidAugmentor(
                args,
                CAMERA_PRIM_PATH,
                cutoff=cfg["cutoff"],
                gain=cfg["gain"],
                frequency=cfg["frequency"],
                prefix=prefix,
            )
        )
    except Exception as exc:
        print(f"[AUG] Failed to configure adjustsigmoid augmentor #{idx + 1}: {exc}")

for idx, cfg in enumerate(contrast_configs):
    prefix = "contrast" if len(contrast_configs) == 1 else f"contrast{idx + 1}"
    try:
        augmentors.append(
            ContrastAugmentor(
                args,
                CAMERA_PRIM_PATH,
                contrast_factor=cfg["contrast_factor"],
                frequency=cfg["frequency"],
                prefix=prefix,
            )
        )
    except Exception as exc:
        print(f"[AUG] Failed to configure contrast augmentor #{idx + 1}: {exc}")

for idx, cfg in enumerate(conv2d_configs):
    prefix = "conv2d" if len(conv2d_configs) == 1 else f"conv2d{idx + 1}"
    try:
        augmentors.append(
            Conv2dAugmentor(
                args,
                CAMERA_PRIM_PATH,
                kernel=cfg["kernel"],
                frequency=cfg["frequency"],
                prefix=prefix,
            )
        )
    except Exception as exc:
        print(f"[AUG] Failed to configure conv2d augmentor #{idx + 1}: {exc}")

for idx, cfg in enumerate(canny_configs):
    prefix = "canny" if len(canny_configs) == 1 else f"canny{idx + 1}"
    try:
        augmentors.append(
            CannyAugmentor(
                args,
                CAMERA_PRIM_PATH,
                threshold_low=cfg["threshold_low"],
                threshold_high=cfg["threshold_high"],
                frequency=cfg["frequency"],
                prefix=prefix,
            )
        )
    except Exception as exc:
        print(f"[AUG] Failed to configure canny augmentor #{idx + 1}: {exc}")

for idx, cfg in enumerate(shotnoise_configs):
    prefix = "shotnoise" if len(shotnoise_configs) == 1 else f"shotnoise{idx + 1}"
    try:
        augmentors.append(
            ShotNoiseAugmentor(
                args,
                CAMERA_PRIM_PATH,
                sigma=cfg["sigma"],
                seed=cfg["seed"],
                frequency=cfg["frequency"],
                prefix=prefix,
            )
        )
    except Exception as exc:
        print(f"[AUG] Failed to configure shotnoise augmentor #{idx + 1}: {exc}")

for idx, cfg in enumerate(specklenoise_configs):
    prefix = "specklenoise" if len(specklenoise_configs) == 1 else f"specklenoise{idx + 1}"
    try:
        augmentors.append(
            SpeckleNoiseAugmentor(
                args,
                CAMERA_PRIM_PATH,
                sigma=cfg["sigma"],
                seed=cfg["seed"],
                frequency=cfg["frequency"],
                prefix=prefix,
            )
        )
    except Exception as exc:
        print(f"[AUG] Failed to configure speckle noise augmentor #{idx + 1}: {exc}")

for idx, cfg in enumerate(cropresize_configs):
    prefix = "cropresize" if len(cropresize_configs) == 1 else f"cropresize{idx + 1}"
    try:
        augmentors.append(
            CropResizeAugmentor(
                args,
                CAMERA_PRIM_PATH,
                crop_factor=cfg["crop_factor"],
                offset_factor=cfg["offset_factor"],
                seed=cfg["seed"],
                frequency=cfg["frequency"],
                prefix=prefix,
            )
        )
    except Exception as exc:
        print(f"[AUG] Failed to configure cropresize augmentor #{idx + 1}: {exc}")

for idx, cfg in enumerate(rotate_configs):
    prefix = "rotate" if len(rotate_configs) == 1 else f"rotate{idx + 1}"
    try:
        augmentors.append(
            RotateAugmentor(
                args,
                CAMERA_PRIM_PATH,
                angle=cfg["angle"],
                frequency=cfg["frequency"],
                prefix=prefix,
            )
        )
    except Exception as exc:
        print(f"[AUG] Failed to configure rotate augmentor #{idx + 1}: {exc}")

for idx, cfg in enumerate(randconv_configs):
    prefix = "randconv" if len(randconv_configs) == 1 else f"randconv{idx + 1}"
    try:
        augmentors.append(
            RandConvAugmentor(
                args,
                CAMERA_PRIM_PATH,
                kernel_width=cfg["kernel_width"],
                alpha=cfg["alpha"],
                frequency=cfg["frequency"],
                prefix=prefix,
            )
        )
    except Exception as exc:
        print(f"[AUG] Failed to configure randconv augmentor #{idx + 1}: {exc}")

for idx, cfg in enumerate(cutmix_configs):
    prefix = "cutmix" if len(cutmix_configs) == 1 else f"cutmix{idx + 1}"
    try:
        augmentors.append(
            CutMixAugmentor(
                args,
                CAMERA_PRIM_PATH,
                folderpath=cfg["folderpath"],
                frequency=cfg["frequency"],
                prefix=prefix,
            )
        )
    except Exception as exc:
        print(f"[AUG] Failed to configure cutmix augmentor #{idx + 1}: {exc}")

for idx, cfg in enumerate(rgb2hsv_configs):
    prefix = "rgb2hsv" if len(rgb2hsv_configs) == 1 else f"rgb2hsv{idx + 1}"
    try:
        augmentors.append(
            RgbToHsvAugmentor(
                args,
                CAMERA_PRIM_PATH,
                frequency=cfg["frequency"],
                prefix=prefix,
            )
        )
    except Exception as exc:
        print(f"[AUG] Failed to configure rgb2hsv augmentor #{idx + 1}: {exc}")

for idx, cfg in enumerate(hsv2rgb_configs):
    prefix = "hsv2rgb" if len(hsv2rgb_configs) == 1 else f"hsv2rgb{idx + 1}"
    try:
        augmentors.append(
            HsvToRgbAugmentor(
                args,
                CAMERA_PRIM_PATH,
                frequency=cfg["frequency"],
                prefix=prefix,
            )
        )
    except Exception as exc:
        print(f"[AUG] Failed to configure hsv2rgb augmentor #{idx + 1}: {exc}")

for idx, cfg in enumerate(imgblend_configs):
    prefix = "imgblend" if len(imgblend_configs) == 1 else f"imgblend{idx + 1}"
    try:
        augmentors.append(
            ImgBlendAugmentor(
                args,
                CAMERA_PRIM_PATH,
                folderpath=cfg["folderpath"],
                frequency=cfg["frequency"],
                prefix=prefix,
            )
        )
    except Exception as exc:
        print(f"[AUG] Failed to configure imgblend augmentor #{idx + 1}: {exc}")

active_writers = [(base_writer, output_dir, _BaseAug())]
active_render_products = [render_product]
for aug in augmentors:
    try:
        w, rp, tmp_dir = aug.attach(output_dir, (args.width, args.height))
        active_writers.append((w, tmp_dir, aug))
        active_render_products.append(rp)
    except Exception as exc:
        print(f"[AUG] Failed to initialize augmentor '{aug.name}': {exc}")

if args.frames > 1:
    denom = args.frames - 1
    position_step = tuple(
        (camera_position_end[i] - camera_position[i]) / denom for i in range(3)
    )
else:
    position_step = (0.0, 0.0, 0.0)

if camera_position_end != camera_position:
    print(f"Camera start position: {camera_position}")
    print(f"Camera end position  : {camera_position_end}")
    print(f"Per-frame step       : {position_step}")

try:
    for i in range(max(0, args.frames)):
        current_pos = tuple(
            camera_position[j] + position_step[j] * i for j in range(3)
        )
        if args.frames > 0 and i == args.frames - 1:
            current_pos = camera_position_end

        if camera_xform_api is not None:
            camera_xform_api.SetTranslate(Gf.Vec3d(*current_pos))
            if camera_rotation:
                camera_xform_api.SetRotate(
                    Gf.Vec3f(*camera_rotation), UsdGeom.XformCommonAPI.RotationOrderXYZ
                )
        elif camera_transform_op is not None:
            camera_transform_op.Set(
                Gf.Matrix4d().SetLookAt(
                    Gf.Vec3d(*current_pos),
                    Gf.Vec3d(*camera_look_at),
                    Gf.Vec3d(0.0, 0.0, 1.0),
                )
            )
        # Schedule augmentor writes based on frequency
        for w, tmp_dir, aug in active_writers[1:]:
            if i % aug.frequency == 0:
                try:
                    print(f"[DEBUG] Scheduling write for {aug.name} at frame {i}")
                    w.schedule_write()
                except Exception as exc:
                    print(f"[ERR] Failed to schedule write for {aug.name}: {exc}")
        if args.rt_subframes > 0:
            rep.orchestrator.step(rt_subframes=args.rt_subframes)
        else:
            rep.orchestrator.step()
        print(f"Capturing frame {i}: camera_pos={current_pos}")
    rep.orchestrator.wait_until_complete()
finally:
    for w, tmp_dir, aug in active_writers:
        try:
            w.detach()
        except Exception:
            pass
        moved = aug.finalize(tmp_dir, output_dir)
        if moved:
            print(f"[AUG] Prefixed {moved} files from {tmp_dir} into {output_dir} with '{aug.prefix}_'.")
    render_product.destroy()
    for rp in active_render_products[1:]:
        try:
            rp.destroy()
        except Exception:
            pass



if args.keep_open:
    print("Capture finished. SimulationApp left running (--keep-open). Press Ctrl+C to exit manually.")
    try:
        while kit.is_running():
            kit.update()
    except KeyboardInterrupt:
        pass
else:
    kit.close()
