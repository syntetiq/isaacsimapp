"""
REST API for triggering Omniverse Replicator dataset runs.

Launch with:
    uvicorn scripts.dataset_server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator, validator

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    
    # Look for .env file in project root (parent of scripts directory)
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print(f"[CONFIG] Loaded environment variables from {env_path}")
    else:
        print(f"[CONFIG] No .env file found at {env_path}, using system environment variables")
except ImportError:
    print("[CONFIG] python-dotenv not installed, using system environment variables only")

SCRIPT_DIR = Path(__file__).resolve().parent
TOOLS_DIR = SCRIPT_DIR / "tools"
LOAD_STAGE_SCRIPT = SCRIPT_DIR / "load_stage.py"
CONVERT_VOC_SCRIPT = TOOLS_DIR / "convert_to_pascal_voc.py"
CONVERT_YOLO_SCRIPT = TOOLS_DIR / "convert_to_yolo.py"
ZIP_DATASET_SCRIPT = TOOLS_DIR / "zip_dataset.py"
RUN_LOCK = asyncio.Lock()
PYTHON_OMNI = "python" # or isaac-sim/python.bat
GCS_IMPORT_SUBDIR = "import_tmp"

Float3 = List[float]

app = FastAPI(title="Replicator Dataset Service")


class PixellateItem(BaseModel):
    kernel: int = Field(default=8, ge=1, description="Pixellate kernel size.")
    frequency: int = Field(default=1, ge=1, description="Capture every Nth frame for this augmentor.")


class MotionBlurItem(BaseModel):
    angle: float = Field(default=45.0, description="Motion blur angle in degrees.")
    strength: float = Field(default=0.7, description="Motion blur strength (-1 to 1).")
    kernel: int = Field(default=11, ge=1, description="Motion blur kernel size.")
    frequency: int = Field(default=1, ge=1, description="Capture every Nth frame for this augmentor.")

class GlassBlurItem(BaseModel):
    frequency: int = Field(default=1, ge=1, description="Capture every Nth frame for this augmentor.")
    delta: int = Field(default=4, ge=1, description="Pixel displacement (delta) for glass blur.")
    seed: int = Field(default=0, description="Random seed for glass blur.")

class BrightnessItem(BaseModel):
    frequency: int = Field(default=1, ge=1, description="Capture every Nth frame for this augmentor.")
    brightness_factor: float = Field(default=0.0, ge=-100.0, le=100.0, description="Brightness adjustment in [-100, 100].")

class ColorizeDepthItem(BaseModel):
    frequency: int = Field(default=1, ge=1, description="Capture every Nth frame for this augmentor.")

class ColorizeNormalsItem(BaseModel):
    frequency: int = Field(default=1, ge=1, description="Capture every Nth frame for this augmentor.")

class SobelItem(BaseModel):
    frequency: int = Field(default=1, ge=1, description="Capture every Nth frame for this augmentor.")

class AdjustSigmoidItem(BaseModel):
    frequency: int = Field(default=1, ge=1, description="Capture every Nth frame for this augmentor.")
    cutoff: float = Field(default=0.5, description="Shifts the sigmoid horizontally.")
    gain: float = Field(default=1.0, description="Multiplier in the exponent’s power.")

class ContrastItem(BaseModel):
    frequency: int = Field(default=1, ge=1, description="Capture every Nth frame for this augmentor.")
    contrastFactor: float = Field(default=1.0, ge=0.0, description="How much to adjust contrast (0=gray, 1=identity, 2=double).")

class Conv2dItem(BaseModel):
    frequency: int = Field(default=1, ge=1, description="Capture every Nth frame for this augmentor.")
    kernel: List[float] = Field(..., description="Flattened kernel values of size N*N (e.g., 3x3 => 9 entries).")

class CannyItem(BaseModel):
    frequency: int = Field(default=1, ge=1, description="Capture every Nth frame for this augmentor.")
    thresholdLow: float = Field(default=50.0, description="Low threshold for hysteresis.")
    thresholdHigh: float = Field(default=150.0, description="High threshold for hysteresis.")

class ShotNoiseItem(BaseModel):
    frequency: int = Field(default=1, ge=1, description="Capture every Nth frame for this augmentor.")
    sigma: float = Field(default=0.1, ge=0.0, description="Noise amount; larger is noisier.")
    seed: int = Field(default=0, ge=0, description="Random seed for shot noise.")

class SpeckleNoiseItem(BaseModel):
    frequency: int = Field(default=1, ge=1, description="Capture every Nth frame for this augmentor.")
    sigma: float = Field(default=0.1, ge=0.0, description="Noise strength; higher values add more noise.")
    seed: int = Field(default=0, ge=0, description="Random seed for speckle noise.")

class CropResizeItem(BaseModel):
    frequency: int = Field(default=1, ge=1, description="Capture every Nth frame for this augmentor.")
    cropFactor: float = Field(default=1.0, gt=0.0, le=1.0, description="Portion of the image to keep (0-1].")
    offsetFactor: List[float] = Field(
        default_factory=lambda: [0.0, 0.0],
        description="Translation offset (vertical, horizontal) in [-1,1].",
    )
    seed: int = Field(default=0, ge=0, description="Random seed for crop sampler.")

    @validator("offsetFactor", pre=True)
    def _normalize_offset(cls, value):
        seq = [0.0, 0.0] if value is None else list(value)
        if len(seq) != 2:
            raise ValueError("offsetFactor must have exactly two elements (vertical, horizontal).")
        return [max(-1.0, min(float(seq[0]), 1.0)), max(-1.0, min(float(seq[1]), 1.0))]

class RotateItem(BaseModel):
    angle: float = Field(default=15.0, description="Rotation angle in degrees.")
    frequency: int = Field(default=1, ge=1, description="Capture every Nth frame for this augmentor.")

class RandConvItem(BaseModel):
    frequency: int = Field(default=1, ge=1, description="Capture every Nth frame for this augmentor.")
    kernel_width: int = Field(default=3, ge=1, description="Kernel width.")
    alpha: float = Field(default=0.7, description="Alpha value.")

class CutMixItem(BaseModel):
    frequency: int = Field(default=1, ge=1, description="Capture every Nth frame for this augmentor.")
    folderpath: str = Field(..., description="Folder containing source images for CutMix.")

class RgbToHsvItem(BaseModel):
    frequency: int = Field(default=1, ge=1, description="Capture every Nth frame for this augmentor.")

class HsvToRgbItem(BaseModel):
    frequency: int = Field(default=1, ge=1, description="Capture every Nth frame for this augmentor.")

class ImgBlendItem(BaseModel):
    frequency: int = Field(default=1, ge=1, description="Capture every Nth frame for this augmentor.")
    folderpath: str = Field(..., description="Folder containing source images for random blending.")


class AugmentationItem(BaseModel):
    pixellate: Optional[PixellateItem] = None
    motion_blur: Optional[MotionBlurItem] = None
    glass_blur: Optional[GlassBlurItem] = None
    brightness: Optional[BrightnessItem] = None
    colorize_depth: Optional[ColorizeDepthItem] = None
    colorize_normals: Optional[ColorizeNormalsItem] = None
    sobel: Optional[SobelItem] = None
    adjust_sigmoid: Optional[AdjustSigmoidItem] = None
    contrast: Optional[ContrastItem] = None
    conv2d: Optional[Conv2dItem] = None
    canny: Optional[CannyItem] = None
    shot_noise: Optional[ShotNoiseItem] = None
    speckle_noise: Optional[SpeckleNoiseItem] = None
    cropresize: Optional[CropResizeItem] = None
    rotate: Optional[RotateItem] = None
    rand_conv: Optional[RandConvItem] = None
    cutmix: Optional[CutMixItem] = None
    rgb2hsv: Optional[RgbToHsvItem] = None
    hsv2rgb: Optional[HsvToRgbItem] = None
    imgblend: Optional[ImgBlendItem] = None

    @model_validator(mode="before")
    def _one_kind_only(cls, data):
        data = data or {}
        kinds = [
            key
            for key in (
                "pixellate",
                "motion_blur",
                "glass_blur",
                "brightness",
                "colorize_depth",
                "colorize_normals",
                "sobel",
                "adjust_sigmoid",
                "contrast",
                "conv2d",
                "canny",
                "shot_noise",
                "speckle_noise",
                "cropresize",
                "rotate",
                "rand_conv",
                "cutmix",
                "rgb2hsv",
                "hsv2rgb",
                "imgblend",
            )
            if data.get(key) is not None
        ]
        if len(kinds) != 1:
            raise ValueError(
                "Each augmentation entry must define exactly one type (pixellate, motion_blur, glass_blur, brightness, colorize_depth, colorize_normals, sobel, adjust_sigmoid, contrast, conv2d, canny, shot_noise, speckle_noise, cropresize, rotate, rand_conv, cutmix, rgb2hsv, hsv2rgb, or imgblend)."
            )
        return data


class LoadStageRequest(BaseModel):
    usd_path: str = Field(..., description="USD file to load inside the simulation.")
    frames: int = Field(default=5, gt=0, description="Number of frames to capture.")
    warmup_frames: int = Field(default=2, ge=0, description="Frames to run before capturing to avoid first-frame artifacts.")
    width: int = Field(default=640, gt=0, description="Render width in pixels.")
    height: int = Field(default=480, gt=0, description="Render height in pixels.")
    focal_length: float = Field(default=35.0, gt=0, description="Camera focal length.")
    camera_pos: Float3 = Field(
        default_factory=lambda: [0.0, 300.0, 400.0],
        description="Starting camera position (XYZ).",
    )
    camera_pos_end: Optional[Float3] = Field(
        default=None,
        description="Optional end position (XYZ). The camera interpolates linearly from start to end.",
    )
    camera_look_at: Optional[Float3] = Field(
        default=None,
        description="Optional look-at target (XYZ). When omitted the script default is used.",
    )
    camera_rotation: Optional[Float3] = Field(
        default=None,
        description="Optional camera Euler XYZ rotation in degrees.",
    )
    label_name: Optional[str] = Field(
        default=None,
        description="Optional semantic label assigned to the target prim or spawned cube.",
    )
    distance_scale: Optional[float] = Field(
        default=None,
        description="Optional distance scale used when auto-positioning the camera.",
    )
    spawn_cube: bool = Field(
        default=False,
        description="Spawn a helper cube into the scene (useful when no target prim is available).",
    )
    cube_translate: Optional[Float3] = Field(
        default=None,
        description="Cube translation applied when --spawn-cube is used.",
    )
    cube_scale: Optional[Float3] = Field(
        default=None,
        description="Cube scale applied when --spawn-cube is used.",
    )
    cube_size: Optional[float] = Field(
        default=None,
        description="Cube size (in stage units) when --spawn-cube is used.",
    )
    augmentation: Optional[List[AugmentationItem]] = Field(
        default=None,
        description="Optional list of augmentations; allows multiple entries of the same type with unique parameters.",
    )
    tmp_root: Optional[str] = Field(
        default=None,
        description="Optional directory where temporary run folders will be created.",
    )
    # Converter options
    convert_images_to_jpeg: bool = Field(
        default=False,
        description="Convert images to JPEG when generating Pascal VOC output.",
    )
    jpeg_quality: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="JPEG quality (1-100) used when convert_images_to_jpeg is true.",
    )
    # Filter options (apply during VOC conversion)
    include_labels: Optional[List[str]] = Field(
        default=None,
        description="Only include bboxes with these labels (exact match).",
    )
    cleanup_after_zip: bool = Field(
        default=False,
        description="Delete intermediate run data after zipping (keeps only the zip).",
    )
    hash_request: Optional[str] = Field(
        default=None,
        description="Optional hash identifier for this dataset generation request.",
    )
    dataset_format: str = Field(
        default="yolo",
        description="Format of the output dataset (pascal_voc or yolo).",
    )
    disable_async_rendering: bool = Field(
        default=True,
        description="Disable async rendering to prevent random black frames. Trades speed for reliability.",
    )

    @validator(
        "camera_pos",
        "camera_pos_end",
        "camera_look_at",
        "camera_rotation",
        "cube_translate",
        "cube_scale",
        pre=True,
        always=True,
    )
    def _validate_float3(cls, value):
        if value is None:
            return value
        seq = list(value)
        if len(seq) != 3:
            raise ValueError("Expected exactly three elements")
        return [float(x) for x in seq]


def _extend_vector_args(cmd: List[str], flag: str, values: Optional[Float3]) -> None:
    if values is None:
        return
    cmd.append(flag)
    cmd.extend(str(v) for v in values)


def _build_load_stage_command(payload: LoadStageRequest, output_dir: Path) -> List[str]:
    if not LOAD_STAGE_SCRIPT.exists():
        raise FileNotFoundError(f"load_stage.py not found at {LOAD_STAGE_SCRIPT}")

    cmd: List[str] = [
        PYTHON_OMNI,
        str(LOAD_STAGE_SCRIPT),
        # "--keep-open",
        "--usd_path",
        payload.usd_path,
        "--frames",
        str(payload.frames),
        "--width",
        str(payload.width),
        "--height",
        str(payload.height),
        "--warmup-frames",
        str(payload.warmup_frames),
        "--focal-length",
        str(payload.focal_length),
        "--output-dir",
        str(output_dir),
    ]

    _extend_vector_args(cmd, "--camera-pos", payload.camera_pos)
    _extend_vector_args(cmd, "--camera-pos-end", payload.camera_pos_end)
    _extend_vector_args(cmd, "--camera-look-at", payload.camera_look_at)
    _extend_vector_args(cmd, "--camera-rotation", payload.camera_rotation)

    if payload.label_name:
        cmd.extend(["--label-name", payload.label_name])
    if payload.distance_scale is not None:
        cmd.extend(["--distance-scale", str(payload.distance_scale)])
    if payload.spawn_cube:
        cmd.append("--spawn-cube")
        _extend_vector_args(cmd, "--cube-translate", payload.cube_translate or [0.0, 0.0, 0.0])
        _extend_vector_args(cmd, "--cube-scale", payload.cube_scale or [1.0, 1.0, 1.0])
        if payload.cube_size is not None:
            cmd.extend(["--cube-size", str(payload.cube_size)])

    if payload.disable_async_rendering:
        cmd.append("--disable-async-rendering")

    cmd.extend([])

    def _append_aug(flag: str, config: dict) -> None:
        cmd.extend([flag, json.dumps(config)])

    aug_entries: List[dict] = []
    if payload.augmentation:
        for item in payload.augmentation:
            if item.pixellate:
                aug_entries.append({"type": "pixellate", **item.pixellate.dict()})
            elif item.motion_blur:
                aug_entries.append({"type": "motion_blur", **item.motion_blur.dict()})
            elif item.glass_blur:
                aug_entries.append({"type": "glass_blur", **item.glass_blur.dict()})
            elif item.brightness:
                aug_entries.append({"type": "brightness", **item.brightness.dict()})
            elif item.colorize_depth:
                aug_entries.append({"type": "colorize_depth", **item.colorize_depth.dict()})
            elif item.colorize_normals:
                aug_entries.append({"type": "colorize_normals", **item.colorize_normals.dict()})
            elif item.sobel:
                aug_entries.append({"type": "sobel", **item.sobel.dict()})
            elif item.adjust_sigmoid:
                aug_entries.append({"type": "adjust_sigmoid", **item.adjust_sigmoid.dict()})
            elif item.contrast:
                aug_entries.append({"type": "contrast", **item.contrast.dict()})
            elif item.conv2d:
                aug_entries.append({"type": "conv2d", **item.conv2d.dict()})
            elif item.canny:
                aug_entries.append({"type": "canny", **item.canny.dict()})
            elif item.shot_noise:
                aug_entries.append({"type": "shot_noise", **item.shot_noise.dict()})
            elif item.speckle_noise:
                aug_entries.append({"type": "speckle_noise", **item.speckle_noise.dict()})
            elif item.cropresize:
                aug_entries.append({"type": "cropresize", **item.cropresize.dict()})
            elif item.rotate:
                aug_entries.append({"type": "rotate", **item.rotate.dict()})
            elif item.rand_conv:
                aug_entries.append({"type": "rand_conv", **item.rand_conv.dict()})
            elif item.cutmix:
                aug_entries.append({"type": "cutmix", **item.cutmix.dict()})
            elif item.rgb2hsv:
                aug_entries.append({"type": "rgb2hsv", **item.rgb2hsv.dict()})
            elif item.hsv2rgb:
                aug_entries.append({"type": "hsv2rgb", **item.hsv2rgb.dict()})
            elif item.imgblend:
                aug_entries.append({"type": "imgblend", **item.imgblend.dict()})

    for entry in aug_entries:
        if entry.get("type") == "pixellate":
            cfg = dict(entry)
            cfg.pop("type", None)
            _append_aug("--aug-pixellate", cfg)
        elif entry.get("type") == "motion_blur":
            cfg = dict(entry)
            cfg.pop("type", None)
            _append_aug("--aug-motionblur", cfg)
        elif entry.get("type") == "glass_blur":
            cfg = dict(entry)
            cfg.pop("type", None)
            _append_aug("--aug-glassblur", cfg)
        elif entry.get("type") == "brightness":
            cfg = dict(entry)
            cfg.pop("type", None)
            _append_aug("--aug-brightness", cfg)
        elif entry.get("type") == "colorize_depth":
            cfg = dict(entry)
            cfg.pop("type", None)
            _append_aug("--aug-colorizedepth", cfg)
        elif entry.get("type") == "colorize_normals":
            cfg = dict(entry)
            cfg.pop("type", None)
            _append_aug("--aug-colorizenormals", cfg)
        elif entry.get("type") == "sobel":
            cfg = dict(entry)
            cfg.pop("type", None)
            _append_aug("--aug-sobel", cfg)
        elif entry.get("type") == "adjust_sigmoid":
            cfg = dict(entry)
            cfg.pop("type", None)
            _append_aug("--aug-adjustsigmoid", cfg)
        elif entry.get("type") == "contrast":
            cfg = dict(entry)
            cfg.pop("type", None)
            _append_aug("--aug-contrast", cfg)
        elif entry.get("type") == "conv2d":
            cfg = dict(entry)
            cfg.pop("type", None)
            _append_aug("--aug-conv2d", cfg)
        elif entry.get("type") == "canny":
            cfg = dict(entry)
            cfg.pop("type", None)
            _append_aug("--aug-canny", cfg)
        elif entry.get("type") == "shot_noise":
            cfg = dict(entry)
            cfg.pop("type", None)
            _append_aug("--aug-shotnoise", cfg)
        elif entry.get("type") == "speckle_noise":
            cfg = dict(entry)
            cfg.pop("type", None)
            _append_aug("--aug-specklenoise", cfg)
        elif entry.get("type") == "cropresize":
            cfg = dict(entry)
            cfg.pop("type", None)
            _append_aug("--aug-cropresize", cfg)
        elif entry.get("type") == "rotate":
            cfg = dict(entry)
            cfg.pop("type", None)
            _append_aug("--aug-rotate", cfg)
        elif entry.get("type") == "rand_conv":
            cfg = dict(entry)
            cfg.pop("type", None)
            _append_aug("--aug-randconv", cfg)
        elif entry.get("type") == "cutmix":
            cfg = dict(entry)
            cfg.pop("type", None)
            _append_aug("--aug-cutmix", cfg)
        elif entry.get("type") == "rgb2hsv":
            cfg = dict(entry)
            cfg.pop("type", None)
            _append_aug("--aug-rgb2hsv", cfg)
        elif entry.get("type") == "hsv2rgb":
            cfg = dict(entry)
            cfg.pop("type", None)
            _append_aug("--aug-hsv2rgb", cfg)
        elif entry.get("type") == "imgblend":
            cfg = dict(entry)
            cfg.pop("type", None)
            _append_aug("--aug-imgblend", cfg)

    return cmd

def _find_latest_zip(output_dir: Path) -> Optional[Path]:
    try:
        zips = [p for p in output_dir.glob("*.zip") if p.is_file()]
        if not zips:
            return None
        return max(zips, key=lambda p: p.stat().st_mtime)
    except Exception:
        return None


def _send_callback(hash_value: Optional[str], file_name: str) -> None:
    """Send callback to external API after successful upload.
    
    Args:
        hash_value: Hash identifier from the request
        file_name: Name of the uploaded file in GCS bucket
    """
    callback_url = os.environ.get("CALLBACK_URL", "")
    # Allow skipping SSL verification for local development
    verify_ssl_str = os.environ.get("CALLBACK_VERIFY_SSL", "true").lower()
    verify_ssl = verify_ssl_str not in ("false", "0", "no")
    
    if not callback_url:
        print(json.dumps({"status": "callback_skipped", "reason": "callback_url_not_configured"}))
        return
    
    if not hash_value:
        print(json.dumps({"status": "callback_skipped", "reason": "no_hash_provided", "file": file_name}))
        return
    
    payload = {
        "hash": hash_value,
        "fileName": file_name
    }
    
    try:
        import requests
        
        response = requests.post(
            callback_url,
            json=payload,
            timeout=30,
            headers={"Content-Type": "application/json"},
            verify=verify_ssl
        )
        
        if response.status_code in (200, 201, 202, 204):
            print(
                json.dumps(
                    {
                        "status": "callback_completed",
                        "url": callback_url,
                        "payload": payload,
                        "response_status": response.status_code,
                    }
                )
            )
        else:
            print(
                json.dumps(
                    {
                        "status": "callback_failed",
                        "url": callback_url,
                        "payload": payload,
                        "response_status": response.status_code,
                        "response_body": response.text[:200],
                    }
                ),
                file=sys.stderr,
            )
    except ImportError:
        print(
            json.dumps(
                {
                    "status": "callback_failed",
                    "error": "requests library not available",
                    "url": callback_url,
                    "payload": payload,
                }
            ),
            file=sys.stderr,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "callback_failed",
                    "error": str(exc),
                    "url": callback_url,
                    "payload": payload,
                }
            ),
            file=sys.stderr,
        )


def _upload_zip_to_gcs(zip_path: Path) -> bool:
    """Upload zip file to Google Cloud Storage.
    
    Args:
        zip_path: Path to the zip file to upload
    
    Returns:
        True if upload was successful, False otherwise
    
    Reads configuration from environment variables:
    - GCS_BUCKET_NAME: Name of the GCS bucket
    - GCS_PROJECT_ID: GCP project ID
    - GCS_BUCKET_LOCATION: Bucket location (unused, kept for compatibility)
    - GCS_BUCKET_DIRECTORY: Base directory in bucket
    - STORAGE_EMULATOR_HOST: If set, uses storage emulator for local testing
    """
    bucket_name = os.environ.get("GCS_BUCKET_NAME", "")
    bucket_dir = os.environ.get("GCS_BUCKET_DIRECTORY", "")
    project_id = os.environ.get("GCS_PROJECT_ID", "")
    emulator_host = os.environ.get("STORAGE_EMULATOR_HOST", "")
    key_file = ""

    if not zip_path.exists():
        print(json.dumps({"status": "gcs_upload_skipped", "reason": "zip_missing", "file": str(zip_path)}), file=sys.stderr)
        return False

    if not bucket_name:
        print(json.dumps({"status": "gcs_upload_skipped", "reason": "bucket_not_configured", "file": str(zip_path)}))
        return False

    blob_path_parts = [part for part in (bucket_dir, GCS_IMPORT_SUBDIR, zip_path.name) if part]
    destination_blob_name = "/".join(blob_path_parts)

    try:
        from google.cloud import storage  # type: ignore
    except Exception as exc:  # pragma: no cover - import failure is logged
        print(
            json.dumps(
                {
                    "status": "gcs_upload_failed",
                    "error": f"google-cloud-storage not available: {exc}",
                    "bucket": bucket_name,
                    "destination": destination_blob_name,
                    "file": str(zip_path),
                }
            ),
            file=sys.stderr,
        )
        return False

    try:
        # Configure client for emulator or production
        if emulator_host:
            # Use storage emulator for local testing
            os.environ["STORAGE_EMULATOR_HOST"] = emulator_host
            client = storage.Client(project=project_id or "test-project")
            print(json.dumps({"status": "gcs_using_emulator", "host": emulator_host}))
        elif key_file:
            client = storage.Client.from_service_account_json(key_file, project=project_id)
        elif project_id:
            client = storage.Client(project=project_id)
        else:
            client = storage.Client()

        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(str(zip_path))
        print(
            json.dumps(
                {
                    "status": "gcs_upload_completed",
                    "bucket": bucket_name,
                    "destination": destination_blob_name,
                    "file": str(zip_path),
                }
            )
        )
        
        return True
    except Exception as exc:  # pragma: no cover
        print(
            json.dumps(
                {
                    "status": "gcs_upload_failed",
                    "error": str(exc),
                    "bucket": bucket_name,
                    "destination": destination_blob_name,
                    "file": str(zip_path),
                }
            ),
            file=sys.stderr,
        )
        return False


def _run_load_stage_and_convert(
    command: List[str],
    output_dir: Path,
    dataset_name: str,
    img_width: int,
    img_height: int,
    convert_images_to_jpeg: bool,
    jpeg_quality: Optional[int],
    include_labels: Optional[List[str]],
    cleanup_after_zip: bool,
    hash_value: Optional[str] = None,
    dataset_format: str = "yolo",
) -> None:
    dataset_subdir = "voc" if dataset_format == "pascal_voc" else "yolo"
    dataset_dir = output_dir / dataset_subdir

    try:
        subprocess.run(
            command,
            check=True,
            cwd=str(SCRIPT_DIR),
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        print(
            json.dumps(
                {
                    "status": "stage_completed",
                    "output_dir": str(output_dir),
                    "command": command,
                }
            )
        )

        convert_script = CONVERT_VOC_SCRIPT if dataset_format == "pascal_voc" else CONVERT_YOLO_SCRIPT
        if convert_script.exists():
            convert_cmd = [
                sys.executable,
                str(convert_script),
                "--run-dir",
                str(output_dir),
                "--output-dir",
                str(dataset_dir),
                "--dataset-name",
                dataset_name,
            ]
            if dataset_format == "pascal_voc":
                convert_cmd.extend([
                    "--dataset-image-width",
                    str(img_width),
                    "--dataset-image-height",
                    str(img_height),
                ])
            if convert_images_to_jpeg:
                convert_cmd.append("--convert-images-to-jpeg")
                if jpeg_quality is not None:
                    convert_cmd.extend(["--jpeg-quality", str(jpeg_quality)])
            # Forward filters
            if include_labels:
                for label in include_labels:
                    convert_cmd.extend(["--include-label", str(label)])
            subprocess.run(
                convert_cmd,
                check=True,
                cwd=str(SCRIPT_DIR),
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            print(
                json.dumps(
                    {
                        "status": "convert_completed",
                        "output_dir": str(dataset_dir),
                        "command": convert_cmd,
                    }
                )
            )
            # Zip the dataset subset after conversion
            if ZIP_DATASET_SCRIPT.exists():
                zip_cmd = [
                    sys.executable,
                    str(ZIP_DATASET_SCRIPT),
                    "--input-dir",
                    str(dataset_dir),
                    "--format",
                    dataset_format,
                    "--dataset-name",
                    dataset_name,
                    "--output-dir",
                    str(output_dir),
                ]
                subprocess.run(
                    zip_cmd,
                    check=True,
                    cwd=str(SCRIPT_DIR),
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                )
                print(
                    json.dumps(
                        {
                            "status": "zip_completed",
                            "input_dir": str(dataset_dir),
                            "output_dir": str(output_dir),
                            "command": zip_cmd,
                        }
                    )
                )
                
                # Upload zip to GCS if configured
                latest_zip = _find_latest_zip(output_dir)
                if latest_zip:
                    upload_success = _upload_zip_to_gcs(latest_zip)
                    
                    # Send callback after successful upload
                    if upload_success and hash_value:
                        _send_callback(hash_value, latest_zip.name)
                    else:
                        print(json.dumps({"status": "callback_skipped", "reason": "no_hash_provided or upload_failed", "file": latest_zip.name}))
                else:
                    print(json.dumps({"status": "gcs_upload_skipped", "reason": "no_zip_found", "output_dir": str(output_dir)}))
                
                if cleanup_after_zip:
                    try:
                        removed = []
                        for entry in output_dir.iterdir():
                            try:
                                if entry.is_file() and entry.suffix.lower() == ".zip":
                                    continue
                                if entry.is_dir():
                                    shutil.rmtree(entry, ignore_errors=True)
                                    removed.append(str(entry))
                                else:
                                    entry.unlink()
                                    removed.append(str(entry))
                            except Exception:
                                pass
                        print(json.dumps({"status": "cleanup_completed", "removed": removed, "output_dir": str(output_dir)}))
                    except Exception as exc:
                        print(json.dumps({"status": "cleanup_failed", "error": str(exc), "output_dir": str(output_dir)}), file=sys.stderr)
            else:
                print(json.dumps({"status": "skip_zip", "reason": "zip script not found"}))
        else:
            print(json.dumps({"status": "skip_voc", "reason": "converter not found"}))
    except subprocess.CalledProcessError as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "returncode": exc.returncode,
                    "cmd": exc.cmd,
                    "output_dir": str(output_dir),
                }
            ),
            file=sys.stderr,
        )
    except Exception as exc:  # pragma: no cover
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": str(exc),
                    "output_dir": str(output_dir),
                }
            ),
            file=sys.stderr,
        )
    finally:
        if RUN_LOCK.locked():
            RUN_LOCK.release()


@app.post("/load-stage")
async def create_load_stage(request: LoadStageRequest, tasks: BackgroundTasks):
    if RUN_LOCK.locked():
        raise HTTPException(status_code=409, detail="Another load_stage run is already in progress.")

    await RUN_LOCK.acquire()

    try:
        if request.tmp_root:
            tmp_root = Path(request.tmp_root).resolve()
            tmp_root.mkdir(parents=True, exist_ok=True)
            tmp_dir = Path(tempfile.mkdtemp(prefix="load_stage_", dir=str(tmp_root)))
        else:
            tmp_dir = Path(tempfile.mkdtemp(prefix="load_stage_"))
        command = _build_load_stage_command(request, tmp_dir)
    except Exception as exc:
        RUN_LOCK.release()
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Derive dataset metadata
    dataset_name = Path(request.usd_path).name
    tasks.add_task(
        _run_load_stage_and_convert,
        command,
        tmp_dir,
        dataset_name,
        request.width,
        request.height,
        request.convert_images_to_jpeg,
        request.jpeg_quality,
        request.include_labels,
        request.cleanup_after_zip,
        request.hash_request,
        request.dataset_format,
    )

    return {"status": "accepted", "output_dir": str(tmp_dir), "command": command}


@app.get("/healthz")
async def healthcheck():
    return {"status": "ok"}
