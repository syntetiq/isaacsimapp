# isaacsimapp

A Python toolkit built on top of [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) and Omni Replicator for generating synthetic computer vision datasets. It loads a USD stage, captures RGB frames and bounding boxes, applies a configurable pipeline of image augmentations, exports to YOLO or Pascal VOC, zips the result with a content-based hash, and optionally uploads to Google Cloud Storage with a webhook callback.

It can be used either as a CLI ([scripts/load_stage.py](scripts/load_stage.py)) or as a FastAPI REST server ([scripts/dataset_server.py](scripts/dataset_server.py)).

## Features

- USD stage loading with configurable camera (position, rotation, focal length, motion path).
- Optional procedural cube spawning for quick smoke tests.
- 20+ augmentations with per-augmentor `frequency` (capture every Nth frame): brightness, contrast, motion blur, glass blur, pixellate, rotate, crop & resize, CutMix, Sobel, Canny, conv2d, random conv, shot noise, speckle noise, RGB↔HSV, colorize depth, colorize normals, sigmoid adjust, image blend.
- Bounding box export via Replicator `BasicWriter` with automatic conversion to **YOLO** or **Pascal VOC**.
- Content-hash zipping, optional GCS upload, optional webhook callback keyed by `hash_request`.
- FastAPI REST server with single-job locking (one dataset run at a time).

## Requirements

- NVIDIA Isaac Sim with a configured conda environment (install instructions are out of scope; see NVIDIA's official documentation).
- Python 3 inside the Isaac Sim environment (uses `isaacsim`, `omni.replicator.core`).
- Python packages: `fastapi`, `uvicorn`, `pydantic`, `python-dotenv`, `numpy`, `pillow`, `google-cloud-storage`, `requests`.

## Setup

1. Clone this repository.
2. Install Isaac Sim and create a conda environment for it.
3. Copy the env template and fill in your values:
   ```bash
   cp .env.example .env
   ```
4. Install Python dependencies into the Isaac Sim environment:
   ```bash
   pip install fastapi uvicorn pydantic python-dotenv numpy pillow google-cloud-storage requests
   ```

## Usage — REST server

Start the server (from inside the Isaac Sim Python environment):

```bash
uvicorn scripts.dataset_server:app --host 0.0.0.0 --port 8000
```

Endpoints:

| Method | Path          | Description                                  |
| ------ | ------------- | -------------------------------------------- |
| POST   | `/load-stage` | Trigger a dataset generation job             |
| GET    | `/healthz`    | Health check                                 |

`POST /load-stage` accepts a JSON body matching the `LoadStageRequest` model in [scripts/dataset_server.py:205](scripts/dataset_server.py:205). Minimal example:

```json
{
  "usd_path": "/Isaac/Environments/Simple_Room/simple_room.usd",
  "frames": 5,
  "width": 640,
  "height": 640,
  "focal_length": 10,
  "camera_pos": [0.38605, 4.22024, 0.60473],
  "camera_pos_end": [2.0, 4.22024, 0.60473],
  "camera_rotation": [76.0, 0.0, 175.0],
  "label_name": "table_low_123",
  "spawn_cube": true,
  "cube_translate": [0.8, 3.0, 0.2],
  "cube_scale": [0.2, 0.2, 0.2],
  "cube_size": 1.0,
  "augmentation": [
    {"pixellate":   {"kernel": 12, "frequency": 5}},
    {"motion_blur": {"angle": 45.0, "strength": 0.7, "kernel": 11, "frequency": 3}}
  ],
  "tmp_root": "data/tmp",
  "convert_images_to_jpeg": true,
  "jpeg_quality": 92,
  "cleanup_after_zip": true,
  "include_labels": ["cube", "table"]
}
```

Response codes:

| Code | Meaning                                    |
| ---- | ------------------------------------------ |
| 202  | Job accepted; runs in the background       |
| 409  | Another job is already running             |
| 400  | Invalid request                            |

Pass `hash_request` in the body to receive a callback once the zip has been uploaded. The service will `POST` `{"hash": "<request_hash>", "fileName": "<zip_filename>"}` to `CALLBACK_URL`. See [examples/callback_example.py](examples/callback_example.py) and [examples/gcs_upload_example.py](examples/gcs_upload_example.py).

## Usage — CLI

`scripts/load_stage.py` runs the simulation directly. It must be executed inside the Isaac Sim Python environment:

```bash
python scripts/load_stage.py \
  --usd-path /Isaac/Environments/Simple_Room/simple_room.usd \
  --frames 5 \
  --width 640 --height 640 \
  --augmentation '[{"motion_blur": {"angle": 45, "strength": 0.7, "kernel": 11, "frequency": 3}}]'
```

The script accepts 50+ flags for camera, cube spawning, augmentations, and output. Run with `--help` for the full list.

## Output formats

| Format     | Layout                                                                    |
| ---------- | ------------------------------------------------------------------------- |
| YOLO       | Normalized bboxes in `.txt` files, split into `train`/`val`/`test`        |
| Pascal VOC | XML in `Annotations/`, JPEGs in `JPEGImages/`                             |

Conversion is handled by [scripts/tools/convert_to_yolo.py](scripts/tools/convert_to_yolo.py) and [scripts/tools/convert_to_pascal_voc.py](scripts/tools/convert_to_pascal_voc.py); zipping by [scripts/tools/zip_dataset.py](scripts/tools/zip_dataset.py); GCS upload by [scripts/tools/upload_to_gcs.py](scripts/tools/upload_to_gcs.py).

## Project layout

```
isaacsimapp/
├── scripts/
│   ├── load_stage.py          # Main CLI: load USD, capture frames, run augmentations
│   ├── dataset_server.py      # FastAPI server (POST /load-stage, GET /healthz)
│   ├── augmentators/          # Augmentation modules (one per technique)
│   └── tools/
│       ├── convert_to_yolo.py
│       ├── convert_to_pascal_voc.py
│       ├── zip_dataset.py
│       ├── upload_to_gcs.py
│       └── show_npy.py        # Debug utility for .npy bbox arrays
├── examples/
│   ├── gcs_upload_example.py
│   └── callback_example.py
├── .env.example
├── LICENSE
└── README.md
```

## Environment variables

Configured via `.env` (loaded with `python-dotenv`). See [.env.example](.env.example).

| Variable                | Purpose                                                                              |
| ----------------------- | ------------------------------------------------------------------------------------ |
| `GCS_BUCKET_NAME`       | Target GCS bucket name                                                               |
| `GCS_PROJECT_ID`        | GCP project ID                                                                       |
| `GCS_BUCKET_DIRECTORY`  | Base directory in the bucket; zips go to `{GCS_BUCKET_DIRECTORY}/import_tmp/`        |
| `GCS_BUCKET_LOCATION`   | Bucket location (currently unused, kept for compatibility)                           |
| `STORAGE_EMULATOR_HOST` | Optional GCS emulator URL for local dev (e.g. `http://localhost:4443`)               |
| `CALLBACK_URL`          | URL that receives `{hash, fileName}` after a successful upload                       |
| `CALLBACK_VERIFY_SSL`   | Set to `false` to skip SSL verification on callbacks (not recommended in production) |
| `CONDA_ENV_PATH`        | Path to the Isaac Sim conda environment                                              |

## License

Apache License 2.0. See [LICENSE](LICENSE).
