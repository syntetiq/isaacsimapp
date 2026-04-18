#!/usr/bin/env python3
"""
Example: How to set up and use GCS upload with the dataset server.

This script demonstrates:
1. Setting environment variables for GCS upload
2. Making a request to the dataset server
3. Verifying the upload to GCS
"""

import os
import requests
import json

# Set GCS configuration via environment variables
# Option 1: For production GCS
os.environ["GCS_BUCKET_NAME"] = "your-actual-bucket-name"
os.environ["GCS_PROJECT_ID"] = "your-project-id"
os.environ["GCS_BUCKET_DIRECTORY"] = "data"
os.environ["GCS_BUCKET_LOCATION"] = "us-central1"

# Option 2: For local development with GCS emulator
# Uncomment the following line to use a local emulator:
# os.environ["STORAGE_EMULATOR_HOST"] = "http://localhost:4443"

# Example request payload
payload = {
    "usd_path": "/Isaac/Environments/Simple_Room/simple_room.usd",
    "frames": 5,
    "width": 640,
    "height": 640,
    "focal_length": 10,
    "camera_pos": [0.38605, 4.22024, 0.60473],
    "camera_pos_end": [2.0, 4.22024, 0.60473],
    "camera_rotation": [76.0, 0.0, 175.0],
    "label_name": "table_low_123",
    "spawn_cube": True,
    "cube_translate": [0.8, 3.0, 0.2],
    "cube_scale": [0.2, 0.2, 0.2],
    "cube_size": 1.0,
    "augmentation": [
        {"pixellate": {"kernel": 12, "frequency": 5}},
        {"motion_blur": {"angle": 45.0, "strength": 0.7, "kernel": 11, "frequency": 3}},
    ],
    "tmp_root": "data/tmp",
    "convert_images_to_jpeg": True,
    "jpeg_quality": 92,
    "cleanup_after_zip": True,
    "include_labels": ["cube", "table"],
}

# Make request to dataset server
api_url = "http://localhost:8000/load-stage"

try:
    print(f"Sending request to {api_url}...")
    response = requests.post(api_url, json=payload, timeout=10)
    
    if response.status_code == 202:
        result = response.json()
        print(f"✓ Request accepted!")
        print(f"  Output directory: {result.get('output_dir')}")
        print(f"\nThe dataset will be generated, zipped, and uploaded to:")
        print(f"  gs://{os.environ.get('GCS_BUCKET_NAME')}/{os.environ.get('GCS_BUCKET_DIRECTORY')}/import_tmp/")
        print(f"\nMonitor the uvicorn console for upload status.")
    elif response.status_code == 409:
        print("⚠ Another job is already running. Please wait and try again.")
    else:
        print(f"✗ Request failed with status {response.status_code}")
        print(f"  Response: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("✗ Could not connect to the dataset server.")
    print("  Make sure the server is running with:")
    print("  uvicorn scripts.dataset_server:app --host 0.0.0.0 --port 8000")
except Exception as e:
    print(f"✗ Error: {e}")
