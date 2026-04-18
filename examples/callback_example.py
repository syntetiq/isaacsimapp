#!/usr/bin/env python3
"""
Example: Using the callback feature with hash_request parameter.

This demonstrates how to include a hash_request in your request to receive
a callback notification after the dataset is uploaded to GCS.
"""

import requests
import json

# Example request with hash parameter
payload = {
    "usd_path": "/Isaac/Environments/Simple_Room/simple_room.usd",
    "frames": 5,
    "width": 640,
    "height": 640,
    "focal_length": 10,
    "camera_pos": [0.38605, 4.22024, 0.60473],
    "spawn_cube": True,
    "cube_translate": [0.8, 3.0, 0.2],
    "cube_scale": [0.2, 0.2, 0.2],
    "cube_size": 1.0,
    "tmp_root": "data/tmp",
    "convert_images_to_jpeg": True,
    "jpeg_quality": 92,
    "cleanup_after_zip": True,
    
    # IMPORTANT: Add hash_request to receive callback notification
    "hash_request": "c4ca4238a0b923820dcc509a6f75849b"
}

api_url = "http://localhost:9999/load-stage"

try:
    print(f"Sending request to {api_url}...")
    print(f"Hash Request: {payload['hash_request']}")
    
    response = requests.post(api_url, json=payload, timeout=10)
    
    if response.status_code == 202:
        result = response.json()
        print(f"✓ Request accepted!")
        print(f"  Output directory: {result.get('output_dir')}")
        print(f"\nWorkflow:")
        print(f"  1. Dataset will be generated and zipped")
        print(f"  2. Zip will be uploaded to GCS")
        print(f"  3. Callback will be sent to: https://syntetiq.docker.localhost/api/omniverse/callback")
        print(f"     with payload: {{'hash': '{payload['hash_request']}', 'fileName': '<zip_filename>'}}")
        print(f"\nMonitor the uvicorn console for status updates.")
    elif response.status_code == 409:
        print("⚠ Another job is already running. Please wait and try again.")
    else:
        print(f"✗ Request failed with status {response.status_code}")
        print(f"  Response: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("✗ Could not connect to the dataset server.")
    print("  Make sure the server is running with:")
    print("  uvicorn scripts.dataset_server:app --host 0.0.0.0 --port 9999")
except Exception as e:
    print(f"✗ Error: {e}")
