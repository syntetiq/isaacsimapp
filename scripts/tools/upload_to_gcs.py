import argparse
import os
import sys
from google.cloud import storage

def upload_blob(bucket_name, source_file_name, destination_blob_name, project_id=None):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    try:
        if project_id:
            storage_client = storage.Client(project=project_id)
        else:
            storage_client = storage.Client()

        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(source_file_name)

        print(
            f"File {source_file_name} uploaded to {destination_blob_name} in bucket {bucket_name}."
        )
    except Exception as e:
        print(f"Error uploading file: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Upload a file to Google Cloud Storage.")
    parser.add_argument("--file", required=True, help="Path to the file to upload.")
    parser.add_argument("--bucket", required=True, help="Name of the GCS bucket.")
    parser.add_argument("--destination", help="Destination blob name (optional, defaults to file name).")
    parser.add_argument("--project", help="GCS project ID (optional).")

    args = parser.parse_args()

    if not os.path.isfile(args.file):
        print(f"Error: File '{args.file}' not found.")
        sys.exit(1)

    destination_blob_name = args.destination if args.destination else os.path.basename(args.file)

    upload_blob(args.bucket, args.file, destination_blob_name, args.project)

if __name__ == "__main__":
    main()
