# %%
import os
from pathlib import Path

import boto3
import start_research
from tqdm import tqdm


# %%
def download_s3_bucket(
    bucket_name: str,
    prefix: str = "",
    output_dir: Path = Path("./data"),
    region: str = "us-east-2",
    endpoint_url: str = "https://s3.us-east-2.amazonaws.com",
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
    max_workers: int = 10,
):
    """
    Downloads all objects from an S3 bucket to a local directory.

    Args:
        bucket_name (str): Name of the S3 bucket
        prefix (str): Prefix to filter objects (like a directory in S3)
        output_dir (str): Local directory to save downloaded files
        region (str): AWS region
        endpoint_url (str): S3 endpoint URL
        aws_access_key_id (str): AWS access key ID
        aws_secret_access_key (str): AWS secret access key
        max_workers (int): Maximum number of concurrent downloads
    """
    import concurrent.futures

    # Create S3 client
    s3_client = boto3.client(
        "s3",
        region_name=region,
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_access_key_id or os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=aws_secret_access_key or os.environ["AWS_SECRET_ACCESS_KEY"],
    )

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # List all objects in the bucket with the given prefix
    print(f"Listing objects in s3://{bucket_name}/{prefix}...")
    paginator = s3_client.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    # Count total objects for progress tracking
    objects = []
    for page in page_iterator:
        if "Contents" in page:
            objects.extend(page["Contents"])

    if not objects:
        print(f"No objects found in s3://{bucket_name}/{prefix}")
        return

    print(f"Found {len(objects)} objects. Starting download...")

    # Function to download a single object
    def download_object(obj):
        # Get the object key (path)
        key = obj["Key"]

        # Create local directory structure
        rel_path = key
        if prefix:
            # Remove prefix if specified to maintain relative structure
            if key.startswith(prefix):
                rel_path = key[len(prefix) :]
                if rel_path.startswith("/"):
                    rel_path = rel_path[1:]

        local_path = output_path / rel_path

        # Check if file already exists
        if local_path.exists():
            return key, "skipped"

        # Create parent directories if they don't exist
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Download the file
        try:
            s3_client.download_file(bucket_name, key, str(local_path))
            return key, "downloaded"
        except Exception as e:
            return key, f"error: {str(e)}"

    # Use ThreadPoolExecutor for concurrent downloads
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_key = {executor.submit(download_object, obj): obj["Key"] for obj in objects}

        # Process results as they complete
        downloaded = 0
        skipped = 0
        errors = 0

        for future in tqdm(
            concurrent.futures.as_completed(future_to_key), total=len(objects), desc="Downloading"
        ):
            key, status = future.result()
            if status == "downloaded":
                downloaded += 1
            elif status == "skipped":
                skipped += 1
            else:
                errors += 1
                print(f"Error downloading {key}: {status}")

    print(f"Download complete. Files saved to {output_dir}")
    print(f"Summary: {downloaded} downloaded, {skipped} skipped, {errors} errors")


# %%

download_s3_bucket(
    bucket_name="emg2qwerty",
    prefix="dataset/emg2qwerty-data-2021-08",
    output_dir=Path("mnt/dataset/"),
    region="us-east-2",
    endpoint_url="https://s3.us-east-2.amazonaws.com",
)

# s3://emg2qwerty/dataset/emg2qwerty-data-2021-08/2020-08-13-1597354281-keystrokes.hdf5

# %%
