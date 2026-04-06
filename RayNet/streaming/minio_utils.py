"""
MinIO utilities for uploading and downloading MDS shards, and configuring streaming access.

MinIO provides S3-compatible object storage. MosaicML Streaming connects
to MinIO using standard S3 environment variables.

Usage:
    # Upload shards
    python -m RayNet.streaming.minio_utils upload \
        --shard_dir ./mds_shards/train \
        --bucket gazegene \
        --prefix train \
        --endpoint http://localhost:9000

    # Download shards
    python -m RayNet.streaming.minio_utils download \
        --local_dir ./mds_shards/train_local \
        --bucket gazegene \
        --prefix train \
        --endpoint http://localhost:9000

    # Set env vars for streaming
    from RayNet.streaming.minio_utils import configure_minio_env
    configure_minio_env('http://localhost:9000', 'minioadmin', 'minioadmin')
"""

import os
import glob

try:
    from minio import Minio
except ImportError:
    Minio = None


def configure_minio_env(endpoint_url, access_key, secret_key):
    """
    Set environment variables so mosaicml-streaming connects to MinIO
    instead of AWS S3.

    Must be called BEFORE creating StreamingDataset instances.

    Args:
        endpoint_url: MinIO endpoint (e.g. 'http://localhost:9000')
        access_key: MinIO access key
        secret_key: MinIO secret key
    """
    os.environ['S3_ENDPOINT_URL'] = endpoint_url
    os.environ['AWS_ACCESS_KEY_ID'] = access_key
    os.environ['AWS_SECRET_ACCESS_KEY'] = secret_key
    # Disable SSL verification for local MinIO
    if endpoint_url.startswith('http://'):
        os.environ['S3_USE_SSL'] = '0'
    print(f"MinIO configured: {endpoint_url}")


def upload_to_minio(shard_dir, bucket, prefix='',
                    endpoint='http://localhost:9000',
                    access_key=None, secret_key=None,
                    make_bucket=True):
    """
    Upload MDS shard directory to MinIO.

    Uploads all files (index.json, shard.*.mds, etc.) preserving
    the flat directory structure expected by mosaicml-streaming.

    Args:
        shard_dir: Local directory containing MDS shard files
        bucket: MinIO bucket name (e.g. 'gazegene')
        prefix: Object key prefix (e.g. 'train' or 'val')
        endpoint: MinIO endpoint URL
        access_key: MinIO access key (default: from env AWS_ACCESS_KEY_ID)
        secret_key: MinIO secret key (default: from env AWS_SECRET_ACCESS_KEY)
        make_bucket: Create bucket if it doesn't exist
    """
    assert Minio is not None, "pip install minio"

    access_key = access_key or os.environ.get('AWS_ACCESS_KEY_ID', 'minioadmin')
    secret_key = secret_key or os.environ.get('AWS_SECRET_ACCESS_KEY', 'minioadmin')

    # Parse endpoint
    secure = endpoint.startswith('https://')
    host = endpoint.replace('https://', '').replace('http://', '')

    client = Minio(
        host,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure,
    )

    # Create bucket
    if make_bucket and not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        print(f"Created bucket: {bucket}")

    # Upload all files
    files = sorted(glob.glob(os.path.join(shard_dir, '*')))
    print(f"Uploading {len(files)} files to s3://{bucket}/{prefix}/")

    for filepath in files:
        filename = os.path.basename(filepath)
        object_name = f"{prefix}/{filename}" if prefix else filename

        client.fput_object(bucket, object_name, filepath)
        size_mb = os.path.getsize(filepath) / 1e6
        print(f"  {object_name} ({size_mb:.1f} MB)")

    total_mb = sum(os.path.getsize(f) for f in files) / 1e6
    print(f"Done. {total_mb:.0f} MB uploaded to s3://{bucket}/{prefix}/")


def download_from_minio(local_dir, bucket, prefix='',
                        endpoint='http://localhost:9000',
                        access_key=None, secret_key=None):
    """
    Download MDS shard directory from MinIO to avoid streaming bottlenecks.

    Downloads all files under the specified bucket and prefix into
    a flat local directory.

    Args:
        local_dir: Local directory to save the downloaded MDS shard files
        bucket: MinIO bucket name (e.g. 'gazegene')
        prefix: Object key prefix (e.g. 'train' or 'val')
        endpoint: MinIO endpoint URL
        access_key: MinIO access key (default: from env AWS_ACCESS_KEY_ID)
        secret_key: MinIO secret key (default: from env AWS_SECRET_ACCESS_KEY)
    """
    assert Minio is not None, "pip install minio"

    access_key = access_key or os.environ.get('AWS_ACCESS_KEY_ID', 'minioadmin')
    secret_key = secret_key or os.environ.get('AWS_SECRET_ACCESS_KEY', 'minioadmin')

    # Parse endpoint
    secure = endpoint.startswith('https://')
    host = endpoint.replace('https://', '').replace('http://', '')

    client = Minio(
        host,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure,
    )

    if not client.bucket_exists(bucket):
        raise ValueError(f"Bucket '{bucket}' does not exist.")

    # Ensure local directory exists
    os.makedirs(local_dir, exist_ok=True)

    # Fetch objects in the bucket matching the prefix
    objects = client.list_objects(bucket, prefix=prefix, recursive=True)

    files_downloaded = 0
    total_bytes = 0

    print(f"Downloading from s3://{bucket}/{prefix} to {local_dir}/")

    for obj in objects:
        if obj.is_dir:
            continue

        # Extract filename to maintain the flat directory structure locally
        filename = os.path.basename(obj.object_name)
        if not filename:
            continue

        local_filepath = os.path.join(local_dir, filename)

        # Download the file
        client.fget_object(bucket, obj.object_name, local_filepath)

        size_mb = obj.size / 1e6
        total_bytes += obj.size
        files_downloaded += 1
        print(f"  {obj.object_name} -> {local_filepath} ({size_mb:.1f} MB)")

    total_mb = total_bytes / 1e6
    print(f"Done. {files_downloaded} files ({total_mb:.0f} MB) downloaded to {local_dir}/")


def minio_shard_url(bucket, prefix, endpoint=None):
    """
    Build the remote URL string for mosaicml-streaming.

    Args:
        bucket: MinIO bucket name
        prefix: Object key prefix (e.g. 'train')
        endpoint: MinIO endpoint (uses S3_ENDPOINT_URL env var if None)

    Returns:
        URL string like 's3://gazegene/train'

    Note:
        S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
        must be set as environment variables for streaming to connect.
    """
    url = f"s3://{bucket}/{prefix}" if prefix else f"s3://{bucket}"
    if endpoint:
        print(f"Note: Set S3_ENDPOINT_URL={endpoint} for MinIO access")
    return url


def verify_minio_connection(endpoint='http://localhost:9000',
                            access_key=None, secret_key=None):
    """
    Verify MinIO is reachable and list available buckets.

    Returns:
        list of bucket names, or raises on failure
    """
    assert Minio is not None, "pip install minio"

    access_key = access_key or os.environ.get('AWS_ACCESS_KEY_ID', 'minioadmin')
    secret_key = secret_key or os.environ.get('AWS_SECRET_ACCESS_KEY', 'minioadmin')

    secure = endpoint.startswith('https://')
    host = endpoint.replace('https://', '').replace('http://', '')

    client = Minio(host, access_key=access_key, secret_key=secret_key,
                   secure=secure)

    buckets = [b.name for b in client.list_buckets()]
    print(f"MinIO connected: {endpoint}")
    print(f"  Buckets: {buckets}")
    return buckets


def main():
    import argparse
    parser = argparse.ArgumentParser(description='MinIO utilities')
    subparsers = parser.add_subparsers(dest='command')

    # Upload command
    sp_upload = subparsers.add_parser('upload', help='Upload MDS shards to MinIO')
    sp_upload.add_argument('--shard_dir', type=str, required=True)
    sp_upload.add_argument('--bucket', type=str, default='gazegene')
    sp_upload.add_argument('--prefix', type=str, default='train')
    sp_upload.add_argument('--endpoint', type=str, default='http://localhost:9000')
    sp_upload.add_argument('--access_key', type=str, default=None)
    sp_upload.add_argument('--secret_key', type=str, default=None)

    # Download command
    sp_download = subparsers.add_parser('download', help='Download MDS shards from MinIO')
    sp_download.add_argument('--local_dir', type=str, required=True)
    sp_download.add_argument('--bucket', type=str, default='gazegene')
    sp_download.add_argument('--prefix', type=str, default='train')
    sp_download.add_argument('--endpoint', type=str, default='http://localhost:9000')
    sp_download.add_argument('--access_key', type=str, default=None)
    sp_download.add_argument('--secret_key', type=str, default=None)

    # Verify command
    sp_verify = subparsers.add_parser('verify', help='Verify MinIO connection')
    sp_verify.add_argument('--endpoint', type=str, default='http://localhost:9000')

    args = parser.parse_args()

    if args.command == 'upload':
        upload_to_minio(
            shard_dir=args.shard_dir,
            bucket=args.bucket,
            prefix=args.prefix,
            endpoint=args.endpoint,
            access_key=args.access_key,
            secret_key=args.secret_key,
        )
    elif args.command == 'download':
        download_from_minio(
            local_dir=args.local_dir,
            bucket=args.bucket,
            prefix=args.prefix,
            endpoint=args.endpoint,
            access_key=args.access_key,
            secret_key=args.secret_key,
        )
    elif args.command == 'verify':
        verify_minio_connection(endpoint=args.endpoint)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
