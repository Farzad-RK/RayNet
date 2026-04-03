#!/usr/bin/env bash
# =============================================================================
# RayNet MinIO Setup Script
#
# Run after docker compose up to:
#   1. Create the gazegene bucket
#   2. Set anonymous read access (for streaming without per-request auth)
#   3. Verify the setup
#
# Usage:
#   chmod +x setup_minio.sh
#   ./setup_minio.sh
#
# Or with custom endpoint:
#   MINIO_ENDPOINT=http://myserver:9000 ./setup_minio.sh
# =============================================================================

set -euo pipefail

MINIO_ENDPOINT="${MINIO_ENDPOINT:-http://localhost:9000}"
MINIO_USER="${MINIO_ROOT_USER:-minioadmin}"
MINIO_PASS="${MINIO_ROOT_PASSWORD:-minioadmin}"
BUCKET="${MINIO_BUCKET:-gazegene}"
ALIAS="raynet"

echo "=== RayNet MinIO Setup ==="
echo "Endpoint: ${MINIO_ENDPOINT}"
echo "Bucket:   ${BUCKET}"
echo ""

# Check if mc (MinIO client) is installed locally
if ! command -v mc &> /dev/null; then
    echo "MinIO client (mc) not found. Using docker exec..."
    docker exec raynet-minio mc alias set ${ALIAS} http://localhost:9000 "${MINIO_USER}" "${MINIO_PASS}"
    docker exec raynet-minio mc mb --ignore-existing ${ALIAS}/${BUCKET}
    docker exec raynet-minio mc anonymous set download ${ALIAS}/${BUCKET}
    echo ""
    echo "Verifying..."
    docker exec raynet-minio mc ls ${ALIAS}/${BUCKET}/
    echo ""
    echo "=== Setup complete ==="
    echo "S3 endpoint:  ${MINIO_ENDPOINT}"
    echo "Bucket:       s3://${BUCKET}"
    echo "Console:      http://localhost:9001"
    exit 0
fi

# Using local mc
mc alias set ${ALIAS} "${MINIO_ENDPOINT}" "${MINIO_USER}" "${MINIO_PASS}"
mc mb --ignore-existing ${ALIAS}/${BUCKET}
mc anonymous set download ${ALIAS}/${BUCKET}

echo ""
echo "Verifying..."
mc ls ${ALIAS}/
mc admin info ${ALIAS}

echo ""
echo "=== Setup complete ==="
echo "S3 endpoint:  ${MINIO_ENDPOINT}"
echo "Bucket:       s3://${BUCKET}"
echo "Console:      http://localhost:9001"
echo ""
echo "Next steps:"
echo "  1. Convert dataset:  python -m RayNet.streaming.convert_to_mds --data_dir /path/to/GazeGene --output_dir ./mds_shards/train --split train"
echo "  2. Upload to MinIO:  python -m RayNet.streaming.minio_utils upload --shard_dir ./mds_shards/train --bucket ${BUCKET} --prefix train"
echo "  3. Stream in Colab:  See notebooks/train_colab_a100.ipynb"
