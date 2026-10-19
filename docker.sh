#!/bin/bash
IMAGE_NAME="local/audio-processor"
HOST_USER_MAPPING="$(id -u):$(id -g)"
LOCAL_CACHE="./.cache"

if [ ! -f .env ]; then
    echo "ERROR: .env not found. Copy the template first:"
    echo "  cp .env.example .env"
    exit 1
fi

set -a
source .env
set +a

echo "=> Building image"
docker build . -t $IMAGE_NAME

echo "=> Execute container"
docker run --rm -u "$HOST_USER_MAPPING" \
                -v "$INPUT_DIR:/data/input" \
                -v "$OUTPUT_DIR:/data/output" \
                -v "$LOCAL_CACHE:/data/cache" \
                -e XDG_CACHE_HOME=/data/cache \
                -e HF_HOME=/data/cache \
                -e TORCH_HOME=/data/cache/torch \
                -e NUMBA_CACHE_DIR=/tmp \
                -e USER=runner \
                -e INPUT_DIR=/data/input \
                -e OUTPUT_DIR=/data/output \
                "$IMAGE_NAME"