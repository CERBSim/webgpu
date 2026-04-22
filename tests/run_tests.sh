#!/usr/bin/env bash
# Usage: ./tests/run_tests.sh [--update-baselines]
set -e
cd "$(dirname "$0")/.."

BASE_IMAGE=webgpu-base
IMAGE=webgpu-tests
EXTRA=()

for arg in "$@"; do
    case "$arg" in
        --update-baselines) EXTRA+=(-e UPDATE_BASELINES=1) ;;
        *) echo "Usage: $0 [--update-baselines]"; exit 1 ;;
    esac
done

echo "==> Building base image..."
docker build -f tests/Dockerfile.base -t "$BASE_IMAGE" .

echo "==> Building test image..."
docker build -f tests/Dockerfile --build-arg BASE_IMAGE="$BASE_IMAGE" -t "$IMAGE" .

echo "==> Running tests..."
docker run --rm \
    -v "$(pwd)/tests/output:/app/tests/output" \
    -v "$(pwd)/tests/baselines:/app/tests/baselines" \
    "${EXTRA[@]}" \
    "$IMAGE"

echo "==> Done. Screenshots in tests/output/"
if [[ " ${EXTRA[*]} " == *"UPDATE_BASELINES=1"* ]]; then
    echo "==> Baselines updated in tests/baselines/"
fi