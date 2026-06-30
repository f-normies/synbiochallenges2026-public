#!/usr/bin/env bash
# Build the per-tool micromamba image. Run from the repo root or anywhere.
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="${IMAGE:-synbio2026-mamba:dev}"

# --platform forced because the build host is darwin/arm64 but the target
# (enroot/V100 cluster) is linux/amd64.
docker buildx build \
  --platform linux/amd64 \
  -f docker/Dockerfile \
  -t "$IMAGE" \
  --load \
  .

echo "built ${IMAGE}"
echo "next: enroot import dockerd://${IMAGE}   # -> ${IMAGE/:/+}.sqsh"
