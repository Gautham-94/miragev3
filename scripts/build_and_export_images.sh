#!/bin/sh
# Builds the three mirage Docker images (pipeline/api/frontend) and exports each to a
# .tar file under dist/ -- run this yourself wherever Docker is available (needs the
# full repo checkout: Dockerfile.pipeline/.api/.frontend, mirage/, frontend/,
# requirements*.txt, pyproject.toml, docker/nginx.conf). The client only ever receives
# the resulting .tar files + docker-compose.client.yml + DOCKER_PREREQUISITES.md, never
# this script or the source it builds from.
#
# Usage: ./scripts/build_and_export_images.sh [tag]
# Default tag: whatever's in VERSION below, or pass one explicitly, e.g.
#   ./scripts/build_and_export_images.sh 1.2.0

set -eu

VERSION="${1:-latest}"
IMAGE_PREFIX="mirage"
OUT_DIR="dist"

echo "Building images tagged :${VERSION} ..."
docker build -f Dockerfile.pipeline -t "${IMAGE_PREFIX}-pipeline:${VERSION}" .
docker build -f Dockerfile.api      -t "${IMAGE_PREFIX}-api:${VERSION}" .
docker build -f Dockerfile.frontend -t "${IMAGE_PREFIX}-frontend:${VERSION}" .

mkdir -p "${OUT_DIR}"

echo "Exporting images to ${OUT_DIR}/*.tar ..."
docker save "${IMAGE_PREFIX}-pipeline:${VERSION}"  -o "${OUT_DIR}/mirage-pipeline-${VERSION}.tar"
docker save "${IMAGE_PREFIX}-api:${VERSION}"       -o "${OUT_DIR}/mirage-api-${VERSION}.tar"
docker save "${IMAGE_PREFIX}-frontend:${VERSION}"  -o "${OUT_DIR}/mirage-frontend-${VERSION}.tar"

echo
echo "Done. Ship these to the client:"
echo "  ${OUT_DIR}/mirage-pipeline-${VERSION}.tar"
echo "  ${OUT_DIR}/mirage-api-${VERSION}.tar"
echo "  ${OUT_DIR}/mirage-frontend-${VERSION}.tar"
echo "  docker-compose.client.yml  (edit its image: tags to match \"${VERSION}\" if not \"latest\")"
echo "  DOCKER_PREREQUISITES.md"
