#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

SAM_VERSION="${SAM_VERSION:-small}"
VITE_BASE_PATH="${VITE_BASE_PATH:-/freeze-me/}"

echo "Building rm_freeze_me_backend:latest with SAM_VERSION=${SAM_VERSION}"
cd "${REPO_ROOT}/backend"
podman build --build-arg "SAM_VERSION=${SAM_VERSION}" -t rm_freeze_me_backend:latest .

echo "Building rm_freeze_me_frontend:latest with VITE_BASE_PATH=${VITE_BASE_PATH}"
cd "${REPO_ROOT}/frontend"
podman build --build-arg "VITE_BASE_PATH=${VITE_BASE_PATH}" -t rm_freeze_me_frontend:latest .

echo "Freeze Me server images built successfully."
