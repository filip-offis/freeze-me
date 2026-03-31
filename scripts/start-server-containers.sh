#!/usr/bin/env bash

set -euo pipefail

NETWORK_NAME="${NETWORK_NAME:-mosaic-network}"
BACKEND_CONTAINER_NAME="${BACKEND_CONTAINER_NAME:-rm_freeze_me_backend}"
FRONTEND_CONTAINER_NAME="${FRONTEND_CONTAINER_NAME:-rm_freeze_me_frontend}"
BACKEND_IMAGE="${BACKEND_IMAGE:-rm_freeze_me_backend:latest}"
FRONTEND_IMAGE="${FRONTEND_IMAGE:-rm_freeze_me_frontend:latest}"
VIDEOS_VOLUME="${VIDEOS_VOLUME:-freeze_me_videos}"

podman volume create "${VIDEOS_VOLUME}" >/dev/null

run_container() {
    local container_name="$1"
    local image_name="$2"
    shift 2

    echo "Starting ${container_name} container"

    if podman run --userns=host --rm -d \
        --name "${container_name}" \
        --network "${NETWORK_NAME}" \
        "$@" \
        "${image_name}"; then
        podman logs "${container_name}" || true
        echo "${container_name} container started successfully"
    else
        local exit_code=$?
        podman logs "${container_name}" || true
        echo "Failed to start ${container_name} container. Exit code: ${exit_code}"
        return "${exit_code}"
    fi
}

run_container "${BACKEND_CONTAINER_NAME}" "${BACKEND_IMAGE}" \
    -v "${VIDEOS_VOLUME}:/app/src/videos"

run_container "${FRONTEND_CONTAINER_NAME}" "${FRONTEND_IMAGE}"
