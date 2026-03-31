FROM python:3.10-slim

# System dependencies for OpenCV, ffmpeg, and general build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (CPU-only, no cupy)
COPY requirements.txt .
RUN grep -v "cupy-cuda12x" requirements.txt > requirements-cpu.txt && \
    pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements-cpu.txt && \
    rm requirements-cpu.txt

# Copy application source
COPY src/ src/

WORKDIR /app/src

# Create runtime directories
RUN mkdir -p videos

# Download the checkpoint matching the configured SAM version.
ARG SAM_VERSION=small
ENV SAM_VERSION=${SAM_VERSION}
RUN mkdir -p /app/checkpoints && \
    case "$SAM_VERSION" in \
        large) checkpoint_name="sam2.1_hiera_large.pt" ;; \
        b_plus) checkpoint_name="sam2.1_hiera_base_plus.pt" ;; \
        tiny) checkpoint_name="sam2.1_hiera_tiny.pt" ;; \
        *) checkpoint_name="sam2.1_hiera_small.pt" ;; \
    esac && \
    curl -fSL -o "/app/checkpoints/${checkpoint_name}" \
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/${checkpoint_name}"

ENV PYTORCH_ENABLE_MPS_FALLBACK=1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
