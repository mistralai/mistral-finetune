# Dockerfile for Mistral AI Fine-Tuning
# Supports LoRA fine-tuning of Mistral models with GPU acceleration
#
# Build:
#   docker build -t mistral-finetune .
#
# Run (single GPU):
#   docker run --gpus all -v /path/to/data:/data -v /path/to/model:/model \
#     mistral-finetune --config /data/config.yaml
#
# Run (multi-GPU with torchrun):
#   docker run --gpus all -v /path/to/data:/data -v /path/to/model:/model \
#     --entrypoint torchrun mistral-finetune \
#     --nproc_per_node=4 /app/train.py --config /data/config.yaml

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Default entrypoint for single-GPU training
ENTRYPOINT ["python", "-m", "train"]
