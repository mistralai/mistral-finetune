#!/bin/bash
# Setup script for RTX 5090 support with CUDA 12.9

set -e

echo "Setting up mistral-finetune for RTX 5090 support..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Set pip cache to avoid filling disk
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-"/tmp/pip-cache"}
mkdir -p $PIP_CACHE_DIR

# Install PyTorch nightly with CUDA 12.9 support
echo "Installing PyTorch nightly with CUDA 12.9..."
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129

# Install other requirements
echo "Installing other requirements..."
pip install -r requirements.txt

# Install xformers with no-deps to avoid conflicts
echo "Installing xformers..."
pip install xformers==0.0.31.post1 --no-deps

# Additional dependencies that might be needed
pip install accelerate>=0.34.2
pip install datasets>=2.19.0
pip install transformers>=4.53.0
pip install bitsandbytes>=0.44.1
pip install ninja  # For faster builds

echo "Setup complete!"
echo "To activate the environment: source venv/bin/activate"