# RTX 5090 Support for mistral-finetune

This fork adds support for NVIDIA RTX 5090 GPUs and fixes compatibility issues with mistral-common v1.8.1.

## Key Changes

### 1. RTX 5090 Support
- Updated to PyTorch nightly builds with CUDA 12.9 support
- RTX 5090 requires CUDA 12.9 due to its sm_120 compute capability
- Updated all dependencies to be compatible with the latest PyTorch

### 2. mistral-common v1.8.1 Compatibility
- Fixed import errors: `InstructTokenizerBase` → `MistralTokenizer`
- Updated tokenizer initialization to use new API
- Fixed validation issues with training data ending in assistant messages
- Added workaround for `request.tools` vs `request.available_tools` bug in v1.8.1

### 3. Installation

#### Quick Setup
```bash
# Use the provided setup script for RTX 5090
chmod +x setup_rtx5090.sh
./setup_rtx5090.sh
```

#### Manual Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch nightly with CUDA 12.9
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129

# Install other requirements
pip install -r requirements.txt

# Install xformers (optional, may have compatibility issues)
pip install xformers==0.0.31.post1 --no-deps
```

### 4. Usage with RTX 5090

When using multiple GPUs, note that PyTorch may see GPUs in a different order than nvidia-smi:

```python
# Check GPU order in PyTorch
import torch
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")
```

To use RTX 5090 specifically:
```bash
# If RTX 5090 is device 1 in PyTorch
export CUDA_VISIBLE_DEVICES=1
python train.py your_config.yaml
```

### 5. Known Issues

1. **xformers compatibility**: xformers may not work properly with PyTorch nightly builds. The code will fall back to standard PyTorch attention if xformers fails.

2. **GPU monitoring**: nvidia-smi may show 100% utilization on RTX 5090 even when idle. This appears to be a driver/monitoring issue and doesn't affect functionality.

3. **mistral-common v1.8.1 bugs**: The validator in v1.8.1 has a bug where it tries to access `request.tools` instead of `request.available_tools`. We've added a workaround for this.

### 6. Testing

Create a simple test file to verify everything works:

```jsonl
{"messages": [{"role": "user", "content": "What is Python?"}, {"role": "assistant", "content": "Python is a high-level programming language."}]}
```

Then run:
```bash
python train.py example/example_config.yaml
```

## Contributing

If you encounter issues or have improvements, please submit a pull request or open an issue.

## Changes Made

### Files Modified:
- `requirements.txt` - Updated for RTX 5090 support
- `finetune/data/tokenize.py` - Fixed mistral-common v1.8.1 compatibility
- `finetune/checkpointing.py` - Updated imports
- `train.py` - Updated tokenizer initialization
- Various test files - Updated for new API

### New Files:
- `setup_rtx5090.sh` - Setup script for RTX 5090
- `fix_mistral_common_v1_8_1.py` - Compatibility fix script
- `README_RTX5090.md` - This documentation