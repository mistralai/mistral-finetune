# Contributing to mistral-finetune

Thank you for your interest in contributing to mistral-finetune! This guide will help you get started.

## Prerequisites

- **Python** >= 3.9
- **NVIDIA GPU** with CUDA 12.1+ support (required for torch 2.2 and xformers)
- **Git**

## Development Setup

1. **Fork and clone** the repository:

```bash
git clone https://github.com/<your-username>/mistral-finetune.git
cd mistral-finetune
```

2. **Create a virtual environment** (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
pip install -r requirements.dev.txt
```

4. **Set up pre-commit hooks** (optional but recommended):

```bash
pre-commit install
```

## Code Style

This project uses the following tools, all configured in `pyproject.toml`:

| Tool | Purpose | Config |
|------|---------|--------|
| **ruff** | Linting | `pyproject.toml` [tool.ruff] |
| **black** | Formatting (line-length=88) | `pyproject.toml` [tool.black] |
| **isort** | Import sorting | `pyproject.toml` [tool.isort] |
| **mypy** | Type checking (Python 3.9) | `pyproject.toml` [tool.mypy] |

Run all checks before submitting a PR:

```bash
ruff check .
black --check .
mypy --ignore-missing-imports .
```

Or fix formatting automatically:

```bash
black .
isort .
```

## Testing

Tests are located in `tests/` and require a GPU with a dummy model:

```bash
# Set the path to a compatible model checkpoint
export DUMMY_MODEL=/path/to/dummy/model

# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_data.py -v
```

> **Note:** Tests require an NVIDIA GPU and a model checkpoint. They cannot be run in CPU-only environments.

## Submitting a Pull Request

1. **Create a branch** from `main`:
   ```bash
   git checkout -b fix/your-fix-description main
   ```

2. **Make your changes** and ensure all lint checks pass.

3. **Commit** with a clear message:
   - `fix:` for bug fixes
   - `feat:` for new features
   - `docs:` for documentation
   - `ci:` for CI/CD changes
   - `test:` for test additions

4. **Push** and open a PR against `main`:
   ```bash
   git push origin fix/your-fix-description
   ```

5. **Describe your changes** in the PR description, including:
   - What the change does
   - Why it's needed
   - Related issues (use `Fixes #123` or `Relates to #123`)

## Project Structure

```
mistral-finetune/
├── train.py              # Main training entry point
├── finetune/             # Core fine-tuning package
│   ├── args.py           # Training arguments
│   ├── data/             # Data loading and tokenization
│   ├── distributed.py    # torch.distributed utilities
│   ├── eval.py           # Evaluation
│   ├── loss.py           # Loss computation
│   └── wrapped_model.py  # Model loading with LoRA
├── model/                # Model architecture
│   ├── transformer.py    # Transformer + LoRA layers
│   ├── lora.py           # LoRA implementation
│   └── moe.py            # Mixture of Experts
├── utils/                # Utility scripts
│   ├── validate_data.py  # Data format validation
│   └── reformat_data.py  # Data reformatting
└── tests/                # Unit tests (GPU required)
```

## License

By contributing, you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE).
