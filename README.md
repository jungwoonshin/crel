# crel

**Centered Residual Energy Loss for Structured Prediction**

## Installation

```bash
pip install -e .
```

With dev dependencies (pytest, etc.):

```bash
pip install -e ".[dev]"
```

With NLP extras (transformers):

```bash
pip install -e ".[nlp]"
```

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- See [pyproject.toml](pyproject.toml) for full dependencies.

## Project structure

- `crel/` — core library (energy models, losses, training, data)
- `configs/` — configuration files
- `scripts/` — training and analysis scripts
- `tests/` — tests

## License

See repository for license information.
