# TorchGeo: PyTorch Geospatial Deep Learning Library

TorchGeo is a PyTorch domain library providing datasets, samplers, transforms, and pre-trained models for geospatial data and remote sensing applications. The library enables machine learning experts to work with geospatial data and remote sensing experts to explore machine learning solutions.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Bootstrap, Build, and Test the Repository
**CRITICAL: NEVER CANCEL long-running commands. All timeouts below are minimum required times.**

1. **Install TorchGeo in Development Mode** (5 minutes, NEVER CANCEL):
   ```bash
   pip install -e .
   ```
   - Takes ~2.5 minutes due to PyTorch and geospatial dependencies (GDAL, rasterio, etc.)
   - NEVER CANCEL: Set timeout to 10+ minutes
   - Network issues may cause timeouts - retry if needed

2. **Install Test Dependencies** (30 seconds):
   ```bash
   pip install -r requirements/tests.txt
   ```

3. **Install Style/Linting Dependencies** (30 seconds):
   ```bash
   pip install -r requirements/style.txt
   ```

4. **Install Documentation Dependencies** (2 minutes, may fail):
   ```bash
   pip install -r requirements/docs.txt
   cd docs
   pip install -r requirements.txt  # May fail due to pytorch_sphinx_theme network issues
   ```

### Testing
**CRITICAL: Set appropriate timeouts for test commands. NEVER CANCEL.**

- **Run All Tests** (30+ minutes, NEVER CANCEL):
  ```bash
  pytest --cov=torchgeo --cov-report=xml
  ```
  - NEVER CANCEL: Set timeout to 60+ minutes
  - Full test suite is comprehensive and takes time

- **Run Specific Test Files** (1-5 minutes per file):
  ```bash
  pytest tests/test_main.py -v
  pytest tests/trainers/test_utils.py -v
  ```

- **Run Single Tests** (10-30 seconds):
  ```bash
  pytest tests/test_main.py::test_help -v
  ```

### CLI Interface
**CLI startup takes ~10 seconds due to heavy imports - this is normal.**

- **Test CLI Help** (10 seconds):
  ```bash
  python -m torchgeo --help
  ```

- **Test CLI with Config** (30-60 seconds):
  ```bash
  python -m torchgeo fit --config tests/conf/eurosat.yaml --trainer.fast_dev_run=true --trainer.accelerator=cpu
  ```

- **Alternative CLI Command**:
  ```bash
  python3 -m torchgeo
  ```

### Linting and Code Quality
**Always run linters before committing changes.**

- **Ruff Code Formatting** (<5 seconds):
  ```bash
  ruff format
  ruff check
  ```

- **MyPy Type Checking** (30-120 seconds):
  ```bash
  mypy .
  ```
  - NEVER CANCEL: Set timeout to 5+ minutes for full codebase
  - Can run on individual modules: `mypy torchgeo/datasets`

- **Prettier Formatting** (<5 seconds):
  ```bash
  npx prettier --write .
  ```

- **Pre-commit Hooks** (2-5 minutes first run):
  ```bash
  pip install pre-commit
  pre-commit install
  pre-commit run --all-files  # First run takes longer, subsequent runs are fast
  ```
  - CAUTION: May fail due to network timeouts when installing hook environments
  - If pre-commit fails, run linting tools individually: `ruff format`, `ruff check`, `mypy .`

### Documentation Building
**Documentation builds may fail due to network dependencies.**

```bash
cd docs
make clean
make html  # May fail if pytorch_sphinx_theme cannot be installed
```
- Documentation requires additional dependencies that may have network issues
- Focus on testing functionality rather than documentation builds

### I/O Benchmarking
**For testing geospatial data loading performance:**

```bash
python -m torchgeo fit --config tests/conf/io_raw.yaml
python -m torchgeo fit --config tests/conf/io_preprocessed.yaml
```
- These download datasets and may take significant time
- Used to profile RandomGeoSampler and GridGeoSampler performance

## Validation Scenarios

### CRITICAL Validation Steps
**Always test these scenarios after making changes:**

1. **CLI Functionality Test**:
   ```bash
   python -m torchgeo --help
   python -m torchgeo fit --config tests/conf/eurosat.yaml --trainer.fast_dev_run=true --trainer.accelerator=cpu
   ```

2. **Core Functionality Test**:
   ```bash
   pytest tests/test_main.py -v
   ```

3. **Linting Validation**:
   ```bash
   ruff format --check .
   ruff check .
   mypy torchgeo/main.py  # Test a single file for speed
   ```

4. **Import Test**:
   ```python
   python -c "import torchgeo; print(torchgeo.__version__)"
   ```

## Installation Options

### PyPI Installation
```bash
pip install torchgeo
pip install torchgeo[datasets,models]  # With optional dependencies
pip install torchgeo[all]  # All optional dependencies
```

### Conda Installation
```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install torchgeo
```

### Development Installation
```bash
git clone https://github.com/torchgeo/torchgeo.git
cd torchgeo
pip install -e .
```

## Key Project Structure

### Repository Root
```
.
├── README.md              # Main documentation
├── pyproject.toml          # Dependencies and configuration
├── torchgeo/              # Main source code
├── tests/                 # Test suite
├── docs/                  # Documentation source
├── requirements/          # Dependency files
├── .pre-commit-config.yaml # Pre-commit hooks config
└── .github/workflows/     # CI/CD pipelines
```

### Key Source Modules
```
torchgeo/
├── __init__.py
├── main.py               # CLI entry point
├── datasets/             # Geospatial and benchmark datasets
├── datamodules/          # Lightning datamodules
├── trainers/             # Lightning trainers/tasks
├── models/               # Pre-trained models
├── samplers/             # Geospatial samplers
└── transforms/           # Data transformations
```

### Key Test Directories
```
tests/
├── test_main.py          # CLI tests
├── conf/                 # Configuration files for testing
├── data/                 # Fake test data
├── datasets/             # Dataset tests
├── trainers/             # Trainer tests
└── ...
```

## Common Issues and Solutions

### Build Issues
- **Slow pip install**: Normal due to PyTorch and geospatial dependencies
- **Network timeouts**: Retry the installation command
- **Documentation build fails**: Skip documentation builds, focus on code functionality
- **Pre-commit failures**: Use individual linting tools if pre-commit fails due to network issues

### Test Issues
- **Slow test startup**: Normal due to PyTorch imports (~10 seconds)
- **Tests requiring data**: Use tests/conf/ configurations with fake data
- **GPU tests**: Use `--trainer.accelerator=cpu` for CPU-only testing

### CLI Issues
- **Slow CLI startup**: Normal due to Lightning and PyTorch imports
- **Missing datasets**: Use configurations in tests/conf/ for testing

## Dependencies Summary

**Core Dependencies**: PyTorch, Lightning, GDAL, rasterio, geopandas, kornia, timm, torchvision
**Test Dependencies**: pytest, pytest-cov, nbmake
**Style Dependencies**: ruff, mypy
**Docs Dependencies**: sphinx, nbsphinx (may fail to install)

## Timeout Recommendations

**CRITICAL: Use these minimum timeout values in commands:**

- `pip install -e .`: 10+ minutes
- `pytest` (full suite): 60+ minutes  
- `mypy .`: 5+ minutes
- `python -m torchgeo`: 30+ seconds
- `pre-commit run --all-files`: 10+ minutes (first run)

**NEVER CANCEL any command marked with timeouts above.**