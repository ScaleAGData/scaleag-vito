## ScaleAGData Vito

### Installation

To install the environment with all dependencies needed to run the `few_shot_learning.ipynb` notebook, follow these steps:

#### 0. Clone repository and switch to most updated branch
```bash
# Clone the repository
git clone https://github.com/your-username/scaleag-vito.git

# switch to prometheo-integration branch (this is the most updated branch, soon to be merged to main)
git checkout prometheo-integration
```
#### 1. Create a new conda environment

```bash
# Create a new conda environment with Python 3.10
conda create -n scaleag-env python=3.10
conda activate scaleag-env
```

#### 2. Install the package with dependencies

```bash
# Navigate to the project directory
cd /path/to/scaleag-vito

# Install the package in editable mode with all optional dependencies
pip install -e ".[dev,notebooks,train]"
```

This will install:
- Core dependencies for the ScaleAG package
- Development tools (`dev` dependencies)
- Jupyter notebook dependencies including `ipyleaflet` and interactive widgets (`notebooks` dependencies)
- Training dependencies for model fine-tuning (`train` dependencies)


### Requirements

- Python >= 3.8
- Access to [Copernicus Data Space Ecosystem (CDSE)](https://dataspace.copernicus.eu/) account (free registration)
- Sufficient disk space for EO data extractions

### Dependencies

The installation includes:
- **Presto**: Foundation model for remote sensing data
- **OpenEO**: For Earth observation data extraction
- **PyTorch**: Deep learning framework
- **Jupyter ecosystem**: Interactive notebooks with mapping widgets
- **Geospatial libraries**: For handling geographic data

For a complete list of dependencies, see `pyproject.toml`.
