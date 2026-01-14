# Examples Directory

This directory contains example notebooks and scripts demonstrating how to use LineamentLearning.

## Quick Links

### Python Scripts
- `train_example.py` - Simple training example
- `predict_example.py` - Prediction example
- `config_example.py` - Configuration examples
- `postprocessing_example.py` - Post-processing and clustering demonstration
- `train_with_data_generator.py` - Data loading and augmentation examples
- `mat_conversion_examples.py` - **NEW**: MATLAB .mat to PyData conversion examples

### Data Format Conversion
- `mat_conversion_examples.py` - Complete examples for converting .mat files to NumPy, HDF5, and Zarr formats
- See [MAT_TO_PYDATA_GUIDE.md](../MAT_TO_PYDATA_GUIDE.md) for detailed documentation

### Future Additions
- Jupyter notebooks for interactive exploration
- Visualization examples
- Custom architecture examples
- Advanced training techniques

## Running Examples

### Prerequisites

```bash
# Install package
cd ..
pip install -e ".[full]"
```

### Run Scripts

```bash
# Configuration examples
python config_example.py

# Training example
python train_example.py --help

# Prediction example (requires trained model)
python predict_example.py --model ../models/best_model.h5

# Post-processing example (demonstrates clustering)
python postprocessing_example.py

# Data conversion examples
python mat_conversion_examples.py
```

## Data Conversion Quick Start

```bash
# Inspect a .mat file
python -m mat_converter --inspect ../Dataset/sample.mat

# Convert to HDF5 (recommended)
python -m mat_converter ../Dataset/sample.mat ../Dataset/sample.h5

# Convert to NumPy
python -m mat_converter --format numpy ../Dataset/sample.mat ../Dataset/sample.npz

# See examples for more details
python mat_conversion_examples.py
```

## Need Help?

- Check the [QUICKSTART.md](../QUICKSTART.md) guide
- Read the full [README.md](../README.md)
- See [MAT_TO_PYDATA_GUIDE.md](../MAT_TO_PYDATA_GUIDE.md) for data conversion
- See [POSTPROCESSING_GUIDE.md](../POSTPROCESSING_GUIDE.md) for clustering details
- Open an issue on GitHub
