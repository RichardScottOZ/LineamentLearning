# Examples Directory

This directory contains example notebooks and scripts demonstrating how to use LineamentLearning.

## Quick Links

### Python Scripts
- `train_example.py` - Simple training example
- `predict_example.py` - Prediction example
- `config_example.py` - Configuration examples
- `postprocessing_example.py` - Post-processing and clustering demonstration

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
```

## Need Help?

- Check the [QUICKSTART.md](../QUICKSTART.md) guide
- Read the full [README.md](../README.md)
- See [POSTPROCESSING_GUIDE.md](../POSTPROCESSING_GUIDE.md) for clustering details
- Open an issue on GitHub
