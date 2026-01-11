# Quick Start Guide

This guide will help you get started with LineamentLearning in 5 minutes.

## Installation

```bash
# Clone repository
git clone https://github.com/RichardScottOZ/LineamentLearning.git
cd LineamentLearning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Your First Training

### 1. Prepare Your Data

Your data should be in MATLAB `.mat` format with the following structure:
- `I1, I2, ..., I8`: Input layers (8 geophysical datasets)
- `output`: Ground truth lineaments
- `mask`: Valid data region
- `train_mask`: Training region
- `test_mask`: Testing region
- `DEGREES`: Lineament orientations

### 2. Create Configuration

Create `my_config.json`:

```json
{
    "model": {
        "architecture": "RotateNet",
        "window_size": 45,
        "batch_size": 32,
        "epochs": 50,
        "learning_rate": 0.001
    },
    "data": {
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15
    }
}
```

### 3. Train Model

```bash
lineament-train \
    --config my_config.json \
    --data ./Dataset/Australia/Rotations/Australia_strip.mat \
    --output ./my_first_model \
    --tensorboard
```

Monitor training with TensorBoard:
```bash
tensorboard --logdir ./my_first_model/logs
```

### 4. Run Prediction

```bash
lineament-predict \
    --model ./my_first_model/best_model.h5 \
    --data ./Dataset/test_data.mat \
    --output ./results \
    --visualize
```

## Using Python API

```python
from config import Config
from model_modern import build_model

# Create and configure model
config = Config()
config.model.architecture = 'UNet'
model = build_model(config)

# View model architecture
model.summary()
```

## Trying Different Architectures

### U-Net (Better for Spatial Context)

```bash
lineament-train \
    --architecture UNet \
    --window-size 64 \
    --data ./data/train.mat \
    --output ./models/unet
```

### ResNet (Deeper Network)

```bash
lineament-train \
    --architecture ResNet \
    --window-size 64 \
    --epochs 100 \
    --data ./data/train.mat \
    --output ./models/resnet
```

## Common Issues

### Issue: Out of Memory
**Solution**: Reduce batch size
```bash
lineament-train --batch-size 16 ...
```

### Issue: Slow Training
**Solutions**:
1. Enable GPU: `--gpu 0`
2. Use mixed precision: Add to config: `"use_mixed_precision": true`
3. Reduce window size: `--window-size 32`

### Issue: Model Not Learning
**Solutions**:
1. Check data quality
2. Adjust learning rate: `--learning-rate 0.0001`
3. Increase epochs: `--epochs 200`
4. Try different architecture: `--architecture UNet`

## Next Steps

1. **Read Full Documentation**: See README.md
2. **Explore Examples**: Check `examples/` directory
3. **Customize Model**: Edit `model_modern.py`
4. **Optimize Hyperparameters**: Experiment with config
5. **Visualize Results**: Use TensorBoard and visualization tools

## Tips for Best Results

1. **Data Quality**: Ensure clean, properly labeled data
2. **Data Augmentation**: Enable augmentation for small datasets
3. **Early Stopping**: Use early stopping to prevent overfitting
4. **Model Selection**: Try multiple architectures
5. **Hyperparameter Tuning**: Experiment with learning rates and batch sizes
6. **Ensemble Methods**: Combine multiple models for better results

## Getting Help

- **Documentation**: README.md, CHANGELOG.md
- **Issues**: https://github.com/RichardScottOZ/LineamentLearning/issues
- **Original Thesis**: http://hdl.handle.net/2429/68438
