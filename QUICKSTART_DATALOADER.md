# Data Loading and Rotation Augmentation - Quick Start

This guide provides a quick introduction to the newly integrated data loading and rotation augmentation features.

## What's New

### 1. Automatic Data Loading

No more manual data loading! The `ModelTrainer` now automatically loads data from .mat files:

```python
from config import Config
from model_modern import ModelTrainer

config = Config()
trainer = ModelTrainer(config, output_dir='./models')

# Automatic data loading from .mat file
history = trainer.train(
    data_path='./Dataset/Australia/Rotations/Australia_strip.mat',
    train_ratio=0.1,
    val_ratio=0.5
)
```

### 2. Rotation Augmentation

Enable rotation augmentation through configuration:

```python
config = Config()
config.augmentation.enable_rotation = True
config.augmentation.rotation_probability = 0.5  # 50% chance
config.augmentation.rotation_angles = [0, 90, 180, 270]

trainer = ModelTrainer(config, output_dir='./models')
history = trainer.train(data_path='dataset.mat')
```

### 3. Command-Line Interface

Use the enhanced CLI for training:

```bash
# Basic training
python cli.py train --data dataset.mat --output ./models

# With rotation augmentation
python cli.py train \
    --data dataset.mat \
    --output ./models \
    --enable-rotation \
    --rotation-prob 0.5

# Full configuration
python cli.py train \
    --data dataset.mat \
    --output ./models \
    --architecture UNet \
    --epochs 50 \
    --batch-size 32 \
    --train-ratio 0.2 \
    --enable-rotation \
    --enable-flipping \
    --tensorboard
```

## Configuration File Example

Create a configuration file `config.json`:

```json
{
    "model": {
        "architecture": "RotateNet",
        "window_size": 45,
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001
    },
    "augmentation": {
        "enable_rotation": true,
        "rotation_probability": 0.5,
        "rotation_angles": [0, 90, 180, 270],
        "enable_flipping": true
    }
}
```

Then train with:

```bash
python cli.py train --config config.json --data dataset.mat --output ./models
```

## Python API Examples

### Example 1: Basic Training

```python
from config import Config
from model_modern import ModelTrainer

config = Config()
trainer = ModelTrainer(config, './models')
history = trainer.train(data_path='dataset.mat', train_ratio=0.1)
```

### Example 2: With DataGenerator

```python
from config import Config
from data_generator import DataGenerator
from model_modern import ModelTrainer

config = Config()
data_gen = DataGenerator(config, 'dataset.mat')

# Get dataset info
info = data_gen.get_dataset_info()
print(f"Dataset shape: {info['shape']}")
print(f"Fault pixels: {info['fault_pixels']}")

# Train with data generator
trainer = ModelTrainer(config, './models', data_generator=data_gen)
history = trainer.train(train_ratio=0.1)
```

### Example 3: Full Augmentation

```python
from config import Config
from model_modern import ModelTrainer

config = Config()
config.model.architecture = 'UNet'
config.model.epochs = 10

# Enable augmentations
config.augmentation.enable_rotation = True
config.augmentation.rotation_probability = 0.5
config.augmentation.enable_flipping = True

trainer = ModelTrainer(config, './models')
history = trainer.train(
    data_path='dataset.mat',
    train_ratio=0.2,
    val_ratio=0.5,
    use_tensorboard=True
)
```

## DataGenerator API

The `DataGenerator` class provides tf.data.Dataset integration:

```python
from data_generator import DataGenerator
from config import Config

config = Config()
data_gen = DataGenerator(config, 'dataset.mat')

# Create training dataset
train_ds = data_gen.create_training_dataset(
    ratio=0.1,        # Use 10% of data
    choosy=False,     # Use all mask locations
    shuffle=True,     # Shuffle data
    cache=False       # Don't cache (for large datasets)
)

# Create validation dataset
val_ds = data_gen.create_validation_dataset(
    ratio=0.5,        # Use 50% of validation data
    cache=True        # Cache (validation sets are usually smaller)
)

# Get dataset information
info = data_gen.get_dataset_info()
```

## Augmentation Options

### Rotation Augmentation

```python
config.augmentation.enable_rotation = True
config.augmentation.rotation_probability = 0.5
config.augmentation.rotation_angles = [0, 90, 180, 270]

# Or use FILTER.py rotation matrices
config.augmentation.rotation_filter_path = "./Filters/Default.mat"
```

### Flipping Augmentation

```python
config.augmentation.enable_flipping = True
config.augmentation.flip_probability = 0.5
```

## Backward Compatibility

All existing code continues to work:

```python
# Old way still works
from DATASET import DATASET
from model_modern import build_model

ds = DATASET('data.mat')
X, Y, _ = ds.generateDS(ds.OUTPUT, ds.trainMask)
model = build_model(config)
model.fit(X, Y, epochs=10)

# New way (recommended)
from model_modern import ModelTrainer

trainer = ModelTrainer(config, './models')
trainer.train(data_path='data.mat')
```

## Performance Tips

1. **For small datasets**: Enable caching
   ```python
   train_ds = data_gen.create_training_dataset(cache=True)
   ```

2. **For large datasets**: Use smaller ratios and prefetching
   ```python
   train_ds = data_gen.create_training_dataset(
       ratio=0.05,  # Use less data
       cache=False  # Don't cache
   )
   ```

3. **For faster training**: Disable augmentation during testing
   ```python
   config.augmentation.enable_rotation = False
   ```

4. **For better results**: Enable multiple augmentations
   ```python
   config.augmentation.enable_rotation = True
   config.augmentation.enable_flipping = True
   ```

## Troubleshooting

### Out of Memory

Reduce batch size or train ratio:
```python
config.model.batch_size = 16  # Reduce from 32
history = trainer.train(data_path='dataset.mat', train_ratio=0.05)
```

### Slow Training

Enable prefetching (already default in DataGenerator):
```python
# Prefetching is enabled by default
train_ds = data_gen.create_training_dataset()
```

### No Validation Data

The validation dataset is optional:
```python
# Training without validation
trainer = ModelTrainer(config, './models')
# Just provide training data, validation will be None if not available
```

## Complete Working Example

```python
#!/usr/bin/env python3
"""Complete training example."""

from config import Config
from model_modern import ModelTrainer

def main():
    # Configure
    config = Config()
    config.model.architecture = 'RotateNet'
    config.model.epochs = 10
    config.model.batch_size = 32
    
    # Enable augmentation
    config.augmentation.enable_rotation = True
    config.augmentation.rotation_probability = 0.5
    config.augmentation.enable_flipping = True
    
    # Create trainer
    trainer = ModelTrainer(config, output_dir='./outputs/my_model')
    
    # Train
    history = trainer.train(
        data_path='./Dataset/Australia/Rotations/Australia_strip.mat',
        train_ratio=0.1,
        val_ratio=0.5,
        use_tensorboard=True
    )
    
    print("Training complete!")
    print(f"Final accuracy: {history.history['accuracy'][-1]:.4f}")

if __name__ == '__main__':
    main()
```

## More Information

- Full specification: `DATA_LOADING_ROTATION_IMPROVEMENTS.md`
- Pipeline coverage: `PIPELINE_COVERAGE.md`
- More examples: `examples/train_with_data_generator.py`
