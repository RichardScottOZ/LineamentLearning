# Data Loading and Rotation Improvements Specification

This document provides detailed specifications for improving data loading and rotation augmentation integration in LineamentLearning, as referenced in PIPELINE_COVERAGE.md.

## Overview

The modern LineamentLearning pipeline has been enhanced with new model architectures, CLI tools, and configuration management. However, two critical components need better integration:

1. **Data Loading** - Integration of DATASET.py with modern ModelTrainer
2. **Rotation Augmentation** - Integration of FILTER.py with modern training pipeline

## Current State

### Data Loading (DATASET.py)
**Status**: ⚠️ Available but not fully integrated

**What Exists**:
- ✅ Original DATASET class can load .mat files
- ✅ Bridge adapter (`DatasetAdapter`) provides basic integration
- ✅ Can generate training/validation data in original format

**What's Missing**:
- ❌ No tf.data.Dataset pipeline for efficient data loading
- ❌ No built-in data augmentation during training
- ❌ No batch prefetching and parallel loading
- ❌ No integration with ModelTrainer's fit() method
- ❌ No streaming for large datasets
- ❌ CLI commands assume data integration exists but it doesn't work out-of-the-box

### Rotation Augmentation (FILTER.py)
**Status**: ⚠️ Available but not integrated

**What Exists**:
- ✅ Original FILTER class can load rotation matrices from .mat files
- ✅ Bridge adapter (`FilterAdapter`) provides access to rotation filters

**What's Missing**:
- ❌ No integration with tf.keras data augmentation layers
- ❌ No automatic rotation during training
- ❌ No configuration option to enable/disable rotation augmentation
- ❌ Cannot use rotation augmentation with modern ModelTrainer
- ❌ No random rotation angle generation using modern TensorFlow operations

## Detailed Improvement Specifications

### 1. Data Loading Improvements

#### 1.1 Create TensorFlow Data Pipeline

**Goal**: Create a `DataGenerator` class that wraps DATASET.py and provides tf.data.Dataset compatibility.

**Implementation Requirements**:

```python
class DataGenerator:
    """Modern data generator wrapping original DATASET class."""
    
    def __init__(self, config: Config, dataset_path: str):
        """Initialize with configuration and dataset path."""
        pass
    
    def create_training_dataset(self) -> tf.data.Dataset:
        """Create tf.data.Dataset for training with prefetching."""
        # - Load data using DATASET.generateDS()
        # - Convert to tf.data.Dataset
        # - Add batch processing
        # - Add prefetching
        # - Add shuffling
        pass
    
    def create_validation_dataset(self) -> tf.data.Dataset:
        """Create tf.data.Dataset for validation."""
        pass
```

**Benefits**:
- Efficient batch loading
- GPU/CPU parallelism
- Memory efficiency for large datasets
- Compatible with model.fit()

#### 1.2 Integrate with ModelTrainer

**Goal**: Modify `model_modern.py` ModelTrainer to accept DataGenerator.

**Changes Needed**:

```python
class ModelTrainer:
    def __init__(self, config: Config, data_generator: Optional[DataGenerator] = None):
        """Accept optional DataGenerator."""
        self.data_generator = data_generator
    
    def train(self):
        """Use data_generator if provided."""
        if self.data_generator:
            train_ds = self.data_generator.create_training_dataset()
            val_ds = self.data_generator.create_validation_dataset()
            self.model.fit(train_ds, validation_data=val_ds, ...)
```

**Benefits**:
- End-to-end training without manual data loading
- Works with existing CLI commands
- Backward compatible with manual data loading

#### 1.3 Update CLI Integration

**Goal**: Make `lineament-train` command work with .mat files directly.

**Changes Needed in cli.py**:

```python
@click.command()
@click.option('--data', required=True, help='Path to .mat dataset file')
def train(data, ...):
    """Train a lineament detection model."""
    config = Config.load(config_path)
    
    # Create data generator from .mat file
    data_gen = DataGenerator(config, data)
    
    # Create trainer with data generator
    trainer = ModelTrainer(config, data_generator=data_gen)
    
    # Train model
    trainer.train()
```

**Benefits**:
- Users can train directly: `lineament-train --data dataset.mat`
- No manual data loading code required
- Professional user experience

### 2. Rotation Augmentation Improvements

#### 2.1 Add TensorFlow Augmentation Layer

**Goal**: Create modern rotation augmentation using tf.keras layers.

**Implementation Requirements**:

```python
class RotationAugmentation(tf.keras.layers.Layer):
    """Custom layer for rotation augmentation compatible with FILTER.py."""
    
    def __init__(self, filter_path: Optional[str] = None, **kwargs):
        """Initialize with optional FILTER.py matrices or use tf.image.rot90."""
        super().__init__(**kwargs)
        if filter_path:
            self.filter = FILTER(filter_path)
            self.use_original_filters = True
        else:
            self.use_original_filters = False
    
    def call(self, inputs, training=None):
        """Apply random rotation during training."""
        if not training:
            return inputs
        
        if self.use_original_filters:
            # Use FILTER.py rotation matrices
            return self._apply_original_rotation(inputs)
        else:
            # Use tf.image rotation
            return self._apply_tf_rotation(inputs)
```

**Benefits**:
- Works with both original FILTER.py and modern TensorFlow
- Integrates seamlessly with model architecture
- Can be enabled/disabled via configuration

#### 2.2 Add Configuration Options

**Goal**: Add rotation augmentation settings to config.py.

**Changes Needed**:

```python
@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""
    
    # Rotation
    enable_rotation: bool = False
    rotation_filter_path: Optional[str] = None  # Path to FILTER.py .mat file
    rotation_probability: float = 0.5  # Probability of applying rotation
    
    # Other augmentations
    enable_flipping: bool = False
    enable_brightness: bool = False
    brightness_delta: float = 0.1

@dataclass
class Config:
    """Complete configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)  # NEW
```

**Benefits**:
- User can enable/disable rotation via config file
- Support for both FILTER.py and TensorFlow rotation
- Extensible for future augmentation types

#### 2.3 Integrate with Model Building

**Goal**: Apply rotation augmentation when building models.

**Changes in model_modern.py**:

```python
def build_model(config: Config) -> keras.Model:
    """Build model with optional augmentation."""
    
    inputs = layers.Input(
        shape=(config.model.window_size, config.model.window_size, config.model.layers)
    )
    
    x = inputs
    
    # Add augmentation layers if enabled
    if config.augmentation.enable_rotation:
        x = RotationAugmentation(
            filter_path=config.augmentation.rotation_filter_path
        )(x)
    
    if config.augmentation.enable_flipping:
        x = layers.RandomFlip("horizontal_and_vertical")(x)
    
    # Continue with model architecture
    if config.model.architecture == 'RotateNet':
        model_outputs = create_rotatenet_core(x, config.model)
    ...
```

**Benefits**:
- Augmentation applied automatically during training
- Configured via JSON/YAML files
- No code changes needed by users

### 3. Integration Workflow Examples

#### 3.1 Training with Data Loading + Rotation

**Configuration File (config.json)**:
```json
{
  "model": {
    "architecture": "RotateNet",
    "window_size": 45,
    "epochs": 50
  },
  "augmentation": {
    "enable_rotation": true,
    "rotation_filter_path": "./Dataset/filters/Default.mat",
    "rotation_probability": 0.5,
    "enable_flipping": true
  }
}
```

**Command Line**:
```bash
lineament-train \
    --config config.json \
    --data ./Dataset/Australia/Rotations/Australia_strip.mat \
    --output ./models/my_model
```

**Python API**:
```python
from config import Config
from model_modern import build_model, ModelTrainer, DataGenerator

# Load configuration
config = Config.from_json('config.json')

# Create data generator
data_gen = DataGenerator(config, './Dataset/Australia/Rotations/Australia_strip.mat')

# Build model with augmentation
model = build_model(config)

# Train with integrated pipeline
trainer = ModelTrainer(config, data_generator=data_gen)
trainer.train()
```

#### 3.2 Training without Rotation (Modern TensorFlow only)

```json
{
  "model": {
    "architecture": "UNet",
    "window_size": 64
  },
  "augmentation": {
    "enable_rotation": false,
    "enable_flipping": true,
    "enable_brightness": true
  }
}
```

**Benefits**:
- Can train without FILTER.py dependency
- Uses modern TensorFlow augmentation
- Faster and simpler for new users

## Implementation Roadmap

### Phase 1: Data Loading Integration (Priority: HIGH)
**Estimated Time**: 1-2 days

Tasks:
1. Create `DataGenerator` class in new file `data_generator.py`
2. Add unit tests for DataGenerator
3. Modify `ModelTrainer.__init__()` to accept DataGenerator
4. Update `cli.py train()` command to use DataGenerator
5. Add example in `examples/train_with_data_generator.py`
6. Update documentation

**Success Criteria**:
- ✅ Can run: `lineament-train --data dataset.mat --output ./models`
- ✅ Training works end-to-end without manual data loading
- ✅ Backward compatible with existing code

### Phase 2: Rotation Augmentation Integration (Priority: MEDIUM)
**Estimated Time**: 1 day

Tasks:
1. Create `RotationAugmentation` layer in `model_modern.py`
2. Add `AugmentationConfig` to `config.py`
3. Integrate augmentation in `build_model()`
4. Add unit tests for rotation augmentation
5. Add example in `examples/train_with_rotation.py`
6. Update documentation

**Success Criteria**:
- ✅ Can enable rotation via config file
- ✅ Works with both FILTER.py and TensorFlow rotation
- ✅ Can disable rotation for faster training

### Phase 3: Additional Augmentations (Priority: LOW)
**Estimated Time**: 0.5 days

Tasks:
1. Add flipping, brightness, contrast augmentation
2. Add noise augmentation
3. Document all augmentation options
4. Add visualization of augmented samples

**Success Criteria**:
- ✅ Full suite of augmentation options available
- ✅ Well documented with examples
- ✅ Can visualize augmented data

## Testing Strategy

### Unit Tests
```python
# test_data_generator.py
def test_data_generator_creates_dataset():
    """Test DataGenerator creates valid tf.data.Dataset."""
    
def test_data_generator_batch_shape():
    """Test batch shape matches configuration."""

# test_augmentation.py
def test_rotation_augmentation_shape():
    """Test rotation preserves tensor shape."""
    
def test_rotation_augmentation_training_only():
    """Test rotation only applied during training."""
```

### Integration Tests
```python
# test_training_integration.py
def test_end_to_end_training():
    """Test complete training pipeline with data loading."""
    
def test_training_with_rotation():
    """Test training with rotation augmentation enabled."""
```

### Manual Testing
1. Train small model on sample data (5 epochs)
2. Verify rotation augmentation visually
3. Test CLI commands work as documented
4. Verify backward compatibility

## Documentation Updates

### Files to Update:
1. **PIPELINE_COVERAGE.md**: 
   - Change status from ⚠️ to ✅ after implementation
   - Update integration examples
   - Remove "What's Missing" sections

2. **README.md**:
   - Update quick start examples
   - Show data loading integration
   - Show rotation augmentation example

3. **QUICKSTART.md**:
   - Update training command examples
   - Add augmentation configuration example

4. **New File: DATA_LOADING_GUIDE.md**:
   - Complete guide to data loading
   - Examples with different dataset types
   - Troubleshooting section

5. **New File: AUGMENTATION_GUIDE.md**:
   - Complete guide to data augmentation
   - Configuration options
   - Visual examples

## Backward Compatibility

### Ensure These Still Work:
```python
# Original way (must still work)
from DATASET import DATASET
ds = DATASET('data.mat')
X, Y, _ = ds.generateDS(ds.OUTPUT, ds.trainMask)

# Bridge way (must still work)
from bridge import DatasetAdapter
adapter = DatasetAdapter(config, 'data.mat')
X, Y, _ = adapter.generate_training_data()

# New way (after implementation)
from data_generator import DataGenerator
gen = DataGenerator(config, 'data.mat')
train_ds = gen.create_training_dataset()
```

## Performance Considerations

### Data Loading:
- Use `tf.data.Dataset.prefetch()` for pipelining
- Use `num_parallel_calls` for parallel data loading
- Cache small datasets in memory
- Use generators for datasets that don't fit in memory

### Rotation Augmentation:
- Apply rotation on GPU when possible
- Use compiled TensorFlow operations
- Batch augmentation operations
- Consider pre-generating rotated samples for very large datasets

## Common Issues and Solutions

### Issue 1: Out of Memory
**Solution**: Use DataGenerator with smaller batch sizes and enable prefetching but not caching.

### Issue 2: Slow Data Loading
**Solution**: Enable parallel loading and prefetching in DataGenerator configuration.

### Issue 3: Rotation Changes Data Distribution
**Solution**: Adjust rotation_probability or use validation set without augmentation.

### Issue 4: FILTER.py Not Found
**Solution**: Make rotation_filter_path optional, fall back to TensorFlow rotation.

## Summary

This specification provides a complete roadmap for integrating data loading and rotation augmentation with the modern LineamentLearning pipeline. The improvements will:

1. **Enable end-to-end training** without manual data loading code
2. **Provide flexible augmentation** with easy configuration
3. **Maintain backward compatibility** with existing code
4. **Improve user experience** with CLI integration
5. **Enhance performance** with TensorFlow data pipelines

**Total Implementation Time**: 2-3 days for complete implementation and testing.

**Priority Order**:
1. Data Loading (HIGH) - Blocks end-to-end training
2. Rotation Augmentation (MEDIUM) - Enhances model performance
3. Additional Augmentations (LOW) - Nice to have features

## References

- **PIPELINE_COVERAGE.md**: Current state analysis
- **bridge.py**: Existing adapter implementation
- **DATASET.py**: Original data loading implementation
- **FILTER.py**: Original rotation filter implementation
- **model_modern.py**: Modern model architectures
- **config.py**: Configuration system
