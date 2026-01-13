"""
Example: Training with DataGenerator and Rotation Augmentation

This example demonstrates the new integrated data loading and rotation
augmentation features.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from data_generator import DataGenerator
from model_modern import ModelTrainer


def example_1_basic_training():
    """Example 1: Basic training with DataGenerator."""
    print("=" * 70)
    print("Example 1: Basic Training with DataGenerator")
    print("=" * 70)
    
    # Create configuration
    config = Config()
    config.model.architecture = 'RotateNet'
    config.model.window_size = 45
    config.model.epochs = 5  # Small number for demo
    config.model.batch_size = 32
    
    # No augmentation in this example
    config.augmentation.enable_rotation = False
    config.augmentation.enable_flipping = False
    
    print("\nConfiguration:")
    print(f"  Architecture: {config.model.architecture}")
    print(f"  Window size: {config.model.window_size}")
    print(f"  Epochs: {config.model.epochs}")
    print(f"  Batch size: {config.model.batch_size}")
    
    # Example dataset path (replace with actual path)
    dataset_path = "./Dataset/Australia/Rotations/Australia_strip.mat"
    
    print(f"\nDataset path: {dataset_path}")
    print("\nNote: This is a demonstration. Replace dataset_path with your actual data.")
    print("\nTo run training:")
    print(f"  1. Ensure dataset exists at: {dataset_path}")
    print("  2. Uncomment the training code below")
    
    # Uncomment to run actual training:
    """
    # Create trainer with DataGenerator
    trainer = ModelTrainer(
        config=config,
        output_dir='./outputs/example1'
    )
    
    # Train with automatic data loading
    history = trainer.train(
        data_path=dataset_path,
        train_ratio=0.1,  # Use 10% of training data for quick demo
        val_ratio=0.5,
        use_tensorboard=False
    )
    
    print("\nTraining complete!")
    """


def example_2_with_rotation_augmentation():
    """Example 2: Training with rotation augmentation."""
    print("\n" + "=" * 70)
    print("Example 2: Training with Rotation Augmentation")
    print("=" * 70)
    
    # Create configuration with rotation augmentation
    config = Config()
    config.model.architecture = 'RotateNet'
    config.model.window_size = 45
    config.model.epochs = 5
    config.model.batch_size = 32
    
    # Enable rotation augmentation
    config.augmentation.enable_rotation = True
    config.augmentation.rotation_probability = 0.5  # 50% chance of rotation
    config.augmentation.rotation_angles = [0, 90, 180, 270]  # 90-degree rotations
    
    # Optionally use FILTER.py rotation matrices
    # config.augmentation.rotation_filter_path = "./Filters/Default.mat"
    
    print("\nConfiguration:")
    print(f"  Architecture: {config.model.architecture}")
    print(f"  Rotation augmentation: ENABLED")
    print(f"  Rotation probability: {config.augmentation.rotation_probability}")
    print(f"  Rotation angles: {config.augmentation.rotation_angles}")
    
    dataset_path = "./Dataset/Australia/Rotations/Australia_strip.mat"
    
    print(f"\nDataset path: {dataset_path}")
    print("\nNote: This is a demonstration. Replace dataset_path with your actual data.")
    print("\nTo run training:")
    print("  1. Ensure dataset exists")
    print("  2. Uncomment the training code below")
    
    # Uncomment to run actual training:
    """
    trainer = ModelTrainer(
        config=config,
        output_dir='./outputs/example2_with_rotation'
    )
    
    history = trainer.train(
        data_path=dataset_path,
        train_ratio=0.1,
        val_ratio=0.5,
        use_tensorboard=False
    )
    
    print("\nTraining complete with rotation augmentation!")
    """


def example_3_separate_data_generator():
    """Example 3: Using DataGenerator separately."""
    print("\n" + "=" * 70)
    print("Example 3: Using DataGenerator Separately")
    print("=" * 70)
    
    # Create configuration
    config = Config()
    config.model.window_size = 45
    config.model.batch_size = 32
    
    dataset_path = "./Dataset/Australia/Rotations/Australia_strip.mat"
    
    print("\nThis example shows how to use DataGenerator separately")
    print("for more control over data loading.")
    
    print("\nTo run:")
    print("  1. Ensure dataset exists")
    print("  2. Uncomment the code below")
    
    # Uncomment to run:
    """
    # Create DataGenerator
    data_gen = DataGenerator(config, dataset_path)
    
    # Get dataset info
    info = data_gen.get_dataset_info()
    print("\nDataset Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Create tf.data.Dataset objects
    train_ds = data_gen.create_training_dataset(ratio=0.1, shuffle=True)
    val_ds = data_gen.create_validation_dataset(ratio=0.5)
    
    # Create trainer with data generator
    trainer = ModelTrainer(
        config=config,
        output_dir='./outputs/example3',
        data_generator=data_gen
    )
    
    # Train using the pre-configured data generator
    history = trainer.train(train_ratio=0.1, val_ratio=0.5)
    
    print("\nTraining complete!")
    """


def example_4_full_augmentation():
    """Example 4: Training with all augmentation options."""
    print("\n" + "=" * 70)
    print("Example 4: Training with Full Augmentation")
    print("=" * 70)
    
    # Create configuration with all augmentations
    config = Config()
    config.model.architecture = 'UNet'  # Try different architecture
    config.model.window_size = 64  # Larger window
    config.model.epochs = 10
    config.model.batch_size = 16
    config.model.use_early_stopping = True
    config.model.early_stopping_patience = 3
    
    # Enable all augmentations
    config.augmentation.enable_rotation = True
    config.augmentation.rotation_probability = 0.5
    config.augmentation.rotation_angles = [0, 90, 180, 270]
    
    config.augmentation.enable_flipping = True
    config.augmentation.flip_probability = 0.5
    
    print("\nConfiguration:")
    print(f"  Architecture: {config.model.architecture}")
    print(f"  Window size: {config.model.window_size}")
    print(f"  Epochs: {config.model.epochs}")
    print(f"  Early stopping: {config.model.use_early_stopping}")
    print("\nAugmentation:")
    print(f"  Rotation: ENABLED (p={config.augmentation.rotation_probability})")
    print(f"  Flipping: ENABLED (p={config.augmentation.flip_probability})")
    
    dataset_path = "./Dataset/Australia/Rotations/Australia_strip.mat"
    
    print(f"\nDataset path: {dataset_path}")
    print("\nNote: This is a demonstration. Replace dataset_path with your actual data.")
    
    # Uncomment to run:
    """
    trainer = ModelTrainer(
        config=config,
        output_dir='./outputs/example4_full_augmentation'
    )
    
    history = trainer.train(
        data_path=dataset_path,
        train_ratio=0.2,  # Use more data
        val_ratio=0.5,
        use_tensorboard=True  # Enable TensorBoard
    )
    
    print("\nTraining complete with full augmentation!")
    print("View TensorBoard logs:")
    print("  tensorboard --logdir=./outputs/example4_full_augmentation/logs")
    """


def main():
    """Run all examples."""
    print("\n")
    print("=" * 70)
    print("LineamentLearning - Data Loading & Rotation Examples")
    print("=" * 70)
    print("\nThese examples demonstrate the new integrated features:")
    print("  1. DataGenerator for efficient data loading")
    print("  2. Rotation augmentation")
    print("  3. End-to-end training pipeline")
    print("\n")
    
    # Run examples (demonstrations only - training code is commented out)
    example_1_basic_training()
    example_2_with_rotation_augmentation()
    example_3_separate_data_generator()
    example_4_full_augmentation()
    
    print("\n" + "=" * 70)
    print("Examples Complete")
    print("=" * 70)
    print("\nTo run actual training:")
    print("  1. Ensure you have a .mat dataset file")
    print("  2. Edit the dataset_path in each example")
    print("  3. Uncomment the training code")
    print("  4. Run: python examples/train_with_data_generator.py")
    print("\nFor more information, see:")
    print("  - DATA_LOADING_ROTATION_IMPROVEMENTS.md")
    print("  - PIPELINE_COVERAGE.md")
    print("\n")


if __name__ == '__main__':
    main()
