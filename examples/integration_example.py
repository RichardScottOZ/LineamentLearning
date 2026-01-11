"""
Integration example showing how to use original and modern pipelines together.

This example demonstrates:
1. Using original DATASET with modern models
2. Accessing all original functionality
3. Backward compatibility
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config


def example_1_dataset_adapter():
    """Example 1: Using DatasetAdapter to load original data."""
    print("=" * 60)
    print("Example 1: DatasetAdapter")
    print("=" * 60)
    
    from bridge import DatasetAdapter
    
    config = Config()
    config.model.window_size = 45
    
    print("\nDatasetAdapter bridges original DATASET with modern pipeline")
    print("It provides:")
    print("  - generate_training_data()")
    print("  - generate_validation_data()")
    print("  - get_dataset_info()")
    
    print("\nUsage:")
    print("------")
    print("adapter = DatasetAdapter(config, 'path/to/data.mat')")
    print("X, Y, IDX = adapter.generate_training_data(ratio=0.1)")
    print("info = adapter.get_dataset_info()")
    print()


def example_2_filter_adapter():
    """Example 2: Using FilterAdapter for rotation augmentation."""
    print("=" * 60)
    print("Example 2: FilterAdapter")
    print("=" * 60)
    
    from bridge import FilterAdapter
    
    print("\nFilterAdapter provides access to rotation filters")
    print("It provides:")
    print("  - get_random_rotation()")
    print("  - get_rotation_by_index(index)")
    print("  - get_all_rotations()")
    
    print("\nUsage:")
    print("------")
    print("adapter = FilterAdapter('path/to/filters.mat')")
    print("fnum, filter_matrix = adapter.get_random_rotation()")
    print("# Apply filter for augmentation")
    print()


def example_3_legacy_trainer():
    """Example 3: Using LegacyTrainer for complete workflow."""
    print("=" * 60)
    print("Example 3: LegacyTrainer")
    print("=" * 60)
    
    from bridge import LegacyTrainer
    
    config = Config()
    config.model.architecture = 'UNet'  # Can use any modern architecture!
    config.model.window_size = 45
    config.model.batch_size = 32
    
    print("\nLegacyTrainer combines original data with modern models")
    print("Configuration:")
    print(f"  Architecture: {config.model.architecture}")
    print(f"  Window Size: {config.model.window_size}")
    print(f"  Batch Size: {config.model.batch_size}")
    
    print("\nUsage:")
    print("------")
    print("trainer = LegacyTrainer(config, 'path/to/data.mat')")
    print("history = trainer.train_simple(ratio=0.1, epochs=5)")
    print("metrics = trainer.evaluate(ratio=0.5)")
    print()


def example_4_convenience_function():
    """Example 4: Using convenience function for quick training."""
    print("=" * 60)
    print("Example 4: Convenience Function")
    print("=" * 60)
    
    from bridge import train_with_original_pipeline
    
    print("\ntrain_with_original_pipeline() provides one-line training")
    
    print("\nUsage:")
    print("------")
    print("from config import Config")
    print("from bridge import train_with_original_pipeline")
    print("")
    print("config = Config()")
    print("model_path = train_with_original_pipeline(")
    print("    config=config,")
    print("    dataset_path='./Dataset/Australia/Rotations/Australia_360.mat',")
    print("    output_dir='./models/integrated',")
    print("    epochs=10")
    print(")")
    print()


def example_5_backward_compatibility():
    """Example 5: Everything from original still works."""
    print("=" * 60)
    print("Example 5: Backward Compatibility")
    print("=" * 60)
    
    print("\nAll original components still work:")
    
    print("\n1. Original MODEL:")
    print("   from MODEL import MODEL")
    print("   model = MODEL(w=45)")
    
    print("\n2. Original DATASET:")
    print("   from DATASET import DATASET")
    print("   ds = DATASET('path/to/data.mat')")
    print("   X, Y, IDX = ds.generateDS(ds.OUTPUT, ds.trainMask)")
    
    print("\n3. Original FILTER:")
    print("   from FILTER import FILTER")
    print("   flt = FILTER('path/to/filters.mat')")
    
    print("\n4. Original Prob2Line:")
    print("   from Prob2Line import prob2map")
    print("   p2l = prob2map(pmap)")
    print("   clusters = p2l.getClusters()")
    
    print("\n5. Original GUI:")
    print("   python Demo.py")
    
    print("\n6. Original Training:")
    print("   python RotateLearning.py train-choosy")
    print()


def example_6_modern_with_original_data():
    """Example 6: Complete example combining both."""
    print("=" * 60)
    print("Example 6: Complete Integration Example")
    print("=" * 60)
    
    print("\nComplete workflow using original data with modern stack:")
    
    print("\nStep 1: Configure")
    print("--------")
    print("from config import Config")
    print("config = Config()")
    print("config.model.architecture = 'ResNet'")
    print("config.model.window_size = 64")
    print("config.model.use_batch_normalization = True")
    
    print("\nStep 2: Load Data (Original)")
    print("--------")
    print("from bridge import DatasetAdapter")
    print("adapter = DatasetAdapter(config, 'data.mat')")
    print("X_train, Y_train, _ = adapter.generate_training_data(ratio=0.2)")
    
    print("\nStep 3: Build Model (Modern)")
    print("--------")
    print("from model_modern import build_model")
    print("model = build_model(config)")
    
    print("\nStep 4: Train")
    print("--------")
    print("history = model.fit(X_train, Y_train, epochs=10)")
    
    print("\nStep 5: Predict")
    print("--------")
    print("predictions = model.predict(X_test)")
    
    print("\nStep 6: Post-process (Modern)")
    print("--------")
    print("from postprocessing import PostProcessor")
    print("processor = PostProcessor(config.inference)")
    print("clusters, lineaments = processor.extract_lineaments(pmap)")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Original + Modern Pipeline Integration Examples")
    print("=" * 60)
    print()
    
    example_1_dataset_adapter()
    example_2_filter_adapter()
    example_3_legacy_trainer()
    example_4_convenience_function()
    example_5_backward_compatibility()
    example_6_modern_with_original_data()
    
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print()
    print("The bridge module provides seamless integration between:")
    print("  ✓ Original DATASET.py → Modern ModelTrainer")
    print("  ✓ Original FILTER.py → Modern augmentation")
    print("  ✓ Original workflows → Modern architectures")
    print()
    print("All original code still works (100% backward compatible)")
    print()
    print("For complete details, see:")
    print("  - PIPELINE_COVERAGE.md - Feature comparison")
    print("  - bridge.py - Integration code")
    print("  - POSTPROCESSING_GUIDE.md - Post-processing details")
    print()
    print("=" * 60)


if __name__ == '__main__':
    main()
