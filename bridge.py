"""
Bridge module connecting original and modern pipelines.

This module provides adapters to use original DATASET, FILTER, and other
components with the modern ModelTrainer and configuration system.
"""

import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path

from config import Config
from DATASET import DATASET
from FILTER import FILTER
from Utility import myNormalizer


class DatasetAdapter:
    """Adapter to use original DATASET class with modern pipeline.
    
    This class bridges the gap between the original data loading
    and the modern training infrastructure.
    """
    
    def __init__(self, config: Config, dataset_path: str):
        """Initialize adapter.
        
        Args:
            config: Modern configuration object
            dataset_path: Path to .mat dataset file
        """
        self.config = config
        self.dataset = DATASET(dataset_path)
        self.dataset_path = dataset_path
    
    def generate_training_data(self, 
                              ratio: float = 1.0,
                              choosy: bool = False,
                              output_type: float = 0) -> Tuple[np.ndarray, np.ndarray, tuple]:
        """Generate training data using original DATASET class.
        
        Args:
            ratio: Ratio of samples to use
            choosy: Whether to only pick fault locations
            output_type: 0 for binary, np.pi/2.0 for angle detection
            
        Returns:
            Tuple of (X, Y, IDX) where:
            - X: Input patches (N, W, W, layers)
            - Y: Labels (N, 1)
            - IDX: Indices of samples
        """
        return self.dataset.generateDS(
            output=self.dataset.OUTPUT,
            mask=self.dataset.trainMask,
            w=self.config.model.window_size,
            choosy=choosy,
            ratio=ratio,
            output_type=output_type
        )
    
    def generate_validation_data(self,
                                 ratio: float = 1.0) -> Tuple[np.ndarray, np.ndarray, tuple]:
        """Generate validation data.
        
        Args:
            ratio: Ratio of samples to use
            
        Returns:
            Tuple of (X, Y, IDX)
        """
        return self.dataset.generateDS(
            output=self.dataset.OUTPUT,
            mask=self.dataset.testMask,
            w=self.config.model.window_size,
            choosy=False,
            ratio=ratio,
            output_type=0
        )
    
    def get_dataset_info(self) -> dict:
        """Get information about the dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        return {
            'shape': (self.dataset.x, self.dataset.y),
            'layers': self.dataset.INPUTS.shape[2],
            'train_mask_size': np.sum(self.dataset.trainMask),
            'test_mask_size': np.sum(self.dataset.testMask),
            'total_mask_size': np.sum(self.dataset.MASK),
            'fault_pixels': np.sum(self.dataset.OUTPUT > 0)
        }


class FilterAdapter:
    """Adapter to use original FILTER class for rotation augmentation.
    
    Provides easy access to rotation matrices for data augmentation.
    """
    
    def __init__(self, filter_path: str):
        """Initialize adapter.
        
        Args:
            filter_path: Path to filter .mat file
        """
        self.filter = FILTER(filter_path)
        self.n_filters = self.filter.N
    
    def get_random_rotation(self) -> Tuple[int, np.ndarray]:
        """Get a random rotation filter.
        
        Returns:
            Tuple of (filter_number, filter_matrix)
        """
        return self.filter.getFilter(n=1)
    
    def get_rotation_by_index(self, index: int) -> Tuple[int, np.ndarray]:
        """Get specific rotation filter.
        
        Args:
            index: Filter index
            
        Returns:
            Tuple of (filter_number, filter_matrix)
        """
        return self.filter.getFilterbyNumber(index)
    
    def get_all_rotations(self) -> List[Tuple[int, np.ndarray]]:
        """Get all rotation filters.
        
        Returns:
            List of (filter_number, filter_matrix) tuples
        """
        return [self.get_rotation_by_index(i) for i in range(self.n_filters)]


class LegacyTrainer:
    """Trainer that uses original data loading with modern architecture.
    
    This class demonstrates how to use original DATASET and FILTER
    classes with the modern model architectures.
    """
    
    def __init__(self, config: Config, dataset_path: str):
        """Initialize legacy trainer.
        
        Args:
            config: Modern configuration
            dataset_path: Path to dataset .mat file
        """
        self.config = config
        self.dataset_adapter = DatasetAdapter(config, dataset_path)
        
        # Build modern model
        from model_modern import build_model
        self.model = build_model(config)
    
    def train_simple(self, 
                    ratio: float = 0.1,
                    epochs: int = 1) -> dict:
        """Simple training using original data loader.
        
        Args:
            ratio: Ratio of training data to use
            epochs: Number of epochs
            
        Returns:
            Training history
        """
        # Load data using original DATASET
        print("Loading training data...")
        X_train, Y_train, _ = self.dataset_adapter.generate_training_data(
            ratio=ratio,
            choosy=False,
            output_type=0
        )
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Labels shape: {Y_train.shape}")
        
        # Train using modern model
        print("Training model...")
        history = self.model.fit(
            X_train, Y_train,
            batch_size=self.config.model.batch_size,
            epochs=epochs,
            validation_split=0.2,
            verbose=1
        )
        
        return history.history
    
    def evaluate(self, ratio: float = 0.5) -> dict:
        """Evaluate model on test data.
        
        Args:
            ratio: Ratio of test data to use
            
        Returns:
            Evaluation metrics
        """
        print("Loading test data...")
        X_test, Y_test, _ = self.dataset_adapter.generate_validation_data(ratio=ratio)
        
        print(f"Test data shape: {X_test.shape}")
        
        print("Evaluating model...")
        results = self.model.evaluate(X_test, Y_test, verbose=1)
        
        # Get metric names
        metric_names = self.model.metrics_names
        
        return dict(zip(metric_names, results))


def train_with_original_pipeline(config: Config,
                                 dataset_path: str,
                                 output_dir: str,
                                 epochs: int = 10) -> str:
    """Convenience function to train using original data pipeline.
    
    This demonstrates the complete integration between original and
    modern components.
    
    Args:
        config: Configuration object
        dataset_path: Path to dataset .mat file
        output_dir: Directory to save model
        epochs: Number of training epochs
        
    Returns:
        Path to saved model
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Create trainer
    trainer = LegacyTrainer(config, dataset_path)
    
    # Get dataset info
    info = trainer.dataset_adapter.get_dataset_info()
    print("\nDataset Information:")
    print(f"  Shape: {info['shape']}")
    print(f"  Layers: {info['layers']}")
    print(f"  Train mask size: {info['train_mask_size']}")
    print(f"  Test mask size: {info['test_mask_size']}")
    print(f"  Fault pixels: {info['fault_pixels']}")
    
    # Train
    print(f"\nTraining for {epochs} epochs...")
    history = trainer.train_simple(ratio=0.1, epochs=epochs)
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = trainer.evaluate(ratio=0.5)
    print("Test metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Save model
    model_path = os.path.join(output_dir, 'model.h5')
    trainer.model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    return model_path


# Example usage
if __name__ == '__main__':
    from config import Config
    
    # Create configuration
    config = Config()
    config.model.architecture = 'RotateNet'
    config.model.window_size = 45
    config.model.epochs = 5
    
    print("=" * 60)
    print("Legacy Pipeline Integration Example")
    print("=" * 60)
    
    print("\nThis module demonstrates how to use original DATASET")
    print("and FILTER classes with modern model architectures.")
    
    print("\nUsage:")
    print("------")
    print("from bridge import DatasetAdapter, LegacyTrainer")
    print("")
    print("# Create adapter")
    print("adapter = DatasetAdapter(config, 'path/to/data.mat')")
    print("")
    print("# Generate training data")
    print("X, Y, IDX = adapter.generate_training_data(ratio=0.1)")
    print("")
    print("# Or use complete trainer")
    print("trainer = LegacyTrainer(config, 'path/to/data.mat')")
    print("history = trainer.train_simple(ratio=0.1, epochs=5)")
    print("")
    print("# Or use convenience function")
    print("model_path = train_with_original_pipeline(")
    print("    config=config,")
    print("    dataset_path='path/to/data.mat',")
    print("    output_dir='./models',")
    print("    epochs=10")
    print(")")
    
    print("\n" + "=" * 60)
    print("See PIPELINE_COVERAGE.md for complete integration details")
    print("=" * 60)
