"""
Data generator for LineamentLearning with tf.data.Dataset support.

This module provides modern data loading capabilities that wrap the original
DATASET class and provide efficient tf.data.Dataset pipelines.
"""

import tensorflow as tf
import numpy as np
from typing import Optional, Tuple
from pathlib import Path

from config import Config
from DATASET import DATASET


class DataGenerator:
    """Modern data generator wrapping original DATASET class.
    
    This class bridges the gap between original DATASET.py and modern
    TensorFlow 2.x training pipelines, providing:
    - tf.data.Dataset compatibility
    - Efficient batch loading
    - Prefetching and parallel processing
    - Integration with model.fit()
    """
    
    def __init__(self, config: Config, dataset_path: str, mode: str = 'normal', file_format: str = 'auto'):
        """Initialize data generator.
        
        Args:
            config: Configuration object
            dataset_path: Path to dataset file (.mat, .npz, or .h5)
            mode: Dataset mode ('normal' or other modes supported by DATASET)
            file_format: Format of dataset file:
                - 'auto': Auto-detect from extension (default)
                - 'mat': MATLAB .mat format
                - 'numpy' or 'npz': NumPy .npz format
                - 'hdf5' or 'h5': HDF5 format
        """
        self.config = config
        self.dataset_path = dataset_path
        self.mode = mode
        self.file_format = file_format
        
        # Load dataset using DATASET class (now supports multiple formats)
        self.dataset = DATASET(dataset_path, mode=mode, file_format=file_format)
        
        # Cache for generated data
        self._train_data = None
        self._val_data = None
        self._test_data = None
    
    def generate_training_data(self, 
                               ratio: float = 1.0,
                               choosy: bool = False,
                               output_type: float = 0) -> Tuple[np.ndarray, np.ndarray, tuple]:
        """Generate training data using original DATASET class.
        
        Args:
            ratio: Ratio of samples to use (0.0 to 1.0)
            choosy: Whether to only pick fault locations
            output_type: 0 for binary, np.pi/2.0 for angle detection
            
        Returns:
            Tuple of (X, Y, IDX) where:
            - X: Input patches (N, W, W, layers)
            - Y: Labels (N, 1)
            - IDX: Indices of samples
        """
        if self._train_data is None:
            print(f"Generating training data (ratio={ratio}, choosy={choosy})...")
            self._train_data = self.dataset.generateDS(
                output=self.dataset.OUTPUT,
                mask=self.dataset.trainMask,
                w=self.config.model.window_size,
                choosy=choosy,
                ratio=ratio,
                output_type=output_type
            )
        return self._train_data
    
    def generate_validation_data(self,
                                 ratio: float = 1.0) -> Tuple[np.ndarray, np.ndarray, tuple]:
        """Generate validation data.
        
        Args:
            ratio: Ratio of samples to use
            
        Returns:
            Tuple of (X, Y, IDX), or None if no validation data available
        """
        if not hasattr(self.dataset, 'testMask'):
            print("Warning: No testMask found in dataset. Validation data not available.")
            print("  This is expected for datasets loaded in non-normal mode.")
            return None
            
        if self._val_data is None:
            print(f"Generating validation data (ratio={ratio})...")
            self._val_data = self.dataset.generateDS(
                output=self.dataset.OUTPUT,
                mask=self.dataset.testMask,
                w=self.config.model.window_size,
                choosy=False,
                ratio=ratio,
                output_type=0
            )
        return self._val_data
    
    def create_training_dataset(self,
                               ratio: float = 0.1,
                               choosy: bool = False,
                               shuffle: bool = True,
                               cache: bool = False) -> tf.data.Dataset:
        """Create tf.data.Dataset for training with prefetching.
        
        Args:
            ratio: Ratio of training data to use
            choosy: Whether to only use fault locations
            shuffle: Whether to shuffle the data
            cache: Whether to cache the dataset in memory
            
        Returns:
            tf.data.Dataset configured for training
        """
        # Generate data using original DATASET
        X, Y, IDX = self.generate_training_data(ratio=ratio, choosy=choosy, output_type=0)
        
        print(f"Training dataset shape: X={X.shape}, Y={Y.shape}")
        
        # Create tf.data.Dataset
        dataset = tf.data.Dataset.from_tensor_slices((X, Y))
        
        # Cache if requested (useful for small datasets)
        if cache:
            dataset = dataset.cache()
        
        # Shuffle
        if shuffle:
            buffer_size = min(len(X), 10000)  # Limit buffer size for memory
            dataset = dataset.shuffle(buffer_size, seed=self.config.random_seed)
        
        # Batch
        dataset = dataset.batch(self.config.model.batch_size)
        
        # Prefetch for performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def create_validation_dataset(self,
                                 ratio: float = 0.5,
                                 cache: bool = True) -> Optional[tf.data.Dataset]:
        """Create tf.data.Dataset for validation.
        
        Args:
            ratio: Ratio of validation data to use
            cache: Whether to cache the dataset in memory
            
        Returns:
            tf.data.Dataset configured for validation, or None if no validation data
        """
        # Generate validation data
        val_data = self.generate_validation_data(ratio=ratio)
        
        if val_data is None:
            return None
        
        X_val, Y_val, _ = val_data
        print(f"Validation dataset shape: X={X_val.shape}, Y={Y_val.shape}")
        
        # Create tf.data.Dataset
        dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
        
        # Cache validation data (usually smaller and used multiple times)
        if cache:
            dataset = dataset.cache()
        
        # Batch
        dataset = dataset.batch(self.config.model.batch_size)
        
        # Prefetch
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_dataset_info(self) -> dict:
        """Get information about the dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        info = {
            'shape': (self.dataset.x, self.dataset.y),
            'layers': self.dataset.INPUTS.shape[2],
            'train_mask_size': int(np.sum(self.dataset.trainMask)),
            'total_mask_size': int(np.sum(self.dataset.MASK)),
        }
        
        # Add test mask info if available
        if hasattr(self.dataset, 'testMask'):
            info['test_mask_size'] = int(np.sum(self.dataset.testMask))
        
        # Add fault pixels info if available
        if hasattr(self.dataset, 'OUTPUT'):
            info['fault_pixels'] = int(np.sum(self.dataset.OUTPUT > 0))
        
        return info
    
    def clear_cache(self):
        """Clear cached data to free memory."""
        self._train_data = None
        self._val_data = None
        self._test_data = None


# Backward compatibility alias
TFDataGenerator = DataGenerator
