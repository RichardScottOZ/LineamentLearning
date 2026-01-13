"""
Configuration management for LineamentLearning.

This module provides a modern, flexible configuration system using dataclasses
and supports loading from YAML/JSON files.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
import json


@dataclass
class ModelConfig:
    """Configuration for model architecture and training."""
    
    window_size: int = 45
    layers: int = 8
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 150
    
    # Modern architectures
    architecture: str = "RotateNet"  # Options: RotateNet, UNet, ResNet
    use_batch_normalization: bool = True
    use_dropout: bool = True
    dropout_rate: float = 0.3
    
    # Advanced training options
    use_mixed_precision: bool = False
    use_early_stopping: bool = True
    early_stopping_patience: int = 10
    
    # Data augmentation
    use_augmentation: bool = True
    rotation_range: int = 360
    flip_horizontal: bool = False
    flip_vertical: bool = False


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    mask_threshold: float = 0.9
    radian_threshold: float = 0.2618  # π/12
    
    # Directories
    dataset_dir: str = "./Dataset/Australia/Rotations/"
    results_dir: str = "./Results/"
    callbacks_dir: str = "./CallBacks/Rotate/"
    figures_dir: str = "./Figures/Rotate/"
    filters_dir: str = "./Filters/"
    pmap_dir: str = "./Pmaps/"
    
    # Data processing
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    normalize_inputs: bool = True


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    
    # Rotation augmentation
    enable_rotation: bool = False
    rotation_filter_path: Optional[str] = None  # Path to FILTER.py .mat file
    rotation_probability: float = 0.5  # Probability of applying rotation
    rotation_angles: List[int] = field(default_factory=lambda: [0, 90, 180, 270])  # TF rotation angles
    
    # Flipping augmentation
    enable_flipping: bool = False
    flip_probability: float = 0.5
    
    # Brightness/contrast augmentation
    enable_brightness: bool = False
    brightness_delta: float = 0.1
    enable_contrast: bool = False
    contrast_range: Tuple[float, float] = (0.9, 1.1)


@dataclass
class InferenceConfig:
    """Configuration for model inference."""
    
    threshold: float = 0.5
    cutoff: float = 0.3
    eps: float = 0.3
    min_cluster_size: int = 20
    
    # Post-processing
    use_clustering: bool = True
    clustering_method: str = "DBSCAN"  # Options: DBSCAN, HDBSCAN
    line_fitting_method: str = "BestCurve"  # Options: Linear, Curve, BestCurve
    polynomial_degrees: List[int] = field(default_factory=lambda: [1, 3, 5])


@dataclass
class Config:
    """Main configuration container."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # General settings
    debug_mode: bool = True
    random_seed: int = 42
    num_workers: int = 4
    device: str = "auto"  # Options: auto, cpu, gpu, tpu
    
    @classmethod
    def from_file(cls, filepath: str) -> 'Config':
        """Load configuration from JSON file.
        
        Args:
            filepath: Path to JSON configuration file
            
        Returns:
            Config object with loaded settings
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            data=DataConfig(**config_dict.get('data', {})),
            augmentation=AugmentationConfig(**config_dict.get('augmentation', {})),
            inference=InferenceConfig(**config_dict.get('inference', {})),
            debug_mode=config_dict.get('debug_mode', True),
            random_seed=config_dict.get('random_seed', 42),
            num_workers=config_dict.get('num_workers', 4),
            device=config_dict.get('device', 'auto'),
        )
    
    def to_file(self, filepath: str):
        """Save configuration to JSON file.
        
        Args:
            filepath: Path where to save configuration
        """
        from dataclasses import asdict
        
        config_dict = {
            'model': asdict(self.model),
            'data': asdict(self.data),
            'augmentation': asdict(self.augmentation),
            'inference': asdict(self.inference),
            'debug_mode': self.debug_mode,
            'random_seed': self.random_seed,
            'num_workers': self.num_workers,
            'device': self.device,
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    def validate(self) -> bool:
        """Validate configuration settings.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate ratios sum to 1
        total_ratio = self.data.train_ratio + self.data.val_ratio + self.data.test_ratio
        if not (0.99 <= total_ratio <= 1.01):
            raise ValueError(f"Data split ratios must sum to 1.0, got {total_ratio}")
        
        # Validate directories exist or can be created
        dirs_to_check = [
            self.data.results_dir,
            self.data.callbacks_dir,
            self.data.figures_dir,
        ]
        
        for dir_path in dirs_to_check:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        return True


# Default configuration instance
DEFAULT_CONFIG = Config()


def get_config(config_file: Optional[str] = None) -> Config:
    """Get configuration object.
    
    Args:
        config_file: Optional path to configuration file
        
    Returns:
        Config object
    """
    if config_file is not None:
        return Config.from_file(config_file)
    return DEFAULT_CONFIG
