"""
Modern model architectures for LineamentLearning.

This module provides updated model architectures using TensorFlow 2.x/Keras
with support for multiple architectures and modern training techniques.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Optional, Tuple, TYPE_CHECKING
import numpy as np

from config import Config, ModelConfig

if TYPE_CHECKING:
    from data_generator import DataGenerator


def create_rotatenet(config: ModelConfig) -> keras.Model:
    """Create the original RotateNet architecture with modern improvements.
    
    Args:
        config: Model configuration
        
    Returns:
        Keras model
    """
    inputs = layers.Input(
        shape=(config.window_size, config.window_size, config.layers),
        name='input_layer'
    )
    
    # Convolutional layer
    x = layers.Conv2D(
        8, 
        kernel_size=3, 
        padding='valid', 
        activation='relu',
        name='conv2d'
    )(inputs)
    
    # Optional batch normalization
    if config.use_batch_normalization:
        x = layers.BatchNormalization()(x)
    
    # Flatten
    x = layers.Flatten()(x)
    
    # Dense layers with optional dropout
    x = layers.Dense(300, activation='relu', name='dense1')(x)
    if config.use_dropout:
        x = layers.Dropout(config.dropout_rate)(x)
    
    if config.use_batch_normalization:
        x = layers.BatchNormalization()(x)
    
    x = layers.Dense(300, activation='relu', name='dense2')(x)
    if config.use_dropout:
        x = layers.Dropout(config.dropout_rate)(x)
    
    # Output layer
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='RotateNet')
    
    return model


def create_unet(config: ModelConfig) -> keras.Model:
    """Create a U-Net architecture for lineament detection.
    
    U-Net is excellent for image segmentation tasks and can better
    capture spatial context than the original architecture.
    
    Args:
        config: Model configuration
        
    Returns:
        Keras model
    """
    inputs = layers.Input(
        shape=(config.window_size, config.window_size, config.layers),
        name='input_layer'
    )
    
    # Encoder
    # Block 1
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(c1)
    if config.use_batch_normalization:
        c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling2D(2)(c1)
    if config.use_dropout:
        p1 = layers.Dropout(config.dropout_rate * 0.5)(p1)
    
    # Block 2
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(c2)
    if config.use_batch_normalization:
        c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D(2)(c2)
    if config.use_dropout:
        p2 = layers.Dropout(config.dropout_rate * 0.5)(p2)
    
    # Block 3
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(c3)
    if config.use_batch_normalization:
        c3 = layers.BatchNormalization()(c3)
    p3 = layers.MaxPooling2D(2)(c3)
    if config.use_dropout:
        p3 = layers.Dropout(config.dropout_rate)(p3)
    
    # Bottleneck
    c4 = layers.Conv2D(128, 3, activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(128, 3, activation='relu', padding='same')(c4)
    if config.use_batch_normalization:
        c4 = layers.BatchNormalization()(c4)
    
    # Decoder
    # Block 5
    u5 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv2D(64, 3, activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(64, 3, activation='relu', padding='same')(c5)
    if config.use_batch_normalization:
        c5 = layers.BatchNormalization()(c5)
    
    # Block 6
    u6 = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(32, 3, activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(32, 3, activation='relu', padding='same')(c6)
    if config.use_batch_normalization:
        c6 = layers.BatchNormalization()(c6)
    
    # Block 7
    u7 = layers.Conv2DTranspose(16, 2, strides=2, padding='same')(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(16, 3, activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(16, 3, activation='relu', padding='same')(c7)
    
    # Global pooling and classification
    x = layers.GlobalAveragePooling2D()(c7)
    x = layers.Dense(64, activation='relu')(x)
    if config.use_dropout:
        x = layers.Dropout(config.dropout_rate)(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='UNet')
    
    return model


def create_resnet_block(x, filters: int, kernel_size: int = 3, 
                       stride: int = 1, use_bn: bool = True):
    """Create a ResNet block with skip connection.
    
    Args:
        x: Input tensor
        filters: Number of filters
        kernel_size: Kernel size
        stride: Stride
        use_bn: Whether to use batch normalization
        
    Returns:
        Output tensor
    """
    shortcut = x
    
    # First conv
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Second conv
    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    
    # Match dimensions if needed
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        if use_bn:
            shortcut = layers.BatchNormalization()(shortcut)
    
    # Add skip connection
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x


def create_resnet(config: ModelConfig) -> keras.Model:
    """Create a ResNet-inspired architecture.
    
    ResNet with skip connections can help with training deeper networks
    and capturing complex patterns.
    
    Args:
        config: Model configuration
        
    Returns:
        Keras model
    """
    inputs = layers.Input(
        shape=(config.window_size, config.window_size, config.layers),
        name='input_layer'
    )
    
    # Initial convolution
    x = layers.Conv2D(32, 7, strides=2, padding='same')(inputs)
    if config.use_batch_normalization:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # ResNet blocks
    x = create_resnet_block(x, 32, use_bn=config.use_batch_normalization)
    x = create_resnet_block(x, 32, use_bn=config.use_batch_normalization)
    
    x = create_resnet_block(x, 64, stride=2, use_bn=config.use_batch_normalization)
    x = create_resnet_block(x, 64, use_bn=config.use_batch_normalization)
    
    x = create_resnet_block(x, 128, stride=2, use_bn=config.use_batch_normalization)
    x = create_resnet_block(x, 128, use_bn=config.use_batch_normalization)
    
    # Global pooling and classification
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    if config.use_dropout:
        x = layers.Dropout(config.dropout_rate)(x)
    x = layers.Dense(128, activation='relu')(x)
    if config.use_dropout:
        x = layers.Dropout(config.dropout_rate)(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='ResNet')
    
    return model


class RotationAugmentation(layers.Layer):
    """Custom augmentation layer for rotation during training.
    
    This layer applies random rotations to input images during training.
    It can use either TensorFlow's built-in rotation or FILTER.py rotation matrices.
    """
    
    def __init__(self, 
                 filter_path: Optional[str] = None,
                 rotation_angles: Optional[list] = None,
                 probability: float = 0.5,
                 **kwargs):
        """Initialize rotation augmentation layer.
        
        Args:
            filter_path: Optional path to FILTER.py .mat file
            rotation_angles: List of angles in degrees for random rotation (e.g., [0, 90, 180, 270])
            probability: Probability of applying rotation (0.0 to 1.0)
        """
        super().__init__(**kwargs)
        self.filter_path = filter_path
        self.rotation_angles = rotation_angles or [0, 90, 180, 270]
        self.probability = probability
        self.use_original_filters = filter_path is not None
        
        # Load FILTER if path provided
        if self.use_original_filters:
            try:
                from FILTER import FILTER
                self.filter = FILTER(filter_path)
            except Exception as e:
                print(f"Warning: Could not load FILTER from {filter_path}: {e}")
                print("Falling back to TensorFlow rotation")
                self.use_original_filters = False
    
    def call(self, inputs, training=None):
        """Apply rotation augmentation during training.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Augmented input tensor
        """
        if not training:
            return inputs
        
        # Apply rotation with given probability
        if tf.random.uniform([]) < self.probability:
            return self._apply_rotation(inputs)
        
        return inputs
    
    def _apply_rotation(self, inputs):
        """Apply rotation to inputs using TensorFlow operations.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Rotated tensor
        """
        # For TensorFlow rotation, use random angle from the list
        # Convert angles to radians for tf.image operations
        angles_rad = [angle * np.pi / 180.0 for angle in self.rotation_angles]
        
        # Randomly select an angle
        angle_idx = tf.random.uniform([], 0, len(angles_rad), dtype=tf.int32)
        angle = tf.constant(angles_rad)[angle_idx]
        
        # Use tf.image.rot90 for 90-degree rotations (more efficient)
        # For k rotations: k=1 means 90°, k=2 means 180°, k=3 means 270°
        if len(self.rotation_angles) == 4 and all(a % 90 == 0 for a in self.rotation_angles):
            k = angle_idx  # Direct mapping: 0->0°, 1->90°, 2->180°, 3->270°
            return tf.image.rot90(inputs, k=k)
        else:
            # For arbitrary angles, use scipy-based rotation
            # Note: This is less efficient and may not work in graph mode
            return tf.py_function(
                func=lambda x, a: self._scipy_rotate(x.numpy(), a.numpy()),
                inp=[inputs, angle],
                Tout=inputs.dtype
            )
    
    def _scipy_rotate(self, inputs, angle):
        """Rotate using scipy (for arbitrary angles).
        
        Args:
            inputs: Numpy array
            angle: Rotation angle in radians
            
        Returns:
            Rotated numpy array
        """
        import scipy.ndimage
        angle_deg = angle * 180.0 / np.pi
        return scipy.ndimage.rotate(inputs, angle_deg, reshape=False, order=1)
    
    def get_config(self):
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            'filter_path': self.filter_path,
            'rotation_angles': self.rotation_angles,
            'probability': self.probability,
        })
        return config


def build_model(config: Config, apply_augmentation: bool = True) -> keras.Model:
    """Build a model based on configuration.
    
    Args:
        config: Configuration object
        apply_augmentation: Whether to add augmentation layers to the model
        
    Returns:
        Compiled Keras model
    """
    # Create base model architecture
    base_inputs = layers.Input(
        shape=(config.model.window_size, config.model.window_size, config.model.layers),
        name='input_layer'
    )
    
    x = base_inputs
    
    # Add augmentation layers if enabled (applied during training only)
    if apply_augmentation and config.augmentation.enable_rotation:
        x = RotationAugmentation(
            filter_path=config.augmentation.rotation_filter_path,
            rotation_angles=config.augmentation.rotation_angles,
            probability=config.augmentation.rotation_probability
        )(x)
    
    if apply_augmentation and config.augmentation.enable_flipping:
        x = layers.RandomFlip(
            "horizontal_and_vertical",
            seed=config.random_seed
        )(x)
    
    # Create core model architecture (without input layer since we have augmentation)
    if config.model.architecture == 'RotateNet':
        # For RotateNet, we need to rebuild without the input layer
        # Conv layer
        x = layers.Conv2D(8, kernel_size=3, padding='valid', activation='relu', name='conv2d')(x)
        if config.model.use_batch_normalization:
            x = layers.BatchNormalization()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(300, activation='relu', name='dense1')(x)
        if config.model.use_dropout:
            x = layers.Dropout(config.model.dropout_rate)(x)
        if config.model.use_batch_normalization:
            x = layers.BatchNormalization()(x)
        x = layers.Dense(300, activation='relu', name='dense2')(x)
        if config.model.use_dropout:
            x = layers.Dropout(config.model.dropout_rate)(x)
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        model = keras.Model(inputs=base_inputs, outputs=outputs, name='RotateNet')
        
    elif config.model.architecture == 'UNet':
        # Build UNet on augmented input
        # Encoder Block 1
        c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(x)
        c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(c1)
        if config.model.use_batch_normalization:
            c1 = layers.BatchNormalization()(c1)
        p1 = layers.MaxPooling2D(2)(c1)
        if config.model.use_dropout:
            p1 = layers.Dropout(config.model.dropout_rate * 0.5)(p1)
        
        # Encoder Block 2
        c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(p1)
        c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(c2)
        if config.model.use_batch_normalization:
            c2 = layers.BatchNormalization()(c2)
        p2 = layers.MaxPooling2D(2)(c2)
        if config.model.use_dropout:
            p2 = layers.Dropout(config.model.dropout_rate * 0.5)(p2)
        
        # Bottleneck
        c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(p2)
        c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(c3)
        if config.model.use_batch_normalization:
            c3 = layers.BatchNormalization()(c3)
        
        # Decoder Block 1
        u1 = layers.UpSampling2D(2)(c3)
        u1 = layers.Concatenate()([u1, c2])
        c4 = layers.Conv2D(32, 3, activation='relu', padding='same')(u1)
        c4 = layers.Conv2D(32, 3, activation='relu', padding='same')(c4)
        if config.model.use_batch_normalization:
            c4 = layers.BatchNormalization()(c4)
        
        # Decoder Block 2
        u2 = layers.UpSampling2D(2)(c4)
        u2 = layers.Concatenate()([u2, c1])
        c5 = layers.Conv2D(16, 3, activation='relu', padding='same')(u2)
        c5 = layers.Conv2D(16, 3, activation='relu', padding='same')(c5)
        if config.model.use_batch_normalization:
            c5 = layers.BatchNormalization()(c5)
        
        # Global pooling and output
        x = layers.GlobalAveragePooling2D()(c5)
        x = layers.Dense(128, activation='relu')(x)
        if config.model.use_dropout:
            x = layers.Dropout(config.model.dropout_rate)(x)
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        model = keras.Model(inputs=base_inputs, outputs=outputs, name='UNet')
        
    elif config.model.architecture == 'ResNet':
        # Build ResNet on augmented input
        x = layers.Conv2D(64, 7, strides=2, padding='same')(x)
        if config.model.use_batch_normalization:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
        
        # Residual blocks (simplified)
        for filters in [64, 64, 128, 128]:
            shortcut = x
            x = layers.Conv2D(filters, 3, padding='same')(x)
            if config.model.use_batch_normalization:
                x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(filters, 3, padding='same')(x)
            if config.model.use_batch_normalization:
                x = layers.BatchNormalization()(x)
            
            # Adjust shortcut if needed
            if shortcut.shape[-1] != filters:
                shortcut = layers.Conv2D(filters, 1)(shortcut)
            x = layers.Add()([x, shortcut])
            x = layers.Activation('relu')(x)
        
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        if config.model.use_dropout:
            x = layers.Dropout(config.model.dropout_rate)(x)
        x = layers.Dense(128, activation='relu')(x)
        if config.model.use_dropout:
            x = layers.Dropout(config.model.dropout_rate)(x)
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        model = keras.Model(inputs=base_inputs, outputs=outputs, name='ResNet')
    else:
        raise ValueError(f"Unknown architecture: {config.model.architecture}")
    
    # Enable mixed precision training if configured
    if config.model.use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Setup optimizer with learning rate
    optimizer = keras.optimizers.Adam(learning_rate=config.model.learning_rate)
    
    # Wrap optimizer for mixed precision if needed
    if config.model.use_mixed_precision:
        optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
        ]
    )
    
    return model


class ModelTrainer:
    """Wrapper class for model training with modern features."""
    
    def __init__(self, config: Config, output_dir: str, data_generator: Optional['DataGenerator'] = None):
        """Initialize trainer.
        
        Args:
            config: Configuration object
            output_dir: Directory to save models and logs
            data_generator: Optional DataGenerator for automatic data loading
        """
        self.config = config
        self.output_dir = output_dir
        self.data_generator = data_generator
        self.model = build_model(config)
        
        # Create output directory
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def get_callbacks(self, use_tensorboard: bool = False) -> list:
        """Get training callbacks.
        
        Args:
            use_tensorboard: Whether to enable TensorBoard logging
            
        Returns:
            List of Keras callbacks
        """
        callbacks = []
        
        # Model checkpoint
        checkpoint_path = f"{self.output_dir}/best_model.h5"
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        )
        
        # Early stopping
        if self.config.model.use_early_stopping:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.model.early_stopping_patience,
                    restore_best_weights=True,
                    verbose=1
                )
            )
        
        # Reduce learning rate on plateau
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                verbose=1,
                min_lr=1e-7
            )
        )
        
        # TensorBoard
        if use_tensorboard:
            callbacks.append(
                keras.callbacks.TensorBoard(
                    log_dir=f"{self.output_dir}/logs",
                    histogram_freq=1,
                    write_graph=True
                )
            )
        
        # CSV logger
        callbacks.append(
            keras.callbacks.CSVLogger(
                f"{self.output_dir}/training_history.csv"
            )
        )
        
        return callbacks
    
    def train(self, 
              data_path: Optional[str] = None,
              train_ratio: float = 0.1,
              val_ratio: float = 0.5,
              use_tensorboard: bool = False,
              choosy: bool = False):
        """Train the model.
        
        Args:
            data_path: Path to training data (.mat file). If None, uses data_generator.
            train_ratio: Ratio of training data to use
            val_ratio: Ratio of validation data to use
            use_tensorboard: Whether to enable TensorBoard
            choosy: Whether to only use fault locations for training
            
        Returns:
            Training history
        """
        # If data_generator is provided, use it
        if self.data_generator is not None:
            print("Using DataGenerator for training...")
            train_ds = self.data_generator.create_training_dataset(
                ratio=train_ratio,
                choosy=choosy,
                shuffle=True,
                cache=False
            )
            val_ds = self.data_generator.create_validation_dataset(
                ratio=val_ratio,
                cache=True
            )
            
            # Print dataset info
            info = self.data_generator.get_dataset_info()
            print("\nDataset Information:")
            for key, value in info.items():
                print(f"  {key}: {value}")
            
        elif data_path is not None:
            # Create DataGenerator from data_path
            print(f"Loading data from {data_path}...")
            from data_generator import DataGenerator
            self.data_generator = DataGenerator(self.config, data_path)
            
            train_ds = self.data_generator.create_training_dataset(
                ratio=train_ratio,
                choosy=choosy,
                shuffle=True,
                cache=False
            )
            val_ds = self.data_generator.create_validation_dataset(
                ratio=val_ratio,
                cache=True
            )
            
            # Print dataset info
            info = self.data_generator.get_dataset_info()
            print("\nDataset Information:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        else:
            print("ERROR: No data source provided!")
            print("Please provide either:")
            print("  1. data_path parameter to train() method")
            print("  2. data_generator in ModelTrainer constructor")
            print("\nModel architecture: " + self.config.model.architecture)
            self.model.summary()
            return None
        
        # Get callbacks
        callbacks = self.get_callbacks(use_tensorboard=use_tensorboard)
        
        # Train model
        print(f"\nTraining {self.config.model.architecture} for {self.config.model.epochs} epochs...")
        print(f"Batch size: {self.config.model.batch_size}")
        print(f"Learning rate: {self.config.model.learning_rate}")
        
        if self.config.augmentation.enable_rotation:
            print(f"Rotation augmentation: ENABLED (p={self.config.augmentation.rotation_probability})")
        if self.config.augmentation.enable_flipping:
            print(f"Flipping augmentation: ENABLED")
        
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config.model.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        final_model_path = f"{self.output_dir}/final_model.h5"
        self.model.save(final_model_path)
        print(f"\nFinal model saved to: {final_model_path}")
        
        return history
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model weights from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        self.model.load_weights(checkpoint_path)


class ModelPredictor:
    """Wrapper class for model prediction with post-processing.
    
    This class handles the full prediction pipeline:
    1. Load model and run predictions
    2. Generate probability maps
    3. Apply post-processing (clustering, line fitting)
    4. Save results and visualizations
    """
    
    def __init__(self, config: Config, model_path: str):
        """Initialize predictor.
        
        Args:
            config: Configuration object
            model_path: Path to trained model
        """
        self.config = config
        self.model = keras.models.load_model(model_path)
    
    def predict(self, data_path: str, output_dir: str, 
                visualize: bool = False):
        """Run prediction on data.
        
        Args:
            data_path: Path to input data
            output_dir: Directory to save results
            visualize: Whether to generate visualizations
            
        Returns:
            Dictionary with prediction results:
            - 'probability_map': Raw probability map
            - 'cluster_map': Cluster assignments (if clustering enabled)
            - 'lineaments': Extracted lineaments
            - 'statistics': Clustering statistics
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("Prediction not yet fully implemented - requires data loading")
        print("However, post-processing pipeline is ready:")
        print(f"  - Clustering: {self.config.inference.use_clustering}")
        print(f"  - Method: {self.config.inference.clustering_method}")
        print(f"  - Line fitting: {self.config.inference.line_fitting_method}")
        
        return {
            'probability_map': None,
            'cluster_map': None,
            'lineaments': [],
            'statistics': {}
        }
    
    def predict_and_postprocess(self, probability_map: np.ndarray,
                                output_dir: str,
                                visualize: bool = False):
        """Run post-processing on a probability map.
        
        This method demonstrates the post-processing pipeline on a given
        probability map. Can be used once data loading is implemented.
        
        Args:
            probability_map: Probability map from model predictions (H x W)
            output_dir: Directory to save results
            visualize: Whether to generate visualizations
            
        Returns:
            Dictionary with results:
            - 'probability_map': Input probability map
            - 'cluster_map': Cluster assignments
            - 'lineaments': Extracted lineaments
            - 'statistics': Clustering statistics
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Import post-processing module
        from postprocessing import PostProcessor
        
        # Initialize post-processor
        processor = PostProcessor(self.config.inference)
        
        # Extract lineaments
        print("Running post-processing pipeline...")
        cluster_map, lineaments = processor.extract_lineaments(probability_map)
        stats = processor.get_cluster_statistics(cluster_map)
        
        print(f"Post-processing complete:")
        print(f"  - Clusters found: {stats.get('n_clusters', 0)}")
        print(f"  - Lineaments extracted: {len(lineaments)}")
        
        # Save results
        np.save(os.path.join(output_dir, 'probability_map.npy'), probability_map)
        np.save(os.path.join(output_dir, 'cluster_map.npy'), cluster_map)
        
        # Save lineaments as JSON
        import json
        lineaments_json = []
        for lineament in lineaments:
            lineaments_json.append({
                'cluster_id': lineament['cluster_id'],
                'type': lineament['type'],
                'points': lineament['points'].tolist()
            })
        
        with open(os.path.join(output_dir, 'lineaments.json'), 'w') as f:
            json.dump(lineaments_json, f, indent=2)
        
        # Save statistics
        with open(os.path.join(output_dir, 'statistics.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        
        if visualize:
            self._visualize_results(probability_map, cluster_map, lineaments, output_dir)
        
        return {
            'probability_map': probability_map,
            'cluster_map': cluster_map,
            'lineaments': lineaments,
            'statistics': stats
        }
    
    def _visualize_results(self, probability_map: np.ndarray,
                          cluster_map: np.ndarray,
                          lineaments: list,
                          output_dir: str):
        """Generate visualizations of results.
        
        Args:
            probability_map: Input probability map
            cluster_map: Cluster assignments
            lineaments: Extracted lineaments
            output_dir: Directory to save visualizations
        """
        try:
            import matplotlib.pyplot as plt
            
            # Create figure with subplots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Plot probability map
            axes[0].imshow(probability_map, cmap='hot')
            axes[0].set_title('Probability Map')
            axes[0].axis('off')
            
            # Plot clusters
            axes[1].imshow(cluster_map, cmap='tab20')
            axes[1].set_title(f'Clusters (n={len(np.unique(cluster_map)) - 1})')
            axes[1].axis('off')
            
            # Plot lineaments
            axes[2].imshow(probability_map, cmap='gray')
            for lineament in lineaments:
                points = lineament['points']
                axes[2].plot(points[:, 1], points[:, 0], 'r-', linewidth=2)
            axes[2].set_title(f'Lineaments (n={len(lineaments)})')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/results_visualization.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Visualization saved to {output_dir}/results_visualization.png")
        except Exception as e:
            print(f"Warning: Could not generate visualization: {e}")
