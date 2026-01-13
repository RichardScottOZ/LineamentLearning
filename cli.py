"""
Command-line interface for LineamentLearning.

This module provides a modern CLI for training and inference operations.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from config import Config, get_config


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        description='LineamentLearning: Deep Learning for Geoscience Lineament Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--config', type=str, help='Path to configuration file')
    train_parser.add_argument('--data', type=str, required=True, help='Path to training data (.mat file)')
    train_parser.add_argument('--output', type=str, default='./models', help='Output directory for models')
    train_parser.add_argument('--window-size', type=int, help='Window size for patches')
    train_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, help='Batch size')
    train_parser.add_argument('--architecture', type=str, choices=['RotateNet', 'UNet', 'ResNet'],
                            help='Model architecture')
    train_parser.add_argument('--train-ratio', type=float, default=0.1,
                            help='Ratio of training data to use (0.0 to 1.0)')
    train_parser.add_argument('--val-ratio', type=float, default=0.5,
                            help='Ratio of validation data to use (0.0 to 1.0)')
    train_parser.add_argument('--choosy', action='store_true',
                            help='Only use fault locations for training')
    
    # Augmentation options
    train_parser.add_argument('--enable-rotation', action='store_true',
                            help='Enable rotation augmentation')
    train_parser.add_argument('--rotation-prob', type=float, default=0.5,
                            help='Probability of applying rotation (0.0 to 1.0)')
    train_parser.add_argument('--enable-flipping', action='store_true',
                            help='Enable flipping augmentation')
    
    train_parser.add_argument('--resume', type=str, help='Resume training from checkpoint')
    train_parser.add_argument('--tensorboard', action='store_true', help='Enable TensorBoard logging')
    train_parser.add_argument('--gpu', type=int, help='GPU device ID to use')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Run prediction on data')
    predict_parser.add_argument('--config', type=str, help='Path to configuration file')
    predict_parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    predict_parser.add_argument('--data', type=str, required=True, help='Path to input data')
    predict_parser.add_argument('--output', type=str, required=True, help='Output directory for results')
    predict_parser.add_argument('--threshold', type=float, help='Probability threshold')
    predict_parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    predict_parser.add_argument('--batch-size', type=int, help='Batch size for inference')
    predict_parser.add_argument('--gpu', type=int, help='GPU device ID to use')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model performance')
    eval_parser.add_argument('--config', type=str, help='Path to configuration file')
    eval_parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    eval_parser.add_argument('--data', type=str, required=True, help='Path to test data')
    eval_parser.add_argument('--output', type=str, default='./evaluation', help='Output directory')
    eval_parser.add_argument('--metrics', type=str, nargs='+', 
                           default=['accuracy', 'precision', 'recall', 'f1'],
                           help='Metrics to compute')
    
    # Convert command (legacy to modern format)
    convert_parser = subparsers.add_parser('convert', help='Convert legacy models/data')
    convert_parser.add_argument('--input', type=str, required=True, help='Input file/directory')
    convert_parser.add_argument('--output', type=str, required=True, help='Output file/directory')
    convert_parser.add_argument('--format', type=str, choices=['model', 'data'], 
                              required=True, help='What to convert')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export model for deployment')
    export_parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    export_parser.add_argument('--output', type=str, required=True, help='Output path')
    export_parser.add_argument('--format', type=str, choices=['onnx', 'tflite', 'savedmodel'],
                             default='savedmodel', help='Export format')
    
    return parser


def train_command(args: argparse.Namespace) -> int:
    """Execute training command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success)
    """
    print("=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    # Load configuration
    config = get_config(args.config)
    
    # Override config with command-line arguments
    if args.window_size:
        config.model.window_size = args.window_size
    if args.epochs:
        config.model.epochs = args.epochs
    if args.batch_size:
        config.model.batch_size = args.batch_size
    if args.architecture:
        config.model.architecture = args.architecture
    
    # Augmentation settings
    if args.enable_rotation:
        config.augmentation.enable_rotation = True
        config.augmentation.rotation_probability = args.rotation_prob
    if args.enable_flipping:
        config.augmentation.enable_flipping = True
    
    # Set device
    if args.gpu is not None:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    print(f"Configuration:")
    print(f"  Data: {args.data}")
    print(f"  Output: {args.output}")
    print(f"  Architecture: {config.model.architecture}")
    print(f"  Window Size: {config.model.window_size}")
    print(f"  Epochs: {config.model.epochs}")
    print(f"  Batch Size: {config.model.batch_size}")
    print(f"  Train Ratio: {args.train_ratio}")
    print(f"  Val Ratio: {args.val_ratio}")
    
    if config.augmentation.enable_rotation:
        print(f"  Rotation: ENABLED (p={config.augmentation.rotation_probability})")
    if config.augmentation.enable_flipping:
        print(f"  Flipping: ENABLED")
    
    # Import here to avoid loading TensorFlow unnecessarily
    try:
        from model_modern import ModelTrainer
        
        trainer = ModelTrainer(config, args.output)
        
        if args.resume:
            print(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        # Train model with new integrated data loading
        trainer.train(
            data_path=args.data,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            use_tensorboard=args.tensorboard,
            choosy=args.choosy
        )
        
        print("\nTraining completed successfully!")
        print(f"Model saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"Error during training: {e}", file=sys.stderr)
        if config.debug_mode:
            raise
        return 1


def predict_command(args: argparse.Namespace) -> int:
    """Execute prediction command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success)
    """
    print("=" * 60)
    print("Starting Prediction")
    print("=" * 60)
    
    # Load configuration
    config = get_config(args.config)
    
    # Override config
    if args.threshold:
        config.inference.threshold = args.threshold
    if args.batch_size:
        config.model.batch_size = args.batch_size
    
    # Set device
    if args.gpu is not None:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Input: {args.data}")
    print(f"  Output: {args.output}")
    print(f"  Threshold: {config.inference.threshold}")
    
    try:
        from model_modern import ModelPredictor
        
        predictor = ModelPredictor(config, args.model)
        
        # Run prediction
        results = predictor.predict(
            data_path=args.data,
            output_dir=args.output,
            visualize=args.visualize
        )
        
        print("\nPrediction completed successfully!")
        print(f"Results saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        if config.debug_mode:
            raise
        return 1


def main():
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute appropriate command
    if args.command == 'train':
        return train_command(args)
    elif args.command == 'predict':
        return predict_command(args)
    elif args.command == 'evaluate':
        print("Evaluate command not yet implemented")
        return 1
    elif args.command == 'convert':
        print("Convert command not yet implemented")
        return 1
    elif args.command == 'export':
        print("Export command not yet implemented")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
