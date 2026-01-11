"""Simple training example for LineamentLearning."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from model_modern import build_model, ModelTrainer


def main():
    parser = argparse.ArgumentParser(description='Train a lineament detection model')
    parser.add_argument('--architecture', default='RotateNet', 
                       choices=['RotateNet', 'UNet', 'ResNet'])
    parser.add_argument('--window-size', type=int, default=45)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--output', default='./outputs/trained_model')
    args = parser.parse_args()
    
    print("LineamentLearning - Training Example")
    print(f"Architecture: {args.architecture}")
    
    config = Config()
    config.model.architecture = args.architecture
    config.model.window_size = args.window_size
    config.model.epochs = args.epochs
    
    model = build_model(config)
    model.summary()
    
    print(f"\nModel would be saved to: {args.output}")
    print("Note: Actual training requires data loading implementation")

if __name__ == '__main__':
    import tensorflow as tf
    main()
