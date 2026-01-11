"""Simple prediction example for LineamentLearning."""
import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from model_modern import build_model


def main():
    parser = argparse.ArgumentParser(description='Run predictions')
    parser.add_argument('--model', help='Path to trained model')
    parser.add_argument('--window-size', type=int, default=45)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    
    print("LineamentLearning - Prediction Example")
    
    config = Config()
    config.model.window_size = args.window_size
    
    # Create demo model if no model provided
    if args.model:
        print(f"Loading model from: {args.model}")
        import tensorflow as tf
        try:
            model = tf.keras.models.load_model(args.model)
        except:
            print("Error loading model, creating demo model")
            model = build_model(config)
    else:
        print("Creating demo model")
        model = build_model(config)
    
    # Create dummy data
    test_data = np.random.randn(5, args.window_size, args.window_size, 8).astype(np.float32)
    predictions = model.predict(test_data, verbose=0)
    
    print(f"\nPredictions: {predictions.shape}")
    print(f"Mean: {predictions.mean():.4f}")
    print(f"Detections (>{args.threshold}): {(predictions >= args.threshold).sum()}")

if __name__ == '__main__':
    import tensorflow as tf
    main()
