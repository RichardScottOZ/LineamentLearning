"""Configuration examples for LineamentLearning."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config


def main():
    print("LineamentLearning - Configuration Examples\n")
    
    # Example 1: Default config
    print("1. Default Configuration")
    config = Config()
    print(f"   Architecture: {config.model.architecture}")
    print(f"   Window Size: {config.model.window_size}\n")
    
    # Example 2: Custom config
    print("2. Custom Configuration")
    config = Config()
    config.model.architecture = 'UNet'
    config.model.window_size = 64
    config.model.use_dropout = True
    print(f"   Architecture: {config.model.architecture}")
    print(f"   Window Size: {config.model.window_size}")
    print(f"   Dropout: {config.model.use_dropout}\n")
    
    # Example 3: Save and load
    print("3. Save and Load Configuration")
    config_path = './outputs/demo_config.json'
    Path('./outputs').mkdir(exist_ok=True)
    config.to_file(config_path)
    print(f"   Saved to: {config_path}")
    
    loaded = Config.from_file(config_path)
    print(f"   Loaded: {loaded.model.architecture}\n")
    
    # Example 4: Validation
    print("4. Configuration Validation")
    try:
        config.validate()
        print("   ✓ Configuration is valid\n")
    except ValueError as e:
        print(f"   ✗ Error: {e}\n")
    
    print("All examples completed!")

if __name__ == '__main__':
    main()
