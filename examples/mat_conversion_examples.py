#!/usr/bin/env python3
"""
Examples: MATLAB .mat to PyData Format Conversion

This script demonstrates various ways to convert LineamentLearning .mat files
to PyData formats (NumPy, HDF5, Zarr) for use in Python workflows.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import scipy.io as sio


def example_1_inspect_mat_file():
    """Example 1: Inspect a .mat file to understand its structure."""
    print("=" * 70)
    print("Example 1: Inspecting .mat File Structure")
    print("=" * 70)
    
    # This is a demonstration - replace with your actual file path
    mat_path = "./Dataset/Australia/Rotations/Australia_strip.mat"
    
    print(f"\nDataset path: {mat_path}")
    print("\nTo inspect a .mat file:")
    print("  1. Use the mat_converter tool")
    print("  2. Or use scipy.io.loadmat directly")
    
    print("\n# Method 1: Using mat_converter (recommended)")
    print("from mat_converter import inspect_mat_file")
    print(f"inspect_mat_file('{mat_path}')")
    
    print("\n# Method 2: Using scipy directly")
    print("import scipy.io as sio")
    print(f"mat_data = sio.loadmat('{mat_path}')")
    print("for key, value in mat_data.items():")
    print("    if not key.startswith('__'):")
    print("        print(f'{key}: shape={value.shape}, dtype={value.dtype}')")
    
    print("\nExpected output:")
    print("  I1: shape=(2000, 2000), dtype=float64")
    print("  I2: shape=(2000, 2000), dtype=float64")
    print("  ...")
    print("  mask: shape=(2000, 2000), dtype=float64")
    print("  train_mask: shape=(2000, 2000), dtype=float64")


def example_2_simple_numpy_conversion():
    """Example 2: Simple conversion to NumPy .npz format."""
    print("\n" + "=" * 70)
    print("Example 2: Simple NumPy Conversion")
    print("=" * 70)
    
    print("\n# Step 1: Load .mat file")
    print("import scipy.io as sio")
    print("mat_data = sio.loadmat('dataset.mat')")
    
    print("\n# Step 2: Save as compressed NumPy archive")
    print("import numpy as np")
    print("np.savez_compressed('dataset.npz', **{")
    print("    key: value for key, value in mat_data.items()")
    print("    if not key.startswith('__')")
    print("})")
    
    print("\n# Step 3: Load back")
    print("data = np.load('dataset.npz')")
    print("I1 = data['I1']")
    print("mask = data['mask']")
    
    print("\nAdvantages:")
    print("  ✓ Simple and fast")
    print("  ✓ Good compression")
    print("  ✓ Native Python support")
    
    print("\nLimitations:")
    print("  - Loads entire file into memory")
    print("  - No chunked access")


def example_3_hdf5_conversion():
    """Example 3: Conversion to HDF5 with compression."""
    print("\n" + "=" * 70)
    print("Example 3: HDF5 Conversion (Recommended)")
    print("=" * 70)
    
    print("\n# Step 1: Load .mat file")
    print("import scipy.io as sio")
    print("import h5py")
    print("mat_data = sio.loadmat('dataset.mat')")
    
    print("\n# Step 2: Create HDF5 file with organized structure")
    print("with h5py.File('dataset.h5', 'w') as f:")
    print("    # Organize data in groups")
    print("    inputs = f.create_group('inputs')")
    print("    masks = f.create_group('masks')")
    print("    labels = f.create_group('labels')")
    print("    ")
    print("    # Save input layers with compression")
    print("    for i in range(1, 9):")
    print("        inputs.create_dataset(")
    print("            f'I{i}',")
    print("            data=mat_data[f'I{i}'],")
    print("            compression='gzip',")
    print("            compression_opts=4,")
    print("            chunks=True")
    print("        )")
    print("    ")
    print("    # Save masks")
    print("    masks.create_dataset('mask', data=mat_data['mask'], compression='gzip')")
    print("    masks.create_dataset('train_mask', data=mat_data['train_mask'], compression='gzip')")
    print("    ")
    print("    # Add metadata")
    print("    f.attrs['source'] = 'LineamentLearning'")
    print("    f.attrs['shape'] = mat_data['I1'].shape")
    
    print("\n# Step 3: Load back (can load specific arrays)")
    print("with h5py.File('dataset.h5', 'r') as f:")
    print("    # Load specific layer")
    print("    I1 = f['inputs/I1'][:]")
    print("    ")
    print("    # Or load just a slice (memory efficient!)")
    print("    I1_subset = f['inputs/I1'][0:1000, 0:1000]")
    
    print("\nAdvantages:")
    print("  ✓ Excellent compression")
    print("  ✓ Chunked/partial loading")
    print("  ✓ Industry standard")
    print("  ✓ Organized structure")
    print("  ✓ Metadata support")


def example_4_using_mat_converter():
    """Example 4: Using the built-in mat_converter module."""
    print("\n" + "=" * 70)
    print("Example 4: Using mat_converter Module (Easiest)")
    print("=" * 70)
    
    print("\n# Import the converter")
    print("from mat_converter import MatConverter")
    
    print("\n# Method 1: Convert to NumPy")
    print("converter = MatConverter()")
    print("converter.convert_to_numpy(")
    print("    mat_path='dataset.mat',")
    print("    output_path='dataset.npz'")
    print(")")
    
    print("\n# Method 2: Convert to HDF5 (recommended)")
    print("converter.convert_to_hdf5(")
    print("    mat_path='dataset.mat',")
    print("    output_path='dataset.h5',")
    print("    compression='gzip',")
    print("    compression_opts=4")
    print(")")
    
    print("\n# Method 3: Generic convert with auto-detection")
    print("converter.convert(")
    print("    input_path='dataset.mat',")
    print("    output_path='dataset.h5',")
    print("    format='hdf5'")
    print(")")
    
    print("\nAdvantages:")
    print("  ✓ Handles edge cases automatically")
    print("  ✓ Validates input/output")
    print("  ✓ Organized HDF5 structure")
    print("  ✓ Progress reporting")


def example_5_command_line_tool():
    """Example 5: Using the command-line tool."""
    print("\n" + "=" * 70)
    print("Example 5: Command-Line Tool")
    print("=" * 70)
    
    print("\n# Inspect a .mat file")
    print("python -m mat_converter --inspect dataset.mat")
    
    print("\n# Convert to HDF5 (default)")
    print("python -m mat_converter dataset.mat dataset.h5")
    
    print("\n# Convert to NumPy")
    print("python -m mat_converter --format numpy dataset.mat dataset.npz")
    
    print("\n# Batch convert all .mat files in a directory")
    print("python -m mat_converter --batch \\")
    print("    --input-dir Dataset/Australia/Rotations/ \\")
    print("    --output-dir Dataset/Converted/ \\")
    print("    --format hdf5 \\")
    print("    --compression gzip \\")
    print("    --compression-level 4")
    
    print("\n# Validate conversion")
    print("python -m mat_converter --validate dataset.mat dataset.h5")
    
    print("\nTip: Use --help for full documentation:")
    print("python -m mat_converter --help")


def example_6_use_with_lineamentlearning():
    """Example 6: Using converted data with LineamentLearning."""
    print("\n" + "=" * 70)
    print("Example 6: Using Converted Data with LineamentLearning")
    print("=" * 70)
    
    print("\nOnce converted, you can use the data with LineamentLearning:")
    
    print("\n# Option 1: Direct loading with NumPy")
    print("import numpy as np")
    print("from config import Config")
    print("from model_modern import ModelTrainer")
    print("")
    print("# Load converted data")
    print("data = np.load('dataset.npz')")
    print("# ... process and train manually")
    
    print("\n# Option 2: Using modified DATASET class (supports multiple formats)")
    print("from DATASET import DATASET")
    print("")
    print("# Load from HDF5")
    print("dataset = DATASET('dataset.h5', file_format='hdf5')")
    print("")
    print("# Load from NumPy")
    print("dataset = DATASET('dataset.npz', file_format='numpy')")
    print("")
    print("# Use as normal")
    print("X, Y, IDX = dataset.generateDS(")
    print("    output=dataset.OUTPUT,")
    print("    mask=dataset.trainMask,")
    print("    w=45")
    print(")")
    
    print("\n# Option 3: Using DataGenerator")
    print("from data_generator import DataGenerator")
    print("")
    print("config = Config()")
    print("data_gen = DataGenerator(")
    print("    config=config,")
    print("    dataset_path='dataset.h5',")
    print("    file_format='hdf5'  # or 'numpy'")
    print(")")
    print("")
    print("trainer = ModelTrainer(config, data_generator=data_gen)")
    print("history = trainer.train(train_ratio=0.1)")
    
    print("\n# Option 4: Command-line interface")
    print("lineament-train \\")
    print("    --data dataset.h5 \\")
    print("    --format hdf5 \\")
    print("    --output ./models \\")
    print("    --epochs 50")


def example_7_complete_workflow():
    """Example 7: Complete conversion and training workflow."""
    print("\n" + "=" * 70)
    print("Example 7: Complete Workflow")
    print("=" * 70)
    
    print("\nComplete workflow from .mat to trained model:")
    
    print("\n# Step 1: Inspect original .mat file")
    print("python -m mat_converter --inspect dataset.mat")
    
    print("\n# Step 2: Convert to HDF5")
    print("python -m mat_converter dataset.mat dataset.h5 --format hdf5")
    
    print("\n# Step 3: Validate conversion")
    print("python -m mat_converter --validate dataset.mat dataset.h5")
    
    print("\n# Step 4: Train model with converted data")
    print("lineament-train \\")
    print("    --data dataset.h5 \\")
    print("    --format hdf5 \\")
    print("    --output ./models \\")
    print("    --architecture UNet \\")
    print("    --epochs 50 \\")
    print("    --tensorboard")
    
    print("\n# Or using Python API:")
    print("from config import Config")
    print("from data_generator import DataGenerator")
    print("from model_modern import ModelTrainer")
    print("")
    print("config = Config()")
    print("config.model.architecture = 'UNet'")
    print("config.model.epochs = 50")
    print("")
    print("data_gen = DataGenerator(config, 'dataset.h5', file_format='hdf5')")
    print("trainer = ModelTrainer(config, output_dir='./models', data_generator=data_gen)")
    print("history = trainer.train(train_ratio=0.1, val_ratio=0.5)")


def example_8_advanced_hdf5():
    """Example 8: Advanced HDF5 usage."""
    print("\n" + "=" * 70)
    print("Example 8: Advanced HDF5 Usage")
    print("=" * 70)
    
    print("\n# Memory-efficient loading of large datasets")
    print("import h5py")
    print("import numpy as np")
    print("")
    print("with h5py.File('large_dataset.h5', 'r') as f:")
    print("    # Get dataset info without loading")
    print("    shape = f['inputs/I1'].shape")
    print("    print(f'Dataset shape: {shape}')")
    print("    ")
    print("    # Load only a region of interest")
    print("    roi = f['inputs/I1'][1000:2000, 1000:2000]")
    print("    ")
    print("    # Iterate through chunks")
    print("    chunk_size = 500")
    print("    for i in range(0, shape[0], chunk_size):")
    print("        for j in range(0, shape[1], chunk_size):")
    print("            chunk = f['inputs/I1'][i:i+chunk_size, j:j+chunk_size]")
    print("            # Process chunk...")
    print("    ")
    print("    # Access metadata")
    print("    source = f.attrs['source']")
    print("    original_shape = f.attrs['shape']")
    
    print("\nTip: HDF5 allows memory-mapped access without loading entire file!")


def main():
    """Run all examples."""
    print("\n")
    print("=" * 70)
    print("LineamentLearning - MATLAB to PyData Conversion Examples")
    print("=" * 70)
    print("\nThese examples demonstrate converting .mat files to PyData formats")
    print("for better performance and integration with Python workflows.")
    print("\n")
    
    example_1_inspect_mat_file()
    example_2_simple_numpy_conversion()
    example_3_hdf5_conversion()
    example_4_using_mat_converter()
    example_5_command_line_tool()
    example_6_use_with_lineamentlearning()
    example_7_complete_workflow()
    example_8_advanced_hdf5()
    
    print("\n" + "=" * 70)
    print("Examples Complete")
    print("=" * 70)
    print("\nFor more information:")
    print("  - See MAT_TO_PYDATA_GUIDE.md for detailed documentation")
    print("  - Run: python -m mat_converter --help")
    print("  - Check examples in this file")
    print("\nQuick start:")
    print("  python -m mat_converter --inspect your_dataset.mat")
    print("  python -m mat_converter your_dataset.mat your_dataset.h5")
    print("\n")


if __name__ == '__main__':
    main()
