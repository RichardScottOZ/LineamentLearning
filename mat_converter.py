#!/usr/bin/env python3
"""
MATLAB .mat to PyData Format Converter

This module provides utilities to convert LineamentLearning .mat files
to various PyData formats (NumPy, HDF5, Zarr).

Note: While the companion documentation (MAT_TO_PYDATA_GUIDE.md) discusses
Parquet format for reference, this module currently focuses on array-based
formats (NumPy, HDF5, Zarr) which are most suitable for spatial data.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
import warnings

import numpy as np
import scipy.io as sio


class MatConverter:
    """Converter for MATLAB .mat files to PyData formats."""
    
    # Expected fields in LineamentLearning .mat files
    INPUT_LAYERS = [f'I{i}' for i in range(1, 9)]
    REQUIRED_FIELDS = INPUT_LAYERS + ['mask', 'train_mask', 'DEGREES']
    OPTIONAL_FIELDS = ['test_mask', 'output', 'R2M', 'M2R']
    FILTER_FIELDS = ['filters', 'rotations']
    
    def __init__(self, verbose: bool = True):
        """Initialize converter.
        
        Args:
            verbose: Print progress messages
        """
        self.verbose = verbose
    
    def log(self, message: str):
        """Print message if verbose mode enabled."""
        if self.verbose:
            print(message)
    
    def load_mat_file(self, mat_path: Union[str, Path]) -> Dict[str, np.ndarray]:
        """Load .mat file, handling both old and new formats.
        
        Args:
            mat_path: Path to .mat file
            
        Returns:
            Dictionary of arrays from .mat file
            
        Raises:
            ValueError: If file cannot be loaded
        """
        mat_path = Path(mat_path)
        
        if not mat_path.exists():
            raise FileNotFoundError(f"File not found: {mat_path}")
        
        self.log(f"Loading {mat_path}...")
        
        try:
            # Try loading with scipy (works for MATLAB v7 and earlier)
            mat_data = sio.loadmat(str(mat_path))
            self.log(f"  Loaded with scipy.io.loadmat (MATLAB v7 format)")
            
        except NotImplementedError:
            # MATLAB v7.3 files are HDF5 format
            self.log(f"  Detected MATLAB v7.3 format (HDF5)")
            try:
                import h5py
            except ImportError:
                raise ImportError(
                    "h5py required for MATLAB v7.3 files. Install with: pip install h5py"
                )
            
            mat_data = {}
            with h5py.File(mat_path, 'r') as f:
                for key in f.keys():
                    if not key.startswith('#'):
                        data = np.array(f[key])
                        # MATLAB v7.3 arrays may need transposing
                        if data.ndim == 2:
                            data = data.T
                        mat_data[key] = data
                        
            self.log(f"  Loaded with h5py (MATLAB v7.3 format)")
        
        # Filter out metadata fields
        mat_data = {
            k: v for k, v in mat_data.items()
            if not k.startswith('__')
        }
        
        self.log(f"  Found {len(mat_data)} fields")
        return mat_data
    
    def inspect(self, mat_path: Union[str, Path]) -> Dict:
        """Inspect .mat file structure and contents.
        
        Args:
            mat_path: Path to .mat file
            
        Returns:
            Dictionary with file information
        """
        mat_path = Path(mat_path)
        mat_data = self.load_mat_file(mat_path)
        
        file_size_mb = mat_path.stat().st_size / (1024**2)
        
        info = {
            'path': str(mat_path),
            'size_mb': file_size_mb,
            'fields': {},
            'is_dataset': False,
            'is_filter': False,
            'missing_required': [],
            'available_optional': []
        }
        
        # Analyze fields
        for key, value in mat_data.items():
            field_info = {
                'shape': value.shape,
                'dtype': str(value.dtype),
                'size_mb': value.nbytes / (1024**2),
                'min': float(np.min(value)) if value.size > 0 else None,
                'max': float(np.max(value)) if value.size > 0 else None,
                'mean': float(np.mean(value)) if value.size > 0 else None,
            }
            info['fields'][key] = field_info
        
        # Check if it's a dataset file
        required_present = sum(1 for f in self.REQUIRED_FIELDS if f in mat_data)
        if required_present >= len(self.INPUT_LAYERS) + 1:  # At least inputs + mask
            info['is_dataset'] = True
            info['missing_required'] = [f for f in self.REQUIRED_FIELDS if f not in mat_data]
            info['available_optional'] = [f for f in self.OPTIONAL_FIELDS if f in mat_data]
        
        # Check if it's a filter file
        if all(f in mat_data for f in self.FILTER_FIELDS):
            info['is_filter'] = True
        
        return info
    
    def print_inspection(self, mat_path: Union[str, Path]):
        """Print human-readable inspection of .mat file."""
        info = self.inspect(mat_path)
        
        print("=" * 70)
        print(f"MATLAB File Inspection: {info['path']}")
        print("=" * 70)
        print(f"\nFile size: {info['size_mb']:.2f} MB")
        print(f"\nFile type:")
        if info['is_dataset']:
            print("  ✓ LineamentLearning Dataset")
            if info['missing_required']:
                print(f"  ⚠ Missing required fields: {', '.join(info['missing_required'])}")
            if info['available_optional']:
                print(f"  ✓ Optional fields present: {', '.join(info['available_optional'])}")
        elif info['is_filter']:
            print("  ✓ Filter/Rotation File")
        else:
            print("  ? Unknown format")
        
        print(f"\nFields ({len(info['fields'])}):")
        print(f"{'Field':<15} {'Shape':<20} {'Dtype':<10} {'Size (MB)':<12} {'Range':<30}")
        print("-" * 90)
        
        for key, field in sorted(info['fields'].items()):
            shape_str = str(field['shape'])
            size_str = f"{field['size_mb']:.2f}"
            
            if field['min'] is not None and field['max'] is not None:
                range_str = f"[{field['min']:.3e}, {field['max']:.3e}]"
            else:
                range_str = "N/A"
            
            print(f"{key:<15} {shape_str:<20} {field['dtype']:<10} {size_str:<12} {range_str:<30}")
        
        print("\n" + "=" * 70)
    
    def convert_to_numpy(self,
                        mat_path: Union[str, Path],
                        output_path: Union[str, Path],
                        compress: bool = True) -> Path:
        """Convert .mat file to NumPy .npz format.
        
        Args:
            mat_path: Input .mat file path
            output_path: Output .npz file path
            compress: Use compression (recommended)
            
        Returns:
            Path to created file
        """
        mat_path = Path(mat_path)
        output_path = Path(output_path)
        
        # Load data
        mat_data = self.load_mat_file(mat_path)
        
        # Save as NumPy
        self.log(f"Saving to {output_path}...")
        if compress:
            np.savez_compressed(output_path, **mat_data)
        else:
            np.savez(output_path, **mat_data)
        
        # Report results
        original_size = mat_path.stat().st_size / (1024**2)
        converted_size = output_path.stat().st_size / (1024**2)
        ratio = original_size / converted_size if converted_size > 0 else 0
        
        self.log(f"Conversion complete!")
        self.log(f"  Original size: {original_size:.2f} MB")
        self.log(f"  Converted size: {converted_size:.2f} MB")
        self.log(f"  Compression ratio: {ratio:.2f}x")
        
        return output_path
    
    def convert_to_hdf5(self,
                       mat_path: Union[str, Path],
                       output_path: Union[str, Path],
                       compression: str = 'gzip',
                       compression_opts: int = 4,
                       chunks: bool = True) -> Path:
        """Convert .mat file to HDF5 format.
        
        Args:
            mat_path: Input .mat file path
            output_path: Output .h5 file path
            compression: Compression algorithm ('gzip', 'lzf', or None)
            compression_opts: Compression level (0-9 for gzip)
            chunks: Enable chunking for better partial access
            
        Returns:
            Path to created file
        """
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required. Install with: pip install h5py")
        
        mat_path = Path(mat_path)
        output_path = Path(output_path)
        
        # Load data
        mat_data = self.load_mat_file(mat_path)
        
        # Inspect to determine file type
        info = self.inspect(mat_path)
        
        # Save as HDF5
        self.log(f"Creating HDF5 file: {output_path}...")
        with h5py.File(output_path, 'w') as f:
            if info['is_dataset']:
                # Organize dataset files with groups
                self._save_dataset_hdf5(f, mat_data, compression, compression_opts, chunks)
            elif info['is_filter']:
                # Save filter files directly
                self._save_filter_hdf5(f, mat_data, compression, compression_opts, chunks)
            else:
                # Save all fields at root level
                self._save_flat_hdf5(f, mat_data, compression, compression_opts, chunks)
            
            # Add metadata
            f.attrs['source_file'] = str(mat_path)
            f.attrs['original_format'] = '.mat file'
            f.attrs['converter'] = 'LineamentLearning mat_converter'
        
        # Report results
        original_size = mat_path.stat().st_size / (1024**2)
        converted_size = output_path.stat().st_size / (1024**2)
        ratio = original_size / converted_size if converted_size > 0 else 0
        
        self.log(f"Conversion complete!")
        self.log(f"  Original size: {original_size:.2f} MB")
        self.log(f"  Converted size: {converted_size:.2f} MB")
        self.log(f"  Compression ratio: {ratio:.2f}x")
        
        return output_path
    
    def _save_dataset_hdf5(self, f, mat_data, compression, compression_opts, chunks):
        """Save dataset .mat as organized HDF5."""
        # Input layers
        inputs_group = f.create_group('inputs')
        for i in range(1, 9):
            key = f'I{i}'
            if key in mat_data:
                inputs_group.create_dataset(
                    key,
                    data=mat_data[key],
                    compression=compression,
                    compression_opts=compression_opts,
                    chunks=chunks
                )
        
        # Masks
        masks_group = f.create_group('masks')
        for key in ['mask', 'train_mask', 'test_mask']:
            if key in mat_data:
                masks_group.create_dataset(
                    key,
                    data=mat_data[key],
                    compression=compression,
                    compression_opts=compression_opts,
                    chunks=chunks
                )
        
        # Labels and other data
        labels_group = f.create_group('labels')
        for key in ['output', 'DEGREES', 'R2M', 'M2R']:
            if key in mat_data:
                labels_group.create_dataset(
                    key,
                    data=mat_data[key],
                    compression=compression,
                    compression_opts=compression_opts,
                    chunks=chunks
                )
        
        # Add metadata
        if 'I1' in mat_data:
            f.attrs['shape'] = mat_data['I1'].shape
            f.attrs['num_layers'] = sum(1 for i in range(1, 9) if f'I{i}' in mat_data)
    
    def _save_filter_hdf5(self, f, mat_data, compression, compression_opts, chunks):
        """Save filter .mat as HDF5."""
        for key in ['filters', 'rotations']:
            if key in mat_data:
                f.create_dataset(
                    key,
                    data=mat_data[key],
                    compression=compression,
                    compression_opts=compression_opts,
                    chunks=chunks
                )
        
        if 'filters' in mat_data:
            f.attrs['n_filters'] = mat_data['filters'].shape[0]
    
    def _save_flat_hdf5(self, f, mat_data, compression, compression_opts, chunks):
        """Save all fields at root level."""
        for key, value in mat_data.items():
            f.create_dataset(
                key,
                data=value,
                compression=compression,
                compression_opts=compression_opts,
                chunks=chunks
            )
    
    def convert(self,
               input_path: Union[str, Path],
               output_path: Union[str, Path],
               format: str = 'hdf5',
               **kwargs) -> Path:
        """Convert .mat file to specified format.
        
        Args:
            input_path: Input .mat file
            output_path: Output file path
            format: Output format ('numpy', 'hdf5', 'zarr')
            **kwargs: Format-specific options
            
        Returns:
            Path to converted file
        """
        format = format.lower()
        
        if format in ['numpy', 'npz']:
            # Filter kwargs for numpy conversion
            numpy_kwargs = {k: v for k, v in kwargs.items() if k in ['compress']}
            return self.convert_to_numpy(input_path, output_path, **numpy_kwargs)
        elif format in ['hdf5', 'h5']:
            # Filter kwargs for hdf5 conversion
            hdf5_kwargs = {k: v for k, v in kwargs.items() 
                          if k in ['compression', 'compression_opts', 'chunks']}
            return self.convert_to_hdf5(input_path, output_path, **hdf5_kwargs)
        elif format == 'zarr':
            # Filter kwargs for zarr conversion
            zarr_kwargs = {k: v for k, v in kwargs.items() if k in ['chunks', 'compressor']}
            return self.convert_to_zarr(input_path, output_path, **zarr_kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def convert_to_zarr(self,
                       mat_path: Union[str, Path],
                       output_path: Union[str, Path],
                       chunks: Tuple[int, int] = (500, 500),
                       compressor: str = 'zstd') -> Path:
        """Convert .mat file to Zarr format.
        
        Args:
            mat_path: Input .mat file path
            output_path: Output .zarr directory path
            chunks: Chunk size for arrays
            compressor: Compression algorithm ('zstd', 'blosc', etc.)
            
        Returns:
            Path to created directory
        """
        try:
            import zarr
        except ImportError:
            raise ImportError("zarr required. Install with: pip install zarr")
        
        mat_path = Path(mat_path)
        output_path = Path(output_path)
        
        # Load data
        mat_data = self.load_mat_file(mat_path)
        
        # Create Zarr store
        self.log(f"Creating Zarr store: {output_path}...")
        store = zarr.DirectoryStore(str(output_path))
        root = zarr.group(store=store, overwrite=True)
        
        # Determine compressor
        if compressor == 'zstd':
            comp = zarr.Blosc(cname='zstd', clevel=3)
        elif compressor == 'blosc':
            comp = zarr.Blosc(cname='lz4', clevel=5)
        else:
            comp = None
        
        # Inspect to determine organization
        info = self.inspect(mat_path)
        
        if info['is_dataset']:
            # Organize dataset files
            inputs = root.create_group('inputs')
            for i in range(1, 9):
                key = f'I{i}'
                if key in mat_data:
                    inputs.array(key, mat_data[key], chunks=chunks, compressor=comp)
            
            masks = root.create_group('masks')
            for key in ['mask', 'train_mask', 'test_mask']:
                if key in mat_data:
                    masks.array(key, mat_data[key], chunks=chunks, compressor=comp)
            
            labels = root.create_group('labels')
            for key in ['output', 'DEGREES', 'R2M', 'M2R']:
                if key in mat_data:
                    labels.array(key, mat_data[key], chunks=chunks, compressor=comp)
        else:
            # Save all fields at root
            for key, value in mat_data.items():
                root.array(key, value, chunks=chunks, compressor=comp)
        
        # Add metadata
        root.attrs['source_file'] = str(mat_path)
        root.attrs['original_format'] = '.mat file'
        
        self.log(f"Conversion complete!")
        return output_path
    
    def validate_conversion(self,
                          original_path: Union[str, Path],
                          converted_path: Union[str, Path],
                          format: str = 'hdf5',
                          tolerance: float = 1e-10) -> bool:
        """Validate that conversion preserved data accurately.
        
        Args:
            original_path: Original .mat file
            converted_path: Converted file
            format: Format of converted file
            tolerance: Numerical tolerance for comparison
            
        Returns:
            True if validation passed
        """
        self.log("Validating conversion...")
        
        # Load original
        original_data = self.load_mat_file(original_path)
        
        # Load converted based on format
        if format in ['numpy', 'npz']:
            converted_data = dict(np.load(converted_path))
        elif format in ['hdf5', 'h5']:
            import h5py
            converted_data = {}
            with h5py.File(converted_path, 'r') as f:
                # Recursively load all datasets
                def load_recursive(group):
                    for key in group.keys():
                        if isinstance(group[key], h5py.Dataset):
                            converted_data[key] = np.array(group[key])
                        else:
                            load_recursive(group[key])
                load_recursive(f)
        elif format == 'zarr':
            try:
                import zarr
            except ImportError:
                raise ImportError("zarr required for validation. Install with: pip install zarr")
            
            converted_data = {}
            root = zarr.open(str(converted_path), mode='r')
            
            def load_zarr_recursive(group):
                """Recursively load arrays from Zarr group."""
                for key in group.keys():
                    item = group[key]
                    if isinstance(item, zarr.core.Array):
                        converted_data[key] = np.array(item[:])
                    else:
                        load_zarr_recursive(item)
            
            load_zarr_recursive(root)
        else:
            raise ValueError(f"Validation not implemented for format: {format}")
        
        # Compare all fields
        all_match = True
        for key in original_data.keys():
            if key not in converted_data:
                self.log(f"  ✗ Field '{key}' missing in converted file")
                all_match = False
                continue
            
            original = original_data[key]
            converted = converted_data[key]
            
            if not np.allclose(original, converted, rtol=tolerance, atol=tolerance):
                self.log(f"  ✗ Field '{key}' values don't match")
                max_diff = np.max(np.abs(original - converted))
                self.log(f"    Max difference: {max_diff}")
                all_match = False
            else:
                self.log(f"  ✓ Field '{key}' matches")
        
        if all_match:
            self.log("\n✓ Validation passed!")
        else:
            self.log("\n✗ Validation failed!")
        
        return all_match


def inspect_mat_file(mat_path: Union[str, Path]):
    """Inspect and print information about a .mat file.
    
    Args:
        mat_path: Path to .mat file
    """
    converter = MatConverter(verbose=True)
    converter.print_inspection(mat_path)


def batch_convert(input_dir: Union[str, Path],
                 output_dir: Union[str, Path],
                 format: str = 'hdf5',
                 pattern: str = '*.mat',
                 **kwargs):
    """Batch convert multiple .mat files.
    
    Args:
        input_dir: Directory containing .mat files
        output_dir: Directory for converted files
        format: Output format
        pattern: Glob pattern for selecting files
        **kwargs: Conversion options
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .mat files
    mat_files = list(input_dir.glob(pattern))
    
    if not mat_files:
        print(f"No files matching pattern '{pattern}' found in {input_dir}")
        return
    
    print(f"Found {len(mat_files)} .mat files")
    print("=" * 70)
    
    converter = MatConverter(verbose=True)
    
    # Determine output extension
    ext = '.npz' if format == 'numpy' else f'.{format}'
    
    for i, mat_file in enumerate(mat_files, 1):
        print(f"\n[{i}/{len(mat_files)}] Converting {mat_file.name}...")
        
        output_file = output_dir / (mat_file.stem + ext)
        
        try:
            converter.convert(mat_file, output_file, format=format, **kwargs)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    print("\n" + "=" * 70)
    print(f"Batch conversion complete! Files saved to: {output_dir}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Convert MATLAB .mat files to PyData formats',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect a .mat file
  python mat_converter.py --inspect dataset.mat
  
  # Convert to HDF5 (recommended)
  python mat_converter.py dataset.mat dataset.h5
  
  # Convert to NumPy
  python mat_converter.py --format numpy dataset.mat dataset.npz
  
  # Batch convert
  python mat_converter.py --batch --input-dir Dataset/ --output-dir Converted/
  
  # Validate conversion
  python mat_converter.py --validate dataset.mat dataset.h5
        """
    )
    
    # Main arguments
    parser.add_argument('input', nargs='?', help='Input .mat file')
    parser.add_argument('output', nargs='?', help='Output file')
    
    # Action flags
    parser.add_argument('--inspect', action='store_true',
                       help='Inspect .mat file structure')
    parser.add_argument('--validate', action='store_true',
                       help='Validate conversion')
    parser.add_argument('--batch', action='store_true',
                       help='Batch convert multiple files')
    
    # Format options
    parser.add_argument('--format', default='hdf5',
                       choices=['numpy', 'npz', 'hdf5', 'h5', 'zarr'],
                       help='Output format (default: hdf5)')
    
    # HDF5 options
    parser.add_argument('--compression', default='gzip',
                       help='HDF5 compression (default: gzip)')
    parser.add_argument('--compression-level', type=int, default=4,
                       help='Compression level 0-9 (default: 4)')
    
    # Batch options
    parser.add_argument('--input-dir', help='Input directory for batch mode')
    parser.add_argument('--output-dir', help='Output directory for batch mode')
    parser.add_argument('--pattern', default='*.mat',
                       help='File pattern for batch mode (default: *.mat)')
    
    # Other options
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output')
    
    args = parser.parse_args()
    
    # Create converter
    converter = MatConverter(verbose=not args.quiet)
    
    try:
        if args.inspect:
            # Inspect mode
            if not args.input:
                parser.error("Input file required for --inspect")
            converter.print_inspection(args.input)
            
        elif args.validate:
            # Validate mode
            if not args.input or not args.output:
                parser.error("Both input and output required for --validate")
            success = converter.validate_conversion(
                args.input, args.output,
                format=args.format
            )
            sys.exit(0 if success else 1)
            
        elif args.batch:
            # Batch mode
            if not args.input_dir or not args.output_dir:
                parser.error("--input-dir and --output-dir required for --batch")
            batch_convert(
                args.input_dir,
                args.output_dir,
                format=args.format,
                pattern=args.pattern,
                compression=args.compression,
                compression_opts=args.compression_level
            )
            
        else:
            # Convert mode
            if not args.input or not args.output:
                parser.error("Both input and output required for conversion")
            converter.convert(
                args.input,
                args.output,
                format=args.format,
                compression=args.compression,
                compression_opts=args.compression_level
            )
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
