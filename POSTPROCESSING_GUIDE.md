# Post-Processing and Clustering Integration Guide

## Overview

The modernized LineamentLearning pipeline includes comprehensive post-processing capabilities that integrate DBSCAN clustering and line/curve fitting from the original `Prob2Line.py` module.

## Architecture

### Post-Processing Pipeline

```
Model Predictions → Probability Maps → Thresholding → Clustering → Line Fitting → Lineaments
```

### Key Components

1. **`postprocessing.py`**: New module providing modern post-processing
2. **`config.py`**: InferenceConfig with clustering parameters
3. **`model_modern.py`**: ModelPredictor with integrated post-processing
4. **`Prob2Line.py`**: Original implementation (preserved for compatibility)

## Usage

### 1. Basic Post-Processing

```python
from config import Config
from postprocessing import PostProcessor
import numpy as np

# Configure
config = Config()
config.inference.use_clustering = True
config.inference.threshold = 0.5
config.inference.eps = 5.0
config.inference.min_cluster_size = 20
config.inference.line_fitting_method = 'BestCurve'

# Initialize processor
processor = PostProcessor(config.inference)

# Process probability map
probability_map = model.predict(data)  # Your model predictions
cluster_map, lineaments = processor.extract_lineaments(probability_map)

# Get statistics
stats = processor.get_cluster_statistics(cluster_map)
print(f"Found {stats['n_clusters']} clusters")
print(f"Extracted {len(lineaments)} lineaments")
```

### 2. Integrated with ModelPredictor

```python
from config import Config
from model_modern import ModelPredictor

config = Config()
predictor = ModelPredictor(config, 'path/to/model.h5')

# Full prediction + post-processing pipeline
results = predictor.predict_and_postprocess(
    probability_map=pmap,
    output_dir='./results',
    visualize=True
)

# Access results
cluster_map = results['cluster_map']
lineaments = results['lineaments']
statistics = results['statistics']
```

### 3. Configuration Options

```json
{
  "inference": {
    "threshold": 0.5,
    "cutoff": 0.3,
    "eps": 0.3,
    "min_cluster_size": 20,
    "use_clustering": true,
    "clustering_method": "DBSCAN",
    "line_fitting_method": "BestCurve",
    "polynomial_degrees": [1, 3, 5]
  }
}
```

#### Clustering Parameters

- **`threshold`**: Probability threshold for detection (0-1)
- **`cutoff`**: Alternative threshold for clustering
- **`eps`**: DBSCAN epsilon parameter (spatial distance)
- **`min_cluster_size`**: Minimum points to form a cluster
- **`use_clustering`**: Enable/disable clustering
- **`clustering_method`**: Algorithm to use (currently "DBSCAN")

#### Line Fitting Parameters

- **`line_fitting_method`**: Method for fitting
  - `"Linear"`: RANSAC linear regression
  - `"Curve"`: Polynomial curve (degree 3)
  - `"BestCurve"`: Try multiple degrees, select best
- **`polynomial_degrees`**: Degrees to try for BestCurve (e.g., [1, 3, 5])

## API Reference

### PostProcessor Class

#### Methods

**`apply_threshold(pmap, threshold=None)`**
- Applies probability threshold to map
- Returns binary detection map

**`cluster_detections(pmap, threshold=None, eps=None, min_samples=None)`**
- Clusters detections using DBSCAN
- Returns cluster map with cluster IDs

**`fit_line_to_cluster(cluster_map, cluster_id)`**
- Fits linear line to cluster using RANSAC
- Returns (start_point, end_point) tuple

**`fit_curve_to_cluster(cluster_map, cluster_id, degree=3)`**
- Fits polynomial curve to cluster
- Returns array of curve points (Nx2)

**`fit_best_curve_to_cluster(cluster_map, cluster_id, degrees=None)`**
- Tries multiple polynomial degrees
- Selects curve with lowest error
- Returns array of curve points (Nx2)

**`extract_lineaments(pmap)`**
- Complete pipeline: threshold → cluster → fit
- Returns (cluster_map, lineaments) tuple
- Lineaments is list of dicts with:
  - `'cluster_id'`: int
  - `'type'`: 'line', 'curve', or 'best_curve'
  - `'points'`: np.ndarray of shape (N, 2)

**`get_cluster_statistics(cluster_map)`**
- Computes cluster statistics
- Returns dict with counts and sizes

### Convenience Function

**`process_probability_map(pmap, config)`**
- Single function for full pipeline
- Returns (cluster_map, lineaments, statistics)

## Output Format

### Lineaments Structure

Each lineament is a dictionary:

```python
{
    'cluster_id': 5,              # Cluster ID from DBSCAN
    'type': 'best_curve',         # Fitting method used
    'points': np.array([          # Array of (x, y) coordinates
        [10.5, 20.3],
        [11.2, 21.1],
        ...
    ])
}
```

### Statistics Structure

```python
{
    'n_clusters': 12,                    # Number of clusters found
    'cluster_ids': [1, 2, 3, ...],      # List of cluster IDs
    'cluster_sizes': [45, 38, 52, ...], # Size of each cluster
    'mean_cluster_size': 45.3,          # Average cluster size
    'max_cluster_size': 89,             # Largest cluster
    'min_cluster_size': 12              # Smallest cluster
}
```

## Integration with Original Code

The new post-processing integrates with the original `Prob2Line.py`:

### Original (Prob2Line.py)
```python
from Prob2Line import prob2map

p2l = prob2map(pmap)
cmap = p2l.getClusters(cutoff=0.3, eps=0.3)
lines = p2l.makeConversion(cutoff=0.3, eps=0.3)
```

### Modern (postprocessing.py)
```python
from postprocessing import PostProcessor

processor = PostProcessor(config.inference)
cluster_map, lineaments = processor.extract_lineaments(pmap)
```

Both approaches work and are compatible. The modern version:
- ✅ Uses configuration system
- ✅ Supports multiple fitting methods
- ✅ Better error handling
- ✅ Type hints for IDE support
- ✅ Comprehensive statistics

## Examples

### Example 1: Simple Clustering

```python
from postprocessing import PostProcessor
from config import InferenceConfig

config = InferenceConfig()
processor = PostProcessor(config)

# Cluster probability map
cluster_map = processor.cluster_detections(pmap)
stats = processor.get_cluster_statistics(cluster_map)

print(f"Clusters: {stats['n_clusters']}")
```

### Example 2: Different Fitting Methods

```python
# Try linear fitting
config.line_fitting_method = 'Linear'
processor = PostProcessor(config)
_, lineaments_linear = processor.extract_lineaments(pmap)

# Try best curve fitting
config.line_fitting_method = 'BestCurve'
processor = PostProcessor(config)
_, lineaments_curve = processor.extract_lineaments(pmap)

print(f"Linear: {len(lineaments_linear)} lineaments")
print(f"Curves: {len(lineaments_curve)} lineaments")
```

### Example 3: Custom Workflow

```python
processor = PostProcessor(config)

# Step by step processing
binary_map = processor.apply_threshold(pmap, threshold=0.6)
cluster_map = processor.cluster_detections(pmap, eps=10.0)

# Fit specific clusters
for cluster_id in [1, 2, 3]:
    line = processor.fit_line_to_cluster(cluster_map, cluster_id)
    if line:
        print(f"Cluster {cluster_id}: {line}")
```

## Visualization

The `ModelPredictor.predict_and_postprocess()` method includes automatic visualization:

```python
results = predictor.predict_and_postprocess(
    probability_map=pmap,
    output_dir='./results',
    visualize=True  # Generates results_visualization.png
)
```

Output visualization shows:
1. **Probability Map**: Raw model predictions
2. **Clusters**: Color-coded cluster assignments
3. **Lineaments**: Fitted lines/curves overlaid on probability map

## Testing

Run the post-processing example:

```bash
cd examples
python postprocessing_example.py
```

This demonstrates:
- Synthetic probability map generation
- Complete post-processing pipeline
- Different fitting methods
- Statistics computation
- Visualization (if matplotlib available)

## Performance Considerations

### DBSCAN Parameters

- **`eps`**: Larger values merge nearby clusters
  - Typical range: 0.3 to 10.0
  - Depends on data resolution and scale
  
- **`min_cluster_size`**: Filters out noise
  - Typical range: 5 to 50
  - Higher values = fewer but larger clusters

### Fitting Methods

- **Linear**: Fastest, good for straight features
- **Curve**: Medium speed, captures curvature
- **BestCurve**: Slowest, most accurate for varied shapes

## Future Enhancements

Potential improvements (see FUTURE_IMPROVEMENTS.md):

1. **Alternative Clustering**
   - HDBSCAN for hierarchical clustering
   - Mean-shift for variable density
   - OPTICS for ordering

2. **Advanced Fitting**
   - Spline interpolation
   - Bezier curves
   - B-splines

3. **Quality Metrics**
   - Line confidence scores
   - Cluster compactness
   - Fitting residuals

4. **Parallel Processing**
   - Multi-threaded clustering
   - Batch processing
   - GPU acceleration

## Troubleshooting

### Too Many Clusters

- Increase `eps` parameter
- Increase `min_cluster_size`
- Increase `threshold`

### Too Few Clusters

- Decrease `eps` parameter
- Decrease `min_cluster_size`
- Decrease `threshold`

### Poor Line Fitting

- Try different `line_fitting_method`
- Adjust `polynomial_degrees`
- Check cluster quality first

## Summary

The modernized post-processing provides:

✅ **Complete Integration**: Works seamlessly with ModelPredictor  
✅ **Flexible Configuration**: JSON-based parameter control  
✅ **Multiple Methods**: Linear, curve, and best-curve fitting  
✅ **Comprehensive Output**: Clusters, lineaments, and statistics  
✅ **Backward Compatible**: Original Prob2Line.py still available  
✅ **Well Documented**: API reference and examples provided  

The clustering and line extraction pipeline is fully implemented and ready to use once data loading is completed.
