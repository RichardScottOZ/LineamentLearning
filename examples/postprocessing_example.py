"""
Post-processing example for LineamentLearning.

Demonstrates how to use the clustering and line fitting pipeline.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from postprocessing import PostProcessor, process_probability_map


def create_synthetic_probability_map(size=(200, 200), n_lineaments=3):
    """Create a synthetic probability map with lineaments for testing.
    
    Args:
        size: Size of the map (height, width)
        n_lineaments: Number of lineaments to generate
        
    Returns:
        Synthetic probability map
    """
    pmap = np.zeros(size)
    
    # Add some lineaments
    for i in range(n_lineaments):
        # Random line parameters
        x_start = np.random.randint(0, size[0])
        y_start = np.random.randint(0, size[1])
        angle = np.random.uniform(0, np.pi)
        length = np.random.randint(30, 80)
        
        # Draw line with varying probability
        for t in range(length):
            x = int(x_start + t * np.sin(angle))
            y = int(y_start + t * np.cos(angle))
            
            if 0 <= x < size[0] and 0 <= y < size[1]:
                # Gaussian profile around line
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < size[0] and 0 <= ny < size[1]:
                            dist = np.sqrt(dx**2 + dy**2)
                            prob = np.exp(-dist**2 / 2) * np.random.uniform(0.7, 1.0)
                            pmap[nx, ny] = max(pmap[nx, ny], prob)
    
    # Add some noise
    pmap += np.random.uniform(0, 0.1, size)
    pmap = np.clip(pmap, 0, 1)
    
    return pmap


def main():
    """Run post-processing examples."""
    print("=" * 60)
    print("LineamentLearning - Post-processing Example")
    print("=" * 60)
    
    # Create configuration
    config = Config()
    config.inference.use_clustering = True
    config.inference.clustering_method = 'DBSCAN'
    config.inference.threshold = 0.5
    config.inference.eps = 5.0
    config.inference.min_cluster_size = 10
    
    print("\nConfiguration:")
    print(f"  Clustering: {config.inference.use_clustering}")
    print(f"  Method: {config.inference.clustering_method}")
    print(f"  Threshold: {config.inference.threshold}")
    print(f"  DBSCAN eps: {config.inference.eps}")
    print(f"  Min cluster size: {config.inference.min_cluster_size}")
    print(f"  Line fitting: {config.inference.line_fitting_method}")
    
    # Create synthetic probability map
    print("\nGenerating synthetic probability map...")
    pmap = create_synthetic_probability_map(size=(200, 200), n_lineaments=5)
    print(f"Probability map shape: {pmap.shape}")
    print(f"Value range: [{pmap.min():.3f}, {pmap.max():.3f}]")
    print(f"Mean probability: {pmap.mean():.3f}")
    
    # Initialize post-processor
    print("\n" + "=" * 60)
    print("Running Post-processing Pipeline")
    print("=" * 60)
    
    processor = PostProcessor(config.inference)
    
    # Step 1: Apply threshold
    print("\n1. Applying threshold...")
    binary_map = processor.apply_threshold(pmap)
    n_detections = np.sum(binary_map > 0)
    print(f"   Detections above threshold: {n_detections}")
    
    # Step 2: Cluster detections
    print("\n2. Clustering detections...")
    cluster_map = processor.cluster_detections(pmap)
    stats = processor.get_cluster_statistics(cluster_map)
    print(f"   Clusters found: {stats['n_clusters']}")
    if stats['cluster_sizes']:
        print(f"   Cluster sizes: min={stats['min_cluster_size']}, "
              f"max={stats['max_cluster_size']}, "
              f"mean={stats['mean_cluster_size']:.1f}")
    
    # Step 3: Extract lineaments
    print("\n3. Extracting lineaments...")
    
    # Try Linear fitting
    config.inference.line_fitting_method = 'Linear'
    cluster_map, lineaments = processor.extract_lineaments(pmap)
    print(f"   Linear fitting: {len(lineaments)} lineaments")
    
    # Try BestCurve fitting
    config.inference.line_fitting_method = 'BestCurve'
    config.inference.polynomial_degrees = [1, 2, 3]
    processor = PostProcessor(config.inference)
    cluster_map, lineaments = processor.extract_lineaments(pmap)
    print(f"   BestCurve fitting: {len(lineaments)} lineaments")
    
    # Show lineament details
    if lineaments:
        print("\n   Lineament details:")
        for i, lineament in enumerate(lineaments[:3]):  # Show first 3
            print(f"     Lineament {i+1}:")
            print(f"       Cluster ID: {lineament['cluster_id']}")
            print(f"       Type: {lineament['type']}")
            print(f"       Points: {len(lineament['points'])} points")
    
    # Step 4: Use convenience function
    print("\n4. Using convenience function...")
    cluster_map, lineaments, stats = process_probability_map(pmap, config.inference)
    print(f"   Complete processing:")
    print(f"     Clusters: {stats['n_clusters']}")
    print(f"     Lineaments: {len(lineaments)}")
    
    # Save results
    output_dir = Path('./outputs/postprocessing_example')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / 'probability_map.npy', pmap)
    np.save(output_dir / 'cluster_map.npy', cluster_map)
    
    print(f"\n   Results saved to: {output_dir}")
    
    # Try visualization if matplotlib available
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Probability map
        im1 = axes[0].imshow(pmap, cmap='hot')
        axes[0].set_title('Probability Map')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046)
        
        # Clusters
        im2 = axes[1].imshow(cluster_map, cmap='tab20')
        axes[1].set_title(f'Clusters (n={stats["n_clusters"]})')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046)
        
        # Lineaments
        axes[2].imshow(pmap, cmap='gray', alpha=0.5)
        for lineament in lineaments:
            points = lineament['points']
            axes[2].plot(points[:, 1], points[:, 0], 'r-', linewidth=2, alpha=0.8)
        axes[2].set_title(f'Lineaments (n={len(lineaments)})')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'visualization.png', dpi=150, bbox_inches='tight')
        print(f"   Visualization saved to: {output_dir / 'visualization.png'}")
        plt.close()
        
    except ImportError:
        print("   (matplotlib not available for visualization)")
    
    print("\n" + "=" * 60)
    print("Post-processing example completed!")
    print("=" * 60)
    
    print("\nIntegration with ModelPredictor:")
    print("  1. Run model.predict() to get probability maps")
    print("  2. Use predictor.predict_and_postprocess() for full pipeline")
    print("  3. Or use PostProcessor directly for custom workflows")


if __name__ == '__main__':
    main()
