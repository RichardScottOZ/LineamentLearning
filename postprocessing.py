"""
Post-processing module for LineamentLearning.

This module provides modern post-processing capabilities including clustering
and line fitting for converting probability maps to lineament predictions.
"""

import numpy as np
from typing import Tuple, List, Optional
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

from config import InferenceConfig


class PostProcessor:
    """Post-processing for probability maps to extract lineaments.
    
    This class provides functionality to:
    1. Apply thresholding to probability maps
    2. Cluster detected regions using DBSCAN
    3. Fit lines or curves to clusters
    4. Generate final lineament predictions
    """
    
    def __init__(self, config: InferenceConfig):
        """Initialize post-processor.
        
        Args:
            config: Inference configuration with clustering parameters
        """
        self.config = config
    
    def apply_threshold(self, pmap: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Apply threshold to probability map.
        
        Args:
            pmap: Probability map (H x W)
            threshold: Probability threshold (uses config if None)
            
        Returns:
            Binary map with values above threshold
        """
        if threshold is None:
            threshold = self.config.threshold
        
        binary_map = np.zeros_like(pmap)
        binary_map[pmap >= threshold] = 1
        return binary_map
    
    def cluster_detections(self, pmap: np.ndarray, 
                          threshold: Optional[float] = None,
                          eps: Optional[float] = None,
                          min_samples: Optional[int] = None) -> np.ndarray:
        """Cluster detected regions using DBSCAN.
        
        Args:
            pmap: Probability map (H x W)
            threshold: Probability threshold (uses config if None)
            eps: DBSCAN epsilon parameter (uses config if None)
            min_samples: Minimum samples for cluster (uses config if None)
            
        Returns:
            Cluster map with cluster IDs (H x W)
        """
        # Apply threshold
        if threshold is None:
            threshold = self.config.cutoff
        binary_map = self.apply_threshold(pmap, threshold)
        
        # Get coordinates of detections
        coords = np.transpose(np.where(binary_map > 0))
        
        if len(coords) == 0:
            return np.zeros_like(pmap, dtype=np.int32)
        
        # Apply DBSCAN
        if eps is None:
            eps = self.config.eps
        if min_samples is None:
            min_samples = self.config.min_cluster_size
        
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clusterer.fit_predict(coords)
        
        # Create cluster map
        cluster_map = np.zeros_like(pmap, dtype=np.int32)
        for i, (x, y) in enumerate(coords):
            cluster_map[x, y] = labels[i] + 1  # +1 to make noise cluster 0
        
        return cluster_map
    
    def fit_line_to_cluster(self, cluster_map: np.ndarray, 
                           cluster_id: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Fit a line to a cluster using RANSAC.
        
        Args:
            cluster_map: Map with cluster IDs
            cluster_id: ID of cluster to fit
            
        Returns:
            Tuple of (start_point, end_point) or None if fitting fails
        """
        # Get cluster coordinates
        coords = np.where(cluster_map == cluster_id)
        if len(coords[0]) == 0:
            return None
        
        X = coords[0].reshape(-1, 1)
        y = coords[1]
        
        # Fit line using RANSAC
        try:
            ransac = RANSACRegressor(random_state=42)
            ransac.fit(X, y)
            
            # Get line endpoints
            x_min, x_max = X.min(), X.max()
            y_min = ransac.predict([[x_min]])[0]
            y_max = ransac.predict([[x_max]])[0]
            
            start_point = np.array([x_min, y_min])
            end_point = np.array([x_max, y_max])
            
            return start_point, end_point
        except Exception:
            return None
    
    def fit_curve_to_cluster(self, cluster_map: np.ndarray,
                             cluster_id: int,
                             degree: int = 3) -> Optional[np.ndarray]:
        """Fit a polynomial curve to a cluster.
        
        Args:
            cluster_map: Map with cluster IDs
            cluster_id: ID of cluster to fit
            degree: Polynomial degree
            
        Returns:
            Array of curve points (Nx2) or None if fitting fails
        """
        # Get cluster coordinates
        coords = np.where(cluster_map == cluster_id)
        if len(coords[0]) == 0:
            return None
        
        X = coords[0].reshape(-1, 1)
        y = coords[1]
        
        # Fit polynomial curve
        try:
            model = make_pipeline(PolynomialFeatures(degree), Ridge())
            model.fit(X, y)
            
            # Generate curve points
            x_curve = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            y_curve = model.predict(x_curve)
            
            curve_points = np.column_stack([x_curve.ravel(), y_curve])
            return curve_points
        except Exception:
            return None
    
    def fit_best_curve_to_cluster(self, cluster_map: np.ndarray,
                                  cluster_id: int,
                                  degrees: Optional[List[int]] = None) -> Optional[np.ndarray]:
        """Fit the best polynomial curve to a cluster.
        
        Tries multiple polynomial degrees and selects the one with lowest error.
        
        Args:
            cluster_map: Map with cluster IDs
            cluster_id: ID of cluster to fit
            degrees: List of degrees to try (uses config if None)
            
        Returns:
            Array of curve points (Nx2) or None if fitting fails
        """
        if degrees is None:
            degrees = self.config.polynomial_degrees
        
        # Get cluster coordinates
        coords = np.where(cluster_map == cluster_id)
        if len(coords[0]) == 0:
            return None
        
        X = coords[0].reshape(-1, 1)
        y = coords[1]
        
        best_model = None
        best_error = float('inf')
        
        # Try each degree
        for degree in degrees:
            try:
                model = make_pipeline(PolynomialFeatures(degree), Ridge())
                model.fit(X, y)
                
                # Calculate error
                y_pred = model.predict(X)
                error = mean_squared_error(y, y_pred)
                
                if error < best_error:
                    best_error = error
                    best_model = model
            except Exception:
                continue
        
        if best_model is None:
            return None
        
        # Generate curve points with best model
        x_curve = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_curve = best_model.predict(x_curve)
        
        curve_points = np.column_stack([x_curve.ravel(), y_curve])
        return curve_points
    
    def extract_lineaments(self, pmap: np.ndarray) -> Tuple[np.ndarray, List]:
        """Extract lineaments from probability map.
        
        Full pipeline: threshold → cluster → fit lines/curves
        
        Args:
            pmap: Probability map (H x W)
            
        Returns:
            Tuple of (cluster_map, lineaments)
            - cluster_map: Map with cluster IDs (H x W)
            - lineaments: List of fitted lines/curves, each as dict with:
                - 'cluster_id': int
                - 'type': 'line' or 'curve'
                - 'points': np.ndarray of shape (N, 2)
        """
        # Step 1: Cluster detections
        if not self.config.use_clustering:
            # No clustering - just return thresholded map
            binary_map = self.apply_threshold(pmap)
            return binary_map.astype(np.int32), []
        
        cluster_map = self.cluster_detections(pmap)
        
        # Step 2: Fit lines or curves to each cluster
        lineaments = []
        cluster_ids = np.unique(cluster_map)
        cluster_ids = cluster_ids[cluster_ids > 0]  # Exclude noise (0)
        
        for cluster_id in cluster_ids:
            # Choose fitting method based on config
            if self.config.line_fitting_method == 'Linear':
                result = self.fit_line_to_cluster(cluster_map, cluster_id)
                if result is not None:
                    start, end = result
                    points = np.array([start, end])
                    lineaments.append({
                        'cluster_id': int(cluster_id),
                        'type': 'line',
                        'points': points
                    })
            
            elif self.config.line_fitting_method == 'Curve':
                points = self.fit_curve_to_cluster(cluster_map, cluster_id, degree=3)
                if points is not None:
                    lineaments.append({
                        'cluster_id': int(cluster_id),
                        'type': 'curve',
                        'points': points
                    })
            
            elif self.config.line_fitting_method == 'BestCurve':
                points = self.fit_best_curve_to_cluster(cluster_map, cluster_id)
                if points is not None:
                    lineaments.append({
                        'cluster_id': int(cluster_id),
                        'type': 'best_curve',
                        'points': points
                    })
        
        return cluster_map, lineaments
    
    def get_cluster_statistics(self, cluster_map: np.ndarray) -> dict:
        """Get statistics about clusters.
        
        Args:
            cluster_map: Map with cluster IDs
            
        Returns:
            Dictionary with statistics
        """
        cluster_ids = np.unique(cluster_map)
        cluster_ids = cluster_ids[cluster_ids > 0]  # Exclude background/noise
        
        stats = {
            'n_clusters': len(cluster_ids),
            'cluster_sizes': [],
            'cluster_ids': cluster_ids.tolist()
        }
        
        for cluster_id in cluster_ids:
            size = np.sum(cluster_map == cluster_id)
            stats['cluster_sizes'].append(int(size))
        
        if stats['cluster_sizes']:
            stats['mean_cluster_size'] = float(np.mean(stats['cluster_sizes']))
            stats['max_cluster_size'] = int(np.max(stats['cluster_sizes']))
            stats['min_cluster_size'] = int(np.min(stats['cluster_sizes']))
        
        return stats


def process_probability_map(pmap: np.ndarray, 
                            config: InferenceConfig) -> Tuple[np.ndarray, List, dict]:
    """Convenience function to process a probability map.
    
    Args:
        pmap: Probability map (H x W)
        config: Inference configuration
        
    Returns:
        Tuple of (cluster_map, lineaments, statistics)
    """
    processor = PostProcessor(config)
    cluster_map, lineaments = processor.extract_lineaments(pmap)
    stats = processor.get_cluster_statistics(cluster_map)
    
    return cluster_map, lineaments, stats
