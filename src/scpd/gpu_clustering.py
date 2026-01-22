"""
GPU-accelerated clustering optimization module.

Main optimizations:
1. GPU-accelerated KMeans using CuPy
2. Optimized memory usage and data transfer
3. Automatic fallback to CPU implementation
4. Performance benchmarking
"""

import numpy as np
import warnings
from typing import Optional, Tuple
import time

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU acceleration available (CuPy)")
except ImportError:
    GPU_AVAILABLE = False
    print("GPU acceleration unavailable, using CPU")


class GPUKMeans:
    """GPU-accelerated KMeans implementation"""
    
    def __init__(self, n_clusters: int, max_iter: int = 100, tol: float = 1e-4, random_state: int = 0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        
    def _init_centroids_gpu(self, X_gpu):
        """Initialize cluster centers on GPU"""
        n_samples, n_features = X_gpu.shape
        
        cp.random.seed(self.random_state)
        
        centroids = cp.zeros((self.n_clusters, n_features))
        centroids[0] = X_gpu[cp.random.randint(0, n_samples)]
        
        for i in range(1, self.n_clusters):
            distances = cp.full(n_samples, cp.inf)
            for j in range(i):
                dist = cp.sum((X_gpu - centroids[j]) ** 2, axis=1)
                distances = cp.minimum(distances, dist)
            
            probabilities = distances / cp.sum(distances)
            cumulative_probs = cp.cumsum(probabilities)
            r = cp.random.rand()
            centroids[i] = X_gpu[cp.searchsorted(cumulative_probs, r)]
            
        return centroids
    
    def fit_predict(self, X):
        """Fit and predict cluster labels"""
        if not GPU_AVAILABLE:
            raise RuntimeError("GPU unavailable, use CPU version")
            
        X_gpu = cp.asarray(X, dtype=cp.float32)
        n_samples, n_features = X_gpu.shape
        
        centroids = self._init_centroids_gpu(X_gpu)
        
        prev_labels = cp.zeros(n_samples, dtype=cp.int32)
        
        for iteration in range(self.max_iter):
            distances = cp.zeros((n_samples, self.n_clusters))
            for i in range(self.n_clusters):
                distances[:, i] = cp.sum((X_gpu - centroids[i]) ** 2, axis=1)
            
            labels = cp.argmin(distances, axis=1)
            
            if iteration > 0 and cp.all(labels == prev_labels):
                break
                
            new_centroids = cp.zeros_like(centroids)
            for i in range(self.n_clusters):
                mask = labels == i
                if cp.sum(mask) > 0:
                    new_centroids[i] = cp.mean(X_gpu[mask], axis=0)
                else:
                    new_centroids[i] = centroids[i]
            
            center_shift = cp.sum((centroids - new_centroids) ** 2)
            centroids = new_centroids
            
            if center_shift < self.tol:
                break
                
            prev_labels = labels.copy()
        
        self.cluster_centers_ = cp.asnumpy(centroids)
        self.labels_ = cp.asnumpy(labels)
        
        return self.labels_


def gpu_kmeans_clustering(
    X: np.ndarray,
    n_clusters: int,
    random_state: int = 0,
    max_iter: int = 100
) -> np.ndarray:
    """
    GPU-accelerated KMeans clustering.
    
    Parameters
    ----------
    X : ndarray
        Input feature matrix.
    n_clusters : int
        Number of clusters.
    random_state : int
        Random seed.
    max_iter : int
        Maximum iterations.
        
    Returns
    -------
    labels : ndarray
        Cluster labels.
    """
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU unavailable")
    
    kmeans = GPUKMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
        random_state=random_state
    )
    
    return kmeans.fit_predict(X)


def optimized_minibatch_kmeans(
    X: np.ndarray,
    n_clusters: int,
    random_state: int = 0,
    batch_size: Optional[int] = None
) -> np.ndarray:
    """
    Optimized MiniBatch KMeans implementation.
    """
    from sklearn.cluster import MiniBatchKMeans
    
    n_samples = X.shape[0]
    
    if batch_size is None:
        batch_size = min(max(256, n_samples // 50), 4096)
    
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=batch_size,
        n_init=10,
        max_iter=50,
        tol=1e-3,
        reassignment_ratio=0.01
    )
    
    return kmeans.fit_predict(X)


def perform_fast_clustering(
    s: np.ndarray,
    feature_matrix: Optional[np.ndarray],
    n_landmarks: int,
    method: str = "auto",
    random_state: int = 0
) -> Tuple[np.ndarray, str]:
    """
    Perform fast clustering.
    
    Parameters
    ----------
    s : ndarray
        State coordinates.
    feature_matrix : ndarray or None
        Feature matrix.
    n_landmarks : int
        Number of clusters.
    method : str
        Clustering method: "auto", "gpu", "cpu_optimized", "cpu_standard".
    random_state : int
        Random seed.
        
    Returns
    -------
    labels : ndarray
        Cluster labels.
    method_used : str
        Actual method used.
    """
    n_cells = len(s)
    
    if feature_matrix is None:
        X = s.reshape(-1, 1)
        data_type = "1D"
    else:
        X = feature_matrix
        data_type = f"{X.shape[1]}D"
    
    if method == "auto":
        if GPU_AVAILABLE and n_cells > 5000:
            method = "gpu"
        elif n_cells > 10000:
            method = "cpu_optimized"
        else:
            method = "cpu_standard"
    
    start_time = time.time()
    
    try:
        if method == "gpu" and GPU_AVAILABLE:
            labels = gpu_kmeans_clustering(X, n_landmarks, random_state)
            method_used = f"GPU_KMeans_{data_type}"
            
        elif method == "cpu_optimized":
            labels = optimized_minibatch_kmeans(X, n_landmarks, random_state)
            method_used = f"OptimizedMiniBatch_{data_type}"
            
        else:
            from sklearn.cluster import MiniBatchKMeans
            kmeans = MiniBatchKMeans(
                n_clusters=n_landmarks,
                random_state=random_state,
                batch_size=min(1024, n_cells),
                n_init=3
            )
            labels = kmeans.fit_predict(X)
            method_used = f"StandardMiniBatch_{data_type}"
            
    except Exception as e:
        warnings.warn(f"Clustering method {method} failed: {e}, falling back to standard method")
        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(
            n_clusters=n_landmarks,
            random_state=random_state,
            batch_size=min(1024, n_cells),
            n_init=3
        )
        labels = kmeans.fit_predict(X)
        method_used = f"Fallback_MiniBatch_{data_type}"
    
    elapsed_time = time.time() - start_time
    print(f"Clustering complete: {method_used}, time: {elapsed_time:.2f}s")
    
    return labels, method_used


def benchmark_clustering_performance(
    X: np.ndarray,
    n_clusters: int,
    random_state: int = 0
) -> dict:
    """
    Benchmark performance of different clustering methods.
    """
    results = {}
    
    print(f"Benchmarking clustering - data shape: {X.shape}, n_clusters: {n_clusters}")
    
    print("Testing standard MiniBatch KMeans...")
    start_time = time.time()
    try:
        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state)
        labels_standard = kmeans.fit_predict(X)
        time_standard = time.time() - start_time
        results['Standard_MiniBatch'] = {
            'time': time_standard,
            'n_clusters': len(np.unique(labels_standard)),
            'success': True
        }
        print(f"  Complete, time: {time_standard:.2f}s")
    except Exception as e:
        results['Standard_MiniBatch'] = {'success': False, 'error': str(e)}
        print(f"  Failed: {e}")
    
    print("Testing optimized MiniBatch KMeans...")
    start_time = time.time()
    try:
        labels_optimized = optimized_minibatch_kmeans(X, n_clusters, random_state)
        time_optimized = time.time() - start_time
        results['Optimized_MiniBatch'] = {
            'time': time_optimized,
            'n_clusters': len(np.unique(labels_optimized)),
            'success': True
        }
        print(f"  Complete, time: {time_optimized:.2f}s")
    except Exception as e:
        results['Optimized_MiniBatch'] = {'success': False, 'error': str(e)}
        print(f"  Failed: {e}")
    
    if GPU_AVAILABLE:
        print("Testing GPU KMeans...")
        start_time = time.time()
        try:
            labels_gpu = gpu_kmeans_clustering(X, n_clusters, random_state)
            time_gpu = time.time() - start_time
            results['GPU_KMeans'] = {
                'time': time_gpu,
                'n_clusters': len(np.unique(labels_gpu)),
                'success': True
            }
            print(f"  Complete, time: {time_gpu:.2f}s")
        except Exception as e:
            results['GPU_KMeans'] = {'success': False, 'error': str(e)}
            print(f"  Failed: {e}")
    else:
        print("GPU unavailable, skipping GPU test")
    
    return results


if __name__ == "__main__":
    np.random.seed(42)
    n_samples = 10000
    n_features = 50
    n_clusters = 100
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    print("Starting performance benchmark...")
    results = benchmark_clustering_performance(X, n_clusters)
    
    print("\n=== Benchmark Results ===")
    for method, result in results.items():
        if result['success']:
            print(f"{method}: {result['time']:.2f}s, {result['n_clusters']} clusters")
        else:
            print(f"{method}: Failed - {result['error']}")
