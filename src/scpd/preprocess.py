"""
Data preprocessing and input preparation.

Handles input validation, landmarking (over-clustering for computational speedup),
and construction of PreparedData objects for fitting.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Dict, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class LandmarkInfo:
    """
    Information about landmark clustering.
    
    Attributes
    ----------
    enabled : bool
        Whether landmarking is active.
    n_landmarks : int
        Number of landmark clusters (K).
    cluster_labels : ndarray of shape (n_cells,)
        Cluster assignment for each cell.
    landmark_s : ndarray of shape (n_landmarks,)
        Representative s value (median) for each cluster.
    cluster_sizes : ndarray of shape (n_landmarks,)
        Number of cells in each cluster.
    cluster_method : str
        Clustering algorithm used.
    """
    enabled: bool
    n_landmarks: int = 0
    cluster_labels: Optional[np.ndarray] = None
    landmark_s: Optional[np.ndarray] = None
    cluster_sizes: Optional[np.ndarray] = None
    cluster_method: str = "none"


@dataclass
class PreparedData:
    """
    Prepared input data for model fitting.
    
    Attributes
    ----------
    s : ndarray of shape (n_cells,)
        State coordinate for each cell, normalized to [0, 1].
    time_labels : ndarray of shape (n_cells,)
        Time point / stage label for each cell.
    unique_times : ndarray of shape (n_times,)
        Sorted unique time points.
    time_values : ndarray of shape (n_times,)
        Numeric time values for simulation.
    n_cells : int
        Total number of cells.
    n_times : int
        Number of unique time points.
    N_obs : Optional[ndarray of shape (n_times,)]
        Observed population size at each time point (if provided).
    sigma_N : Optional[ndarray of shape (n_times,)]
        Uncertainty in N_obs.
    replicate_labels : Optional[ndarray of shape (n_cells,)]
        Replicate labels for uncertainty estimation.
    landmark_info : LandmarkInfo
        Information about landmark clustering.
    feature_matrix : Optional[ndarray]
        Feature matrix used for clustering (if landmarking enabled).
    """
    s: np.ndarray
    time_labels: np.ndarray
    unique_times: np.ndarray
    time_values: np.ndarray
    n_cells: int
    n_times: int
    N_obs: Optional[np.ndarray] = None
    sigma_N: Optional[np.ndarray] = None
    replicate_labels: Optional[np.ndarray] = None
    landmark_info: LandmarkInfo = field(default_factory=lambda: LandmarkInfo(enabled=False))
    feature_matrix: Optional[np.ndarray] = None
    
    # Per-time-point data (computed during preparation)
    cells_per_time: Optional[List[np.ndarray]] = None
    s_per_time: Optional[List[np.ndarray]] = None
    weights_per_time: Optional[List[np.ndarray]] = None


def _determine_n_landmarks(
    n_cells: int,
    landmarks: str,
    n_landmarks: Optional[int],
    target_cells_per_landmark: int = 20,
    max_landmarks: int = 5000,
    min_cells_for_landmark: int = 2500
) -> Tuple[bool, int]:
    """
    Determine whether to use landmarking and how many landmarks.
    
    Parameters
    ----------
    n_cells : int
        Total number of cells.
    landmarks : str
        "auto", "on", or "off".
    n_landmarks : int or None
        User-specified number of landmarks.
    target_cells_per_landmark : int
        Target cells per landmark for auto mode.
    max_landmarks : int
        Maximum number of landmarks.
    min_cells_for_landmark : int
        Minimum cells required to enable landmarking in auto mode.
        
    Returns
    -------
    use_landmarks : bool
        Whether to use landmarking.
    K : int
        Number of landmarks (0 if not using).
    """
    if landmarks == "off":
        return False, 0
    
    if landmarks == "on" and n_landmarks is not None:
        K = min(n_landmarks, n_cells)
        return True, K
    
    if landmarks == "on":
        # Force on with default calculation
        K = max(500, int(np.ceil(n_cells / target_cells_per_landmark)))
        K = min(K, max_landmarks, n_cells)
        return True, K
    
    # landmarks == "auto"
    if n_cells < min_cells_for_landmark:
        return False, 0
    
    K = max(500, int(np.ceil(n_cells / target_cells_per_landmark)))
    K = min(K, max_landmarks, n_cells)
    return True, K


def _perform_landmarking(
    s: np.ndarray,
    feature_matrix: Optional[np.ndarray],
    n_landmarks: int,
    random_state: int = 0,
    use_optimized: bool = True
) -> LandmarkInfo:
    """
    Perform landmark clustering with optimizations.
    
    Parameters
    ----------
    s : ndarray of shape (n_cells,)
        State coordinates.
    feature_matrix : ndarray or None
        Feature matrix for clustering. If None, uses s for 1D clustering.
    n_landmarks : int
        Number of clusters (K).
    random_state : int
        Random state for reproducibility.
    use_optimized : bool
        Whether to use optimized clustering parameters.
        
    Returns
    -------
    info : LandmarkInfo
        Landmark clustering information.
    """
    import time
    from sklearn.cluster import MiniBatchKMeans
    
    n_cells = len(s)
    
    if feature_matrix is None:
        X = s.reshape(-1, 1)
        method = "MiniBatchKMeans_1D"
    else:
        X = feature_matrix
        method = "MiniBatchKMeans_PCA"
    
    start_time = time.time()
    
    if use_optimized:
        if n_cells > 50000:
            batch_size = min(2048, n_cells // 25)
            n_init = 3
            max_iter = 30
            tol = 1e-2
        else:
            batch_size = min(max(512, n_cells // 20), 1024)
            n_init = 3
            max_iter = 100
            tol = 1e-4
        
        kmeans = MiniBatchKMeans(
            n_clusters=n_landmarks,
            random_state=random_state,
            batch_size=batch_size,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol
        )
        method += "_optimized"
    else:
        kmeans = MiniBatchKMeans(
            n_clusters=n_landmarks,
            random_state=random_state,
            batch_size=min(1024, n_cells),
            n_init=3
        )
        method += "_standard"
    
    labels = kmeans.fit_predict(X)
    clustering_time = time.time() - start_time
    
    unique_labels = np.unique(labels)
    n_actual_clusters = len(unique_labels)
    
    if n_actual_clusters != n_landmarks:
        label_map = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_map[label] for label in labels])
    
    landmark_s = np.zeros(n_actual_clusters)
    cluster_sizes = np.bincount(labels, minlength=n_actual_clusters)
    
    for c in range(n_actual_clusters):
        mask = labels == c
        if cluster_sizes[c] > 0:
            landmark_s[c] = np.median(s[mask])
    
    print(f"Clustering complete: {method}, {n_cells} cells -> {n_actual_clusters} clusters, time: {clustering_time:.2f}s")
    
    return LandmarkInfo(
        enabled=True,
        n_landmarks=n_actual_clusters,
        cluster_labels=labels,
        landmark_s=landmark_s,
        cluster_sizes=cluster_sizes,
        cluster_method=method
    )


def prepare_inputs(
    s: np.ndarray,
    time_labels: np.ndarray,
    time_values: Optional[np.ndarray] = None,
    N_obs: Optional[np.ndarray] = None,
    sigma_N: Optional[np.ndarray] = None,
    sigma_N_frac: float = 0.05,
    replicate_labels: Optional[np.ndarray] = None,
    feature_matrix: Optional[np.ndarray] = None,
    landmarks: str = "auto",
    n_landmarks: Optional[int] = None,
    target_cells_per_landmark: int = 20,
    max_landmarks: int = 5000,
    min_cells_for_landmark: int = 2500,
    n_pca_dims: int = 50,
    random_state: int = 0,
    use_optimized_clustering: bool = True
) -> PreparedData:
    """
    Prepare input data for model fitting.
    
    Parameters
    ----------
    s : ndarray of shape (n_cells,)
        State coordinate for each cell. Will be normalized to [0, 1].
    time_labels : ndarray of shape (n_cells,)
        Time point or stage label for each cell.
    time_values : ndarray of shape (n_times,), optional
        Numeric time values for simulation. If None, uses 0, 1, 2, ...
    N_obs : ndarray of shape (n_times,), optional
        Observed population size at each time point. Required for mode="with_population".
    sigma_N : ndarray of shape (n_times,), optional
        Uncertainty in N_obs. If None and N_obs provided, uses sigma_N_frac * N_obs.
    sigma_N_frac : float, default=0.05
        Fraction of N_obs to use for sigma_N if not provided.
    replicate_labels : ndarray of shape (n_cells,), optional
        Replicate labels for estimating sigma_A from replicate variation.
    feature_matrix : ndarray of shape (n_cells, n_features), optional
        Feature matrix for landmark clustering (e.g., PCA coordinates).
    landmarks : str, default="auto"
        Landmarking mode: "auto", "on", or "off".
    n_landmarks : int, optional
        Number of landmarks. If None, determined automatically.
    target_cells_per_landmark : int, default=20
        Target cells per landmark for automatic K selection.
    max_landmarks : int, default=5000
        Maximum number of landmarks.
    min_cells_for_landmark : int, default=2500
        Minimum cells required to enable landmarking in auto mode.
    n_pca_dims : int, default=50
        Number of PCA dimensions to use if feature_matrix has more.
    random_state : int, default=0
        Random state for reproducibility.
    use_optimized_clustering : bool, default=True
        Whether to use optimized clustering parameters for speed.
        
    Returns
    -------
    prepared : PreparedData
        Prepared data ready for fitting.
        
    Notes
    -----
    State coordinate s is normalized to [0, 1] using min-max scaling.
    
    Landmarking is automatically enabled when n_cells >= 2500 (configurable).
    When enabled, cells are clustered into K landmarks, and ECDF/bootstrap
    operations use cluster-level weights for efficiency.
    """
    s = np.asarray(s, dtype=float).ravel()
    time_labels = np.asarray(time_labels).ravel()
    n_cells = len(s)
    
    if len(time_labels) != n_cells:
        raise ValueError(f"s and time_labels must have same length: {n_cells} vs {len(time_labels)}")
    
    # Normalize s to [0, 1]
    s_min, s_max = np.min(s), np.max(s)
    if s_max > s_min:
        s_normalized = (s - s_min) / (s_max - s_min)
    else:
        s_normalized = np.zeros_like(s) + 0.5
    
    # Get unique times
    unique_times = np.unique(time_labels)
    n_times = len(unique_times)
    
    # Time values for simulation
    if time_values is None:
        time_values = np.arange(n_times, dtype=float)
    else:
        time_values = np.asarray(time_values, dtype=float)
        if len(time_values) != n_times:
            raise ValueError(f"time_values length {len(time_values)} != n_times {n_times}")
    
    # Handle N_obs and sigma_N
    if N_obs is not None:
        N_obs = np.asarray(N_obs, dtype=float)
        if len(N_obs) != n_times:
            raise ValueError(f"N_obs length {len(N_obs)} != n_times {n_times}")
        if sigma_N is None:
            sigma_N = sigma_N_frac * N_obs
        else:
            sigma_N = np.asarray(sigma_N, dtype=float)
    
    # Replicate labels
    if replicate_labels is not None:
        replicate_labels = np.asarray(replicate_labels).ravel()
    
    # Feature matrix for clustering
    if feature_matrix is not None:
        feature_matrix = np.asarray(feature_matrix)
        if feature_matrix.shape[0] != n_cells:
            raise ValueError(f"feature_matrix rows {feature_matrix.shape[0]} != n_cells {n_cells}")
        # Limit PCA dimensions
        if feature_matrix.shape[1] > n_pca_dims:
            feature_matrix = feature_matrix[:, :n_pca_dims]
    
    # Determine landmarking
    use_landmarks, K = _determine_n_landmarks(
        n_cells=n_cells,
        landmarks=landmarks,
        n_landmarks=n_landmarks,
        target_cells_per_landmark=target_cells_per_landmark,
        max_landmarks=max_landmarks,
        min_cells_for_landmark=min_cells_for_landmark
    )
    
    if use_landmarks:
        landmark_info = _perform_landmarking(
            s=s_normalized,
            feature_matrix=feature_matrix,
            n_landmarks=K,
            random_state=random_state,
            use_optimized=use_optimized_clustering
        )
    else:
        landmark_info = LandmarkInfo(enabled=False)
    
    cells_per_time = []
    s_per_time = []
    weights_per_time = []
    
    for t in unique_times:
        mask = time_labels == t
        indices = np.where(mask)[0]
        cells_per_time.append(indices)
        s_per_time.append(s_normalized[mask])
        
        if use_landmarks:
            cluster_labels_t = landmark_info.cluster_labels[mask]
            weights = np.bincount(cluster_labels_t, minlength=landmark_info.n_landmarks)
            weights_per_time.append(weights.astype(float))
        else:
            weights_per_time.append(np.ones(len(indices)))
    
    return PreparedData(
        s=s_normalized,
        time_labels=time_labels,
        unique_times=unique_times,
        time_values=time_values,
        n_cells=n_cells,
        n_times=n_times,
        N_obs=N_obs,
        sigma_N=sigma_N,
        replicate_labels=replicate_labels,
        landmark_info=landmark_info,
        feature_matrix=feature_matrix,
        cells_per_time=cells_per_time,
        s_per_time=s_per_time,
        weights_per_time=weights_per_time
    )


def prepare_from_anndata(
    adata,
    s_key: str = "dpt_pseudotime",
    time_key: str = "time",
    N_obs: Optional[np.ndarray] = None,
    pca_key: str = "X_pca",
    **kwargs
) -> PreparedData:
    """
    Prepare inputs from an AnnData object.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    s_key : str, default="dpt_pseudotime"
        Key in adata.obs for the state coordinate.
    time_key : str, default="time"
        Key in adata.obs for time labels.
    N_obs : ndarray, optional
        Observed population sizes.
    pca_key : str, default="X_pca"
        Key in adata.obsm for PCA coordinates (used for landmarking).
    **kwargs
        Additional arguments passed to prepare_inputs.
        
    Returns
    -------
    prepared : PreparedData
        Prepared data ready for fitting.
    """
    s = adata.obs[s_key].values
    time_labels = adata.obs[time_key].values
    
    # Get PCA features if available
    feature_matrix = None
    if pca_key in adata.obsm:
        feature_matrix = adata.obsm[pca_key]
    
    return prepare_inputs(
        s=s,
        time_labels=time_labels,
        N_obs=N_obs,
        feature_matrix=feature_matrix,
        **kwargs
    )


def find_robust_root(adata, day_column='day', day_value=0.0, pca_key='X_pca'):
    """
    Find root cell as nearest neighbor to geometric centroid of specified time point.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing cell data.
    day_column : str, default='day'
        Column name for time points.
    day_value : float, default=0.0
        Time point value for root cell selection.
    pca_key : str, default='X_pca'
        Key for PCA coordinates in obsm.
        
    Returns
    -------
    root_index : int
        Index of selected root cell.
    """
    from sklearn.metrics import pairwise_distances
    
    day_indices = np.flatnonzero(adata.obs[day_column] == day_value)
    
    if len(day_indices) > 0:
        day_pca = adata.obsm[pca_key][day_indices]
        centroid = np.mean(day_pca, axis=0)
        dists = pairwise_distances(day_pca, centroid.reshape(1, -1))
        root_index_in_subset = np.argmin(dists)
        best_root = day_indices[root_index_in_subset]
        adata.uns['iroot'] = best_root
        print(f"Selected robust root cell index: {best_root}")
        return best_root
    else:
        print("Warning: No cells found at specified time point. Using default index 0.")
        adata.uns['iroot'] = 0
        return 0


def compute_normalized_pseudotime(adata, n_dcs=10, percentile=99):
    """
    Compute and normalize pseudotime to [0,1].
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing cell data.
    n_dcs : int, default=10
        Number of diffusion components.
    percentile : float, default=99
        Percentile for normalization.
        
    Returns
    -------
    s : ndarray
        Normalized pseudotime [0,1].
    """
    import scanpy as sc
    
    sc.tl.dpt(adata, n_dcs=n_dcs)
    s_raw = adata.obs['dpt_pseudotime'].values
    s_robust_max = np.percentile(s_raw, percentile)
    
    print(f"{percentile}% of cells at s <= {s_robust_max:.4f}")
    
    s = s_raw / s_robust_max 
    s[s > 1.0] = 1.0
    
    return s

