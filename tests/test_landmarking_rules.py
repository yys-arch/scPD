"""
Test landmarking (over-clustering) rules and behavior.
"""

import numpy as np
import pytest

from scpd.preprocess import (
    prepare_inputs,
    _determine_n_landmarks,
    _perform_landmarking,
    LandmarkInfo
)


class TestLandmarkingRules:
    """Test automatic K selection rules."""
    
    def test_small_dataset_no_landmarks(self):
        """Datasets < 2500 cells should not use landmarks by default."""
        use, K = _determine_n_landmarks(
            n_cells=2000,
            landmarks="auto",
            n_landmarks=None
        )
        
        assert use is False
        assert K == 0
    
    def test_large_dataset_uses_landmarks(self):
        """Datasets >= 2500 cells should use landmarks by default."""
        use, K = _determine_n_landmarks(
            n_cells=3000,
            landmarks="auto",
            n_landmarks=None
        )
        
        assert use is True
        assert K >= 500
    
    def test_minimum_landmarks_500(self):
        """K should be at least 500 when landmarks enabled."""
        use, K = _determine_n_landmarks(
            n_cells=5000,
            landmarks="auto",
            n_landmarks=None,
            target_cells_per_landmark=100  # Would give K=50
        )
        
        assert K >= 500
    
    def test_k_scales_with_n_cells(self):
        """K should increase with n_cells."""
        _, K1 = _determine_n_landmarks(n_cells=5000, landmarks="auto", n_landmarks=None)
        _, K2 = _determine_n_landmarks(n_cells=20000, landmarks="auto", n_landmarks=None)
        
        assert K2 > K1
    
    def test_max_landmarks_respected(self):
        """K should not exceed max_landmarks."""
        use, K = _determine_n_landmarks(
            n_cells=100000,
            landmarks="auto",
            n_landmarks=None,
            max_landmarks=3000
        )
        
        assert K <= 3000
    
    def test_force_off(self):
        """landmarks='off' should disable landmarks regardless of size."""
        use, K = _determine_n_landmarks(
            n_cells=50000,
            landmarks="off",
            n_landmarks=None
        )
        
        assert use is False
        assert K == 0
    
    def test_force_on(self):
        """landmarks='on' should enable landmarks regardless of size."""
        use, K = _determine_n_landmarks(
            n_cells=1000,
            landmarks="on",
            n_landmarks=None
        )
        
        assert use is True
        assert K >= 500
    
    def test_explicit_k(self):
        """User-specified K should be used."""
        use, K = _determine_n_landmarks(
            n_cells=5000,
            landmarks="on",
            n_landmarks=100
        )
        
        assert use is True
        assert K == 100


class TestLandmarkClustering:
    """Test the clustering process."""
    
    def test_cluster_labels_shape(self):
        """Cluster labels should match n_cells."""
        np.random.seed(42)
        s = np.random.uniform(0, 1, 1000)
        
        info = _perform_landmarking(s, None, n_landmarks=50)
        
        assert len(info.cluster_labels) == 1000
        assert info.n_landmarks == 50
    
    def test_all_clusters_assigned(self):
        """All cells should be assigned to some cluster."""
        np.random.seed(42)
        s = np.random.uniform(0, 1, 500)
        
        info = _perform_landmarking(s, None, n_landmarks=20)
        
        # All labels should be in [0, n_landmarks)
        assert np.all(info.cluster_labels >= 0)
        assert np.all(info.cluster_labels < 20)
    
    def test_landmark_s_in_range(self):
        """Landmark s values should be in [0, 1]."""
        np.random.seed(42)
        s = np.random.uniform(0, 1, 500)
        
        info = _perform_landmarking(s, None, n_landmarks=30)
        
        assert np.all(info.landmark_s >= 0)
        assert np.all(info.landmark_s <= 1)
    
    def test_cluster_sizes_sum(self):
        """Cluster sizes should sum to n_cells."""
        np.random.seed(42)
        s = np.random.uniform(0, 1, 500)
        
        info = _perform_landmarking(s, None, n_landmarks=25)
        
        assert np.sum(info.cluster_sizes) == 500
    
    def test_uses_feature_matrix_if_provided(self):
        """Should use feature matrix for clustering when available."""
        np.random.seed(42)
        n = 500
        s = np.random.uniform(0, 1, n)
        features = np.random.randn(n, 10)  # 10D features
        
        info = _perform_landmarking(s, features, n_landmarks=30)
        
        assert info.cluster_method in ["MiniBatchKMeans_PCA_optimized", "MiniBatchKMeans_PCA_standard"]
    
    def test_falls_back_to_1d_without_features(self):
        """Should use 1D clustering without feature matrix."""
        np.random.seed(42)
        s = np.random.uniform(0, 1, 500)
        
        info = _perform_landmarking(s, None, n_landmarks=30)
        
        assert info.cluster_method in ["MiniBatchKMeans_1D_optimized", "MiniBatchKMeans_1D_standard"]


class TestPrepareInputsWithLandmarks:
    """Test prepare_inputs with landmarking."""
    
    def test_prepare_inputs_auto_landmarks(self):
        """prepare_inputs should enable landmarks for large datasets."""
        np.random.seed(42)
        n_cells = 3000
        s = np.random.uniform(0, 1, n_cells)
        time_labels = np.random.choice([0, 1, 2], n_cells)
        
        prepared = prepare_inputs(s, time_labels, landmarks="auto")
        
        assert prepared.landmark_info.enabled is True
        assert prepared.landmark_info.n_landmarks >= 480  # Allow for clustering randomness
    
    def test_prepare_inputs_no_landmarks_small(self):
        """prepare_inputs should not enable landmarks for small datasets."""
        np.random.seed(42)
        n_cells = 1000
        s = np.random.uniform(0, 1, n_cells)
        time_labels = np.random.choice([0, 1, 2], n_cells)
        
        prepared = prepare_inputs(s, time_labels, landmarks="auto")
        
        assert prepared.landmark_info.enabled is False
    
    def test_weights_per_time_with_landmarks(self):
        """weights_per_time should reflect cluster membership per time."""
        np.random.seed(42)
        n_cells = 3000
        s = np.random.uniform(0, 1, n_cells)
        time_labels = np.random.choice([0, 1, 2], n_cells)
        
        prepared = prepare_inputs(s, time_labels, landmarks="on", n_landmarks=100)
        
        assert len(prepared.weights_per_time) == 3  # 3 time points
        
        # Each weight array should have length = n_landmarks
        for w in prepared.weights_per_time:
            assert len(w) == 100
        
        # Weights should sum to cells in that time point
        for k, t in enumerate(prepared.unique_times):
            expected_count = np.sum(time_labels == t)
            assert np.sum(prepared.weights_per_time[k]) == expected_count
    
    def test_landmarks_reproducible(self):
        """Same random_state should give same landmarks."""
        np.random.seed(42)
        n_cells = 3000
        s = np.random.uniform(0, 1, n_cells)
        time_labels = np.random.choice([0, 1, 2], n_cells)
        
        p1 = prepare_inputs(s, time_labels, landmarks="on", n_landmarks=50, random_state=123)
        p2 = prepare_inputs(s, time_labels, landmarks="on", n_landmarks=50, random_state=123)
        
        assert np.allclose(p1.landmark_info.landmark_s, p2.landmark_info.landmark_s)
        assert np.allclose(p1.landmark_info.cluster_labels, p2.landmark_info.cluster_labels)


class TestLandmarkConsistency:
    """Test that landmark mode gives consistent results with non-landmark mode."""
    
    def test_ecdf_computed_from_original_cells(self):
        """ECDF should always use original cell s values, not landmarks."""
        np.random.seed(42)
        n_cells = 1000
        s = np.random.uniform(0, 1, n_cells)
        time_labels = np.random.choice([0, 1], n_cells)
        
        # With and without landmarks
        prepared_off = prepare_inputs(s, time_labels, landmarks="off")
        prepared_on = prepare_inputs(s, time_labels, landmarks="on", n_landmarks=50)
        
        # s_per_time should be the same
        for k in range(2):
            assert np.allclose(
                np.sort(prepared_off.s_per_time[k]), 
                np.sort(prepared_on.s_per_time[k])
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
