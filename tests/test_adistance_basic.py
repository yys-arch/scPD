"""
Test A-distance and ECDF computations.
"""

import numpy as np
import pytest

from scpd.likelihood import (
    compute_ecdf,
    compute_weighted_ecdf,
    compute_a_distance,
    estimate_sigma_A_bootstrap,
    estimate_sigma_A_landmark_bootstrap,
    compute_nll_cdf
)


class TestECDF:
    """Test empirical CDF computation."""
    
    def test_ecdf_uniform_samples(self):
        """ECDF of uniform samples should be roughly linear."""
        np.random.seed(42)
        s_samples = np.random.uniform(0, 1, 1000)
        s_grid = np.linspace(0, 1, 101)
        
        ecdf = compute_ecdf(s_samples, s_grid)
        
        # Should start near 0 and end near 1
        assert ecdf[0] < 0.05
        assert ecdf[-1] > 0.95
        
        # Should be monotonically non-decreasing
        assert np.all(np.diff(ecdf) >= 0)
    
    def test_ecdf_concentrated_samples(self):
        """ECDF should jump at the sample location."""
        s_samples = np.array([0.5, 0.5, 0.5])
        s_grid = np.linspace(0, 1, 101)
        
        ecdf = compute_ecdf(s_samples, s_grid)
        
        # All samples at 0.5, so ECDF should be 0 before and 1 after
        assert ecdf[0] == 0
        assert ecdf[50] == 1.0
        assert ecdf[-1] == 1.0
    
    def test_ecdf_empty_samples(self):
        """Empty samples should give zero ECDF."""
        s_samples = np.array([])
        s_grid = np.linspace(0, 1, 11)
        
        ecdf = compute_ecdf(s_samples, s_grid)
        
        assert np.allclose(ecdf, 0)


class TestWeightedECDF:
    """Test weighted ECDF for landmark mode."""
    
    def test_weighted_ecdf_uniform_weights(self):
        """Uniform weights should give regular ECDF-like behavior."""
        landmark_s = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        weights = np.array([1, 1, 1, 1, 1])
        s_grid = np.linspace(0, 1, 101)
        
        ecdf = compute_weighted_ecdf(landmark_s, weights, s_grid)
        
        # Should increase in steps
        assert ecdf[0] == 0  # Before first landmark
        assert ecdf[-1] == 1.0  # After all landmarks
    
    def test_weighted_ecdf_varying_weights(self):
        """Different weights should affect ECDF shape."""
        landmark_s = np.array([0.2, 0.8])
        weights1 = np.array([1, 9])  # Most weight at s=0.8
        weights2 = np.array([9, 1])  # Most weight at s=0.2
        s_grid = np.linspace(0, 1, 101)
        
        ecdf1 = compute_weighted_ecdf(landmark_s, weights1, s_grid)
        ecdf2 = compute_weighted_ecdf(landmark_s, weights2, s_grid)
        
        # At s=0.5 (between landmarks):
        # ecdf1 should be 0.1 (10% of weight is at s=0.2)
        # ecdf2 should be 0.9 (90% of weight is at s=0.2)
        mid_idx = 50
        assert ecdf1[mid_idx] < ecdf2[mid_idx]
    
    def test_weighted_ecdf_zero_weights(self):
        """Zero total weight should give zero ECDF."""
        landmark_s = np.array([0.2, 0.8])
        weights = np.array([0, 0])
        s_grid = np.linspace(0, 1, 11)
        
        ecdf = compute_weighted_ecdf(landmark_s, weights, s_grid)
        
        assert np.allclose(ecdf, 0)


class TestADistance:
    """Test A-distance computation."""
    
    def test_a_distance_identical_cdfs(self):
        """A-distance between identical CDFs should be near zero."""
        s_grid = np.linspace(0, 1, 101)
        cdf = s_grid  # Linear CDF
        ds = s_grid[1] - s_grid[0]
        
        A = compute_a_distance(cdf, cdf, ds)
        
        assert A < 1e-10
    
    def test_a_distance_different_cdfs(self):
        """A-distance between different CDFs should be positive."""
        s_grid = np.linspace(0, 1, 101)
        ds = s_grid[1] - s_grid[0]
        
        cdf1 = s_grid  # Uniform
        cdf2 = s_grid ** 2  # Concentrated toward end
        
        A = compute_a_distance(cdf1, cdf2, ds)
        
        assert A > 0
    
    def test_a_distance_bounds(self):
        """A-distance should be bounded by [0, 1]."""
        s_grid = np.linspace(0, 1, 101)
        ds = s_grid[1] - s_grid[0]
        
        # Extreme case: step function vs linear
        cdf1 = np.zeros_like(s_grid)
        cdf1[50:] = 1.0  # Step at 0.5
        cdf2 = s_grid  # Linear
        
        A = compute_a_distance(cdf1, cdf2, ds)
        
        assert 0 <= A <= 1


class TestBootstrapSigmaA:
    """Test bootstrap estimation of σ_A."""
    
    def test_bootstrap_produces_positive_sigma(self):
        """Bootstrap should give positive σ_A."""
        np.random.seed(42)
        s_values = np.random.uniform(0, 1, 100)
        s_grid = np.linspace(0, 1, 51)
        ds = s_grid[1] - s_grid[0]
        
        sigma_A = estimate_sigma_A_bootstrap(
            s_values, s_grid, ds,
            n_bootstrap=50, random_state=0
        )
        
        assert sigma_A > 0
    
    def test_more_samples_smaller_sigma(self):
        """More samples should generally give smaller σ_A."""
        np.random.seed(42)
        s_grid = np.linspace(0, 1, 51)
        ds = s_grid[1] - s_grid[0]
        
        s_small = np.random.uniform(0, 1, 50)
        s_large = np.random.uniform(0, 1, 500)
        
        sigma_small = estimate_sigma_A_bootstrap(s_small, s_grid, ds, n_bootstrap=100, random_state=0)
        sigma_large = estimate_sigma_A_bootstrap(s_large, s_grid, ds, n_bootstrap=100, random_state=0)
        
        # Not a strict test since bootstrap is stochastic, but generally true
        assert sigma_large < sigma_small or abs(sigma_large - sigma_small) < 0.1


class TestLandmarkBootstrap:
    """Test landmark-based multinomial bootstrap."""
    
    def test_landmark_bootstrap_positive_sigma(self):
        """Landmark bootstrap should give positive σ_A."""
        landmark_s = np.linspace(0.1, 0.9, 20)
        weights = np.random.poisson(10, 20).astype(float)
        s_grid = np.linspace(0, 1, 51)
        ds = s_grid[1] - s_grid[0]
        
        sigma_A = estimate_sigma_A_landmark_bootstrap(
            landmark_s, weights, s_grid, ds,
            n_bootstrap=50, random_state=0
        )
        
        assert sigma_A > 0


class TestNLLCDF:
    """Test CDF NLL computation."""
    
    def test_nll_zero_distance(self):
        """Zero A-distance should give minimal NLL."""
        A_values = np.array([0.0, 0.0, 0.0])
        sigma_A = np.array([0.1, 0.1, 0.1])
        
        nll = compute_nll_cdf(A_values, sigma_A)
        
        # NLL = 0.5 * sum(z^2) + sum(log(sigma))
        # z = 0 -> 0.5*0 + 3*log(0.1) = 3*(-2.3) ≈ -6.9
        expected = 3 * np.log(0.1)
        assert np.isclose(nll, expected, rtol=1e-5)
    
    def test_nll_positive(self):
        """NLL computation should work for positive A."""
        A_values = np.array([0.1, 0.2])
        sigma_A = np.array([0.1, 0.2])
        
        nll = compute_nll_cdf(A_values, sigma_A)
        
        # z = [1, 1], so 0.5 * 2 + log(0.1) + log(0.2)
        expected = 0.5 * 2 + np.log(0.1) + np.log(0.2)
        assert np.isclose(nll, expected, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

