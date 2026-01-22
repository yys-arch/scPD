"""
Test basic fitting functionality on synthetic data.
"""

import numpy as np
import pytest
from scipy.interpolate import interp1d

from scpd.synthetic import generate_synthetic_dataset
from scpd.preprocess import prepare_inputs
from scpd.solver import PseudodynamicsModel
from scpd.utils import shape_similarity


def interpolate_to_grid(values, from_grid, to_grid):
    """Interpolate values from one grid to another."""
    f = interp1d(from_grid, values, kind='linear', fill_value='extrapolate')
    return f(to_grid)


class TestFitRecovery:
    """Test basic fitting on synthetic data."""
    
    def test_fit_completes_distribution_only(self):
        """Fitting in distribution_only mode should complete successfully."""
        dataset = generate_synthetic_dataset(
            n_times=4,
            n_cells_per_time=200,
            D_type="constant",
            v_type="linear",
            g_type="zero",
            random_state=42
        )
        
        prepared = prepare_inputs(
            s=dataset['s'],
            time_labels=dataset['time_labels'],
            time_values=dataset['time_values'],
            landmarks="off"
        )
        
        model = PseudodynamicsModel(n_grid=50, spline_df=4)
        result = model.fit(
            prepared,
            mode="distribution_only",
            rho=0.5,
            n_starts=2,
            n_bootstrap=20,
            random_state=0
        )
        
        assert np.isfinite(result.diagnostics.total_nll)
        assert result.g is not None
        assert np.allclose(result.g, 0), "g should be zero in distribution_only mode"
    
    def test_fit_completes_with_population(self):
        """Fitting in with_population mode should complete successfully."""
        dataset = generate_synthetic_dataset(
            n_times=4,
            n_cells_per_time=200,
            D_type="constant",
            v_type="constant",
            g_type="constant",
            g_scale=0.1,
            random_state=42
        )
        
        prepared = prepare_inputs(
            s=dataset['s'],
            time_labels=dataset['time_labels'],
            time_values=dataset['time_values'],
            N_obs=dataset['N_obs'],
            landmarks="off"
        )
        
        model = PseudodynamicsModel(n_grid=50, spline_df=4)
        result = model.fit(
            prepared,
            mode="with_population",
            rho=0.5,
            n_starts=2,
            n_bootstrap=20,
            random_state=0
        )
        
        assert np.isfinite(result.diagnostics.total_nll)
        # In with_population mode, g can be non-zero
        assert result.g is not None
    
    def test_developmental_potential_computed(self):
        """Developmental potential W should be computed correctly."""
        dataset = generate_synthetic_dataset(
            n_times=3,
            n_cells_per_time=100,
            random_state=42
        )
        
        prepared = prepare_inputs(
            s=dataset['s'],
            time_labels=dataset['time_labels'],
            landmarks="off"
        )
        
        model = PseudodynamicsModel(n_grid=50, spline_df=4)
        result = model.fit(prepared, mode="distribution_only", rho=1.0, n_starts=2, random_state=0)
        
        # W should be computed
        assert result.W is not None
        assert len(result.W) == 50
        
        # W(0) should be 0 (or close to it)
        assert abs(result.W[0]) < 0.1
    
    def test_result_save_load(self):
        """Results should save and load correctly."""
        import tempfile
        import os
        
        dataset = generate_synthetic_dataset(
            n_times=3,
            n_cells_per_time=100,
            random_state=42
        )
        
        prepared = prepare_inputs(
            s=dataset['s'],
            time_labels=dataset['time_labels'],
            landmarks="off"
        )
        
        model = PseudodynamicsModel(n_grid=50, spline_df=4)
        result = model.fit(prepared, mode="distribution_only", rho=1.0, n_starts=2, random_state=0)
        
        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "result.npz")
            result.save(path)
            
            from scpd.results import PseudodynamicsResult
            loaded = PseudodynamicsResult.load(path)
            
            assert np.allclose(result.D, loaded.D)
            assert np.allclose(result.v, loaded.v)
            assert np.allclose(result.W, loaded.W)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
