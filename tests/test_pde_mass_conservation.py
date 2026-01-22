"""
Test mass conservation in PDE solver.

When g=0 (no growth/death), total mass should be conserved:
    d/dt ∫u ds = ∫g*u ds = 0 if g=0
"""

import numpy as np
import pytest

from scpd.grid import create_grid
from scpd.pde import pde_rhs, compute_total_mass
from scpd.synthetic import simulate_forward


class TestMassConservation:
    """Test mass conservation with g=0."""
    
    def test_mass_conservation_constant_D_zero_v(self):
        """With constant D, v=0, g=0: pure diffusion conserves mass."""
        n_grid = 100
        s_grid, s_faces, ds = create_grid(n_grid)
        
        # Constant diffusion, no drift, no growth
        D = np.ones(n_grid) * 0.01
        v = np.zeros(n_grid)
        g = np.zeros(n_grid)
        
        # Initial condition: localized bump
        u0 = np.exp(-((s_grid - 0.3) ** 2) / 0.02)
        u0 /= np.sum(u0) * ds  # Normalize
        
        initial_mass = compute_total_mass(u0, ds)
        
        # Simulate
        times = np.linspace(0, 1, 11)
        u = simulate_forward(D, v, g, s_grid, ds, times, u0=u0)
        
        # Check mass at each time
        for k in range(len(times)):
            mass_k = compute_total_mass(u[:, k], ds)
            rel_error = abs(mass_k - initial_mass) / initial_mass
            assert rel_error < 0.01, f"Mass not conserved at t={times[k]}: {rel_error:.4f}"
    
    def test_mass_conservation_with_drift(self):
        """With non-zero v but g=0: mass still conserved."""
        n_grid = 100
        s_grid, s_faces, ds = create_grid(n_grid)
        
        D = np.ones(n_grid) * 0.01
        v = np.ones(n_grid) * 0.3  # Constant positive drift
        g = np.zeros(n_grid)
        
        # Initial condition
        u0 = np.exp(-((s_grid - 0.2) ** 2) / 0.02)
        u0 /= np.sum(u0) * ds
        
        initial_mass = compute_total_mass(u0, ds)
        
        times = np.linspace(0, 0.5, 6)
        u = simulate_forward(D, v, g, s_grid, ds, times, u0=u0)
        
        for k in range(len(times)):
            mass_k = compute_total_mass(u[:, k], ds)
            rel_error = abs(mass_k - initial_mass) / initial_mass
            assert rel_error < 0.02, f"Mass not conserved at t={times[k]}: {rel_error:.4f}"
    
    def test_mass_conservation_variable_D(self):
        """With spatially varying D and v, g=0: mass conserved."""
        n_grid = 100
        s_grid, s_faces, ds = create_grid(n_grid)
        
        D = 0.005 + 0.01 * s_grid  # Increasing D
        v = 0.2 * (1 - s_grid)    # Decreasing v
        g = np.zeros(n_grid)
        
        u0 = np.exp(-((s_grid - 0.3) ** 2) / 0.02)
        u0 /= np.sum(u0) * ds
        
        initial_mass = compute_total_mass(u0, ds)
        
        times = np.linspace(0, 0.5, 6)
        u = simulate_forward(D, v, g, s_grid, ds, times, u0=u0)
        
        for k in range(len(times)):
            mass_k = compute_total_mass(u[:, k], ds)
            rel_error = abs(mass_k - initial_mass) / initial_mass
            assert rel_error < 0.02, f"Mass not conserved at t={times[k]}: {rel_error:.4f}"
    
    def test_mass_increases_with_positive_g(self):
        """With positive g, total mass should increase."""
        n_grid = 100
        s_grid, s_faces, ds = create_grid(n_grid)
        
        D = np.ones(n_grid) * 0.01
        v = np.zeros(n_grid)
        g = np.ones(n_grid) * 0.5  # Positive growth
        
        u0 = np.exp(-((s_grid - 0.5) ** 2) / 0.05)
        u0 /= np.sum(u0) * ds
        
        initial_mass = compute_total_mass(u0, ds)
        
        times = np.linspace(0, 0.5, 6)
        u = simulate_forward(D, v, g, s_grid, ds, times, u0=u0)
        
        final_mass = compute_total_mass(u[:, -1], ds)
        assert final_mass > initial_mass, "Mass should increase with positive g"
    
    def test_mass_decreases_with_negative_g(self):
        """With negative g, total mass should decrease."""
        n_grid = 100
        s_grid, s_faces, ds = create_grid(n_grid)
        
        D = np.ones(n_grid) * 0.01
        v = np.zeros(n_grid)
        g = np.ones(n_grid) * (-0.5)  # Negative growth (death)
        
        u0 = np.exp(-((s_grid - 0.5) ** 2) / 0.05)
        u0 /= np.sum(u0) * ds
        
        initial_mass = compute_total_mass(u0, ds)
        
        times = np.linspace(0, 0.5, 6)
        u = simulate_forward(D, v, g, s_grid, ds, times, u0=u0)
        
        final_mass = compute_total_mass(u[:, -1], ds)
        assert final_mass < initial_mass, "Mass should decrease with negative g"


class TestNoFluxBoundary:
    """Test that no-flux boundary conditions are respected."""
    
    def test_boundary_flux_zero(self):
        """Verify flux at boundaries is zero."""
        from scpd.pde import compute_flux_at_faces
        
        n_grid = 50
        s_grid, s_faces, ds = create_grid(n_grid)
        
        # Various density profiles
        for u_type in ["uniform", "left_peak", "right_peak"]:
            if u_type == "uniform":
                u = np.ones(n_grid)
            elif u_type == "left_peak":
                u = np.exp(-((s_grid - 0.1) ** 2) / 0.01)
            else:
                u = np.exp(-((s_grid - 0.9) ** 2) / 0.01)
            
            D = np.ones(n_grid) * 0.01
            v = np.ones(n_grid) * 0.2
            
            J = compute_flux_at_faces(u, D, v, ds)
            
            assert J[0] == 0, f"Left boundary flux should be 0 for {u_type}"
            assert J[-1] == 0, f"Right boundary flux should be 0 for {u_type}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

