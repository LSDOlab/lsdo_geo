import numpy as np
import pytest


def test_simpson_quadrature_exactness():
    """Verify that the 5-point composite Simpson's rule over [0.20, 0.60] integrates polynomials up to degree 3 exactly."""
    x_fracs_box = np.array([0.20, 0.30, 0.40, 0.50, 0.60])
    weights = np.array([1.0, 4.0, 2.0, 4.0, 1.0]) / 12.0
    
    # 1. Weights sum to 1.0
    np.testing.assert_allclose(np.sum(weights), 1.0, atol=1e-15)
    
    # 2. Test quadratic polynomial: t(x) = 3x^2 - 2x + 1
    # Analytical integral of t(x) over [0.2, 0.6]:
    # [x^3 - x^2 + x] from 0.2 to 0.6 = (0.216 - 0.36 + 0.6) - (0.008 - 0.04 + 0.2) = 0.456 - 0.168 = 0.288
    # Average = 0.288 / 0.40 = 0.720
    poly_quad = lambda x: 3.0 * x**2 - 2.0 * x + 1.0
    quad_avg = np.sum(weights * poly_quad(x_fracs_box))
    np.testing.assert_allclose(quad_avg, 0.720, atol=1e-14)
    
    # 3. Test cubic polynomial: t(x) = 4x^3
    # Analytical integral: [x^4] from 0.2 to 0.6 = 0.1296 - 0.0016 = 0.1280
    # Average = 0.1280 / 0.40 = 0.320
    poly_cube = lambda x: 4.0 * x**3
    cube_avg = np.sum(weights * poly_cube(x_fracs_box))
    np.testing.assert_allclose(cube_avg, 0.320, atol=1e-14)


def test_naca0012_baseline_weighted_box_height():
    """Verify that on the baseline NACA 0012 profile, the weighted average thickness across [0.20, 0.60] matches analytical expectation."""
    x_fracs_box = np.array([0.20, 0.30, 0.40, 0.50, 0.60])
    weights = np.array([1.0, 4.0, 2.0, 4.0, 1.0]) / 12.0
    
    # Standard NACA 0012 thickness formula (t/c = 0.12)
    def naca0012_t(x):
        return 10.0 * 0.12 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    
    t_vals = naca0012_t(x_fracs_box)
    t_weighted = np.sum(weights * t_vals)
    t_quarter = naca0012_t(0.25)
    
    # Quarter chord thickness is ~0.1188c
    assert 0.118 < t_quarter < 0.119
    # Weighted average is ~0.1118c (reflecting rear taper)
    assert 0.111 < t_weighted < 0.112
    # Ratio is ~94.1%
    ratio = t_weighted / t_quarter
    assert 0.93 < ratio < 0.95


def test_ex_rectangular_wing_bwb_box_dimensions():
    """Verify box dimensions in ex_rectangular_wing_to_bwb without optimization."""
    import os, sys
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.join(repo_root, "examples/showcase_examples/rectangular_wing"))
    os.environ['SKIP_OPTIMIZATION'] = '1'
    os.environ['USE_GEONIC'] = '0'
    
    import examples.showcase_examples.rectangular_wing.ex_rectangular_wing_to_bwb as bwb
    
    # Verify weighting matrix properties
    assert hasattr(bwb, 'W_box_mat')
    W = bwb.W_box_mat
    assert W.shape == (bwb.num_beam_nodes, 5 * bwb.num_beam_nodes)
    # Every row of W sums to 1.0
    np.testing.assert_allclose(np.sum(W, axis=1), np.ones(bwb.num_beam_nodes), atol=1e-15)
    
    # Verify unscaled box_height equals local_height
    assert bwb.box_height is bwb.local_height
    
    # Check evaluated outputs from jax_sim
    sim = bwb.jax_sim
    sim.run()
    
    h_elem = np.asarray(sim[bwb.local_height]).flatten()
    w_elem = np.asarray(sim[bwb.box_width]).flatten()
    c_elem = np.asarray(sim[bwb.local_chord]).flatten()
    
    # All dimensions positive and non-empty
    assert len(h_elem) == bwb.num_beam_nodes - 1
    assert len(w_elem) == bwb.num_beam_nodes - 1
    assert np.all(h_elem > 0)
    assert np.all(w_elem > 0)
    assert np.all(c_elem > 0)
    
    # Width is exactly 40% of chord
    np.testing.assert_allclose(w_elem, 0.40 * c_elem, rtol=1e-5)
    
    # Height is approximately 0.1118 * chord (NACA 0012 weighted average)
    scale = bwb.scale_factor
    expected_height = 0.1118 * scale
    np.testing.assert_allclose(h_elem[0], expected_height, rtol=0.01)
