from __future__ import annotations

import os
import sys

SHOWCASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SHOWCASE_DIR not in sys.path:
    sys.path.insert(0, SHOWCASE_DIR)

import pytest
import numpy as np
import csdl_alpha as csdl

from physics_models.geonic_subsystems import (
    compute_actuator_sizing,
    ACTUATOR_BASE_DIMS_M,
    ACTUATOR_HEIGHT_FLOOR,
)


def test_level1_actuator_sizing_bounds_and_values():
    """Verify that Level 1 actuator scaling respects the physical packaging floor
    and produces realistic dimensions across different chord stations."""
    rec = csdl.Recorder(inline=True)
    rec.start()

    c_ref = 7.5
    scale_factor = 7.5

    # Test chords: root/reference (7.5m), mid-span (6.0m), outboard (4.0m), small (1.0m), near-zero (0.01m)
    test_chords = [7.5, 6.0, 4.0, 1.0, 0.01]
    c_vars = [csdl.Variable(value=np.array([c])) for c in test_chords]
    hl_vars = [
        compute_actuator_sizing(c_v, c_ref_local=c_ref, scale_factor=scale_factor, dynamic_sizing=True, sizing_method='level1_hinge_moment')
        for c_v in c_vars
    ]

    sim = csdl.experimental.JaxSimulator(
        rec,
        additional_inputs=c_vars,
        additional_outputs=hl_vars,
    )
    sim.run()

    H_base = ACTUATOR_BASE_DIMS_M[2]
    H_min_floor = H_base * ACTUATOR_HEIGHT_FLOOR

    for i, c_val in enumerate(test_chords):
        dims = 2.0 * np.asarray(sim[hl_vars[i]]).flatten()
        L, W, H = dims[0], dims[1], dims[2]
        
        # 1. Height must never shrink below the physical floor H_min_floor
        assert H >= H_min_floor - 1e-6, f"Height {H:.4f}m shrank below floor {H_min_floor:.4f}m for chord {c_val}m"
        
        # 2. At reference chord (7.5m), dimensions should equal base dimensions
        if np.isclose(c_val, c_ref):
            assert np.allclose(dims, ACTUATOR_BASE_DIMS_M, atol=1e-5)
            
        # 3. Dimensions must be positive and physically realistic
        assert 0.2 <= L <= 1.2, f"Length {L} out of realistic range"
        assert 0.08 <= W <= 0.40, f"Width {W} out of realistic range"
        assert 0.03 <= H <= 0.20, f"Height {H} out of realistic range"

    # Outboard station (4.0m) should scale according to Level 1 formula
    dims_4m = 2.0 * np.asarray(sim[hl_vars[2]]).flatten()
    expected_H_4m = ACTUATOR_BASE_DIMS_M[2] * (ACTUATOR_HEIGHT_FLOOR + (1.0 - ACTUATOR_HEIGHT_FLOOR) * np.sqrt(4.0 / 7.5))
    assert np.isclose(dims_4m[2], expected_H_4m, atol=1e-3)


def test_level1_actuator_derivatives():
    """Verify that JAX analytical derivatives of Level 1 actuator scaling match centered FD."""
    rec = csdl.Recorder(inline=True)
    rec.start()

    c_local = csdl.Variable(value=4.0)
    c_ref = 7.5
    scale_factor = 7.5

    act_hl = compute_actuator_sizing(
        c_local=c_local,
        c_ref_local=c_ref,
        scale_factor=scale_factor,
        dynamic_sizing=True,
        sizing_method='level1_hinge_moment',
    )

    sim = csdl.experimental.JaxSimulator(
        rec,
        additional_inputs=[c_local],
        additional_outputs=[act_hl],
    )
    sim.run()
    totals = sim.compute_totals()

    d_act_dc_analytic = np.asarray(totals[(act_hl, c_local)]).flatten()

    h = 1e-5
    sim[c_local] = 4.0 + h
    sim.run()
    f_plus = np.asarray(sim[act_hl]).flatten()

    sim[c_local] = 4.0 - h
    sim.run()
    f_minus = np.asarray(sim[act_hl]).flatten()

    d_act_dc_fd = (f_plus - f_minus) / (2.0 * h)

    assert np.allclose(d_act_dc_analytic, d_act_dc_fd, rtol=1e-5)
    # Derivative w.r.t chord must be positive for all 3 dimensions
    assert np.all(d_act_dc_analytic > 0.0)
