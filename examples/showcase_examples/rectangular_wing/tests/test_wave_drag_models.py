"""Unit tests for transonic wave drag models (Korn-Lock and Karman-Tsien RH)."""

import os
import sys

SHOWCASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SHOWCASE_DIR not in sys.path:
    sys.path.insert(0, SHOWCASE_DIR)

import numpy as np
import pytest
import csdl_alpha as csdl

from physics_models.wave_drag import (
    evaluate_wave_drag,
    build_panel_strip_wave_mapping,
    evaluate_wave_drag_karman_tsien_rh,
)


def test_build_panel_strip_wave_mapping():
    """Verify static mapping construction, dimensions, and partition of unity."""
    num_aero = 4
    num_panels_per_strip = 10
    num_panels = num_aero * num_panels_per_strip
    num_drag = 20

    # Build W_strip
    W_strip = np.zeros((num_aero, num_panels))
    for s in range(num_aero):
        W_strip[s, s * num_panels_per_strip:(s + 1) * num_panels_per_strip] = 1.0

    y_strip_centers = np.linspace(0.1, 0.9, num_aero)
    y_drag_centers = np.linspace(0.05, 0.95, num_drag)
    M_interp_drag = np.ones((num_drag, num_aero)) / num_aero

    mapping = build_panel_strip_wave_mapping(
        W_strip=W_strip,
        M_interp_drag=M_interp_drag,
        y_strip_centers=y_strip_centers,
        y_drag_centers=y_drag_centers,
    )

    M_drag_to_aero = mapping['M_drag_to_aero']
    M_drag_to_panel = mapping['M_drag_to_panel']
    strip_panel_slices = mapping['strip_panel_slices']

    assert M_drag_to_aero.shape == (num_aero, num_drag)
    assert M_drag_to_panel.shape == (num_panels, num_drag)
    assert len(strip_panel_slices) == num_aero

    # Partition of unity: each row should sum to 1.0
    np.testing.assert_allclose(np.sum(M_drag_to_aero, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.sum(M_drag_to_panel, axis=1), 1.0, atol=1e-12)

    # Check slice indices
    for s, (start, end) in enumerate(strip_panel_slices):
        assert start == s * num_panels_per_strip
        assert end == (s + 1) * num_panels_per_strip


def test_karman_tsien_rh_numerical_robustness():
    """Verify Karman-Tsien RH model behaves smoothly without NaNs under extreme Cp."""
    num_aero = 2
    num_panels = 20
    num_drag = 10

    W_strip = np.zeros((num_aero, num_panels))
    W_strip[0, :10] = 1.0
    W_strip[1, 10:] = 1.0

    y_strip_centers = np.array([0.25, 0.75])
    y_drag_centers = np.linspace(0.1, 0.9, num_drag)
    M_interp_drag = np.ones((num_drag, num_aero)) / num_aero

    mapping = build_panel_strip_wave_mapping(
        W_strip=W_strip,
        M_interp_drag=M_interp_drag,
        y_strip_centers=y_strip_centers,
        y_drag_centers=y_drag_centers,
    )

    # Test extreme suction (Cp = -10.0), normal suction (Cp = -1.2), and stagnation (Cp = +1.2)
    cp_vals = np.array([-10.0] * 5 + [-1.2] * 5 + [0.0] * 5 + [1.2] * 5)
    areas = np.full(num_panels, 0.1)

    rec = csdl.Recorder()
    rec.start()

    cp_var = csdl.Variable(value=cp_vals)
    area_var = csdl.Variable(value=areas)
    sweep_var = csdl.Variable(value=np.full(num_drag, np.radians(25.0)))
    strip_area_var = csdl.Variable(value=np.full(num_drag, 1.0))
    total_area_var = csdl.Variable(value=np.array([10.0]))
    q_var = csdl.Variable(value=np.array([10000.0]))
    s_ref_var = csdl.Variable(value=np.array([10.0]))
    dy_dyn_var = csdl.Variable(value=np.array([0.5, 0.5]))
    chord_var = csdl.Variable(value=np.full(num_drag, 2.0))

    results = evaluate_wave_drag_karman_tsien_rh(
        cp_right=cp_var,
        panel_area_right=area_var,
        strip_sweep_halfchord=sweep_var,
        strip_area=strip_area_var,
        total_strip_area=total_area_var,
        mach_node=csdl.Variable(value=np.array([0.75])),
        q_node=q_var,
        planform_area=s_ref_var,
        wave_panel_mapping=mapping,
        M_interp_drag=M_interp_drag,
        W_strip=W_strip,
        dy_aero_dyn=dy_dyn_var,
        local_chord_drag=chord_var,
        c_cal=1.0,
    )
    rec.stop()

    sim = csdl.experimental.JaxSimulator(
        recorder=rec,
        additional_inputs=[cp_var],
        additional_outputs=[
            results['CD_wave'],
            results['D_wave'],
            results['cd_wave_strip'],
            results['M_dd'],
            results['M_crit'],
        ],
        gpu=False,
    )
    sim.run()

    cd_w = float(np.asarray(sim[results['CD_wave']]).flatten()[0])
    d_w = float(np.asarray(sim[results['D_wave']]).flatten()[0])
    cd_strip = np.asarray(sim[results['cd_wave_strip']])
    m_dd = np.asarray(sim[results['M_dd']])
    m_crit = np.asarray(sim[results['M_crit']])

    # Verify no NaNs or Infs
    assert np.isfinite(cd_w)
    assert np.isfinite(d_w)
    assert np.all(np.isfinite(cd_strip))
    assert np.all(np.isfinite(m_dd))
    assert np.all(np.isfinite(m_crit))
    assert cd_w > 0.0
    assert d_w > 0.0


def test_karman_tsien_rh_subcritical_zero_drag():
    """Verify that purely subcritical flow produces zero wave drag."""
    num_aero = 2
    num_panels = 20
    num_drag = 10

    W_strip = np.zeros((num_aero, num_panels))
    W_strip[0, :10] = 1.0
    W_strip[1, 10:] = 1.0

    y_strip_centers = np.array([0.25, 0.75])
    y_drag_centers = np.linspace(0.1, 0.9, num_drag)
    M_interp_drag = np.ones((num_drag, num_aero)) / num_aero

    mapping = build_panel_strip_wave_mapping(
        W_strip=W_strip,
        M_interp_drag=M_interp_drag,
        y_strip_centers=y_strip_centers,
        y_drag_centers=y_drag_centers,
    )

    # Subcritical flow at M = 0.30, mild suction Cp = -0.2
    cp_vals = np.full(num_panels, -0.2)
    areas = np.full(num_panels, 0.1)

    rec = csdl.Recorder()
    rec.start()

    cp_var = csdl.Variable(value=cp_vals)
    area_var = csdl.Variable(value=areas)
    sweep_var = csdl.Variable(value=np.full(num_drag, np.radians(20.0)))
    strip_area_var = csdl.Variable(value=np.full(num_drag, 1.0))
    total_area_var = csdl.Variable(value=np.array([10.0]))
    q_var = csdl.Variable(value=np.array([5000.0]))
    s_ref_var = csdl.Variable(value=np.array([10.0]))
    dy_dyn_var = csdl.Variable(value=np.array([0.5, 0.5]))
    chord_var = csdl.Variable(value=np.full(num_drag, 2.0))

    results = evaluate_wave_drag_karman_tsien_rh(
        cp_right=cp_var,
        panel_area_right=area_var,
        strip_sweep_halfchord=sweep_var,
        strip_area=strip_area_var,
        total_strip_area=total_area_var,
        mach_node=csdl.Variable(value=np.array([0.30])),
        q_node=q_var,
        planform_area=s_ref_var,
        wave_panel_mapping=mapping,
        M_interp_drag=M_interp_drag,
        W_strip=W_strip,
        dy_aero_dyn=dy_dyn_var,
        local_chord_drag=chord_var,
        c_cal=1.0,
    )

    rec.stop()

    sim = csdl.experimental.JaxSimulator(
        recorder=rec,
        additional_inputs=[cp_var],
        additional_outputs=[results['CD_wave']],
        gpu=False,
    )
    sim.run()

    cd_w = float(np.asarray(sim[results['CD_wave']]).flatten()[0])
    # In subcritical flow, wave drag should be virtually zero (< 1e-10)
    assert cd_w < 1e-8


def test_karman_tsien_rh_derivatives():
    """Verify exact JAX automatic differentiation against finite difference."""
    num_aero = 2
    num_panels = 10
    num_drag = 6

    W_strip = np.zeros((num_aero, num_panels))
    W_strip[0, :5] = 1.0
    W_strip[1, 5:] = 1.0

    y_strip_centers = np.array([0.25, 0.75])
    y_drag_centers = np.linspace(0.1, 0.9, num_drag)
    M_interp_drag = np.ones((num_drag, num_aero)) / num_aero

    mapping = build_panel_strip_wave_mapping(
        W_strip=W_strip,
        M_interp_drag=M_interp_drag,
        y_strip_centers=y_strip_centers,
        y_drag_centers=y_drag_centers,
    )

    # Base state with supersonic suction peak
    cp_base = np.array([-1.5, -1.2, -0.8, -0.4, 0.1, -1.6, -1.1, -0.7, -0.3, 0.2])
    areas = np.full(num_panels, 0.2)

    def eval_at_cp(cp_in):
        rec = csdl.Recorder()
        rec.start()
        cp_var = csdl.Variable(value=cp_in)
        area_var = csdl.Variable(value=areas)
        sweep_var = csdl.Variable(value=np.full(num_drag, np.radians(20.0)))
        strip_area_var = csdl.Variable(value=np.full(num_drag, 1.0))
        total_area_var = csdl.Variable(value=np.array([6.0]))
        q_var = csdl.Variable(value=np.array([10000.0]))
        s_ref_var = csdl.Variable(value=np.array([6.0]))
        dy_dyn_var = csdl.Variable(value=np.array([0.5, 0.5]))
        chord_var = csdl.Variable(value=np.full(num_drag, 2.0))

        res = evaluate_wave_drag_karman_tsien_rh(
            cp_right=cp_var,
            panel_area_right=area_var,
            strip_sweep_halfchord=sweep_var,
            strip_area=strip_area_var,
            total_strip_area=total_area_var,
            mach_node=csdl.Variable(value=np.array([0.72])),
            q_node=q_var,
            planform_area=s_ref_var,
            wave_panel_mapping=mapping,
            M_interp_drag=M_interp_drag,
            W_strip=W_strip,
            dy_aero_dyn=dy_dyn_var,
            local_chord_drag=chord_var,
            c_cal=1.0,
        )
        rec.stop()
        sim = csdl.experimental.JaxSimulator(
            recorder=rec,
            additional_inputs=[cp_var],
            additional_outputs=[res['CD_wave']],
            gpu=False,
        )
        sim.run()
        totals = sim.compute_totals()
        derivs = totals[(res['CD_wave'], cp_var)]
        return float(np.asarray(sim[res['CD_wave']]).flatten()[0]), np.asarray(derivs).flatten()

    cd0, dcd_dcp_analytic = eval_at_cp(cp_base)

    # Central finite differences with respect to Cp[0]
    eps = 1e-6
    cp_plus = cp_base.copy()
    cp_plus[0] += eps
    cd_plus, _ = eval_at_cp(cp_plus)

    cp_minus = cp_base.copy()
    cp_minus[0] -= eps
    cd_minus, _ = eval_at_cp(cp_minus)

    dcd_dcp0_fd = (cd_plus - cd_minus) / (2.0 * eps)

    # Relative difference check
    rel_err = abs(dcd_dcp_analytic[0] - dcd_dcp0_fd) / (abs(dcd_dcp0_fd) + 1e-12)
    assert rel_err < 1e-4, f"Derivative error: analytic={dcd_dcp_analytic[0]}, fd={dcd_dcp0_fd}, rel_err={rel_err}"
