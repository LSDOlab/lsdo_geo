"""Unit and regression tests for ISA atmosphere, flight conditions, and Korn-Lock wave drag."""

import numpy as np
import pytest
import csdl_alpha as csdl

from examples.showcase_examples.rectangular_wing.physics_models.flight_conditions import (
    compute_isa_troposphere,
    get_nominal_flight_conditions,
    CRUISE_ALTITUDE_M,
    CRUISE_MACH,
    SIZING_EQUIVALENT_SPEED_FACTOR,
)
from examples.showcase_examples.rectangular_wing.physics_models.wave_drag import (
    setup_strip_projection_points,
    compute_half_chord_sweep,
    compute_strip_thickness_to_chord,
    evaluate_wave_drag,
    KAPPA_A,
    DELTA_M_DD_TO_CRIT,
)


def test_atmosphere_regression():
    """Verify atmospheric condition table from AGENTS.md to floating-point tolerance."""
    conds = get_nominal_flight_conditions()
    cruise = conds['cruise']
    sizing = conds['sizing']

    # Cruise / stability at 9,144 m (30,000 ft)
    assert np.isclose(cruise['altitude_m'], 9144.0, atol=1e-3)
    assert np.isclose(cruise['temperature_K'], 228.714, atol=1e-3)
    assert np.isclose(cruise['density_kg_m3'], 0.458312, atol=1e-4)
    assert np.isclose(cruise['speed_of_sound_m_s'], 303.174, atol=1e-3)
    assert np.isclose(cruise['speed_m_s'], 212.221, atol=1e-3)
    assert np.isclose(cruise['mach'], 0.70, atol=1e-5)
    assert np.isclose(cruise['dynamic_pressure_Pa'], 10320.720, atol=1e-2)

    # Sea-level sizing at 0 m
    assert np.isclose(sizing['altitude_m'], 0.0, atol=1e-3)
    assert np.isclose(sizing['temperature_K'], 288.15, atol=1e-3)
    assert np.isclose(sizing['density_kg_m3'], 1.225000, atol=1e-4)
    assert np.isclose(sizing['speed_of_sound_m_s'], 340.294, atol=1e-3)
    assert np.isclose(sizing['speed_m_s'], 162.260, atol=1e-3)
    assert np.isclose(sizing['mach'], 0.476824, atol=1e-4)
    assert np.isclose(sizing['dynamic_pressure_Pa'], 16126.125, atol=1e-2)

    # Ratio q_sizing / q_cruise == 1.25^2 = 1.5625
    q_ratio = sizing['dynamic_pressure_Pa'] / cruise['dynamic_pressure_Pa']
    assert np.isclose(q_ratio, 1.5625, atol=1e-5)


def test_wave_drag_equations_and_trends():
    """Verify Korn-Lock equations, softplus near-zero pre-onset, and sensitivity trends."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    num_strips = 10
    tc_val = 0.12
    sweep_val = np.radians(20.0)
    cl_val = 0.40
    mach_val = 0.75
    q_val = 11847.765
    area_val = np.ones(num_strips)
    tot_area_val = float(num_strips)
    sref_val = 10.0

    tc = csdl.Variable(value=np.full(num_strips, tc_val))
    sweep = csdl.Variable(value=np.full(num_strips, sweep_val))
    cl = csdl.Variable(value=np.full(num_strips, cl_val))
    area = csdl.Variable(value=area_val)
    tot_area = csdl.Variable(value=tot_area_val)
    sref = csdl.Variable(value=sref_val)

    # 1. Base evaluation
    res = evaluate_wave_drag(
        strip_tc=tc,
        strip_sweep_halfchord=sweep,
        strip_cl=cl,
        strip_area=area,
        total_strip_area=tot_area,
        mach_node=mach_val,
        q_node=q_val,
        planform_area=sref,
    )

    cos_L = np.cos(sweep_val)
    mdd_expected = KAPPA_A / cos_L - tc_val / (cos_L**2) - cl_val / (10.0 * (cos_L**3))
    mcrit_expected = mdd_expected - DELTA_M_DD_TO_CRIT
    delta_m_expected = mach_val - mcrit_expected
    delta_m_eff_expected = np.log(1.0 + np.exp(100.0 * delta_m_expected)) / 100.0
    cd_expected = 20.0 * (delta_m_eff_expected**4)

    assert np.isclose(float(res['M_dd'].value[0]), mdd_expected, atol=1e-5)
    assert np.isclose(float(res['M_crit'].value[0]), mcrit_expected, atol=1e-5)
    assert np.isclose(float(res['delta_M'].value[0]), delta_m_expected, atol=1e-5)
    assert np.isclose(float(res['delta_M_eff'].value[0]), delta_m_eff_expected, atol=1e-5)
    assert np.isclose(float(res['cd_wave_strip'].value[0]), cd_expected, atol=1e-7)

    # 2. Well below onset (M << Mcrit): wave drag must be effectively zero (< 1e-12)
    res_sub = evaluate_wave_drag(
        strip_tc=tc,
        strip_sweep_halfchord=sweep,
        strip_cl=cl,
        strip_area=area,
        total_strip_area=tot_area,
        mach_node=0.50,
        q_node=q_val,
        planform_area=sref,
    )
    assert float(res_sub['CD_wave'].value[0]) < 1.0e-12

    # 3. Higher Mach increases wave drag
    res_sup = evaluate_wave_drag(
        strip_tc=tc,
        strip_sweep_halfchord=sweep,
        strip_cl=cl,
        strip_area=area,
        total_strip_area=tot_area,
        mach_node=0.80,
        q_node=q_val,
        planform_area=sref,
    )
    assert float(res_sup['CD_wave'].value[0]) > float(res['CD_wave'].value[0])

    # 4. Thicker airfoil increases wave drag
    tc_thick = csdl.Variable(value=np.full(num_strips, 0.16))
    res_thick = evaluate_wave_drag(
        strip_tc=tc_thick,
        strip_sweep_halfchord=sweep,
        strip_cl=cl,
        strip_area=area,
        total_strip_area=tot_area,
        mach_node=mach_val,
        q_node=q_val,
        planform_area=sref,
    )
    assert float(res_thick['CD_wave'].value[0]) > float(res['CD_wave'].value[0])


def test_half_chord_sweep_computation():
    """Verify half-chord sweep derivation from mid-chord points."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    num_strips = 20
    # Linear sweep with slope dy/dx = 2.0 => sweep = atan2(dx, dy) = atan2(0.5, 1.0)
    y_coords = np.linspace(0.5, 20.0, num_strips)
    x_coords = 0.5 * y_coords
    r_pts = np.column_stack([x_coords, y_coords, np.zeros(num_strips)])

    r_var = csdl.Variable(value=r_pts)
    sweep_var = compute_half_chord_sweep(r_var)

    expected_sweep = np.arctan2(0.5, 1.0)
    sweep_vals = np.asarray(sweep_var.value)
    assert np.allclose(sweep_vals, expected_sweep, atol=1e-5)


def test_wave_drag_distribution_processing():
    """Verify strip wave drag processing, spline smoothing, and integration."""
    from examples.showcase_examples.rectangular_wing.optimization_analyses.extract_wave_drag_distribution import (
        process_wave_drag_strips,
        compute_korn_lock_numpy,
    )
    num_strips = 50
    y_strip_pts = np.linspace(0.2, 30.0, num_strips)
    tc = np.full(num_strips, 0.12)
    sweep = np.full(num_strips, np.radians(25.0))
    cl = np.full(num_strips, 0.45)
    mach = 0.70

    # Korn-Lock numpy evaluation
    kl_res = compute_korn_lock_numpy(tc, sweep, cl, mach_cruise=mach)
    assert kl_res['cd_wave_strip'].shape == (num_strips,)
    assert np.all(kl_res['M_crit'] < kl_res['M_dd'])

    # Strip processing
    res = process_wave_drag_strips(
        y_strip_pts=y_strip_pts,
        strip_wave_cd=kl_res['cd_wave_strip'],
        strip_M_crit=kl_res['M_crit'],
        strip_M_dd=kl_res['M_dd'],
        strip_tc=tc,
        strip_sweep_halfchord=sweep,
        strip_cl=cl,
        mach_cruise=mach,
        q_cruise=10320.72,
        sref_val=500.0,
    )

    assert len(res['y_fine']) == 300
    assert len(res['dD_fine']) == 300
    assert res['tot_d_wave_total'] >= 0.0
    assert res['integrated_cd_wave'] >= 0.0
    assert np.isclose(res['tot_d_wave_total'], 2.0 * res['tot_d_wave_half'])


