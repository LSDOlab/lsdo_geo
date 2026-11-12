"""
Unit and derivative verification tests for physics_models/geonic_subsystems.py.
"""
from __future__ import annotations

import os
import sys

import csdl_alpha as csdl
import numpy as np
import pytest

SHOWCASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SHOWCASE_DIR not in sys.path:
    sys.path.insert(0, SHOWCASE_DIR)

from physics_models.geonic_subsystems import (
    BoxSignedDistanceFunction,
    aggregate_constraints_by_mode,
    build_box_bspline_geometry,
    build_box_representation,
    build_box_surface_samples,
    build_local_frame_from_chord_vector,
    build_payload_box_sdfs,
    build_payload_representations,
    compute_actuator_sizing,
    compute_landing_gear_sizing,
    evaluate_exterior_non_interference,
    get_box_corners_3d,
    parse_geonic_aggregation_mode,
    parse_geonic_subsystem_mode,
    plot_geometry_with_payloads,
)


def test_subsystem_mode_parsers():
    assert parse_geonic_subsystem_mode(env={}) == 'none'
    assert parse_geonic_subsystem_mode(env={'GEONIC_SUBSYSTEM_MODE': 'landing_gear'}) == 'landing_gear'
    assert parse_geonic_subsystem_mode(mode='FULL') == 'full'
    with pytest.raises(ValueError):
        parse_geonic_subsystem_mode(mode='invalid')

    assert parse_geonic_aggregation_mode(env={}) == 'partial_aggregation'
    assert parse_geonic_aggregation_mode(env={'GEONIC_AGGREGATION_MODE': 'pairwise'}) == 'pairwise'
    with pytest.raises(ValueError):
        parse_geonic_aggregation_mode(mode='bad_agg')


def test_box_sdf_values_and_rotation():
    rec = csdl.Recorder(inline=True)
    rec.start()

    # Axis-aligned box centered at [1, 2, 3], half-lengths [2, 1, 0.5]
    center = csdl.Variable(value=np.array([1.0, 2.0, 3.0]))
    hl = csdl.Variable(value=np.array([2.0, 1.0, 0.5]))
    sdf_op = BoxSignedDistanceFunction(half_lengths=hl, center=center, sharpness=500.0)

    pts = np.array([
        [1.0, 2.0, 3.0],    # Center -> inside, min distance = -0.5
        [3.0, 2.0, 3.0],    # +x face -> 0.0
        [4.0, 2.0, 3.0],    # 1.0 m outside +x face -> +1.0
        [4.0, 3.0, 3.5],    # Corner outside by (1, 0, 0) -> +1.0
        [4.0, 4.0, 3.5],    # Outside (1, 1, 0) -> sqrt(2)
    ])
    pts_var = csdl.Variable(value=pts)
    vals = sdf_op.evaluate(pts_var)

    # Verify JAX simulation and finite derivatives
    sim = csdl.experimental.JaxSimulator(rec, additional_inputs=[pts_var, center], additional_outputs=[vals])
    sim.run()
    v = np.asarray(sim[vals]).flatten()
    assert np.isclose(v[0], -0.5, atol=5e-3)
    assert np.isclose(v[1], 0.0, atol=5e-3)
    assert np.isclose(v[2], 1.0, atol=5e-3)
    assert np.isclose(v[3], 1.0, atol=5e-3)
    assert np.isclose(v[4], np.sqrt(2.0), atol=5e-3)

    # Check rotated box (90 deg about z-axis: local x -> world +y, local y -> world -x)
    theta = np.pi / 2.0
    R_z = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0],
    ])
    sdf_rot = BoxSignedDistanceFunction(
        half_lengths=np.array([2.0, 0.5, 0.5]),
        center=np.array([0.0, 0.0, 0.0]),
        rotation=R_z,
        sharpness=500.0,
    )
    # Point at world [0, 3.0, 0] is along local +x (half_length = 2.0) -> outside by 1.0
    # Point at world [1.5, 0, 0] is along local -y (half_length = 0.5) -> outside by 1.0
    pts_rot = csdl.Variable(value=np.array([[0.0, 3.0, 0.0], [1.5, 0.0, 0.0]]))
    vals_rot = sdf_rot.evaluate(pts_rot)
    sim_rot = csdl.experimental.JaxSimulator(rec, additional_inputs=[pts_rot], additional_outputs=[vals_rot])
    sim_rot.run()
    v_rot = np.asarray(sim_rot[vals_rot]).flatten()
    assert np.isclose(v_rot[0], 1.0, atol=5e-3)
    assert np.isclose(v_rot[1], 1.0, atol=5e-3)


def test_dynamic_sizing_toggles_and_derivatives():
    rec = csdl.Recorder(inline=True)
    rec.start()

    W_total = csdl.Variable(value=1.5e6)
    W_ref = 1.2e6
    dyn = compute_landing_gear_sizing(W_total, W_ref_lg=W_ref, scale_factor=7.5, dynamic_sizing=True)
    stat = compute_landing_gear_sizing(W_total, W_ref_lg=W_ref, scale_factor=7.5, dynamic_sizing=False)

    c_local = csdl.Variable(value=4.0)
    c_ref = 3.0
    act_dyn = compute_actuator_sizing(c_local, c_ref_local=c_ref, scale_factor=7.5, dynamic_sizing=True)
    act_stat = compute_actuator_sizing(c_local, c_ref_local=c_ref, scale_factor=7.5, dynamic_sizing=False)

    sim = csdl.experimental.JaxSimulator(
        rec,
        additional_inputs=[W_total, c_local],
        additional_outputs=[
            dyn['mlg_half_lengths'], dyn['mlg_mass'], dyn['lg_dim_scale'],
            stat['mlg_half_lengths'], stat['mlg_mass'],
            act_dyn, act_stat,
        ],
    )
    sim.run()
    totals = sim.compute_totals()

    expected_dim_scale = (1.5e6 / 1.2e6) ** (1.0 / 3.0)
    assert np.allclose(sim[dyn['lg_dim_scale']], expected_dim_scale)
    assert np.allclose(sim[stat['mlg_mass']], 1950.0)
    assert np.allclose(sim[dyn['mlg_mass']], 1950.0 * (1.5e6 / 1.2e6))

    # Derivative check vs FD for dynamic landing gear sizing
    d_mlg_dW_analytic = np.asarray(totals[(dyn['mlg_half_lengths'], W_total)]).flatten()
    h = 10.0
    sim[W_total] = 1.5e6 + h
    sim.run()
    f_plus = np.asarray(sim[dyn['mlg_half_lengths']]).flatten()
    sim[W_total] = 1.5e6 - h
    sim.run()
    f_minus = np.asarray(sim[dyn['mlg_half_lengths']]).flatten()
    d_mlg_dW_fd = (f_plus - f_minus) / (2.0 * h)
    assert np.allclose(d_mlg_dW_analytic, d_mlg_dW_fd, rtol=1e-5)

    # Static toggle must have zero derivative wrt W_total
    d_stat_dW = np.asarray(totals[(stat['mlg_half_lengths'], W_total)]).flatten()
    assert np.allclose(d_stat_dW, 0.0)

    # Actuator derivative check vs FD
    d_act_dc_analytic = np.asarray(totals[(act_dyn, c_local)]).flatten()
    h_c = 1e-4
    sim[c_local] = 4.0 + h_c
    sim.run()
    a_plus = np.asarray(sim[act_dyn]).flatten()
    sim[c_local] = 4.0 - h_c
    sim.run()
    a_minus = np.asarray(sim[act_dyn]).flatten()
    d_act_dc_fd = (a_plus - a_minus) / (2.0 * h_c)
    assert np.allclose(d_act_dc_analytic, d_act_dc_fd, rtol=1e-5)
    assert np.allclose(np.asarray(totals[(act_stat, c_local)]).flatten(), 0.0)


def test_local_chord_frame_and_box_corners():
    rec = csdl.Recorder(inline=True)
    rec.start()

    # Swept + twisted chord vector: dx=3, dy=1, dz=-0.5
    c_vec = csdl.Variable(value=np.array([3.0, 1.0, -0.5]))
    R = build_local_frame_from_chord_vector(c_vec)
    sim = csdl.experimental.JaxSimulator(rec, additional_inputs=[c_vec], additional_outputs=[R])
    sim.run()
    R_val = np.asarray(sim[R])

    # Verify orthonormal right-handed frame
    assert np.allclose(R_val.T @ R_val, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(R_val), 1.0, atol=1e-10)

    corners = get_box_corners_3d(np.array([1.0, 2.0, 0.0]), np.array([1.0, 0.5, 0.25]), rotation=R_val)
    assert corners.shape == (8, 3)


def test_smooth_union_and_partial_aggregation_derivatives():
    rec = csdl.Recorder(inline=True)
    rec.start()

    nlg_x = csdl.Variable(value=1.0)
    nlg_c = csdl.concatenate([csdl.reshape(nlg_x, (1,)), csdl.Variable(value=np.array([0.0, 0.0]))])
    nlg_hl = csdl.Variable(value=np.array([1.0, 0.5, 0.5]))

    pts_nlg, part_groups = build_box_surface_samples(
        center=nlg_c,
        half_lengths=nlg_hl,
        grid_res=(3, 3),
        num_partitions_per_axis=2,
        symmetry_y_nonneg=True,
    )
    # 5 faces (symmetry_y_nonneg) * 3*3 = 45 pts; 2*2=4 partitions per face -> 20 local constraints
    assert pts_nlg.shape == (45, 3)
    assert len(part_groups) == 20

    # Two obstacles: payload box at x=8.0 and MLG box at x=10.0, y=3.0
    pay_sdfs = build_payload_box_sdfs('oversized', payload_center_x=csdl.Variable(value=8.0))
    mlg_sdf = BoxSignedDistanceFunction(
        half_lengths=np.array([1.5, 0.75, 0.6]),
        center=np.array([10.0, 3.0, 0.0]),
    )
    obstacles = pay_sdfs['oversized'] + [mlg_sdf]

    raw_ext, union_d = evaluate_exterior_non_interference(
        query_points=pts_nlg,
        obstacle_sdfs=obstacles,
        clearance_buffer=0.2,
        rho_union=50.0,
        aggregation_mode='partial_aggregation',
    )
    agg_partial = aggregate_constraints_by_mode(raw_ext, part_groups, 'partial_aggregation')
    agg_full = aggregate_constraints_by_mode(raw_ext, part_groups, 'full_aggregation')

    sim = csdl.experimental.JaxSimulator(
        rec,
        additional_inputs=[nlg_x],
        additional_outputs=[agg_partial, agg_full],
    )
    sim.run()
    totals = sim.compute_totals()

    # NLG is well separated from payload at x=8 (closest face x=2 vs payload front x=8-5.029=2.971) -> dist ~0.97 > 0.2 -> feasible (< 0)
    assert np.all(np.asarray(sim[agg_partial]) < 0.0)
    assert float(np.asarray(sim[agg_full]).flatten()[0]) < 0.0

    # Check derivative wrt nlg_x against centered FD
    d_full_analytic = float(np.asarray(totals[(agg_full, nlg_x)]).flatten()[0])
    h = 1e-5
    sim[nlg_x] = 1.0 + h
    sim.run()
    f_p = float(np.asarray(sim[agg_full]).flatten()[0])
    sim[nlg_x] = 1.0 - h
    sim.run()
    f_m = float(np.asarray(sim[agg_full]).flatten()[0])
    d_full_fd = (f_p - f_m) / (2.0 * h)
    assert np.isclose(d_full_analytic, d_full_fd, rtol=1e-4, atol=1e-6)


def test_bspline_oml_subsystems_and_aggregation_modes(tmp_path, monkeypatch):
    """Fast targeted integration test without VortexAD/aframe:
    exercises lsdo_geo B-spline OML `evaluate_signed_distance(sign_mode='winding_number')`,
    dynamic landing gear & actuator sizing, local-frame actuator attachment,
    all aggregation modes ('partial_aggregation', 'pairwise', 'full_aggregation', 'unaggregated'),
    and pitch-rotation invariance of body-frame containment constraints.
    """
    from pathlib import Path
    import lsdo_geo

    repo_root = Path(__file__).resolve().parents[4]
    cad_path = repo_root / "examples" / "example_geometries" / "rectangular_wing" / "rectangular_wing_naca0012_10ar.stp"

    # Isolate lsdo_function_spaces' stored_files/imports cache in tmp_path
    monkeypatch.chdir(tmp_path)

    rec = csdl.Recorder(inline=True)
    rec.start()

    geom = lsdo_geo.import_geometry(str(cad_path), parallelize=False)
    scale_factor = 7.5
    # Scale geometry around origin so root chord is 7.5 m and half-span is 37.5 m
    for fn in geom.functions.values():
        fn.coefficients = fn.coefficients * scale_factor

    # Dynamic variables
    nlg_center_x = csdl.Variable(value=1.5, name='nlg_center_x')
    mlg_center_x = csdl.Variable(value=4.5, name='mlg_center_x')
    mlg_center_y = csdl.Variable(value=3.0, name='mlg_center_y')
    payload_center_x = csdl.Variable(value=3.75, name='payload_center_x')
    W_total = csdl.Variable(value=1.2e6, name='W_total')
    pitch = csdl.Variable(value=np.deg2rad(5.0), name='pitch')

    # Landing gear dynamic sizing + samples
    lg_sizing = compute_landing_gear_sizing(
        W_total=W_total,
        W_ref_lg=1.2e6,
        scale_factor=scale_factor,
        dynamic_sizing=True,
    )
    nlg_center = csdl.concatenate([
        csdl.reshape(nlg_center_x, (1,)),
        csdl.Variable(value=np.array([0.0, 0.0])),
    ])
    mlg_center = csdl.concatenate([
        csdl.reshape(mlg_center_x, (1,)),
        csdl.reshape(mlg_center_y, (1,)),
        csdl.Variable(value=np.array([0.0])),
    ])
    nlg_pts, nlg_groups = build_box_surface_samples(
        nlg_center, lg_sizing['nlg_half_lengths'], grid_res=(2, 2), num_partitions_per_axis=2, symmetry_y_nonneg=True
    )
    mlg_pts, mlg_groups = build_box_surface_samples(
        mlg_center, lg_sizing['mlg_half_lengths'], grid_res=(2, 2), num_partitions_per_axis=2, symmetry_y_nonneg=False
    )

    # Outboard actuator projected onto wing + local frame + chord sizing
    y_act = 2.5 * scale_factor
    le_proj = geom.project(np.array([0.0, y_act, 0.0]), direction=np.array([1.0, 0.0, 0.0]), plot=False)
    te_proj = geom.project(np.array([scale_factor, y_act, 0.0]), direction=np.array([-1.0, 0.0, 0.0]), plot=False)
    u75_proj = geom.project(np.array([0.75 * scale_factor, y_act, 1.0]), direction=np.array([0.0, 0.0, -1.0]), plot=False)
    l75_proj = geom.project(np.array([0.75 * scale_factor, y_act, -1.0]), direction=np.array([0.0, 0.0, 1.0]), plot=False)

    le_pt = csdl.reshape(geom.evaluate(le_proj), (3,))
    te_pt = csdl.reshape(geom.evaluate(te_proj), (3,))
    u75_pt = csdl.reshape(geom.evaluate(u75_proj), (3,))
    l75_pt = csdl.reshape(geom.evaluate(l75_proj), (3,))

    chord_vec = te_pt - le_pt
    c_local = csdl.sqrt(csdl.sum(chord_vec**2) + 1e-12)
    act_hl = compute_actuator_sizing(c_local=c_local, c_ref_local=scale_factor, scale_factor=scale_factor, dynamic_sizing=True)
    act_rot = build_local_frame_from_chord_vector(chord_vec)
    act_center = 0.5 * (u75_pt + l75_pt)
    act_pts, act_groups = build_box_surface_samples(
        act_center, act_hl, rotation=act_rot, grid_res=(2, 2), num_partitions_per_axis=2, symmetry_y_nonneg=False
    )

    # Batched projection-based OML SDF in body frame (no BSM3)
    all_pts = csdl.concatenate([nlg_pts, mlg_pts, act_pts], axis=0)
    all_oml_sdf = csdl.reshape(geom.evaluate_signed_distance(all_pts, sign_mode='winding_number'), (all_pts.shape[0],))
    n_nlg = nlg_pts.shape[0]
    n_mlg = mlg_pts.shape[0]
    raw_oml_nlg = all_oml_sdf[:n_nlg] + 0.2
    raw_oml_mlg = all_oml_sdf[n_nlg:n_nlg + n_mlg] + 0.2
    raw_oml_act = all_oml_sdf[n_nlg + n_mlg:] + 0.2

    # Exterior non-interference against oversized payload & each other
    payload_sdfs = build_payload_box_sdfs('oversized', payload_center_x=payload_center_x)
    nlg_sdf = BoxSignedDistanceFunction(lg_sizing['nlg_half_lengths'], center=nlg_center)
    mlg_r_sdf = BoxSignedDistanceFunction(lg_sizing['mlg_half_lengths'], center=mlg_center)
    mlg_l_center = csdl.concatenate([
        csdl.reshape(mlg_center_x, (1,)),
        csdl.reshape(-mlg_center_y, (1,)),
        csdl.Variable(value=np.array([0.0])),
    ])
    mlg_l_sdf = BoxSignedDistanceFunction(lg_sizing['mlg_half_lengths'], center=mlg_l_center)

    nlg_ext_raw, _ = evaluate_exterior_non_interference(
        nlg_pts, payload_sdfs['oversized'] + [mlg_r_sdf, mlg_l_sdf], clearance_buffer=0.2, aggregation_mode='partial_aggregation'
    )
    nlg_ext_partial = aggregate_constraints_by_mode(nlg_ext_raw, nlg_groups, 'partial_aggregation')
    mlg_ext_raw, _ = evaluate_exterior_non_interference(
        mlg_pts, payload_sdfs['oversized'] + [nlg_sdf, mlg_l_sdf], clearance_buffer=0.2, aggregation_mode='partial_aggregation'
    )
    mlg_ext_pairwise_raw, _ = evaluate_exterior_non_interference(
        mlg_pts, payload_sdfs['oversized'] + [nlg_sdf, mlg_l_sdf], clearance_buffer=0.2, aggregation_mode='pairwise'
    )

    # Verify all aggregation modes build valid shapes
    agg_outputs = {}
    for mode in ('partial_aggregation', 'pairwise', 'full_aggregation', 'unaggregated'):
        agg_outputs[mode] = aggregate_constraints_by_mode(raw_oml_mlg, mlg_groups, mode)

    # Unified pitch kinematics (copying payload pattern)
    rot_origin = csdl.Variable(value=np.array([0.25 * scale_factor, 0.0, 0.0]))
    nlg_center_inertial = lsdo_geo.rotate(
        nlg_center, rotation_origin=rot_origin, axis_vector=np.array([0.0, 1.0, 0.0]), angles=pitch, units='radians'
    )
    act_center_inertial = lsdo_geo.rotate(
        act_center, rotation_origin=rot_origin, axis_vector=np.array([0.0, 1.0, 0.0]), angles=pitch, units='radians'
    )

    rec.stop()

    outs = [
        raw_oml_nlg, raw_oml_mlg, raw_oml_act,
        nlg_ext_partial, nlg_ext_raw, mlg_ext_pairwise_raw, mlg_ext_raw,
        nlg_center_inertial, act_center_inertial,
        agg_outputs['partial_aggregation'],
        agg_outputs['pairwise'],
        agg_outputs['full_aggregation'],
        agg_outputs['unaggregated'],
    ]
    sim = csdl.experimental.JaxSimulator(
        recorder=rec,
        additional_inputs=[nlg_center_x, mlg_center_x, mlg_center_y, payload_center_x, W_total, pitch],
        additional_outputs=outs,
    )
    sim.run()
    totals = sim.compute_totals()

    for var in outs:
        val = np.asarray(sim[var])
        assert np.all(np.isfinite(val))

    # Check shapes of aggregation outputs
    assert np.asarray(sim[agg_outputs['partial_aggregation']]).shape == (len(mlg_groups),)
    assert np.asarray(sim[agg_outputs['pairwise']]).shape == (n_mlg,)
    assert np.asarray(sim[agg_outputs['full_aggregation']]).shape == (1,)
    assert np.asarray(sim[agg_outputs['unaggregated']]).shape == (n_mlg,)

    # Pitch-rotation invariance of body-frame OML constraint & exact pitch rotation of center
    nlg_rot_val = np.asarray(sim[nlg_center_inertial]).flatten()
    theta = np.deg2rad(5.0)
    dx = 1.5 - 0.25 * scale_factor
    expected_x = 0.25 * scale_factor + dx * np.cos(theta)
    expected_z = -dx * np.sin(theta)
    assert np.allclose(nlg_rot_val, [expected_x, 0.0, expected_z], atol=1e-6)

    # Check derivatives w.r.t. mlg_center_x and W_total exist and are finite
    for dv in (nlg_center_x, mlg_center_x, mlg_center_y, W_total):
        assert np.all(np.isfinite(totals[(agg_outputs['full_aggregation'], dv)]))
        assert np.all(np.isfinite(totals[(nlg_ext_partial, dv)]))
    # Body-frame SDF must have 0 sensitivity to pitch
    assert np.allclose(totals[(agg_outputs['full_aggregation'], pitch)], 0.0, atol=1e-12)


def test_pallets_and_both_mode_subsystems_cg_coupling():
    """Fast check of 'both' and 'pallets' payload-to-subsystem exterior SDFs,
    landing-gear mass carve-out from misc_mass, composite CG, and MLG tip-back derivatives.
    """
    rec = csdl.Recorder(inline=True)
    rec.start()

    pay_cx = csdl.Variable(value=3.75, name='payload_center_x')
    pal_cx = csdl.Variable(value=3.75, name='pallet_stack_center_x')
    nlg_x = csdl.Variable(value=-7.0, name='nlg_center_x')
    mlg_x = csdl.Variable(value=5.5, name='mlg_center_x')
    mlg_y = csdl.Variable(value=5.5, name='mlg_center_y')
    W_struct = csdl.Variable(value=3.0e5)
    payload_weight = csdl.Variable(value=7.117e5)
    misc_weight = csdl.Variable(value=6.0e5)

    W_total_body = W_struct + payload_weight + misc_weight
    lg = compute_landing_gear_sizing(W_total_body, W_ref_lg=1.6e6, scale_factor=7.5, dynamic_sizing=True)
    nlg_mass = lg['nlg_mass'][0]
    mlg_mass = lg['mlg_mass'][0]

    misc_mass = misc_weight / 9.81
    misc_mass_rem = csdl.maximum(misc_mass - (nlg_mass + 2.0 * mlg_mass), csdl.Variable(value=100.0), rho=50.0)
    total_mass = (W_struct + payload_weight) / 9.81 + nlg_mass + 2.0 * mlg_mass + misc_mass_rem
    x_cg = ((W_struct / 9.81) * 4.0 + (payload_weight / 9.81) * pay_cx + nlg_mass * nlg_x + 2.0 * mlg_mass * mlg_x + misc_mass_rem * 3.0) / total_mass
    tipback_margin = (mlg_x - x_cg) - 0.5

    nlg_c = csdl.concatenate([csdl.reshape(nlg_x, (1,)), csdl.Variable(value=np.array([0.0, 0.0]))])
    mlg_r_c = csdl.concatenate([csdl.reshape(mlg_x, (1,)), csdl.reshape(mlg_y, (1,)), csdl.Variable(value=np.array([0.0]))])
    mlg_l_c = csdl.concatenate([csdl.reshape(mlg_x, (1,)), csdl.reshape(-mlg_y, (1,)), csdl.Variable(value=np.array([0.0]))])
    nlg_pts, nlg_groups = build_box_surface_samples(nlg_c, lg['nlg_half_lengths'], grid_res=(2, 2), symmetry_y_nonneg=True)

    # 'both' mode creates 1 oversized SDF + 6 pallet SDFs
    pay_both = build_payload_box_sdfs('both', payload_center_x=pay_cx, pallet_stack_center_x=pal_cx)
    assert len(pay_both['oversized']) == 1 and len(pay_both['pallets']) == 6

    mlg_r_sdf = BoxSignedDistanceFunction(lg['mlg_half_lengths'], center=mlg_r_c)
    mlg_l_sdf = BoxSignedDistanceFunction(lg['mlg_half_lengths'], center=mlg_l_c)
    nlg_ext_raw, _ = evaluate_exterior_non_interference(
        nlg_pts, pay_both['oversized'] + pay_both['pallets'] + [mlg_r_sdf, mlg_l_sdf], clearance_buffer=0.2
    )
    nlg_ext_partial = aggregate_constraints_by_mode(nlg_ext_raw, nlg_groups, 'partial_aggregation')

    rec.stop()

    sim = csdl.experimental.JaxSimulator(
        recorder=rec,
        additional_inputs=[pay_cx, pal_cx, nlg_x, mlg_x, mlg_y, W_struct],
        additional_outputs=[misc_mass_rem, total_mass, tipback_margin, nlg_ext_partial],
    )
    sim.run()
    totals = sim.compute_totals()

    # Mass conservation: misc_mass_rem + nlg_mass + 2*mlg_mass ~ misc_mass
    assert np.isclose(float(np.asarray(sim[total_mass]).flatten()[0]), (3.0e5 + 7.117e5 + 6.0e5) / 9.81, rtol=1e-4)
    # All NLG exterior residuals feasible (x=-7 far ahead of payload & pallets & MLG)
    assert np.all(np.asarray(sim[nlg_ext_partial]) < 0.0)
    # Check analytical tipback derivative wrt mlg_x matches centered FD
    d_tip_analytic = float(np.asarray(totals[(tipback_margin, mlg_x)]).flatten()[0])
    h = 1e-5
    sim[mlg_x] = 5.5 + h
    sim.run()
    tp = float(np.asarray(sim[tipback_margin]).flatten()[0])
    sim[mlg_x] = 5.5 - h
    sim.run()
    tm = float(np.asarray(sim[tipback_margin]).flatten()[0])
    assert np.isclose(d_tip_analytic, (tp - tm) / (2.0 * h), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("subsys_mode,agg_mode,pay_mode,do_plot", [
    ("full", "partial_aggregation", "both", True),
    ("landing_gear", "volume_only", "oversized", False),
])
def test_bwb_script_subsystem_modes_py_diagnostic(subsys_mode, agg_mode, pay_mode, do_plot, tmp_path):
    """Targeted end-to-end BWB script diagnostic checks using PySimulator (~15s each)
    instead of full XLA compilation or multi-hour animation replay.
    """
    import subprocess
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[4]
    screenshot_path = tmp_path / "bwb_payload_plot.png"
    env = os.environ.copy()
    env.update({
        'BWB_SIMULATOR': 'py',
        'DIAGNOSTIC_ONLY': 'true',
        'RUN_PRE_DIAGNOSTICS': '1',
        'VISCOUS_DRAG_MODE': 'constant_cd0',
        'BWB_RESOLUTION': 'fast',
        'GEONIC_PAYLOAD_MODE': pay_mode,
        'GEONIC_SUBSYSTEM_MODE': subsys_mode,
        'GEONIC_AGGREGATION_MODE': agg_mode,
    })
    if do_plot:
        env.update({
            'PLOT_GEOMETRY_WITH_PAYLOAD': '1',
            'PLOT_GEOMETRY_WITH_PAYLOAD_INTERACTIVE': '0',
            'PLOT_GEOMETRY_WITH_PAYLOAD_SCREENSHOT': str(screenshot_path),
            'PYVISTA_OFF_SCREEN': 'true',
        })
    res = subprocess.run(
        [sys.executable, "examples/showcase_examples/rectangular_wing/ex_rectangular_wing_to_bwb.py"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert res.returncode == 0, f"BWB script failed:\n{res.stderr}\n{res.stdout[-1000:]}"
    assert f"Subsystem Mode: {subsys_mode.upper()} | Aggregation: {agg_mode.upper()}" in res.stdout
    assert "MLG Tip-Back Margin:" in res.stdout
    if subsys_mode == 'full' and agg_mode != 'volume_only':
        assert "Actuator OML Margin:" in res.stdout
    if do_plot:
        assert screenshot_path.is_file() and screenshot_path.stat().st_size > 1000


def test_dual_explicit_bspline_and_implicit_sdf_representations(tmp_path):
    """Verify every payload and subsystem box has BOTH a watertight 6-patch
    B-spline lsdo_geo.Geometry and an implicit BoxSignedDistanceFunction that agree.
    """
    import lsdo_geo
    import pyvista as pv

    rec = csdl.Recorder(inline=True)
    rec.start()

    pay_cx = csdl.Variable(value=3.75, name='payload_center_x')
    pal_cx = csdl.Variable(value=3.75, name='pallet_stack_center_x')
    pay_reps = build_payload_representations('both', payload_center_x=pay_cx, pallet_stack_center_x=pal_cx)
    assert len(pay_reps['oversized']) == 1
    assert len(pay_reps['pallets']) == 6

    all_reps = pay_reps['oversized'] + pay_reps['pallets']
    for rep in all_reps:
        assert isinstance(rep.explicit_geometry, lsdo_geo.Geometry)
        assert len(rep.explicit_geometry.functions) == 6
        assert isinstance(rep.implicit_sdf, BoxSignedDistanceFunction)

    # Verify explicit B-spline signed_distance matches implicit BoxSignedDistanceFunction on oversized box
    over_rep = pay_reps['oversized'][0]
    probe_pts = csdl.Variable(value=np.array([
        [3.75, 0.0, 0.0],    # Center (interior: -1.524 m)
        [3.75, 0.0, 3.0],    # Outside top face (+z half-height = 1.524 -> +1.476 m)
        [10.5, 0.0, 0.0],    # Outside rear face
    ]))
    explicit_sdf = over_rep.explicit_geometry.evaluate_signed_distance(probe_pts, sign_mode='winding_number')
    implicit_sdf = over_rep.implicit_sdf.evaluate(probe_pts)

    # Also build a rotated subsystem representation & verify headless plot_geometry_with_payloads
    rot = build_local_frame_from_chord_vector(csdl.Variable(value=np.array([2.0, 0.2, -0.1])))
    act_rep = build_box_representation(
        name='actuator_test',
        half_lengths=np.array([0.4, 0.15, 0.08]),
        center=np.array([6.0, 10.0, 0.0]),
        rotation=rot,
    )
    assert len(act_rep.explicit_geometry.functions) == 6
    wing_mock = build_box_bspline_geometry(np.array([15.0, 20.0, 5.0]), center=np.array([5.0, 0.0, 0.0]), name='mock_oml')

    rec.stop()

    sim = csdl.experimental.PySimulator(recorder=rec)
    sim.run()
    exp_vals = np.asarray(sim[explicit_sdf]).flatten()
    imp_vals = np.asarray(sim[implicit_sdf]).flatten()
    assert np.allclose(exp_vals, imp_vals, atol=1e-2)

    # Render without blocking using PyVista off-screen helper
    pv.OFF_SCREEN = True
    shot = str(tmp_path / 'dual_repr_plot.png')
    elems = plot_geometry_with_payloads(
        wing_geometry=wing_mock,
        payload_representations=pay_reps,
        subsystem_representations=[act_rep],
        sample_points=[('payload', probe_pts)],
        interactive=False,
        screenshot=shot,
        show=True,
    )
    assert len(elems) > 0
    assert os.path.isfile(shot) and os.path.getsize(shot) > 1000
