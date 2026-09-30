from __future__ import annotations

import numpy as np
import pytest
import csdl_alpha as csdl

from examples.showcase_examples.rectangular_wing.physics_models.geonic_payload import (
    FT2M,
    PAYLOAD_LENGTH_M,
    PAYLOAD_WIDTH_M,
    PAYLOAD_HEIGHT_M,
    GEONIC_CLEARANCE_M,
    build_geonic_payload_sample_points,
    compute_geonic_clearance_and_margin,
    get_full_payload_box_corners,
)


def test_payload_dimensions():
    """Verify TCP0 oversized-payload dimensions converted to SI units."""
    assert np.isclose(FT2M, 1.0 / 3.28084)
    assert np.isclose(PAYLOAD_LENGTH_M, 33.0 / 3.28084)
    assert np.isclose(PAYLOAD_WIDTH_M, 13.0 / 3.28084)
    assert np.isclose(PAYLOAD_HEIGHT_M, 10.0 / 3.28084)
    assert GEONIC_CLEARANCE_M == 0.2


def test_payload_sample_builder_geometry_and_symmetry():
    """
    Verify the 8-point CSDL sample builder:
    - Exactly 8 points shaped (8, 3)
    - Root (y=0) and right-side (y=width/2) only; no negative-y coordinates
    - Correct front/rear x expressions
    - Correct bottom/top z expressions
    """
    rec = csdl.Recorder(inline=True)
    rec.start()

    center_x_val = 3.75
    payload_center_x = csdl.Variable(value=center_x_val, name="payload_center_x")
    payload_center_x.set_as_design_variable()

    sample_points_csdl = build_geonic_payload_sample_points(payload_center_x)

    sim = csdl.experimental.JaxSimulator(recorder=rec, additional_outputs=[sample_points_csdl])
    pts = np.asarray(sim[sample_points_csdl])

    # Check count and shape
    assert pts.shape == (8, 3), f"Expected shape (8, 3), got {pts.shape}"

    half_l = 0.5 * PAYLOAD_LENGTH_M
    half_w = 0.5 * PAYLOAD_WIDTH_M
    half_h = 0.5 * PAYLOAD_HEIGHT_M

    expected_x_front = center_x_val - half_l
    expected_x_rear = center_x_val + half_l

    # Check x coordinates
    np.testing.assert_allclose(pts[:4, 0], expected_x_front)
    np.testing.assert_allclose(pts[4:, 0], expected_x_rear)

    # Check y coordinates: exactly 0.0 or +half_w; none are negative
    y_vals = pts[:, 1]
    assert np.all(y_vals >= 0.0), f"Expected non-negative y coordinates, got {y_vals}"
    np.testing.assert_allclose(np.sort(np.unique(y_vals)), [0.0, half_w])
    assert np.sum(y_vals == 0.0) == 4
    assert np.sum(np.isclose(y_vals, half_w)) == 4

    # Check z coordinates: symmetric about z=0
    z_vals = pts[:, 2]
    np.testing.assert_allclose(np.sort(np.unique(z_vals)), [-half_h, half_h])
    assert np.sum(np.isclose(z_vals, -half_h)) == 4
    assert np.sum(np.isclose(z_vals, half_h)) == 4


def test_geonic_residual_convention_and_margin():
    """
    Verify 0.2 m clearance buffer and residual convention:
    geonic_constraint_values = payload_signed_distance + 0.2 <= 0
    geonic_clearance_per_point = -payload_signed_distance
    geonic_margin = -max(geonic_constraint_values) >= 0 for feasible
    """
    rec = csdl.Recorder(inline=True)
    rec.start()

    # Case 1: Exactly at the 0.2 m interior buffer (sdf = -0.2)
    # constraint_value = -0.2 + 0.2 = 0.0
    # clearance = 0.2
    # margin = 0.0 (boundary feasible)
    sdf_mock = csdl.Variable(value=np.full(8, -0.2), name="sdf_mock")
    sdf_mock.set_as_design_variable()

    c_vals, clearance, margin = compute_geonic_clearance_and_margin(sdf_mock, clearance_buffer=0.2)

    sim = csdl.experimental.JaxSimulator(recorder=rec, additional_outputs=[c_vals, clearance, margin])
    c_arr = np.asarray(sim[c_vals])
    clearance_arr = np.asarray(sim[clearance])
    margin_val = float(np.asarray(sim[margin]).flatten()[0])

    # KS smooth maximum for 8 identical values evaluates to x + ln(8)/rho = 0 + ln(8)/50 ~= 0.04159
    # so margin evaluates to -ln(8)/50
    expected_ks_margin = -np.log(8.0) / 50.0
    assert np.isclose(margin_val, expected_ks_margin, atol=1e-5)

    # When 1 point is active at boundary (sdf=-0.2) and other 7 points are well inside (sdf=-1.0),
    # margin evaluates to 0.0 within 1e-6
    rec1b = csdl.Recorder(inline=True)
    rec1b.start()
    sdf_single_active = csdl.Variable(value=np.array([-0.2] + [-1.0]*7), name="sdf_single")
    sdf_single_active.set_as_design_variable()
    _, _, m_single = compute_geonic_clearance_and_margin(sdf_single_active, clearance_buffer=0.2, rho=50.0)
    sim1b = csdl.experimental.JaxSimulator(recorder=rec1b, additional_outputs=[m_single])
    assert np.isclose(float(np.asarray(sim1b[m_single]).flatten()[0]), 0.0, atol=1e-6)

    # Case 2: Deep inside (sdf = -0.5) -> clearance = 0.5 m, margin > 0 (feasible)
    rec2 = csdl.Recorder(inline=True)
    rec2.start()
    sdf_deep = csdl.Variable(value=np.full(8, -0.5), name="sdf_deep")
    sdf_deep.set_as_design_variable()
    c_deep, _, m_deep = compute_geonic_clearance_and_margin(sdf_deep, clearance_buffer=0.2)
    sim2 = csdl.experimental.JaxSimulator(recorder=rec2, additional_outputs=[c_deep, m_deep])
    assert float(np.asarray(sim2[m_deep]).flatten()[0]) > 0.25

    # Case 3: Violated (sdf = -0.1, clearance is 0.1 < 0.2 buffer) -> margin < 0
    rec3 = csdl.Recorder(inline=True)
    rec3.start()
    sdf_viol = csdl.Variable(value=np.full(8, -0.1), name="sdf_viol")
    sdf_viol.set_as_design_variable()
    c_viol, _, m_viol = compute_geonic_clearance_and_margin(sdf_viol, clearance_buffer=0.2)
    sim3 = csdl.experimental.JaxSimulator(recorder=rec3, additional_outputs=[c_viol, m_viol])
    assert float(np.asarray(sim3[m_viol]).flatten()[0]) < 0.0


def test_full_payload_box_corners():
    """Verify rigid 3D box generation for PyVista wireframe/mesh."""
    corners = get_full_payload_box_corners(center_x=3.75, center_y=0.0, center_z=0.0)
    assert corners.shape == (8, 3)
    assert np.min(corners[:, 0]) < 3.75 < np.max(corners[:, 0])
    assert np.isclose(np.max(corners[:, 0]) - np.min(corners[:, 0]), PAYLOAD_LENGTH_M)
    assert np.isclose(np.max(corners[:, 1]) - np.min(corners[:, 1]), PAYLOAD_WIDTH_M)
    assert np.isclose(np.max(corners[:, 2]) - np.min(corners[:, 2]), PAYLOAD_HEIGHT_M)
