import os
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import numpy as np
import pytest
import csdl_alpha as csdl

SHOWCASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
if SHOWCASE_DIR not in sys.path:
    sys.path.insert(0, SHOWCASE_DIR)

from physics_models.geonic_payload import (
    FT2M,
    IN2M,
    PAYLOAD_LENGTH_M,
    PAYLOAD_WIDTH_M,
    PAYLOAD_HEIGHT_M,
    PALLET_L_SHORT_M,
    PALLET_WIDTH_M,
    PALLET_HEIGHT_M,
    PALLET_L_LONG_M,
    PALLET_STACK_FRONT_M,
    PALLET_STACK_REFERENCE_CG_X,
    GEONIC_CLEARANCE_M,
    VALID_GEONIC_PAYLOAD_MODES,
    parse_geonic_payload_mode,
    build_geonic_payload_sample_points,
    build_geonic_pallet_sample_points,
    compute_geonic_clearance_and_margin,
    get_full_payload_box_corners,
    get_full_pallet_boxes_corners,
)


def test_payload_dimensions():
    # TCP0 oversized payload: 33 x 13 x 10 ft
    assert np.isclose(PAYLOAD_LENGTH_M, 33.0 / 3.28084)
    assert np.isclose(PAYLOAD_WIDTH_M, 13.0 / 3.28084)
    assert np.isclose(PAYLOAD_HEIGHT_M, 10.0 / 3.28084)
    assert GEONIC_CLEARANCE_M == 0.2

    # TCP0 pallets: 88 x 108 x 96 in shorts, 5x length longs
    assert np.isclose(PALLET_L_SHORT_M, 88.0 / 39.3701)
    assert np.isclose(PALLET_WIDTH_M, 108.0 / 39.3701)
    assert np.isclose(PALLET_HEIGHT_M, 96.0 / 39.3701)
    assert np.isclose(PALLET_L_LONG_M, 5.0 * (88.0 / 39.3701))

    # Reference CG calculation
    l_s = 88.0 / 39.3701
    calc_cg = 5.0 + (71.0 / 18.0) * l_s
    assert np.isclose(PALLET_STACK_REFERENCE_CG_X, calc_cg, atol=1e-12)
    assert np.isclose(PALLET_STACK_REFERENCE_CG_X, 13.816617461248795, atol=1e-12)


def test_oversized_payload_sample_points():
    rec = csdl.Recorder()
    rec.start()
    cx = csdl.Variable(value=3.75, name='payload_center_x')
    pts = build_geonic_payload_sample_points(cx)
    pts_legacy = build_geonic_payload_sample_points(cx, num_chord_points=2)
    rec.stop()

    sim = csdl.experimental.JaxSimulator(
        recorder=rec,
        additional_inputs=[cx],
        additional_outputs=[pts, pts_legacy],
        gpu=False,
    )
    sim.run()
    arr = np.asarray(sim[pts])
    arr_legacy = np.asarray(sim[pts_legacy])

    # Default chordwise discretization: 5 stations * 4 points = 20 points
    assert arr.shape == (20, 3)
    # Legacy: 2 stations * 4 points = 8 points
    assert arr_legacy.shape == (8, 3)

    half_l = 0.5 * PAYLOAD_LENGTH_M
    half_w = 0.5 * PAYLOAD_WIDTH_M
    half_h = 0.5 * PAYLOAD_HEIGHT_M

    # All x points must lie within [front, rear]
    assert np.all(arr[:, 0] >= 3.75 - half_l - 1e-12)
    assert np.all(arr[:, 0] <= 3.75 + half_l + 1e-12)
    # All y points must be root (0) or right edge (half_w) - no negative y mirrors
    assert np.all(np.isclose(arr[:, 1], 0.0) | np.isclose(arr[:, 1], half_w))
    # All z points must be bottom (-half_h) or top (half_h)
    assert np.all(np.isclose(arr[:, 2], -half_h) | np.isclose(arr[:, 2], half_h))


def test_pallet_sample_points_construction():
    rec = csdl.Recorder()
    rec.start()
    pal_cx = csdl.Variable(value=PALLET_STACK_REFERENCE_CG_X, name='pallet_stack_center_x')
    pts = build_geonic_pallet_sample_points(pal_cx)
    pts_legacy = build_geonic_pallet_sample_points(pal_cx, num_chord_points_short=2, num_chord_points_long=2)
    rec.stop()

    sim = csdl.experimental.JaxSimulator(
        recorder=rec,
        additional_inputs=[pal_cx],
        additional_outputs=[pts, pts_legacy],
        gpu=False,
    )
    sim.run()
    arr = np.asarray(sim[pts])
    arr_legacy = np.asarray(sim[pts_legacy])

    # Default chordwise discretization: (3 + 3 + 6 + 6) * 4 = 72 points
    assert arr.shape == (72, 3)
    # Legacy: (2 + 2 + 2 + 2) * 4 = 32 points
    assert arr_legacy.shape == (32, 3)

    # Zero shift since pal_cx == PALLET_STACK_REFERENCE_CG_X
    # Check that y points are non-negative (symmetry reduction)
    assert np.all(arr[:, 1] >= 0.0)
    # Check z bounds
    half_h = 0.5 * PALLET_HEIGHT_M
    assert np.all(np.isclose(arr[:, 2], -half_h) | np.isclose(arr[:, 2], half_h))


def test_clearance_and_margin_convention():
    rec = csdl.Recorder()
    rec.start()
    # Signed distance: -0.25 (inside by 0.25m, clearance = +0.25m, constraint = -0.25 + 0.2 = -0.05 <= 0)
    sdf = csdl.Variable(value=np.array([-0.25, -0.30]), name='sdf')
    con, clr, margin = compute_geonic_clearance_and_margin(sdf, clearance_buffer=0.2)
    rec.stop()

    sim = csdl.experimental.JaxSimulator(
        recorder=rec,
        additional_inputs=[sdf],
        additional_outputs=[con, clr, margin],
        gpu=False,
    )
    sim.run()

    con_vals = np.asarray(sim[con])
    clr_vals = np.asarray(sim[clr])
    margin_val = float(np.asarray(sim[margin]).flatten()[0])

    assert np.allclose(con_vals, np.array([-0.05, -0.10]))
    assert np.allclose(clr_vals, np.array([0.25, 0.30]))
    assert margin_val > 0.0  # Feasible


def test_full_box_corners():
    corners_over = get_full_payload_box_corners(center_x=5.0)
    assert corners_over.shape == (8, 3)
    assert np.all(corners_over[:, 1] <= 0.5 * PAYLOAD_WIDTH_M)
    assert np.all(corners_over[:, 1] >= -0.5 * PAYLOAD_WIDTH_M)

    pallet_boxes = get_full_pallet_boxes_corners(pallet_stack_center_x=PALLET_STACK_REFERENCE_CG_X)
    assert len(pallet_boxes) == 6
    for b in pallet_boxes:
        assert b.shape == (8, 3)


def test_chordwise_discretization_points():
    rec = csdl.Recorder()
    rec.start()
    cx_pay = csdl.Variable(value=4.0, name='cx_pay')
    cx_pal = csdl.Variable(value=10.0, name='cx_pal')

    pts_pay = build_geonic_payload_sample_points(cx_pay, num_chord_points=5)
    pts_pal = build_geonic_pallet_sample_points(cx_pal, num_chord_points_short=3, num_chord_points_long=6)
    rec.stop()

    sim = csdl.experimental.JaxSimulator(
        recorder=rec,
        additional_inputs=[cx_pay, cx_pal],
        additional_outputs=[pts_pay, pts_pal],
        gpu=False,
    )
    sim.run()

    arr_pay = np.asarray(sim[pts_pay])
    arr_pal = np.asarray(sim[pts_pal])

    # Oversized chordwise stations
    unique_x_pay = np.unique(np.round(arr_pay[:, 0], 6))
    assert len(unique_x_pay) == 5
    # Uniform spacing ~2.51 m
    diffs_pay = np.diff(unique_x_pay)
    assert np.allclose(diffs_pay, PAYLOAD_LENGTH_M / 4.0)

    # Pallets chordwise stations: maximum gap anywhere along any box must be <= PALLET_L_SHORT_M (~2.24m)
    # Box 1: 3 x stations
    # Box 2: 3 x stations
    # Box 3: 6 x stations
    # Box 4: 6 x stations
    box1_pts = arr_pal[:12]
    box2_pts = arr_pal[12:24]
    box3_pts = arr_pal[24:48]
    box4_pts = arr_pal[48:72]

    for b_pts, expected_stations in [(box1_pts, 3), (box2_pts, 3), (box3_pts, 6), (box4_pts, 6)]:
        u_x = np.unique(np.round(b_pts[:, 0], 6))
        assert len(u_x) == expected_stations
        diffs = np.diff(u_x)
        assert np.all(diffs <= PALLET_L_SHORT_M + 1e-6)

