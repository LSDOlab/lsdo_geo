import os
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

from pathlib import Path
import numpy as np
import pytest
import csdl_alpha as csdl
import lsdo_function_spaces as lfs

SHOWCASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
if SHOWCASE_DIR not in sys.path:
    sys.path.insert(0, SHOWCASE_DIR)

from physics_models.geonic_payload import (
    GEONIC_CLEARANCE_M,
    PAYLOAD_LENGTH_M,
    PAYLOAD_WIDTH_M,
    PAYLOAD_HEIGHT_M,
    PALLET_L_SHORT_M,
    PALLET_WIDTH_M,
    PALLET_HEIGHT_M,
    PALLET_L_LONG_M,
    PALLET_STACK_FRONT_M,
    PALLET_STACK_REFERENCE_CG_X,
    VALID_GEONIC_PAYLOAD_MODES,
    parse_geonic_payload_mode,
    build_geonic_payload_sample_points,
    build_geonic_pallet_sample_points,
    compute_geonic_clearance_and_margin,
)


def test_mode_resolution_and_isolation():
    # Priority check: GEONIC_PAYLOAD_MODE over USE_GEONIC
    mode, use_geo = parse_geonic_payload_mode(env={'GEONIC_PAYLOAD_MODE': 'none', 'USE_GEONIC': '1'})
    assert mode == 'none'
    assert not use_geo

    mode, use_geo = parse_geonic_payload_mode(env={'GEONIC_PAYLOAD_MODE': 'both', 'USE_GEONIC': '0'})
    assert mode == 'both'
    assert use_geo

    # Deprecated fallback check
    mode, use_geo = parse_geonic_payload_mode(env={'USE_GEONIC': '1'})
    assert mode == 'oversized'
    assert use_geo

    mode, use_geo = parse_geonic_payload_mode(env={'USE_GEONIC': '0'})
    assert mode == 'none'
    assert not use_geo

    # Invalid mode
    with pytest.raises(ValueError, match="Invalid geonic_payload_mode"):
        parse_geonic_payload_mode(env={'GEONIC_PAYLOAD_MODE': 'nonexistent'})

    # Subprocess check: GEONIC_PAYLOAD_MODE='none' must not import bsm3
    import subprocess
    cmd = [
        sys.executable, '-c',
        "import os, sys\n"
        "os.environ['GEONIC_PAYLOAD_MODE'] = 'none'\n"
        "from examples.showcase_examples.rectangular_wing.physics_models.geonic_payload import parse_geonic_payload_mode\n"
        "mode, use_geo = parse_geonic_payload_mode()\n"
        "assert not use_geo\n"
        "assert 'bsm3' not in sys.modules\n"
    ]
    res = subprocess.run(cmd, env={**os.environ, 'GEONIC_PAYLOAD_MODE': 'none'}, capture_output=True, text=True)
    assert res.returncode == 0, f"Failed isolation check: {res.stderr}"


def test_multi_mode_point_counts_and_cg_consistency():
    recorder = csdl.Recorder()
    recorder.start()

    cx_pay = csdl.Variable(value=5.0, name='payload_center_x')
    cx_pal = csdl.Variable(value=5.0, name='pallet_stack_center_x')

    # Oversized points (20 points with default 5 chordwise stations)
    pts_over = build_geonic_payload_sample_points(cx_pay)
    # Pallet points (72 points with default chordwise stations)
    pts_pal = build_geonic_pallet_sample_points(cx_pal)
    # Both points (92 points)
    pts_both = csdl.concatenate([pts_over, pts_pal])

    # CG consistency constraint
    cg_consistency = cx_pay - cx_pal

    # Common CG (arithmetic mean)
    common_cg = 0.5 * (cx_pay + cx_pal)

    recorder.stop()

    sim = csdl.experimental.JaxSimulator(
        recorder=recorder,
        additional_inputs=[cx_pay, cx_pal],
        additional_outputs=[pts_over, pts_pal, pts_both, cg_consistency, common_cg],
        gpu=False,
    )
    sim.run()

    arr_over = np.asarray(sim[pts_over])
    arr_pal = np.asarray(sim[pts_pal])
    arr_both = np.asarray(sim[pts_both])
    val_cons = float(np.asarray(sim[cg_consistency]).flatten()[0])
    val_cg = float(np.asarray(sim[common_cg]).flatten()[0])

    n_over = arr_over.shape[0]
    assert arr_over.shape == (20, 3)
    assert arr_pal.shape == (72, 3)
    assert arr_both.shape == (92, 3)
    assert np.allclose(arr_both[:n_over], arr_over)
    assert np.allclose(arr_both[n_over:], arr_pal)

    # Coincident CG
    assert np.isclose(val_cons, 0.0)
    assert np.isclose(val_cg, 5.0)

    # Shift pallet CG by +2.0
    sim[cx_pal] = 7.0
    sim.run()
    val_cons_shifted = float(np.asarray(sim[cg_consistency]).flatten()[0])
    val_cg_shifted = float(np.asarray(sim[common_cg]).flatten()[0])
    assert np.isclose(val_cons_shifted, -2.0)
    assert np.isclose(val_cg_shifted, 6.0)

    # Derivative checks
    totals = sim.compute_totals()
    d_cons_wrt_pay = float(totals[(cg_consistency, cx_pay)].flatten()[0])
    d_cons_wrt_pal = float(totals[(cg_consistency, cx_pal)].flatten()[0])
    assert np.isclose(d_cons_wrt_pay, 1.0)
    assert np.isclose(d_cons_wrt_pal, -1.0)


def test_bsm3_derivatives_vs_finite_difference():
    import bsm3

    rec = csdl.Recorder()
    rec.start()

    from lsdo_geo import import_geometry

    repo_root = Path(__file__).resolve().parents[4]
    cad_file = str(
        repo_root / "examples" / "example_geometries" / "rectangular_wing_naca0012_10ar.stp"
    )
    wing_geom = import_geometry(cad_file, parallelize=False)

    cx_pal = csdl.Variable(value=5.0, name='pallet_stack_center_x')
    pallet_pts = build_geonic_pallet_sample_points(cx_pal)

    model = bsm3.FunctionSetProjectionModel(
        function_set=wing_geom,
        warm_start_nu=25,
        warm_start_nv=25,
        sdf=True,
        sdf_sign_mode='enclosed',
    )
    sdf_op = bsm3.FunctionSetClosestDistanceOperation(model=model)
    pallet_signed_distance = sdf_op.evaluate(
        coefficients=wing_geom.stack_coefficients(),
        points=pallet_pts,
    )
    pallet_con, pallet_clr, pallet_margin = compute_geonic_clearance_and_margin(
        pallet_signed_distance, clearance_buffer=GEONIC_CLEARANCE_M, rho=50.0
    )

    rec.stop()

    sim = csdl.experimental.JaxSimulator(
        recorder=rec,
        additional_inputs=[cx_pal],
        additional_outputs=[pallet_margin],
        gpu=False,
    )
    sim.run()

    totals = sim.compute_totals()
    d_margin_exact = float(totals[(pallet_margin, cx_pal)].flatten()[0])

    # Centered finite differences
    h = 1e-5
    sim[cx_pal] = 5.0 + h
    sim.run()
    m_plus = float(np.asarray(sim[pallet_margin]).flatten()[0])

    sim[cx_pal] = 5.0 - h
    sim.run()
    m_minus = float(np.asarray(sim[pallet_margin]).flatten()[0])

    d_margin_fd = (m_plus - m_minus) / (2.0 * h)

    rel_err = abs(d_margin_exact - d_margin_fd) / (abs(d_margin_fd) + 1e-8)
    assert rel_err < 1e-3, f"Derivative mismatch: exact={d_margin_exact}, fd={d_margin_fd}, rel_err={rel_err}"
