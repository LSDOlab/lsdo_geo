"""
Tests for sdfpack geometric non-interference enforcement.

Verifies:
1. Method parsing and defaults ('selective_aggregation' vs 'sdfpack').
2. Analytic box and pallet stack SDFs and spatial/translation gradients.
3. CSDL CustomExplicitOperation forward evaluation and analytic sensitivities via JaxSimulator.
4. Agreement between analytic derivatives and centered finite differences.
"""
from __future__ import annotations

import os
import pytest
import numpy as np
import csdl_alpha as csdl
import bsm3
import lsdo_function_spaces as lfs
import sys
from pathlib import Path

SHOWCASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
if SHOWCASE_DIR not in sys.path:
    sys.path.insert(0, SHOWCASE_DIR)

from physics_models.geonic_payload import (
    GEONIC_CLEARANCE_M,
    VALID_GEONIC_METHODS,
    parse_geonic_method,
)
from physics_models.geonic_sdfpack import (
    compute_box_sdf_and_grad,
    pallet_stack_sdf_and_grad,
    SDFPackOversizedInterferenceOperation,
    SDFPackPalletsInterferenceOperation,
    evaluate_sdfpack_geonic,
)

DEFAULT_STEP_PATH = Path("/home/andrew/optimization/BSM3/examples/basic_examples/swept_wing.stp")


def test_geonic_method_parsing():
    """Verify method parsing, default fallback, and invalid value rejection."""
    assert parse_geonic_method(None, env={}) == "selective_aggregation"
    assert parse_geonic_method("selective_aggregation", env={}) == "selective_aggregation"
    assert parse_geonic_method("sdfpack", env={}) == "sdfpack"
    assert parse_geonic_method("SDFPACK", env={}) == "sdfpack"
    assert parse_geonic_method(None, env={"GEONIC_METHOD": "sdfpack"}) == "sdfpack"

    with pytest.raises(ValueError, match="Invalid geonic_method"):
        parse_geonic_method("unsupported_method", env={})


def test_box_sdf_and_grad_accuracy():
    """Verify analytical box SDF values and spatial gradients against finite differences."""
    center = np.array([1.0, 2.0, -0.5])
    half_lengths = np.array([2.5, 1.0, 0.75])

    test_points = np.array([
        [1.5, 2.2, -0.4],       # inside point
        [3.0, 2.3, -0.3],       # inside along x
        [4.0, 2.2, -0.3],       # outside along x
        [4.0, 3.5, 1.0],        # outside corner
        [0.0, 2.8, -0.2],       # inside general
    ])

    sdf, grad = compute_box_sdf_and_grad(test_points, center, half_lengths)
    assert sdf[0] < 0.0, "Interior point should have negative SDF"
    assert sdf[2] > 0.0, "Exterior point should have positive SDF"

    # Centered finite differences for spatial gradient
    eps = 1e-6
    for i in range(len(test_points)):
        p = test_points[i]
        fd_grad = np.zeros(3)
        for dim in range(3):
            p_plus = p.copy()
            p_plus[dim] += eps
            p_minus = p.copy()
            p_minus[dim] -= eps
            s_plus, _ = compute_box_sdf_and_grad(p_plus, center, half_lengths)
            s_minus, _ = compute_box_sdf_and_grad(p_minus, center, half_lengths)
            fd_grad[dim] = (s_plus[0] - s_minus[0]) / (2 * eps)

        np.testing.assert_allclose(
            grad[i],
            fd_grad,
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"Box gradient mismatch at point {p}",
        )


def test_pallet_stack_sdf_and_grad():
    """Verify composite 6-pallet stack SDF and translation sensitivity."""
    cx = 13.8166
    pts = np.array([
        [5.5, 0.0, 0.0],     # inside front short
        [10.0, 1.0, 0.0],    # inside long pallet
        [20.0, 5.0, 2.0],    # far outside
    ])

    sdf, grad = pallet_stack_sdf_and_grad(pts, cx)
    assert sdf[0] < 0.0, "Point inside front pallet must have negative SDF"
    assert sdf[2] > 0.0, "Far point must have positive SDF"

    # Finite difference w.r.t pallet stack center x
    eps = 1e-6
    sdf_p, _ = pallet_stack_sdf_and_grad(pts, cx + eps)
    sdf_m, _ = pallet_stack_sdf_and_grad(pts, cx - eps)
    fd_dcx = (sdf_p - sdf_m) / (2 * eps)

    analytic_dcx = -grad[:, 0]
    np.testing.assert_allclose(analytic_dcx, fd_dcx, rtol=1e-5, atol=1e-6)


def test_sdfpack_oversized_op_jax_derivatives():
    """Verify SDFPackOversizedInterferenceOperation CSDL/JAX sensitivity matches finite difference."""
    wing_fun_set = lfs.import_file_patched(file_name=str(DEFAULT_STEP_PATH), parallelize=False)
    projection_model = bsm3.FunctionSetProjectionModel(
        function_set=wing_fun_set,
        warm_start_nu=15,
        warm_start_nv=15,
        sdf=True,
        sdf_sign_mode="enclosed",
    )
    reshaped_coeffs = []
    for _, fun in wing_fun_set.functions.items():
        coeffs = fun.coefficients
        reshaped_coeffs.append(coeffs.value.reshape((-1, 3)))
    scale = 7.5
    coeffs_np = np.vstack(reshaped_coeffs) * scale

    rec = csdl.Recorder(inline=True)
    rec.start()
    c_var = csdl.Variable(value=coeffs_np)
    c_var.set_as_design_variable(lower=-100.0, upper=100.0)
    cx_var = csdl.Variable(value=np.array([3.75]))
    cx_var.set_as_design_variable(lower=0.0, upper=10.0)

    # Use a penetrating box so interference is positive and gradients active
    op = SDFPackOversizedInterferenceOperation(
        projection_model=projection_model,
        clearance=0.2,
        max_grid_subdivisions=4,
        smoothing_radius=0.01,
    )
    # Artificially enlarge half lengths for test to ensure penetration
    op.half_lengths = np.array([2.0, 1.0, 1.0])

    interf = op.evaluate(c_var, cx_var)
    interf.set_as_objective()

    sim = csdl.experimental.JaxSimulator(recorder=rec, gpu=False)
    sim.run()
    val = float(sim[interf][0])
    assert val > 0.0, f"Expected positive interference for penetrating test box, got {val}"

    totals = sim.compute_totals()
    d_val_dcx_analytic = float(totals[interf, cx_var][0, 0])

    # Centered finite difference for d_val / d_cx
    eps = 1e-5
    sim[cx_var] = np.array([3.75 + eps])
    sim.run()
    val_p = float(sim[interf][0])

    sim[cx_var] = np.array([3.75 - eps])
    sim.run()
    val_m = float(sim[interf][0])

    fd_dcx = (val_p - val_m) / (2 * eps)

    np.testing.assert_allclose(
        d_val_dcx_analytic,
        fd_dcx,
        rtol=1e-4,
        atol=1e-5,
        err_msg="Analytic d_interference/d_payload_center_x does not match finite difference",
    )


def test_sdfpack_pallets_op_jax_derivatives():
    """Verify SDFPackPalletsInterferenceOperation CSDL/JAX sensitivity matches finite difference."""
    wing_fun_set = lfs.import_file_patched(file_name=str(DEFAULT_STEP_PATH), parallelize=False)
    projection_model = bsm3.FunctionSetProjectionModel(
        function_set=wing_fun_set,
        warm_start_nu=15,
        warm_start_nv=15,
        sdf=True,
        sdf_sign_mode="enclosed",
    )
    reshaped_coeffs = []
    for _, fun in wing_fun_set.functions.items():
        coeffs = fun.coefficients
        reshaped_coeffs.append(coeffs.value.reshape((-1, 3)))
    scale = 7.5
    coeffs_np = np.vstack(reshaped_coeffs) * scale

    rec = csdl.Recorder(inline=True)
    rec.start()
    c_var = csdl.Variable(value=coeffs_np)
    c_var.set_as_design_variable(lower=-100.0, upper=100.0)
    cx_var = csdl.Variable(value=np.array([12.0]))
    cx_var.set_as_design_variable(lower=0.0, upper=20.0)

    op = SDFPackPalletsInterferenceOperation(
        projection_model=projection_model,
        clearance=0.2,
        max_grid_subdivisions=4,
        smoothing_radius=0.01,
        bbox=[np.array([-5.0, -5.0, -2.5]), np.array([20.0, 5.0, 2.5])],
    )

    interf = op.evaluate(c_var, cx_var)
    interf.set_as_objective()

    sim = csdl.experimental.JaxSimulator(recorder=rec, gpu=False)
    sim.run()
    val = float(sim[interf][0])
    assert val > 0.0, f"Expected positive interference for penetrating pallets, got {val}"

    totals = sim.compute_totals()
    d_val_dcx_analytic = float(totals[interf, cx_var][0, 0])

    # Centered finite difference for d_val / d_cx
    eps = 1e-5
    sim[cx_var] = np.array([12.0 + eps])
    sim.run()
    val_p = float(sim[interf][0])

    sim[cx_var] = np.array([12.0 - eps])
    sim.run()
    val_m = float(sim[interf][0])

    fd_dcx = (val_p - val_m) / (2 * eps)

    np.testing.assert_allclose(
        d_val_dcx_analytic,
        fd_dcx,
        rtol=1e-4,
        atol=1e-5,
        err_msg="Analytic d_interference/d_pallet_stack_center_x does not match finite difference",
    )


def test_sdfpack_both_mode_evaluation_and_derivatives():
    """Verify evaluate_sdfpack_geonic in 'both' mode on NACA0012 geometry."""
    from lsdo_geo import import_geometry

    repo_root = Path(__file__).resolve().parents[4]
    cad_path = repo_root / "examples" / "example_geometries" / "rectangular_wing" / "rectangular_wing_naca0012_10ar.stp"

    rec = csdl.Recorder(inline=True)
    rec.start()

    wing_geom = import_geometry(str(cad_path), parallelize=False)

    model = bsm3.FunctionSetProjectionModel(
        function_set=wing_geom,
        warm_start_nu=15,
        warm_start_nv=15,
        sdf=True,
        sdf_sign_mode="enclosed",
    )

    cx_pay = csdl.Variable(value=np.array([3.75]))
    cx_pay.set_as_design_variable(lower=0.0, upper=10.0)
    cx_pal = csdl.Variable(value=np.array([5.0]))
    cx_pal.set_as_design_variable(lower=0.0, upper=10.0)

    coeffs = wing_geom.stack_coefficients()

    interf_pay, interf_pal = evaluate_sdfpack_geonic(
        projection_model=model,
        coefficients=coeffs,
        payload_mode="both",
        payload_center_x=cx_pay,
        pallet_stack_center_x=cx_pal,
        clearance=0.2,
        max_grid_subdivisions=4,
        smoothing_radius=0.01,
    )
    combined = interf_pay + interf_pal
    combined.set_as_objective()

    sim = csdl.experimental.JaxSimulator(
        recorder=rec,
        additional_outputs=[interf_pay, interf_pal],
        gpu=False,
    )
    sim.run()

    val_pay = float(sim[interf_pay][0])
    val_pal = float(sim[interf_pal][0])
    assert val_pay > 0.0, f"Expected positive oversized interference, got {val_pay}"
    assert val_pal > 0.0, f"Expected positive pallet interference, got {val_pal}"

    totals = sim.compute_totals()
    d_comb_dcx_pay_analytic = float(totals[combined, cx_pay][0, 0])
    d_comb_dcx_pal_analytic = float(totals[combined, cx_pal][0, 0])

    eps = 1e-5
    # FD w.r.t cx_pay
    sim[cx_pay] = np.array([3.75 + eps])
    sim.run()
    comb_p = float(sim[combined][0])
    sim[cx_pay] = np.array([3.75 - eps])
    sim.run()
    comb_m = float(sim[combined][0])
    fd_dcx_pay = (comb_p - comb_m) / (2 * eps)
    sim[cx_pay] = np.array([3.75])

    # FD w.r.t cx_pal
    sim[cx_pal] = np.array([5.0 + eps])
    sim.run()
    comb_p = float(sim[combined][0])
    sim[cx_pal] = np.array([5.0 - eps])
    sim.run()
    comb_m = float(sim[combined][0])
    fd_dcx_pal = (comb_p - comb_m) / (2 * eps)

    np.testing.assert_allclose(
        d_comb_dcx_pay_analytic,
        fd_dcx_pay,
        rtol=1e-4,
        atol=1e-5,
        err_msg="Both mode d_interference/d_payload_center_x mismatch",
    )
    np.testing.assert_allclose(
        d_comb_dcx_pal_analytic,
        fd_dcx_pal,
        rtol=1e-4,
        atol=1e-5,
        err_msg="Both mode d_interference/d_pallet_stack_center_x mismatch",
    )


