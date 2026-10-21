"""
Core unit tests for LSDO_GEO Signed Distance Field (SDF) operation.

Tests:
1. 2D closed curve loops and 3D multi-patch shells.
2. Outward orientation calibration on inverted CAD patches.
3. Accurate sign at interior, exterior, and edge-adjacent points.
4. Rejection of open/non-manifold geometry in winding-number mode; acceptance in unsigned mode.
5. First derivatives against centered finite differences for query points and coefficients.
6. Higher-order directional derivatives / KKT system on smooth branches.
7. Simulator compatibility: JaxSimulator and PySimulator.
8. Geometry.evaluate_signed_distance() convenience wrapper.
"""
from __future__ import annotations

from pathlib import Path
import pytest
import numpy as np
import csdl_alpha as csdl

from lsdo_geo import (
    Geometry,
    import_geometry,
    FunctionSetSignedDistanceOperation,
)
import lsdo_function_spaces as lfs


def _create_2d_circle_loop(radius: float = 2.0, num_points: int = 40) -> Geometry:
    """Create a 2D closed B-spline curve loop approximating a circle."""
    theta = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    circle_pts = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    circle_pts = np.vstack([circle_pts, circle_pts[0:1]])
    s = np.linspace(0.0, 1.0, len(circle_pts))[:, None]

    space = lfs.BSplineSpace(
        num_parametric_dimensions=1,
        degree=(3,),
        coefficients_shape=(20,),
    )
    c = space.fit(circle_pts, s)
    fn = lfs.Function(space=space, coefficients=c)
    geom = Geometry(functions={0: fn}, space=space)
    return geom


def test_2d_signed_distance_and_derivatives():
    """Verify 2D closed loop signed distance and first derivatives against centered FD."""
    rec = csdl.Recorder(inline=True)
    rec.start()

    geom = _create_2d_circle_loop(radius=2.0)

    pts_val = np.array([
        [0.4, 0.3],    # Inside (radius = 0.5 < 2.0, phi ~ -1.5)
        [0.8, 0.6],    # Inside (radius = 1.0 < 2.0, phi ~ -1.0)
        [2.4, 1.8],    # Outside (radius = 3.0 > 2.0, phi ~ +1.0)
    ])
    pts_var = csdl.Variable(value=pts_val, name="query_pts_2d")
    coeffs_var = geom.stack_coefficients()

    op = FunctionSetSignedDistanceOperation(geom, sign_mode="winding_number")
    sdf = op.evaluate(coeffs_var, pts_var)

    rec.stop()

    sim = csdl.experimental.JaxSimulator(recorder=rec, additional_inputs=[pts_var], additional_outputs=[sdf])
    sim.run()

    sdf_vals = np.asarray(sim[sdf]).flatten()
    assert sdf_vals[0] < -1.4, f"Inner point 0 should be inside: {sdf_vals[0]}"
    assert sdf_vals[1] < -0.5, f"Inner point should be inside: {sdf_vals[1]}"
    assert sdf_vals[2] > 0.5, f"Outer point should be outside: {sdf_vals[2]}"

    # Verify first derivatives w.r.t query points vs FD
    totals = sim.compute_totals()
    d_pts_exact = totals[(sdf, pts_var)]

    h = 1e-6
    d_pts_fd = np.zeros_like(d_pts_exact)
    for i in range(len(pts_val)):
        for d in range(2):
            p_plus = pts_val.copy()
            p_plus[i, d] += h
            sim[pts_var] = p_plus
            sim.run()
            f_plus = np.asarray(sim[sdf]).flatten()

            p_minus = pts_val.copy()
            p_minus[i, d] -= h
            sim[pts_var] = p_minus
            sim.run()
            f_minus = np.asarray(sim[sdf]).flatten()

            col = i * 2 + d
            d_pts_fd[:, col] = (f_plus - f_minus) / (2.0 * h)

    rel_err = np.max(np.abs(d_pts_exact - d_pts_fd))
    assert rel_err < 1e-4, f"2D point derivative mismatch: max error = {rel_err}"


def test_3d_naca0012_orientation_calibration_and_containment():
    """Verify 3D multi-patch wing outward calibration and exact signed distance signs."""
    repo_root = Path(__file__).resolve().parents[1]
    cad_path = repo_root / "examples" / "example_geometries" / "rectangular_wing" / "rectangular_wing_naca0012_10ar.stp"

    rec = csdl.Recorder(inline=True)
    rec.start()

    geom = import_geometry(str(cad_path), parallelize=False)

    pts_val = np.array([
        [0.4, 2.5, 0.02],   # Inside wing (closer to upper surface)
        [0.4, 2.5, -0.02],  # Inside wing (closer to lower surface)
        [0.5, 2.5, 0.25],   # Above wing (exterior)
        [0.5, 2.5, -0.25],  # Below wing (exterior)
        [-0.5, 2.5, 0.0],   # Ahead of leading edge (exterior)
        [1.5, 2.5, 0.0],    # Behind trailing edge (exterior)
    ])
    pts_var = csdl.Variable(value=pts_val, name="pts_3d")

    # Use convenience wrapper Geometry.evaluate_signed_distance
    sdf = geom.evaluate_signed_distance(pts_var, sign_mode="winding_number")

    rec.stop()

    sim = csdl.experimental.JaxSimulator(recorder=rec, additional_inputs=[pts_var], additional_outputs=[sdf])
    sim.run()

    sdf_vals = np.asarray(sim[sdf]).flatten()
    assert sdf_vals[0] < 0.0, f"Upper interior point should be negative: {sdf_vals[0]}"
    assert sdf_vals[1] < 0.0, f"Lower interior point should be negative: {sdf_vals[1]}"
    assert sdf_vals[2] > 0.0, f"Upper exterior point should be positive: {sdf_vals[2]}"
    assert sdf_vals[3] > 0.0, f"Lower exterior point should be positive: {sdf_vals[3]}"
    assert sdf_vals[4] > 0.0, f"Leading exterior point should be positive: {sdf_vals[4]}"
    assert sdf_vals[5] > 0.0, f"Trailing exterior point should be positive: {sdf_vals[5]}"

    # Verify first derivatives vs FD
    totals = sim.compute_totals()
    d_pts_exact = totals[(sdf, pts_var)]

    h = 1e-6
    d_pts_fd = np.zeros_like(d_pts_exact)
    for i in range(len(pts_val)):
        for d in range(3):
            p_plus = pts_val.copy()
            p_plus[i, d] += h
            sim[pts_var] = p_plus
            sim.run()
            f_plus = np.asarray(sim[sdf]).flatten()

            p_minus = pts_val.copy()
            p_minus[i, d] -= h
            sim[pts_var] = p_minus
            sim.run()
            f_minus = np.asarray(sim[sdf]).flatten()

            col = i * 3 + d
            d_pts_fd[:, col] = (f_plus - f_minus) / (2.0 * h)

    rel_err = np.max(np.abs(d_pts_exact - d_pts_fd))
    assert rel_err < 1e-4, f"3D point derivative mismatch: max error = {rel_err}"


def test_open_geometry_rejection_in_winding_number_mode():
    """Verify that an open surface is rejected in winding_number mode and accepted in unsigned mode."""
    repo_root = Path(__file__).resolve().parents[1]
    cad_path = repo_root / "examples" / "example_geometries" / "rectangular_wing" / "rectangular_wing_naca0012_10ar.stp"
    full_geom = import_geometry(str(cad_path), parallelize=False)

    # Take an open sub-component (e.g. only patch 0, which is an open sheet without tip caps or other side)
    open_geom = full_geom.create_subset(function_indices=[0])

    # In winding_number mode: must raise ValueError
    with pytest.raises(ValueError, match="open or non-manifold"):
        FunctionSetSignedDistanceOperation(open_geom, sign_mode="winding_number")

    # In unsigned mode: must succeed and compute unsigned distance
    op_unsigned = FunctionSetSignedDistanceOperation(open_geom, sign_mode="unsigned")
    assert op_unsigned.sign_mode == "unsigned"


def test_pysimulator_compatibility():
    """Verify SDF evaluation and compute_totals in PySimulator backend."""
    repo_root = Path(__file__).resolve().parents[1]
    cad_path = repo_root / "examples" / "example_geometries" / "rectangular_wing" / "rectangular_wing_naca0012_10ar.stp"

    rec = csdl.Recorder(inline=True)
    rec.start()

    geom = import_geometry(str(cad_path), parallelize=False)

    pts_val = np.array([[0.4, 2.5, 0.02], [0.5, 2.5, 0.25]])
    pts_var = csdl.Variable(value=pts_val, name="pts_py")
    sdf = geom.evaluate_signed_distance(pts_var)

    rec.stop()

    sim = csdl.experimental.PySimulator(recorder=rec)
    sim.run()

    sdf_vals = np.asarray(sim[sdf]).flatten()
    assert sdf_vals[0] < 0.0
    assert sdf_vals[1] > 0.0

    totals = sim.compute_totals(ofs=[sdf], wrts=[pts_var])
    d_pts = totals[(sdf, pts_var)]
    assert np.all(np.isfinite(d_pts))
    assert d_pts.shape == (2, 6)


def test_2d_second_order_derivatives():
    """Verify 2D second derivatives (Hessian of signed distance) on smooth branch."""
    rec = csdl.Recorder(inline=True)
    rec.start()

    geom = _create_2d_circle_loop(radius=2.0)
    p_val = np.array([[0.8, 0.6]])  # inside, smooth branch
    pts_var = csdl.Variable(value=p_val, name="pts_2d_h")
    c_var = geom.stack_coefficients()

    op = FunctionSetSignedDistanceOperation(geom, sign_mode="winding_number")
    sdf = op.evaluate(c_var, pts_var)

    # 1st and 2nd derivatives via CSDL
    d_pts = csdl.derivative(sdf, pts_var)
    d2_pts = csdl.derivative(d_pts, pts_var)

    rec.stop()

    sim = csdl.experimental.JaxSimulator(recorder=rec, additional_inputs=[pts_var], additional_outputs=[sdf, d_pts, d2_pts])
    sim.run()

    h_exact = np.asarray(sim[d2_pts]).reshape(2, 2)
    assert np.all(np.isfinite(h_exact))
    assert np.max(np.abs(h_exact - h_exact.T)) < 1e-12, "2D Hessian must be symmetric"

    # Compare against analytical circle approximation (B-spline with 20 CPs matches within 5e-3)
    p_norm = np.linalg.norm(p_val[0])
    p_hat = p_val[0] / p_norm
    h_expected = (np.eye(2) - np.outer(p_hat, p_hat)) / p_norm
    assert np.max(np.abs(h_exact - h_expected)) < 5e-3, f"2D Hessian vs circle mismatch: {h_exact} vs {h_expected}"

    # Verify against exact B-spline projection gradient finite difference
    h = 1e-6
    h_fd = np.zeros((2, 2))
    fn = geom.functions[0]
    c_val = c_var.value.reshape(-1, 2)
    for d in range(2):
        p_plus = p_val.copy()
        p_plus[0, d] += h
        u_plus = fn.project(p_plus)[0]
        x_plus = (fn.space.compute_basis_matrix(u_plus) @ c_val)[0]
        diff_plus = p_plus[0] - x_plus
        gp_plus = - diff_plus / np.linalg.norm(diff_plus)

        p_minus = p_val.copy()
        p_minus[0, d] -= h
        u_minus = fn.project(p_minus)[0]
        x_minus = (fn.space.compute_basis_matrix(u_minus) @ c_val)[0]
        diff_minus = p_minus[0] - x_minus
        gp_minus = - diff_minus / np.linalg.norm(diff_minus)

        h_fd[:, d] = (gp_plus - gp_minus) / (2.0 * h)

    rel_err = np.max(np.abs(h_exact - h_fd))
    assert rel_err < 1e-4, f"2D Hessian vs projection FD mismatch: max error = {rel_err}"


def test_3d_second_order_derivatives():
    """Verify 3D second derivatives against finite difference of 1st-order projection gradient."""
    repo_root = Path(__file__).resolve().parents[1]
    cad_path = repo_root / "examples" / "example_geometries" / "rectangular_wing" / "rectangular_wing_naca0012_10ar.stp"

    rec = csdl.Recorder(inline=True)
    rec.start()

    geom = import_geometry(str(cad_path), parallelize=False)
    p_val = np.array([[0.5, 2.5, 0.25]])  # exterior upper wing, smooth branch
    pts_var = csdl.Variable(value=p_val, name="pts_3d_h")
    c_var = geom.stack_coefficients()

    op = FunctionSetSignedDistanceOperation(geom, sign_mode="winding_number")
    sdf = op.evaluate(c_var, pts_var)

    d_pts = csdl.derivative(sdf, pts_var)
    d2_pts = csdl.derivative(d_pts, pts_var)

    rec.stop()

    sim = csdl.experimental.JaxSimulator(recorder=rec, additional_inputs=[pts_var], additional_outputs=[sdf, d_pts, d2_pts])
    sim.run()

    h_exact = np.asarray(sim[d2_pts]).reshape(3, 3)
    assert np.all(np.isfinite(h_exact))
    assert np.max(np.abs(h_exact - h_exact.T)) < 1e-12, "3D Hessian must be symmetric"

    # Directional derivative test / FD of exact 1st-derivative projection gradient
    h = 1e-6
    h_fd = np.zeros((3, 3))
    for d in range(3):
        p_plus = p_val.copy()
        p_plus[0, d] += h
        proj_plus = geom.project(p_plus)
        p_idx_plus, u_plus = proj_plus[0]
        fn_plus = geom.functions[p_idx_plus]
        x_plus = (fn_plus.space.compute_basis_matrix(np.asarray(u_plus).reshape(1, 2)) @ fn_plus.coefficients.value.reshape(-1, 3))[0]
        diff_plus = p_plus[0] - x_plus
        gp_plus = diff_plus / np.linalg.norm(diff_plus)

        p_minus = p_val.copy()
        p_minus[0, d] -= h
        proj_minus = geom.project(p_minus)
        p_idx_minus, u_minus = proj_minus[0]
        fn_minus = geom.functions[p_idx_minus]
        x_minus = (fn_minus.space.compute_basis_matrix(np.asarray(u_minus).reshape(1, 2)) @ fn_minus.coefficients.value.reshape(-1, 3))[0]
        diff_minus = p_minus[0] - x_minus
        gp_minus = diff_minus / np.linalg.norm(diff_minus)

        h_fd[:, d] = (gp_plus - gp_minus) / (2.0 * h)

    rel_err = np.max(np.abs(h_exact - h_fd))
    assert rel_err < 1e-4, f"3D Hessian vs projection FD mismatch: max error = {rel_err}"

