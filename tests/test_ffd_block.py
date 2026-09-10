import pytest
import numpy as np
import csdl_alpha as csdl
import lsdo_geo as lg
import os

SAMPLE_STP = os.path.join(
    os.path.dirname(__file__), "..", "examples", "example_geometries", "rectangular_wing.stp"
)

def test_ffd_block_construction_around_points():
    """Test creating an FFD block enclosing a set of 3D points."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    pts = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [2.0, 3.0, 1.0],
    ])
    pts_var = csdl.Variable(value=pts)

    ffd = lg.construct_ffd_block_around_entities(
        name="box_ffd",
        entities=pts_var,
        num_coefficients=(3, 3, 2),
        degree=(1, 1, 1),
    )

    assert ffd.name == "box_ffd"
    assert ffd.coefficients.shape == (3, 3, 2, 3)

def test_ffd_block_parametric_mapping():
    """Test that evaluating parametric coordinates (u, v, w) through FFD maps to correct physical box."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    pts = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 4.0, 6.0],
    ])
    pts_var = csdl.Variable(value=pts)

    ffd = lg.construct_ffd_block_around_entities(
        name="identity_ffd",
        entities=pts_var,
        num_coefficients=(2, 2, 2),
        degree=(1, 1, 1),
    )

    uvw = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5],
    ])
    evaluated_pts = ffd.evaluate(uvw)
    
    expected = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 4.0, 6.0],
        [1.0, 2.0, 3.0],
    ])
    np.testing.assert_allclose(evaluated_pts.value, expected, atol=1e-5)

def test_ffd_block_construction_around_geometry():
    """Test creating an FFD block enclosing an imported CAD geometry."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    geo = lg.import_geometry(SAMPLE_STP, parallelize=False)
    ffd = lg.construct_ffd_block_around_entities(
        name="wing_ffd",
        entities=geo,
        num_coefficients=(2, 5, 2),
        degree=(1, 2, 1),
    )

    assert ffd.name == "wing_ffd"
    assert ffd.coefficients.shape == (2, 5, 2, 3)
    assert not np.isnan(ffd.coefficients.value).any()


def test_construct_ffd_block_from_corners():
    """Test constructing an FFD block directly from corner coordinates."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    # Define a unit cube with 8 corners of shape (2, 2, 2, 3)
    corners = np.zeros((2, 2, 2, 3))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                corners[i, j, k] = [float(i), float(j), float(k)]

    pts = np.array([[0.25, 0.5, 0.75]])
    pts_var = csdl.Variable(value=pts)

    ffd = lg.construct_ffd_block_from_corners(
        entities=pts_var,
        corners=corners,
        num_coefficients=(2, 2, 2),
        degree=(1, 1, 1),
        name="corners_ffd",
    )

    assert ffd.name == "corners_ffd"
    assert ffd.coefficients.shape == (2, 2, 2, 3)

    out = ffd.evaluate_ffd(ffd.coefficients)
    np.testing.assert_allclose(out.value, pts, atol=1e-5)


def test_ffd_block_evaluate_ffd_perturbation():
    """Test that perturbing FFD block coefficients deforms embedded points as expected."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    corners = np.zeros((2, 2, 2, 3))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                corners[i, j, k] = [float(i), float(j), float(k)]

    pts = np.array([[0.5, 0.5, 0.5]])
    pts_var = csdl.Variable(value=pts)

    ffd = lg.construct_ffd_block_from_corners(
        entities=pts_var,
        corners=corners,
        num_coefficients=(2, 2, 2),
        degree=(1, 1, 1),
    )

    # Perturb the (1, 1, 1) corner by +1 in x direction
    delta = np.zeros_like(ffd.coefficients.value)
    delta[1, 1, 1, 0] = 1.0
    new_coeffs = csdl.Variable(value=ffd.coefficients.value + delta)

    deformed_pts = ffd.evaluate_ffd(new_coeffs)
    # The center point (0.5, 0.5, 0.5) should move by 1/8 in x
    assert deformed_pts.value[0, 0] > pts[0, 0]
    np.testing.assert_allclose(deformed_pts.value[0, 1:], pts[0, 1:], atol=1e-5)


def test_ffd_block_multiple_entities_and_non_csdl():
    """Test embedding multiple entities and evaluating with non_csdl=True."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    pts1 = np.array([[0.2, 0.3, 0.4]])
    pts2 = np.array([[0.6, 0.7, 0.8], [0.1, 0.2, 0.3]])

    ffd = lg.construct_ffd_block_around_entities(
        entities=[pts1, pts2],
        num_coefficients=(2, 2, 2),
        degree=(1, 1, 1),
        name="multi_ffd",
    )

    assert len(ffd.embedded_entities) == 2

    # CSDL evaluation
    outputs_csdl = ffd.evaluate_ffd(ffd.coefficients)
    assert len(outputs_csdl) == 2
    np.testing.assert_allclose(outputs_csdl[0].value, pts1, atol=1e-4)
    np.testing.assert_allclose(outputs_csdl[1].value, pts2, atol=1e-4)

    # non_csdl evaluation
    outputs_np = ffd.evaluate_ffd(ffd.coefficients, non_csdl=True)
    assert len(outputs_np) == 2
    assert isinstance(outputs_np[0], np.ndarray)
    np.testing.assert_allclose(outputs_np[0].reshape(-1, 3), pts1, atol=1e-4)
    np.testing.assert_allclose(outputs_np[1], pts2, atol=1e-4)

