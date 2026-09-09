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
