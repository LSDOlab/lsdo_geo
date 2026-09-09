import pytest
import numpy as np
import csdl_alpha as csdl
import lsdo_geo as lg
import os

SAMPLE_STP = os.path.join(
    os.path.dirname(__file__), "..", "examples", "example_geometries", "rectangular_wing.stp"
)

def test_geometry_import():
    """Test importing CAD geometry from STEP file."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    geo = lg.import_geometry(SAMPLE_STP, name="test_wing", parallelize=False)
    assert geo.name == "test_wing"
    assert len(geo.functions) > 0

def test_geometry_copy():
    """Test creating an independent copy of a geometry."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    geo = lg.import_geometry(SAMPLE_STP, name="orig_wing", parallelize=False)
    geo_copy = geo.copy()
    assert geo_copy.name == geo.name
    assert len(geo_copy.functions) == len(geo.functions)

def test_geometry_translate():
    """Test rigid body translation of geometry coefficients."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    geo = lg.import_geometry(SAMPLE_STP, parallelize=False)
    f0 = list(geo.functions.values())[0]
    orig_coeffs = np.copy(f0.coefficients.value)
    
    translation = np.array([2.5, -1.0, 3.0])
    geo.translate(translation)
    
    new_coeffs = f0.coefficients.value
    np.testing.assert_allclose(new_coeffs, orig_coeffs + translation, atol=1e-7)

def test_geometry_rotate():
    """Test rotating geometry about an axis."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    geo = lg.import_geometry(SAMPLE_STP, parallelize=False)
    origin = np.array([0.0, 0.0, 0.0])
    axis = np.array([0.0, 0.0, 1.0])
    
    geo.rotate(rotation_origin=origin, axis_vector=axis, angles=90.0)
    for func in geo.functions.values():
        assert not np.isnan(func.coefficients.value).any()

def test_geometry_rotate_using_quaternion():
    """Test rotating geometry using quaternion representation."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    geo = lg.import_geometry(SAMPLE_STP, parallelize=False)
    origin = np.array([0.0, 0.0, 0.0])
    # 90 degrees around Z axis in quaternion [w, x, y, z] -> [cos(45°), 0, 0, sin(45°)]
    theta = np.pi / 4.0
    quat = np.array([np.cos(theta), 0.0, 0.0, np.sin(theta)])

    geo.rotate_using_quaternion(rotation_origin=origin, quaternion=quat)
    for func in geo.functions.values():
        assert not np.isnan(func.coefficients.value).any()
