import pytest
import numpy as np
import csdl_alpha as csdl
from lsdo_geo.core.geometry.geometry_functions import rotate

def test_rotation_z_axis():
    """Test 90-degree rotation of [1, 0, 0] around Z-axis yields [0, 1, 0]."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    points = np.array([[1.0, 0.0, 0.0]])
    origin = np.array([0.0, 0.0, 0.0])
    axis = np.array([0.0, 0.0, 1.0])
    angle = np.pi / 2.0

    rotated = rotate(points, origin, axis, angle)
    np.testing.assert_allclose(rotated.value, np.array([[0.0, 1.0, 0.0]]), atol=1e-7)

def test_rotation_x_axis():
    """Test 90-degree rotation of [0, 1, 0] around X-axis yields [0, 0, 1]."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    points = np.array([[0.0, 1.0, 0.0]])
    origin = np.array([0.0, 0.0, 0.0])
    axis = np.array([1.0, 0.0, 0.0])
    angle = np.pi / 2.0

    rotated = rotate(points, origin, axis, angle)
    np.testing.assert_allclose(rotated.value, np.array([[0.0, 0.0, 1.0]]), atol=1e-7)

def test_rotation_degrees():
    """Test rotation specifying angle in degrees."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    points = np.array([[1.0, 0.0, 0.0]])
    origin = np.array([0.0, 0.0, 0.0])
    axis = np.array([0.0, 0.0, 1.0])
    angle = 90.0

    rotated = rotate(points, origin, axis, angle, units='degrees')
    np.testing.assert_allclose(rotated.value, np.array([[0.0, 1.0, 0.0]]), atol=1e-7)

def test_rotation_multi_points():
    """Test rotating multiple points simultaneously."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    points = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    origin = np.array([0.0, 0.0, 0.0])
    axis = np.array([0.0, 0.0, 1.0])
    angle = np.pi # 180 degrees around Z

    rotated = rotate(points, origin, axis, angle)
    expected = np.array([
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    np.testing.assert_allclose(rotated.value, expected, atol=1e-7)
