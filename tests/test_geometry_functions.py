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

def test_rotation_non_unit_axis():
    """Test rotation around a non-unit axis rotates by the exact angle."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    points = np.array([[1.0, 0.0, 0.0]])
    origin = np.array([0.0, 0.0, 0.0])
    axis = np.array([0.0, 0.0, 5.0]) # non-unit axis
    angle = np.pi / 2.0

    rotated = rotate(points, origin, axis, angle)
    np.testing.assert_allclose(rotated.value, np.array([[0.0, 1.0, 0.0]]), atol=1e-7)

    # Oblique non-unit axis
    axis_oblique = np.array([1.0, 1.0, 0.0]) # norm is sqrt(2)
    rotated_oblique = rotate(points, origin, axis_oblique, np.pi)
    # 180 degrees around [1, 1, 0] reflects (1, 0, 0) to (0, 1, 0)
    np.testing.assert_allclose(rotated_oblique.value, np.array([[0.0, 1.0, 0.0]]), atol=1e-7)

def test_rotation_multiple_angles():
    """Test rotating points with multiple angles simultaneously."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    points = np.array([[1.0, 0.0, 0.0]])
    origin = np.array([0.0, 0.0, 0.0])
    axis = np.array([0.0, 0.0, 1.0])
    angles = np.array([0.0, np.pi / 2.0, np.pi])

    rotated = rotate(points, origin, axis, angles)
    assert rotated.shape == (3, 1, 3)
    expected = np.array([
        [[1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0]],
        [[-1.0, 0.0, 0.0]],
    ])
    np.testing.assert_allclose(rotated.value, expected, atol=1e-7)


def test_rodrigues_vs_quaternion_equivalence():
    """Verify that Rodrigues rotation matches Quaternion rotation to machine precision."""
    from lsdo_geo.core.geometry.geometry_functions import (
        rotate_using_rodrigues,
        rotate_using_quaternion,
        compute_rodrigues_rotation_matrix,
    )

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [-1.0, 0.5, 2.0]])
    origin = np.array([0.1, -0.2, 0.3])
    axis = np.array([1.0, 2.0, 3.0])
    angle = 0.7

    rot_rodrigues = rotate(pts, origin, axis, angle, method='rodrigues')
    rot_quaternion = rotate(pts, origin, axis, angle, method='quaternion')
    rot_using_rodrigues = rotate_using_rodrigues(pts, origin, axis, angle)
    rot_using_quaternion = rotate_using_quaternion(pts, origin, axis, angle)

    np.testing.assert_allclose(rot_rodrigues.value, rot_quaternion.value, atol=1e-12)
    np.testing.assert_allclose(rot_using_rodrigues.value, rot_quaternion.value, atol=1e-12)
    np.testing.assert_allclose(rot_using_quaternion.value, rot_quaternion.value, atol=1e-12)

    # Test orthogonality of compute_rodrigues_rotation_matrix
    R = compute_rodrigues_rotation_matrix(axis, angle)
    np.testing.assert_allclose(R.value @ R.value.T, np.eye(3), atol=1e-12)


def test_rodrigues_derivatives_equivalence():
    """Verify that analytic derivatives of Rodrigues rotation match Quaternion rotation."""
    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [-1.0, 0.5, 2.0]])
    origin = np.array([0.1, -0.2, 0.3])
    axis = np.array([1.0, 2.0, 3.0])
    angle = 0.7

    # Quaternion derivatives
    rec1 = csdl.Recorder(inline=True)
    rec1.start()
    pts_v1 = csdl.Variable(value=pts)
    axis_v1 = csdl.Variable(value=axis)
    angle_v1 = csdl.Variable(value=np.array([angle]))
    rot1 = rotate(pts_v1, origin, axis_v1, angle_v1, method='quaternion')
    loss1 = csdl.sum(rot1)
    d_angle1 = csdl.derivative(loss1, angle_v1).value
    d_pts1 = csdl.derivative(loss1, pts_v1).value
    d_axis1 = csdl.derivative(loss1, axis_v1).value
    rec1.stop()

    # Rodrigues derivatives
    rec2 = csdl.Recorder(inline=True)
    rec2.start()
    pts_v2 = csdl.Variable(value=pts)
    axis_v2 = csdl.Variable(value=axis)
    angle_v2 = csdl.Variable(value=np.array([angle]))
    rot2 = rotate(pts_v2, origin, axis_v2, angle_v2, method='rodrigues')
    loss2 = csdl.sum(rot2)
    d_angle2 = csdl.derivative(loss2, angle_v2).value
    d_pts2 = csdl.derivative(loss2, pts_v2).value
    d_axis2 = csdl.derivative(loss2, axis_v2).value
    rec2.stop()

    np.testing.assert_allclose(d_angle1, d_angle2, atol=1e-12)
    np.testing.assert_allclose(d_pts1, d_pts2, atol=1e-12)
    np.testing.assert_allclose(d_axis1, d_axis2, atol=1e-12)


