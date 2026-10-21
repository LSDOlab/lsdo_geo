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


def test_geometry_rotate_using_rodrigues_and_equivalence():
    """Test rotating geometry using Rodrigues formula and verify equivalence with quaternion."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    origin = np.array([0.5, 1.0, -0.5])
    axis = np.array([1.0, 2.0, 3.0])
    angle = 37.5  # degrees

    # 1. Rotate using Rodrigues
    geo_rodrigues = lg.import_geometry(SAMPLE_STP, parallelize=False)
    geo_rodrigues.rotate_using_rodrigues(rotation_origin=origin, axis_vector=axis, angles=angle, units='degrees')

    # 2. Rotate using Quaternion (with axis and angles)
    geo_quaternion = lg.import_geometry(SAMPLE_STP, parallelize=False)
    geo_quaternion.rotate_using_quaternion(rotation_origin=origin, axis_vector=axis, angles=angle, units='degrees')

    # 3. Rotate using general rotate (default rodrigues)
    geo_general = lg.import_geometry(SAMPLE_STP, parallelize=False)
    geo_general.rotate(rotation_origin=origin, axis_vector=axis, angles=angle, units='degrees')

    for key in geo_rodrigues.functions.keys():
        coeff_rod = geo_rodrigues.functions[key].coefficients.value
        coeff_quat = geo_quaternion.functions[key].coefficients.value
        coeff_gen = geo_general.functions[key].coefficients.value

        np.testing.assert_allclose(coeff_rod, coeff_quat, atol=1e-12)
        np.testing.assert_allclose(coeff_rod, coeff_gen, atol=1e-12)

    # Also test an axis-aligned rotation (e.g. Y axis) where rotate() uses coordinate matrix
    # but rotate_using_rodrigues strictly enforces Rodrigues
    axis_y = np.array([0.0, 1.0, 0.0])
    geo_rod_y = lg.import_geometry(SAMPLE_STP, parallelize=False)
    geo_rod_y.rotate_using_rodrigues(rotation_origin=origin, axis_vector=axis_y, angles=angle, units='degrees')

    geo_rot_y = lg.import_geometry(SAMPLE_STP, parallelize=False)
    geo_rot_y.rotate(rotation_origin=origin, axis_vector=axis_y, angles=angle, units='degrees')

    for key in geo_rod_y.functions.keys():
        coeff_rod_y = geo_rod_y.functions[key].coefficients.value
        coeff_rot_y = geo_rot_y.functions[key].coefficients.value
        np.testing.assert_allclose(coeff_rod_y, coeff_rot_y, atol=1e-12)


def test_geometry_declare_component():
    """Test declaring a sub-component of a geometry."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    geo = lg.import_geometry(SAMPLE_STP, parallelize=False)
    comp = geo.declare_component(function_indices=[0], name="wing_tip")
    assert comp.name == "wing_tip"
    assert len(comp.functions) == 1
    assert 0 in comp.functions


def test_geometry_create_component_copy():
    """Test creating an independent copy of a geometry sub-component."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    geo = lg.import_geometry(SAMPLE_STP, parallelize=False)
    comp_copy = geo.create_component_copy(function_indices=[0, 1], name="wing_root_copy")
    assert comp_copy.name == "wing_root_copy"
    assert len(comp_copy.functions) == 2


def test_geometry_get_function_space():
    """Test retrieving function space from geometry."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    geo = lg.import_geometry(SAMPLE_STP, parallelize=False)
    space = geo.get_function_space()
    assert space is not None


def test_geometry_export_iges_and_obj(tmp_path):
    """Test exporting geometry to IGES and OBJ formats."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    geo = lg.import_geometry(SAMPLE_STP, parallelize=False)

    iges_file = tmp_path / "wing.igs"
    geo.export_iges(str(iges_file))
    assert iges_file.exists()
    assert iges_file.stat().st_size > 1000

    obj_file = tmp_path / "wing.obj"
    geo.export_obj(str(obj_file))


def test_geometry_translate_single_function():
    """Test translating a single function of a geometry (function_indices=[0])."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    geo = lg.import_geometry(SAMPLE_STP, parallelize=False)
    f0 = geo.functions[0]
    orig_f0 = np.copy(f0.coefficients.value)
    f1 = geo.functions[1]
    orig_f1 = np.copy(f1.coefficients.value)

    translation = np.array([1.0, 2.0, 3.0])
    geo.translate(translation, function_indices=[0])

    # f0 should move, f1 should remain unchanged
    np.testing.assert_allclose(f0.coefficients.value, orig_f0 + translation, atol=1e-7)
    np.testing.assert_allclose(f1.coefficients.value, orig_f1, atol=1e-7)


def test_geometry_project_multi_point_line_search():
    """Verify that multi-point batch projection on CAD geometry works with use_line_search=True."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    geo = lg.import_geometry(SAMPLE_STP, parallelize=False)
    pts = np.zeros((10, 3))
    pts[:, 0] = np.linspace(0.1, 0.9, 10)
    pts[:, 1] = 0.5
    pts[:, 2] = 0.01

    proj_no_ls = geo.project(pts, use_line_search=False, do_pickles=False)
    proj_ls = geo.project(pts, use_line_search=True, do_pickles=False)

    eval_no_ls = geo.evaluate(proj_no_ls, non_csdl=True)
    eval_ls = geo.evaluate(proj_ls, non_csdl=True)

    # All points in the batch must converge identically
    diffs = np.linalg.norm(eval_ls - eval_no_ls, axis=1)
    np.testing.assert_allclose(diffs, 0.0, atol=1e-4)



