import pytest
import numpy as np
import pyvista as pv
import csdl_alpha as csdl
import lsdo_function_spaces as lfs

import lsdo_geo as lg
from lsdo_geo import (
    Geometry,
    Mesh,
    FFDBlock,
    SectionalParameterization,
    SectionalParameters,
    construct_ffd_block_around_entities,
    import_geometry,
)


@pytest.fixture(autouse=True)
def setup_headless_pyvista():
    """Ensure all PyVista plotting is off-screen during tests."""
    pv.OFF_SCREEN = True


def test_rectangular_wing_parameterization_pipeline():
    """Test the end-to-end geometry and FFD sectional parameterization pipeline
    from showcase_examples/rectangular_wing/ex_rectangular_wing.py.
    """
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    stp_path = "examples/example_geometries/rectangular_wing.stp"
    geometry = import_geometry(stp_path, parallelize=False)
    assert isinstance(geometry, Geometry)

    # 1. Project key geometric landmarks
    quarter_chord_left = geometry.project(np.array([0.25, -4.0, 0.0]))
    quarter_chord_right = geometry.project(np.array([0.25, 4.0, 0.0]))
    quarter_chord_center = geometry.project(np.array([0.25, 0.0, 0.0]))

    pt_center = geometry.evaluate(quarter_chord_center)
    np.testing.assert_allclose(pt_center.value, np.array([0.25, 0.0, 0.0]), atol=1e-3)

    # 2. Construct FFD block around geometry
    num_ffd_sections = 3
    ffd_block = construct_ffd_block_around_entities(
        entities=geometry,
        num_coefficients=(4, num_ffd_sections, 2),
        degree=(2, 1, 1),
    )
    assert isinstance(ffd_block, FFDBlock)

    # 3. Setup Sectional Parameterization
    ffd_sectional_sp = SectionalParameterization(
        name="test_ffd_sp",
        parameterized_points=ffd_block.coefficients,
        principal_parametric_dimension=1,
    )

    # 4. Define B-splines for chord stretch and wingspan stretch
    space_linear_3 = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(3,))
    chord_stretch_bs = lfs.Function(
        space=space_linear_3,
        coefficients=csdl.Variable(shape=(3,), value=np.array([0.1, 0.0, 0.1])),
        name="test_chord_stretch",
    )

    parametric_inputs = np.linspace(0.0, 1.0, num_ffd_sections).reshape((-1, 1))
    chord_stretch_params = chord_stretch_bs.evaluate(parametric_inputs)

    # 5. Apply sectional parameters
    params = SectionalParameters()
    params.add_stretch(axis=0, stretch=chord_stretch_params)
    params.add_translation(axis=1, translation=csdl.Variable(shape=(3,), value=np.zeros(3)))
    params.add_rotation(axis=1, rotation=csdl.Variable(shape=(3,), value=np.zeros(3)))

    ffd_coeffs = ffd_sectional_sp.evaluate(params, plot=False)
    assert ffd_coeffs.shape == ffd_block.coefficients.shape

    # 6. Evaluate FFD perturbation
    new_geo_coeffs = ffd_block.evaluate_ffd(coefficients=ffd_coeffs, plot=False)
    geometry.set_coefficients(new_geo_coeffs)

    # Evaluate deformed point
    deformed_center = geometry.evaluate(quarter_chord_center)
    assert deformed_center.shape == (3,)


def test_lift_plus_cruise_component_and_ffd_pipeline():
    """Test component declaration and FFD parameterization pipeline from
    showcase_examples/lift_plus_cruise/ex_lift_plus_cruise.py.
    """
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    stp_path = "examples/example_geometries/lift_plus_cruise_final.stp"
    geometry = import_geometry(stp_path, parallelize=False)

    # 1. Component declaration
    wing = geometry.declare_component(function_search_names=['Wing'], name='wing')
    h_tail = geometry.declare_component(function_search_names=['Tail_1'], name='h_tail')
    v_tail = geometry.declare_component(function_search_names=['Tail_2'], name='v_tail')
    fuselage = geometry.declare_component(function_search_names=['Fuselage_***.main'], name='fuselage')

    assert wing.name == 'wing'
    assert h_tail.name == 'h_tail'
    assert v_tail.name == 'v_tail'
    assert fuselage.name == 'fuselage'

    # 2. Construct FFD block around wing component
    wing_ffd = construct_ffd_block_around_entities(
        entities=wing,
        num_coefficients=(3, 4, 2),
        degree=(1, 1, 1),
    )
    assert isinstance(wing_ffd, FFDBlock)

    # 3. Setup Sectional Parameterization on wing FFD
    wing_sp = SectionalParameterization(
        name="wing_ffd_sp",
        parameterized_points=wing_ffd.coefficients,
        principal_parametric_dimension=1,
    )

    # 4. Apply twist rotation along spanwise direction
    twist_angles = csdl.Variable(shape=(4, 1), value=np.linspace(0.0, 5.0, 4).reshape(-1, 1) * np.pi / 180.0)
    sp_params = SectionalParameters()
    sp_params.add_rotation(axis=1, rotation=twist_angles)

    deformed_ffd_coeffs = wing_sp.evaluate(sp_params, plot=False)
    assert deformed_ffd_coeffs.shape == wing_ffd.coefficients.shape

    # 5. Update wing geometry coefficients
    wing_updated_coeffs = wing_ffd.evaluate_ffd(coefficients=deformed_ffd_coeffs, plot=False)
    wing.set_coefficients(wing_updated_coeffs)
    assert wing_updated_coeffs is not None


@pytest.mark.slow
def test_uav_parameterization_spaces_pipeline():
    """Test B-spline space creation and function refitting from
    additional_examples/hand_launched_uavs/hand_launched_uavs_parameterization.py.
    """
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    stp_path = "examples/example_geometries/rectangular_wing.stp"
    wing = import_geometry(stp_path, parallelize=False)

    # Test BSplineSpace construction with degree and shape
    num_cp_u = 3
    num_cp_v = wing.functions[3].coefficients.shape[1]
    wing_space = lfs.BSplineSpace(
        num_parametric_dimensions=2,
        degree=(2, 3),
        coefficients_shape=(num_cp_u, num_cp_v),
        knots=(np.array([0., 0., 0., 1., 1., 1.]), wing.functions[3].space.knots[1]),
    )
    assert wing_space.degree == (2, 3)

    # Fuselage B-spline space
    fuselage_space = lfs.BSplineSpace(
        num_parametric_dimensions=2,
        degree=(1, 4),
        coefficients_shape=(2, 13),
    )
    assert fuselage_space.num_parametric_dimensions == 2

    # Nose cone B-spline space
    nose_space = lfs.BSplineSpace(
        num_parametric_dimensions=2,
        degree=(2, 4),
        coefficients_shape=(5, 13, 3),
    )
    assert nose_space.coefficients_shape == (5, 13, 3)


@pytest.mark.slow
def test_toy_optimization_setup_smoke():
    """Smoke test ensuring ex_rectangular_wing_toy_optimization sets up without errors."""
    pytest.importorskip("modopt")

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    stp_path = "examples/example_geometries/rectangular_wing.stp"
    geometry = import_geometry(stp_path, parallelize=False)

    # FFD Block
    ffd_block = construct_ffd_block_around_entities(
        entities=geometry,
        num_coefficients=(4, 3, 2),
        degree=(2, 1, 1),
    )
    ffd_sp = SectionalParameterization(
        name="ffd_sp",
        parameterized_points=ffd_block.coefficients,
        principal_parametric_dimension=1,
    )

    # Sectional parameters
    space_2 = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))
    wingspan_bs = lfs.Function(
        space=space_2,
        coefficients=csdl.Variable(shape=(2,), value=np.array([0.0, 0.0])),
        name="span_coeffs",
    )
    parametric_inputs = np.linspace(0.0, 1.0, 3).reshape((-1, 1))
    span_params = wingspan_bs.evaluate(parametric_inputs)

    params = SectionalParameters()
    params.add_translation(axis=1, translation=span_params)

    ffd_coeffs = ffd_sp.evaluate(params, plot=False)
    geom_coeffs = ffd_block.evaluate_ffd(coefficients=ffd_coeffs, plot=False)
    geometry.set_coefficients(geom_coeffs)
    assert geom_coeffs is not None
