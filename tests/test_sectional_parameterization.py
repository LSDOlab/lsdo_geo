import pytest
import warnings
import numpy as np
import csdl_alpha as csdl
import lsdo_geo as lg


def test_sectional_parameterization_evaluate():
    """Verify SectionalParameterization evaluation with stretch, translation, and rotation."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    # Create a 3D FFD-like block with shape (4, 3, 2, 3)
    grid = np.zeros((4, 3, 2, 3))
    for i in range(4):
        for j in range(3):
            for k in range(2):
                grid[i, j, k] = [i * 2.0, j * 3.0, k * 1.5]

    pts = csdl.Variable(value=grid)
    sp = lg.SectionalParameterization(
        parameterized_points=pts,
        principal_parametric_dimension=1,
    )

    # Plot before evaluate should work
    elements_before = sp.plot(show=False)
    assert len(elements_before) > 0

    params = lg.SectionalParameters()
    params.add_stretch(axis=0, stretch=csdl.Variable(value=np.array([1.0, 1.2, 1.5])))
    params.add_translation(axis=2, translation=csdl.Variable(value=np.array([0.1, 0.2, 0.3])))
    params.add_rotation(axis=1, rotation=csdl.Variable(value=np.array([0.05, 0.1, 0.15])))

    updated = sp.evaluate(params)
    assert updated.shape == (4, 3, 2, 3)
    # The output values should differ from the original grid
    assert not np.allclose(updated.value, grid)

    # Plot after evaluate should also work
    elements_after = sp.plot(show=False)
    assert len(elements_after) > 0


def test_deprecated_aliases_and_warnings():
    """Verify VolumeSectionalParameterization and VolumeSectionalParameterizationInputs emit DeprecationWarnings."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    grid = np.zeros((4, 3, 2, 3))
    for i in range(4):
        for j in range(3):
            for k in range(2):
                grid[i, j, k] = [i * 1.0, j * 1.0, k * 1.0]
    pts = csdl.Variable(value=grid)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        vsp = lg.VolumeSectionalParameterization(
            parameterized_points=pts,
            principal_parametric_dimension=1,
        )
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "VolumeSectionalParameterization is deprecated" in str(w[0].message)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        vsp_inputs = lg.VolumeSectionalParameterizationInputs()
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "VolumeSectionalParameterizationInputs is deprecated" in str(w[0].message)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        vsp_inputs.add_sectional_stretch(0, csdl.Variable(value=np.ones(3)))
        vsp_inputs.add_sectional_translation(2, csdl.Variable(value=np.ones(3)))
        vsp_inputs.add_sectional_rotation(1, csdl.Variable(value=np.ones(3)))
        assert len(w) == 3
        assert issubclass(w[0].category, DeprecationWarning)
        assert "add_sectional_stretch is deprecated" in str(w[0].message)
        assert issubclass(w[1].category, DeprecationWarning)
        assert "add_sectional_translation is deprecated" in str(w[1].message)
        assert issubclass(w[2].category, DeprecationWarning)
        assert "add_sectional_rotation is deprecated" in str(w[2].message)

    # Verify that deprecated classes still successfully evaluate
    updated = vsp.evaluate(vsp_inputs)
    assert updated.shape == (4, 3, 2, 3)


def test_section_axis_computation_all_principal_dimensions():
    """Verify _compute_section_axis works across all principal parametric dimensions."""
    for principal_dim in [0, 1, 2]:
        recorder = csdl.Recorder(inline=True)
        recorder.start()

        shape = [4, 4, 4, 3]
        grid = np.zeros(shape)
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    grid[i, j, k] = [i * 1.0, j * 2.0, k * 3.0]

        pts = csdl.Variable(value=grid)
        sp = lg.SectionalParameterization(
            parameterized_points=pts,
            principal_parametric_dimension=principal_dim,
        )

        params = lg.SectionalParameters()
        non_principal = [i for i in range(3) if i != principal_dim]
        # Stretch along first non-principal dimension
        params.add_stretch(axis=non_principal[0], stretch=csdl.Variable(value=np.ones(4) * 1.1))
        # Translation along principal dimension (normal to section)
        params.add_translation(axis=principal_dim, translation=csdl.Variable(value=np.ones(4) * 0.5))
        # Rotation about principal dimension (twist in section plane)
        params.add_rotation(axis=principal_dim, rotation=csdl.Variable(value=np.ones(4) * 0.1))

        updated = sp.evaluate(params)
        assert updated.shape == (4, 4, 4, 3)


def test_sectional_parameterization_vector_axes():
    """Verify SectionalParameterization with csdl.Variable and np.ndarray vector axes."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    grid = np.zeros((4, 3, 2, 3))
    for i in range(4):
        for j in range(3):
            for k in range(2):
                grid[i, j, k] = [i * 1.0, j * 1.0, k * 1.0]

    pts = csdl.Variable(value=grid)
    sp = lg.SectionalParameterization(
        parameterized_points=pts,
        principal_parametric_dimension=1,
    )

    params = lg.SectionalParameters()
    # csdl.Variable vector translation axis
    params.add_translation(
        axis=csdl.Variable(value=np.array([0.0, 1.0, 0.0])),
        translation=csdl.Variable(value=np.array([0.1, 0.2, 0.3])),
    )
    # np.ndarray vector translation axis
    params.add_translation(
        axis=np.array([1.0, 0.0, 0.0]),
        translation=csdl.Variable(value=np.array([0.2, 0.1, 0.0])),
    )
    # csdl.Variable vector rotation axis
    params.add_rotation(
        axis=csdl.Variable(value=np.array([0.0, 0.0, 1.0])),
        rotation=csdl.Variable(value=np.array([0.05, 0.1, 0.15])),
    )
    # np.ndarray vector rotation axis
    params.add_rotation(
        axis=np.array([0.0, 1.0, 0.0]),
        rotation=csdl.Variable(value=np.array([0.02, 0.04, 0.06])),
    )

    updated = sp.evaluate(params)
    assert updated.shape == (4, 3, 2, 3)
    assert not np.allclose(updated.value, grid)


def test_sectional_parameterization_custom_coordinates():
    """Verify SectionalParameterization with 1D and 2D custom parametric coordinates."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    grid = np.zeros((4, 3, 2, 3))
    for i in range(4):
        for j in range(3):
            for k in range(2):
                grid[i, j, k] = [i * 2.0, j * 3.0, k * 1.0]

    pts = csdl.Variable(value=grid)
    sp = lg.SectionalParameterization(
        parameterized_points=pts,
        principal_parametric_dimension=1,
    )

    # 1D coordinate (applied uniformly to all sections)
    coord_1d = np.array([0.25, 0.75])
    params1 = lg.SectionalParameters()
    params1.add_stretch(axis=0, stretch=csdl.Variable(value=np.ones(3) * 1.1), parametric_coordinate=coord_1d)
    params1.add_rotation(axis=1, rotation=csdl.Variable(value=np.ones(3) * 0.1), parametric_coordinate=coord_1d)
    updated1 = sp.evaluate(params1)
    assert updated1.shape == (4, 3, 2, 3)

    # 2D coordinate array (per-section coordinates, shape (num_sections, 2)) for stretch & rotation
    coord_2d = np.array([
        [0.2, 0.8],
        [0.5, 0.5],
        [0.8, 0.2],
    ])
    params2 = lg.SectionalParameters()
    params2.add_stretch(axis=0, stretch=csdl.Variable(value=np.ones(3) * 1.2), parametric_coordinate=coord_2d)
    params2.add_rotation(axis=1, rotation=csdl.Variable(value=np.ones(3) * 0.05), parametric_coordinate=coord_2d)
    updated2 = sp.evaluate(params2)
    assert updated2.shape == (4, 3, 2, 3)


def test_sectional_parameterization_validation_errors():
    """Verify SectionalParameterization error handling on invalid inputs."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    # 1D parameterized points error
    with pytest.raises(Exception, match="structured shape"):
        lg.SectionalParameterization(
            parameterized_points=csdl.Variable(value=np.ones((10,))),
            principal_parametric_dimension=0,
        )

    # 2D shape (1D set of physical points) error
    with pytest.raises(Exception, match="Can't make a sectional parameterization for a 1D set of points"):
        lg.SectionalParameterization(
            parameterized_points=csdl.Variable(value=np.ones((10, 3))),
            principal_parametric_dimension=0,
        )

    # Out of bounds principal_parametric_dimension
    with pytest.raises(Exception, match="Principal parametric dimension is greater"):
        lg.SectionalParameterization(
            parameterized_points=csdl.Variable(value=np.ones((4, 3, 2, 3))),
            principal_parametric_dimension=3,
        )

    # Parameterized points shape size mismatch
    with pytest.raises(Exception, match="not the same size"):
        lg.SectionalParameterization(
            parameterized_points=csdl.Variable(value=np.ones((4, 3, 2, 3))),
            parameterized_points_shape=(5, 5, 5, 3),
        )

    # Invalid axis type during evaluation
    sp = lg.SectionalParameterization(
        parameterized_points=csdl.Variable(value=np.ones((4, 3, 2, 3))),
        principal_parametric_dimension=1,
    )
    bad_params = lg.SectionalParameters()
    bad_params.add_translation(axis="invalid_string", translation=csdl.Variable(value=np.ones(3)))
    with pytest.raises(Exception, match="Invalid axis type"):
        sp.evaluate(bad_params)

    # Mismatched 2D parametric coordinate rows vs num_sections for stretch
    bad_coord_params = lg.SectionalParameters()
    bad_coord_params.add_stretch(
        axis=0,
        stretch=csdl.Variable(value=np.ones(3)),
        parametric_coordinate=np.ones((10, 2)),  # 10 != num_sections (3)
    )
    with pytest.raises(Exception, match="Invalid parametric_coordinate shape"):
        sp.evaluate(bad_coord_params)

    # Mismatched 2D parametric coordinate rows vs num_sections for rotation
    bad_rot_params = lg.SectionalParameters()
    bad_rot_params.add_rotation(
        axis=1,
        rotation=csdl.Variable(value=np.ones(3)),
        parametric_coordinate=np.ones((10, 2)),  # 10 != num_sections (3)
    )
    with pytest.raises(Exception, match="Invalid parametric_coordinate shape"):
        sp.evaluate(bad_rot_params)


def test_sectional_parameterization_2d_surface():
    """Verify SectionalParameterization on a 2D surface in 3D space (shape (nx, ny, 3))."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    # 2D surface mesh in 3D physical space
    surface_grid = np.zeros((5, 4, 3))
    for i in range(5):
        for j in range(4):
            surface_grid[i, j] = [i * 1.0, j * 2.0, 0.0]

    pts = csdl.Variable(value=surface_grid)
    sp = lg.SectionalParameterization(
        parameterized_points=pts,
        principal_parametric_dimension=0,
    )
    assert sp.num_sections == 5

    # Plot curve sections (2D surface -> 1D section curves)
    elements = sp.plot(show=False)
    assert len(elements) > 0

    params = lg.SectionalParameters()
    params.add_translation(axis=1, translation=csdl.Variable(value=np.linspace(0.0, 0.5, 5)))
    updated = sp.evaluate(params)
    assert updated.shape == (5, 4, 3)
