import numpy as np
import pytest
import csdl_alpha as csdl
import lsdo_function_spaces as lfs
import lsdo_geo as lg


def test_marker_and_connection():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    c1 = np.array([[[0., 0., 0.], [0., 1., 0.]], [[1., 0., 0.], [1., 1., 0.]]])
    c2 = np.array([[[0., 0., 0.], [0., 1., 0.]], [[1., 0., 0.], [1., 1., 0.]]])

    g1 = lg.Geometry(functions={0: lfs.Function(space=space, coefficients=c1)}, space=space, name='g1')
    g2 = lg.Geometry(functions={0: lfs.Function(space=space, coefficients=c2)}, space=space, name='g2')

    m1 = g1.add_marker((0, np.array([0.5, 0.5])), name='m1')
    m2 = g2.add_marker(locations=(0, np.array([0.5, 0.5])), name='m2')

    assert m1.name == 'm1'
    assert np.allclose(m1.evaluate().value, [0.5, 0.5, 0.0])
    assert np.allclose(m1.compute_surface_normal().value, [0.0, 0.0, 1.0])
    assert np.allclose(m1.compute_surface_normal(flip=True).value, [0.0, 0.0, -1.0])

    connection = lg.Connection(component_1=g1, marker_1=m1, component_2=g2, marker_2=m2, name='c1')
    assembly = lg.Assembly(name='test_assembly')
    assembly.add_component(g1, 'g1')
    assembly.add_component(g2, 'g2')
    assembly.add_connection(connection)

    assert assembly.get_component('g1') is g1
    assert assembly.get_connection('c1') is connection

    trans_err = connection.evaluate_translational_offset()
    align_err = connection.evaluate_alignment_constraint()
    assert np.allclose(trans_err.value, [0.0, 0.0, 0.0])
    assert np.allclose(align_err.value, [0.0, 0.0])


def test_marker_and_connection_optional_names():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    c1 = np.array([[[0., 0., 0.], [0., 1., 0.]], [[1., 0., 0.], [1., 1., 0.]]])
    c2 = np.array([[[0., 0., 0.], [0., 1., 0.]], [[1., 0., 0.], [1., 1., 0.]]])

    g1 = lg.Geometry(functions={0: lfs.Function(space=space, coefficients=c1)}, space=space)
    g2 = lg.Geometry(functions={0: lfs.Function(space=space, coefficients=c2)}, space=space)

    # Optional name in add_marker
    m1 = g1.add_marker((0, np.array([0.5, 0.5])))
    m2 = g2.add_marker(locations=(0, np.array([0.5, 0.5])))

    assert m1.name is not None
    assert m2.name is not None
    assert m1.name in g1.markers
    assert g1.markers[m1.name] is m1

    # Optional name in Connection
    connection = lg.Connection(component_1=g1, marker_1=m1, component_2=g2, marker_2=m2)
    assert connection.name is not None

    assembly = lg.Assembly()
    assembly.add_component(g1)
    assembly.add_component(g2)
    assembly.add_connection(connection)

    assert connection.name in assembly.connections
    assert assembly.get_connection(connection.name) is connection


def test_marker_physical_projection():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    c1 = np.array([[[0., 0., 0.], [0., 1., 0.]], [[1., 0., 0.], [1., 1., 0.]]])
    g1 = lg.Geometry(functions={0: lfs.Function(space=space, coefficients=c1)}, space=space)

    # Physical 3D Cartesian point
    m = g1.add_marker(np.array([0.5, 0.5, 0.0]), name='phys_marker')
    assert m.name == 'phys_marker'
    assert np.allclose(m.evaluate().value, [0.5, 0.5, 0.0], atol=1e-5)
    assert np.allclose(m.initial_point, [0.5, 0.5, 0.0])


def test_marker_compute_null_space():
    # Test non_csdl
    space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    c1 = np.array([[[0., 0., 0.], [0., 1., 0.]], [[1., 0., 0.], [1., 1., 0.]]])
    g1 = lg.Geometry(functions={0: lfs.Function(space=space, coefficients=c1)}, space=space)
    m = g1.add_marker((0, np.array([0.5, 0.5])))

    ns_non_csdl = m.compute_null_space(non_csdl=True)
    assert ns_non_csdl.shape == (3, 2)
    # Check orthogonality
    assert np.isclose(np.dot(ns_non_csdl[:, 0], ns_non_csdl[:, 1]), 0.0)
    assert np.isclose(np.linalg.norm(ns_non_csdl[:, 0]), 1.0)
    assert np.isclose(np.linalg.norm(ns_non_csdl[:, 1]), 1.0)

    # Test CSDL inline=False (graph mode)
    rec = csdl.Recorder(inline=False)
    rec.start()
    g2 = lg.Geometry(functions={0: lfs.Function(space=space, coefficients=c1)}, space=space)
    m2 = g2.add_marker((0, np.array([0.5, 0.5])))
    ns_graph = m2.compute_null_space(non_csdl=False)
    assert ns_graph.shape == (3, 2)
    rec.stop()






def test_assembly_as_function_network():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    c1 = np.array([[[0., 0., 0.], [0., 1., 0.]], [[1., 0., 0.], [1., 1., 0.]]])
    c2 = np.array([[[1., 0., 0.], [1., 1., 0.]], [[2., 0., 0.], [2., 1., 0.]]])

    g1 = lg.Geometry(functions={0: lfs.Function(space=space, coefficients=c1)}, space=space, name='g1')
    g2 = lg.Geometry(functions={0: lfs.Function(space=space, coefficients=c2)}, space=space, name='g2')

    m1 = g1.add_marker((0, np.array([1.0, 0.5])), name='m1')
    m2 = g2.add_marker((0, np.array([0.0, 0.5])), name='m2')

    conn = lg.Connection(component_1=g1, marker_1=m1, component_2=g2, marker_2=m2, connection_type='spherical', name='c1')
    assembly = lg.Assembly(name='test_geom_network')
    assembly.add_component(g1, 'g1')
    assembly.add_component(g2, 'g2')
    assembly.add_connection(conn)

    # FunctionNetwork inheritance capabilities
    assert isinstance(assembly, lfs.FunctionNetwork)
    assert len(assembly.functions) == 2
    assert len(assembly.interfaces) == 1
    assert len(assembly.conditions) == 1

    # Connected components
    comps = assembly.compute_connected_components()
    assert len(comps) == 1
    assert len(comps[0]) == 2

    # Groupings & classification
    assembly.component_groupings = {
        'comp1': [(g1, 0)],
        'comp2': [(g2, 0)],
    }
    internal, between = assembly.classify_interfaces()
    assert len(internal) == 0
    assert len(between) == 1

    # Residual evaluation
    res = assembly.evaluate_residuals()
    # At (1.0, 0.5) on g1, pos is [1.0, 0.5, 0.0]
    # At (0.0, 0.5) on g2, pos is [1.0, 0.5, 0.0]
    # Difference is 0
    assert np.allclose(res.value, [0.0, 0.0, 0.0])


def test_assembly_multi_patch_function_set_freedom():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))

    # Wing: 2 patches along x
    c_w0 = np.array([[[0., 0., 0.], [0., 1., 0.]], [[1., 0., 0.], [1., 1., 0.]]])
    c_w1 = np.array([[[1., 0., 0.], [1., 1., 0.]], [[2., 0., 0.], [2., 1., 0.]]])
    wing = lg.Geometry(
        functions={0: lfs.Function(space=space, coefficients=c_w0), 1: lfs.Function(space=space, coefficients=c_w1)},
        space=space,
        name='wing'
    )

    # Fuselage: 2 patches along x at y in [1, 2]
    c_f0 = np.array([[[0., 1., 0.], [0., 2., 0.]], [[1., 1., 0.], [1., 2., 0.]]])
    c_f1 = np.array([[[1., 1., 0.], [1., 2., 0.]], [[2., 1., 0.], [2., 2., 0.]]])
    fuselage = lg.Geometry(
        functions={0: lfs.Function(space=space, coefficients=c_f0), 1: lfs.Function(space=space, coefficients=c_f1)},
        space=space,
        name='fuselage'
    )

    # 1. Internal seam on wing between patch 0 (at x=1.0) and patch 1 (at x=1.0)
    m_w0 = wing.add_marker((0, np.array([1.0, 0.5])), name='wing_p0_seam')
    m_w1 = wing.add_marker((1, np.array([0.0, 0.5])), name='wing_p1_seam')
    conn_seam = lg.Connection(
        component_1=wing,
        marker_1=m_w0,
        component_2=wing,
        marker_2=m_w1,
        connection_type='spherical',
        name='wing_seam',
    )

    # 2. Inter-component connection between wing patch 1 and fuselage patch 1
    # Created via physical Cartesian point projection
    m_w_attach = wing.add_marker(np.array([1.5, 0.5, 0.0]), name='wing_attach')
    m_f_attach = fuselage.add_marker(np.array([1.5, 1.0, 0.0]), name='fuse_attach')

    # Verify physical projection correctly mapped to patch 1 for both entities
    assert m_w_attach.parametric_coordinates[0][0] == 1
    assert m_f_attach.parametric_coordinates[0][0] == 1

    conn_attach = lg.Connection(
        component_1=wing,
        marker_1=m_w_attach,
        component_2=fuselage,
        marker_2=m_f_attach,
        connection_type='spherical',
        offset=np.array([0.0, -0.5, 0.0]),
        name='wing_fuse_connection',
    )

    assembly = lg.Assembly(name='airplane_assembly')
    assembly.add_component(wing, 'wing')
    assembly.add_component(fuselage, 'fuselage')
    assembly.add_connection(conn_seam)
    assembly.add_connection(conn_attach)

    # Component groupings
    assembly.component_groupings = {
        'wing': [(wing, 0), (wing, 1)],
        'fuselage': [(fuselage, 0), (fuselage, 1)],
    }

    # Classify interfaces
    internal, between = assembly.classify_interfaces()
    assert len(internal) == 1
    assert internal[0] is conn_seam.interface
    assert len(between) == 1
    assert between[0] is conn_attach.interface

    # Connected components
    comps = assembly.compute_connected_components()
    assert len(comps) == 2
    comp_sizes = sorted([len(c) for c in comps])
    assert comp_sizes == [1, 3]

    # Cycle rank
    assert assembly.compute_cycle_rank() == (0, 0)

    # Freedom analysis
    report = assembly.analyze_freedom()
    assert report.num_variables == 48
    assert report.num_constraints == 6
    assert report.rank == 6
    assert report.redundancy == 0
    assert report.nullity == 42
    assert report.is_licq_satisfied is True

    # Residual evaluation
    res = assembly.evaluate_residuals()
    assert np.allclose(res.value, np.zeros(6), atol=1e-6)


def test_assembly_rebind_and_copy():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    c1 = np.array([[[0., 0., 0.], [0., 1., 0.]], [[1., 0., 0.], [1., 1., 0.]]])
    c2 = np.array([[[1., 0., 0.], [1., 1., 0.]], [[2., 0., 0.], [2., 1., 0.]]])

    g1 = lg.Geometry(functions={0: lfs.Function(space=space, coefficients=c1)}, space=space, name='g1')
    g2 = lg.Geometry(functions={0: lfs.Function(space=space, coefficients=c2)}, space=space, name='g2')

    m1 = g1.add_marker((0, np.array([1.0, 0.5])), name='m1')
    m2 = g2.add_marker((0, np.array([0.0, 0.5])), name='m2')

    conn = lg.Connection(
        component_1=g1,
        marker_1=m1,
        component_2=g2,
        marker_2=m2,
        connection_type='revolute',
        name='conn_1',
    )

    assembly = lg.Assembly(name='orig_assembly')
    assembly.add_component(g1, 'g1')
    assembly.add_component(g2, 'g2')
    assembly.add_connection(conn)

    assembly.component_groupings = {
        'comp1': [(g1, 0)],
        'comp2': [(g2, 0)],
    }

    # 1. Deep copy of Assembly
    assembly_copy = assembly.copy(name='copied_assembly')
    assert assembly_copy.name == 'copied_assembly'

    # Verify components are independent instances
    g1_copy = assembly_copy.get_component('g1')
    g2_copy = assembly_copy.get_component('g2')
    assert g1_copy is not g1
    assert g2_copy is not g2

    # Verify markers are cloned and bound to the copied geometry
    m1_copy = g1_copy.markers['m1']
    m2_copy = g2_copy.markers['m2']
    assert m1_copy is not m1
    assert m2_copy is not m2
    assert m1_copy.entity is g1_copy
    assert m2_copy.entity is g2_copy

    # Verify connection is rebound to copied components and markers
    conn_copy = assembly_copy.get_connection('conn_1')
    assert conn_copy is not conn
    assert conn_copy.component_1 is g1_copy
    assert conn_copy.component_2 is g2_copy
    assert conn_copy.marker_1 is m1_copy
    assert conn_copy.marker_2 is m2_copy
    assert conn_copy.interface.markers[0] is m1_copy
    assert conn_copy.interface.markers[1] is m2_copy

    # Groupings rebound
    assert assembly_copy.component_groupings['comp1'][0][0] is g1_copy
    assert assembly_copy.component_groupings['comp2'][0][0] is g2_copy

    # Residual evaluations match
    res_orig = conn.evaluate_residual()
    res_copy = conn_copy.evaluate_residual()
    np.testing.assert_allclose(res_orig.value, res_copy.value, atol=1e-7)

    # 2. Rebind with explicit target map
    g1_deformed = lg.Geometry(functions={0: lfs.Function(space=space, coefficients=c1 + 1.0)}, space=space, name='g1_def')
    g2_deformed = lg.Geometry(functions={0: lfs.Function(space=space, coefficients=c2 + 1.0)}, space=space, name='g2_def')
    m1_def = g1_deformed.add_marker((0, np.array([1.0, 0.5])), name='m1')
    m2_def = g2_deformed.add_marker((0, np.array([0.0, 0.5])), name='m2')

    rebound_assembly = assembly.rebind({g1: g1_deformed, g2: g2_deformed}, name='rebound_assembly')
    assert rebound_assembly.get_component('g1') is g1_deformed
    assert rebound_assembly.get_component('g2') is g2_deformed
    rebound_conn = rebound_assembly.get_connection('conn_1')
    assert rebound_conn.component_1 is g1_deformed
    assert rebound_conn.component_2 is g2_deformed
    assert rebound_conn.marker_1 is m1_def
    assert rebound_conn.marker_2 is m2_def


def test_assembly_discover_internal_seams():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    # Wing with 2 patches meeting at x=1.0
    c0 = np.array([[[0., 0., 0.], [0., 1., 0.]], [[1., 0., 0.], [1., 1., 0.]]])
    c1 = np.array([[[1., 0., 0.], [1., 1., 0.]], [[2., 0., 0.], [2., 1., 0.]]])
    wing = lg.Geometry(
        functions={0: lfs.Function(space=space, coefficients=c0), 1: lfs.Function(space=space, coefficients=c1)},
        space=space,
        name='wing',
    )

    assembly = lg.Assembly(name='wing_assembly')
    assembly.add_component(wing, 'wing')

    # Automatically discover seams across all multi-patch components
    discovered = assembly.discover_internal_seams(tolerance=1e-5, search_n=6, create_conditions=True)
    assert len(discovered) == 1
    assert len(assembly.interfaces) == 1
    assert len(assembly.conditions) == 1

    # Verify connected components
    comps = assembly.compute_connected_components()
    assert len(comps) == 1
    assert len(comps[0]) == 2

    # Verify residual evaluation
    res = assembly.evaluate_residuals()
    assert res.shape == (6, 3)
    np.testing.assert_allclose(res.value, np.zeros((6, 3)), atol=1e-8)


def test_geometry_higher_dimensional_marker():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
    c = np.array([[[0., 0., 0.], [0., 1., 0.]], [[1., 0., 0.], [1., 1., 0.]]])
    geom = lg.Geometry(functions={0: lfs.Function(space=space, coefficients=c)}, space=space, name='geom')

    # 2D surface function map: [0, 1]^2 -> [0.1, 0.9]^2
    c_map = np.array([
        [[0.1, 0.1], [0.1, 0.9]],
        [[0.9, 0.1], [0.9, 0.9]],
    ])
    alpha_fn = lfs.Function(space=space, coefficients=c_map, name='subpatch_map')

    # Add 2D marker to geometry directly via add_marker
    m = geom.add_marker(alpha_fn, name='subpatch_marker')
    assert m.domain.dimension == 2
    assert m.domain.is_finite is False

    # Evaluate marker
    val = m.evaluate()
    assert val.shape == (100, 3)



