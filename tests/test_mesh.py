import pytest
import numpy as np
import csdl_alpha as csdl
import lsdo_geo as lg
import os

SAMPLE_STP = os.path.join(
    os.path.dirname(__file__), "..", "examples", "example_geometries", "rectangular_wing.stp"
)


def test_mesh_initialization_and_shape_geometry():
    """Verify Mesh initialization and shape inference with Geometry (FunctionSet)."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    geo = lg.import_geometry(SAMPLE_STP, parallelize=False)
    parametric_coords = [
        (0, np.array([0.2, 0.3])),
        (0, np.array([0.5, 0.5])),
        (0, np.array([0.8, 0.7])),
    ]

    mesh = lg.Mesh(geometry=geo, parametric_coordinates=parametric_coords)
    assert mesh.name.startswith("mesh_")
    assert mesh.shape == (3, 3)

    evaluated = mesh.evaluate(geo)
    assert isinstance(evaluated, csdl.Variable)
    assert evaluated.shape == (3, 3)
    assert not np.isnan(evaluated.value).any()


def test_mesh_initialization_and_shape_function():
    """Verify Mesh initialization and shape inference with a single Function."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    geo = lg.import_geometry(SAMPLE_STP, parallelize=False)
    f0 = list(geo.functions.values())[0]
    uv = np.array([[0.25, 0.25], [0.75, 0.75]])

    mesh = lg.Mesh(geometry=f0, parametric_coordinates=uv, name="custom_mesh")
    assert mesh.name == "custom_mesh"
    assert mesh.shape == (2, 3)

    evaluated = mesh.evaluate(f0)
    assert isinstance(evaluated, csdl.Variable)
    assert evaluated.shape == (2, 3)
    assert not np.isnan(evaluated.value).any()


def test_geometry_representations():
    """Verify Geometry add_representation and evaluate_representations methods."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    geo = lg.import_geometry(SAMPLE_STP, parallelize=False)
    coords1 = [(0, np.array([0.1, 0.1])), (0, np.array([0.9, 0.9]))]
    coords2 = [(0, np.array([0.5, 0.5]))]

    mesh1 = lg.Mesh(geometry=geo, parametric_coordinates=coords1, name="rep1")
    mesh2 = lg.Mesh(geometry=geo, parametric_coordinates=coords2, name="rep2")

    geo.add_representation(mesh1)
    geo.add_representation(mesh2)

    assert "rep1" in geo.representations
    assert "rep2" in geo.representations

    # Evaluate single representation directly
    res1 = geo.evaluate_representations(mesh1)
    assert isinstance(res1, csdl.Variable)
    assert res1.shape == (2, 3)

    # Evaluate multiple representations as a list
    res_list = geo.evaluate_representations([mesh1, mesh2])
    assert isinstance(res_list, list)
    assert len(res_list) == 2
    assert res_list[0].shape == (2, 3)
    assert res_list[1].shape == (3,)


def test_custom_mesh_subclass():
    """Verify subclassing Mesh with custom post-evaluation logic."""
    class MidpointMesh(lg.Mesh):
        def evaluate(self, geometry):
            pts = super().evaluate(geometry)
            return (pts[0] + pts[1]) / 2.0

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    geo = lg.import_geometry(SAMPLE_STP, parallelize=False)
    coords = [(0, np.array([0.0, 0.0])), (0, np.array([1.0, 1.0]))]
    m = MidpointMesh(geometry=geo, parametric_coordinates=coords)
    midpoint = m.evaluate(geo)

    assert isinstance(midpoint, csdl.Variable)
    assert midpoint.shape == (3,)
    assert not np.isnan(midpoint.value).any()


def test_mesh_with_markers_and_subroutine():
    """Verify Mesh constructed from constituent Markers and a custom subroutine."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    geo = lg.import_geometry(SAMPLE_STP, parallelize=False)
    m1 = lg.Marker.from_parametric(geo, [(0, np.array([0.2, 0.2]))], name="m1")
    m2 = lg.Marker.from_parametric(geo, [(0, np.array([0.8, 0.8]))], name="m2")

    # Mesh formed by averaging two markers via a custom subroutine
    def average_subroutine(evaluated_dict):
        return (evaluated_dict["m1"] + evaluated_dict["m2"]) / 2.0

    mesh = lg.Mesh(
        markers={"m1": m1, "m2": m2},
        subroutine=average_subroutine,
        name="average_mesh",
    )
    assert mesh.name == "average_mesh"
    assert mesh.geometry is geo

    val = mesh.evaluate()
    assert isinstance(val, csdl.Variable)
    assert val.shape == (3,)
    assert not np.isnan(val.value).any()

    # Verify non-CSDL evaluation
    val_np = mesh.evaluate(non_csdl=True)
    assert isinstance(val_np, np.ndarray)
    assert np.allclose(val.value, val_np)


def test_camber_surface_mesh():
    """Verify specialized CamberSurfaceMesh averaging upper and lower surface wireframes."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    geo = lg.import_geometry(SAMPLE_STP, parallelize=False)
    upper_coords = [(0, np.array([0.2, 0.5])), (0, np.array([0.4, 0.5]))]
    lower_coords = [(0, np.array([0.2, 0.1])), (0, np.array([0.4, 0.1]))]

    upper_marker = lg.Marker.from_parametric(geo, upper_coords, name="upper")
    lower_marker = lg.Marker.from_parametric(geo, lower_coords, name="lower")

    camber_mesh = lg.CamberSurfaceMesh(
        upper_surface_marker=upper_marker,
        lower_surface_marker=lower_marker,
        shape=(2, 3),
        name="camber",
    )
    res = camber_mesh.evaluate()
    assert isinstance(res, csdl.Variable)
    assert res.shape == (2, 3)
    assert not np.isnan(res.value).any()

    # Direct validation against expected average
    up_val = upper_marker.evaluate()
    lo_val = lower_marker.evaluate()
    expected = (up_val.value + lo_val.value) / 2.0
    assert np.allclose(res.value, expected)


def test_unstructured_mesh_centroids_normals_and_meshio():
    """Verify UnstructuredMesh cell connectivity, centroids, normals, areas, and meshio export."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    geo = lg.import_geometry(SAMPLE_STP, parallelize=False)
    coords = [
        (0, np.array([0.1, 0.1])),
        (0, np.array([0.5, 0.1])),
        (0, np.array([0.5, 0.5])),
        (0, np.array([0.1, 0.5])),
    ]

    cells = {
        "quad": np.array([[0, 1, 2, 3]]),
        "triangle": np.array([[0, 1, 2], [0, 2, 3]]),
    }

    mesh = lg.UnstructuredMesh(
        geometry=geo,
        parametric_coordinates=coords,
        cells=cells,
        name="quad_tri_mesh",
    )

    # 1. Evaluate element centroids in CSDL
    centroids = mesh.compute_element_centroids()
    assert "quad" in centroids and "triangle" in centroids
    assert centroids["quad"].shape == (1, 3)
    assert centroids["triangle"].shape == (2, 3)

    # 2. Evaluate element normals and areas in CSDL
    normals_areas = mesh.compute_element_normals_and_areas()
    quad_n, quad_a = normals_areas["quad"]
    tri_n, tri_a = normals_areas["triangle"]
    assert quad_n.shape == (1, 3)
    assert tri_n.shape == (2, 3)
    assert quad_a.shape == (1,)
    assert tri_a.shape == (2,)

    # 3. Non-CSDL computation check
    centroids_np = mesh.compute_element_centroids(non_csdl=True)
    assert np.allclose(centroids["quad"].value, centroids_np["quad"])
    assert np.allclose(centroids["triangle"].value, centroids_np["triangle"])

    # 4. to_meshio conversion check
    try:
        import meshio
        m_obj = mesh.to_meshio()
        assert len(m_obj.points) == 4
        assert len(m_obj.cells) == 2
    except ImportError:
        pass


def test_iga_mesh_from_function():
    """Verify continuous Mesh created via from_function for Isogeometric Analysis (IGA)."""
    import lsdo_function_spaces as lfs

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    geo = lg.import_geometry(SAMPLE_STP, parallelize=False)

    # Create a 1D continuous curve in the (u, v) parameter space of patch 0: alpha(gamma) = [gamma, 0.5]
    space_1d = lfs.BSplineSpace(num_parametric_dimensions=1, degree=2, coefficients_shape=(4,))
    n_coef = space_1d.coefficients_shape[0]
    ctrl_pts = np.zeros((n_coef, 2))
    ctrl_pts[:, 0] = np.linspace(0.1, 0.9, n_coef)
    ctrl_pts[:, 1] = 0.5
    curve_fn = lfs.Function(space=space_1d, coefficients=ctrl_pts)

    # Instantiate Mesh as a continuous function map for IGA
    iga_mesh = lg.Mesh.from_function(
        geo,
        function_map=(0, curve_fn),
        name="iga_chord_curve",
    )
    assert isinstance(iga_mesh, lg.Mesh)
    assert isinstance(iga_mesh, lg.Marker)
    assert iga_mesh.domain is not None

    # Evaluate at quadrature / sample parameters
    sample_gammas = np.linspace(0.0, 1.0, 7).reshape((-1, 1))
    eval_pts = iga_mesh.evaluate(sample_parameters=sample_gammas)
    assert isinstance(eval_pts, csdl.Variable)
    assert eval_pts.shape == (7, 3)
    assert not np.isnan(eval_pts.value).any()

    # Evaluate surface tangents along the IGA mesh
    tu, tv = iga_mesh.compute_tangents()
    assert isinstance(tu, csdl.Variable)
    assert isinstance(tv, csdl.Variable)

