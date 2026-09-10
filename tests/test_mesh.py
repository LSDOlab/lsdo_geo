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
