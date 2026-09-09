import pytest
import numpy as np
import csdl_alpha as csdl
import lsdo_geo as lg
import pyvista as pv
import os

SAMPLE_STP = os.path.join(
    os.path.dirname(__file__), "..", "examples", "example_geometries", "rectangular_wing.stp"
)

# Set PyVista to offscreen mode for automated headless testing
pv.OFF_SCREEN = True


def test_geometry_plot():
    """Verify geometry.plot returns valid PyVista plotting elements and renders offscreen."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    geo = lg.import_geometry(SAMPLE_STP, name="test_wing", parallelize=False)
    elements = geo.plot(show=False)

    assert isinstance(elements, list)
    assert len(elements) > 0

    # Verify rendering elements with PyVista offscreen
    plotter = pv.Plotter(off_screen=True)
    for el in elements:
        if isinstance(el, dict) and "mesh" in el:
            plotter.add_mesh(el["mesh"], **el.get("kwargs", {}))
        elif isinstance(el, pv.DataSet):
            plotter.add_mesh(el)
    # Render without errors
    plotter.render()
    plotter.close()


def test_geometry_plot_meshes():
    """Verify geometry.plot_meshes with surface, curve, point cloud, and arrow inputs."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    geo = lg.import_geometry(SAMPLE_STP, name="test_wing", parallelize=False)

    # 1. Surface mesh (3D array)
    u = np.linspace(0, 1, 5)
    v = np.linspace(0, 1, 4)
    uu, vv = np.meshgrid(u, v, indexing='ij')
    surface_mesh = np.stack([uu, vv, np.zeros_like(uu)], axis=-1)
    surface_var = csdl.Variable(value=surface_mesh)

    # 2. Curve mesh (2D array)
    curve_mesh = np.column_stack([np.linspace(0, 1, 10), np.zeros(10), np.zeros(10)])

    # 3. Point cloud
    point_cloud = np.array([[0.0, 0.0, 0.5], [1.0, 0.5, 0.5]])

    # 4. Vector arrow tuple
    arrow_vector = (csdl.Variable(value=np.array([0.0, 0.0, 0.0])), csdl.Variable(value=np.array([1.0, 0.0, 0.0])))

    elements = geo.plot_meshes(
        meshes=[surface_var, curve_mesh, point_cloud, arrow_vector],
        mesh_plot_types=['surface', 'wireframe', 'point_cloud'],
        show=False,
    )

    assert isinstance(elements, list)
    assert len(elements) > 0

    # Verify rendering with PyVista
    plotter = pv.Plotter(off_screen=True)
    for el in elements:
        if isinstance(el, dict) and "mesh" in el:
            plotter.add_mesh(el["mesh"], **el.get("kwargs", {}))
        elif isinstance(el, pv.DataSet):
            plotter.add_mesh(el)
    plotter.render()
    plotter.close()


def test_ffd_block_plot():
    """Verify FFDBlock.plot returns valid PyVista plotting elements and renders offscreen."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    pts = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 4.0, 6.0],
    ])
    pts_var = csdl.Variable(value=pts)

    ffd = lg.construct_ffd_block_around_entities(
        name="test_ffd",
        entities=pts_var,
        num_coefficients=(3, 3, 2),
        degree=(1, 1, 1),
    )

    elements = ffd.plot(show=False)
    assert isinstance(elements, list)
    assert len(elements) > 0

    # Verify rendering with PyVista
    plotter = pv.Plotter(off_screen=True)
    for el in elements:
        if isinstance(el, dict) and "mesh" in el:
            plotter.add_mesh(el["mesh"], **el.get("kwargs", {}))
        elif isinstance(el, pv.DataSet):
            plotter.add_mesh(el)
    plotter.render()
    plotter.close()


def test_volume_sectional_parameterization_plot():
    """Verify VolumeSectionalParameterization.plot returns valid PyVista plotting elements."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    geo = lg.import_geometry(SAMPLE_STP, parallelize=False)
    ffd = lg.construct_ffd_block_around_entities(
        name="wing_ffd",
        entities=geo,
        num_coefficients=(4, 3, 2),
        degree=(1, 1, 1),
    )

    vsp = lg.VolumeSectionalParameterization(
        name="test_vsp",
        parameterized_points=ffd.coefficients,
        principal_parametric_dimension=1,
    )

    elements = vsp.plot(show=False)
    assert isinstance(elements, list)
    assert len(elements) > 0

    # Verify rendering with PyVista
    plotter = pv.Plotter(off_screen=True)
    for el in elements:
        if isinstance(el, dict) and "mesh" in el:
            plotter.add_mesh(el["mesh"], **el.get("kwargs", {}))
        elif isinstance(el, pv.DataSet):
            plotter.add_mesh(el)
    plotter.render()
    plotter.close()


def test_pyvista_movie_recording(tmp_path):
    """Verify PyVista off-screen movie recording works as used in examples."""
    video_path = str(tmp_path / "test_movie.mp4")
    plotter = pv.Plotter(off_screen=True, window_size=[640, 480])
    plotter.open_movie(video_path, framerate=10)

    sphere = pv.Sphere()
    plotter.add_mesh(sphere, color="lightblue")
    plotter.add_text("Test Frame", position="lower_left", font_size=12)
    plotter.write_frame()
    plotter.close()

    assert os.path.exists(video_path)
    assert os.path.getsize(video_path) > 0
