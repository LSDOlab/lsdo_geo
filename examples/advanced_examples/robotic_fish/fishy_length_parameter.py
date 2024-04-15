import lsdo_geo
import lsdo_geo.splines.b_splines.b_spline_functions as bsp

import numpy as np
import m3l
import vedo

from panel_method.utils.generate_meshes import generate_fish_v_m1,get_connectivity_matrix_from_pyvista
from panel_method.utils.generate_eel_verifcation import (
    generate_eel_carling,get_connectivity_matrix_from_pyvista,neighbor_cell_idx, get_swimming_eel_geometry)

num_pts_L = 50
num_pts_R = 23
L = 1.
s_1_ind = 5
s_2_ind = 45
num_fish = 1

grid = generate_eel_carling(num_pts_L,num_pts_R,L,s_1_ind,s_2_ind)
grid_shape = grid.dimensions[:-1] + (3,)
grid_points = np.array(grid.points).reshape(grid_shape, order='F')

num_coefficients_u = 23
num_coefficients_v = 15

fish_surface = bsp.fit_b_spline(fitting_points=grid_points, order=(4,4), 
                                num_coefficients=(num_coefficients_u,num_coefficients_v), name='fisho_surface')

num_nodes = (num_pts_L//2, num_pts_R*3)
volume_elements_mesh_parametetric_coordinates = bsp.generate_parametric_grid(num_nodes)
quad_mesh = fish_surface.evaluate(volume_elements_mesh_parametetric_coordinates).reshape((num_pts_L//2,num_pts_R*3,3))

num_control_points_u = quad_mesh.shape[0]
num_control_points_v = quad_mesh.shape[1]
vertices = []
faces = []
for u_index in range(num_control_points_u):
    for v_index in range(num_control_points_v):
        vertex = tuple(quad_mesh.value[u_index, v_index, :])
        vertices.append(vertex)
        if u_index != 0 and v_index != 0:
            face = tuple((
                (u_index-1)*num_control_points_v+(v_index-1),
                (u_index-1)*num_control_points_v+(v_index),
                (u_index)*num_control_points_v+(v_index),
                (u_index)*num_control_points_v+(v_index-1),
            ))
            faces.append(face)


fish = fish_surface.plot(show=False, opacity=0.3)
mesh = vedo.Mesh([vertices, faces]).opacity(1.).color('green').wireframe()
# plotter = vedo.Plotter()
# plotter.show([fish[0], mesh], axes=1)

# video = vedo.Video('test.mp4', duration=None, fps=10, backend="cv")
camera = dict(
    position=(-1, -1.25, 0.25),
    focal_point=(0.2, 0, -0.01),
    viewup=(0, 0, 1),
    distance=0,
)

# region length translation

fisho_b_spline_set = bsp.create_b_spline_set(name='fisho', b_splines={'fisho_surface':fish_surface})
fishy = lsdo_geo.Geometry(name='fishy', space=fisho_b_spline_set.space, coefficients=fisho_b_spline_set.coefficients, 
                        num_physical_dimensions={'fisho_surface': 3})
# fishy.plot()

panel_mesh_parametric_coordinates = []
for i in range(volume_elements_mesh_parametetric_coordinates.shape[0]):
    panel_mesh_parametric_coordinates.append(('fisho_surface', volume_elements_mesh_parametetric_coordinates[i,:]))
panel_mesh = fishy.evaluate(panel_mesh_parametric_coordinates, plot=False).reshape((num_pts_L//2,num_pts_R*3,3))
undeformed_mesh = fishy.plot_meshes([panel_mesh], show=False)


from lsdo_geo.core.parameterization.free_form_deformation_functions import construct_ffd_block_around_entities
ffd_block_coefficients_shape = (10,2,2)
ffd_block = construct_ffd_block_around_entities(name='ffd_block', entities=fishy, num_coefficients=ffd_block_coefficients_shape)
# ffd_block.plot()

from lsdo_geo.core.parameterization.volume_sectional_parameterization import VolumeSectionalParameterization
volume_sectional_parameterization = VolumeSectionalParameterization(
    name='sectional_parameterization',
    parameterized_points=ffd_block.coefficients,
    principal_parametric_dimension=0,
    parameterized_points_shape=ffd_block_coefficients_shape + (3,),
)

# volume_sectional_parameterization.plot(additional_plotting_elements=undeformed_mesh, show=True)
volume_sectional_parameterization.add_sectional_translation(name='length_stretch', axis=0)
# volume_sectional_parameterization.add_sectional_stretch(name='width_stretch', axis=1)
# volume_sectional_parameterization.add_sectional_stretch(name='height_stretch', axis=2)
# volume_sectional_parameterization.add_sectional_translation(name='wiggle_translations', axis=1)

from lsdo_geo.splines.b_splines.b_spline import BSpline
import lsdo_geo.splines.b_splines as bsp

space_of_linear_2_dof_b_splines = bsp.BSplineSpace(name='space_of_linear_2_dof_b_splines', order=2, parametric_coefficients_shape=(2,))
space_of_cubic_5_dof_b_splines = bsp.BSplineSpace(name='space_of_cubic_5_dof_b_splines', order=4, parametric_coefficients_shape=(5,))

length_stretch_coefficients = m3l.Variable(name='length_stretch_coefficients', shape=(2,), value=np.array([0., 0.]))
length_stretch_b_spline = bsp.BSpline(name='length_stretch', space=space_of_linear_2_dof_b_splines, coefficients=length_stretch_coefficients,
                                    num_physical_dimensions=1)

parametric_coordinates = np.linspace(0., 1., 10).reshape((10,1))
length_stretch = length_stretch_b_spline.evaluate(parametric_coordinates)

sectional_parameters = {
    'length_stretch': length_stretch,
}

ffd_block_coefficients = volume_sectional_parameterization.evaluate(sectional_parameters=sectional_parameters, plot=False)

fishy_coefficients = ffd_block.evaluate(ffd_block_coefficients, plot=False)

fishy.assign_coefficients(fishy_coefficients)

fishy_nose = fishy.evaluate([('fisho_surface', np.array([0., 0.]))], plot=False)
fishy_tail = fishy.evaluate([('fisho_surface', np.array([0., 1.]))], plot=False)
fishy_length = m3l.norm(fishy_nose - fishy_tail)

from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver
parameterization_solver = ParameterizationSolver()
parameterization_solver.declare_input('fishy_length', input=fishy_length)
parameterization_solver.declare_state('length_stretch_coefficients', length_stretch_coefficients)
video = vedo.Video('length_parameter.mp4', duration=4, backend="cv")
parameter_values = np.hstack((np.linspace(1, 2, 20), np.linspace(2, 0.5, 30), np.linspace(0.5, 1, 10)))
for step in range(len(parameter_values)):
    parameter_value = parameter_values[step]

    fishy_length_dv = m3l.Variable(name='fishy_length', shape=(1,), value=np.array([parameter_value]))

    outputs_dict = parameterization_solver.evaluate(inputs={'fishy_length': fishy_length_dv})
    length_stretch_coefficients = outputs_dict['length_stretch_coefficients']

    length_stretch_b_spline = bsp.BSpline(name='length_stretch', space=space_of_linear_2_dof_b_splines, coefficients=length_stretch_coefficients,
                                        num_physical_dimensions=1)


    parametric_coordinates = np.linspace(0., 1., 10).reshape((10,1))
    length_stretch = length_stretch_b_spline.evaluate(parametric_coordinates)

    sectional_parameters = {
        'length_stretch': length_stretch,
    }

    ffd_block_coefficients = volume_sectional_parameterization.evaluate(sectional_parameters=sectional_parameters, plot=False)

    fishy_coefficients = ffd_block.evaluate(ffd_block_coefficients, plot=False)

    fishy.assign_coefficients(fishy_coefficients)

    panel_mesh_parametric_coordinates = []
    for i in range(volume_elements_mesh_parametetric_coordinates.shape[0]):
        panel_mesh_parametric_coordinates.append(('fisho_surface', volume_elements_mesh_parametetric_coordinates[i,:]))
    panel_mesh = fishy.evaluate(panel_mesh_parametric_coordinates, plot=False).reshape((num_pts_L//2,num_pts_R*3,3))

    sectional_ffd_plots = volume_sectional_parameterization.plot(show=False)
    # plotting_elements = fishy.plot_meshes([panel_mesh], additional_plotting_elements=sectional_ffd_plots, show=False)
    plotting_elements = fishy.plot_meshes([panel_mesh], show=False)
    text = vedo.Text2D(f'length: {parameter_value:.2f}', pos='bottom-middle', s=5, c='black')

    import vedo
    plotter = vedo.Plotter(size=(1920,1000),offscreen=True)
    plotter.show(plotting_elements, text, camera=camera, axes=1)

    video.add_frame()
video.close()
# endregion

print("I'm done.")