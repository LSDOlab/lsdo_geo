import lsdo_geo
import lsdo_geo.splines.b_splines.b_spline_functions as bsp

import numpy as np
import m3l
import vedo

from panel_method.utils.generate_meshes import generate_fish_v_m1,get_connectivity_matrix_from_pyvista
from panel_method.utils.generate_eel_verifcation import (
    generate_eel_carling,get_connectivity_matrix_from_pyvista,neighbor_cell_idx, get_swimming_eel_geometry)
# RADIUS      = 0.5
# # N_L         = 31            # number of points in length direction
# # N_R         = 11            # number of points in radial direction (odd number is better)

# N_L         = 51            # number of points in length direction
# N_R         = 25            # number of points in radial direction (odd number is better)
# N_taper     = int(N_L - 4)  # start of the tail taper
# num_fish = 1

# angles= np.zeros(num_fish)             # bulk rotation of fish


# points_rigid, _, _ = generate_fish_v_m1(angles, N_L, N_R, N_taper)
# points_rigid = points_rigid.reshape((N_L,N_R,3))/100

# import vedo
# vedo.show(vedo.Points(points_rigid.reshape((-1,3))), interactive=True)

num_pts_L = 50
num_pts_R = 23
L = 1.
s_1_ind = 5
s_2_ind = 45
num_fish = 1

grid = generate_eel_carling(num_pts_L,num_pts_R,L,s_1_ind,s_2_ind)
grid_shape = grid.dimensions[:-1] + (3,)
grid_points = np.array(grid.points).reshape(grid_shape, order='F')

# import vedo
# vedo.show(vedo.Points(grid_points[:,:,:].reshape((-1,3))), interactive=True, axes=1)


num_coefficients_u = 23
num_coefficients_v = 15

fish_surface = bsp.fit_b_spline(fitting_points=grid_points, order=(4,4), 
                                num_coefficients=(num_coefficients_u,num_coefficients_v), name='fisho_surface')
# fish_surface = bsp.fit_b_spline(fitting_points=points_rigid, order=(2,3), 
#                                 num_coefficients=(num_coefficients_u,num_coefficients_v), name='fisho_surface')
num_nodes = (num_pts_L*2, num_pts_R)
volume_elements_mesh_parametetric_coordinates = bsp.generate_parametric_grid(num_nodes)
# parametric_coordinates = []
# for i in range(volume_elements_mesh_parametetric_coordinates.shape[0]):
#     parametric_coordinates.append(('fisho_volume', volume_elements_mesh_parametetric_coordinates[i,:].reshape((1,-1))))
# quad_mesh = fish_surface.evaluate(parametric_coordinates).reshape((N_L*2,N_R,3))
quad_mesh = fish_surface.evaluate(volume_elements_mesh_parametetric_coordinates).reshape((num_pts_L*2,num_pts_R,3))

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
plotter = vedo.Plotter()
plotter.show([fish[0], mesh], axes=1)

# fish_surface.plot()

# parametric_coordinates = np.hstack((np.linspace(0.999, 0.999, 51).reshape((-1,1)), np.linspace(0., 0.75, 51).reshape((-1,1))))
# points = fish_surface.evaluate(parametric_coordinates=parametric_coordinates).value
# vedo.show(vedo.Points(points.reshape((-1,3))), interactive=True, axes=1)

# num_coefficients_u = 21
# num_coefficients_v = 11
# grid_resolution = (101,51)
# parametric_coordinates = bsp.generate_parametric_grid(grid_resolution)

# grid_points = fish_surface.evaluate(parametric_coordinates)
# grid_u_derivative = fish_surface.evaluate(parametric_coordinates=parametric_coordinates, parametric_derivative_order=(1,0))
# grid_v_derivative = fish_surface.evaluate(parametric_coordinates=parametric_coordinates, parametric_derivative_order=(0,1))
# thickness = 0.01

# b_spline_evaluated_points = grid_points.value.reshape(grid_resolution + (3,))
# b_spline_evaluated_points_u_derivative = grid_u_derivative.value.reshape(grid_resolution + (3,))
# b_spline_evaluated_points_v_derivative = grid_v_derivative.value.reshape(grid_resolution + (3,))
# try:    # this is only here because numpy has bug where pylance thinks cross doesn't return anything and code after is unreachable
#     b_spline_normal_vectors = np.cross(b_spline_evaluated_points_u_derivative, -b_spline_evaluated_points_v_derivative)
# except:
#     pass
# corner_indices = np.where(np.linalg.norm(b_spline_evaluated_points_v_derivative, axis=-1) < 1e-10)
# b_spline_normal_vectors[np.where((np.linalg.norm(b_spline_evaluated_points_v_derivative, axis=-1) < 1e-10) \
#                                  & (np.linalg.norm(b_spline_evaluated_points, axis=-1) < 0.5))] = np.array([-1., 0., 0.])
# # b_spline_normal_vectors[np.where((np.linalg.norm(b_spline_evaluated_points_v_derivative, axis=-1) < 1e-2) \
# #                                  & (np.linalg.norm(b_spline_evaluated_points, axis=-1) > 0.5)\
# #                                  & (b_spline_evaluated_points[:,:,2] > RADIUS/2/100))] = np.array([0., 0., 1.])
# # b_spline_normal_vectors[np.where((np.linalg.norm(b_spline_evaluated_points_v_derivative, axis=-1) < 1e-2) \
# #                                  & (np.linalg.norm(b_spline_evaluated_points, axis=-1) > 0.5)\
# #                                  & (b_spline_evaluated_points[:,:,2] < RADIUS/2/100))] = np.array([0., 0., 1.])
# b_spline_normal_vectors[-1,:,:] = np.array([1., 0., 0.])
# # plotting_points_3 = vedo.Points(b_spline_evaluated_points[-1,:,:].reshape((-1,3)), c="green", r=6)
# # else:
# #     b_spline_normal_vectors[np.where(np.linalg.norm(b_spline_evaluated_points_v_derivative, axis=-1) < 1e-10)] = np.array([1., 0., 0.])
# b_spline_normal_vectors /= np.linalg.norm(b_spline_normal_vectors, axis=-1, keepdims=True)

# b_spline_offset_points = b_spline_evaluated_points + thickness*b_spline_normal_vectors
# current_height = np.max(b_spline_evaluated_points[-1,:,2]) - np.min(b_spline_evaluated_points[-1,:,2])
# b_spline_offset_points[-1,:,2] *= (current_height + 2*thickness)/current_height

# # plotting_points_1 = vedo.Points(b_spline_evaluated_points.reshape((-1,3)), c="blue")
# # plotting_points_2 = vedo.Points(b_spline_offset_points.reshape((-1,3)), c="red")
# # plotter = vedo.Plotter()
# # plotter.show([plotting_points_1, plotting_points_2], axes=1)
# # plotter.show([plotting_points_3], axes=1)

# # vedo.show([plotting_points_1, plotting_points_2], axes=1)
# b_spline_volume_points = np.stack((b_spline_evaluated_points, b_spline_offset_points), axis=-2)

# parametric_coordinates = bsp.generate_parametric_grid(grid_resolution + (2,))

# fisho = bsp.fit_b_spline(b_spline_volume_points, parametric_coordinates, order=(3,3,2), 
#                                 num_coefficients=(num_coefficients_u,num_coefficients_v,2), name='fisho_volume')
# # fisho.plot()
# fisho_b_spline_set = bsp.create_b_spline_set(name='fisho', b_splines={'fisho_volume':fisho})

fisho_b_spline_set = bsp.create_b_spline_set(name='fisho', b_splines={'fisho_surface':fish_surface})

fishy = lsdo_geo.Geometry(name='fishy', space=fisho_b_spline_set.space, coefficients=fisho_b_spline_set.coefficients, 
                          num_physical_dimensions={'fisho_surface': 3})
# Note: In the future, the surface that we fit to the generated mesh could be the imported geometry and we can use M3L operations to
#       create the thickness in the meshes. I should think about this vs importing the volume.
#      The oml/surface could be the initial geometry, and we can use operations to create the volume coefficients, but we also have a
#       python object for the volume so we can project into that?
#     No, we make the volume object with the coefficients as an M3L variable where the offset points are computed using M3L operations.
#   For now though, we use numpy to make it easier for indexing, etc.

fishy.plot()

# num_nodes = (num_pts_L*2, num_pts_R, 2)
num_nodes = (num_pts_L*2, num_pts_R)
volume_elements_mesh_parametetric_coordinates = bsp.generate_parametric_grid(num_nodes)
parametric_coordinates = []
for i in range(volume_elements_mesh_parametetric_coordinates.shape[0]):
    parametric_coordinates.append(('fisho_surface', volume_elements_mesh_parametetric_coordinates[i,:].reshape((1,-1))))
quad_mesh = fishy.evaluate(parametric_coordinates)

plotting_points = vedo.Points(quad_mesh.value.reshape((-1,3)), c="blue")
plotter = vedo.Plotter()
plotter.show(plotting_points, axes=1)

# fishy.plot_meshes(meshes=[quad_mesh], plot_types=['wirframe'], show=True) # Volume wireframe mesh plotting not implemented.




# geometry = lsdo_geo.import_geometry('lsdo_geo/splines/b_splines/sample_geometries/openvsp_fish.stp', parallelize=False)
# # geometry = import_geometry('lsdo_geo/splines/b_splines/sample_geometries/lift_plus_cruise_final.stp')
# geometry.refit(parallelize=False)

# # geometry.plot()
# # for b_spline_name in list(geometry.coefficient_indices.keys()):
# #     print(b_spline_name)
# #     geometry.plot(b_splines=[b_spline_name], plot_types=['surface'], show=True)

# num_coefficients_u = 50
# num_coefficients_v = 15
# grid_resolution = (500,50)
# grid_points, grid_point_indices = geometry.evaluate_grid(grid_resolution=grid_resolution)
# grid_u_derivative, grid_point_indices = geometry.evaluate_grid(grid_resolution=grid_resolution, parametric_derivative_order=(1,0))
# grid_v_derivative, grid_point_indices = geometry.evaluate_grid(grid_resolution=grid_resolution, parametric_derivative_order=(0,1))
# thickness = 0.01

# # b_spline_volume_points = {}
# new_b_splines = {}
# for b_spline_name in geometry.coefficient_indices.keys():
#     b_spline_evaluated_points = grid_points.reshape((-1,3))[grid_point_indices[b_spline_name],:].reshape(grid_resolution + (3,))
#     b_spline_evaluated_points_u_derivative = grid_u_derivative.reshape((-1,3))[grid_point_indices[b_spline_name],:].reshape(grid_resolution + (3,))
#     b_spline_evaluated_points_v_derivative = grid_v_derivative.reshape((-1,3))[grid_point_indices[b_spline_name],:].reshape(grid_resolution + (3,))
#     b_spline_normal_vectors = np.cross(b_spline_evaluated_points_u_derivative, -b_spline_evaluated_points_v_derivative)
#     b_spline_normal_vectors /= np.linalg.norm(b_spline_normal_vectors, axis=-1, keepdims=True)
#     corner_indices = np.where(np.linalg.norm(b_spline_evaluated_points_v_derivative, axis=-1) < 1e-10)
#     if b_spline_evaluated_points[corner_indices][0,0] < 0.5:
#         b_spline_normal_vectors[np.where(np.linalg.norm(b_spline_evaluated_points_v_derivative, axis=-1) < 1e-10)] = np.array([-1., 0., 0.])
#     else:
#         b_spline_normal_vectors[np.where(np.linalg.norm(b_spline_evaluated_points_v_derivative, axis=-1) < 1e-10)] = np.array([1., 0., 0.])
#     b_spline_offset_points = b_spline_evaluated_points + thickness*b_spline_normal_vectors

#     b_spline_volume_points = np.stack((b_spline_evaluated_points, b_spline_offset_points), axis=-2)

#     mesh_grid_input = []
#     num_parametric_dimensions = 3
#     b_spline_grid_resolution = grid_resolution + (2,)
#     for dimension_index in range(num_parametric_dimensions):
#         mesh_grid_input.append(np.linspace(0., 1., b_spline_grid_resolution[dimension_index]))

#     parametric_coordinates_tuple = np.meshgrid(*mesh_grid_input, indexing='ij')
#     for dimensions_index in range(num_parametric_dimensions):
#         parametric_coordinates_tuple[dimensions_index] = parametric_coordinates_tuple[dimensions_index].reshape((-1,1))

#     parametric_coordinates = np.hstack(parametric_coordinates_tuple)

#     new_b_spline = bsp.fit_b_spline(b_spline_volume_points, parametric_coordinates, order=(3,3,2), 
#                                     num_coefficients=(num_coefficients_u,num_coefficients_v,2), name=b_spline_name)
#     new_b_spline.plot()   # need to change fit B-spline name in order to plot
#     new_b_splines[b_spline_name] = new_b_spline

# fisho = bsp.create_b_spline_set(name='fisho', b_splines=new_b_splines)
# fisho.plot()



print("I'm done.")