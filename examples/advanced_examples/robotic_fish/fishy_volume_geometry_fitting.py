import lsdo_geo
import meshio
import vedo
import numpy as np

# Preprocessing to get geometry volume from mesh surfaces
msh = meshio.read("examples/advanced_examples/robotic_fish/meshes/fish_v1_finer_finer_finer_finer.msh")
mesh_nodes = msh.points/1000
# print(mesh_nodes.shape)

# from lsdo_geo.splines.b_splines.b_spline import BSpline, BSplineSpace
# # coefficients = np.meshgrid(np.linspace(0., 1., 10), np.linspace(0., 2., 10), np.linspace(0., 3., 10), indexing='ij')
# coefficients = np.meshgrid(np.array([0., 1.]), 2*np.array([0., 1.]), 3*np.array([0., 1.]), indexing='ij')
# coefficients = np.column_stack((coefficients[0].flatten(), coefficients[1].flatten(), coefficients[2].flatten()))
# test_cube_space = BSplineSpace(name='test_cube_space', order=(2,2,2), parametric_coefficients_shape=(2,2,2))
# test_cube = BSpline(name='test_cube', space=test_cube_space, coefficients=coefficients, num_physical_dimensions=3)
# # test_cube.plot()
# parametric_coordinates = np.meshgrid(np.linspace(0., 1., 21), np.linspace(0., 1., 21), np.linspace(0., 1., 21), indexing='ij')
# parametric_coordinates = np.column_stack((parametric_coordinates[0].flatten(), parametric_coordinates[1].flatten(), parametric_coordinates[2].flatten()))
# mesh_nodes = test_cube.evaluate(parametric_coordinates, plot=True).value.reshape((-1,3))

# Create an enclosure volume around the mesh

# "Project" the enclosure volume outer surfaces onto the mesh points (find the closest mesh point to each volume outer surface point)

# For each point pair, give the closest mesh point the parametric coordinate of the enclosure volume outer surface point

# Use the mesh coordinates (now physical and parametric) to fit a B-spline volume

# (Optional idea): Create a level-set volume to get the internal geometry information


# Create an enclosure volume around the mesh
from lsdo_geo.splines.b_splines.b_spline_functions import create_cartesian_enclosure_block

enclosure_volume = create_cartesian_enclosure_block(name='enclosure_volume', points=mesh_nodes, num_coefficients=(21,15,15), order=(3,3,3))
plotting_volume = enclosure_volume.plot(opacity=0.3, show=False)
plotting_points = vedo.Points(mesh_nodes, r=4, c='gold')
# plotter = vedo.Plotter()
# plotter.show([plotting_volume, plotting_points], axes=1)

# "Project" the enclosure volume outer surfaces onto the mesh points (find the closest mesh point to each volume outer surface point)
total_fitting_points = []
total_fitting_parametric_coordinates = []
num_fitting_points = (101,51,51)
# -- Back surface
back_surface_parametric_coordinates_u = np.zeros((num_fitting_points[1]*num_fitting_points[2],))
back_surface_parametric_coordinates_v = np.einsum('i,j->ij', np.linspace(0., 1., num_fitting_points[1]), np.ones(num_fitting_points[2])).flatten()
back_surface_parametric_coordinates_w = np.einsum('i,j->ij', np.ones(num_fitting_points[1]), np.linspace(0., 1., num_fitting_points[2])).flatten()
back_surface_parametric_coordinates = np.column_stack(
    (back_surface_parametric_coordinates_u, back_surface_parametric_coordinates_v, back_surface_parametric_coordinates_w))
back_surface_enclosure_points = enclosure_volume.evaluate(back_surface_parametric_coordinates, plot=False).value.reshape((-1,3))
# ---- Get closest mesh points
back_surface_mesh_points = np.zeros_like(back_surface_enclosure_points)
for i, point in enumerate(back_surface_enclosure_points):
    back_surface_mesh_points[i] = mesh_nodes[np.argmin(np.linalg.norm(mesh_nodes - point, axis=1))]
# back_surface_mesh_points_plotting = vedo.Points(back_surface_mesh_points, r=4, c='red')
# plotter = vedo.Plotter()
# plotter.show([plotting_volume, plotting_points, back_surface_mesh_points_plotting], axes=1)
back_surface_mesh_points = back_surface_mesh_points.reshape((num_fitting_points[1], num_fitting_points[2], 3))
total_fitting_points.append(back_surface_mesh_points)
total_fitting_parametric_coordinates.append(back_surface_parametric_coordinates)

# -- Front surface
front_surface_parametric_coordinates_u = np.ones((num_fitting_points[1]*num_fitting_points[2],))
front_surface_parametric_coordinates_v = np.einsum('i,j->ij', np.linspace(0., 1., num_fitting_points[1]), np.ones(num_fitting_points[2])).flatten()
front_surface_parametric_coordinates_w = np.einsum('i,j->ij', np.ones(num_fitting_points[1]), np.linspace(0., 1., num_fitting_points[2])).flatten()
front_surface_parametric_coordinates = np.column_stack(
    (front_surface_parametric_coordinates_u, front_surface_parametric_coordinates_v, front_surface_parametric_coordinates_w))
front_surface_enclosure_points = enclosure_volume.evaluate(front_surface_parametric_coordinates, plot=False).value.reshape((-1,3))
# ---- Get closest mesh points
front_surface_mesh_points = np.zeros_like(front_surface_enclosure_points)
for i, point in enumerate(front_surface_enclosure_points):
    front_surface_mesh_points[i] = mesh_nodes[np.argmin(np.linalg.norm(mesh_nodes - point, axis=1))]
# front_surface_mesh_points_plotting = vedo.Points(front_surface_mesh_points, r=4, c='red')
# plotter = vedo.Plotter()
# plotter.show([plotting_volume, plotting_points, front_surface_mesh_points_plotting], axes=1)
front_surface_mesh_points = front_surface_mesh_points.reshape((num_fitting_points[1], num_fitting_points[2], 3))
total_fitting_points.append(front_surface_mesh_points)
total_fitting_parametric_coordinates.append(front_surface_parametric_coordinates)

# -- Bottom surface
bottom_surface_parametric_coordinates_u = np.einsum('i,j->ij', np.linspace(0., 1., num_fitting_points[0]), np.ones(num_fitting_points[2])).flatten()
bottom_surface_parametric_coordinates_v = np.zeros((num_fitting_points[0]*num_fitting_points[2],))
bottom_surface_parametric_coordinates_w = np.einsum('i,j->ij', np.ones(num_fitting_points[0]), np.linspace(0., 1., num_fitting_points[2])).flatten()
bottom_surface_parametric_coordinates = np.column_stack(
    (bottom_surface_parametric_coordinates_u, bottom_surface_parametric_coordinates_v, bottom_surface_parametric_coordinates_w))
bottom_surface_enclosure_points = enclosure_volume.evaluate(bottom_surface_parametric_coordinates, plot=False).value.reshape((-1,3))
# ---- Get closest mesh points
bottom_surface_mesh_points = np.zeros_like(bottom_surface_enclosure_points)
for i, point in enumerate(bottom_surface_enclosure_points):
    bottom_surface_mesh_points[i] = mesh_nodes[np.argmin(np.linalg.norm(mesh_nodes - point, axis=1))]
# bottom_surface_mesh_points_plotting = vedo.Points(bottom_surface_mesh_points, r=4, c='red')
# plotter = vedo.Plotter()
# plotter.show([plotting_volume, plotting_points, bottom_surface_mesh_points_plotting], axes=1)
bottom_surface_mesh_points = bottom_surface_mesh_points.reshape((num_fitting_points[0], num_fitting_points[2], 3))
total_fitting_points.append(bottom_surface_mesh_points)
total_fitting_parametric_coordinates.append(bottom_surface_parametric_coordinates)

# -- Top surface
top_surface_parametric_coordinates_u = np.einsum('i,j->ij', np.linspace(0., 1., num_fitting_points[0]), np.ones(num_fitting_points[2])).flatten()
top_surface_parametric_coordinates_v = np.ones((num_fitting_points[0]*num_fitting_points[2],))
top_surface_parametric_coordinates_w = np.einsum('i,j->ij', np.ones(num_fitting_points[0]), np.linspace(0., 1., num_fitting_points[2])).flatten()
top_surface_parametric_coordinates = np.column_stack(
    (top_surface_parametric_coordinates_u, top_surface_parametric_coordinates_v, top_surface_parametric_coordinates_w))
top_surface_enclosure_points = enclosure_volume.evaluate(top_surface_parametric_coordinates, plot=False).value.reshape((-1,3))
# ---- Get closest mesh points
top_surface_mesh_points = np.zeros_like(top_surface_enclosure_points)
for i, point in enumerate(top_surface_enclosure_points):
    top_surface_mesh_points[i] = mesh_nodes[np.argmin(np.linalg.norm(mesh_nodes - point, axis=1))]
# top_surface_mesh_points_plotting = vedo.Points(top_surface_mesh_points, r=4, c='red')
# plotter = vedo.Plotter()
# plotter.show([plotting_volume, plotting_points, top_surface_mesh_points_plotting], axes=1)
top_surface_mesh_points = top_surface_mesh_points.reshape((num_fitting_points[0], num_fitting_points[2], 3))
total_fitting_points.append(top_surface_mesh_points)
total_fitting_parametric_coordinates.append(top_surface_parametric_coordinates)

# -- Left surface
left_surface_parametric_coordinates_u = np.einsum('i,j->ij', np.linspace(0., 1., num_fitting_points[0]), np.ones(num_fitting_points[1])).flatten()
left_surface_parametric_coordinates_v = np.einsum('i,j->ij', np.ones(num_fitting_points[0]), np.linspace(0., 1., num_fitting_points[1])).flatten()
left_surface_parametric_coordinates_w = np.zeros((num_fitting_points[0]*num_fitting_points[1],))
left_surface_parametric_coordinates = np.column_stack(
    (left_surface_parametric_coordinates_u, left_surface_parametric_coordinates_v, left_surface_parametric_coordinates_w))
left_surface_enclosure_points = enclosure_volume.evaluate(left_surface_parametric_coordinates, plot=False).value.reshape((-1,3))
# ---- Get closest mesh points
left_surface_mesh_points = np.zeros_like(left_surface_enclosure_points)
for i, point in enumerate(left_surface_enclosure_points):
    left_surface_mesh_points[i] = mesh_nodes[np.argmin(np.linalg.norm(mesh_nodes - point, axis=1))]
# left_surface_mesh_points_plotting = vedo.Points(left_surface_mesh_points, r=4, c='red')
# plotter = vedo.Plotter()
# plotter.show([plotting_volume, plotting_points, left_surface_mesh_points_plotting], axes=1)
left_surface_mesh_points = left_surface_mesh_points.reshape((num_fitting_points[0], num_fitting_points[1], 3))
total_fitting_points.append(left_surface_mesh_points)
total_fitting_parametric_coordinates.append(left_surface_parametric_coordinates)

# -- Right surface
right_surface_parametric_coordinates_u = np.einsum('i,j->ij', np.linspace(0., 1., num_fitting_points[0]), np.ones(num_fitting_points[1])).flatten()
right_surface_parametric_coordinates_v = np.einsum('i,j->ij', np.ones(num_fitting_points[0]), np.linspace(0., 1., num_fitting_points[1])).flatten()
right_surface_parametric_coordinates_w = np.ones((num_fitting_points[0]*num_fitting_points[1],))
right_surface_parametric_coordinates = np.column_stack(
    (right_surface_parametric_coordinates_u, right_surface_parametric_coordinates_v, right_surface_parametric_coordinates_w))
right_surface_enclosure_points = enclosure_volume.evaluate(right_surface_parametric_coordinates, plot=False).value.reshape((-1,3))
# ---- Get closest mesh points
right_surface_mesh_points = np.zeros_like(right_surface_enclosure_points)
for i, point in enumerate(right_surface_enclosure_points):
    right_surface_mesh_points[i] = mesh_nodes[np.argmin(np.linalg.norm(mesh_nodes - point, axis=1))]
# right_surface_mesh_points_plotting = vedo.Points(right_surface_mesh_points, r=4, c='red')
# plotter = vedo.Plotter()
# plotter.show([plotting_volume, plotting_points, right_surface_mesh_points_plotting], axes=1)
right_surface_mesh_points = right_surface_mesh_points.reshape((num_fitting_points[0], num_fitting_points[1], 3))
total_fitting_points.append(right_surface_mesh_points)
total_fitting_parametric_coordinates.append(right_surface_parametric_coordinates)


# Perform transfinite interpolation to get full grid of points to fit
u_linspace = np.linspace(0., 1., num_fitting_points[0])
v_linspace = np.linspace(0., 1., num_fitting_points[1])
w_linspace = np.linspace(0., 1., num_fitting_points[2])
num_points = np.prod(num_fitting_points)
boundary_maps = []  # there should be 6 of these for 3D (u0, u1, v0, v1, w0, w1) (back, front, bottom, top, left, right)
boundary_maps.append(np.zeros((num_points, num_fitting_points[1]*num_fitting_points[2])))
boundary_maps.append(np.zeros((num_points, num_fitting_points[1]*num_fitting_points[2])))
boundary_maps.append(np.zeros((num_points, num_fitting_points[0]*num_fitting_points[2])))
boundary_maps.append(np.zeros((num_points, num_fitting_points[0]*num_fitting_points[2])))
boundary_maps.append(np.zeros((num_points, num_fitting_points[0]*num_fitting_points[1])))
boundary_maps.append(np.zeros((num_points, num_fitting_points[0]*num_fitting_points[1])))
point_counter = 0
for i in range(num_fitting_points[0]):
    for j in range(num_fitting_points[1]):
        for k in range(num_fitting_points[2]):
            # N = 3
            # Add the linear combinations (surfaces)    # 6 surfaces (2*N)
            vw_index = num_fitting_points[2]*j + k
            uw_index = num_fitting_points[2]*i + k
            uv_index = num_fitting_points[1]*i + j
            boundary_maps[0][point_counter,vw_index] = 1-u_linspace[i]
            boundary_maps[1][point_counter,vw_index] = u_linspace[i]
            boundary_maps[2][point_counter,uw_index] = 1-v_linspace[j]
            boundary_maps[3][point_counter,uw_index] = v_linspace[j]
            boundary_maps[4][point_counter,uv_index] = 1-w_linspace[k]
            boundary_maps[5][point_counter,uv_index] = w_linspace[k]


            # Subtract intersecting boundaries (curves) # 12 curves (2*N**2)
            # - S1: back surface
            boundary_maps[0][point_counter,k] -= (1-u_linspace[i])*(1-v_linspace[j]) # back-bottom curve (C_13)
            boundary_maps[0][point_counter,-num_fitting_points[2]+k] -= (1-u_linspace[i])*v_linspace[j] # back-top curve (C_14)
            boundary_maps[0][point_counter,num_fitting_points[2]*j] -= (1-u_linspace[i])*(1-w_linspace[k]) # back-left curve (C_15)
            boundary_maps[0][point_counter,num_fitting_points[2]*(j+1)-1] -= (1-u_linspace[i])*w_linspace[k] # back-right curve (C_16)
            # - S2: front surface
            boundary_maps[1][point_counter,k] -= u_linspace[i]*(1-v_linspace[j]) # front-bottom curve (C_23)
            boundary_maps[1][point_counter,-num_fitting_points[2]+k] -= u_linspace[i]*v_linspace[j] # front-top curve (C_24)
            boundary_maps[1][point_counter,num_fitting_points[2]*j] -= u_linspace[i]*(1-w_linspace[k]) # front-left curve (C_25)
            boundary_maps[1][point_counter,num_fitting_points[2]*(j+1)-1] -= u_linspace[i]*w_linspace[k] # front-right curve (C_26)
            # - S3: bottom surface
            # -- C13 repeated so skip
            # -- C23 repeated so skip
            boundary_maps[2][point_counter,num_fitting_points[2]*i] -= (1-v_linspace[j])*(1-w_linspace[k]) # bottom-left curve (C_35)
            boundary_maps[2][point_counter,num_fitting_points[2]*(i+1)-1] -= (1-v_linspace[j])*w_linspace[k] # bottom-right curve (C_36)
            # - S4: top surface
            # -- C14 repeated so skip
            # -- C24 repeated so skip
            boundary_maps[3][point_counter,num_fitting_points[2]*i] -= v_linspace[j]*(1-w_linspace[k]) # top-left curve (C_45)
            boundary_maps[3][point_counter,num_fitting_points[2]*(i+1)-1] -= v_linspace[j]*w_linspace[k] # top-right curve (C_46)
            # - S5: left surface
            # -- C15 repeated so skip
            # -- C25 repeated so skip
            # -- C35 repeated so skip
            # -- C45 repeated so skip
            # - S6: right surface
            # -- C16 repeated so skip
            # -- C26 repeated so skip
            # -- C36 repeated so skip
            # -- C46 repeated so skip


            # Add back on the corners (points)  # 8 corners (2*N**(N-1) = 2**(N))
            boundary_maps[0][point_counter,0] += (1-u_linspace[i])*(1-v_linspace[j])*(1-w_linspace[k]) # back-bottom-left corner (P_135)
            boundary_maps[0][point_counter,num_fitting_points[2]-1] += (1-u_linspace[i])*(1-v_linspace[j])*w_linspace[k] # back-bottom-right corner (P_136)
            boundary_maps[0][point_counter,num_fitting_points[2]*(num_fitting_points[1]-1)] += (1-u_linspace[i])*v_linspace[j]*(1-w_linspace[k]) # back-top-left corner (P_145)
            boundary_maps[0][point_counter, -1] += (1-u_linspace[i])*v_linspace[j]*w_linspace[k] # back-top-right corner (P_146)
            boundary_maps[1][point_counter,0] += u_linspace[i]*(1-v_linspace[j])*(1-w_linspace[k]) # front-bottom-left corner (P_235)
            boundary_maps[1][point_counter,num_fitting_points[2]-1] += u_linspace[i]*(1-v_linspace[j])*w_linspace[k] # front-bottom-right corner (P_236)
            boundary_maps[1][point_counter,num_fitting_points[2]*(num_fitting_points[1]-1)] += u_linspace[i]*v_linspace[j]*(1-w_linspace[k]) # front-top-left corner (P_245)
            boundary_maps[1][point_counter, -1] += u_linspace[i]*v_linspace[j]*w_linspace[k] # front-top-right corner (P_246)

            point_counter += 1

contribution_from_surface_1 = boundary_maps[0].dot(back_surface_mesh_points.reshape((-1,3)))
contribution_from_surface_2 = boundary_maps[1].dot(front_surface_mesh_points.reshape((-1,3)))
contribution_from_surface_3 = boundary_maps[2].dot(bottom_surface_mesh_points.reshape((-1,3)))
contribution_from_surface_4 = boundary_maps[3].dot(top_surface_mesh_points.reshape((-1,3)))
contribution_from_surface_5 = boundary_maps[4].dot(left_surface_mesh_points.reshape((-1,3)))
contribution_from_surface_6 = boundary_maps[5].dot(right_surface_mesh_points.reshape((-1,3)))

total_fitting_points = contribution_from_surface_1 + contribution_from_surface_2 + contribution_from_surface_3 \
                     + contribution_from_surface_4 + contribution_from_surface_5 + contribution_from_surface_6
total_fitting_parametric_coordinates_tuple = np.meshgrid(u_linspace, v_linspace, w_linspace, indexing='ij')
total_fitting_parametric_coordinates = np.column_stack(
    (total_fitting_parametric_coordinates_tuple[0].flatten(), total_fitting_parametric_coordinates_tuple[1].flatten(), total_fitting_parametric_coordinates_tuple[2].flatten()))

# total_fitting_points = np.vstack(total_fitting_points)
# total_fitting_parametric_coordinates = np.vstack(total_fitting_parametric_coordinates)

# Use the mesh coordinates (now physical and parametric) to fit a B-spline volume
from lsdo_geo.splines.b_splines.b_spline_functions import fit_b_spline

fishy = fit_b_spline(fitting_points=total_fitting_points, parametric_coordinates=total_fitting_parametric_coordinates, order=(3,3,3),
                      num_coefficients=(31,15,15), regularization_parameter=1.e-1)
fishy.plot()


# (Optional idea): Create a level-set volume to get the internal geometry information

# Save geometry in pickle file so this doesn't have to be redone each run
import os
import pickle
file_name = "examples/advanced_examples/robotic_fish/pickle_files/fishy_volume_geometry.pickle"
# fn = os.path.basename(file_name)
# fn_wo_ext = fn[:fn.rindex('.')]

# saved_geometry = IMPORT_FOLDER / f'{fn_wo_ext}_stored_import_dict.pickle'

# if name == 'geometry':
#     name = fn_wo_ext

# saved_geometry_file = Path(saved_geometry) 

# if saved_geometry_file.is_file():
#     with open(saved_geometry, 'rb') as handle:
#         import_dict = pickle.load(handle)
#         b_splines = import_dict['b_splines']

# else:
with open(file_name, 'wb+') as handle:
    pickle.dump(fishy, handle, protocol=pickle.HIGHEST_PROTOCOL)
