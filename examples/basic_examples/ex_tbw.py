import csdl_alpha as csdl
import numpy as np
import lsdo_function_spaces as lfs

from lsdo_geo.core.parameterization.free_form_deformation_functions import (
    construct_tight_fit_ffd_block,construct_ffd_block_around_entities,construct_ffd_block_from_corners
)
from lsdo_geo.core.parameterization.volume_sectional_parameterization import (
    VolumeSectionalParameterization,
    VolumeSectionalParameterizationInputs
)


import lsdo_geo


recorder = csdl.Recorder(inline=True)
recorder.start()

geometry = lsdo_geo.import_geometry(
    "examples/example_geometries/tbw.stp",
    # "examples/example_geometries/simple_wing.stp",
    parallelize=False,
)
# geometry.plot()

# region Key locations
# leading_edge_left = geometry.project(np.array([68.0, -80.0, 8.0]))
# leading_edge_right = geometry.project(np.array([68.0, 80.0, 8.0]))
# trailing_edge_left = geometry.project(np.array([72.0, -80.0, 8.0]))
# trailing_edge_right = geometry.project(np.array([72.0, 80.0, 8.0]))
# leading_edge_center = geometry.project(np.array([45.0, 0.0, 9.0]))
# trailing_edge_center = geometry.project(np.array([53.0, 0.0, 9.0]))
# quarter_chord_left = geometry.project(np.array([69., -80.0, 0.0]))
# quarter_chord_right = geometry.project(np.array([69., 80.0, 0.0]))
# quarter_chord_center = geometry.project(np.array([47., 0.0, 0.0]))

# points = np.array([
# [
#     [
#         [68.136 - 21.0, 85.291 + 0.5, 4.741 - 9.5],  # right tip chord
#         [68.136 - 21.0, 85.291 + 0.5, 4.741 + 4.6]
#     ],
#     [
#         [71.664 + 2.0, 85.291 + 0.5, 4.741 - 9.5],
#         [71.664 + 2.0, 85.291 + 0.5, 4.741 + 4.6]
#     ]
# ],
# [
#     [
#         [68.136 - 21.0, -85.291 - 0.5, 4.741 - 9.5],  # left tip chord
#         [68.136 - 21.0, -85.291 - 0.5, 4.741 + 4.6]
#     ],
#     [
#         [71.664 + 1.0, -85.291 - 0.5, 4.741 - 9.5],
#         [71.664 + 1.0, -85.291 - 0.5, 4.741 + 4.6]
#     ]
# ]
# ])

point20 = np.array([68.136 - 21.0, 85.291 + 0.5, 4.741 + 4.6])
point21 = np.array([68.136 - 21.0, -85.291 - 0.5, 4.741 + 4.6])

right_tip_leading_edge = np.array([68.035, 85.291, 4.704 + 0.1])  # * ft2m # Right tip leading edge
right_tip_trailing_edge = np.array([71.790, 85.291, 4.708 + 0.1])  # * ft2m # Right tip trailing edge
center_leading_edge = np.array([47.231, 0.000, 6.937 + 0.1])  # * ft2m # Center Leading Edge
center_trailing_edge = np.array([57.953, 0.000, 6.574 + 0.1])  # * ft2m # Center Trailing edge
left_tip_leading_edge = np.array([68.035, -85.291, 4.704 + 0.1])  # * ft2m # Left tip leading edge
left_tip_trailing_edge = np.array([71.790, -85.291, 4.708 + 0.1])  # * ft2m # Left tip trailing edge

right_tip_leading_edge_parametric = geometry.project(right_tip_leading_edge)
right_tip_trailing_edge_parametric = geometry.project(right_tip_trailing_edge)
center_leading_edge_parametric = geometry.project(center_leading_edge)
center_trailing_edge_parametric = geometry.project(center_trailing_edge)
left_tip_leading_edge_parametric = geometry.project(left_tip_leading_edge)
left_tip_trailing_edge_parametric = geometry.project(left_tip_trailing_edge)
# endregion

# Define geometry components
wing = geometry.declare_component(function_search_names=['Wing'])

# endregion Define geometry components

# region Mesh definitions
# region Wing Camber Surface
num_spanwise = 21
num_chordwise = 6
points_to_project_on_leading_edge_left = np.linspace(left_tip_leading_edge, center_leading_edge, num_spanwise//2, endpoint=False)
points_to_project_on_leading_edge_right = np.linspace(center_leading_edge, right_tip_leading_edge, num_spanwise//2+1)
points_to_project_leading_edge = np.vstack((points_to_project_on_leading_edge_left, points_to_project_on_leading_edge_right))

points_to_project_on_trailing_edge_left = np.linspace(left_tip_trailing_edge, center_trailing_edge, num_spanwise//2, endpoint=False)
points_to_project_on_trailing_edge_right = np.linspace(center_trailing_edge, right_tip_trailing_edge, num_spanwise//2+1)
points_to_project_trailing_edge = np.vstack((points_to_project_on_trailing_edge_left, points_to_project_on_trailing_edge_right))

leading_edge_parametric = wing.project(points_to_project_leading_edge, direction=np.array([0., 0., -1.]), plot=False)
leading_edge_physical = wing.evaluate(leading_edge_parametric, plot=False)
trailing_edge_parametric = wing.project(points_to_project_trailing_edge, direction=np.array([0., 0., -1.]), plot=False)
trailing_edge_physical = wing.evaluate(trailing_edge_parametric)

chord_surface = csdl.linear_combination(leading_edge_physical, trailing_edge_physical, num_chordwise).value.reshape((num_chordwise, num_spanwise, 3))
upper_surface_wireframe_parametric = wing.project(chord_surface + np.array([0., 0., 1]), direction=np.array([0., 0., -1.]), 
                                                      grid_search_density_parameter=10, plot=False)
lower_surface_wireframe_parametric = wing.project(chord_surface - np.array([0., 0., 1]), direction=np.array([0., 0., -1.]), 
                                                      grid_search_density_parameter=10, plot=False)

class CamberSurfaceMesh(lsdo_geo.Mesh):
    def evaluate(self, geometry:lsdo_geo.Geometry):
        evaluated_points = super().evaluate(geometry)
        upper_surface_wireframe = evaluated_points[:num_chordwise*num_spanwise]
        lower_surface_wireframe = evaluated_points[num_chordwise*num_spanwise:]

        camber_surface = csdl.linear_combination(upper_surface_wireframe, lower_surface_wireframe, 1).reshape((num_chordwise, num_spanwise, 3))
        return camber_surface
    

# wing_parametric_grid = wing.generate_parametric_grid(grid_resolution=(10,10))
# wing_grid = wing.evaluate(wing_parametric_grid, plot=True)
# counter = 0
# total_area = 0
# for function_index, function in wing.functions.items():
#     function_grid_parametric = function.space.generate_parametric_grid(grid_resolution=(10,10))
#     function_grid = function.evaluate(function_grid_parametric, plot=False).reshape((10,10,3))
#     u_vectors = function_grid[1:] - function_grid[:-1]
#     v_vectors = function_grid[:,1:] - function_grid[:,:-1]
#     area_vectors = csdl.cross(u_vectors[:,:-1], v_vectors[:-1,:], axis=2)
#     areas = csdl.norm(area_vectors, axes=(2,))
#     total_area = total_area + csdl.sum(areas)

# print(total_area.value)
# exit()
camber_surface_mesh = CamberSurfaceMesh(geometry=geometry, parametric_coordinates=upper_surface_wireframe_parametric+lower_surface_wireframe_parametric)

# upper_surface_wireframe = wing.evaluate(upper_surface_wireframe_parametric, plot=False)
# lower_surface_wireframe = wing.evaluate(lower_surface_wireframe_parametric)
# camber_surface = csdl.linear_combination(upper_surface_wireframe, lower_surface_wireframe, 1).reshape((num_chordwise, num_spanwise, 3))
camber_surface = camber_surface_mesh.evaluate(geometry)
# camber_surface = geometry.evaluate_representations([camber_surface_mesh], plot=True)
# geometry.plot_meshes([camber_surface])

# shell_mesh_parametric = wing.project(shell_mesh)

# endregion

# endregion

# region Parameterization

# region Create Parameterization Objects

num_ffd_sections = 3
num_wing_secctions = 2
# ffd_block = construct_ffd_block_from_corners(entities=geometry, corners=points, num_coefficients=(2, num_ffd_sections, 2), degree=(1,1,1))
# ffd_block = construct_ffd_block_around_entities(entities=wing, num_coefficients=(2, num_ffd_sections, 2), degree=(1,1,1))
wing_ffd_block = construct_tight_fit_ffd_block(entities=wing, num_coefficients=(2, (num_ffd_sections // num_wing_secctions + 1), 2), degree=(1,1,1))
# ffd_block = construct_tight_fit_ffd_block(entities=geometry, num_coefficients=(2, 3, 2), degree=(1,1,1))
# ffd_block.plot()

ffd_sectional_parameterization = VolumeSectionalParameterization(
    name="ffd_sectional_parameterization",
    parameterized_points=wing_ffd_block.coefficients,
    principal_parametric_dimension=1,
)
# ffd_sectional_parameterization.plot()

space_of_linear_3_dof_b_splines = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(3,))
space_of_linear_2_dof_b_splines = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))

chord_stretching_b_spline = lfs.Function(space=space_of_linear_3_dof_b_splines,
                                         coefficients=csdl.Variable(shape=(3,), value=np.array([-0.8, 3., -0.8])), name='chord_stretching_b_spline_coefficients')

wingspan_stretching_b_spline = lfs.Function(space=space_of_linear_2_dof_b_splines,
                                             coefficients=csdl.Variable(shape=(2,), value=np.array([-4., 4.])), name='wingspan_stretching_b_spline_coefficients')

sweep_translation_b_spline = lfs.Function(space=space_of_linear_3_dof_b_splines,
                                            coefficients=csdl.Variable(shape=(3,), value=np.array([4.0, 0.0, 4.0])), name='sweep_translation_b_spline_coefficients')
# sweep_translation_b_spline.plot()

twist_b_spline = lfs.Function(space=space_of_linear_3_dof_b_splines,
                                coefficients=csdl.Variable(shape=(3,), value=np.array([15, 0., 15])*np.pi/180), name='twist_b_spline_coefficients')

# endregion

# region Evaluate Parameterization To Define Parameterization Forward Model For Parameterization Solver
parametric_b_spline_inputs = np.linspace(0.0, 1.0, 7).reshape((-1, 1))
chord_stretch_sectional_parameters = chord_stretching_b_spline.evaluate(
    parametric_b_spline_inputs,
)
wingspan_stretch_sectional_parameters = wingspan_stretching_b_spline.evaluate(
    parametric_b_spline_inputs
)
sweep_translation_sectional_parameters = sweep_translation_b_spline.evaluate(
    parametric_b_spline_inputs
)

twist_sectional_parameters = twist_b_spline.evaluate(
    parametric_b_spline_inputs
)


sectional_parameters = VolumeSectionalParameterizationInputs()
sectional_parameters.add_sectional_stretch(axis=0, stretch=chord_stretch_sectional_parameters)
sectional_parameters.add_sectional_translation(axis=1, translation=wingspan_stretch_sectional_parameters)
sectional_parameters.add_sectional_translation(axis=0, translation=sweep_translation_sectional_parameters)
sectional_parameters.add_sectional_rotation(axis=1, rotation=twist_sectional_parameters)

ffd_coefficients = ffd_sectional_parameterization.evaluate(sectional_parameters, plot=False)

wing_coefficients = wing_ffd_block.evaluate(ffd_coefficients, plot=False)
wing.set_coefficients(wing_coefficients)



# vertical_stabilizer = geometry.declare_component(function_search_names=['Vertical Stabilizer'])
# vertical_stabilizer_rigid_body_translation = csdl.Variable(shape=(3,), value=np.array([0., 0., 0.]))
# translated_vertical_stabilizer = vertical_stabilizer.coefficients + vertical_stabilizer_rigid_body_translation
# vertical_stabilizer.set_coefficients(translated_vertical_stabilizer)

# vertical_stabilizer_fuselage_connection = ...


wingspan = csdl.norm(
    geometry.evaluate(left_tip_leading_edge_parametric) - geometry.evaluate(right_tip_leading_edge_parametric)
)
root_chord = csdl.norm(
    geometry.evaluate(center_leading_edge_parametric) - geometry.evaluate(center_trailing_edge_parametric)
)
tip_chord_left = csdl.norm(
    geometry.evaluate(left_tip_trailing_edge_parametric) - geometry.evaluate(left_tip_leading_edge_parametric)
)
tip_chord_right = csdl.norm(
    geometry.evaluate(right_tip_trailing_edge_parametric) - geometry.evaluate(right_tip_leading_edge_parametric)
)

spanwise_direction_left = geometry.evaluate(left_tip_leading_edge_parametric) - geometry.evaluate(center_leading_edge_parametric)
spanwise_direction_right = geometry.evaluate(right_tip_leading_edge_parametric) - geometry.evaluate(center_leading_edge_parametric)

sweep_angle_left = csdl.arccos(-spanwise_direction_left[1] / csdl.norm(spanwise_direction_left))
sweep_angle_right = csdl.arccos(spanwise_direction_right[1] / csdl.norm(spanwise_direction_right))

print("Wingspan: ", wingspan.value)
print("Root Chord: ", root_chord.value)
print("Tip Chord Left: ", tip_chord_left.value)
print("Tip Chord Right: ", tip_chord_right.value)
print("Sweep Angle Left: ", sweep_angle_left.value*180/np.pi)
print("Sweep Angle Right: ", sweep_angle_right.value*180/np.pi)

# Create Newton solver for inner optimization
chord_stretching_b_spline.coefficients.add_name('chord_stretching_b_spline_coefficients')
wingspan_stretching_b_spline.coefficients.add_name('wingspan_stretching_b_spline_coefficients')
sweep_translation_b_spline.coefficients.add_name('sweep_translation_b_spline_coefficients')


# objective = (csdl.vdot(chord_stretching_b_spline.coefficients, chord_stretching_b_spline.coefficients)
#             + csdl.vdot(wingspan_stretching_b_spline.coefficients, wingspan_stretching_b_spline.coefficients)
#             + csdl.vdot(sweep_translation_b_spline.coefficients, sweep_translation_b_spline.coefficients))

wingspan_outer_dv = csdl.Variable(shape=(1,), value=np.array([150.0]))
root_chord_outer_dv = csdl.Variable(shape=(1,), value=np.array([30.0]))
tip_chord_outer_dv = csdl.Variable(shape=(1,), value=np.array([0.5]))
sweep_angle_outer_dv = csdl.Variable(shape=(1,), value=np.array([45*np.pi/180]))


geometry_solver = lsdo_geo.ParameterizationSolver()
geometry_solver.add_parameter(chord_stretching_b_spline.coefficients)
geometry_solver.add_parameter(wingspan_stretching_b_spline.coefficients)
geometry_solver.add_parameter(sweep_translation_b_spline.coefficients)

# geometry_solver.add_parameter(vertical_stabilizer_rigid_body_translation)

geometric_variables = lsdo_geo.GeometricVariables()
geometric_variables.add_variable(wingspan, wingspan_outer_dv)
geometric_variables.add_variable(root_chord, root_chord_outer_dv)
geometric_variables.add_variable(tip_chord_left, tip_chord_outer_dv)
geometric_variables.add_variable(tip_chord_right, tip_chord_outer_dv)
geometric_variables.add_variable(sweep_angle_left, sweep_angle_outer_dv)
geometric_variables.add_variable(sweep_angle_right, sweep_angle_outer_dv)

# geometric_variables.add_variable(wing_jury_constraint, desired_value=wing_jury_constraint.value)
# geometric_variables.add_variable(vertical_stabilizer_fuselage_connection, vertical_stabilizer_fuselage_connection.value)

# geometry.plot()
geometry_solver.evaluate(geometric_variables)
# geometry.plot()

# wing.set_coefficients(wing.coefficients + control_point_deltas)

# upper_surface_wireframe = wing.evaluate(upper_surface_wireframe_parametric, plot=False)
# lower_surface_wireframe = wing.evaluate(lower_surface_wireframe_parametric)
# camber_surface = csdl.linear_combination(upper_surface_wireframe, lower_surface_wireframe, 1).reshape((num_chordwise, num_spanwise, 3))
camber_surface = camber_surface_mesh.evaluate(geometry)
# geometry.plot_meshes([camber_surface])

# shell_mesh = wing.evaluate(shell_mesh_parametric)

# rotation_axis = np.array([0., 0., 1.])
# rotation_origin = geometry.project(np.array([0.0, 0.0, 0.0]))
# rotation_angle = 45
# geometry.rotate(rotation_origin, rotation_axis, rotation_angle)


print("Wingspan: ", wingspan.value)
print("Root Chord: ", root_chord.value)
print("Tip Chord Left: ", tip_chord_left.value)
print("Tip Chord Right: ", tip_chord_right.value)
print("Sweep Angle Left: ", sweep_angle_left.value*180/np.pi)
print("Sweep Angle Right: ", sweep_angle_right.value*180/np.pi)
print("Chord Stretching: ", chord_stretching_b_spline.coefficients.value)
print("Wingspan Stretching: ", wingspan_stretching_b_spline.coefficients.value)
print("Sweep Translation: ", sweep_translation_b_spline.coefficients.value)

# recorder.visualize_graph('my_graph')
# csdl.save_all_variables()
# # csdl.inline_save('variables')
# recorder.save_graph('graph')


# Pretend analysis
# pretend_pressures = np.linspace(0, 1, 100)
# wing.functions[3].plot()

# pressure_space = lfs.BSplineSpace(
#     num_parametric_dimensions=3,
#     degree=(2,0,0),
#     # coefficients_shape=(5,3,3))
#     coefficients_shape=(5,1,1))
# pressures = np.random.rand(num_chordwise, num_spanwise)

# # pressure_space = wing.create_parallel_space(pressure_space)
# geometry_space = wing.space
# pressure_function = geometry_space.fit_function_set(pressures, parametric_coordinates=upper_surface_wireframe_parametric, regularization_parameter=1e-3)
# geometry.plot(color=pressure_function)
