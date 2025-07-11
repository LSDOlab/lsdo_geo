# region Imports and Setup

import csdl_alpha as csdl
import numpy as np
import lsdo_function_spaces as lfs

from lsdo_geo.core.parameterization.free_form_deformation_functions import (
    construct_tight_fit_ffd_block,construct_ffd_block_around_entities
)
from lsdo_geo.core.parameterization.volume_sectional_parameterization import (
    VolumeSectionalParameterization,
    VolumeSectionalParameterizationInputs
)
from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver, GeometricVariables

import lsdo_geo

recorder = csdl.Recorder(inline=True)
recorder.start()

geometry = lsdo_geo.import_geometry(
    "examples/example_geometries/rectangular_wing.stp",
    parallelize=False,
)
# geometry.plot()

new_function_space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=(3,3), coefficients_shape=(10,30))
geometry = geometry.refit(new_function_space)
geometry = lsdo_geo.Geometry(functions=geometry.functions, function_names=geometry.function_names, name=geometry.name, space=geometry.space)
# geometry.plot()
# endregion Imports

# region Key locations
leading_edge_left = geometry.project(np.array([0.0, -4.0, 0.0]))
leading_edge_right = geometry.project(np.array([0.0, 4.0, 0.0]))
trailing_edge_left = geometry.project(np.array([1.0, -4.0, 0.0]))
trailing_edge_right = geometry.project(np.array([1.0, 4.0, 0.0]))
leading_edge_center = geometry.project(np.array([0.0, 0.0, 0.0]))
trailing_edge_center = geometry.project(np.array([1.0, 0.0, 0.0]))
quarter_chord_left = geometry.project(np.array([0.25, -4.0, 0.0]))
quarter_chord_right = geometry.project(np.array([0.25, 4.0, 0.0]))
quarter_chord_center = geometry.project(np.array([0.25, 0.0, 0.0]))
# endregion

# region Mesh definitions
# region Wing Camber Surface
num_spanwise = 11
num_chordwise = 4
points_to_project_on_leading_edge = np.linspace(np.array([0., -4., 1.]), np.array([0., 4., 1.]), num_spanwise)
points_to_project_on_trailing_edge = np.linspace(np.array([1., -4., 1.]), np.array([1., 4., 1.]), num_spanwise)

leading_edge_parametric = geometry.project(points_to_project_on_leading_edge, direction=np.array([0., 0., -1.]), plot=False)
leading_edge_physical = geometry.evaluate(leading_edge_parametric, plot=False)
trailing_edge_parametric = geometry.project(points_to_project_on_trailing_edge, direction=np.array([0., 0., -1.]), plot=False)
trailing_edge_physical = geometry.evaluate(trailing_edge_parametric)

chord_surface = csdl.linear_combination(leading_edge_physical, trailing_edge_physical, num_chordwise).value.reshape((num_chordwise, num_spanwise, 3))
upper_surface_wireframe_parametric = geometry.project(chord_surface + np.array([0., 0., 0.1]), direction=np.array([0., 0., -1.]),
                                                      grid_search_density_parameter=10, plot=False)
lower_surface_wireframe_parametric = geometry.project(chord_surface - np.array([0., 0., 0.1]), direction=np.array([0., 0., -1.]),
                                                      grid_search_density_parameter=10, plot=False)
upper_surface_wireframe = geometry.evaluate(upper_surface_wireframe_parametric, plot=False)
lower_surface_wireframe = geometry.evaluate(lower_surface_wireframe_parametric, plot=False)
camber_surface = csdl.linear_combination(upper_surface_wireframe, lower_surface_wireframe, 1).reshape((num_chordwise, num_spanwise, 3))
# geometry.plot_meshes([camber_surface])
# endregion

# endregion

# region Create Parameterization Objects
num_ffd_coefficients_chordwise = 8
num_ffd_sections = 3
ffd_block = construct_ffd_block_around_entities(entities=geometry, 
                                                num_coefficients=(num_ffd_coefficients_chordwise, num_ffd_sections, 2), degree=(3,1,1))
# ffd_block.plot()

ffd_sectional_parameterization = VolumeSectionalParameterization(
    name="ffd_sectional_parameterization",
    parameterized_points=ffd_block.coefficients,
    principal_parametric_dimension=1,
)
# ffd_sectional_parameterization.plot()

space_of_linear_3_dof_b_splines = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(3,))
space_of_linear_2_dof_b_splines = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))

chord_stretching_b_spline = lfs.Function(space=space_of_linear_3_dof_b_splines,
                                         coefficients=csdl.ImplicitVariable(shape=(3,), value=np.array([0., 0., -0.])), name='chord_stretching_b_spline_coefficients')

wingspan_stretching_b_spline = lfs.Function(space=space_of_linear_2_dof_b_splines,
                                             coefficients=csdl.ImplicitVariable(shape=(2,), value=np.array([0., 0.])), name='wingspan_stretching_b_spline_coefficients')

sweep_translation_b_spline = lfs.Function(space=space_of_linear_3_dof_b_splines,
                                            coefficients=csdl.ImplicitVariable(shape=(3,), value=np.array([0., 0., 0.])), name='sweep_translation_b_spline_coefficients')
# sweep_translation_b_spline.plot()

twist_b_spline = lfs.Function(space=space_of_linear_3_dof_b_splines,
                                coefficients=csdl.Variable(shape=(3,), value=np.array([0., 0., 0.])*np.pi/180), name='twist_b_spline_coefficients')

# endregion Create Parameterization Objects

# region Evaluate Inner Parameterization Map To Define Forward Model For Parameterization Solver
parametric_b_spline_inputs = np.linspace(0.0, 1.0, num_ffd_sections).reshape((-1, 1))
chord_stretch_sectional_parameters = chord_stretching_b_spline.evaluate(parametric_b_spline_inputs)
wingspan_stretch_sectional_parameters = wingspan_stretching_b_spline.evaluate(parametric_b_spline_inputs)
sweep_translation_sectional_parameters = sweep_translation_b_spline.evaluate(parametric_b_spline_inputs)
twist_sectional_parameters = twist_b_spline.evaluate(parametric_b_spline_inputs)

sectional_parameters = VolumeSectionalParameterizationInputs()
sectional_parameters.add_sectional_stretch(axis=0, stretch=chord_stretch_sectional_parameters)
sectional_parameters.add_sectional_translation(axis=1, translation=wingspan_stretch_sectional_parameters)
sectional_parameters.add_sectional_translation(axis=0, translation=sweep_translation_sectional_parameters)
sectional_parameters.add_sectional_rotation(axis=1, rotation=twist_sectional_parameters)

ffd_coefficients = ffd_sectional_parameterization.evaluate(sectional_parameters, plot=False)

# Apply shape variables
original_block_thickness = ffd_block.coefficients.value[0, 0, 1, 2] - ffd_block.coefficients.value[0, 0, 0, 2]

percent_change_in_thickness = csdl.Variable(shape=(num_ffd_coefficients_chordwise,num_ffd_sections), value=0.)
percent_change_in_thickness_dof = csdl.Variable(shape=(num_ffd_coefficients_chordwise, num_ffd_sections//2+1),
                                                    value=np.array([[0., 0.], [0., 0.], [0., 0.],
                                                                    [0., 0.], [0., 0.], [0., 0.],
                                                                    [0., 0.], [0., 0.]]))
percent_change_in_thickness = percent_change_in_thickness.set(csdl.slice[:,:num_ffd_sections//2+1], percent_change_in_thickness_dof)
percent_change_in_thickness = percent_change_in_thickness.set(csdl.slice[:,num_ffd_sections//2+1:], percent_change_in_thickness_dof[:,-2::-1])
delta_block_thickness = (percent_change_in_thickness / 100) * original_block_thickness
thickness_upper_translation = 1/2 * delta_block_thickness
thickness_lower_translation = -thickness_upper_translation
ffd_coefficients = ffd_coefficients.set(csdl.slice[:,:,1,2], ffd_coefficients[:,:,1,2] + thickness_upper_translation)
ffd_coefficients = ffd_coefficients.set(csdl.slice[:,:,0,2], ffd_coefficients[:,:,0,2] + thickness_lower_translation)

# Parameterize camber change as normalized by the original block (kind of like chord) length
normalized_percent_camber_change = csdl.Variable(shape=(num_ffd_coefficients_chordwise,num_ffd_sections),
                                            value=0.)
normalized_percent_camber_change_dof = csdl.Variable(shape=(num_ffd_coefficients_chordwise-2, num_ffd_sections//2+1),
                                                       value=np.array([[0., 0.], [0., 0.],
                                                                       [0., 0.], [0., 0.],
                                                                       [0., 0.], [0., 0.]]))
normalized_percent_camber_change = normalized_percent_camber_change.set(csdl.slice[1:-1,:num_ffd_sections//2+1],
                                                                         normalized_percent_camber_change_dof)
normalized_percent_camber_change = normalized_percent_camber_change.set(csdl.slice[1:-1,num_ffd_sections//2+1:], 
                                                                        normalized_percent_camber_change_dof[:,-2::-1])
block_length = ffd_block.coefficients.value[1, 0, 0, 0] - ffd_block.coefficients.value[0, 0, 0, 0]
camber_change = (normalized_percent_camber_change / 100) * block_length
ffd_coefficients = ffd_coefficients.set(csdl.slice[:,:,:,2], 
                                        ffd_coefficients[:,:,:,2] + 
                                        csdl.expand(camber_change, (num_ffd_coefficients_chordwise, num_ffd_sections, 2), 'ij->ijk'))

geometry_coefficients = ffd_block.evaluate_ffd(coefficients=ffd_coefficients, plot=False)
geometry.set_coefficients(geometry_coefficients) # type: ignore
geometry.plot()

rotation_axis = np.array([0., 1., 0.])
rotation_origin = geometry.evaluate(geometry.project(np.array([0.0, 0.0, 0.0])))
rotation_angle = 15
geometry.rotate(rotation_origin, rotation_axis, rotation_angle, units='degrees')

wingspan = csdl.norm(geometry.evaluate(leading_edge_right) - geometry.evaluate(leading_edge_left)) # type: ignore
root_chord = csdl.norm(geometry.evaluate(trailing_edge_center) - geometry.evaluate(leading_edge_center)) # type: ignore
tip_chord_left = csdl.norm(geometry.evaluate(trailing_edge_left) - geometry.evaluate(leading_edge_left)) # type: ignore
tip_chord_right = csdl.norm(geometry.evaluate(trailing_edge_right) - geometry.evaluate(leading_edge_right)) # type: ignore

spanwise_direction_left = geometry.evaluate(quarter_chord_left) - geometry.evaluate(quarter_chord_center)
spanwise_direction_right = geometry.evaluate(quarter_chord_right) - geometry.evaluate(quarter_chord_center)
sweep_angle_left = csdl.arctan(-spanwise_direction_left[0] / spanwise_direction_left[1]) # type: ignore
sweep_angle_right = csdl.arctan(spanwise_direction_right[0] / spanwise_direction_right[1]) # type: ignore
# endregion Evaluate Parameterization To Define Parameterization Forward Model For Parameterization Solver

# region Set Up and Evaluate Geometry Parameterization Solver
wingspan_outer_dv = csdl.Variable(shape=(1,), value=np.array([1.0]))
root_chord_outer_dv = csdl.Variable(shape=(1,), value=np.array([2.0]))
tip_chord_outer_dv = csdl.Variable(shape=(1,), value=np.array([0.5]))
sweep_angle_outer_dv = csdl.Variable(shape=(1,), value=np.array([30*np.pi/180]))

tan_sweep_left = -spanwise_direction_left[0]/spanwise_direction_left[1]
tan_sweep_right = spanwise_direction_right[0]/spanwise_direction_right[1]

geometry_solver = ParameterizationSolver()
geometry_solver.add_state(chord_stretching_b_spline.coefficients)
geometry_solver.add_state(wingspan_stretching_b_spline.coefficients)
geometry_solver.add_state(sweep_translation_b_spline.coefficients)

geometric_variables = GeometricVariables()
geometric_variables.add_variable(wingspan, wingspan_outer_dv, penalty_value=None)
geometric_variables.add_variable(root_chord, root_chord_outer_dv, penalty_value=None)
geometric_variables.add_variable(tip_chord_left, tip_chord_outer_dv, penalty_value=None)
geometric_variables.add_variable(tip_chord_right, tip_chord_outer_dv, penalty_value=None)
# geometric_variables.add_variable(sweep_angle_left, sweep_angle_outer_dv, penalty_value=None)
# geometric_variables.add_variable(sweep_angle_right, sweep_angle_outer_dv, penalty_value=None)
geometric_variables.add_variable(tan_sweep_left, csdl.tan(sweep_angle_outer_dv), penalty_value=None)
geometric_variables.add_variable(tan_sweep_right, csdl.tan(sweep_angle_outer_dv), penalty_value=None)

print("Wingspan: ", wingspan.value) # type: ignore
print("Root Chord: ", root_chord.value) # type: ignore
print("Tip Chord Left: ", tip_chord_left.value) # type: ignore
print("Tip Chord Right: ", tip_chord_right.value) # type: ignore
print("Sweep Angle Left: ", sweep_angle_left.value*180/np.pi) # type: ignore
print("Sweep Angle Right: ", sweep_angle_right.value*180/np.pi) # type: ignore

# geometry.plot()
geometry_solver.evaluate(geometric_variables)
geometry.plot()

print()
print("Wingspan: ", wingspan.value) # type: ignore
print("Root Chord: ", root_chord.value) # type: ignore
print("Tip Chord Left: ", tip_chord_left.value) # type: ignore
print("Tip Chord Right: ", tip_chord_right.value) # type: ignore
print("Sweep Angle Left: ", sweep_angle_left.value*180/np.pi) # type: ignore
print("Sweep Angle Right: ", sweep_angle_right.value*180/np.pi) # type: ignore
print("Chord Stretching: ", chord_stretching_b_spline.coefficients.value) # type: ignore
print("Wingspan Stretching: ", wingspan_stretching_b_spline.coefficients.value) # type: ignore
print("Sweep Translation: ", sweep_translation_b_spline.coefficients.value) # type: ignore
# endregion Setup and Evaluate Geometry Parameterization Solver

# recorder.visualize_graph('my_graph')
# csdl.save_all_variables()
# # csdl.inline_save('variables')
# recorder.save_graph('graph')

import os
import pickle
from pathlib import Path
 
file_path = f"stored_files/deformed_geometries/my_deformed_geometry.pickle"
 
Path("stored_files/deformed_geometries").mkdir(parents=True, exist_ok=True)
with open(file_path, 'wb+') as handle:
    geometry_copy = geometry.copy()
    for i, function in geometry_copy.functions.items():
        function_copy = function.copy()
        function_copy.coefficients = function.coefficients.value.copy()
        geometry_copy.functions[i] = function_copy
    pickle.dump(geometry_copy, handle, protocol=pickle.HIGHEST_PROTOCOL)

# b_spline_set = lfs.FunctionSet(b_spline_list, name='imported_geometry')
# fn = os.path.basename(file_name)
# fn_wo_ext = fn[:fn.rindex('.')]
# file_path = f"stored_files/imports/{fn_wo_ext}_stored_import.pickle"

# Path("stored_files/imports").mkdir(parents=True, exist_ok=True)
# with open(file_path, 'wb+') as handle:
#     b_spline_set_copy = b_spline_set.copy()
#     for i, function in b_spline_set.functions.items():
#         function_copy = function.copy()
#         function_copy.coefficients = function.coefficients.value.copy()
#         b_spline_set_copy.functions[i] = function_copy

#     pickle.dump(b_spline_set_copy, handle, protocol=pickle.HIGHEST_PROTOCOL)