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


import lsdo_geo

'''
TODO:
1. Set up inner optimization class
2. Ruff Ruff

'''

recorder = csdl.Recorder(inline=True)
recorder.start()

geometry = lsdo_geo.import_geometry(
    "examples/example_geometries/rectangular_wing.stp",
    # "examples/example_geometries/simple_wing.stp",
    parallelize=False,
)
geometry.plot()

# dummy_basis_matrix1 = geometry.functions[3].space.compute_basis_matrix(np.array([0., 0.5]))
# dummy_basis_matrix2 = geometry.functions[9].space.compute_basis_matrix(np.array([0., 0.5]))
# # geometry.functions[9].coefficients.value[:,:,:] = np.flip(geometry.functions[9].coefficients.value[:,:,:], axis=1)
# plotting_elements = geometry.functions[3].plot(show=False, point_types=['coefficients'], plot_types=['wireframe'])
# plotting_elements = geometry.functions[9].plot(show=False, point_types=['coefficients'], plot_types=['wireframe'], additional_plotting_elements=plotting_elements)
# plotting_elements = geometry.functions[3].plot(show=False, point_types=['evaluated_points'], plot_types=['point_cloud'], additional_plotting_elements=plotting_elements)
# geometry.functions[9].plot(point_types=['evaluated_points'], additional_plotting_elements=plotting_elements, plot_types=['point_cloud'])
# # geometry.refit(parallelize=False) # New API if you want to do this!
# # geometry.plot()


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
upper_surface_wireframe_parametric = geometry.project(chord_surface + np.array([0., 0., 1]), direction=np.array([0., 0., -1.]),
                                                      grid_search_density_parameter=10, plot=True)
lower_surface_wireframe_parametric = geometry.project(chord_surface - np.array([0., 0., 1]), direction=np.array([0., 0., -1.]),
                                                      grid_search_density_parameter=10, plot=True)
upper_surface_wireframe = geometry.evaluate(upper_surface_wireframe_parametric, plot=False)
lower_surface_wireframe = geometry.evaluate(lower_surface_wireframe_parametric, plot=False)
camber_surface = csdl.linear_combination(upper_surface_wireframe, lower_surface_wireframe, 1).reshape((num_chordwise, num_spanwise, 3))
geometry.plot_meshes([camber_surface])
# endregion

# endregion

# region Parameterization

# region Create Parameterization Objects

num_ffd_sections = 3
num_wing_secctions = 2
# ffd_block = construct_ffd_block_around_entities(entities=geometry, num_coefficients=(2, num_ffd_sections, 2), degree=(1,1,1))
ffd_block = construct_tight_fit_ffd_block(entities=geometry, num_coefficients=(2, (num_ffd_sections // num_wing_secctions + 1), 2), degree=(1,1,1))
# ffd_block = construct_tight_fit_ffd_block(entities=geometry, num_coefficients=(2, 3, 2), degree=(1,1,1))
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
                                         coefficients=csdl.ImplicitVariable(shape=(3,), value=np.array([-0.8, 3., -0.8])), name='chord_stretching_b_spline_coefficients')

wingspan_stretching_b_spline = lfs.Function(space=space_of_linear_2_dof_b_splines,
                                             coefficients=csdl.ImplicitVariable(shape=(2,), value=np.array([-4., 4.])), name='wingspan_stretching_b_spline_coefficients')

sweep_translation_b_spline = lfs.Function(space=space_of_linear_3_dof_b_splines,
                                            coefficients=csdl.ImplicitVariable(shape=(3,), value=np.array([4.0, 0.0, 4.0])), name='sweep_translation_b_spline_coefficients')
# sweep_translation_b_spline.plot()

twist_b_spline = lfs.Function(space=space_of_linear_3_dof_b_splines,
                                coefficients=csdl.Variable(shape=(3,), value=np.array([15, 0., 15])*np.pi/180), name='twist_b_spline_coefficients')

# endregion

# region Evaluate Parameterization To Define Parameterization Forward Model For Parameterization Solver
parametric_b_spline_inputs = np.linspace(0.0, 1.0, num_ffd_sections).reshape((-1, 1))
chord_stretch_sectional_parameters = chord_stretching_b_spline.evaluate(
    parametric_b_spline_inputs
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

ffd_coefficients = ffd_sectional_parameterization.evaluate(sectional_parameters, plot=False)    # TODO: Fix plot function
ffd_coefficients.name = 'ffd_coefficients'
# ffd_coefficients._save = True

geometry_coefficients = ffd_block.evaluate(ffd_coefficients, plot=False)
print(geometry_coefficients)
test = geometry_coefficients[4]
test._save = True
geometry.set_coefficients(geometry_coefficients)
# geometry.plot()


wingspan = csdl.norm(
    geometry.evaluate(leading_edge_right) - geometry.evaluate(leading_edge_left)
)
root_chord = csdl.norm(
    geometry.evaluate(trailing_edge_center) - geometry.evaluate(leading_edge_center)
)
tip_chord_left = csdl.norm(
    geometry.evaluate(trailing_edge_left) - geometry.evaluate(leading_edge_left)
)
tip_chord_right = csdl.norm(
    geometry.evaluate(trailing_edge_right) - geometry.evaluate(leading_edge_right)
)

spanwise_direction_left = geometry.evaluate(quarter_chord_left) - geometry.evaluate(quarter_chord_center)
spanwise_direction_right = geometry.evaluate(quarter_chord_right) - geometry.evaluate(quarter_chord_center)
# sweep_angle = csdl.arccos(csdl.vdot(spanwise_direction, np.array([0., -1., 0.])) / csdl.norm(spanwise_direction))
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

wingspan_outer_dv = csdl.Variable(shape=(1,), value=np.array([6.0]))
root_chord_outer_dv = csdl.Variable(shape=(1,), value=np.array([2.0]))
tip_chord_outer_dv = csdl.Variable(shape=(1,), value=np.array([0.5]))
sweep_angle_outer_dv = csdl.Variable(shape=(1,), value=np.array([45*np.pi/180]))

# wingspan_constraint = wingspan - wingspan_outer_dv
# wingspan_constraint.add_name('wingspan_constraint')
# root_chord_constraint = root_chord - root_chord_outer_dv
# root_chord_constraint.add_name('root_chord_constraint')
# tip_chord_left_constraint = tip_chord_left - tip_chord_outer_dv
# tip_chord_left_constraint.add_name('tip_chord_left_constraint')
# tip_chord_right_constraint = tip_chord_right - tip_chord_outer_dv
# tip_chord_right_constraint.add_name('tip_chord_right_constraint')
# sweep_angle_left_constraint = sweep_angle_left - sweep_angle_outer_dv
# sweep_angle_left_constraint.add_name('sweep_angle_left_constraint')
# sweep_angle_right_constraint = sweep_angle_right - sweep_angle_outer_dv
# sweep_angle_right_constraint.add_name('sweep_angle_right_constraint')

# geometry.plot()

# from lsdo_geo.csdl.optimization import Optimization, NewtonOptimizer
# geometry_optimization = Optimization()
# geometry_optimization.add_objective(objective)
# geometry_optimization.add_design_variable(chord_stretching_b_spline.coefficients)
# geometry_optimization.add_design_variable(wingspan_stretching_b_spline.coefficients)
# geometry_optimization.add_design_variable(sweep_translation_b_spline.coefficients)
# geometry_optimization.add_constraint(wingspan_constraint)
# geometry_optimization.add_constraint(root_chord_constraint)
# geometry_optimization.add_constraint(tip_chord_left_constraint)
# geometry_optimization.add_constraint(tip_chord_right_constraint)
# geometry_optimization.add_constraint(sweep_angle_left_constraint)
# geometry_optimization.add_constraint(sweep_angle_right_constraint)

# geometry_optimizer = NewtonOptimizer()
# geometry_optimizer.add_optimization(geometry_optimization)
# geometry_optimizer.run()


from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver, GeometricVariables
geometry_solver = ParameterizationSolver()
geometry_solver.add_parameter(chord_stretching_b_spline.coefficients)
geometry_solver.add_parameter(wingspan_stretching_b_spline.coefficients)
geometry_solver.add_parameter(sweep_translation_b_spline.coefficients)

geometric_variables = GeometricVariables()
geometric_variables.add_variable(wingspan, wingspan_outer_dv)
geometric_variables.add_variable(root_chord, root_chord_outer_dv)
geometric_variables.add_variable(tip_chord_left, tip_chord_outer_dv)
geometric_variables.add_variable(tip_chord_right, tip_chord_outer_dv)
geometric_variables.add_variable(sweep_angle_left, sweep_angle_outer_dv)
geometric_variables.add_variable(sweep_angle_right, sweep_angle_outer_dv)


print("Wingspan: ", wingspan.value)
print("Root Chord: ", root_chord.value)
print("Tip Chord Left: ", tip_chord_left.value)
print("Tip Chord Right: ", tip_chord_right.value)
print("Sweep Angle Left: ", sweep_angle_left.value*180/np.pi)
print("Sweep Angle Right: ", sweep_angle_right.value*180/np.pi)
print("Chord Stretching: ", chord_stretching_b_spline.coefficients.value)
print("Wingspan Stretching: ", wingspan_stretching_b_spline.coefficients.value)
print("Sweep Translation: ", sweep_translation_b_spline.coefficients.value)

geometry.plot()
geometry_solver.evaluate(geometric_variables)
geometry.plot()


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