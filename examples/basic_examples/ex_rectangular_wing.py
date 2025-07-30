# region Imports and Setup

import csdl_alpha as csdl
import numpy as np
import lsdo_function_spaces as lfs

from lsdo_geo.core.parameterization.free_form_deformation_functions import construct_ffd_block_around_entities
from lsdo_geo.core.parameterization.volume_sectional_parameterization import (
    VolumeSectionalParameterization,
    VolumeSectionalParameterizationInputs
)
from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver, GeometricVariables

import lsdo_geo

recorder = csdl.Recorder(inline=True)
recorder.start()

# Import initiail geometry that will be deformed
geometry = lsdo_geo.import_geometry(
    "examples/example_geometries/rectangular_wing.stp",
    parallelize=False,
)
# geometry.plot()
# endregion Imports

# region Key locations

# The following points are used to define the key locations of the geometry 
# that can be used to define meshes and/or design parameters. The inputs are numpy arrays
# with the initial locations in physical space. The output of the projection is the parametric 
# location of the point on the geometry. It is important to have the coordinates in parametric space
# because the parametric coordinates will not change as the geometry is deformed.

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
# region Wing Camber Surface (for VLM solver)

# Project (physical -> closest parametric locations on geometry) and evaluate (parametric -> physical) 
# to find the exact locations of the leading and trailing edges
num_spanwise = 11
num_chordwise = 4
points_to_project_on_leading_edge = np.linspace(np.array([0., -4., 1.]), np.array([0., 4., 1.]), num_spanwise)
points_to_project_on_trailing_edge = np.linspace(np.array([1., -4., 1.]), np.array([1., 4., 1.]), num_spanwise)
leading_edge_parametric = geometry.project(points_to_project_on_leading_edge, direction=np.array([0., 0., -1.]), plot=False)
leading_edge_physical = geometry.evaluate(leading_edge_parametric, plot=False)
trailing_edge_parametric = geometry.project(points_to_project_on_trailing_edge, direction=np.array([0., 0., -1.]), plot=False)
trailing_edge_physical = geometry.evaluate(trailing_edge_parametric)

# Compute the chord surface -> upper+lower surfaces to get the camber surface
# Note: This is not the only way to get the camber surface
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
# Construct a Free Form Deformation (FFD) block around the geometry
num_ffd_coefficients_chordwise = 8
num_ffd_sections = 3
# Note: This FFD block construction is one of a few helper functions that can be used to create a FFD block.
#       The "manual" method is to use construct_ffd_block_from_corners, which allows for defining the coefficients directly.
ffd_block = construct_ffd_block_around_entities(entities=geometry, 
                                                num_coefficients=(num_ffd_coefficients_chordwise, num_ffd_sections, 2), degree=(3,1,1))
# ffd_block.plot()

# Define an axial sectional parameterization for the FFD volume. 
# This views the FFD volume as a series of 2D sections (as defined by the control points) 
# that can be allowed to stretch, translate, and rotate independently.
# The sectional parameterization is chosen to have the spanwise direction as the principal 
# parametric dimension (0,1,2 corresponds to u,v,w of the FFD block, which in this case corresponds to x,y,z).
ffd_sectional_parameterization = VolumeSectionalParameterization(
    name="ffd_sectional_parameterization",
    parameterized_points=ffd_block.coefficients,
    principal_parametric_dimension=1,
)
# ffd_sectional_parameterization.plot()

# Although unnecessary for this example, this section defines B-spline functions that can be used to independently
# parameterize the sectional parameters (this method is commonly used, so it's included here for completeness).
# The coefficients will be used as the states of the parameterization solver, which will be manipulated to solve
# for the desired geometry (satisfies the design parameters and constraints). The initial values are mainly for
# debugging to see what the deformation modes do to the geometry since the solver will solve for the actual values.
space_of_linear_3_dof_b_splines = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(3,))
space_of_linear_2_dof_b_splines = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))

chord_stretching_b_spline = lfs.Function(space=space_of_linear_3_dof_b_splines,
                                         coefficients=csdl.Variable(shape=(3,), value=np.array([-0.8, 3., -0.8])), name='chord_stretching_b_spline_coefficients')

wingspan_stretching_b_spline = lfs.Function(space=space_of_linear_2_dof_b_splines,
                                             coefficients=csdl.Variable(shape=(2,), value=np.array([-20., 20.])), name='wingspan_stretching_b_spline_coefficients')

sweep_translation_b_spline = lfs.Function(space=space_of_linear_3_dof_b_splines,
                                            coefficients=csdl.Variable(shape=(3,), value=np.array([14., 0., 14.])), name='sweep_translation_b_spline_coefficients')
# sweep_translation_b_spline.plot()

twist_b_spline = lfs.Function(space=space_of_linear_3_dof_b_splines,
                                coefficients=csdl.Variable(shape=(3,), value=np.array([0., 0., 0.])*np.pi/180), name='twist_b_spline_coefficients')

# endregion Create Parameterization Objects

# region Evaluate Inner Parameterization Map To Define Forward Model For Parameterization Solver
# Evaluate the B-splines to get the sectional parameters
parametric_b_spline_inputs = np.linspace(0.0, 1.0, num_ffd_sections).reshape((-1, 1))
chord_stretch_sectional_parameters = chord_stretching_b_spline.evaluate(parametric_b_spline_inputs)
wingspan_stretch_sectional_parameters = wingspan_stretching_b_spline.evaluate(parametric_b_spline_inputs)
sweep_translation_sectional_parameters = sweep_translation_b_spline.evaluate(parametric_b_spline_inputs)
twist_sectional_parameters = twist_b_spline.evaluate(parametric_b_spline_inputs)

# Evaluate the sectional parameterization to get the FFD coefficients
sectional_parameters = VolumeSectionalParameterizationInputs()
sectional_parameters.add_sectional_stretch(axis=0, stretch=chord_stretch_sectional_parameters)
sectional_parameters.add_sectional_translation(axis=1, translation=wingspan_stretch_sectional_parameters)
sectional_parameters.add_sectional_translation(axis=0, translation=sweep_translation_sectional_parameters)
sectional_parameters.add_sectional_rotation(axis=1, rotation=twist_sectional_parameters)
ffd_coefficients = ffd_sectional_parameterization.evaluate(sectional_parameters, plot=False)

# Evaluate the FFD and set the coefficients of the geometry
geometry_coefficients = ffd_block.evaluate_ffd(coefficients=ffd_coefficients, plot=False)
geometry.set_coefficients(geometry_coefficients) # type: ignore
# geometry.plot()

# Apply any other geometry transformations, such as rotation
rotation_axis = np.array([0., 1., 0.])
rotation_origin = geometry.evaluate(geometry.project(np.array([0.0, 0.0, 0.0])))
rotation_angle = 15
geometry.rotate(rotation_origin, rotation_axis, rotation_angle, units='degrees')

# Define the design parameters as a function of the geometry (which is now a function of the parameterization states)
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
# Define design variables for the optimizer (for the solver, these are desired values)
wingspan_outer_dv = csdl.Variable(shape=(1,), value=np.array([1.0]))
root_chord_outer_dv = csdl.Variable(shape=(1,), value=np.array([2.0]))
tip_chord_outer_dv = csdl.Variable(shape=(1,), value=np.array([0.5]))
sweep_angle_outer_dv = csdl.Variable(shape=(1,), value=np.array([30*np.pi/180]))

geometry_solver = ParameterizationSolver()

# Define the states for the parameterization solver (solver will manipulate these to achieve the variables)
geometry_solver.add_state(chord_stretching_b_spline.coefficients)
geometry_solver.add_state(wingspan_stretching_b_spline.coefficients)
geometry_solver.add_state(sweep_translation_b_spline.coefficients)

# Define the geometric variables/constraints that the solver will enforce.
geometric_variables = GeometricVariables()
geometric_variables.add_variable(wingspan, wingspan_outer_dv, penalty_value=None)
geometric_variables.add_variable(root_chord, root_chord_outer_dv, penalty_value=None)
geometric_variables.add_variable(tip_chord_left, tip_chord_outer_dv, penalty_value=None)
geometric_variables.add_variable(tip_chord_right, tip_chord_outer_dv, penalty_value=None)
geometric_variables.add_variable(sweep_angle_left, sweep_angle_outer_dv, penalty_value=None)
geometric_variables.add_variable(sweep_angle_right, sweep_angle_outer_dv, penalty_value=None)

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
