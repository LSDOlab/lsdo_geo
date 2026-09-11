# region Imports and Setup

import csdl_alpha as csdl
import numpy as np
import lsdo_function_spaces as lfs

import lsdo_geo as lg
from lsdo_geo import (
    Geometry,
    construct_ffd_block_around_entities,
    SectionalParameterization,
    SectionalParameters,
    ParameterizationSolver,
    GeometricVariables,
    import_geometry,
)
import VortexAD
import meshio
import pickle

recorder = csdl.Recorder(inline=True)
recorder.start()

# Import initial geometry that will be deformed
geometry_directory = "examples/example_geometries/"
file_name = "rectangular_wing_naca0012_10ar"
geometry = import_geometry(geometry_directory + file_name + ".stp", parallelize=False)
# geometry.plot()

# endregion Imports

# region Key locations

# The following points are used to define the key locations of the geometry 
# that can be used to define meshes and/or design parameters. The inputs are numpy arrays
# with the initial locations in physical space. The output of the projection is the parametric 
# location of the point on the geometry. It is important to have the coordinates in parametric space
# because the parametric coordinates will not change as the geometry is deformed.

# for i, function in geometry.functions.items():
#     print(f"Function {i}:")
#     function.plot()

leading_edge_left = geometry.project(np.array([0.0, -5.0, 0.0]))
leading_edge_right = geometry.project(np.array([0.0, 5.0, 0.0]))
trailing_edge_left = geometry.project(np.array([1.0, -5.0, 0.0]))
trailing_edge_right = geometry.project(np.array([1.0, 5.0, 0.0]))
leading_edge_center = geometry.project(np.array([0.0, 0.0, 0.0]))
trailing_edge_center = geometry.project(np.array([1.0, 0.0, 0.0]))
quarter_chord_left = geometry.project(np.array([0.25, -5.0, 0.0]))
quarter_chord_right = geometry.project(np.array([0.25, 5.0, 0.0]))
quarter_chord_center = geometry.project(np.array([0.25, 0.0, 0.0]))
# endregion

# region Mesh definitions
mesh = meshio.read(geometry_directory + file_name + ".msh")

points_orig = mesh.points
cells = mesh.cells
cells_dict = mesh.cells_dict
cell_adjacency_data = VortexAD.find_cell_adjacency(points=points_orig, cells=cells_dict)

points_orig = cell_adjacency_data[0] 
cells_dict = cell_adjacency_data[1] 
cell_adjacency = cell_adjacency_data[2] 
edges2cells = cell_adjacency_data[3]
points2cells = cell_adjacency_data[4]

TE_properties = VortexAD.TE_detection(points=points_orig,
                             cells=cells_dict,
                             edges2cells=edges2cells,
                             points2cells=points2cells,
                             threshold_theta=125.,
                             )

upper_TE_cells = TE_properties[0] 
lower_TE_cells = TE_properties[1] 
TE_edges = TE_properties[2] 
TE_node_indices = TE_properties[3]

cell_types = cells_dict.keys()
combined_cells = []
for cell_type in cell_types:
    combined_cells += cells_dict[cell_type].tolist()

projected_panel_mesh = geometry.project(points_orig, 
                                        grid_search_density_parameter=1, 
                                        newton_tolerance=1.e-10, 
                                        grid_search_density_cutoff=30,
                                        projection_tolerance=1.e-3,
                                        force_reprojection=False, 
                                        plot=False
                                        )

# project panel centers
cell_types = cells_dict.keys()
combined_cells = []
for cell_type in cell_types:
    combined_cells += cells_dict[cell_type].tolist()

panel_centers = np.zeros((len(combined_cells), 3))
for i, cell in enumerate(combined_cells):
    panel_centers[i] = np.mean(points_orig[cell], axis=0)

panel_centers = geometry.project(panel_centers, 
                            grid_search_density_parameter=1,
                            newton_tolerance=1.e-10,
                            grid_search_density_cutoff=30,
                            projection_tolerance=1.e-2,
                            force_reprojection=False, 
                            plot=False,
                            )

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
ffd_sectional_parameterization = SectionalParameterization(
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
                                         coefficients=csdl.Variable(shape=(3,), value=np.array([0., 0., 0.])), name='chord_stretching_b_spline_coefficients')

wingspan_stretching_b_spline = lfs.Function(space=space_of_linear_2_dof_b_splines,
                                             coefficients=csdl.Variable(shape=(2,), value=np.array([-0., 0.])), name='wingspan_stretching_b_spline_coefficients')

sweep_translation_b_spline = lfs.Function(space=space_of_linear_3_dof_b_splines,
                                            coefficients=csdl.Variable(shape=(3,), value=np.array([0., 0., 0.])), name='sweep_translation_b_spline_coefficients')
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
sectional_parameters = SectionalParameters()
sectional_parameters.add_stretch(axis=0, stretch=chord_stretch_sectional_parameters)
sectional_parameters.add_translation(axis=1, translation=wingspan_stretch_sectional_parameters)
sectional_parameters.add_translation(axis=0, translation=sweep_translation_sectional_parameters)
sectional_parameters.add_rotation(axis=1, rotation=twist_sectional_parameters)
ffd_coefficients = ffd_sectional_parameterization.evaluate(sectional_parameters, plot=False)

# Evaluate the FFD and set the coefficients of the geometry
geometry_coefficients = ffd_block.evaluate_ffd(coefficients=ffd_coefficients, plot=False)
geometry.set_coefficients(geometry_coefficients) # type: ignore
# geometry.plot()

# Define the design parameters as a function of the geometry (which is now a function of the parameterization states)
wingspan = geometry.evaluate(leading_edge_right)[1] - geometry.evaluate(leading_edge_left)[1] # type: ignore
root_chord = geometry.evaluate(trailing_edge_center)[0] - geometry.evaluate(leading_edge_center)[0] # type: ignore
tip_chord_left = geometry.evaluate(trailing_edge_left)[0] - geometry.evaluate(leading_edge_left)[0] # type: ignore
tip_chord_right = geometry.evaluate(trailing_edge_right)[0] - geometry.evaluate(leading_edge_right)[0] # type: ignore

spanwise_direction_left = geometry.evaluate(quarter_chord_left) - geometry.evaluate(quarter_chord_center)
spanwise_direction_right = geometry.evaluate(quarter_chord_right) - geometry.evaluate(quarter_chord_center)
sweep_angle_left = csdl.arctan(-spanwise_direction_left[0] / spanwise_direction_left[1]) # type: ignore
sweep_angle_right = csdl.arctan(spanwise_direction_right[0] / spanwise_direction_right[1]) # type: ignore
# endregion Evaluate Parameterization To Define Parameterization Forward Model For Parameterization Solver

# region Set Up and Evaluate Geometry Parameterization Solver
# Define design variables for the optimizer (for the solver, these are desired values)
wingspan_outer_dv = csdl.Variable(shape=(1,), value=np.array([10.0]))
root_chord_outer_dv = csdl.Variable(shape=(1,), value=np.array([1.0]))
tip_chord_outer_dv = csdl.Variable(shape=(1,), value=np.array([1.0]))
sweep_angle_outer_dv = csdl.Variable(shape=(1,), value=np.array([0.*np.pi/180]))

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
# geometric_variables.add_variable(sweep_angle_left, sweep_angle_outer_dv, penalty_value=None)
# geometric_variables.add_variable(sweep_angle_right, sweep_angle_outer_dv, penalty_value=None)
geometric_variables.add_variable(-spanwise_direction_left[0], csdl.tan(sweep_angle_outer_dv) * spanwise_direction_left[1], penalty_value=None)
geometric_variables.add_variable(spanwise_direction_right[0], csdl.tan(sweep_angle_outer_dv) * spanwise_direction_right[1], penalty_value=None)

print("Wingspan: ", wingspan.value) # type: ignore
print("Root Chord: ", root_chord.value) # type: ignore
print("Tip Chord Left: ", tip_chord_left.value) # type: ignore
print("Tip Chord Right: ", tip_chord_right.value) # type: ignore
print("Sweep Angle Left: ", sweep_angle_left.value*180/np.pi) # type: ignore
print("Sweep Angle Right: ", sweep_angle_right.value*180/np.pi) # type: ignore

# geometry.plot()
geometry_solver.evaluate(geometric_variables)
# geometry.plot()

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

pitch = csdl.Variable(value=0.*np.pi/180) # pitch angle in radians
geometry.rotate(rotation_origin=geometry.evaluate(quarter_chord_center), axis_vector=np.array([0., 1., 0.]), angles=pitch, units='radians')


recorder.inline = False

cruise_speed = csdl.Variable(value=1.)
velocity = csdl.concatenate([cruise_speed, csdl.Variable(value=0.), csdl.Variable(value=0.)])

# region Aerodynamic solver (panel method)
num_nodes = 1

panel_mesh = geometry.evaluate(projected_panel_mesh, plot=False)
panel_mesh = panel_mesh.expand((1,) + panel_mesh.shape, 'ij->aij')

point_velocities = csdl.expand(velocity, (num_nodes,) + panel_mesh.shape[1:], 'j->iaj')
rho_array = csdl.Variable(shape=(num_nodes,), value=np.array([1.225]))
sos_array = csdl.Variable(shape=(num_nodes,), value=np.array([343.0]))

planform_area = (root_chord_outer_dv + tip_chord_outer_dv) / 2 * wingspan_outer_dv

pm_solver_inputs = {
    'V_inf': -point_velocities,
    'rho': rho_array,
    'sos': sos_array,
    'compressibility': True,
    'Cp cutoff': -5.,
    'partition_size': 1,
    'reuse_AIC': True,
    # 'mesh_path': file_path+file_name, # already done externally
    'ref_area': planform_area # does not matter bc we don't use the coefficients,
}
# we leave out the mesh path because we need FFD to move the mesh

panel_method = VortexAD.PanelMethod(
    solver_input_dict=pm_solver_inputs,
    skip_geometry=True # not running geometry
)
# inserting grid data from above
panel_method.insert_grid_data(
    mesh=panel_mesh[0,:],
    cell_adjacency_data=cell_adjacency_data,
    TE_properties=TE_properties
)

panel_method.declare_outputs([
    'Cp',
    'L',
    'Di',
    'M',
    'panel_forces',
    'CL',
    'CDi',
    'CM',
])
outputs = panel_method.evaluate()


CL = outputs['CL']
CDi = outputs['CDi']
Cp = outputs['Cp']

# endregion Aerodynamic solver (panel method)

# region Structural solver (beam model)
# TODO: I will come back to this

# endregion Structural solver (beam model)


# TODO: Define design variables, constraints, and objective function for optimization problem (look at geometric optimization example)
# objective should be CDi
# constraint that CL = 0.5
# Variables should include wingspan_outer_dv, root_chord_outer_dv, tip_chord_outer_dv, sweep_angle_outer_dv, and pitch
# -- Be sure to include variable bounds, particularly for span and pitch



jax_sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=[wingspan_outer_dv, root_chord_outer_dv, tip_chord_outer_dv, sweep_angle_outer_dv, pitch],
    additional_outputs=[CL, CDi, Cp, panel_mesh],
    gpu=False
)

# Run and plot panel method solution to make sure simulation is working
# -- This does not need to be run for optimization, but is useful for debugging
jax_sim.run()
print("CL: ", CL.value)
print("CDi: ", CDi.value)
panel_method.points_orig = panel_mesh.value
panel_method.plot(Cp.value, bounds=[-0.5,1])
# region Optimization
# TODO: Define optimization problem and run optimization (look at ModOpt docs or geometric optimization example)

# endregion Optimization

