# region Imports and Setup

from dataclasses import dataclass
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
import aframe
import modopt
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
                             threshold_theta=125.
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

# Project line of beam nodes along the span at 50% chord
num_beam_nodes = 11
y_beam_span = np.linspace(-4.99, 4.99, num_beam_nodes)
beam_line_physical = np.zeros((num_beam_nodes, 3))
beam_line_physical[:, 0] = 0.50  # 50% chord
beam_line_physical[:, 1] = y_beam_span
beam_line_physical[:, 2] = 0.0

projected_beam_mesh = geometry.project(
    beam_line_physical,
    grid_search_density_parameter=1,
    newton_tolerance=1.e-10,
    grid_search_density_cutoff=30,
    projection_tolerance=1.e-2,
    force_reprojection=False,
    plot=False
)

# Project leading edge and trailing edge lines along the span for local chord computation
le_line_physical = np.zeros((num_beam_nodes, 3))
le_line_physical[:, 0] = 0.00  # Leading edge (0% chord)
le_line_physical[:, 1] = y_beam_span
le_line_physical[:, 2] = 0.0

projected_le_mesh = geometry.project(
    le_line_physical,
    grid_search_density_parameter=1,
    newton_tolerance=1.e-10,
    grid_search_density_cutoff=30,
    projection_tolerance=1.e-2,
    force_reprojection=False,
    plot=False
)

te_line_physical = np.zeros((num_beam_nodes, 3))
te_line_physical[:, 0] = 1.00  # Trailing edge (100% chord)
te_line_physical[:, 1] = y_beam_span
te_line_physical[:, 2] = 0.0

projected_te_mesh = geometry.project(
    te_line_physical,
    grid_search_density_parameter=1,
    newton_tolerance=1.e-10,
    grid_search_density_cutoff=30,
    projection_tolerance=1.e-2,
    force_reprojection=False,
    plot=False
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

# pitch = csdl.Variable(value=0.*np.pi/180) # pitch angle in radians
pitch = csdl.Variable(value=5.*np.pi/180) # pitch angle in radians
geometry.rotate(rotation_origin=geometry.evaluate(quarter_chord_center), axis_vector=np.array([0., 1., 0.]), angles=pitch, units='radians')


recorder.inline = False

cruise_speed = csdl.Variable(value=20.)
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
L = outputs['L']
Di = outputs['Di']
Cp = outputs['Cp']

# endregion Aerodynamic solver (panel method)

# region Structural solver (beam model)
beam_mesh = geometry.evaluate(projected_beam_mesh, plot=False)
le_mesh = geometry.evaluate(projected_le_mesh, plot=False)
te_mesh = geometry.evaluate(projected_te_mesh, plot=False)

# Compute local chord from projected mesh (leading edge to trailing edge distance)
node_chords = csdl.norm(te_mesh - le_mesh, axes=(1,))
local_chord = 0.5 * (node_chords[:-1] + node_chords[1:])

# Define wingbox cross-section along the span (20% to 80% of chord -> width = 0.60 * local_chord)
box_width = 0.60 * local_chord
box_height = 0.10 * local_chord

ttop = csdl.Variable(value=np.ones(num_beam_nodes - 1) * 0.005)
tweb = csdl.Variable(value=np.ones(num_beam_nodes - 1) * 0.005)

beam_cs = aframe.CSBox(
    height=box_height,
    width=box_width,
    ttop=ttop,
    tbot=ttop,
    tweb=tweb,
)

# Beam material (Aluminum: E = 69 GPa, G = 26 GPa, density = 2700 kg/m^3)
beam = aframe.Beam(name='wing_spar', mesh=beam_mesh, E=69e9, G=26e9, density=2700, cs=beam_cs)

# Fix center node (wing root boundary condition)
center_node_idx = num_beam_nodes // 2
beam.fix(node=center_node_idx)

# Distribute aerodynamic lift force across spanwise beam nodes (elliptical distribution)
y_norm = np.linspace(-1.0, 1.0, num_beam_nodes)
elliptic_weights = np.sqrt(np.maximum(0.0, 1.0 - y_norm**2))
elliptic_weights = elliptic_weights / np.sum(elliptic_weights)

beam_loads_matrix = []
for n in range(num_beam_nodes):
    fz_n = L[0] * elliptic_weights[n]
    node_load = csdl.concatenate([
        csdl.Variable(value=np.array([0.0])),
        csdl.Variable(value=np.array([0.0])),
        fz_n,
        csdl.Variable(value=np.array([0.0])),
        csdl.Variable(value=np.array([0.0])),
        csdl.Variable(value=np.array([0.0])),
    ])
    beam_loads_matrix.append(node_load)

beam_loads = csdl.reshape(csdl.concatenate(beam_loads_matrix), (num_beam_nodes, 6))
beam.add_load(beam_loads)

# Solve structural beam model using aframe
frame = aframe.Frame(beams=[beam])
frame.solve()

beam_displacement = frame.displacement['wing_spar']
structural_mass = beam.mass

# Structural constraint: Limit maximum tip deflection (z-displacement)
tip_deflection = csdl.norm(beam_displacement[0, 0:3])
# tip_deflection.set_as_constraint(upper=0.3)

# endregion Structural solver (beam model)


# Define design variables, constraints, and objective for optimization problem
# Objective: Minimize Aerodynamic Drag (Di)
objective = Di
# objective = CDi
objective.set_as_objective()

# Lift = Weight constraint (Weight = 365 N)
weight_N = 135.0
L.set_as_constraint(equals=weight_N)

# CL.set_as_constraint(equals=0.5)

@dataclass
class DVInfo:
    variable: csdl.Variable
    lower: float
    upper: float
    scaler: float = 1.0

design_variables: dict[str, DVInfo] = {
    'wingspan': DVInfo(variable=wingspan_outer_dv, lower=1.0, upper=30.0, scaler=1.e-1),
    'root_chord': DVInfo(variable=root_chord_outer_dv, lower=0.1, upper=5.0),
    'tip_chord': DVInfo(variable=tip_chord_outer_dv, lower=0.1, upper=5.0),
    'sweep_angle': DVInfo(variable=sweep_angle_outer_dv, lower=-30.0*np.pi/180, upper=45.0*np.pi/180),
    'pitch': DVInfo(variable=pitch, lower=-10.0*np.pi/180, upper=15.0*np.pi/180),
    # 'ttop': DVInfo(variable=ttop, lower=0.001, upper=0.02, scaler=100.0),
    # 'tweb': DVInfo(variable=tweb, lower=0.001, upper=0.02, scaler=100.0),
}

for dv_info in design_variables.values():
    dv_info.variable.set_as_design_variable(lower=dv_info.lower, upper=dv_info.upper, scaler=dv_info.scaler)

geometry_coefficients = [geometry_function.coefficients for geometry_function in geometry.functions.values()]

jax_sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=[dv_info.variable for dv_info in design_variables.values()],
    additional_outputs=[CL, CDi, L, Di, Cp, panel_mesh, structural_mass, beam_displacement] + geometry_coefficients,
    gpu=False
)

# # region Run Model
# jax_sim.run()
# print(f"Drag: {jax_sim[Di]}")
# print(f"Lift: {jax_sim[L]}")
# exit()
# # endregion Run Model


# region Optimization
optimization_problem = modopt.CSDLAlphaProblem(problem_name='rectangular_wing_aerostructural_optimization', simulator=jax_sim)
optimizer = modopt.PySLSQP(optimization_problem, solver_options={'maxiter': 1000, 'acc': 1.e-10}, readable_outputs=['x'])
optimizer.solve()
optimizer.print_results()

# endregion Optimization


# region Plot Optimization History
import pyvista as pv
import os, glob

# Find the latest output folder
output_base_dir = 'rectangular_wing_aerostructural_optimization_outputs'
output_folders = sorted(glob.glob(os.path.join(output_base_dir, '*')))
latest_folder = output_folders[-1]
print(f"Reading optimization history from: {latest_folder}")

# Read design variable history from x.out (preferred) or record.hdf5
x_out_path = os.path.join(latest_folder, 'x.out')
if os.path.exists(x_out_path):
    x_history = np.loadtxt(x_out_path)
    print(f"Loaded {x_history.shape[0]} iterations from x.out")
else:
    import h5py
    hdf5_path = os.path.join(latest_folder, 'record.hdf5')
    with h5py.File(hdf5_path, 'r') as f:
        obj_callbacks = sorted(
            [k for k in f.keys() if k.startswith('callback_') and 'obj' in f[k]['outputs']],
            key=lambda k: int(k.split('_')[1])
        )
        x_history_list = []
        for cb in obj_callbacks:
            x = f[cb]['inputs']['x'][:]
            x_history_list.append(x)
        # Deduplicate consecutive identical x vectors
        unique_x = [x_history_list[0]]
        for i in range(1, len(x_history_list)):
            if not np.allclose(x_history_list[i], x_history_list[i-1]):
                unique_x.append(x_history_list[i])
        x_history = np.array(unique_x)
    print(f"Loaded {x_history.shape[0]} unique iterations from record.hdf5")

num_iterations = x_history.shape[0]

# Set up pyvista offscreen rendering and video
pv.OFF_SCREEN = True
video_path = os.path.join(latest_folder, 'optimization_history.mp4')
plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])

plotter.open_movie(video_path, framerate=4)

camera = {
    'position': (-20.0, -15.0, 10.0),
    'focal_point': (0.0, 0.0, 0.0),
    'viewup': (0, 0, 1),
}

# Calculate index slices for design variables
dv_slices = {}
curr_idx = 0
for name, dv_info in design_variables.items():
    var_size = dv_info.variable.shape[0] if len(dv_info.variable.shape) > 0 else 1
    dv_slices[name] = slice(curr_idx, curr_idx + var_size)
    curr_idx += var_size

for iteration in range(num_iterations):
    x_scaled = x_history[iteration]

    # Undo scaling for each design variable to set physical (unscaled) values on jax_sim
    unscaled_values = {}
    for name, dv_info in design_variables.items():
        slc = dv_slices[name]
        unscaled_val = x_scaled[slc] / dv_info.scaler
        jax_sim[dv_info.variable] = unscaled_val
        unscaled_values[name] = unscaled_val

    # Run the simulator to update geometry coefficients
    jax_sim.run()

    # Get plotting elements from geometry.plot (returns list of pyvista objects)
    plotting_elements = geometry.plot(show=False)

    # Clear previous frame and add new geometry
    plotter.clear()

    # Add each plotting element to the plotter
    for element in plotting_elements:
        if isinstance(element, dict) and 'mesh' in element:
            mesh = element['mesh']
            kwargs = element.get('kwargs', {})
            plotter.add_mesh(mesh, **kwargs)
        elif isinstance(element, tuple) and len(element) == 2:
            mesh, kwargs = element
            plotter.add_mesh(mesh, **kwargs)
        elif isinstance(element, pv.Actor):
            plotter.add_actor(element)
        elif isinstance(element, pv.DataSet):
            plotter.add_mesh(element)

    span_val = float(unscaled_values['wingspan'].item() if hasattr(unscaled_values['wingspan'], 'item') else unscaled_values['wingspan'][0])
    root_val = float(unscaled_values['root_chord'].item() if hasattr(unscaled_values['root_chord'], 'item') else unscaled_values['root_chord'][0])
    tip_val = float(unscaled_values['tip_chord'].item() if hasattr(unscaled_values['tip_chord'], 'item') else unscaled_values['tip_chord'][0])
    sweep_val = float(unscaled_values['sweep_angle'].item() if hasattr(unscaled_values['sweep_angle'], 'item') else unscaled_values['sweep_angle'][0])
    pitch_val = float(unscaled_values['pitch'].item() if hasattr(unscaled_values['pitch'], 'item') else unscaled_values['pitch'][0])
    drag_val = float(Di.value.item() if hasattr(Di.value, 'item') else Di.value[0])
    lift_val = float(L.value.item() if hasattr(L.value, 'item') else L.value[0])
    mass_val = float(structural_mass.value.item() if hasattr(structural_mass.value, 'item') else structural_mass.value[0])

    # Add iteration counter label using unscaled physical values
    plotter.add_text(
        f"Iteration {iteration}/{num_iterations - 1}\n"
        f"Span={span_val:.2f} m  Root={root_val:.3f} m  Tip={tip_val:.3f} m\n"
        f"Sweep={np.degrees(sweep_val):.1f}°  Pitch={np.degrees(pitch_val):.1f}°\n"
        f"Lift={lift_val:.1f} N  Drag={drag_val:.3f} N  Struct Mass={mass_val:.2f} kg",
        position='upper_left',
        font_size=12,
        color='white',
        shadow=True,
    )

    # Set camera
    plotter.camera.position = camera['position']
    plotter.camera.focal_point = camera['focal_point']
    plotter.camera.up = camera['viewup']
    plotter.set_background('black')

    plotter.write_frame()
    print(f"  Frame {iteration}/{num_iterations - 1} written")

plotter.close()
print(f"Video saved to: {video_path}")

# endregion Plot Optimization History