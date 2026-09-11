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
import modopt
import meshio
import pickle

recorder = csdl.Recorder(inline=True)
recorder.start()

# Import initial geometry that will be deformed
geometry_directory = "examples/example_geometries/"
file_name = "rectangular_wing_naca0012_10ar"
geometry = import_geometry(geometry_directory + file_name + ".stp", parallelize=False)

# Add more spanwise control points to each geometry function via linear interpolation.
# Since this is a rectangular wing (uniform cross-section), linearly interpolating
# control points along the spanwise axis is exact and preserves the chordwise resolution.
# Each function's spanwise axis is detected independently since B-splines may be oriented differently.
num_spanwise_cp_target = 15  # more than the 9 FFD sections

for idx in list(geometry.functions.keys()):
    function = geometry.functions[idx]
    coeffs = function.coefficients.value if hasattr(function.coefficients, 'value') else function.coefficients

    # Determine which axis is spanwise by checking y-coordinate range along each axis
    axis0_y_range = np.ptp([coeffs[i, :, 1].mean() for i in range(coeffs.shape[0])])
    axis1_y_range = np.ptp([coeffs[:, j, 1].mean() for j in range(coeffs.shape[1])])

    # Skip tip caps and other non-spanwise surfaces (y-extent < 1.0)
    if max(axis0_y_range, axis1_y_range) < 1.0:
        continue

    spanwise_axis = 0 if axis0_y_range >= axis1_y_range else 1

    orig_degree = function.space.degree

    if spanwise_axis == 0:
        n_chord = coeffs.shape[1]
        new_coeffs = np.zeros((num_spanwise_cp_target, n_chord, 3))
        for j in range(n_chord):
            for k in range(3):
                new_coeffs[:, j, k] = np.linspace(coeffs[0, j, k], coeffs[-1, j, k], num_spanwise_cp_target)
        new_shape = (num_spanwise_cp_target, n_chord)
        new_degree = (min(2, num_spanwise_cp_target - 1), orig_degree[1])
    else:
        n_chord = coeffs.shape[0]
        new_coeffs = np.zeros((n_chord, num_spanwise_cp_target, 3))
        for j in range(n_chord):
            for k in range(3):
                new_coeffs[j, :, k] = np.linspace(coeffs[j, 0, k], coeffs[j, -1, k], num_spanwise_cp_target)
        new_shape = (n_chord, num_spanwise_cp_target)
        new_degree = (orig_degree[0], min(2, num_spanwise_cp_target - 1))

    new_space = lfs.BSplineSpace(
        num_parametric_dimensions=2,
        degree=new_degree,
        coefficients_shape=new_shape,
    )
    geometry.functions[idx] = lfs.Function(space=new_space, coefficients=new_coeffs)

geometry = lsdo_geo.Geometry(functions=geometry.functions, function_names=geometry.function_names,
                             name=geometry.name, space=geometry.space)
# geometry.plot()
# geometry.plot(point_types=['coefficients'], plot_types=['point_cloud'])
# exit()
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

# Project chord measurement points at 5 evenly spaced stations across the right half-span
# Stations: y = 0.0 (root), 1.25, 2.5, 3.75, 5.0 (tip)
num_chord_stations = 5
chord_station_y = np.linspace(0.0, 5.0, num_chord_stations)
chord_le_projections = []
chord_te_projections = []
for y_val in chord_station_y:
    chord_le_projections.append(geometry.project(np.array([0.0, y_val, 0.0])))
    chord_te_projections.append(geometry.project(np.array([1.0, y_val, 0.0])))

# Also project symmetric (left half) chord measurement points for the solver
chord_le_projections_left = []
chord_te_projections_left = []
for y_val in chord_station_y[1:]:  # skip root (index 0) since it's already at center
    chord_le_projections_left.append(geometry.project(np.array([0.0, -y_val, 0.0])))
    chord_te_projections_left.append(geometry.project(np.array([1.0, -y_val, 0.0])))

# Project wireframes along upper and lower wing skins to compute chord surface for planform area
nx_area = 21  # chordwise grid resolution
ny_area = 41  # spanwise grid resolution
x_grid = np.linspace(0.0, 1.0, nx_area)
y_grid = np.linspace(-5.0, 5.0, ny_area)
X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid, indexing='ij')

upper_seed_pts = np.column_stack([X_mesh.ravel(), Y_mesh.ravel(), np.full(X_mesh.size, 0.05)])
lower_seed_pts = np.column_stack([X_mesh.ravel(), Y_mesh.ravel(), np.full(X_mesh.size, -0.05)])

projected_upper_skin = geometry.project(upper_seed_pts, force_reprojection=False, direction=np.array([0, 0, -1]), plot=False)
projected_lower_skin = geometry.project(lower_seed_pts, force_reprojection=False, direction=np.array([0, 0, 1]), plot=False)
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

# endregion

# endregion

# region Create Parameterization Objects
# Construct a Free Form Deformation (FFD) block around the geometry
num_ffd_coefficients_chordwise = 2
num_ffd_sections = 9
# Note: This FFD block construction is one of a few helper functions that can be used to create a FFD block.
#       The "manual" method is to use construct_ffd_block_from_corners, which allows for defining the coefficients directly.
ffd_block = construct_ffd_block_around_entities(entities=geometry, 
                                                num_coefficients=(num_ffd_coefficients_chordwise, num_ffd_sections, 2), degree=(1,3,1))
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

# region Define Design Variables and CSDL Parameterization Map
# Flag to alternate between two formulations:
# True  -> Target physical chord design variables using ParameterizationSolver
# False -> Explicit sectional chord stretch design variables directly
use_parameterization_solver = False

wingspan_outer_dv = csdl.Variable(shape=(1,), value=np.array([10.0]))
sweep_angle_outer_dv = csdl.Variable(shape=(1,), value=np.array([0.*np.pi/180]))

space_of_linear_9_dof_b_splines = lfs.BSplineSpace(num_parametric_dimensions=1, degree=2, coefficients_shape=(9,))
space_of_linear_2_dof_b_splines = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))

if use_parameterization_solver:
    # 5 target physical chord design variables: index 0 = root/midline, index 4 = right tip
    chord_outer_dvs = csdl.Variable(shape=(num_chord_stations,), value=np.ones(num_chord_stations))
    
    chord_stretching_b_spline = lfs.Function(
        space=space_of_linear_9_dof_b_splines,
        coefficients=csdl.Variable(shape=(9,), value=np.zeros(9)),
        name='chord_stretching_b_spline_coefficients'
    )
else:
    # 5 explicit chord stretch design variables across half-span (0 = root, 4 = right tip)
    # stretch = 0.0 corresponds to baseline unit chord (1.0 m)
    chord_stretch_dvs = csdl.Variable(shape=(num_chord_stations,), value=np.zeros(num_chord_stations))
    
    # Symmetrically construct 9 B-spline control point coefficients across full span:
    # Index 0: Left Tip (station 4), 1: Left 75%, 2: Left 50%, 3: Left 25%, 4: Root (station 0), 5..8: Right span
    b_spline_coeffs = csdl.concatenate([
        chord_stretch_dvs[4], chord_stretch_dvs[3], chord_stretch_dvs[2], chord_stretch_dvs[1],
        chord_stretch_dvs[0],
        chord_stretch_dvs[1], chord_stretch_dvs[2], chord_stretch_dvs[3], chord_stretch_dvs[4]
    ])
    
    chord_stretching_b_spline = lfs.Function(
        space=space_of_linear_9_dof_b_splines,
        coefficients=b_spline_coeffs,
        name='chord_stretching_b_spline_coefficients'
    )

wingspan_stretching_b_spline = lfs.Function(space=space_of_linear_2_dof_b_splines,
                                             coefficients=csdl.Variable(shape=(2,), value=np.array([-0., 0.])), name='wingspan_stretching_b_spline_coefficients')

sweep_translation_b_spline = lfs.Function(space=space_of_linear_9_dof_b_splines,
                                            coefficients=csdl.Variable(shape=(9,), value=np.zeros(9)), name='sweep_translation_b_spline_coefficients')

twist_b_spline = lfs.Function(space=space_of_linear_9_dof_b_splines,
                                coefficients=csdl.Variable(shape=(9,), value=np.zeros(9)), name='twist_b_spline_coefficients')

# endregion Create Parameterization Objects

# region Evaluate Parameterization Map To Define Geometry Deformation
# Evaluate the B-splines to get the sectional parameters
parametric_b_spline_inputs = np.linspace(0.0, 1.0, num_ffd_sections).reshape((-1, 1))
chord_stretch_sectional_parameters = chord_stretching_b_spline.evaluate(parametric_b_spline_inputs)
wingspan_stretch_sectional_parameters = wingspan_stretching_b_spline.evaluate(parametric_b_spline_inputs)
sweep_translation_sectional_parameters = sweep_translation_b_spline.evaluate(parametric_b_spline_inputs)
twist_sectional_parameters = twist_b_spline.evaluate(parametric_b_spline_inputs)

# Evaluate the sectional parameterization to get the FFD coefficients
sectional_parameters = SectionalParameters()
sectional_parameters.add_stretch(axis=0, stretch=chord_stretch_sectional_parameters)
# sectional_parameters.add_translation(axis=1, translation=wingspan_stretch_sectional_parameters)
# sectional_parameters.add_translation(axis=0, translation=sweep_translation_sectional_parameters)
# sectional_parameters.add_rotation(axis=1, rotation=twist_sectional_parameters)
ffd_coefficients = ffd_sectional_parameterization.evaluate(sectional_parameters, plot=False)

# Evaluate the FFD and set the coefficients of the geometry
geometry_coefficients = ffd_block.evaluate_ffd(coefficients=ffd_coefficients, plot=False)
geometry.set_coefficients(geometry_coefficients) # type: ignore

# Define the design parameters as a function of the geometry
wingspan = geometry.evaluate(leading_edge_right)[1] - geometry.evaluate(leading_edge_left)[1] # type: ignore

# Compute chord at each of the 5 right half-span stations from projected LE/TE points
chord_right = []  # right half-span chords (root to tip)
for i in range(num_chord_stations):
    chord_i = geometry.evaluate(chord_te_projections[i])[0] - geometry.evaluate(chord_le_projections[i])[0]  # type: ignore
    chord_right.append(chord_i)

# Compute chord at each of the 4 left half-span stations (symmetric, skip root)
chord_left = []  # left half-span chords (25% to tip)
for i in range(num_chord_stations - 1):
    chord_i = geometry.evaluate(chord_te_projections_left[i])[0] - geometry.evaluate(chord_le_projections_left[i])[0]  # type: ignore
    chord_left.append(chord_i)

spanwise_direction_left = geometry.evaluate(quarter_chord_left) - geometry.evaluate(quarter_chord_center)
spanwise_direction_right = geometry.evaluate(quarter_chord_right) - geometry.evaluate(quarter_chord_center)
sweep_angle_left = csdl.arctan(-spanwise_direction_left[0] / spanwise_direction_left[1]) # type: ignore
sweep_angle_right = csdl.arctan(spanwise_direction_right[0] / spanwise_direction_right[1]) # type: ignore

# region Set Up and Evaluate Geometry Parameterization Solver
if use_parameterization_solver:
    geometry_solver = ParameterizationSolver()
    geometry_solver.add_state(chord_stretching_b_spline.coefficients)

    geometric_variables = GeometricVariables()

    for i in range(num_chord_stations):
        geometric_variables.add_variable(chord_right[i], chord_outer_dvs[i], penalty_value=None)

    for i in range(num_chord_stations - 1):
        geometric_variables.add_variable(chord_left[i], chord_outer_dvs[i + 1], penalty_value=None)

    geometry_solver.evaluate(geometric_variables)
# endregion Parameterization Solver

pitch = csdl.Variable(value=5.*np.pi/180) # pitch angle in radians
geometry.rotate(rotation_origin=geometry.evaluate(quarter_chord_center), axis_vector=np.array([0., 1., 0.]), angles=pitch, units='radians')



cruise_speed = csdl.Variable(value=1.)
# cruise_speed = csdl.Variable(value=20.)
velocity = csdl.concatenate([cruise_speed, csdl.Variable(value=0.), csdl.Variable(value=0.)])

# region Aerodynamic solver (panel method)
num_nodes = 1

panel_mesh = geometry.evaluate(projected_panel_mesh, plot=False)
panel_mesh = panel_mesh.expand((1,) + panel_mesh.shape, 'ij->aij')

point_velocities = csdl.expand(velocity, (num_nodes,) + panel_mesh.shape[1:], 'j->iaj')
rho_array = csdl.Variable(shape=(num_nodes,), value=np.array([1.225]))
sos_array = csdl.Variable(shape=(num_nodes,), value=np.array([343.0]))

# Planform area computed from upper and lower skin wireframes:
# 1. Evaluate matching wireframes on upper and lower wing skins to get the chord surface
upper_skin_pts = geometry.evaluate(projected_upper_skin, plot=False)
lower_skin_pts = geometry.evaluate(projected_lower_skin, plot=False)
chord_surface_pts = 0.5 * (upper_skin_pts + lower_skin_pts)

# 2. Reshape into structured 3D grid (nx_area, ny_area, 3)
chord_surface_grid = csdl.reshape(chord_surface_pts, (nx_area, ny_area, 3))

# 3. Compute element edge vectors along chordwise (x) and spanwise (y) directions
v_x = chord_surface_grid[1:, :-1, :] - chord_surface_grid[:-1, :-1, :]
v_y = chord_surface_grid[:-1, 1:, :] - chord_surface_grid[:-1, :-1, :]

# 4. Cross product of element edge vectors (v_x x v_y)
area_vectors = csdl.cross(v_x, v_y, axis=2)

# 5. Sum of the norm of the cross product of each quad element
element_areas = csdl.norm(area_vectors, axes=(2,))
planform_area = csdl.sum(element_areas)
print("Planform area:", planform_area.value)

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
    'CDi_Trefftz',
    'CM',
])

recorder.inline = False

outputs = panel_method.evaluate()


CL = outputs['CL']
CDi = outputs['CDi_Trefftz']
L = outputs['L']
Di = outputs['Di']
Cp = outputs['Cp']

# endregion Aerodynamic solver (panel method)

# region Structural solver (beam model)
# TODO: I will come back to this

# endregion Structural solver (beam model)


# Define design variables, constraints, and objective for optimization problem
objective = CDi
objective.set_as_objective(scaler=1.e4)

CL.set_as_constraint(equals=0.5, scaler=1.e1)
planform_area.set_as_constraint(equals=10.0, scaler=1.0)

@dataclass
class DVInfo:
    variable: csdl.Variable
    lower: float
    upper: float
    scaler: float = 1.0

if use_parameterization_solver:
    design_variables: dict[str, DVInfo] = {
        'chord_dvs': DVInfo(variable=chord_outer_dvs, lower=0.15, upper=5.0),
        'pitch': DVInfo(variable=pitch, lower=-10.0*np.pi/180, upper=15.0*np.pi/180, scaler=1.e1),
    }
else:
    design_variables: dict[str, DVInfo] = {
        'chord_stretch_dvs': DVInfo(variable=chord_stretch_dvs, lower=-0.85, upper=4.0),
        'pitch': DVInfo(variable=pitch, lower=-10.0*np.pi/180, upper=15.0*np.pi/180, scaler=1.e1),
    }

for dv_info in design_variables.values():
    dv_info.variable.set_as_design_variable(lower=dv_info.lower, upper=dv_info.upper, scaler=dv_info.scaler)

geometry_coefficients = [geometry_function.coefficients for geometry_function in geometry.functions.values()]

jax_sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=[dv_info.variable for dv_info in design_variables.values()],
    additional_outputs=[Di, L, CL, CDi, Cp, panel_mesh, planform_area] + geometry_coefficients,
    gpu=False
)

# Run and plot panel method solution to make sure simulation is working
# -- This does not need to be run for optimization, but is useful for debugging
# jax_sim.run()
# print(f"Lift: {jax_sim[L]}")
# print(f"Drag: {jax_sim[Di]}")
# exit()
# panel_method.points_orig = panel_mesh.value
# panel_method.plot(Cp.value, bounds=[-0.5,1])

# region Optimization
optimization_problem = modopt.CSDLAlphaProblem(problem_name='rectangular_wing_panel_optimization', simulator=jax_sim)
# optimizer = modopt.IPOPT(optimization_problem, recording=True)
optimizer = modopt.PySLSQP(optimization_problem, solver_options={'maxiter': 1000, 'acc': 1.e-7}, readable_outputs=['x'])
optimizer.solve()
optimizer.print_results()

# endregion Optimization


# region Plot Optimization History
import pyvista as pv
import os, glob

# Find the latest output folder
output_base_dir = 'rectangular_wing_panel_optimization_outputs'
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
    x_history_list = []
    if os.path.exists(hdf5_path):
        with h5py.File(hdf5_path, 'r') as f:
            valid_keys = [k for k in f.keys() if k.isdigit() or (k.startswith('callback_') and k.split('_')[1].isdigit())]
            cbs = sorted(valid_keys, key=lambda k: int(k.split('_')[1]) if '_' in k else int(k))
            for cb in cbs:
                if 'inputs' in f[cb]:
                    inp_grp = f[cb]['inputs']
                    if 'pitch' in inp_grp:
                        pitch_val = inp_grp['pitch'][:]
                        if 'chord_dvs' in inp_grp:
                            chords = inp_grp['chord_dvs'][:]
                            x_vec = np.concatenate([chords, pitch_val])
                            x_history_list.append(x_vec)
                        elif 'chord_stretch_dvs' in inp_grp:
                            stretches = inp_grp['chord_stretch_dvs'][:]
                            x_vec = np.concatenate([stretches, pitch_val])
                            x_history_list.append(x_vec)
                    elif 'x' in inp_grp:
                        x_history_list.append(inp_grp['x'][:])
            
            if len(x_history_list) > 0:
                unique_x = [x_history_list[0]]
                for i in range(1, len(x_history_list)):
                    if not np.allclose(x_history_list[i], x_history_list[i-1]):
                        unique_x.append(x_history_list[i])
                x_history = np.array(unique_x)
                print(f"Loaded {x_history.shape[0]} unique iterations from record.hdf5")
            else:
                x_history = np.array([])
    else:
        x_history = np.array([])

num_iterations = x_history.shape[0]
if num_iterations == 0:
    print(f"No optimization history found in {latest_folder}. Skipping post-processing plot rendering.")
    exit()

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

cd_history = []
cl_history = []
sref_history = []
wing_img_path = os.path.join(latest_folder, 'final_wing.png')

for iteration in range(num_iterations):
    x_scaled = x_history[iteration]

    # Undo scaling for each design variable to set physical (unscaled) values on jax_sim
    # Use slicing to handle vector-valued design variables
    unscaled_values = {}
    curr_idx = 0
    for name, dv_info in design_variables.items():
        var_size = dv_info.variable.shape[0] if len(dv_info.variable.shape) > 0 else 1
        slc = slice(curr_idx, curr_idx + var_size)
        unscaled_val = x_scaled[slc] / dv_info.scaler
        jax_sim[dv_info.variable] = unscaled_val
        unscaled_values[name] = unscaled_val
        curr_idx += var_size

    # Run the simulator to update geometry coefficients
    jax_sim.run()

    # Record history metrics
    cl_val = float(jax_sim[CL].item() if hasattr(jax_sim[CL], 'item') else jax_sim[CL][0])
    cd_val = float(jax_sim[CDi].item() if hasattr(jax_sim[CDi], 'item') else jax_sim[CDi][0])
    sref_val = float(jax_sim[planform_area].item() if hasattr(jax_sim[planform_area], 'item') else jax_sim[planform_area][0])

    cd_history.append(cd_val * 1e4)  # CD in drag counts (x 1e4)
    cl_history.append(cl_val)
    sref_history.append(sref_val)

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

    # Build chord info string for all 5 stations
    if 'chord_dvs' in unscaled_values:
        chord_vals = unscaled_values['chord_dvs']
    elif 'chord_stretch_dvs' in unscaled_values:
        chord_vals = 1.0 + unscaled_values['chord_stretch_dvs']
    else:
        chord_vals = np.ones(5)
    chord_str = "  ".join([f"c{i}={chord_vals[i]:.3f}" for i in range(len(chord_vals))])

    span_val = float(unscaled_values['wingspan'].item() if hasattr(unscaled_values['wingspan'], 'item') else unscaled_values['wingspan'][0]) if 'wingspan' in unscaled_values else 10.0
    sweep_val = float(unscaled_values['sweep_angle'].item() if hasattr(unscaled_values['sweep_angle'], 'item') else unscaled_values['sweep_angle'][0]) if 'sweep_angle' in unscaled_values else 0.0
    pitch_val = float(unscaled_values['pitch'].item() if hasattr(unscaled_values['pitch'], 'item') else unscaled_values['pitch'][0]) if 'pitch' in unscaled_values else 0.0

    # Add iteration counter label using unscaled physical values
    plotter.add_text(
        f"Iteration {iteration}/{num_iterations - 1}\n"
        f"Span={span_val:.2f}  {chord_str}\n"
        f"Sweep={np.degrees(sweep_val):.1f}°  Pitch={np.degrees(pitch_val):.1f}°",
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

    # Save final wing render on light gray background for the summary plot
    if iteration == num_iterations - 1:
        pv_temp = pv.Plotter(off_screen=True, window_size=[1000, 1000])
        pv_temp.set_background('#f4f4f4')
        for element in plotting_elements:
            if isinstance(element, dict) and 'mesh' in element:
                pv_temp.add_mesh(element['mesh'], **element.get('kwargs', {}))
            elif isinstance(element, tuple) and len(element) == 2:
                pv_temp.add_mesh(element[0], **element[1])
            elif isinstance(element, pv.Actor):
                pv_temp.add_actor(element)
            elif isinstance(element, pv.DataSet):
                pv_temp.add_mesh(element)
        pv_temp.camera.position = camera['position']
        pv_temp.camera.focal_point = camera['focal_point']
        pv_temp.camera.up = camera['viewup']
        pv_temp.add_axes()
        pv_temp.screenshot(wing_img_path)
        pv_temp.close()

plotter.close()
print(f"Video saved to: {video_path}")

# region Plot Summary Figure (Wing geometry vs Theory)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Compute theoretical Cd = Cl^2 / (pi * AR) in drag counts (x 1e4)
# AR = b^2 / S_ref; with b = 10.0 and S_ref = 10.0 -> AR = 10
span_b = 10.0
target_cl = 0.5
target_sref = 10.0
ar = (span_b ** 2) / target_sref
cd_theory_counts = (target_cl ** 2) / (np.pi * ar) * 1e4

fig = plt.figure(figsize=(14, 6), dpi=150)
gs = fig.add_gridspec(3, 2, width_ratios=[1.1, 1.5], wspace=0.25, hspace=0.2)

# Left panel: 3D Render of final optimized wing geometry
ax_img = fig.add_subplot(gs[:, 0])
if os.path.exists(wing_img_path):
    img = mpimg.imread(wing_img_path)
    ax_img.imshow(img)
ax_img.axis('off')

# Right panels: Optimization history plots
subplot_bg = '#eaeaf2'
grid_color = '#ffffff'
iters = np.arange(num_iterations)

# 1. Top Subplot: CD
ax_cd = fig.add_subplot(gs[0, 1])
ax_cd.set_facecolor(subplot_bg)
ax_cd.grid(True, color=grid_color, linewidth=1.2)
ax_cd.plot(iters, cd_history, 'o-', color='#3b6998', linewidth=2, markersize=5)
ax_cd.axhline(cd_theory_counts, color='#7a9bbd', linestyle='--', linewidth=1.8)
ax_cd.text(1.02, cd_theory_counts, 'theory', color='#7a9bbd', transform=ax_cd.get_yaxis_transform(),
            va='center', fontsize=11, fontweight='bold')
ax_cd.set_ylabel('CD', fontsize=11)
plt.setp(ax_cd.get_xticklabels(), visible=False)
for spine in ax_cd.spines.values():
    spine.set_visible(False)

# 2. Middle Subplot: CL
ax_cl = fig.add_subplot(gs[1, 1], sharex=ax_cd)
ax_cl.set_facecolor(subplot_bg)
ax_cl.grid(True, color=grid_color, linewidth=1.2)
ax_cl.plot(iters, cl_history, 'o-', color='#4fa86c', linewidth=2, markersize=5)
ax_cl.axhline(target_cl, color='#87c79d', linestyle='--', linewidth=1.8)
ax_cl.text(1.02, target_cl, 'con', color='#87c79d', transform=ax_cl.get_yaxis_transform(),
            va='center', fontsize=11, fontweight='bold')
ax_cl.set_ylabel('CL', fontsize=11)
plt.setp(ax_cl.get_xticklabels(), visible=False)
for spine in ax_cl.spines.values():
    spine.set_visible(False)

# 3. Bottom Subplot: S_ref
ax_sref = fig.add_subplot(gs[2, 1], sharex=ax_cd)
ax_sref.set_facecolor(subplot_bg)
ax_sref.grid(True, color=grid_color, linewidth=1.2)
ax_sref.plot(iters, sref_history, 'o-', color='#c54b4b', linewidth=2, markersize=5)
ax_sref.axhline(target_sref, color='#e08585', linestyle='--', linewidth=1.8)
ax_sref.text(1.02, target_sref, 'con', color='#e08585', transform=ax_sref.get_yaxis_transform(),
            va='center', fontsize=11, fontweight='bold')
ax_sref.set_ylabel('S_ref', fontsize=11)
ax_sref.set_xlabel('Iterations', fontsize=11)
for spine in ax_sref.spines.values():
    spine.set_visible(False)

fig.suptitle("Optimal wing geometry vs theory", y=0.03, fontsize=15, fontweight='bold')

summary_fig_path = os.path.join(latest_folder, 'optimization_summary.png')
plt.savefig(summary_fig_path, bbox_inches='tight', dpi=200)
plt.close()
print(f"Summary figure saved to: {summary_fig_path}")

# endregion Plot Summary Figure
# endregion Plot Optimization History