# region Imports and Setup

from dataclasses import dataclass
import csdl_alpha as csdl
import numpy as np
import lsdo_function_spaces as lfs

from lsdo_geo.core.parameterization.free_form_deformation_functions import construct_ffd_block_around_entities
from lsdo_geo.core.parameterization.sectional_parameterization import (
    SectionalParameterization,
    SectionalParameters
)
from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver, GeometricVariables

import lsdo_geo
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
imported_function_set = lfs.import_file_patched(file_name=geometry_directory + file_name + ".stp", parallelize=False)
geometry = lsdo_geo.Geometry(functions=imported_function_set.functions, 
                                      function_names=imported_function_set.function_names,
                                      name='imported_geometry',
                                      space=imported_function_set.space)

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

    new_space = lfs.BSplineSpaceNew(
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

# Project max thickness measurement points at x=0.3 for the 5 stations
upper_thickness_projections = []
lower_thickness_projections = []
for y_val in chord_station_y:
    upper_thickness_projections.append(geometry.project(np.array([0.3, y_val, 0.05]), direction=np.array([0, 0, -1])))
    lower_thickness_projections.append(geometry.project(np.array([0.3, y_val, -0.05]), direction=np.array([0, 0, 1])))

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

# Project line of beam nodes along the span at 25% chord
num_beam_nodes = 21
y_beam_span = np.linspace(-4.99, 4.99, num_beam_nodes)
beam_line_physical = np.zeros((num_beam_nodes, 3))
beam_line_physical[:, 0] = 0.25  # 25% chord elastic axis
beam_line_physical[:, 1] = y_beam_span
beam_line_physical[:, 2] = 0.0

projected_beam_mesh = geometry.project(beam_line_physical, plot=False)

le_line_physical = np.zeros((num_beam_nodes, 3))
le_line_physical[:, 0] = 0.00
le_line_physical[:, 1] = y_beam_span
le_line_physical[:, 2] = 0.0
projected_le_mesh = geometry.project(le_line_physical, plot=False)

te_line_physical = np.zeros((num_beam_nodes, 3))
te_line_physical[:, 0] = 1.00
te_line_physical[:, 1] = y_beam_span
te_line_physical[:, 2] = 0.0
projected_te_mesh = geometry.project(te_line_physical, plot=False)

upper_beam_seed = np.column_stack([np.full(num_beam_nodes, 0.25), y_beam_span, np.full(num_beam_nodes, 0.05)])
lower_beam_seed = np.column_stack([np.full(num_beam_nodes, 0.25), y_beam_span, np.full(num_beam_nodes, -0.05)])
projected_upper_beam_mesh = geometry.project(upper_beam_seed, direction=np.array([0, 0, -1]), plot=False)
projected_lower_beam_mesh = geometry.project(lower_beam_seed, direction=np.array([0, 0, 1]), plot=False)
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

projected_panel_centers = geometry.project(panel_centers, 
                            grid_search_density_parameter=1,
                            newton_tolerance=1.e-10,
                            grid_search_density_cutoff=30,
                            projection_tolerance=1.e-2,
                            force_reprojection=False, 
                            plot=False,
                            )

# Compute static spanwise (Y) interpolation mapping matrix
num_panels = len(panel_centers)
W_matrix = np.zeros((num_beam_nodes, num_panels))
Y_beam = beam_line_physical[:, 1]
Y_panels = panel_centers[:, 1]

for i in range(num_panels):
    y_p = Y_panels[i]
    if y_p <= Y_beam[0]:
        W_matrix[0, i] = 1.0
    elif y_p >= Y_beam[-1]:
        W_matrix[-1, i] = 1.0
    else:
        n = np.searchsorted(Y_beam, y_p) - 1
        dy = Y_beam[n+1] - Y_beam[n]
        W_matrix[n, i] = (Y_beam[n+1] - y_p) / dy
        W_matrix[n+1, i] = (y_p - Y_beam[n]) / dy

# endregion

# endregion

# region Create Parameterization Objects
# Construct a Free Form Deformation (FFD) block around the geometry
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
# Formulation flag:
# 'ar_area'   -> Design variables: 5 chord DVs, Aspect Ratio (AR), and pitch (planform area fixed at 10)
# 'chord_span' -> Design variables: 5 chord stretch DVs, span stretch DV, and pitch
# formulation = 'ar_area'  # Options: 'ar_area' or 'chord_span'
formulation = 'chord_span'  # Options: 'ar_area' or 'chord_span'

pitch = csdl.Variable(value=5.*np.pi/180) # pitch angle in radians

@dataclass
class DVInfo:
    variable: csdl.Variable
    lower: float
    upper: float
    scaler: float = 1.0

num_chord_stations = 5

num_beam_elements = 20
ttop = csdl.Variable(shape=(num_beam_elements,), value=np.ones(num_beam_elements) * 0.005)
tweb = csdl.Variable(shape=(num_beam_elements,), value=np.ones(num_beam_elements) * 0.005)

space_of_linear_9_dof_b_splines = lfs.BSplineSpaceNew(num_parametric_dimensions=1, degree=2, coefficients_shape=(9,))
space_of_linear_2_dof_b_splines = lfs.BSplineSpaceNew(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))

if formulation == 'ar_area':
    # Formulation 1: 4 taper ratio DVs (stations 1 to 4) + Aspect Ratio (AR) design variable
    taper_dvs = csdl.Variable(shape=(num_chord_stations - 1,), value=np.ones(num_chord_stations - 1))
    aspect_ratio = csdl.Variable(shape=(1,), value=np.array([10.0]))

    design_variables: dict[str, DVInfo] = {
        'taper_dvs': DVInfo(variable=taper_dvs, lower=0.15, upper=5.0, scaler=2.0),
        'aspect_ratio': DVInfo(variable=aspect_ratio, lower=2.0, upper=15.0, scaler=0.5),
        'pitch': DVInfo(variable=pitch, lower=-10.0*np.pi/180, upper=15.0*np.pi/180, scaler=1.e1),
        'ttop': DVInfo(variable=ttop, lower=0.001, upper=0.1, scaler=1.e2),
        'tweb': DVInfo(variable=tweb, lower=0.001, upper=0.1, scaler=1.e2),
    }

    # ParameterizationSolver drives 5 chord stretch states, 5 thickness stretch states, and 1 span stretch state
    chord_stretch_states = csdl.Variable(shape=(num_chord_stations,), value=np.zeros(num_chord_stations))
    thickness_stretch_states = csdl.Variable(shape=(num_chord_stations,), value=np.zeros(num_chord_stations))
    span_stretch_state = csdl.Variable(value=0.0)

    # Symmetrically construct 9 B-spline control point coefficients across full span
    chord_coeffs = csdl.concatenate([
        chord_stretch_states[4], chord_stretch_states[3], chord_stretch_states[2], chord_stretch_states[1],
        chord_stretch_states[0],
        chord_stretch_states[1], chord_stretch_states[2], chord_stretch_states[3], chord_stretch_states[4]
    ])
    thickness_coeffs = csdl.concatenate([
        thickness_stretch_states[4], thickness_stretch_states[3], thickness_stretch_states[2], thickness_stretch_states[1],
        thickness_stretch_states[0],
        thickness_stretch_states[1], thickness_stretch_states[2], thickness_stretch_states[3], thickness_stretch_states[4]
    ])
    wingspan_coeffs = csdl.concatenate([-span_stretch_state, span_stretch_state])

elif formulation == 'chord_span':
    # Formulation 2: 5 chord stretch DVs, 5 thickness stretch DVs + Span stretch DV
    chord_stretch_dvs = csdl.Variable(shape=(num_chord_stations,), value=np.zeros(num_chord_stations))
    thickness_stretch_dvs = csdl.Variable(shape=(num_chord_stations,), value=np.zeros(num_chord_stations))
    span_stretch_dv = csdl.Variable(shape=(1,), value=np.array([0.0]))

    design_variables: dict[str, DVInfo] = {
        'chord_stretch_dvs': DVInfo(variable=chord_stretch_dvs, lower=-0.85, upper=4.0, scaler=1.0),
        'thickness_stretch_dvs': DVInfo(variable=thickness_stretch_dvs, lower=-0.85, upper=4.0, scaler=1.0),
        'span_stretch_dv': DVInfo(variable=span_stretch_dv, lower=-4.5, upper=20.0, scaler=1.e-1),
        'pitch': DVInfo(variable=pitch, lower=-10.0*np.pi/180, upper=15.0*np.pi/180, scaler=1.e1),
        'ttop': DVInfo(variable=ttop, lower=0.001, upper=0.1, scaler=1.e2),
        'tweb': DVInfo(variable=tweb, lower=0.001, upper=0.1, scaler=1.e2),
    }

    chord_coeffs = csdl.concatenate([
        chord_stretch_dvs[4], chord_stretch_dvs[3], chord_stretch_dvs[2], chord_stretch_dvs[1],
        chord_stretch_dvs[0],
        chord_stretch_dvs[1], chord_stretch_dvs[2], chord_stretch_dvs[3], chord_stretch_dvs[4]
    ])
    thickness_coeffs = csdl.concatenate([
        thickness_stretch_dvs[4], thickness_stretch_dvs[3], thickness_stretch_dvs[2], thickness_stretch_dvs[1],
        thickness_stretch_dvs[0],
        thickness_stretch_dvs[1], thickness_stretch_dvs[2], thickness_stretch_dvs[3], thickness_stretch_dvs[4]
    ])
    wingspan_coeffs = csdl.concatenate([-span_stretch_dv, span_stretch_dv])

chord_stretching_b_spline = lfs.Function(
    space=space_of_linear_9_dof_b_splines,
    coefficients=chord_coeffs,
    name='chord_stretching_b_spline_coefficients'
)
thickness_stretching_b_spline = lfs.Function(
    space=space_of_linear_9_dof_b_splines,
    coefficients=thickness_coeffs,
    name='thickness_stretching_b_spline_coefficients'
)

wingspan_stretching_b_spline = lfs.Function(
    space=space_of_linear_2_dof_b_splines,
    coefficients=wingspan_coeffs,
    name='wingspan_stretching_b_spline_coefficients'
)

parametric_b_spline_inputs = np.linspace(0.0, 1.0, num_ffd_sections).reshape((-1, 1))
chord_stretch_sectional_parameters = chord_stretching_b_spline.evaluate(parametric_b_spline_inputs)
thickness_stretch_sectional_parameters = thickness_stretching_b_spline.evaluate(parametric_b_spline_inputs)
wingspan_stretch_sectional_parameters = wingspan_stretching_b_spline.evaluate(parametric_b_spline_inputs)

sectional_parameters = SectionalParameters()
sectional_parameters.add_stretch(axis=0, stretch=chord_stretch_sectional_parameters)
sectional_parameters.add_stretch(axis=2, stretch=thickness_stretch_sectional_parameters)
sectional_parameters.add_translation(axis=1, translation=wingspan_stretch_sectional_parameters)

ffd_coefficients = ffd_sectional_parameterization.evaluate(sectional_parameters, plot=False)

geometry_coefficients = ffd_block.evaluate_ffd(coefficients=ffd_coefficients, plot=False)
geometry.set_coefficients(geometry_coefficients) # type: ignore

wingspan = geometry.evaluate(leading_edge_right)[1] - geometry.evaluate(leading_edge_left)[1] # type: ignore

# Evaluate local sectional chords and thicknesses at the 5 spanwise stations
local_chords = [
    geometry.evaluate(chord_te_projections[i])[0] - geometry.evaluate(chord_le_projections[i])[0]
    for i in range(num_chord_stations)
]
local_thicknesses = [
    geometry.evaluate(upper_thickness_projections[i])[2] - geometry.evaluate(lower_thickness_projections[i])[2]
    for i in range(num_chord_stations)
]

# Planform area computed from upper and lower skin surface grid projections:
upper_skin_pts = geometry.evaluate(projected_upper_skin, plot=False)
lower_skin_pts = geometry.evaluate(projected_lower_skin, plot=False)
chord_surface_pts = 0.5 * (upper_skin_pts + lower_skin_pts)
chord_surface_grid = csdl.reshape(chord_surface_pts, (nx_area, ny_area, 3))

v_x = chord_surface_grid[1:, :-1, :] - chord_surface_grid[:-1, :-1, :]
v_y = chord_surface_grid[:-1, 1:, :] - chord_surface_grid[:-1, :-1, :]
area_vectors = csdl.cross(v_x, v_y, axis=2)
element_areas = csdl.norm(area_vectors, axes=(2,))
planform_area = csdl.sum(element_areas)

aspect_ratio_calc = (wingspan**2) / planform_area

if formulation == 'ar_area':
    # ParameterizationSolver manipulates states to match targets
    geometry_solver = ParameterizationSolver()
    geometry_solver.add_state(chord_stretch_states)
    geometry_solver.add_state(thickness_stretch_states)
    geometry_solver.add_state(span_stretch_state)

    geometric_variables = GeometricVariables()
    # Enforce normalized chord profile (taper ratios) at stations 1 to 4
    for i in range(1, num_chord_stations):
        normalized_chord = local_chords[i] / local_chords[0]
        geometric_variables.add_variable(normalized_chord, taper_dvs[i - 1], penalty_value=None)
    
    # Enforce constant thickness-to-chord ratio = 0.12 at all stations
    for i in range(num_chord_stations):
        tc_ratio = local_thicknesses[i] / local_chords[i]
        geometric_variables.add_variable(tc_ratio, 0.12, penalty_value=None)
    
    # Enforce planform area and aspect ratio simultaneously
    geometric_variables.add_variable(planform_area, 10.0, penalty_value=None)
    geometric_variables.add_variable(aspect_ratio_calc, aspect_ratio, penalty_value=None)

    geometry_solver.evaluate(geometric_variables)

geometry.rotate(rotation_origin=geometry.evaluate(quarter_chord_center), axis_vector=np.array([0., 1., 0.]), angles=pitch, units='radians')

# cruise_speed = csdl.Variable(value=1.)
cruise_speed = csdl.Variable(value=20.)
velocity = csdl.concatenate([cruise_speed, csdl.Variable(value=0.), csdl.Variable(value=0.)])

# region Aerodynamic solver (panel method)
num_nodes = 1

panel_mesh = geometry.evaluate(projected_panel_mesh, plot=False)
panel_mesh = panel_mesh.expand((1,) + panel_mesh.shape, 'ij->aij')

point_velocities = csdl.expand(velocity, (num_nodes,) + panel_mesh.shape[1:], 'j->iaj')
rho_array = csdl.Variable(shape=(num_nodes,), value=np.array([1.225]))
sos_array = csdl.Variable(shape=(num_nodes,), value=np.array([343.0]))

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
beam_mesh = geometry.evaluate(projected_beam_mesh, plot=False)
le_mesh = geometry.evaluate(projected_le_mesh, plot=False)
te_mesh = geometry.evaluate(projected_te_mesh, plot=False)
upper_beam_mesh = geometry.evaluate(projected_upper_beam_mesh, plot=False)
lower_beam_mesh = geometry.evaluate(projected_lower_beam_mesh, plot=False)

# Compute local chord from projected mesh (leading edge to trailing edge distance)
node_chords = te_mesh[:,0] - le_mesh[:,0]
local_chord = 0.5 * (node_chords[:-1] + node_chords[1:])

node_heights = upper_beam_mesh[:, 2] - lower_beam_mesh[:, 2]
local_height = 0.5 * (node_heights[:-1] + node_heights[1:])

# Define wingbox cross-section along the span
box_width = 0.50 * local_chord
box_height = local_height

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

# Rigorously map aerodynamic panel forces to beam structural nodes
dynamic_panel_centers = geometry.evaluate(projected_panel_centers, plot=False)
panel_forces_var = outputs['panel_forces'][0] # shape (num_panels, 3)

W_var = csdl.Variable(value=W_matrix)

# 1. Force Mapping: F_node = W * F_panel
F_node = csdl.matmat(W_var, panel_forces_var)

# 2. Moment Mapping: M_node = sum_i (W_i * (r_i x F_i))
B_expand = csdl.expand(beam_mesh, (num_beam_nodes, num_panels, 3), 'nj->nij')
C_expand = csdl.expand(dynamic_panel_centers, (num_beam_nodes, num_panels, 3), 'ij->nij')
F_expand = csdl.expand(panel_forces_var, (num_beam_nodes, num_panels, 3), 'ij->nij')
W_expand = csdl.expand(W_var, (num_beam_nodes, num_panels, 3), 'ni->nij')

r = C_expand - B_expand
r_cross_F = csdl.cross(r, F_expand, axis=2)
M_node = csdl.sum(W_expand * r_cross_F, axes=(1,))

# Combine mapped forces and moments into 6-DOF beam loads
beam_loads = csdl.concatenate([F_node, M_node], axis=1) # shape (num_beam_nodes, 6)
beam.add_load(beam_loads)

# Solve structural beam model using aframe
frame = aframe.Frame(beams=[beam])
frame.solve()

beam_displacement = frame.displacement['wing_spar']
structural_mass = beam.mass
beam_stress = frame.compute_stress()['wing_spar']
beam_stress.set_as_constraint(upper=276e6, scaler=1.e-8) # Yield stress for Aluminum 6061 is ~276 MPa
# endregion Structural solver (beam model)


# Compute calculated aspect ratio from geometry (wingspan and planform area)
# Define design variables, constraints, and objective for optimization problem
# Compute dimensional Trefftz drag as objective
Di_Trefftz = CDi * 0.5 * rho_array[0] * (cruise_speed**2) * planform_area
objective = Di_Trefftz
objective.set_as_objective(scaler=1.e1)

# L = W constraint
# Set a fixed positive payload weight so W_total cannot reach zero!
# A payload of 1150.0 N + structural weight will require ~1225 N of lift.
# At V = 20 m/s and S = 10 m^2, 1225 N corresponds to a CL of ~0.5!
# payload_weight = csdl.Variable(value=1150.0)
payload_weight = csdl.Variable(value=1000.0)
W_total = structural_mass * 9.81 + payload_weight
lift_trim = L - W_total
lift_trim.set_as_constraint(equals=0.0, scaler=1.e-3)

if formulation == 'chord_span':
    # For chord and span stretch formulation (no ParameterizationSolver),
    # keep planform area constraint and aspect ratio inequality constraint AR <= 15.0
    planform_area.set_as_constraint(equals=10.0, scaler=1.e-1)
    aspect_ratio_calc.set_as_constraint(upper=15.0, scaler=1.e-1)
    # Enforce constant thickness-to-chord ratio = 0.12 at all stations
    for i in range(num_chord_stations):
        tc_ratio = local_thicknesses[i] / local_chords[i]
        tc_ratio.set_as_constraint(equals=0.12, scaler=1.e1)
else:
    # For AR and Area formulation, ParameterizationSolver explicitly enforces
    # taper ratios, planform area, and aspect ratio.
    pass

for dv_info in design_variables.values():
    dv_info.variable.set_as_design_variable(lower=dv_info.lower, upper=dv_info.upper, scaler=dv_info.scaler)

geometry_coefficients = [geometry_function.coefficients for geometry_function in geometry.functions.values()]

jax_sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=[dv_info.variable for dv_info in design_variables.values()],
    additional_outputs=[Di, L, CL, CDi, Cp, panel_mesh, planform_area, aspect_ratio_calc, structural_mass, beam_displacement, beam_stress, Di_Trefftz, W_total] + geometry_coefficients,
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
optimization_problem = modopt.CSDLAlphaProblem(problem_name='rectangular_wing_aerostructural_shape_optimization_with_chord_profile', simulator=jax_sim)
# optimizer = modopt.IPOPT(optimization_problem, recording=True)
optimizer = modopt.PySLSQP(optimization_problem, solver_options={'maxiter': 100, 'acc': 1.e-7}, readable_outputs=['x'])
optimizer.solve()
optimizer.print_results()

# endregion Optimization


# region Plot Optimization History
import pyvista as pv
import os, glob

# Find the latest output folder
output_base_dir = 'rectangular_wing_aerostructural_shape_optimization_with_chord_profile_outputs'
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
                        if 'aspect_ratio' in inp_grp and 'planform_area_dv' in inp_grp:
                            ar_val = inp_grp['aspect_ratio'][:]
                            s_val = inp_grp['planform_area_dv'][:]
                            x_vec = np.concatenate([ar_val, s_val, pitch_val])
                            x_history_list.append(x_vec)
                        elif 'chord_stretch_dv' in inp_grp and 'span_stretch_dv' in inp_grp:
                            cs_val = inp_grp['chord_stretch_dv'][:]
                            ss_val = inp_grp['span_stretch_dv'][:]
                            x_vec = np.concatenate([cs_val, ss_val, pitch_val])
                            x_history_list.append(x_vec)
                        elif 'taper_dvs' in inp_grp:
                            taper = inp_grp['taper_dvs'][:]
                            ar_val = inp_grp['aspect_ratio'][:] if 'aspect_ratio' in inp_grp else np.array([10.0])
                            x_vec = np.concatenate([taper, ar_val, pitch_val])
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

    # Build parameter info string depending on active formulation
    if formulation == 'ar_area':
        ar_val = float(unscaled_values['aspect_ratio'].item() if hasattr(unscaled_values['aspect_ratio'], 'item') else unscaled_values['aspect_ratio'][0]) if 'aspect_ratio' in unscaled_values else 10.0
        t_vals = unscaled_values['taper_dvs'] if 'taper_dvs' in unscaled_values else np.ones(4)
        c_vals = [1.0] + list(t_vals)
        c_str = " ".join([f"t{i}={c_vals[i]:.2f}" for i in range(len(c_vals))])
        dv_str = f"AR={ar_val:.2f}  {c_str}"
    elif formulation == 'chord_span':
        ss_val = float(unscaled_values['span_stretch_dv'].item() if hasattr(unscaled_values['span_stretch_dv'], 'item') else unscaled_values['span_stretch_dv'][0]) if 'span_stretch_dv' in unscaled_values else 0.0
        cs_vals = unscaled_values['chord_stretch_dvs'] if 'chord_stretch_dvs' in unscaled_values else np.zeros(5)
        c_str = " ".join([f"c{i}={1.0+cs_vals[i]:.2f}" for i in range(len(cs_vals))])
        dv_str = f"b_stretch={ss_val:.2f}  {c_str}"

    pitch_val = float(unscaled_values['pitch'].item() if hasattr(unscaled_values['pitch'], 'item') else unscaled_values['pitch'][0]) if 'pitch' in unscaled_values else 0.0

    # Add iteration counter label using unscaled physical values
    plotter.add_text(
        f"Iteration {iteration}/{num_iterations - 1}\n"
        f"Formulation: {formulation}\n"
        f"{dv_str}\n"
        f"Pitch={np.degrees(pitch_val):.1f}°",
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
# ar = (span_b ** 2) / target_sref
target_ar = 15
cd_theory_counts = (target_cl ** 2) / (np.pi * target_ar) * 1e4

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