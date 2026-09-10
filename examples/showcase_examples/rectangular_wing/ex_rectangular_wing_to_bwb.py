# region Imports and Setup

from dataclasses import dataclass
from typing import Union
import numpy.typing as npt
import csdl_alpha as csdl
import numpy as np
import lsdo_function_spaces as lfs

from lsdo_geo.core.parameterization.free_form_deformation_functions import construct_ffd_block_around_entities
from lsdo_geo.core.parameterization.sectional_parameterization import (
    SectionalParameterization,
    SectionalParameters,
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
# Refined mesh generated from OpenVSP (2,872 quads, 35 spanwise strips per half-span)
mesh_file_name = "rectangular_wing_naca0012_35sect"
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
elevator_hinge = geometry.project(np.array([0.80, 0.0, 0.0]))

# Project chord measurement points at 8 evenly spaced stations across the right half-span
# Stations: y = 0.0 (root) to 5.0 (tip)
num_chord_stations = 8
chord_station_y = np.linspace(0.0, 5.0, num_chord_stations)
chord_le_projections = []
chord_te_projections = []
quarter_chord_projections = []
for y_val in chord_station_y:
    chord_le_projections.append(geometry.project(np.array([0.0, y_val, 0.0])))
    chord_te_projections.append(geometry.project(np.array([1.0, y_val, 0.0])))
    quarter_chord_projections.append(geometry.project(np.array([0.25, y_val, 0.0])))

# Project max thickness measurement points at x=0.3 for the 8 stations
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

# Project line of beam nodes along the span at 25% chord for right half-span (y >= 0)
num_beam_nodes = 15
y_beam_span = np.linspace(0.0, 4.99, num_beam_nodes)
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
mesh = meshio.read(geometry_directory + mesh_file_name + ".msh")

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

# Filter panels for right half of wing (y > 0)
right_panel_indices = np.where(panel_centers[:, 1] > 0.0)[0]
num_right_panels = len(right_panel_indices)
Y_panels_right = panel_centers[right_panel_indices, 1]
Y_beam = beam_line_physical[:, 1]

# Compute static spanwise (Y) interpolation mapping matrix for right half-wing
W_matrix = np.zeros((num_beam_nodes, num_right_panels))
for i in range(num_right_panels):
    y_p = Y_panels_right[i]
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
num_ffd_sections = 15
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
# # Formulation flag:
# 'ar_area'   -> Design variables: 8 chord DVs, Aspect Ratio (AR), and pitch (planform area fixed at 10)
# 'chord_span' -> Design variables: 8 chord stretch DVs, 8 sweep DVs, span stretch DV, elevator angle, and pitch
# formulation = 'ar_area'  # Options: 'ar_area' or 'chord_span'
formulation = 'chord_span'  # Options: 'ar_area' or 'chord_span'

pitch = csdl.Variable(value=5.*np.pi/180) # pitch angle in radians
elevator_angle = csdl.Variable(value=0.0) # elevator deflection angle in radians

@dataclass
class DVInfo:
    variable: csdl.Variable
    lower: Union[float, npt.NDArray[np.float64]]
    upper: Union[float, npt.NDArray[np.float64]]
    scaler: float = 1.0

# Define B-spline space of 2-dof linear B-splines
space_of_linear_2_dof_b_splines = lfs.BSplineSpaceNew(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))

init_file = 'rectangular_wing_to_bwb_aerostructural_optimization_outputs/2026-09-09_12.44.58.866121/x.out'

num_thickness_stations = 8
thickness_space = lfs.BSplineSpaceNew(num_parametric_dimensions=1, degree=2, coefficients_shape=(num_thickness_stations,))
ttop_dvs = csdl.Variable(shape=(num_thickness_stations,), value=np.ones(num_thickness_stations) * 0.005)
tweb_dvs = csdl.Variable(shape=(num_thickness_stations,), value=np.ones(num_thickness_stations) * 0.005)

num_chord_stations = 8
num_span_coeffs = 2 * num_chord_stations - 1  # 15 coefficients across full span
space_of_linear_15_dof_b_splines = lfs.BSplineSpaceNew(num_parametric_dimensions=1, degree=2, coefficients_shape=(num_span_coeffs,))
space_of_linear_2_dof_b_splines = lfs.BSplineSpaceNew(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))

if formulation == 'ar_area':
    # Formulation 1: Taper ratio DVs (stations 1 to 7) + Aspect Ratio (AR) + Sectional Sweep Angles + Elevator + Pitch
    taper_dvs = csdl.Variable(shape=(num_chord_stations - 1,), value=np.ones(num_chord_stations - 1))
    aspect_ratio = csdl.Variable(shape=(1,), value=np.array([10.0]))
    sweep_angle_dvs = csdl.Variable(shape=(num_chord_stations - 1,), value=np.zeros(num_chord_stations - 1))

    design_variables: dict[str, DVInfo] = {
        'taper_dvs': DVInfo(variable=taper_dvs, lower=0.15, upper=5.0, scaler=2.0),
        'aspect_ratio': DVInfo(variable=aspect_ratio, lower=2.0, upper=15.0, scaler=0.5),
        'sweep_angle_dvs': DVInfo(variable=sweep_angle_dvs, lower=-10.0*np.pi/180, upper=45.0*np.pi/180, scaler=1.e1),
        'elevator_angle': DVInfo(variable=elevator_angle, lower=-25.0*np.pi/180, upper=25.0*np.pi/180, scaler=1.e1),
        'pitch': DVInfo(variable=pitch, lower=-10.0*np.pi/180, upper=15.0*np.pi/180, scaler=1.e1),
        'ttop_dvs': DVInfo(variable=ttop_dvs, lower=0.001, upper=0.1, scaler=1.e2),
        'tweb_dvs': DVInfo(variable=tweb_dvs, lower=0.001, upper=0.1, scaler=1.e2),
    }

    # ParameterizationSolver drives 8 chord stretch states, 8 thickness stretch states, 1 span stretch state, and 7 sweep translation states
    chord_stretch_states = csdl.Variable(shape=(num_chord_stations,), value=np.zeros(num_chord_stations))
    thickness_stretch_states = csdl.Variable(shape=(num_chord_stations,), value=np.zeros(num_chord_stations))
    span_stretch_state = csdl.Variable(value=0.0)
    sweep_translation_states = csdl.Variable(shape=(num_chord_stations - 1,), value=np.zeros(num_chord_stations - 1))

    sweep_full_half = csdl.concatenate([csdl.Variable(value=0.0), sweep_translation_states])

    chord_coeffs = csdl.concatenate(
        [chord_stretch_states[i] for i in range(num_chord_stations - 1, 0, -1)] +
        [chord_stretch_states[i] for i in range(num_chord_stations)]
    )
    sweep_coeffs = csdl.concatenate(
        [sweep_full_half[i] for i in range(num_chord_stations - 1, 0, -1)] +
        [sweep_full_half[i] for i in range(num_chord_stations)]
    )
    thickness_coeffs = csdl.concatenate(
        [thickness_stretch_states[i] for i in range(num_chord_stations - 1, 0, -1)] +
        [thickness_stretch_states[i] for i in range(num_chord_stations)]
    )
    wingspan_coeffs = csdl.concatenate([-span_stretch_state, span_stretch_state])

elif formulation == 'chord_span':
    # Formulation 2: 8 chord stretch DVs, 8 sweep DVs, 8 thickness stretch DVs, span stretch DV, elevator angle, pitch
    init_chord = np.zeros(num_chord_stations)
    init_sweep = np.zeros(num_chord_stations)
    init_thick = np.zeros(num_chord_stations)
    init_span = np.array([0.0])
    init_elev = 0.0
    init_pitch = 5.0 * np.pi / 180.0
    init_ttop_val = np.ones(num_thickness_stations) * 0.005
    init_tweb_val = np.ones(num_thickness_stations) * 0.005

    warm_start = False

    chord_stretch_dvs = csdl.Variable(shape=(num_chord_stations,), value=init_chord)
    sweep_dvs = csdl.Variable(shape=(num_chord_stations,), value=init_sweep)
    thickness_stretch_dvs = csdl.Variable(shape=(num_chord_stations,), value=init_thick)
    span_stretch_dv = csdl.Variable(shape=(1,), value=init_span)
    pitch.value = init_pitch
    elevator_angle.value = init_elev
    ttop_dvs.value = init_ttop_val
    tweb_dvs.value = init_tweb_val

    sweep_lower = np.full(num_chord_stations, -0.5)
    sweep_lower[0] = 0.0  # fix root sweep to 0
    sweep_upper = np.full(num_chord_stations, 4.0)
    sweep_upper[0] = 0.0

    design_variables: dict[str, DVInfo] = {
        'chord_stretch_dvs': DVInfo(variable=chord_stretch_dvs, lower=-0.85, upper=4.0, scaler=1.0),
        'sweep_dvs': DVInfo(variable=sweep_dvs, lower=sweep_lower, upper=sweep_upper, scaler=1.0),
        'thickness_stretch_dvs': DVInfo(variable=thickness_stretch_dvs, lower=-0.85, upper=4.0, scaler=1.0),
        'span_stretch_dv': DVInfo(variable=span_stretch_dv, lower=-4.5, upper=20.0, scaler=1.e-1),
        'elevator_angle': DVInfo(variable=elevator_angle, lower=-25.0*np.pi/180, upper=25.0*np.pi/180, scaler=1.e1),
        'pitch': DVInfo(variable=pitch, lower=-10.0*np.pi/180, upper=15.0*np.pi/180, scaler=1.e1),
        'ttop_dvs': DVInfo(variable=ttop_dvs, lower=0.001, upper=0.1, scaler=1.e2),
        'tweb_dvs': DVInfo(variable=tweb_dvs, lower=0.001, upper=0.1, scaler=1.e2),
    }

    chord_coeffs = csdl.concatenate(
        [chord_stretch_dvs[i] for i in range(num_chord_stations - 1, 0, -1)] +
        [chord_stretch_dvs[i] for i in range(num_chord_stations)]
    )
    sweep_coeffs = csdl.concatenate(
        [sweep_dvs[i] for i in range(num_chord_stations - 1, 0, -1)] +
        [sweep_dvs[i] for i in range(num_chord_stations)]
    )
    thickness_coeffs = csdl.concatenate(
        [thickness_stretch_dvs[i] for i in range(num_chord_stations - 1, 0, -1)] +
        [thickness_stretch_dvs[i] for i in range(num_chord_stations)]
    )
    wingspan_coeffs = csdl.concatenate([-span_stretch_dv, span_stretch_dv])

chord_stretching_b_spline = lfs.Function(
    space=space_of_linear_15_dof_b_splines,
    coefficients=chord_coeffs,
    name='chord_stretching_b_spline_coefficients'
)
sweep_translation_b_spline = lfs.Function(
    space=space_of_linear_15_dof_b_splines,
    coefficients=sweep_coeffs,
    name='sweep_translation_b_spline_coefficients'
)
thickness_stretching_b_spline = lfs.Function(
    space=space_of_linear_15_dof_b_splines,
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
sweep_sectional_parameters = sweep_translation_b_spline.evaluate(parametric_b_spline_inputs)
thickness_stretch_sectional_parameters = thickness_stretching_b_spline.evaluate(parametric_b_spline_inputs)
wingspan_stretch_sectional_parameters = wingspan_stretching_b_spline.evaluate(parametric_b_spline_inputs)

sectional_parameters = SectionalParameters()
sectional_parameters.add_stretch(axis=0, stretch=chord_stretch_sectional_parameters)
sectional_parameters.add_translation(axis=0, translation=sweep_sectional_parameters)
sectional_parameters.add_stretch(axis=2, stretch=thickness_stretch_sectional_parameters)
sectional_parameters.add_translation(axis=1, translation=wingspan_stretch_sectional_parameters)

ffd_coefficients = ffd_sectional_parameterization.evaluate(sectional_parameters, plot=False)

geometry_coefficients = ffd_block.evaluate_ffd(coefficients=ffd_coefficients, plot=False)
geometry.set_coefficients(geometry_coefficients) # type: ignore

# Apply elevator deflection to the back 20% of chord and middle quarter of wing
# Functions 0 & 5: lower surfaces, trailing edge is at rows 0:6 (x in [0.789, 1.0])
# Functions 1 & 4: upper surfaces, trailing edge is at rows 97:103 (x in [0.789, 1.0])
# Spanwise columns 0:4 cover |y| <= 1.07 m (~21.4% of span, matching middle quarter)
hinge_origin = geometry.evaluate(elevator_hinge)
for f_idx, row_slc in [(0, slice(0, 6)), (5, slice(0, 6)), (1, slice(97, None)), (4, slice(97, None))]:
    func = geometry.functions[f_idx]
    sub_pts = func.coefficients[:4, row_slc, :]
    rot_sub = lsdo_geo.rotate(
        points=sub_pts,
        rotation_origin=hinge_origin,
        axis_vector=np.array([0., 1., 0.]),
        angles=elevator_angle,
        units='radians'
    )
    func.coefficients = func.coefficients.set(csdl.slice[:4, row_slc, :], rot_sub)

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
    geometry_solver.add_state(sweep_translation_states)

    geometric_variables = GeometricVariables()
    # Enforce normalized chord profile (taper ratios) at stations 1 to 7
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

    # Enforce sectional sweep angles between adjacent quarter chord stations
    qc_pts = [geometry.evaluate(quarter_chord_projections[i]) for i in range(num_chord_stations)]
    for i in range(num_chord_stations - 1):
        dx = qc_pts[i + 1][0] - qc_pts[i][0]
        dy = qc_pts[i + 1][1] - qc_pts[i][1]
        sectional_sweep = csdl.arctan(dx / dy)
        geometric_variables.add_variable(sectional_sweep, sweep_angle_dvs[i], penalty_value=None)

    geometry_solver.evaluate(geometric_variables)

geometry.rotate(rotation_origin=geometry.evaluate(quarter_chord_center), axis_vector=np.array([0., 1., 0.]), angles=pitch, units='radians')

# cruise_speed = csdl.Variable(value=1.)
cruise_speed = csdl.Variable(value=20.)

# region Aerodynamic solver (panel method)
# 2 nodes: Node 0 = cruise condition, Node 1 = stability condition (+1.0 deg alpha perturbation)
num_nodes = 2

panel_mesh = geometry.evaluate(projected_panel_mesh, plot=False)
panel_mesh = panel_mesh.expand((1,) + panel_mesh.shape, 'ij->aij')

dalpha_rad = 1.0 * np.pi / 180.0
v0 = csdl.concatenate([cruise_speed, csdl.Variable(value=0.0), csdl.Variable(value=0.0)])
v1 = csdl.concatenate([cruise_speed * np.cos(dalpha_rad), csdl.Variable(value=0.0), cruise_speed * np.sin(dalpha_rad)])
v_stacked = csdl.reshape(csdl.concatenate([v0, v1]), (2, 3))
point_velocities = csdl.expand(v_stacked, (num_nodes,) + panel_mesh.shape[1:], 'ij->iaj')

rho_array = csdl.Variable(shape=(num_nodes,), value=np.array([1.225, 1.225]))
sos_array = csdl.Variable(shape=(num_nodes,), value=np.array([343.0, 343.0]))

# region Structural beam model geometry and Center of Mass calculation
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

# Evaluate B-spline thickness parameterization at element midpoints
num_beam_elements = num_beam_nodes - 1
y_elem_np = 0.5 * (y_beam_span[:-1] + y_beam_span[1:])
y_norm_elem = (y_elem_np / 4.99).reshape((-1, 1))

ttop_func = lfs.Function(space=thickness_space, coefficients=ttop_dvs)
tweb_func = lfs.Function(space=thickness_space, coefficients=tweb_dvs)
ttop_elem = ttop_func.evaluate(y_norm_elem)
tweb_elem = tweb_func.evaluate(y_norm_elem)

beam_cs = aframe.CSBox(
    height=box_height,
    width=box_width,
    ttop=ttop_elem,
    tbot=ttop_elem,
    tweb=tweb_elem,
)

# Beam material (Aluminum: E = 69 GPa, G = 26 GPa, density = 2700 kg/m^3)
beam = aframe.Beam(name='wing_spar', mesh=beam_mesh, E=69e9, G=26e9, density=2700, cs=beam_cs)

# Fix root node at y=0 (clamped cantilever symmetry boundary condition)
beam.fix(node=0)

# Compute structural mass (doubled for full wing) and structural center of mass from beam model
structural_mass = 2.0 * beam.mass
structural_cg = beam.cg
x_struct = structural_cg[0]
z_struct = structural_cg[2]

# Compute payload location: 40% chord at the root of the wing (y = 0.0, node 0)
x_le_root = le_mesh[0, 0]
x_te_root = te_mesh[0, 0]
root_chord = x_te_root - x_le_root
x_payload = x_le_root + 0.40 * root_chord
y_payload = csdl.Variable(value=np.array([0.0]))
z_payload = 0.5 * (upper_beam_mesh[0, 2] + lower_beam_mesh[0, 2])

# Fixed positive payload weight of 1000.0 N
payload_weight = csdl.Variable(value=1000.0)
payload_mass = payload_weight / 9.81
W_total = structural_mass * 9.81 + payload_weight
total_mass = structural_mass + payload_mass

# Dynamic composite aircraft Center of Mass (CG) updated each iteration
x_cg = (structural_mass * x_struct + payload_mass * x_payload) / total_mass
z_cg = (structural_mass * z_struct + payload_mass * z_payload) / total_mass
r_cg = csdl.concatenate([csdl.reshape(x_cg, (1,)), y_payload, csdl.reshape(z_cg, (1,))])
# endregion Structural beam model geometry and Center of Mass calculation

pm_solver_inputs = {
    'V_inf': -point_velocities,
    'rho': rho_array,
    'sos': sos_array,
    'compressibility': True,
    'Cp cutoff': -5.,
    'partition_size': 1,
    'reuse_AIC': True,
    # 'mesh_path': file_path+file_name, # already done externally
    'ref_area': planform_area, # does not matter bc we don't use the coefficients,
    'moment_reference': r_cg,
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
M = outputs['M']
CM = outputs['CM']
Cp = outputs['Cp']

# endregion Aerodynamic solver (panel method)

# region Structural solver (beam loads & stress)
# Rigorously map aerodynamic panel forces from right half-span panels to beam structural nodes
dynamic_panel_centers = geometry.evaluate(projected_panel_centers, plot=False)
dynamic_panel_centers_right = dynamic_panel_centers[:num_right_panels, :]
panel_forces_right = outputs['panel_forces'][0, :num_right_panels, :] # shape (num_right_panels, 3)

W_var = csdl.Variable(value=W_matrix)

# 1. Force Mapping: F_node = W * F_panel_right
F_node = csdl.matmat(W_var, panel_forces_right)

# 2. Moment Mapping: M_node = sum_i (W_i * (r_i x F_i))
B_expand = csdl.expand(beam_mesh, (num_beam_nodes, num_right_panels, 3), 'nj->nij')
C_expand = csdl.expand(dynamic_panel_centers_right, (num_beam_nodes, num_right_panels, 3), 'ij->nij')
F_expand = csdl.expand(panel_forces_right, (num_beam_nodes, num_right_panels, 3), 'ij->nij')
W_expand = csdl.expand(W_var, (num_beam_nodes, num_right_panels, 3), 'ni->nij')

r = C_expand - B_expand
r_cross_F = csdl.cross(r, F_expand, axis=2)
M_node = csdl.sum(W_expand * r_cross_F, axes=(1,))

# 3. 2.5g Sizing Case: Scale mapped forces and moments by 2.5x
beam_loads_2_5g = 2.5 * csdl.concatenate([F_node, M_node], axis=1) # shape (num_beam_nodes, 6)
beam.add_load(beam_loads_2_5g)

# Solve structural beam model using aframe
frame = aframe.Frame(beams=[beam])
frame.solve()

beam_displacement = frame.displacement['wing_spar']
beam_stress = frame.compute_stress()['wing_spar'] # shape (num_beam_elements, 5)

# Cross-sectional stress aggregation per element using csdl.maximum with rho=1.0
elem_max_stress = csdl.maximum(beam_stress, axes=(1,), rho=1.0) # shape (num_beam_elements,)

# Fit 1D B-spline function to aggregated element stresses across the half-span
stress_space = lfs.BSplineSpaceNew(num_parametric_dimensions=1, degree=2, coefficients_shape=(num_thickness_stations,))
stress_coeffs = stress_space.fit(
    values=csdl.reshape(elem_max_stress, (num_beam_elements, 1)),
    parametric_coordinates=y_norm_elem
)
stress_func = lfs.Function(space=stress_space, coefficients=stress_coeffs)

# Evaluate fitted stress at the 8 structural DV locations
dv_locs = np.linspace(0.0, 1.0, num_thickness_stations).reshape((-1, 1))
dv_stresses = stress_func.evaluate(dv_locs) # shape (8,)

# Yield stress constraint enforced only at the 8 DV locations (Yield stress for Aluminum 6061 is 276 MPa)
dv_stresses.set_as_constraint(upper=276e6, scaler=1.e-8)
# endregion Structural solver (beam loads & stress)


# Compute calculated aspect ratio from geometry (wingspan and planform area)
# Define design variables, constraints, and objective for optimization problem
# Compute dimensional Trefftz drag as objective (Node 0: cruise condition)
Di_Trefftz = CDi[0] * 0.5 * rho_array[0] * (cruise_speed**2) * planform_area
objective = Di_Trefftz
objective.set_as_objective(scaler=1.e1)

# L = W constraint (Node 0: cruise condition)
lift_trim = L[0] - W_total
lift_trim.set_as_constraint(equals=0.0, scaler=1.e-3)

# Pitch / Moment trim constraint: My = 0 about dynamic center of mass (x_cg)
pitch_moment = M[0, 1]
pitch_trim = pitch_moment
pitch_trim.set_as_constraint(equals=0.0, scaler=1.e-2)

# Static Margin constraint: SM >= 0.03 relative to the dynamic center of mass (x_cg)
# SM = (x_np - x_cg) / mean_chord = -dMy_cg / (dL * mean_chord)
mean_chord = planform_area / wingspan
dL_stab = L[1] - L[0]
dMy_stab = M[1, 1] - M[0, 1]
neutral_point_x = x_cg - dMy_stab / dL_stab
static_margin = (neutral_point_x - x_cg) / mean_chord
static_margin.set_as_constraint(lower=0.03, scaler=1.e1)

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
    additional_outputs=[Di, L, CL, CDi, Cp, panel_mesh, planform_area, aspect_ratio_calc, structural_mass, beam_displacement, beam_stress, elem_max_stress, dv_stresses, ttop_elem, tweb_elem, ttop_dvs, tweb_dvs, Di_Trefftz, W_total, M, CM, pitch_trim, static_margin, neutral_point_x, x_cg, x_payload, x_struct, r_cg, local_chord, local_height, box_width, beam_mesh, F_node] + geometry_coefficients,
    gpu=False
)

# Populate design variables with initial/warm-started values
for dv_info in design_variables.values():
    val = dv_info.variable.value
    if val is not None:
        jax_sim[dv_info.variable] = np.asarray(val)

jax_sim.run()
print(f"Final Lift (Cruise Node 0): {float(np.asarray(jax_sim[L]).flatten()[0]):.2f} N (Total Weight W: {float(np.asarray(jax_sim[W_total]).flatten()[0]):.2f} N)")
print(f"Final Pitching Moment (about CG): {float(np.asarray(jax_sim[M][0, 1]).flatten()[0]):.4f} N*m")
print(f"Final Center of Mass (x_cg): {float(np.asarray(jax_sim[x_cg]).flatten()[0]):.4f} m (Struct CG: {float(np.asarray(jax_sim[x_struct]).flatten()[0]):.4f} m, Payload: {float(np.asarray(jax_sim[x_payload]).flatten()[0]):.4f} m)")
print(f"Final Static Margin: {float(np.asarray(jax_sim[static_margin]).flatten()[0]):.4f} (Neutral Point: {float(np.asarray(jax_sim[neutral_point_x]).flatten()[0]):.4f} m, Margin Lever: {float(np.asarray(jax_sim[neutral_point_x] - jax_sim[x_cg]).flatten()[0]):.4f} m)")
print(f"Half-Beam Mass: {float(np.asarray(jax_sim[structural_mass]).flatten()[0])/2.0:.2f} kg (Full Structural Mass: {float(np.asarray(jax_sim[structural_mass]).flatten()[0]):.2f} kg)")

elem_stress_arr = np.asarray(jax_sim[elem_max_stress]).flatten()
dv_stress_arr = np.asarray(jax_sim[dv_stresses]).flatten()
chords_arr = np.asarray(jax_sim[local_chord]).flatten()
heights_arr = np.asarray(jax_sim[local_height]).flatten()
widths_arr = np.asarray(jax_sim[box_width]).flatten()
beam_pts = np.asarray(jax_sim[beam_mesh])
f_nodes_arr = np.asarray(jax_sim[F_node])
ttop_elem_arr = np.asarray(jax_sim[ttop_elem]).flatten()
tweb_elem_arr = np.asarray(jax_sim[tweb_elem]).flatten()
ttop_dv_arr = np.asarray(jax_sim[ttop_dvs]).flatten()
tweb_dv_arr = np.asarray(jax_sim[tweb_dvs]).flatten()

print(f"\n================ ELEMENT-BY-ELEMENT BEAM DIAGNOSTIC ({num_beam_elements} Elements, 2.5g Load Case) ================")
print(f"{'Elem':4s} | {'y_mid [m]':9s} | {'Chord [m]':9s} | {'Height [m]':10s} | {'ttop [mm]':9s} | {'2.5g Fz [N]':11s} | {'Max Stress [MPa]':16s}")
print("-" * 88)
y_elem_mid = 0.5 * (beam_pts[:-1, 1] + beam_pts[1:, 1])
for i in range(num_beam_elements):
    print(f"{i:4d} | {y_elem_mid[i]:9.3f} | {chords_arr[i]:9.3f} | {heights_arr[i]:10.4f} | {ttop_elem_arr[i]*1e3:9.2f} | {f_nodes_arr[i, 2]*2.5:11.2f} | {elem_stress_arr[i]/1e6:16.2f}")

print("\n================ 8 STRUCTURAL DV STATIONS (Fitted Stress Constraints) ================")
print(f"{'DV':3s} | {'eta':5s} | {'Span y [m]':10s} | {'ttop [mm]':9s} | {'tweb [mm]':9s} | {'Fitted Stress [MPa]':19s} | {'Yield Limit [MPa]':17s} | {'Status':8s}")
print("-" * 92)
eta_dvs = np.linspace(0.0, 1.0, num_thickness_stations)
y_dv_span = eta_dvs * 4.99
for j in range(num_thickness_stations):
    st_val = dv_stress_arr[j] / 1e6
    status = "FEASIBLE" if st_val <= 276.0 else "VIOLATED"
    print(f"{j:3d} | {eta_dvs[j]:5.2f} | {y_dv_span[j]:10.3f} | {ttop_dv_arr[j]*1e3:9.2f} | {tweb_dv_arr[j]*1e3:9.2f} | {st_val:19.2f} | {276.0:17.1f} | {status:8s}")

print("\nTrefftz Induced Drag:", float(np.asarray(jax_sim[Di_Trefftz]).flatten()[0]), "N")
exit(0) # Stop here for inspection
# endregion Optimization


# region Plot Optimization History
import pyvista as pv
import os, glob

# Find the latest output folder
output_base_dir = 'rectangular_wing_to_bwb_aerostructural_optimization_outputs'
output_folders = glob.glob(os.path.join(output_base_dir, '*'))
latest_folder = max(output_folders, key=os.path.getmtime)
print(f"Reading optimization history from: {latest_folder}")

# Read design variable history from x.out (preferred) or record.hdf5
x_out_path = os.path.join(latest_folder, 'x.out')
if os.path.exists(x_out_path):
    x_history = np.loadtxt(x_out_path)
    if len(x_history.shape) == 1:
        x_history = x_history.reshape(1, -1)
    print(f"Loaded {x_history.shape[0]} iterations from current run x.out")

    # If warm-started from prior file, prepend all previous iterations so video & summary show the full trajectory
    if warm_start and os.path.exists(init_file):
        x_prior = np.loadtxt(init_file)
        if len(x_prior.shape) > 1 and x_prior.shape[0] > 0:
            x_history = np.vstack([x_prior[:-1], x_history])
            print(f"Combined total: {x_history.shape[0]} iterations from initial rectangular wing to converged optimum")
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
    cl_val = float(np.asarray(jax_sim[CL]).flatten()[0])
    cd_val = float(np.asarray(jax_sim[CDi]).flatten()[0])
    sref_val = float(np.asarray(jax_sim[planform_area]).flatten()[0])

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
        ar_val = float(np.asarray(unscaled_values['aspect_ratio']).flatten()[0]) if 'aspect_ratio' in unscaled_values else 10.0
        t_vals = unscaled_values['taper_dvs'] if 'taper_dvs' in unscaled_values else np.ones(num_chord_stations - 1)
        c_vals = [1.0] + list(t_vals)
        c_str = " ".join([f"t{i}={c_vals[i]:.2f}" for i in range(len(c_vals))])
        sw_vals = unscaled_values['sweep_angle_dvs'] if 'sweep_angle_dvs' in unscaled_values else np.zeros(num_chord_stations - 1)
        sw_str = " ".join([f"sw{i}={np.degrees(sw_vals[i]):.1f}°" for i in range(len(sw_vals))])
        dv_str = f"AR={ar_val:.2f}  {c_str}\n{sw_str}"
    elif formulation == 'chord_span':
        ss_val = float(np.asarray(unscaled_values['span_stretch_dv']).flatten()[0]) if 'span_stretch_dv' in unscaled_values else 0.0
        cs_vals = unscaled_values['chord_stretch_dvs'] if 'chord_stretch_dvs' in unscaled_values else np.zeros(num_chord_stations)
        c_str = " ".join([f"c{i}={1.0+cs_vals[i]:.2f}" for i in range(len(cs_vals))])
        sw_vals = unscaled_values['sweep_dvs'] if 'sweep_dvs' in unscaled_values else np.zeros(num_chord_stations)
        sw_str = " ".join([f"sw{i}={sw_vals[i]:.2f}" for i in range(len(sw_vals))])
        dv_str = f"b_stretch={ss_val:.2f}  {c_str}\n{sw_str}"

    pitch_val = float(np.asarray(unscaled_values['pitch']).flatten()[0]) if 'pitch' in unscaled_values else 0.0
    elev_val = float(np.asarray(unscaled_values['elevator_angle']).flatten()[0]) if 'elevator_angle' in unscaled_values else 0.0
    sm_val = float(np.asarray(jax_sim[static_margin]).flatten()[0])
    xnp_val = float(np.asarray(jax_sim[neutral_point_x]).flatten()[0])
    xcg_val = float(np.asarray(jax_sim[x_cg]).flatten()[0])

    # Add iteration counter label using unscaled physical values
    plotter.add_text(
        f"Iteration {iteration}/{num_iterations - 1}\n"
        f"Formulation: {formulation}\n"
        f"{dv_str}\n"
        f"Elevator={np.degrees(elev_val):.1f}°  Pitch={np.degrees(pitch_val):.1f}°\n"
        f"SM={sm_val:.4f} (x_cg={xcg_val:.3f}m, x_np={xnp_val:.3f}m)",
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

import shutil
artifact_dir = '/home/andrew/.gemini/antigravity/brain/6430b318-3da2-48b6-9610-a3a9a37d089f'
if os.path.exists(artifact_dir):
    if os.path.exists(video_path):
        shutil.copy2(video_path, os.path.join(artifact_dir, 'optimization_history.mp4'))
    if os.path.exists(summary_fig_path):
        shutil.copy2(summary_fig_path, os.path.join(artifact_dir, 'optimization_summary.png'))
    if os.path.exists(wing_img_path):
        shutil.copy2(wing_img_path, os.path.join(artifact_dir, 'final_wing.png'))
    print("Artifacts successfully copied to brain artifact directory!")

# endregion Plot Summary Figure
# endregion Plot Optimization History