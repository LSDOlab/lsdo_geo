# region Imports and Setup

from dataclasses import dataclass
from typing import Union
import numpy.typing as npt
import csdl_alpha as csdl
import numpy as np
import lsdo_function_spaces as lfs

import lsdo_geo
from lsdo_geo import (
    import_geometry,
    rotate,
    construct_ffd_block_around_entities,
    SectionalParameterization,
    SectionalParameters,
    ParameterizationSolver,
    GeometricVariables,
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

# Resolution toggle: 'fast' (5 spanwise DVs, coarse mesh ~1,272 quads) vs 'full' (8 spanwise DVs, refined mesh 2,872 quads)
resolution = 'fast'  # Options: 'fast' or 'full'
# resolution = 'full'  # Options: 'fast' or 'full'

if resolution == 'fast':
    num_stations = 5
    num_ffd_sections = 2 * num_stations - 1  # 9 FFD sections across full span
    mesh_file_name = "rectangular_wing_naca0012_10ar"  # Coarser mesh (1,272 quads) for rapid testing
elif resolution == 'full':
    num_stations = 8
    num_ffd_sections = 2 * num_stations - 1  # 15 FFD sections across full span
    mesh_file_name = "rectangular_wing_naca0012_35sect"  # Refined mesh (2,872 quads)
else:
    raise ValueError(f"Unknown resolution: {resolution}. Must be 'fast' or 'full'.")

num_chord_stations = num_stations
num_thickness_stations = num_stations

# Sizing condition toggle: -1.0g push-down maneuver
# When False, drops flight condition Node 3, pitch_neg1g DV, lift_neg1g trim constraint,
# and beam_neg1g FEA solve, accelerating optimization iterations by ~20-25%.
include_neg1g_sizing = False

# Geometric CAD control points are kept equivalent across resolutions (15 spanwise)
num_spanwise_cp_target = 15

geometry = import_geometry(
    geometry_directory + file_name + ".stp",
    name='imported_geometry',
    parallelize=False,
)

# Scale CAD geometry from baseline 10 m^2 wing to 1.0 m^2 tactical UAV wing
# scale_factor = 1.0 / np.sqrt(10.0)  # = 0.31622776601683794

# Scale CAD geometry from baseline 10 m span to 50 m span full-scale BWB size
scale_factor = 7.5

for function in geometry.functions.values():
    function.coefficients = function.coefficients * scale_factor

# Add more spanwise control points to each geometry function via linear interpolation.
# Since this is a rectangular wing (uniform cross-section), linearly interpolating
# control points along the spanwise axis is exact and preserves the chordwise resolution.
# Each function's spanwise axis is detected independently since B-splines may be oriented differently.

for idx in list(geometry.functions.keys()):
    function = geometry.functions[idx]
    coeffs = function.coefficients.value if hasattr(function.coefficients, 'value') else function.coefficients

    # Determine which axis is spanwise by checking y-coordinate range along each axis
    axis0_y_range = np.ptp([coeffs[i, :, 1].mean() for i in range(coeffs.shape[0])])
    axis1_y_range = np.ptp([coeffs[:, j, 1].mean() for j in range(coeffs.shape[1])])

    # Skip tip caps and other non-spanwise surfaces (y-extent < 1.0 * scale_factor)
    if max(axis0_y_range, axis1_y_range) < 1.0 * scale_factor:
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
    geometry.space.spaces[idx] = new_space
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

leading_edge_left = geometry.project(np.array([0.0, -5.0 * scale_factor, 0.0]))
leading_edge_right = geometry.project(np.array([0.0, 5.0 * scale_factor, 0.0]))
trailing_edge_left = geometry.project(np.array([1.0 * scale_factor, -5.0 * scale_factor, 0.0]))
trailing_edge_right = geometry.project(np.array([1.0 * scale_factor, 5.0 * scale_factor, 0.0]))
leading_edge_center = geometry.project(np.array([0.0, 0.0, 0.0]))
trailing_edge_center = geometry.project(np.array([1.0 * scale_factor, 0.0, 0.0]))
quarter_chord_left = geometry.project(np.array([0.25 * scale_factor, -5.0 * scale_factor, 0.0]))
quarter_chord_right = geometry.project(np.array([0.25 * scale_factor, 5.0 * scale_factor, 0.0]))
quarter_chord_center = geometry.project(np.array([0.25 * scale_factor, 0.0, 0.0]))
elevator_hinge = geometry.project(np.array([0.80 * scale_factor, 0.0, 0.0]))

# Project chord measurement points across the right half-span (y = 0.0 to 5.0 * scale_factor)
chord_station_y = np.linspace(0.0, 5.0 * scale_factor, num_chord_stations)
chord_le_projections = []
chord_te_projections = []
quarter_chord_projections = []
for y_val in chord_station_y:
    chord_le_projections.append(geometry.project(np.array([0.0, y_val, 0.0])))
    chord_te_projections.append(geometry.project(np.array([1.0 * scale_factor, y_val, 0.0])))
    quarter_chord_projections.append(geometry.project(np.array([0.25 * scale_factor, y_val, 0.0])))

# Project max thickness measurement points at x=0.3 for the 8 stations
upper_thickness_projections = []
lower_thickness_projections = []
for y_val in chord_station_y:
    upper_thickness_projections.append(geometry.project(np.array([0.3 * scale_factor, y_val, 0.05 * scale_factor]), direction=np.array([0, 0, -1])))
    lower_thickness_projections.append(geometry.project(np.array([0.3 * scale_factor, y_val, -0.05 * scale_factor]), direction=np.array([0, 0, 1])))

# Also project symmetric (left half) chord measurement points for the solver
chord_le_projections_left = []
chord_te_projections_left = []
for y_val in chord_station_y[1:]:  # skip root (index 0) since it's already at center
    chord_le_projections_left.append(geometry.project(np.array([0.0, -y_val, 0.0])))
    chord_te_projections_left.append(geometry.project(np.array([1.0 * scale_factor, -y_val, 0.0])))

# Project wireframes along upper and lower wing skins to compute chord surface for planform area
nx_area = 21  # chordwise grid resolution
ny_area = 41  # spanwise grid resolution
x_grid = np.linspace(0.0, 1.0 * scale_factor, nx_area)
y_grid = np.linspace(-5.0 * scale_factor, 5.0 * scale_factor, ny_area)
X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid, indexing='ij')

upper_seed_pts = np.column_stack([X_mesh.ravel(), Y_mesh.ravel(), np.full(X_mesh.size, 0.05 * scale_factor)])
lower_seed_pts = np.column_stack([X_mesh.ravel(), Y_mesh.ravel(), np.full(X_mesh.size, -0.05 * scale_factor)])

projected_upper_skin = geometry.project(upper_seed_pts, force_reprojection=False, direction=np.array([0, 0, -1]), plot=False)
projected_lower_skin = geometry.project(lower_seed_pts, force_reprojection=False, direction=np.array([0, 0, 1]), plot=False)

# Project line of beam nodes along the span at 25% chord for right half-span (y >= 0)
num_beam_nodes = 15
y_beam_span = np.linspace(0.0, 4.99 * scale_factor, num_beam_nodes)
beam_line_physical = np.zeros((num_beam_nodes, 3))
beam_line_physical[:, 0] = 0.25 * scale_factor  # 25% chord elastic axis
beam_line_physical[:, 1] = y_beam_span
beam_line_physical[:, 2] = 0.0

projected_beam_mesh = geometry.project(beam_line_physical, plot=False)

le_line_physical = np.zeros((num_beam_nodes, 3))
le_line_physical[:, 0] = 0.00
le_line_physical[:, 1] = y_beam_span
le_line_physical[:, 2] = 0.0
projected_le_mesh = geometry.project(le_line_physical, plot=False)

te_line_physical = np.zeros((num_beam_nodes, 3))
te_line_physical[:, 0] = 1.00 * scale_factor
te_line_physical[:, 1] = y_beam_span
te_line_physical[:, 2] = 0.0
projected_te_mesh = geometry.project(te_line_physical, plot=False)

upper_beam_seed = np.column_stack([np.full(num_beam_nodes, 0.25 * scale_factor), y_beam_span, np.full(num_beam_nodes, 0.05 * scale_factor)])
lower_beam_seed = np.column_stack([np.full(num_beam_nodes, 0.25 * scale_factor), y_beam_span, np.full(num_beam_nodes, -0.05 * scale_factor)])
projected_upper_beam_mesh = geometry.project(upper_beam_seed, direction=np.array([0, 0, -1]), plot=False)
projected_lower_beam_mesh = geometry.project(lower_beam_seed, direction=np.array([0, 0, 1]), plot=False)

# Project 51 points along leading and trailing edges for 50-strip drag & geometry-based alpha
num_drag_strips = 50
num_drag_nodes = num_drag_strips + 1
y_drag_span = np.linspace(0.0, 4.99 * scale_factor, num_drag_nodes)

le_drag_physical = np.zeros((num_drag_nodes, 3))
le_drag_physical[:, 0] = 0.00
le_drag_physical[:, 1] = y_drag_span
le_drag_physical[:, 2] = 0.0
projected_le_drag_mesh = geometry.project(le_drag_physical, plot=False)

te_drag_physical = np.zeros((num_drag_nodes, 3))
te_drag_physical[:, 0] = 1.00 * scale_factor
te_drag_physical[:, 1] = y_drag_span
te_drag_physical[:, 2] = 0.0
projected_te_drag_mesh = geometry.project(te_drag_physical, plot=False)
# endregion

# region Mesh definitions
mesh = meshio.read(geometry_directory + mesh_file_name + ".msh")
mesh.points = mesh.points * scale_factor

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
                            projection_tolerance=1.e-1,
                            use_line_search=True,
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
# Camber formulation option: adds 3x5 (fast) or 3x8 (full) FFD camber DVs
include_camber = True  # Options: True or False

# Elevator formulation option: default off when camber is on
include_elevator = False if include_camber else True  # Options: True or False

# Construct a Free Form Deformation (FFD) block around the geometry
# 5 chordwise control points when camber is active (excluding LE & TE gives 3 interior points)
num_ffd_coefficients_chordwise = 5 if include_camber else 2
ffd_degree_chordwise = 2 if include_camber else 1
# num_ffd_sections is dynamically set by resolution ('fast': 9, 'full': 15)
# Note: This FFD block construction is one of a few helper functions that can be used to create a FFD block.
#       The "manual" method is to use construct_ffd_block_from_corners, which allows for defining the coefficients directly.
ffd_block = construct_ffd_block_around_entities(entities=geometry, 
                                                num_coefficients=(num_ffd_coefficients_chordwise, num_ffd_sections, 2), degree=(ffd_degree_chordwise, 3, 1))
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
# 'ar_area'   -> Design variables: chord DVs, Aspect Ratio (AR), and pitch (planform area fixed at 10)
# 'chord_span' -> Design variables: chord stretch DVs, sweep DVs, span stretch DV, elevator angle, pitch, pitch_3g
formulation = 'ar_area'  # Options: 'ar_area' or 'chord_span'

pitch = csdl.Variable(value=5.*np.pi/180) # pitch angle in radians
elevator_angle = csdl.Variable(value=0.0) # elevator deflection angle in radians
pitch_ss = csdl.Variable(value=10.0*np.pi/180) # pitch angle for pull-up structural sizing maneuver in radians
if include_neg1g_sizing:
    pitch_neg1g = csdl.Variable(value=-5.0*np.pi/180) # pitch angle for -1.0g push-down sizing maneuver in radians
payload_cg = csdl.Variable(value=0.40) # payload CG location as fraction of root chord (0.05 to 0.95)

@dataclass
class DVInfo:
    variable: csdl.Variable
    lower: Union[float, npt.NDArray[np.float64]]
    upper: Union[float, npt.NDArray[np.float64]]
    scaler: float = 1.0

init_file = 'rectangular_wing_to_bwb_aerostructural_optimization_outputs/2026-09-09_12.44.58.866121/x.out'

thickness_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=2, coefficients_shape=(num_thickness_stations,))
ttop_dvs = csdl.Variable(shape=(num_thickness_stations,), value=np.ones(num_thickness_stations) * 0.001)
tweb_dvs = csdl.Variable(shape=(num_thickness_stations,), value=np.ones(num_thickness_stations) * 0.001)

twist_lower = np.full(num_chord_stations, -15.0 * np.pi / 180.0)
twist_lower[0] = 0.0  # fix root twist to 0
twist_upper = np.full(num_chord_stations, 15.0 * np.pi / 180.0)
twist_upper[0] = 0.0

camber_max_percent = 5.0  # 5.0% chord max camber displacement
camber_lower = -camber_max_percent
camber_upper = camber_max_percent
camber_scaler = 1.0 / camber_max_percent  # Scales DVs in [-5.0, 5.0] to [-1, 1] range for optimizer
warm_start = False

if formulation == 'ar_area':
    # Formulation 1: Taper ratio DVs (stations 1 to num_stations-1) + Aspect Ratio (AR) + Sectional Sweep Angles + Elevator + Pitch + Pitch 2.5g + Linear Twist
    taper_dvs = csdl.Variable(shape=(num_chord_stations - 1,), value=np.ones(num_chord_stations - 1))
    aspect_ratio = csdl.Variable(shape=(1,), value=np.array([10.0]))
    sweep_angle_dvs = csdl.Variable(shape=(num_chord_stations - 1,), value=np.zeros(num_chord_stations - 1))
    twist_dvs = csdl.Variable(shape=(num_chord_stations,), value=np.zeros(num_chord_stations))

    design_variables: dict[str, DVInfo] = {
        'taper_dvs': DVInfo(variable=taper_dvs, lower=0.10, upper=5.0, scaler=2.0),
        'aspect_ratio': DVInfo(variable=aspect_ratio, lower=2.0, upper=15.0, scaler=0.5),
        # 'sweep_angle_dvs': DVInfo(variable=sweep_angle_dvs, lower=-10.0*np.pi/180, upper=45.0*np.pi/180, scaler=1.e1),
        'sweep_angle_dvs': DVInfo(variable=sweep_angle_dvs, lower=0.0*np.pi/180, upper=70.0*np.pi/180, scaler=1.e1),
        'twist_dvs': DVInfo(variable=twist_dvs, lower=twist_lower, upper=twist_upper, scaler=1.e1),
        'pitch': DVInfo(variable=pitch, lower=-10.0*np.pi/180, upper=15.0*np.pi/180, scaler=1.e1),
        'pitch_ss': DVInfo(variable=pitch_ss, lower=0.0*np.pi/180, upper=35.0*np.pi/180, scaler=1.e1),
        'payload_cg': DVInfo(variable=payload_cg, lower=0.05, upper=0.95, scaler=1.e1),
        'ttop_dvs': DVInfo(variable=ttop_dvs, lower=0.0001, upper=0.1, scaler=5.e3),
        'tweb_dvs': DVInfo(variable=tweb_dvs, lower=0.0001, upper=0.1, scaler=5.e3),
    }
    if include_elevator:
        design_variables['elevator_angle'] = DVInfo(variable=elevator_angle, lower=-25.0*np.pi/180, upper=25.0*np.pi/180, scaler=1.e1)
    if include_camber:
        camber_dvs = csdl.Variable(shape=(3, num_chord_stations), value=np.zeros((3, num_chord_stations)))
        design_variables['camber_dvs'] = DVInfo(variable=camber_dvs, lower=camber_lower, upper=camber_upper, scaler=camber_scaler)
    if include_neg1g_sizing:
        design_variables['pitch_neg1g'] = DVInfo(variable=pitch_neg1g, lower=-35.0*np.pi/180, upper=10.0*np.pi/180, scaler=1.e1)

    # ParameterizationSolver drives chord stretch states, thickness stretch states, 1 span stretch state, and sweep translation states
    chord_stretch_states = csdl.Variable(shape=(num_chord_stations,), value=np.zeros(num_chord_stations))
    thickness_stretch_states = csdl.Variable(shape=(num_chord_stations,), value=np.zeros(num_chord_stations))
    span_stretch_state = csdl.Variable(value=0.0)
    sweep_translation_states = csdl.Variable(shape=(num_chord_stations - 1,), value=np.zeros(num_chord_stations - 1))

    sweep_full_half = csdl.concatenate([csdl.Variable(value=0.0), sweep_translation_states])

    chord_params = csdl.concatenate(
        [chord_stretch_states[i] for i in range(num_chord_stations - 1, 0, -1)] +
        [chord_stretch_states[i] for i in range(num_chord_stations)]
    )
    sweep_params = csdl.concatenate(
        [sweep_full_half[i] for i in range(num_chord_stations - 1, 0, -1)] +
        [sweep_full_half[i] for i in range(num_chord_stations)]
    )
    thickness_params = csdl.concatenate(
        [thickness_stretch_states[i] for i in range(num_chord_stations - 1, 0, -1)] +
        [thickness_stretch_states[i] for i in range(num_chord_stations)]
    )
    twist_params = csdl.concatenate(
        [twist_dvs[i] for i in range(num_chord_stations - 1, 0, -1)] +
        [twist_dvs[i] for i in range(num_chord_stations)]
    )
    span_weights = np.linspace(-1.0, 1.0, num_ffd_sections)
    span_params = csdl.expand(span_stretch_state, (num_ffd_sections,)) * span_weights

elif formulation == 'chord_span':
    # Formulation 2: chord stretch DVs, sweep DVs, thickness stretch DVs, linear twist DV, span stretch DV, elevator angle, pitch, pitch_ss
    init_chord = np.zeros(num_chord_stations)
    init_sweep = np.zeros(num_chord_stations)
    init_thick = np.zeros(num_chord_stations)
    init_twist = np.zeros(num_chord_stations)
    init_span = np.array([0.0])
    init_elev = 0.0
    init_pitch = 5.0 * np.pi / 180.0
    init_pitch_ss = 10.0 * np.pi / 180.0
    if include_neg1g_sizing:
        init_pitch_neg1g = -5.0 * np.pi / 180.0
    init_ttop_val = np.ones(num_thickness_stations) * 0.001
    init_tweb_val = np.ones(num_thickness_stations) * 0.001

    warm_start = False

    chord_stretch_dvs = csdl.Variable(shape=(num_chord_stations,), value=init_chord)
    sweep_dvs = csdl.Variable(shape=(num_chord_stations,), value=init_sweep)
    thickness_stretch_dvs = csdl.Variable(shape=(num_chord_stations,), value=init_thick)
    twist_dvs = csdl.Variable(shape=(num_chord_stations,), value=init_twist)
    span_stretch_dv = csdl.Variable(shape=(1,), value=init_span)
    pitch.value = init_pitch
    elevator_angle.value = init_elev
    pitch_ss.value = init_pitch_ss
    if include_neg1g_sizing:
        pitch_neg1g.value = init_pitch_neg1g
    ttop_dvs.value = init_ttop_val
    tweb_dvs.value = init_tweb_val

    initial_chord = 1.0 * scale_factor  # 0.3162 m baseline chord
    chord_stretch_lower = -0.9 * initial_chord  # -0.3004 m (prevents chord collapsing below 5% of baseline)
    chord_stretch_upper = 4.0 * initial_chord    # 1.2649 m

    initial_thickness = 0.12 * initial_chord  # 0.0379 m baseline NACA 0012 thickness
    thickness_stretch_lower = -0.95 * initial_thickness
    thickness_stretch_upper = 4.0 * initial_thickness    # 0.1518 m

    sweep_lower = np.full(num_chord_stations, -0.5 * scale_factor)
    sweep_lower[0] = 0.0
    sweep_upper = np.full(num_chord_stations, 4.0 * scale_factor)
    sweep_upper[0] = 0.0

    span_stretch_lower = -4.5 * scale_factor
    span_stretch_upper = 20.0 * scale_factor

    design_variables: dict[str, DVInfo] = {
        'chord_stretch_dvs': DVInfo(variable=chord_stretch_dvs, lower=chord_stretch_lower, upper=chord_stretch_upper, scaler=1.0 / scale_factor),
        'sweep_dvs': DVInfo(variable=sweep_dvs, lower=sweep_lower, upper=sweep_upper, scaler=1.0 / scale_factor),
        'thickness_stretch_dvs': DVInfo(variable=thickness_stretch_dvs, lower=thickness_stretch_lower, upper=thickness_stretch_upper, scaler=1.0 / initial_thickness),
        'twist_dvs': DVInfo(variable=twist_dvs, lower=twist_lower, upper=twist_upper, scaler=1.e1),
        'span_stretch_dv': DVInfo(variable=span_stretch_dv, lower=span_stretch_lower, upper=span_stretch_upper, scaler=1.0 / scale_factor),
        'pitch': DVInfo(variable=pitch, lower=-10.0*np.pi/180, upper=15.0*np.pi/180, scaler=1.e1),
        'pitch_ss': DVInfo(variable=pitch_ss, lower=0.0*np.pi/180, upper=35.0*np.pi/180, scaler=1.e1),
        'payload_cg': DVInfo(variable=payload_cg, lower=0.05, upper=0.95, scaler=1.e1),
        'ttop_dvs': DVInfo(variable=ttop_dvs, lower=0.0001, upper=0.05, scaler=5.e3),
        'tweb_dvs': DVInfo(variable=tweb_dvs, lower=0.0001, upper=0.05, scaler=5.e3),
    }
    if include_elevator:
        design_variables['elevator_angle'] = DVInfo(variable=elevator_angle, lower=-25.0*np.pi/180, upper=25.0*np.pi/180, scaler=1.e1)
    if include_camber:
        camber_dvs = csdl.Variable(shape=(3, num_chord_stations), value=np.zeros((3, num_chord_stations)))
        design_variables['camber_dvs'] = DVInfo(variable=camber_dvs, lower=camber_lower, upper=camber_upper, scaler=camber_scaler)
    if include_neg1g_sizing:
        design_variables['pitch_neg1g'] = DVInfo(variable=pitch_neg1g, lower=-35.0*np.pi/180, upper=10.0*np.pi/180, scaler=1.e1)

    chord_params = csdl.concatenate(
        [chord_stretch_dvs[i] for i in range(num_chord_stations - 1, 0, -1)] +
        [chord_stretch_dvs[i] for i in range(num_chord_stations)]
    )
    sweep_params = csdl.concatenate(
        [sweep_dvs[i] for i in range(num_chord_stations - 1, 0, -1)] +
        [sweep_dvs[i] for i in range(num_chord_stations)]
    )
    thickness_params = csdl.concatenate(
        [thickness_stretch_dvs[i] for i in range(num_chord_stations - 1, 0, -1)] +
        [thickness_stretch_dvs[i] for i in range(num_chord_stations)]
    )
    twist_params = csdl.concatenate(
        [twist_dvs[i] for i in range(num_chord_stations - 1, 0, -1)] +
        [twist_dvs[i] for i in range(num_chord_stations)]
    )
    span_weights = np.linspace(-1.0, 1.0, num_ffd_sections)
    span_params = csdl.expand(span_stretch_dv, (num_ffd_sections,)) * span_weights

sectional_parameters = SectionalParameters()
sectional_parameters.add_stretch(axis=np.array([1., 0., 0.]), stretch=chord_params)
sectional_parameters.add_translation(axis=np.array([1., 0., 0.]), translation=sweep_params)
sectional_parameters.add_stretch(axis=np.array([0., 0., 1.]), stretch=thickness_params)
sectional_parameters.add_translation(axis=np.array([0., 1., 0.]), translation=span_params)
sectional_parameters.add_rotation(axis=np.array([0., 1., 0.]), rotation=twist_params, parametric_coordinate=np.array([0.25, 0.5]))

ffd_coefficients = ffd_sectional_parameterization.evaluate(sectional_parameters, plot=False)

if include_camber:
    # Section chord calculated from difference in x coordinate between leading and trailing FFD control points
    section_chords = ffd_coefficients[-1, :, 0, 0] - ffd_coefficients[0, :, 0, 0]

    full_span_camber_list = []
    for c in range(3):
        row = csdl.concatenate(
            [camber_dvs[c, i] for i in range(num_chord_stations - 1, 0, -1)] +
            [camber_dvs[c, i] for i in range(num_chord_stations)]
        )
        full_span_camber_list.append(csdl.reshape(row, (1, num_ffd_sections)))
    full_span_camber = csdl.concatenate(full_span_camber_list, axis=0)  # shape (3, num_ffd_sections)

    # Convert chord percentage to physical vertical displacement for each section
    camber_displacement = (full_span_camber / 100.0) * csdl.expand(section_chords, (3, num_ffd_sections), 'j->ij')
    camber_delta = csdl.expand(camber_displacement, (3, num_ffd_sections, 2), 'ij->ijk')
    ffd_coefficients = ffd_coefficients.set(csdl.slice[1:4, :, :, 2], ffd_coefficients[1:4, :, :, 2] + camber_delta)

geometry_coefficients = ffd_block.evaluate_ffd(coefficients=ffd_coefficients, plot=False)
geometry.set_coefficients(geometry_coefficients) # type: ignore

if include_elevator:
    # Apply elevator deflection to the back 20% of chord and middle quarter of wing
    # Functions 0 & 5: lower surfaces, trailing edge is at rows 0:6 (x in [0.789, 1.0])
    # Functions 1 & 4: upper surfaces, trailing edge is at rows 97:103 (x in [0.789, 1.0])
    # Spanwise columns 0:4 cover |y| <= 1.07 m (~21.4% of span, matching middle quarter)
    hinge_origin = geometry.evaluate(elevator_hinge)
    for f_idx, row_slc in [(0, slice(0, 6)), (5, slice(0, 6)), (1, slice(97, None)), (4, slice(97, None))]:
        func = geometry.functions[f_idx]
        sub_pts = func.coefficients[:4, row_slc, :]
        rot_sub = rotate(
            points=sub_pts,
            rotation_origin=hinge_origin,
            axis_vector=np.array([0., 1., 0.]),
            angles=-elevator_angle,
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
    
    # Enforce constant thickness-to-chord ratio = 0.12 at all stations (normalized by 0.12)
    for i in range(num_chord_stations):
        tc_ratio = local_thicknesses[i] / local_chords[i]
        geometric_variables.add_variable(tc_ratio / 0.12, 1.0, penalty_value=None)
    
    # Enforce planform area and aspect ratio simultaneously (AR normalized by reference 10.0)
    geometric_variables.add_variable(planform_area / (scale_factor**2), planform_area.value / (scale_factor**2), penalty_value=None)
    geometric_variables.add_variable(aspect_ratio_calc / 10.0, aspect_ratio / 10.0, penalty_value=None)

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
if scale_factor == 1.0 or scale_factor == 1.0/np.sqrt(10.0):
    cruise_speed = csdl.Variable(value=20.)
    sizing_speed = 1.5 * cruise_speed  # 30.0 m/s (1.5x cruise speed) structural sizing maneuver speed
    if scale_factor == 1.0/np.sqrt(10.0):
        # Fixed positive payload weight of 100.0 N (10.19 kg tactical UAV payload)
        payload_weight = csdl.Variable(value=100.0)
        load_factor = csdl.Variable(value=4.0)  # 4.0g load factor for sizing maneuver
    elif scale_factor == 1.0:
        # Fixed positive payload weight of 1000.0 N
        payload_weight = csdl.Variable(value=1000.0)
        load_factor = csdl.Variable(value=3.0)  # 3.0g load factor for sizing maneuver
elif scale_factor == 7.5:
    cruise_speed = csdl.Variable(value=140.)    # This is matching normal dynamic pressure without going to altitude
    sizing_speed = 1.25 * cruise_speed  # (1.25x cruise speed) structural sizing maneuver speed
    payload_weight = csdl.Variable(value=160000.*4.44822)  # 711,86 kN = 160,000 lbf (4.44822 N/lbf) payload weight
    load_factor = csdl.Variable(value=2.5)  # 2.5g load factor for sizing maneuver
else:
    raise Exception("Cruise speed not defined for scale factor = {}. Set the cruise speed for this scale factor.".format(scale_factor))

load_factor_val = float(np.asarray(load_factor.value).flatten()[0]) if hasattr(load_factor, 'value') else float(load_factor)

# region Aerodynamic solver (panel method)
# Flight conditions:
# Node 0 = cruise condition (20 m/s)
# Node 1 = stability condition (+1.0 deg alpha perturbation, 20 m/s)
# Node 2 = structural sizing pull-up condition (sizing_speed, pitch angle = pitch_ss)
# Node 3 = -1.0g push-down structural sizing condition (optional, pitch angle = pitch_neg1g)
if include_neg1g_sizing:
    num_nodes = 4
    dalpha_rad = 1.0 * np.pi / 180.0
    dalpha_ss = pitch_ss - pitch
    dalpha_neg1g = pitch_neg1g - pitch
    v0 = csdl.concatenate([cruise_speed, csdl.Variable(value=0.0), csdl.Variable(value=0.0)])
    v1 = csdl.concatenate([cruise_speed * np.cos(dalpha_rad), csdl.Variable(value=0.0), cruise_speed * np.sin(dalpha_rad)])
    v2 = csdl.concatenate([sizing_speed * csdl.cos(dalpha_ss), csdl.Variable(value=0.0), sizing_speed * csdl.sin(dalpha_ss)])
    v3 = csdl.concatenate([sizing_speed * csdl.cos(dalpha_neg1g), csdl.Variable(value=0.0), sizing_speed * csdl.sin(dalpha_neg1g)])
    v_stacked = csdl.reshape(csdl.concatenate([v0, v1, v2, v3]), (4, 3))
    rho_array = csdl.Variable(shape=(num_nodes,), value=np.array([1.225, 1.225, 1.225, 1.225]))
    sos_array = csdl.Variable(shape=(num_nodes,), value=np.array([343.0, 343.0, 343.0, 343.0]))
else:
    num_nodes = 3
    dalpha_rad = 1.0 * np.pi / 180.0
    dalpha_ss = pitch_ss - pitch
    v0 = csdl.concatenate([cruise_speed, csdl.Variable(value=0.0), csdl.Variable(value=0.0)])
    v1 = csdl.concatenate([cruise_speed * np.cos(dalpha_rad), csdl.Variable(value=0.0), cruise_speed * np.sin(dalpha_rad)])
    v2 = csdl.concatenate([sizing_speed * csdl.cos(dalpha_ss), csdl.Variable(value=0.0), sizing_speed * csdl.sin(dalpha_ss)])
    v_stacked = csdl.reshape(csdl.concatenate([v0, v1, v2]), (3, 3))
    rho_array = csdl.Variable(shape=(num_nodes,), value=np.array([1.225, 1.225, 1.225]))
    sos_array = csdl.Variable(shape=(num_nodes,), value=np.array([343.0, 343.0, 343.0]))

panel_mesh = geometry.evaluate(projected_panel_mesh, plot=False)
panel_mesh = panel_mesh.expand((1,) + panel_mesh.shape, 'ij->aij')
point_velocities = csdl.expand(v_stacked, (num_nodes,) + panel_mesh.shape[1:], 'ij->iaj')

# region Structural beam model geometry and Center of Mass calculation
beam_mesh = geometry.evaluate(projected_beam_mesh, plot=False)
le_mesh = geometry.evaluate(projected_le_mesh, plot=False)
te_mesh = geometry.evaluate(projected_te_mesh, plot=False)
upper_beam_mesh = geometry.evaluate(projected_upper_beam_mesh, plot=False)
lower_beam_mesh = geometry.evaluate(projected_lower_beam_mesh, plot=False)
le_drag_mesh = geometry.evaluate(projected_le_drag_mesh, plot=False)
te_drag_mesh = geometry.evaluate(projected_te_drag_mesh, plot=False)

# Compute local chord from projected mesh (leading edge to trailing edge distance)
node_chords = te_mesh[:,0] - le_mesh[:,0]
local_chord = 0.5 * (node_chords[:-1] + node_chords[1:])

node_heights = upper_beam_mesh[:, 2] - lower_beam_mesh[:, 2]
local_height = 0.5 * (node_heights[:-1] + node_heights[1:])

# Define wingbox cross-section along the span
# Wingbox width is 40% of local chord; height is 50% of local maximum thickness
box_width = 0.40 * local_chord
box_height = 0.50 * local_height

# Evaluate B-spline thickness parameterization at element midpoints
num_beam_elements = num_beam_nodes - 1
y_elem_np = 0.5 * (y_beam_span[:-1] + y_beam_span[1:])
y_norm_elem = (y_elem_np / (4.99 * scale_factor)).reshape((-1, 1))

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

# Compute payload location: payload_cg fraction of root chord (y = 0.0, node 0)
x_le_root = le_mesh[0, 0]
x_te_root = te_mesh[0, 0]
root_chord = x_te_root - x_le_root
x_payload = x_le_root + payload_cg * root_chord
y_payload = csdl.Variable(value=np.array([0.0]))
z_payload = 0.5 * (upper_beam_mesh[0, 2] + lower_beam_mesh[0, 2])

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
panel_forces_right_cruise = outputs['panel_forces'][0, :num_right_panels, :] # shape (num_right_panels, 3) from Node 0 (cruise)
panel_forces_right_ss = outputs['panel_forces'][2, :num_right_panels, :] # shape (num_right_panels, 3) from Node 2 (structural sizing pull-up)
if include_neg1g_sizing:
    panel_forces_right_neg1g = outputs['panel_forces'][3, :num_right_panels, :] # shape (num_right_panels, 3) from Node 3 (-1.0g push-down)

W_var = csdl.Variable(value=W_matrix)

# 1. Force & Moment Mapping for Structural Sizing Case
F_node = csdl.matmat(W_var, panel_forces_right_ss)

B_expand = csdl.expand(beam_mesh, (num_beam_nodes, num_right_panels, 3), 'nj->nij')
C_expand = csdl.expand(dynamic_panel_centers_right, (num_beam_nodes, num_right_panels, 3), 'ij->nij')
F_expand = csdl.expand(panel_forces_right_ss, (num_beam_nodes, num_right_panels, 3), 'ij->nij')
W_expand = csdl.expand(W_var, (num_beam_nodes, num_right_panels, 3), 'ni->nij')

r = C_expand - B_expand
r_cross_F = csdl.cross(r, F_expand, axis=2)
M_node = csdl.sum(W_expand * r_cross_F, axes=(1,))

# Structural Sizing Case: Direct mapped forces and moments from Node 2 pull-up maneuver
beam_loads_ss = csdl.concatenate([F_node, M_node], axis=1) # shape (num_beam_nodes, 6)
beam.add_load(beam_loads_ss)

# Solve structural beam model using aframe for structural sizing
frame = aframe.Frame(beams=[beam])
frame.solve()

beam_displacement = frame.displacement['wing_spar']
beam_rotation = frame.rotation['wing_spar']
tip_twist_ss = beam_rotation[-1, 1]
beam_stress = frame.compute_stress()['wing_spar'] # shape (num_beam_elements, 5)

# Cross-sectional stress aggregation per element using csdl.maximum with rho=1.0
elem_max_stress = csdl.maximum(beam_stress, axes=(1,), rho=1.0) # shape (num_beam_elements,)

if include_neg1g_sizing:
    # 2. Force & Moment Mapping for -1.0g Sizing Case
    F_node_neg1g = csdl.matmat(W_var, panel_forces_right_neg1g)
    F_expand_neg1g = csdl.expand(panel_forces_right_neg1g, (num_beam_nodes, num_right_panels, 3), 'ij->nij')
    r_cross_F_neg1g = csdl.cross(r, F_expand_neg1g, axis=2)
    M_node_neg1g = csdl.sum(W_expand * r_cross_F_neg1g, axes=(1,))

    beam_loads_neg1g = csdl.concatenate([F_node_neg1g, M_node_neg1g], axis=1) # shape (num_beam_nodes, 6)
    beam_neg1g = aframe.Beam(name='wing_spar_neg1g', mesh=beam_mesh, E=69e9, G=26e9, density=2700, cs=beam_cs)
    beam_neg1g.fix(node=0)
    beam_neg1g.add_load(beam_loads_neg1g)

    frame_neg1g = aframe.Frame(beams=[beam_neg1g])
    frame_neg1g.solve()

    beam_displacement_neg1g = frame_neg1g.displacement['wing_spar_neg1g']
    beam_rotation_neg1g = frame_neg1g.rotation['wing_spar_neg1g']
    beam_stress_neg1g = frame_neg1g.compute_stress()['wing_spar_neg1g'] # shape (num_beam_elements, 5)
    elem_max_stress_neg1g = csdl.maximum(beam_stress_neg1g, axes=(1,), rho=1.0) # shape (num_beam_elements,)

# Fit B-spline stress functions (Fast: 15-CP cubic with S'(0)=0; Full: 8-CP quadratic via space.fit)
if resolution == 'fast':
    tau_colloc = np.concatenate([[0.0], y_norm_elem.flatten()])
    int_knots = [np.mean(tau_colloc[j:j+3]) for j in range(1, 12)]
    knots_stress_15 = np.concatenate([[0.0, 0.0, 0.0, 0.0], int_knots, [1.0, 1.0, 1.0, 1.0]])

    stress_space = lfs.BSplineSpace(
        num_parametric_dimensions=1,
        degree=3,
        coefficients_shape=(15,),
        knots=(knots_stress_15,),
    )

    B14_matrix = stress_space.compute_basis_matrix(y_norm_elem).toarray()
    # Root symmetry condition: S'(0) = 0 <=> c_1 - c_0 = 0 (since clamped cubic has S'(0) = 3/t_4 * (c_1 - c_0))
    d_root_row = np.zeros((1, 15))
    d_root_row[0, 0] = -1.0
    d_root_row[0, 1] = 1.0

    M_stress_sys = np.vstack([B14_matrix, d_root_row])
    M_stress_inv = np.linalg.inv(M_stress_sys)

    # Solve for 15 B-spline coefficients via exact linear system solve in CSDL (3.0g)
    stress_rhs = csdl.concatenate([elem_max_stress, csdl.Variable(value=np.array([0.0]))])
    stress_coeffs_flat = csdl.matvec(csdl.Variable(value=M_stress_inv), stress_rhs)
    stress_coeffs = csdl.reshape(stress_coeffs_flat, (15, 1))
    stress_func = lfs.Function(space=stress_space, coefficients=stress_coeffs)

    if include_neg1g_sizing:
        # Solve for 15 B-spline coefficients via exact linear system solve in CSDL (-1.0g)
        stress_rhs_neg1g = csdl.concatenate([elem_max_stress_neg1g, csdl.Variable(value=np.array([0.0]))])
        stress_coeffs_flat_neg1g = csdl.matvec(csdl.Variable(value=M_stress_inv), stress_rhs_neg1g)
        stress_coeffs_neg1g = csdl.reshape(stress_coeffs_flat_neg1g, (15, 1))
        stress_func_neg1g = lfs.Function(space=stress_space, coefficients=stress_coeffs_neg1g)

elif resolution == 'full':
    # Fit 8-CP B-spline matching the 8 spanwise thickness stations
    stress_space = lfs.BSplineSpace(
        num_parametric_dimensions=1,
        degree=2,
        coefficients_shape=(8,),
    )
    stress_coeffs = stress_space.fit(
        values=csdl.reshape(elem_max_stress, (num_beam_elements, 1)),
        parametric_coordinates=y_norm_elem,
    )
    stress_func = lfs.Function(space=stress_space, coefficients=stress_coeffs)

    if include_neg1g_sizing:
        stress_coeffs_neg1g = stress_space.fit(
            values=csdl.reshape(elem_max_stress_neg1g, (num_beam_elements, 1)),
            parametric_coordinates=y_norm_elem,
        )
        stress_func_neg1g = lfs.Function(space=stress_space, coefficients=stress_coeffs_neg1g)

# 1. Parametric locations corresponding to the peaks of the thickness control (Greville abscissae of thickness_space)
knots_thick = thickness_space.knots[0]
thickness_peaks = np.array([(knots_thick[i+1] + knots_thick[i+2]) / 2.0 for i in range(num_thickness_stations)])

# 2. Evaluate two equally spaced points in between each of these peak points
eval_points_list = []
for i in range(num_thickness_stations - 1):
    eval_points_list.append(thickness_peaks[i])
    pts = np.linspace(thickness_peaks[i], thickness_peaks[i+1], 4)[1:3]
    eval_points_list.extend(pts)
eval_points_list.append(thickness_peaks[-1])
eval_points_arr = np.array(eval_points_list).reshape((-1, 1))  # shape (3 * num_thickness_stations - 2, 1)

# Evaluate the stress function at all peak and intermediate points
all_eval_stresses = stress_func.evaluate(eval_points_arr)  # shape (3 * num_thickness_stations - 2,)

# 3. Aggregate each peak point with its immediate neighboring points (within 1 point on either side)
aggregated_stress_list = []
for i in range(num_thickness_stations):
    if i == 0:
        grp = [0, 1]
    elif i == num_thickness_stations - 1:
        grp = [3*i - 1, 3*i]
    else:
        grp = [3*i - 1, 3*i, 3*i + 1]
    sub_stresses = csdl.concatenate([all_eval_stresses[idx] for idx in grp])
    grp_max = csdl.maximum(sub_stresses, axes=(0,), rho=1.0)
    aggregated_stress_list.append(csdl.reshape(grp_max, (1,)))

# dv_stresses has shape (num_thickness_stations,) -> 5 constraints in fast, 8 in full
dv_stresses = csdl.concatenate(aggregated_stress_list)
# don't enforce stresses at tips since load goes to 0
# stresses_to_enforce = dv_stresses[:-2] if resolution == 'fast' else dv_stresses[:-3]
stresses_to_enforce = dv_stresses[:-1] if resolution == 'fast' else dv_stresses[:-1]

# Yield stress for Aluminum 6061 is 276 MPa.
# Safety factor of 1.5 applied to obtain the allowable stress:
safety_factor = 1.5
yield_stress = 276.0e6
allowable_stress = yield_stress / safety_factor  # 69.0 MPa
# dv_stresses.set_as_constraint(upper=allowable_stress, scaler=1.0 / allowable_stress)
stresses_to_enforce.set_as_constraint(upper=allowable_stress, scaler=1.0 / allowable_stress)

if include_neg1g_sizing:
    # Root stress constraint for -1.0g sizing condition
    root_stress_neg1g = stress_func_neg1g.evaluate(np.array([[0.0]]))
    root_stress_neg1g.set_as_constraint(upper=allowable_stress, scaler=1.0 / allowable_stress)
# endregion Structural solver (beam loads & stress)


# Compute calculated aspect ratio from geometry (wingspan and planform area)
# Define design variables, constraints, and objective for optimization problem
# 1. Induced Drag (Node 0: cruise condition from Trefftz plane)
Di_Trefftz = CDi[0] * 0.5 * rho_array[0] * (cruise_speed**2) * planform_area

# 2. Strip-Wise Sectional Profile & Stall Drag Model (50 strips, geometry-derived alpha via arctan2)
# Evaluate strip center leading and trailing edge points from evaluated geometry:
le_strip_center = 0.5 * (le_drag_mesh[:-1, :] + le_drag_mesh[1:, :])
te_strip_center = 0.5 * (te_drag_mesh[:-1, :] + te_drag_mesh[1:, :])

# Chord vector components: dx (chordwise) and dz (vertical, nose up positive)
dx_strip = te_strip_center[:, 0] - le_strip_center[:, 0]
dz_strip = le_strip_center[:, 2] - te_strip_center[:, 2]

# Exact local angle of attack from geometry using arctan2:
alpha_local_elem = csdl.arctan2(dz_strip, dx_strip)  # shape (50,)

# Geometric local chord length and spanwise strip width dy:
local_chord_drag = csdl.sqrt(dx_strip**2 + dz_strip**2)
y_strip_pts = 0.5 * (le_drag_mesh[:-1, 1] + le_drag_mesh[1:, 1])
dy_strip = le_drag_mesh[1:, 1] - le_drag_mesh[:-1, 1]

# Strip planform area across full wingspan (factor of 2.0 for both wings):
strip_area = 2.0 * local_chord_drag * dy_strip

# Base parasite drag coefficient (NACA 0012 base skin friction + form drag: CD0 = 0.0080)
CD0_base = 0.0080

# Smooth stall drag and lift deficit penalty parameters (NACA 0012 attached flow up to ~10 deg, soft stall onset)
alpha_crit = 10.0 * np.pi / 180.0     # critical angle of attack: 10 degrees
delta_alpha_ref = 4.0 * np.pi / 180.0 # scaling width for post-stall drag rise
beta_stall = 20.0                     # softplus transition sharpness
k_stall = 0.20                        # stall drag scaling factor
k_stall_lift = 0.35                   # stall lift deficit scaling factor

# Penalize stall for both positive and negative angles of attack (Cruise):
alpha_abs = csdl.absolute(alpha_local_elem)
delta_alpha = alpha_abs - alpha_crit

# Numerically stabilized native csdl.softplus:
softplus_val = (1.0 / beta_stall) * csdl.softplus(beta_stall * delta_alpha)
cd_stall_elem = k_stall * ((softplus_val / delta_alpha_ref) ** 2)

# Total profile and stall drag coefficient per strip
cd_profile_elem = CD0_base + cd_stall_elem

# Sectional and total profile drag [N]
q_inf = 0.5 * rho_array[0] * (cruise_speed ** 2)
D_profile_elem = cd_profile_elem * q_inf * strip_area
D_profile = csdl.sum(D_profile_elem)

# Sectional lift deficit penalty for cruise
sign_alpha = alpha_local_elem / (alpha_abs + 1.e-6)
cl_stall_loss_elem = k_stall_lift * (softplus_val / delta_alpha_ref)
L_loss_cruise = csdl.sum(cl_stall_loss_elem * sign_alpha * q_inf * strip_area)
lift_effective_cruise = L[0] - L_loss_cruise

# Total aircraft drag objective: Induced Drag + Strip-Wise Profile & Stall Drag
D_total = Di_Trefftz + D_profile

# Reference values for scaling constraints and objective function
payload_weight_val = float(np.asarray(payload_weight.value).flatten()[0]) if hasattr(payload_weight, 'value') else float(payload_weight)
W_ref = 1.15 * payload_weight_val  # reference cruise weight [N] (~1.15x payload weight)
D_ref = W_ref / 20. # reference drag [N] (~1/20 of payload weight if we assume L/D ~ 20)
c_ref = scale_factor * 1.0 # Initial chord length

objective = D_total
# objective.set_as_objective(scaler=1.e1)
objective.set_as_objective(scaler=1.e1 / D_ref)

# L = W constraint (Node 0: cruise condition)
lift_trim = lift_effective_cruise - W_total
lift_trim.set_as_constraint(equals=0.0, scaler=1.0 / W_ref)

# Pitch / Moment trim constraint: My = 0 about dynamic center of mass (x_cg)

pitch_moment = M[0, 1]
pitch_trim = pitch_moment
pitch_trim.set_as_constraint(equals=0.0, scaler=1.0 / (W_ref * c_ref))

# Sizing Lift constraint: L = load_factor * W (Node 2: structural sizing pull-up condition)
alpha_local_ss = alpha_local_elem + dalpha_ss
alpha_abs_ss = csdl.absolute(alpha_local_ss)
delta_alpha_ss = alpha_abs_ss - alpha_crit
softplus_val_ss = (1.0 / beta_stall) * csdl.softplus(beta_stall * delta_alpha_ss)
cl_stall_loss_ss = k_stall_lift * (softplus_val_ss / delta_alpha_ref)
sign_alpha_ss = alpha_local_ss / (alpha_abs_ss + 1.e-6)
q_inf_ss = 0.5 * rho_array[2] * (sizing_speed ** 2)
L_loss_ss = csdl.sum(cl_stall_loss_ss * sign_alpha_ss * q_inf_ss * strip_area)
lift_effective_ss = L[2] - L_loss_ss

lift_ss = lift_effective_ss - load_factor * W_total
lift_ss.set_as_constraint(equals=0.0, scaler=1.0 / (load_factor_val * W_ref))

if include_neg1g_sizing:
    # Sizing Lift constraint: L = -1.0 * W (Node 3: -1.0g push-down sizing condition)
    lift_neg1g = L[3] - (-1.0 * W_total)
    lift_neg1g.set_as_constraint(equals=0.0, scaler=1.0 / W_ref)

# Static Margin constraint: SM >= 0.10 relative to the dynamic center of mass (x_cg)
# SM = (x_np - x_cg) / mean_chord = -dMy_cg / (dL * mean_chord)
mean_chord = planform_area / wingspan
dL_stab = L[1] - L[0]
dMy_stab = M[1, 1] - M[0, 1]
neutral_point_x = x_cg - dMy_stab / dL_stab
static_margin = (neutral_point_x - x_cg) / mean_chord
# static_margin.set_as_constraint(lower=0.1, scaler=1.e1)
static_margin.set_as_constraint(lower=0.05, scaler=2.e1)
# static_margin.set_as_constraint(equals=0.05, scaler=2.e1)

# # 16-CP Cubic B-spline Fit to Beam Twist under 4.0g Maneuver Load
# # Conditions: f(0) = 0 (root zero twist), f'(0) = 0 (symmetry), and f(y_k) = theta_k for nodes 1..14
# y_norm_node = (y_beam_span / (4.99 * scale_factor)).reshape((-1, 1))
# int_knots_twist_16 = np.linspace(0.0, 1.0, 14)[1:-1]
# knots_twist_16 = np.concatenate([[0.0, 0.0, 0.0, 0.0], int_knots_twist_16, [1.0, 1.0, 1.0, 1.0]])

# twist_fit_space = lfs.BSplineSpace(
#     num_parametric_dimensions=1,
#     degree=3,
#     coefficients_shape=(16,),
#     knots=(knots_twist_16,),
# )

# B16_matrix = twist_fit_space.compute_basis_matrix(y_norm_node).toarray()  # shape (15, 16)
# M_twist_sys = np.zeros((16, 16))
# M_twist_sys[0, 0] = 1.0
# M_twist_sys[1, 0] = -1.0
# M_twist_sys[1, 1] = 1.0
# M_twist_sys[2:, :] = B16_matrix[1:, :]
# M_twist_inv = np.linalg.inv(M_twist_sys)

# beam_twist_all_nodes = beam_rotation[:, 1]  # shape (15,)
# twist_rhs = csdl.concatenate([csdl.Variable(value=np.zeros(2)), beam_twist_all_nodes[1:]])
# twist_coeffs_flat = csdl.matvec(csdl.Variable(value=M_twist_inv), twist_rhs)
# twist_coeffs = csdl.reshape(twist_coeffs_flat, (16, 1))
# twist_spline_func = lfs.Function(space=twist_fit_space, coefficients=twist_coeffs)

# # Evaluate twist at the spanwise sweep stations (stations 1 to num_stations-1, excluding root station 0)
# station_eta_eval = np.clip(chord_station_y / (4.99 * scale_factor), 0.0, 1.0).reshape((-1, 1))
# all_station_twists = twist_spline_func.evaluate(station_eta_eval)  # shape (num_chord_stations,)
# outboard_station_twists = all_station_twists[1:]  # shape (num_chord_stations - 1,)

# # Enforce absolute twist <= 0 at each outboard station individually for static aeroelastic stability
# # Sign convention: theta_y < 0 corresponds to pitch-down / twist-down about spanwise Y-axis (washout)
# outboard_station_twists.set_as_constraint(upper=0.0, scaler=180.0 / np.pi)

if formulation == 'chord_span':
    # For chord and span stretch formulation (no ParameterizationSolver),
    # keep planform area constraint (1.0 m^2) and aspect ratio inequality constraint AR <= 15.0
    planform_area.set_as_constraint(equals=1.0 * scale_factor**2, scaler=1.0 / (scale_factor**2))
    aspect_ratio_calc.set_as_constraint(upper=15.0, scaler=1.e-1)
    # Enforce constant thickness-to-chord ratio = 0.12 at all stations
    for i in range(num_chord_stations):
        tc_ratio = local_thicknesses[i] / local_chords[i]
        tc_ratio.set_as_constraint(equals=0.12, scaler=10.0)
else:
    # For AR and Area formulation, ParameterizationSolver explicitly enforces
    # taper ratios, planform area, and aspect ratio.
    pass

for dv_info in design_variables.values():
    dv_info.variable.set_as_design_variable(lower=dv_info.lower, upper=dv_info.upper, scaler=dv_info.scaler)

geometry_coefficients = [geometry_function.coefficients for geometry_function in geometry.functions.values()]

additional_outs = [Di, L, CL, CDi, Cp, panel_mesh, planform_area, aspect_ratio_calc, structural_mass, beam_displacement, beam_rotation, tip_twist_ss, beam_stress, elem_max_stress, dv_stresses, stress_coeffs, ttop_elem, tweb_elem, ttop_dvs, tweb_dvs, twist_dvs, Di_Trefftz, D_profile, D_total, alpha_local_elem, y_strip_pts, W_total, M, CM, pitch_trim, lift_ss, static_margin, neutral_point_x, x_cg, x_payload, payload_cg, x_struct, r_cg, local_chord, local_height, box_width, beam_mesh, F_node, pitch_ss, dynamic_panel_centers_right, panel_forces_right_cruise, panel_forces_right_ss, lift_effective_cruise, lift_effective_ss, L_loss_cruise, L_loss_ss]
if include_camber:
    additional_outs += [camber_dvs]
if include_elevator:
    additional_outs += [elevator_angle]
if include_neg1g_sizing:
    additional_outs += [lift_neg1g, root_stress_neg1g, stress_coeffs_neg1g, elem_max_stress_neg1g, pitch_neg1g, panel_forces_right_neg1g]
additional_outs += geometry_coefficients

jax_sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=[dv_info.variable for dv_info in design_variables.values()],
    additional_outputs=additional_outs,
    gpu=False
)

# Populate design variables with initial/warm-started values
for dv_info in design_variables.values():
    val = dv_info.variable.value
    if val is not None:
        jax_sim[dv_info.variable] = np.asarray(val)

# Optional initial diagnostic pass (disabled by default to avoid slow initial JAX compilation)
run_pre_diagnostics = False
if run_pre_diagnostics:
    jax_sim.run()
    # print(f"Final Lift (Cruise Node 0): Effective = {float(np.asarray(jax_sim[lift_effective_cruise]).flatten()[0]):.2f} N (VLM: {float(np.asarray(jax_sim[L]).flatten()[0]):.2f} N, Loss: {float(np.asarray(jax_sim[L_loss_cruise]).flatten()[0]):.2f} N, Total Weight W: {float(np.asarray(jax_sim[W_total]).flatten()[0]):.2f} N)")
    # print(f"Final Lift ({load_factor_val:.1f}g Sizing Node 2): Effective = {float(np.asarray(jax_sim[lift_effective_ss]).flatten()[0]):.2f} N (VLM: {float(np.asarray(jax_sim[L]).flatten()[2]):.2f} N, Loss: {float(np.asarray(jax_sim[L_loss_ss]).flatten()[0]):.2f} N, Target {load_factor_val:.1f}x Weight: {load_factor_val * float(np.asarray(jax_sim[W_total]).flatten()[0]):.2f} N)")
    # if include_neg1g_sizing:
    #     print(f"Final Lift (-1.0g Sizing Node 3): {float(np.asarray(jax_sim[L]).flatten()[3]):.2f} N (Target -1.0x Weight: {-1.0 * float(np.asarray(jax_sim[W_total]).flatten()[0]):.2f} N)")
    # print(f"Pitch (Cruise Node 0): {float(np.asarray(jax_sim[pitch]).flatten()[0]) * 180 / np.pi:.2f} deg")
    # print(f"Pitch ({load_factor_val:.1f}g Sizing Node 2): {float(np.asarray(jax_sim[pitch_ss]).flatten()[0]) * 180 / np.pi:.2f} deg")
    # if include_neg1g_sizing:
    #     print(f"Pitch (-1.0g Sizing Node 3): {float(np.asarray(jax_sim[pitch_neg1g]).flatten()[0]) * 180 / np.pi:.2f} deg")
    #     print(f"Root Stress (-1.0g Sizing): {float(np.asarray(jax_sim[root_stress_neg1g]).flatten()[0])/1e6:.2f} MPa (Allowable: {allowable_stress/1e6:.1f} MPa)")
    # print(f"Final Pitching Moment (about CG): {float(np.asarray(jax_sim[M][0, 1]).flatten()[0]):.4f} N*m")
    # print(f"Final Center of Mass (x_cg): {float(np.asarray(jax_sim[x_cg]).flatten()[0]):.4f} m (Struct CG: {float(np.asarray(jax_sim[x_struct]).flatten()[0]):.4f} m, Payload: {float(np.asarray(jax_sim[x_payload]).flatten()[0]):.4f} m [{float(np.asarray(jax_sim[payload_cg]).flatten()[0])*100:.1f}% root chord])")
    # print(f"Total Drag (Objective): {float(np.asarray(jax_sim[D_total]).flatten()[0]):.2f} N (Induced: {float(np.asarray(jax_sim[Di_Trefftz]).flatten()[0]):.2f} N, Profile+Stall: {float(np.asarray(jax_sim[D_profile]).flatten()[0]):.2f} N)")
    # alpha_local_deg_all = np.degrees(np.asarray(jax_sim[alpha_local_elem]).flatten())
    # print(f"Local Strip Alpha (Cruise): Min = {np.min(alpha_local_deg_all):.2f} deg, Max = {np.max(alpha_local_deg_all):.2f} deg")
    # print(f"Half-Beam Mass: {float(np.asarray(jax_sim[structural_mass]).flatten()[0])/2.0:.2f} kg (Full Structural Mass: {float(np.asarray(jax_sim[structural_mass]).flatten()[0]):.2f} kg)")
    # tip_rot_ss_val = np.asarray(jax_sim[beam_rotation])[-1]
    # tip_twist_deg = float(np.asarray(jax_sim[tip_twist_ss]).flatten()[0]) * 180.0 / np.pi
    # print(f"Wing Tip Rotation ({load_factor_val:.1f}g Load): θx (roll slope) = {tip_rot_ss_val[0]*180/np.pi:+.3f}°, θy (twist/pitch) = {tip_twist_deg:+.3f}°, θz (yaw slope) = {tip_rot_ss_val[2]*180/np.pi:+.3f}°")

    # station_twists_deg = np.asarray(jax_sim[all_station_twists]).flatten() * 180.0 / np.pi
    # outboard_twists_deg = np.asarray(jax_sim[outboard_station_twists]).flatten() * 180.0 / np.pi
    # print(f"\n================ 16-CP CUBIC B-SPLINE AEROELASTIC TWIST CONSTRAINTS ({load_factor_val:.1f}g Load Case) ================")
    # print(f"{'Station':7s} | {'eta':6s} | {'y [m]':8s} | {'Twist θy [deg]':16s} | {'Constraint (θy <= 0)':22s} | {'Status':8s}")
    # print("-" * 75)
    # print(f"{0:7d} | {station_eta_eval[0,0]:6.3f} | {chord_station_y[0]:8.3f} | {station_twists_deg[0]:16.6f} | [Fixed Root Boundary] | FEASIBLE")
    # for i in range(1, num_chord_stations):
    #     tw_val = station_twists_deg[i]
    #     st = "FEASIBLE" if tw_val <= 1e-6 else "VIOLATED"
    #     print(f"{i:7d} | {station_eta_eval[i,0]:6.3f} | {chord_station_y[i]:8.3f} | {tw_val:16.6f} | {tw_val:+.4f}° <= 0.0°        | {st:8s}")

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
    twist_dv_arr = np.asarray(jax_sim[twist_dvs]).flatten()

    print(f"\n================ ELEMENT-BY-ELEMENT BEAM DIAGNOSTIC ({num_beam_elements} Elements, {load_factor_val:.1f}g Load Case) ================")
    print(f"{'Elem':4s} | {'y_mid [m]':9s} | {'Chord [m]':9s} | {'Height [m]':10s} | {'ttop [mm]':9s} | {f'{load_factor_val:.1f}g Fz [N]':11s} | {'Max Stress [MPa]':16s}")
    print("-" * 88)
    y_elem_mid = 0.5 * (beam_pts[:-1, 1] + beam_pts[1:, 1])
    for i in range(num_beam_elements):
        print(f"{i:4d} | {y_elem_mid[i]:9.3f} | {chords_arr[i]:9.3f} | {heights_arr[i]:10.4f} | {ttop_elem_arr[i]*1e3:9.2f} | {f_nodes_arr[i, 2]:11.2f} | {elem_stress_arr[i]/1e6:16.2f}")

    print(f"\n================ {num_thickness_stations} THICKNESS-STATION AGGREGATED STRESS CONSTRAINTS (Allowable = {allowable_stress/1e6:.1f} MPa) ================")
    print(f"{'Station':7s} | {'eta_peak':8s} | {'Span y [m]':10s} | {'Aggregated Stress [MPa]':24s} | {'Allowable [MPa]':16s} | {'Status':8s}")
    print("-" * 95)
    for j in range(num_thickness_stations):
        st_val = dv_stress_arr[j] / 1e6
        y_p_span = thickness_peaks[j] * 4.99 * scale_factor
        status = "FEASIBLE" if st_val <= (allowable_stress / 1e6) else "VIOLATED"
        print(f"{j:7d} | {thickness_peaks[j]:8.4f} | {y_p_span:10.3f} | {st_val:24.2f} | {allowable_stress/1e6:16.1f} | {status:8s}")

    stress_coeffs_arr = np.asarray(jax_sim[stress_coeffs]).flatten()
    if resolution == 'fast':
        root_deriv_val = (stress_coeffs_arr[1] - stress_coeffs_arr[0]) * 3.0 / knots_stress_15[4]
        print(f"\n================ 15-CP CUBIC STRESS SPLINE DIAGNOSTIC ================")
        print(f"Root symmetry check: c0 = {stress_coeffs_arr[0]/1e6:.4f} MPa, c1 = {stress_coeffs_arr[1]/1e6:.4f} MPa, dS/du(0) = {root_deriv_val/1e6:.6f} MPa/unit")
        print(f"Stress control points (MPa): {np.round(stress_coeffs_arr/1e6, 2)}")
    elif resolution == 'full':
        print(f"\n================ 8-CP B-SPLINE STRESS DIAGNOSTIC ================")
        print(f"Stress control points (MPa): {np.round(stress_coeffs_arr/1e6, 2)}")

optimization_problem = modopt.CSDLAlphaProblem(
    problem_name='rectangular_wing_to_bwb_aerostructural_optimization',
    simulator=jax_sim,
)
optimizer = modopt.PySLSQP(
    optimization_problem,
    # solver_options={'maxiter': 100, 'acc': 1.e-7},
    solver_options={'maxiter': 200, 'acc': 1.e-5},
    readable_outputs=['x'],
)
optimizer.solve()
optimizer.print_results()
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
    'position': (-20.0 * scale_factor, -15.0 * scale_factor, 10.0 * scale_factor),
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
        var_size = int(np.prod(dv_info.variable.shape))
        slc = slice(curr_idx, curr_idx + var_size)
        unscaled_val = (x_scaled[slc] / dv_info.scaler).reshape(dv_info.variable.shape)
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
        if 'twist_dvs' in unscaled_values:
            tw_vals = unscaled_values['twist_dvs']
            tw_str = " ".join([f"tw{i}={np.degrees(tw_vals[i]):.1f}°" for i in range(len(tw_vals))])
            dv_str = f"AR={ar_val:.2f}  {c_str}\n{sw_str}\n{tw_str}"
        else:
            dv_str = f"AR={ar_val:.2f}  {c_str}\n{sw_str}"
    elif formulation == 'chord_span':
        ss_val = float(np.asarray(unscaled_values['span_stretch_dv']).flatten()[0]) if 'span_stretch_dv' in unscaled_values else 0.0
        cs_vals = unscaled_values['chord_stretch_dvs'] if 'chord_stretch_dvs' in unscaled_values else np.zeros(num_chord_stations)
        c_str = " ".join([f"c{i}={initial_chord+cs_vals[i]:.3f}m" for i in range(len(cs_vals))])
        sw_vals = unscaled_values['sweep_dvs'] if 'sweep_dvs' in unscaled_values else np.zeros(num_chord_stations)
        sw_str = " ".join([f"sw{i}={sw_vals[i]:.2f}" for i in range(len(sw_vals))])
        if 'tip_twist' in unscaled_values:
            tw_tip_val = float(np.asarray(unscaled_values['tip_twist']).flatten()[0])
            tw_str = f"tw_tip={np.degrees(tw_tip_val):.1f}° (linear)"
        elif 'twist_dvs' in unscaled_values:
            tw_vals = unscaled_values['twist_dvs']
            tw_str = " ".join([f"tw{i}={np.degrees(tw_vals[i]):.1f}°" for i in range(len(tw_vals))])
        else:
            tw_str = ""
    if 'camber_dvs' in unscaled_values:
        cam_vals = unscaled_values['camber_dvs']
        max_cam_pct = np.max(np.abs(cam_vals))
        dv_str += f"\nmax|camber|={max_cam_pct:.2f}% chord"

    pitch_val = float(np.asarray(unscaled_values['pitch']).flatten()[0]) if 'pitch' in unscaled_values else 0.0
    pitch_ss_val = float(np.asarray(unscaled_values['pitch_ss']).flatten()[0]) if 'pitch_ss' in unscaled_values else 0.0
    pitch_neg1g_val = float(np.asarray(unscaled_values['pitch_neg1g']).flatten()[0]) if 'pitch_neg1g' in unscaled_values else 0.0
    pitch_neg1g_str = f"  Pitch_-1g={np.degrees(pitch_neg1g_val):.1f}°" if 'pitch_neg1g' in unscaled_values else ""
    elev_val = float(np.asarray(unscaled_values['elevator_angle']).flatten()[0]) if 'elevator_angle' in unscaled_values else 0.0
    elev_str = f"Elevator={np.degrees(elev_val):.1f}°  " if include_elevator else ""
    pay_cg_val = float(np.asarray(unscaled_values['payload_cg']).flatten()[0]) if 'payload_cg' in unscaled_values else 0.40
    sm_val = float(np.asarray(jax_sim[static_margin]).flatten()[0])
    xnp_val = float(np.asarray(jax_sim[neutral_point_x]).flatten()[0])
    xcg_val = float(np.asarray(jax_sim[x_cg]).flatten()[0])
    xpay_val = float(np.asarray(jax_sim[x_payload]).flatten()[0])

    # Add iteration counter label using unscaled physical values
    plotter.add_text(
        f"Iteration {iteration}/{num_iterations - 1}\n"
        f"Formulation: {formulation} | Res: {resolution}\n"
        f"{dv_str}\n"
        f"{elev_str}Pitch={np.degrees(pitch_val):.1f}°  Pitch_ss={np.degrees(pitch_ss_val):.1f}°{pitch_neg1g_str}\n"
        f"SM={sm_val:.4f} (x_cg={xcg_val:.3f}m, x_np={xnp_val:.3f}m, x_pay={xpay_val:.3f}m [{pay_cg_val*100:.1f}%])",
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

        # Cache panel telemetry for extract_lift_and_moment_distributions.py
        cache_file_opt = os.path.join(latest_folder, 'lift_and_moment_data.npz')
        try:
            xcg_val_opt = float(np.asarray(jax_sim[x_cg]).flatten()[0])
            zcg_val_opt = float(np.asarray(jax_sim[r_cg]).flatten()[2]) if 'r_cg' in globals() else 0.0
            panel_centers_right_opt = np.asarray(jax_sim[dynamic_panel_centers_right])
            f_cruise_opt = np.asarray(jax_sim[panel_forces_right_cruise])
            f_ss_opt = np.asarray(jax_sim[panel_forces_right_ss])
            np.savez_compressed(
                cache_file_opt,
                panel_centers_right=panel_centers_right_opt,
                f_cruise=f_cruise_opt,
                f_ss=f_ss_opt,
                xcg_val=xcg_val_opt,
                zcg_val=zcg_val_opt
            )
            print(f"Cached panel telemetry saved to: {cache_file_opt}")
        except Exception as e:
            print(f"Warning: could not save panel telemetry cache: {e}")

plotter.close()
print(f"Video saved to: {video_path}")

# region Plot Summary Figure (Wing geometry vs Theory)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Compute theoretical Cd = Cl^2 / (pi * AR) in drag counts (x 1e4)
span_b = 10.0 * scale_factor
target_cl = 0.5
target_sref = 1.0
target_ar = 20
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
artifact_dir = '/home/andrew/.gemini/antigravity/brain/47a5c338-9be3-486f-abb2-1deefc5b2d19'
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