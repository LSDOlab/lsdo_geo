# region Imports and Setup

from dataclasses import dataclass
from typing import Union, Literal
import sys
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
SHOWCASE_DIR = os.path.dirname(os.path.abspath(__file__))
if SHOWCASE_DIR not in sys.path:
    sys.path.insert(0, SHOWCASE_DIR)
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

from physics_models.flight_conditions import (
    compute_isa_troposphere,
    get_nominal_flight_conditions,
    CRUISE_ALTITUDE_M,
    CRUISE_MACH,
    SIZING_EQUIVALENT_SPEED_FACTOR,
)
from physics_models.wave_drag import (
    setup_strip_projection_points,
    compute_strip_sweep,
    compute_quarter_chord_sweep,
    compute_half_chord_sweep,
    compute_strip_thickness_to_chord,
    evaluate_wave_drag,
)
from physics_models.ar_area_bspline import BsplineTargetRegularization

class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            if module.startswith('numpy._core'):
                module = module.replace('numpy._core', 'numpy.core')
            elif module.startswith('numpy.core'):
                module = module.replace('numpy.core', 'numpy._core')
            return super().find_class(module, name)

_orig_pickle_load = pickle.load
pickle.load = lambda f, **kwargs: CompatUnpickler(f, **kwargs).load()

recorder = csdl.Recorder(inline=True)
recorder.start()

# =============================================================================
# TOP-LEVEL CONFIGURATION AND TOGGLES
# =============================================================================

# 1. Parameterization formulation: 'chord_span' or 'ar_area'
#   'chord_span' -> Direct chord stretch, sweep translation, span stretch, twist, camber/thickness shape DVs
#   'ar_area'    -> Regularized cubic B-spline taper ratios, aspect ratio, Galerkin sweep weak-form, planform area target
ParamerizationType = Literal['ar_area', 'chord_span']
formulation: ParamerizationType = 'chord_span'  # Options: 'chord_span' or 'ar_area'
formulation = os.environ.get('BWB_FORMULATION', formulation)

# 2. Mesh and design variable resolution: 'fast' or 'full'
#   'fast' -> 5 design stations, 9 FFD sections, 1,272 aero quads (fast iteration/testing)
#   'full' -> 8 design stations, 15 FFD sections, 2,872 aero quads (production resolution)
resolution: Literal['fast', 'full'] = 'full'  # Options: 'fast' or 'full'
resolution = os.environ.get('BWB_RESOLUTION', resolution)

# 3. Geometric Non-Interference (Geonic) payload containment mode:
#   'none'      -> Baseline behavior: unconstrained OML, legacy payload_cg DV, no BSM3 import
#   'oversized' -> Constrain TCP0 oversized box (8 sample points, payload_center_x DV, 0.2 m buffer)
#   'pallets'   -> Constrain TCP0 6-pallet stack (32 sample points, pallet_stack_center_x DV, 0.2 m buffer)
#   'both'      -> Constrain both payload cases (40-point batch, equal CG consistency constraint)
geonic_payload_mode: str = 'none'  # Options: 'none', 'oversized', 'pallets', 'both'
geonic_payload_mode = os.environ.get('GEONIC_PAYLOAD_MODE', geonic_payload_mode)
if 'GEONIC_PAYLOAD_MODE' not in os.environ and 'USE_GEONIC' in os.environ:
    geonic_payload_mode = 'oversized' if str(os.environ['USE_GEONIC']).strip() == '1' else 'none'

# 4. Viscous drag mode:
#   'ibl'          -> Integral boundary layer drag model & attachment constraint H_max <= 2.4
#   'constant_cd0' -> Legacy fixed baseline profile drag (CD0_base = 0.0080)
viscous_drag_mode: Literal['ibl', 'constant_cd0'] = 'ibl'  # Options: 'ibl' or 'constant_cd0'
viscous_drag_mode = os.environ.get('VISCOUS_DRAG_MODE', viscous_drag_mode)

# 5. Sizing condition toggle: -1.0g push-down maneuver
#   True  -> includes -1.0g sizing maneuver (Flight condition Node 3)
#   False -> drops Node 3, pitch_neg1g DV, lift_neg1g trim constraint for ~25% speedup
include_neg1g_sizing: bool = False  # Options: True or False

# 5b. Structural wingbox depth factor:
#   Effective structural height factor relative to OML (accounts for spar cap flange
#   centroid recession beneath OML skins/stringers, shear lag, and internal cutouts)
box_height_factor: float = 0.7     # Options: 0.50 - 0.70 (standard preliminary design factor)
box_height_factor = float(os.environ.get('BOX_HEIGHT_FACTOR', box_height_factor))

# 6. Airfoil shape & control surface options:
include_camber: bool = True           # Mean camber line deformation (3 rows of control points)
include_thickness_shape: bool = True  # Airfoil thickness distribution deformation
include_te_thickness: bool = False    # True: blunt TE (5 rows); False: pinned sharp TE (4 rows)
include_elevator: bool = False        # Trailing-edge elevator deflection DV

# 7. Aerodynamics and drag formulation:
induced_drag_objective: Literal['trefftz', 'fourier', 'mixed'] = 'mixed'  # Options: 'trefftz', 'fourier', 'mixed'
include_cl_polar: bool = False        # Apply quadratic drag polar bucket k_polar * (Cl - Cl_ideal)^2
include_stall_drag: bool = True       # Apply smooth aerodynamic stall penalty when Cl > Cl_crit

# 8. Scale factor & warm start:
scale_factor: float = 7.5             # Scale factor (7.5 for 50 m full-scale BWB; 1.0 for wind-tunnel scale)
warm_start: bool = False              # Warm start DVs from prior run

# 9. Diagnostics & pre-flight verification:
run_pre_diagnostics: bool = True      # Run initial JAX evaluation and print comprehensive diagnostic report before optimization
# =============================================================================

# Validate viscous drag mode
if viscous_drag_mode not in ('ibl', 'constant_cd0'):
    raise ValueError(f"Unknown viscous_drag_mode: '{viscous_drag_mode}'. Must be 'ibl' or 'constant_cd0'.")

from optimization_analyses.bwb_viscous_ibl import (
    build_ibl_mesh_topology,
    evaluate_bwb_viscous_ibl,
)

# Parse and validate geonic payload mode
from physics_models.geonic_payload import (
    VALID_GEONIC_PAYLOAD_MODES,
    parse_geonic_payload_mode,
)
geonic_payload_mode, use_geonic = parse_geonic_payload_mode(mode=geonic_payload_mode)

if use_geonic:
    try:
        import bsm3
    except ImportError as e:
        raise ImportError(
            f"geonic_payload_mode='{geonic_payload_mode}' requires 'bsm3' to be installed. "
            "Please install the repaired checkout at /home/andrew/optimization/BSM3 "
            "via `pip install -e /home/andrew/optimization/BSM3 --no-deps`."
        ) from e
    from physics_models.geonic_payload import (
        GEONIC_CLEARANCE_M,
        PAYLOAD_LENGTH_M,
        PAYLOAD_WIDTH_M,
        PAYLOAD_HEIGHT_M,
        PALLET_L_SHORT_M,
        PALLET_WIDTH_M,
        PALLET_HEIGHT_M,
        PALLET_L_LONG_M,
        PALLET_STACK_FRONT_M,
        PALLET_STACK_REFERENCE_CG_X,
        build_geonic_payload_sample_points,
        build_geonic_pallet_sample_points,
        compute_geonic_clearance_and_margin,
        get_full_payload_box_corners,
        get_full_pallet_boxes_corners,
    )

# Import initial geometry that will be deformed
geometry_directory = "examples/example_geometries/"
file_name = "rectangular_wing_naca0012_10ar"

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

# Diagnostic-only override: retain active FFD parameterization while evaluating aero on different mesh
aero_mesh_file_name = os.environ.get('BWB_AERO_MESH_FILE', mesh_file_name)
num_chord_stations = num_stations
num_thickness_stations = num_stations

# Geometric CAD control points are kept equivalent across resolutions (15 spanwise)
num_spanwise_cp_target = 25

geometry = import_geometry(
    geometry_directory + file_name + ".stp",
    name='imported_geometry',
    parallelize=False,
)

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
        new_degree = (min(3, num_spanwise_cp_target - 1), orig_degree[1])
    else:
        n_chord = coeffs.shape[0]
        new_coeffs = np.zeros((n_chord, num_spanwise_cp_target, 3))
        for j in range(n_chord):
            for k in range(3):
                new_coeffs[j, :, k] = np.linspace(coeffs[j, 0, k], coeffs[j, -1, k], num_spanwise_cp_target)
        new_shape = (n_chord, num_spanwise_cp_target)
        new_degree = (orig_degree[0], min(3, num_spanwise_cp_target - 1))

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

# Construct a Free Form Deformation (FFD) block around the geometry
# 5 chordwise control points when camber or thickness shape is active (excluding LE & TE gives 3 interior points)
num_ffd_coefficients_chordwise = 5 if (include_camber or include_thickness_shape) else 2
ffd_degree_chordwise = 2 if (include_camber or include_thickness_shape) else 1
# num_ffd_sections is dynamically set by resolution ('fast': 9, 'full': 15)
ffd_block = construct_ffd_block_around_entities(
    entities=geometry, 
    num_coefficients=(num_ffd_coefficients_chordwise, num_ffd_sections, 2),
    degree=(ffd_degree_chordwise, 3, 1),
)
# Define an axial sectional parameterization for the FFD volume.
ffd_sectional_parameterization = SectionalParameterization(
    name="ffd_sectional_parameterization",
    parameterized_points=ffd_block.coefficients,
    principal_parametric_dimension=1,
)

# Project chord, thickness, and sweep measurement points across the right half-span (y = 0.0 to 5.0 * scale_factor).
# In 'ar_area', map stations directly from uniform parametric spanwise coordinates eta in [0, 1] of the cubic B-spline FFD volume.
# This ensures the physical geometry evaluation stations exactly match the cubic B-spline basis collocation stations
# for all 3 spanwise distribution variables (taper, thickness, sweep: T = inv(B_actual) @ B_stations = I),
# preventing coordinate mismatch distortion from producing negative FFD control point stretches and mid-span inversion.
eta_stations = np.linspace(0.0, 1.0, num_chord_stations)
if formulation == 'ar_area':
    v_stations = 0.5 + 0.5 * eta_stations
    station_eval_pts = np.column_stack([
        np.zeros(num_chord_stations),
        v_stations,
        np.full(num_chord_stations, 0.5),
    ])
    station_phys_pts = ffd_block.evaluate(station_eval_pts, non_csdl=True)
    chord_station_y = np.clip(np.asarray(station_phys_pts)[:, 1], 0.0, 5.0 * scale_factor)
    chord_station_y[0] = 0.0
    chord_station_y[-1] = 5.0 * scale_factor
else:
    chord_station_y = np.linspace(0.0, 5.0 * scale_factor, num_chord_stations)
chord_le_projections = []
chord_te_projections = []
quarter_chord_projections = []
for y_val in chord_station_y:
    chord_le_projections.append(geometry.project(np.array([0.0, y_val, 0.0])))
    chord_te_projections.append(geometry.project(np.array([1.0 * scale_factor, y_val, 0.0])))
    quarter_chord_projections.append(geometry.project(np.array([0.25 * scale_factor, y_val, 0.0])))

# High-resolution quarter-chord sweep evaluation projections to strictly enforce sweep >= 0 across span
num_sweep_eval_stations = 25
sweep_eval_y = np.linspace(0.0, 5.0 * scale_factor, num_sweep_eval_stations)
quarter_chord_sweep_projections = []
for y_val in sweep_eval_y:
    quarter_chord_sweep_projections.append(geometry.project(np.array([0.25 * scale_factor, y_val, 0.0])))

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

# Project dense spanwise leading and trailing edge lines to compute planform area via trapezoid rule
ny_area = 31  # Dense spanwise resolution
area_station_y = np.linspace(0.0, 5.0 * scale_factor, ny_area)

area_le_physical = np.zeros((ny_area, 3))
area_le_physical[:, 0] = 0.0
area_le_physical[:, 1] = area_station_y
area_le_physical[:, 2] = 0.0

area_te_physical = np.zeros((ny_area, 3))
area_te_physical[:, 0] = 1.0 * scale_factor
area_te_physical[:, 1] = area_station_y
area_te_physical[:, 2] = 0.0

projected_area_le = geometry.project(area_le_physical, plot=False)
projected_area_te = geometry.project(area_te_physical, plot=False)

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

# Sample wingbox upper/lower surfaces at 5 chordwise stations across 20% to 70% chord
# Using composite Simpson's rule: weights = [1, 4, 2, 4, 1] / 12
# Integrates quadratic B-spline thickness profiles with zero algebraic truncation error
x_fracs_box = np.array([0.20, 0.325, 0.45, 0.575, 0.70])
num_box_samples = len(x_fracs_box)
simpson_weights_box = np.array([1.0, 4.0, 2.0, 4.0, 1.0]) / 12.0

x_box_grid, y_box_grid = np.meshgrid(x_fracs_box * scale_factor, y_beam_span, indexing='ij')
upper_box_seed = np.column_stack([
    x_box_grid.ravel(),
    y_box_grid.ravel(),
    np.full(x_box_grid.size, 0.05 * scale_factor),
])
lower_box_seed = np.column_stack([
    x_box_grid.ravel(),
    y_box_grid.ravel(),
    np.full(x_box_grid.size, -0.05 * scale_factor),
])
projected_upper_box_mesh = geometry.project(upper_box_seed, direction=np.array([0, 0, -1]), plot=False)
projected_lower_box_mesh = geometry.project(lower_box_seed, direction=np.array([0, 0, 1]), plot=False)
projected_upper_beam_mesh = projected_upper_box_mesh
projected_lower_beam_mesh = projected_lower_box_mesh

# Weighting matrix for csdl.matvec: shape (num_beam_nodes, num_box_samples * num_beam_nodes)
# Computes node_heights[i] = sum_k w[k] * (upper_z[k, i] - lower_z[k, i])
W_box_mat = np.zeros((num_beam_nodes, num_box_samples * num_beam_nodes))
for i in range(num_beam_nodes):
    for k in range(num_box_samples):
        W_box_mat[i, k * num_beam_nodes + i] = simpson_weights_box[k]

# Project 101 points along leading and trailing edges for 100-strip refined drag & local Cl
num_drag_strips = 100
num_drag_nodes = num_drag_strips + 1
b_tip_ref = 4.99 * scale_factor
eta_drag = np.sin(0.5 * np.pi * np.linspace(0.0, 1.0, num_drag_nodes))
y_drag_span = eta_drag * b_tip_ref

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

# Fixed paired upper/lower surface projection coordinates at 41 chordwise fractions
# for every drag-strip center for live thickness-to-chord (t/c)_i evaluation:
y_drag_centers_init = 0.5 * (y_drag_span[:-1] + y_drag_span[1:])
strip_upper_seed, strip_lower_seed, strip_grid_shape = setup_strip_projection_points(
    y_strip_centers=y_drag_centers_init,
    scale_factor=scale_factor,
    num_chord_fractions=41,
    z_offset_ratio=0.05,
)
projected_strip_upper_skin = geometry.project(
    strip_upper_seed,
    force_reprojection=False,
    direction=np.array([0, 0, -1]),
    use_line_search=True,
    plot=False,
)
projected_strip_lower_skin = geometry.project(
    strip_lower_seed,
    force_reprojection=False,
    direction=np.array([0, 0, 1]),
    use_line_search=True,
    plot=False,
)
# endregion

# region Mesh definitions
mesh = meshio.read(geometry_directory + aero_mesh_file_name + ".msh")
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

# Build static IBL mesh topology index maps and interpolation matrix
ibl_topology = build_ibl_mesh_topology(
    points=points_orig,
    cells_dict=cells_dict,
    scale_factor=1.0,  # points_orig is already scaled by scale_factor at line 451
    num_drag_strips=num_drag_strips,
    y_drag_centers=y_drag_centers_init,
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

# Compute static panel-to-Fourier transformation matrix for lifting-line induced drag
y_round = np.round(Y_panels_right, 4)
y_unique = np.unique(y_round)
main_stations = [y for y in y_unique if np.sum(y_round == y) == 40]
num_strips = len(main_stations)

W_strip = np.zeros((num_strips, num_right_panels))
y_strip_centers = []
for s_idx, y_val in enumerate(main_stations):
    mask = (y_round == y_val)
    W_strip[s_idx, mask] = 1.0
    y_strip_centers.append(float(np.mean(Y_panels_right[mask])))

# Merge tip cap panels into the final tip strip to guarantee complete vertical load accounting
tip_mask = ~np.isin(y_round, main_stations)
if np.any(tip_mask):
    W_strip[-1, tip_mask] = 1.0

y_strip_centers = np.array(y_strip_centers)
b_tip_initial = float(np.max(Y_panels_right))
eta_strips = y_strip_centers / b_tip_initial
# Invariant polar angle theta in [pi/2, pi] corresponding to spanwise coordinate y in [0, b_tip]
theta_strips = np.pi - np.arccos(eta_strips)

num_fourier_terms = 16  if resolution == 'full' else 8
# Odd harmonics n = 1, 3, 5, ..., 15
n_fourier_odd = np.arange(1, 2 * num_fourier_terms, 2)
K_fourier = np.sin(np.outer(n_fourier_odd, theta_strips)) / np.sin(theta_strips)
T_fourier_panel = K_fourier @ W_strip  # shape (num_fourier_terms, num_right_panels)

# Compute interpolation matrix from aero strip centers to refined drag strip centers
y_aero_c = np.array(y_strip_centers)
y_mid = 0.5 * (y_aero_c[:-1] + y_aero_c[1:])
y_aero_bounds = np.concatenate([[0.0], y_mid, [b_tip_initial]])
dy_aero_init = np.diff(y_aero_bounds)

y_drag_centers_init = 0.5 * (y_drag_span[:-1] + y_drag_span[1:])
num_aero_strips = len(y_aero_c)
M_interp_drag = np.zeros((num_drag_strips, num_aero_strips))
for i, yd in enumerate(y_drag_centers_init):
    if yd <= y_aero_c[0]:
        M_interp_drag[i, 0] = 1.0
    elif yd >= y_aero_c[-1]:
        M_interp_drag[i, -1] = 1.0
    else:
        k = np.searchsorted(y_aero_c, yd) - 1
        t = (yd - y_aero_c[k]) / (y_aero_c[k+1] - y_aero_c[k])
        M_interp_drag[i, k] = 1.0 - t
        M_interp_drag[i, k+1] = t

# Compute B-spline spaces and matrices for drag strip lift coefficient fitting
# Exact 101-CP cubic B-spline fit with root symmetry S'(0)=0 for the 100 drag strip collocation points
tau_drag_init = y_drag_centers_init / b_tip_ref
tau_drag_colloc = np.concatenate([[0.0], tau_drag_init])
int_knots_cl_101 = [np.mean(tau_drag_colloc[j:j+3]) for j in range(1, 98)]
knots_cl_101 = np.concatenate([[0.0, 0.0, 0.0, 0.0], int_knots_cl_101, [1.0, 1.0, 1.0, 1.0]])

cl_ss_fit_space = lfs.BSplineSpace(
    num_parametric_dimensions=1,
    degree=3,
    coefficients_shape=(101,),
    knots=(knots_cl_101,),
)

B100_cl_matrix = cl_ss_fit_space.compute_basis_matrix(tau_drag_init.reshape((-1, 1))).toarray()
d_root_cl_row = np.zeros((1, 101))
d_root_cl_row[0, 0] = -1.0
d_root_cl_row[0, 1] = 1.0

M_cl_ss_sys = np.vstack([B100_cl_matrix, d_root_cl_row])
M_cl_ss_inv = np.linalg.inv(M_cl_ss_sys)

# Evaluation points corresponding to the design variable stations (peaks) and dense intermediate points
# to ensure continuous stall progression envelopment across the entire span
station_cl_peaks = np.linspace(0.0, 1.0, num_stations)
num_intermediate_cl = 10
m_cl = num_intermediate_cl // 2

eval_cl_points_list = []
station_cl_peak_indices = []

for i in range(num_stations - 1):
    station_cl_peak_indices.append(len(eval_cl_points_list))
    eval_cl_points_list.append(station_cl_peaks[i])
    pts = np.linspace(station_cl_peaks[i], station_cl_peaks[i+1], num_intermediate_cl + 2)[1:-1]
    eval_cl_points_list.extend(pts)
station_cl_peak_indices.append(len(eval_cl_points_list))
eval_cl_points_list.append(station_cl_peaks[-1])

eval_cl_points_arr = np.array(eval_cl_points_list).reshape((-1, 1))

# Build station grouping indices for aggregation with smooth boundary overlap
station_cl_group_indices = []
total_eval_cl_pts = len(eval_cl_points_list)
for i in range(num_stations):
    p_idx = station_cl_peak_indices[i]
    left = 0 if i == 0 else p_idx - m_cl
    right = total_eval_cl_pts - 1 if i == num_stations - 1 else p_idx + m_cl
    left_ol = max(0, left - 1) if i > 0 else 0
    right_ol = min(total_eval_cl_pts - 1, right + 1) if i < num_stations - 1 else right
    station_cl_group_indices.append(list(range(left_ol, right_ol + 1)))

# endregion


# endregion

# region Parameterization Objects (ffd_block and ffd_sectional_parameterization constructed above)
# endregion

# region Define Design Variables and CSDL Parameterization Map
# # Formulation flag:
# 'ar_area'   -> Design variables: chord DVs, Aspect Ratio (AR), and pitch (planform area fixed at 10)
# 'chord_span' -> Design variables: chord stretch DVs, sweep DVs, span stretch DV, elevator angle, pitch, pitch_3g

pitch = csdl.Variable(value=5.*np.pi/180) # pitch angle in radians
elevator_angle = csdl.Variable(value=0.0) # elevator deflection angle in radians
pitch_ss = csdl.Variable(value=10.0*np.pi/180) # pitch angle for pull-up structural sizing maneuver in radians
if include_neg1g_sizing:
    pitch_neg1g = csdl.Variable(value=-5.0*np.pi/180) # pitch angle for -1.0g push-down sizing maneuver in radians
if geonic_payload_mode == 'none':
    payload_cg = csdl.Variable(value=0.25, name='payload_cg') # payload CG location as fraction of root chord (0.05 to 0.95)
if geonic_payload_mode in {'oversized', 'both'}:
    # Undeformed root mid-chord = 0.5 * scale_factor (3.75 m for scale_factor = 7.5)
    payload_center_x = csdl.Variable(value=0.5 * scale_factor, name='payload_center_x')
if geonic_payload_mode in {'pallets', 'both'}:
    # Pallet stack volume-weighted CG DV
    pallet_stack_center_x = csdl.Variable(value=0.5 * scale_factor, name='pallet_stack_center_x')

@dataclass
class DVInfo:
    variable: csdl.Variable
    lower: Union[float, npt.NDArray[np.float64]]
    upper: Union[float, npt.NDArray[np.float64]]
    scaler: float = 1.0

init_file = 'rectangular_wing_to_bwb_aerostructural_optimization_outputs/2026-09-09_12.44.58.866121/x.out'

thickness_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=2, coefficients_shape=(num_thickness_stations,))
ttop_dvs = csdl.Variable(shape=(num_thickness_stations,), value=np.ones(num_thickness_stations) * 0.01)
tweb_dvs = csdl.Variable(shape=(num_thickness_stations,), value=np.ones(num_thickness_stations) * 0.01)

twist_lower = np.full(num_chord_stations, -15.0 * np.pi / 180.0)
twist_lower[0] = 0.0  # fix root twist to 0
twist_upper = np.full(num_chord_stations, 15.0 * np.pi / 180.0)
twist_upper[0] = 0.0

camber_max_percent = 6.0  # 6.0% chord max camber displacement
camber_lower = -camber_max_percent
camber_upper = camber_max_percent
camber_scaler = 1.0  # Scales DVs with unit scaler (dampens aggressive optimizer camber updates)

num_chordwise_thick_dvs = 5 if include_te_thickness else 4
thick_shape_lower = np.full((num_chordwise_thick_dvs, num_chord_stations), -50.0)  # Max -50% thickness reduction
thick_shape_lower[0, :] = -10.0  # LE cannot have negative thickness change
if include_te_thickness:
    thick_shape_lower[-1, :] = 0.0  # TE cannot have negative thickness change

thick_shape_upper = np.full((num_chordwise_thick_dvs, num_chord_stations), 50.0)  # Max +50% thickness increase
if include_te_thickness:
    thick_shape_upper[-1, :] = 10.0  # Max blunt TE

thick_shape_scaler = 1.0 / 10.0  # Scales [-50.0, 50.0] with 1/10 scaler (dampens aggressive thickness updates)

if formulation == 'ar_area':
    # Formulation 1: Cubic B-spline Target Regularization
    # Taper ratio control points (stations 1 to num_stations-1) + Aspect Ratio (AR) + Sweep Angle Control Points + Elevator + Pitch + Pitch 2.5g + Linear Twist
    taper_control_points = csdl.Variable(shape=(num_chord_stations - 1,), value=np.ones(num_chord_stations - 1))
    taper_dvs = taper_control_points  # backward-compatibility alias
    aspect_ratio = csdl.Variable(shape=(1,), value=np.array([10.0]))
    sweep_angle_control_points = csdl.Variable(shape=(num_chord_stations - 1,), value=np.zeros(num_chord_stations - 1))
    sweep_angle_dvs = sweep_angle_control_points  # backward-compatibility alias
    tc_target_control_points = csdl.Variable(shape=(num_chord_stations,), value=np.full(num_chord_stations, 0.12), name='tc_target_control_points')
    twist_dvs = csdl.Variable(shape=(num_chord_stations,), value=np.zeros(num_chord_stations))

    tc_lower = np.full(num_chord_stations, 0.06)
    tc_upper = np.full(num_chord_stations, 0.35)
    tc_scaler = 10.0

    design_variables: dict[str, DVInfo] = {
        'taper_control_points': DVInfo(variable=taper_control_points, lower=0.05, upper=1.25, scaler=1.0),
        'aspect_ratio': DVInfo(variable=aspect_ratio, lower=2.0, upper=15.0, scaler=0.5),
        'sweep_angle_control_points': DVInfo(variable=sweep_angle_control_points, lower=0.0*np.pi/180, upper=60.0*np.pi/180, scaler=1.e1),
        'tc_target_control_points': DVInfo(variable=tc_target_control_points, lower=tc_lower, upper=tc_upper, scaler=tc_scaler),
        'twist_dvs': DVInfo(variable=twist_dvs, lower=twist_lower, upper=twist_upper, scaler=1.e1),
        'pitch': DVInfo(variable=pitch, lower=-10.0*np.pi/180, upper=15.0*np.pi/180, scaler=1.e1),
        'pitch_ss': DVInfo(variable=pitch_ss, lower=0.0*np.pi/180, upper=35.0*np.pi/180, scaler=1.e1),
    }
    if geonic_payload_mode == 'none':
        design_variables['payload_cg'] = DVInfo(
            variable=payload_cg, lower=0.05, upper=0.95, scaler=1.e1
        )
    if geonic_payload_mode in {'oversized', 'both'}:
        design_variables['payload_center_x'] = DVInfo(
            variable=payload_center_x, lower=0.0, upper=10.0, scaler=1.0 / scale_factor
        )
    if geonic_payload_mode in {'pallets', 'both'}:
        design_variables['pallet_stack_center_x'] = DVInfo(
            variable=pallet_stack_center_x, lower=0.0, upper=10.0, scaler=1.0 / scale_factor
        )
    if use_geonic:
        planform_area_target = csdl.Variable(value=10.0 * scale_factor**2, name='planform_area_target')
        area_lower = 1.0 * scale_factor**2   # 56.25 m^2 (vs 562.5 m^2 baseline)
        area_upper = 30.0 * scale_factor**2  # 1687.5 m^2
        area_scaler = 1.0 / (10.0 * scale_factor**2)
        design_variables['planform_area_target'] = DVInfo(
            variable=planform_area_target, lower=area_lower, upper=area_upper, scaler=area_scaler
        )
    design_variables['ttop_dvs'] = DVInfo(variable=ttop_dvs, lower=0.0001, upper=0.5, scaler=5.e1)
    design_variables['tweb_dvs'] = DVInfo(variable=tweb_dvs, lower=0.0001, upper=0.5, scaler=5.e1)
    if include_elevator:
        design_variables['elevator_angle'] = DVInfo(variable=elevator_angle, lower=-25.0*np.pi/180, upper=25.0*np.pi/180, scaler=1.e1)
    if include_camber:
        camber_dvs = csdl.Variable(shape=(3, num_chord_stations), value=np.zeros((3, num_chord_stations)))
        design_variables['camber_dvs'] = DVInfo(variable=camber_dvs, lower=camber_lower, upper=camber_upper, scaler=camber_scaler)
    if include_thickness_shape:
        thickness_shape_dvs = csdl.Variable(
            shape=(num_chordwise_thick_dvs, num_chord_stations),
            value=np.zeros((num_chordwise_thick_dvs, num_chord_stations)),
            name='thickness_shape_dvs',
        )
        design_variables['thickness_shape_dvs'] = DVInfo(
            variable=thickness_shape_dvs, lower=thick_shape_lower, upper=thick_shape_upper, scaler=thick_shape_scaler
        )
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
    init_ttop_val = np.ones(num_thickness_stations) * 0.02
    init_tweb_val = np.ones(num_thickness_stations) * 0.02

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
    chord_stretch_lower = -0.85 * initial_chord  # (prevents chord collapsing below 15% of baseline)
    chord_stretch_upper = 4.0 * initial_chord    

    initial_thickness = 0.12 * initial_chord  # 0.0379 m baseline NACA 0012 thickness
    thickness_stretch_lower = -0.95 * initial_thickness
    thickness_stretch_upper = 4.0 * initial_thickness    # 0.1518 m

    # sweep_lower = np.full(num_chord_stations, -0.5 * scale_factor)
    sweep_lower = np.full(num_chord_stations, -0. * scale_factor)
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
    }
    if geonic_payload_mode == 'none':
        design_variables['payload_cg'] = DVInfo(
            variable=payload_cg, lower=0.05, upper=0.95, scaler=1.e1
        )
    if geonic_payload_mode in {'oversized', 'both'}:
        design_variables['payload_center_x'] = DVInfo(
            variable=payload_center_x, lower=0.0, upper=10.0, scaler=1.0 / scale_factor
        )
    if geonic_payload_mode in {'pallets', 'both'}:
        design_variables['pallet_stack_center_x'] = DVInfo(
            variable=pallet_stack_center_x, lower=0.0, upper=10.0, scaler=1.0 / scale_factor
        )
    design_variables['ttop_dvs'] = DVInfo(variable=ttop_dvs, lower=0.0001, upper=0.5, scaler=5.e1)
    design_variables['tweb_dvs'] = DVInfo(variable=tweb_dvs, lower=0.0001, upper=0.5, scaler=5.e1)
    if include_elevator:
        design_variables['elevator_angle'] = DVInfo(variable=elevator_angle, lower=-25.0*np.pi/180, upper=25.0*np.pi/180, scaler=1.e1)
    if include_camber:
        camber_dvs = csdl.Variable(shape=(3, num_chord_stations), value=np.zeros((3, num_chord_stations)))
        design_variables['camber_dvs'] = DVInfo(variable=camber_dvs, lower=camber_lower, upper=camber_upper, scaler=camber_scaler)
    if include_thickness_shape:
        thickness_shape_dvs = csdl.Variable(
            shape=(num_chordwise_thick_dvs, num_chord_stations),
            value=np.zeros((num_chordwise_thick_dvs, num_chord_stations)),
            name='thickness_shape_dvs',
        )
        design_variables['thickness_shape_dvs'] = DVInfo(
            variable=thickness_shape_dvs, lower=thick_shape_lower, upper=thick_shape_upper, scaler=thick_shape_scaler
        )
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

if include_camber or include_thickness_shape:
    # Section chord calculated from difference in x coordinate between leading and trailing FFD control points
    section_chords = ffd_coefficients[-1, :, 0, 0] - ffd_coefficients[0, :, 0, 0]

    if include_camber:
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

    if include_thickness_shape:
        full_span_thick_list = []
        for c in range(num_chordwise_thick_dvs):
            row = csdl.concatenate(
                [thickness_shape_dvs[c, i] for i in range(num_chord_stations - 1, 0, -1)] +
                [thickness_shape_dvs[c, i] for i in range(num_chord_stations)]
            )
            full_span_thick_list.append(csdl.reshape(row, (1, num_ffd_sections)))
        full_span_thick = csdl.concatenate(full_span_thick_list, axis=0)  # shape (num_chordwise_thick_dvs, num_ffd_sections)

        # Convert thickness percentage to physical vertical displacement for each section (half-thickness delta)
        row_end = 5 if include_te_thickness else 4
        section_thicknesses = ffd_coefficients[0:row_end, :, 1, 2] - ffd_coefficients[0:row_end, :, 0, 2]
        thick_displacement = (full_span_thick / 100.0) * section_thicknesses
        half_dt = 0.5 * thick_displacement

        # Combine lower (-0.5 * dt) and upper (+0.5 * dt) shifts into single atomic FFD slice update
        dt_pair = csdl.concatenate(
            [csdl.reshape(-half_dt, (row_end, num_ffd_sections, 1)),
             csdl.reshape(half_dt, (row_end, num_ffd_sections, 1))],
            axis=2
        )
        ffd_coefficients = ffd_coefficients.set(
            csdl.slice[0:row_end, :, :, 2],
            ffd_coefficients[0:row_end, :, :, 2] + dt_pair
        )

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

# Evaluate local sectional chords, thicknesses, and spanwise coordinates at right half-span stations
chord_le_pts = [geometry.evaluate(chord_le_projections[i]) for i in range(num_chord_stations)]
chord_te_pts = [geometry.evaluate(chord_te_projections[i]) for i in range(num_chord_stations)]
local_chords = [
    chord_te_pts[i][0] - chord_le_pts[i][0]
    for i in range(num_chord_stations)
]
local_thicknesses = [
    geometry.evaluate(upper_thickness_projections[i])[2] - geometry.evaluate(lower_thickness_projections[i])[2]
    for i in range(num_chord_stations)
]
# Planform area computed via midpoint/trapezoid rule on dense spanwise stations matching wireframe grid (half-wing * 2 for symmetry):
area_le_pts = geometry.evaluate(projected_area_le, plot=False)
area_te_pts = geometry.evaluate(projected_area_te, plot=False)
dense_chords = area_te_pts[:, 0] - area_le_pts[:, 0]
dense_y = area_le_pts[:, 1]

c_mid = 0.5 * (dense_chords[:-1] + dense_chords[1:])
dy = dense_y[1:] - dense_y[:-1]
half_planform_area = csdl.sum(c_mid * dy)
planform_area = 2.0 * half_planform_area

aspect_ratio_calc = (wingspan**2) / planform_area

if formulation == 'ar_area':
    # ParameterizationSolver manipulates states to match targets
    geometry_solver = ParameterizationSolver()
    geometry_solver.add_state(chord_stretch_states)
    geometry_solver.add_state(thickness_stretch_states)
    geometry_solver.add_state(span_stretch_state)
    geometry_solver.add_state(sweep_translation_states)

    bspline_reg = BsplineTargetRegularization(
        num_chord_stations=num_chord_stations,
        scale_factor=scale_factor,
        eta_stations=eta_stations,
    )

    geometric_variables = GeometricVariables()

    local_chords_vec = csdl.concatenate([csdl.reshape(c, (1,)) for c in local_chords])
    local_thicknesses_vec = csdl.concatenate([csdl.reshape(t, (1,)) for t in local_thicknesses])

    # 1. Enforce normalized chord profile (taper ratios) via Galerkin weak form (n - 1 equations)
    c_ref_scale = float(np.asarray(local_chords[0].value).flatten()[0]) if local_chords[0].value is not None else 1.0 * scale_factor
    res_taper = bspline_reg.compute_taper_residual(local_chords_vec, taper_control_points, c_ref=c_ref_scale)
    # for i in range(num_chord_stations - 1):
    #     geometric_variables.add_variable(res_taper[i], 0.0, penalty_value=None)
    geometric_variables.add_variable(res_taper, 0.0, penalty_value=None)  # shape (num_chord_stations - 1,)

    # 2. Enforce thickness-to-chord ratio target via Galerkin weak form (n equations)
    t_ref_scale = 0.12 * c_ref_scale
    res_thick = bspline_reg.compute_thickness_residual(
        local_thicknesses_vec, local_chords_vec, tc_target_control_points, t_ref=t_ref_scale
    )
    # for i in range(num_chord_stations):
    #     geometric_variables.add_variable(res_thick[i], 0.0, penalty_value=None)
    geometric_variables.add_variable(res_thick, 0.0, penalty_value=None)  # shape (num_chord_stations,)

    # 3. Enforce planform area and aspect ratio simultaneously (2 equations)
    if use_geonic:
        geometric_variables.add_variable(
            planform_area / (10*scale_factor**2),
            planform_area_target / (10*scale_factor**2),
            penalty_value=None
        )
        geometric_variables.add_variable(
            wingspan**2 / (10.0 * 10*scale_factor**2),
            aspect_ratio * planform_area_target / (10.0 * 10*scale_factor**2),
            penalty_value=None
        )
    else:
        geometric_variables.add_variable(
            planform_area / (10*scale_factor**2),
            planform_area.value / (10*scale_factor**2),
            penalty_value=None
        )
        geometric_variables.add_variable(
            wingspan**2 / (10.0 * 10*scale_factor**2),
            aspect_ratio * planform_area.value / (10.0 * 10*scale_factor**2),
            penalty_value=None
        )

    # 4. Enforce sectional sweep angles via Galerkin weak form (n - 1 equations)
    qc_pts = [geometry.evaluate(quarter_chord_projections[i]) for i in range(num_chord_stations)]
    dx_list = []
    dy_list = []
    for i in range(num_chord_stations - 1):
        dx_list.append(csdl.reshape(qc_pts[i + 1][0] - qc_pts[i][0], (1,)))
        dy_list.append(csdl.reshape(qc_pts[i + 1][1] - qc_pts[i][1], (1,)))
    dx_qc_vec = csdl.concatenate(dx_list)
    dy_qc_vec = csdl.concatenate(dy_list)

    y_scale = 5.0 * scale_factor
    res_sweep = bspline_reg.compute_sweep_residual(
        dx_qc_vec, dy_qc_vec, sweep_angle_control_points, y_scale=y_scale
    )
    # for i in range(num_chord_stations - 1):
    #     geometric_variables.add_variable(res_sweep[i], 0.0, penalty_value=None)
    geometric_variables.add_variable(res_sweep, 0.0, penalty_value=None)  # shape (num_chord_stations - 1,)

    geometry_solver.evaluate(geometric_variables)

    # Dense diagnostics
    ar_area_diagnostics = bspline_reg.compute_diagnostics(
        local_chords=local_chords_vec,
        local_thicknesses=local_thicknesses_vec,
        dx_qc=dx_qc_vec,
        dy_qc=dy_qc_vec,
        taper_control_points=taper_control_points,
        tc_target_control_points=tc_target_control_points,
        sweep_angle_control_points=sweep_angle_control_points,
    )

# Right-half sectional quarter-chord sweep between successive evaluated quarter-chord stations:
# Lambda_qc[i] = atan2(x_qc[i+1]-x_qc[i], y_qc[i+1]-y_qc[i])
qc_pts_right = [geometry.evaluate(quarter_chord_projections[i]) for i in range(num_chord_stations)]
qc_pts_sweep = [geometry.evaluate(quarter_chord_sweep_projections[i]) for i in range(num_sweep_eval_stations)]
lambda_qc_list = []
for i in range(num_sweep_eval_stations - 1):
    dx_qc = qc_pts_sweep[i + 1][0] - qc_pts_sweep[i][0]
    dy_qc = qc_pts_sweep[i + 1][1] - qc_pts_sweep[i][1]
    sw_qc = csdl.arctan2(dx_qc, dy_qc)
    lambda_qc_list.append(csdl.reshape(sw_qc, (1,)))

lambda_qc_vec = csdl.concatenate(lambda_qc_list)  # shape (num_sweep_eval_stations - 1,)
max_quarter_chord_sweep = csdl.maximum(lambda_qc_vec, axes=(0,), rho=50.0)
quarter_chord_sweep_margin = (60.0 * np.pi / 180.0) - max_quarter_chord_sweep

quarter_chord_rot_origin = geometry.evaluate(quarter_chord_center)

if use_geonic:
    # Build sample points according to mode
    if geonic_payload_mode == 'oversized':
        payload_sample_points = build_geonic_payload_sample_points(payload_center_x)
        payload_sample_points.name = 'payload_sample_points'
        active_sample_points = payload_sample_points
    elif geonic_payload_mode == 'pallets':
        pallet_stack_x_shift = pallet_stack_center_x - PALLET_STACK_REFERENCE_CG_X
        pallet_stack_x_shift.name = 'pallet_stack_x_shift'
        pallet_sample_points = build_geonic_pallet_sample_points(pallet_stack_center_x)
        pallet_sample_points.name = 'pallet_sample_points'
        active_sample_points = pallet_sample_points
    elif geonic_payload_mode == 'both':
        pallet_stack_x_shift = pallet_stack_center_x - PALLET_STACK_REFERENCE_CG_X
        pallet_stack_x_shift.name = 'pallet_stack_x_shift'
        payload_sample_points = build_geonic_payload_sample_points(payload_center_x)
        payload_sample_points.name = 'payload_sample_points'
        pallet_sample_points = build_geonic_pallet_sample_points(pallet_stack_center_x)
        pallet_sample_points.name = 'pallet_sample_points'
        active_sample_points = csdl.concatenate([payload_sample_points, pallet_sample_points])

    # Complete geometry function set passed for enclosed-volume signed distance
    projection_model = bsm3.FunctionSetProjectionModel(
        function_set=geometry,
        warm_start_nu=50,
        warm_start_nv=50,
        sdf=True,
        sdf_sign_mode='enclosed',
    )
    sdf_op = bsm3.FunctionSetClosestDistanceOperation(model=projection_model)
    all_signed_distance = sdf_op.evaluate(
        coefficients=geometry.stack_coefficients(),
        points=active_sample_points,
    )

    if geonic_payload_mode == 'oversized':
        payload_signed_distance = all_signed_distance
        payload_signed_distance.name = 'payload_signed_distance'
        geonic_constraint_values, geonic_clearance_per_point, geonic_margin = (
            compute_geonic_clearance_and_margin(payload_signed_distance, clearance_buffer=GEONIC_CLEARANCE_M, rho=50.0)
        )
        geonic_constraint_values.name = 'geonic_payload_oml_clearance'
        geonic_clearance_per_point.name = 'geonic_clearance_per_point'
        geonic_margin.name = 'geonic_margin'
        geonic_constraint_values.set_as_constraint(upper=0.0, scaler=1.0)

    elif geonic_payload_mode == 'pallets':
        pallet_signed_distance = all_signed_distance
        pallet_signed_distance.name = 'pallet_signed_distance'
        pallet_constraint_values, pallet_clearance_per_point, pallet_margin = (
            compute_geonic_clearance_and_margin(pallet_signed_distance, clearance_buffer=GEONIC_CLEARANCE_M, rho=50.0)
        )
        pallet_constraint_values.name = 'geonic_pallet_oml_clearance'
        pallet_clearance_per_point.name = 'pallet_clearance_per_point'
        pallet_margin.name = 'pallet_margin'
        pallet_constraint_values.set_as_constraint(upper=0.0, scaler=1.0)
        geonic_margin = pallet_margin

    elif geonic_payload_mode == 'both':
        num_pay_pts = int(payload_sample_points.shape[0])
        payload_signed_distance = all_signed_distance[:num_pay_pts]
        payload_signed_distance.name = 'payload_signed_distance'
        pallet_signed_distance = all_signed_distance[num_pay_pts:]
        pallet_signed_distance.name = 'pallet_signed_distance'

        geonic_constraint_values, geonic_clearance_per_point, geonic_margin = (
            compute_geonic_clearance_and_margin(payload_signed_distance, clearance_buffer=GEONIC_CLEARANCE_M, rho=50.0)
        )
        geonic_constraint_values.name = 'geonic_payload_oml_clearance'
        geonic_clearance_per_point.name = 'geonic_clearance_per_point'
        geonic_margin.name = 'geonic_margin'
        geonic_constraint_values.set_as_constraint(upper=0.0, scaler=1.0)

        pallet_constraint_values, pallet_clearance_per_point, pallet_margin = (
            compute_geonic_clearance_and_margin(pallet_signed_distance, clearance_buffer=GEONIC_CLEARANCE_M, rho=50.0)
        )
        pallet_constraint_values.name = 'geonic_pallet_oml_clearance'
        pallet_clearance_per_point.name = 'pallet_clearance_per_point'
        pallet_margin.name = 'pallet_margin'
        pallet_constraint_values.set_as_constraint(upper=0.0, scaler=1.0)

        # Overall diagnostic margin
        all_constraint_values = all_signed_distance + GEONIC_CLEARANCE_M
        overall_geonic_margin = -csdl.maximum(all_constraint_values, axes=(0,), rho=50.0)
        overall_geonic_margin.name = 'overall_geonic_margin'

        # CG consistency equality constraint: payload_center_x - pallet_stack_center_x = 0
        payload_cg_consistency = payload_center_x - pallet_stack_center_x
        payload_cg_consistency.name = 'payload_cg_consistency_constraint'
        payload_cg_consistency.set_as_constraint(equals=0.0, scaler=1.0 / scale_factor)

    # Compute rotated inertial CGs for active cases
    if geonic_payload_mode in {'oversized', 'both'}:
        payload_center_body = csdl.reshape(
            csdl.concatenate([
                csdl.reshape(payload_center_x, (1,)),
                csdl.Variable(value=np.array([0.0])),
                csdl.Variable(value=np.array([0.0])),
            ]),
            (1, 3),
        )
        payload_center_body.name = 'payload_center_body'
        payload_center_inertial = rotate(
            points=payload_center_body,
            rotation_origin=quarter_chord_rot_origin,
            axis_vector=np.array([0., 1., 0.]),
            angles=pitch,
            units='radians',
        )
        payload_center_inertial = csdl.reshape(payload_center_inertial, (3,))
        payload_center_inertial.name = 'payload_center_inertial'

    if geonic_payload_mode in {'pallets', 'both'}:
        pallet_stack_center_body = csdl.reshape(
            csdl.concatenate([
                csdl.reshape(pallet_stack_center_x, (1,)),
                csdl.Variable(value=np.array([0.0])),
                csdl.Variable(value=np.array([0.0])),
            ]),
            (1, 3),
        )
        pallet_stack_center_body.name = 'pallet_stack_center_body'
        pallet_stack_center_inertial = rotate(
            points=pallet_stack_center_body,
            rotation_origin=quarter_chord_rot_origin,
            axis_vector=np.array([0., 1., 0.]),
            angles=pitch,
            units='radians',
        )
        pallet_stack_center_inertial = csdl.reshape(pallet_stack_center_inertial, (3,))
        pallet_stack_center_inertial.name = 'pallet_stack_center_inertial'

geometry.rotate(rotation_origin=quarter_chord_rot_origin, axis_vector=np.array([0., 1., 0.]), angles=pitch, units='radians')

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
    sizing_speed = 1.2 * cruise_speed  # (1.2x cruise speed) structural sizing maneuver speed
    payload_weight = csdl.Variable(value=160000.*4.44822)  # 711.86 kN = 160,000 lbf (4.44822 N/lbf) payload weight
    # misc_weight scaled with (q_M0.70 / q_M0.75) = (0.70/0.75)^2 to keep cruise CL consistent:
    # Total fixed weight: 500,000 lbf * (0.70/0.75)^2 = 574,933.33 lbf -> misc = 414,933.33 lbf
    # misc_weight = csdl.Variable(value=(500000. * (0.70 / 0.75)**2 - 160000.) * 4.44822)
    misc_weight = csdl.Variable(value=6.e5)  # Total payload + misc should be roughly around 130k kg
    # for now, just add misc weight to payload weight cause it's easier, but separate these out later
    payload_weight = payload_weight + misc_weight
    load_factor = csdl.Variable(value=2.5)  # 2.5g load factor for sizing maneuver
    # load_factor = csdl.Variable(value=3.0)  # 3.0g load factor for sizing maneuver
else:
    raise Exception("Cruise speed not defined for scale factor = {}. Set the cruise speed for this scale factor.".format(scale_factor))

load_factor_val = float(np.asarray(load_factor.value).flatten()[0]) if hasattr(load_factor, 'value') else float(load_factor)

# Flight conditions:
# Node 0 = cruise condition (ISA 30,000 ft, Mach 0.70)
# Node 1 = stability condition (+1.0 deg alpha perturbation, ISA 30,000 ft, Mach 0.70)
# Node 2 = structural sizing pull-up condition (sea level, q_sizing = 1.25^2 * q_cruise, pitch angle = pitch_ss)
# Node 3 = -1.0g push-down structural sizing condition (optional, sea level, pitch angle = pitch_neg1g)
flight_cond_dict = get_nominal_flight_conditions()
cruise_cond = flight_cond_dict['cruise']
sizing_cond = flight_cond_dict['sizing']

V_cruise_val = cruise_cond['speed_m_s']
V_sizing_val = sizing_cond['speed_m_s']
q_cruise_val = cruise_cond['dynamic_pressure_Pa']
q_sizing_val = sizing_cond['dynamic_pressure_Pa']

cruise_speed = csdl.Variable(value=V_cruise_val)
sizing_speed = csdl.Variable(value=V_sizing_val)

# Atmospheric state vectors for all nodes
# Node 0: cruise, Node 1: stability (+1 deg), Node 2: pull-up, Node 3: optional -1.0g
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
    
    node_altitudes = np.array([CRUISE_ALTITUDE_M, CRUISE_ALTITUDE_M, 0.0, 0.0])
    node_rho = np.array([cruise_cond['density_kg_m3'], cruise_cond['density_kg_m3'], sizing_cond['density_kg_m3'], sizing_cond['density_kg_m3']])
    node_sos = np.array([cruise_cond['speed_of_sound_m_s'], cruise_cond['speed_of_sound_m_s'], sizing_cond['speed_of_sound_m_s'], sizing_cond['speed_of_sound_m_s']])
    node_mach = np.array([cruise_cond['mach'], cruise_cond['mach'], sizing_cond['mach'], sizing_cond['mach']])
    node_temp = np.array([cruise_cond['temperature_K'], cruise_cond['temperature_K'], sizing_cond['temperature_K'], sizing_cond['temperature_K']])
    node_pressure = np.array([cruise_cond['pressure_Pa'], cruise_cond['pressure_Pa'], sizing_cond['pressure_Pa'], sizing_cond['pressure_Pa']])
    node_viscosity = np.array([cruise_cond['viscosity_Pa_s'], cruise_cond['viscosity_Pa_s'], sizing_cond['viscosity_Pa_s'], sizing_cond['viscosity_Pa_s']])
    node_q = np.array([q_cruise_val, q_cruise_val, q_sizing_val, q_sizing_val])
    node_speeds = np.array([V_cruise_val, V_cruise_val, V_sizing_val, V_sizing_val])
    
    rho_array = csdl.Variable(shape=(num_nodes,), value=node_rho)
    sos_array = csdl.Variable(shape=(num_nodes,), value=node_sos)
else:
    num_nodes = 3
    dalpha_rad = 1.0 * np.pi / 180.0
    dalpha_ss = pitch_ss - pitch
    v0 = csdl.concatenate([cruise_speed, csdl.Variable(value=0.0), csdl.Variable(value=0.0)])
    v1 = csdl.concatenate([cruise_speed * np.cos(dalpha_rad), csdl.Variable(value=0.0), cruise_speed * np.sin(dalpha_rad)])
    v2 = csdl.concatenate([sizing_speed * csdl.cos(dalpha_ss), csdl.Variable(value=0.0), sizing_speed * csdl.sin(dalpha_ss)])
    v_stacked = csdl.reshape(csdl.concatenate([v0, v1, v2]), (3, 3))
    
    node_altitudes = np.array([CRUISE_ALTITUDE_M, CRUISE_ALTITUDE_M, 0.0])
    node_rho = np.array([cruise_cond['density_kg_m3'], cruise_cond['density_kg_m3'], sizing_cond['density_kg_m3']])
    node_sos = np.array([cruise_cond['speed_of_sound_m_s'], cruise_cond['speed_of_sound_m_s'], sizing_cond['speed_of_sound_m_s']])
    node_mach = np.array([cruise_cond['mach'], cruise_cond['mach'], sizing_cond['mach']])
    node_temp = np.array([cruise_cond['temperature_K'], cruise_cond['temperature_K'], sizing_cond['temperature_K']])
    node_pressure = np.array([cruise_cond['pressure_Pa'], cruise_cond['pressure_Pa'], sizing_cond['pressure_Pa']])
    node_viscosity = np.array([cruise_cond['viscosity_Pa_s'], cruise_cond['viscosity_Pa_s'], sizing_cond['viscosity_Pa_s']])
    node_q = np.array([q_cruise_val, q_cruise_val, q_sizing_val])
    node_speeds = np.array([V_cruise_val, V_cruise_val, V_sizing_val])
    
    rho_array = csdl.Variable(shape=(num_nodes,), value=node_rho)
    sos_array = csdl.Variable(shape=(num_nodes,), value=node_sos)

# CSDL variables for atmospheric node diagnostics
atm_altitudes = csdl.Variable(shape=(num_nodes,), value=node_altitudes)
atm_densities = rho_array
atm_sound_speeds = sos_array
atm_mach_numbers = csdl.Variable(shape=(num_nodes,), value=node_mach)
atm_temperatures = csdl.Variable(shape=(num_nodes,), value=node_temp)
atm_pressures = csdl.Variable(shape=(num_nodes,), value=node_pressure)
atm_viscosities = csdl.Variable(shape=(num_nodes,), value=node_viscosity)
atm_dynamic_pressures = csdl.Variable(shape=(num_nodes,), value=node_q)
atm_speeds = csdl.Variable(shape=(num_nodes,), value=node_speeds)
atm_reynolds_per_unit_chord = csdl.Variable(shape=(num_nodes,), value=node_rho * node_speeds / node_viscosity)

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
strip_upper_mesh = geometry.evaluate(projected_strip_upper_skin, plot=False)
strip_lower_mesh = geometry.evaluate(projected_strip_lower_skin, plot=False)

# Compute local chord from projected mesh (leading edge to trailing edge distance)
node_chords = te_mesh[:,0] - le_mesh[:,0]
local_chord = 0.5 * (node_chords[:-1] + node_chords[1:])

# Compute area-weighted wingbox height across 20% to 70% chord via composite Simpson's rule
box_diff_z = upper_beam_mesh[:, 2] - lower_beam_mesh[:, 2]
node_heights = csdl.matvec(W_box_mat, box_diff_z)
local_height = 0.5 * (node_heights[:-1] + node_heights[1:])

# Define wingbox cross-section along the span
# Wingbox width is 50% of local chord; height is scaled by box_height_factor
box_width = 0.50 * local_chord
box_height = box_height_factor * local_height

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
# beam = aframe.Beam(name='wing_spar', mesh=beam_mesh, E=69e9, G=26e9, density=1., cs=beam_cs)

# Fix root node at y=0 (clamped cantilever symmetry boundary condition)
beam.fix(node=0)

# Compute structural mass (doubled for full wing) and structural center of mass from beam model
structural_mass = 2.0 * beam.mass
structural_cg = beam.cg
x_struct = structural_cg[0]
z_struct = structural_cg[2]

# Compute payload location
x_le_root = le_mesh[0, 0]
x_te_root = te_mesh[0, 0]
root_chord = x_te_root - x_le_root
if use_geonic:
    # Use rotated inertial payload center for mass/CG consistency
    if geonic_payload_mode == 'oversized':
        active_inertial_cg = payload_center_inertial
    elif geonic_payload_mode == 'pallets':
        active_inertial_cg = pallet_stack_center_inertial
    elif geonic_payload_mode == 'both':
        # Arithmetic mean of the two inertial CGs during optimization; equals both at feasibility
        active_inertial_cg = 0.5 * (payload_center_inertial + pallet_stack_center_inertial)
    x_payload = active_inertial_cg[0]
    y_payload = active_inertial_cg[1]
    z_payload = active_inertial_cg[2]
else:
    x_payload = x_le_root + payload_cg * root_chord
    y_payload = csdl.Variable(value=np.array([0.0]))
    z_payload = 0.5 * (upper_beam_mesh[0, 2] + lower_beam_mesh[0, 2])

payload_mass = payload_weight / 9.81
W_total = structural_mass * 9.81 + payload_weight
total_mass = structural_mass + payload_mass

# Dynamic composite aircraft Center of Mass (CG) updated each iteration
x_cg = (structural_mass * x_struct + payload_mass * x_payload) / total_mass
z_cg = (structural_mass * z_struct + payload_mass * z_payload) / total_mass
r_cg = csdl.concatenate([csdl.reshape(x_cg, (1,)), csdl.reshape(y_payload, (1,)), csdl.reshape(z_cg, (1,))])
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
    'drag_type' : 'Trefftz_2D_consistent',
    'nwpcl': 200,
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
    'L_panel',
    'CL',
    'CDi',
    'CDi_Trefftz',
    'lift_ratio',
    'CM',
    'mu_w',
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
mu_w = outputs['mu_w']
trefftz_lift_ratio = outputs['lift_ratio']

# endregion Aerodynamic solver (panel method)

# region Structural solver (beam loads & stress)
# Rigorously map aerodynamic panel forces from right half-span panels to beam structural nodes
dynamic_panel_centers = geometry.evaluate(projected_panel_centers, plot=False)
dynamic_panel_centers_right = dynamic_panel_centers[:num_right_panels, :]
panel_forces_right_cruise = outputs['panel_forces'][0, :num_right_panels, :] # shape (num_right_panels, 3) from Node 0 (cruise)
panel_lift_right_cruise = outputs['L_panel'][0, :num_right_panels]
panel_forces_right_ss = outputs['panel_forces'][2, :num_right_panels, :] # shape (num_right_panels, 3) from Node 2 (structural sizing pull-up)
if include_neg1g_sizing:
    panel_forces_right_neg1g = outputs['panel_forces'][3, :num_right_panels, :] # shape (num_right_panels, 3) from Node 3 (-1.0g push-down)

# Lifting-line Fourier induced drag calculation from cruise vertical panel forces
F_z_cruise = panel_forces_right_cruise[:, 2]
T_fourier_var = csdl.Variable(value=T_fourier_panel)
A_fourier_raw = csdl.matvec(T_fourier_var, F_z_cruise)  # Harmonic expansion coefficients (odd n = 1, 3, ..., 15)
A_fourier_ratios = A_fourier_raw[1:] / A_fourier_raw[0]

n_fourier_weights = csdl.Variable(value=n_fourier_odd[1:])
delta_fourier = csdl.sum(n_fourier_weights * (A_fourier_ratios ** 2))
e_fourier = 1.0 / (1.0 + delta_fourier)

CDi_Fourier = (CL[0] ** 2 / (np.pi * aspect_ratio_calc)) * (1.0 + delta_fourier)
Di_Fourier = CDi_Fourier * 0.5 * rho_array[0] * (cruise_speed ** 2) * planform_area

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

# Fit B-spline stress functions: exact 15-CP cubic B-spline with root symmetry S'(0)=0 for both fast and full resolutions
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

# 2. Strip-Wise Sectional Profile & Stall Drag Model (100 strips, aerodynamic Cl & polar model)
# Evaluate strip center leading and trailing edge points from evaluated geometry:
le_strip_center = 0.5 * (le_drag_mesh[:-1, :] + le_drag_mesh[1:, :])
te_strip_center = 0.5 * (te_drag_mesh[:-1, :] + te_drag_mesh[1:, :])

# Chord vector components: dx (chordwise) and dz (vertical, nose up positive)
dx_strip = te_strip_center[:, 0] - le_strip_center[:, 0]
dz_strip = le_strip_center[:, 2] - te_strip_center[:, 2]

# Exact local angle of attack from geometry using arctan2:
alpha_local_elem = csdl.arctan2(dz_strip, dx_strip)  # shape (num_drag_strips,)

# Geometric local chord length and spanwise strip width dy:
local_chord_drag = csdl.sqrt(dx_strip**2 + dz_strip**2)
y_strip_pts = 0.5 * (le_drag_mesh[:-1, 1] + le_drag_mesh[1:, 1])
dy_strip = le_drag_mesh[1:, 1] - le_drag_mesh[:-1, 1]

# Strip planform area across full wingspan (factor of 2.0 for both wings):
strip_area = 2.0 * local_chord_drag * dy_strip
total_strip_area = csdl.sum(strip_area)

# Live geometric strip thickness-to-chord ratio (t/c)_i evaluated from paired surface projections:
strip_tc = compute_strip_thickness_to_chord(
    upper_surface_pts=strip_upper_mesh,
    lower_surface_pts=strip_lower_mesh,
    grid_shape=strip_grid_shape,
    chord_lengths=local_chord_drag,
    rho=50.0,
)

# Strip quarter-chord points and aerodynamic quarter-chord sweep angle Lambda_c/4:
qc_drag_pts = 0.75 * le_strip_center + 0.25 * te_strip_center  # shape (num_drag_strips, 3)
strip_sweep_qc = compute_quarter_chord_sweep(qc_drag_pts)  # shape (num_drag_strips,)
max_strip_qc_sweep = csdl.maximum(strip_sweep_qc, axes=(0,), rho=50.0)
strip_qc_sweep_excess = max_strip_qc_sweep - (60.0 * np.pi / 180.0)

# Backward-compatible aliases for diagnostics and logging:
mid_chord_drag_pts = 0.5 * (le_strip_center + te_strip_center)
strip_sweep_halfchord = strip_sweep_qc  # Korn-Lock wave drag now uses quarter-chord sweep
max_half_chord_sweep = max_strip_qc_sweep
half_chord_sweep_excess = strip_qc_sweep_excess

# Dynamic spanwise scaling: dy scales with wingspan stretch
span_scale = wingspan / (10.0 * scale_factor)
dy_aero_dyn = csdl.Variable(value=dy_aero_init) * span_scale

# Map VortexAD's L_panel for local section lift coefficient cl across nodes:
# Lprime_i = L_strip_i / dy_aero_dyn
# cl_i = Lprime_i / (q_node * c_i)
q_inf = 0.5 * rho_array[0] * (cruise_speed ** 2)
lift_aero_strips = csdl.matvec(csdl.Variable(value=W_strip), outputs['L_panel'][0, :num_right_panels])
lift_prime_aero = lift_aero_strips / dy_aero_dyn
lift_prime_drag = csdl.matvec(csdl.Variable(value=M_interp_drag), lift_prime_aero)
cl_local_elem = lift_prime_drag / (q_inf * local_chord_drag)

# Evaluate wave drag across all flight nodes (computed for every node, cruise added to objective):
wave_drag_results = []
CD_wave_nodes = []
D_wave_nodes = []
for node_idx in range(num_nodes):
    q_node_val = 0.5 * rho_array[node_idx] * (atm_speeds[node_idx] ** 2)
    l_strip_node = csdl.matvec(csdl.Variable(value=W_strip), outputs['L_panel'][node_idx, :num_right_panels])
    lp_aero_node = l_strip_node / dy_aero_dyn
    lp_drag_node = csdl.matvec(csdl.Variable(value=M_interp_drag), lp_aero_node)
    cl_strip_node = lp_drag_node / (q_node_val * local_chord_drag)
    
    node_wave = evaluate_wave_drag(
        strip_tc=strip_tc,
        strip_sweep=strip_sweep_qc,
        strip_cl=cl_strip_node,
        strip_area=strip_area,
        total_strip_area=total_strip_area,
        mach_node=atm_mach_numbers[node_idx],
        q_node=q_node_val,
        planform_area=planform_area,
    )
    wave_drag_results.append(node_wave)
    CD_wave_nodes.append(csdl.reshape(node_wave['CD_wave'], (1,)))
    D_wave_nodes.append(csdl.reshape(node_wave['D_wave'], (1,)))

CD_wave_array = csdl.concatenate(CD_wave_nodes)
D_wave_array = csdl.concatenate(D_wave_nodes)

# Cruise node wave drag outputs:
CD_wave_cruise = CD_wave_array[0]
D_wave_cruise = D_wave_array[0]
strip_wave_cd_cruise = wave_drag_results[0]['cd_wave_strip']
strip_M_dd_cruise = wave_drag_results[0]['M_dd']
strip_M_crit_cruise = wave_drag_results[0]['M_crit']
strip_delta_M_cruise = wave_drag_results[0]['delta_M']
strip_delta_M_eff_cruise = wave_drag_results[0]['delta_M_eff']
min_strip_mach_margin_cruise = -csdl.maximum(strip_delta_M_cruise, axes=(0,), rho=50.0)  # min(Mcrit - M)

# Base parasite drag coefficient (NACA 0012 base skin friction + form drag: CD0 = 0.0080)
# Viscous drag mode evaluation: 'ibl' (direct integral boundary layer) vs 'constant_cd0' (legacy CD0_base = 0.0080)
CD0_base = 0.0080

ibl_results = evaluate_bwb_viscous_ibl(
    cp_node0=Cp[0],
    dynamic_panel_centers=dynamic_panel_centers,
    topology=ibl_topology,
    v_cruise=cruise_speed,
    rho_cruise=rho_array[0],
    mu_air=csdl.Variable(value=cruise_cond['viscosity_Pa_s']),
    q_inf=q_inf,
    strip_area=strip_area,
    total_strip_area=total_strip_area,
)

CD_viscous_ibl = ibl_results['CD_viscous']
D_viscous_ibl = ibl_results['D_viscous']
CD_ibl_section = ibl_results['CD_ibl_section']
H_max_ibl = ibl_results['H_max_ibl']
ibl_attachment_margin = ibl_results['ibl_attachment_margin']
ibl_min_cp = ibl_results['ibl_min_cp']
ibl_cp_cutoff_margin = ibl_results['ibl_cp_cutoff_margin']
theta_te_upper = ibl_results['theta_te_upper']
theta_te_lower = ibl_results['theta_te_lower']
H_strip = ibl_results['H_strip']
H_section_upper = ibl_results['H_section_upper']
H_section_lower = ibl_results['H_section_lower']
H_section_both = ibl_results['H_section_both']

if viscous_drag_mode == 'ibl':
    cd_viscous_elem = ibl_results['cd_viscous_elem']
    CD_viscous = CD_viscous_ibl
    D_viscous = D_viscous_ibl

    # Fit 101-CP cubic B-spline with root symmetry S'(0)=0 to the 100 drag strip section max H values
    # Reuses the exact precomputed matrix inverse M_cl_ss_inv and B-spline space cl_ss_fit_space
    H_rhs = csdl.concatenate([H_strip, csdl.Variable(value=np.array([0.0]))])
    H_coeffs_flat = csdl.matvec(csdl.Variable(value=M_cl_ss_inv), H_rhs)
    H_coeffs = csdl.reshape(H_coeffs_flat, (101, 1))
    H_func = lfs.Function(space=cl_ss_fit_space, coefficients=H_coeffs)

    # Evaluate at the num_stations design variable station peaks (5 in fast, 8 in full)
    dv_H_stations = H_func.evaluate(station_cl_peaks.reshape((-1, 1)))
    dv_H_stations.name = 'dv_H_stations'

    # Enforce active attachment constraint at each station individually: H <= 2.4, scaled by 0.15
    dv_H_stations.set_as_constraint(upper=2.4, scaler=0.15)
else:
    cd_viscous_elem = CD0_base
    CD_viscous = csdl.Variable(value=CD0_base, name='CD_viscous')
    D_viscous = CD0_base * q_inf * total_strip_area
    D_viscous.name = 'D_viscous'
    dv_H_stations = csdl.Variable(value=np.full((num_stations,), 1.4), name='dv_H_stations')

# 1. Sectional Cl drag polar bucket: regularizes planform by penalizing extreme section Cl
k_polar = 0.010              # Curvature of polar bucket
cl_ideal = 0.50              # Design cruise lift coefficient
cd_polar_elem = k_polar * ((cl_local_elem - cl_ideal) ** 2) if include_cl_polar else 0.0

# 2. Unified aerodynamic section stall parameters (applicable to all flight conditions)
cl_crit = 1.40               # Critical section lift coefficient before stall onset
delta_cl_ref = 0.25          # Scaling width for post-stall transition
beta_cl_stall = 40.0         # Softplus transition sharpness
k_stall_drag = 0.20          # Stall drag scaling factor (cruise)
k_stall_lift = 0.35          # Stall lift deficit scaling factor (cruise & maneuver)

cl_abs = csdl.absolute(cl_local_elem)
delta_cl = cl_abs - cl_crit
softplus_cl_val = (1.0 / beta_cl_stall) * csdl.softplus(beta_cl_stall * delta_cl)
cd_stall_elem = k_stall_drag * ((softplus_cl_val / delta_cl_ref) ** 2) if include_stall_drag else 0.0

# Total profile drag coefficient per strip
cd_profile_elem = CD0_base + cd_polar_elem + cd_stall_elem
cd_profile_elem = cd_viscous_elem + cd_polar_elem + cd_stall_elem

# Sectional and total profile drag [N]
# Normalized strictly by total_strip_area to prevent area-shrinkage numerical loopholes
CD_profile = csdl.sum(cd_profile_elem * strip_area) / total_strip_area
D_profile_elem = cd_profile_elem * q_inf * strip_area
D_profile = csdl.sum(D_profile_elem)

# Sectional lift deficit penalty for cruise
sign_cl = cl_local_elem / (cl_abs + 1.e-6)
cl_stall_loss_elem = k_stall_lift * (softplus_cl_val / delta_cl_ref) if include_stall_drag else 0.0 * cl_local_elem
CL_loss_cruise = csdl.sum(cl_stall_loss_elem * sign_cl * strip_area) / total_strip_area
L_loss_cruise = csdl.sum(cl_stall_loss_elem * sign_cl * q_inf * strip_area)
lift_effective_cruise = L[0] - L_loss_cruise

# Total aircraft drag: Induced Drag + Strip-Wise Profile & Stall Drag + Transonic Wave Drag
Di_chosen = Di_Fourier if induced_drag_objective == 'fourier' else Di_Trefftz
D_total = Di_chosen + D_profile + D_wave_cruise

# Reference values for scaling constraints and objective function
payload_weight_val = float(np.asarray(payload_weight.value).flatten()[0]) if hasattr(payload_weight, 'value') else float(payload_weight)
W_ref = 2.0 * payload_weight_val  # reference cruise weight [N] (~1.15x payload weight)
D_ref = W_ref / 50. # reference drag [N] (~1/20 of payload weight if we assume L/D ~ 20)
c_ref = scale_factor * 1.0 # Initial chord length

# Set active optimization objective based on induced_drag_objective toggle:
# Physical drag force objective: D = 0.5 * rho * V^2 * S * CD
if induced_drag_objective == 'fourier':
    CD_active = CDi_Fourier + CD_profile + CD_wave_cruise
elif induced_drag_objective == 'mixed':
    CD_active = csdl.maximum(CDi_Fourier, CDi[0], rho=2.*1.e3) + CD_profile + CD_wave_cruise
else:
    CD_active = CDi[0] + CD_profile + CD_wave_cruise

# D_total in Newtons = 0.5 * rho * V^2 * S * CD
objective = 0.5 * rho_array[0] * (cruise_speed ** 2) * planform_area * CD_active
objective.set_as_objective(scaler=1.0 / D_ref)

# L = W constraint (Node 0: cruise condition)
lift_trim = lift_effective_cruise - W_total
lift_trim.set_as_constraint(equals=0.0, scaler=1.0 / (5*W_ref))
# lift_trim = CL[0]# - CL_loss_cruise
# lift_trim.set_as_constraint(equals=0.5, scaler=2.)

# Pitch / Moment trim constraint: My = 0 about dynamic center of mass (x_cg)

pitch_moment = M[0, 1]
pitch_trim = pitch_moment
pitch_trim.set_as_constraint(equals=0.0, scaler=1.0 / (W_ref * c_ref))

# Sizing Lift constraint: L = load_factor * W (Node 2: structural sizing pull-up condition)
# Extract aerodynamic strip lift from panel vertical forces at Node 2 (sizing pull-up maneuver):
F_z_ss = panel_forces_right_ss[:, 2]
lift_aero_strips_ss = csdl.matvec(csdl.Variable(value=W_strip), F_z_ss)
lift_prime_aero_ss = lift_aero_strips_ss / dy_aero_dyn
lift_prime_drag_ss = csdl.matvec(csdl.Variable(value=M_interp_drag), lift_prime_aero_ss)

# Sectional lift coefficient during sizing maneuver on each drag strip: Cl_ss = L'_ss / (q_inf_ss * c)
q_inf_ss = 0.5 * rho_array[2] * (sizing_speed ** 2)
cl_local_elem_ss = lift_prime_drag_ss / (q_inf_ss * local_chord_drag)

# Aerodynamic stall lift loss penalty during sizing maneuver (unified parameters):
cl_abs_ss = csdl.absolute(cl_local_elem_ss)
delta_cl_ss = cl_abs_ss - cl_crit
softplus_cl_val_ss = (1.0 / beta_cl_stall) * csdl.softplus(beta_cl_stall * delta_cl_ss)
cl_stall_loss_ss = k_stall_lift * (softplus_cl_val_ss / delta_cl_ref)
sign_cl_ss = cl_local_elem_ss / (cl_abs_ss + 1.e-6)

# Preserve geometric alpha_local_ss for reference / inspection
alpha_local_ss = alpha_local_elem + dalpha_ss

# Total lift deficit across both wings during sizing maneuver:
L_loss_ss = csdl.sum(cl_stall_loss_ss * sign_cl_ss * q_inf_ss * strip_area)
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
# static_margin.set_as_constraint(lower=0.00, scaler=2.e1)
static_margin.set_as_constraint(lower=0.02, scaler=2.e1)
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

# =========================================================================
# 4B: Stall Progression & Maneuver Tip Margin Constraint via B-Spline Fit
# =========================================================================
# Fit 101-CP cubic B-spline to the 100 drag strip maneuver lift coefficients (Cl_ss)
# with root symmetry S'(0)=0, matching the stress fitting/evaluation/aggregation architecture.
cl_ss_rhs = csdl.concatenate([cl_local_elem_ss, csdl.Variable(value=np.array([0.0]))])
cl_ss_coeffs_flat = csdl.matvec(csdl.Variable(value=M_cl_ss_inv), cl_ss_rhs)
cl_ss_coeffs = csdl.reshape(cl_ss_coeffs_flat, (101, 1))
cl_ss_func = lfs.Function(space=cl_ss_fit_space, coefficients=cl_ss_coeffs)

# Evaluate at the dense spanwise evaluation points across the half-span
all_eval_cl_ss = cl_ss_func.evaluate(eval_cl_points_arr)

# Aggregate each station's span sector using smooth maximum
aggregated_cl_ss_list = []
for i in range(num_stations):
    grp = station_cl_group_indices[i]
    sub_cl_ss = csdl.concatenate([all_eval_cl_ss[idx] for idx in grp])
    grp_cl_max = csdl.maximum(sub_cl_ss, axes=(0,), rho=50.0)
    aggregated_cl_ss_list.append(csdl.reshape(grp_cl_max, (1,)))

# dv_cl_ss has shape (num_stations,) -> exactly 5 constraints in fast, 8 in full
dv_cl_ss = csdl.concatenate(aggregated_cl_ss_list)

# Stall progression ceiling: decreases monotonically from root to tip
# Root allowable = 1.40, Tip allowable = 1.15
cl_root_max = 1.4
cl_tip_max = 1.15
cl_ss_ceiling = np.linspace(cl_root_max, cl_tip_max, num_stations)

# for i in range(num_stations):
    # dv_cl_ss[i].set_as_constraint(upper=cl_ss_ceiling[i], scaler=1.0 / cl_ss_ceiling[i])
cl_constraints_to_enforce = dv_cl_ss[-3:] if resolution == 'fast' else dv_cl_ss[-4:]
cl_ceiling_to_enforce = cl_ss_ceiling[-3:] if resolution == 'fast' else cl_ss_ceiling[-4:]
# cl_constraints_to_enforce.set_as_constraint(upper=cl_ceiling_to_enforce, scaler=1.0 / cl_ceiling_to_enforce)

# =========================================================================
# 4C: Transonic Drag Divergence / Buffet Constraint via B-Spline Fit
# =========================================================================
# Fit 101-CP cubic B-spline to the 100 drag strip drag divergence Mach numbers (M_dd)
# in a fully-determined manner with root symmetry S'(0)=0, matching cl_ss architecture.
mdd_rhs = csdl.concatenate([strip_M_dd_cruise, csdl.Variable(value=np.array([0.0]))])
mdd_coeffs_flat = csdl.matvec(csdl.Variable(value=M_cl_ss_inv), mdd_rhs)
mdd_coeffs = csdl.reshape(mdd_coeffs_flat, (101, 1))
mdd_func = lfs.Function(space=cl_ss_fit_space, coefficients=mdd_coeffs)

# Evaluate at the dense spanwise evaluation points across the half-span
all_eval_mdd = mdd_func.evaluate(eval_cl_points_arr)
all_eval_mdd_excess = cruise_cond['mach'] - all_eval_mdd

# Aggregate each station's span sector using smooth maximum
aggregated_mdd_excess_list = []
for i in range(num_stations):
    grp = station_cl_group_indices[i]
    sub_mdd_excess = csdl.concatenate([all_eval_mdd_excess[idx] for idx in grp])
    grp_mdd_max = csdl.maximum(sub_mdd_excess, axes=(0,), rho=50.0)
    aggregated_mdd_excess_list.append(csdl.reshape(grp_mdd_max, (1,)))

# dv_mdd_excess has shape (num_stations,) -> exactly 5 locally aggregated values in fast, 8 in full
dv_mdd_excess = csdl.concatenate(aggregated_mdd_excess_list)

# Additional aggregation at the end to reduce to a single global constraint
global_mdd_excess = csdl.maximum(dv_mdd_excess, axes=(0,), rho=50.0)
min_strip_mdd_margin_cruise = -global_mdd_excess  # min(M_dd - M_cruise) across the wing

# Enforce M_inf <= M_dd (can be switched to dv_mdd_excess in the future)
# global_mdd_excess.set_as_constraint(upper=0.0, scaler=10.0)

if formulation == 'chord_span':
    # For chord and span stretch formulation (no ParameterizationSolver),
    # keep planform area constraint (10.0 m^2) unless geonic is active
    if not use_geonic:
        planform_area.set_as_constraint(equals=10.0 * scale_factor**2, scaler=1.0 / (10*scale_factor**2))
    aspect_ratio_calc.set_as_constraint(upper=15.0, scaler=1.e-1)
    # Enforce thickness-to-chord ratio bounds at all stations only if thickness or chord DVs are active
    if 'thickness_stretch_dvs' in design_variables and 'chord_stretch_dvs' in design_variables:
        for i in range(num_chord_stations):
            tc_ratio = local_thicknesses[i] / local_chords[i]
            tc_ratio.set_as_constraint(lower=0.06, upper=0.35, scaler=10.0)

    # Enforce evaluated Lambda_qc in [0, 60 deg] at every adjacent evaluation section:
    sweep_max_rad = 60.0 * np.pi / 180.0
    for i in range(num_sweep_eval_stations - 1):
        lambda_qc_vec[i].set_as_constraint(lower=0.0, upper=sweep_max_rad, scaler=1.0 / sweep_max_rad)
else:
    # For AR and Area formulation, ParameterizationSolver explicitly enforces
    # taper ratios, planform area, and aspect ratio, and sweep is bounded by sweep_angle_dvs.
    pass

for dv_info in design_variables.values():
    dv_info.variable.set_as_design_variable(lower=dv_info.lower, upper=dv_info.upper, scaler=dv_info.scaler)

geometry_coefficients = [geometry_function.coefficients for geometry_function in geometry.functions.values()]

additional_outs = [
    Di, L, CL, CDi, Cp, panel_mesh, planform_area, aspect_ratio_calc, structural_mass,
    beam_displacement, beam_rotation, tip_twist_ss, beam_stress, elem_max_stress,
    dv_stresses, stress_coeffs, ttop_elem, tweb_elem, ttop_dvs, tweb_dvs, twist_dvs,
    Di_Trefftz, CD_profile, D_profile, D_total, alpha_local_elem, cl_local_elem,
    cl_local_elem_ss, dv_cl_ss, cl_ss_coeffs, y_strip_pts, W_total, M, CM, pitch_trim, lift_ss, static_margin,
    neutral_point_x, x_cg, x_payload, x_struct, r_cg, local_chord, local_height,
    box_height, node_heights,
    box_width, beam_mesh, F_node, pitch_ss, dynamic_panel_centers_right,
    panel_forces_right_cruise, panel_lift_right_cruise, panel_forces_right_ss,
    lift_effective_cruise, lift_effective_ss, L_loss_cruise, L_loss_ss,
    CDi_Fourier, Di_Fourier, delta_fourier, e_fourier, A_fourier_raw, mu_w, trefftz_lift_ratio,
    # Atmospheric condition outputs:
    atm_altitudes, atm_densities, atm_sound_speeds, atm_mach_numbers,
    atm_temperatures, atm_pressures, atm_viscosities, atm_dynamic_pressures,
    atm_speeds, atm_reynolds_per_unit_chord,
    # Quarter-chord and half-chord sweep outputs:
    lambda_qc_vec, max_quarter_chord_sweep, quarter_chord_sweep_margin,
    strip_sweep_halfchord, max_half_chord_sweep, half_chord_sweep_excess,
    # Wave-drag outputs:
    strip_tc, CD_wave_array, D_wave_array, CD_wave_cruise, D_wave_cruise,
    strip_wave_cd_cruise, strip_M_dd_cruise, strip_M_crit_cruise,
    strip_delta_M_cruise, strip_delta_M_eff_cruise, min_strip_mach_margin_cruise,
    global_mdd_excess, dv_mdd_excess, min_strip_mdd_margin_cruise, mdd_coeffs,
    local_chord_drag, dy_strip, strip_area,
    # Viscous drag outputs:
    CD_viscous, D_viscous, CD_ibl_section, theta_te_upper,
    theta_te_lower, H_max_ibl, ibl_attachment_margin,
    ibl_min_cp, ibl_cp_cutoff_margin, dv_H_stations, H_strip, H_section_upper,
    H_section_lower, H_section_both,
]
if use_geonic:
    if geonic_payload_mode in {'oversized', 'both'}:
        additional_outs += [
            payload_center_x,
            payload_sample_points,
            payload_signed_distance,
            geonic_constraint_values,
            geonic_clearance_per_point,
            geonic_margin,
            payload_center_inertial,
            payload_center_body,
        ]
    if geonic_payload_mode in {'pallets', 'both'}:
        additional_outs += [
            pallet_stack_center_x,
            pallet_stack_x_shift,
            pallet_sample_points,
            pallet_signed_distance,
            pallet_constraint_values,
            pallet_clearance_per_point,
            pallet_margin,
            pallet_stack_center_inertial,
            pallet_stack_center_body,
        ]
    if geonic_payload_mode == 'both':
        additional_outs += [
            payload_cg_consistency,
            overall_geonic_margin,
        ]
else:
    additional_outs += [payload_cg]
if include_camber:
    additional_outs += [camber_dvs]
if include_thickness_shape:
    additional_outs += [thickness_shape_dvs]
if include_elevator:
    additional_outs += [elevator_angle]
if include_neg1g_sizing:
    additional_outs += [lift_neg1g, root_stress_neg1g, stress_coeffs_neg1g, elem_max_stress_neg1g, pitch_neg1g, panel_forces_right_neg1g]
if formulation == 'ar_area':
    additional_outs += list(ar_area_diagnostics.values())
    additional_outs += [res_taper, res_thick, res_sweep, tc_target_control_points]
    if use_geonic:
        additional_outs += [planform_area_target]
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
    # print(f"Total Drag (Objective): {float(np.asarray(jax_sim[D_total]).flatten()[0]):.2f} N (Induced: {float(np.asarray(jax_sim[Di_Trefftz]).flatten()[0]):.2f} N, Profile: {float(np.asarray(jax_sim[D_profile]).flatten()[0]):.2f} N, Wave: {float(np.asarray(jax_sim[D_wave_cruise]).flatten()[0]):.2f} N)")
    # print(f"Wave Drag CD_wave: {float(np.asarray(jax_sim[CD_wave_cruise]).flatten()[0])*1e4:.2f} counts | min(Mcrit - M): {float(np.asarray(jax_sim[min_strip_mach_margin_cruise]).flatten()[0]):+.4f}")
    # print(f"Quarter-Chord Sweep: Max = {np.degrees(float(np.asarray(jax_sim[max_quarter_chord_sweep]).flatten()[0])):.2f}° (Margin to 60°: {np.degrees(float(np.asarray(jax_sim[quarter_chord_sweep_margin]).flatten()[0])):.2f}°)")
    # print(f"Half-Chord Sweep: Max = {np.degrees(float(np.asarray(jax_sim[max_half_chord_sweep]).flatten()[0])):.2f}° (Excess above 60°: {np.degrees(float(np.asarray(jax_sim[half_chord_sweep_excess]).flatten()[0])):.2f}°)")
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
    root_deriv_val = (stress_coeffs_arr[1] - stress_coeffs_arr[0]) * 3.0 / knots_stress_15[4]
    print(f"\n================ 15-CP CUBIC STRESS SPLINE DIAGNOSTIC ({resolution.upper()}) ================")
    print(f"Root symmetry check: c0 = {stress_coeffs_arr[0]/1e6:.4f} MPa, c1 = {stress_coeffs_arr[1]/1e6:.4f} MPa, dS/du(0) = {root_deriv_val/1e6:.6f} MPa/unit")
    print(f"Stress control points (MPa): {np.round(stress_coeffs_arr/1e6, 2)}")

    dv_cl_ss_arr = np.asarray(jax_sim[dv_cl_ss]).flatten()
    print(f"\n================ {num_stations} STATION 4B MANEUVER LIFT COEFFICIENT CONSTRAINTS ({load_factor_val:.1f}g Pull-Up) ================")
    print(f"{'Station':7s} | {'eta_peak':8s} | {'Cl_ss (aggregated)':20s} | {'Allowable Ceiling':18s} | {'Status':8s}")
    print("-" * 80)
    for j in range(num_stations):
        cl_val_st = dv_cl_ss_arr[j]
        ceil_val = cl_ss_ceiling[j]
        st_status = "FEASIBLE" if cl_val_st <= ceil_val else "VIOLATED"
        print(f"{j:7d} | {station_cl_peaks[j]:8.4f} | {cl_val_st:20.4f} | {ceil_val:18.4f} | {st_status:8s}")

    dv_mdd_excess_arr = np.asarray(jax_sim[dv_mdd_excess]).flatten()
    global_mdd_excess_val = float(np.asarray(jax_sim[global_mdd_excess]))
    print(f"\n================ {num_stations} STATION TRANSONIC DRAG DIVERGENCE / BUFFET MARGIN (Cruise M_inf = {cruise_cond['mach']:.3f}) ================")
    print(f"{'Station':7s} | {'eta_peak':8s} | {'M_inf - M_dd':14s} | {'M_dd - M_inf Margin':20s} | {'Status':8s}")
    print("-" * 75)
    for j in range(num_stations):
        excess_st = dv_mdd_excess_arr[j]
        mdd_margin = -excess_st
        st_status = "FEASIBLE" if excess_st <= 0.0 else "VIOLATED"
        print(f"{j:7d} | {station_cl_peaks[j]:8.4f} | {excess_st:14.4f} | {mdd_margin:20.4f} | {st_status:8s}")
    print(f"Global M_dd margin min(M_dd - M_inf): {-global_mdd_excess_val:.4f} (Status: {'FEASIBLE' if global_mdd_excess_val <= 0.0 else 'VIOLATED'})\n")

    if use_geonic:
        print(f"================ GEONIC PAYLOAD NON-INTERFERENCE DIAGNOSTIC ================")
        print(f"Mode: {geonic_payload_mode.upper()} | Buffer Requirement: {GEONIC_CLEARANCE_M:.2f} m")

        if geonic_payload_mode in {'oversized', 'both'}:
            pay_cx_val = float(np.asarray(jax_sim[payload_center_x]).flatten()[0])
            clearance_vals = np.asarray(jax_sim[geonic_clearance_per_point]).flatten()
            min_clearance = float(np.min(clearance_vals))
            g_margin = float(np.asarray(jax_sim[geonic_margin]).flatten()[0])
            g_status = "FEASIBLE" if g_margin >= 0.0 else "VIOLATED"
            pay_inertial_val = np.asarray(jax_sim[payload_center_inertial]).flatten()
            print(f"--- OVERSIZED PAYLOAD ---")
            print(f"Payload Center x: {pay_cx_val:.3f} m (Inertial: [{pay_inertial_val[0]:.3f}, {pay_inertial_val[1]:.3f}, {pay_inertial_val[2]:.3f}] m)")
            print(f"Min Clearance: {min_clearance:.4f} m | Geonic Margin: {g_margin:+.4f} m | Status: {g_status}")
            print(f"{'Sample Point':14s} | {'SDF [m]':10s} | {'Clearance [m]':14s} | {'Buffer [m]':12s} | {'Status':8s}")
            print("-" * 65)
            for pt_i in range(len(clearance_vals)):
                c_val = clearance_vals[pt_i]
                pt_status = "FEASIBLE" if c_val >= GEONIC_CLEARANCE_M else "VIOLATED"
                print(f"Oversized Pt {pt_i:2d} | {-c_val:10.4f} | {c_val:14.4f} | {GEONIC_CLEARANCE_M:12.2f} | {pt_status:8s}")

        if geonic_payload_mode in {'pallets', 'both'}:
            pal_cx_val = float(np.asarray(jax_sim[pallet_stack_center_x]).flatten()[0])
            pal_shift_val = float(np.asarray(jax_sim[pallet_stack_x_shift]).flatten()[0])
            pal_clearance_vals = np.asarray(jax_sim[pallet_clearance_per_point]).flatten()
            min_pal_clearance = float(np.min(pal_clearance_vals))
            pal_margin_val = float(np.asarray(jax_sim[pallet_margin]).flatten()[0])
            pal_status = "FEASIBLE" if pal_margin_val >= 0.0 else "VIOLATED"
            pal_inertial_val = np.asarray(jax_sim[pallet_stack_center_inertial]).flatten()
            print(f"--- PALLET STACK (6 PALLETS) ---")
            print(f"Pallet Stack CG x: {pal_cx_val:.3f} m (Shift: {pal_shift_val:+.3f} m, Inertial: [{pal_inertial_val[0]:.3f}, {pal_inertial_val[1]:.3f}, {pal_inertial_val[2]:.3f}] m)")
            print(f"Min Clearance: {min_pal_clearance:.4f} m | Pallet Margin: {pal_margin_val:+.4f} m | Status: {pal_status}")
            print(f"{'Sample Point':14s} | {'SDF [m]':10s} | {'Clearance [m]':14s} | {'Buffer [m]':12s} | {'Status':8s}")
            print("-" * 65)
            for pt_i in range(len(pal_clearance_vals)):
                c_val = pal_clearance_vals[pt_i]
                pt_status = "FEASIBLE" if c_val >= GEONIC_CLEARANCE_M else "VIOLATED"
                print(f"Pallet Pt {pt_i:2d}    | {-c_val:10.4f} | {c_val:14.4f} | {GEONIC_CLEARANCE_M:12.2f} | {pt_status:8s}")

        if geonic_payload_mode == 'both':
            cg_diff_val = float(np.asarray(jax_sim[payload_cg_consistency]).flatten()[0])
            cg_status = "FEASIBLE (COINCIDENT)" if abs(cg_diff_val) <= 1e-4 else "VIOLATED"
            ov_margin_val = float(np.asarray(jax_sim[overall_geonic_margin]).flatten()[0])
            ov_status = "FEASIBLE" if ov_margin_val >= 0.0 else "VIOLATED"
            print(f"--- BOTH-MODE CONSISTENCY ---")
            print(f"CG Consistency (pay_x - pal_x): {cg_diff_val:+.6f} m | Status: {cg_status}")
            print(f"Overall Diagnostic Margin: {ov_margin_val:+.4f} m | Overall Status: {ov_status}")
        print()
    else:
        print(f"Geonic Mode: DISABLED (Legacy payload_cg active)\n")

    # Viscous Drag & Attachment Diagnostic
    cd_visc_val = float(np.asarray(jax_sim[CD_viscous]).flatten()[0])
    d_visc_val = float(np.asarray(jax_sim[D_viscous]).flatten()[0])
    hmax_val = float(np.asarray(jax_sim[H_max_ibl]).flatten()[0])
    att_margin_val = float(np.asarray(jax_sim[ibl_attachment_margin]).flatten()[0])
    min_cp_val = float(np.asarray(jax_sim[ibl_min_cp]).flatten()[0])
    cp_cut_margin = float(np.asarray(jax_sim[ibl_cp_cutoff_margin]).flatten()[0])
    att_status = "ATTACHED (FEASIBLE)" if att_margin_val >= 0.0 else "SEPARATED (VIOLATED)"
    cutoff_status = "VALID" if cp_cut_margin >= 0.1 else "CUTOFF WARNING"
    print(f"================ VISCOUS DRAG & IBL ATTACHMENT DIAGNOSTIC ================")
    print(f"Mode: {viscous_drag_mode.upper()} | CD_viscous: {cd_visc_val*1e4:.2f} counts ({cd_visc_val:.6f}) | D_viscous: {d_visc_val:.1f} N")
    print(f"H_max_ibl: {hmax_val:.4f} (Limit: 2.40) | Attachment Margin: {att_margin_val:+.4f} ({att_status})")
    print(f"Min Cp (Node 0): {min_cp_val:.4f} (Cutoff: -5.0) | Cutoff Margin: {cp_cut_margin:+.4f} ({cutoff_status})\n")


if __name__ == '__main__':
    if os.environ.get('SKIP_OPTIMIZATION', '0') != '1':
        optimization_problem = modopt.CSDLAlphaProblem(
            problem_name='rectangular_wing_to_bwb_aerostructural_optimization',
            simulator=jax_sim,
        )
        max_iter = int(os.environ.get('MAX_ITER', '500'))
        optimizer = modopt.PySLSQP(
            optimization_problem,
            solver_options={'maxiter': max_iter, 'acc': 1.e-5},
            readable_outputs=['x'],
        )
        optimizer.solve()
        optimizer.print_results()
    else:
        print("SKIP_OPTIMIZATION=1 detected: Skipping solve and running post-processing on latest folder.")
    # endregion Optimization
    
    
    # region Plot Optimization History
    import pyvista as pv
    import os, glob
    
    # Find the latest output folder (or TARGET_OUTPUT_FOLDER if provided)
    output_base_dir = 'rectangular_wing_to_bwb_aerostructural_optimization_outputs'
    output_folders = glob.glob(os.path.join(output_base_dir, '*'))
    target_folder = os.environ.get('TARGET_OUTPUT_FOLDER', None)
    if target_folder and os.path.exists(target_folder):
        latest_folder = target_folder
    else:
        valid_folders = [f for f in output_folders if os.path.isfile(os.path.join(f, 'x.out')) and os.path.getsize(os.path.join(f, 'x.out')) > 0]
        latest_folder = max(valid_folders, key=os.path.getmtime) if valid_folders else max(output_folders, key=os.path.getmtime)
    print(f"Reading optimization history from: {latest_folder}")
    
    # Read design variable history from x.out (preferred) or record.hdf5
    x_out_path = os.path.join(latest_folder, 'x.out')
    expected_dvs = sum(int(np.prod(dv_info.variable.shape)) for dv_info in design_variables.values())
    if os.path.exists(x_out_path):
        x_history = np.loadtxt(x_out_path)
        if len(x_history.shape) == 1:
            x_history = x_history.reshape(1, -1)
        if x_history.shape[1] != expected_dvs:
            print(f"Warning: {x_out_path} has {x_history.shape[1]} DVs, expected {expected_dvs}. Searching for matching run...")
            matching_folders = []
            for f in valid_folders:
                try:
                    f_x = np.loadtxt(os.path.join(f, 'x.out'))
                    if len(f_x.shape) == 1:
                        f_x = f_x.reshape(1, -1)
                    if f_x.shape[1] == expected_dvs:
                        matching_folders.append(f)
                except Exception:
                    pass
            if matching_folders:
                latest_folder = max(matching_folders, key=os.path.getmtime)
                x_out_path = os.path.join(latest_folder, 'x.out')
                x_history = np.loadtxt(x_out_path)
                if len(x_history.shape) == 1:
                    x_history = x_history.reshape(1, -1)
                print(f"Loaded matching history from: {latest_folder} ({x_history.shape[0]} iters)")
            else:
                print(f"No previous run found matching {expected_dvs} DVs. Post-processing animation replay skipped.")
                x_history = np.array([])
        else:
            print(f"Loaded {x_history.shape[0]} iterations from current run x.out")

        # If warm-started from prior file, prepend all previous iterations so video & summary show the full trajectory
        if warm_start and os.path.exists(init_file) and len(x_history) > 0:
            x_prior = np.loadtxt(init_file)
            if len(x_prior.shape) > 1 and x_prior.shape[0] > 0 and x_prior.shape[1] == expected_dvs:
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
                            elif 'taper_control_points' in inp_grp:
                                taper = inp_grp['taper_control_points'][:]
                                ar_val = inp_grp['aspect_ratio'][:] if 'aspect_ratio' in inp_grp else np.array([10.0])
                                x_vec = np.concatenate([taper, ar_val, pitch_val])
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
    
    # Set up pyvista offscreen rendering and video frames directory
    pv.OFF_SCREEN = True
    video_path = os.path.join(latest_folder, 'optimization_history.mp4')
    frames_dir = os.path.join(latest_folder, 'temp_video_frames')
    os.makedirs(frames_dir, exist_ok=True)
    plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
    plotter.set_background('black')
    
    # Top-down planform camera (~77.7 deg elevation): span horizontal, nose pointing up,
    # minimal planform foreshortening (2.3%) while allowing 3D thickness/camber to catch light.
    camera = {
        'position': (1.2 * scale_factor - 3.5 * scale_factor, 0.0, 16.0 * scale_factor),
        'focal_point': (1.2 * scale_factor, 0.0, 0.0),
        'viewup': (-1, 0, 0),
    }

    # Setup 3-point scene lighting for smooth Phong shading and realistic surface highlights
    plotter.renderer.RemoveAllLights()
    # Key light: upper-front-left (illuminates leading edge and upper surface curvature)
    plotter.add_light(pv.Light(position=(camera['position'][0] - 10*scale_factor, -15*scale_factor, camera['position'][2] + 5*scale_factor),
                               focal_point=camera['focal_point'], color='white', intensity=0.85, light_type='scene light'))
    # Fill light: upper-right (softens shadows across starboard wing)
    plotter.add_light(pv.Light(position=(camera['position'][0] + 5*scale_factor, 15*scale_factor, camera['position'][2] + 3*scale_factor),
                               focal_point=camera['focal_point'], color='#d0e4f7', intensity=0.45, light_type='scene light'))
    # Headlight / camera light: fills dead zones and ensures uniform depth definition
    plotter.add_light(pv.Light(position=camera['position'], focal_point=camera['focal_point'], color='white', intensity=0.4, light_type='camera light'))
    
    cd_history = []
    cl_history = []
    sref_history = []
    geonic_margin_history = []
    from optimization_analyses.wake_load_consistency import (
        WakeCirculationClosureHistory,
        WakeLoadConsistencyHistory,
    )
    wake_load_diagnostic = WakeLoadConsistencyHistory()
    wake_closure_diagnostic = WakeCirculationClosureHistory()
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

        wake_load_diagnostic.record(
            iteration=iteration,
            panel_centers=np.asarray(jax_sim[dynamic_panel_centers_right]),
            panel_forces=np.asarray(jax_sim[panel_forces_right_cruise]),
            panel_mesh=np.asarray(jax_sim[panel_mesh]),
            mu_w=np.asarray(jax_sim[mu_w]),
            te_edges=np.asarray(TE_properties[2]),
            rho_inf=float(np.asarray(rho_array.value).reshape(-1)[0]),
            velocity_inf=float(np.asarray(cruise_speed.value).reshape(-1)[0]),
            sound_speed=float(np.asarray(sos_array.value).reshape(-1)[0]),
            cl=float(np.asarray(jax_sim[CL]).reshape(-1)[0]),
            cdi_fourier=float(np.asarray(jax_sim[CDi_Fourier]).reshape(-1)[0]),
            cdi_trefftz=float(np.asarray(jax_sim[CDi]).reshape(-1)[0]),
            trefftz_lift_ratio=float(np.asarray(jax_sim[trefftz_lift_ratio]).reshape(-1)[0]),
        )
        wake_closure_diagnostic.record(
            iteration=iteration,
            panel_centers=np.asarray(jax_sim[dynamic_panel_centers_right]),
            panel_forces=np.asarray(jax_sim[panel_forces_right_cruise]),
            panel_lift=np.asarray(jax_sim[panel_lift_right_cruise]),
            panel_mesh=np.asarray(jax_sim[panel_mesh]),
            mu_w=np.asarray(jax_sim[mu_w]),
            te_edges=np.asarray(TE_properties[2]),
            rho_inf=float(np.asarray(rho_array.value).reshape(-1)[0]),
            velocity_inf=float(np.asarray(cruise_speed.value).reshape(-1)[0]),
            sound_speed=float(np.asarray(sos_array.value).reshape(-1)[0]),
            cl=float(np.asarray(jax_sim[CL]).reshape(-1)[0]),
            cdi_fourier=float(np.asarray(jax_sim[CDi_Fourier]).reshape(-1)[0]),
            cdi_trefftz=float(np.asarray(jax_sim[CDi]).reshape(-1)[0]),
            trefftz_lift_ratio=float(np.asarray(jax_sim[trefftz_lift_ratio]).reshape(-1)[0]),
        )
    
        # Record history metrics
        cl_val = float(np.asarray(jax_sim[CL]).flatten()[0])
        cd_trefftz_val = float(np.asarray(jax_sim[CDi]).flatten()[0])
        cdi_fourier_val = float(np.asarray(jax_sim[CDi_Fourier]).flatten()[0])
        e_fourier_val = float(np.asarray(jax_sim[e_fourier]).flatten()[0])
        sref_val = float(np.asarray(jax_sim[planform_area]).flatten()[0])
        cd_wave_val = float(np.asarray(jax_sim[CD_wave_cruise]).flatten()[0])
        d_wave_val = float(np.asarray(jax_sim[D_wave_cruise]).flatten()[0])
        cd_profile_val = float(np.asarray(jax_sim[CD_profile]).flatten()[0])
        d_profile_val = float(np.asarray(jax_sim[D_profile]).flatten()[0])
        d_total_val = float(np.asarray(jax_sim[D_total]).flatten()[0])
        max_qc_sw_deg = np.degrees(float(np.asarray(jax_sim[max_quarter_chord_sweep]).flatten()[0]))
        max_hc_sw_deg = np.degrees(float(np.asarray(jax_sim[max_half_chord_sweep]).flatten()[0]))
        min_mach_margin = float(np.asarray(jax_sim[min_strip_mach_margin_cruise]).flatten()[0])
        min_mdd_margin = float(np.asarray(jax_sim[min_strip_mdd_margin_cruise]).flatten()[0])

        cd_active_val = (cdi_fourier_val if induced_drag_objective == 'fourier' else cd_trefftz_val) + cd_profile_val + cd_wave_val
        cd_history.append(cd_active_val * 1e4)  # Active total CD in drag counts (x 1e4)
        cl_history.append(cl_val)
        sref_history.append(sref_val)
        if use_geonic:
            if geonic_payload_mode == 'both':
                margin_record = float(np.asarray(jax_sim[overall_geonic_margin]).flatten()[0])
            elif geonic_payload_mode == 'pallets':
                margin_record = float(np.asarray(jax_sim[pallet_margin]).flatten()[0])
            else:
                margin_record = float(np.asarray(jax_sim[geonic_margin]).flatten()[0])
            geonic_margin_history.append(margin_record)
    
        # Get plotting elements from geometry.plot (returns list of pyvista objects)
        plotting_elements = geometry.plot(show=False)
    
        # Clear previous frame actors while preserving lighting setup
        plotter.clear_actors()
    
        # Add each plotting element to the plotter with smooth Phong shading and surface normals
        for element in plotting_elements:
            if isinstance(element, dict) and 'mesh' in element:
                mesh = element['mesh']
                kwargs = element.get('kwargs', {})
            elif isinstance(element, tuple) and len(element) == 2:
                mesh, kwargs = element
            elif isinstance(element, pv.Actor):
                plotter.add_actor(element)
                continue
            elif isinstance(element, pv.DataSet):
                mesh = element
                kwargs = {}
            else:
                continue

            # Convert StructuredGrid to PolyData with exact point normals for smooth Phong shading
            if hasattr(mesh, 'extract_surface'):
                try:
                    surf = mesh.extract_surface(algorithm='dataset_surface')
                except TypeError:
                    surf = mesh.extract_surface()
                if hasattr(surf, 'compute_normals'):
                    surf = surf.compute_normals(auto_orient_normals=True)
            else:
                surf = mesh

            mesh_kwargs = {
                'smooth_shading': True,
                'specular': 0.5,
                'specular_power': 30,
                'ambient': 0.25,
                'diffuse': 0.75,
            }
            mesh_kwargs.update(kwargs)
            # Ensure shading parameters are active
            mesh_kwargs['smooth_shading'] = True
            mesh_kwargs['specular'] = 0.5
            mesh_kwargs['specular_power'] = 30
            mesh_kwargs['ambient'] = 0.25
            mesh_kwargs['diffuse'] = 0.75
            plotter.add_mesh(surf, **mesh_kwargs)
    
        # Build parameter info string depending on active formulation
        dv_str = ""
        if formulation == 'ar_area':
            ar_val = float(np.asarray(unscaled_values['aspect_ratio']).flatten()[0]) if 'aspect_ratio' in unscaled_values else 10.0
            if 'taper_control_points' in unscaled_values:
                t_vals = unscaled_values['taper_control_points']
            elif 'taper_dvs' in unscaled_values:
                t_vals = unscaled_values['taper_dvs']
            else:
                t_vals = np.ones(num_chord_stations - 1)
            p1 = float(np.asarray(t_vals).flatten()[0]) if len(t_vals) > 0 else 1.0
            p0 = 1.5 - 0.5 * p1
            c_vals = [p0] + list(t_vals)
            c_str = " ".join([f"cp_t{i}={c_vals[i]:.2f}" for i in range(len(c_vals))])
            if 'sweep_angle_control_points' in unscaled_values:
                sw_vals = unscaled_values['sweep_angle_control_points']
            elif 'sweep_angle_dvs' in unscaled_values:
                sw_vals = unscaled_values['sweep_angle_dvs']
            else:
                sw_vals = np.zeros(num_chord_stations - 1)
            sw_str = " ".join([f"cp_sw{i}={np.degrees(float(np.asarray(sw_vals[i]).flatten()[0])):.1f}°" for i in range(len(sw_vals))])
            if 'twist_dvs' in unscaled_values:
                tw_vals = unscaled_values['twist_dvs']
                tw_str = " ".join([f"tw{i}={np.degrees(float(np.asarray(tw_vals[i]).flatten()[0])):.1f}°" for i in range(len(tw_vals))])
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
            dv_str = f"b_stretch={ss_val:.2f}  {c_str}\n{sw_str}" + (f"\n{tw_str}" if tw_str else "")
        if 'camber_dvs' in unscaled_values:
            cam_vals = unscaled_values['camber_dvs']
            max_cam_pct = np.max(np.abs(cam_vals))
            dv_str += (f"\nmax camber={max_cam_pct:.2f}% chord" if dv_str else f"max camber={max_cam_pct:.2f}% chord")
        if 'thickness_shape_dvs' in unscaled_values:
            th_vals = unscaled_values['thickness_shape_dvs']
            max_th_pct = np.max(np.abs(th_vals))
            dv_str += (f"\nmax thick shape={max_th_pct:.2f}% thickness" if dv_str else f"max thick shape={max_th_pct:.2f}% thickness")
        if 'tc_target_control_points' in unscaled_values:
            tc_vals = np.asarray(unscaled_values['tc_target_control_points']).flatten()
            dv_str += (f"\nt/c=[{tc_vals[0]:.3f}..{tc_vals[-1]:.3f}]" if dv_str else f"t/c=[{tc_vals[0]:.3f}..{tc_vals[-1]:.3f}]")
        if 'planform_area_target' in unscaled_values:
            ar_tgt_val = float(np.asarray(unscaled_values['planform_area_target']).flatten()[0])
            dv_str += (f"\nS_target={ar_tgt_val:.1f}m²" if dv_str else f"S_target={ar_tgt_val:.1f}m²")
    
        pitch_val = float(np.asarray(unscaled_values['pitch']).flatten()[0]) if 'pitch' in unscaled_values else 0.0
        pitch_ss_val = float(np.asarray(unscaled_values['pitch_ss']).flatten()[0]) if 'pitch_ss' in unscaled_values else 0.0
        pitch_neg1g_val = float(np.asarray(unscaled_values['pitch_neg1g']).flatten()[0]) if 'pitch_neg1g' in unscaled_values else 0.0
        pitch_neg1g_str = f"  Pitch_-1g={np.degrees(pitch_neg1g_val):.1f}°" if 'pitch_neg1g' in unscaled_values else ""
        elev_val = float(np.asarray(unscaled_values['elevator_angle']).flatten()[0]) if 'elevator_angle' in unscaled_values else 0.0
        elev_str = f"Elevator={np.degrees(elev_val):.1f}°  " if include_elevator else ""
        sm_val = float(np.asarray(jax_sim[static_margin]).flatten()[0])
        xnp_val = float(np.asarray(jax_sim[neutral_point_x]).flatten()[0])
        xcg_val = float(np.asarray(jax_sim[x_cg]).flatten()[0])
        xpay_val = float(np.asarray(jax_sim[x_payload]).flatten()[0])

        if use_geonic:
            cos_p = np.cos(pitch_val)
            sin_p = np.sin(pitch_val)
            box_faces = [
                4, 0, 1, 3, 2,  # front
                4, 4, 6, 7, 5,  # rear
                4, 0, 4, 5, 1,  # left
                4, 2, 3, 7, 6,  # right
                4, 0, 2, 6, 4,  # bottom
                4, 1, 5, 7, 3,  # top
            ]

            if geonic_payload_mode in {'oversized', 'both'}:
                pay_cx_val = float(np.asarray(unscaled_values['payload_center_x']).flatten()[0]) if 'payload_center_x' in unscaled_values else float(np.asarray(jax_sim[payload_center_x]).flatten()[0])
                clearance_vals = np.asarray(jax_sim[geonic_clearance_per_point]).flatten()
                min_clearance = float(np.min(clearance_vals))
                g_margin = float(np.asarray(jax_sim[geonic_margin]).flatten()[0])
                g_status = "FEAS" if g_margin >= 0.0 else "VIOL"

                box_corners_body = get_full_payload_box_corners(pay_cx_val, 0.0, 0.0)
                pay_inertial_val = np.asarray(jax_sim[payload_center_inertial]).flatten()
                box_corners_rot = np.empty_like(box_corners_body)
                dx = box_corners_body[:, 0] - pay_cx_val
                dz = box_corners_body[:, 2]
                box_corners_rot[:, 0] = pay_inertial_val[0] + dx * cos_p + dz * sin_p
                box_corners_rot[:, 1] = box_corners_body[:, 1]
                box_corners_rot[:, 2] = pay_inertial_val[2] - dx * sin_p + dz * cos_p
                payload_poly = pv.PolyData(box_corners_rot, box_faces)
                plotter.add_mesh(payload_poly, color='#9467bd', opacity=0.4, style='wireframe', line_width=2.0)
                plotter.add_mesh(payload_poly, color='#9467bd', opacity=0.15, style='surface')

            if geonic_payload_mode in {'pallets', 'both'}:
                pal_cx_val = float(np.asarray(unscaled_values['pallet_stack_center_x']).flatten()[0]) if 'pallet_stack_center_x' in unscaled_values else float(np.asarray(jax_sim[pallet_stack_center_x]).flatten()[0])
                pal_clearance_vals = np.asarray(jax_sim[pallet_clearance_per_point]).flatten()
                min_pal_clearance = float(np.min(pal_clearance_vals))
                pal_margin = float(np.asarray(jax_sim[pallet_margin]).flatten()[0])
                pal_status = "FEAS" if pal_margin >= 0.0 else "VIOL"

                pallet_boxes = get_full_pallet_boxes_corners(pal_cx_val)
                pal_inertial_val = np.asarray(jax_sim[pallet_stack_center_inertial]).flatten()
                for p_box in pallet_boxes:
                    p_box_rot = np.empty_like(p_box)
                    dx = p_box[:, 0] - pal_cx_val
                    dz = p_box[:, 2]
                    p_box_rot[:, 0] = pal_inertial_val[0] + dx * cos_p + dz * sin_p
                    p_box_rot[:, 1] = p_box[:, 1]
                    p_box_rot[:, 2] = pal_inertial_val[2] - dx * sin_p + dz * cos_p
                    pal_poly = pv.PolyData(p_box_rot, box_faces)
                    plotter.add_mesh(pal_poly, color='cyan', opacity=0.35, style='wireframe', line_width=1.5)
                    plotter.add_mesh(pal_poly, color='cyan', opacity=0.12, style='surface')

            if geonic_payload_mode == 'oversized':
                geonic_hud_str = f"Geonic: OVERSIZED | pay_x={pay_cx_val:.2f}m | min_clr={min_clearance:.3f}m | margin={g_margin:+.3f}m ({g_status})\n"
                pay_cg_str = f"x_pay={xpay_val:.3f}m (pay_x={pay_cx_val:.3f}m)"
            elif geonic_payload_mode == 'pallets':
                geonic_hud_str = f"Geonic: PALLETS | pal_x={pal_cx_val:.2f}m | min_clr={min_pal_clearance:.3f}m | margin={pal_margin:+.3f}m ({pal_status})\n"
                pay_cg_str = f"x_pay={xpay_val:.3f}m (pal_x={pal_cx_val:.3f}m)"
            elif geonic_payload_mode == 'both':
                cg_diff_val = float(np.asarray(jax_sim[payload_cg_consistency]).flatten()[0])
                ov_margin_val = float(np.asarray(jax_sim[overall_geonic_margin]).flatten()[0])
                ov_st = "FEAS" if ov_margin_val >= 0.0 else "VIOL"
                geonic_hud_str = f"Geonic: BOTH | pay_x={pay_cx_val:.2f}m, pal_x={pal_cx_val:.2f}m (Δ={cg_diff_val:+.3f}m) | margin={ov_margin_val:+.3f}m ({ov_st})\n"
                pay_cg_str = f"x_pay={xpay_val:.3f}m (pay={pay_cx_val:.2f}m, pal={pal_cx_val:.2f}m)"
        else:
            pay_cg_val = float(np.asarray(unscaled_values['payload_cg']).flatten()[0]) if 'payload_cg' in unscaled_values else 0.40
            geonic_hud_str = ""
            pay_cg_str = f"x_pay={xpay_val:.3f}m [{pay_cg_val*100:.1f}%]"
    
        if viscous_drag_mode == 'ibl':
            cd_visc_val_iter = float(np.asarray(jax_sim[CD_viscous]).flatten()[0])
            att_m_iter = float(np.asarray(jax_sim[ibl_attachment_margin]).flatten()[0])
            att_st_iter = "FEAS" if att_m_iter >= 0.0 else "VIOL"
            viscous_hud_str = f"Viscous: IBL | CD_visc={cd_visc_val_iter*1e4:.1f} cts | Att_margin={att_m_iter:+.3f} ({att_st_iter})\n"
        else:
            viscous_hud_str = f"Viscous: CONST_CD0 (CD0=80.0 cts)\n"

        # Add iteration counter label using unscaled physical values
        plotter.add_text(
            f"Iteration {iteration}/{num_iterations - 1}\n"
            f"Formulation: {formulation} | Res: {resolution} | Obj: {induced_drag_objective}\n"
            f"CD_tot={cd_active_val*1e4:.1f} cts (CDi={(cdi_fourier_val if induced_drag_objective=='fourier' else cd_trefftz_val)*1e4:.1f}, CDprof={cd_profile_val*1e4:.1f}, CDwave={cd_wave_val*1e4:.1f})\n"
            f"D_tot={d_total_val:.1f} N | Λ_qc_max={max_qc_sw_deg:.1f}° | Λ_0.5_max={max_hc_sw_deg:.1f}° | min(Mdd-M)={min_mdd_margin:+.4f}\n"
            f"{viscous_hud_str}"
            f"{geonic_hud_str}"
            f"{dv_str}\n"
            f"{elev_str}Pitch={np.degrees(pitch_val):.1f}°  Pitch_ss={np.degrees(pitch_ss_val):.1f}°{pitch_neg1g_str}\n"
            f"SM={sm_val:.4f} (x_cg={xcg_val:.3f}m, x_np={xnp_val:.3f}m, {pay_cg_str})",
            position='upper_left',
            font_size=11,
            color='white',
            shadow=False,
        )
    
        # Set camera
        plotter.camera.position = camera['position']
        plotter.camera.focal_point = camera['focal_point']
        plotter.camera.up = camera['viewup']
    
        frame_file = os.path.join(frames_dir, f"frame_{iteration:04d}.png")
        plotter.screenshot(frame_file)
        print(f"  Frame {iteration}/{num_iterations - 1} rendered")
    
        # Save final wing render for the summary plot
        if iteration == num_iterations - 1:
            try:
                import shutil
                shutil.copyfile(frame_file, wing_img_path)
            except Exception as e:
                print(f"Warning: could not save wing image: {e}")
    
            # Cache panel telemetry for extract_lift_and_moment_distributions.py and extract_lift_distribution.py
            cache_file_opt = os.path.join(latest_folder, 'lift_and_moment_data.npz')
            try:
                xcg_val_opt = float(np.asarray(jax_sim[x_cg]).flatten()[0])
                zcg_val_opt = float(np.asarray(jax_sim[r_cg]).flatten()[2]) if 'r_cg' in globals() else 0.0
                panel_centers_right_opt = np.asarray(jax_sim[dynamic_panel_centers_right])
                f_cruise_opt = np.asarray(jax_sim[panel_forces_right_cruise])
                f_ss_opt = np.asarray(jax_sim[panel_forces_right_ss])
                cl_val_opt = float(np.asarray(jax_sim[CL]).flatten()[0])
                cdi_val_opt = float(np.asarray(jax_sim[CDi]).flatten()[0])
                lift_ratio_val_opt = float(np.asarray(jax_sim[trefftz_lift_ratio]).flatten()[0])
                cdi_fourier_val_opt = float(np.asarray(jax_sim[CDi_Fourier]).flatten()[0])
                delta_fourier_val_opt = float(np.asarray(jax_sim[delta_fourier]).flatten()[0])
                e_fourier_val_opt = float(np.asarray(jax_sim[e_fourier]).flatten()[0])
                A_fourier_raw_opt = np.asarray(jax_sim[A_fourier_raw]).flatten()
                sref_val_opt = float(np.asarray(jax_sim[planform_area]).flatten()[0])
                ar_val_opt = float(np.asarray(jax_sim[aspect_ratio_calc]).flatten()[0])
                cd_wave_val_opt = float(np.asarray(jax_sim[CD_wave_cruise]).flatten()[0])
                d_wave_val_opt = float(np.asarray(jax_sim[D_wave_cruise]).flatten()[0])
                cd_profile_val_opt = float(np.asarray(jax_sim[CD_profile]).flatten()[0])
                d_profile_val_opt = float(np.asarray(jax_sim[D_profile]).flatten()[0])
                d_total_val_opt = float(np.asarray(jax_sim[D_total]).flatten()[0])
                strip_tc_opt = np.asarray(jax_sim[strip_tc]).flatten()
                strip_sweep_opt = np.asarray(jax_sim[strip_sweep_halfchord]).flatten()
                y_strip_pts_opt = np.asarray(jax_sim[y_strip_pts]).flatten()
                strip_wave_cd_opt = np.asarray(jax_sim[strip_wave_cd_cruise]).flatten()
                strip_M_crit_opt = np.asarray(jax_sim[strip_M_crit_cruise]).flatten()
                strip_M_dd_opt = np.asarray(jax_sim[strip_M_dd_cruise]).flatten()
                strip_delta_M_opt = np.asarray(jax_sim[strip_delta_M_cruise]).flatten()
                cl_local_elem_opt = np.asarray(jax_sim[cl_local_elem]).flatten()
                local_chord_drag_opt = np.asarray(jax_sim[local_chord_drag]).flatten()
                dy_strip_opt = np.asarray(jax_sim[dy_strip]).flatten()
                strip_area_opt = np.asarray(jax_sim[strip_area]).flatten()

                cache_dict = dict(
                    panel_centers_right=panel_centers_right_opt,
                    f_cruise=f_cruise_opt,
                    f_ss=f_ss_opt,
                    xcg_val=xcg_val_opt,
                    zcg_val=zcg_val_opt,
                    cl_val=cl_val_opt,
                    cdi_val=cdi_val_opt,
                    lift_ratio=lift_ratio_val_opt,
                    cdi_fourier_val=cdi_fourier_val_opt,
                    delta_fourier_val=delta_fourier_val_opt,
                    e_fourier_val=e_fourier_val_opt,
                    A_fourier_raw=A_fourier_raw_opt,
                    cd_wave_val=cd_wave_val_opt,
                    d_wave_val=d_wave_val_opt,
                    cd_profile_val=cd_profile_val_opt,
                    d_profile_val=d_profile_val_opt,
                    d_total_val=d_total_val_opt,
                    strip_tc=strip_tc_opt,
                    strip_sweep_qc=strip_sweep_opt,
                    strip_sweep_halfchord=strip_sweep_opt,
                    y_strip_pts=y_strip_pts_opt,
                    strip_wave_cd=strip_wave_cd_opt,
                    strip_M_crit=strip_M_crit_opt,
                    strip_M_dd=strip_M_dd_opt,
                    cl_local_elem=cl_local_elem_opt,
                    cl_local_elem_ss=np.asarray(jax_sim[cl_local_elem_ss]).flatten(),
                    cl_ss_coeffs=np.asarray(jax_sim[cl_ss_coeffs]).flatten(),
                    eval_cl_points=eval_cl_points_arr.flatten(),
                    station_cl_peaks=np.asarray(station_cl_peaks).flatten(),
                    dv_cl_ss=np.asarray(jax_sim[dv_cl_ss]).flatten(),
                    cl_ss_ceiling=cl_ss_ceiling,
                    cl_constraints_enforced=False,
                    local_chord_drag=local_chord_drag_opt,
                    dy_strip=dy_strip_opt,
                    strip_area=strip_area_opt,
                    induced_drag_objective=str(induced_drag_objective),
                    sref_val=sref_val_opt,
                    ar_val=ar_val_opt,
                    formulation=str(formulation),
                    resolution=str(resolution),
                    include_camber=bool(include_camber),
                    include_thickness_shape=bool(include_thickness_shape),
                    include_te_thickness=bool(include_te_thickness),
                    include_elevator=bool(include_elevator),
                    scale_factor=float(scale_factor),
                    use_geonic=bool(use_geonic),
                    geonic_payload_mode=str(geonic_payload_mode),
                    num_chord_stations=int(num_chord_stations),
                    num_stations=int(num_stations),
                    dv_names=np.array(list(design_variables.keys()), dtype=str),
                )
                for dv_name, dv_info in design_variables.items():
                    cache_dict[f'dv_opt_{dv_name}'] = np.asarray(jax_sim[dv_info.variable])

                cache_dict['geonic_payload_mode'] = str(geonic_payload_mode)
                if use_geonic:
                    if geonic_payload_mode in {'oversized', 'both'}:
                        cache_dict['payload_center_x'] = float(np.asarray(jax_sim[payload_center_x]).flatten()[0])
                        cache_dict['payload_sample_points'] = np.asarray(jax_sim[payload_sample_points])
                        cache_dict['payload_signed_distance'] = np.asarray(jax_sim[payload_signed_distance]).flatten()
                        cache_dict['geonic_margin'] = float(np.asarray(jax_sim[geonic_margin]).flatten()[0])
                        cache_dict['geonic_clearance_per_point'] = np.asarray(jax_sim[geonic_clearance_per_point]).flatten()
                        cache_dict['payload_center_inertial'] = np.asarray(jax_sim[payload_center_inertial]).flatten()
                    if geonic_payload_mode in {'pallets', 'both'}:
                        cache_dict['pallet_stack_center_x'] = float(np.asarray(jax_sim[pallet_stack_center_x]).flatten()[0])
                        cache_dict['pallet_stack_x_shift'] = float(np.asarray(jax_sim[pallet_stack_x_shift]).flatten()[0])
                        cache_dict['pallet_sample_points'] = np.asarray(jax_sim[pallet_sample_points])
                        cache_dict['pallet_signed_distance'] = np.asarray(jax_sim[pallet_signed_distance]).flatten()
                        cache_dict['pallet_margin'] = float(np.asarray(jax_sim[pallet_margin]).flatten()[0])
                        cache_dict['pallet_clearance_per_point'] = np.asarray(jax_sim[pallet_clearance_per_point]).flatten()
                        cache_dict['pallet_stack_center_inertial'] = np.asarray(jax_sim[pallet_stack_center_inertial]).flatten()
                    if geonic_payload_mode == 'both':
                        cache_dict['payload_cg_consistency'] = float(np.asarray(jax_sim[payload_cg_consistency]).flatten()[0])
                        cache_dict['overall_geonic_margin'] = float(np.asarray(jax_sim[overall_geonic_margin]).flatten()[0])
                else:
                    cache_dict['payload_cg'] = float(np.asarray(jax_sim[payload_cg]).flatten()[0])

                cache_dict['viscous_drag_mode'] = str(viscous_drag_mode)
                cache_dict['cd_viscous_val'] = float(np.asarray(jax_sim[CD_viscous]).flatten()[0])
                cache_dict['d_viscous_val'] = float(np.asarray(jax_sim[D_viscous]).flatten()[0])
                cache_dict['cd_ibl_section'] = np.asarray(jax_sim[CD_ibl_section]).flatten()
                cache_dict['theta_te_upper'] = np.asarray(jax_sim[theta_te_upper]).flatten()
                cache_dict['theta_te_lower'] = np.asarray(jax_sim[theta_te_lower]).flatten()
                cache_dict['h_max_ibl'] = float(np.asarray(jax_sim[H_max_ibl]).flatten()[0])
                cache_dict['ibl_attachment_margin'] = float(np.asarray(jax_sim[ibl_attachment_margin]).flatten()[0])
                cache_dict['ibl_min_cp'] = float(np.asarray(jax_sim[ibl_min_cp]).flatten()[0])
                cache_dict['ibl_cp_cutoff_margin'] = float(np.asarray(jax_sim[ibl_cp_cutoff_margin]).flatten()[0])
                cache_dict['dv_H_stations'] = np.asarray(jax_sim[dv_H_stations]).flatten()
                if viscous_drag_mode == 'ibl':
                    cache_dict['H_section_upper'] = np.asarray(jax_sim[H_section_upper]).flatten()

                np.savez_compressed(cache_file_opt, **cache_dict)
                print(f"Cached panel telemetry saved to: {cache_file_opt}")

                # Dedicated cache for extract_wave_drag_distribution.py
                wave_cache_opt = os.path.join(latest_folder, 'wave_drag_distribution_data.npz')
                np.savez_compressed(
                    wave_cache_opt,
                    y_strip=y_strip_pts_opt,
                    strip_wave_cd=strip_wave_cd_opt,
                    strip_M_crit=strip_M_crit_opt,
                    strip_M_dd=strip_M_dd_opt,
                    strip_delta_M=strip_delta_M_opt,
                    strip_tc=strip_tc_opt,
                    strip_sweep=strip_sweep_opt,
                    strip_sweep_qc=strip_sweep_opt,
                    strip_sweep_halfchord=strip_sweep_opt,
                    strip_cl=cl_local_elem_opt,
                    local_chord_drag=local_chord_drag_opt,
                    dy_strip=dy_strip_opt,
                    strip_area=strip_area_opt,
                    sref_val=sref_val_opt,
                    ar_val=ar_val_opt,
                    cd_wave_val=cd_wave_val_opt,
                    d_wave_val=d_wave_val_opt,
                    mach_cruise=0.70,
                    q_cruise=float(cruise_cond['dynamic_pressure_Pa']),
                )
                print(f"Cached wave drag telemetry saved to: {wave_cache_opt}")
            except Exception as e:
                print(f"Warning: could not save wave drag telemetry cache: {e}")
    
            # Cache structural telemetry for generate_structural_plots.py
            struct_cache_opt = os.path.join(latest_folder, 'structural_data.npz')
            try:
                import json
                dv_metadata = {}
                for dv_k, dv_v in design_variables.items():
                    dv_metadata[dv_k] = {
                        'shape': list(dv_v.variable.shape),
                        'scaler': dv_v.scaler.tolist() if isinstance(dv_v.scaler, np.ndarray) else float(dv_v.scaler),
                        'lower': dv_v.lower.tolist() if isinstance(dv_v.lower, np.ndarray) else float(dv_v.lower),
                        'upper': dv_v.upper.tolist() if isinstance(dv_v.upper, np.ndarray) else float(dv_v.upper),
                    }
                dv_metadata_json = json.dumps(dv_metadata)
                x_scaler_opt = np.asarray(optimization_problem.x_scaler).flatten() if 'optimization_problem' in locals() and hasattr(optimization_problem, 'x_scaler') else np.array([])
                x_lower_opt = np.asarray(optimization_problem.x_lower).flatten() if 'optimization_problem' in locals() and hasattr(optimization_problem, 'x_lower') else np.array([])
                x_upper_opt = np.asarray(optimization_problem.x_upper).flatten() if 'optimization_problem' in locals() and hasattr(optimization_problem, 'x_upper') else np.array([])

                np.savez_compressed(
                    struct_cache_opt,
                    beam_pts_opt=np.asarray(jax_sim[beam_mesh]),
                    scale_factor=float(scale_factor),
                    allowable_stress=float(allowable_stress),
                    chords_opt=np.asarray(jax_sim[local_chord]).flatten(),
                    heights_opt=np.asarray(jax_sim[local_height]).flatten(),
                    box_heights_opt=np.asarray(jax_sim[box_height]).flatten(),
                    box_height_factor=float(box_height_factor),
                    widths_opt=np.asarray(jax_sim[box_width]).flatten(),
                    ttop_elem_opt=np.asarray(jax_sim[ttop_elem]).flatten(),
                    tweb_elem_opt=np.asarray(jax_sim[tweb_elem]).flatten(),
                    elem_stress_ss=np.asarray(jax_sim[elem_max_stress]).flatten(),
                    dv_stress_ss=np.asarray(jax_sim[dv_stresses]).flatten(),
                    stress_coeffs_ss=np.asarray(jax_sim[stress_coeffs]).flatten(),
                    ttop_dvs_opt=np.asarray(jax_sim[ttop_dvs]).flatten(),
                    tweb_dvs_opt=np.asarray(jax_sim[tweb_dvs]).flatten(),
                    thickness_peaks=np.asarray(thickness_peaks),
                    knots_stress_15=np.asarray(knots_stress_15) if 'knots_stress_15' in globals() else np.array([]),
                    resolution=str(resolution),
                    formulation=str(formulation),
                    num_chord_stations=int(num_chord_stations),
                    num_stations=int(num_stations),
                    use_geonic=bool(use_geonic),
                    geonic_payload_mode=str(geonic_payload_mode),
                    dv_names=np.array(list(design_variables.keys()), dtype=str),
                    dv_metadata_json=str(dv_metadata_json),
                    x_scaler=x_scaler_opt,
                    x_lower=x_lower_opt,
                    x_upper=x_upper_opt,
                    camber_max_percent=float(camber_max_percent) if 'camber_max_percent' in globals() else 6.0,
                    camber_scaler=float(np.mean(camber_scaler)) if 'camber_scaler' in globals() else 1.0,
                    thick_shape_scaler=float(np.mean(thick_shape_scaler)) if 'thick_shape_scaler' in globals() else 1.0,
                    load_factor_val=float(load_factor_val) if 'load_factor_val' in globals() else 2.5,
                )
                print(f"Cached structural telemetry saved to: {struct_cache_opt}")
            except Exception as e:
                print(f"Warning: could not save structural telemetry cache: {e}")
    
    plotter.close()
    
    # Compile rendered frames into MP4 video using bundled imageio_ffmpeg binary
    try:
        import imageio_ffmpeg, subprocess, shutil
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, '-y',
            '-framerate', '4',
            '-i', os.path.join(frames_dir, 'frame_%04d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            video_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if res.returncode == 0:
            print(f"Video successfully saved to: {video_path}")
            shutil.rmtree(frames_dir, ignore_errors=True)
        else:
            print(f"Warning: ffmpeg video encoding failed: {res.stderr.decode('utf-8', errors='ignore')}")
    except Exception as e:
        print(f"Warning: video compilation encountered an error: {e}")
    
    # region Plot Summary Figure (Wing geometry vs Theory)
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    
    # Compute theoretical Cd = Cl^2 / (pi * AR) in drag counts (x 1e4)
    target_sref = 10.0 * scale_factor ** 2
    final_sref = float(sref_history[-1]) if len(sref_history) > 0 else float(target_sref)
    final_ar = float(np.asarray(jax_sim[aspect_ratio_calc]).flatten()[0])
    final_w_total = float(np.asarray(jax_sim[W_total]).flatten()[0])
    q_cruise_val = float(cruise_cond['dynamic_pressure_Pa'])
    target_cl = float(final_w_total / (q_cruise_val * final_sref))
    cd_theory_counts = float((target_cl ** 2) / (np.pi * final_ar) * 1e4)
    
    has_geonic_history = use_geonic and len(geonic_margin_history) == num_iterations
    n_rows = 4 if has_geonic_history else 3
    fig_height = 8 if has_geonic_history else 6
    fig = plt.figure(figsize=(14, fig_height), dpi=150)
    gs = fig.add_gridspec(n_rows, 2, width_ratios=[1.1, 1.5], wspace=0.25, hspace=0.25)
    
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
    ax_cd.set_ylabel(f'CD ({induced_drag_objective})', fontsize=11)
    plt.setp(ax_cd.get_xticklabels(), visible=False)
    for spine in ax_cd.spines.values():
        spine.set_visible(False)
    
    # 2. Middle Subplot: CL
    ax_cl = fig.add_subplot(gs[1, 1], sharex=ax_cd)
    ax_cl.set_facecolor(subplot_bg)
    ax_cl.grid(True, color=grid_color, linewidth=1.2)
    ax_cl.plot(iters, cl_history, 'o-', color='#4fa86c', linewidth=2, markersize=5)
    ax_cl.axhline(target_cl, color='#87c79d', linestyle='--', linewidth=1.8)
    ax_cl.text(1.02, target_cl, 'L=W trim', color='#87c79d', transform=ax_cl.get_yaxis_transform(),
                va='center', fontsize=11, fontweight='bold')
    ax_cl.set_ylabel('CL', fontsize=11)
    plt.setp(ax_cl.get_xticklabels(), visible=False)
    for spine in ax_cl.spines.values():
        spine.set_visible(False)
    
    # 3. Third Subplot: S_ref
    ax_sref = fig.add_subplot(gs[2, 1], sharex=ax_cd)
    ax_sref.set_facecolor(subplot_bg)
    ax_sref.grid(True, color=grid_color, linewidth=1.2)
    ax_sref.plot(iters, sref_history, 'o-', color='#c54b4b', linewidth=2, markersize=5)
    ax_sref.axhline(target_sref, color='#e08585', linestyle='--', linewidth=1.8)
    ax_sref.text(1.02, target_sref, 'con', color='#e08585', transform=ax_sref.get_yaxis_transform(),
                va='center', fontsize=11, fontweight='bold')
    ax_sref.set_ylabel('S_ref [m²]', fontsize=11)
    if has_geonic_history:
        plt.setp(ax_sref.get_xticklabels(), visible=False)
    else:
        ax_sref.set_xlabel('Iterations', fontsize=11)
    for spine in ax_sref.spines.values():
        spine.set_visible(False)

    # 4. Fourth Subplot (if geonic): Geonic Margin
    if has_geonic_history:
        ax_geo = fig.add_subplot(gs[3, 1], sharex=ax_cd)
        ax_geo.set_facecolor(subplot_bg)
        ax_geo.grid(True, color=grid_color, linewidth=1.2)
        ax_geo.plot(iters, geonic_margin_history, 'o-', color='#9467bd', linewidth=2, markersize=5)
        ax_geo.axhline(0.0, color='#2ca02c', linestyle='--', linewidth=1.8)
        ax_geo.text(1.02, 0.0, 'con (≥0)', color='#2ca02c', transform=ax_geo.get_yaxis_transform(),
                    va='center', fontsize=11, fontweight='bold')
        ax_geo.set_ylabel('Geonic Margin [m]', fontsize=11)
        ax_geo.set_xlabel('Iterations', fontsize=11)
        for spine in ax_geo.spines.values():
            spine.set_visible(False)
    
    fig.suptitle("Optimal wing geometry vs theory", y=0.03, fontsize=15, fontweight='bold')
    
    summary_fig_path = os.path.join(latest_folder, 'optimization_summary.png')
    plt.savefig(summary_fig_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Summary figure saved to: {summary_fig_path}")

    try:
        diagnostic_data_path, diagnostic_figure_path = wake_load_diagnostic.save_and_plot(latest_folder)
        print(f"Wake-load consistency diagnostic saved to: {diagnostic_data_path}")
        print(f"Wake-load consistency figure saved to: {diagnostic_figure_path}")
    except Exception as exc:
        print(f"Warning: unable to save wake-load consistency diagnostic: {exc}")
    try:
        closure_data_path, closure_figure_path = wake_closure_diagnostic.save_and_plot(latest_folder)
        print(f"Wake-circulation closure diagnostic saved to: {closure_data_path}")
        print(f"Wake-circulation closure figure saved to: {closure_figure_path}")
    except Exception as exc:
        print(f"Warning: unable to save wake-circulation closure diagnostic: {exc}")

    # Viscous Drag Mode and Attachment Final Summary
    cd_visc_final = float(np.asarray(jax_sim[CD_viscous]).flatten()[0])
    d_visc_final = float(np.asarray(jax_sim[D_viscous]).flatten()[0])
    hmax_final = float(np.asarray(jax_sim[H_max_ibl]).flatten()[0])
    att_margin_final = float(np.asarray(jax_sim[ibl_attachment_margin]).flatten()[0])
    min_cp_final = float(np.asarray(jax_sim[ibl_min_cp]).flatten()[0])
    cp_cut_margin_final = float(np.asarray(jax_sim[ibl_cp_cutoff_margin]).flatten()[0])
    att_st_final = "ATTACHED (FEASIBLE)" if att_margin_final >= 0.0 else "SEPARATED (VIOLATED)"
    print(f"\n================ VISCOUS DRAG FINAL SUMMARY ================")
    print(f"Selected Mode: {viscous_drag_mode.upper()}")
    print(f"CD_viscous:    {cd_visc_final*1e4:.2f} counts ({cd_visc_final:.6f})")
    print(f"D_viscous:     {d_visc_final:.1f} N")
    print(f"H_max_ibl:     {hmax_final:.4f} (Limit: 2.40)")
    print(f"Attach Margin: {att_margin_final:+.4f} ({att_st_final})")
    print(f"Min Cp:        {min_cp_final:.4f} (Cutoff Margin: {cp_cut_margin_final:+.4f})")
    print(f"============================================================\n")
    
    import shutil
    current_conv_id = '0ac41e2d-0335-41d9-9f48-d6791df931f1'
    candidate_ids = [
        current_conv_id,
        'ce0e9874-a39a-41db-b481-5008cc744a13',
        '0c0a47e5-2e16-41bb-9139-10357c23c5ee',
        '3256dd7c-d4c8-4c73-887d-131361f9d0c3',
        '680209fd-293d-4fa0-9f6c-ae59a72a6987',
    ]
    artifact_dir = os.environ.get('ARTIFACT_DIR', None)
    if not artifact_dir or not os.path.exists(artifact_dir):
        for c_id in candidate_ids:
            cand_path = f'/home/andrew/.gemini/antigravity/brain/{c_id}'
            if os.path.exists(cand_path):
                artifact_dir = cand_path
                break

    if os.path.exists(artifact_dir):
        if os.path.exists(video_path):
            shutil.copy2(video_path, os.path.join(artifact_dir, 'optimization_history.mp4'))
        if os.path.exists(summary_fig_path):
            shutil.copy2(summary_fig_path, os.path.join(artifact_dir, 'optimization_summary.png'))
        if os.path.exists(wing_img_path):
            shutil.copy2(wing_img_path, os.path.join(artifact_dir, 'final_wing.png'))
        if 'diagnostic_figure_path' in locals() and os.path.exists(diagnostic_figure_path):
            shutil.copy2(diagnostic_figure_path, os.path.join(artifact_dir, 'wake_load_consistency.png'))
        if 'closure_figure_path' in locals() and os.path.exists(closure_figure_path):
            shutil.copy2(closure_figure_path, os.path.join(artifact_dir, 'wake_circulation_closure.png'))
        print("Base artifacts successfully copied to brain artifact directory!")

    import sys

    # region Automatic Lift Distribution Analysis
    try:
        from optimization_analyses.extract_lift_distribution import extract_and_plot_lift_distribution
        extract_and_plot_lift_distribution(
            output_folder=latest_folder,
            artifact_dir=artifact_dir,
            jax_sim=jax_sim,
            main_script=sys.modules[__name__],
        )
        if artifact_dir and os.path.exists(artifact_dir):
            lift_fig = os.path.join(latest_folder, 'lift_distribution.png')
            if os.path.exists(lift_fig):
                shutil.copy2(lift_fig, os.path.join(artifact_dir, 'lift_distribution.png'))
        print("Automatic lift distribution analysis completed successfully.")
    except Exception as exc:
        print(f"Warning: automatic lift distribution analysis failed: {exc}")
    # endregion

    # region Automatic Lift and Pitching Moment Distribution Analysis
    try:
        from optimization_analyses.extract_lift_and_moment_distributions import extract_and_plot_lift_and_moment
        extract_and_plot_lift_and_moment(
            output_folder=latest_folder,
            artifact_dir=artifact_dir,
            jax_sim=jax_sim,
            main_script=sys.modules[__name__],
        )
        print("Automatic lift and moment distribution analysis completed successfully.")
    except Exception as exc:
        print(f"Warning: automatic lift and moment distribution analysis failed: {exc}")
    # endregion

    # region Automatic Section Cl & Stall Margin Analysis
    try:
        from optimization_analyses.extract_cl_distribution import extract_and_plot_cl_distribution
        extract_and_plot_cl_distribution(
            output_folder=latest_folder,
            artifact_dir=artifact_dir,
        )
        print("Automatic section cl & stall margin analysis completed successfully.")
    except Exception as exc:
        print(f"Warning: automatic section cl distribution analysis failed: {exc}")
    # endregion

    # region Automatic Transonic Wave Drag Distribution Analysis
    try:
        from optimization_analyses.extract_wave_drag_distribution import extract_and_plot_wave_drag_distribution
        extract_and_plot_wave_drag_distribution(
            output_folder=latest_folder,
            jax_sim=jax_sim,
            main_script=sys.modules[__name__],
            artifact_dir=artifact_dir,
        )
        print("Automatic wave drag distribution analysis completed successfully.")
    except Exception as exc:
        print(f"Warning: automatic wave drag distribution analysis failed: {exc}")
    # endregion

    # region Automatic Structural Thickness & Stress Analysis
    try:
        from optimization_analyses.generate_structural_plots import generate_structural_plots
        generate_structural_plots(
            output_folder=latest_folder,
            artifact_dir=artifact_dir,
        )
        print("Automatic structural analysis completed successfully.")
    except Exception as exc:
        print(f"Warning: automatic structural analysis failed: {exc}")
    # endregion

    # region Automatic Airfoil Cross-Section Analysis
    try:
        from optimization_analyses.plot_airfoil_cross_sections import (
            parse_design_variables,
            build_and_evaluate_geometry,
            extract_station_airfoils,
            generate_airfoil_gallery_plot,
            generate_shape_evolution_plot,
            save_airfoil_telemetry,
        )
        x_out_file = os.path.join(latest_folder, "x.out")
        if os.path.exists(x_out_file):
            x_hist = np.loadtxt(x_out_file)
            x_opt_last = x_hist[-1] if x_hist.ndim > 1 else x_hist
            p_dvs = parse_design_variables(x_opt_last, scale_factor=float(scale_factor), target_dir=latest_folder)
            repo_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
            geom_eval, wingspan_m, half_span_m, l_chords, c_stretches, cad_prof = build_and_evaluate_geometry(p_dvs, repo_root_dir)
            stn_data = extract_station_airfoils(geom_eval, half_span_m, num_stations=p_dvs['num_stations'])
            gal_path = os.path.join(latest_folder, "airfoil_cross_sections_gallery.png")
            evo_path = os.path.join(latest_folder, "airfoil_shape_evolution.png")
            tel_path = os.path.join(latest_folder, "airfoil_cross_sections_data.npz")
            generate_airfoil_gallery_plot(stn_data, gal_path, half_span_m, parsed_dvs=p_dvs)
            generate_shape_evolution_plot(stn_data, evo_path, half_span_m, p_dvs, l_chords, c_stretches, cad_prof)
            save_airfoil_telemetry(stn_data, tel_path, half_span_m)
            if os.path.exists(artifact_dir):
                shutil.copy2(gal_path, os.path.join(artifact_dir, "airfoil_cross_sections_gallery.png"))
                shutil.copy2(evo_path, os.path.join(artifact_dir, "airfoil_shape_evolution.png"))
            print("Airfoil cross-section analysis completed automatically.")
    except Exception as e:
        print(f"Warning: automatic airfoil cross-section analysis skipped: {e}")
    # endregion
    
    # endregion Plot Summary Figure
    # endregion Plot Optimization History
    
