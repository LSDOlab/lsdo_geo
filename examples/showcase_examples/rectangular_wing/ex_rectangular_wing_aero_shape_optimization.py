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

geometry = Geometry(functions=geometry.functions, function_names=geometry.function_names,
                             name=geometry.name, space=geometry.space)
base_geometry_coefficients = {idx: (func.coefficients.value.copy() if hasattr(func.coefficients, 'value') else np.array(func.coefficients).copy()) for idx, func in geometry.functions.items()}
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

projected_panel_centers = geometry.project(panel_centers, 
                            grid_search_density_parameter=1,
                            newton_tolerance=1.e-10,
                            grid_search_density_cutoff=30,
                            projection_tolerance=1.e-2,
                            force_reprojection=False, 
                            plot=False,
                            )
right_panel_indices = np.where(panel_centers[:, 1] > 0.0)[0]
num_right_panels = len(right_panel_indices)

# endregion

# endregion

# region Create Parameterization Objects and Model Builder

@dataclass
class DVInfo:
    variable: csdl.Variable
    lower: float
    upper: float
    scaler: float = 1.0


def build_optimization_model(formulation='chord_span',
                             obj_scaler=1.e3,
                             cl_scaler=1.e1,
                             area_scaler=1.e-1,
                             ar_scaler=1.e-1,
                             dv_scalers=None,
                             include_cl_constraint=False,
                             include_pitch_dv=False):
    """
    Builds the CSDL computational graph, VortexAD aerodynamic panel solver,
    parameterization and constraints, and compiles the JaxSimulator.

    Parameters
    ----------
    formulation : str
        'ar_area' or 'mdf' (implicit method via ParameterizationSolver),
        'sand' (Simultaneous Analysis and Design with optimizer solving parameterization constraints), or
        'chord_span' (explicit method with direct stretch DVs and constraints).
    obj_scaler : float
        Objective scaling factor for CDi (default 1e3).
    cl_scaler : float
        Constraint scaling factor for CL (default 1e1).
    area_scaler : float
        Constraint scaling factor for planform area (default 1e-1).
    ar_scaler : float
        Constraint scaling factor for aspect ratio (default 1e-1).
    dv_scalers : dict, optional
        Custom design variable scalers dict.
    include_cl_constraint : bool
        Whether to add CL == 0.5 constraint (default False).
    include_pitch_dv : bool
        Whether to include pitch as an active design variable (default False).

    Returns
    -------
    tuple
        (jax_sim, design_variables, outputs_dict, geometry, ffd_block)
    """
    if dv_scalers is None:
        dv_scalers = {}

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    # Reset geometry coefficients to un-deformed base geometry for this recorder
    for idx, coeff_val in base_geometry_coefficients.items():
        geometry.functions[idx].coefficients = csdl.Variable(value=coeff_val.copy())

    # Construct Free Form Deformation (FFD) block around the geometry
    num_ffd_coefficients_chordwise = 2
    num_ffd_sections = 2
    ffd_block = construct_ffd_block_around_entities(
        entities=geometry, 
        num_coefficients=(num_ffd_coefficients_chordwise, num_ffd_sections, 2),
        degree=(1, 1, 1)
    )

    ffd_sectional_parameterization = SectionalParameterization(
        name="ffd_sectional_parameterization",
        parameterized_points=ffd_block.coefficients,
        principal_parametric_dimension=1,
    )

    pitch = csdl.Variable(value=5.*np.pi/180) # pitch angle in radians
    pitch_scaler = dv_scalers.get('pitch', 1.e1)

    space_of_linear_2_dof_b_splines = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))

    if formulation in ['ar_area', 'mdf']:
        # Formulation 1 (Implicit / MDF): Aspect Ratio (AR) and Planform Area (S)
        aspect_ratio = csdl.Variable(shape=(1,), value=np.array([10.0]))
        planform_area_dv = csdl.Variable(shape=(1,), value=np.array([10.0]))
        ar_dv_scaler = dv_scalers.get('aspect_ratio', 1.e-1)

        design_variables: dict[str, DVInfo] = {
            'aspect_ratio': DVInfo(variable=aspect_ratio, lower=2.0, upper=15.0, scaler=ar_dv_scaler),
        }
        if include_pitch_dv:
            design_variables['pitch'] = DVInfo(variable=pitch, lower=-10.0*np.pi/180, upper=15.0*np.pi/180, scaler=pitch_scaler)

        # ParameterizationSolver states
        chord_stretch_state = csdl.Variable(value=0.0)
        span_stretch_state = csdl.Variable(value=0.0)

        chord_coeffs = csdl.expand(chord_stretch_state, (num_ffd_sections,))
        wingspan_coeffs = csdl.concatenate([-span_stretch_state, span_stretch_state])

    elif formulation == 'sand':
        # Formulation 3 (SAND): Optimizer solves parameterization states with equality constraints
        aspect_ratio = csdl.Variable(shape=(1,), value=np.array([10.0]))
        ar_dv_scaler = dv_scalers.get('aspect_ratio', 1.e-1)

        chord_stretch_state = csdl.Variable(shape=(1,), value=np.array([0.0]))
        span_stretch_state = csdl.Variable(shape=(1,), value=np.array([0.0]))
        cs_scaler = dv_scalers.get('chord_stretch_state', 1.0)
        ss_scaler = dv_scalers.get('span_stretch_state', 1.e-1)

        design_variables: dict[str, DVInfo] = {
            'aspect_ratio': DVInfo(variable=aspect_ratio, lower=2.0, upper=15.0, scaler=ar_dv_scaler),
            'chord_stretch_state': DVInfo(variable=chord_stretch_state, lower=-0.85, upper=4.0, scaler=cs_scaler),
            'span_stretch_state': DVInfo(variable=span_stretch_state, lower=-4.5, upper=20.0, scaler=ss_scaler),
        }
        if include_pitch_dv:
            design_variables['pitch'] = DVInfo(variable=pitch, lower=-10.0*np.pi/180, upper=15.0*np.pi/180, scaler=pitch_scaler)

        chord_coeffs = csdl.expand(chord_stretch_state, (num_ffd_sections,))
        wingspan_coeffs = csdl.concatenate([-span_stretch_state, span_stretch_state])

    elif formulation == 'chord_span':
        # Formulation 2 (Explicit): Chord stretch and Span stretch DVs
        chord_stretch_dv = csdl.Variable(shape=(1,), value=np.array([0.0]))
        span_stretch_dv = csdl.Variable(shape=(1,), value=np.array([0.0]))
        cs_dv_scaler = dv_scalers.get('chord_stretch_dv', 1.0)
        ss_dv_scaler = dv_scalers.get('span_stretch_dv', 1.e-1)

        design_variables: dict[str, DVInfo] = {
            'chord_stretch_dv': DVInfo(variable=chord_stretch_dv, lower=-0.85, upper=4.0, scaler=cs_dv_scaler),
            'span_stretch_dv': DVInfo(variable=span_stretch_dv, lower=-4.5, upper=20.0, scaler=ss_dv_scaler),
        }
        if include_pitch_dv:
            design_variables['pitch'] = DVInfo(variable=pitch, lower=-10.0*np.pi/180, upper=15.0*np.pi/180, scaler=pitch_scaler)

        chord_coeffs = csdl.expand(chord_stretch_dv, (num_ffd_sections,))
        wingspan_coeffs = csdl.concatenate([-span_stretch_dv, span_stretch_dv])

    chord_stretching_b_spline = lfs.Function(
        space=space_of_linear_2_dof_b_splines,
        coefficients=chord_coeffs,
        name='chord_stretching_b_spline_coefficients'
    )

    wingspan_stretching_b_spline = lfs.Function(
        space=space_of_linear_2_dof_b_splines,
        coefficients=wingspan_coeffs,
        name='wingspan_stretching_b_spline_coefficients'
    )

    parametric_b_spline_inputs = np.linspace(0.0, 1.0, num_ffd_sections).reshape((-1, 1))
    chord_stretch_sectional_parameters = chord_stretching_b_spline.evaluate(parametric_b_spline_inputs)
    wingspan_stretch_sectional_parameters = wingspan_stretching_b_spline.evaluate(parametric_b_spline_inputs)

    sectional_parameters = SectionalParameters()
    sectional_parameters.add_stretch(axis=0, stretch=chord_stretch_sectional_parameters)
    sectional_parameters.add_translation(axis=1, translation=wingspan_stretch_sectional_parameters)

    ffd_coefficients = ffd_sectional_parameterization.evaluate(sectional_parameters, plot=False)

    geometry_coefficients = ffd_block.evaluate_ffd(coefficients=ffd_coefficients, plot=False)
    geometry.set_coefficients(geometry_coefficients)

    wingspan = geometry.evaluate(leading_edge_right)[1] - geometry.evaluate(leading_edge_left)[1]
    chord_root = geometry.evaluate(chord_te_projections[0])[0] - geometry.evaluate(chord_le_projections[0])[0]

    planform_area_geom = wingspan * chord_root
    aspect_ratio_geom = wingspan / chord_root

    if formulation in ['ar_area', 'mdf']:
        # ParameterizationSolver explicitly enforces geometry to match aspect_ratio and planform_area_dv
        geometry_solver = ParameterizationSolver()
        geometry_solver.add_state(chord_stretch_state)
        geometry_solver.add_state(span_stretch_state)

        geometric_variables = GeometricVariables()
        geometric_variables.add_variable(planform_area_geom, planform_area_dv, penalty_value=None)
        geometric_variables.add_variable(aspect_ratio_geom, aspect_ratio, penalty_value=None)

        geometry_solver.evaluate(geometric_variables)

    geometry.rotate(rotation_origin=geometry.evaluate(quarter_chord_center), axis_vector=np.array([0., 1., 0.]), angles=pitch, units='radians')

    cruise_speed = csdl.Variable(value=1.)
    velocity = csdl.concatenate([cruise_speed, csdl.Variable(value=0.), csdl.Variable(value=0.)])

    num_nodes = 1
    panel_mesh = geometry.evaluate(projected_panel_mesh, plot=False)
    panel_mesh = panel_mesh.expand((1,) + panel_mesh.shape, 'ij->aij')

    point_velocities = csdl.expand(velocity, (num_nodes,) + panel_mesh.shape[1:], 'j->iaj')
    rho_array = csdl.Variable(shape=(num_nodes,), value=np.array([1.225]))
    sos_array = csdl.Variable(shape=(num_nodes,), value=np.array([343.0]))

    upper_skin_pts = geometry.evaluate(projected_upper_skin, plot=False)
    lower_skin_pts = geometry.evaluate(projected_lower_skin, plot=False)
    chord_surface_pts = 0.5 * (upper_skin_pts + lower_skin_pts)

    chord_surface_grid = csdl.reshape(chord_surface_pts, (nx_area, ny_area, 3))
    v_x = chord_surface_grid[1:, :-1, :] - chord_surface_grid[:-1, :-1, :]
    v_y = chord_surface_grid[:-1, 1:, :] - chord_surface_grid[:-1, :-1, :]
    area_vectors = csdl.cross(v_x, v_y, axis=2)
    element_areas = csdl.norm(area_vectors, axes=(2,))
    planform_area = csdl.sum(element_areas)

    pm_solver_inputs = {
        'V_inf': -point_velocities,
        'rho': rho_array,
        'sos': sos_array,
        'compressibility': True,
        'Cp cutoff': -5.,
        'partition_size': 1,
        'reuse_AIC': True,
        'ref_area': planform_area
    }

    panel_method = VortexAD.PanelMethod(
        solver_input_dict=pm_solver_inputs,
        skip_geometry=True
    )
    panel_method.insert_grid_data(
        mesh=panel_mesh[0,:],
        cell_adjacency_data=cell_adjacency_data,
        TE_properties=TE_properties
    )

    panel_method.declare_outputs([
        'Cp', 'L', 'Di', 'M', 'panel_forces', 'CL', 'CDi', 'CDi_Trefftz', 'CM',
    ])

    recorder.inline = False
    outputs = panel_method.evaluate()

    CL = outputs['CL']
    CDi = outputs['CDi_Trefftz']
    L = outputs['L']
    Di = outputs['Di']
    Cp = outputs['Cp']

    dynamic_panel_centers = geometry.evaluate(projected_panel_centers, plot=False)
    dynamic_panel_centers_right = dynamic_panel_centers[:num_right_panels, :]
    panel_forces_right = outputs['panel_forces'][0, :num_right_panels, :]

    aspect_ratio_calc = (wingspan**2) / planform_area

    objective = CDi
    objective.set_as_objective(scaler=obj_scaler)

    if include_cl_constraint:
        CL.set_as_constraint(equals=0.5, scaler=cl_scaler)

    if formulation == 'chord_span':
        planform_area.set_as_constraint(equals=10.0, scaler=area_scaler)
        aspect_ratio_calc.set_as_constraint(upper=15.0, scaler=ar_scaler)
    elif formulation == 'sand':
        planform_area_geom.set_as_constraint(equals=10.0, scaler=area_scaler)
        (aspect_ratio_geom - aspect_ratio).set_as_constraint(equals=0.0, scaler=ar_scaler)

    for dv_info in design_variables.values():
        dv_info.variable.set_as_design_variable(lower=dv_info.lower, upper=dv_info.upper, scaler=dv_info.scaler)

    geom_coeffs = [geometry_function.coefficients for geometry_function in geometry.functions.values()]

    additional_outputs_list = [
        Di, L, CL, CDi, Cp, panel_mesh, planform_area, aspect_ratio_calc,
        dynamic_panel_centers_right, panel_forces_right, planform_area_geom, aspect_ratio_geom
    ] + geom_coeffs

    jax_sim = csdl.experimental.JaxSimulator(
        recorder=recorder,
        additional_inputs=[dv_info.variable for dv_info in design_variables.values()],
        additional_outputs=additional_outputs_list,
        gpu=False
    )

    outputs_dict = {
        'CL': CL,
        'CDi': CDi,
        'L': L,
        'Di': Di,
        'Cp': Cp,
        'panel_mesh': panel_mesh,
        'planform_area': planform_area,
        'planform_area_geom': planform_area_geom,
        'aspect_ratio_geom': aspect_ratio_geom,
        'aspect_ratio_calc': aspect_ratio_calc,
        'wingspan': wingspan,
        'chord_root': chord_root,
        'dynamic_panel_centers_right': dynamic_panel_centers_right,
        'panel_forces_right': panel_forces_right,
    }

    return jax_sim, design_variables, outputs_dict, geometry, ffd_block

# Default formulation when running standalone: 'chord_span' or 'ar_area'
formulation = 'chord_span'
jax_sim, design_variables, outputs_dict, geometry, ffd_block = build_optimization_model(
    formulation=formulation, include_cl_constraint=True, include_pitch_dv=True
)
CL = outputs_dict['CL']
CDi = outputs_dict['CDi']
L = outputs_dict['L']
Di = outputs_dict['Di']
Cp = outputs_dict['Cp']
panel_mesh = outputs_dict['panel_mesh']
planform_area = outputs_dict['planform_area']
aspect_ratio_calc = outputs_dict['aspect_ratio_calc']
dynamic_panel_centers_right = outputs_dict['dynamic_panel_centers_right']
panel_forces_right = outputs_dict['panel_forces_right']

# Run and plot panel method solution to make sure simulation is working
# -- This does not need to be run for optimization, but is useful for debugging
# jax_sim.run()
# print(f"Lift: {jax_sim[L]}")
# print(f"Drag: {jax_sim[Di]}")
# exit()
# panel_method.points_orig = panel_mesh.value
# panel_method.plot(Cp.value, bounds=[-0.5,1])

if __name__ == '__main__':
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
                            elif 'chord_dvs' in inp_grp:
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

        # Build parameter info string depending on active formulation
        if formulation == 'ar_area':
            ar_val = float(unscaled_values['aspect_ratio'].item() if hasattr(unscaled_values['aspect_ratio'], 'item') else unscaled_values['aspect_ratio'][0]) if 'aspect_ratio' in unscaled_values else 10.0
            s_val = float(unscaled_values['planform_area_dv'].item() if hasattr(unscaled_values['planform_area_dv'], 'item') else unscaled_values['planform_area_dv'][0]) if 'planform_area_dv' in unscaled_values else 10.0
            dv_str = f"AR={ar_val:.2f}  S={s_val:.2f}"
        elif formulation == 'chord_span':
            ss_val = float(unscaled_values['span_stretch_dv'].item() if hasattr(unscaled_values['span_stretch_dv'], 'item') else unscaled_values['span_stretch_dv'][0]) if 'span_stretch_dv' in unscaled_values else 0.0
            if 'chord_dvs' in unscaled_values:
                c_vals = unscaled_values['chord_dvs']
                c_str = " ".join([f"c{i}={c_vals[i]:.2f}" for i in range(len(c_vals))])
                dv_str = f"b_stretch={ss_val:.2f}  {c_str}"
            elif 'chord_stretch_dvs' in unscaled_values:
                cs_vals = unscaled_values['chord_stretch_dvs']
                c_str = " ".join([f"c{i}={1.0+cs_vals[i]:.2f}" for i in range(len(cs_vals))])
                dv_str = f"b_stretch={ss_val:.2f}  {c_str}"
            else:
                cs_val = float(unscaled_values['chord_stretch_dv'].item() if hasattr(unscaled_values['chord_stretch_dv'], 'item') else unscaled_values['chord_stretch_dv'][0]) if 'chord_stretch_dv' in unscaled_values else 0.0
                dv_str = f"b_stretch={ss_val:.2f}  c_stretch={cs_val:.2f}"

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

    # region Automatic Lift Distribution Analysis
    from optimization_analyses.extract_lift_distribution import extract_and_plot_lift_distribution
    import sys
    extract_and_plot_lift_distribution(
        output_folder=latest_folder,
        jax_sim=jax_sim,
        main_script=sys.modules[__name__],
    )
    # endregion

    # endregion Plot Summary Figure
    # endregion Plot Optimization History