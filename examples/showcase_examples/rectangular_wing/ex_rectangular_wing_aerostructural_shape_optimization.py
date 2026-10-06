"""
Aerostructural Shape Optimization of Rectangular Wing with Aframe Beam Model
=============================================================================
Coupled aerodynamic (VortexAD) and structural (aframe) shape optimization.
Structural thicknesses are held constant to drive the optimal aspect ratio (AR)
to an interior optimum between initial (AR0 = 10.0) and upper bound (AR = 15.0).

Supports formulations:
  - 'ar_area' (or 'mdf'): Implicit geometric parameterization via ParameterizationSolver.
  - 'chord_span': Explicit geometric parameterization with direct FFD stretch DVs.
  - 'sand': Simultaneous Analysis and Design with parameterization constraints in SLSQP.
"""

from dataclasses import dataclass
import os
import sys
import numpy as np
import csdl_alpha as csdl
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

# Initialize CSDL Recorder for initial geometry and mesh imports
recorder = csdl.Recorder(inline=True)
recorder.start()

geometry_directory = "examples/example_geometries/"
file_name = "rectangular_wing_naca0012_10ar"
geometry = import_geometry(geometry_directory + file_name + ".stp", parallelize=False)

num_spanwise_cp_target = 15
for idx in list(geometry.functions.keys()):
    function = geometry.functions[idx]
    coeffs = function.coefficients.value if hasattr(function.coefficients, 'value') else function.coefficients

    axis0_y_range = np.ptp([coeffs[i, :, 1].mean() for i in range(coeffs.shape[0])])
    axis1_y_range = np.ptp([coeffs[:, j, 1].mean() for j in range(coeffs.shape[1])])

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
                new_coeffs[j, :, k] = np.linspace(coeffs[j, 0, k], coeffs[-1, j, k], num_spanwise_cp_target)
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

leading_edge_left = geometry.project(np.array([0.0, -5.0, 0.0]))
leading_edge_right = geometry.project(np.array([0.0, 5.0, 0.0]))
quarter_chord_center = geometry.project(np.array([0.25, 0.0, 0.0]))

num_chord_stations = 5
chord_station_y = np.linspace(0.0, 5.0, num_chord_stations)
chord_le_projections = [geometry.project(np.array([0.0, y, 0.0])) for y in chord_station_y]
chord_te_projections = [geometry.project(np.array([1.0, y, 0.0])) for y in chord_station_y]

chord_le_projections_left = [geometry.project(np.array([0.0, -y, 0.0])) for y in chord_station_y[1:]]
chord_te_projections_left = [geometry.project(np.array([1.0, -y, 0.0])) for y in chord_station_y[1:]]

nx_area = 21
ny_area = 41
x_grid = np.linspace(0.0, 1.0, nx_area)
y_grid = np.linspace(-5.0, 5.0, ny_area)
X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid, indexing='ij')

upper_seed_pts = np.column_stack([X_mesh.ravel(), Y_mesh.ravel(), np.full(X_mesh.size, 0.05)])
lower_seed_pts = np.column_stack([X_mesh.ravel(), Y_mesh.ravel(), np.full(X_mesh.size, -0.05)])

projected_upper_skin = geometry.project(upper_seed_pts, force_reprojection=False, direction=np.array([0, 0, -1]), plot=False)
projected_lower_skin = geometry.project(lower_seed_pts, force_reprojection=False, direction=np.array([0, 0, 1]), plot=False)

mesh = meshio.read(geometry_directory + file_name + ".msh")
points_orig = mesh.points
cells_dict = mesh.cells_dict
cell_adjacency_data = VortexAD.find_cell_adjacency(points=points_orig, cells=cells_dict)
points_orig, cells_dict, cell_adjacency, edges2cells, points2cells = cell_adjacency_data[:5]
TE_properties = VortexAD.TE_detection(points=points_orig, cells=cells_dict, edges2cells=edges2cells, points2cells=points2cells, threshold_theta=125.)

projected_panel_mesh = geometry.project(
    points_orig, grid_search_density_parameter=1, newton_tolerance=1.e-10,
    grid_search_density_cutoff=30, projection_tolerance=1.e-3, force_reprojection=False, plot=False
)

@dataclass
class DVInfo:
    variable: csdl.Variable
    lower: float
    upper: float
    scaler: float = 1.0


def build_optimization_model(formulation='chord_span',
                             obj_scaler=100.0,
                             area_scaler=1.e-1,
                             ar_scaler=1.e-1,
                             disp_scaler=50.0,
                             dv_scalers=None,
                             spar_thickness=0.003,
                             max_tip_disp=0.020,
                             cruise_speed_val=20.0):
    if dv_scalers is None:
        dv_scalers = {}

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    for idx, coeff_val in base_geometry_coefficients.items():
        geometry.functions[idx].coefficients = csdl.Variable(value=coeff_val.copy())

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

    pitch = csdl.Variable(value=5.0 * np.pi / 180)
    space_of_linear_2_dof_b_splines = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))

    if formulation in ['ar_area', 'mdf']:
        aspect_ratio = csdl.Variable(shape=(1,), value=np.array([10.0]))
        planform_area_dv = csdl.Variable(shape=(1,), value=np.array([10.0]))
        ar_dv_scaler = dv_scalers.get('aspect_ratio', 1.e-1)
        design_variables: dict[str, DVInfo] = {
            'aspect_ratio': DVInfo(variable=aspect_ratio, lower=2.0, upper=15.0, scaler=ar_dv_scaler),
        }
        chord_stretch_state = csdl.Variable(value=0.0)
        span_stretch_state = csdl.Variable(value=0.0)
        chord_coeffs = csdl.expand(chord_stretch_state, (num_ffd_sections,))
        wingspan_coeffs = csdl.concatenate([-span_stretch_state, span_stretch_state])

    elif formulation == 'sand':
        aspect_ratio = csdl.Variable(shape=(1,), value=np.array([10.0]))
        chord_stretch_state = csdl.Variable(shape=(1,), value=np.array([0.0]))
        span_stretch_state = csdl.Variable(shape=(1,), value=np.array([0.0]))
        ar_dv_scaler = dv_scalers.get('aspect_ratio', 1.e-1)
        cs_scaler = dv_scalers.get('chord_stretch_state', 1.0)
        ss_scaler = dv_scalers.get('span_stretch_state', 1.e-1)
        design_variables: dict[str, DVInfo] = {
            'aspect_ratio': DVInfo(variable=aspect_ratio, lower=2.0, upper=15.0, scaler=ar_dv_scaler),
            'chord_stretch_state': DVInfo(variable=chord_stretch_state, lower=-0.85, upper=4.0, scaler=cs_scaler),
            'span_stretch_state': DVInfo(variable=span_stretch_state, lower=-4.5, upper=20.0, scaler=ss_scaler),
        }
        chord_coeffs = csdl.expand(chord_stretch_state, (num_ffd_sections,))
        wingspan_coeffs = csdl.concatenate([-span_stretch_state, span_stretch_state])

    elif formulation == 'chord_span':
        chord_stretch_dv = csdl.Variable(shape=(1,), value=np.array([0.0]))
        span_stretch_dv = csdl.Variable(shape=(1,), value=np.array([0.0]))
        cs_dv_scaler = dv_scalers.get('chord_stretch_dv', 1.0)
        ss_dv_scaler = dv_scalers.get('span_stretch_dv', 1.e-1)
        design_variables: dict[str, DVInfo] = {
            'chord_stretch_dv': DVInfo(variable=chord_stretch_dv, lower=-0.85, upper=4.0, scaler=cs_dv_scaler),
            'span_stretch_dv': DVInfo(variable=span_stretch_dv, lower=-4.5, upper=20.0, scaler=ss_dv_scaler),
        }
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
    geom_coefficients = ffd_block.evaluate_ffd(coefficients=ffd_coefficients, plot=False)
    geometry.set_coefficients(geom_coefficients)

    wingspan = geometry.evaluate(leading_edge_right)[1] - geometry.evaluate(leading_edge_left)[1]
    chord_root = geometry.evaluate(chord_te_projections[0])[0] - geometry.evaluate(chord_le_projections[0])[0]

    planform_area_geom = wingspan * chord_root
    aspect_ratio_geom = wingspan / chord_root

    if formulation in ['ar_area', 'mdf']:
        geometry_solver = ParameterizationSolver()
        geometry_solver.add_state(chord_stretch_state)
        geometry_solver.add_state(span_stretch_state)

        geometric_variables = GeometricVariables()
        geometric_variables.add_variable(planform_area_geom, planform_area_dv, penalty_value=None)
        geometric_variables.add_variable(aspect_ratio_geom, aspect_ratio, penalty_value=None)
        geometry_solver.evaluate(geometric_variables)

    geometry.rotate(rotation_origin=geometry.evaluate(quarter_chord_center), axis_vector=np.array([0., 1., 0.]), angles=pitch, units='radians')

    cruise_speed = csdl.Variable(value=cruise_speed_val)
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

    panel_method = VortexAD.PanelMethod(solver_input_dict=pm_solver_inputs, skip_geometry=True)
    panel_method.insert_grid_data(mesh=panel_mesh[0, :], cell_adjacency_data=cell_adjacency_data, TE_properties=TE_properties)
    panel_method.declare_outputs(['Cp', 'L', 'Di', 'CL', 'CDi', 'CDi_Trefftz'])

    recorder.inline = False
    outputs = panel_method.evaluate()

    CL = outputs['CL']
    CDi = outputs['CDi_Trefftz']
    L = outputs['L']
    Di = outputs['Di']
    aspect_ratio_calc = (wingspan**2) / planform_area

    # Objective: Minimize Aerodynamic Drag
    objective = CDi
    objective.set_as_objective(scaler=obj_scaler)

    # Structural Beam Model (aframe) with CONSTANT Thickness
    le_all = chord_le_projections_left[::-1] + chord_le_projections
    te_all = chord_te_projections_left[::-1] + chord_te_projections
    num_beam_nodes = len(le_all)

    beam_nodes_list = []
    chord_list = []
    for i in range(num_beam_nodes):
        le_pt = geometry.evaluate(le_all[i])
        te_pt = geometry.evaluate(te_all[i])
        node_pt = le_pt + 0.50 * (te_pt - le_pt)
        beam_nodes_list.append(node_pt)
        chord_list.append(csdl.norm(te_pt - le_pt))

    beam_mesh = csdl.reshape(csdl.concatenate(beam_nodes_list), (num_beam_nodes, 3))
    node_chords = csdl.reshape(csdl.concatenate(chord_list), (num_beam_nodes,))
    local_chord = 0.5 * (node_chords[:-1] + node_chords[1:])

    ttop = csdl.Variable(value=np.ones(num_beam_nodes - 1) * spar_thickness)
    tweb = csdl.Variable(value=np.ones(num_beam_nodes - 1) * spar_thickness)

    beam_cs = aframe.CSBox(
        height=0.10 * local_chord,
        width=0.60 * local_chord,
        ttop=ttop,
        tbot=ttop,
        tweb=tweb,
    )
    beam = aframe.Beam(name='wing_spar', mesh=beam_mesh, E=69e9, G=26e9, density=2700, cs=beam_cs)
    beam.fix(node=num_beam_nodes // 2)

    y_norm = np.linspace(-1.0, 1.0, num_beam_nodes)
    weights = np.sqrt(np.maximum(0.0, 1.0 - y_norm**2))
    weights = weights / np.sum(weights)

    loads_list = []
    for n in range(num_beam_nodes):
        fn = L[0] * weights[n]
        loads_list.append(csdl.concatenate([
            csdl.Variable(value=np.array([0.0])),
            csdl.Variable(value=np.array([0.0])),
            fn,
            csdl.Variable(value=np.array([0.0])),
            csdl.Variable(value=np.array([0.0])),
            csdl.Variable(value=np.array([0.0])),
        ]))
    beam_loads = csdl.reshape(csdl.concatenate(loads_list), (num_beam_nodes, 6))
    beam.add_load(beam_loads)

    frame = aframe.Frame(beams=[beam])
    frame.solve()

    disp = frame.displacement['wing_spar']
    tip_deflection = csdl.norm(disp[0, 0:3])

    tip_deflection.set_as_constraint(upper=max_tip_disp, scaler=disp_scaler)

    if formulation == 'chord_span':
        planform_area.set_as_constraint(equals=10.0, scaler=area_scaler)
        aspect_ratio_calc.set_as_constraint(upper=15.0, scaler=ar_scaler)
    elif formulation == 'sand':
        planform_area_geom.set_as_constraint(equals=10.0, scaler=area_scaler)
        (aspect_ratio_geom - aspect_ratio).set_as_constraint(equals=0.0, scaler=ar_scaler)

    for dv_info in design_variables.values():
        dv_info.variable.set_as_design_variable(lower=dv_info.lower, upper=dv_info.upper, scaler=dv_info.scaler)

    additional_outputs_list = [
        Di, L, CL, CDi, planform_area, aspect_ratio_calc,
        planform_area_geom, aspect_ratio_geom, tip_deflection
    ]

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
        'planform_area': planform_area,
        'planform_area_geom': planform_area_geom,
        'aspect_ratio_geom': aspect_ratio_geom,
        'aspect_ratio_calc': aspect_ratio_calc,
        'wingspan': wingspan,
        'chord_root': chord_root,
        'tip_deflection': tip_deflection,
    }

    return jax_sim, design_variables, outputs_dict, geometry, ffd_block
