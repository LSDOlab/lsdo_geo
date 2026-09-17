#!/usr/bin/env python
"""
Airfoil Cross-Section Extraction & Analysis for BWB Optimization Outputs.

This script reads the last/optimal design variable vector from an optimization
run of ex_rectangular_wing_to_bwb.py, reconstructs the authentic deformed CAD geometry,
extracts 2D airfoil cross-sections along 8 spanwise stations (half-span from root to tip),
and produces comprehensive visualization figures and numerical telemetry.

Usage:
    python plot_airfoil_cross_sections.py [output_directory]

If output_directory is omitted, the latest directory under
'rectangular_wing_to_bwb_aerostructural_optimization_outputs' is analyzed.
"""

import os
import sys
import glob
import shutil
import pickle
from typing import Union
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import scipy.interpolate as si
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Ensure CPU backend for JAX if used
os.environ["JAX_PLATFORMS"] = "cpu"

# -----------------------------------------------------------------------------
# Compatibility layer for unpickling across numpy 1.x and 2.x
# -----------------------------------------------------------------------------
class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)

_orig_pickle_load = pickle.load
pickle.load = lambda f, **kwargs: CompatUnpickler(f, **kwargs).load()

import csdl_alpha as csdl
import lsdo_function_spaces as lfs
import lsdo_geo
from lsdo_geo import (
    import_geometry,
    construct_ffd_block_around_entities,
    SectionalParameterization,
    SectionalParameters,
    ParameterizationSolver,
    GeometricVariables,
)


@dataclass
class DVInfo:
    variable: csdl.Variable
    lower: Union[float, npt.NDArray[np.float64]]
    upper: Union[float, npt.NDArray[np.float64]]
    scaler: float = 1.0


def find_latest_output_dir(base_dir: str) -> str:
    """Find the latest modified subdirectory in base_dir."""
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Base output directory not found: {base_dir}")
    folders = [os.path.join(base_dir, d) for d in os.listdir(base_dir)
               if os.path.isdir(os.path.join(base_dir, d))]
    if not folders:
        raise FileNotFoundError(f"No run subdirectories found in: {base_dir}")
    latest = max(folders, key=os.path.getmtime)
    return os.path.abspath(latest)


def parse_design_variables(x_opt: np.ndarray):
    """
    Auto-detect configuration and parse unscaled physical design variable values.
    Supports 5-station ('fast') and 8-station ('full') runs with or without camber/elevator.
    """
    n_dv = len(x_opt)
    print(f"\nAnalyzing design variable vector of length {n_dv}...")

    if n_dv == 42:
        resolution = 'fast'
        include_camber = True
        include_elevator = False
    elif n_dv == 28:
        resolution = 'fast'
        include_camber = False
        include_elevator = True
    elif n_dv == 66:
        resolution = 'full'
        include_camber = True
        include_elevator = False
    elif n_dv == 43:
        resolution = 'full'
        include_camber = False
        include_elevator = True
    elif n_dv >= 60:
        resolution = 'full'
        include_camber = True
        include_elevator = False
    else:
        # Default fallback to fast with camber
        resolution = 'fast'
        include_camber = True
        include_elevator = False

    num_stations = 5 if resolution == 'fast' else 8
    num_chord_stations = num_stations
    scale_factor = 1.0 / np.sqrt(10.0)

    curr = 0
    taper_dvs_val = x_opt[curr : curr + num_chord_stations - 1] / 2.0
    curr += num_chord_stations - 1

    ar_val = x_opt[curr : curr + 1] / 0.5
    curr += 1

    sweep_angle_dvs_val = x_opt[curr : curr + num_chord_stations - 1] / 10.0
    curr += num_chord_stations - 1

    twist_dvs_val = x_opt[curr : curr + num_chord_stations] / 10.0
    curr += num_chord_stations

    pitch_val = x_opt[curr : curr + 1] / 10.0
    curr += 1

    pitch_ss_val = x_opt[curr : curr + 1] / 10.0
    curr += 1

    payload_cg_val = x_opt[curr : curr + 1] / 10.0
    curr += 1

    ttop_val = x_opt[curr : curr + num_stations] / 5000.0
    curr += num_stations

    tweb_val = x_opt[curr : curr + num_stations] / 5000.0
    curr += num_stations

    camber_max_percent = 5.0  # 5.0% chord max camber displacement
    camber_scaler = 1.0 / camber_max_percent  # Scales DVs in [-5.0, 5.0] to [-1, 1] range for optimizer

    if include_camber:
        camber_dvs_val = (x_opt[curr : curr + 3 * num_chord_stations] / camber_scaler).reshape((3, num_chord_stations))
        curr += 3 * num_chord_stations
    else:
        camber_dvs_val = np.zeros((3, num_chord_stations))

    if include_elevator and curr < n_dv:
        elevator_val = float(x_opt[curr : curr + 1] / 10.0)
        curr += 1
    else:
        elevator_val = 0.0

    parsed = {
        'resolution': resolution,
        'include_camber': include_camber,
        'include_elevator': include_elevator,
        'num_stations': num_stations,
        'scale_factor': scale_factor,
        'taper_dvs': taper_dvs_val,
        'aspect_ratio': ar_val,
        'sweep_angle_dvs': sweep_angle_dvs_val,
        'twist_dvs': twist_dvs_val,
        'pitch': pitch_val,
        'pitch_ss': pitch_ss_val,
        'payload_cg': payload_cg_val,
        'ttop_dvs': ttop_val,
        'tweb_dvs': tweb_val,
        'camber_dvs': camber_dvs_val,
        'elevator_angle': elevator_val,
    }

    print(f"  Detected Formulation : Aspect Ratio & Area ('ar_area')")
    print(f"  Resolution           : {resolution} ({num_stations} design stations)")
    print(f"  Camber DVs Active    : {include_camber}")
    print(f"  Elevator Active      : {include_elevator}")
    print(f"  Aspect Ratio         : {float(ar_val[0]):.2f}")
    print(f"  Taper Ratios         : {np.round(taper_dvs_val, 4)}")
    print(f"  Twist Angles (deg)   : {np.round(np.degrees(twist_dvs_val), 2)}")
    return parsed


def build_and_evaluate_geometry(parsed_dvs: dict, repo_root: str):
    """
    Builds the CSDL parameterization and evaluates ParameterizationSolver inline.
    Returns the updated Geometry object, total wingspan, and half-span.
    """
    scale_factor = parsed_dvs['scale_factor']
    num_stations = parsed_dvs['num_stations']
    num_chord_stations = num_stations
    num_ffd_sections = 2 * num_stations - 1
    include_camber = parsed_dvs['include_camber']
    include_elevator = parsed_dvs['include_elevator']

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    stp_path = os.path.join(repo_root, "examples/example_geometries/rectangular_wing_naca0012_10ar.stp")
    if not os.path.exists(stp_path):
        raise FileNotFoundError(f"CAD geometry file not found at: {stp_path}")

    print("\nImporting CAD geometry...")
    geometry = import_geometry(stp_path, name='imported_geometry', parallelize=False)
    for f in geometry.functions.values():
        f.coefficients = f.coefficients * scale_factor

    # Interpolate spanwise control points to 15
    num_spanwise_cp_target = 15
    for idx in list(geometry.functions.keys()):
        function = geometry.functions[idx]
        coeffs = function.coefficients.value if hasattr(function.coefficients, 'value') else function.coefficients
        axis0_y_range = np.ptp([coeffs[i, :, 1].mean() for i in range(coeffs.shape[0])])
        axis1_y_range = np.ptp([coeffs[:, j, 1].mean() for j in range(coeffs.shape[1])])
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
        new_space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=new_degree, coefficients_shape=new_shape)
        geometry.functions[idx] = lfs.Function(space=new_space, coefficients=new_coeffs)
        geometry.space.spaces[idx] = new_space

    # Projections
    leading_edge_left = geometry.project(np.array([0.0, -5.0 * scale_factor, 0.0]))
    leading_edge_right = geometry.project(np.array([0.0, 5.0 * scale_factor, 0.0]))

    chord_station_y = np.linspace(0.0, 5.0 * scale_factor, num_chord_stations)
    chord_le_projections = [geometry.project(np.array([0.0, y, 0.0])) for y in chord_station_y]
    chord_te_projections = [geometry.project(np.array([1.0 * scale_factor, y, 0.0])) for y in chord_station_y]
    quarter_chord_projections = [geometry.project(np.array([0.25 * scale_factor, y, 0.0])) for y in chord_station_y]
    upper_thickness_projections = [geometry.project(np.array([0.3 * scale_factor, y, 0.05 * scale_factor]), direction=np.array([0, 0, -1])) for y in chord_station_y]
    lower_thickness_projections = [geometry.project(np.array([0.3 * scale_factor, y, -0.05 * scale_factor]), direction=np.array([0, 0, 1])) for y in chord_station_y]

    nx_area, ny_area = 21, 41
    x_grid = np.linspace(0.0, 1.0 * scale_factor, nx_area)
    y_grid = np.linspace(-5.0 * scale_factor, 5.0 * scale_factor, ny_area)
    X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid, indexing='ij')
    upper_seed_pts = np.column_stack([X_mesh.ravel(), Y_mesh.ravel(), np.full(X_mesh.size, 0.05 * scale_factor)])
    lower_seed_pts = np.column_stack([X_mesh.ravel(), Y_mesh.ravel(), np.full(X_mesh.size, -0.05 * scale_factor)])
    projected_upper_skin = geometry.project(upper_seed_pts, force_reprojection=False, direction=np.array([0, 0, -1]), plot=False)
    projected_lower_skin = geometry.project(lower_seed_pts, force_reprojection=False, direction=np.array([0, 0, 1]), plot=False)

    num_ffd_coefficients_chordwise = 5 if include_camber else 2
    ffd_degree_chordwise = 2 if include_camber else 1
    ffd_block = construct_ffd_block_around_entities(
        entities=geometry,
        num_coefficients=(num_ffd_coefficients_chordwise, num_ffd_sections, 2),
        degree=(ffd_degree_chordwise, 3, 1)
    )
    ffd_sectional_parameterization = SectionalParameterization(
        name='ffd_param',
        parameterized_points=ffd_block.coefficients,
        principal_parametric_dimension=1
    )

    # Initialize variables directly with optimal values
    taper_dvs = csdl.Variable(shape=(num_chord_stations - 1,), value=parsed_dvs['taper_dvs'])
    aspect_ratio = csdl.Variable(shape=(1,), value=parsed_dvs['aspect_ratio'])
    sweep_angle_dvs = csdl.Variable(shape=(num_chord_stations - 1,), value=parsed_dvs['sweep_angle_dvs'])
    twist_dvs = csdl.Variable(shape=(num_chord_stations,), value=parsed_dvs['twist_dvs'])
    if include_camber:
        camber_dvs = csdl.Variable(shape=(3, num_chord_stations), value=parsed_dvs['camber_dvs'])

    chord_stretch_states = csdl.Variable(shape=(num_chord_stations,), value=np.zeros(num_chord_stations))
    thickness_stretch_states = csdl.Variable(shape=(num_chord_stations,), value=np.zeros(num_chord_stations))
    span_stretch_state = csdl.Variable(value=0.0)
    sweep_translation_states = csdl.Variable(shape=(num_chord_stations - 1,), value=np.zeros(num_chord_stations - 1))
    sweep_full_half = csdl.concatenate([csdl.Variable(value=0.0), sweep_translation_states])

    chord_params = csdl.concatenate([chord_stretch_states[i] for i in range(num_chord_stations - 1, 0, -1)] + [chord_stretch_states[i] for i in range(num_chord_stations)])
    sweep_params = csdl.concatenate([sweep_full_half[i] for i in range(num_chord_stations - 1, 0, -1)] + [sweep_full_half[i] for i in range(num_chord_stations)])
    thickness_params = csdl.concatenate([thickness_stretch_states[i] for i in range(num_chord_stations - 1, 0, -1)] + [thickness_stretch_states[i] for i in range(num_chord_stations)])
    twist_params = csdl.concatenate([twist_dvs[i] for i in range(num_chord_stations - 1, 0, -1)] + [twist_dvs[i] for i in range(num_chord_stations)])
    span_weights = np.linspace(-1.0, 1.0, num_ffd_sections)
    span_params = csdl.expand(span_stretch_state, (num_ffd_sections,)) * span_weights

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
            row = csdl.concatenate([camber_dvs[c, i] for i in range(num_chord_stations - 1, 0, -1)] + [camber_dvs[c, i] for i in range(num_chord_stations)])
            full_span_camber_list.append(csdl.reshape(row, (1, num_ffd_sections)))
        full_span_camber = csdl.concatenate(full_span_camber_list, axis=0)

        # Convert chord percentage to physical vertical displacement for each section
        camber_displacement = (full_span_camber / 100.0) * csdl.expand(section_chords, (3, num_ffd_sections), 'j->ij')
        camber_delta = csdl.expand(camber_displacement, (3, num_ffd_sections, 2), 'ij->ijk')
        ffd_coefficients = ffd_coefficients.set(csdl.slice[1:4, :, :, 2], ffd_coefficients[1:4, :, :, 2] + camber_delta)

    geometry_coefficients = ffd_block.evaluate_ffd(coefficients=ffd_coefficients, plot=False)
    geometry.set_coefficients(geometry_coefficients)

    wingspan = geometry.evaluate(leading_edge_right)[1] - geometry.evaluate(leading_edge_left)[1]
    local_chords = [geometry.evaluate(chord_te_projections[i])[0] - geometry.evaluate(chord_le_projections[i])[0] for i in range(num_chord_stations)]
    local_thicknesses = [geometry.evaluate(upper_thickness_projections[i])[2] - geometry.evaluate(lower_thickness_projections[i])[2] for i in range(num_chord_stations)]

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

    geometry_solver = ParameterizationSolver()
    geometry_solver.add_state(chord_stretch_states)
    geometry_solver.add_state(thickness_stretch_states)
    geometry_solver.add_state(span_stretch_state)
    geometry_solver.add_state(sweep_translation_states)

    geometric_variables = GeometricVariables()
    for i in range(1, num_chord_stations):
        normalized_chord = local_chords[i] / local_chords[0]
        geometric_variables.add_variable(normalized_chord, taper_dvs[i - 1], penalty_value=None)
    for i in range(num_chord_stations):
        tc_ratio = local_thicknesses[i] / local_chords[i]
        geometric_variables.add_variable(tc_ratio / 0.12, 1.0, penalty_value=None)
    geometric_variables.add_variable(planform_area, 1.0, penalty_value=None)
    geometric_variables.add_variable(aspect_ratio_calc / 10.0, aspect_ratio / 10.0, penalty_value=None)

    qc_pts = [geometry.evaluate(quarter_chord_projections[i]) for i in range(num_chord_stations)]
    for i in range(num_chord_stations - 1):
        dx = qc_pts[i + 1][0] - qc_pts[i][0]
        dy = qc_pts[i + 1][1] - qc_pts[i][1]
        sectional_sweep = csdl.arctan(dx / dy)
        geometric_variables.add_variable(sectional_sweep, sweep_angle_dvs[i], penalty_value=None)

    print("Executing ParameterizationSolver (in-line Newton solver)...")
    geometry_solver.evaluate(geometric_variables)

    total_wingspan = float(wingspan.value[0])
    half_span = total_wingspan / 2.0
    s_ref = float(planform_area.value[0])
    ar_final = float(aspect_ratio_calc.value[0])

    print(f"Deformed Geometry Evaluated:")
    print(f"  Wingspan (b)     = {total_wingspan:.4f} m (Half-Span = {half_span:.4f} m)")
    print(f"  Planform Area (S)= {s_ref:.4f} m^2")
    print(f"  Aspect Ratio (AR)= {ar_final:.2f}")

    return geometry, total_wingspan, half_span


def extract_station_airfoils(geometry, half_span: float, num_stations: int = 8):
    """
    Extracts airfoil cross-sections at num_stations along the right half-span
    using direct B-spline function evaluation at high chordwise resolution.

    Instead of slicing coarse PyVista meshes, this evaluates the upper (function 1)
    and lower (function 0) B-spline surfaces directly in parametric space at 300
    chordwise points per station, producing smooth, exact profiles.
    """
    print(f"\nExtracting airfoil cross-sections at {num_stations} spanwise stations "
          f"(y in [0.0, {half_span:.3f}] m) via direct B-spline evaluation...")

    # Identify upper and lower surface functions for the right half-span (y >= 0).
    # After geometry import and interpolation:
    #   Function 0: lower surface (z < 0), spanwise axis = u, chordwise axis = v
    #   Function 1: upper surface (z > 0), spanwise axis = u, chordwise axis = v
    # Functions 2,3 are tiny tip caps; Functions 4-7 are the left half.
    func_lower = geometry.functions[0]
    func_upper = geometry.functions[1]

    # Verify which parametric axis is spanwise by checking coefficient y-extent
    for label, func in [("Lower (fn 0)", func_lower), ("Upper (fn 1)", func_upper)]:
        coeffs = func.coefficients.value if hasattr(func.coefficients, 'value') else func.coefficients
        axis0_y_range = np.ptp([coeffs[i, :, 1].mean() for i in range(coeffs.shape[0])])
        axis1_y_range = np.ptp([coeffs[:, j, 1].mean() for j in range(coeffs.shape[1])])
        spanwise_axis = 0 if axis0_y_range >= axis1_y_range else 1
        print(f"  {label}: shape={coeffs.shape}, spanwise_axis={spanwise_axis} "
              f"(axis0_y_range={axis0_y_range:.4f}, axis1_y_range={axis1_y_range:.4f})")
        if spanwise_axis != 0:
            print(f"  WARNING: Expected spanwise_axis=0 for {label}, got {spanwise_axis}")

    # Number of chordwise evaluation points (high resolution for smooth profiles)
    n_chord_pts = 300

    # Station locations: root (eta=0.0) to near-tip (eta=0.995)
    stations_eta = np.linspace(0.0, 0.995, num_stations)
    stations_y = stations_eta * half_span

    station_data = []

    for k, (eta, y_val) in enumerate(zip(stations_eta, stations_y)):
        # Parametric u coordinate for this spanwise station.
        # u ∈ [0, 1] maps to y ∈ [0, half_span] (linearly for the interpolated CPs).
        u_param = np.clip(eta, 0.0, 0.999)

        # Sweep v from 0 to 1 at high resolution for chordwise profile
        v_vals = np.linspace(0.0, 1.0, n_chord_pts)
        u_vals = np.full_like(v_vals, u_param)
        param_coords = np.column_stack([u_vals, v_vals])  # (n_chord_pts, 2)

        # Evaluate upper and lower surfaces
        pts_lower = func_lower.evaluate(param_coords, non_csdl=True).reshape(-1, 3)
        pts_upper = func_upper.evaluate(param_coords, non_csdl=True).reshape(-1, 3)

        x_lo_raw = pts_lower[:, 0]
        z_lo_raw = pts_lower[:, 2]
        x_up_raw = pts_upper[:, 0]
        z_up_raw = pts_upper[:, 2]

        # Find leading and trailing edges from combined point set
        all_x = np.concatenate([x_lo_raw, x_up_raw])
        all_z = np.concatenate([z_lo_raw, z_up_raw])
        le_idx = np.argmin(all_x)
        te_idx = np.argmax(all_x)
        x_le = float(all_x[le_idx])
        z_le = float(all_z[le_idx])
        x_te = float(all_x[te_idx])
        z_te = float(all_z[te_idx])
        chord = x_te - x_le

        # Sort each surface by x and remove duplicate x values
        sort_lo = np.argsort(x_lo_raw)
        x_lo_sorted, z_lo_sorted = x_lo_raw[sort_lo], z_lo_raw[sort_lo]
        sort_up = np.argsort(x_up_raw)
        x_up_sorted, z_up_sorted = x_up_raw[sort_up], z_up_raw[sort_up]

        _, uniq_lo = np.unique(x_lo_sorted, return_index=True)
        _, uniq_up = np.unique(x_up_sorted, return_index=True)
        x_lo_sorted, z_lo_sorted = x_lo_sorted[uniq_lo], z_lo_sorted[uniq_lo]
        x_up_sorted, z_up_sorted = x_up_sorted[uniq_up], z_up_sorted[uniq_up]

        # Resample on uniform chordwise grid
        x_grid = np.linspace(x_le, x_te, 300)

        if len(x_up_sorted) >= 2:
            f_up = si.interp1d(x_up_sorted, z_up_sorted, kind='cubic', fill_value='extrapolate')
            z_up_grid = f_up(x_grid)
        else:
            z_up_grid = np.full_like(x_grid, z_le)

        if len(x_lo_sorted) >= 2:
            f_lo = si.interp1d(x_lo_sorted, z_lo_sorted, kind='cubic', fill_value='extrapolate')
            z_lo_grid = f_lo(x_grid)
        else:
            z_lo_grid = np.full_like(x_grid, z_le)

        # Ensure upper >= lower (handles any crossover from interpolation)
        z_up_grid = np.maximum(z_up_grid, z_lo_grid)

        # Compute aerodynamic properties
        thickness_dist = z_up_grid - z_lo_grid
        t_max = float(np.max(thickness_dist))
        tc_ratio = (t_max / chord) * 100.0 if chord > 0 else 0.0

        slope = (z_te - z_le) / max(chord, 1e-8)
        chord_line = z_le + slope * (x_grid - x_le)
        camber_line = 0.5 * (z_up_grid + z_lo_grid)
        camber_dist = camber_line - chord_line
        max_camber = float(np.max(np.abs(camber_dist)))
        camber_ratio = (max_camber / chord) * 100.0 if chord > 0 else 0.0

        twist_deg = float(np.degrees(np.arctan2(z_le - z_te, x_te - x_le)))

        station_dict = {
            'station_idx': k + 1,
            'eta': eta,
            'y': y_val,
            'x_le': x_le,
            'z_le': z_le,
            'x_te': x_te,
            'z_te': z_te,
            'chord': chord,
            't_max': t_max,
            'tc_ratio': tc_ratio,
            'max_camber': max_camber,
            'camber_ratio': camber_ratio,
            'twist_deg': twist_deg,
            'x_up': x_up_sorted,
            'z_up': z_up_sorted,
            'x_lo': x_lo_sorted,
            'z_lo': z_lo_sorted,
            'x_grid': x_grid,
            'z_up_grid': z_up_grid,
            'z_lo_grid': z_lo_grid,
            'camber_line': camber_line,
            'chord_line': chord_line,
        }
        station_data.append(station_dict)

        print(f"  Station {k+1:2d}: eta = {eta:.3f}, y = {y_val:.3f} m | Chord = {chord:.4f} m, "
              f"t_max = {t_max*1e3:5.2f} mm, t/c = {tc_ratio:5.2f}%, twist = {twist_deg:+5.2f}°, "
              f"camber = {max_camber*1e3:4.2f} mm ({camber_ratio:4.2f}%)")

    return station_data


def generate_airfoil_gallery_plot(station_data: list, output_path: str, half_span: float):
    """
    Figure 1: 4x2 Grid of individual stations with detailed annotations.
    """
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(4, 2, figsize=(15, 14), dpi=200)
    axes = axes.flatten()

    c_upper = '#1f77b4'
    c_lower = '#2ca02c'
    c_camber = '#d62728'
    c_chord = '#7f7f7f'

    for k, data in enumerate(station_data):
        ax = axes[k]
        x_g = data['x_grid']
        z_u = data['z_up_grid']
        z_l = data['z_lo_grid']
        z_cam = data['camber_line']
        z_ch = data['chord_line']

        # Fill profile
        ax.fill_between(x_g, z_l, z_u, color='#e6f2ff', alpha=0.7, label='Airfoil Section')
        ax.plot(x_g, z_u, '-', color=c_upper, linewidth=2.0, label='Upper Surface')
        ax.plot(x_g, z_l, '-', color=c_lower, linewidth=2.0, label='Lower Surface')
        ax.plot(x_g, z_cam, '--', color=c_camber, linewidth=1.5, label='Mean Camber Line')
        ax.plot(x_g, z_ch, ':', color=c_chord, linewidth=1.2, label='Chord Line')

        # Internal spar box representation (25% to 75% chord)
        x_spar_front = data['x_le'] + 0.25 * data['chord']
        x_spar_rear = data['x_le'] + 0.75 * data['chord']
        ax.axvline(x_spar_front, color='#ff7f0e', linestyle=':', alpha=0.5, label='Front/Rear Spar (25%/75%)' if k==0 else None)
        ax.axvline(x_spar_rear, color='#ff7f0e', linestyle=':', alpha=0.5)

        # Mark LE and TE
        ax.plot(data['x_le'], data['z_le'], 'ko', markersize=5)
        ax.plot(data['x_te'], data['z_te'], 'ks', markersize=5)

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("x [m]", fontsize=10, fontweight='bold')
        ax.set_ylabel("z [m]", fontsize=10, fontweight='bold')

        title_str = f"Station {data['station_idx']}: $\\eta = {data['eta']:.3f}$ ($y = {data['y']:.3f}$ m)"
        ax.set_title(title_str, fontsize=11, fontweight='bold', pad=6)

        # Metric annotation text box
        info_box = (
            f"Chord: {data['chord']:.3f} m\n"
            f"$t_{{max}}$: {data['t_max']*1e3:.1f} mm\n"
            f"$t/c$: {data['tc_ratio']:.1f}%\n"
            f"Twist: {data['twist_deg']:+.2f}°\n"
            f"Camber: {data['max_camber']*1e3:.1f} mm ({data['camber_ratio']:.1f}%)"
        )
        ax.text(
            0.97, 0.05, info_box,
            transform=ax.transAxes,
            verticalalignment='bottom',
            horizontalalignment='right',
            fontsize=8.5,
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.88, edgecolor='#cccccc')
        )
        ax.grid(True, linestyle=':', alpha=0.6)
        if k == 0:
            ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=7.5)

    fig.suptitle(
        f"Optimized Airfoil Cross-Sections Across 8 Spanwise Stations\n"
        f"BWB Aerostructural Optimization | Half-Span $b/2 = {half_span:.3f}$ m",
        fontsize=14, fontweight='bold', y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved Airfoil Gallery Plot to: {output_path}")


def generate_shape_evolution_plot(station_data: list, output_path: str, half_span: float):
    """
    Figure 2: Multi-panel comparative analysis:
    (a) Planform cutline map (x-y plane)
    (b) Leading-edge aligned physical cross-section overlay
    (c) Normalized airfoil profile comparison (x/c vs z/c)
    (d) Spanwise distribution metrics (c, t_max, t/c, twist, camber)
    """
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig = plt.figure(figsize=(16, 12), dpi=200)
    gs = GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.22)

    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(station_data)))

    # -------------------------------------------------------------------------
    # Panel (a): Planform View with Station Cutlines
    # -------------------------------------------------------------------------
    ax_plan = fig.add_subplot(gs[0, 0])
    y_stations = [d['y'] for d in station_data]
    x_les = [d['x_le'] for d in station_data]
    x_tes = [d['x_te'] for d in station_data]

    # Smooth planform edges
    y_dense = np.linspace(0.0, half_span, 200)
    spl_le = si.CubicSpline(y_stations, x_les, bc_type='natural')
    spl_te = si.CubicSpline(y_stations, x_tes, bc_type='natural')
    x_le_dense = spl_le(y_dense)
    x_te_dense = spl_te(y_dense)

    ax_plan.fill_betweenx(y_dense, x_le_dense, x_te_dense, color='#e9ecef', alpha=0.8, label='Right Wing Planform')
    ax_plan.plot(x_le_dense, y_dense, 'k-', linewidth=1.8, label='Leading Edge')
    ax_plan.plot(x_te_dense, y_dense, 'k--', linewidth=1.5, label='Trailing Edge')

    # Draw 25% chord spar axis
    x_spar_dense = x_le_dense + 0.25 * (x_te_dense - x_le_dense)
    ax_plan.plot(x_spar_dense, y_dense, '-.', color='#d95f02', linewidth=1.5, label='Internal Spar (25% c)')

    # Draw station cutting lines
    for k, (data, col) in enumerate(zip(station_data, colors)):
        ax_plan.plot([data['x_le'], data['x_te']], [data['y'], data['y']], '-', color=col, linewidth=2.2)
        ax_plan.text(data['x_te'] + 0.02, data['y'], f"Stn {k+1} ($\\eta={data['eta']:.2f}$)",
                     color=col, va='center', fontsize=8, fontweight='bold')

    ax_plan.set_xlabel("Chordwise Position x [m]", fontsize=10, fontweight='bold')
    ax_plan.set_ylabel("Spanwise Position y [m]", fontsize=10, fontweight='bold')
    ax_plan.set_title("(a) Wing Planform & Slicing Station Locations", fontsize=11, fontweight='bold')
    ax_plan.set_aspect('equal')
    ax_plan.grid(True, linestyle=':', alpha=0.6)
    ax_plan.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9, fontsize=8)

    # -------------------------------------------------------------------------
    # Panel (b): Leading-Edge Aligned Physical Cross Sections
    # -------------------------------------------------------------------------
    ax_align = fig.add_subplot(gs[0, 1])
    for k, (data, col) in enumerate(zip(station_data, colors)):
        x_rel = data['x_grid'] - data['x_le']
        ax_align.plot(x_rel, data['z_up_grid'], '-', color=col, linewidth=1.8, label=f"Stn {k+1} ($\\eta={data['eta']:.2f}$)")
        ax_align.plot(x_rel, data['z_lo_grid'], '-', color=col, linewidth=1.8)

    ax_align.set_xlabel("Relative Chordwise Position $(x - x_{LE})$ [m]", fontsize=10, fontweight='bold')
    ax_align.set_ylabel("Vertical Coordinate z [m]", fontsize=10, fontweight='bold')
    ax_align.set_title("(b) Leading-Edge Aligned Physical Cross Sections (Taper & Scale)", fontsize=11, fontweight='bold')
    ax_align.set_aspect('equal')
    ax_align.grid(True, linestyle=':', alpha=0.6)
    ax_align.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=8)

    # -------------------------------------------------------------------------
    # Panel (c): Normalized Airfoil Profile Shapes (x/c vs z/c)
    # -------------------------------------------------------------------------
    ax_norm = fig.add_subplot(gs[1, 0])
    for k, (data, col) in enumerate(zip(station_data, colors)):
        c = max(data['chord'], 1e-6)
        x_norm = (data['x_grid'] - data['x_le']) / c
        z_up_norm = (data['z_up_grid'] - data['z_le']) / c
        z_lo_norm = (data['z_lo_grid'] - data['z_le']) / c
        z_cam_norm = (data['camber_line'] - data['z_le']) / c

        ax_norm.plot(x_norm, z_up_norm, '-', color=col, linewidth=1.8, label=f"Stn {k+1} ($\\eta={data['eta']:.2f}$)")
        ax_norm.plot(x_norm, z_lo_norm, '-', color=col, linewidth=1.8)
        ax_norm.plot(x_norm, z_cam_norm, ':', color=col, alpha=0.7, linewidth=1.0)

    ax_norm.set_xlabel("Normalized Chord $x/c$", fontsize=10, fontweight='bold')
    ax_norm.set_ylabel("Normalized Thickness & Camber $z/c$", fontsize=10, fontweight='bold')
    ax_norm.set_title("(c) Normalized Airfoil Profiles (Intrinsic Aerodynamic Shapes)", fontsize=11, fontweight='bold')
    ax_norm.set_aspect('equal')
    ax_norm.grid(True, linestyle=':', alpha=0.6)
    ax_norm.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=8)

    # -------------------------------------------------------------------------
    # Panel (d) & (e): Spanwise Distributions in bottom-right quadrant
    # -------------------------------------------------------------------------
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    sub_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1], hspace=0.35)
    ax_metric1 = fig.add_subplot(sub_gs[0, 0])
    ax_metric2 = fig.add_subplot(sub_gs[1, 0])

    etas = [d['eta'] for d in station_data]
    chords = [d['chord'] for d in station_data]
    tcs = [d['tc_ratio'] for d in station_data]
    twists = [d['twist_deg'] for d in station_data]
    cambers = [d['camber_ratio'] for d in station_data]

    # Subplot 1: Chord and Twist
    ax_twist = ax_metric1.twinx()
    l1 = ax_metric1.plot(etas, chords, 'o-', color='#d62728', linewidth=2.0, markersize=5, label='Chord $c$ [m]')
    l2 = ax_twist.plot(etas, twists, 's--', color='#9467bd', linewidth=1.8, markersize=5, label='Twist [deg]')
    ax_metric1.set_ylabel('Chord $c$ [m]', color='#d62728', fontweight='bold', fontsize=9)
    ax_twist.set_ylabel('Twist [deg]', color='#9467bd', fontweight='bold', fontsize=9)
    ax_metric1.tick_params(axis='y', labelcolor='#d62728')
    ax_twist.tick_params(axis='y', labelcolor='#9467bd')
    ax_metric1.set_title('(d) Planform & Twist Distribution', fontsize=10, fontweight='bold', pad=4)
    lines_top = l1 + l2
    ax_metric1.legend(lines_top, [l.get_label() for l in lines_top], loc='upper right', fontsize=8, frameon=True, framealpha=0.85)
    ax_metric1.grid(True, linestyle=':', alpha=0.6)

    # Subplot 2: t/c ratio and Camber ratio
    ax_cam = ax_metric2.twinx()
    l3 = ax_metric2.plot(etas, tcs, 'o-', color='#1f77b4', linewidth=2.0, markersize=5, label='$t/c$ [%]')
    l4 = ax_cam.plot(etas, cambers, '^--', color='#2ca02c', linewidth=1.8, markersize=5, label='Camber [% $c$]')
    ax_metric2.set_xlabel('Normalized Spanwise Coordinate $\\eta = y/(b/2)$', fontsize=9, fontweight='bold')
    ax_metric2.set_ylabel('$t/c$ [%]', color='#1f77b4', fontweight='bold', fontsize=9)
    ax_cam.set_ylabel('Camber [% $c$]', color='#2ca02c', fontweight='bold', fontsize=9)
    ax_metric2.tick_params(axis='y', labelcolor='#1f77b4')
    ax_cam.tick_params(axis='y', labelcolor='#2ca02c')
    ax_metric2.set_title('(e) Thickness Ratio & Camber Distribution', fontsize=10, fontweight='bold', pad=4)
    lines_bot = l3 + l4
    ax_metric2.legend(lines_bot, [l.get_label() for l in lines_bot], loc='upper right', fontsize=8, frameon=True, framealpha=0.85)
    ax_metric2.grid(True, linestyle=':', alpha=0.6)

    fig.suptitle(
        f"Spanwise Airfoil Shape Transition & Geometric Parameter Evolution\n"
        f"Half-Span $b/2 = {half_span:.3f}$ m",
        fontsize=14, fontweight='bold', y=0.99
    )
    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92, hspace=0.30, wspace=0.25)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved Shape Evolution Plot to: {output_path}")


def save_airfoil_telemetry(station_data: list, output_npz_path: str, half_span: float):
    """Saves extracted station cross-sections and telemetry to an npz file."""
    export_dict = {
        'half_span': half_span,
        'stations_eta': np.array([d['eta'] for d in station_data]),
        'stations_y': np.array([d['y'] for d in station_data]),
        'chords': np.array([d['chord'] for d in station_data]),
        't_max': np.array([d['t_max'] for d in station_data]),
        'tc_ratio': np.array([d['tc_ratio'] for d in station_data]),
        'max_camber': np.array([d['max_camber'] for d in station_data]),
        'camber_ratio': np.array([d['camber_ratio'] for d in station_data]),
        'twist_deg': np.array([d['twist_deg'] for d in station_data]),
    }
    for k, d in enumerate(station_data):
        export_dict[f"station_{k+1}_x_grid"] = d['x_grid']
        export_dict[f"station_{k+1}_z_upper"] = d['z_up_grid']
        export_dict[f"station_{k+1}_z_lower"] = d['z_lo_grid']
        export_dict[f"station_{k+1}_camber_line"] = d['camber_line']

    np.savez_compressed(output_npz_path, **export_dict)
    print(f"Saved numerical cross-section telemetry to: {output_npz_path}")


def main():
    # Determine repo root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "../../../.."))

    # Determine target run output folder
    if len(sys.argv) > 1:
        target_dir = os.path.abspath(sys.argv[1])
    else:
        base_output_dir = os.path.join(repo_root, "rectangular_wing_to_bwb_aerostructural_optimization_outputs")
        target_dir = find_latest_output_dir(base_output_dir)

    print(f"================================================================================")
    print(f"Airfoil Cross-Section Extraction Analysis")
    print(f"Target Output Directory: {target_dir}")
    print(f"================================================================================")

    x_out_file = os.path.join(target_dir, "x.out")
    if not os.path.exists(x_out_file):
        raise FileNotFoundError(f"Optimization design variable file not found: {x_out_file}")

    x_history = np.loadtxt(x_out_file)
    if x_history.ndim == 1:
        x_opt = x_history
        num_iters = 1
    else:
        x_opt = x_history[-1]
        num_iters = x_history.shape[0]

    print(f"Loaded {num_iters} iterations. Extracting final optimal design vector.")

    # Parse DVs
    parsed_dvs = parse_design_variables(x_opt)

    # Reconstruct authentic geometry
    geometry, total_wingspan, half_span = build_and_evaluate_geometry(parsed_dvs, repo_root)

    # Extract 8 cross-section stations along half-span
    station_data = extract_station_airfoils(geometry, half_span, num_stations=8)

    # Generate Output Figures and Data
    gallery_fig_path = os.path.join(target_dir, "airfoil_cross_sections_gallery.png")
    evolution_fig_path = os.path.join(target_dir, "airfoil_shape_evolution.png")
    telemetry_path = os.path.join(target_dir, "airfoil_cross_sections_data.npz")

    generate_airfoil_gallery_plot(station_data, gallery_fig_path, half_span)
    generate_shape_evolution_plot(station_data, evolution_fig_path, half_span)
    save_airfoil_telemetry(station_data, telemetry_path, half_span)

    # Also copy artifacts to the active agent brain directory if available
    artifact_dir = "/home/andrew/.gemini/antigravity/brain/5d53fe94-f3d4-4c8e-aec8-721ba87a6dbf"
    if os.path.exists(artifact_dir):
        shutil.copy2(gallery_fig_path, os.path.join(artifact_dir, "airfoil_cross_sections_gallery.png"))
        shutil.copy2(evolution_fig_path, os.path.join(artifact_dir, "airfoil_shape_evolution.png"))
        print(f"Artifacts successfully copied to brain directory: {artifact_dir}")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY: AIRFOIL CROSS SECTIONS ALONG 8 SPANWISE STATIONS")
    print("=" * 80)
    print(f"{'Stn':4s} | {'eta':6s} | {'y [m]':8s} | {'Chord [m]':10s} | {'t_max [mm]':11s} | {'t/c [%]':8s} | {'Twist [deg]':12s} | {'Camber [mm]':12s}")
    print("-" * 80)
    for d in station_data:
        print(f"{d['station_idx']:4d} | {d['eta']:6.3f} | {d['y']:8.3f} | {d['chord']:10.4f} | {d['t_max']*1e3:11.2f} | {d['tc_ratio']:8.2f} | {d['twist_deg']:+12.2f} | {d['max_camber']*1e3:12.2f}")
    print("=" * 80)
    print("Analysis complete successfully!")


if __name__ == "__main__":
    main()
