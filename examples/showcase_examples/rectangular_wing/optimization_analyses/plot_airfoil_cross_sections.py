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
import argparse
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
    GeometricVariables,
    ParameterizationSolver,
)

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from physics_models.ar_area_bspline import BsplineTargetRegularization


@dataclass
class DVInfo:
    variable: csdl.Variable
    lower: Union[float, npt.NDArray[np.float64]]
    upper: Union[float, npt.NDArray[np.float64]]
    scaler: float = 1.0


def find_latest_output_dir(base_dir: str) -> str:
    """Find the latest completed (or active) output subdirectory in base_dir."""
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Base output directory not found: {base_dir}")
    folders = [os.path.join(base_dir, d) for d in os.listdir(base_dir)
               if os.path.isdir(os.path.join(base_dir, d))]
    if not folders:
        raise FileNotFoundError(f"No run subdirectories found in: {base_dir}")
    
    # Prefer completed runs containing modopt_results.out
    completed = [d for d in folders if os.path.exists(os.path.join(d, 'modopt_results.out'))]
    if completed:
        latest = max(completed, key=os.path.getmtime)
    else:
        # Fallback to any directory with x.out
        with_x = [d for d in folders if os.path.exists(os.path.join(d, 'x.out'))]
        latest = max(with_x if with_x else folders, key=os.path.getmtime)
    return os.path.abspath(latest)


def detect_scale_factor(output_dir: str, repo_root: str) -> float:
    """
    Detects the scale factor used in the optimization run:
    1. Check structural_data.npz if present in output_dir.
    2. Check lift_and_moment_data.npz to infer from actual half-span.
    3. Read scale_factor definition from ex_rectangular_wing_to_bwb.py.
    """
    # 1. Check structural_data.npz
    struct_npz = os.path.join(output_dir, 'structural_data.npz')
    if os.path.exists(struct_npz):
        try:
            data = np.load(struct_npz, allow_pickle=True)
            if 'scale_factor' in data:
                sf = float(data['scale_factor'])
                print(f"  Detected scale_factor = {sf} from structural_data.npz")
                return sf
        except Exception:
            pass

    # 2. Check lift_and_moment_data.npz
    lm_npz = os.path.join(output_dir, 'lift_and_moment_data.npz')
    if os.path.exists(lm_npz):
        try:
            data = np.load(lm_npz, allow_pickle=True)
            pc = data['panel_centers_right']
            y_max = float(np.max(pc[:, 1]))
            if y_max > 15.0:
                sf = 7.5
            elif y_max > 2.5:
                sf = 1.0
            else:
                sf = 1.0 / np.sqrt(10.0)
            print(f"  Inferred scale_factor = {sf} from panel telemetry (half-span = {y_max:.2f} m)")
            return sf
        except Exception:
            pass

    # 3. Read from ex_rectangular_wing_to_bwb.py
    run_script = os.path.join(repo_root, 'examples/showcase_examples/rectangular_wing/ex_rectangular_wing_to_bwb.py')
    if os.path.exists(run_script):
        try:
            with open(run_script, 'r') as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith('scale_factor =') or stripped.startswith('scale_factor='):
                        val_str = stripped.split('=')[1].split('#')[0].strip()
                        sf = float(eval(val_str))
                        print(f"  Read scale_factor = {sf} from ex_rectangular_wing_to_bwb.py")
                        return sf
        except Exception:
            pass

    return 7.5


def parse_design_variables(x_opt: np.ndarray, scale_factor: float = 7.5, target_dir: str = None, formulation_override: str = None, resolution_override: str = None):
    """
    Auto-detect configuration and parse unscaled physical design variable values.
    Supports both 'ar_area' and 'chord_span' formulations across 'fast' (5-station)
    and 'full' (8-station) resolutions, with or without camber and elevator.
    """
    n_dv = len(x_opt)
    print(f"\nAnalyzing design variable vector of length {n_dv}...")

    formulation = formulation_override
    resolution = resolution_override
    include_camber = None
    include_elevator = None
    include_thickness_shape = None
    include_te_thickness = None

    # 1. Check saved metadata in output folder
    dv_names = None
    if target_dir is not None:
        for fn in ['lift_and_moment_data.npz', 'structural_data.npz']:
            fpath = os.path.join(target_dir, fn)
            if os.path.exists(fpath):
                try:
                    data = np.load(fpath, allow_pickle=True)
                    if dv_names is None and 'dv_names' in data:
                        dv_names = [str(x) for x in data['dv_names']]
                    if formulation is None and 'formulation' in data:
                        formulation = str(data['formulation'])
                    if resolution is None and 'resolution' in data:
                        resolution = str(data['resolution'])
                    if include_camber is None and 'include_camber' in data:
                        include_camber = bool(data['include_camber'])
                    if include_elevator is None and 'include_elevator' in data:
                        include_elevator = bool(data['include_elevator'])
                    if include_thickness_shape is None and 'include_thickness_shape' in data:
                        include_thickness_shape = bool(data['include_thickness_shape'])
                    if include_te_thickness is None and 'include_te_thickness' in data:
                        include_te_thickness = bool(data['include_te_thickness'])
                except Exception:
                    pass

    # 2. Dynamic parsing if dv_names metadata is available
    if dv_names is not None:
        if include_thickness_shape is None:
            include_thickness_shape = 'thickness_shape_dvs' in dv_names
        if include_te_thickness is None:
            include_te_thickness = False

        if formulation is None:
            if any(k in dv_names for k in ['chord_stretch_dvs', 'span_stretch_dv', 'thickness_stretch_dvs']):
                formulation = 'chord_span'
            else:
                formulation = 'ar_area'

        if resolution is None:
            for res_candidate, n_stn in [('fast', 5), ('full', 8)]:
                expected_len = 0
                for name in dv_names:
                    if name in ['chord_stretch_dvs', 'sweep_dvs', 'thickness_stretch_dvs', 'twist_dvs', 'ttop_dvs', 'tweb_dvs', 'tc_target_control_points']:
                        expected_len += n_stn
                    elif name in ['taper_dvs', 'taper_control_points', 'sweep_angle_dvs', 'sweep_angle_control_points']:
                        expected_len += n_stn - 1
                    elif name == 'camber_dvs':
                        expected_len += 3 * n_stn
                    elif name == 'thickness_shape_dvs':
                        n_rows_th = 5 if include_te_thickness else 4
                        expected_len += n_rows_th * n_stn
                    else:
                        expected_len += 1
                if expected_len == n_dv:
                    resolution = res_candidate
                    break
            if resolution is None:
                resolution = 'fast' if n_dv <= 35 else 'full'

        num_stations = 5 if resolution == 'fast' else 8
        num_chord_stations = num_stations
        initial_chord = 1.0 * scale_factor
        initial_thickness = 0.12 * initial_chord
        camber_max_percent = 5.0

        curr = 0
        dv_dict = {}
        for name in dv_names:
            if name in ['chord_stretch_dvs', 'sweep_dvs']:
                dv_dict[name] = x_opt[curr : curr + num_chord_stations] * scale_factor
                curr += num_chord_stations
            elif name == 'thickness_stretch_dvs':
                dv_dict[name] = x_opt[curr : curr + num_chord_stations] * initial_thickness
                curr += num_chord_stations
            elif name == 'twist_dvs':
                dv_dict[name] = x_opt[curr : curr + num_chord_stations] / 10.0
                curr += num_chord_stations
            elif name == 'span_stretch_dv':
                dv_dict[name] = x_opt[curr : curr + 1] * scale_factor
                curr += 1
            elif name in ['pitch', 'pitch_ss', 'pitch_neg1g', 'payload_cg']:
                dv_dict[name] = x_opt[curr : curr + 1] / 10.0
                curr += 1
            elif name == 'payload_center_x':
                payload_cx_scaler = 1.0 / scale_factor
                dv_dict[name] = x_opt[curr : curr + 1] / payload_cx_scaler
                curr += 1
            elif name in ['ttop_dvs', 'tweb_dvs']:
                dv_dict[name] = x_opt[curr : curr + num_stations] / 5000.0
                curr += num_stations
            elif name == 'elevator_angle':
                dv_dict[name] = float(x_opt[curr : curr + 1] / 10.0)
                curr += 1
            elif name == 'camber_dvs':
                camber_scaler = 1.0 / camber_max_percent
                dv_dict[name] = (x_opt[curr : curr + 3 * num_chord_stations] / camber_scaler).reshape((3, num_chord_stations))
                curr += 3 * num_chord_stations
            elif name in ['taper_dvs', 'taper_control_points']:
                taper_scaler = 1.0 if name == 'taper_control_points' else 2.0
                dv_dict['taper_dvs'] = x_opt[curr : curr + num_chord_stations - 1] / taper_scaler
                curr += num_chord_stations - 1
            elif name == 'aspect_ratio':
                dv_dict[name] = x_opt[curr : curr + 1] / 0.5
                curr += 1
            elif name in ['sweep_angle_dvs', 'sweep_angle_control_points']:
                dv_dict['sweep_angle_dvs'] = x_opt[curr : curr + num_chord_stations - 1] / 10.0
                curr += num_chord_stations - 1
            elif name == 'tc_target_control_points':
                dv_dict[name] = x_opt[curr : curr + num_chord_stations] / 10.0
                curr += num_chord_stations
            elif name == 'planform_area_target':
                dv_dict[name] = x_opt[curr : curr + 1] * (10.0 * (scale_factor ** 2))
                curr += 1
            elif name == 'thickness_shape_dvs':
                n_rows_th = 5 if include_te_thickness else 4
                thick_shape_scaler = 1.0 / 5.0
                dv_dict[name] = (x_opt[curr : curr + n_rows_th * num_chord_stations] / thick_shape_scaler).reshape((n_rows_th, num_chord_stations))
                curr += n_rows_th * num_chord_stations
            else:
                print(f"  Warning: Unrecognized DV name '{name}', assuming size 1")
                dv_dict[name] = x_opt[curr : curr + 1]
                curr += 1

        if formulation == 'chord_span':
            include_camber = 'camber_dvs' in dv_dict
            include_elevator = 'elevator_angle' in dv_dict
            include_sweep = 'sweep_dvs' in dv_dict
            include_thickness_shape = 'thickness_shape_dvs' in dv_dict
            parsed = {
                'formulation': 'chord_span',
                'resolution': resolution,
                'include_camber': include_camber,
                'include_elevator': include_elevator,
                'include_sweep': include_sweep,
                'include_thickness_shape': include_thickness_shape,
                'include_te_thickness': include_te_thickness,
                'num_stations': num_stations,
                'scale_factor': scale_factor,
                'chord_stretch_dvs': dv_dict.get('chord_stretch_dvs', np.zeros(num_chord_stations)),
                'sweep_dvs': dv_dict.get('sweep_dvs', np.zeros(num_chord_stations)),
                'thickness_stretch_dvs': dv_dict.get('thickness_stretch_dvs', np.zeros(num_chord_stations)),
                'twist_dvs': dv_dict.get('twist_dvs', np.zeros(num_chord_stations)),
                'span_stretch_dv': dv_dict.get('span_stretch_dv', np.array([0.0])),
                'pitch': dv_dict.get('pitch', np.array([0.0])),
                'pitch_ss': dv_dict.get('pitch_ss', np.array([0.0])),
                'camber_dvs': dv_dict.get('camber_dvs', np.zeros((3, num_chord_stations))),
                'thickness_shape_dvs': dv_dict.get('thickness_shape_dvs', None),
                'elevator_angle': dv_dict.get('elevator_angle', 0.0),
            }
            print(f"  Detected Formulation : Chord & Span Stretch ('chord_span') [via dv_names metadata]")
            print(f"  Resolution           : {resolution} ({num_stations} design stations)")
            print(f"  Active DVs ({len(dv_names)}): {dv_names}")
            print(f"  Camber DVs Active    : {include_camber}")
            print(f"  Elevator Active      : {include_elevator}")
            print(f"  Thick Shape Active   : {include_thickness_shape}")
            print(f"  Chord Stretches (m)  : {np.round(parsed['chord_stretch_dvs'], 3)}")
            print(f"  Effective Chords (m) : {np.round(initial_chord + parsed['chord_stretch_dvs'], 3)}")
            print(f"  Twist Angles (deg)   : {np.round(np.degrees(parsed['twist_dvs']), 2)}")
            print(f"  Span Stretch (m)     : {float(parsed['span_stretch_dv'][0]):.3f}")
            return parsed
        else:
            include_camber = 'camber_dvs' in dv_dict
            include_elevator = 'elevator_angle' in dv_dict
            include_thickness_shape = 'thickness_shape_dvs' in dv_dict
            parsed = {
                'formulation': 'ar_area',
                'resolution': resolution,
                'include_camber': include_camber,
                'include_elevator': include_elevator,
                'include_thickness_shape': include_thickness_shape,
                'include_te_thickness': include_te_thickness,
                'num_stations': num_stations,
                'scale_factor': scale_factor,
                'taper_dvs': dv_dict.get('taper_dvs', np.ones(num_chord_stations - 1)),
                'aspect_ratio': dv_dict.get('aspect_ratio', np.array([10.0])),
                'sweep_angle_dvs': dv_dict.get('sweep_angle_dvs', np.zeros(num_chord_stations - 1)),
                'tc_target_control_points': dv_dict.get('tc_target_control_points', None),
                'planform_area_target': dv_dict.get('planform_area_target', None),
                'twist_dvs': dv_dict.get('twist_dvs', np.zeros(num_chord_stations)),
                'pitch': dv_dict.get('pitch', np.array([0.0])),
                'pitch_ss': dv_dict.get('pitch_ss', np.array([0.0])),
                'payload_cg': dv_dict.get('payload_cg', np.array([0.4])),
                'payload_center_x': dv_dict.get('payload_center_x', np.array([0.5 * scale_factor])),
                'ttop_dvs': dv_dict.get('ttop_dvs', np.full(num_stations, 0.001)),
                'tweb_dvs': dv_dict.get('tweb_dvs', np.full(num_stations, 0.001)),
                'camber_dvs': dv_dict.get('camber_dvs', np.zeros((3, num_chord_stations))),
                'thickness_shape_dvs': dv_dict.get('thickness_shape_dvs', None),
                'elevator_angle': dv_dict.get('elevator_angle', 0.0),
            }
            print(f"  Detected Formulation : Aspect Ratio & Area ('ar_area') [via dv_names metadata]")
            print(f"  Resolution           : {resolution} ({num_stations} design stations)")
            print(f"  Active DVs ({len(dv_names)}): {dv_names}")
            print(f"  Camber DVs Active    : {include_camber}")
            print(f"  Elevator Active      : {include_elevator}")
            print(f"  Aspect Ratio         : {float(parsed['aspect_ratio'][0]):.2f}")
            print(f"  Taper Ratios         : {np.round(parsed['taper_dvs'], 4)}")
            print(f"  Twist Angles (deg)   : {np.round(np.degrees(parsed['twist_dvs']), 2)}")
            if 'payload_center_x' in dv_dict:
                print(f"  Payload Center x (m) : {float(parsed['payload_center_x'][0]):.3f}")
            return parsed

    # 3. Fallback heuristic detection from vector length if dv_names is not available
    if formulation is None:
        if n_dv in [50, 26, 32, 17, 58, 34, 27]:
            formulation = 'chord_span'
        elif n_dv in [66, 43, 42, 28]:
            formulation = 'ar_area'
        elif n_dv >= 48:
            formulation = 'chord_span'
        else:
            formulation = 'ar_area'

    if resolution is None:
        if formulation == 'chord_span':
            resolution = 'fast' if n_dv in [17, 32, 22, 37, 27] else 'full'
        else:
            resolution = 'fast' if n_dv in [27, 28, 42] else 'full'

    if include_camber is None:
        if formulation == 'chord_span':
            include_camber = (n_dv in [50, 32, 58, 37, 27])
        else:
            include_camber = (n_dv in [66, 42])

    if include_elevator is None:
        if formulation == 'chord_span':
            include_elevator = False
        else:
            include_elevator = (n_dv in [28, 43])

    num_stations = 5 if resolution == 'fast' else 8
    num_chord_stations = num_stations
    initial_chord = 1.0 * scale_factor
    initial_thickness = 0.12 * initial_chord

    if formulation == 'chord_span':
        curr = 0
        chord_stretch_dvs_val = x_opt[curr : curr + num_chord_stations] * scale_factor
        curr += num_chord_stations

        include_sweep = (n_dv in [58, 34, 37, 22])
        if include_sweep:
            sweep_dvs_val = x_opt[curr : curr + num_chord_stations] * scale_factor
            curr += num_chord_stations
        else:
            sweep_dvs_val = np.zeros(num_chord_stations)

        thickness_stretch_dvs_val = x_opt[curr : curr + num_chord_stations] * initial_thickness
        curr += num_chord_stations

        include_twist = (n_dv not in [27])
        if include_twist:
            twist_dvs_val = x_opt[curr : curr + num_chord_stations] / 10.0
            curr += num_chord_stations
        else:
            twist_dvs_val = np.zeros(num_chord_stations)

        if n_dv in [26, 17]:  # historical runs with fixed span stretch but pitch and pitch_ss
            span_stretch_dv_val = np.array([0.0])
            pitch_val = x_opt[curr : curr + 1] / 10.0
            curr += 1
            pitch_ss_val = x_opt[curr : curr + 1] / 10.0
            curr += 1
        else:
            span_stretch_dv_val = x_opt[curr : curr + 1] * scale_factor if curr < n_dv else np.array([0.0])
            curr += 1
            pitch_val = x_opt[curr : curr + 1] / 10.0 if curr < n_dv else np.array([0.0])
            curr += 1
            pitch_ss_val = np.array([0.0])

        if include_elevator and curr < n_dv:
            elevator_val = float(x_opt[curr : curr + 1] / 10.0)
            curr += 1
        else:
            elevator_val = 0.0

        camber_max_percent = 5.0
        camber_scaler = 1.0 / camber_max_percent
        if include_camber and curr < n_dv and (curr + 3 * num_chord_stations <= n_dv):
            camber_dvs_val = (x_opt[curr : curr + 3 * num_chord_stations] / camber_scaler).reshape((3, num_chord_stations))
            curr += 3 * num_chord_stations
        else:
            camber_dvs_val = np.zeros((3, num_chord_stations))

        parsed = {
            'formulation': 'chord_span',
            'resolution': resolution,
            'include_camber': include_camber,
            'include_elevator': include_elevator,
            'include_sweep': include_sweep,
            'num_stations': num_stations,
            'scale_factor': scale_factor,
            'chord_stretch_dvs': chord_stretch_dvs_val,
            'sweep_dvs': sweep_dvs_val,
            'thickness_stretch_dvs': thickness_stretch_dvs_val,
            'twist_dvs': twist_dvs_val,
            'span_stretch_dv': span_stretch_dv_val,
            'pitch': pitch_val,
            'camber_dvs': camber_dvs_val,
            'elevator_angle': elevator_val,
        }

        print(f"  Detected Formulation : Chord & Span Stretch ('chord_span') [fallback heuristic]")
        print(f"  Resolution           : {resolution} ({num_stations} design stations)")
        print(f"  Camber DVs Active    : {include_camber}")
        print(f"  Elevator Active      : {include_elevator}")
        print(f"  Chord Stretches (m)  : {np.round(chord_stretch_dvs_val, 3)}")
        print(f"  Effective Chords (m) : {np.round(initial_chord + chord_stretch_dvs_val, 3)}")
        print(f"  Twist Angles (deg)   : {np.round(np.degrees(twist_dvs_val), 2)}")
        print(f"  Span Stretch (m)     : {float(span_stretch_dv_val[0]):.3f}")
        return parsed

    else:  # ar_area formulation
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

        pitch_ss_val = x_opt[curr : curr + 1] / 10.0 if curr < n_dv else np.array([0.0])
        curr += 1

        payload_cg_val = x_opt[curr : curr + 1] / 10.0 if curr < n_dv else np.array([0.4])
        curr += 1

        ttop_val = x_opt[curr : curr + num_stations] / 5000.0 if curr < n_dv else np.full(num_stations, 0.001)
        curr += num_stations

        tweb_val = x_opt[curr : curr + num_stations] / 5000.0 if curr < n_dv else np.full(num_stations, 0.001)
        curr += num_stations

        camber_max_percent = 5.0
        camber_scaler = 1.0 / camber_max_percent
        if include_camber and curr < n_dv and (curr + 3 * num_chord_stations <= n_dv):
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
            'formulation': 'ar_area',
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

        print(f"  Detected Formulation : Aspect Ratio & Area ('ar_area') [fallback heuristic]")
        print(f"  Resolution           : {resolution} ({num_stations} design stations)")
        print(f"  Camber DVs Active    : {include_camber}")
        print(f"  Elevator Active      : {include_elevator}")
        print(f"  Aspect Ratio         : {float(ar_val[0]):.2f}")
        print(f"  Taper Ratios         : {np.round(taper_dvs_val, 4)}")
        print(f"  Twist Angles (deg)   : {np.round(np.degrees(twist_dvs_val), 2)}")
        return parsed


def build_and_evaluate_geometry(parsed_dvs: dict, repo_root: str):
    """
    Builds the CSDL parameterization and evaluates deformed CAD geometry.
    Supports both 'chord_span' (direct FFD) and 'ar_area' (ParameterizationSolver).
    Returns geometry, total_wingspan, half_span, local_chords_vals, chord_stretch_vals, cad_chord_profile.
    """
    scale_factor = parsed_dvs['scale_factor']
    num_stations = parsed_dvs['num_stations']
    num_chord_stations = num_stations
    num_ffd_sections = 2 * num_stations - 1
    include_camber = parsed_dvs['include_camber']
    include_elevator = parsed_dvs['include_elevator']
    formulation = parsed_dvs['formulation']
    resolution = parsed_dvs['resolution']

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

    # Project dense spanwise leading and trailing edge lines to compute planform area via trapezoid rule
    ny_area = 31
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

    include_thickness_shape = parsed_dvs.get('include_thickness_shape', False)
    include_te_thickness = parsed_dvs.get('include_te_thickness', False)
    num_ffd_coefficients_chordwise = 5 if (include_camber or include_thickness_shape) else 2
    ffd_degree_chordwise = 2 if (include_camber or include_thickness_shape) else 1
    ffd_spanwise_degree = 2 if (formulation == 'chord_span' or resolution == 'full') else 3
    ffd_block = construct_ffd_block_around_entities(
        entities=geometry,
        num_coefficients=(num_ffd_coefficients_chordwise, num_ffd_sections, 2),
        degree=(ffd_degree_chordwise, ffd_spanwise_degree, 1)
    )
    ffd_sectional_parameterization = SectionalParameterization(
        name='ffd_param',
        parameterized_points=ffd_block.coefficients,
        principal_parametric_dimension=1
    )

    if formulation == 'chord_span':
        print("Executing Direct Sectional FFD Parameterization ('chord_span')...")
        chord_stretch_dvs = parsed_dvs['chord_stretch_dvs']
        sweep_dvs = parsed_dvs['sweep_dvs']
        thickness_stretch_dvs = parsed_dvs['thickness_stretch_dvs']
        twist_dvs = parsed_dvs['twist_dvs']
        span_stretch_dv = parsed_dvs['span_stretch_dv']

        chord_params = csdl.concatenate(
            [csdl.Variable(value=chord_stretch_dvs[i]) for i in range(num_chord_stations - 1, 0, -1)] +
            [csdl.Variable(value=chord_stretch_dvs[i]) for i in range(num_chord_stations)]
        )
        sweep_params = csdl.concatenate(
            [csdl.Variable(value=sweep_dvs[i]) for i in range(num_chord_stations - 1, 0, -1)] +
            [csdl.Variable(value=sweep_dvs[i]) for i in range(num_chord_stations)]
        )
        thickness_params = csdl.concatenate(
            [csdl.Variable(value=thickness_stretch_dvs[i]) for i in range(num_chord_stations - 1, 0, -1)] +
            [csdl.Variable(value=thickness_stretch_dvs[i]) for i in range(num_chord_stations)]
        )
        twist_params = csdl.concatenate(
            [csdl.Variable(value=twist_dvs[i]) for i in range(num_chord_stations - 1, 0, -1)] +
            [csdl.Variable(value=twist_dvs[i]) for i in range(num_chord_stations)]
        )
        span_weights = np.linspace(-1.0, 1.0, num_ffd_sections)
        span_params = csdl.Variable(value=span_stretch_dv[0] * span_weights)

        sectional_parameters = SectionalParameters()
        sectional_parameters.add_stretch(axis=np.array([1., 0., 0.]), stretch=chord_params)
        sectional_parameters.add_translation(axis=np.array([1., 0., 0.]), translation=sweep_params)
        sectional_parameters.add_stretch(axis=np.array([0., 0., 1.]), stretch=thickness_params)
        sectional_parameters.add_translation(axis=np.array([0., 1., 0.]), translation=span_params)
        sectional_parameters.add_rotation(axis=np.array([0., 1., 0.]), rotation=twist_params, parametric_coordinate=np.array([0.25, 0.5]))

        ffd_coefficients = ffd_sectional_parameterization.evaluate(sectional_parameters, plot=False)

        if include_camber:
            camber_dvs = parsed_dvs['camber_dvs']
            section_chords = ffd_coefficients[-1, :, 0, 0] - ffd_coefficients[0, :, 0, 0]
            full_span_camber_list = []
            for c in range(3):
                row = csdl.concatenate(
                    [csdl.Variable(value=camber_dvs[c, i]) for i in range(num_chord_stations - 1, 0, -1)] +
                    [csdl.Variable(value=camber_dvs[c, i]) for i in range(num_chord_stations)]
                )
                full_span_camber_list.append(csdl.reshape(row, (1, num_ffd_sections)))
            full_span_camber = csdl.concatenate(full_span_camber_list, axis=0)
            camber_displacement = (full_span_camber / 100.0) * csdl.expand(section_chords, (3, num_ffd_sections), 'j->ij')
            camber_delta = csdl.expand(camber_displacement, (3, num_ffd_sections, 2), 'ij->ijk')
            ffd_coefficients = ffd_coefficients.set(csdl.slice[1:4, :, :, 2], ffd_coefficients[1:4, :, :, 2] + camber_delta)

        if include_thickness_shape and 'thickness_shape_dvs' in parsed_dvs and parsed_dvs['thickness_shape_dvs'] is not None:
            thickness_shape_dvs_val = parsed_dvs['thickness_shape_dvs']
            n_rows_th = thickness_shape_dvs_val.shape[0]
            section_chords = ffd_coefficients[-1, :, 0, 0] - ffd_coefficients[0, :, 0, 0]
            full_span_thick_list = []
            for c in range(n_rows_th):
                row = csdl.concatenate(
                    [csdl.Variable(value=thickness_shape_dvs_val[c, i]) for i in range(num_chord_stations - 1, 0, -1)] +
                    [csdl.Variable(value=thickness_shape_dvs_val[c, i]) for i in range(num_chord_stations)]
                )
                full_span_thick_list.append(csdl.reshape(row, (1, num_ffd_sections)))
            full_span_thick = csdl.concatenate(full_span_thick_list, axis=0)
            thick_displacement = (full_span_thick / 100.0) * csdl.expand(section_chords, (n_rows_th, num_ffd_sections), 'j->ij')
            half_dt = 0.5 * thick_displacement
            row_end = n_rows_th
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
        geometry.set_coefficients(geometry_coefficients)

        if include_elevator and abs(parsed_dvs['elevator_angle']) > 1e-6:
            from lsdo_geo import rotate
            elevator_angle = csdl.Variable(value=parsed_dvs['elevator_angle'])
            elevator_hinge = geometry.project(np.array([0.8 * scale_factor, 0.0, 0.0]))
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
                func.coefficients[:4, row_slc, :] = rot_sub

        wingspan_eval = geometry.evaluate(leading_edge_right)[1] - geometry.evaluate(leading_edge_left)[1]
        wingspan_val = wingspan_eval.value if hasattr(wingspan_eval, 'value') else wingspan_eval
        total_wingspan = float(np.abs(np.squeeze(wingspan_val)))
        half_span = total_wingspan / 2.0
        local_chords_vals = []
        for i in range(num_chord_stations):
            ch_eval = geometry.evaluate(chord_te_projections[i])[0] - geometry.evaluate(chord_le_projections[i])[0]
            val = ch_eval.value if hasattr(ch_eval, 'value') else ch_eval
            local_chords_vals.append(float(np.squeeze(val)))
        chord_stretch_vals = [float(s) for s in chord_stretch_dvs]

    else:  # ar_area formulation with ParameterizationSolver
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
            section_chords = ffd_coefficients[-1, :, 0, 0] - ffd_coefficients[0, :, 0, 0]
            full_span_camber_list = []
            for c in range(3):
                row = csdl.concatenate([camber_dvs[c, i] for i in range(num_chord_stations - 1, 0, -1)] + [camber_dvs[c, i] for i in range(num_chord_stations)])
                full_span_camber_list.append(csdl.reshape(row, (1, num_ffd_sections)))
            full_span_camber = csdl.concatenate(full_span_camber_list, axis=0)
            camber_displacement = (full_span_camber / 100.0) * csdl.expand(section_chords, (3, num_ffd_sections), 'j->ij')
            camber_delta = csdl.expand(camber_displacement, (3, num_ffd_sections, 2), 'ij->ijk')
            ffd_coefficients = ffd_coefficients.set(csdl.slice[1:4, :, :, 2], ffd_coefficients[1:4, :, :, 2] + camber_delta)

        if include_thickness_shape and 'thickness_shape_dvs' in parsed_dvs and parsed_dvs['thickness_shape_dvs'] is not None:
            thickness_shape_dvs_val = parsed_dvs['thickness_shape_dvs']
            n_rows_th = thickness_shape_dvs_val.shape[0]
            section_chords = ffd_coefficients[-1, :, 0, 0] - ffd_coefficients[0, :, 0, 0]
            full_span_thick_list = []
            for c in range(n_rows_th):
                row = csdl.concatenate(
                    [csdl.Variable(value=thickness_shape_dvs_val[c, i]) for i in range(num_chord_stations - 1, 0, -1)] +
                    [csdl.Variable(value=thickness_shape_dvs_val[c, i]) for i in range(num_chord_stations)]
                )
                full_span_thick_list.append(csdl.reshape(row, (1, num_ffd_sections)))
            full_span_thick = csdl.concatenate(full_span_thick_list, axis=0)
            thick_displacement = (full_span_thick / 100.0) * csdl.expand(section_chords, (n_rows_th, num_ffd_sections), 'j->ij')
            half_dt = 0.5 * thick_displacement
            row_end = n_rows_th
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
        geometry.set_coefficients(geometry_coefficients)

        wingspan = geometry.evaluate(leading_edge_right)[1] - geometry.evaluate(leading_edge_left)[1]
        local_chords = [geometry.evaluate(chord_te_projections[i])[0] - geometry.evaluate(chord_le_projections[i])[0] for i in range(num_chord_stations)]
        local_thicknesses = [geometry.evaluate(upper_thickness_projections[i])[2] - geometry.evaluate(lower_thickness_projections[i])[2] for i in range(num_chord_stations)]

        area_le_pts = geometry.evaluate(projected_area_le, plot=False)
        area_te_pts = geometry.evaluate(projected_area_te, plot=False)
        dense_chords = area_te_pts[:, 0] - area_le_pts[:, 0]
        dense_y = area_le_pts[:, 1]

        c_mid = 0.5 * (dense_chords[:-1] + dense_chords[1:])
        dy = dense_y[1:] - dense_y[:-1]
        half_planform_area = csdl.sum(c_mid * dy)
        planform_area = 2.0 * half_planform_area
        aspect_ratio_calc = (wingspan**2) / planform_area

        geometry_solver = ParameterizationSolver()
        geometry_solver.add_state(chord_stretch_states)
        geometry_solver.add_state(thickness_stretch_states)
        geometry_solver.add_state(span_stretch_state)
        geometry_solver.add_state(sweep_translation_states)

        bspline_reg = BsplineTargetRegularization(num_chord_stations=num_chord_stations, scale_factor=scale_factor)

        geometric_variables = GeometricVariables()

        local_chords_vec = csdl.concatenate([csdl.reshape(c, (1,)) for c in local_chords])
        local_thicknesses_vec = csdl.concatenate([csdl.reshape(t, (1,)) for t in local_thicknesses])

        c_ref_scale = float(np.asarray(local_chords[0].value).flatten()[0]) if local_chords[0].value is not None else 1.0 * scale_factor
        res_taper = bspline_reg.compute_taper_residual(local_chords_vec, taper_dvs, c_ref=c_ref_scale)
        geometric_variables.add_variable(res_taper, 0.0, penalty_value=None)

        t_ref_scale = 0.12 * c_ref_scale
        if 'tc_target_control_points' in parsed_dvs and parsed_dvs['tc_target_control_points'] is not None:
            tc_target_control_points = csdl.Variable(shape=(num_chord_stations,), value=parsed_dvs['tc_target_control_points'])
        else:
            tc_target_control_points = csdl.Variable(shape=(num_chord_stations,), value=np.full(num_chord_stations, 0.12))
        res_thick = bspline_reg.compute_thickness_residual(
            local_thicknesses_vec, local_chords_vec, tc_target_control_points, t_ref=t_ref_scale
        )
        geometric_variables.add_variable(res_thick, 0.0, penalty_value=None)

        if 'planform_area_target' in parsed_dvs and parsed_dvs['planform_area_target'] is not None:
            target_area = csdl.Variable(value=float(np.asarray(parsed_dvs['planform_area_target']).flatten()[0]))
            geometric_variables.add_variable(planform_area / (10.0 * scale_factor**2), target_area / (10.0 * scale_factor**2), penalty_value=None)
            geometric_variables.add_variable(wingspan**2 / (10.0 * 10.0 * scale_factor**2),
                                             aspect_ratio * target_area / (10.0 * 10.0 * scale_factor**2), penalty_value=None)
        else:
            target_area = 10.0 * (scale_factor ** 2)
            geometric_variables.add_variable(planform_area / target_area, 1.0, penalty_value=None)
            geometric_variables.add_variable(wingspan**2 / (10.0 * target_area),
                                             aspect_ratio * 1.0 / 10.0, penalty_value=None)

        qc_pts = [geometry.evaluate(quarter_chord_projections[i]) for i in range(num_chord_stations)]
        dx_list = [csdl.reshape(qc_pts[i + 1][0] - qc_pts[i][0], (1,)) for i in range(num_chord_stations - 1)]
        dy_list = [csdl.reshape(qc_pts[i + 1][1] - qc_pts[i][1], (1,)) for i in range(num_chord_stations - 1)]
        dx_qc_vec = csdl.concatenate(dx_list)
        dy_qc_vec = csdl.concatenate(dy_list)

        y_scale = 5.0 * scale_factor
        res_sweep = bspline_reg.compute_sweep_residual(
            dx_qc_vec, dy_qc_vec, sweep_angle_dvs, y_scale=y_scale
        )
        geometric_variables.add_variable(res_sweep, 0.0, penalty_value=None)

        print("Executing ParameterizationSolver (in-line Newton solver)...")
        geometry_solver.evaluate(geometric_variables)

        total_wingspan = float(wingspan.value[0])
        half_span = total_wingspan / 2.0
        s_ref = float(planform_area.value[0])
        ar_final = float(aspect_ratio_calc.value[0])
        local_chords_vals = [float(c.value[0]) for c in local_chords]
        chord_stretch_vals = [float(s) for s in chord_stretch_states.value]

    print(f"Deformed Geometry Evaluated:")
    print(f"  Wingspan (b)     = {total_wingspan:.4f} m (Half-Span = {half_span:.4f} m)")
    if 's_ref' in locals():
        print(f"  Planform Area (S)= {s_ref:.4f} m^2")
    if 'ar_final' in locals():
        print(f"  Aspect Ratio (AR)= {ar_final:.2f}")
    for i, cv in enumerate(local_chords_vals):
        print(f"  Station {i} Design Chord = {cv:.3f} m (Stretch = {chord_stretch_vals[i]:+.3f} m)")

    # Fine spanwise continuous chord and planform profile sampled directly from CAD surface
    print("Evaluating fine continuous CAD surface chord profile...")
    fn_lo = geometry.functions[0]
    fn_up = geometry.functions[1]
    n_fine_cad = 120
    cad_eta = np.linspace(0.0, 0.999, n_fine_cad)
    v_fine_pts = np.linspace(0.0, 1.0, 120)
    cad_chords = np.zeros(n_fine_cad)
    cad_x_le = np.zeros(n_fine_cad)
    cad_x_te = np.zeros(n_fine_cad)
    cad_y = np.zeros(n_fine_cad)
    for idx_cad, eta_val in enumerate(cad_eta):
        coords = np.column_stack([np.full_like(v_fine_pts, eta_val), v_fine_pts])
        p_lo = fn_lo.evaluate(coords, non_csdl=True)
        p_up = fn_up.evaluate(coords, non_csdl=True)
        all_x = np.concatenate([p_lo[:, 0], p_up[:, 0]])
        all_y = np.concatenate([p_lo[:, 1], p_up[:, 1]])
        cad_x_le[idx_cad] = float(np.min(all_x))
        cad_x_te[idx_cad] = float(np.max(all_x))
        cad_y[idx_cad] = float(np.mean(all_y))
        cad_chords[idx_cad] = cad_x_te[idx_cad] - cad_x_le[idx_cad]

    cad_chord_profile = {
        'eta': cad_eta,
        'chord': cad_chords,
        'x_le': cad_x_le,
        'x_te': cad_x_te,
        'y': cad_y,
    }

    return geometry, total_wingspan, half_span, local_chords_vals, chord_stretch_vals, cad_chord_profile


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


def generate_airfoil_gallery_plot(station_data: list, output_path: str, half_span: float, parsed_dvs: dict = None):
    """
    Figure 1: 4x2 Grid of individual stations with detailed annotations.
    """
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    n_stns = len(station_data)
    nrows = (n_stns + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(15, 3.5 * nrows), dpi=200)
    axes = axes.flatten()
    for extra_ax in axes[n_stns:]:
        fig.delaxes(extra_ax)

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

        # Draw payload box on root section if available
        if k == 0 and parsed_dvs and 'payload_center_x' in parsed_dvs:
            from matplotlib.patches import Rectangle
            pay_x = float(np.asarray(parsed_dvs['payload_center_x']).flatten()[0])
            L_box = 33.0 / 3.28084
            H_box = 10.0 / 3.28084
            rect = Rectangle(
                (pay_x - L_box / 2.0, -H_box / 2.0),
                L_box,
                H_box,
                linewidth=1.8,
                edgecolor='#d62728',
                facecolor='#ff9896',
                alpha=0.30,
                linestyle='--',
                label=f'TCP0 Payload ({L_box:.2f}m × {H_box:.2f}m)',
                zorder=3,
            )
            ax.add_patch(rect)

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
        f"Optimized Airfoil Cross-Sections Across {n_stns} Spanwise Stations\n"
        f"BWB Aerostructural Optimization | Half-Span $b/2 = {half_span:.3f}$ m",
        fontsize=14, fontweight='bold', y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved Airfoil Gallery Plot to: {output_path}")


def evaluate_symmetric_bspline(right_half_cps: np.ndarray, num_eval_pts: int = 300):
    """
    Constructs a symmetric degree-3 (cubic) B-spline across full span (N = 2*n - 1 CPs),
    and evaluates it along the right half-span (v in [0.5, 1.0], eta in [0.0, 1.0]).

    Guarantees zero slope at the root (eta = 0, v = 0.5) by construction of symmetry.
    Returns:
        eta_fine: (num_eval_pts,) normalized spanwise coordinates [0, 1]
        curve_fine: (num_eval_pts,) evaluated continuous B-spline field
        greville_eta: (len(right_half_cps),) Greville abscissae (eta locations) of control points
    """
    n = len(right_half_cps)
    N = 2 * n - 1
    # Symmetrically mirror control points across root
    full_cps = np.concatenate([right_half_cps[::-1][:-1], right_half_cps])

    sp = lfs.BSplineSpace(num_parametric_dimensions=1, degree=3, coefficients_shape=(N,))
    knots = sp.knots[0]

    # Greville abscissae across full span
    p = 3
    greville_full = np.array([np.mean(knots[i + 1 : i + 1 + p]) for i in range(N)])
    right_idx = np.where(greville_full >= 0.5 - 1e-9)[0]
    greville_eta = (greville_full[right_idx] - 0.5) / 0.5

    eta_fine = np.linspace(0.0, 1.0, num_eval_pts)
    v_fine = (0.5 + 0.5 * eta_fine).reshape(-1, 1)
    B_matrix = sp.compute_basis_matrix(v_fine).toarray()
    curve_fine = B_matrix @ full_cps

    return eta_fine, curve_fine, greville_eta


def generate_shape_evolution_plot(
    station_data: list,
    output_path: str,
    half_span: float,
    parsed_dvs: dict = None,
    local_chords_vals: list = None,
    chord_stretch_vals: list = None,
    cad_chord_profile: dict = None,
):
    """
    Figure 2: Multi-panel comparative analysis:
    (a) Planform cutline map (x-y plane)
    (b) Leading-edge aligned physical cross-section overlay
    (c) Normalized airfoil profile comparison (x/c vs z/c)
    (d) Continuous Spanwise Chord Distribution (Symmetric Cubic B-Spline)
    (e) Continuous Spanwise Aerodynamic Twist Distribution (Symmetric Cubic B-Spline)
    """
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig = plt.figure(figsize=(16, 12), dpi=200)
    gs = GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.22)

    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(station_data)))

    # -------------------------------------------------------------------------
    # Panel (a): Planform View with Station Cutlines
    # -------------------------------------------------------------------------
    ax_plan = fig.add_subplot(gs[0, 0])
    if cad_chord_profile is not None and 'x_le' in cad_chord_profile:
        y_dense = cad_chord_profile['y']
        x_le_dense = cad_chord_profile['x_le']
        x_te_dense = cad_chord_profile['x_te']
    else:
        y_stations = [d['y'] for d in station_data]
        x_les = [d['x_le'] for d in station_data]
        x_tes = [d['x_te'] for d in station_data]
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
    label_offset = max(0.02, 0.012 * half_span)
    for k, (data, col) in enumerate(zip(station_data, colors)):
        ax_plan.plot([data['x_le'], data['x_te']], [data['y'], data['y']], '-', color=col, linewidth=2.2)
        ax_plan.text(data['x_te'] + label_offset, data['y'], f"Stn {k+1} ($\\eta={data['eta']:.2f}$)",
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
    # Panel (d) & (e): Symmetric Cubic B-Spline Spanwise Distributions
    # -------------------------------------------------------------------------
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    sub_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1], hspace=0.35)
    ax_chord = fig.add_subplot(sub_gs[0, 0])
    ax_twist = fig.add_subplot(sub_gs[1, 0])

    # 1. Chord B-Spline & CAD Surface
    scale_factor = parsed_dvs.get('scale_factor', 7.5) if parsed_dvs is not None else 7.5
    baseline_chord = 1.0 * scale_factor

    if chord_stretch_vals is not None:
        chord_cps = baseline_chord + np.array(chord_stretch_vals)
    elif local_chords_vals is not None:
        chord_cps = np.array(local_chords_vals)
    else:
        chord_cps = np.array([d['chord'] for d in station_data])

    eta_fine, ffd_chord_curve, chord_cp_eta = evaluate_symmetric_bspline(chord_cps)
    c_root_val = float(ffd_chord_curve[0])

    # FFD continuous B-spline curve
    ax_chord.plot(
        eta_fine, ffd_chord_curve, '-', color='#d62728', linewidth=2.5,
        label=f'FFD Chord B-spline $c(y)$ ($c(0)={c_root_val:.2f}$ m)'
    )
    # FFD chord control points
    ax_chord.plot(
        chord_cp_eta, chord_cps, 'o', color='#d62728', markersize=7,
        markeredgecolor='black', zorder=5, label=f'Chord CPs ($c_0+\\Delta c_k$, $N={len(chord_cps)}$)'
    )

    # CAD surface physical chord profile
    if cad_chord_profile is not None:
        ax_chord.plot(
            cad_chord_profile['eta'], cad_chord_profile['chord'], '--',
            color='#1f77b4', linewidth=2.0, alpha=0.9,
            label='Physical CAD Surface Chord'
        )

    # Sliced stations for comparison
    sliced_etas = [d['eta'] for d in station_data]
    sliced_chords = [d['chord'] for d in station_data]
    ax_chord.scatter(
        sliced_etas, sliced_chords, marker='d', color='#2ca02c', s=35, zorder=4,
        alpha=0.85, label=f'Sliced Sections ($N={len(sliced_chords)}$)'
    )

    # Secondary axis: Taper Ratio c / c_root
    ax_taper = ax_chord.twinx()
    ax_taper.plot(eta_fine, ffd_chord_curve / c_root_val, ':', color='#7f7f7f', alpha=0.5, linewidth=1.2)
    ax_taper.set_ylabel(r'Taper Ratio $c / c_{\mathrm{root}}$', color='#555555', fontweight='bold', fontsize=9)
    ax_taper.tick_params(axis='y', labelcolor='#555555')
    ax_taper.grid(False)

    ax_chord.set_ylabel('Chord $c$ [m]', color='#d62728', fontweight='bold', fontsize=9.5)
    ax_chord.tick_params(axis='y', labelcolor='#d62728')
    ax_chord.set_title('(d) Spanwise Chord Distribution (FFD B-Spline & CAD Surface)', fontsize=10.5, fontweight='bold', pad=5)
    ax_chord.set_xlim([0.0, 1.0])
    y_max_chord = max(np.max(ffd_chord_curve), np.max(chord_cps))
    if cad_chord_profile is not None:
        y_max_chord = max(y_max_chord, np.max(cad_chord_profile['chord']))
    chord_ylim = [0.0, y_max_chord * 1.15]
    ax_chord.set_ylim(chord_ylim)
    ax_taper.set_ylim([chord_ylim[0] / c_root_val, chord_ylim[1] / c_root_val])
    ax_chord.grid(True, linestyle=':', alpha=0.6)
    ax_chord.legend(loc='upper right', frameon=True, framealpha=0.92, fontsize=7.5)

    # 2. Twist B-Spline & Section Incidence
    twist_dvs_deg = np.degrees(parsed_dvs['twist_dvs']) if parsed_dvs is not None else np.zeros(5)
    eta_fine_tw, twist_curve, twist_cp_eta = evaluate_symmetric_bspline(twist_dvs_deg)

    # Continuous aerodynamic twist B-spline
    ax_twist.plot(
        eta_fine_tw, twist_curve, '-', color='#9467bd', linewidth=2.5,
        label=r'FFD Aerodynamic Twist $\theta(y)$'
    )
    # Twist design variable control points
    ax_twist.plot(
        twist_cp_eta, twist_dvs_deg, 's', color='#9467bd', markersize=7,
        markeredgecolor='black', zorder=5, label=f'Twist DVs ($tw_k$, Peak: {np.min(twist_dvs_deg):.1f}°)'
    )

    # Sliced section geometric incidence angle
    sliced_twists = [d['twist_deg'] for d in station_data]
    ax_twist.plot(
        sliced_etas, sliced_twists, '^--', color='#2ca02c', markersize=6, linewidth=1.5,
        alpha=0.85, label=r'CAD Surface Incidence $\arctan(\Delta z / c)$'
    )

    # Mark active lower bound (-15 deg)
    ax_twist.axhline(
        -15.0, color='#d62728', linestyle=':', linewidth=1.5, alpha=0.8,
        label=r'Washout Lower Bound ($-15^{\circ}$)'
    )

    ax_twist.set_xlabel(r'Normalized Spanwise Coordinate $\eta = y / (b/2)$', fontsize=9.5, fontweight='bold')
    ax_twist.set_ylabel('Twist / Incidence [deg]', color='#9467bd', fontweight='bold', fontsize=9.5)
    ax_twist.tick_params(axis='y', labelcolor='#9467bd')
    ax_twist.set_title('(e) Spanwise Aerodynamic Twist & Surface Incidence Distribution', fontsize=10.5, fontweight='bold', pad=5)
    ax_twist.set_xlim([0.0, 1.0])
    y_min_tw = min(-18.0, float(np.min(twist_curve)) - 3.0, float(np.min(twist_dvs_deg)) - 2.5)
    y_max_tw = max(4.0, float(np.max(twist_curve)) + 2.0, float(np.max(twist_dvs_deg)) + 2.0)
    ax_twist.set_ylim([y_min_tw, y_max_tw])
    ax_twist.grid(True, linestyle=':', alpha=0.6)
    ax_twist.legend(loc='lower left', ncol=2, frameon=True, framealpha=0.92, fontsize=7.5)

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

    parser = argparse.ArgumentParser(description="Extract and plot airfoil cross-sections from optimization results.")
    parser.add_argument("run_dir", nargs="?", default=None, help="Path to optimization output folder containing x.out")
    parser.add_argument("--formulation", choices=["auto", "chord_span", "ar_area"], default="auto", help="Wing parameterization formulation")
    parser.add_argument("--resolution", choices=["auto", "fast", "full"], default="auto", help="Spanwise resolution ('fast' = 5 stations, 'full' = 8 stations)")
    args = parser.parse_args()

    # Determine target run output folder
    if args.run_dir:
        target_dir = os.path.abspath(args.run_dir)
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

    # Detect scale factor
    scale_factor = detect_scale_factor(target_dir, repo_root)

    formulation_override = None if args.formulation == "auto" else args.formulation
    resolution_override = None if args.resolution == "auto" else args.resolution

    # Parse DVs
    parsed_dvs = parse_design_variables(
        x_opt,
        scale_factor=scale_factor,
        target_dir=target_dir,
        formulation_override=formulation_override,
        resolution_override=resolution_override,
    )
    num_stations = parsed_dvs['num_stations']

    # Reconstruct authentic geometry
    geometry, total_wingspan, half_span, local_chords_vals, chord_stretch_vals, cad_chord_profile = build_and_evaluate_geometry(parsed_dvs, repo_root)

    # Extract cross-section stations along half-span
    station_data = extract_station_airfoils(geometry, half_span, num_stations=num_stations)

    # Generate Output Figures and Data
    gallery_fig_path = os.path.join(target_dir, "airfoil_cross_sections_gallery.png")
    evolution_fig_path = os.path.join(target_dir, "airfoil_shape_evolution.png")
    telemetry_path = os.path.join(target_dir, "airfoil_cross_sections_data.npz")

    generate_airfoil_gallery_plot(station_data, gallery_fig_path, half_span, parsed_dvs=parsed_dvs)
    generate_shape_evolution_plot(
        station_data,
        evolution_fig_path,
        half_span,
        parsed_dvs=parsed_dvs,
        local_chords_vals=local_chords_vals,
        chord_stretch_vals=chord_stretch_vals,
        cad_chord_profile=cad_chord_profile,
    )
    save_airfoil_telemetry(station_data, telemetry_path, half_span)

    # Also copy artifacts to the active agent brain directory if available
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
    if artifact_dir and os.path.exists(artifact_dir):
        shutil.copy2(gallery_fig_path, os.path.join(artifact_dir, "airfoil_cross_sections_gallery.png"))
        shutil.copy2(evolution_fig_path, os.path.join(artifact_dir, "airfoil_shape_evolution.png"))
        print(f"Artifacts successfully copied to brain directory: {artifact_dir}")

    # Summary table
    print("\n" + "=" * 80)
    print(f"SUMMARY: AIRFOIL CROSS SECTIONS ALONG {len(station_data)} SPANWISE STATIONS")
    print("=" * 80)
    print(f"{'Stn':4s} | {'eta':6s} | {'y [m]':8s} | {'Chord [m]':10s} | {'t_max [mm]':11s} | {'t/c [%]':8s} | {'Twist [deg]':12s} | {'Camber [mm]':12s}")
    print("-" * 80)
    for d in station_data:
        print(f"{d['station_idx']:4d} | {d['eta']:6.3f} | {d['y']:8.3f} | {d['chord']:10.4f} | {d['t_max']*1e3:11.2f} | {d['tc_ratio']:8.2f} | {d['twist_deg']:+12.2f} | {d['max_camber']*1e3:12.2f}")
    print("=" * 80)
    print("Analysis complete successfully!")


if __name__ == "__main__":
    main()
