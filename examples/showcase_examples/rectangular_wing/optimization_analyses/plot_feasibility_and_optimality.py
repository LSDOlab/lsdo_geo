"""Plot and analyze optimization feasibility and optimality history for rectangular wing to BWB.

Visualizes the convergence trajectory across iterations, broken down by:
1. Feasibility history: Separate curves for each constraint type
   (Separation, Structural Stress, Flight Trim, Static Margin, Maneuver Attachment,
    Payload Containment, Geometry Limits).
2. Optimality history: Separate curves for each design variable group
   (Chord & Planform, Sweep, Twist, Airfoil Thickness, Airfoil Camber,
    Structural Sizing, Flight Angles & Trim, Payload & CG).

Supports:
- Group metrics: 'max' (worst-case violation / max step) or 'norm' (L2 norm)
- Standalone CLI execution on any output folder:
    python plot_feasibility_and_optimality.py [output_dir] [--metric max|norm|both]
- Instantaneous loading from 'feasibility_optimality_data.npz' or 'slsqp_recorder.hdf5'
- Seamless fallback reconstructing histories from 'x.out', 'slsqp_summary.out',
  and 'structural_data.npz'
- In-situ recording during optimization or replay via OptimizationConvergenceHistory
"""

from __future__ import annotations
import os
import sys
import glob
import json
import shutil
import argparse
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Ensure working directory and relevant repos are in path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../../../..'))
RW_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

for p in [REPO_ROOT, RW_DIR]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

KNOWN_OUTPUT_DIRS = [
    os.path.join(REPO_ROOT, 'rectangular_wing_to_bwb_aerostructural_optimization_outputs'),
    os.path.join(REPO_ROOT, 'rectangular_wing_aerostructural_optimization_outputs'),
]

# Color palettes and styling
CONSTRAINT_STYLE = {
    'Separation': {'color': '#e41a1c', 'marker': 'o', 'label': 'Separation (Stratford)'},
    'Structural Stress': {'color': '#ff7f00', 'marker': 's', 'label': 'Structural Stress (Spar & Web)'},
    'Flight Trim': {'color': '#377eb8', 'marker': '^', 'label': 'Flight Trim (Lift & Pitch)'},
    'Static Margin': {'color': '#984ea3', 'marker': 'd', 'label': 'Static Margin (SM ≥ 0.02)'},
    'Maneuver Attachment': {'color': '#e6ab02', 'marker': 'v', 'label': 'Maneuver Attachment (2.5g)'},
    'Payload Containment': {'color': '#f781bf', 'marker': 'p', 'label': 'Payload Containment (Geonic)'},
    'Geometry Limits': {'color': '#4daf4a', 'marker': '*', 'label': 'Geometry Limits (AR, Sref, t/c)'},
}

DV_STYLE = {
    'Chord & Planform': {'color': '#1f77b4', 'marker': 'o', 'label': 'Chord & Planform'},
    'Sweep': {'color': '#ff7f0e', 'marker': 's', 'label': 'Sweep Angle'},
    'Twist': {'color': '#2ca02c', 'marker': '^', 'label': 'Twist / Washout'},
    'Airfoil Thickness': {'color': '#d62728', 'marker': 'd', 'label': 'Airfoil Thickness (t/c & Shape)'},
    'Airfoil Camber': {'color': '#9467bd', 'marker': 'v', 'label': 'Airfoil Camber'},
    'Structural Sizing': {'color': '#8c564b', 'marker': 'p', 'label': 'Structural Thickness (Caps & Webs)'},
    'Flight Angles & Trim': {'color': '#e377c2', 'marker': '*', 'label': 'Flight Angles (Pitch & Elevator)'},
    'Payload & CG': {'color': '#17becf', 'marker': 'X', 'label': 'Payload & CG Locations'},
}

# Variable name to DV Group mapping
DV_NAME_TO_GROUP = {
    'chord_stretch_dvs': 'Chord & Planform',
    'span_stretch_dv': 'Chord & Planform',
    'taper_control_points': 'Chord & Planform',
    'taper_dvs': 'Chord & Planform',
    'aspect_ratio': 'Chord & Planform',
    'planform_area_target': 'Chord & Planform',
    'sweep_dvs': 'Sweep',
    'sweep_angle_control_points': 'Sweep',
    'sweep_angle_dvs': 'Sweep',
    'twist_dvs': 'Twist',
    'tip_twist': 'Twist',
    'thickness_stretch_dvs': 'Airfoil Thickness',
    'tc_target_control_points': 'Airfoil Thickness',
    'thickness_shape_dvs': 'Airfoil Thickness',
    'camber_dvs': 'Airfoil Camber',
    'ttop_dvs': 'Structural Sizing',
    'tweb_dvs': 'Structural Sizing',
    'pitch': 'Flight Angles & Trim',
    'pitch_ss': 'Flight Angles & Trim',
    'pitch_neg1g': 'Flight Angles & Trim',
    'elevator_angle': 'Flight Angles & Trim',
    'payload_center_x': 'Payload & CG',
    'pallet_stack_center_x': 'Payload & CG',
    'payload_cg': 'Payload & CG',
    'misc_cg': 'Payload & CG',
}


def find_target_output_dir(base_dir: Optional[str] = None) -> str:
    """Find target optimization output directory."""
    if base_dir:
        if os.path.isdir(base_dir):
            return os.path.abspath(base_dir)
        for cand_base in KNOWN_OUTPUT_DIRS:
            cand_path = os.path.join(cand_base, base_dir)
            if os.path.isdir(cand_path):
                return os.path.abspath(cand_path)
            # Match partial name
            matches = [d for d in glob.glob(os.path.join(cand_base, f"*{base_dir}*")) if os.path.isdir(d)]
            if matches:
                return os.path.abspath(max(matches, key=os.path.getmtime))

    for cand_base in KNOWN_OUTPUT_DIRS:
        if os.path.isdir(cand_base):
            subdirs = [
                d for d in glob.glob(os.path.join(cand_base, '*'))
                if os.path.isdir(d) and (
                    os.path.isfile(os.path.join(d, 'slsqp_summary.out')) or
                    os.path.isfile(os.path.join(d, 'x.out')) or
                    os.path.isfile(os.path.join(d, 'feasibility_optimality_data.npz'))
                )
            ]
            if subdirs:
                return os.path.abspath(max(subdirs, key=os.path.getmtime))

    raise FileNotFoundError("No valid optimization output directories found.")


def parse_slsqp_summary(summary_path: str) -> Optional[Dict[str, np.ndarray]]:
    """Parse slsqp_summary.out into a dictionary of arrays."""
    if not os.path.exists(summary_path):
        return None
    data = []
    with open(summary_path, 'r') as f:
        header = f.readline().split()
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 9:
                try:
                    data.append([float(p) for p in parts[:9]])
                except ValueError:
                    pass
    if not data:
        return None
    arr = np.array(data)
    return {
        'major': arr[:, 0].astype(int),
        'nfev': arr[:, 1].astype(int),
        'ngev': arr[:, 2].astype(int),
        'objfun': arr[:, 3],
        'gnorm': arr[:, 4],
        'cnorm': arr[:, 5],
        'feas': arr[:, 6],
        'opt': arr[:, 7],
        'step': arr[:, 8],
    }


class OptimizationConvergenceHistory:
    """Accumulates and computes feasibility and optimality histories per group."""

    def __init__(
        self,
        design_variables: Optional[Dict[str, Any]] = None,
        formulation: str = 'chord_span',
        scale_factor: float = 7.5,
        use_geonic: bool = True,
        geonic_payload_mode: str = 'both',
        allowable_stress: float = 2.4e8,
        stall_model: str = 'stratford_closure',
        w_ref: float = 50000.0,
        c_ref: float = 2.5,
        load_factor_val: float = 2.5,
        include_neg1g_sizing: bool = False,
    ):
        self.design_variables = design_variables or {}
        self.formulation = formulation
        self.scale_factor = scale_factor
        self.use_geonic = use_geonic
        self.geonic_payload_mode = geonic_payload_mode
        self.allowable_stress = allowable_stress
        self.stall_model = stall_model
        self.w_ref = w_ref
        self.c_ref = c_ref
        self.load_factor_val = load_factor_val
        self.include_neg1g_sizing = include_neg1g_sizing

        # Slices and group mapping for design variables
        self.dv_slices: Dict[str, slice] = {}
        self.group_to_dv_slices: Dict[str, List[slice]] = {}
        curr = 0
        for name, dv_info in self.design_variables.items():
            if hasattr(dv_info, 'variable') and hasattr(dv_info.variable, 'shape'):
                sz = int(np.prod(dv_info.variable.shape))
            elif hasattr(dv_info, 'shape'):
                sz = int(np.prod(dv_info.shape))
            elif isinstance(dv_info, (int, np.integer)):
                sz = int(dv_info)
            elif isinstance(dv_info, (tuple, list)):
                sz = int(np.prod(dv_info))
            else:
                sz = 1
            slc = slice(curr, curr + sz)
            self.dv_slices[name] = slc
            grp = DV_NAME_TO_GROUP.get(name, 'Other DVs')
            self.group_to_dv_slices.setdefault(grp, []).append(slc)
            curr += sz
        self.total_dvs = curr

        # History containers
        self.iterations: List[int] = []
        self.con_violations_max: Dict[str, List[float]] = {grp: [] for grp in CONSTRAINT_STYLE}
        self.con_violations_norm: Dict[str, List[float]] = {grp: [] for grp in CONSTRAINT_STYLE}
        self.dv_steps_max: Dict[str, List[float]] = {grp: [] for grp in DV_STYLE}
        self.dv_steps_norm: Dict[str, List[float]] = {grp: [] for grp in DV_STYLE}
        self.dv_total_max: List[float] = []
        self.dv_total_norm: List[float] = []
        self.prev_x_scaled: Optional[np.ndarray] = None

        # First-Order Optimality containers: df/dx + lambda * dc/dx
        self.has_kkt_optimality: bool = False
        self.dv_kkt_max: Dict[str, List[float]] = {grp: [] for grp in DV_STYLE}
        self.dv_kkt_norm: Dict[str, List[float]] = {grp: [] for grp in DV_STYLE}
        self.dv_total_kkt_max: List[float] = []
        self.dv_total_kkt_norm: List[float] = []

    def record(
        self,
        iteration: int,
        jax_sim: Any,
        x_scaled: np.ndarray,
        main_script: Any = None,
        kkt_vector: Optional[np.ndarray] = None,
        gradient: Optional[np.ndarray] = None,
        jacobian: Optional[np.ndarray] = None,
        multipliers: Optional[np.ndarray] = None,
    ):
        """Record telemetry for current iteration and compute group feasibility & optimality."""
        self.iterations.append(iteration)
        x_scaled = np.asarray(x_scaled).flatten()

        # ------------------ Compute DV Optimality / Steps ------------------
        if self.prev_x_scaled is None:
            # First iteration: zero step
            dx = np.zeros_like(x_scaled)
        else:
            dx = x_scaled - self.prev_x_scaled
        self.prev_x_scaled = x_scaled.copy()

        self.dv_total_max.append(float(np.max(np.abs(dx))))
        self.dv_total_norm.append(float(np.linalg.norm(dx)))

        for grp, slices in self.group_to_dv_slices.items():
            if grp not in self.dv_steps_max:
                continue
            grp_dx = np.concatenate([dx[s] for s in slices])
            if len(grp_dx) > 0:
                self.dv_steps_max[grp].append(float(np.max(np.abs(grp_dx))))
                self.dv_steps_norm[grp].append(float(np.linalg.norm(grp_dx)))
            else:
                self.dv_steps_max[grp].append(0.0)
                self.dv_steps_norm[grp].append(0.0)

        # ------------------ Compute KKT Stationarity: df/dx + lambda * dc/dx ------------------
        if kkt_vector is None and gradient is not None and jacobian is not None and multipliers is not None:
            g = np.asarray(gradient).flatten()
            jac = np.asarray(jacobian)
            mult = np.asarray(multipliers).flatten()
            if jac.shape[0] == len(mult) and jac.shape[1] == len(g):
                kkt_vector = g - jac.T @ mult
            elif jac.shape[1] == len(mult) and jac.shape[0] == len(g):
                kkt_vector = g - jac @ mult
            else:
                kkt_vector = g

        if kkt_vector is not None:
            kkt_vector = np.asarray(kkt_vector).flatten()
            self.has_kkt_optimality = True
            self.dv_total_kkt_max.append(float(np.max(np.abs(kkt_vector))))
            self.dv_total_kkt_norm.append(float(np.linalg.norm(kkt_vector)))
            for grp, slices in self.group_to_dv_slices.items():
                if grp not in self.dv_kkt_max:
                    continue
                grp_kkt = np.concatenate([kkt_vector[s] for s in slices])
                if len(grp_kkt) > 0:
                    self.dv_kkt_max[grp].append(float(np.max(np.abs(grp_kkt))))
                    self.dv_kkt_norm[grp].append(float(np.linalg.norm(grp_kkt)))
                else:
                    self.dv_kkt_max[grp].append(0.0)
                    self.dv_kkt_norm[grp].append(0.0)

        # ------------------ Compute Constraint Feasibility ------------------
        # Helper to extract value safely from jax_sim or script
        def get_val(var_name_or_obj, default=None):
            try:
                if var_name_or_obj is None:
                    return default
                if isinstance(jax_sim, dict):
                    if var_name_or_obj in jax_sim:
                        val = np.asarray(jax_sim[var_name_or_obj]).flatten()
                        return val if val.size > 0 else default
                    elif str(var_name_or_obj) in jax_sim:
                        val = np.asarray(jax_sim[str(var_name_or_obj)]).flatten()
                        return val if val.size > 0 else default
                val = np.asarray(jax_sim[var_name_or_obj]).flatten()
                return val if val.size > 0 else default
            except Exception:
                if main_script and hasattr(main_script, str(var_name_or_obj)):
                    try:
                        return np.asarray(jax_sim[getattr(main_script, str(var_name_or_obj))]).flatten()
                    except Exception:
                        pass
                return default

        # 1. Separation constraints
        v_sep = []
        strat_val = get_val('dv_stratford_constraints')
        if strat_val is not None:
            v_sep.append(np.maximum(0.0, strat_val - 0.39) / 0.39)
        h_val = get_val('dv_H_stations')
        if h_val is not None and len(v_sep) == 0:
            v_sep.append(np.maximum(0.0, h_val - 2.40) / 2.40)
        v_sep_arr = np.concatenate(v_sep) if v_sep else np.array([0.0])
        self.con_violations_max['Separation'].append(float(np.max(v_sep_arr)))
        self.con_violations_norm['Separation'].append(float(np.linalg.norm(v_sep_arr)))

        # 2. Structural Stress constraints
        v_stress = []
        stress_val = get_val('stresses_to_enforce')
        if stress_val is None:
            stress_val = get_val('dv_stresses')
        if stress_val is not None:
            v_stress.append(np.maximum(0.0, stress_val - self.allowable_stress) / self.allowable_stress)
        stress_neg1g = get_val('root_stress_neg1g')
        if stress_neg1g is not None:
            v_stress.append(np.maximum(0.0, stress_neg1g - self.allowable_stress) / self.allowable_stress)
        v_stress_arr = np.concatenate(v_stress) if v_stress else np.array([0.0])
        self.con_violations_max['Structural Stress'].append(float(np.max(v_stress_arr)))
        self.con_violations_norm['Structural Stress'].append(float(np.linalg.norm(v_stress_arr)))

        # 3. Flight Trim constraints
        v_trim = []
        l_trim = get_val('lift_trim')
        if l_trim is not None:
            v_trim.append(np.abs(l_trim) / (5.0 * self.w_ref))
        p_trim = get_val('pitch_trim')
        if p_trim is not None:
            v_trim.append(np.abs(p_trim) / (self.w_ref * self.c_ref))
        l_ss = get_val('lift_ss')
        if l_ss is not None:
            v_trim.append(np.abs(l_ss) / (self.load_factor_val * self.w_ref))
        l_neg1g = get_val('lift_neg1g')
        if l_neg1g is not None:
            v_trim.append(np.abs(l_neg1g) / self.w_ref)
        v_trim_arr = np.concatenate(v_trim) if v_trim else np.array([0.0])
        self.con_violations_max['Flight Trim'].append(float(np.max(v_trim_arr)))
        self.con_violations_norm['Flight Trim'].append(float(np.linalg.norm(v_trim_arr)))

        # 4. Static Margin
        sm_val = get_val('static_margin')
        if sm_val is not None:
            v_sm = np.maximum(0.0, 0.02 - sm_val[0]) * 20.0
            self.con_violations_max['Static Margin'].append(float(v_sm))
            self.con_violations_norm['Static Margin'].append(float(v_sm))
        else:
            self.con_violations_max['Static Margin'].append(0.0)
            self.con_violations_norm['Static Margin'].append(0.0)

        # 5. Maneuver Attachment
        f_att = get_val('f_attached_ss_near_tip')
        if f_att is not None:
            # Lower limit is typically 0.80 or f_attached_ss_min
            f_min = 0.80
            v_att = np.maximum(0.0, f_min - f_att) / f_min
            self.con_violations_max['Maneuver Attachment'].append(float(np.max(v_att)))
            self.con_violations_norm['Maneuver Attachment'].append(float(np.linalg.norm(v_att)))
        else:
            self.con_violations_max['Maneuver Attachment'].append(0.0)
            self.con_violations_norm['Maneuver Attachment'].append(0.0)

        # 6. Payload Containment
        v_geo = []
        if self.use_geonic:
            g_ov = get_val('geonic_constraint_values')
            if g_ov is not None:
                v_geo.append(np.maximum(0.0, g_ov))
            g_pal = get_val('pallet_constraint_values')
            if g_pal is not None:
                v_geo.append(np.maximum(0.0, g_pal))
            cg_cons = get_val('payload_cg_consistency')
            if cg_cons is not None:
                v_geo.append(np.abs(cg_cons) / self.scale_factor)
        v_geo_arr = np.concatenate(v_geo) if v_geo else np.array([0.0])
        self.con_violations_max['Payload Containment'].append(float(np.max(v_geo_arr)))
        self.con_violations_norm['Payload Containment'].append(float(np.linalg.norm(v_geo_arr)))

        # 7. Geometry Limits
        v_geom = []
        s_val = get_val('planform_area')
        if s_val is not None and self.formulation == 'ar_area':
            target_s = 10.0 * self.scale_factor**2
            v_geom.append(np.array([abs(s_val[0] - target_s) / target_s]))
        ar_calc = get_val('aspect_ratio_calc')
        if ar_calc is not None:
            v_geom.append(np.array([max(0.0, ar_calc[0] - 15.0) * 0.1]))
        tc_r = get_val('tc_ratio')
        if tc_r is not None and 'tc_target_control_points' not in self.design_variables:
            v_geom.append(np.maximum(0.0, 0.06 - tc_r) * 10.0 + np.maximum(0.0, tc_r - 0.35) * 10.0)
        v_geom_arr = np.concatenate(v_geom) if v_geom else np.array([0.0])
        self.con_violations_max['Geometry Limits'].append(float(np.max(v_geom_arr)))
        self.con_violations_norm['Geometry Limits'].append(float(np.linalg.norm(v_geom_arr)))

    def save_and_plot(
        self,
        output_folder: str,
        artifact_dir: Optional[str] = None,
        metric: str = 'max',
    ) -> Tuple[str, str]:
        """Save telemetry npz and generate convergence figure."""
        os.makedirs(output_folder, exist_ok=True)
        data_path = os.path.join(output_folder, 'feasibility_optimality_data.npz')

        save_dict = {
            'iterations': np.array(self.iterations),
            'con_groups': np.array(list(CONSTRAINT_STYLE.keys()), dtype=str),
            'dv_groups': np.array(list(DV_STYLE.keys()), dtype=str),
            'has_group_feasibility': True,
            'has_kkt_lagrangian': self.has_kkt_optimality,
            'dv_total_max': np.array(self.dv_total_max),
            'dv_total_norm': np.array(self.dv_total_norm),
        }
        if self.has_kkt_optimality:
            save_dict['dv_total_kkt_max'] = np.array(self.dv_total_kkt_max)
            save_dict['dv_total_kkt_norm'] = np.array(self.dv_total_kkt_norm)
            for grp in DV_STYLE:
                save_dict[f'dv_{grp}_kkt_max'] = np.array(self.dv_kkt_max[grp])
                save_dict[f'dv_{grp}_kkt_norm'] = np.array(self.dv_kkt_norm[grp])
        for grp in CONSTRAINT_STYLE:
            save_dict[f'con_{grp}_max'] = np.array(self.con_violations_max[grp])
            save_dict[f'con_{grp}_norm'] = np.array(self.con_violations_norm[grp])
        for grp in DV_STYLE:
            save_dict[f'dv_{grp}_max'] = np.array(self.dv_steps_max[grp])
            save_dict[f'dv_{grp}_norm'] = np.array(self.dv_steps_norm[grp])

        # Also store slsqp_summary if present
        summary_path = os.path.join(output_folder, 'slsqp_summary.out')
        summary_dict = parse_slsqp_summary(summary_path)
        if summary_dict is not None:
            for k, v in summary_dict.items():
                save_dict[f'slsqp_{k}'] = v

        np.savez_compressed(data_path, **save_dict)
        print(f"Feasibility and optimality telemetry saved to: {data_path}")

        fig_path = plot_feasibility_and_optimality(
            output_folder=output_folder,
            artifact_dir=artifact_dir,
            metric=metric,
        )
        return data_path, fig_path


def load_convergence_data(output_folder: str) -> Dict[str, Any]:
    """Load convergence data from feasibility_optimality_data.npz, slsqp_recorder.hdf5, or reconstruct from x.out & summary."""
    data_path = os.path.join(output_folder, 'feasibility_optimality_data.npz')
    recorder_path = os.path.join(output_folder, 'slsqp_recorder.hdf5')
    if not os.path.isfile(recorder_path):
        recorder_path = os.path.join(output_folder, 'record.hdf5')
    summary_path = os.path.join(output_folder, 'slsqp_summary.out')
    x_out_path = os.path.join(output_folder, 'x.out')
    struct_path = os.path.join(output_folder, 'structural_data.npz')

    summary_data = parse_slsqp_summary(summary_path)

    # Load DV metadata if available
    dv_meta = {}
    if os.path.isfile(struct_path):
        try:
            s_data = np.load(struct_path, allow_pickle=True)
            if 'dv_metadata_json' in s_data:
                dv_meta = json.loads(str(s_data['dv_metadata_json']))
        except Exception:
            pass

    # Compute DV slices
    slices_by_group: Dict[str, List[slice]] = {grp: [] for grp in DV_STYLE}
    if dv_meta:
        curr = 0
        for name, info in dv_meta.items():
            sz = int(np.prod(info['shape']))
            slc = slice(curr, curr + sz)
            grp = DV_NAME_TO_GROUP.get(name, 'Other DVs')
            if grp in slices_by_group:
                slices_by_group[grp].append(slc)
            curr += sz

    # 1. First priority: precomputed feasibility_optimality_data.npz
    if os.path.isfile(data_path):
        raw = np.load(data_path, allow_pickle=True)
        out = {k: raw[k] for k in raw.files}
        if summary_data and 'slsqp_feas' not in out:
            for k, v in summary_data.items():
                out[f'slsqp_{k}'] = v
        if 'has_group_feasibility' not in out:
            out['has_group_feasibility'] = True
        out['has_kkt_lagrangian'] = bool(out.get('has_kkt_lagrangian', False))
        if out['has_kkt_lagrangian']:
            # Map KKT Lagrangian gradient arrays to active plotted keys
            if 'dv_total_kkt_max' in out:
                out['dv_total_max'] = out['dv_total_kkt_max']
                out['dv_total_norm'] = out['dv_total_kkt_norm']
            for g in DV_STYLE:
                if f'dv_{g}_kkt_max' in out:
                    out[f'dv_{g}_max'] = out[f'dv_{g}_kkt_max']
                    out[f'dv_{g}_norm'] = out[f'dv_{g}_kkt_norm']
        else:
            if 'dv_total_max' not in out:
                all_max = [out[f'dv_{g}_max'] for g in DV_STYLE if f'dv_{g}_max' in out]
                if all_max:
                    out['dv_total_max'] = np.max(np.vstack(all_max), axis=0)
            if 'dv_total_norm' not in out:
                all_norm_sq = [out[f'dv_{g}_norm']**2 for g in DV_STYLE if f'dv_{g}_norm' in out]
                if all_norm_sq:
                    out['dv_total_norm'] = np.sqrt(np.sum(np.vstack(all_norm_sq), axis=0))
        return out

    # 2. Second priority: slsqp_recorder.hdf5 with exact Lagrangian gradients (df/dx + lambda * dc/dx)
    if os.path.isfile(recorder_path):
        try:
            import h5py
            with h5py.File(recorder_path, 'r') as hf:
                iter_keys = sorted(
                    [k for k in hf.keys() if (k.startswith('iter_') and k.split('_')[1].isdigit()) or k.isdigit()],
                    key=lambda k: int(k.split('_')[1]) if '_' in k else int(k)
                )
                if iter_keys and 'gradient' in hf[iter_keys[0]] and 'jacobian' in hf[iter_keys[0]] and 'multipliers' in hf[iter_keys[0]]:
                    kkt_iters = np.array([int(k.split('_')[1]) if '_' in k else int(k) for k in iter_keys])
                    g_lag_list = []
                    for k in iter_keys:
                        grp = hf[k]
                        g = np.asarray(grp['gradient'][:]).flatten()
                        jac = np.asarray(grp['jacobian'][:])
                        mult = np.asarray(grp['multipliers'][:]).flatten()
                        # KKT Lagrangian gradient: grad_L = grad - jac.T @ mult (SLSQP convention)
                        if jac.shape[0] == len(mult) and jac.shape[1] == len(g):
                            glag = g - jac.T @ mult
                        elif jac.shape[1] == len(mult) and jac.shape[0] == len(g):
                            glag = g - jac @ mult
                        else:
                            glag = g
                        g_lag_list.append(glag)
                    g_lag_arr = np.array(g_lag_list)
                    num_iters, nx = g_lag_arr.shape

                    # Partition slices if not already populated
                    if not any(slices_by_group.values()):
                        if nx == 84:
                            slices_by_group['Chord & Planform'] = [slice(0, 6), slice(24, 25)]
                            slices_by_group['Sweep'] = [slice(6, 12)]
                            slices_by_group['Airfoil Thickness'] = [slice(12, 18), slice(60, 84)]
                            slices_by_group['Twist'] = [slice(18, 24)]
                            slices_by_group['Flight Angles & Trim'] = [slice(25, 27)]
                            slices_by_group['Payload & CG'] = [slice(27, 30)]
                            slices_by_group['Structural Sizing'] = [slice(30, 42)]
                            slices_by_group['Airfoil Camber'] = [slice(42, 60)]
                        else:
                            slices_by_group['Chord & Planform'] = [slice(0, nx)]

                    out_data = {
                        'iterations': kkt_iters,
                        'con_groups': np.array(list(CONSTRAINT_STYLE.keys()), dtype=str),
                        'dv_groups': np.array(list(DV_STYLE.keys()), dtype=str),
                        'has_group_feasibility': False,
                        'has_kkt_lagrangian': True,
                        'dv_total_max': np.max(np.abs(g_lag_arr), axis=1),
                        'dv_total_norm': np.linalg.norm(g_lag_arr, axis=1),
                    }
                    for grp, slcs in slices_by_group.items():
                        if slcs:
                            grp_glag = np.concatenate([g_lag_arr[:, s] for s in slcs], axis=1)
                            out_data[f'dv_{grp}_max'] = np.max(np.abs(grp_glag), axis=1)
                            out_data[f'dv_{grp}_norm'] = np.linalg.norm(grp_glag, axis=1)
                        else:
                            out_data[f'dv_{grp}_max'] = np.zeros(num_iters)
                            out_data[f'dv_{grp}_norm'] = np.zeros(num_iters)

                    if summary_data:
                        for k, v in summary_data.items():
                            out_data[f'slsqp_{k}'] = v
                    return out_data
        except Exception:
            pass

    # 3. Fallback: Reconstruct step history from x.out, slsqp_summary.out
    if not os.path.isfile(x_out_path):
        raise FileNotFoundError(f"Neither feasibility_optimality_data.npz, slsqp_recorder.hdf5, nor x.out found in {output_folder}")

    x_hist = np.loadtxt(x_out_path)
    if x_hist.ndim == 1:
        x_hist = x_hist.reshape(1, -1)
    num_iters, nx = x_hist.shape
    iters = np.arange(num_iters)

    # Compute fallback slices if metadata was not available
    if not any(slices_by_group.values()):
        if nx == 84:
            slices_by_group['Chord & Planform'] = [slice(0, 6), slice(24, 25)]
            slices_by_group['Sweep'] = [slice(6, 12)]
            slices_by_group['Airfoil Thickness'] = [slice(12, 18), slice(60, 84)]
            slices_by_group['Twist'] = [slice(18, 24)]
            slices_by_group['Flight Angles & Trim'] = [slice(25, 27)]
            slices_by_group['Payload & CG'] = [slice(27, 30)]
            slices_by_group['Structural Sizing'] = [slice(30, 42)]
            slices_by_group['Airfoil Camber'] = [slice(42, 60)]
        else:
            slices_by_group['Chord & Planform'] = [slice(0, nx)]

    # Compute step differences
    dx = np.zeros_like(x_hist)
    dx[1:] = x_hist[1:] - x_hist[:-1]

    out_data = {
        'iterations': iters,
        'con_groups': np.array(list(CONSTRAINT_STYLE.keys()), dtype=str),
        'dv_groups': np.array(list(DV_STYLE.keys()), dtype=str),
        'has_group_feasibility': False,
        'has_kkt_lagrangian': False,
        'dv_total_max': np.max(np.abs(dx), axis=1),
        'dv_total_norm': np.linalg.norm(dx, axis=1),
    }

    for grp, slcs in slices_by_group.items():
        if slcs:
            grp_dx = np.concatenate([dx[:, s] for s in slcs], axis=1)
            out_data[f'dv_{grp}_max'] = np.max(np.abs(grp_dx), axis=1)
            out_data[f'dv_{grp}_norm'] = np.linalg.norm(grp_dx, axis=1)
        else:
            out_data[f'dv_{grp}_max'] = np.zeros(num_iters)
            out_data[f'dv_{grp}_norm'] = np.zeros(num_iters)

    # Store slsqp_summary if present
    if summary_data:
        for k, v in summary_data.items():
            out_data[f'slsqp_{k}'] = v

    return out_data


def generate_convergence_plot(
    data: Dict[str, Any],
    output_fig_path: str,
    metric: str = 'max',
    dpi: int = 200,
) -> str:
    """Generate the two-panel Feasibility and Optimality convergence history plot."""
    metric = metric.lower()
    if metric not in ('max', 'norm'):
        metric = 'max'

    metric_suffix = 'max' if metric == 'max' else 'norm'
    metric_label = 'Max Violation / Step' if metric == 'max' else 'L2 Norm'

    iters = np.asarray(data['iterations'])
    num_iters = len(iters)

    # Floor for log scale plotting
    log_floor = 1e-7

    # Setup publication styling
    fig = plt.figure(figsize=(13, 9), dpi=dpi)
    gs = GridSpec(2, 1, height_ratios=[1.0, 1.0], hspace=0.28)

    bg_color = '#ffffff'
    plot_bg = '#fcfdfe'
    grid_color = '#e2e8f0'
    text_color = '#1e293b'

    fig.patch.set_facecolor(bg_color)

    # ------------------ TOP PANEL: Feasibility History ------------------
    ax_feas = fig.add_subplot(gs[0, 0])
    ax_feas.set_facecolor(plot_bg)
    ax_feas.grid(True, which='both', color=grid_color, linestyle='-', linewidth=0.8, alpha=0.8)

    # Plot each constraint group
    active_con_plotted = 0
    mark_every = max(1, num_iters // 25)

    has_group_feas = data.get('has_group_feasibility', True)
    if has_group_feas:
        for grp, style in CONSTRAINT_STYLE.items():
            key = f'con_{grp}_{metric_suffix}'
            if key in data:
                vals = np.asarray(data[key])
                if len(vals) == num_iters and np.any(np.isfinite(vals)):
                    # If entire history is strictly zero, skip or plot near floor
                    max_observed = np.max(vals)
                    if max_observed > 0.0:
                        clipped_vals = np.maximum(vals, log_floor)
                        ax_feas.semilogy(
                            iters, clipped_vals,
                            color=style['color'],
                            marker=style['marker'],
                            markersize=5,
                            markevery=mark_every,
                            linewidth=1.8,
                            alpha=0.9,
                            label=style['label'],
                        )
                        active_con_plotted += 1
    else:
        ax_feas.text(
            0.5, 0.88,
            'Note: Per-group constraint breakdown was not recorded for this historical run.\nShowing overall solver feasibility trajectory (FEAS). Replay through simulator to generate group breakdown.',
            transform=ax_feas.transAxes,
            ha='center', va='center',
            fontsize=9.5, color='#334155',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8fafc', edgecolor='#cbd5e1', alpha=0.95)
        )

    # Plot SLSQP overall feasibility if available
    if 'slsqp_feas' in data:
        slsqp_feas = np.asarray(data['slsqp_feas'])
        s_iters = data.get('slsqp_major', np.arange(len(slsqp_feas)))
        # Filter out initial 99.0 code
        valid_mask = (slsqp_feas < 90.0) & (slsqp_feas >= 0.0)
        if np.any(valid_mask):
            ax_feas.semilogy(
                s_iters[valid_mask], np.maximum(slsqp_feas[valid_mask], log_floor),
                color='#0f172a',
                linestyle='--',
                linewidth=2.4,
                alpha=0.85,
                label='SLSQP Total Feasibility (FEAS)',
            )

    # Feasibility tolerance guideline
    ax_feas.axhline(1e-5, color='#64748b', linestyle=':', linewidth=1.4, alpha=0.7)
    ax_feas.text(
        0.015, 1.3e-5, 'Feasibility Tolerance (1e-5)',
        transform=ax_feas.get_yaxis_transform(),
        color='#475569', fontsize=9, fontweight='medium', va='bottom'
    )

    ax_feas.set_ylabel(f'Constraint Feasibility [{metric_label}]', fontsize=11, fontweight='bold', color=text_color)
    ax_feas.set_title(
        f'BWB Optimization Convergence History — Feasibility & Optimality ({metric_label} per Group)',
        fontsize=13, fontweight='bold', color=text_color, pad=12
    )
    ax_feas.set_ylim(bottom=log_floor * 0.5, top=max(10.0, ax_feas.get_ylim()[1]))
    h_feas, l_feas = ax_feas.get_legend_handles_labels()
    if h_feas:
        ax_feas.legend(loc='upper right', frameon=True, framealpha=0.95, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=9, ncol=2)

    # ------------------ BOTTOM PANEL: Optimality History ------------------
    ax_opt = fig.add_subplot(gs[1, 0], sharex=ax_feas)
    ax_opt.set_facecolor(plot_bg)
    ax_opt.grid(True, which='both', color=grid_color, linestyle='-', linewidth=0.8, alpha=0.8)

    # Plot each design variable group
    # Note: step at iter 0 is 0; start plotting from iter 1
    plot_slice = slice(1, None) if num_iters > 1 else slice(0, None)
    plot_iters = iters[plot_slice]
    mark_every_opt = max(1, len(plot_iters) // 25)

    has_kkt = data.get('has_kkt_lagrangian', False)

    for grp, style in DV_STYLE.items():
        if has_kkt and f'dv_{grp}_kkt_{metric_suffix}' in data:
            key = f'dv_{grp}_kkt_{metric_suffix}'
        else:
            key = f'dv_{grp}_{metric_suffix}'
        if key in data:
            vals = np.asarray(data[key])[plot_slice]
            if len(vals) == len(plot_iters) and np.any(np.isfinite(vals)):
                max_observed = np.max(vals)
                if max_observed > 0.0:
                    clipped_vals = np.maximum(vals, log_floor)
                    ax_opt.semilogy(
                        plot_iters, clipped_vals,
                        color=style['color'],
                        marker=style['marker'],
                        markersize=5,
                        markevery=mark_every_opt,
                        linewidth=1.8,
                        alpha=0.9,
                        label=style['label'],
                    )

    # Plot Total across ALL DVs (Mathematically >= all individual groups)
    if has_kkt and f'dv_total_kkt_{metric_suffix}' in data:
        total_key = f'dv_total_kkt_{metric_suffix}'
        total_label = rf'Total Optimality $\left\|\nabla_x \mathcal{{L}}\right\|$ ({metric_label})'
    elif f'dv_total_{metric_suffix}' in data:
        total_key = f'dv_total_{metric_suffix}'
        total_label = rf'Total Optimality $\left\|\nabla_x \mathcal{{L}}\right\|$ ({metric_label})' if has_kkt else f'Total Step (All DVs, {metric_label})'
    else:
        total_key = None

    if total_key and total_key in data:
        total_vals = np.asarray(data[total_key])[plot_slice]
        if len(total_vals) == len(plot_iters) and np.any(np.isfinite(total_vals)):
            clipped_total = np.maximum(total_vals, log_floor)
            ax_opt.semilogy(
                plot_iters, clipped_total,
                color='#0f172a',
                linestyle='-',
                linewidth=2.5,
                alpha=0.95,
                label=total_label,
            )

    # Plot SLSQP overall solver optimality (h1 = merit function multiplier residual) if available
    if 'slsqp_opt' in data:
        slsqp_opt = np.asarray(data['slsqp_opt'])
        s_iters = data.get('slsqp_major', np.arange(len(slsqp_opt)))
        valid_mask = (slsqp_opt < 90.0) & (slsqp_opt >= 0.0) & (s_iters >= (1 if num_iters > 1 else 0))
        if np.any(valid_mask):
            ax_opt.semilogy(
                s_iters[valid_mask], np.maximum(slsqp_opt[valid_mask], log_floor),
                color='#64748b',
                linestyle='--',
                linewidth=1.8,
                alpha=0.85,
                label=r'SLSQP Merit Residual (h1 = $\mu^T \max(0, -c)$)',
            )

    # Optimality tolerance guideline
    ax_opt.axhline(1e-5, color='#64748b', linestyle=':', linewidth=1.4, alpha=0.7)
    ax_opt.text(
        0.015, 1.3e-5, 'Optimality Tolerance (1e-5)',
        transform=ax_opt.get_yaxis_transform(),
        color='#475569', fontsize=9, fontweight='medium', va='bottom'
    )

    if has_kkt:
        ax_opt.set_ylabel(rf'First-Order Optimality: $\left\|\frac{{\partial f}}{{\partial x}} + \lambda^T \frac{{\partial c}}{{\partial x}}\right\|$ [{metric_label}]', fontsize=11, fontweight='bold', color=text_color)
    else:
        ax_opt.set_ylabel(f'Design Variable Step [Δx, {metric_label}]', fontsize=11, fontweight='bold', color=text_color)
        ax_opt.text(
            0.5, 0.88,
            'Note: First-order KKT gradient (df/dx + lambda*dc/dx) was not recorded for this historical run.\nShowing design variable step displacement (Δx). Future runs record full KKT gradient automatically via slsqp_recorder.hdf5.',
            transform=ax_opt.transAxes,
            ha='center', va='center',
            fontsize=9.5, color='#334155',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8fafc', edgecolor='#cbd5e1', alpha=0.95)
        )
    ax_opt.set_xlabel('Major Optimization Iteration', fontsize=11, fontweight='bold', color=text_color)
    ax_opt.set_ylim(bottom=log_floor * 0.5, top=max(10.0, ax_opt.get_ylim()[1]))
    ax_opt.legend(loc='upper right', frameon=True, framealpha=0.95, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=9, ncol=2)

    # Clean borders
    for ax in [ax_feas, ax_opt]:
        for spine in ax.spines.values():
            spine.set_color('#94a3b8')
            spine.set_linewidth(1.0)

    plt.savefig(output_fig_path, bbox_inches='tight', dpi=dpi, facecolor=bg_color)
    plt.close()
    return output_fig_path


def plot_feasibility_and_optimality(
    output_folder: Optional[str] = None,
    artifact_dir: Optional[str] = None,
    metric: str = 'max',
    show: bool = False,
) -> str:
    """Find target folder, load data, generate convergence plots, and copy to artifact directory."""
    target_folder = find_target_output_dir(output_folder)
    print(f"Reading optimization convergence data from: {target_folder}")

    data = load_convergence_data(target_folder)

    primary_fig = os.path.join(target_folder, 'optimization_feasibility_optimality.png')

    if metric == 'both':
        fig_max = os.path.join(target_folder, 'optimization_feasibility_optimality_max.png')
        fig_norm = os.path.join(target_folder, 'optimization_feasibility_optimality_norm.png')
        generate_convergence_plot(data, fig_max, metric='max')
        generate_convergence_plot(data, fig_norm, metric='norm')
        # Default primary is max
        shutil.copy2(fig_max, primary_fig)
        print(f"Generated convergence plots:\n  {fig_max}\n  {fig_norm}\n  {primary_fig}")
    else:
        generate_convergence_plot(data, primary_fig, metric=metric)
        print(f"Convergence plot successfully generated: {primary_fig}")

    # Copy to brain artifact directory if specified or found
    if not artifact_dir:
        artifact_dir = os.environ.get('ARTIFACT_DIR', None)
        if not artifact_dir or not os.path.exists(artifact_dir):
            candidate_ids = [
                '09524c37-f7d1-40a1-85ac-170d91a90d0c',
                'ae98c491-cb6a-4455-8a09-a0df739801c2',
                '0ac41e2d-0335-41d9-9f48-d6791df931f1',
            ]
            for c_id in candidate_ids:
                cand = os.path.expanduser(f'~/.gemini/antigravity/brain/{c_id}')
                if os.path.isdir(cand):
                    artifact_dir = cand
                    break

    if artifact_dir and os.path.isdir(artifact_dir):
        dest_fig = os.path.join(artifact_dir, 'optimization_feasibility_optimality.png')
        shutil.copy2(primary_fig, dest_fig)
        print(f"Copied convergence plot to artifact directory: {dest_fig}")
        folder_tag = os.path.basename(os.path.normpath(target_folder))
        shutil.copy2(primary_fig, os.path.join(artifact_dir, f'optimization_feasibility_optimality_{folder_tag}.png'))
        if metric == 'both':
            shutil.copy2(fig_max, os.path.join(artifact_dir, 'optimization_feasibility_optimality_max.png'))
            shutil.copy2(fig_norm, os.path.join(artifact_dir, 'optimization_feasibility_optimality_norm.png'))
            shutil.copy2(fig_max, os.path.join(artifact_dir, f'optimization_feasibility_optimality_max_{folder_tag}.png'))
            shutil.copy2(fig_norm, os.path.join(artifact_dir, f'optimization_feasibility_optimality_norm_{folder_tag}.png'))

    return primary_fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot optimization feasibility and optimality histories by group.")
    parser.add_argument('output_dir', nargs='?', default=None, help="Path to optimization output folder")
    parser.add_argument('--metric', choices=['max', 'norm', 'both'], default='max', help="Group aggregation metric (default: max)")
    parser.add_argument('--artifact-dir', default=None, help="Optional artifact directory to copy figures to")
    args = parser.parse_args()

    plot_feasibility_and_optimality(
        output_folder=args.output_dir,
        artifact_dir=args.artifact_dir,
        metric=args.metric,
    )

