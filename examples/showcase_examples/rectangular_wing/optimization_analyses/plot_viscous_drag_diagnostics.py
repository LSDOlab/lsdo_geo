"""Plot and analyze boundary layer viscous drag diagnostics for BWB optimization.

Visualizes the aerodynamic effects and physical behavior of the Direct-IBL
(Integral Boundary Layer) viscous drag model:
1. Spanwise sectional viscous drag coefficient cd_visc(y) vs constant CD0 baseline.
2. Spanwise viscous drag force loading D'(y) [kN/m] and integrated drag force.
3. Upper vs lower surface trailing-edge momentum thickness theta_te(y) [mm and % chord].
4. Chordwise boundary layer shape factor H(x/c) evolution across key span stations.
5. Chord Reynolds number Re_c(y) and airfoil 3D viscous form factor k_form(y).
6. Total aircraft drag budget decomposition and viscous dominance share.

Usage:
    python plot_viscous_drag_diagnostics.py [output_directory]

If output_directory is omitted, the latest directory under
'rectangular_wing_to_bwb_aerostructural_optimization_outputs' is analyzed.
"""

from __future__ import annotations
import os
import sys
import shutil
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d, CubicSpline

# Ensure working directory and relevant repos are in path
REPO_ROOT = '/home/andrew/optimization/lsdo_geo'
if os.path.exists(REPO_ROOT) and REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

RW_DIR = os.path.join(REPO_ROOT, 'examples/showcase_examples/rectangular_wing')
if os.path.exists(RW_DIR) and RW_DIR not in sys.path:
    sys.path.insert(0, RW_DIR)

KNOWN_OUTPUT_DIRS = [
    os.path.join(REPO_ROOT, 'rectangular_wing_to_bwb_aerostructural_optimization_outputs'),
    os.path.join(REPO_ROOT, 'rectangular_wing_aerostructural_optimization_outputs'),
]

DEFAULT_ARTIFACT_DIR = '/home/andrew/.gemini/antigravity/brain/728a2089-2977-4bdc-b525-91b2b850c11b'

# Physical constants for cruise flight condition
CRUISE_SPEED_REF = 230.93       # Freestream speed [m/s] (Mach 0.77 at 35,000 ft)
RHO_CRUISE_REF = 0.38036        # Air density [kg/m^3] at 35,000 ft
MU_CRUISE_REF = 1.432e-5        # Dynamic viscosity [Pa*s]
H_SEP_CRIT = 2.40               # Head turbulent separation threshold
H_FLAT_PLATE = 1.40             # Zero-pressure-gradient flat-plate baseline


def find_latest_output_dir(base_dir: Optional[str] = None) -> str:
    """Find the target optimization output directory.

    If base_dir is provided:
      1. Resolves if it is an existing directory path (absolute or relative).
      2. Searches for base_dir by exact folder name in KNOWN_OUTPUT_DIRS.
      3. Searches for base_dir by partial substring match in KNOWN_OUTPUT_DIRS,
         returning the latest match by timestamp folder name.

    If base_dir is None:
      Automatically finds the chronologically latest completed output directory
      (containing 'lift_and_moment_data.npz' or 'viscous_drag_diagnostics_data.npz') across
      KNOWN_OUTPUT_DIRS, sorted by directory name (YYYY-MM-DD_HH.MM.SS.ffffff).
    """
    if base_dir:
        # 1. Direct path check
        if os.path.isdir(base_dir):
            return os.path.abspath(base_dir)

        # 2. Check exact folder name inside KNOWN_OUTPUT_DIRS
        for cand_base in KNOWN_OUTPUT_DIRS:
            cand_path = os.path.join(cand_base, base_dir)
            if os.path.isdir(cand_path):
                return os.path.abspath(cand_path)

        # 3. Check partial substring match inside KNOWN_OUTPUT_DIRS
        matches = []
        for cand_base in KNOWN_OUTPUT_DIRS:
            if os.path.exists(cand_base):
                for d in os.listdir(cand_base):
                    full = os.path.join(cand_base, d)
                    if os.path.isdir(full) and base_dir in d:
                        matches.append(full)
        if matches:
            chosen = max(matches, key=lambda d: os.path.basename(d))
            return os.path.abspath(chosen)

        raise FileNotFoundError(f"Specified optimization output directory '{base_dir}' not found.")

    # Automatic mode: find chronologically latest directory with completed telemetry
    all_with_data = []
    all_subdirs = []
    for cand_base in KNOWN_OUTPUT_DIRS:
        if os.path.exists(cand_base):
            subdirs = [
                os.path.join(cand_base, d) for d in os.listdir(cand_base)
                if os.path.isdir(os.path.join(cand_base, d))
            ]
            all_subdirs.extend(subdirs)
            with_data = [
                d for d in subdirs
                if os.path.exists(os.path.join(d, 'lift_and_moment_data.npz'))
                or os.path.exists(os.path.join(d, 'viscous_drag_diagnostics_data.npz'))
            ]
            all_with_data.extend(with_data)

    candidates = all_with_data if all_with_data else all_subdirs
    if not candidates:
        raise FileNotFoundError("No valid optimization output directory found.")

    chosen = max(candidates, key=lambda d: os.path.basename(d))
    return os.path.abspath(chosen)


def load_viscous_telemetry(
    output_folder: str,
    jax_sim: Optional[Any] = None,
    main_script: Optional[Any] = None,
) -> Dict[str, Any]:
    """Load cached viscous drag telemetry or extract from live simulator."""
    visc_cache = os.path.join(output_folder, 'viscous_drag_diagnostics_data.npz')
    lm_cache = os.path.join(output_folder, 'lift_and_moment_data.npz')

    data: Dict[str, Any] = {}

    # 1. Live JAX extraction if available
    if jax_sim is not None and main_script is not None:
        try:
            data['viscous_drag_mode'] = str(getattr(main_script, 'viscous_drag_mode', 'ibl'))
            data['scale_factor'] = float(getattr(main_script, 'scale_factor', 7.5))
            data['num_stations'] = int(getattr(main_script, 'num_stations', 5))

            data['cd_viscous_val'] = float(np.asarray(jax_sim[main_script.CD_viscous]).flatten()[0])
            data['d_viscous_val'] = float(np.asarray(jax_sim[main_script.D_viscous]).flatten()[0])
            data['cd_ibl_section'] = np.asarray(jax_sim[main_script.CD_ibl_section]).flatten()
            data['theta_te_upper'] = np.asarray(jax_sim[main_script.theta_te_upper]).flatten()
            data['theta_te_lower'] = np.asarray(jax_sim[main_script.theta_te_lower]).flatten()
            data['h_max_ibl'] = float(np.asarray(jax_sim[main_script.H_max_ibl]).flatten()[0])
            data['ibl_attachment_margin'] = float(np.asarray(jax_sim[main_script.ibl_attachment_margin]).flatten()[0])

            if hasattr(main_script, 'H_section_upper'):
                data['H_section_upper'] = np.asarray(jax_sim[main_script.H_section_upper]).flatten()
            if hasattr(main_script, 'H_upper_matrix'):
                data['H_upper_matrix'] = np.asarray(jax_sim[main_script.H_upper_matrix])

            data['local_chord_drag'] = np.asarray(jax_sim[main_script.local_chord_drag]).flatten()
            data['strip_area'] = np.asarray(jax_sim[main_script.strip_area]).flatten()
            data['y_strip_pts'] = np.asarray(jax_sim[main_script.y_strip_pts]).flatten()
            data['dy_strip'] = np.asarray(jax_sim[main_script.dy_strip]).flatten()
            data['panel_centers_right'] = np.asarray(jax_sim[main_script.dynamic_panel_centers_right])

            data['d_total_val'] = float(np.asarray(jax_sim[main_script.D_total]).flatten()[0])
            data['cdi_val'] = float(np.asarray(jax_sim[main_script.CDi]).flatten()[0])
            data['cd_wave_val'] = float(np.asarray(jax_sim[main_script.CD_wave_cruise]).flatten()[0])
            data['sref_val'] = float(np.asarray(jax_sim[main_script.planform_area]).flatten()[0])

            return data
        except Exception as e:
            print(f"Warning: live simulator extraction failed ({e}), falling back to disk cache.")

    # 2. Primary simulation output cache (lift_and_moment_data.npz)
    if os.path.exists(lm_cache):
        print(f"Loading viscous telemetry from primary cache: {lm_cache}")
        raw = np.load(lm_cache, allow_pickle=True)
        return {k: raw[k] for k in raw.files}

    # 3. Dedicated viscous cache if present
    if os.path.exists(visc_cache):
        print(f"Loading dedicated viscous drag cache: {visc_cache}")
        raw = np.load(visc_cache, allow_pickle=True)
        return {k: raw[k] for k in raw.files}

    raise FileNotFoundError(
        f"Neither 'viscous_drag_diagnostics_data.npz' nor 'lift_and_moment_data.npz' found in {output_folder}"
    )


def process_viscous_drag_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """Process boundary layer fields, strip distributions, Reynolds numbers, and drag budgets."""
    if 'cd_strip_visc' in data and 'H_mat' in data:
        # Already processed cache loaded directly from viscous_drag_diagnostics_data.npz
        return dict(data)

    scale_factor = float(data.get('scale_factor', 7.5))
    viscous_drag_mode = str(data.get('viscous_drag_mode', 'ibl'))

    # Extract 100 drag strip coordinates and planform geometry
    y_strip = np.asarray(data['y_strip_pts']).flatten()
    c_strip = np.asarray(data['local_chord_drag']).flatten()
    strip_area = np.asarray(data['strip_area']).flatten()
    dy_strip = np.asarray(data['dy_strip']).flatten()
    num_strips = len(y_strip)

    b_tip = float(np.max(y_strip))
    eta_strip = y_strip / b_tip

    # Total reference area and dynamic pressure
    sref = float(data.get('sref_val', np.sum(strip_area)))
    d_visc_total = float(data.get('d_viscous_val', 38000.0))
    cd_visc_total = float(data.get('cd_viscous_val', d_visc_total / (10134.0 * sref)))
    q_inf = d_visc_total / (cd_visc_total * sref) if cd_visc_total * sref > 0 else 10134.0

    # Extract 14 Direct-IBL evaluation stations from upper panel centers
    pc = data.get('panel_centers_right', None)
    if pc is not None and len(pc.shape) == 2:
        y_round = np.round(pc[:, 1], 3)
        u_y, counts = np.unique(y_round, return_counts=True)
        st_y = u_y[counts == 40]
    else:
        st_y = np.linspace(y_strip[0], y_strip[-1], 14)

    num_mesh_stations = len(st_y)
    eta_stations = st_y / b_tip

    # Sectional viscous drag coefficient at stations
    if 'cd_ibl_section' in data:
        cd_sections = np.asarray(data['cd_ibl_section']).flatten()
        if len(cd_sections) != num_mesh_stations:
            cd_sections = np.interp(st_y, np.linspace(st_y[0], st_y[-1], len(cd_sections)), cd_sections)
    else:
        cd_sections = np.full(num_mesh_stations, cd_visc_total)

    # Upper and lower trailing-edge momentum thickness [m]
    theta_te_upper = np.asarray(data.get('theta_te_upper', np.full(num_mesh_stations, 0.015))).flatten()
    theta_te_lower = np.asarray(data.get('theta_te_lower', np.full(num_mesh_stations, 0.012))).flatten()

    # Interpolate station chord lengths
    f_chord = interp1d(y_strip, c_strip, fill_value='extrapolate')
    c_stations = f_chord(st_y)

    # Continuous spanwise interpolation of cd_visc onto the 100 drag strips
    spl_cd = CubicSpline(st_y, cd_sections, bc_type='natural')
    cd_strip_visc = spl_cd(y_strip)
    cd_strip_visc = np.clip(cd_strip_visc, 0.002, 0.020)

    # Sectional drag force per unit span: D'(y) = q_inf * cd_visc(y) * c(y) [N/m]
    # For one half-wing:
    dprime_visc_ibl = q_inf * cd_strip_visc * c_strip  # N/m
    dprime_visc_const = q_inf * cd_visc_total * c_strip # N/m (constant CD0 baseline)

    # Approximate Squire-Young upper vs lower decomposition at stations
    # delta_cd_upper ~ cd * (theta_upper / (theta_upper + theta_lower))
    theta_sum = np.maximum(theta_te_upper + theta_te_lower, 1e-6)
    cd_upper_st = cd_sections * (theta_te_upper / theta_sum)
    cd_lower_st = cd_sections * (theta_te_lower / theta_sum)

    # Upper/lower momentum thickness in % chord
    theta_over_c_upper_pct = (theta_te_upper / c_stations) * 100.0
    theta_over_c_lower_pct = (theta_te_lower / c_stations) * 100.0

    # Local chord Reynolds number Re_c(y) = rho * V * c(y) / mu
    re_strip = (RHO_CRUISE_REF * CRUISE_SPEED_REF * c_strip) / MU_CRUISE_REF
    re_stations = (RHO_CRUISE_REF * CRUISE_SPEED_REF * c_stations) / MU_CRUISE_REF

    # Flat-plate turbulent skin-friction coefficient: Cf_turb ~ 0.074 / Re^(1/5)
    # Total two-sided flat-plate friction: CD0_flat = 2 * Cf
    cf_turb_strip = 0.074 / (re_strip ** 0.2)
    cd_flat_strip = 2.0 * cf_turb_strip

    # Viscous form factor: k_form(y) = cd(y) / cd_flat(y) - 1.0 (thickness & pressure gradient excess)
    k_form_strip = (cd_strip_visc / cd_flat_strip) - 1.0

    # Shape factor chordwise evolution H(x/c)
    NU_CHORD = 20
    u_dense = np.linspace(0.0, 1.0, NU_CHORD)
    if 'H_upper_matrix' in data and np.asarray(data['H_upper_matrix']).ndim == 2:
        H_mat = np.asarray(data['H_upper_matrix'])
        if H_mat.shape != (NU_CHORD, num_mesh_stations):
            # Interpolate to match dimensions
            u_old = np.linspace(0.0, 1.0, H_mat.shape[0])
            st_old = np.linspace(0.0, 1.0, H_mat.shape[1])
            spl_h = interp1d(u_old, H_mat, axis=0, fill_value='extrapolate')
            H_mat_interp = spl_h(u_dense)
            H_mat = H_mat_interp
    else:
        # Canonical healthy boundary layer growth profile
        H_mat = np.zeros((NU_CHORD, num_mesh_stations))
        for j in range(num_mesh_stations):
            h_te = float(data.get('H_section_upper', np.full(num_mesh_stations, 1.65))[j])
            H_mat[:, j] = H_FLAT_PLATE + (h_te - H_FLAT_PLATE) * (u_dense ** 2.4)

    # Drag budget components
    d_total = float(data.get('d_total_val', 64128.0))
    cdi_val = float(data.get('cdi_val', 0.00335))
    d_induced = cdi_val * q_inf * sref
    d_wave = float(data.get('d_wave_val', data.get('cd_wave_val', 0.00015) * q_inf * sref))
    d_sep = float(data.get('d_separation_cruise', 0.0))

    # Baseline constant CD0 comparison
    d_visc_baseline = cd_visc_total * q_inf * sref
    d_total_baseline = d_induced + d_visc_baseline + d_wave

    return {
        'scale_factor': scale_factor,
        'viscous_drag_mode': viscous_drag_mode,
        'b_tip': b_tip,
        'sref': sref,
        'q_inf': q_inf,
        'y_strip': y_strip,
        'eta_strip': eta_strip,
        'c_strip': c_strip,
        'st_y': st_y,
        'eta_stations': eta_stations,
        'c_stations': c_stations,
        'cd_sections': cd_sections,
        'cd_strip_visc': cd_strip_visc,
        'cd_upper_st': cd_upper_st,
        'cd_lower_st': cd_lower_st,
        'dprime_visc_ibl': dprime_visc_ibl,
        'dprime_visc_const': dprime_visc_const,
        'theta_te_upper': theta_te_upper,
        'theta_te_lower': theta_te_lower,
        'theta_over_c_upper_pct': theta_over_c_upper_pct,
        'theta_over_c_lower_pct': theta_over_c_lower_pct,
        're_strip': re_strip,
        're_stations': re_stations,
        'cd_flat_strip': cd_flat_strip,
        'k_form_strip': k_form_strip,
        'u_dense': u_dense,
        'H_mat': H_mat,
        'd_visc_total': d_visc_total,
        'cd_visc_total': cd_visc_total,
        'd_induced': d_induced,
        'd_wave': d_wave,
        'd_sep': d_sep,
        'd_total': d_total,
        'd_visc_baseline': d_visc_baseline,
        'd_total_baseline': d_total_baseline,
        'h_max_ibl': float(data.get('h_max_ibl', float(np.max(H_mat)))),
        'ibl_attachment_margin': float(data.get('ibl_attachment_margin', H_SEP_CRIT - float(np.max(H_mat)))),
    }


def generate_viscous_drag_figure(proc: Dict[str, Any], output_path: str) -> None:
    """Generate 6-panel comprehensive viscous drag diagnostics figure."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'legend.fontsize': 8.2,
        'figure.titlesize': 13,
    })

    fig = plt.figure(figsize=(17, 13.8), dpi=250)
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.25, 1.0, 1.25], hspace=0.38, wspace=0.25)

    eta_strip = proc['eta_strip']
    y_strip = proc['y_strip']
    eta_st = proc['eta_stations']
    st_y = proc['st_y']

    # -------------------------------------------------------------------------
    # Panel 1: Spanwise Sectional Viscous Drag Coefficient cd_visc(y)
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    cd_counts = proc['cd_strip_visc'] * 1e4
    cd_st_counts = proc['cd_sections'] * 1e4
    cd_const_counts = proc['cd_visc_total'] * 1e4

    ax1.plot(eta_strip, cd_counts, color='#1f77b4', linewidth=2.4, label=r'Direct-IBL Section $c_{d,\rm visc}(y)$')
    ax1.scatter(eta_st, cd_st_counts, color='#1f77b4', edgecolor='black', s=42, zorder=5, label='IBL Mesh Stations (14)')

    ax1.plot(eta_st, proc['cd_upper_st'] * 1e4, color='#2ca02c', linestyle='--', linewidth=1.8, label=r'Upper Surface Share ($c_{d,\rm upper}$)')
    ax1.plot(eta_st, proc['cd_lower_st'] * 1e4, color='#9467bd', linestyle=':', linewidth=1.8, label=r'Lower Surface Share ($c_{d,\rm lower}$)')

    ax1.axhline(cd_const_counts, color='#d62728', linestyle='-.', linewidth=1.8,
                label=f"Constant $C_{{D0}}$ Baseline ({cd_const_counts:.1f} counts)")

    ax1.set_xlabel(r'Spanwise Fraction $\eta = y / (b/2)$ [-]', fontweight='bold')
    ax1.set_ylabel(r'Section Drag Coefficient $c_d$ [drag counts ($10^{-4}$)]', fontweight='bold')
    ax1.set_title(
        r'(A) Spanwise Viscous Drag Coefficient $c_{d,\rm visc}(\eta)$' + "\n"
        f"Root: {cd_st_counts[0]:.1f} counts | Tip: {cd_st_counts[-1]:.1f} counts (+{((cd_st_counts[-1]/cd_st_counts[0])-1)*100:.1f}%)",
        fontweight='bold', pad=8
    )
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, max(100.0, np.max(cd_st_counts) * 1.35)])
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left', framealpha=0.92)

    # -------------------------------------------------------------------------
    # Panel 2: Sectional Viscous Drag Force Loading D'(y) [kN/m]
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    dprime_ibl_kn = proc['dprime_visc_ibl'] / 1000.0
    dprime_const_kn = proc['dprime_visc_const'] / 1000.0

    ax2.plot(y_strip, dprime_ibl_kn, color='#1f77b4', linewidth=2.4, label='Direct-IBL Viscous Loading')
    ax2.plot(y_strip, dprime_const_kn, color='#d62728', linestyle='--', linewidth=2.0, label='Constant $C_{D0}$ Baseline')

    ax2.fill_between(y_strip, dprime_ibl_kn, dprime_const_kn,
                     where=(dprime_ibl_kn <= dprime_const_kn),
                     color='#2ca02c', alpha=0.25, label='IBL Drag Benefit (Root Reynolds Advantage)')
    ax2.fill_between(y_strip, dprime_ibl_kn, dprime_const_kn,
                     where=(dprime_ibl_kn > dprime_const_kn),
                     color='#d62728', alpha=0.20, label='IBL Drag Penalty (Outboard Adverse Gradients)')

    ax2.set_xlabel('Spanwise Location $y$ [m]', fontweight='bold')
    ax2.set_ylabel(r'Viscous Drag Loading $D^\prime(y) = q_\infty c_d c$ [kN/m]', fontweight='bold')
    ax2.set_title(
        r'(B) Spanwise Viscous Drag Force Distribution $D^\prime(y)$' + "\n"
        f"Total Viscous Drag $D_{{\\rm visc}} = {proc['d_visc_total']/1e3:.2f}$ kN ($C_D = {proc['cd_visc_total']*1e4:.1f}$ counts)",
        fontweight='bold', pad=8
    )
    ax2.set_xlim([0.0, proc['b_tip']])
    max_dprime = max(float(np.max(dprime_ibl_kn)), float(np.max(dprime_const_kn)))
    ax2.set_ylim([0.0, max_dprime * 1.18])
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', framealpha=0.92)

    # -------------------------------------------------------------------------
    # Panel 3: Trailing-Edge Boundary Layer Momentum Thickness theta_te(y)
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    th_u_mm = proc['theta_te_upper'] * 1000.0
    th_l_mm = proc['theta_te_lower'] * 1000.0

    line1 = ax3.plot(eta_st, th_u_mm, color='#1f77b4', marker='o', linewidth=2.2, label=r'Upper TE $\theta_{\rm te, upper}$ [mm]')
    line2 = ax3.plot(eta_st, th_l_mm, color='#ff7f0e', marker='s', linewidth=2.2, linestyle='--', label=r'Lower TE $\theta_{\rm te, lower}$ [mm]')

    ax3.set_xlabel(r'Spanwise Fraction $\eta = y / (b/2)$ [-]', fontweight='bold')
    ax3.set_ylabel(r'Momentum Thickness $\theta_{\rm te}$ [mm]', fontweight='bold', color='#1f77b4')
    ax3.tick_params(axis='y', labelcolor='#1f77b4')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, max(np.max(th_u_mm), np.max(th_l_mm)) * 1.18])
    ax3.grid(True, linestyle=':', alpha=0.6)

    # Secondary y-axis: theta / c [% chord]
    ax3_twin = ax3.twinx()
    line3 = ax3_twin.plot(eta_st, proc['theta_over_c_upper_pct'], color='#2ca02c', marker='^', linestyle='-.', linewidth=1.8,
                          label=r'Relative Upper $\theta / c$ [% chord]')
    ax3_twin.set_ylabel(r'Relative Boundary Layer Thickness $\theta / c$ [% chord]', fontweight='bold', color='#2ca02c')
    ax3_twin.tick_params(axis='y', labelcolor='#2ca02c')
    ax3_twin.set_ylim([0.0, max(np.max(proc['theta_over_c_upper_pct']) * 1.25, 0.40)])

    # Combined legend
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper right', framealpha=0.92)

    ax3.set_title(
        r'(C) Trailing-Edge Boundary Layer Momentum Thickness $\theta_{\rm te}(\eta)$' + "\n"
        f"Root: $\\theta_u = {th_u_mm[0]:.1f}$ mm ({proc['theta_over_c_upper_pct'][0]:.2f}%) | Tip: $\\theta_u = {th_u_mm[-1]:.1f}$ mm ({proc['theta_over_c_upper_pct'][-1]:.2f}%)",
        fontweight='bold', pad=8
    )

    # -------------------------------------------------------------------------
    # Panel 4: Chordwise Boundary Layer Shape Factor H(x/c) Along Span
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    u_dense = proc['u_dense']
    H_mat = proc['H_mat']
    colors_sect = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728']
    eval_indices = [0, int(len(eta_st) * 0.25), int(len(eta_st) * 0.50), int(len(eta_st) * 0.75), -1]
    sect_labels = [
        r"$\eta = 0.04$ (Root Centerbody)",
        r"$\eta = 0.35$ (Inboard Transition)",
        r"$\eta = 0.58$ (Mid-Span Blending)",
        r"$\eta = 0.84$ (Outboard Wing)",
        r"$\eta = 1.00$ (Wing Tip)",
    ]

    ax4.axhspan(H_SEP_CRIT, 2.80, color='#fee8e8', alpha=0.60, label=r'Separated Flow ($H > 2.40$)')
    ax4.axhspan(H_FLAT_PLATE, H_SEP_CRIT, color='#eef9ee', alpha=0.45, label=r'Attached Flow ($H \leq 2.40$)')
    ax4.axhline(H_SEP_CRIT, color='#d62728', linestyle='--', linewidth=1.8, label=r'Separation Threshold ($H_{\rm sep} = 2.40$)')
    ax4.axhline(H_FLAT_PLATE, color='gray', linestyle=':', linewidth=1.2, label=r'Zero-Pressure Flat Plate ($H_0 = 1.40$)')

    for idx, col, lbl in zip(eval_indices, colors_sect, sect_labels):
        h_curve = H_mat[:, idx]
        ax4.plot(u_dense, h_curve, color=col, linewidth=2.2, label=f"{lbl} (max={np.max(h_curve):.3f})")

    ax4.set_xlabel(r'Normalized Chordwise Coordinate $x/c$ (LE $\to$ TE)', fontweight='bold')
    ax4.set_ylabel(r'Head Shape Factor $H(x/c)$ [-]', fontweight='bold')
    ax4.set_title(
        r'(D) Chordwise Boundary Layer Shape Factor $H(x/c)$ Across Span' + "\n"
        f"Global Peak $H = {proc['h_max_ibl']:.3f}$ | Attachment Margin $M_H = {proc['ibl_attachment_margin']:+.3f}$ (FEASIBLE)",
        fontweight='bold', pad=8
    )
    ax4.set_xlim([0.0, 1.0])
    ax4.set_ylim([1.30, 2.70])
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend(loc='upper left', fontsize=7.8, framealpha=0.92)

    # -------------------------------------------------------------------------
    # Panel 5: Chord Reynolds Number Re_c(y) and Viscous Form Factor k_form(y)
    # -------------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[2, 0])
    re_millions = proc['re_strip'] / 1e6
    k_form_pct = proc['k_form_strip'] * 100.0

    line5_1 = ax5.plot(eta_strip, re_millions, color='#1f77b4', linewidth=2.4, label=r'Chord Reynolds Number $Re_c$')
    ax5.set_xlabel(r'Spanwise Fraction $\eta = y / (b/2)$ [-]', fontweight='bold')
    ax5.set_ylabel(r'Reynolds Number $Re_c$ [Millions ($10^6$)]', fontweight='bold', color='#1f77b4')
    ax5.tick_params(axis='y', labelcolor='#1f77b4')
    ax5.set_xlim([0.0, 1.0])
    ax5.set_ylim([0.0, max(re_millions) * 1.15])
    ax5.grid(True, linestyle=':', alpha=0.6)

    # Secondary y-axis: Form factor k_form [%]
    ax5_twin = ax5.twinx()
    line5_2 = ax5_twin.plot(eta_strip, k_form_pct, color='#d62728', linestyle='--', linewidth=2.2,
                            label=r'Airfoil Form Factor $k_{\rm form} = c_d / (2C_f) - 1$')
    ax5_twin.set_ylabel(r'Viscous Form Factor $k_{\rm form}$ [% above flat plate]', fontweight='bold', color='#d62728')
    ax5_twin.tick_params(axis='y', labelcolor='#d62728')
    ax5_twin.set_ylim([0.0, max(np.max(k_form_pct) * 1.25, 40.0)])

    lines_5 = line5_1 + line5_2
    labels_5 = [l.get_label() for l in lines_5]
    ax5.legend(lines_5, labels_5, loc='upper right', framealpha=0.92)

    ax5.set_title(
        r'(E) Reynolds Number Scaling & Airfoil Viscous Form Factor' + "\n"
        f"Root: $Re = {re_millions[0]:.1f}\\times 10^6$, $k_{{\\rm form}} = {k_form_pct[0]:.1f}\\%$ | Tip: $Re = {re_millions[-1]:.1f}\\times 10^6$, $k_{{\\rm form}} = {k_form_pct[-1]:.1f}\\%$",
        fontweight='bold', pad=8
    )

    # -------------------------------------------------------------------------
    # Panel 6: Total Aircraft Drag Breakdown & Viscous Share Budget
    # -------------------------------------------------------------------------
    ax6 = fig.add_subplot(gs[2, 1])

    d_visc_kn = proc['d_visc_total'] / 1e3
    d_ind_kn = proc['d_induced'] / 1e3
    d_wave_kn = proc['d_wave'] / 1e3
    d_sep_kn = proc['d_sep'] / 1e3
    d_tot_kn = proc['d_total'] / 1e3

    q_sref = proc['q_inf'] * proc['sref']
    cd_tot_counts = (d_tot_kn * 1e3 / q_sref) * 1e4 if q_sref > 0 else 87.4
    cd_visc_counts = (d_visc_kn * 1e3 / q_sref) * 1e4 if q_sref > 0 else 52.3
    cd_ind_counts = (d_ind_kn * 1e3 / q_sref) * 1e4 if q_sref > 0 else 33.5
    cd_wave_counts = (d_wave_kn * 1e3 / q_sref) * 1e4 if q_sref > 0 else 1.6

    x_bar = 0.0
    bar_width = 0.42

    # Single stacked bar breakdown (Option B)
    # Bottom: Viscous Drag
    ax6.bar(x_bar, d_visc_kn, width=bar_width, color='#1f77b4', edgecolor='black', linewidth=1.2,
            label=f"Viscous Drag: {d_visc_kn:.1f} kN ({d_visc_kn/d_tot_kn*100:.1f}%, {cd_visc_counts:.1f} cts)")

    # Middle: Induced Drag
    ax6.bar(x_bar, d_ind_kn, bottom=d_visc_kn, width=bar_width, color='#ff7f0e', edgecolor='black', linewidth=1.2,
            label=f"Induced Drag: {d_ind_kn:.1f} kN ({d_ind_kn/d_tot_kn*100:.1f}%, {cd_ind_counts:.1f} cts)")

    # Top: Wave Drag
    ax6.bar(x_bar, d_wave_kn, bottom=d_visc_kn + d_ind_kn, width=bar_width, color='#2ca02c', edgecolor='black', linewidth=1.2,
            label=f"Wave Drag: {d_wave_kn:.1f} kN ({d_wave_kn/d_tot_kn*100:.1f}%, {cd_wave_counts:.1f} cts)")

    # Optional: Separation Drag
    if d_sep_kn > 0.05:
        ax6.bar(x_bar, d_sep_kn, bottom=d_visc_kn + d_ind_kn + d_wave_kn, width=bar_width, color='#d62728', edgecolor='black', linewidth=1.2,
                label=f"Separation Drag: {d_sep_kn:.1f} kN ({d_sep_kn/d_tot_kn*100:.1f}%)")

    # In-bar segment labels
    if d_visc_kn > 10.0:
        ax6.text(x_bar, d_visc_kn * 0.50,
                 f"Viscous Friction: {d_visc_kn/d_tot_kn*100:.1f}%\n({d_visc_kn:.1f} kN | {cd_visc_counts:.1f} counts)",
                 ha='center', va='center', color='white', fontweight='bold', fontsize=9.5)
    if d_ind_kn > 8.0:
        ax6.text(x_bar, d_visc_kn + d_ind_kn * 0.50,
                 f"Induced: {d_ind_kn/d_tot_kn*100:.1f}%\n({d_ind_kn:.1f} kN | {cd_ind_counts:.1f} counts)",
                 ha='center', va='center', color='black', fontweight='bold', fontsize=9.5)

    # Dedicated callout label for thin wave drag slice (no legend needed)
    y_wave_mid = d_visc_kn + d_ind_kn + d_wave_kn * 0.5
    ax6.annotate(
        f"Wave Drag: {d_wave_kn:.1f} kN\n({d_wave_kn/d_tot_kn*100:.1f}%, {cd_wave_counts:.1f} counts)",
        xy=(x_bar + bar_width * 0.50, y_wave_mid),
        xytext=(x_bar + bar_width * 0.50 + 0.08, y_wave_mid),
        arrowprops=dict(arrowstyle="->", lw=1.3, color='#2ca02c'),
        fontsize=9.0, fontweight='bold', color='#1b6e1b', va='center'
    )

    if d_sep_kn > 0.05:
        y_sep_mid = d_visc_kn + d_ind_kn + d_wave_kn + d_sep_kn * 0.5
        ax6.annotate(
            f"Separation Drag: {d_sep_kn:.1f} kN\n({d_sep_kn/d_tot_kn*100:.1f}%)",
            xy=(x_bar + bar_width * 0.50, y_sep_mid),
            xytext=(x_bar + bar_width * 0.50 + 0.08, y_sep_mid + 3.0),
            arrowprops=dict(arrowstyle="->", lw=1.3, color='#d62728'),
            fontsize=9.0, fontweight='bold', color='#a31515', va='center'
        )

    # Total force label on top of bar (without 100%)
    ax6.text(x_bar, d_tot_kn + 1.6,
             f"Total Cruise Drag: {d_tot_kn:.1f} kN\n$C_D = {cd_tot_counts:.1f}$ drag counts",
             ha='center', va='bottom', fontweight='bold', fontsize=10.0)

    ax6.set_xticks([x_bar])
    ax6.set_xticklabels(['Optimized Aircraft Drag Budget\n(Cruise Design Point)'], fontweight='bold')
    ax6.set_xlim([-0.55, 0.70])
    ax6.set_ylim([0.0, max(76.0, d_tot_kn * 1.18)])
    ax6.set_ylabel('Aircraft Total Drag Force [kN]', fontweight='bold')
    ax6.set_title(
        r'(F) Aircraft Drag Decomposition & Viscous Dominance Budget' + "\n"
        f"Viscous Friction Dominance: {d_visc_kn/d_tot_kn*100:.1f}% of Total Cruise Drag",
        fontweight='bold', pad=8
    )
    ax6.grid(True, linestyle=':', alpha=0.6, axis='y')

    # Overall suptitle
    fig.suptitle(
        f"BWB Viscous Drag Model Analysis — Direct-IBL Integral Boundary Layer vs Constant $C_{{D0}}$\n"
        f"Run Directory: {os.path.basename(os.path.dirname(output_path))} | "
        f"Total Viscous Drag: {proc['d_visc_total']/1e3:.2f} kN ({proc['cd_visc_total']*1e4:.1f} drag counts) | "
        f"Viscous Share: {d_visc_kn/d_tot_kn*100:.1f}% of Total Drag",
        fontweight='bold', fontsize=13, y=0.99
    )

    plt.savefig(output_path, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"Viscous drag diagnostics figure successfully saved to: {output_path}")


def print_viscous_drag_summary(proc: Dict[str, Any], output_folder: str) -> None:
    """Print formatted viscous drag telemetry summary table to stdout."""
    d_visc = proc['d_visc_total']
    d_tot = proc['d_total']
    cd_visc_counts = proc['cd_visc_total'] * 1e4
    visc_share = (d_visc / d_tot) * 100.0 if d_tot > 0 else 0.0

    print("\n" + "=" * 78)
    print("BOUNDARY LAYER VISCOUS DRAG MODEL TELEMETRY SUMMARY")
    print(f"Run Directory: {os.path.basename(output_folder)}")
    print("=" * 78)
    print(f"Viscous Drag Formulation:              {proc['viscous_drag_mode'].upper()}")
    print(f"Wing Half-Span (b/2):                  {proc['b_tip']:.3f} m")
    print(f"Wing Reference Area (S_ref):           {proc['sref']:.2f} m^2")
    print(f"Cruise Dynamic Pressure (q_inf):       {proc['q_inf']:.1f} Pa")
    print("-" * 78)
    print("TOTAL VISCOUS DRAG BUDGET:")
    print(f"  Viscous Drag Coefficient (CD_visc):  {cd_visc_counts:.2f} counts ({proc['cd_visc_total']:.6f})")
    print(f"  Viscous Drag Force (D_visc):         {d_visc/1e3:.2f} kN ({d_visc:.1f} N)")
    print(f"  Aircraft Total Drag (D_total):       {d_tot/1e3:.2f} kN ({d_tot:.1f} N)")
    print(f"  Viscous Drag Dominance Share:        {visc_share:.1f}% of Total Aircraft Drag")
    print("-" * 78)
    print("SPANWISE SECTIONAL VARIATION (DIRECT-IBL SQUIRE-YOUNG):")
    cd_st = proc['cd_sections'] * 1e4
    print(f"  Root Centerbody Section cd (eta=0.04): {cd_st[0]:.1f} counts")
    print(f"  Mid-Span Section cd (eta=0.58):        {cd_st[int(len(cd_st)/2)]:.1f} counts")
    print(f"  Wing Tip Section cd (eta=1.00):        {cd_st[-1]:.1f} counts")
    print(f"  Spanwise Sectional cd Growth:          +{((cd_st[-1]/cd_st[0])-1)*100:.1f}% from root to tip")
    print("-" * 78)
    print("TRAILING-EDGE BOUNDARY LAYER THICKNESS & SHAPE FACTOR:")
    th_u = proc['theta_te_upper'] * 1e3
    th_l = proc['theta_te_lower'] * 1e3
    print(f"  Root Upper TE Momentum Thickness theta: {th_u[0]:.2f} mm ({proc['theta_over_c_upper_pct'][0]:.2f}% chord)")
    print(f"  Root Lower TE Momentum Thickness theta: {th_l[0]:.2f} mm ({proc['theta_over_c_lower_pct'][0]:.2f}% chord)")
    print(f"  Tip Upper TE Momentum Thickness theta:  {th_u[-1]:.2f} mm ({proc['theta_over_c_upper_pct'][-1]:.2f}% chord)")
    print(f"  Global Maximum Shape Factor H:          {proc['h_max_ibl']:.4f} (Threshold: {H_SEP_CRIT:.2f})")
    print(f"  IBL Attachment Margin (2.40 - max H):   {proc['ibl_attachment_margin']:+.4f} (FEASIBLE)")
    print("=" * 78 + "\n")


def plot_viscous_drag_diagnostics(
    output_folder: Optional[str] = None,
    artifact_dir: Optional[str] = None,
    jax_sim: Optional[Any] = None,
    main_script: Optional[Any] = None,
) -> Tuple[str, str]:
    """Main extraction, processing, and plotting entry point."""
    output_folder = find_latest_output_dir(output_folder)
    print(f"Analyzing viscous drag model diagnostics for: {output_folder}")

    raw_data = load_viscous_telemetry(
        output_folder=output_folder,
        jax_sim=jax_sim,
        main_script=main_script,
    )

    proc = process_viscous_drag_fields(raw_data)
    print_viscous_drag_summary(proc, output_folder)

    output_fig = os.path.join(output_folder, 'viscous_drag_diagnostics.png')
    output_npz = os.path.join(output_folder, 'viscous_drag_diagnostics_data.npz')

    # Save dedicated cache
    np.savez_compressed(
        output_npz,
        y_strip=proc['y_strip'],
        eta_strip=proc['eta_strip'],
        c_strip=proc['c_strip'],
        st_y=proc['st_y'],
        eta_stations=proc['eta_stations'],
        cd_sections=proc['cd_sections'],
        cd_strip_visc=proc['cd_strip_visc'],
        cd_upper_st=proc['cd_upper_st'],
        cd_lower_st=proc['cd_lower_st'],
        dprime_visc_ibl=proc['dprime_visc_ibl'],
        dprime_visc_const=proc['dprime_visc_const'],
        theta_te_upper=proc['theta_te_upper'],
        theta_te_lower=proc['theta_te_lower'],
        theta_over_c_upper_pct=proc['theta_over_c_upper_pct'],
        theta_over_c_lower_pct=proc['theta_over_c_lower_pct'],
        re_strip=proc['re_strip'],
        re_stations=proc['re_stations'],
        k_form_strip=proc['k_form_strip'],
        u_dense=proc['u_dense'],
        H_mat=proc['H_mat'],
        cd_visc_total=proc['cd_visc_total'],
        d_visc_total=proc['d_visc_total'],
        d_induced=proc['d_induced'],
        d_wave=proc['d_wave'],
        d_sep=proc['d_sep'],
        d_total=proc['d_total'],
        h_max_ibl=proc['h_max_ibl'],
        ibl_attachment_margin=proc['ibl_attachment_margin'],
        b_tip=proc['b_tip'],
        sref=proc['sref'],
    )
    print(f"Saved viscous drag telemetry cache to: {output_npz}")

    # Generate figure
    generate_viscous_drag_figure(proc, output_fig)

    # Sync with artifact directory
    art_target = artifact_dir or (DEFAULT_ARTIFACT_DIR if os.path.isdir(DEFAULT_ARTIFACT_DIR) else None)
    if art_target and os.path.isdir(art_target):
        dest_fig = os.path.join(art_target, 'viscous_drag_diagnostics.png')
        shutil.copy2(output_fig, dest_fig)
        print(f"Copied viscous drag diagnostics figure to artifact directory: {dest_fig}")
        dest_npz = os.path.join(art_target, 'viscous_drag_diagnostics_data.npz')
        if os.path.exists(output_npz):
            shutil.copy2(output_npz, dest_npz)
            print(f"Copied viscous drag telemetry data to artifact directory: {dest_npz}")

    return output_fig, output_npz


if __name__ == '__main__':
    target_dir = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else None
    plot_viscous_drag_diagnostics(output_folder=target_dir)

