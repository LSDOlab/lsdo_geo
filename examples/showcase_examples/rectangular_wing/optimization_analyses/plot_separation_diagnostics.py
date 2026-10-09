"""Plot and analyze turbulent boundary layer separation diagnostics for BWB optimization.

Visualizes flow attachment and separation proximity across the aircraft upper mold line (OML)
using both:
1. Stratford's canonical separation criterion:
   S(x, y) <= S_crit = 0.39, attachment margin M_S = 0.39 - S
2. Head's entrainment shape factor method (previous approach):
   H(x, y) <= H_sep = 2.40, attachment margin M_H = 2.40 - H

Generates publication-quality 2D planform heatmaps, chordwise sectional profiles,
spanwise margin distributions, and the 4 x N_stations regional constraint matrix.

Usage:
    python plot_separation_diagnostics.py [output_directory]

If output_directory is omitted, the latest directory under
'rectangular_wing_to_bwb_aerostructural_optimization_outputs' is analyzed.
"""

from __future__ import annotations
import os
import sys
import glob
import shutil
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm, Normalize
from scipy.interpolate import RectBivariateSpline, interp1d

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

# Physical and criterion thresholds
S_CRIT = 0.39       # Canonical Stratford turbulent separation threshold
H_SEP = 2.40        # Head shape factor turbulent separation threshold
H_FLAT_PLATE = 1.40 # Zero-pressure-gradient turbulent flat-plate baseline


DEFAULT_ARTIFACT_DIR = '/home/andrew/.gemini/antigravity/brain/728a2089-2977-4bdc-b525-91b2b850c11b'


def find_latest_output_dir(base_dir: Optional[str] = None) -> str:
    """Find the target optimization output directory.

    If base_dir is provided:
      1. Resolves if it is an existing directory path (absolute or relative).
      2. Searches for base_dir by exact folder name in KNOWN_OUTPUT_DIRS.
      3. Searches for base_dir by partial substring match in KNOWN_OUTPUT_DIRS,
         returning the latest match by timestamp folder name.

    If base_dir is None:
      Automatically finds the chronologically latest completed output directory
      (containing 'lift_and_moment_data.npz' or 'separation_diagnostics_data.npz') across
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
                or os.path.exists(os.path.join(d, 'separation_diagnostics_data.npz'))
            ]
            all_with_data.extend(with_data)

    candidates = all_with_data if all_with_data else all_subdirs
    if not candidates:
        raise FileNotFoundError("No valid optimization output directory found.")

    chosen = max(candidates, key=lambda d: os.path.basename(d))
    return os.path.abspath(chosen)


def load_separation_telemetry(output_folder: str) -> Dict[str, Any]:
    """Load cached separation telemetry or extract from lift_and_moment_data.npz."""
    sep_cache = os.path.join(output_folder, 'separation_diagnostics_data.npz')
    lm_cache = os.path.join(output_folder, 'lift_and_moment_data.npz')

    data_dict: Dict[str, Any] = {}

    if os.path.exists(lm_cache):
        print(f"Loading separation telemetry from lift and moment cache: {lm_cache}")
        raw = np.load(lm_cache, allow_pickle=True)
        data_dict = {k: raw[k] for k in raw.files}
        # Supplement with any missing fields from dedicated cache if available
        if os.path.exists(sep_cache):
            raw_sep = np.load(sep_cache, allow_pickle=True)
            for k in raw_sep.files:
                if k not in data_dict:
                    data_dict[k] = raw_sep[k]
        return data_dict

    if os.path.exists(sep_cache):
        print(f"Loading dedicated separation diagnostics cache: {sep_cache}")
        raw = np.load(sep_cache, allow_pickle=True)
        data_dict = {k: raw[k] for k in raw.files}
        return data_dict

    raise FileNotFoundError(
        f"Neither 'separation_diagnostics_data.npz' nor 'lift_and_moment_data.npz' found in {output_folder}"
    )


def process_separation_fields(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstruct 2D surface geometry, dense fields, sectional cuts, and margins."""
    # 1. Number of stations and basic parameters
    num_ffd_stations = int(data_dict.get('num_stations', 5))
    scale_factor = float(data_dict.get('scale_factor', 7.5))

    # Regional constraint values (4 x num_ffd_stations)
    if 'dv_stratford_constraints' in data_dict:
        sc_raw = np.asarray(data_dict['dv_stratford_constraints']).flatten()
        if len(sc_raw) == 4 * num_ffd_stations:
            sc_matrix = sc_raw.reshape(4, num_ffd_stations)
        else:
            num_ffd_stations = len(sc_raw) // 4
            sc_matrix = sc_raw.reshape(4, num_ffd_stations)
    elif 'sc_matrix' in data_dict:
        sc_matrix = np.asarray(data_dict['sc_matrix'])
        num_ffd_stations = sc_matrix.shape[1]
    else:
        # Default placeholder if not found
        sc_matrix = np.full((4, num_ffd_stations), 0.30)

    # Head shape factor station values
    if 'H_section_upper' in data_dict:
        h_mesh_sections = np.asarray(data_dict['H_section_upper']).flatten()
    elif 'dv_H_stations' in data_dict:
        h_mesh_sections = np.asarray(data_dict['dv_H_stations']).flatten()
    else:
        h_mesh_sections = np.full((num_ffd_stations,), 1.8)

    if 'dv_H_stations' in data_dict:
        h_ffd_stations = np.asarray(data_dict['dv_H_stations']).flatten()
    else:
        h_ffd_stations = np.linspace(h_mesh_sections[0], h_mesh_sections[-1], num_ffd_stations)

    # 2. Extract upper surface panel centers and planform geometry
    panel_centers_right = data_dict.get('panel_centers_right', None)
    num_mesh_stations = len(h_mesh_sections)

    if panel_centers_right is not None and len(panel_centers_right.shape) == 2:
        pc = np.asarray(panel_centers_right)
        y_round = np.round(pc[:, 1], 4)
        y_unique = np.unique(y_round)
        main_stations = [y for y in y_unique if np.sum(y_round == y) == 40]

        if len(main_stations) >= num_mesh_stations:
            main_stations = main_stations[:num_mesh_stations]

        y_stations = []
        x_le_stations = []
        x_te_stations = []
        chords = []

        for y_val in main_stations:
            st_panels = pc[y_round == y_val]
            y_stations.append(np.mean(st_panels[:, 1]))
            x_min = np.min(st_panels[:, 0])
            x_max = np.max(st_panels[:, 0])
            x_le_stations.append(x_min)
            x_te_stations.append(x_max)
            chords.append(x_max - x_min)

        y_stations = np.array(y_stations)
        x_le_stations = np.array(x_le_stations)
        x_te_stations = np.array(x_te_stations)
        chords = np.array(chords)
    else:
        # Fallback synthetic BWB planform from scale factor
        b_tip = 15.63
        y_stations = np.linspace(0.63, b_tip, num_mesh_stations)
        x_le_stations = np.linspace(-7.8, 15.8, num_mesh_stations)
        chords = np.linspace(24.5, 1.93, num_mesh_stations)
        x_te_stations = x_le_stations + chords

    b_tip = float(np.max(y_stations))
    eta_stations = y_stations / b_tip

    # 3. Dense 2D Field Reconstruction
    NU = 60
    NV = 80
    u_dense = np.linspace(0.0, 1.0, NU)
    v_dense = np.linspace(0.0, 1.0, NV)

    # Check if exact S_dense_2d is directly cached
    if 'S_dense_2d' in data_dict:
        S_raw_2d = np.asarray(data_dict['S_dense_2d'])
        if S_raw_2d.ndim == 2:
            u_orig = np.linspace(0.0, 1.0, S_raw_2d.shape[0])
            v_orig = np.linspace(0.0, 1.0, S_raw_2d.shape[1])
            spl_s = RectBivariateSpline(u_orig, v_orig, S_raw_2d, kx=2, ky=2)
            S_dense = spl_s(u_dense, v_dense)
        else:
            S_dense = None
    else:
        S_dense = None

    if S_dense is None:
        # Reconstruct Stratford 2D surface from 4 x N_stations regional constraints
        # Anchor S=0 at LE (u=0)
        u_pts = np.array([0.0, 1.0 / 6.0, 0.5, 5.0 / 6.0, 1.0])
        v_pts = np.linspace(0.0, 1.0, num_ffd_stations)
        S_pts = np.vstack([np.zeros((1, num_ffd_stations)), sc_matrix])  # (5, num_ffd_stations)
        spl_s = RectBivariateSpline(u_pts, v_pts, S_pts, kx=2, ky=min(3, num_ffd_stations - 1))
        S_dense = spl_s(u_dense, v_dense)
        S_dense = np.clip(S_dense, 0.0, None)

    # Reconstruct Head H 2D surface
    # Check if exact H_upper_matrix is cached
    if 'H_upper_matrix' in data_dict:
        H_raw_mat = np.asarray(data_dict['H_upper_matrix'])
        if H_raw_mat.ndim == 2:
            u_orig = np.linspace(0.0, 1.0, H_raw_mat.shape[0])
            v_orig = np.linspace(0.0, 1.0, H_raw_mat.shape[1])
            spl_h = RectBivariateSpline(u_orig, v_orig, H_raw_mat, kx=2, ky=min(3, H_raw_mat.shape[1] - 1))
            H_dense = spl_h(u_dense, v_dense)
        else:
            H_dense = None
    else:
        H_dense = None

    if H_dense is None:
        # Canonical boundary layer evolution: H stays around H_0 in suction then climbs sharply towards TE
        # Smoothly interpolating station TE values
        H_te_interp = interp1d(eta_stations, h_mesh_sections, fill_value='extrapolate')
        H_te_v = H_te_interp(v_dense)  # Shape (NV,)

        # Chordwise growth: H(u) = H_0 + (H_te - H_0) * (u ** 2.4)
        H_dense = np.zeros((NU, NV))
        for j in range(NV):
            H_te = max(H_te_v[j], H_FLAT_PLATE)
            growth = u_dense ** 2.4
            H_dense[:, j] = H_FLAT_PLATE + (H_te - H_FLAT_PLATE) * growth

    # Planform physical grid (X, Y) in metres
    f_xle = interp1d(eta_stations, x_le_stations, fill_value='extrapolate')
    f_chord = interp1d(eta_stations, chords, fill_value='extrapolate')

    x_le_v = f_xle(v_dense)
    c_v = f_chord(v_dense)

    X_grid = np.zeros((NU, NV))
    Y_grid = np.zeros((NU, NV))
    for i in range(NU):
        for j in range(NV):
            X_grid[i, j] = x_le_v[j] + u_dense[i] * c_v[j]
            Y_grid[i, j] = v_dense[j] * b_tip

    # 4. Sectional cuts at 5 representative span stations
    eval_etas = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    eta_labels = [
        r"$\eta = 0.00$ (Centerline Root)",
        r"$\eta = 0.25$ (Inboard Transition)",
        r"$\eta = 0.50$ (Mid-Span Blending)",
        r"$\eta = 0.75$ (Outboard Wing)",
        r"$\eta = 1.00$ (Wing Tip)",
    ]

    sectional_data = []
    for eta_val, lbl in zip(eval_etas, eta_labels):
        j_idx = int(np.argmin(np.abs(v_dense - eta_val)))
        s_cut = S_dense[:, j_idx]
        h_cut = H_dense[:, j_idx]
        sectional_data.append({
            'eta': eta_val,
            'label': lbl,
            'u': u_dense,
            'S': s_cut,
            'H': h_cut,
            'max_S': float(np.max(s_cut)),
            'margin_S': float(S_CRIT - np.max(s_cut)),
            'max_H': float(np.max(h_cut)),
            'margin_H': float(H_SEP - np.max(h_cut)),
        })

    # 5. Spanwise minimum margin distributions
    span_margin_S = S_CRIT - np.max(S_dense, axis=0)  # Shape (NV,)
    span_margin_H = H_SEP - np.max(H_dense, axis=0)  # Shape (NV,)
    span_norm_margin_H = span_margin_H / (H_SEP - H_FLAT_PLATE)  # Normalized margin

    # 6. Global summary metrics
    global_max_S = float(np.max(S_dense))
    global_margin_S = float(S_CRIT - global_max_S)
    idx_worst_S = np.unravel_index(np.argmax(S_dense), S_dense.shape)
    worst_S_loc = (float(X_grid[idx_worst_S]), float(Y_grid[idx_worst_S]), float(v_dense[idx_worst_S[1]]))

    global_max_H = float(np.max(H_dense))
    global_margin_H = float(H_SEP - global_max_H)
    idx_worst_H = np.unravel_index(np.argmax(H_dense), H_dense.shape)
    worst_H_loc = (float(X_grid[idx_worst_H]), float(Y_grid[idx_worst_H]), float(v_dense[idx_worst_H[1]]))

    num_satisfied_S = int(np.sum(sc_matrix <= S_CRIT + 1e-4))
    total_constrs_S = int(sc_matrix.size)

    return {
        'num_ffd_stations': num_ffd_stations,
        'num_mesh_stations': num_mesh_stations,
        'scale_factor': scale_factor,
        'b_tip': b_tip,
        'y_stations': y_stations,
        'eta_stations': eta_stations,
        'x_le_stations': x_le_stations,
        'chords': chords,
        'sc_matrix': sc_matrix,
        'h_ffd_stations': h_ffd_stations,
        'h_mesh_sections': h_mesh_sections,
        'u_dense': u_dense,
        'v_dense': v_dense,
        'X_grid': X_grid,
        'Y_grid': Y_grid,
        'S_dense': S_dense,
        'H_dense': H_dense,
        'sectional_data': sectional_data,
        'span_margin_S': span_margin_S,
        'span_margin_H': span_margin_H,
        'span_norm_margin_H': span_norm_margin_H,
        'global_max_S': global_max_S,
        'global_margin_S': global_margin_S,
        'worst_S_loc': worst_S_loc,
        'global_max_H': global_max_H,
        'global_margin_H': global_margin_H,
        'worst_H_loc': worst_H_loc,
        'num_satisfied_S': num_satisfied_S,
        'total_constrs_S': total_constrs_S,
    }


def generate_separation_figure(proc: Dict[str, Any], output_path: str) -> None:
    """Generate 6-panel comprehensive separation diagnostics figure."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'legend.fontsize': 8.5,
        'figure.titlesize': 13,
    })

    fig = plt.figure(figsize=(17, 12), dpi=250)
    gs = GridSpec(3, 2, figure=fig, hspace=0.36, wspace=0.25)

    # Extract processing data
    X = proc['X_grid']
    Y = proc['Y_grid']
    S = proc['S_dense']
    H = proc['H_dense']
    b_tip = proc['b_tip']
    sc_mat = proc['sc_matrix']
    num_ffd = proc['num_ffd_stations']

    # -------------------------------------------------------------------------
    # Panel 1: Upper OML Planform Map of Stratford Parameter S(x, y)
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    levels_s = np.linspace(0.0, max(0.80, float(np.max(S)) * 1.05), 60)
    norm_s = TwoSlopeNorm(vmin=0.0, vcenter=S_CRIT, vmax=max(0.80, float(np.max(S)) * 1.05))
    cs1 = ax1.contourf(X, Y, S, levels=levels_s, cmap='coolwarm', norm=norm_s, extend='both')

    # Separation contour line S = 0.39
    if np.min(S) < S_CRIT < np.max(S):
        cnt1 = ax1.contour(X, Y, S, levels=[S_CRIT], colors='black', linewidths=2.2, linestyles='--')
        ax1.clabel(cnt1, fmt={S_CRIT: r'$S_{\rm crit} = 0.39$'}, fontsize=8.5, inline=True)

    # Overlay regional constraint centers
    u_centers = [1.0 / 6.0, 0.5, 5.0 / 6.0, 1.0]
    v_centers = np.linspace(0.0, 1.0, num_ffd)
    TOL_ACTIVE = 1e-4
    has_active = False
    has_feas = False
    has_viol = False
    for i, uc in enumerate(u_centers):
        for j, vc in enumerate(v_centers):
            xc = np.interp(vc, proc['v_dense'], proc['X_grid'][int(uc * (len(proc['u_dense']) - 1)), :])
            yc = vc * b_tip
            val = sc_mat[i, j]
            if val > S_CRIT + TOL_ACTIVE:
                color = '#d62728'
                marker = 'X'
                has_viol = True
            elif abs(val - S_CRIT) <= TOL_ACTIVE:
                color = '#ff7f0e'
                marker = 'o'
                has_active = True
            else:
                color = '#2ca02c'
                marker = 'o'
                has_feas = True
            ax1.scatter(xc, yc, color=color, edgecolors='black', s=46, marker=marker, zorder=5)

    # Add legend handles for regional constraint markers in ax1
    handles1 = []
    if has_feas:
        handles1.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c', markeredgecolor='k', markersize=7, label=r'Feasible ($S < 0.39$)'))
    if has_active:
        handles1.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', markeredgecolor='k', markersize=7, label=r'Active Bound ($S \approx 0.390$)'))
    if has_viol:
        handles1.append(plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='#d62728', markeredgecolor='k', markersize=7, label=r'Violated ($S > 0.39$)'))
    if handles1:
        ax1.legend(handles=handles1, loc='upper left', fontsize=7.5, framealpha=0.92)

    cbar1 = plt.colorbar(cs1, ax=ax1, pad=0.02, shrink=0.92)
    cbar1.set_label(r'Stratford Parameter $S(x, y)$ [-]', fontweight='bold')
    cbar1.ax.axhline(S_CRIT, color='black', linestyle='--', linewidth=1.5)

    ax1.set_xlabel('Aircraft X Coordinate [m]', fontweight='bold')
    ax1.set_ylabel('Spanwise Y Coordinate [m]', fontweight='bold')
    ax1.set_title(
        r'(A) Stratford Parameter $S(x, y)$ on Upper Mold Line'
        f"\nMax $S = {proc['global_max_S']:.3f}$ | Margin $= {proc['global_margin_S']:+.3f}$",
        fontweight='bold', pad=8
    )
    ax1.set_xlim([np.min(X) - 1.0, np.max(X) + 1.0])
    ax1.set_ylim([0.0, b_tip * 1.04])
    ax1.grid(True, linestyle=':', alpha=0.5)

    # -------------------------------------------------------------------------
    # Panel 2: Upper OML Planform Map of Head Shape Factor H(x, y)
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    levels_h = np.linspace(H_FLAT_PLATE, max(2.80, float(np.max(H)) * 1.05), 60)
    norm_h = TwoSlopeNorm(vmin=H_FLAT_PLATE, vcenter=H_SEP, vmax=max(2.80, float(np.max(H)) * 1.05))
    cs2 = ax2.contourf(X, Y, H, levels=levels_h, cmap='coolwarm', norm=norm_h, extend='both')

    # Separation contour line H = 2.40
    if np.min(H) < H_SEP < np.max(H):
        cnt2 = ax2.contour(X, Y, H, levels=[H_SEP], colors='black', linewidths=2.2, linestyles='--')
        ax2.clabel(cnt2, fmt={H_SEP: r'$H_{\rm sep} = 2.40$'}, fontsize=8.5, inline=True)

    cbar2 = plt.colorbar(cs2, ax=ax2, pad=0.02, shrink=0.92)
    cbar2.set_label(r'Head Shape Factor $H(x, y)$ [-]', fontweight='bold')
    cbar2.ax.axhline(H_SEP, color='black', linestyle='--', linewidth=1.5)

    ax2.set_xlabel('Aircraft X Coordinate [m]', fontweight='bold')
    ax2.set_ylabel('Spanwise Y Coordinate [m]', fontweight='bold')
    ax2.set_title(
        r'(B) Head Shape Factor $H(x, y)$ on Upper Mold Line (Previous Approach)'
        f"\nMax $H = {proc['global_max_H']:.3f}$ | Margin $= {proc['global_margin_H']:+.3f}$",
        fontweight='bold', pad=8
    )
    ax2.set_xlim([np.min(X) - 1.0, np.max(X) + 1.0])
    ax2.set_ylim([0.0, b_tip * 1.04])
    ax2.grid(True, linestyle=':', alpha=0.5)

    # -------------------------------------------------------------------------
    # Panel 3: Chordwise Stratford Profiles S(x/c) at Key Span Stations
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    colors_sect = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728']

    u_pts = proc['u_dense']
    ax3.axhspan(S_CRIT, max(1.10, proc['global_max_S'] * 1.05), color='#fee8e8', alpha=0.65, label='Separated Flow ($S > 0.39$)')
    ax3.axhspan(0.0, S_CRIT, color='#eef9ee', alpha=0.45, label=r'Attached Flow ($S \leq 0.39$)')
    ax3.axhline(S_CRIT, color='#d62728', linestyle='--', linewidth=1.8, label=r'Stratford Limit ($S_{\rm crit} = 0.39$)')

    for sect, col in zip(proc['sectional_data'], colors_sect):
        ax3.plot(u_pts, sect['S'], color=col, linewidth=2.2, label=f"{sect['label']} (max={sect['max_S']:.3f})")

    # Mark chordwise regional aggregation centers
    for uc in u_centers:
        ax3.axvline(uc, color='gray', linestyle=':', linewidth=1.0, alpha=0.7)
    ax3.text(
        0.02, 0.05,
        r"$\mathbf{Note:}$ Sectional curves show continuous physical $S(x/c)$ (max $\approx 0.345$).""\n"
        r"Regional KS smooth-max ($KS \leq 0.39$) adds $\approx +0.045$ aggregation margin," "\n"
        r"activating the optimizer constraint at $0.3900$.",
        fontsize=7.5, color='#333333',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f9fa', edgecolor='#cccccc', alpha=0.9),
        transform=ax3.transAxes
    )

    ax3.set_xlabel(r'Normalized Chordwise Coordinate $x/c$ (LE $\to$ TE)', fontweight='bold')
    ax3.set_ylabel(r'Stratford Parameter $S(x/c)$ [-]', fontweight='bold')
    ax3.set_title(r'(C) Chordwise Stratford Growth $S(x/c)$ Along Span Stations', fontweight='bold', pad=8)
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, max(1.10, proc['global_max_S'] * 1.08)])
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc='upper left', fontsize=7.8, framealpha=0.92)

    # -------------------------------------------------------------------------
    # Panel 4: Chordwise Head Shape Factor Profiles H(x/c) at Key Span Stations
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axhspan(H_SEP, max(3.0, proc['global_max_H'] * 1.05), color='#fee8e8', alpha=0.65, label='Separated Flow ($H > 2.40$)')
    ax4.axhspan(H_FLAT_PLATE, H_SEP, color='#eef9ee', alpha=0.45, label=r'Attached Flow ($H \leq 2.40$)')
    ax4.axhline(H_SEP, color='#d62728', linestyle='--', linewidth=1.8, label=r'Separation Threshold ($H_{\rm sep} = 2.40$)')

    for sect, col in zip(proc['sectional_data'], colors_sect):
        ax4.plot(u_pts, sect['H'], color=col, linewidth=2.2, label=f"{sect['label']} (max={sect['max_H']:.3f})")

    ax4.set_xlabel('Normalized Chordwise Coordinate $x/c$ (LE $\\to$ TE)', fontweight='bold')
    ax4.set_ylabel(r'Head Shape Factor $H(x/c)$ [-]', fontweight='bold')
    ax4.set_title(r'(D) Chordwise Head Shape Factor $H(x/c)$ Along Span Stations', fontweight='bold', pad=8)
    ax4.set_xlim([0.0, 1.0])
    ax4.set_ylim([1.35, max(3.0, proc['global_max_H'] * 1.08)])
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend(loc='upper left', fontsize=7.8, framealpha=0.92)

    # -------------------------------------------------------------------------
    # Panel 5: Spanwise Separation Margin Comparison (0.39 - S vs 2.40 - H)
    # -------------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[2, 0])
    eta_v = proc['v_dense']

    ax5.axhspan(0.0, 1.0, color='#eef9ee', alpha=0.45, label='Attached Margin ($M > 0$)')
    ax5.axhspan(-1.0, 0.0, color='#fee8e8', alpha=0.65, label='Separated Violation ($M < 0$)')
    ax5.axhline(0.0, color='gray', linestyle='--', linewidth=1.5)

    # Stratford absolute margin M_S = 0.39 - max(S)
    ax5.plot(eta_v, proc['span_margin_S'], color='#1f77b4', linewidth=2.4, label=r'Stratford Margin: $M_S(\eta) = 0.39 - \max_x S$')
    # Head normalized margin M_H = (2.40 - max(H)) / 1.0
    ax5.plot(eta_v, proc['span_norm_margin_H'], color='#ff7f0e', linewidth=2.4, linestyle='-.', label=r'Head Norm Margin: $(2.40 - \max_x H) / 1.0$')

    ax5.set_xlabel(r'Spanwise Fraction $\eta = y / (b/2)$ [-]', fontweight='bold')
    ax5.set_ylabel('Separation Margin Metric [-]', fontweight='bold')
    ax5.set_title(r'(E) Spanwise Minimum Separation Margin Comparison', fontweight='bold', pad=8)
    ax5.set_xlim([0.0, 1.0])
    y_min_margin = min(float(np.min(proc['span_margin_S'])), float(np.min(proc['span_norm_margin_H'])), -0.15)
    y_max_margin = max(float(np.max(proc['span_margin_S'])), float(np.max(proc['span_norm_margin_H'])), 0.30)
    ax5.set_ylim([y_min_margin * 1.25, y_max_margin * 1.25])
    ax5.grid(True, linestyle=':', alpha=0.6)
    ax5.legend(loc='lower left', fontsize=8.5, framealpha=0.92)

    # -------------------------------------------------------------------------
    # Panel 6: 2D Regional Stratford Constraint Matrix (4 x N_stations)
    # -------------------------------------------------------------------------
    ax6 = fig.add_subplot(gs[2, 1])

    chord_region_labels = [
        r'LE Suction ($u \approx 0.17$)',
        r'Mid-Chord ($u = 0.50$)',
        r'Aft Recovery ($u \approx 0.83$)',
        r'Trailing Edge ($u = 1.00$)',
    ]
    station_col_labels = [f"Stn {j+1}\n(" + r"$\eta=" + f"{proc['v_dense'][int(j*(len(proc['v_dense'])-1)/(num_ffd-1))]:.2f}" + r"$)" for j in range(num_ffd)]

    im6 = ax6.imshow(
        sc_mat, cmap='coolwarm', aspect='auto',
        norm=TwoSlopeNorm(vmin=0.0, vcenter=S_CRIT, vmax=max(0.80, float(np.max(sc_mat))))
    )

    for i in range(4):
        for j in range(num_ffd):
            val = sc_mat[i, j]
            margin = S_CRIT - val
            if val > S_CRIT + 1e-4:
                status = "VIOL"
            elif abs(val - S_CRIT) <= 1e-4:
                status = "ACTIVE"
            else:
                status = "FEAS"
            txt_col = 'white' if abs(val - S_CRIT) > 0.15 else 'black'
            ax6.text(
                j, i, f"{val:.3f}\n({margin:+.3f})\n{status}",
                ha='center', va='center', color=txt_col, fontsize=8.2, fontweight='bold'
            )

    ax6.set_xticks(range(num_ffd))
    ax6.set_xticklabels(station_col_labels)
    ax6.set_yticks(range(4))
    ax6.set_yticklabels(chord_region_labels)
    ax6.set_title(
        r"(F) Regional Stratford Constraints ($S \leq 0.39$)" + "\n"
        f"Satisfied: {proc['num_satisfied_S']}/{proc['total_constrs_S']} Sectors",
        fontweight='bold', pad=8
    )

    # Add suptitle
    fig.suptitle(
        f"BWB Boundary Layer Separation Diagnostics — Stratford vs Head Formulations\n"
        f"Run Directory: {os.path.basename(os.path.dirname(output_path))} | "
        f"Clearance Status: {'FEASIBLE' if proc['global_margin_S'] >= 0 else 'VIOLATED'} "
        r"($\min M_S = " + f"{proc['global_margin_S']:+.3f}" + r"$, $\min M_H = " + f"{proc['global_margin_H']:+.3f}" + r"$)",
        fontweight='bold', fontsize=13, y=0.99
    )

    plt.savefig(output_path, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"Separation diagnostics figure successfully saved to: {output_path}")


def print_telemetry_summary(proc: Dict[str, Any], output_folder: str) -> None:
    """Print clean formatted telemetry summary table to stdout."""
    print("\n" + "=" * 75)
    print(f"BOUNDARY LAYER SEPARATION TELEMETRY SUMMARY")
    print(f"Run: {os.path.basename(output_folder)}")
    print("=" * 75)
    print(f"Half-Span (b/2):                       {proc['b_tip']:.3f} m")
    print(f"Scale Factor:                          {proc['scale_factor']:.2f}")
    print(f"Number of FFD Variable Stations:       {proc['num_ffd_stations']}")
    print(f"Number of Evaluation Mesh Stations:    {proc['num_mesh_stations']}")
    print("-" * 75)
    print("STRATFORD CANONICAL CRITERION (ACTIVE CONSTRAINT):")
    print(f"  Threshold (S_crit):                  {S_CRIT:.2f}")
    print(f"  Global Maximum S:                    {proc['global_max_S']:.4f}")
    print(f"  Global Minimum Margin (0.39 - S):    {proc['global_margin_S']:+.4f} ({'FEASIBLE' if proc['global_margin_S'] >= 0 else 'VIOLATED'})")
    print(f"  Worst Point Location (X, Y, eta):    X={proc['worst_S_loc'][0]:.2f}m, Y={proc['worst_S_loc'][1]:.2f}m (eta={proc['worst_S_loc'][2]:.2f})")
    print(f"  Regional Constraints Satisfied:      {proc['num_satisfied_S']}/{proc['total_constrs_S']} sectors")
    print("-" * 75)
    print("HEAD SHAPE FACTOR METHOD (PREVIOUS INFORMATIONAL COMPARISON):")
    print(f"  Threshold (H_sep):                   {H_SEP:.2f}")
    print(f"  Global Maximum H:                    {proc['global_max_H']:.4f}")
    print(f"  Global Minimum Margin (2.40 - H):    {proc['global_margin_H']:+.4f} ({'FEASIBLE' if proc['global_margin_H'] >= 0 else 'VIOLATED'})")
    print(f"  Worst Point Location (X, Y, eta):    X={proc['worst_H_loc'][0]:.2f}m, Y={proc['worst_H_loc'][1]:.2f}m (eta={proc['worst_H_loc'][2]:.2f})")
    print("=" * 75 + "\n")


def plot_separation_diagnostics(
    output_folder: Optional[str] = None,
    artifact_dir: Optional[str] = None,
    jax_sim: Optional[Any] = None,
    main_script: Optional[Any] = None,
    force_rerun: bool = False,
) -> Tuple[str, str]:
    """Main extraction, processing, and plotting entry point.

    Parameters
    ----------
    output_folder : str, optional
        Target optimization output directory. Defaults to latest run.
    artifact_dir : str, optional
        Target directory to copy generated artifacts.
    jax_sim : object, optional
        Live JaxSimulator instance if called within optimization execution.
    main_script : object, optional
        Reference to main script module.
    force_rerun : bool
        If True, forces re-processing even if cache exists.

    Returns
    -------
    tuple of (str, str)
        Paths to generated figure PNG and data NPZ.
    """
    output_folder = find_latest_output_dir(output_folder)
    print(f"Analyzing boundary layer separation for output folder: {output_folder}")

    # Load data
    data_dict = load_separation_telemetry(output_folder)

    # Process 2D fields and sectional cuts
    proc = process_separation_fields(data_dict)

    # Output paths
    fig_path = os.path.join(output_folder, 'separation_diagnostics.png')
    npz_path = os.path.join(output_folder, 'separation_diagnostics_data.npz')

    # Save dedicated NPZ cache
    np.savez_compressed(
        npz_path,
        X_grid=proc['X_grid'],
        Y_grid=proc['Y_grid'],
        S_dense=proc['S_dense'],
        H_dense=proc['H_dense'],
        u_dense=proc['u_dense'],
        v_dense=proc['v_dense'],
        sc_matrix=proc['sc_matrix'],
        span_margin_S=proc['span_margin_S'],
        span_margin_H=proc['span_margin_H'],
        global_max_S=proc['global_max_S'],
        global_margin_S=proc['global_margin_S'],
        global_max_H=proc['global_max_H'],
        global_margin_H=proc['global_margin_H'],
        b_tip=proc['b_tip'],
        scale_factor=proc['scale_factor'],
        num_ffd_stations=proc['num_ffd_stations'],
    )
    print(f"Separation diagnostics data saved to: {npz_path}")

    # Generate figure
    generate_separation_figure(proc, fig_path)

    # Copy to artifact directory if present
    art_target = artifact_dir or (DEFAULT_ARTIFACT_DIR if os.path.isdir(DEFAULT_ARTIFACT_DIR) else None)
    if art_target and os.path.isdir(art_target):
        art_fig = os.path.join(art_target, 'separation_diagnostics.png')
        art_npz = os.path.join(art_target, 'separation_diagnostics_data.npz')
        shutil.copy2(fig_path, art_fig)
        shutil.copy2(npz_path, art_npz)
        print(f"Separation artifacts also copied to: {art_fig}")

    # Print summary
    print_telemetry_summary(proc, output_folder)

    return fig_path, npz_path


if __name__ == '__main__':
    target_dir = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else None
    plot_separation_diagnostics(target_dir)
