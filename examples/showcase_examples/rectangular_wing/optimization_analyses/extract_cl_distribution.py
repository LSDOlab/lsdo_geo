"""Extract and plot spanwise section lift coefficient (cl) distribution and stall margin diagnostics.

Visualizes:
1. Sectional lift coefficient cl(y) across 100 spanwise drag strips during:
   - 2.5g Structural Sizing Pull-Up Maneuver (Node 2)
   - 1.0g Cruise Condition (Node 0)
2. Continuous cubic B-spline fit S(y) to the maneuver cl distribution with root symmetry S'(0) = 0.
3. Multi-point evaluation and smooth maximum aggregation at design variable stations.
4. Monotonic stall ceiling profile cl_ceiling(y) (e.g. 1.30 at root down to 0.80 at tip)
   enforcing that the root must stall before the tip.
5. Local stall margin Delta_cl(y) = cl_ceiling(y) - cl_ss(y) and global critical cl_crit = 1.40.

Can be run:
1. Automatically at the end of optimization in `ex_rectangular_wing_to_bwb.py` via:
       from optimization_analyses.extract_cl_distribution import extract_and_plot_cl_distribution
       extract_and_plot_cl_distribution(output_folder)
2. Standalone from CLI:
       python extract_cl_distribution.py [optional_path_to_output_folder]
"""

import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Compatibility shim for stored geometry caches pickled under NumPy 2.x
if 'numpy._core' not in sys.modules:
    sys.modules['numpy._core'] = np.core
if 'numpy._core.numeric' not in sys.modules:
    sys.modules['numpy._core.numeric'] = np.core.numeric

# Ensure working directory is repo root so relative paths resolve
REPO_ROOT = '/home/andrew/optimization/lsdo_geo'
if os.path.exists(REPO_ROOT):
    try:
        os.chdir(REPO_ROOT)
    except Exception:
        pass

# Ensure all relevant paths are on sys.path
sys.path.insert(0, '/home/andrew/optimization/VortexAD')
sys.path.insert(0, '/home/andrew/optimization/aframe')
sys.path.insert(0, '/home/andrew/optimization/csdl')
sys.path.insert(0, '/home/andrew/optimization/CSDL_alpha')
sys.path.insert(0, '/home/andrew/optimization/modopt')
sys.path.insert(0, '/home/andrew/optimization/lsdo_geo')
sys.path.insert(0, '/home/andrew/optimization/lsdo_geo/examples/showcase_examples/rectangular_wing')

KNOWN_OUTPUT_DIRS = [
    '/home/andrew/optimization/lsdo_geo/rectangular_wing_to_bwb_aerostructural_optimization_outputs',
    '/home/andrew/optimization/lsdo_geo/rectangular_wing_aerostructural_optimization_outputs',
    '/home/andrew/optimization/lsdo_geo/rectangular_wing_panel_optimization_outputs',
]


def evaluate_bspline_fit(y_strip, coeffs, num_eval_points=300):
    """Evaluate continuous cubic B-spline fit curve over spanwise coordinate.
    
    Uses lsdo_function_spaces.BSplineSpace matching the formulation in ex_rectangular_wing_to_bwb.py.
    Falls back to scipy CubicSpline if lsdo_function_spaces is not importable.
    """
    b_tip = float(np.max(y_strip))
    eta_eval = np.linspace(0.0, 1.0, num_eval_points)
    y_eval = eta_eval * b_tip

    try:
        import lsdo_function_spaces as lfs
        tau_drag_init = y_strip / b_tip
        tau_drag_colloc = np.concatenate([[0.0], tau_drag_init])
        int_knots_cl_101 = [np.mean(tau_drag_colloc[j:j+3]) for j in range(1, 98)]
        knots_cl_101 = np.concatenate([[0.0, 0.0, 0.0, 0.0], int_knots_cl_101, [1.0, 1.0, 1.0, 1.0]])

        space = lfs.BSplineSpace(
            num_parametric_dimensions=1,
            degree=3,
            coefficients_shape=(101,),
            knots=(knots_cl_101,),
        )
        B_eval = space.compute_basis_matrix(eta_eval.reshape((-1, 1))).toarray()
        cl_eval = np.asarray(B_eval @ coeffs).flatten()
        return y_eval, eta_eval, cl_eval
    except Exception as exc:
        print(f"Notice: evaluating via scipy CubicSpline ({exc})")
        # Enforce zero slope at root
        y_sym = np.concatenate([-y_strip[::-1], y_strip])
        coeffs_sym = np.concatenate([coeffs[:len(y_strip)][::-1], coeffs[:len(y_strip)]])
        spl = CubicSpline(y_sym, coeffs_sym, bc_type='natural')
        return y_eval, eta_eval, spl(y_eval)


def extract_and_plot_cl_distribution(output_folder=None, artifact_dir=None):
    """Main extraction and visualization function."""
    if output_folder is None:
        if len(sys.argv) > 1:
            output_folder = os.path.abspath(sys.argv[1])
        else:
            candidates = []
            for base_dir in KNOWN_OUTPUT_DIRS:
                if os.path.exists(base_dir):
                    subdirs = glob.glob(os.path.join(base_dir, '*'))
                    for d in subdirs:
                        if os.path.isdir(d):
                            candidates.append(d)
            if not candidates:
                raise FileNotFoundError("No optimization output folders found.")
            output_folder = max(candidates, key=os.path.getmtime)

    output_folder = os.path.abspath(output_folder)
    print(f"\n--- Extracting Section Lift Coefficient (cl) Distribution for: {os.path.basename(output_folder)} ---")

    # Locate telemetry cache
    cache_path = os.path.join(output_folder, 'lift_and_moment_data.npz')
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Telemetry cache not found at: {cache_path}")

    data = np.load(cache_path, allow_pickle=True)

    # Required data keys
    y_strip = data['y_strip_pts']
    b_tip = float(np.max(y_strip))
    eta_strip = y_strip / b_tip

    cl_cruise = data['cl_local_elem'] if 'cl_local_elem' in data else None
    cl_ss = data['cl_local_elem_ss'] if 'cl_local_elem_ss' in data else None

    if cl_ss is None:
        raise ValueError("Cache does not contain 'cl_local_elem_ss' (maneuver section lift coefficients).")

    # Station ceiling and aggregation data
    num_stations = 5
    if 'station_cl_peaks' in data:
        station_peaks_eta = data['station_cl_peaks']
        num_stations = len(station_peaks_eta)
    else:
        num_stations = 5
        station_peaks_eta = np.linspace(0.0, 1.0, num_stations)

    station_peaks_y = station_peaks_eta * b_tip

    if 'cl_ss_ceiling' in data:
        cl_ss_ceiling = data['cl_ss_ceiling']
    else:
        cl_ss_ceiling = np.linspace(1.30, 0.80, num_stations)

    if 'dv_cl_ss' in data:
        dv_cl_ss = data['dv_cl_ss']
    else:
        # Fallback approximate sampling
        indices = [int(np.argmin(np.abs(y_strip - y))) for y in station_peaks_y]
        dv_cl_ss = cl_ss[indices]

    cl_crit_global = 1.40

    # Evaluate B-spline fit curve
    if 'cl_ss_coeffs' in data:
        coeffs = data['cl_ss_coeffs']
        y_eval, eta_eval, cl_ss_spline = evaluate_bspline_fit(y_strip, coeffs, num_eval_points=400)
    else:
        # Reconstruct CubicSpline
        y_sym = np.concatenate([-y_strip[::-1], y_strip])
        cl_sym = np.concatenate([cl_ss[::-1], cl_ss])
        cs = CubicSpline(y_sym, cl_sym, bc_type='natural')
        eta_eval = np.linspace(0.0, 1.0, 400)
        y_eval = eta_eval * b_tip
        cl_ss_spline = cs(y_eval)

    # Smooth curve for cruise cl if available
    if cl_cruise is not None:
        y_sym_c = np.concatenate([-y_strip[::-1], y_strip])
        cl_sym_c = np.concatenate([cl_cruise[::-1], cl_cruise])
        cs_c = CubicSpline(y_sym_c, cl_sym_c, bc_type='natural')
        cl_cruise_spline = cs_c(y_eval)
    else:
        cl_cruise_spline = None

    # Continuous ceiling profile across span
    ceiling_spline = np.interp(eta_eval, station_peaks_eta, cl_ss_ceiling)
    
    # Stall margin profile Delta cl = ceiling - cl_ss
    stall_margin_spline = ceiling_spline - cl_ss_spline
    station_margins = cl_ss_ceiling - dv_cl_ss

    # -------------------------------------------------------------------------
    # Visualization: 2-Panel Diagnostic Figure
    # -------------------------------------------------------------------------
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 9.8), dpi=200, sharex=True,
                                         gridspec_kw={'height_ratios': [1.3, 1.0], 'hspace': 0.16})

    fig.suptitle(
        f"Spanwise Section Lift Coefficient ($c_l$) & Monotonic Stall Margin Analysis\n"
        f"Run: {os.path.basename(output_folder)}  |  Half-Span: $b/2 = {b_tip:.2f}$ m  |  Sizing Pull-Up ($N_z = 2.5$g) & Cruise (1.0g)",
        fontsize=13, fontweight='bold', y=0.985
    )

    # Color definitions
    c_spline = '#b2182b'     # Rich deep red
    c_strip = '#d6604d'      # Coral red
    c_ceiling = '#762a83'    # Purple for constraint boundary
    c_crit = '#800000'       # Maroon for global stall
    c_cruise = '#2166ac'     # Classic navy blue
    c_cruise_pts = '#4393c3' # Light blue
    c_safe = '#2ca02c'       # Green
    c_violation = '#d62728'  # Red

    # -------------------------------------------------------------------------
    # Panel 1 (Top): Maneuver cl, B-spline Fit, and Allowable Ceiling
    # -------------------------------------------------------------------------
    # Forbidden Stall Region (above ceiling)
    ax_top.fill_between(y_eval, ceiling_spline, 1.65, color=c_ceiling, alpha=0.10,
                        label=r"Forbidden Stall Regime ($c_l > c_{l,\mathrm{ceiling}}$)")

    # Global critical stall limit cl_crit = 1.40
    ax_top.axhline(cl_crit_global, color=c_crit, linestyle=':', linewidth=1.8, alpha=0.85,
                   label=rf"Global Stall Inception Limit ($c_{{l,\mathrm{{crit}}}} = {cl_crit_global:.2f}$)")
    ax_top.text(b_tip * 0.02, cl_crit_global + 0.02, rf"Global Stall Inception Limit $c_{{l,\mathrm{{crit}}}} = {cl_crit_global:.2f}$",
                fontsize=9.5, fontweight='bold', color=c_crit)

    # Monotonic Allowable Stall Ceiling profile
    ax_top.plot(y_eval, ceiling_spline, '--', color=c_ceiling, linewidth=2.4, zorder=6,
                label=r"Allowable Stall Ceiling $c_{l,\mathrm{ceiling}}(y)$ [1.30 root $\rightarrow$ 0.80 tip]")

    # Raw 100 drag strip cl points
    ax_top.scatter(y_strip, cl_ss, color=c_strip, s=26, alpha=0.75, zorder=4,
                   edgecolors='white', linewidth=0.5, label=r"Strip $c_{l,\mathrm{ss}}$ (100 Drag Strips)")

    # Continuous B-Spline Fit Curve
    ax_top.plot(y_eval, cl_ss_spline, '-', color=c_spline, linewidth=2.8, zorder=5,
                label=r"Continuous Cubic B-Spline Fit $S(y)$ [with $S'(0) = 0$]")

    # Evaluated points from cache if available
    if 'eval_cl_points' in data:
        eval_pts_y = data['eval_cl_points'] * b_tip
        try:
            eval_pts_val = np.interp(data['eval_cl_points'], eta_eval, cl_ss_spline)
            ax_top.scatter(eval_pts_y, eval_pts_val, marker='o', s=20, facecolors='none',
                           edgecolors='#333333', linewidth=1.0, alpha=0.7, zorder=7,
                           label="Intermediate Evaluation Points (3 pts/station)")
        except Exception:
            pass

    # Classify station status with numerical tolerance for active boundary constraints
    active_tol = 1.0e-3
    station_status = []
    for marg in station_margins:
        if marg < -active_tol:
            station_status.append("VIOLATED")
        elif abs(marg) <= active_tol:
            station_status.append("ACTIVE")
        else:
            station_status.append("FEASIBLE")

    # Aggregated Station Peaks (Design Variable Stations)
    annotation_offsets = [
        (+1.2, +0.13),  # Stn 1 (root): shift right and up
        ( 0.0, +0.12),  # Stn 2
        ( 0.0, +0.12),  # Stn 3
        (-1.2, -0.16),  # Stn 4: shift left and down
        (-1.4, -0.16),  # Stn 5 (tip): shift left and down
    ]
    for i, (sy, scl, clim, smarg, stat) in enumerate(zip(station_peaks_y, dv_cl_ss, cl_ss_ceiling, station_margins, station_status)):
        if stat == "VIOLATED":
            m_color = c_violation
            marker = 'X'
            tag = " [VIOL]"
        elif stat == "ACTIVE":
            m_color = '#1f77b4'
            marker = 's'
            tag = " [ACTIVE]"
        else:
            m_color = c_safe
            marker = 'D'
            tag = ""

        # Mark aggregated point
        ax_top.scatter([sy], [scl], marker=marker, s=95, color=m_color, edgecolors='black',
                       linewidth=1.2, zorder=8, label=r"Aggregated Station Peak $c_{l,\mathrm{ss}}$" if i == 0 else "")

        # Mark ceiling station point
        ax_top.scatter([sy], [clim], marker='s', s=45, color=c_ceiling, edgecolors='white',
                       linewidth=0.8, zorder=7)

        # Annotation text with custom offsets
        dx, dy_off = annotation_offsets[i] if i < len(annotation_offsets) else (0.0, 0.10)
        ax_top.annotate(
            f"Stn {i+1} ({station_peaks_eta[i]:.2f})\n$c_l={scl:.3f}$\nCeil: {clim:.2f}{tag}",
            xy=(sy, scl), xytext=(sy + dx, scl + dy_off),
            ha='center', va='bottom' if dy_off > 0 else 'top', fontsize=8.2, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=m_color, alpha=0.92),
            arrowprops=dict(arrowstyle='->', color=m_color, lw=1.2)
        )

    y_min_top = min(float(np.min(cl_ss_spline)), float(np.min(cl_ss)))
    ax_top.set_ylabel(r"Section Lift Coefficient $c_l$", fontsize=11, fontweight='bold')
    ax_top.set_xlim([0.0, b_tip * 1.02])
    ax_top.set_ylim([min(0.15, y_min_top - 0.08), 1.60])
    ax_top.grid(True, linestyle=':', alpha=0.6)
    ax_top.legend(loc='upper right', fontsize=8.2, framealpha=0.95, ncol=2)

    # -------------------------------------------------------------------------
    # Panel 2 (Bottom): Local Stall Margin & Cruise Operating Point
    # -------------------------------------------------------------------------
    # Zero margin reference line
    ax_bot.axhline(0.0, color='black', linestyle='-', linewidth=1.2, alpha=0.7)

    # Shaded safe vs violation regions for margin
    ax_bot.fill_between(y_eval, stall_margin_spline, 0.0, where=(stall_margin_spline >= 0.0),
                        color=c_safe, alpha=0.20, interpolate=True, label=r"Feasible Stall Margin ($\Delta c_l \geq 0$)")
    ax_bot.fill_between(y_eval, stall_margin_spline, 0.0, where=(stall_margin_spline < 0.0),
                        color=c_violation, alpha=0.25, interpolate=True, label=r"Stall Ceiling Exceeded ($\Delta c_l < 0$)")

    # Continuous stall margin curve
    ax_bot.plot(y_eval, stall_margin_spline, '-', color='#4d004b', linewidth=2.4, zorder=5,
                label=r"Local Stall Margin $\Delta c_l(y) = c_{l,\mathrm{ceiling}}(y) - S(y)$")

    # Mark station margin points
    for i, (sy, smarg, stat) in enumerate(zip(station_peaks_y, station_margins, station_status)):
        if stat == "VIOLATED":
            m_color = c_violation
        elif stat == "ACTIVE":
            m_color = '#1f77b4'
        else:
            m_color = c_safe
        ax_bot.scatter([sy], [smarg], marker='o', s=60, color=m_color, edgecolors='black',
                       linewidth=1.0, zorder=7)
        offset_val = +0.05 if smarg >= 0 else -0.05
        label_text = f"{smarg:+.3f}" if stat != "ACTIVE" else f"{smarg:+.3f} (act)"
        ax_bot.annotate(
            label_text,
            xy=(sy, smarg), xytext=(sy, smarg + offset_val),
            ha='center', va='bottom' if offset_val > 0 else 'top', fontsize=8, fontweight='bold',
            color=m_color
        )

    # 1.0g Cruise Lift Coefficient
    if cl_cruise is not None:
        ax_bot.scatter(y_strip, cl_cruise, color=c_cruise_pts, s=20, alpha=0.6, zorder=3,
                       label=r"Cruise $c_{l,\mathrm{cruise}}$ (100 Drag Strips)")
        if cl_cruise_spline is not None:
            ax_bot.plot(y_eval, cl_cruise_spline, '-', color=c_cruise, linewidth=2.0, zorder=4,
                        label=r"Cruise Section $c_l(y)$ (1.0g Level Flight)")

    ax_bot.set_xlabel("Spanwise Coordinate $y$ [m]", fontsize=11, fontweight='bold')
    ax_bot.set_ylabel(r"Stall Margin $\Delta c_l$ / Cruise $c_l$", fontsize=11, fontweight='bold')
    ax_bot.set_title(r"Local Section Stall Margin ($\Delta c_l = c_{l,\mathrm{ceiling}} - c_{l,\mathrm{ss}}$) & Cruise Baseline",
                     fontsize=11.5, fontweight='bold', pad=8)
    ax_bot.set_xlim([0.0, b_tip * 1.02])

    # Dynamic bottom y-limits covering all margin and cruise data
    all_bot = [stall_margin_spline]
    if cl_cruise is not None:
        all_bot.append(cl_cruise)
        if cl_cruise_spline is not None:
            all_bot.append(cl_cruise_spline)
    bot_flat = np.concatenate([np.asarray(arr).flatten() for arr in all_bot])
    y_min_bot = float(np.min(bot_flat))
    y_max_bot = float(np.max(bot_flat))
    pad_bot = (y_max_bot - y_min_bot) * 0.12
    ax_bot.set_ylim([min(-0.25, y_min_bot - pad_bot), max(0.90, y_max_bot + pad_bot)])
    ax_bot.grid(True, linestyle=':', alpha=0.6)
    ax_bot.legend(loc='upper right', fontsize=8.2, framealpha=0.95, ncol=2)

    # Secondary X-axis for non-dimensional span eta = y / (b/2)
    ax_top_sec = ax_top.secondary_xaxis('top', functions=(lambda y: y / b_tip, lambda eta: eta * b_tip))
    ax_top_sec.set_xlabel(r"Fractional Spanwise Station $\eta = y / (b/2)$", fontsize=11, fontweight='bold', labelpad=8)
    ax_top_sec.set_ticks(np.linspace(0.0, 1.0, 11))

    # Summary Table text box in lower left (reflects exact latest run data, no optimization advice)
    lines = ["STATION STALL CONSTRAINTS (Sizing Pull-Up):"]
    for i in range(num_stations):
        lines.append(
            f"  • Stn {i+1} (η={station_peaks_eta[i]:.2f}): cl={dv_cl_ss[i]:.3f}  Ceil={cl_ss_ceiling[i]:.2f}  "
            f"Margin={station_margins[i]:+.3f} [{station_status[i]}]"
        )
    lines.append("STRIP LIFT COEFFICIENTS:")
    lines.append(f"  • Sizing Pull-Up (2.5g): cl ∈ [{np.min(cl_ss):.3f}, {np.max(cl_ss):.3f}], Tip cl = {cl_ss[-1]:.3f}")
    if cl_cruise is not None:
        lines.append(f"  • Cruise (1.0g):         cl ∈ [{np.min(cl_cruise):.3f}, {np.max(cl_cruise):.3f}], Tip cl = {cl_cruise[-1]:.3f}")

    summary_text = "\n".join(lines)

    has_violation = any(s == "VIOLATED" for s in station_status)
    box_color = '#fee8c8' if has_violation else '#e5f5e0'
    border_color = '#e34a33' if has_violation else '#31a354'
    ax_bot.text(
        0.02, 0.05, summary_text, transform=ax_bot.transAxes,
        fontsize=8.0, family='monospace', verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=box_color, edgecolor=border_color, alpha=0.92, lw=1.2)
    )

    plt.subplots_adjust(top=0.88, bottom=0.07, left=0.07, right=0.97, hspace=0.20)

    # Save figure
    fig_path = os.path.join(output_folder, 'cl_distribution.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Section lift coefficient (cl) distribution saved to: {fig_path}")

    # Copy to artifact directory
    if artifact_dir is None:
        curr_id = '0c0a47e5-2e16-41bb-9139-10357c23c5ee'
        artifact_dir = os.environ.get('ARTIFACT_DIR', f'/home/andrew/.gemini/antigravity/brain/{curr_id}')
        if not os.path.exists(artifact_dir):
            for old_id in ['3256dd7c-d4c8-4c73-887d-131361f9d0c3', '680209fd-293d-4fa0-9f6c-ae59a72a6987']:
                check_dir = f'/home/andrew/.gemini/antigravity/brain/{old_id}'
                if os.path.exists(check_dir):
                    artifact_dir = check_dir
                    break

    if os.path.exists(artifact_dir):
        import shutil
        art_path = os.path.join(artifact_dir, 'cl_distribution.png')
        shutil.copy2(fig_path, art_path)
        print(f"Figure also saved to artifact directory: {art_path}")

    # Console Summary Table
    print("\n" + "=" * 76)
    print(f"SECTION LIFT COEFFICIENT & STALL CEILING SUMMARY ({os.path.basename(output_folder)})")
    print("=" * 76)
    print(f"{'Stn':<4} {'eta':<6} {'y [m]':<8} {'cl (ss)':<10} {'Ceiling':<10} {'Margin':<10} {'Status':<16}")
    print("-" * 76)
    for i, (eta_i, y_i, cl_i, ceil_i, marg_i, stat_i) in enumerate(zip(station_peaks_eta, station_peaks_y, dv_cl_ss, cl_ss_ceiling, station_margins, station_status)):
        print(f"{i+1:<4} {eta_i:<6.3f} {y_i:<8.2f} {cl_i:<10.4f} {ceil_i:<10.3f} {marg_i:<+10.4f} {stat_i:<16}")
    print("=" * 76)
    print(f"Global Critical Stall Threshold (cl_crit): {cl_crit_global:.2f}")
    if has_violation:
        print(f"WARNING: Stall ceiling exceeded at {sum(s == 'VIOLATED' for s in station_status)} station(s).")
    else:
        print("ALL STATIONS SATISFY THE MONOTONIC ROOT-BEFORE-TIP STALL CEILING CONSTRAINT.")
    print("=" * 76 + "\n")

    return fig_path


if __name__ == '__main__':
    target_folder = sys.argv[1] if len(sys.argv) > 1 else None
    extract_and_plot_cl_distribution(target_folder)
