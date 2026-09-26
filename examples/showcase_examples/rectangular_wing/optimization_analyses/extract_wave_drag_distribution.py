"""Extract and plot spanwise transonic wave-drag distribution for BWB optimization.

Evaluates and visualizes the Korn-Lock transonic wave-drag model evaluated across
100 spanwise drag strips, including:
1. Sectional wave drag force density D'(y) = dD_wave/dy [N/m] and drag counts cd_wave(y).
2. Transonic Mach envelope: local critical Mach M_crit(y), drag divergence Mach M_dd(y),
   and cruise Mach M_inf = 0.70 showing the shock inception / drag-rise exposure zones.
3. Live geometric drivers: thickness-to-chord ratio (t/c)(y) and half-chord sweep Lambda_0.5(y).

Can be run:
1. Automatically at the end of optimization in `ex_rectangular_wing_to_bwb.py` via:
       from optimization_analyses.extract_wave_drag_distribution import extract_and_plot_wave_drag_distribution
       extract_and_plot_wave_drag_distribution(output_folder, jax_sim, main_script)
2. Standalone from CLI:
       python extract_wave_drag_distribution.py [optional_path_to_output_folder]
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

# Physical and Korn-Lock constants
KAPPA_A = 0.87
LOCK_COEFFICIENT = 20.0
LOCK_ACTIVATION_SHARPNESS = 100.0
DELTA_M_DD_TO_CRIT = (0.1 / 80.0) ** (1.0 / 3.0)  # ~0.1077217
Q_CRUISE_DEFAULT = 10320.72  # Pa at 30,000 ft ISA, M = 0.70
MACH_CRUISE_DEFAULT = 0.70


def compute_korn_lock_numpy(strip_tc, strip_sweep_halfchord, strip_cl, mach_cruise=MACH_CRUISE_DEFAULT):
    """Vectorized NumPy evaluation of Korn-Lock wave drag relations."""
    eps_cl = 1.0e-5
    abs_cl = np.sqrt(strip_cl ** 2 + eps_cl ** 2)
    cos_lambda = np.cos(strip_sweep_halfchord)
    cos2 = cos_lambda ** 2
    cos3 = cos_lambda ** 3

    m_dd = (KAPPA_A / cos_lambda) - (strip_tc / cos2) - (abs_cl / (10.0 * cos3))
    m_crit = m_dd - DELTA_M_DD_TO_CRIT
    delta_m = mach_cruise - m_crit
    # Softplus: log(1 + exp(100 * delta_m)) / 100
    z = LOCK_ACTIVATION_SHARPNESS * delta_m
    delta_m_eff = np.where(z > 40.0, delta_m, np.log1p(np.exp(np.clip(z, -80.0, 40.0))) / LOCK_ACTIVATION_SHARPNESS)
    cd_wave_strip = LOCK_COEFFICIENT * (delta_m_eff ** 4)

    return {
        'cd_wave_strip': cd_wave_strip,
        'M_dd': m_dd,
        'M_crit': m_crit,
        'delta_M': delta_m,
        'delta_M_eff': delta_m_eff,
    }


def process_wave_drag_strips(
    y_strip_pts,
    strip_wave_cd,
    strip_M_crit,
    strip_M_dd,
    strip_tc,
    strip_sweep_halfchord,
    strip_cl,
    local_chord_drag=None,
    dy_strip=None,
    strip_area=None,
    mach_cruise=MACH_CRUISE_DEFAULT,
    q_cruise=Q_CRUISE_DEFAULT,
    sref_val=10.0,
    b_tip=None,
):
    """Process strip wave drag telemetry into sectional densities and smooth spline curves."""
    num_strips = len(y_strip_pts)
    y_strip = np.asarray(y_strip_pts, dtype=float).flatten()

    # Reconstruct strip width dy and wing half-span b_tip if needed
    if b_tip is None or b_tip <= 0:
        if dy_strip is not None and len(dy_strip) == num_strips:
            b_tip = float(y_strip[-1] + 0.5 * dy_strip[-1])
        else:
            b_tip = float(y_strip[-1] * 1.01)

    if dy_strip is None or len(dy_strip) != num_strips:
        bounds = np.zeros(num_strips + 1)
        bounds[0] = 0.0
        bounds[-1] = b_tip
        bounds[1:-1] = 0.5 * (y_strip[:-1] + y_strip[1:])
        dy_strip = np.diff(bounds)
    else:
        dy_strip = np.asarray(dy_strip, dtype=float).flatten()

    # Local chord length c(y)
    if local_chord_drag is None or len(local_chord_drag) != num_strips:
        if strip_area is not None and len(strip_area) == num_strips:
            # strip_area = 2 * c * dy
            local_chord_drag = strip_area / (2.0 * dy_strip)
        else:
            # Default approximation
            local_chord_drag = np.full(num_strips, sref_val / (2.0 * b_tip))
    else:
        local_chord_drag = np.asarray(local_chord_drag, dtype=float).flatten()

    if strip_area is None or len(strip_area) != num_strips:
        strip_area = 2.0 * local_chord_drag * dy_strip
    else:
        strip_area = np.asarray(strip_area, dtype=float).flatten()

    strip_wave_cd = np.asarray(strip_wave_cd, dtype=float).flatten()
    strip_M_crit = np.asarray(strip_M_crit, dtype=float).flatten()
    strip_M_dd = np.asarray(strip_M_dd, dtype=float).flatten()
    strip_tc = np.asarray(strip_tc, dtype=float).flatten()
    strip_sweep_halfchord = np.asarray(strip_sweep_halfchord, dtype=float).flatten()
    strip_cl = np.asarray(strip_cl, dtype=float).flatten()

    # Sectional wave drag force on right half-wing strip:
    # D_half,i = cd_wave,i * q_inf * c_i * dy_i [N]
    d_wave_half_strip = strip_wave_cd * q_cruise * local_chord_drag * dy_strip
    d_wave_total_strip = 2.0 * d_wave_half_strip

    # Sectional wave drag force density per unit span (half-wing basis):
    # dD_half / dy = cd_wave * q_inf * c [N/m]
    dD_wave_density = strip_wave_cd * q_cruise * local_chord_drag
    cd_wave_counts = strip_wave_cd * 1.0e4

    # Integrated quantities
    tot_d_wave_half = float(np.sum(d_wave_half_strip))
    tot_d_wave_total = 2.0 * tot_d_wave_half
    total_strip_area = float(np.sum(strip_area))
    integrated_cd_wave = float(np.sum(strip_wave_cd * strip_area) / total_strip_area)

    # Smooth spline interpolation grid
    y_fine = np.linspace(0.0, b_tip, 300)

    # Enforce root symmetry dD'/dy = 0 at y = 0 via symmetric mirroring
    y_sym = np.concatenate([-y_strip[::-1], y_strip])
    dD_sym = np.concatenate([dD_wave_density[::-1], dD_wave_density])
    cd_counts_sym = np.concatenate([cd_wave_counts[::-1], cd_wave_counts])
    mcrit_sym = np.concatenate([strip_M_crit[::-1], strip_M_crit])
    mdd_sym = np.concatenate([strip_M_dd[::-1], strip_M_dd])
    tc_sym = np.concatenate([strip_tc[::-1], strip_tc])
    sweep_sym = np.concatenate([strip_sweep_halfchord[::-1], strip_sweep_halfchord])

    spl_dD = CubicSpline(y_sym, dD_sym, bc_type='natural')
    spl_cd = CubicSpline(y_sym, cd_counts_sym, bc_type='natural')
    spl_mcrit = CubicSpline(y_sym, mcrit_sym, bc_type='natural')
    spl_mdd = CubicSpline(y_sym, mdd_sym, bc_type='natural')
    spl_tc = CubicSpline(y_sym, tc_sym, bc_type='natural')
    spl_sweep = CubicSpline(y_sym, sweep_sym, bc_type='natural')

    dD_fine = np.maximum(spl_dD(y_fine), 0.0)
    cd_counts_fine = np.maximum(spl_cd(y_fine), 0.0)
    mcrit_fine = spl_mcrit(y_fine)
    mdd_fine = spl_mdd(y_fine)
    tc_fine = spl_tc(y_fine)
    sweep_deg_fine = np.degrees(spl_sweep(y_fine))

    return {
        'y_strip': y_strip,
        'y_fine': y_fine,
        'b_tip': b_tip,
        'dy_strip': dy_strip,
        'local_chord_drag': local_chord_drag,
        'strip_area': strip_area,
        'strip_wave_cd': strip_wave_cd,
        'cd_wave_counts': cd_wave_counts,
        'strip_M_crit': strip_M_crit,
        'strip_M_dd': strip_M_dd,
        'strip_tc': strip_tc,
        'strip_sweep_deg': np.degrees(strip_sweep_halfchord),
        'strip_cl': strip_cl,
        'd_wave_half_strip': d_wave_half_strip,
        'dD_wave_density': dD_wave_density,
        'dD_fine': dD_fine,
        'cd_counts_fine': cd_counts_fine,
        'mcrit_fine': mcrit_fine,
        'mdd_fine': mdd_fine,
        'tc_fine': tc_fine,
        'sweep_deg_fine': sweep_deg_fine,
        'tot_d_wave_half': tot_d_wave_half,
        'tot_d_wave_total': tot_d_wave_total,
        'integrated_cd_wave': integrated_cd_wave,
        'integrated_cd_wave_counts': integrated_cd_wave * 1.0e4,
    }


def plot_wave_drag_distribution(results, output_folder, run_label, sref_val, ar_val, mach_cruise=0.70, q_cruise=Q_CRUISE_DEFAULT, artifact_dir=None):
    """Generate high-resolution diagnostic plot matching the aesthetic of lift_distribution.png."""
    y_strip = results['y_strip']
    y_fine = results['y_fine']
    b_tip = results['b_tip']

    # Two-panel layout matching lift_and_moment_combined_analysis.png styling
    fig, (ax_drag, ax_mach) = plt.subplots(1, 2, figsize=(14, 5.8), dpi=200)

    # Color scheme
    c_wave = '#d62728'      # Crimson red for wave drag
    c_counts = '#8e24aa'    # Purple for drag counts
    c_mcrit = '#e65100'     # Amber / deep orange for Mcrit
    c_mdd = '#2e7d32'       # Forest green for Mdd
    c_minf = '#1565c0'      # Navy blue for M_infinity
    c_geom = '#546e7a'      # Slate grey for geometry annotations

    # Figure suptitle
    fig.suptitle(
        f"Spanwise Transonic Wave Drag & Mach Divergence Analysis (Korn–Lock Model)\n"
        f"Right Half-Span: b/2 = {b_tip:.3f} m (Full b = {2*b_tip:.3f} m) | AR = {ar_val:.2f} | S_ref = {sref_val:.2f} m² | "
        f"Flight Condition: 30,000 ft ISA (M_∞ = {mach_cruise:.2f}, q_∞ = {q_cruise:,.0f} Pa)",
        fontsize=12, fontweight='bold', y=0.98
    )

    # -------------------------------------------------------------------------
    # Panel 1: Spanwise Wave Drag Distribution (Force Density & Drag Counts)
    # -------------------------------------------------------------------------
    # Primary axis: Sectional Wave Drag Force Density D'(y) [N/m]
    ax_drag.plot(
        y_fine, results['dD_fine'], '-', color=c_wave, linewidth=2.4,
        label=f"Wave Drag Density $D'_{{\\text{{wave}}}}(y)$ ($D_{{\\text{{half}}}} = {results['tot_d_wave_half']:.1f}$ N, $D_{{\\text{{total}}}} = {results['tot_d_wave_total']:.1f}$ N)"
    )
    ax_drag.scatter(
        y_strip, results['dD_wave_density'], color=c_wave, s=36, zorder=5,
        edgecolors='white', linewidth=0.8, label="Strip Force Densities"
    )
    ax_drag.fill_between(y_fine, results['dD_fine'], alpha=0.15, color=c_wave)

    ax_drag.set_xlabel("Spanwise Coordinate $y$ [m]", fontsize=11, fontweight='bold')
    ax_drag.set_ylabel("Wave Drag Force Density $dD_{\\text{half}}/dy$ [N/m]", fontsize=11, fontweight='bold', color=c_wave)
    ax_drag.tick_params(axis='y', labelcolor=c_wave)
    ax_drag.set_title("Spanwise Wave Drag Distribution", fontsize=12, fontweight='bold', pad=10)
    ax_drag.set_xlim([0.0, b_tip * 1.02])
    ax_drag.set_ylim(bottom=0.0)
    ax_drag.grid(True, linestyle=':', alpha=0.6)

    # Secondary axis (twinx): Sectional Wave Drag Coefficient in Drag Counts
    ax_drag_twin = ax_drag.twinx()
    ax_drag_twin.plot(
        y_fine, results['cd_counts_fine'], '--', color=c_counts, linewidth=1.8, alpha=0.85,
        label=f"Sectional $c_{{d,\\text{{wave}}}}(y)$ (Net $C_{{D,\\text{{wave}}}} = {results['integrated_cd_wave_counts']:.2f}$ cts)"
    )
    ax_drag_twin.set_ylabel("Sectional Drag Coefficient $c_{d,\\text{wave}}$ [counts, $10^{-4}$]", fontsize=11, fontweight='bold', color=c_counts)
    ax_drag_twin.tick_params(axis='y', labelcolor=c_counts)
    ax_drag_twin.set_ylim(bottom=0.0)

    # Unified legend for Subplot 1
    lines_1, labels_1 = ax_drag.get_legend_handles_labels()
    lines_2, labels_2 = ax_drag_twin.get_legend_handles_labels()
    ax_drag.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize=8.5, framealpha=0.92)

    # Callout info box for drag metrics
    peak_dD = float(np.max(results['dD_fine']))
    y_peak_dD = float(y_fine[np.argmax(results['dD_fine'])])
    peak_counts = float(np.max(results['cd_counts_fine']))
    y_peak_counts = float(y_fine[np.argmax(results['cd_counts_fine'])])

    drag_info_text = (
        f"Cruise Wave Drag Metrics:\n"
        f"• Integrated $C_{{D,\\text{{wave}}}} = {results['integrated_cd_wave_counts']:.2f}$ drag counts\n"
        f"• Total Wave Force $D_{{\\text{{total}}}} = {results['tot_d_wave_total']:.1f}$ N ($D_{{\\text{{half}}}} = {results['tot_d_wave_half']:.1f}$ N)\n"
        f"• Peak Force Density: {peak_dD:.1f} N/m at $y = {y_peak_dD:.2f}$ m\n"
        f"• Peak Sectional Drag: {peak_counts:.1f} counts at $y = {y_peak_counts:.2f}$ m"
    )
    # Position callout in subcritical quiet zone (y ~ 5 to 13 m)
    ax_drag.text(
        0.14, 0.45, drag_info_text, transform=ax_drag.transAxes,
        fontsize=8.5, verticalalignment='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#fffbfa', edgecolor='#ffcdd2', alpha=0.94)
    )

    # -------------------------------------------------------------------------
    # Panel 2: Transonic Mach Envelope & Drag Divergence Diagnostics
    # -------------------------------------------------------------------------
    # Flight Mach line
    ax_mach.axhline(mach_cruise, color=c_minf, linestyle='-', linewidth=2.2, label=f"Flight Mach $M_\\infty = {mach_cruise:.2f}$")

    # Critical Mach and Drag Divergence Mach
    ax_mach.plot(y_fine, results['mcrit_fine'], '-', color=c_mcrit, linewidth=2.2, label="Critical Mach $M_{\\text{crit}}(y)$ (Shock Onset)")
    ax_mach.scatter(y_strip, results['strip_M_crit'], color=c_mcrit, s=28, zorder=5, alpha=0.7, edgecolors='white', linewidth=0.6)

    ax_mach.plot(y_fine, results['mdd_fine'], '-', color=c_mdd, linewidth=2.0, label="Drag Divergence Mach $M_{\\text{dd}}(y)$ ($dC_D/dM = 0.1$)")
    ax_mach.scatter(y_strip, results['strip_M_dd'], color=c_mdd, s=28, zorder=5, alpha=0.7, edgecolors='white', linewidth=0.6)

    # Shaded drag-rise zone where M_inf > M_crit
    active_mask = mach_cruise > results['mcrit_fine']
    if np.any(active_mask):
        ax_mach.fill_between(
            y_fine, results['mcrit_fine'], mach_cruise,
            where=active_mask, color='#ffcdd2', alpha=0.45,
            label="Active Drag Rise Zone ($M_\\infty > M_{\\text{crit}}$)"
        )

    ax_mach.set_xlabel("Spanwise Coordinate $y$ [m]", fontsize=11, fontweight='bold')
    ax_mach.set_ylabel("Mach Number $M$", fontsize=11, fontweight='bold')
    ax_mach.set_title("Transonic Mach Envelope & Drag Divergence Margin", fontsize=12, fontweight='bold', pad=10)
    ax_mach.set_xlim([0.0, b_tip * 1.02])
    y_min_m = min(float(np.min(results['mcrit_fine'])), mach_cruise) - 0.08
    y_max_m = max(float(np.max(results['mdd_fine'])), mach_cruise) + 0.06
    ax_mach.set_ylim([max(0.35, y_min_m), y_max_m])
    ax_mach.grid(True, linestyle=':', alpha=0.6)

    # Secondary axis (twinx): Geometric drivers (thickness-to-chord and sweep)
    ax_mach_twin = ax_mach.twinx()
    ax_mach_twin.plot(y_fine, results['sweep_deg_fine'], ':', color=c_geom, linewidth=1.5, alpha=0.8, label="Half-Chord Sweep $\\Lambda_{0.5}$ [deg]")
    ax_mach_twin.set_ylabel("Sweep $\\Lambda_{0.5}$ [deg]", fontsize=10, color=c_geom)
    ax_mach_twin.tick_params(axis='y', labelcolor=c_geom)

    # Unified legend for Subplot 2
    handles_m1, labels_m1 = ax_mach.get_legend_handles_labels()
    handles_m2, labels_m2 = ax_mach_twin.get_legend_handles_labels()
    ax_mach.legend(handles_m1 + handles_m2, labels_m1 + labels_m2, loc='upper right', fontsize=8.5, framealpha=0.92)

    # Diagnostic callout box
    min_mach_margin = float(np.min(results['strip_M_crit'] - mach_cruise))
    shock_exposure_pct = 100.0 * float(np.mean(results['strip_M_crit'] < mach_cruise))

    mach_info_text = (
        f"Korn–Lock Divergence Diagnostics:\n"
        f"• Technology Factor: $\\kappa_A = {KAPPA_A:.2f}$ (Conventional Supercritical)\n"
        f"• Drag Divergence Offset: $\\Delta M_{{\\text{{dd}}\\to\\text{{crit}}}} = {DELTA_M_DD_TO_CRIT:.3f}$\n"
        f"• Min Mach Margin $(M_{{\\text{{crit}}}} - M_\\infty)$: {min_mach_margin:+.4f}\n"
        f"• Active Drag-Rise Span Fraction: {shock_exposure_pct:.1f}% of wingspan"
    )
    ax_mach.text(
        0.04, 0.05, mach_info_text, transform=ax_mach.transAxes,
        fontsize=8.5, verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#cfd8dc', alpha=0.94)
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.87)

    # Save figure in output folder
    wave_fig_path = os.path.join(output_folder, 'wave_drag_distribution.png')
    plt.savefig(wave_fig_path, dpi=200, bbox_inches='tight')
    print(f"Spanwise wave drag distribution figure saved to: {wave_fig_path}")

    # Copy / save to artifact directory if available
    if artifact_dir and os.path.exists(artifact_dir):
        artifact_fig_path = os.path.join(artifact_dir, f"wave_drag_distribution_{run_label}.png")
        plt.savefig(artifact_fig_path, dpi=200, bbox_inches='tight')
        plt.savefig(os.path.join(artifact_dir, 'wave_drag_distribution.png'), dpi=200, bbox_inches='tight')
        print(f"Figure also saved to artifact directory: {artifact_fig_path}")

    plt.close()


def extract_and_plot_wave_drag_distribution(
    output_folder,
    jax_sim=None,
    main_script=None,
    artifact_dir=None,
    force_rerun=False,
):
    """Core analysis routine: extracts wave drag telemetry, processes strips, plots, and saves cache."""
    output_folder = os.path.abspath(output_folder)
    run_label = os.path.basename(output_folder)
    cache_file = os.path.join(output_folder, 'wave_drag_distribution_data.npz')
    shared_cache_file = os.path.join(output_folder, 'lift_and_moment_data.npz')

    if artifact_dir is None:
        candidate_ids = [
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

    mach_cruise = MACH_CRUISE_DEFAULT
    q_cruise = Q_CRUISE_DEFAULT

    # 1. Obtain Telemetry
    if jax_sim is not None and main_script is not None:
        print(f"\n--- Extracting Transonic Wave Drag Distribution for: {run_label} ---")
        y_strip = np.asarray(jax_sim[main_script.y_strip_pts]).flatten()
        strip_wave_cd = np.asarray(jax_sim[main_script.strip_wave_cd_cruise]).flatten()
        strip_M_crit = np.asarray(jax_sim[main_script.strip_M_crit_cruise]).flatten()
        strip_M_dd = np.asarray(jax_sim[main_script.strip_M_dd_cruise]).flatten()
        strip_tc = np.asarray(jax_sim[main_script.strip_tc]).flatten()
        strip_sweep = np.asarray(jax_sim[main_script.strip_sweep_halfchord]).flatten()
        strip_cl = np.asarray(jax_sim[main_script.cl_local_elem]).flatten()

        sref_val = float(np.asarray(jax_sim[main_script.planform_area]).flatten()[0]) if hasattr(main_script, 'planform_area') else 10.0
        ar_val = float(np.asarray(jax_sim[main_script.aspect_ratio_calc]).flatten()[0]) if hasattr(main_script, 'aspect_ratio_calc') else 10.0
        cd_wave_val = float(np.asarray(jax_sim[main_script.CD_wave_cruise]).flatten()[0])
        d_wave_val = float(np.asarray(jax_sim[main_script.D_wave_cruise]).flatten()[0])

        try:
            local_chord_drag = np.asarray(jax_sim[main_script.local_chord_drag]).flatten()
        except Exception:
            local_chord_drag = None

        try:
            dy_strip = np.asarray(jax_sim[main_script.dy_strip]).flatten()
        except Exception:
            dy_strip = None

        try:
            strip_area = np.asarray(jax_sim[main_script.strip_area]).flatten()
        except Exception:
            strip_area = None

    elif os.path.exists(cache_file) and not force_rerun:
        print(f"Loading cached wave drag telemetry from: {cache_file}")
        data = np.load(cache_file)
        y_strip = data['y_strip']
        strip_wave_cd = data['strip_wave_cd']
        strip_M_crit = data['strip_M_crit']
        strip_M_dd = data['strip_M_dd']
        strip_tc = data['strip_tc']
        strip_sweep = data['strip_sweep']
        strip_cl = data['strip_cl']
        local_chord_drag = data['local_chord_drag'] if 'local_chord_drag' in data else None
        dy_strip = data['dy_strip'] if 'dy_strip' in data else None
        strip_area = data['strip_area'] if 'strip_area' in data else None
        sref_val = float(data['sref_val'])
        ar_val = float(data['ar_val'])
        cd_wave_val = float(data['cd_wave_val'])
        d_wave_val = float(data['d_wave_val'])

    elif os.path.exists(shared_cache_file) and not force_rerun:
        print(f"Loading aerodynamic telemetry from: {shared_cache_file}")
        data = np.load(shared_cache_file)
        sref_val = float(data['sref_val']) if 'sref_val' in data else 10.0
        ar_val = float(data['ar_val']) if 'ar_val' in data else 10.0
        cd_wave_val = float(data['cd_wave_val']) if 'cd_wave_val' in data else 0.0
        d_wave_val = float(data['d_wave_val']) if 'd_wave_val' in data else 0.0
        strip_tc = data['strip_tc']
        strip_sweep = data['strip_sweep_halfchord']

        # If y_strip or strip_cl not directly stored in shared cache, evaluate simulator
        x_out_path = os.path.join(output_folder, 'x.out')
        if os.path.exists(x_out_path):
            print(f"Evaluating exact simulator variables from {x_out_path}...")
            import examples.showcase_examples.rectangular_wing.ex_rectangular_wing_to_bwb as main_script
            jax_sim = main_script.jax_sim
            x_history = np.loadtxt(x_out_path)
            x_opt = x_history[-1] if x_history.ndim > 1 else x_history

            curr_idx = 0
            for name, dv_info in main_script.design_variables.items():
                var_size = int(np.prod(dv_info.variable.shape))
                slc = slice(curr_idx, curr_idx + var_size)
                unscaled_val = (x_opt[slc] / dv_info.scaler).reshape(dv_info.variable.shape)
                jax_sim[dv_info.variable] = unscaled_val
                curr_idx += var_size

            jax_sim.run()

            y_strip = np.asarray(jax_sim[main_script.y_strip_pts]).flatten()
            strip_wave_cd = np.asarray(jax_sim[main_script.strip_wave_cd_cruise]).flatten()
            strip_M_crit = np.asarray(jax_sim[main_script.strip_M_crit_cruise]).flatten()
            strip_M_dd = np.asarray(jax_sim[main_script.strip_M_dd_cruise]).flatten()
            strip_tc = np.asarray(jax_sim[main_script.strip_tc]).flatten()
            strip_sweep = np.asarray(jax_sim[main_script.strip_sweep_halfchord]).flatten()
            strip_cl = np.asarray(jax_sim[main_script.cl_local_elem]).flatten()
            local_chord_drag = None
            dy_strip = None
            strip_area = None
        else:
            raise FileNotFoundError(f"Cannot reconstruct strip wave drag without x.out or complete cache in {output_folder}")

    else:
        # Re-run from x.out
        x_out_path = os.path.join(output_folder, 'x.out')
        if not os.path.exists(x_out_path):
            raise FileNotFoundError(f"x.out not found in {output_folder}")
        print(f"Evaluating simulator at optimal design variables from: {x_out_path}")
        import examples.showcase_examples.rectangular_wing.ex_rectangular_wing_to_bwb as main_script
        jax_sim = main_script.jax_sim
        x_history = np.loadtxt(x_out_path)
        x_opt = x_history[-1] if x_history.ndim > 1 else x_history

        curr_idx = 0
        for name, dv_info in main_script.design_variables.items():
            var_size = int(np.prod(dv_info.variable.shape))
            slc = slice(curr_idx, curr_idx + var_size)
            unscaled_val = (x_opt[slc] / dv_info.scaler).reshape(dv_info.variable.shape)
            jax_sim[dv_info.variable] = unscaled_val
            curr_idx += var_size

        jax_sim.run()

        y_strip = np.asarray(jax_sim[main_script.y_strip_pts]).flatten()
        strip_wave_cd = np.asarray(jax_sim[main_script.strip_wave_cd_cruise]).flatten()
        strip_M_crit = np.asarray(jax_sim[main_script.strip_M_crit_cruise]).flatten()
        strip_M_dd = np.asarray(jax_sim[main_script.strip_M_dd_cruise]).flatten()
        strip_tc = np.asarray(jax_sim[main_script.strip_tc]).flatten()
        strip_sweep = np.asarray(jax_sim[main_script.strip_sweep_halfchord]).flatten()
        strip_cl = np.asarray(jax_sim[main_script.cl_local_elem]).flatten()
        sref_val = float(np.asarray(jax_sim[main_script.planform_area]).flatten()[0]) if hasattr(main_script, 'planform_area') else 10.0
        ar_val = float(np.asarray(jax_sim[main_script.aspect_ratio_calc]).flatten()[0]) if hasattr(main_script, 'aspect_ratio_calc') else 10.0
        cd_wave_val = float(np.asarray(jax_sim[main_script.CD_wave_cruise]).flatten()[0])
        d_wave_val = float(np.asarray(jax_sim[main_script.D_wave_cruise]).flatten()[0])
        local_chord_drag = None
        dy_strip = None
        strip_area = None

    # 2. Process Strips & Build Splines
    results = process_wave_drag_strips(
        y_strip_pts=y_strip,
        strip_wave_cd=strip_wave_cd,
        strip_M_crit=strip_M_crit,
        strip_M_dd=strip_M_dd,
        strip_tc=strip_tc,
        strip_sweep_halfchord=strip_sweep,
        strip_cl=strip_cl,
        local_chord_drag=local_chord_drag,
        dy_strip=dy_strip,
        strip_area=strip_area,
        mach_cruise=mach_cruise,
        q_cruise=q_cruise,
        sref_val=sref_val,
    )

    # 3. Save Cache for instantaneous reload
    try:
        np.savez_compressed(
            cache_file,
            y_strip=y_strip,
            strip_wave_cd=strip_wave_cd,
            strip_M_crit=strip_M_crit,
            strip_M_dd=strip_M_dd,
            strip_tc=strip_tc,
            strip_sweep=strip_sweep,
            strip_cl=strip_cl,
            local_chord_drag=results['local_chord_drag'],
            dy_strip=results['dy_strip'],
            strip_area=results['strip_area'],
            sref_val=sref_val,
            ar_val=ar_val,
            cd_wave_val=results['integrated_cd_wave'],
            d_wave_val=results['tot_d_wave_total'],
            mach_cruise=mach_cruise,
            q_cruise=q_cruise,
        )
        print(f"Saved wave drag telemetry cache to: {cache_file}")
    except Exception as e:
        print(f"Warning: could not save cache to {cache_file}: {e}")

    # 4. Generate Plot
    plot_wave_drag_distribution(
        results=results,
        output_folder=output_folder,
        run_label=run_label,
        sref_val=sref_val,
        ar_val=ar_val,
        mach_cruise=mach_cruise,
        q_cruise=q_cruise,
        artifact_dir=artifact_dir,
    )

    # 5. Print Telemetry Summary
    b_tip = results['b_tip']
    min_mach_margin = float(np.min(strip_M_crit - mach_cruise))
    shock_exposure_pct = 100.0 * float(np.mean(strip_M_crit < mach_cruise))
    peak_cd_counts = float(np.max(results['cd_wave_counts']))
    idx_peak_cd = int(np.argmax(results['cd_wave_counts']))
    y_peak_cd = float(y_strip[idx_peak_cd])

    print("\n" + "=" * 65)
    print(f"TRANSONIC WAVE DRAG TELEMETRY SUMMARY (Run: {run_label})")
    print("=" * 65)
    print(f"Wing Half-Span (b/2):            {b_tip:.3f} m (Full Wingspan: {2*b_tip:.3f} m)")
    print(f"Calculated Aspect Ratio (AR):    {ar_val:.2f}")
    print(f"Planform Area (S_ref):           {sref_val:.3f} m²")
    print(f"Cruise Mach Number (M_inf):      {mach_cruise:.4f}")
    print(f"Cruise Dynamic Pressure (q_inf): {q_cruise:,.2f} Pa (30,000 ft ISA)")
    print(f"Number of Drag Strips:           {len(y_strip)}")
    print(f"Integrated Wave Drag (CD_wave):  {results['integrated_cd_wave']:.6f} ({results['integrated_cd_wave_counts']:.2f} drag counts)")
    print(f"Integrated Wave Drag Force:      {results['tot_d_wave_total']:.1f} N (Half-Wing: {results['tot_d_wave_half']:.1f} N)")
    print(f"Minimum Mach Margin (Mcrit - M): {min_mach_margin:+.4f} ({'Active Drag Rise' if min_mach_margin < 0 else 'Shock Free'})")
    print(f"Peak Sectional cd_wave:          {peak_cd_counts:.2f} counts at y = {y_peak_cd:.2f} m")
    print(f"Drag Divergence Span Fraction:   {shock_exposure_pct:.1f}% of wingspan")
    print("=" * 65 + "\n")

    return results


def main():
    if len(sys.argv) > 1:
        target_folder = os.path.abspath(sys.argv[1])
    else:
        candidate_folders = []
        for base_dir in KNOWN_OUTPUT_DIRS:
            if os.path.exists(base_dir):
                candidate_folders.extend(glob.glob(os.path.join(base_dir, '*')))
        if not candidate_folders:
            raise FileNotFoundError("No optimization output folders found in known output directories.")
        target_folder = max(candidate_folders, key=os.path.getmtime)

    print(f"Target optimization output directory: {target_folder}")
    extract_and_plot_wave_drag_distribution(target_folder)


if __name__ == '__main__':
    main()
