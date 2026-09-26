"""Generate structural analysis plots:
1. Spar cap (ttop) and shear web (tweb) thickness distributions across the span.
2. Authentic beam element stresses and continuous B-spline stress distribution vs allowable stress limit.
3. Additional cross-sectional geometry (wingbox width, height, and chord) along the span.
4. Structural margin of safety profile across the span.

Supports:
- Auto-discovery of the latest run output directory or specified output path via sys.argv[1]
- Instantaneous re-plotting via cached structural_data.npz
- Full fallback to evaluate the optimal state using jax_sim if cache is not yet generated
- Multi-resolution compatibility ('fast' 5-station / 15-CP stress spline vs 'full' 8-station)
"""
import os
import sys
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt

# Ensure all relevant packages are on sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, '../../../..'))
opt_root = os.path.abspath(os.path.join(repo_root, '..'))

for p in [
    os.path.join(opt_root, 'lsdo_b_splines_cython'),
    os.path.join(opt_root, 'VortexAD'),
    os.path.join(opt_root, 'aframe'),
    os.path.join(opt_root, 'csdl'),
    os.path.join(opt_root, 'CSDL_alpha'),
    os.path.join(opt_root, 'lsdo_geo'),
    os.path.join(opt_root, 'lsdo_function_spaces'),
    os.path.join(opt_root, 'modopt'),
    os.path.join(repo_root, 'examples/showcase_examples/rectangular_wing'),
]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

import lsdo_function_spaces as lfs


def generate_structural_plots(output_folder=None, artifact_dir=None, force_rerun=False):
    """Generate structural thickness and stress analysis plots and copy to artifact directory."""
    if output_folder is None:
        if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
            arg_target = os.path.abspath(sys.argv[1])
            if os.path.isfile(arg_target) and arg_target.endswith('.npz'):
                cache_file = arg_target
                output_folder = os.path.dirname(arg_target)
            else:
                output_folder = arg_target
                cache_file = os.path.join(output_folder, 'structural_data.npz')
        else:
            output_base_dir = os.path.join(repo_root, 'rectangular_wing_to_bwb_aerostructural_optimization_outputs')
            output_folders = glob.glob(os.path.join(output_base_dir, '*'))
            if not output_folders:
                raise FileNotFoundError(f"No optimization output runs found in {output_base_dir}")
            output_folder = max(output_folders, key=os.path.getmtime)
            cache_file = os.path.join(output_folder, 'structural_data.npz')
    else:
        output_folder = os.path.abspath(output_folder)
        cache_file = os.path.join(output_folder, 'structural_data.npz')

    print(f"Target optimization output directory: {output_folder}")

    if artifact_dir is None:
        candidate_ids = [
            '0c0a47e5-2e16-41bb-9139-10357c23c5ee',
            '3256dd7c-d4c8-4c73-887d-131361f9d0c3',
            'f4b8c9ca-9e69-4a92-8ee3-962e5e21793d',
            '680209fd-293d-4fa0-9f6c-ae59a72a6987',
        ]
        artifact_dir = os.environ.get('ARTIFACT_DIR', None)
        if not artifact_dir or not os.path.exists(artifact_dir):
            for c_id in candidate_ids:
                cand_path = f'/home/andrew/.gemini/antigravity/brain/{c_id}'
                if os.path.exists(cand_path):
                    artifact_dir = cand_path
                    break

    if '--force' in sys.argv or '-f' in sys.argv:
        force_rerun = True

    # -------------------------------------------------------------------------
    # Load Cached Telemetry or Run Model Evaluation Fallback
    # -------------------------------------------------------------------------
    if os.path.exists(cache_file) and not force_rerun:
        print(f"Loading cached structural telemetry from: {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
    else:
        alt_candidates = (
            glob.glob(os.path.join(output_folder, '*structural*.npz')) +
            glob.glob(os.path.join(output_folder, '*telemetry*.npz'))
        )
        if alt_candidates and not force_rerun:
            alt_file = alt_candidates[0]
            print(f"Loading existing telemetry cache from: {alt_file}")
            data = np.load(alt_file, allow_pickle=True)
        else:
            print(f"No cache found at {cache_file}; running model evaluation...")
            import csdl_alpha as csdl
            recorder = csdl.Recorder(inline=True)
            recorder.start()
            import examples.showcase_examples.rectangular_wing.ex_rectangular_wing_to_bwb as main_script
            jax_sim = main_script.jax_sim
            design_variables = main_script.design_variables

            x_out_path = os.path.join(output_folder, 'x.out')
            if not os.path.exists(x_out_path):
                raise FileNotFoundError(f"Optimization history file not found: {x_out_path}")
            x_history = np.loadtxt(x_out_path)
            if len(x_history.shape) == 1:
                x_history = x_history.reshape(1, -1)
            x_opt = x_history[-1]

            curr_idx = 0
            for name, dv_info in design_variables.items():
                var_size = int(np.prod(dv_info.variable.shape))
                slc = slice(curr_idx, curr_idx + var_size)
                unscaled_val = (x_opt[slc] / dv_info.scaler).reshape(dv_info.variable.shape)
                jax_sim[dv_info.variable] = unscaled_val
                curr_idx += var_size

            jax_sim.run()

            beam_pts_opt = np.asarray(jax_sim[main_script.beam_mesh])
            scale_factor_val = float(main_script.scale_factor)
            allowable_stress_val = float(main_script.allowable_stress)
            chords_opt = np.asarray(jax_sim[main_script.local_chord]).flatten()
            heights_opt = np.asarray(jax_sim[main_script.local_height]).flatten()
            widths_opt = np.asarray(jax_sim[main_script.box_width]).flatten()
            ttop_elem_opt = np.asarray(jax_sim[main_script.ttop_elem]).flatten()
            tweb_elem_opt = np.asarray(jax_sim[main_script.tweb_elem]).flatten()
            elem_stress_ss = np.asarray(jax_sim[main_script.elem_max_stress]).flatten()
            dv_stress_ss = np.asarray(jax_sim[main_script.dv_stresses]).flatten()
            stress_coeffs_ss = np.asarray(jax_sim[main_script.stress_coeffs]).flatten()
            ttop_dvs_opt = np.asarray(jax_sim[main_script.ttop_dvs]).flatten()
            tweb_dvs_opt = np.asarray(jax_sim[main_script.tweb_dvs]).flatten()
            thickness_peaks = np.asarray(main_script.thickness_peaks)
            knots_stress_15 = np.asarray(main_script.knots_stress_15) if hasattr(main_script, 'knots_stress_15') else np.array([])
            resolution_val = str(main_script.resolution)
            load_factor_val = float(main_script.load_factor_val) if hasattr(main_script, 'load_factor_val') else 2.5

            np.savez_compressed(
                cache_file,
                beam_pts_opt=beam_pts_opt,
                scale_factor=scale_factor_val,
                allowable_stress=allowable_stress_val,
                chords_opt=chords_opt,
                heights_opt=heights_opt,
                widths_opt=widths_opt,
                ttop_elem_opt=ttop_elem_opt,
                tweb_elem_opt=tweb_elem_opt,
                elem_stress_ss=elem_stress_ss,
                dv_stress_ss=dv_stress_ss,
                stress_coeffs_ss=stress_coeffs_ss,
                ttop_dvs_opt=ttop_dvs_opt,
                tweb_dvs_opt=tweb_dvs_opt,
                thickness_peaks=thickness_peaks,
                knots_stress_15=knots_stress_15,
                resolution=resolution_val,
                load_factor_val=load_factor_val,
            )
            print(f"Saved structural telemetry cache to: {cache_file}")
            data = np.load(cache_file, allow_pickle=True)

    # Unpack telemetry
    beam_pts = data['beam_pts_opt']
    scale_factor = float(data['scale_factor'])
    allowable_stress = float(data['allowable_stress']) / 1e6
    elem_stress = data['elem_stress_ss'] / 1e6
    dv_stress = data['dv_stress_ss'] / 1e6
    stress_coeffs = data['stress_coeffs_ss'] / 1e6
    ttop_elem = data['ttop_elem_opt'] * 1e3
    tweb_elem = data['tweb_elem_opt'] * 1e3
    ttop_dvs = data['ttop_dvs_opt'] * 1e3
    tweb_dvs = data['tweb_dvs_opt'] * 1e3
    thickness_peaks = data['thickness_peaks']
    chords = data['chords_opt']
    heights = data['heights_opt']
    widths = data['widths_opt']
    resolution = str(data['resolution']) if 'resolution' in data else 'fast'
    load_factor = float(data['load_factor_val']) if 'load_factor_val' in data else 2.5

    # -------------------------------------------------------------------------
    # Spatial Coordinates and B-Spline Continuous Field Evaluation
    # -------------------------------------------------------------------------
    y_nodes = beam_pts[:, 1]
    half_span = float(np.max(y_nodes))
    y_elem_mid = 0.5 * (y_nodes[:-1] + y_nodes[1:])
    y_peaks = thickness_peaks * half_span
    num_dvs = len(thickness_peaks)

    # Continuous B-Spline Stress Field
    N_EVAL = 300
    y_fine = np.linspace(0.0, half_span, N_EVAL)
    tau_fine = y_fine / half_span

    if 'knots_stress_15' in data and len(data['knots_stress_15']) > 0:
        knots = data['knots_stress_15']
    else:
        num_stress_cp = len(stress_coeffs)
        k_int = np.linspace(0.0, 1.0, num_stress_cp - 2)[1:-1]
        knots = np.concatenate([[0.0, 0.0, 0.0, 0.0], k_int, [1.0, 1.0, 1.0, 1.0]])

    space_stress = lfs.BSplineSpace(
        num_parametric_dimensions=1,
        degree=3,
        coefficients_shape=(len(stress_coeffs),),
        knots=knots,
    )
    B_eval = space_stress.compute_basis_matrix(tau_fine.reshape((-1, 1))).toarray()
    stress_field = np.asarray(B_eval @ stress_coeffs).flatten()

    margin_of_safety_field = (allowable_stress / np.maximum(stress_field, 1e-3)) - 1.0

    # -------------------------------------------------------------------------
    # Plotting: 4-Panel Aerostructural Analysis
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=200)
    fig.suptitle(
        f"Transonic BWB Wingbox Structural Analysis ({resolution.upper()} Resolution, {load_factor:.1f}g Pull-Up Maneuver)\n"
        f"Half-Span: b/2 = {half_span:.2f} m | Allowable Stress $\\sigma_{{\\mathrm{{allow}}}} = {allowable_stress:.1f}$ MPa",
        fontsize=14, fontweight='bold', y=0.98
    )

    color_stress = "#b2182b"
    color_allow = "#2166ac"
    color_top = "#2c7bb6"
    color_web = "#d7191c"
    color_chord = "#4d4d4d"
    color_height = "#1a9641"
    color_width = "#fdae61"

    # --- Subplot 1: Spar Cap and Web Thickness ---
    ax_t = axes[0, 0]
    ax_t.plot(y_elem_mid, ttop_elem, 'o-', color=color_top, linewidth=2.2, markersize=5, label='Spar Cap Thickness ($t_{\\mathrm{top}}$)')
    ax_t.plot(y_elem_mid, tweb_elem, 's-', color=color_web, linewidth=2.2, markersize=5, label='Shear Web Thickness ($t_{\\mathrm{web}}$)')
    ax_t.scatter(y_peaks, ttop_dvs, color=color_top, s=70, marker='D', edgecolor='black', zorder=5, label='Design Variables ($t_{\\mathrm{top}}$)')
    ax_t.scatter(y_peaks, tweb_dvs, color=color_web, s=70, marker='D', edgecolor='black', zorder=5, label='Design Variables ($t_{\\mathrm{web}}$)')

    ax_t.set_title("Internal Spar Sizing: Cap & Web Thickness Distributions", fontsize=11, fontweight='bold')
    ax_t.set_xlabel("Spanwise Coordinate $y$ [m]", fontsize=10, fontweight='bold')
    ax_t.set_ylabel("Thickness [mm]", fontsize=10, fontweight='bold')
    ax_t.set_xlim([0, half_span * 1.02])
    ax_t.set_ylim(bottom=0.0)
    ax_t.grid(True, linestyle=":", alpha=0.6)
    ax_t.legend(loc='upper right', frameon=True, framealpha=0.92, fontsize=8.5)

    # --- Subplot 2: Cross-Sectional Geometry ---
    ax_g = axes[0, 1]
    ax_g.plot(y_elem_mid, chords, '^-', color=color_chord, linewidth=2.2, markersize=5, label='Local Chord ($c$)')
    ax_g.set_ylabel("Chord [m]", fontsize=10, fontweight='bold', color=color_chord)
    ax_g.tick_params(axis='y', labelcolor=color_chord)
    ax_g.set_xlabel("Spanwise Coordinate $y$ [m]", fontsize=10, fontweight='bold')
    ax_g.set_xlim([0, half_span * 1.02])
    ax_g.set_ylim(bottom=0.0)
    ax_g.grid(True, linestyle=":", alpha=0.6)

    ax_g2 = ax_g.twinx()
    ax_g2.plot(y_elem_mid, heights * 100.0, 'o-', color=color_height, linewidth=2.0, markersize=4, label='Box Height ($h_{\\mathrm{box}}$)')
    ax_g2.plot(y_elem_mid, widths * 100.0, 's-', color=color_width, linewidth=2.0, markersize=4, label='Box Width ($w_{\\mathrm{box}}$)')
    ax_g2.set_ylabel("Box Height & Width [cm]", fontsize=10, fontweight='bold', color=color_height)
    ax_g2.tick_params(axis='y', labelcolor=color_height)
    ax_g2.set_ylim(bottom=0.0)

    lines_1, labels_1 = ax_g.get_legend_handles_labels()
    lines_2, labels_2 = ax_g2.get_legend_handles_labels()
    ax_g.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', frameon=True, framealpha=0.92, fontsize=8.5)
    ax_g.set_title("Wingbox Sectional Dimensions (Chord, Height, Width)", fontsize=11, fontweight='bold')

    # --- Subplot 3: Stress vs Allowable Limit ---
    ax_s = axes[1, 0]
    ax_s.plot(y_fine, stress_field, '-', color=color_stress, linewidth=2.5, label='Continuous Stress Spline $\\sigma(y)$')
    ax_s.step(np.concatenate([[y_nodes[0]], y_nodes[1:]]),
              np.concatenate([[elem_stress[0]], elem_stress]),
              where='pre', color='#fd8d3c', linestyle='--', linewidth=1.5, label='Beam Element Stress (Discrete)')
    ax_s.scatter(y_elem_mid, elem_stress, color='#fd8d3c', s=35, zorder=4, edgecolor='black', linewidth=0.5)
    ax_s.scatter(y_peaks, dv_stress, marker='s', s=85, color='darkred', edgecolor='black', linewidth=1.2, zorder=6, label='Aggregated Station Stress (Constraints)')
    ax_s.axhline(allowable_stress, color=color_allow, linestyle='--', linewidth=2.2, label=f'Allowable Stress Limit ({allowable_stress:.1f} MPa)')

    ax_s.set_title(f"Von Mises Stress Distribution vs Yield Limit ({load_factor:.1f}g Pull-Up)", fontsize=11, fontweight='bold')
    ax_s.set_xlabel("Spanwise Coordinate $y$ [m]", fontsize=10, fontweight='bold')
    ax_s.set_ylabel("Peak Stress $\\sigma$ [MPa]", fontsize=10, fontweight='bold')
    ax_s.set_xlim([0, half_span * 1.02])
    ax_s.set_ylim([0, max(allowable_stress * 1.18, np.max(stress_field) * 1.12)])
    ax_s.grid(True, linestyle=":", alpha=0.6)
    ax_s.legend(loc='lower left', frameon=True, framealpha=0.92, fontsize=8.5)

    # --- Subplot 4: Margin of Safety ---
    ax_m = axes[1, 1]
    ax_m.plot(y_fine, margin_of_safety_field, color='#252525', linewidth=2.2, label='Margin of Safety $\\mathrm{MS}(y)$')
    ax_m.axhline(0.0, color='red', linestyle='--', linewidth=1.8, label='Critical Limit ($\\mathrm{MS} = 0$)')
    ax_m.fill_between(y_fine, margin_of_safety_field, 0.0, where=(margin_of_safety_field >= 0.0),
                      color='#31a354', alpha=0.25, interpolate=True, label='Safe Structural Region ($\\mathrm{MS} > 0$)')
    ax_m.fill_between(y_fine, margin_of_safety_field, 0.0, where=(margin_of_safety_field < 0.0),
                      color='#de2d26', alpha=0.35, interpolate=True, label='Yield Failure Region ($\\mathrm{MS} < 0$)')

    ax_m.set_title("Structural Margin of Safety $\\mathrm{MS} = (\\sigma_{\\mathrm{allow}} / \\sigma) - 1$", fontsize=11, fontweight='bold')
    ax_m.set_xlabel("Spanwise Coordinate $y$ [m]", fontsize=10, fontweight='bold')
    ax_m.set_ylabel("Margin of Safety [-]", fontsize=10, fontweight='bold')
    ax_m.set_xlim([0, half_span * 1.02])
    ax_m.set_ylim([-0.15, min(4.0, max(2.0, np.max(margin_of_safety_field[:int(N_EVAL * 0.85)]) * 1.2))])
    ax_m.grid(True, linestyle=":", alpha=0.6)
    ax_m.legend(loc='upper right', frameon=True, framealpha=0.92, fontsize=8.5)

    ax_m.annotate(
        "Inboard Sizing Active\n$\\mathrm{MS} \\approx 0.00$ (Fully Stressed Spar)",
        xy=(0.05 * half_span, 0.0), xytext=(0.25 * half_span, 0.8),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.3),
        fontsize=8.5, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#ddffdd", ec="gray", lw=0.8)
    )

    plt.tight_layout()

    out_fig_name = 'structural_thickness_and_stress_analysis.png'
    out_local = os.path.join(output_folder, out_fig_name)
    plt.savefig(out_local, dpi=200, bbox_inches='tight')
    print(f"\nUpdated structural distribution figure saved to: {out_local}")

    if artifact_dir and os.path.exists(artifact_dir):
        out_artifact = os.path.join(artifact_dir, out_fig_name)
        shutil.copy2(out_local, out_artifact)
        print(f"Figure also saved to artifact directory: {out_artifact}")

    plt.close()

    # -------------------------------------------------------------------------
    # Terminal Summary Table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"STRUCTURAL TELEMETRY SUMMARY (Run: {os.path.basename(output_folder)})")
    print("=" * 80)
    print(f"Wing Half-Span (b/2):            {half_span:.3f} m")
    print(f"Allowable Stress (sigma_allow):  {allowable_stress:.1f} MPa")
    print(f"Peak Element Stress:             {np.max(elem_stress):.2f} MPa")
    print(f"Peak Continuous Spline Stress:   {np.max(stress_field):.2f} MPa")
    print(f"Number of Beam Elements:         {len(y_elem_mid)}")
    print(f"Number of Thickness Stations:    {num_dvs}")
    print("-" * 80)
    print(f"{'Elem':4s} | {'y_mid [m]':9s} | {'Chord [m]':9s} | {'Width [cm]':10s} | {'Height [cm]':11s} | {'ttop [mm]':9s} | {'tweb [mm]':9s} | {'Stress [MPa]':12s}")
    print("-" * 80)
    for i in range(len(y_elem_mid)):
        print(f"{i:4d} | {y_elem_mid[i]:9.3f} | {chords[i]:9.3f} | {widths[i]*100.0:10.2f} | {heights[i]*100.0:11.2f} | {ttop_elem[i]:9.2f} | {tweb_elem[i]:9.2f} | {elem_stress[i]:12.2f}")
    print("-" * 80)
    print(f"{'Station':7s} | {'eta_peak':8s} | {'Span y [m]':10s} | {'Aggregated Stress [MPa]':24s} | {'Allowable [MPa]':16s} | {'Status':8s}")
    print("-" * 80)
    for j in range(num_dvs):
        st_val = dv_stress[j]
        status = "FEASIBLE" if st_val <= (allowable_stress + 1e-4) else "VIOLATED"
        print(f"{j:7d} | {thickness_peaks[j]:8.4f} | {y_peaks[j]:10.3f} | {st_val:24.2f} | {allowable_stress:16.1f} | {status:8s}")
    print("=" * 80 + "\n")

    return out_local


if __name__ == '__main__':
    generate_structural_plots()
