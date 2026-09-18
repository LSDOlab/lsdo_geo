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
import os, sys
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

# -------------------------------------------------------------------------
# Determine Target Optimization Output Directory & Cache Path
# -------------------------------------------------------------------------
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

print(f"Target optimization output directory: {output_folder}")
artifact_dir = os.environ.get('ARTIFACT_DIR', '/home/andrew/.gemini/antigravity/brain/f4b8c9ca-9e69-4a92-8ee3-962e5e21793d')

force_rerun = '--force' in sys.argv or '-f' in sys.argv

# -------------------------------------------------------------------------
# Load Cached Telemetry or Run Model Evaluation Fallback
# -------------------------------------------------------------------------
if os.path.exists(cache_file) and not force_rerun:
    print(f"Loading cached structural telemetry from: {cache_file}")
    data = np.load(cache_file, allow_pickle=True)
else:
    # Check for existing telemetry files with matching structural data
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

        # Set optimal design variables on jax_sim
        curr_idx = 0
        for name, dv_info in design_variables.items():
            var_size = int(np.prod(dv_info.variable.shape))
            slc = slice(curr_idx, curr_idx + var_size)
            unscaled_val = (x_opt[slc] / dv_info.scaler).reshape(dv_info.variable.shape)
            jax_sim[dv_info.variable] = unscaled_val
            curr_idx += var_size

        # Evaluate model at optimal point
        jax_sim.run()

        # Extract structural telemetry
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

        # Save cache for instant reuse
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

# -------------------------------------------------------------------------
# Parse Structural Parameters & Normalization
# -------------------------------------------------------------------------
beam_pts = data['beam_pts_opt']
half_span = float(beam_pts[-1, 1])
scale_factor = float(data['scale_factor']) if 'scale_factor' in data else 1.0
allowable_stress = float(data['allowable_stress']) / 1e6  # to MPa
load_factor_val = float(data['load_factor_val']) if 'load_factor_val' in data else 2.5

# Element geometry & locations
y_elem_mid = 0.5 * (beam_pts[:-1, 1] + beam_pts[1:, 1])
chords = data['chords_opt']
heights = data['heights_opt']
widths = data['widths_opt']
ttop_elem = data['ttop_elem_opt'] * 1e3  # mm
tweb_elem = data['tweb_elem_opt'] * 1e3  # mm

# Stresses with flexible key fallback (supports 'ss', '4g', or generic keys)
elem_stress = (data['elem_stress_ss'] if 'elem_stress_ss' in data else (
    data['elem_stress_4g'] if 'elem_stress_4g' in data else data['elem_stress']
)) / 1e6  # MPa

dv_stress = (data['dv_stress_ss'] if 'dv_stress_ss' in data else (
    data['dv_stress_4g'] if 'dv_stress_4g' in data else data['dv_stress']
)) / 1e6  # MPa

stress_coeffs_arr = (data['stress_coeffs_ss'] if 'stress_coeffs_ss' in data else (
    data['stress_coeffs_4g'] if 'stress_coeffs_4g' in data else data['stress_coeffs']
))

thickness_peaks = data['thickness_peaks']
y_peaks = thickness_peaks * half_span
ttop_dvs_opt = data['ttop_dvs_opt']
tweb_dvs_opt = data['tweb_dvs_opt']

# -------------------------------------------------------------------------
# Continuous B-Spline Field Evaluations
# -------------------------------------------------------------------------
y_fine = np.linspace(0.0, half_span, 400)
u_fine = (y_fine / half_span).reshape(-1, 1)

# 1. Thickness Distributions (Degree 2 B-Spline)
num_dvs = len(ttop_dvs_opt)
sp_thick = lfs.BSplineSpace(num_parametric_dimensions=1, degree=2, coefficients_shape=(num_dvs,))
B_thick = sp_thick.compute_basis_matrix(u_fine).toarray()

ttop_fine = (B_thick @ ttop_dvs_opt) * 1e3  # mm
tweb_fine = (B_thick @ tweb_dvs_opt) * 1e3  # mm
ttop_cps = ttop_dvs_opt * 1e3
tweb_cps = tweb_dvs_opt * 1e3
y_cps = thickness_peaks * half_span

# 2. Stress Field B-Spline (Supports 15-CP cubic clamped with S'(0)=0 or 8-CP quadratic)
knots_stress_15 = data['knots_stress_15'] if ('knots_stress_15' in data and len(data['knots_stress_15']) > 0) else None
if len(stress_coeffs_arr) == 15 and knots_stress_15 is not None:
    sp_stress = lfs.BSplineSpace(
        num_parametric_dimensions=1,
        degree=3,
        coefficients_shape=(15,),
        knots=(knots_stress_15,)
    )
elif len(stress_coeffs_arr) == 8:
    sp_stress = lfs.BSplineSpace(
        num_parametric_dimensions=1,
        degree=2,
        coefficients_shape=(8,)
    )
else:
    sp_stress = lfs.BSplineSpace(
        num_parametric_dimensions=1,
        degree=min(2, len(stress_coeffs_arr) - 1),
        coefficients_shape=(len(stress_coeffs_arr),)
    )

B_stress = sp_stress.compute_basis_matrix(u_fine).toarray()
stress_field = (B_stress @ stress_coeffs_arr) / 1e6  # MPa

# -------------------------------------------------------------------------
# Figure: Comprehensive Structural Analysis (2 x 2 Subplots)
# -------------------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(2, 2, figsize=(14, 10.5), dpi=200)
fig.suptitle(
    f"Structural Optimization Analysis: Thickness & Stress Distributions\n"
    f"Structural Sizing Maneuver ({load_factor_val:.1f}g) | Allowable Stress = {allowable_stress:.1f} MPa | Half-Span $b/2 = {half_span:.2f}$ m",
    fontsize=13, fontweight='bold', y=0.98
)

# Colors
c_cap = '#1f77b4'
c_web = '#2ca02c'
c_stress = '#d62728'
c_elem = '#9467bd'
c_allow = '#b2182b'
c_box = '#ff7f0e'

# ---------------- Subplot (0, 0): Spar Cap & Web Thickness Distributions ----------------
ax_t = axes[0, 0]
ax_t.plot(y_fine, ttop_fine, '-', color=c_cap, linewidth=2.5, label=r'Spar Cap Thickness $t_{\mathrm{top}}(y)$ (B-spline)')
ax_t.plot(y_cps, ttop_cps, 'o', color=c_cap, markersize=7, markeredgecolor='black', label=f'Cap Control Points ($N={num_dvs}$)')
ax_t.plot(y_fine, tweb_fine, '--', color=c_web, linewidth=2.2, label=r'Shear Web Thickness $t_{\mathrm{web}}(y)$ (B-spline)')
ax_t.plot(y_cps, tweb_cps, 's', color=c_web, markersize=6, markeredgecolor='black', label='Web Control Points (at 0.10 mm bound)')
ax_t.scatter(y_elem_mid, ttop_elem, color=c_cap, alpha=0.5, s=25, zorder=4, label=r'Element Midpoint $t_{\mathrm{top}}$')

ax_t.set_xlabel("Spanwise Coordinate $y$ [m]", fontsize=11, fontweight='bold')
ax_t.set_ylabel("Thickness [mm]", fontsize=11, fontweight='bold')
ax_t.set_title("(a) Structural Spar Cap & Shear Web Thickness Distributions", fontsize=11.5, fontweight='bold', pad=8)
ax_t.set_xlim([0.0, half_span * 1.02])
ax_t.set_ylim(bottom=0.0, top=max(np.max(ttop_fine), np.max(ttop_cps)) * 1.18)
ax_t.grid(True, linestyle=':', alpha=0.6)
ax_t.legend(loc='upper right', frameon=True, framealpha=0.92, fontsize=8.5)

# ---------------- Subplot (0, 1): Spanwise Stress Distribution vs Allowable ----------------
ax_s = axes[0, 1]
ax_s.plot(y_fine, stress_field, '-', color=c_stress, linewidth=2.5, label=f'Cubic Stress Spline $S(y)$ (Peak: {np.max(stress_field):.1f} MPa)')
ax_s.plot(y_elem_mid, elem_stress, 'o', color=c_elem, markersize=6.5, markeredgecolor='black', label=f'{len(elem_stress)} FEA Beam Elements Maximum Stress')
ax_s.axhline(allowable_stress, color=c_allow, linestyle='--', linewidth=2.0, label=f'Allowable Stress Limit $\\sigma_{{\\mathrm{{allow}}}} = {allowable_stress:.1f}$ MPa')
ax_s.plot(y_peaks, dv_stress, '^', color='black', markersize=8, label='Station Aggregated Constraint Stresses')

ax_s.fill_between(y_fine, 0, stress_field, color=c_stress, alpha=0.12)
ax_s.fill_between(y_fine, allowable_stress, max(80.0, allowable_stress * 1.25), color=c_allow, alpha=0.08, label=f'Infeasible Region ($>{allowable_stress:.0f}$ MPa)')

ax_s.set_xlabel("Spanwise Coordinate $y$ [m]", fontsize=11, fontweight='bold')
ax_s.set_ylabel(r"Peak Cross-Sectional Stress $\sigma$ [MPa]", fontsize=11, fontweight='bold')
ax_s.set_title(f"(b) Spanwise Stress Distribution ({load_factor_val:.1f}g Structural Sizing Load)", fontsize=11.5, fontweight='bold', pad=8)
ax_s.set_xlim([0.0, half_span * 1.02])
ax_s.set_ylim([-5.0, max(allowable_stress * 1.15, np.max(stress_field) * 1.15)])
ax_s.grid(True, linestyle=':', alpha=0.6)
ax_s.legend(loc='lower left', frameon=True, framealpha=0.92, fontsize=8.5)

# Annotation on active root stress
ax_s.annotate(
    f"Active Root Constraint\n$\\sigma(0) = {stress_field[0]:.2f}$ MPa\n$S'(0) = 0$ (Symmetry)",
    xy=(0.0, stress_field[0]), xytext=(0.18 * half_span, min(65.0, allowable_stress * 0.88)),
    arrowprops=dict(arrowstyle="->", color="black", lw=1.3),
    fontsize=8.5, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#ffffdd", ec="gray", lw=0.8)
)

# ---------------- Subplot (1, 0): Internal Wingbox Dimensions ----------------
ax_b = axes[1, 0]
ax_b.plot(y_elem_mid, widths * 100.0, 'o-', color=c_box, linewidth=2.2, label=r'Box Width $w(y) = 0.40 c(y)$ [cm]')
ax_b.plot(y_elem_mid, heights * 100.0, 's-', color='#8c564b', linewidth=2.2, label=r'Box Height $h(y) = 0.50 t_{\mathrm{local}}(y)$ [cm]')
ax_b.plot(y_elem_mid, chords * 100.0, ':', color='gray', linewidth=1.8, label=r'Full Local Chord $c(y)$ [cm]')

ax_b.set_xlabel("Spanwise Coordinate $y$ [m]", fontsize=11, fontweight='bold')
ax_b.set_ylabel("Cross-Section Dimension [cm]", fontsize=11, fontweight='bold')
ax_b.set_title("(c) Internal Wingbox Spar Dimensions Along the Span", fontsize=11.5, fontweight='bold', pad=8)
ax_b.set_xlim([0.0, half_span * 1.02])
ax_b.set_ylim(bottom=0.0)
ax_b.grid(True, linestyle=':', alpha=0.6)
ax_b.legend(loc='upper right', frameon=True, framealpha=0.92, fontsize=8.5)

# ---------------- Subplot (1, 1): Structural Safety Margin Profile ----------------
ax_m = axes[1, 1]
# Margin of Safety: MS = (sigma_allow / sigma_actual) - 1
stress_pos = np.maximum(stress_field, 0.5)
ms_field = (allowable_stress / stress_pos) - 1.0

ax_m.plot(y_fine, ms_field, '-', color='#2b83ba', linewidth=2.5, label=r'Margin of Safety $\mathrm{MS} = \frac{\sigma_{\mathrm{allow}}}{\sigma} - 1$')
ax_m.axhline(0.0, color=c_allow, linestyle='--', linewidth=1.8, label=r'Critical Yield Boundary ($\mathrm{MS} = 0$)')
ax_m.plot(y_peaks, (allowable_stress / dv_stress) - 1.0, '^', color='black', markersize=8, label='Station MS Points')

ax_m.set_xlabel("Spanwise Coordinate $y$ [m]", fontsize=11, fontweight='bold')
ax_m.set_ylabel(r"Margin of Safety $\mathrm{MS}$", fontsize=11, fontweight='bold')
ax_m.set_title("(d) Structural Margin of Safety Across the Span", fontsize=11.5, fontweight='bold', pad=8)
ax_m.set_xlim([0.0, half_span * 1.02])
ax_m.set_ylim([-0.1, 4.0])
ax_m.grid(True, linestyle=':', alpha=0.6)
ax_m.legend(loc='upper right', frameon=True, framealpha=0.92, fontsize=8.5)

ax_m.annotate(
    "Inboard Sizing Active\n$\\mathrm{MS} \\approx 0.00$ (Fully Stressed Spar)",
    xy=(0.05 * half_span, 0.0), xytext=(0.25 * half_span, 0.8),
    arrowprops=dict(arrowstyle="->", color="black", lw=1.3),
    fontsize=8.5, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#ddffdd", ec="gray", lw=0.8)
)

plt.tight_layout()

# Save figure in output folder and artifact directory
out_fig_name = 'structural_thickness_and_stress_analysis.png'
out_local = os.path.join(output_folder, out_fig_name)
plt.savefig(out_local, dpi=200, bbox_inches='tight')
print(f"\nUpdated structural distribution figure saved to: {out_local}")

if os.path.exists(artifact_dir):
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
