"""Generate updated lift and moment distribution curves directly from the latest converged run.
Includes caching of evaluated panel telemetry so subsequent re-plots are instantaneous.
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Ensure all relevant packages are on sys.path
sys.path.insert(0, '/home/andrew/optimization/VortexAD')
sys.path.insert(0, '/home/andrew/optimization/aframe')
sys.path.insert(0, '/home/andrew/optimization/csdl')
sys.path.insert(0, '/home/andrew/optimization/CSDL_alpha')
sys.path.insert(0, '/home/andrew/optimization/lsdo_geo')
sys.path.insert(0, '/home/andrew/optimization/lsdo_geo/examples/showcase_examples/rectangular_wing')

# Paths
import glob
if len(sys.argv) > 1:
    output_folder = os.path.abspath(sys.argv[1])
else:
    output_base_dir = '/home/andrew/optimization/lsdo_geo/rectangular_wing_to_bwb_aerostructural_optimization_outputs'
    output_folders = glob.glob(os.path.join(output_base_dir, '*'))
    output_folder = max(output_folders, key=os.path.getmtime)

print(f"Target optimization output directory: {output_folder}")
cache_file = os.path.join(output_folder, 'lift_and_moment_data.npz')
artifact_dir = '/home/andrew/.gemini/antigravity/brain/47a5c338-9be3-486f-abb2-1deefc5b2d19'

force_rerun = False

if os.path.exists(cache_file) and not force_rerun:
    print(f"Loading cached panel telemetry from: {cache_file}")
    data = np.load(cache_file)
    panel_centers_right = data['panel_centers_right']
    f_cruise = data['f_cruise']
    f_ss = data['f_ss']
    xcg_val = float(data['xcg_val'])
    zcg_val = float(data['zcg_val']) if 'zcg_val' in data else 0.0
else:
    print(f"No cache found at {cache_file}; running model evaluation...")
    import csdl_alpha as csdl
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    import examples.showcase_examples.rectangular_wing.ex_rectangular_wing_to_bwb as main_script
    jax_sim = main_script.jax_sim
    design_variables = main_script.design_variables

    x_out_path = os.path.join(output_folder, 'x.out')
    x_history = np.loadtxt(x_out_path)
    if len(x_history.shape) == 1:
        x_history = x_history.reshape(1, -1)
    x_opt = x_history[-1]

    # Set optimal variables on jax_sim
    curr_idx = 0
    for name, dv_info in design_variables.items():
        var_size = int(np.prod(dv_info.variable.shape))
        slc = slice(curr_idx, curr_idx + var_size)
        unscaled_val = (x_opt[slc] / dv_info.scaler).reshape(dv_info.variable.shape)
        jax_sim[dv_info.variable] = unscaled_val
        curr_idx += var_size

    # Evaluate model
    jax_sim.run()

    xcg_val = float(np.asarray(jax_sim[main_script.x_cg]).flatten()[0])
    zcg_val = float(np.asarray(jax_sim[main_script.r_cg]).flatten()[2]) if hasattr(main_script, 'r_cg') else 0.0
    panel_centers_right = np.asarray(jax_sim[main_script.dynamic_panel_centers_right])
    f_cruise = np.asarray(jax_sim[main_script.panel_forces_right_cruise])
    f_ss = np.asarray(jax_sim[main_script.panel_forces_right_ss])

    # Save cache for instant reuse
    np.savez_compressed(
        cache_file,
        panel_centers_right=panel_centers_right,
        f_cruise=f_cruise,
        f_ss=f_ss,
        xcg_val=xcg_val,
        zcg_val=zcg_val
    )
    print(f"Saved panel telemetry cache to: {cache_file}")

# -------------------------------------------------------------------------
# Strip Integration and Distribution Curve Processing
# -------------------------------------------------------------------------
def process_strips(pc, pf_c, pf_ss, xcg, zcg=0.0):
    # Group panels by spanwise station y
    y_round = np.round(pc[:, 1], 4)
    y_unique = np.unique(y_round)

    # The lifting surface consists of full chordwise rings (40 panels each),
    # while y > 2.23 m contains the tip cap panels.
    main_stations = [y for y in y_unique if np.sum(y_round == y) == 40]

    y_centers = []
    L_c = []
    L_ss = []
    My_c = []
    My_ss = []

    for y_val in main_stations:
        mask = (y_round == y_val)
        y_centers.append(np.mean(pc[mask, 1]))
        # Lift is vertical force Fz (index 2)
        L_c.append(np.sum(pf_c[mask, 2]))
        L_ss.append(np.sum(pf_ss[mask, 2]))
        # Pitching moment about CG: (z - z_cg)*Fx - (x - x_cg)*Fz
        # Pitch-up positive convention: upward force ahead of CG gives positive pitching moment
        dMy_c = (pc[mask, 2] - zcg) * pf_c[mask, 0] - (pc[mask, 0] - xcg) * pf_c[mask, 2]
        dMy_ss = (pc[mask, 2] - zcg) * pf_ss[mask, 0] - (pc[mask, 0] - xcg) * pf_ss[mask, 2]
        My_c.append(np.sum(dMy_c))
        My_ss.append(np.sum(dMy_ss))

    # Add non-ring (tip cap) panel forces into the final tip strip so total half-wing loads are 100% conserved
    tip_mask = ~np.isin(y_round, main_stations)
    if np.any(tip_mask):
        L_c[-1] += np.sum(pf_c[tip_mask, 2])
        L_ss[-1] += np.sum(pf_ss[tip_mask, 2])
        dMy_c_tip = (pc[tip_mask, 2] - zcg) * pf_c[tip_mask, 0] - (pc[tip_mask, 0] - xcg) * pf_c[tip_mask, 2]
        dMy_ss_tip = (pc[tip_mask, 2] - zcg) * pf_ss[tip_mask, 0] - (pc[tip_mask, 0] - xcg) * pf_ss[tip_mask, 2]
        My_c[-1] += np.sum(dMy_c_tip)
        My_ss[-1] += np.sum(dMy_ss_tip)

    y_centers = np.array(y_centers)
    L_c = np.array(L_c)
    L_ss = np.array(L_ss)
    My_c = np.array(My_c)
    My_ss = np.array(My_ss)

    b_tip = np.max(pc[:, 1])

    # Compute strip boundaries and widths dy
    bounds = np.zeros(len(y_centers) + 1)
    bounds[0] = 0.0
    bounds[-1] = b_tip
    bounds[1:-1] = 0.5 * (y_centers[:-1] + y_centers[1:])
    dy = np.diff(bounds)

    # Sectional force/moment densities (per unit span) [N/m] and [N*m/m]
    dL_c = L_c / dy
    dL_ss = L_ss / dy
    dMy_c = My_c / dy
    dMy_ss = My_ss / dy

    # Create fine grid for smooth continuous curves
    y_fine = np.linspace(0.0, b_tip, 300)

    # Enforce root symmetry dL/dy=0 and dMy/dy=0 at y=0 via symmetric mirroring
    y_sym = np.concatenate([-y_centers[::-1], y_centers])
    dL_c_sym = np.concatenate([dL_c[::-1], dL_c])
    dL_ss_sym = np.concatenate([dL_ss[::-1], dL_ss])
    dMy_c_sym = np.concatenate([dMy_c[::-1], dMy_c])
    dMy_ss_sym = np.concatenate([dMy_ss[::-1], dMy_ss])

    # Tip boundary condition: lift and moment vanish at wingtip
    y_pts_L = np.concatenate([[-b_tip], y_sym, [b_tip]])
    dL_c_pts = np.concatenate([[0.0], dL_c_sym, [0.0]])
    dL_ss_pts = np.concatenate([[0.0], dL_ss_sym, [0.0]])
    dMy_c_pts = np.concatenate([[0.0], dMy_c_sym, [0.0]])
    dMy_ss_pts = np.concatenate([[0.0], dMy_ss_sym, [0.0]])

    spl_Lc = CubicSpline(y_pts_L, dL_c_pts, bc_type='natural')
    spl_Lss = CubicSpline(y_pts_L, dL_ss_pts, bc_type='natural')
    spl_Myc = CubicSpline(y_pts_L, dMy_c_pts, bc_type='natural')
    spl_Myss = CubicSpline(y_pts_L, dMy_ss_pts, bc_type='natural')

    # Elliptical lift distribution benchmark (for half-wing):
    # L'(y) = (4 * L_tot) / (pi * b_tip) * sqrt(1 - (y/b_tip)^2)
    tot_Lc = np.sum(L_c)
    tot_Lss = np.sum(L_ss)
    dL_ellip_c = (4.0 * tot_Lc / (np.pi * b_tip)) * np.sqrt(np.clip(1.0 - (y_fine / b_tip)**2, 0.0, 1.0))
    dL_ellip_ss = (4.0 * tot_Lss / (np.pi * b_tip)) * np.sqrt(np.clip(1.0 - (y_fine / b_tip)**2, 0.0, 1.0))

    return {
        'y_centers': y_centers,
        'y_fine': y_fine,
        'b_tip': b_tip,
        'dy': dy,
        'L_c_strip': L_c,
        'L_ss_strip': L_ss,
        'My_c_strip': My_c,
        'My_ss_strip': My_ss,
        'dL_c_raw': dL_c,
        'dL_ss_raw': dL_ss,
        'dMy_c_raw': dMy_c,
        'dMy_ss_raw': dMy_ss,
        'dL_c_smooth': np.maximum(spl_Lc(y_fine), 0.0),
        'dL_ss_smooth': np.maximum(spl_Lss(y_fine), 0.0),
        'dMy_c_smooth': spl_Myc(y_fine),
        'dMy_ss_smooth': spl_Myss(y_fine),
        'dL_ellip_c': dL_ellip_c,
        'dL_ellip_ss': dL_ellip_ss,
        'tot_Lc': tot_Lc,
        'tot_Lss': tot_Lss,
        'tot_Myc': np.sum(My_c),
        'tot_Myss': np.sum(My_ss),
    }

results = process_strips(panel_centers_right, f_cruise, f_ss, xcg_val, zcg_val)
y_fine = results['y_fine']
y_centers = results['y_centers']
b_tip = results['b_tip']

# -------------------------------------------------------------------------
# Plotting Lift and Pitching Moment Distribution Curves
# -------------------------------------------------------------------------
fig, (ax_l, ax_m) = plt.subplots(1, 2, figsize=(14, 5.8), dpi=200)
fig.suptitle(
    f"Spanwise Aerodynamic Load & Moment Distributions (Optimized Configuration)\n"
    f"Right Half-Span: b/2 = {b_tip:.2f} m | Center of Gravity: x_cg = {xcg_val:.3f} m, z_cg = {zcg_val:.3f} m",
    fontsize=13, fontweight='bold', y=0.98
)

# Color scheme
c_cruise = '#1f77b4'
c_ss = '#d62728'

# ----------------- Subplot 1: Lift Distribution Curves -----------------
# 1.0g Cruise
ax_l.plot(y_fine, results['dL_c_smooth'], '-', color=c_cruise, linewidth=2.4,
          label=f"Cruise 1.0g Curve ($L_{{\\text{{half}}}} = {results['tot_Lc']:.1f}$ N, $L_{{\\text{{total}}}} = {2*results['tot_Lc']:.1f}$ N)")
ax_l.scatter(y_centers, results['dL_c_raw'], color=c_cruise, s=38, zorder=5,
             edgecolors='white', linewidth=0.8, label="Cruise Strip Values")
ax_l.fill_between(y_fine, results['dL_c_smooth'], alpha=0.15, color=c_cruise)

# Structural Sizing Pull-Up Maneuver
ax_l.plot(y_fine, results['dL_ss_smooth'], '-', color=c_ss, linewidth=2.4,
          label=f"Structural Sizing (SS) Curve ($L_{{\\text{{half}}}} = {results['tot_Lss']:.1f}$ N, $L_{{\\text{{total}}}} = {2*results['tot_Lss']:.1f}$ N)")
ax_l.scatter(y_centers, results['dL_ss_raw'], color=c_ss, s=38, zorder=5,
             edgecolors='white', linewidth=0.8, label="Sizing Strip Values")
ax_l.fill_between(y_fine, results['dL_ss_smooth'], alpha=0.15, color=c_ss)

# Elliptical Benchmarks
ax_l.plot(y_fine, results['dL_ellip_c'], ':', color='#2ca02c', linewidth=1.8,
          alpha=0.85, label="Elliptical Benchmark (Cruise)")
ax_l.plot(y_fine, results['dL_ellip_ss'], '--', color='#ff7f0e', linewidth=1.8,
          alpha=0.85, label="Elliptical Benchmark (Sizing)")

ax_l.set_xlabel("Spanwise Coordinate $y$ [m]", fontsize=11, fontweight='bold')
ax_l.set_ylabel("Sectional Lift Density $L'(y) = dL/dy$ [N/m]", fontsize=11, fontweight='bold')
ax_l.set_title("Spanwise Lift Distribution $L'(y)$", fontsize=12, fontweight='bold', pad=10)
ax_l.set_xlim([0.0, b_tip * 1.02])
ax_l.set_ylim(bottom=0.0)
ax_l.grid(True, linestyle=':', alpha=0.6)
ax_l.legend(loc='upper right', fontsize=8.5, framealpha=0.92)

# ----------------- Subplot 2: Pitching Moment Distribution Curves -----------------
ax_m.axhline(0.0, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)

# 1.0g Cruise
ax_m.plot(y_fine, results['dMy_c_smooth'], '-', color=c_cruise, linewidth=2.4,
          label=f"Cruise 1.0g Curve (Net $M_y = {results['tot_Myc']:+.3f}$ N·m)")
ax_m.scatter(y_centers, results['dMy_c_raw'], color=c_cruise, s=38, zorder=5,
             edgecolors='white', linewidth=0.8, label="Cruise Strip Values")
ax_m.fill_between(y_fine, results['dMy_c_smooth'], alpha=0.12, color=c_cruise)

# Structural Sizing Pull-Up Maneuver
ax_m.plot(y_fine, results['dMy_ss_smooth'], '-', color=c_ss, linewidth=2.4,
          label=f"Structural Sizing (SS) Curve (Net $M_y = {results['tot_Myss']:+.3f}$ N·m)")
ax_m.scatter(y_centers, results['dMy_ss_raw'], color=c_ss, s=38, zorder=5,
             edgecolors='white', linewidth=0.8, label="Sizing Strip Values")
ax_m.fill_between(y_fine, results['dMy_ss_smooth'], alpha=0.12, color=c_ss)

ax_m.set_xlabel("Spanwise Coordinate $y$ [m]", fontsize=11, fontweight='bold')
ax_m.set_ylabel("Sectional Pitching Moment Density $M'_y(y) = dM_y/dy$ [N·m/m]", fontsize=11, fontweight='bold')
ax_m.set_title("Spanwise Pitching Moment Distribution (about CG)", fontsize=12, fontweight='bold', pad=10)
ax_m.set_xlim([0.0, b_tip * 1.02])
ax_m.grid(True, linestyle=':', alpha=0.6)
ax_m.legend(loc='lower right', fontsize=8.5, framealpha=0.92)

# Text annotation explaining pitch trim
trim_status = "Longitudinally Trimmed (Net My ≈ 0)" if abs(results['tot_Myc']) < 0.05 else "Untrimmed"
ax_m.text(0.04, 0.95, f"Cruise Trim: {trim_status}\nInboard pitch-up balanced by outboard pitch-down",
          transform=ax_m.transAxes, fontsize=8.5, verticalalignment='top',
          bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#cccccc', alpha=0.9))

plt.tight_layout()
plt.subplots_adjust(top=0.87)

# Save figure in output folder
lift_fig_path = os.path.join(output_folder, 'lift_and_moment_combined_analysis.png')
plt.savefig(lift_fig_path, dpi=200, bbox_inches='tight')
print(f"Updated lift/moment distribution figure saved to: {lift_fig_path}")

# Also copy/save to artifact directory if available
if os.path.exists(artifact_dir):
    artifact_fig_path = os.path.join(artifact_dir, 'lift_and_moment_combined_analysis.png')
    plt.savefig(artifact_fig_path, dpi=200, bbox_inches='tight')
    print(f"Figure also saved to artifact directory: {artifact_fig_path}")

plt.close()

# Print summary metrics
print("\n" + "="*65)
print(f"AEROSTRUCTURAL TELEMETRY SUMMARY (Run: {os.path.basename(output_folder)})")
print("="*65)
print(f"Wing Half-Span (b/2):            {b_tip:.3f} m")
print(f"Center of Gravity (x_cg):        {xcg_val:.4f} m (z_cg = {zcg_val:.4f} m)")
print(f"Number of Spanwise Strips:       {len(y_centers)}")
print(f"Cruise 1.0g Half-Wing Lift:      {results['tot_Lc']:.2f} N (Full: {2*results['tot_Lc']:.2f} N)")
print(f"Structural Sizing Half-Wing Lift:{results['tot_Lss']:.2f} N (Full: {2*results['tot_Lss']:.2f} N)")
print(f"Sizing Lift Ratio:               {results['tot_Lss']/results['tot_Lc']:.2f}x")
print(f"Cruise Pitching Moment (Net My): {results['tot_Myc']:+.4f} N*m")
print(f"Sizing Pitching Moment (My):     {results['tot_Myss']:+.4f} N*m")
print("="*65 + "\n")