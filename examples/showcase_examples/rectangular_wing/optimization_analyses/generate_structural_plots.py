"""Generate structural analysis plots:
1. Spar cap (ttop) and shear web (tweb) thickness distributions across the span.
2. Authentic beam element stresses and continuous B-spline stress distribution vs allowable stress limit (69 MPa).
3. Additional cross-sectional geometry (wingbox width, height, and area) along the span.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import lsdo_function_spaces as lfs

# Target output directories
telemetry_path = '/home/andrew/optimization/lsdo_geo/scratch/fast_4g_sf4_ar_area_telemetry.npz'
artifact_dir = '/home/andrew/.gemini/antigravity/brain/47a5c338-9be3-486f-abb2-1deefc5b2d19'

data = np.load(telemetry_path)

# Structural parameters
beam_pts = data['beam_pts_opt']
half_span = float(beam_pts[-1, 1])
scale_factor = float(data['scale_factor'])
allowable_stress = float(data['allowable_stress']) / 1e6  # 69.0 MPa

# Element geometry & locations
y_elem_mid = 0.5 * (beam_pts[:-1, 1] + beam_pts[1:, 1])
chords = data['chords_opt']
heights = data['heights_opt']
widths = data['widths_opt']
ttop_elem = data['ttop_elem_opt'] * 1e3  # mm
tweb_elem = data['tweb_elem_opt'] * 1e3  # mm
elem_stress = data['elem_stress_ss'] / 1e6  # MPa
dv_stress = data['dv_stress_ss'] / 1e6  # MPa
thickness_peaks = data['thickness_peaks']
y_peaks = thickness_peaks * half_span

# -------------------------------------------------------------------------
# Continuous B-Spline Field Evaluations
# -------------------------------------------------------------------------
y_fine = np.linspace(0.0, half_span, 400)
u_fine = (y_fine / half_span).reshape(-1, 1)

# 1. Thickness Distributions (Degree 2 B-Spline, 5 CPs)
num_dvs = len(data['ttop_dvs_opt'])
sp_thick = lfs.BSplineSpace(num_parametric_dimensions=1, degree=2, coefficients_shape=(num_dvs,))
B_thick = sp_thick.compute_basis_matrix(u_fine).toarray()

ttop_fine = (B_thick @ data['ttop_dvs_opt']) * 1e3  # mm
tweb_fine = (B_thick @ data['tweb_dvs_opt']) * 1e3  # mm
ttop_cps = data['ttop_dvs_opt'] * 1e3
tweb_cps = data['tweb_dvs_opt'] * 1e3
y_cps = thickness_peaks * half_span

# 2. Stress Field (Degree 3 Cubic B-Spline, 15 CPs, S'(0)=0)
knots_stress_15 = data['knots_stress_15']
sp_stress = lfs.BSplineSpace(
    num_parametric_dimensions=1,
    degree=3,
    coefficients_shape=(15,),
    knots=(knots_stress_15,)
)
B_stress = sp_stress.compute_basis_matrix(u_fine).toarray()
stress_coeffs_arr = data['stress_coeffs_ss']
stress_field = (B_stress @ stress_coeffs_arr) / 1e6  # MPa

# -------------------------------------------------------------------------
# Figure 1: Comprehensive Structural Analysis (2 x 2 Subplots)
# -------------------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(2, 2, figsize=(14, 10.5), dpi=200)
fig.suptitle(
    f"Structural Optimization Analysis: Thickness & Stress Distributions\n"
    f"Structural Sizing Maneuver | Allowable Stress = {allowable_stress:.1f} MPa | Half-Span $b/2 = {half_span:.2f}$ m",
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
ax_t.plot(y_fine, ttop_fine, '-', color=c_cap, linewidth=2.5, label='Spar Cap Thickness $t_{\\text{top}}(y)$ (B-spline)')
ax_t.plot(y_cps, ttop_cps, 'o', color=c_cap, markersize=7, markeredgecolor='black', label=f'Cap Control Points ($N={num_dvs}$)')
ax_t.plot(y_fine, tweb_fine, '--', color=c_web, linewidth=2.2, label='Shear Web Thickness $t_{\\text{web}}(y)$ (B-spline)')
ax_t.plot(y_cps, tweb_cps, 's', color=c_web, markersize=6, markeredgecolor='black', label='Web Control Points (at 0.10 mm bound)')
ax_t.scatter(y_elem_mid, ttop_elem, color=c_cap, alpha=0.5, s=25, zorder=4, label='Element Midpoint $t_{\\text{top}}$')

ax_t.set_xlabel("Spanwise Coordinate $y$ [m]", fontsize=11, fontweight='bold')
ax_t.set_ylabel("Thickness [mm]", fontsize=11, fontweight='bold')
ax_t.set_title("(a) Structural Spar Cap & Shear Web Thickness Distributions", fontsize=11.5, fontweight='bold', pad=8)
ax_t.set_xlim([0.0, half_span * 1.02])
ax_t.set_ylim(bottom=0.0, top=max(np.max(ttop_fine), np.max(ttop_cps)) * 1.18)
ax_t.grid(True, linestyle=':', alpha=0.6)
ax_t.legend(loc='upper right', frameon=True, framealpha=0.92, fontsize=8.5)

# ---------------- Subplot (0, 1): Spanwise Stress Distribution vs Allowable ----------------
ax_s = axes[0, 1]
ax_s.plot(y_fine, stress_field, '-', color=c_stress, linewidth=2.5, label=f'15-CP Cubic Stress Spline $S(y)$ (Peak: {np.max(stress_field):.1f} MPa)')
ax_s.plot(y_elem_mid, elem_stress, 'o', color=c_elem, markersize=6.5, markeredgecolor='black', label='14 FEA Beam Elements Maximum Stress')
ax_s.axhline(allowable_stress, color=c_allow, linestyle='--', linewidth=2.0, label=f'Allowable Stress Limit $\\sigma_{{\\text{{allow}}}} = {allowable_stress:.1f}$ MPa')
ax_s.plot(y_peaks, dv_stress, '^', color='black', markersize=8, label='Station Aggregated Constraint Stresses')

ax_s.fill_between(y_fine, 0, stress_field, color=c_stress, alpha=0.12)
ax_s.fill_between(y_fine, allowable_stress, 80, color=c_allow, alpha=0.08, label='Infeasible Region ($>69$ MPa)')

ax_s.set_xlabel("Spanwise Coordinate $y$ [m]", fontsize=11, fontweight='bold')
ax_s.set_ylabel("Peak Cross-Sectional Stress $\\sigma$ [MPa]", fontsize=11, fontweight='bold')
ax_s.set_title("(b) Spanwise Stress Distribution (Structural Sizing Maneuver Load)", fontsize=11.5, fontweight='bold', pad=8)
ax_s.set_xlim([0.0, half_span * 1.02])
ax_s.set_ylim([-5.0, 78.0])
ax_s.grid(True, linestyle=':', alpha=0.6)
ax_s.legend(loc='lower left', frameon=True, framealpha=0.92, fontsize=8.5)

# Annotation on active root stress
ax_s.annotate(
    f"Active Root Constraint\n$\\sigma(0) = {stress_field[0]:.2f}$ MPa\n$S'(0) = 0$ (Symmetry)",
    xy=(0.0, stress_field[0]), xytext=(0.28, 60.0),
    arrowprops=dict(arrowstyle="->", color="black", lw=1.3),
    fontsize=8.5, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#ffffdd", ec="gray", lw=0.8)
)

# ---------------- Subplot (1, 0): Internal Wingbox Dimensions ----------------
ax_b = axes[1, 0]
ax_b.plot(y_elem_mid, widths * 100.0, 'o-', color=c_box, linewidth=2.2, label='Box Width $w(y) = 0.40 c(y)$ [cm]')
ax_b.plot(y_elem_mid, heights * 100.0, 's-', color='#8c564b', linewidth=2.2, label='Box Height $h(y) = 0.50 t_{\\text{local}}(y)$ [cm]')
ax_b.plot(y_elem_mid, chords * 100.0, ':', color='gray', linewidth=1.8, label='Full Local Chord $c(y)$ [cm]')

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
# Clip actual stress to positive for MS calculation
stress_pos = np.maximum(stress_field, 0.5)
ms_field = (allowable_stress / stress_pos) - 1.0

ax_m.plot(y_fine, ms_field, '-', color='#2b83ba', linewidth=2.5, label='Margin of Safety $MS = \\frac{\\sigma_{\\text{allow}}}{\\sigma} - 1$')
ax_m.axhline(0.0, color=c_allow, linestyle='--', linewidth=1.8, label='Critical Yield Boundary ($MS = 0$)')
ax_m.plot(y_peaks, (allowable_stress / dv_stress) - 1.0, '^', color='black', markersize=8, label='Station MS Points')

ax_m.set_xlabel("Spanwise Coordinate $y$ [m]", fontsize=11, fontweight='bold')
ax_m.set_ylabel("Margin of Safety $MS$", fontsize=11, fontweight='bold')
ax_m.set_title("(d) Structural Margin of Safety Across the Span", fontsize=11.5, fontweight='bold', pad=8)
ax_m.set_xlim([0.0, half_span * 1.02])
ax_m.set_ylim([-0.1, 4.0])
ax_m.grid(True, linestyle=':', alpha=0.6)
ax_m.legend(loc='upper right', frameon=True, framealpha=0.92, fontsize=8.5)

ax_m.annotate(
    f"Inboard Sizing Active\n$MS \\approx 0.00$ (Fully Stressed Spar)",
    xy=(0.15, 0.0), xytext=(0.40, 0.8),
    arrowprops=dict(arrowstyle="->", color="black", lw=1.3),
    fontsize=8.5, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#ddffdd", ec="gray", lw=0.8)
)

plt.tight_layout()

# Save figures to current folder and artifact directory
out_local = 'structural_thickness_and_stress_analysis.png'
out_artifact = os.path.join(artifact_dir, 'structural_thickness_and_stress_analysis.png')

plt.savefig(out_local, dpi=200, bbox_inches='tight')
plt.savefig(out_artifact, dpi=200, bbox_inches='tight')
plt.close()

print(f"Successfully generated structural analysis plot:")
print(f"  Local:    {os.path.abspath(out_local)}")
print(f"  Artifact: {out_artifact}")

