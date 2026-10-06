"""
Script to generate the combined 2-row vertically stacked benchmark comparison figure
for Aerostructural Optimization:
Row 1: Implicit vs. Explicit Formulations (with aframe beam model, t=const)
Row 2: MDF vs. SAND Formulations (with aframe beam model, t=const)

Features:
- Subplot titles placed above each axes (pad=10) to strictly avoid any overlap with legends.
- Box-and-whisker plots with jittered LHS scatter points for solve time.
- Major iteration distribution bar charts with baseline markers labeled 'Baseline Geometry'.
- Clean publication-quality styling (300 DPI PNG & vector PDF).
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))

# File paths
pkl_imp_exp = os.path.join(script_dir, 'aerostructural_benchmark_results_implicit_vs_explicit.pkl')
pkl_mdf_sand = os.path.join(script_dir, 'aerostructural_benchmark_results_mdf_vs_sand.pkl')

assert os.path.exists(pkl_imp_exp), f"Missing {pkl_imp_exp}"
assert os.path.exists(pkl_mdf_sand), f"Missing {pkl_mdf_sand}"

with open(pkl_imp_exp, 'rb') as f:
    data_ie = pickle.load(f)

with open(pkl_mdf_sand, 'rb') as f:
    data_ms = pickle.load(f)

# Extract data - Row 1 (Implicit vs Explicit)
lhs_imp = data_ie['lhs']['implicit']
lhs_exp = data_ie['lhs']['explicit']
base_imp = data_ie['baseline']['implicit']
base_exp = data_ie['baseline']['explicit']

iters_imp = np.array([r['iterations'] for r in lhs_imp])
iters_exp = np.array([r['iterations'] for r in lhs_exp])
time_imp = np.array([r['time'] for r in lhs_imp])
time_exp = np.array([r['time'] for r in lhs_exp])
n_ie = len(lhs_imp)

# Extract data - Row 2 (MDF vs SAND)
lhs_mdf = data_ms['lhs']['mdf']
lhs_sand = data_ms['lhs']['sand']
base_mdf = data_ms['baseline']['mdf']
base_sand = data_ms['baseline']['sand']

iters_mdf = np.array([r['iterations'] for r in lhs_mdf])
iters_sand = np.array([r['iterations'] for r in lhs_sand])
time_mdf = np.array([r['time'] for r in lhs_mdf])
time_sand = np.array([r['time'] for r in lhs_sand])
n_ms = len(lhs_mdf)

# Colors
c_imp = '#1f4e79'   # Deep Blue
c_exp = '#d9534f'   # Coral Red
fc_imp = '#d0e1fd'
fc_exp = '#fcd0d0'

c_mdf = '#1f4e79'   # Deep Blue
c_sand = '#16a085'  # Teal Green
fc_mdf = '#d0e1fd'
fc_sand = '#d1f2eb'

c_base = '#ffd700'  # Gold for baseline

# Create 2x2 stacked figure
fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), dpi=300)
(ax1, ax2), (ax3, ax4) = axes

# =========================================================================
# ROW 1, LEFT: Iterations (Implicit vs Explicit)
# =========================================================================
all_iters_ie = np.arange(min(np.min(iters_imp), np.min(iters_exp)), max(np.max(iters_imp), np.max(iters_exp)) + 2)
counts_imp = [np.sum(iters_imp == k) / n_ie * 100 for k in all_iters_ie]
counts_exp = [np.sum(iters_exp == k) / n_ie * 100 for k in all_iters_ie]

w = 0.38
x_idx_ie = np.arange(len(all_iters_ie))
ax1.bar(x_idx_ie - w/2, counts_imp, width=w, color=c_imp, label='Implicit', alpha=0.9, edgecolor='black', linewidth=0.8)
ax1.bar(x_idx_ie + w/2, counts_exp, width=w, color=c_exp, label='Explicit', alpha=0.9, edgecolor='black', linewidth=0.8)

# Baseline markers
base_it_imp = base_imp['iterations']
base_it_exp = base_exp['iterations']
idx_base_imp = np.where(all_iters_ie == base_it_imp)[0][0]
idx_base_exp = np.where(all_iters_ie == base_it_exp)[0][0]

ax1.scatter(idx_base_imp - w/2, counts_imp[idx_base_imp] + 6, color=c_base, edgecolor='black', s=65, marker='D', zorder=5)
ax1.scatter(idx_base_exp + w/2, counts_exp[idx_base_exp] + 6, color=c_base, edgecolor='black', s=65, marker='D', zorder=5, label='Baseline Geometry')

ax1.set_xticks(x_idx_ie)
ax1.set_xticklabels(all_iters_ie, fontsize=10.0)
ax1.set_xlabel('Major optimization iterations', fontsize=10.5)
ax1.set_ylabel('Percentage of LHS runs [%]', fontsize=10.5)
ax1.set_ylim(0, 115)
ax1.grid(axis='y', linestyle='--', alpha=0.35, color='#cccccc')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_title('(a) Iterations: Implicit vs. Explicit', fontsize=11, fontweight='bold', pad=10)
ax1.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#cccccc', framealpha=0.95, fontsize=9.5)

# =========================================================================
# ROW 1, RIGHT: Solve Time Boxplot (Implicit vs Explicit)
# =========================================================================
np.random.seed(42)
jit_imp = np.random.normal(1, 0.04, size=len(time_imp))
jit_exp = np.random.normal(2, 0.04, size=len(time_exp))
ax2.scatter(jit_imp, time_imp, color=c_imp, alpha=0.35, s=22, edgecolors='none', zorder=2)
ax2.scatter(jit_exp, time_exp, color=c_exp, alpha=0.35, s=22, edgecolors='none', zorder=2)

bp1 = ax2.boxplot(
    [time_imp, time_exp], positions=[1, 2], widths=0.46, patch_artist=True,
    showmeans=False, showfliers=False, zorder=3,
    medianprops=dict(color='black', linewidth=2.0),
    whiskerprops=dict(linewidth=1.2, linestyle='--'),
    capprops=dict(linewidth=1.2)
)
bp1['boxes'][0].set(facecolor=fc_imp, edgecolor=c_imp, linewidth=1.4, alpha=0.85)
bp1['boxes'][1].set(facecolor=fc_exp, edgecolor=c_exp, linewidth=1.4, alpha=0.85)
bp1['whiskers'][0].set(color=c_imp); bp1['whiskers'][1].set(color=c_imp)
bp1['whiskers'][2].set(color=c_exp); bp1['whiskers'][3].set(color=c_exp)
bp1['caps'][0].set(color=c_imp); bp1['caps'][1].set(color=c_imp)
bp1['caps'][2].set(color=c_exp); bp1['caps'][3].set(color=c_exp)

ax2.scatter(1, base_imp['time'], color=c_base, edgecolor='black', s=80, marker='D', zorder=5, label='Baseline Geometry')
ax2.scatter(2, base_exp['time'], color=c_base, edgecolor='black', s=80, marker='D', zorder=5)

ax2.text(1, np.max(time_imp) + 0.45, f"{np.mean(time_imp):.2f} s",
         ha='center', va='bottom', fontsize=8.8, fontweight='bold', color=c_imp)
ax2.text(2, np.max(time_exp) + 0.45, f"{np.mean(time_exp):.2f} s",
         ha='center', va='bottom', fontsize=8.8, fontweight='bold', color=c_exp)

ax2.set_xticks([1, 2])
ax2.set_xticklabels(['Implicit', 'Explicit'], fontsize=10.0)
ax2.set_ylabel('Optimization solve time [s]', fontsize=10.5)
ax2.set_ylim(0, max(np.max(time_exp), np.max(time_imp)) * 1.25)
ax2.grid(axis='y', linestyle='--', alpha=0.35, color='#cccccc')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_title('(b) Solve Time: Implicit vs. Explicit', fontsize=11, fontweight='bold', pad=10)
ax2.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#cccccc', framealpha=0.95, fontsize=9.5)

# =========================================================================
# ROW 2, LEFT: Iterations (MDF vs SAND)
# =========================================================================
all_iters_ms = np.arange(min(np.min(iters_mdf), np.min(iters_sand)), max(np.max(iters_mdf), np.max(iters_sand)) + 2)
counts_mdf = [np.sum(iters_mdf == k) / n_ms * 100 for k in all_iters_ms]
counts_sand = [np.sum(iters_sand == k) / n_ms * 100 for k in all_iters_ms]

x_idx_ms = np.arange(len(all_iters_ms))
ax3.bar(x_idx_ms - w/2, counts_mdf, width=w, color=c_mdf, label='MDF', alpha=0.9, edgecolor='black', linewidth=0.8)
ax3.bar(x_idx_ms + w/2, counts_sand, width=w, color=c_sand, label='SAND', alpha=0.9, edgecolor='black', linewidth=0.8)

# Baseline markers
base_it_mdf = base_mdf['iterations']
base_it_sand = base_sand['iterations']
idx_base_mdf = np.where(all_iters_ms == base_it_mdf)[0][0]
idx_base_sand = np.where(all_iters_ms == base_it_sand)[0][0]

ax3.scatter(idx_base_mdf - w/2, counts_mdf[idx_base_mdf] + 6, color=c_base, edgecolor='black', s=65, marker='D', zorder=5)
ax3.scatter(idx_base_sand + w/2, counts_sand[idx_base_sand] + 6, color=c_base, edgecolor='black', s=65, marker='D', zorder=5, label='Baseline Geometry')

ax3.set_xticks(x_idx_ms)
ax3.set_xticklabels(all_iters_ms, fontsize=10.0)
ax3.set_xlabel('Major optimization iterations', fontsize=10.5)
ax3.set_ylabel('Percentage of LHS runs [%]', fontsize=10.5)
ax3.set_ylim(0, 115)
ax3.grid(axis='y', linestyle='--', alpha=0.35, color='#cccccc')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.set_title('(c) Iterations: MDF vs. SAND', fontsize=11, fontweight='bold', pad=10)
ax3.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#cccccc', framealpha=0.95, fontsize=9.5)

# =========================================================================
# ROW 2, RIGHT: Solve Time Boxplot (MDF vs SAND)
# =========================================================================
jit_mdf = np.random.normal(1, 0.04, size=len(time_mdf))
jit_sand = np.random.normal(2, 0.04, size=len(time_sand))
ax4.scatter(jit_mdf, time_mdf, color=c_mdf, alpha=0.35, s=22, edgecolors='none', zorder=2)
ax4.scatter(jit_sand, time_sand, color=c_sand, alpha=0.35, s=22, edgecolors='none', zorder=2)

bp2 = ax4.boxplot(
    [time_mdf, time_sand], positions=[1, 2], widths=0.46, patch_artist=True,
    showmeans=False, showfliers=False, zorder=3,
    medianprops=dict(color='black', linewidth=2.0),
    whiskerprops=dict(linewidth=1.2, linestyle='--'),
    capprops=dict(linewidth=1.2)
)
bp2['boxes'][0].set(facecolor=fc_mdf, edgecolor=c_mdf, linewidth=1.4, alpha=0.85)
bp2['boxes'][1].set(facecolor=fc_sand, edgecolor=c_sand, linewidth=1.4, alpha=0.85)
bp2['whiskers'][0].set(color=c_mdf); bp2['whiskers'][1].set(color=c_mdf)
bp2['whiskers'][2].set(color=c_sand); bp2['whiskers'][3].set(color=c_sand)
bp2['caps'][0].set(color=c_mdf); bp2['caps'][1].set(color=c_mdf)
bp2['caps'][2].set(color=c_sand); bp2['caps'][3].set(color=c_sand)

ax4.scatter(1, base_mdf['time'], color=c_base, edgecolor='black', s=80, marker='D', zorder=5, label='Baseline Geometry')
ax4.scatter(2, base_sand['time'], color=c_base, edgecolor='black', s=80, marker='D', zorder=5)

ax4.text(1, np.max(time_mdf) + 0.45, f"{np.mean(time_mdf):.2f} s",
         ha='center', va='bottom', fontsize=8.8, fontweight='bold', color=c_mdf)
ax4.text(2, np.max(time_sand) + 0.45, f"{np.mean(time_sand):.2f} s",
         ha='center', va='bottom', fontsize=8.8, fontweight='bold', color=c_sand)

ax4.set_xticks([1, 2])
ax4.set_xticklabels(['MDF', 'SAND'], fontsize=10.0)
ax4.set_ylabel('Optimization solve time [s]', fontsize=10.5)
ax4.set_ylim(0, max(np.max(time_sand), np.max(time_mdf)) * 1.25)
ax4.grid(axis='y', linestyle='--', alpha=0.35, color='#cccccc')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.set_title('(d) Solve Time: MDF vs. SAND', fontsize=11, fontweight='bold', pad=10)
ax4.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#cccccc', framealpha=0.95, fontsize=9.5)

plt.tight_layout(h_pad=2.8, w_pad=2.0)

fig_png = os.path.join(script_dir, 'aerostructural_benchmark_comparison_combined_stacked.png')
fig_pdf = os.path.join(script_dir, 'aerostructural_benchmark_comparison_combined_stacked.pdf')

plt.savefig(fig_png, bbox_inches='tight', dpi=300)
plt.savefig(fig_pdf, bbox_inches='tight')
plt.close()

print(f"Aerostructural stacked benchmark figure saved to:")
print(f"  PNG: {fig_png}")
print(f"  PDF: {fig_pdf}")
