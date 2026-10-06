"""
Script to generate the aerodynamic benchmark comparison figures:
1. Major Iterations: Implicit vs. Explicit (no title)
2. Solve Time: Implicit vs. Explicit (no title)
3. Major Iterations: MDF vs. SAND (no title)
4. Solve Time: MDF vs. SAND (no title)
5. Combined 2x2 stacked figure (no titles)

Designed for publication / LaTeX subfigures.
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Compatibility shim for pickles created across numpy 1.x / 2.x
if 'numpy._core' not in sys.modules:
    try:
        import numpy._core
    except ImportError:
        import numpy.core as _core
        sys.modules['numpy._core'] = _core
        sys.modules['numpy._core.numeric'] = _core.numeric

script_dir = os.path.dirname(os.path.abspath(__file__))

pkl_imp_exp = os.path.join(script_dir, 'benchmark_results_implicit_vs_explicit.pkl')
pkl_mdf_sand = os.path.join(script_dir, 'benchmark_results_mdf_vs_sand.pkl')

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


def draw_panel_iters_ie(ax):
    """Draws major iterations histogram for Implicit vs. Explicit (no title)."""
    all_iters_ie = np.arange(min(np.min(iters_imp), np.min(iters_exp)), max(np.max(iters_imp), np.max(iters_exp)) + 2)
    counts_imp = [np.sum(iters_imp == k) / n_ie * 100 for k in all_iters_ie]
    counts_exp = [np.sum(iters_exp == k) / n_ie * 100 for k in all_iters_ie]

    w = 0.38
    x_idx_ie = np.arange(len(all_iters_ie))
    ax.bar(x_idx_ie - w/2, counts_imp, width=w, color=c_imp, label='Implicit', alpha=0.9, edgecolor='black', linewidth=0.8)
    ax.bar(x_idx_ie + w/2, counts_exp, width=w, color=c_exp, label='Explicit', alpha=0.9, edgecolor='black', linewidth=0.8)

    base_it_imp = base_imp['iterations']
    base_it_exp = base_exp['iterations']
    idx_base_imp = np.where(all_iters_ie == base_it_imp)[0][0]
    idx_base_exp = np.where(all_iters_ie == base_it_exp)[0][0]

    ax.scatter(idx_base_imp - w/2, counts_imp[idx_base_imp] + 6, color=c_base, edgecolor='black', s=65, marker='D', zorder=5)
    ax.scatter(idx_base_exp + w/2, counts_exp[idx_base_exp] + 6, color=c_base, edgecolor='black', s=65, marker='D', zorder=5, label='Baseline Geometry')

    ax.set_xticks(x_idx_ie)
    ax.set_xticklabels(all_iters_ie, fontsize=10.0)
    ax.set_xlabel('Major optimization iterations', fontsize=10.5)
    ax.set_ylabel('Percentage of LHS runs [%]', fontsize=10.5)
    ax.set_ylim(0, 115)
    ax.grid(axis='y', linestyle='--', alpha=0.35, color='#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#cccccc', framealpha=0.95, fontsize=9.5)


def draw_panel_time_ie(ax):
    """Draws solve time boxplot and scatter for Implicit vs. Explicit (no title)."""
    np.random.seed(42)
    jit_imp = np.random.normal(1, 0.04, size=len(time_imp))
    jit_exp = np.random.normal(2, 0.04, size=len(time_exp))
    ax.scatter(jit_imp, time_imp, color=c_imp, alpha=0.35, s=22, edgecolors='none', zorder=2)
    ax.scatter(jit_exp, time_exp, color=c_exp, alpha=0.35, s=22, edgecolors='none', zorder=2)

    bp1 = ax.boxplot(
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

    ax.scatter(1, base_imp['time'], color=c_base, edgecolor='black', s=80, marker='D', zorder=5, label='Baseline Geometry')
    ax.scatter(2, base_exp['time'], color=c_base, edgecolor='black', s=80, marker='D', zorder=5)

    ax.text(1, np.max(time_imp) + 0.45, f"{np.mean(time_imp):.2f} s",
            ha='center', va='bottom', fontsize=8.8, fontweight='bold', color=c_imp)
    ax.text(2, np.max(time_exp) + 0.45, f"{np.mean(time_exp):.2f} s",
            ha='center', va='bottom', fontsize=8.8, fontweight='bold', color=c_exp)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Implicit', 'Explicit'], fontsize=10.0)
    ax.set_ylabel('Optimization solve time [s]', fontsize=10.5)
    ax.set_ylim(0, max(np.max(time_exp), np.max(time_imp)) * 1.25)
    ax.grid(axis='y', linestyle='--', alpha=0.35, color='#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#cccccc', framealpha=0.95, fontsize=9.5)


def draw_panel_iters_ms(ax):
    """Draws major iterations histogram for MDF vs. SAND (no title)."""
    all_iters_ms = np.arange(min(np.min(iters_mdf), np.min(iters_sand)), max(np.max(iters_mdf), np.max(iters_sand)) + 2)
    counts_mdf = [np.sum(iters_mdf == k) / n_ms * 100 for k in all_iters_ms]
    counts_sand = [np.sum(iters_sand == k) / n_ms * 100 for k in all_iters_ms]

    w = 0.38
    x_idx_ms = np.arange(len(all_iters_ms))
    ax.bar(x_idx_ms - w/2, counts_mdf, width=w, color=c_mdf, label='MDF', alpha=0.9, edgecolor='black', linewidth=0.8)
    ax.bar(x_idx_ms + w/2, counts_sand, width=w, color=c_sand, label='SAND', alpha=0.9, edgecolor='black', linewidth=0.8)

    base_it_mdf = base_mdf['iterations']
    base_it_sand = base_sand['iterations']
    idx_base_mdf = np.where(all_iters_ms == base_it_mdf)[0][0]
    idx_base_sand = np.where(all_iters_ms == base_it_sand)[0][0]

    ax.scatter(idx_base_mdf - w/2, counts_mdf[idx_base_mdf] + 6, color=c_base, edgecolor='black', s=65, marker='D', zorder=5)
    ax.scatter(idx_base_sand + w/2, counts_sand[idx_base_sand] + 6, color=c_base, edgecolor='black', s=65, marker='D', zorder=5, label='Baseline Geometry')

    ax.set_xticks(x_idx_ms)
    ax.set_xticklabels(all_iters_ms, fontsize=10.0)
    ax.set_xlabel('Major optimization iterations', fontsize=10.5)
    ax.set_ylabel('Percentage of LHS runs [%]', fontsize=10.5)
    ax.set_ylim(0, 115)
    ax.grid(axis='y', linestyle='--', alpha=0.35, color='#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#cccccc', framealpha=0.95, fontsize=9.5)


def draw_panel_time_ms(ax):
    """Draws solve time boxplot and scatter for MDF vs. SAND (no title)."""
    np.random.seed(42)
    jit_mdf = np.random.normal(1, 0.04, size=len(time_mdf))
    jit_sand = np.random.normal(2, 0.04, size=len(time_sand))
    ax.scatter(jit_mdf, time_mdf, color=c_mdf, alpha=0.35, s=22, edgecolors='none', zorder=2)
    ax.scatter(jit_sand, time_sand, color=c_sand, alpha=0.35, s=22, edgecolors='none', zorder=2)

    bp2 = ax.boxplot(
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

    ax.scatter(1, base_mdf['time'], color=c_base, edgecolor='black', s=80, marker='D', zorder=5, label='Baseline Geometry')
    ax.scatter(2, base_sand['time'], color=c_base, edgecolor='black', s=80, marker='D', zorder=5)

    ax.text(1, np.max(time_mdf) + 0.45, f"{np.mean(time_mdf):.2f} s",
            ha='center', va='bottom', fontsize=8.8, fontweight='bold', color=c_mdf)
    ax.text(2, np.max(time_sand) + 0.45, f"{np.mean(time_sand):.2f} s",
            ha='center', va='bottom', fontsize=8.8, fontweight='bold', color=c_sand)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['MDF', 'SAND'], fontsize=10.0)
    ax.set_ylabel('Optimization solve time [s]', fontsize=10.5)
    ax.set_ylim(0, max(np.max(time_sand), np.max(time_mdf)) * 1.25)
    ax.grid(axis='y', linestyle='--', alpha=0.35, color='#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#cccccc', framealpha=0.95, fontsize=9.5)


def generate_all_figures():
    # -------------------------------------------------------------------------
    # 1. Standalone Plot 1: Iterations (Implicit vs Explicit)
    # -------------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(5.5, 4.0), dpi=300)
    draw_panel_iters_ie(ax1)
    plt.tight_layout()
    f1_png = os.path.join(script_dir, 'benchmark_aero_iterations_implicit_vs_explicit.png')
    f1_pdf = os.path.join(script_dir, 'benchmark_aero_iterations_implicit_vs_explicit.pdf')
    plt.savefig(f1_png, bbox_inches='tight', dpi=300)
    plt.savefig(f1_pdf, bbox_inches='tight')
    plt.close(fig1)

    # -------------------------------------------------------------------------
    # 2. Standalone Plot 2: Solve Time (Implicit vs Explicit)
    # -------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(5.5, 4.0), dpi=300)
    draw_panel_time_ie(ax2)
    plt.tight_layout()
    f2_png = os.path.join(script_dir, 'benchmark_aero_time_implicit_vs_explicit.png')
    f2_pdf = os.path.join(script_dir, 'benchmark_aero_time_implicit_vs_explicit.pdf')
    plt.savefig(f2_png, bbox_inches='tight', dpi=300)
    plt.savefig(f2_pdf, bbox_inches='tight')
    plt.close(fig2)

    # -------------------------------------------------------------------------
    # 3. Standalone Plot 3: Iterations (MDF vs SAND)
    # -------------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(5.5, 4.0), dpi=300)
    draw_panel_iters_ms(ax3)
    plt.tight_layout()
    f3_png = os.path.join(script_dir, 'benchmark_aero_iterations_mdf_vs_sand.png')
    f3_pdf = os.path.join(script_dir, 'benchmark_aero_iterations_mdf_vs_sand.pdf')
    plt.savefig(f3_png, bbox_inches='tight', dpi=300)
    plt.savefig(f3_pdf, bbox_inches='tight')
    plt.close(fig3)

    # -------------------------------------------------------------------------
    # 4. Standalone Plot 4: Solve Time (MDF vs SAND)
    # -------------------------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(5.5, 4.0), dpi=300)
    draw_panel_time_ms(ax4)
    plt.tight_layout()
    f4_png = os.path.join(script_dir, 'benchmark_aero_time_mdf_vs_sand.png')
    f4_pdf = os.path.join(script_dir, 'benchmark_aero_time_mdf_vs_sand.pdf')
    plt.savefig(f4_png, bbox_inches='tight', dpi=300)
    plt.savefig(f4_pdf, bbox_inches='tight')
    plt.close(fig4)

    # -------------------------------------------------------------------------
    # 5. Combined 2x2 Stacked Figure (titles removed)
    # -------------------------------------------------------------------------
    fig_all, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(11.0, 8.0), dpi=300)
    draw_panel_iters_ie(a1)
    draw_panel_time_ie(a2)
    draw_panel_iters_ms(a3)
    draw_panel_time_ms(a4)
    plt.tight_layout(h_pad=2.8, w_pad=2.0)
    
    fig_png = os.path.join(script_dir, 'benchmark_comparison_combined_stacked.png')
    fig_pdf = os.path.join(script_dir, 'benchmark_comparison_combined_stacked.pdf')
    plt.savefig(fig_png, bbox_inches='tight', dpi=300)
    plt.savefig(fig_pdf, bbox_inches='tight')
    plt.close(fig_all)

    print("Aerodynamic benchmark figures successfully generated (all titles removed):")
    print(f"  1. Iterations (Imp vs Exp): {f1_png}")
    print(f"                              {f1_pdf}")
    print(f"  2. Solve Time (Imp vs Exp): {f2_png}")
    print(f"                              {f2_pdf}")
    print(f"  3. Iterations (MDF vs SAND):{f3_png}")
    print(f"                              {f3_pdf}")
    print(f"  4. Solve Time (MDF vs SAND):{f4_png}")
    print(f"                              {f4_pdf}")
    print(f"  5. Combined 2x2 Stacked:    {fig_png}")
    print(f"                              {fig_pdf}")


if __name__ == '__main__':
    generate_all_figures()
