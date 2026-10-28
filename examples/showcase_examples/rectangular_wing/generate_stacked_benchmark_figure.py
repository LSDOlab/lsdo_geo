"""
Script to generate the aerodynamic benchmark comparison figures:
1. Major Iterations: Implicit (RS) vs. Explicit vs. Implicit (FS) (no title)
2. Solve Time: Implicit (RS) vs. Explicit vs. Implicit (FS) (no title)
3. Combined 1x2 compact comparison figure (no titles)

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

# Extract data: Implicit (RS) & Explicit from IE benchmark, FS from MS benchmark
lhs_imp = data_ie['lhs']['implicit']
lhs_exp = data_ie['lhs']['explicit']
lhs_sand = data_ms['lhs']['sand']

base_imp = data_ie['baseline']['implicit']
base_exp = data_ie['baseline']['explicit']
base_sand = data_ms['baseline']['sand']

iters_imp = np.array([r['iterations'] for r in lhs_imp])
iters_exp = np.array([r['iterations'] for r in lhs_exp])
iters_sand = np.array([r['iterations'] for r in lhs_sand])

time_imp = np.array([r['time'] for r in lhs_imp])
time_exp = np.array([r['time'] for r in lhs_exp])
time_sand = np.array([r['time'] for r in lhs_sand])

n_imp = len(lhs_imp)
n_exp = len(lhs_exp)
n_sand = len(lhs_sand)

# Colors
c_imp = '#1f4e79'   # Deep Blue (Implicit / RS)
c_exp = '#d9534f'   # Coral Red (Explicit)
c_sand = '#16a085'  # Teal Green (Implicit / FS)

fc_imp = '#d0e1fd'
fc_exp = '#fcd0d0'
fc_sand = '#d1f2eb'

c_base = '#ffd700'  # Gold for initial geometry


def draw_panel_iters(ax):
    """Draws major iterations histogram for Implicit (RS) vs. Explicit vs. Implicit (FS) (no title)."""
    all_iters = np.arange(
        min(np.min(iters_imp), np.min(iters_exp), np.min(iters_sand)),
        max(np.max(iters_imp), np.max(iters_exp), np.max(iters_sand)) + 2,
    )
    counts_imp = [np.sum(iters_imp == k) / n_imp * 100 for k in all_iters]
    counts_exp = [np.sum(iters_exp == k) / n_exp * 100 for k in all_iters]
    counts_sand = [np.sum(iters_sand == k) / n_sand * 100 for k in all_iters]

    w = 0.26
    x_idx = np.arange(len(all_iters))
    ax.bar(x_idx - w, counts_imp, width=w, color=c_imp, label='Implicit (RS)', alpha=0.9, edgecolor='black', linewidth=0.8)
    ax.bar(x_idx,     counts_exp, width=w, color=c_exp, label='Explicit', alpha=0.9, edgecolor='black', linewidth=0.8)
    ax.bar(x_idx + w, counts_sand, width=w, color=c_sand, label='Implicit (FS)', alpha=0.9, edgecolor='black', linewidth=0.8)

    base_it_imp = base_imp['iterations']
    base_it_exp = base_exp['iterations']
    base_it_sand = base_sand['iterations']
    idx_base_imp = np.where(all_iters == base_it_imp)[0][0]
    idx_base_exp = np.where(all_iters == base_it_exp)[0][0]
    idx_base_sand = np.where(all_iters == base_it_sand)[0][0]

    ax.scatter(idx_base_imp - w, counts_imp[idx_base_imp] + 6, color=c_base, edgecolor='black', s=65, marker='D', zorder=5)
    ax.scatter(idx_base_exp, counts_exp[idx_base_exp] + 6, color=c_base, edgecolor='black', s=65, marker='D', zorder=5)
    ax.scatter(idx_base_sand + w, counts_sand[idx_base_sand] + 6, color=c_base, edgecolor='black', s=65, marker='D', zorder=5, label='Initial Geometry')

    ax.set_xticks(x_idx)
    ax.set_xticklabels(all_iters, fontsize=10.0)
    ax.set_xlabel('Major optimization iterations', fontsize=10.5)
    ax.set_ylabel('Percentage of LHS runs [%]', fontsize=10.5)
    ax.set_ylim(0, 115)
    ax.grid(axis='y', linestyle='--', alpha=0.35, color='#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    leg = ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#cccccc', framealpha=0.95, fontsize=9.5)
    for text in leg.get_texts():
        if 'Implicit (RS)' in text.get_text():
            text.set_fontweight('bold')


def draw_panel_time(ax):
    """Draws solve time boxplot and scatter for Implicit (RS) vs. Explicit vs. Implicit (FS) (no title)."""
    np.random.seed(42)
    jit_imp = np.random.normal(1, 0.04, size=len(time_imp))
    jit_exp = np.random.normal(2, 0.04, size=len(time_exp))
    jit_sand = np.random.normal(3, 0.04, size=len(time_sand))
    ax.scatter(jit_imp, time_imp, color=c_imp, alpha=0.35, s=22, edgecolors='none', zorder=2)
    ax.scatter(jit_exp, time_exp, color=c_exp, alpha=0.35, s=22, edgecolors='none', zorder=2)
    ax.scatter(jit_sand, time_sand, color=c_sand, alpha=0.35, s=22, edgecolors='none', zorder=2)

    bp = ax.boxplot(
        [time_imp, time_exp, time_sand], positions=[1, 2, 3], widths=0.46, patch_artist=True,
        showmeans=False, showfliers=False, zorder=3,
        medianprops=dict(color='black', linewidth=2.0),
        whiskerprops=dict(linewidth=1.2, linestyle='--'),
        capprops=dict(linewidth=1.2),
    )
    box_colors = [c_imp, c_exp, c_sand]
    box_fcs = [fc_imp, fc_exp, fc_sand]
    for idx, (bc, fc) in enumerate(zip(box_colors, box_fcs)):
        bp['boxes'][idx].set(facecolor=fc, edgecolor=bc, linewidth=1.4, alpha=0.85)
        bp['whiskers'][2 * idx].set(color=bc)
        bp['whiskers'][2 * idx + 1].set(color=bc)
        bp['caps'][2 * idx].set(color=bc)
        bp['caps'][2 * idx + 1].set(color=bc)

    ax.scatter(1, base_imp['time'], color=c_base, edgecolor='black', s=80, marker='D', zorder=5, label='Initial Geometry')
    ax.scatter(2, base_exp['time'], color=c_base, edgecolor='black', s=80, marker='D', zorder=5)
    ax.scatter(3, base_sand['time'], color=c_base, edgecolor='black', s=80, marker='D', zorder=5)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Implicit (RS)', 'Explicit', 'Implicit (FS)'], fontsize=10.0)
    for tick in ax.get_xticklabels():
        if 'Implicit (RS)' in tick.get_text():
            tick.set_fontweight('bold')

    ax.set_ylabel('Optimization solve time [s]', fontsize=10.5)
    ax.set_ylim(0, max(np.max(time_exp), np.max(time_imp), np.max(time_sand)) * 1.25)
    ax.grid(axis='y', linestyle='--', alpha=0.35, color='#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#cccccc', framealpha=0.95, fontsize=9.5)


# Backward-compatibility aliases
draw_panel_iters_ie = draw_panel_iters
draw_panel_time_ie = draw_panel_time


def generate_all_figures():
    # -------------------------------------------------------------------------
    # 1. Standalone Plot 1: Iterations (Implicit (RS) vs Explicit vs Implicit (FS))
    # -------------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(5.5, 4.0), dpi=300)
    draw_panel_iters(ax1)
    plt.tight_layout()
    f1_png = os.path.join(script_dir, 'benchmark_aero_iterations.png')
    f1_pdf = os.path.join(script_dir, 'benchmark_aero_iterations.pdf')
    f1_compat_png = os.path.join(script_dir, 'benchmark_aero_iterations_implicit_vs_explicit.png')
    f1_compat_pdf = os.path.join(script_dir, 'benchmark_aero_iterations_implicit_vs_explicit.pdf')
    plt.savefig(f1_png, bbox_inches='tight', dpi=300)
    plt.savefig(f1_pdf, bbox_inches='tight')
    plt.savefig(f1_compat_png, bbox_inches='tight', dpi=300)
    plt.savefig(f1_compat_pdf, bbox_inches='tight')
    plt.close(fig1)

    # -------------------------------------------------------------------------
    # 2. Standalone Plot 2: Solve Time (Implicit (RS) vs Explicit vs Implicit (FS))
    # -------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(5.5, 4.0), dpi=300)
    draw_panel_time(ax2)
    plt.tight_layout()
    f2_png = os.path.join(script_dir, 'benchmark_aero_time.png')
    f2_pdf = os.path.join(script_dir, 'benchmark_aero_time.pdf')
    f2_compat_png = os.path.join(script_dir, 'benchmark_aero_time_implicit_vs_explicit.png')
    f2_compat_pdf = os.path.join(script_dir, 'benchmark_aero_time_implicit_vs_explicit.pdf')
    plt.savefig(f2_png, bbox_inches='tight', dpi=300)
    plt.savefig(f2_pdf, bbox_inches='tight')
    plt.savefig(f2_compat_png, bbox_inches='tight', dpi=300)
    plt.savefig(f2_compat_pdf, bbox_inches='tight')
    plt.close(fig2)

    # -------------------------------------------------------------------------
    # 3. Combined 1x2 Compact Comparison Figure (no stack, titles removed)
    # -------------------------------------------------------------------------
    fig_all, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.0), dpi=300)
    draw_panel_iters(a1)
    draw_panel_time(a2)
    plt.tight_layout(w_pad=2.0)
    
    fig_png = os.path.join(script_dir, 'benchmark_comparison_combined.png')
    fig_pdf = os.path.join(script_dir, 'benchmark_comparison_combined.pdf')
    fig_stacked_png = os.path.join(script_dir, 'benchmark_comparison_combined_stacked.png')
    fig_stacked_pdf = os.path.join(script_dir, 'benchmark_comparison_combined_stacked.pdf')
    plt.savefig(fig_png, bbox_inches='tight', dpi=300)
    plt.savefig(fig_pdf, bbox_inches='tight')
    plt.savefig(fig_stacked_png, bbox_inches='tight', dpi=300)
    plt.savefig(fig_stacked_pdf, bbox_inches='tight')
    plt.close(fig_all)

    print("Aerodynamic benchmark figures successfully generated:")
    print(f"  1. Iterations:         {f1_png}")
    print(f"                         {f1_pdf}")
    print(f"  2. Solve Time:         {f2_png}")
    print(f"                         {f2_pdf}")
    print(f"  3. Combined (1x2):     {fig_png}")
    print(f"                         {fig_pdf}")
    print(f"     (Also updated {fig_stacked_png} for compatibility)")


if __name__ == '__main__':
    generate_all_figures()
