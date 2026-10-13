import os
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, "../../../.."))
base_dir = os.path.join(repo_root, 'rectangular_wing_to_bwb_aerostructural_optimization_outputs')
art_dir = '/home/andrewfletcher/.gemini/antigravity/brain/a2f03ba8-1c5a-4fe2-ba15-b4ee2887956e'
out_dir = os.path.join(repo_root, 'examples/showcase_examples/rectangular_wing')

def get_strip_data(folder):
    d = np.load(f'{base_dir}/{folder}/lift_distribution_data.npz')
    pc = d['panel_centers_right']
    pf = d['f_cruise']
    y_round = np.round(pc[:, 1], 3)
    u_y, counts = np.unique(y_round, return_counts=True)
    dom = int(np.max(counts))
    main_y = [y for y, c in zip(u_y, counts) if c == dom]

    y_arr, c_arr, L_strip, x_le_arr, x_te_arr = [], [], [], [], []
    for y_val in main_y:
        mask = (y_round == y_val)
        y_arr.append(np.mean(pc[mask, 1]))
        x_pts = pc[mask, 0]
        x_le_arr.append(np.min(x_pts))
        x_te_arr.append(np.max(x_pts))
        c_arr.append(np.max(x_pts) - np.min(x_pts))
        L_strip.append(np.sum(pf[mask, 2]))

    tip_mask = ~np.isin(y_round, main_y)
    if np.any(tip_mask):
        L_strip[-1] += np.sum(pf[tip_mask, 2])

    y_arr, c_arr, x_le_arr, x_te_arr, L_strip = map(
        np.array, [y_arr, c_arr, x_le_arr, x_te_arr, L_strip]
    )
    b = float(np.max(pc[:, 1]))
    bounds = np.concatenate(([0.0], 0.5 * (y_arr[:-1] + y_arr[1:]), [b]))
    dy = np.diff(bounds)
    L_prime = L_strip / dy
    return {
        'y': y_arr,
        'c': c_arr,
        'x_le': x_le_arr,
        'x_te': x_te_arr,
        'b': b,
        'L_prime': L_prime,
        'L_strip': L_strip,
    }


def make_planform_and_lift_plot(cases_dict, title_plan, title_lift, filename, ref_key=None):
    fig, (ax_plan, ax_lift) = plt.subplots(
        2, 1, figsize=(14, 11.5), dpi=200,
        gridspec_kw={'height_ratios': [1.05, 1.0], 'hspace': 0.44}
    )

    for name, (folder, color, lstyle, label) in cases_dict.items():
        res = get_strip_data(folder)
        y, x_le, x_te, L_prime, b = (
            res['y'],
            res['x_le'],
            res['x_te'],
            res['L_prime'],
            res['b'],
        )
        y_full_le = np.concatenate((-y[::-1], y))
        x_full_le = np.concatenate((x_le[::-1], x_le))
        y_full_te = np.concatenate((-y[::-1], y))
        x_full_te = np.concatenate((x_te[::-1], x_te))

        ax_plan.plot(y_full_le, x_full_le, color=color, linestyle=lstyle, linewidth=2.3, label=label)
        ax_plan.plot(y_full_te, x_full_te, color=color, linestyle=lstyle, linewidth=2.3)
        ax_plan.plot([y[-1], y[-1]], [x_le[-1], x_te[-1]], color=color, linestyle=lstyle, linewidth=2.3)
        ax_plan.plot([-y[-1], -y[-1]], [x_le[-1], x_te[-1]], color=color, linestyle=lstyle, linewidth=2.3)

        ax_lift.plot(y / b, L_prime / 1000.0, color=color, linestyle=lstyle, linewidth=2.4, label=label)

    if ref_key is None:
        ref_key = list(cases_dict.keys())[0]
    res_ref = get_strip_data(cases_dict[ref_key][0])
    eta_dense = np.linspace(0, 1, 200)
    L0_ellip = (4.0 * np.sum(res_ref['L_strip'])) / (np.pi * res_ref['b'])
    ax_lift.plot(
        eta_dense,
        (L0_ellip * np.sqrt(np.maximum(0, 1 - eta_dense**2))) / 1000.0,
        color='#495057',
        linestyle=':',
        linewidth=2.0,
        label='Ideal Elliptical Lift Reference',
    )

    ax_plan.set_title(title_plan, fontsize=13.5, fontweight='bold', pad=12)
    ax_plan.set_xlabel('Spanwise Coordinate $y$ [m]', fontsize=11.5, labelpad=6)
    ax_plan.set_ylabel('Streamwise Coordinate $x$ [m]', fontsize=11.5)
    ax_plan.set_aspect('equal')
    ax_plan.invert_yaxis()
    ax_plan.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.24),
        ncol=min(3, len(cases_dict)),
        frameon=True,
        fontsize=10.0,
    )
    ax_plan.grid(True, linestyle=':', alpha=0.6)

    ax_lift.set_title(title_lift, fontsize=13.5, fontweight='bold', pad=12)
    ax_lift.set_xlabel(r'Normalized Semi-Span Station $\eta = 2y/b$ [-]', fontsize=11.5)
    ax_lift.set_ylabel(r"Sectional Lift Density $L'(y)$ [kN/m]", fontsize=11.5)
    ax_lift.set_xlim([0, 1.02])
    ax_lift.legend(loc='upper right', frameon=True, fontsize=10.5)
    ax_lift.grid(True, linestyle=':', alpha=0.6)

    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    shutil.copy2(out_path, os.path.join(art_dir, filename))
    print(f"Saved: {out_path}")


def make_airfoil_gallery_plot(cases_dict, suptitle, filename):
    stations = [
        (1, r'Station 1: Root Centerbody ($\eta = 0.00$)'),
        (2, r'Station 2: Inboard Blending ($\eta = 0.25$)'),
        (3, r'Station 3: Mid-Span Transition ($\eta = 0.50$)'),
        (4, r'Station 4: Outer Wing ($\eta = 0.75$)'),
        (5, r'Station 5: Wingtip ($\eta = 1.00$)'),
    ]
    fig, axes = plt.subplots(5, 2, figsize=(16, 15.5), dpi=200, sharex=True)
    for idx, (st_num, title) in enumerate(stations):
        ax_inst = axes[idx, 0]
        ax_pure = axes[idx, 1]
        for name, (folder, color, lstyle, label) in cases_dict.items():
            d = np.load(f'{base_dir}/{folder}/airfoil_cross_sections_data.npz')
            x = d[f'station_{st_num}_x_grid']
            zu = d[f'station_{st_num}_z_upper']
            zl = d[f'station_{st_num}_z_lower']
            camber = d[f'station_{st_num}_camber_line']
            c = np.max(x) - np.min(x)
            x_norm = (x - np.min(x)) / c
            t_max_pct = np.max(zu - zl) / c * 100.0

            z_le = 0.5 * (zu[0] + zl[0])
            z_te = 0.5 * (zu[-1] + zl[-1])
            z_chord_line = z_le + (z_te - z_le) * x_norm

            lbl = f'{label} (Root $c={c:.1f}$m, $t/c={t_max_pct:.1f}$%)' if idx == 0 else None

            # Left column: Installed profile (includes geometric twist)
            ax_inst.plot(x_norm, zu / c * 100.0, color=color, linestyle=lstyle, linewidth=2.1, label=lbl)
            ax_inst.plot(x_norm, zl / c * 100.0, color=color, linestyle=lstyle, linewidth=2.1)
            ax_inst.plot(x_norm, camber / c * 100.0, color=color, linestyle=':', alpha=0.55, linewidth=1.2)

            # Right column: De-twisted profile (pure thickness & camber relative to local chord line)
            ax_pure.plot(x_norm, (zu - z_chord_line) / c * 100.0, color=color, linestyle=lstyle, linewidth=2.1)
            ax_pure.plot(x_norm, (zl - z_chord_line) / c * 100.0, color=color, linestyle=lstyle, linewidth=2.1)
            ax_pure.plot(x_norm, (camber - z_chord_line) / c * 100.0, color=color, linestyle=':', alpha=0.55, linewidth=1.2)

        ax_inst.set_title(f"{title} — Installed (With Twist)", fontsize=11.5, fontweight='bold', pad=5)
        ax_pure.set_title(f"{title} — De-Twisted (Pure Camber & Thickness)", fontsize=11.5, fontweight='bold', pad=5)

        for ax in (ax_inst, ax_pure):
            ax.set_ylabel(r'$z / c$ [%]', fontsize=10)
            ax.set_xlim([-0.02, 1.02])
            ax.grid(True, linestyle=':', alpha=0.6)
        ax_inst.set_ylim([-7.5, 10.5])
        ax_pure.set_ylim([-6.5, 8.5])
        if idx == 4:
            ax_inst.set_xlabel(r'Normalized Chord Coordinate $x / c$ [-]', fontsize=10.5)
            ax_pure.set_xlabel(r'Normalized Chord Coordinate $x / c$ [-]', fontsize=10.5)

    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=0.992)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.966), ncol=min(3, len(cases_dict)), frameon=True, fontsize=10.0)
    plt.tight_layout(rect=[0, 0, 1, 0.935])
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    shutil.copy2(out_path, os.path.join(art_dir, filename))
    print(f"Saved: {out_path}")


cases_chordwise = {
    '5DV_35S_20C': ('2026-10-12_17.38.56.154169', '#17a2b8', '--', '20 Chord Panels (10 Upper/Lower, 1,432 Quads)'),
    '5DV_35S_40C': ('2026-10-09_19.38.04.667437', '#28a745', '-',  '40 Chord Panels (20 Upper/Lower, 2,872 Quads, Baseline)'),
    '5DV_35S_60C': ('2026-10-12_20.14.26.055007', '#dc3545', '-.', '60 Chord Panels (30 Upper/Lower, 4,312 Quads)'),
}

make_planform_and_lift_plot(
    cases_chordwise,
    title_plan='35-Section Mesh (5 DVs): Optimal Planform Across Chordwise Mesh Refinement (20C, 40C, 60C)',
    title_lift="35-Section Mesh (5 DVs): Spanwise Lift Density Distribution L'(y) Across Chordwise Refinement",
    filename='planform_and_lift_chordwise_refinement.png',
    ref_key='5DV_35S_40C',
)

make_airfoil_gallery_plot(
    cases_chordwise,
    suptitle='35-Section Mesh (5 DVs): Normalized Airfoil Profiles Across Chordwise Mesh Refinement (20C, 40C, 60C)',
    filename='airfoil_gallery_chordwise_refinement.png',
)
