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

            # Chord line connecting LE midpoint to TE midpoint
            z_le = 0.5 * (zu[0] + zl[0])
            z_te = 0.5 * (zu[-1] + zl[-1])
            z_chord_line = z_le + (z_te - z_le) * x_norm
            twist_deg = float(np.degrees(np.arcsin(np.clip(-(z_te - z_le) / c, -1.0, 1.0))))

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


# Suite 1: 5 DVs across Spanwise Mesh Resolutions (15S, 35S, 60S)
cases_5dv_meshes = {
    '5DV_15S': ('2026-10-09_17.37.55.051059', '#1f77b4', '--', '5 DVs — 15S (14 Span Strips, 1,272 Quads)'),
    '5DV_35S': ('2026-10-09_19.38.04.667437', '#2ca02c', '-',  '5 DVs — 35S (34 Span Strips, 2,872 Quads)'),
    '5DV_60S': ('2026-10-10_07.32.42.035823', '#d62728', '-.', '5 DVs — 60S (59 Span Strips, 4,872 Quads)'),
}
make_planform_and_lift_plot(
    cases_5dv_meshes,
    title_plan='5-DV Parameterization: Optimal Planform Across Spanwise Mesh Resolutions (15S, 35S, 60S)',
    title_lift="5-DV Parameterization: Spanwise Lift Density Distribution L'(y) Across Mesh Resolutions",
    filename='planform_and_lift_5dv_meshes.png',
    ref_key='5DV_35S',
)
make_airfoil_gallery_plot(
    cases_5dv_meshes,
    suptitle='5-DV Parameterization: Normalized Airfoil Profiles Across Spanwise Mesh Resolutions (15S, 35S, 60S)',
    filename='airfoil_gallery_5dv_meshes.png',
)

# Suite 2: Fixed 35S (34 Span Strips) Mesh across DV Counts (5 DV, 6 DV, 8 DV)
cases_35s_dvs = {
    '5DV_35S': ('2026-10-09_19.38.04.667437', '#2ca02c', '--', '5 DVs (35S Mesh, r = 6.80 Strips/DV)'),
    '6DV_35S': ('2026-10-12_11.37.46.801487', '#9467bd', '-.', '6 DVs (35S Mesh, r = 5.67 Strips/DV)'),
    '8DV_35S': ('2026-10-11_09.54.25.037286', '#ff7f0e', '-',  '8 DVs (35S Mesh, r = 4.25 Strips/DV)'),
}
make_planform_and_lift_plot(
    cases_35s_dvs,
    title_plan='Fixed 35S Mesh (34 Spanwise Strips): Optimal Planform for 5 DVs, 6 DVs, and 8 DVs',
    title_lift="Fixed 35S Mesh (34 Spanwise Strips): Spanwise Lift Density Distribution L'(y) for 5, 6, and 8 DVs",
    filename='planform_and_lift_35s_dvs.png',
    ref_key='5DV_35S',
)
make_airfoil_gallery_plot(
    cases_35s_dvs,
    suptitle='Fixed 35S Mesh (34 Spanwise Strips): Normalized Airfoil Profiles for 5 DVs, 6 DVs, and 8 DVs',
    filename='airfoil_gallery_35s_dvs.png',
)
