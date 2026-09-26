"""Extract and plot spanwise lift distribution for rectangular wing optimization examples.

Supported examples:
- ex_rectangular_wing_aero_shape_optimization.py
- ex_rectangular_wing_aero_shape_optimization_with_chord_profile.py
- ex_rectangular_wing_aerostructural_optimization.py
- ex_rectangular_wing_aerostructural_shape_optimization_with_chord_profile.py
- ex_rectangular_wing_to_bwb.py

Can be run:
1. Automatically at the end of an optimization script via `extract_and_plot_lift_distribution(...)`
2. Standalone from CLI:
       python extract_lift_distribution.py [optional_path_to_output_folder]
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

# Ensure working directory is repo root so relative geometry paths resolve
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
    '/home/andrew/optimization/lsdo_geo/rectangular_wing_panel_optimization_outputs',
    '/home/andrew/optimization/lsdo_geo/rectangular_wing_aerostructural_shape_optimization_with_chord_profile_outputs',
    '/home/andrew/optimization/lsdo_geo/rectangular_wing_aerostructural_optimization_outputs',
    '/home/andrew/optimization/lsdo_geo/rectangular_wing_to_bwb_aerostructural_optimization_outputs',
]


def process_strips(pc, pf_c):
    """Integrate aerodynamic panel forces into spanwise strip lift density."""
    # Group panels by spanwise station y
    y_round = np.round(pc[:, 1], 4)
    y_unique = np.unique(y_round)

    # Main lifting surface panels are arranged in chordwise rings (40 panels each)
    main_stations = [y for y in y_unique if np.sum(y_round == y) == 40]

    y_centers = []
    L_c = []

    for y_val in main_stations:
        mask = (y_round == y_val)
        y_centers.append(float(np.mean(pc[mask, 1])))
        # Lift is vertical force Fz (index 2)
        L_c.append(float(np.sum(pf_c[mask, 2])))

    # Add non-ring (tip cap) panel forces into the final tip strip so total half-wing loads are 100% conserved
    tip_mask = ~np.isin(y_round, main_stations)
    if np.any(tip_mask):
        L_c[-1] += float(np.sum(pf_c[tip_mask, 2]))

    y_centers = np.array(y_centers)
    L_c = np.array(L_c)

    b_tip = float(np.max(pc[:, 1]))

    # Compute strip boundaries and widths dy
    bounds = np.zeros(len(y_centers) + 1)
    bounds[0] = 0.0
    bounds[-1] = b_tip
    bounds[1:-1] = 0.5 * (y_centers[:-1] + y_centers[1:])
    dy = np.diff(bounds)

    # Sectional force density (per unit span) [N/m]
    dL_c = L_c / dy

    # Create fine grid for smooth continuous curve
    y_fine = np.linspace(0.0, b_tip, 300)

    # Enforce root symmetry dL/dy = 0 at y = 0 via symmetric mirroring
    y_sym = np.concatenate([-y_centers[::-1], y_centers])
    dL_c_sym = np.concatenate([dL_c[::-1], dL_c])

    # Tip boundary condition: lift vanishes at wingtip
    y_pts_L = np.concatenate([[-b_tip], y_sym, [b_tip]])
    dL_c_pts = np.concatenate([[0.0], dL_c_sym, [0.0]])

    spl_Lc = CubicSpline(y_pts_L, dL_c_pts, bc_type='natural')

    # Elliptical lift distribution benchmark (for half-wing):
    # L'(y) = (4 * L_tot) / (pi * b_tip) * sqrt(1 - (y/b_tip)^2)
    tot_Lc = float(np.sum(L_c))
    dL_ellip_c = (4.0 * tot_Lc / (np.pi * b_tip)) * np.sqrt(np.clip(1.0 - (y_fine / b_tip)**2, 0.0, 1.0))

    return {
        'y_centers': y_centers,
        'y_fine': y_fine,
        'b_tip': b_tip,
        'dy': dy,
        'L_c_strip': L_c,
        'dL_c_raw': dL_c,
        'dL_c_smooth': spl_Lc(y_fine),
        'dL_ellip_c': dL_ellip_c,
        'tot_Lc': tot_Lc,
        'spl_Lc': spl_Lc,
    }


def compute_fourier_trefftz(spl_Lc, b_tip, num_terms=15):
    """Compute Fourier sine series coefficients of circulation to calculate Oswald efficiency factor e."""
    theta = np.linspace(1e-5, np.pi - 1e-5, 2000)
    y = -b_tip * np.cos(theta)
    dL = np.maximum(spl_Lc(y), 0.0)
    n_odd = np.arange(1, 2 * num_terms, 2)
    A = []
    for n in n_odd:
        val = (2.0 / np.pi) * np.trapz(dL * np.sin(n * theta), theta)
        A.append(val)
    A = np.array(A)
    if A[0] <= 1e-12:
        return 1.0
    delta = float(np.sum(n_odd[1:] * (A[1:] / A[0]) ** 2))
    e_val = 1.0 / (1.0 + max(delta, 0.0))
    return float(np.clip(e_val, 0.1, 1.0))


def extract_cdi_from_output_folder(output_folder, sref_val=10.0):
    """Attempt to parse optimal induced drag coefficient CDi from optimization logs."""
    modopt_path = os.path.join(output_folder, 'modopt_results.out')
    if os.path.exists(modopt_path):
        problem_name = None
        objective_val = None
        with open(modopt_path, 'r') as f:
            for line in f:
                if 'Problem' in line:
                    problem_name = line.split(':')[-1].strip()
                elif 'Objective' in line:
                    try:
                        objective_val = float(line.split(':')[-1].strip())
                    except ValueError:
                        pass
        if objective_val is not None:
            if problem_name == 'rectangular_wing_to_bwb_aerostructural_optimization':
                # objective = CDi[0] * 1.e3
                return float(objective_val / 1.e3)
            elif problem_name == 'rectangular_wing_panel_optimization':
                # objective = CDi * 1.e4
                return float(objective_val / 1.e4)
            elif problem_name == 'rectangular_wing_aerostructural_shape_optimization_with_chord_profile':
                # objective = Di_Trefftz * 1.e1
                q = 0.5 * 1.225 * (20.0 ** 2)
                di = objective_val / 1.e1
                return float(di / (q * sref_val))
            elif problem_name == 'rectangular_wing_aerostructural_optimization':
                # objective = Di (V=20 m/s, rho=1.225)
                q = 0.5 * 1.225 * (20.0 ** 2)
                return float(objective_val / (q * sref_val))
    return None


def find_matching_module(output_folder):
    """Determine which rectangular wing script produced this output folder."""
    folder_name = os.path.basename(os.path.normpath(output_folder))
    parent_name = os.path.basename(os.path.dirname(os.path.normpath(output_folder)))

    if 'to_bwb' in parent_name or 'to_bwb' in folder_name:
        import examples.showcase_examples.rectangular_wing.ex_rectangular_wing_to_bwb as mod
        return mod
    elif 'aerostructural_shape_optimization_with_chord_profile' in parent_name or 'aerostructural_shape_optimization_with_chord_profile' in folder_name:
        import examples.showcase_examples.rectangular_wing.ex_rectangular_wing_aerostructural_shape_optimization_with_chord_profile as mod
        return mod
    elif 'aerostructural_optimization' in parent_name or 'aerostructural_optimization' in folder_name:
        import examples.showcase_examples.rectangular_wing.ex_rectangular_wing_aerostructural_optimization as mod
        return mod
    elif 'panel_optimization' in parent_name or 'panel_optimization' in folder_name:
        # Check x.out length: with_chord_profile has 6 DVs (4 taper + AR + pitch),
        # whereas standard aero_shape_optimization has 3 DVs (AR, area, pitch) or 7 DVs
        x_out_path = os.path.join(output_folder, 'x.out')
        if os.path.exists(x_out_path):
            x_hist = np.loadtxt(x_out_path)
            dim = x_hist.shape[1] if len(x_hist.shape) > 1 else len(x_hist)
            if dim == 6:
                import examples.showcase_examples.rectangular_wing.ex_rectangular_wing_aero_shape_optimization_with_chord_profile as mod
                return mod
            elif dim in [3, 7]:
                import examples.showcase_examples.rectangular_wing.ex_rectangular_wing_aero_shape_optimization as mod
                return mod
        import examples.showcase_examples.rectangular_wing.ex_rectangular_wing_aero_shape_optimization_with_chord_profile as mod
        return mod

    # Fallback default
    import examples.showcase_examples.rectangular_wing.ex_rectangular_wing_aero_shape_optimization_with_chord_profile as mod
    return mod


def extract_and_plot_lift_distribution(output_folder, jax_sim=None, main_script=None, artifact_dir=None, force_rerun=False):
    """Core analysis routine: extracts telemetry, integrates strips, plots, and saves cache."""
    output_folder = os.path.abspath(output_folder)
    cache_file = os.path.join(output_folder, 'lift_distribution_data.npz')

    if artifact_dir is None:
        curr_id = '3256dd7c-d4c8-4c73-887d-131361f9d0c3'
        candidate = os.environ.get('ARTIFACT_DIR', f'/home/andrew/.gemini/antigravity/brain/{curr_id}')
        if not os.path.exists(candidate):
            old = '/home/andrew/.gemini/antigravity/brain/680209fd-293d-4fa0-9f6c-ae59a72a6987'
            artifact_dir = old if os.path.exists(old) else None
        else:
            artifact_dir = candidate

    # 1. Obtain Telemetry (from in-memory jax_sim, cache, or fresh evaluation)
    if jax_sim is not None and main_script is not None:
        # In-memory simulator already evaluated
        print(f"\n--- Extracting Lift Distribution for: {os.path.basename(output_folder)} ---")
        panel_centers_right = np.asarray(jax_sim[main_script.dynamic_panel_centers_right])

        if hasattr(main_script, 'panel_forces_right_cruise'):
            f_cruise = np.asarray(jax_sim[main_script.panel_forces_right_cruise])
        elif hasattr(main_script, 'panel_forces_right'):
            f_cruise = np.asarray(jax_sim[main_script.panel_forces_right])
        else:
            num_right = len(panel_centers_right)
            f_cruise = np.asarray(jax_sim[main_script.outputs['panel_forces']][0, :num_right, :])

        cl_val = float(np.asarray(jax_sim[main_script.CL]).flatten()[0])
        cdi_val = float(np.asarray(jax_sim[main_script.CDi]).flatten()[0]) if hasattr(main_script, 'CDi') else None

        if hasattr(main_script, 'planform_area'):
            sref_val = float(np.asarray(jax_sim[main_script.planform_area]).flatten()[0])
        else:
            sref_val = 10.0

        if hasattr(main_script, 'aspect_ratio_calc'):
            ar_val = float(np.asarray(jax_sim[main_script.aspect_ratio_calc]).flatten()[0])
        else:
            b_val = 2.0 * float(np.max(panel_centers_right[:, 1]))
            ar_val = (b_val ** 2) / sref_val

        if cdi_val is None or cdi_val <= 0:
            cdi_val = extract_cdi_from_output_folder(output_folder, sref_val)

        induced_drag_objective = getattr(main_script, 'induced_drag_objective', 'trefftz')
        cdi_fourier_cache = float(np.asarray(jax_sim[main_script.CDi_Fourier]).flatten()[0]) if hasattr(main_script, 'CDi_Fourier') else None
        e_fourier_cache = float(np.asarray(jax_sim[main_script.e_fourier]).flatten()[0]) if hasattr(main_script, 'e_fourier') else None

    elif (os.path.exists(os.path.join(output_folder, 'lift_and_moment_data.npz')) or os.path.exists(cache_file)) and not force_rerun:
        active_cache = os.path.join(output_folder, 'lift_and_moment_data.npz') if os.path.exists(os.path.join(output_folder, 'lift_and_moment_data.npz')) else cache_file
        print(f"Loading cached panel telemetry from: {active_cache}")
        data = np.load(active_cache)
        panel_centers_right = data['panel_centers_right']
        f_cruise = data['f_cruise']

        b_val = 2.0 * float(np.max(panel_centers_right[:, 1]))
        sref_val = float(data['sref_val']) if 'sref_val' in data else 10.0
        ar_val = float(data['ar_val']) if 'ar_val' in data else (b_val ** 2) / sref_val

        if 'cl_val' in data and float(data['cl_val']) > 0:
            cl_val = float(data['cl_val'])
        else:
            q_cruise = 0.5 * 1.225 * (20.0 ** 2)
            tot_L = 2.0 * float(np.sum(f_cruise[:, 2]))
            cl_val = tot_L / (q_cruise * sref_val)

        cdi_val = float(data['cdi_val']) if ('cdi_val' in data and float(data['cdi_val']) > 0) else None
        cdi_fourier_cache = None
        if 'cdi_fourier_val' in data:
            cdi_fourier_cache = float(data['cdi_fourier_val'])
        elif 'cdi_fourier' in data:
            cdi_fourier_cache = float(data['cdi_fourier'])

        e_fourier_cache = None
        if 'e_fourier_val' in data:
            e_fourier_cache = float(data['e_fourier_val'])
        elif 'e_fourier' in data:
            e_fourier_cache = float(data['e_fourier'])

        lift_ratio_cache = float(data['lift_ratio']) if 'lift_ratio' in data else None

        induced_drag_objective = str(data['induced_drag_objective']) if 'induced_drag_objective' in data else 'trefftz'
        if cdi_val is None or cdi_val <= 0:
            cdi_val = extract_cdi_from_output_folder(output_folder, sref_val)

    else:
        induced_drag_objective = 'trefftz'
        cdi_fourier_cache = None
        e_fourier_cache = None
        print(f"No cache found at {cache_file}; identifying model and evaluating at optimal design variables...")
        main_script = find_matching_module(output_folder)
        jax_sim = main_script.jax_sim
        design_variables = main_script.design_variables

        x_out_path = os.path.join(output_folder, 'x.out')
        if not os.path.exists(x_out_path):
            raise FileNotFoundError(f"x.out not found in {output_folder}")

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

        jax_sim.run()

        panel_centers_right = np.asarray(jax_sim[main_script.dynamic_panel_centers_right])
        if hasattr(main_script, 'panel_forces_right_cruise'):
            f_cruise = np.asarray(jax_sim[main_script.panel_forces_right_cruise])
        elif hasattr(main_script, 'panel_forces_right'):
            f_cruise = np.asarray(jax_sim[main_script.panel_forces_right])
        else:
            num_right = len(panel_centers_right)
            f_cruise = np.asarray(jax_sim[main_script.outputs['panel_forces']][0, :num_right, :])

        cl_val = float(np.asarray(jax_sim[main_script.CL]).flatten()[0])
        cdi_val = float(np.asarray(jax_sim[main_script.CDi]).flatten()[0]) if hasattr(main_script, 'CDi') else None
        lift_ratio_cache = float(np.asarray(jax_sim[main_script.trefftz_lift_ratio]).flatten()[0]) if hasattr(main_script, 'trefftz_lift_ratio') else None

        if hasattr(main_script, 'planform_area'):
            sref_val = float(np.asarray(jax_sim[main_script.planform_area]).flatten()[0])
        else:
            sref_val = 10.0

        if hasattr(main_script, 'aspect_ratio_calc'):
            ar_val = float(np.asarray(jax_sim[main_script.aspect_ratio_calc]).flatten()[0])
        else:
            b_val = 2.0 * float(np.max(panel_centers_right[:, 1]))
            ar_val = (b_val ** 2) / sref_val

        if cdi_val is None or cdi_val <= 0:
            cdi_val = extract_cdi_from_output_folder(output_folder, sref_val)

    # Ensure cache is saved/updated with full telemetry for instant subsequent access
    try:
        np.savez_compressed(
            cache_file,
            panel_centers_right=panel_centers_right,
            f_cruise=f_cruise,
            cl_val=cl_val,
            cdi_val=cdi_val if cdi_val is not None else 0.0,
            lift_ratio=lift_ratio_cache if lift_ratio_cache is not None else 1.0,
            sref_val=sref_val,
            ar_val=ar_val,
        )
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_file}: {e}")

    # 2. Strip Integration
    results = process_strips(panel_centers_right, f_cruise)
    y_fine = results['y_fine']
    y_centers = results['y_centers']
    b_tip = results['b_tip']

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
    c_cruise = '#1f77b4'
    c_ellip = '#2ca02c'

    ax.plot(
        y_fine, results['dL_c_smooth'], '-', color=c_cruise, linewidth=2.5,
        label=f"Optimized Lift Distribution ($L_{{\\text{{half}}}} = {results['tot_Lc']:.2f}$ N, $L_{{\\text{{total}}}} = {2*results['tot_Lc']:.2f}$ N)"
    )
    ax.scatter(
        y_centers, results['dL_c_raw'], color=c_cruise, s=42, zorder=5,
        edgecolors='white', linewidth=1.0, label="Integrated Strip Values"
    )
    ax.fill_between(y_fine, results['dL_c_smooth'], alpha=0.15, color=c_cruise)

    ax.plot(
        y_fine, results['dL_ellip_c'], '--', color=c_ellip, linewidth=2.0, alpha=0.9,
        label="Ideal Elliptical Benchmark ($C_{Di,\\text{min}}$)"
    )

    if e_fourier_cache is not None and e_fourier_cache > 0:
        e_fourier = e_fourier_cache
        cdi_fourier = cdi_fourier_cache if (cdi_fourier_cache is not None and cdi_fourier_cache > 0) else ((cl_val**2) / (np.pi * ar_val * e_fourier))
    else:
        e_fourier = compute_fourier_trefftz(results['spl_Lc'], b_tip)
        cdi_fourier = (cl_val**2) / (np.pi * ar_val * e_fourier) if (ar_val > 0 and e_fourier > 0) else cdi_val

    if cdi_val is not None and cdi_val > 0 and ar_val > 0:
        e_solver = (cl_val**2) / (np.pi * ar_val * cdi_val)
    else:
        e_solver = e_fourier
        cdi_val = cdi_fourier

    is_unphysical_e = (e_solver > 1.005)

    # Persist complete cache with all aerodynamic variables
    np.savez_compressed(
        cache_file,
        panel_centers_right=panel_centers_right,
        f_cruise=f_cruise,
        cl_val=cl_val,
        cdi_val=cdi_val,
        cdi_fourier=cdi_fourier,
        cdi_fourier_val=cdi_fourier,
        e_fourier=e_fourier,
        e_fourier_val=e_fourier,
        e_solver=e_solver,
        induced_drag_objective=induced_drag_objective,
        sref_val=sref_val,
        ar_val=ar_val,
    )

    run_label = os.path.basename(output_folder)
    obj_str = f"Obj: {induced_drag_objective.capitalize()} $C_{{Di}}$ | " if induced_drag_objective in ['fourier', 'mixed'] else ""
    perf_line = (
        f"{obj_str}$C_L = {cl_val:.4f}$ | "
        f"$C_{{Di,\\text{{Fourier}}}} = {cdi_fourier*1e4:.1f}$ cts ($e = {e_fourier:.3f}$) | "
        f"$C_{{Di,\\text{{Trefftz}}}} = {cdi_val*1e4:.1f}$ cts ($e = {e_solver:.3f}$)"
    )

    title_lines = [
        f"Spanwise Lift Distribution: {run_label}",
        f"Right Half-Span $b/2 = {b_tip:.3f}$ m (Full $b = {2*b_tip:.3f}$ m) | $AR = {ar_val:.2f}$ | $S_{{\\text{{ref}}}} = {sref_val:.2f}\\text{{ m}}^2$",
        perf_line,
    ]
    ax.set_title("\n".join(title_lines), fontsize=10.5, fontweight='bold', pad=12)
    ax.set_xlabel("Spanwise Coordinate $y$ [m]", fontsize=11, fontweight='bold')
    ax.set_ylabel("Sectional Lift Density $L'(y) = dL/dy$ [N/m]", fontsize=11, fontweight='bold')
    ax.set_xlim([0.0, b_tip * 1.02])
    min_lift_density = min(float(np.min(results['dL_c_smooth'])), float(np.min(results['dL_c_raw'])))
    if min_lift_density < 0.0:
        y_bot = min_lift_density * 1.25
        ax.set_ylim(bottom=y_bot)
        ax.axhline(0.0, color='gray', linestyle='--', linewidth=1.0, alpha=0.7)
    else:
        ax.set_ylim(bottom=0.0)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=9.5, framealpha=0.95)

    obj_info = f"Objective: {induced_drag_objective.capitalize()} Induced Drag\n" if induced_drag_objective in ['fourier', 'mixed'] else ""
    lr_line = f"Wake Lift Ratio = {lift_ratio_cache:.4f}\n" if lift_ratio_cache is not None else ""
    info_text = (
        f"{obj_info}"
        f"Fourier e = {e_fourier:.4f}\n"
        f"Fourier CDi = {cdi_fourier*1e4:.2f} counts\n"
        f"Trefftz e = {e_solver:.4f}\n"
        f"Trefftz CDi = {cdi_val*1e4:.2f} counts\n"
        f"{lr_line}"
        f"Lift Coefficient CL = {cl_val:.4f}"
    )

    ax.text(
        0.04, 0.15,
        info_text,
        transform=ax.transAxes,
        fontsize=9.5,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#cccccc', alpha=0.92)
    )

    plt.tight_layout()
    lift_fig_path = os.path.join(output_folder, 'lift_distribution.png')
    plt.savefig(lift_fig_path, dpi=200, bbox_inches='tight')
    print(f"Lift distribution figure saved to: {lift_fig_path}")

    if os.path.exists(artifact_dir):
        artifact_fig_path = os.path.join(artifact_dir, f"lift_distribution_{run_label}.png")
        plt.savefig(artifact_fig_path, dpi=200, bbox_inches='tight')
        # Also maintain un-suffixed copy
        plt.savefig(os.path.join(artifact_dir, 'lift_distribution.png'), dpi=200, bbox_inches='tight')

    plt.close()

    # 4. Summary Printout
    print("\n" + "="*65)
    print(f"AERODYNAMIC LIFT TELEMETRY SUMMARY (Run: {run_label})")
    print("="*65)
    print(f"Wing Half-Span (b/2):            {b_tip:.3f} m (Full Wingspan: {2*b_tip:.3f} m)")
    print(f"Calculated Aspect Ratio (AR):    {ar_val:.2f}")
    print(f"Planform Area (S_ref):           {sref_val:.3f} m^2")
    print(f"Number of Spanwise Strips:       {len(y_centers)}")
    print(f"Cruise Lift Coefficient (CL):    {cl_val:.5f}")
    print(f"Trefftz Induced Drag (CDi):      {cdi_val:.6f} ({cdi_val*1e4:.2f} drag counts)")
    print(f"Trefftz Efficiency Factor (e):   {e_solver:.4f}")
    if lift_ratio_cache is not None:
        print(f"Wake Lift Ratio (L_surf/L_wake): {lift_ratio_cache:.4f}")
    print(f"Fourier Physical Eff (e_Fourier):{e_fourier:.4f} (CDi: {cdi_fourier*1e4:.2f} counts)")
    print(f"Cruise Half-Wing Lift:           {results['tot_Lc']:.3f} N (Full Wing: {2*results['tot_Lc']:.3f} N)")
    print(f"Peak Lift Density (Root):        {results['dL_c_smooth'][0]:.3f} N/m (Elliptical: {results['dL_ellip_c'][0]:.3f} N/m)")
    print("="*65 + "\n")

    return results


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    include_cl = ('--cl' in sys.argv or '--with-cl' in sys.argv)

    if args:
        target_folder = os.path.abspath(args[0])
    else:
        # Scan all known output directories for the most recent run
        candidate_folders = []
        for base_dir in KNOWN_OUTPUT_DIRS:
            if os.path.exists(base_dir):
                candidate_folders.extend(glob.glob(os.path.join(base_dir, '*')))
        if not candidate_folders:
            raise FileNotFoundError("No optimization output folders found in known output directories.")
        target_folder = max(candidate_folders, key=os.path.getmtime)

    print(f"Target optimization output directory: {target_folder}")
    extract_and_plot_lift_distribution(target_folder)

    if include_cl:
        try:
            from optimization_analyses.extract_cl_distribution import extract_and_plot_cl_distribution
            extract_and_plot_cl_distribution(target_folder)
        except Exception as e:
            print(f"Notice: could not run cl distribution analysis: {e}")


if __name__ == '__main__':
    main()
