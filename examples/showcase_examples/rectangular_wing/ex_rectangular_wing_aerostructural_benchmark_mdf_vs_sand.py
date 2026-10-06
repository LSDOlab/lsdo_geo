"""
Benchmarking Script: Aerostructural Shape Optimization (MDF vs. SAND)
=====================================================================
Experimental Platform: ex_rectangular_wing_aerostructural_shape_optimization.py
Coupled aerodynamic (VortexAD) and structural (aframe beam) optimization.

Compares MDO Architectures:
1. MDF (Multidisciplinary Feasible / Implicit):
   - Parameterization constraints solved internally by ParameterizationSolver.
   - Design variable: aspect_ratio (1 DV)
   - Optimizer constraints: tip_deflection <= 0.020 m (1 inequality constraint)

2. SAND (Simultaneous Analysis and Design):
   - Parameterization constraints solved simultaneously by the optimizer (SLSQP).
   - Design variables: aspect_ratio, chord_stretch_state, span_stretch_state (3 DVs)
   - Optimizer constraints:
     - planform_area_geom == 10.0 (equality constraint)
     - aspect_ratio_geom - aspect_ratio == 0.0 (equality constraint)
     - tip_deflection <= 0.020 m (structural inequality constraint)

Structural wall thicknesses are held constant to drive the optimal aspect ratio (AR)
to an interior optimum (AR* ≈ 12.6) strictly between initial (AR0 = 10.0) and upper bound (AR = 15.0).
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"
import sys
import time
import pickle
import json
import numpy as np
from scipy.stats import qmc
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from ex_rectangular_wing_aerostructural_shape_optimization import build_optimization_model
import modopt


def run_single_optimization(sim, design_variables, outputs_dict, formulation, x0_dict, problem_name="opt", solver_acc=1e-6, maxiter=100):
    prob = modopt.CSDLAlphaProblem(problem_name=problem_name, simulator=sim)

    x0_unscaled_list = []
    x0_scaled_list = []
    curr_idx = 0
    for dv_name, dv_info in design_variables.items():
        var_size = dv_info.variable.shape[0] if len(dv_info.variable.shape) > 0 else 1
        if dv_name in x0_dict:
            val = np.atleast_1d(np.asarray(x0_dict[dv_name], dtype=float))
        else:
            val = np.atleast_1d(np.asarray(dv_info.variable.value, dtype=float))

        x0_unscaled_list.append(val)
        x0_scaled_list.append((val + prob.x_adder[curr_idx:curr_idx+var_size]) * prob.x_scaler[curr_idx:curr_idx+var_size])
        curr_idx += var_size

    x0_scaled = np.concatenate(x0_scaled_list)

    prob.x0 = x0_scaled.copy()
    prob.warm_x = x0_scaled.copy() - 1.0
    prob.warm_x_deriv = x0_scaled.copy() - 2.0

    optimizer = modopt.PySLSQP(
        prob,
        solver_options={'maxiter': maxiter, 'acc': solver_acc, 'iprint': 0},
        turn_off_outputs=True
    )
    optimizer.x0 = x0_scaled.copy()

    t_start = time.perf_counter()
    optimizer.solve()
    t_end = time.perf_counter()
    wall_time = t_end - t_start

    res = optimizer.results

    opt_x_scaled = res.get('x', x0_scaled)
    opt_x_dict = {}
    curr_idx = 0
    for dv_name, dv_info in design_variables.items():
        var_size = dv_info.variable.shape[0] if len(dv_info.variable.shape) > 0 else 1
        sc = prob.x_scaler[curr_idx:curr_idx+var_size]
        ad = prob.x_adder[curr_idx:curr_idx+var_size]
        unscaled_val = opt_x_scaled[curr_idx:curr_idx+var_size] / sc - ad
        opt_x_dict[dv_name] = unscaled_val[0] if var_size == 1 else unscaled_val
        sim[dv_info.variable] = unscaled_val
        curr_idx += var_size

    sim.run()

    cdi_var = outputs_dict.get('CDi')
    cl_var = outputs_dict.get('CL')
    s_var = outputs_dict.get('planform_area')
    ar_var = outputs_dict.get('aspect_ratio_calc')
    disp_var = outputs_dict.get('tip_deflection')

    cdi_val = float(sim[cdi_var][0]) if cdi_var is not None else 0.0
    cl_val = float(sim[cl_var][0]) if cl_var is not None else 0.0
    s_val = float(sim[s_var][0]) if s_var is not None else 10.0
    ar_val = float(sim[ar_var][0]) if ar_var is not None else 10.0
    disp_val = float(sim[disp_var][0]) if disp_var is not None else 0.0

    record = {
        'formulation': formulation,
        'success': bool(res.get('success', False)),
        'status': int(res.get('status', -1)),
        'message': str(res.get('message', '')),
        'iterations': int(res.get('num_majiter', 0)),
        'nfev': int(res.get('nfev', 0)),
        'ngev': int(res.get('ngev', 0)),
        'time': float(wall_time),
        'fev_time': float(res.get('fev_time', 0.0)),
        'gev_time': float(res.get('gev_time', 0.0)),
        'objective': float(res.get('objective', 0.0)),
        'optimality': float(res.get('optimality', 0.0)),
        'feasibility': float(res.get('feasibility', 0.0)),
        'initial_x': {k: (float(v[0]) if hasattr(v, '__len__') else float(v)) for k, v in x0_dict.items()},
        'optimal_x': {k: (float(v[0]) if hasattr(v, '__len__') else float(v)) for k, v in opt_x_dict.items()},
        'cl': cl_val,
        'cdi_counts': cdi_val * 1e4,
        'planform_area': s_val,
        'aspect_ratio': ar_val,
        'tip_deflection_mm': disp_val * 1e3,
    }

    return record


def run_benchmark(n_samples=50, random_seed=42):
    print("=" * 80)
    print("AEROSTRUCTURAL SHAPE OPTIMIZATION BENCHMARK: MDF VS. SAND")
    print(f"Number of LHS samples: {n_samples} | Seed: {random_seed}")
    print("Constant spar thickness: 3.0 mm | Allowable tip deflection: 20.0 mm")
    print("=" * 80)

    print("\n[1/4] Compiling MDF Aerostructural Model ('mdf')...")
    sim_mdf, dvs_mdf, outs_mdf, _, _ = build_optimization_model(formulation='mdf')

    print("[2/4] Compiling SAND Aerostructural Model ('sand')...")
    sim_sand, dvs_sand, outs_sand, _, _ = build_optimization_model(formulation='sand')

    print("\n[3/4] Running Baseline Comparison (Starting at AR = 10.0)...")
    res_base_mdf = run_single_optimization(
        sim=sim_mdf, design_variables=dvs_mdf, outputs_dict=outs_mdf,
        formulation='mdf', x0_dict={'aspect_ratio': 10.0}, problem_name='base_mdf'
    )
    print(f"  MDF Baseline: Iters={res_base_mdf['iterations']}, Time={res_base_mdf['time']:.2f}s, "
          f"Opt AR={res_base_mdf['aspect_ratio']:.3f}, CDi={res_base_mdf['cdi_counts']:.2f} counts, "
          f"Tip Disp={res_base_mdf['tip_deflection_mm']:.2f} mm")

    res_base_sand = run_single_optimization(
        sim=sim_sand, design_variables=dvs_sand, outputs_dict=outs_sand,
        formulation='sand',
        x0_dict={'aspect_ratio': 10.0, 'chord_stretch_state': 0.0, 'span_stretch_state': 0.0},
        problem_name='base_sand'
    )
    print(f"  SAND Baseline: Iters={res_base_sand['iterations']}, Time={res_base_sand['time']:.2f}s, "
          f"Opt AR={res_base_sand['aspect_ratio']:.3f}, CDi={res_base_sand['cdi_counts']:.2f} counts, "
          f"Tip Disp={res_base_sand['tip_deflection_mm']:.2f} mm")

    print(f"\n[4/4] Executing {n_samples} Latin Hypercube Samples...")
    lhs_sampler_1d = qmc.LatinHypercube(d=1, seed=random_seed)
    lhs_samples_ar = qmc.scale(lhs_sampler_1d.random(n=n_samples), l_bounds=[6.0], u_bounds=[12.0])

    lhs_sampler_3d = qmc.LatinHypercube(d=3, seed=random_seed)
    lhs_samples_sand = qmc.scale(
        lhs_sampler_3d.random(n=n_samples),
        l_bounds=[6.0, -0.20, -1.50],
        u_bounds=[12.0, 0.20, 1.50]
    )

    results_lhs_mdf = []
    print("\nRunning MDF LHS Evaluations...")
    t_lhs_mdf_start = time.perf_counter()
    for i, ar_val in enumerate(lhs_samples_ar):
        res = run_single_optimization(
            sim=sim_mdf, design_variables=dvs_mdf, outputs_dict=outs_mdf,
            formulation='mdf', x0_dict={'aspect_ratio': float(ar_val[0])},
            problem_name=f'lhs_mdf_{i}'
        )
        results_lhs_mdf.append(res)
        status_str = "SUCCESS" if res['success'] else f"FAILED({res['status']})"
        if (i + 1) % 10 == 0 or (i + 1) == n_samples or not res['success']:
            print(f"  [MDF {i+1:3d}/{n_samples}] AR0={ar_val[0]:5.2f} | Iters={res['iterations']:2d} Fev={res['nfev']:2d} Time={res['time']:5.2f}s | Opt AR={res['aspect_ratio']:5.2f} | {status_str}")

    t_lhs_mdf_total = time.perf_counter() - t_lhs_mdf_start

    results_lhs_sand = []
    print("\nRunning SAND LHS Evaluations...")
    t_lhs_sand_start = time.perf_counter()
    for i, sample in enumerate(lhs_samples_sand):
        res = run_single_optimization(
            sim=sim_sand, design_variables=dvs_sand, outputs_dict=outs_sand,
            formulation='sand',
            x0_dict={
                'aspect_ratio': float(sample[0]),
                'chord_stretch_state': float(sample[1]),
                'span_stretch_state': float(sample[2])
            },
            problem_name=f'lhs_sand_{i}'
        )
        results_lhs_sand.append(res)
        status_str = "SUCCESS" if res['success'] else f"FAILED({res['status']})"
        if (i + 1) % 10 == 0 or (i + 1) == n_samples or not res['success']:
            print(f"  [SAND {i+1:3d}/{n_samples}] AR0={sample[0]:5.2f} c_str0={sample[1]:5.2f} b_str0={sample[2]:5.2f} | Iters={res['iterations']:2d} Fev={res['nfev']:2d} Time={res['time']:5.2f}s | Opt AR={res['aspect_ratio']:5.2f} | {status_str}")

    t_lhs_sand_total = time.perf_counter() - t_lhs_sand_start

    iters_mdf = np.array([r['iterations'] for r in results_lhs_mdf])
    iters_sand = np.array([r['iterations'] for r in results_lhs_sand])
    time_mdf = np.array([r['time'] for r in results_lhs_mdf])
    time_sand = np.array([r['time'] for r in results_lhs_sand])
    conv_mdf = np.array([r['success'] for r in results_lhs_mdf])
    conv_sand = np.array([r['success'] for r in results_lhs_sand])

    print("\n" + "=" * 80)
    print("STATISTICAL BENCHMARK SUMMARY (MDF VS. SAND AEROSTRUCTURAL)")
    print("=" * 80)
    print(f"{'Metric':<30} | {'MDF':<20} | {'SAND':<20}")
    print("-" * 76)
    print(f"{'Total Samples':<30} | {n_samples:<20d} | {n_samples:<20d}")
    print(f"{'Convergence Rate':<30} | {np.mean(conv_mdf)*100:<19.1f}% | {np.mean(conv_sand)*100:<19.1f}%")
    print(f"{'Mean Iterations':<30} | {np.mean(iters_mdf):<20.2f} | {np.mean(iters_sand):<20.2f}")
    print(f"{'Median Iterations':<30} | {np.median(iters_mdf):<20.1f} | {np.median(iters_sand):<20.1f}")
    print(f"{'Mean Solve Time (s)':<30} | {np.mean(time_mdf):<20.2f} | {np.mean(time_sand):<20.2f}")
    print(f"{'Median Solve Time (s)':<30} | {np.median(time_mdf):<20.2f} | {np.median(time_sand):<20.2f}")
    print("=" * 80)

    results_payload = {
        'metadata': {
            'n_samples': n_samples,
            'random_seed': random_seed,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        'baseline': {
            'mdf': res_base_mdf,
            'sand': res_base_sand,
        },
        'lhs': {
            'mdf': results_lhs_mdf,
            'sand': results_lhs_sand,
        },
        'summary': {
            'iterations': {
                'mdf_mean': float(np.mean(iters_mdf)),
                'sand_mean': float(np.mean(iters_sand)),
                'mdf_median': float(np.median(iters_mdf)),
                'sand_median': float(np.median(iters_sand)),
            },
            'solve_time': {
                'mdf_mean': float(np.mean(time_mdf)),
                'sand_mean': float(np.mean(time_sand)),
                'mdf_median': float(np.median(time_mdf)),
                'sand_median': float(np.median(time_sand)),
            },
            'convergence_rate': {
                'mdf': float(np.mean(conv_mdf)),
                'sand': float(np.mean(conv_sand)),
            }
        }
    }

    pkl_path = os.path.join(script_dir, 'aerostructural_benchmark_results_mdf_vs_sand.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results_payload, f)
    print(f"\nRaw benchmark data saved to: {pkl_path}")

    json_path = os.path.join(script_dir, 'aerostructural_benchmark_summary_mdf_vs_sand.json')
    with open(json_path, 'w') as f:
        json.dump(results_payload['summary'], f, indent=4)
    print(f"Summary JSON saved to:       {json_path}")

    # Plot Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0), dpi=300)
    c_mdf = '#1f4e79'
    c_sand = '#16a085'
    fc_mdf = '#d0e1fd'
    fc_sand = '#d1f2eb'

    all_iters = np.arange(min(np.min(iters_mdf), np.min(iters_sand)), max(np.max(iters_mdf), np.max(iters_sand)) + 2)
    counts1 = [np.sum(iters_mdf == k) / n_samples * 100 for k in all_iters]
    counts2 = [np.sum(iters_sand == k) / n_samples * 100 for k in all_iters]

    w = 0.38
    x_indices = np.arange(len(all_iters))
    ax1.bar(x_indices - w/2, counts1, width=w, color=c_mdf, label='MDF', alpha=0.9, edgecolor='black', linewidth=0.8)
    ax1.bar(x_indices + w/2, counts2, width=w, color=c_sand, label='SAND', alpha=0.9, edgecolor='black', linewidth=0.8)

    base_it1 = res_base_mdf['iterations']
    base_it2 = res_base_sand['iterations']
    ax1.scatter(np.where(all_iters == base_it1)[0][0] - w/2, 105, color='#ffd700', edgecolor='black', s=65, marker='D', zorder=5)
    ax1.scatter(np.where(all_iters == base_it2)[0][0] + w/2, counts2[np.where(all_iters == base_it2)[0][0]] + 6,
                color='#ffd700', edgecolor='black', s=65, marker='D', zorder=5, label='Baseline Geometry')

    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(all_iters, fontsize=10.5)
    ax1.set_xlabel('Major optimization iterations', fontsize=11)
    ax1.set_ylabel('Percentage of LHS runs [%]', fontsize=11)
    ax1.set_ylim(0, 115)
    ax1.grid(axis='y', linestyle='--', alpha=0.35, color='#cccccc')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_title('Major Optimization Iterations', fontsize=11, fontweight='bold', pad=10)
    ax1.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#cccccc', framealpha=0.95, fontsize=10)

    np.random.seed(42)
    jit1 = np.random.normal(1, 0.04, size=len(time_mdf))
    jit2 = np.random.normal(2, 0.04, size=len(time_sand))
    ax2.scatter(jit1, time_mdf, color=c_mdf, alpha=0.35, s=22, edgecolors='none', zorder=2)
    ax2.scatter(jit2, time_sand, color=c_sand, alpha=0.35, s=22, edgecolors='none', zorder=2)

    bp = ax2.boxplot(
        [time_mdf, time_sand], positions=[1, 2], widths=0.46, patch_artist=True,
        showmeans=False, showfliers=False, zorder=3,
        medianprops=dict(color='black', linewidth=2.0),
        whiskerprops=dict(linewidth=1.2, linestyle='--'),
        capprops=dict(linewidth=1.2)
    )
    bp['boxes'][0].set(facecolor=fc_mdf, edgecolor=c_mdf, linewidth=1.4, alpha=0.85)
    bp['boxes'][1].set(facecolor=fc_sand, edgecolor=c_sand, linewidth=1.4, alpha=0.85)
    bp['whiskers'][0].set(color=c_mdf)
    bp['whiskers'][1].set(color=c_mdf)
    bp['whiskers'][2].set(color=c_sand)
    bp['whiskers'][3].set(color=c_sand)
    bp['caps'][0].set(color=c_mdf)
    bp['caps'][1].set(color=c_mdf)
    bp['caps'][2].set(color=c_sand)
    bp['caps'][3].set(color=c_sand)

    ax2.scatter(1, res_base_mdf['time'], color='#ffd700', edgecolor='black', s=80, marker='D', zorder=5, label='Baseline Geometry')
    ax2.scatter(2, res_base_sand['time'], color='#ffd700', edgecolor='black', s=80, marker='D', zorder=5)

    ax2.text(1, np.max(time_mdf) + 0.45, f"{np.mean(time_mdf):.2f} s",
             ha='center', va='bottom', fontsize=8.8, fontweight='bold', color=c_mdf)
    ax2.text(2, np.max(time_sand) + 0.45, f"{np.mean(time_sand):.2f} s",
             ha='center', va='bottom', fontsize=8.8, fontweight='bold', color=c_sand)

    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(['MDF', 'SAND'], fontsize=10.5)
    ax2.set_ylabel('Optimization solve time [s]', fontsize=11)
    ax2.set_ylim(0, max(np.max(time_sand), np.max(time_mdf)) * 1.25)
    ax2.grid(axis='y', linestyle='--', alpha=0.35, color='#cccccc')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title('Optimization Solve Time', fontsize=11, fontweight='bold', pad=10)
    ax2.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#cccccc', framealpha=0.95, fontsize=10)

    plt.tight_layout()
    fig_path = os.path.join(script_dir, 'aerostructural_benchmark_comparison_mdf_vs_sand.png')
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Comparison figure saved to: {fig_path}")


if __name__ == '__main__':
    samples = 25
    if len(sys.argv) > 1:
        try:
            samples = int(sys.argv[1])
        except ValueError:
            pass
    run_benchmark(n_samples=samples)
