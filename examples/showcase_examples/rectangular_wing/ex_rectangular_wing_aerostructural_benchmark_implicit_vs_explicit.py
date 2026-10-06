"""
Benchmarking Script: Aerostructural Shape Optimization (Implicit vs. Explicit)
=============================================================================
Experimental Platform: ex_rectangular_wing_aerostructural_shape_optimization.py
Coupled aerodynamic (VortexAD) and structural (aframe beam) optimization.

Compares Geometric Parameterization Formulations:
1. Implicit Parameterization ('ar_area'):
   - Parameterization constraints solved internally by ParameterizationSolver (Newton).
   - Optimizer design variable: aspect_ratio (1 DV)
   - Optimizer equality constraints: None (0 constraints)
   - Optimizer inequality constraints: tip_deflection <= 0.020 m (1 constraint, normalized to 1.0)

2. Explicit Parameterization ('chord_span'):
   - Direct FFD sectional parameters: chord stretch and span stretch
   - Optimizer design variables: chord_stretch_dv, span_stretch_dv (2 DVs)
   - Optimizer equality constraints: planform_area == 10.0 m^2 (1 constraint, normalized to 1.0)
   - Optimizer inequality constraints: tip_deflection <= 0.020 m (1 constraint, normalized to 1.0)
   - (aspect_ratio <= 15.0 inactive upper bound)

Because the implicit formulation eliminates the equality constraint and reduces
the design space from 2D to 1D, it achieves faster convergence and lower solve times.
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


def run_benchmark(n_samples=10, random_seed=42):
    print("=" * 80)
    print("AEROSTRUCTURAL SHAPE OPTIMIZATION BENCHMARK: IMPLICIT VS. EXPLICIT")
    print(f"Number of LHS samples: {n_samples} | Seed: {random_seed}")
    print("Constant spar thickness: 3.0 mm | Allowable tip deflection: 20.0 mm (scaler=50.0)")
    print("=" * 80)

    print("\n[1/4] Compiling Implicit Aerostructural Model ('ar_area')...")
    sim_imp, dvs_imp, outs_imp, _, _ = build_optimization_model(formulation='ar_area', disp_scaler=50.0)

    print("[2/4] Compiling Explicit Aerostructural Model ('chord_span')...")
    sim_exp, dvs_exp, outs_exp, _, _ = build_optimization_model(formulation='chord_span', disp_scaler=50.0)

    print("\n[3/4] Running Baseline Comparison (Starting at Baseline AR = 10.0)...")
    res_base_imp = run_single_optimization(
        sim=sim_imp, design_variables=dvs_imp, outputs_dict=outs_imp,
        formulation='implicit', x0_dict={'aspect_ratio': 10.0}, problem_name='base_imp'
    )
    print(f"  Implicit Baseline: Iters={res_base_imp['iterations']}, Time={res_base_imp['time']:.2f}s, "
          f"Opt AR={res_base_imp['aspect_ratio']:.3f}, CDi={res_base_imp['cdi_counts']:.2f} counts, "
          f"Tip Disp={res_base_imp['tip_deflection_mm']:.2f} mm")

    res_base_exp = run_single_optimization(
        sim=sim_exp, design_variables=dvs_exp, outputs_dict=outs_exp,
        formulation='explicit',
        x0_dict={'chord_stretch_dv': 0.0, 'span_stretch_dv': 0.0},
        problem_name='base_exp'
    )
    print(f"  Explicit Baseline: Iters={res_base_exp['iterations']}, Time={res_base_exp['time']:.2f}s, "
          f"Opt AR={res_base_exp['aspect_ratio']:.3f}, CDi={res_base_exp['cdi_counts']:.2f} counts, "
          f"Tip Disp={res_base_exp['tip_deflection_mm']:.2f} mm")

    print(f"\n[4/4] Executing {n_samples} Latin Hypercube Samples...")
    # 1D LHS for Implicit
    lhs_sampler_imp = qmc.LatinHypercube(d=1, seed=random_seed)
    lhs_samples_imp = qmc.scale(lhs_sampler_imp.random(n=n_samples), l_bounds=[6.0], u_bounds=[12.0])

    # 2D LHS for Explicit
    lhs_sampler_exp = qmc.LatinHypercube(d=2, seed=random_seed + 100)
    lhs_samples_exp = qmc.scale(
        lhs_sampler_exp.random(n=n_samples),
        l_bounds=[-0.20, -1.0],
        u_bounds=[0.20, 1.0]
    )

    results_lhs_imp = []
    print("\nRunning Implicit LHS Evaluations (1D: aspect_ratio)...")
    t_lhs_imp_start = time.perf_counter()
    for i, ar_val in enumerate(lhs_samples_imp):
        res = run_single_optimization(
            sim=sim_imp, design_variables=dvs_imp, outputs_dict=outs_imp,
            formulation='implicit', x0_dict={'aspect_ratio': float(ar_val[0])},
            problem_name=f'lhs_imp_{i}'
        )
        results_lhs_imp.append(res)
        status_str = "SUCCESS" if res['success'] else f"FAILED({res['status']})"
        if (i + 1) % 5 == 0 or (i + 1) == n_samples or not res['success']:
            print(f"  [Implicit {i+1:3d}/{n_samples}] AR0={ar_val[0]:5.2f} | Iters={res['iterations']:2d} Fev={res['nfev']:2d} Time={res['time']:5.2f}s | Opt AR={res['aspect_ratio']:5.2f} | {status_str}")

    t_lhs_imp_total = time.perf_counter() - t_lhs_imp_start

    results_lhs_exp = []
    print("\nRunning Explicit LHS Evaluations (2D: chord_stretch, span_stretch)...")
    t_lhs_exp_start = time.perf_counter()
    for i, pt in enumerate(lhs_samples_exp):
        res = run_single_optimization(
            sim=sim_exp, design_variables=dvs_exp, outputs_dict=outs_exp,
            formulation='explicit',
            x0_dict={'chord_stretch_dv': float(pt[0]), 'span_stretch_dv': float(pt[1])},
            problem_name=f'lhs_exp_{i}'
        )
        results_lhs_exp.append(res)
        status_str = "SUCCESS" if res['success'] else f"FAILED({res['status']})"
        if (i + 1) % 5 == 0 or (i + 1) == n_samples or not res['success']:
            print(f"  [Explicit {i+1:3d}/{n_samples}] c_str0={pt[0]:5.2f} b_str0={pt[1]:5.2f} | Iters={res['iterations']:2d} Fev={res['nfev']:2d} Time={res['time']:5.2f}s | Opt AR={res['aspect_ratio']:5.2f} | {status_str}")

    t_lhs_exp_total = time.perf_counter() - t_lhs_exp_start

    iters_imp = np.array([r['iterations'] for r in results_lhs_imp])
    iters_exp = np.array([r['iterations'] for r in results_lhs_exp])
    time_imp = np.array([r['time'] for r in results_lhs_imp])
    time_exp = np.array([r['time'] for r in results_lhs_exp])
    conv_imp = np.array([r['success'] for r in results_lhs_imp])
    conv_exp = np.array([r['success'] for r in results_lhs_exp])

    print("\n" + "=" * 80)
    print("STATISTICAL BENCHMARK SUMMARY (IMPLICIT VS. EXPLICIT AEROSTRUCTURAL)")
    print("=" * 80)
    print(f"{'Metric':<30} | {'Implicit':<20} | {'Explicit':<20}")
    print("-" * 76)
    print(f"{'Total Samples':<30} | {n_samples:<20d} | {n_samples:<20d}")
    print(f"{'Convergence Rate':<30} | {np.mean(conv_imp)*100:<19.1f}% | {np.mean(conv_exp)*100:<19.1f}%")
    print(f"{'Mean Iterations':<30} | {np.mean(iters_imp):<20.2f} | {np.mean(iters_exp):<20.2f}")
    print(f"{'Median Iterations':<30} | {np.median(iters_imp):<20.1f} | {np.median(iters_exp):<20.1f}")
    print(f"{'Mean Solve Time (s)':<30} | {np.mean(time_imp):<20.2f} | {np.mean(time_exp):<20.2f}")
    print(f"{'Median Solve Time (s)':<30} | {np.median(time_imp):<20.2f} | {np.median(time_exp):<20.2f}")
    print("=" * 80)

    results_payload = {
        'metadata': {
            'n_samples': n_samples,
            'random_seed': random_seed,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        'baseline': {
            'implicit': res_base_imp,
            'explicit': res_base_exp,
        },
        'lhs': {
            'implicit': results_lhs_imp,
            'explicit': results_lhs_exp,
        },
        'summary': {
            'iterations': {
                'implicit_mean': float(np.mean(iters_imp)),
                'explicit_mean': float(np.mean(iters_exp)),
                'implicit_median': float(np.median(iters_imp)),
                'explicit_median': float(np.median(iters_exp)),
            },
            'solve_time': {
                'implicit_mean': float(np.mean(time_imp)),
                'explicit_mean': float(np.mean(time_exp)),
                'implicit_median': float(np.median(time_imp)),
                'explicit_median': float(np.median(time_exp)),
            },
            'convergence_rate': {
                'implicit': float(np.mean(conv_imp)),
                'explicit': float(np.mean(conv_exp)),
            }
        }
    }

    pkl_path = os.path.join(script_dir, 'aerostructural_benchmark_results_implicit_vs_explicit.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results_payload, f)
    print(f"\nRaw benchmark data saved to: {pkl_path}")

    json_path = os.path.join(script_dir, 'aerostructural_benchmark_summary_implicit_vs_explicit.json')
    with open(json_path, 'w') as f:
        json.dump(results_payload['summary'], f, indent=4)
    print(f"Summary JSON saved to:       {json_path}")

    # Plot Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0), dpi=300)
    c_imp = '#1f4e79'
    c_exp = '#d9534f'
    fc_imp = '#d0e1fd'
    fc_exp = '#fcd0d0'

    all_iters = np.arange(min(np.min(iters_imp), np.min(iters_exp)), max(np.max(iters_imp), np.max(iters_exp)) + 2)
    counts1 = [np.sum(iters_imp == k) / n_samples * 100 for k in all_iters]
    counts2 = [np.sum(iters_exp == k) / n_samples * 100 for k in all_iters]

    w = 0.38
    x_indices = np.arange(len(all_iters))
    ax1.bar(x_indices - w/2, counts1, width=w, color=c_imp, label='Implicit', alpha=0.9, edgecolor='black', linewidth=0.8)
    ax1.bar(x_indices + w/2, counts2, width=w, color=c_exp, label='Explicit', alpha=0.9, edgecolor='black', linewidth=0.8)

    base_it1 = res_base_imp['iterations']
    base_it2 = res_base_exp['iterations']
    ax1.scatter(np.where(all_iters == base_it1)[0][0] - w/2, counts1[np.where(all_iters == base_it1)[0][0]] + 6,
                color='#ffd700', edgecolor='black', s=65, marker='D', zorder=5)
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
    jit1 = np.random.normal(1, 0.04, size=len(time_imp))
    jit2 = np.random.normal(2, 0.04, size=len(time_exp))
    ax2.scatter(jit1, time_imp, color=c_imp, alpha=0.35, s=22, edgecolors='none', zorder=2)
    ax2.scatter(jit2, time_exp, color=c_exp, alpha=0.35, s=22, edgecolors='none', zorder=2)

    bp = ax2.boxplot(
        [time_imp, time_exp], positions=[1, 2], widths=0.46, patch_artist=True,
        showmeans=False, showfliers=False, zorder=3,
        medianprops=dict(color='black', linewidth=2.0),
        whiskerprops=dict(linewidth=1.2, linestyle='--'),
        capprops=dict(linewidth=1.2)
    )
    bp['boxes'][0].set(facecolor=fc_imp, edgecolor=c_imp, linewidth=1.4, alpha=0.85)
    bp['boxes'][1].set(facecolor=fc_exp, edgecolor=c_exp, linewidth=1.4, alpha=0.85)
    bp['whiskers'][0].set(color=c_imp)
    bp['whiskers'][1].set(color=c_imp)
    bp['whiskers'][2].set(color=c_exp)
    bp['whiskers'][3].set(color=c_exp)
    bp['caps'][0].set(color=c_imp)
    bp['caps'][1].set(color=c_imp)
    bp['caps'][2].set(color=c_exp)
    bp['caps'][3].set(color=c_exp)

    ax2.scatter(1, res_base_imp['time'], color='#ffd700', edgecolor='black', s=80, marker='D', zorder=5, label='Baseline Geometry')
    ax2.scatter(2, res_base_exp['time'], color='#ffd700', edgecolor='black', s=80, marker='D', zorder=5)

    ax2.text(1, np.max(time_imp) + 0.45, f"{np.mean(time_imp):.2f} s",
             ha='center', va='bottom', fontsize=8.8, fontweight='bold', color=c_imp)
    ax2.text(2, np.max(time_exp) + 0.45, f"{np.mean(time_exp):.2f} s",
             ha='center', va='bottom', fontsize=8.8, fontweight='bold', color=c_exp)

    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(['Implicit', 'Explicit'], fontsize=10.5)
    ax2.set_ylabel('Optimization solve time [s]', fontsize=11)
    ax2.set_ylim(0, max(np.max(time_exp), np.max(time_imp)) * 1.25)
    ax2.grid(axis='y', linestyle='--', alpha=0.35, color='#cccccc')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title('Optimization Solve Time', fontsize=11, fontweight='bold', pad=10)
    ax2.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#cccccc', framealpha=0.95, fontsize=10)

    plt.tight_layout()
    fig_path = os.path.join(script_dir, 'aerostructural_benchmark_comparison_implicit_vs_explicit.png')
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Comparison figure saved to: {fig_path}")


if __name__ == '__main__':
    samples = 10
    if len(sys.argv) > 1:
        try:
            samples = int(sys.argv[1])
        except ValueError:
            pass
    run_benchmark(n_samples=samples)
