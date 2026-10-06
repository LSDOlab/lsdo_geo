"""
Benchmarking Script: Implicit vs. Explicit Geometry Parameterization
===================================================================
Experimental Platform: ex_rectangular_wing_aero_shape_optimization.py

Isolated Parameterization Benchmark:
- Pitch angle fixed at 5.0 degrees (not a design variable)
- CL constraint removed

Compares:
1. Implicit Formulation ('ar_area'):
   - Geometric variables: Aspect Ratio (AR) and Planform Area (S = 10.0 m^2)
   - Solved via ParameterizationSolver (internal Newton solver)
   - Optimizer design variables: aspect_ratio (1 DV, bounds [2.0, 15.0])
   - Optimizer constraints: None (0 constraints)

2. Explicit Formulation ('chord_span'):
   - Direct FFD sectional parameters: chord stretch and span stretch
   - Optimizer design variables: chord_stretch_dv, span_stretch_dv (2 DVs)
   - Optimizer constraints: planform_area == 10.0, AR <= 15.0 (2 constraints)

Metrics tracked:
- Number of major optimization iterations
- Number of function evaluations (nfev)
- Number of gradient/derivative evaluations (ngev)
- Total optimization solve time (seconds)
- Convergence success rate and final optimality/feasibility
- Final aerodynamic metrics (CL, CDi, AR, S_ref)

Tests:
- Part 1: Baseline comparison starting from initial un-deformed geometry
- Part 2: Statistical comparison over Latin Hypercube Sampling (LHS) of initial conditions
          (1D LHS for implicit, 2D LHS for explicit)
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

# Ensure local script directory is on sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Import model builder from isolated rectangular wing optimization script
from ex_rectangular_wing_aero_shape_optimization import build_optimization_model
import modopt


def run_single_optimization(sim, design_variables, outputs_dict, formulation, x0_dict, problem_name="opt", solver_acc=1e-7, maxiter=100):
    """
    Executes a single PySLSQP optimization starting from specified physical x0 values
    using an already-compiled JaxSimulator.
    """
    prob = modopt.CSDLAlphaProblem(problem_name=problem_name, simulator=sim)

    # Compute scaled x0 vector
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

    # Set x0 in problem and ensure cache is reset
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

    # Unpack optimal solution
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

    # Evaluate simulator at optimal point to retrieve final physical telemetry
    sim.run()

    # Retrieve telemetry from simulator using CSDL Variable keys
    cdi_var = outputs_dict.get('CDi')
    cl_var = outputs_dict.get('CL')
    s_var = outputs_dict.get('planform_area')
    ar_var = outputs_dict.get('aspect_ratio_calc')

    cdi_val = float(sim[cdi_var][0]) if cdi_var is not None else 0.0
    cl_val = float(sim[cl_var][0]) if cl_var is not None else 0.0
    s_val = float(sim[s_var][0]) if s_var is not None else 10.0
    ar_val = float(sim[ar_var][0]) if ar_var is not None else 10.0

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
    }

    return record


def run_benchmark(n_samples=50, random_seed=42):
    """
    Runs the complete benchmark suite comparing implicit vs explicit formulations:
    - Baseline from initial geometry
    - Latin Hypercube Sampling across initial conditions
    """
    print("=" * 80)
    print("IMPLICIT VS. EXPLICIT GEOMETRY PARAMETERIZATION BENCHMARK")
    print("=" * 80)
    print(f"LHS Samples per Formulation: {n_samples}")
    print(f"Random Seed:                 {random_seed}")
    print(f"Refined Objective Scaler:    1.0e3 (CDi)")
    print(f"Refined CL Scaler:           1.0e1 (equals=0.5)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. BUILD MODELS (ONE-TIME COMPILATION FOR EACH FORMULATION)
    # -------------------------------------------------------------------------
    print("\n[1/4] Building and compiling Implicit ('ar_area') Model...")
    t0 = time.perf_counter()
    sim_imp, dvs_imp, outs_imp, geo_imp, ffd_imp = build_optimization_model(formulation='ar_area', obj_scaler=1e3)
    # Warm-up / compile simulator once
    prob_warmup_imp = modopt.CSDLAlphaProblem(problem_name='warmup_imp', simulator=sim_imp)
    opt_warmup_imp = modopt.PySLSQP(prob_warmup_imp, solver_options={'maxiter': 1, 'iprint': 0}, turn_off_outputs=True)
    opt_warmup_imp.solve()
    t_compile_imp = time.perf_counter() - t0
    print(f"  Implicit model ready and compiled in {t_compile_imp:.2f} s.")

    print("\n[2/4] Building and compiling Explicit ('chord_span') Model...")
    t0 = time.perf_counter()
    sim_exp, dvs_exp, outs_exp, geo_exp, ffd_exp = build_optimization_model(formulation='chord_span', obj_scaler=1e3)
    # Warm-up / compile simulator once
    prob_warmup_exp = modopt.CSDLAlphaProblem(problem_name='warmup_exp', simulator=sim_exp)
    opt_warmup_exp = modopt.PySLSQP(prob_warmup_exp, solver_options={'maxiter': 1, 'iprint': 0}, turn_off_outputs=True)
    opt_warmup_exp.solve()
    t_compile_exp = time.perf_counter() - t0
    print(f"  Explicit model ready and compiled in {t_compile_exp:.2f} s.")

    # -------------------------------------------------------------------------
    # 2. BASELINE COMPARISON FROM INITIAL GEOMETRY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[3/4] BENCHMARK PART 1: BASELINE (STARTING FROM INITIAL GEOMETRY)")
    print("=" * 80)

    baseline_x0_imp = {
        'aspect_ratio': 10.0,
    }
    baseline_x0_exp = {
        'chord_stretch_dv': 0.0,
        'span_stretch_dv': 0.0,
    }

    print("\nRunning Baseline Implicit ('ar_area')...")
    res_base_imp = run_single_optimization(
        sim=sim_imp,
        design_variables=dvs_imp,
        outputs_dict=outs_imp,
        formulation='ar_area',
        x0_dict=baseline_x0_imp,
        problem_name='baseline_implicit'
    )
    print(f"  Implicit Baseline: Iters={res_base_imp['iterations']}, Fev={res_base_imp['nfev']}, Time={res_base_imp['time']:.2f}s, CDi={res_base_imp['cdi_counts']:.2f} counts, CL={res_base_imp['cl']:.4f}")

    print("\nRunning Baseline Explicit ('chord_span')...")
    res_base_exp = run_single_optimization(
        sim=sim_exp,
        design_variables=dvs_exp,
        outputs_dict=outs_exp,
        formulation='chord_span',
        x0_dict=baseline_x0_exp,
        problem_name='baseline_explicit'
    )
    print(f"  Explicit Baseline: Iters={res_base_exp['iterations']}, Fev={res_base_exp['nfev']}, Time={res_base_exp['time']:.2f}s, CDi={res_base_exp['cdi_counts']:.2f} counts, CL={res_base_exp['cl']:.4f}")

    # Baseline side-by-side summary table
    print("\n" + "-" * 75)
    print(f"{'Metric':<30} | {'Implicit (ar_area)':<20} | {'Explicit (chord_span)':<20}")
    print("-" * 75)
    print(f"{'Major Iterations':<30} | {res_base_imp['iterations']:<20d} | {res_base_exp['iterations']:<20d}")
    print(f"{'Function Evaluations (nfev)':<30} | {res_base_imp['nfev']:<20d} | {res_base_exp['nfev']:<20d}")
    print(f"{'Gradient Evaluations (ngev)':<30} | {res_base_imp['ngev']:<20d} | {res_base_exp['ngev']:<20d}")
    print(f"{'Optimization Time (s)':<30} | {res_base_imp['time']:<20.2f} | {res_base_exp['time']:<20.2f}")
    print(f"{'Final Trefftz CDi (counts)':<30} | {res_base_imp['cdi_counts']:<20.2f} | {res_base_exp['cdi_counts']:<20.2f}")
    print(f"{'Final CL':<30} | {res_base_imp['cl']:<20.4f} | {res_base_exp['cl']:<20.4f}")
    print(f"{'Final Aspect Ratio':<30} | {res_base_imp['aspect_ratio']:<20.2f} | {res_base_exp['aspect_ratio']:<20.2f}")
    print(f"{'Final Planform Area (m^2)':<30} | {res_base_imp['planform_area']:<20.3f} | {res_base_exp['planform_area']:<20.3f}")
    print(f"{'Solver Status':<30} | {res_base_imp['message']:<20} | {res_base_exp['message']:<20}")
    print("-" * 75)

    # -------------------------------------------------------------------------
    # 3. LATIN HYPERCUBE SAMPLING (LHS) COMPARISON
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"[4/4] BENCHMARK PART 2: LATIN HYPERCUBE SAMPLING ({n_samples} SAMPLES EACH)")
    print("=" * 80)

    # Generate LHS for Implicit: 1D space [aspect_ratio]
    # AR bounds: [5.0, 13.0]
    lhs_bounds_imp_low = np.array([5.0])
    lhs_bounds_imp_high = np.array([13.0])
    sampler_imp = qmc.LatinHypercube(d=1, seed=random_seed)
    unit_samples_imp = sampler_imp.random(n=n_samples)
    lhs_samples_imp = qmc.scale(unit_samples_imp, lhs_bounds_imp_low, lhs_bounds_imp_high)

    # Generate LHS for Explicit: 2D space [chord_stretch_dv, span_stretch_dv]
    # chord_stretch: [-0.25, 0.35], span_stretch: [-1.5, 1.5]
    lhs_bounds_exp_low = np.array([-0.25, -1.5])
    lhs_bounds_exp_high = np.array([0.35, 1.5])
    sampler_exp = qmc.LatinHypercube(d=2, seed=random_seed + 100)
    unit_samples_exp = sampler_exp.random(n=n_samples)
    lhs_samples_exp = qmc.scale(unit_samples_exp, lhs_bounds_exp_low, lhs_bounds_exp_high)

    results_lhs_imp = []
    results_lhs_exp = []

    # Run Implicit LHS
    print(f"\n--- Running Implicit ('ar_area') LHS ({n_samples} samples) ---")
    t_lhs_imp_start = time.perf_counter()
    for i, sample in enumerate(lhs_samples_imp):
        x0_s = {
            'aspect_ratio': sample[0],
        }
        res = run_single_optimization(
            sim=sim_imp,
            design_variables=dvs_imp,
            outputs_dict=outs_imp,
            formulation='ar_area',
            x0_dict=x0_s,
            problem_name=f'lhs_imp_{i}'
        )
        results_lhs_imp.append(res)
        status_str = "SUCCESS" if res['success'] else f"FAILED({res['status']})"
        if (i + 1) % 10 == 0 or (i + 1) == n_samples or not res['success']:
            print(f"  [Implicit {i+1:3d}/{n_samples}] AR0={sample[0]:5.2f} | Iters={res['iterations']:2d} Fev={res['nfev']:2d} Time={res['time']:5.2f}s | CDi={res['cdi_counts']:5.2f} counts | {status_str}")

    t_lhs_imp_total = time.perf_counter() - t_lhs_imp_start
    print(f"Implicit LHS completed in {t_lhs_imp_total:.2f} s ({t_lhs_imp_total/n_samples:.2f} s/solve).")

    # Run Explicit LHS
    print(f"\n--- Running Explicit ('chord_span') LHS ({n_samples} samples) ---")
    t_lhs_exp_start = time.perf_counter()
    for i, sample in enumerate(lhs_samples_exp):
        x0_s = {
            'chord_stretch_dv': sample[0],
            'span_stretch_dv': sample[1],
        }
        res = run_single_optimization(
            sim=sim_exp,
            design_variables=dvs_exp,
            outputs_dict=outs_exp,
            formulation='chord_span',
            x0_dict=x0_s,
            problem_name=f'lhs_exp_{i}'
        )
        results_lhs_exp.append(res)
        status_str = "SUCCESS" if res['success'] else f"FAILED({res['status']})"
        if (i + 1) % 10 == 0 or (i + 1) == n_samples or not res['success']:
            print(f"  [Explicit {i+1:3d}/{n_samples}] c_str0={sample[0]:5.2f} b_str0={sample[1]:5.2f} | Iters={res['iterations']:2d} Fev={res['nfev']:2d} Time={res['time']:5.2f}s | CDi={res['cdi_counts']:5.2f} counts | {status_str}")

    t_lhs_exp_total = time.perf_counter() - t_lhs_exp_start
    print(f"Explicit LHS completed in {t_lhs_exp_total:.2f} s ({t_lhs_exp_total/n_samples:.2f} s/solve).")

    # -------------------------------------------------------------------------
    # 4. STATISTICAL ANALYSIS & TELEMETRY SUMMARY
    # -------------------------------------------------------------------------
    iters_imp = np.array([r['iterations'] for r in results_lhs_imp])
    iters_exp = np.array([r['iterations'] for r in results_lhs_exp])

    fev_imp = np.array([r['nfev'] for r in results_lhs_imp])
    fev_exp = np.array([r['nfev'] for r in results_lhs_exp])

    time_imp = np.array([r['time'] for r in results_lhs_imp])
    time_exp = np.array([r['time'] for r in results_lhs_exp])

    conv_imp = np.array([r['success'] for r in results_lhs_imp])
    conv_exp = np.array([r['success'] for r in results_lhs_exp])

    cdi_imp = np.array([r['cdi_counts'] for r in results_lhs_imp if r['success']])
    cdi_exp = np.array([r['cdi_counts'] for r in results_lhs_exp if r['success']])

    print("\n" + "=" * 80)
    print("STATISTICAL BENCHMARK SUMMARY OVER LHS SAMPLES")
    print("=" * 80)
    print(f"{'Metric':<30} | {'Implicit (ar_area)':<20} | {'Explicit (chord_span)':<20} | {'Comparison (Exp / Imp)':<20}")
    print("-" * 96)
    print(f"{'Total Samples':<30} | {n_samples:<20d} | {n_samples:<20d} | 1.00x")
    print(f"{'Converged Solves':<30} | {np.sum(conv_imp):d} / {n_samples:d} ({np.mean(conv_imp)*100:.1f}%) | {np.sum(conv_exp):d} / {n_samples:d} ({np.mean(conv_exp)*100:.1f}%) | -")
    print("-" * 96)
    print(f"{'Mean Major Iterations':<30} | {np.mean(iters_imp):<20.2f} | {np.mean(iters_exp):<20.2f} | {np.mean(iters_exp)/np.mean(iters_imp):.2f}x")
    print(f"{'Median Major Iterations':<30} | {np.median(iters_imp):<20.1f} | {np.median(iters_exp):<20.1f} | {np.median(iters_exp)/np.median(iters_imp):.2f}x")
    print(f"{'Min / Max Iterations':<30} | {np.min(iters_imp)} / {np.max(iters_imp):<16} | {np.min(iters_exp)} / {np.max(iters_exp):<16} | -")
    print(f"{'Iteration Std Dev':<30} | {np.std(iters_imp):<20.2f} | {np.std(iters_exp):<20.2f} | -")
    print("-" * 96)
    print(f"{'Mean Function Evaluations':<30} | {np.mean(fev_imp):<20.2f} | {np.mean(fev_exp):<20.2f} | {np.mean(fev_exp)/np.mean(fev_imp):.2f}x")
    print(f"{'Median Function Evaluations':<30} | {np.median(fev_imp):<20.1f} | {np.median(fev_exp):<20.1f} | {np.median(fev_exp)/np.median(fev_imp):.2f}x")
    print(f"{'Min / Max Function Evals':<30} | {np.min(fev_imp)} / {np.max(fev_imp):<16} | {np.min(fev_exp)} / {np.max(fev_exp):<16} | -")
    print("-" * 96)
    print(f"{'Mean Solve Time (s)':<30} | {np.mean(time_imp):<20.2f} | {np.mean(time_exp):<20.2f} | {np.mean(time_exp)/np.mean(time_imp):.2f}x")
    print(f"{'Median Solve Time (s)':<30} | {np.median(time_imp):<20.2f} | {np.median(time_exp):<20.2f} | {np.median(time_exp)/np.median(time_imp):.2f}x")
    print(f"{'Total Benchmark Time (s)':<30} | {t_lhs_imp_total:<20.2f} | {t_lhs_exp_total:<20.2f} | {t_lhs_exp_total/t_lhs_imp_total:.2f}x")
    print("-" * 96)
    if len(cdi_imp) > 0 and len(cdi_exp) > 0:
        print(f"{'Mean Optimal CDi (counts)':<30} | {np.mean(cdi_imp):<20.2f} | {np.mean(cdi_exp):<20.2f} | -")
    print("=" * 96)

    # -------------------------------------------------------------------------
    # 5. SAVE DATA FILES
    # -------------------------------------------------------------------------
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
            'function_evals': {
                'implicit_mean': float(np.mean(fev_imp)),
                'explicit_mean': float(np.mean(fev_exp)),
                'implicit_median': float(np.median(fev_imp)),
                'explicit_median': float(np.median(fev_exp)),
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

    pkl_path = os.path.join(script_dir, 'benchmark_results_implicit_vs_explicit.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results_payload, f)
    print(f"\nRaw benchmark data saved to: {pkl_path}")

    json_path = os.path.join(script_dir, 'benchmark_summary.json')
    with open(json_path, 'w') as f:
        json.dump(results_payload['summary'], f, indent=4)
    print(f"Summary JSON saved to:       {json_path}")

    # -------------------------------------------------------------------------
    # 6. GENERATE VISUALIZATION FIGURE (JOURNAL READY, NO TITLES/TAGS)
    # -------------------------------------------------------------------------
    print("\nGenerating publication-quality comparison figure...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0), dpi=300)

    c_imp = '#1f4e79'  # Classic AIAA Navy
    c_exp = '#c0392b'  # AIAA Crimson

    # Panel 1: Major Iterations Distribution (Grouped Discrete Histogram)
    all_iters = np.arange(2, 8)
    counts1 = [np.sum(iters_imp == k) / n_samples * 100 for k in all_iters]
    counts2 = [np.sum(iters_exp == k) / n_samples * 100 for k in all_iters]

    w = 0.38
    x_indices = np.arange(len(all_iters))
    b1 = ax1.bar(x_indices - w/2, counts1, width=w, color=c_imp,
                 label='Implicit (ar_area)', alpha=0.9, edgecolor='black', linewidth=0.8)
    b2 = ax1.bar(x_indices + w/2, counts2, width=w, color=c_exp,
                 label='Explicit (chord_span)', alpha=0.9, edgecolor='black', linewidth=0.8)

    # Baseline markers on Panel A
    base_it1 = res_base_imp['iterations']
    base_it2 = res_base_exp['iterations']
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

    # Panel 2: Solve Time Distribution (Box-and-Whisker + Scatter + Baseline Marker)
    fc_imp = '#d0e1fd'
    fc_exp = '#f8d7da'

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
    ax2.set_xticklabels(['Implicit (ar_area)', 'Explicit (chord_span)'], fontsize=10.5)
    ax2.set_ylabel('Optimization solve time [s]', fontsize=11)
    ax2.set_ylim(0, max(np.max(time_exp), np.max(time_imp)) * 1.25)
    ax2.grid(axis='y', linestyle='--', alpha=0.35, color='#cccccc')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title('Optimization Solve Time', fontsize=11, fontweight='bold', pad=10)
    ax2.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#cccccc', framealpha=0.95, fontsize=10)

    plt.tight_layout()
    fig_path = os.path.join(script_dir, 'benchmark_comparison_implicit_vs_explicit.png')
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Comparison figure saved to: {fig_path}")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == '__main__':
    samples = 100
    if len(sys.argv) > 1:
        try:
            samples = int(sys.argv[1])
        except ValueError:
            pass
    run_benchmark(n_samples=samples)
