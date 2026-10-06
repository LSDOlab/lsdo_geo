"""
Benchmarking Script: MDF vs. SAND Implicit Geometry Parameterization
=====================================================================
Experimental Platform: ex_rectangular_wing_aero_shape_optimization.py

Isolated Parameterization Benchmark:
- Pitch angle fixed at 5.0 degrees (not a design variable)
- CL constraint removed

Compares two implicit-style MDO architectures:
1. MDF (Multidisciplinary Feasible) Formulation ('mdf' / 'ar_area'):
   - Parameterization states (chord stretch, span stretch) are solved internally
     at every function evaluation via ParameterizationSolver (Newton nonlinear solver).
   - Geometric variables: Aspect Ratio (AR) and Planform Area (S = 10.0 m^2).
   - Optimizer design variables: aspect_ratio (1 DV, bounds [2.0, 15.0]).
   - Optimizer constraints: None (0 constraints, area and AR matching handled internally).

2. SAND (Simultaneous Analysis and Design) Formulation ('sand'):
   - Bypasses the nested ParameterizationSolver; promotes parameterization states
     to optimizer design variables and enforces governing geometry equations via constraints.
   - Optimizer design variables:
       * aspect_ratio (bounds [2.0, 15.0])
       * chord_stretch_state (bounds [-0.85, 4.0])
       * span_stretch_state (bounds [-4.5, 20.0])
       (3 DVs total; area is constant 10.0 m^2 so not a design variable).
   - Optimizer constraints:
       * planform_area_geom == 10.0 (equality constraint)
       * aspect_ratio_geom - aspect_ratio == 0.0 (equality constraint)
       (2 equality constraints).

Metrics tracked:
- Number of major optimization iterations
- Number of function evaluations (nfev)
- Number of gradient/derivative evaluations (ngev)
- Total optimization solve time (seconds)
- Convergence success rate and final optimality/feasibility
- Final aerodynamic metrics (CL, CDi, AR, S_ref)

Tests:
- Part 1: Baseline comparison starting from initial un-deformed geometry
- Part 2: Statistical comparison over 100 Latin Hypercube Samples (LHS)
          (1D LHS for MDF, 3D LHS for SAND)
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


def run_benchmark(n_samples=100, random_seed=42):
    """
    Runs the complete benchmark suite comparing MDF vs SAND formulations:
    - Baseline from initial geometry
    - Latin Hypercube Sampling across initial conditions
    """
    print("=" * 80)
    print("MDF VS. SAND GEOMETRY PARAMETERIZATION BENCHMARK")
    print("================================================================================")
    print(f"LHS Samples per Formulation: {n_samples}")
    print(f"Random Seed:                 {random_seed}")
    print(f"Objective Scaler:            1.0e3 (CDi)")
    print(f"Pitch:                       Fixed 5.0 deg (isolated geometric benchmark)")
    print(f"CL Constraint:               None (unconstrained lift)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. BUILD MODELS (ONE-TIME COMPILATION FOR EACH FORMULATION)
    # -------------------------------------------------------------------------
    print("\n[1/4] Building and compiling MDF ('mdf' / 'ar_area') Model...")
    t0 = time.perf_counter()
    sim_mdf, dvs_mdf, outs_mdf, geo_mdf, ffd_mdf = build_optimization_model(formulation='mdf', obj_scaler=1e3)
    # Warm-up / compile simulator once
    prob_warmup_mdf = modopt.CSDLAlphaProblem(problem_name='warmup_mdf', simulator=sim_mdf)
    opt_warmup_mdf = modopt.PySLSQP(prob_warmup_mdf, solver_options={'maxiter': 1, 'iprint': 0}, turn_off_outputs=True)
    opt_warmup_mdf.solve()
    t_compile_mdf = time.perf_counter() - t0
    print(f"  MDF model ready and compiled in {t_compile_mdf:.2f} s. DVs: {list(dvs_mdf.keys())}")

    print("\n[2/4] Building and compiling SAND ('sand') Model...")
    t0 = time.perf_counter()
    sim_sand, dvs_sand, outs_sand, geo_sand, ffd_sand = build_optimization_model(formulation='sand', obj_scaler=1e3)
    # Warm-up / compile simulator once
    prob_warmup_sand = modopt.CSDLAlphaProblem(problem_name='warmup_sand', simulator=sim_sand)
    opt_warmup_sand = modopt.PySLSQP(prob_warmup_sand, solver_options={'maxiter': 1, 'iprint': 0}, turn_off_outputs=True)
    opt_warmup_sand.solve()
    t_compile_sand = time.perf_counter() - t0
    print(f"  SAND model ready and compiled in {t_compile_sand:.2f} s. DVs: {list(dvs_sand.keys())}")

    # -------------------------------------------------------------------------
    # 2. BASELINE COMPARISON FROM INITIAL GEOMETRY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[3/4] BENCHMARK PART 1: BASELINE (STARTING FROM INITIAL GEOMETRY)")
    print("=" * 80)

    baseline_x0_mdf = {
        'aspect_ratio': 10.0,
    }
    baseline_x0_sand = {
        'aspect_ratio': 10.0,
        'chord_stretch_state': 0.0,
        'span_stretch_state': 0.0,
    }

    print("\nRunning Baseline MDF ('mdf')...")
    res_base_mdf = run_single_optimization(
        sim=sim_mdf,
        design_variables=dvs_mdf,
        outputs_dict=outs_mdf,
        formulation='mdf',
        x0_dict=baseline_x0_mdf,
        problem_name='baseline_mdf'
    )
    print(f"  MDF Baseline: Iters={res_base_mdf['iterations']}, Fev={res_base_mdf['nfev']}, Time={res_base_mdf['time']:.2f}s, CDi={res_base_mdf['cdi_counts']:.2f} counts, CL={res_base_mdf['cl']:.4f}")

    print("\nRunning Baseline SAND ('sand')...")
    res_base_sand = run_single_optimization(
        sim=sim_sand,
        design_variables=dvs_sand,
        outputs_dict=outs_sand,
        formulation='sand',
        x0_dict=baseline_x0_sand,
        problem_name='baseline_sand'
    )
    print(f"  SAND Baseline: Iters={res_base_sand['iterations']}, Fev={res_base_sand['nfev']}, Time={res_base_sand['time']:.2f}s, CDi={res_base_sand['cdi_counts']:.2f} counts, CL={res_base_sand['cl']:.4f}")

    # Baseline side-by-side summary table
    print("\n" + "-" * 75)
    print(f"{'Metric':<30} | {'MDF (Nested Solver)':<20} | {'SAND (Optimizer Solver)':<20}")
    print("-" * 75)
    print(f"{'Major Iterations':<30} | {res_base_mdf['iterations']:<20d} | {res_base_sand['iterations']:<20d}")
    print(f"{'Function Evaluations':<30} | {res_base_mdf['nfev']:<20d} | {res_base_sand['nfev']:<20d}")
    print(f"{'Solve Time (s)':<30} | {res_base_mdf['time']:<20.2f} | {res_base_sand['time']:<20.2f}")
    print(f"{'Optimal Aspect Ratio':<30} | {res_base_mdf['aspect_ratio']:<20.3f} | {res_base_sand['aspect_ratio']:<20.3f}")
    print(f"{'Optimal Planform Area (m^2)':<30} | {res_base_mdf['planform_area']:<20.3f} | {res_base_sand['planform_area']:<20.3f}")
    print(f"{'Optimal CDi (counts)':<30} | {res_base_mdf['cdi_counts']:<20.2f} | {res_base_sand['cdi_counts']:<20.2f}")
    print("-" * 75)

    # -------------------------------------------------------------------------
    # 3. LATIN HYPERCUBE SAMPLING (LHS) COMPARISON
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"[4/4] BENCHMARK PART 2: LATIN HYPERCUBE SAMPLING ({n_samples} SAMPLES EACH)")
    print("=" * 80)

    # Generate LHS for MDF: 1D space [aspect_ratio]
    # AR bounds: [5.0, 13.0]
    lhs_bounds_mdf_low = np.array([5.0])
    lhs_bounds_mdf_high = np.array([13.0])
    sampler_mdf = qmc.LatinHypercube(d=1, seed=random_seed)
    unit_samples_mdf = sampler_mdf.random(n=n_samples)
    lhs_samples_mdf = qmc.scale(unit_samples_mdf, lhs_bounds_mdf_low, lhs_bounds_mdf_high)

    # Generate LHS for SAND: 3D space [aspect_ratio, chord_stretch_state, span_stretch_state]
    # AR: [5.0, 13.0], chord_stretch: [-0.25, 0.35], span_stretch: [-1.5, 1.5]
    lhs_bounds_sand_low = np.array([5.0, -0.25, -1.5])
    lhs_bounds_sand_high = np.array([13.0, 0.35, 1.5])
    sampler_sand = qmc.LatinHypercube(d=3, seed=random_seed + 100)
    unit_samples_sand = sampler_sand.random(n=n_samples)
    lhs_samples_sand = qmc.scale(unit_samples_sand, lhs_bounds_sand_low, lhs_bounds_sand_high)

    results_lhs_mdf = []
    results_lhs_sand = []

    # Run MDF LHS
    print(f"\n--- Running MDF ('mdf') LHS ({n_samples} samples) ---")
    t_lhs_mdf_start = time.perf_counter()
    for i, sample in enumerate(lhs_samples_mdf):
        x0_s = {
            'aspect_ratio': sample[0],
        }
        res = run_single_optimization(
            sim=sim_mdf,
            design_variables=dvs_mdf,
            outputs_dict=outs_mdf,
            formulation='mdf',
            x0_dict=x0_s,
            problem_name=f'lhs_mdf_{i}'
        )
        results_lhs_mdf.append(res)
        status_str = "SUCCESS" if res['success'] else f"FAILED({res['status']})"
        if (i + 1) % 10 == 0 or (i + 1) == n_samples or not res['success']:
            print(f"  [MDF  {i+1:3d}/{n_samples}] AR0={sample[0]:5.2f} | Iters={res['iterations']:2d} Fev={res['nfev']:2d} Time={res['time']:5.2f}s | CDi={res['cdi_counts']:5.2f} counts | {status_str}")

    t_lhs_mdf_total = time.perf_counter() - t_lhs_mdf_start
    print(f"MDF LHS completed in {t_lhs_mdf_total:.2f} s ({t_lhs_mdf_total/n_samples:.2f} s/solve).")

    # Run SAND LHS
    print(f"\n--- Running SAND ('sand') LHS ({n_samples} samples) ---")
    t_lhs_sand_start = time.perf_counter()
    for i, sample in enumerate(lhs_samples_sand):
        x0_s = {
            'aspect_ratio': sample[0],
            'chord_stretch_state': sample[1],
            'span_stretch_state': sample[2],
        }
        res = run_single_optimization(
            sim=sim_sand,
            design_variables=dvs_sand,
            outputs_dict=outs_sand,
            formulation='sand',
            x0_dict=x0_s,
            problem_name=f'lhs_sand_{i}'
        )
        results_lhs_sand.append(res)
        status_str = "SUCCESS" if res['success'] else f"FAILED({res['status']})"
        if (i + 1) % 10 == 0 or (i + 1) == n_samples or not res['success']:
            print(f"  [SAND {i+1:3d}/{n_samples}] AR0={sample[0]:5.2f} c_str0={sample[1]:5.2f} b_str0={sample[2]:5.2f} | Iters={res['iterations']:2d} Fev={res['nfev']:2d} Time={res['time']:5.2f}s | CDi={res['cdi_counts']:5.2f} counts | {status_str}")

    t_lhs_sand_total = time.perf_counter() - t_lhs_sand_start
    print(f"SAND LHS completed in {t_lhs_sand_total:.2f} s ({t_lhs_sand_total/n_samples:.2f} s/solve).")

    # -------------------------------------------------------------------------
    # 4. STATISTICAL ANALYSIS & TELEMETRY SUMMARY
    # -------------------------------------------------------------------------
    iters_mdf = np.array([r['iterations'] for r in results_lhs_mdf])
    iters_sand = np.array([r['iterations'] for r in results_lhs_sand])

    fev_mdf = np.array([r['nfev'] for r in results_lhs_mdf])
    fev_sand = np.array([r['nfev'] for r in results_lhs_sand])

    time_mdf = np.array([r['time'] for r in results_lhs_mdf])
    time_sand = np.array([r['time'] for r in results_lhs_sand])

    conv_mdf = np.array([r['success'] for r in results_lhs_mdf])
    conv_sand = np.array([r['success'] for r in results_lhs_sand])

    cdi_mdf = np.array([r['cdi_counts'] for r in results_lhs_mdf if r['success']])
    cdi_sand = np.array([r['cdi_counts'] for r in results_lhs_sand if r['success']])

    print("\n" + "=" * 80)
    print("STATISTICAL BENCHMARK SUMMARY OVER LHS SAMPLES")
    print("=" * 80)
    print(f"{'Metric':<30} | {'MDF (Nested Solver)':<22} | {'SAND (Optimizer Solver)':<22} | {'Ratio (SAND / MDF)':<20}")
    print("-" * 96)
    print(f"{'Total Samples':<30} | {n_samples:<22d} | {n_samples:<22d} | 1.00x")
    print(f"{'Converged Solves':<30} | {np.sum(conv_mdf):d} / {n_samples:d} ({np.mean(conv_mdf)*100:.1f}%) | {np.sum(conv_sand):d} / {n_samples:d} ({np.mean(conv_sand)*100:.1f}%) | -")
    print("-" * 96)
    print(f"{'Mean Major Iterations':<30} | {np.mean(iters_mdf):<22.2f} | {np.mean(iters_sand):<22.2f} | {np.mean(iters_sand)/np.mean(iters_mdf):.2f}x")
    print(f"{'Median Major Iterations':<30} | {np.median(iters_mdf):<22.1f} | {np.median(iters_sand):<22.1f} | {np.median(iters_sand)/np.median(iters_mdf):.2f}x")
    print(f"{'Min / Max Iterations':<30} | {np.min(iters_mdf)} / {np.max(iters_mdf):<18} | {np.min(iters_sand)} / {np.max(iters_sand):<18} | -")
    print(f"{'Iteration Std Dev':<30} | {np.std(iters_mdf):<22.2f} | {np.std(iters_sand):<22.2f} | -")
    print("-" * 96)
    print(f"{'Mean Function Evaluations':<30} | {np.mean(fev_mdf):<22.2f} | {np.mean(fev_sand):<22.2f} | {np.mean(fev_sand)/np.mean(fev_mdf):.2f}x")
    print(f"{'Median Function Evaluations':<30} | {np.median(fev_mdf):<22.1f} | {np.median(fev_sand):<22.1f} | {np.median(fev_sand)/np.median(fev_mdf):.2f}x")
    print(f"{'Min / Max Function Evals':<30} | {np.min(fev_mdf)} / {np.max(fev_mdf):<18} | {np.min(fev_sand)} / {np.max(fev_sand):<18} | -")
    print("-" * 96)
    print(f"{'Mean Solve Time (s)':<30} | {np.mean(time_mdf):<22.2f} | {np.mean(time_sand):<22.2f} | {np.mean(time_sand)/np.mean(time_mdf):.2f}x")
    print(f"{'Median Solve Time (s)':<30} | {np.median(time_mdf):<22.2f} | {np.median(time_sand):<22.2f} | {np.median(time_sand)/np.median(time_mdf):.2f}x")
    print(f"{'Total Benchmark Time (s)':<30} | {t_lhs_mdf_total:<22.2f} | {t_lhs_sand_total:<22.2f} | {t_lhs_sand_total/t_lhs_mdf_total:.2f}x")
    print("-" * 96)
    if len(cdi_mdf) > 0 and len(cdi_sand) > 0:
        print(f"{'Mean Optimal CDi (counts)':<30} | {np.mean(cdi_mdf):<22.2f} | {np.mean(cdi_sand):<22.2f} | -")
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
            'function_evals': {
                'mdf_mean': float(np.mean(fev_mdf)),
                'sand_mean': float(np.mean(fev_sand)),
                'mdf_median': float(np.median(fev_mdf)),
                'sand_median': float(np.median(fev_sand)),
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

    pkl_path = os.path.join(script_dir, 'benchmark_results_mdf_vs_sand.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results_payload, f)
    print(f"\nRaw benchmark data saved to: {pkl_path}")

    json_path = os.path.join(script_dir, 'benchmark_summary_mdf_vs_sand.json')
    with open(json_path, 'w') as f:
        json.dump(results_payload['summary'], f, indent=4)
    print(f"Summary JSON saved to:       {json_path}")

    # -------------------------------------------------------------------------
    # 6. GENERATE VISUALIZATION FIGURE (JOURNAL READY, NO TITLES/TAGS)
    # -------------------------------------------------------------------------
    print("\nGenerating publication-quality comparison figure...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0), dpi=300)

    c_mdf = '#1f4e79'   # Classic AIAA Navy
    c_sand = '#16a085'  # Teal/Emerald for SAND

    # Panel 1: Major Iterations Distribution (Grouped Discrete Histogram)
    all_iters = np.arange(2, 8)
    counts1 = [np.sum(iters_mdf == k) / n_samples * 100 for k in all_iters]
    counts2 = [np.sum(iters_sand == k) / n_samples * 100 for k in all_iters]

    w = 0.38
    x_indices = np.arange(len(all_iters))
    b1 = ax1.bar(x_indices - w/2, counts1, width=w, color=c_mdf,
                 label='MDF', alpha=0.9, edgecolor='black', linewidth=0.8)
    b2 = ax1.bar(x_indices + w/2, counts2, width=w, color=c_sand,
                 label='SAND', alpha=0.9, edgecolor='black', linewidth=0.8)

    # Baseline markers on Panel A
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

    # Panel 2: Solve Time Distribution (Box-and-Whisker + Scatter + Baseline Marker)
    fc_mdf = '#d0e1fd'
    fc_sand = '#d1f2eb'

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
    fig_path = os.path.join(script_dir, 'benchmark_comparison_mdf_vs_sand.png')
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Comparison figure saved to: {fig_path}")

    print("\n" + "=" * 80)
    print("MDF VS. SAND BENCHMARK COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == '__main__':
    samples = 100
    if len(sys.argv) > 1:
        try:
            samples = int(sys.argv[1])
        except ValueError:
            pass
    run_benchmark(n_samples=samples)
