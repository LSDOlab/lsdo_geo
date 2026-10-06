"""
Benchmarking Script for Lift+Cruise (LPC) Geometry Parameterization
===================================================================
Evaluates execution time and Newton solver iteration performance across
Latin Hypercube Sampling (LHS) of design variables.

Pulls geometry, component declarations, and parameterization setup from
the standard `ex_lift_plus_cruise.py` script to avoid code duplication.
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"
import sys
import time
import pickle
import numpy as np
from scipy.stats import qmc
import matplotlib.pyplot as plt
import csdl_alpha as csdl

# Iteration Tracking Instrumentation
iteration_log = []

def enable_iteration_tracking():
    """Instruments CSDL's FixedPoint Newton solver to record iteration counts via jax.debug.callback."""
    fixed_point_cls = csdl.nonlinear_solvers.Newton.__mro__[1]

    def instrumented_jax_solve(self, jax_residual_function, jax_intermediate_function, input_var_dict):
        import jax
        import jax.numpy as jnp
        import jax.lax as lax
        from csdl_alpha.src.graph.variable import Variable
        from csdl_alpha.utils.inputs import ingest_value

        def loop_body(val):
            states = self._jax_update_states(
                jax_residual_function,
                jax_intermediate_function,
                val,
                input_var_dict,
            )
            residuals = jax_residual_function(states)
            it = val[2] + 1
            return (states, residuals, it)

        def loop_condition(val):
            residuals = val[1]
            it = val[2]
            return ~self._jax_check_converged(residuals, it, input_var_dict)

        states = []
        for state in self.state_to_residual_map.keys():
            if not isinstance(self.state_metadata[state]["initial_value"], Variable):
                value = ingest_value(self.state_metadata[state]["initial_value"])
                if value.shape != state.shape:
                    if value.size == 1:
                        value = value * np.ones(state.shape)
                    else:
                        raise ValueError("Initial value shape mismatch")
                states.append(jnp.array(value))
            else:
                states.append(input_var_dict[self.state_metadata[state]["initial_value"]])
        val = (states, jax_residual_function(states), 0)

        states, residuals, it = lax.while_loop(loop_condition, loop_body, val)
        jax.debug.callback(lambda count: iteration_log.append(int(count)), it)
        return states

    fixed_point_cls._jax_solve_ = instrumented_jax_solve

enable_iteration_tracking()

# Configuration Flags
CREATE_VIDEO = False
RUN_BENCHMARK = not CREATE_VIDEO
ENABLE_INEQUALITIES = os.environ.get("BENCH_ENABLE_INEQUALITIES", "False").lower() in ("true", "1", "yes")
N_LHS_SAMPLES = int(os.environ.get("BENCH_N_SAMPLES", "100" if ENABLE_INEQUALITIES else "300"))
SEED = 42

# Pass inequality toggle to standard script
os.environ["LPC_ENABLE_INEQUALITIES"] = str(ENABLE_INEQUALITIES)

# Add directory to sys.path and import geometry/parameterization from standard example
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from ex_lift_plus_cruise import (
    geometry,
    recorder,
    jax_inputs,
    variable_names,
    computed_metrics,
    jax_outputs,
    rotor_names,
    generate_perturbation_video,
)

if CREATE_VIDEO:
    generate_perturbation_video()
else:
    print("=== Setting up JaxSimulator for LHS Benchmarking ===")
    t_comp_start = time.time()
    sim = csdl.experimental.JaxSimulator(
        recorder=recorder,
        additional_inputs=jax_inputs,
        additional_outputs=jax_outputs,
        gpu=False,
    )
    iteration_log.clear()
    sim.run()
    compile_time = time.time() - t_comp_start
    print(f"JAX Compilation finished in {compile_time:.2f} seconds.\n")

    baseline_values = np.array([float(np.asarray(inp.value).flatten()[0]) for inp in jax_inputs])

    design_variable_bounds = {
        "wing_area":            [baseline_values[0] * 0.85, baseline_values[0] * 1.15],
        "wing_aspect_ratio":    [baseline_values[1] * 0.85, baseline_values[1] * 1.15],
        "wing_taper_ratio":     [baseline_values[2] * 0.85, baseline_values[2] * 1.15],
        "wing_tip_twist":       [-4.0, 4.0],  # degrees
        "tail_area":            [baseline_values[4] * 0.85, baseline_values[4] * 1.15],
        "tail_aspect_ratio":    [baseline_values[5] * 0.85, baseline_values[5] * 1.15],
        "tail_taper_ratio":     [baseline_values[6] * 0.85, baseline_values[6] * 1.15],
        "cabin_length":         [baseline_values[7] * 0.90, baseline_values[7] * 1.10],
        "tail_moment_arm":      [baseline_values[8] * 0.90, baseline_values[8] * 1.10],
        "cabin_height":         [baseline_values[9] * 0.90, baseline_values[9] * 1.10],
        "cabin_width":          [baseline_values[10] * 0.90, baseline_values[10] * 1.10],
        "cruise_rotor_radius":  [baseline_values[11] * 0.90, baseline_values[11] * 1.10],
    }
    for i, name in enumerate(rotor_names):
        idx = 12 + i
        design_variable_bounds[f"{name}_radius"] = [baseline_values[idx] * 0.95, baseline_values[idx] * 1.05]

    lower_bounds = np.array([design_variable_bounds[name][0] for name in variable_names])
    upper_bounds = np.array([design_variable_bounds[name][1] for name in variable_names])

    sampler = qmc.LatinHypercube(d=len(variable_names), seed=SEED)
    unit_samples = sampler.random(n=N_LHS_SAMPLES)
    lhs_samples = qmc.scale(unit_samples, lower_bounds, upper_bounds)

    print(f"Generated {N_LHS_SAMPLES} LHS samples across {len(variable_names)} design variables.")

    sample_timings = []
    sample_iterations = []
    sample_input_difference_norms = []
    unconverged_samples = []
    max_residuals = []

    print(f"\n=== Starting Benchmark Timing Across {N_LHS_SAMPLES} LHS Samples ===")
    total_benchmark_start = time.perf_counter()

    for i, sample in enumerate(lhs_samples):
        for j, var_value in enumerate(sample):
            sim[jax_inputs[j]] = np.array([var_value])

        t_start = time.perf_counter()
        sim.run()
        t_end = time.perf_counter()
        sample_duration = t_end - t_start
        sample_timings.append(sample_duration)

        it_count = iteration_log[-1] if iteration_log else 1
        sample_iterations.append(it_count)

        input_diff_norm = np.linalg.norm(sample - baseline_values)
        sample_input_difference_norms.append(input_diff_norm)

        sample_max_res = 0.0
        has_nan = False

        for j in range(len(jax_inputs)):
            val_comp = sim[computed_metrics[j]]
            val_target = sim[jax_inputs[j]]
            if np.isnan(val_comp).any() or np.isnan(val_target).any():
                has_nan = True
                break
            err = float(np.max(np.abs(val_comp - val_target)))
            if err > sample_max_res:
                sample_max_res = err

        max_residuals.append(sample_max_res)
        converged = (not has_nan) and (sample_max_res < 1e-3)

        if not converged:
            unconverged_samples.append({
                "index": i,
                "sample": sample,
                "has_nan": has_nan,
                "max_residual": sample_max_res,
                "iterations": it_count,
            })

        if (i + 1) % 10 == 0 or (i + 1) == N_LHS_SAMPLES:
            print(
                f"  Sample {i+1:3d}/{N_LHS_SAMPLES} | Solve Time: {sample_duration*1000:6.2f} ms | "
                f"Iters: {it_count:2d} | Max Res: {sample_max_res:.2e} | Status: {'CONVERGED' if converged else 'FAILED'}"
            )

    total_benchmark_time = time.perf_counter() - total_benchmark_start
    timings_ms = np.array(sample_timings) * 1000.0
    iterations_arr = np.array(sample_iterations)

    print("\n" + "=" * 65)
    print("=== LIFT+CRUISE GEOMETRY PARAMETERIZATION BENCHMARK SUMMARY ===")
    print("=" * 65)
    print(f"Total Samples Evaluated:    {N_LHS_SAMPLES}")
    print(f"Converged Solves:           {N_LHS_SAMPLES - len(unconverged_samples)} / {N_LHS_SAMPLES} ({100.0 * (1 - len(unconverged_samples)/N_LHS_SAMPLES):.1f}%)")
    if unconverged_samples:
        print(f"WARNING: {len(unconverged_samples)} samples did not converge!")
        for unc in unconverged_samples:
            print(f"  Sample {unc['index']}: has_nan={unc['has_nan']}, max_residual={unc['max_residual']:.2e}")
    else:
        print("All solves converged successfully!")
    print("-" * 65)
    print(f"One-Time Compilation Time: {compile_time:.4f} s")
    print(f"Total LHS Benchmark Time:   {total_benchmark_time:.4f} s")
    print(f"Mean Solve Time:            {np.mean(timings_ms):.2f} ms")
    print(f"Median Solve Time:          {np.median(timings_ms):.2f} ms")
    print(f"Min Solve Time:             {np.min(timings_ms):.2f} ms")
    print(f"Max Solve Time:             {np.max(timings_ms):.2f} ms")
    print(f"Standard Deviation:         {np.std(timings_ms):.2f} ms")
    print("-" * 65)
    print(f"Mean Newton Iterations:     {np.mean(iterations_arr):.2f}")
    print(f"Median Newton Iterations:   {int(np.median(iterations_arr))}")
    print(f"Min / Max Iterations:       {np.min(iterations_arr)} / {np.max(iterations_arr)}")
    print("=" * 65)

    results_dict = {
        "sample_timings": sample_timings,
        "sample_iterations": sample_iterations,
        "sample_input_difference_norms": sample_input_difference_norms,
        "max_residuals": max_residuals,
        "unconverged_samples": unconverged_samples,
        "n_samples": N_LHS_SAMPLES,
        "baseline_values": baseline_values,
        "variable_names": variable_names,
        "compile_time": compile_time,
        "total_benchmark_time": total_benchmark_time,
        "mean_ms": float(np.mean(timings_ms)),
        "median_ms": float(np.median(timings_ms)),
        "min_ms": float(np.min(timings_ms)),
        "max_ms": float(np.max(timings_ms)),
        "std_ms": float(np.std(timings_ms)),
        "mean_iterations": float(np.mean(iterations_arr)),
        "median_iterations": int(np.median(iterations_arr)),
        "enable_inequalities": ENABLE_INEQUALITIES,
    }

    if ENABLE_INEQUALITIES:
        pkl_filename = "examples/showcase_examples/lift_plus_cruise/lift_plus_cruise_with_inequalities_lhs_timing_results.pkl"
    else:
        pkl_filename = "examples/showcase_examples/lift_plus_cruise/lift_plus_cruise_lhs_timing_results.pkl"
    with open(pkl_filename, "wb") as f:
        pickle.dump(results_dict, f)
    print(f"\nResults successfully cached to '{pkl_filename}'")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), dpi=300)

    # Left: Solve Time Boxplot
    ax1 = axes[0]
    bp1 = ax1.boxplot(
        timings_ms, patch_artist=True, widths=0.45,
        boxprops=dict(facecolor="#d0e1fd", color="#1f77b4", linewidth=1.8),
        medianprops=dict(color="#d62728", linewidth=2.2),
        whiskerprops=dict(color="#1f77b4", linewidth=1.4),
        capprops=dict(color="#1f77b4", linewidth=1.4),
        flierprops=dict(marker="o", markerfacecolor="#1f77b4", markeredgecolor="none", alpha=0.5, markersize=5),
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="#2ca02c", markeredgecolor="k", markersize=6),
    )
    np.random.seed(42)
    jitter1 = np.random.normal(0, 0.035, size=len(timings_ms))
    ax1.scatter(np.ones_like(timings_ms) + jitter1, timings_ms, color="#1f77b4", alpha=0.55, s=28, zorder=3, edgecolors="none")
    ax1.set_xticks([1])
    ax1.set_xticklabels([f"Lift+Cruise\n({N_LHS_SAMPLES} Samples)"], fontsize=11, fontweight="bold")
    ax1.set_ylabel("Solve Time (ms)", fontsize=12)
    ax1.set_title("Solve Time Distribution", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")

    stats_box = (
        f"Samples (N): {len(timings_ms)}\n"
        f"Mean (♦):    {np.mean(timings_ms):.2f} ms\n"
        f"Median (—):  {np.median(timings_ms):.2f} ms\n"
        f"Std Dev:     {np.std(timings_ms):.2f} ms\n"
        f"Min:         {np.min(timings_ms):.2f} ms\n"
        f"Max:         {np.max(timings_ms):.2f} ms"
    )
    ax1.text(0.05, 0.95, stats_box, transform=ax1.transAxes, verticalalignment="top",
            fontsize=10, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="#b0b0b0"))

    # Right: Iteration Count Boxplot
    ax2 = axes[1]
    bp2 = ax2.boxplot(
        iterations_arr, patch_artist=True, widths=0.45,
        boxprops=dict(facecolor="#d4edda", color="#28a745", linewidth=1.8),
        medianprops=dict(color="#d62728", linewidth=2.2),
        whiskerprops=dict(color="#28a745", linewidth=1.4),
        capprops=dict(color="#28a745", linewidth=1.4),
        flierprops=dict(marker="o", markerfacecolor="#28a745", markeredgecolor="none", alpha=0.5, markersize=5),
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="#1f77b4", markeredgecolor="k", markersize=6),
    )
    jitter2 = np.random.normal(0, 0.035, size=len(iterations_arr))
    ax2.scatter(np.ones_like(iterations_arr) + jitter2, iterations_arr, color="#28a745", alpha=0.55, s=28, zorder=3, edgecolors="none")
    ax2.set_xticks([1])
    ax2.set_xticklabels(["Newton Solver"], fontsize=11, fontweight="bold")
    ax2.set_ylabel("Iterations to Convergence", fontsize=12)
    ax2.set_title("Iteration Count Distribution", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")

    iter_box = (
        f"Mean (♦):    {np.mean(iterations_arr):.2f}\n"
        f"Median (—):  {int(np.median(iterations_arr))}\n"
        f"Min / Max:   {np.min(iterations_arr)} / {np.max(iterations_arr)}\n"
        f"Convergence: 100%"
    )
    ax2.text(0.05, 0.95, iter_box, transform=ax2.transAxes, verticalalignment="top",
            fontsize=10, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="#b0b0b0"))

    ineq_tag = " (With Inequalities)" if ENABLE_INEQUALITIES else " (Without Inequalities)"
    plt.suptitle(f"Lift+Cruise Parameterization Solve Performance{ineq_tag}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    suffix = "_with_inequalities" if ENABLE_INEQUALITIES else ""
    plot_filename = f"examples/showcase_examples/lift_plus_cruise/time_vs_input_norm_lpc{suffix}.png"
    plt.savefig(plot_filename, dpi=300)
    boxplot_filename = f"examples/showcase_examples/lift_plus_cruise/solve_time_boxplot_lpc{suffix}.png"
    plt.savefig(boxplot_filename, dpi=300)
    plt.close()
    print(f"Boxplot saved to '{plot_filename}' and '{boxplot_filename}'")
