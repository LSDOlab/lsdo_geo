"""
Benchmarking Script for Truss-Braced Wing (TBW) Geometry Parameterization
=========================================================================
Evaluates execution time and Newton solver iteration performance across
Latin Hypercube Sampling (LHS) of design variables.

Pulls geometry, component declarations, and parameterization setup from
the standard `ex_truss_braced_wing.py` script to avoid code duplication.
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

# Add directory to sys.path and import geometry/parameterization from standard example
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from ex_truss_braced_wing import (
    geo,
    recorder,
    jax_inputs,
    variable_names,
    target_comp_pairs,
    tracking_outputs,
    conn_fuse_strut,
    target_conn_fuse_strut,
    conn_wing_strut,
    target_conn_wing_strut,
    conn_strut_jury,
    target_conn_strut_jury,
    conn_fuse_tail,
    target_conn_fuse_tail,
)

print("=== Setting up JaxSimulator for LHS Benchmarking ===")
jax_outputs = [f.coefficients for f in geo.functions.values()] + tracking_outputs

print("Compiling JaxSimulator (one-time compilation)...")
sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=jax_inputs,
    additional_outputs=jax_outputs,
    gpu=False,
)

# Compile the model once
t_compile_start = time.perf_counter()
sim.run()
t_compile_end = time.perf_counter()
compile_time = t_compile_end - t_compile_start
print(f"JaxSimulator compiled successfully in {compile_time:.4f} seconds.\n")

# Baseline values for each design variable
baseline_values = np.array([float(np.squeeze(inp.value)) for inp in jax_inputs])
print("=== Baseline Design Variable Values ===")
for name, val in zip(variable_names, baseline_values):
    print(f"  {name:25s}: {val:10.4f}")
print()

# Reasonable design space bounds (±15% for wing and tail, [-10%, +15%] for cabin & moment arm)
design_variable_bounds = {
    'wing_area':            [baseline_values[0] * 0.85, baseline_values[0] * 1.15],
    'wing_aspect_ratio':    [baseline_values[1] * 0.85, baseline_values[1] * 1.15],
    'wing_mid_taper_ratio': [baseline_values[2] * 0.85, baseline_values[2] * 1.15],
    'wing_tip_taper_ratio': [baseline_values[3] * 0.85, baseline_values[3] * 1.15],
    'wing_inboard_sweep':   [baseline_values[4] * 0.85, baseline_values[4] * 1.15],
    'wing_outboard_sweep':  [baseline_values[5] * 0.85, baseline_values[5] * 1.15],
    'cabin_length':         [baseline_values[6] * 0.90, baseline_values[6] * 1.15],
    'tail_moment_arm':      [baseline_values[7] * 0.90, baseline_values[7] * 1.15],
    'fuselage_radius':      [baseline_values[8] * 0.85, baseline_values[8] * 1.15],
    'tail_area':            [baseline_values[9] * 0.85, baseline_values[9] * 1.15],
    'tail_aspect_ratio':    [baseline_values[10] * 0.85, baseline_values[10] * 1.15],
    'tail_taper_ratio':     [baseline_values[11] * 0.85, baseline_values[11] * 1.15],
}

n_samples = 100
lower_bounds = np.array([design_variable_bounds[name][0] for name in variable_names])
upper_bounds = np.array([design_variable_bounds[name][1] for name in variable_names])

# Generate Latin Hypercube Sampling
sampler = qmc.LatinHypercube(d=len(variable_names), seed=42)
unit_samples = sampler.random(n=n_samples)
lhs_samples = qmc.scale(unit_samples, lower_bounds, upper_bounds)

print(f"Generated {n_samples} Latin Hypercube samples across {len(variable_names)} design variables.")
print(f"Sample array shape: {lhs_samples.shape}")

output_dir = os.path.dirname(os.path.abspath(__file__))

# Benchmarking Loop over Latin Hypercube Samples
sample_timings = []
sample_iterations = []
sample_input_difference_norms = []
unconverged_samples = []
max_residuals = []

print(f"\n=== Starting Benchmark Timing Across {n_samples} LHS Samples ===")
total_benchmark_start = time.perf_counter()

for i, sample in enumerate(lhs_samples):
    # Set design variable values in the compiled simulator
    for j, var_value in enumerate(sample):
        sim[jax_inputs[j]] = np.array([var_value])

    # Time solve with the already-compiled model
    t_start = time.perf_counter()
    sim.run()
    t_end = time.perf_counter()
    sample_duration = t_end - t_start
    sample_timings.append(sample_duration)

    # Track Newton iterations
    it_count = iteration_log[-1] if iteration_log else 1
    sample_iterations.append(it_count)

    # Compute Euclidean distance norm of input difference from baseline
    input_diff_norm = np.linalg.norm(sample - baseline_values)
    sample_input_difference_norms.append(input_diff_norm)

    # Convergence and residual verification
    has_nan = False
    sample_max_res = 0.0

    # 1. Target geometric variables vs computed values
    for target_dv, comp_var in target_comp_pairs:
        val_comp = sim[comp_var]
        val_target = sim[target_dv]
        if np.isnan(val_comp).any() or np.isnan(val_target).any():
            has_nan = True
            break
        err = float(np.max(np.abs(val_comp - val_target)))
        if err > sample_max_res:
            sample_max_res = err

    # 2. Connection invariant constraints
    conn_eval_pairs = [
        (sim[conn_fuse_strut], target_conn_fuse_strut),
        (sim[conn_wing_strut], target_conn_wing_strut),
        (sim[conn_strut_jury], target_conn_strut_jury),
        (sim[conn_fuse_tail][[0, 2]], target_conn_fuse_tail),
    ]
    for conn_val, target_val in conn_eval_pairs:
        if np.isnan(conn_val).any():
            has_nan = True
            break
        err = float(np.max(np.abs(conn_val - target_val)))
        if err > sample_max_res:
            sample_max_res = err

    max_residuals.append(sample_max_res)
    converged = (not has_nan) and (sample_max_res < 1e-3)

    if not converged:
        unconverged_samples.append({
            'index': i,
            'sample': sample,
            'has_nan': has_nan,
            'max_residual': sample_max_res,
            'iterations': it_count,
        })

    if (i + 1) % 10 == 0 or (i + 1) == n_samples:
        print(f"  Sample {i+1:3d}/{n_samples} | Solve Time: {sample_duration*1000:6.2f} ms | Iters: {it_count:2d} | Max Res: {sample_max_res:.2e} | Status: {'CONVERGED' if converged else 'FAILED'}")

total_benchmark_time = time.perf_counter() - total_benchmark_start
timings_ms = np.array(sample_timings) * 1000.0
iterations_arr = np.array(sample_iterations)

print("\n" + "=" * 65)
print("=== TRUSS-BRACED WING (TBW) PARAMETERIZATION BENCHMARK ===")
print("=" * 65)
print(f"Total Samples Evaluated:    {n_samples}")
print(f"Converged Solves:           {n_samples - len(unconverged_samples)} / {n_samples} ({100.0 * (1 - len(unconverged_samples)/n_samples):.1f}%)")
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
print(f"Std Deviation:              {np.std(timings_ms):.2f} ms")
print("-" * 65)
print(f"Mean Newton Iterations:     {np.mean(iterations_arr):.2f}")
print(f"Median Newton Iterations:   {int(np.median(iterations_arr))}")
print(f"Min / Max Iterations:       {np.min(iterations_arr)} / {np.max(iterations_arr)}")
print(f"Max Residual Across All:    {np.max(max_residuals):.2e}")
print("=" * 65 + "\n")

# Cache timing and benchmark results
cache_file = os.path.join(output_dir, "truss_braced_wing_lhs_timing_results.pkl")
timing_data = {
    'variable_names': variable_names,
    'baseline_values': baseline_values,
    'design_variable_bounds': design_variable_bounds,
    'lhs_samples': lhs_samples,
    'sample_timings': sample_timings,
    'sample_iterations': sample_iterations,
    'sample_input_difference_norms': sample_input_difference_norms,
    'max_residuals': max_residuals,
    'unconverged_samples': unconverged_samples,
    'compile_time': compile_time,
    'total_benchmark_time': total_benchmark_time,
    'n_samples': n_samples,
    'stats': {
        'mean_ms': float(np.mean(timings_ms)),
        'median_ms': float(np.median(timings_ms)),
        'min_ms': float(np.min(timings_ms)),
        'max_ms': float(np.max(timings_ms)),
        'std_ms': float(np.std(timings_ms)),
        'mean_iterations': float(np.mean(iterations_arr)),
        'median_iterations': int(np.median(iterations_arr)),
    }
}

with open(cache_file, 'wb') as f:
    pickle.dump(timing_data, f)
print(f"Results successfully cached to '{cache_file}'")

# Generate and save benchmark box and whisker plot (Solve Time & Iterations)
plot_file = os.path.join(output_dir, "time_vs_input_norm_tbw.png")
boxplot_file = os.path.join(output_dir, "solve_time_boxplot_tbw.png")
fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), dpi=300)

# Left: Solve Time Boxplot
ax1 = axes[0]
bp1 = ax1.boxplot(
    timings_ms, patch_artist=True, widths=0.45,
    boxprops=dict(facecolor='#d0e1fd', color='#1f77b4', linewidth=1.8),
    medianprops=dict(color='#d62728', linewidth=2.2),
    whiskerprops=dict(color='#1f77b4', linewidth=1.4),
    capprops=dict(color='#1f77b4', linewidth=1.4),
    flierprops=dict(marker='o', markerfacecolor='#1f77b4', markeredgecolor='none', alpha=0.5, markersize=5),
    showmeans=True,
    meanprops=dict(marker='D', markerfacecolor='#2ca02c', markeredgecolor='k', markersize=6)
)
np.random.seed(42)
jitter1 = np.random.normal(0, 0.035, size=len(timings_ms))
ax1.scatter(np.ones_like(timings_ms) + jitter1, timings_ms, color='#1f77b4', alpha=0.55, s=28, zorder=3, edgecolors='none')
ax1.set_xticks([1])
ax1.set_xticklabels([f'Truss-Braced Wing\n({n_samples} LHS Samples)'], fontsize=11, fontweight='bold')
ax1.set_ylabel('Solve Time (ms)', fontsize=12)
ax1.set_title('Solve Time Distribution', fontsize=12, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.4, axis='y')

stats_box = (
    f'Samples (N): {len(timings_ms)}\n'
    f'Mean (♦):    {np.mean(timings_ms):.2f} ms\n'
    f'Median (—):  {np.median(timings_ms):.2f} ms\n'
    f'Std Dev:     {np.std(timings_ms):.2f} ms\n'
    f'Min:         {np.min(timings_ms):.2f} ms\n'
    f'Max:         {np.max(timings_ms):.2f} ms'
)
ax1.text(0.05, 0.95, stats_box, transform=ax1.transAxes, verticalalignment='top',
        fontsize=10, family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#b0b0b0'))

# Right: Iteration Count Boxplot
ax2 = axes[1]
bp2 = ax2.boxplot(
    iterations_arr, patch_artist=True, widths=0.45,
    boxprops=dict(facecolor='#d4edda', color='#28a745', linewidth=1.8),
    medianprops=dict(color='#d62728', linewidth=2.2),
    whiskerprops=dict(color='#28a745', linewidth=1.4),
    capprops=dict(color='#28a745', linewidth=1.4),
    flierprops=dict(marker='o', markerfacecolor='#28a745', markeredgecolor='none', alpha=0.5, markersize=5),
    showmeans=True,
    meanprops=dict(marker='D', markerfacecolor='#1f77b4', markeredgecolor='k', markersize=6)
)
jitter2 = np.random.normal(0, 0.035, size=len(iterations_arr))
ax2.scatter(np.ones_like(iterations_arr) + jitter2, iterations_arr, color='#28a745', alpha=0.55, s=28, zorder=3, edgecolors='none')
ax2.set_xticks([1])
ax2.set_xticklabels(['Newton Solver'], fontsize=11, fontweight='bold')
ax2.set_ylabel('Iterations to Convergence', fontsize=12)
ax2.set_title('Iteration Count Distribution', fontsize=12, fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.4, axis='y')

iter_box = (
    f'Mean (♦):    {np.mean(iterations_arr):.2f}\n'
    f'Median (—):  {int(np.median(iterations_arr))}\n'
    f'Min / Max:   {np.min(iterations_arr)} / {np.max(iterations_arr)}\n'
    f'Convergence: {n_samples - len(unconverged_samples)}/{n_samples} ({100.0 * (1 - len(unconverged_samples)/n_samples):.1f}%)'
)
ax2.text(0.05, 0.95, iter_box, transform=ax2.transAxes, verticalalignment='top',
        fontsize=10, family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#b0b0b0'))

plt.suptitle('Truss-Braced Wing Parameterization Solve Performance', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
plt.savefig(boxplot_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"Benchmark boxplot successfully saved to '{plot_file}' and '{boxplot_file}'")
