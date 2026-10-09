"""
Benchmarking Script for Quadruped Robot Geometry Parameterization
=================================================================
Evaluates execution time and Newton solver iteration performance across
Latin Hypercube Sampling (LHS) of design variables.

Pulls geometry, component declarations, and parameterization setup from
the standard `ex_quadruped_robot.py` script to avoid code duplication.
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

# Add script directory to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from ex_quadruped_robot import (
    geometry,
    recorder,
    jax_inputs,
    variable_names,
    units,
    target_comp_pairs,
    tracking_outputs,
    jax_outputs,
    baseline_fl_hip_conn,
    baseline_fr_hip_conn,
    baseline_rl_hip_conn,
    baseline_rr_hip_conn,
    baseline_fl_knee_conn,
    baseline_fr_knee_conn,
    baseline_rl_knee_conn,
    baseline_rr_knee_conn,
)

print("=== Setting up JaxSimulator for Quadruped Robot Benchmarking ===")
print("Compiling JaxSimulator (one-time compilation)...")
t_comp_start = time.perf_counter()
sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=jax_inputs,
    additional_outputs=jax_outputs,
    gpu=False,
)
iteration_log.clear()
sim.run()
compile_time = time.perf_counter() - t_comp_start
print(f"JaxSimulator compiled successfully in {compile_time:.2f} seconds.\n")

# Baseline values for design variables
baseline_values = np.array([float(np.squeeze(inp.value)) for inp in jax_inputs])
print("=== Baseline Design Variable Values ===")
for name, val, unit in zip(variable_names, baseline_values, units):
    print(f"  {name:25s}: {val:10.4f} [{unit}]")
print()

# Aggressive design space bounds (±50% on geometric ratios/lengths and posture angles)
design_variable_bounds = {
    'scale_length':           [baseline_values[0] * 0.50, baseline_values[0] * 1.50],
    'leg_aspect_ratio':       [baseline_values[1] * 0.50, baseline_values[1] * 1.50],
    'leg_to_body_ratio':      [baseline_values[2] * 0.50, baseline_values[2] * 1.50],
    'chassis_aspect_ratio':   [baseline_values[3] * 0.50, baseline_values[3] * 1.50],
    'abduction_offset_ratio': [baseline_values[4] * 0.50, baseline_values[4] * 1.50],
    'initial_hip_angle':      [baseline_values[5] * 0.50, baseline_values[5] * 1.50],
    'initial_knee_angle':     [baseline_values[6] * 1.50, baseline_values[6] * 0.50],
}

n_samples = 100
lower_bounds = np.array([design_variable_bounds[name][0] for name in variable_names])
upper_bounds = np.array([design_variable_bounds[name][1] for name in variable_names])

sampler = qmc.LatinHypercube(d=len(variable_names), seed=42)
unit_samples = sampler.random(n=n_samples)
lhs_samples = qmc.scale(unit_samples, lower_bounds, upper_bounds)

print(f"Generated {n_samples} Latin Hypercube samples across {len(variable_names)} design variables.")

sample_timings = []
sample_iterations = []
sample_input_difference_norms = []
unconverged_samples = []
max_residuals = []

print(f"\n=== Starting Benchmark Timing Across {n_samples} LHS Samples ===")
total_benchmark_start = time.perf_counter()

for i, sample in enumerate(lhs_samples):
    for j, var_value in enumerate(sample):
        sim[jax_inputs[j]] = np.array([var_value])

    iteration_log.clear()
    t_start = time.perf_counter()
    sim.run()
    t_end = time.perf_counter()
    duration = t_end - t_start

    iters = iteration_log[-1] if iteration_log else 0
    sample_timings.append(duration)
    sample_iterations.append(iters)

    diff_norm = np.linalg.norm((sample - baseline_values) / baseline_values)
    sample_input_difference_norms.append(diff_norm)

    # Residual checks
    res_list = []
    for pair in target_comp_pairs:
        target_val = float(np.squeeze(sim[pair[0]]))
        comp_val = float(np.squeeze(sim[pair[1]]))
        res_list.append(abs(comp_val - target_val))

    max_res = max(res_list) if res_list else 0.0
    max_residuals.append(max_res)

    if np.isnan(max_res) or max_res > 1e-4:
        unconverged_samples.append({
            'index': i,
            'max_residual': max_res,
            'iterations': iters,
            'sample': sample.tolist(),
        })

    if (i + 1) % 20 == 0 or (i + 1) == n_samples:
        print(f"  Sample {i+1:3d}/{n_samples}: solve_time = {duration*1000.0:6.2f} ms | newton_iters = {iters:2d} | max_res = {max_res:.2e}")

total_benchmark_time = time.perf_counter() - total_benchmark_start
timings_ms = np.array(sample_timings) * 1000.0
iterations_arr = np.array(sample_iterations)

print("\n" + "=" * 65)
print("=== QUADRUPED ROBOT PARAMETERIZATION BENCHMARK SUMMARY ===")
print("=" * 65)
print(f"Total Samples Evaluated:    {n_samples}")
print(f"Converged Solves:           {n_samples - len(unconverged_samples)} / {n_samples} ({100.0 * (1 - len(unconverged_samples)/n_samples):.1f}%)")
print(f"One-Time Compilation Time: {compile_time:.2f} s")
print(f"Total Benchmark Time:       {total_benchmark_time:.2f} s")
print(f"Mean Solve Time:            {np.mean(timings_ms):.2f} ms")
print(f"Median Solve Time:          {np.median(timings_ms):.2f} ms")
print(f"Min Solve Time:             {np.min(timings_ms):.2f} ms")
print(f"Max Solve Time:             {np.max(timings_ms):.2f} ms")
print(f"Std Deviation:              {np.std(timings_ms):.2f} ms")
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
    "n_samples": n_samples,
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
}

pkl_filename = os.path.join(script_dir, "quadruped_robot_lhs_timing_results.pkl")
with open(pkl_filename, "wb") as f:
    pickle.dump(results_dict, f)
with open(os.path.join(script_dir, "robot_dog_lhs_timing_results.pkl"), "wb") as f:
    pickle.dump(results_dict, f)
print(f"\nResults successfully cached to '{pkl_filename}'")

