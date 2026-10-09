#!/usr/bin/env python3
"""
Master Runner for Geometric Parameterization Benchmarks (100 LHS Samples)
========================================================================
Runs all 4 geometric parameterization benchmarks under consistent settings:
  1. Laser-Powered UAV (100 LHS samples)
  2. Quadruped Robot (100 LHS samples)
  3. Lift+Cruise (Equality, 100 LHS samples)
  4. Lift+Cruise (With Inequalities, 100 LHS samples)
  5. Publication Figure Generator (regenerates all PNG & PDF figures)

Usage:
  # Using your Python 3.14 environment:
  python3.14 examples/showcase_examples/run_all_geometric_benchmarks.py

  # Or using specific flags:
  python run_all_geometric_benchmarks.py --all
  python run_all_geometric_benchmarks.py --uav
  python run_all_geometric_benchmarks.py --quadruped
  python run_all_geometric_benchmarks.py --lpc-eq
  python run_all_geometric_benchmarks.py --lpc-ineq
  python run_all_geometric_benchmarks.py --figures-only
"""

import os
import sys
import time
import argparse
import subprocess

def print_environment_info():
    print("=" * 80)
    print("      LSDO_GEO GEOMETRIC PARAMETERIZATION BENCHMARK SUITE (100 SAMPLES)")
    print("=" * 80)
    print(f"Python Executable : {sys.executable}")
    print(f"Python Version    : {sys.version.split()[0]}")
    
    try:
        import jax
        print(f"JAX Version       : {jax.__version__}")
        print(f"JAX Devices       : {jax.devices()}")
    except ImportError:
        print("JAX Version       : [Not found in current environment]")

    try:
        import csdl_alpha as csdl
        print(f"CSDL Version      : {getattr(csdl, '__version__', 'alpha')}")
    except ImportError:
        print("CSDL Version      : [Not found in current environment]")

    try:
        import numpy as np
        print(f"NumPy Version     : {np.__version__}")
    except ImportError:
        pass
    print("=" * 80 + "\n")

def run_benchmark_step(name, script_relpath, env_overrides=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(base_dir, script_relpath)
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")
        
    print("-" * 80)
    print(f"STARTING BENCHMARK: {name}")
    print(f"Script: {script_path}")
    print("-" * 80)
    
    env = os.environ.copy()
    env["BENCH_N_SAMPLES"] = "100"
    if env_overrides:
        env.update(env_overrides)
        
    repo_root = os.path.abspath(os.path.join(base_dir, "../.."))
    
    t_start = time.perf_counter()
    ret = subprocess.run([sys.executable, script_path], env=env, cwd=repo_root)
    t_elapsed = time.perf_counter() - t_start
    
    if ret.returncode != 0:
        print(f"\n[ERROR] Benchmark '{name}' failed with return code {ret.returncode}!")
        return False, t_elapsed
    else:
        print(f"\n[SUCCESS] Benchmark '{name}' completed successfully in {t_elapsed:.2f} seconds.")
        return True, t_elapsed

def main():
    parser = argparse.ArgumentParser(description="Master runner for geometric parameterization benchmarks (100 LHS samples).")
    parser.add_argument("--all", action="store_true", default=True, help="Run all 4 benchmarks and regenerate figures (default).")
    parser.add_argument("--uav", action="store_true", help="Run only Laser-Powered UAV benchmark.")
    parser.add_argument("--quadruped", action="store_true", help="Run only Quadruped Robot benchmark.")
    parser.add_argument("--lpc-eq", action="store_true", help="Run only Lift+Cruise (Equality) benchmark.")
    parser.add_argument("--lpc-ineq", action="store_true", help="Run only Lift+Cruise (Inequalities) benchmark.")
    parser.add_argument("--figures-only", action="store_true", help="Regenerate publication figures from existing pkl data.")
    args = parser.parse_args()

    # If any specific benchmark flag is passed, disable the default --all
    specific_flags = [args.uav, args.quadruped, args.lpc_eq, args.lpc_ineq, args.figures_only]
    if any(specific_flags):
        args.all = False

    print_environment_info()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(base_dir, "../.."))
    results = {}

    if not args.figures_only:
        # 1. Laser-Powered UAV
        if args.all or args.uav:
            success, dur = run_benchmark_step(
                name="Laser-Powered UAV (100 samples)",
                script_relpath="laser_powered_uav/ex_laser_powered_uav_benchmarking.py"
            )
            results["Laser-Powered UAV"] = (success, dur)

        # 2. Quadruped Robot
        if args.all or args.quadruped:
            success, dur = run_benchmark_step(
                name="Quadruped Robot (100 samples)",
                script_relpath="quadruped_robot/ex_quadruped_robot_benchmarking.py"
            )
            results["Quadruped Robot"] = (success, dur)

        # 3. Lift+Cruise (Equality)
        if args.all or args.lpc_eq:
            success, dur = run_benchmark_step(
                name="Lift+Cruise - Equality (100 samples)",
                script_relpath="lift_plus_cruise/ex_lift_plus_cruise_benchmarking.py",
                env_overrides={"BENCH_ENABLE_INEQUALITIES": "False", "BENCH_N_SAMPLES": "100"}
            )
            results["Lift+Cruise (Equality)"] = (success, dur)

        # 4. Lift+Cruise (Inequalities)
        if args.all or args.lpc_ineq:
            success, dur = run_benchmark_step(
                name="Lift+Cruise - With Inequalities (100 samples)",
                script_relpath="lift_plus_cruise/ex_lift_plus_cruise_benchmarking.py",
                env_overrides={"BENCH_ENABLE_INEQUALITIES": "True", "BENCH_N_SAMPLES": "100"}
            )
            results["Lift+Cruise (Inequalities)"] = (success, dur)

    # 5. Publication Figures
    if args.all or args.figures_only or any(r[0] for r in results.values()):
        print("\n" + "=" * 80)
        print("REGENERATING PUBLICATION-READY BENCHMARK FIGURES")
        print("=" * 80)
        fig_script = os.path.join(base_dir, "generate_combined_benchmark_figure.py")
        t_fig_start = time.perf_counter()
        ret = subprocess.run([sys.executable, fig_script], cwd=repo_root)
        t_fig_elapsed = time.perf_counter() - t_fig_start
        results["Figure Generation"] = (ret.returncode == 0, t_fig_elapsed)

    print("\n" + "=" * 80)
    print("BENCHMARK EXECUTION SUMMARY")
    print("=" * 80)
    for name, (ok, dur) in results.items():
        status = "PASSED" if ok else "FAILED"
        print(f"  {name:35s}: [{status}] in {dur:.2f} s")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()

