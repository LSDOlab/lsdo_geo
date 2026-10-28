#!/usr/bin/env python3
"""
Spanwise Mesh Refinement Study: Wake-Load Consistency & Lift Distribution.

Evaluates the converged 6DV design vector from the latest optimization run
(2026-10-13_16.56.18.459801) across spanwise mesh resolutions:
  - 15 spanwise sections (15S_50C)
  - 35 spanwise sections (35S_50C)
  - 60 spanwise sections (60S_50C)

Analyzes whether the mismatch between near-field pressure loading and far-field
wake loading is a spatial discretization artifact or a 3D centerbody effect.
"""

from __future__ import annotations

import os
import sys
import time
import subprocess
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

_trapezoid = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

REPO_ROOT = Path(__file__).resolve().parents[4]
SHOWCASE_DIR = REPO_ROOT / "examples" / "showcase_examples" / "rectangular_wing"
if str(SHOWCASE_DIR) not in sys.path:
    sys.path.insert(0, str(SHOWCASE_DIR))
OUTPUT_BASE = REPO_ROOT / "rectangular_wing_to_bwb_aerostructural_optimization_outputs" / "spanwise_refinement_study"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

OPTIMAL_RUN_DIR = REPO_ROOT / "rectangular_wing_to_bwb_aerostructural_optimization_outputs" / "2026-10-13_16.56.18.459801"
X_OPT_PATH = OPTIMAL_RUN_DIR / "x.out"

PYTHON_BIN = "/home/andrewfletcher/anaconda3/envs/py310/bin/python"

CASES = [
    {
        "name": "15S_50C",
        "n_span": 15,
        "n_chord": 50,
        "n_dvs": 6,
        "color": "#d95f02",  # Rust orange
        "linestyle": ":",
        "label": "15S / 50C",
    },
    {
        "name": "35S_50C",
        "n_span": 35,
        "n_chord": 50,
        "n_dvs": 6,
        "color": "#7570b3",  # Muted purple
        "linestyle": "--",
        "label": "35S / 50C (Baseline)",
    },
    {
        "name": "40S_50C",
        "n_span": 40,
        "n_chord": 50,
        "n_dvs": 6,
        "color": "#e7298a",  # Magenta / pink
        "linestyle": "-.",
        "label": "40S / 50C",
    },
    {
        "name": "50S_50C",
        "n_span": 50,
        "n_chord": 50,
        "n_dvs": 6,
        "color": "#386cb0",  # Strong blue
        "linestyle": "--",
        "label": "50S / 50C",
    },
    {
        "name": "60S_50C",
        "n_span": 60,
        "n_chord": 50,
        "n_dvs": 6,
        "color": "#1b9e77",  # Teal / green
        "linestyle": "-",
        "label": "60S / 50C (Fine)",
    },
]


def extract_converged_vector():
    if not X_OPT_PATH.exists():
        raise FileNotFoundError(f"Optimal vector file not found: {X_OPT_PATH}")
    x_hist = np.loadtxt(X_OPT_PATH)
    x_opt = x_hist[-1] if x_hist.ndim > 1 else x_hist
    print(f"Loaded converged design vector from {X_OPT_PATH}: shape = {x_opt.shape}")
    return x_opt


def run_forward_evaluation(case, x_opt):
    case_name = case["name"]
    case_dir = OUTPUT_BASE / f"eval_{case_name}"
    case_dir.mkdir(parents=True, exist_ok=True)

    # Write single-row x.out for this case
    x_out_file = case_dir / "x.out"
    np.savetxt(x_out_file, x_opt.reshape(1, -1))

    log_file = case_dir / "eval.log"
    print(f"\n=======================================================")
    print(f"Evaluating Case: {case_name}")
    print(f"  Spanwise sections: {case['n_span']}")
    print(f"  Chordwise panels:  {case['n_chord']}")
    print(f"  Design variables:  {case['n_dvs']}")
    print(f"  Output directory:  {case_dir}")
    print(f"=======================================================")

    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["SKIP_OPTIMIZATION"] = "1"
    env["TARGET_OUTPUT_FOLDER"] = str(case_dir)
    env["BWB_NUM_SPANWISE_SECTIONS"] = str(case["n_span"])
    env["BWB_NUM_CHORDWISE_PANELS"] = str(case["n_chord"])
    env["BWB_NUM_DVS"] = str(case["n_dvs"])
    env["GEONIC_PAYLOAD_MODE"] = "none"
    env["VISCOUS_DRAG_MODE"] = "ibl"
    env["STALL_MODEL"] = "stratford_closure"

    cmd = [
        "taskset", "-c", "0-11",
        PYTHON_BIN,
        str(SHOWCASE_DIR / "ex_rectangular_wing_to_bwb.py"),
    ]

    t0 = time.time()
    with open(log_file, "w") as f_log:
        ret = subprocess.run(cmd, env=env, stdout=f_log, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT))
    elapsed = time.time() - t0

    if ret.returncode != 0:
        print(f"ERROR: Evaluation of {case_name} failed with exit code {ret.returncode} ({elapsed:.1f} s). Check {log_file}")
        return False, elapsed
    else:
        print(f"SUCCESS: Evaluation of {case_name} finished in {elapsed:.1f} s.")
        return True, elapsed


def collect_and_plot(artifact_dir=None):
    results = {}
    for case in CASES:
        case_name = case["name"]
        case_dir = OUTPUT_BASE / f"eval_{case_name}"
        wake_data_file = case_dir / "wake_load_consistency_data.npz"
        lift_data_file = case_dir / "lift_distribution_data.npz"

        if not wake_data_file.exists():
            print(f"Warning: {wake_data_file} not found. Skipping.")
            continue

        w_data = np.load(wake_data_file)
        l_data = np.load(lift_data_file) if lift_data_file.exists() else None

        eta = w_data["eta"]
        w_curve = w_data["wake_curve_scaled"][-1]

        # Use updated aggregate_pressure_loading for robust strip integration
        from optimization_analyses.wake_load_consistency import aggregate_pressure_loading, _curve_on_grid
        centers = l_data["panel_centers_right"]
        forces = l_data["f_cruise"]
        yp, dp, pressure_lift, bp = aggregate_pressure_loading(centers, forces)
        b = max(bp, float(np.max(w_data["eta"] * bp)))
        p_curve = _curve_on_grid(yp, dp, b, eta, pressure_lift)
        diff = p_curve - w_curve

        denom = b * _trapezoid(p_curve**2, eta)
        l2_norm = float(np.sqrt(b * _trapezoid(diff**2, eta) / denom))
        max_norm = float(np.max(np.abs(diff)) / np.max(np.abs(p_curve)))

        results[case_name] = {
            "case": case,
            "eta": eta,
            "pressure": p_curve,
            "wake": w_curve,
            "diff": diff,
            "mismatch_l2": l2_norm,
            "mismatch_max": max_norm,
            "cl": float(w_data["cl"][-1]),
            "cdi_trefftz": float(w_data["cdi_trefftz"][-1]),
            "cdi_fourier": float(w_data["cdi_fourier"][-1]),
            "pressure_wake_ratio": float(w_data["pressure_wake_lift_ratio"][-1]),
            "half_span": b,
            "lift_data": l_data,
        }

    # Print Summary Table
    print("\n" + "=" * 110)
    print(f"{'Mesh':10s} | {'N_span':6s} | {'L2 Mismatch':12s} | {'Max Mismatch':13s} | {'P(0) [N/m]':12s} | {'W(0) [N/m]':12s} | {'Delta(0) [%]':13s} | {'CL':8s} | {'CDi_Trefftz':12s}")
    print("-" * 110)
    for case_name, res in results.items():
        case = res["case"]
        p0 = res["pressure"][0]
        w0 = res["wake"][0]
        d0_pct = (p0 - w0) / p0 * 100.0
        print(f"{case_name:10s} | {case['n_span']:6d} | {res['mismatch_l2']*100:10.2f} % | {res['mismatch_max']*100:11.2f} % | {p0:12.1f} | {w0:12.1f} | {d0_pct:+12.2f} % | {res['cl']:8.4f} | {res['cdi_trefftz']*1e4:10.2f} cts")
    print("=" * 110 + "\n")

    # Generate Detailed 4-Panel Comparison Figure
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # Subplot 1: Near-field Pressure vs Wake Loading across Full Semi-Span
    for case_name, res in results.items():
        case = res["case"]
        eta = res["eta"]
        ax1.plot(eta, res["pressure"], color=case["color"], linestyle="-", linewidth=2.0,
                 label=f"{case_name} Pressure")
        ax1.plot(eta, res["wake"], color=case["color"], linestyle=":", linewidth=2.0, alpha=0.8,
                 label=f"{case_name} Wake")

    # Elliptical reference from 35S
    if "35S_50C" in results:
        ref_p = results["35S_50C"]["pressure"]
        eta = results["35S_50C"]["eta"]
        p_tot = _trapezoid(ref_p, eta)
        ell = (4.0 * p_tot / np.pi) * np.sqrt(np.clip(1.0 - eta**2, 0.0, 1.0))
        ax1.plot(eta, ell, "k--", linewidth=1.5, alpha=0.6, label="Elliptical Ref")

    ax1.set_xlabel(r"Semi-Span Fraction $\eta = y / (b/2)$", fontsize=11)
    ax1.set_ylabel("Sectional Lift Density $dL/dy$ [N/m]", fontsize=11)
    ax1.set_title("(a) Pressure-Panel vs. Wake Loading across Span", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.7)
    ax1.legend(loc="lower left", fontsize=9, ncol=2)

    # Subplot 2: Difference Curve (Pressure - Wake Loading)
    for case_name, res in results.items():
        case = res["case"]
        eta = res["eta"]
        ax2.plot(eta, res["diff"], color=case["color"], linestyle=case["linestyle"], linewidth=2.2,
                 label=f"{case_name} ($L_2$={res['mismatch_l2']*100:.1f}%)")
    ax2.axhline(0.0, color="black", linestyle="-", linewidth=1.0, alpha=0.5)
    ax2.set_xlabel(r"Semi-Span Fraction $\eta = y / (b/2)$", fontsize=11)
    ax2.set_ylabel(r"Loading Discrepancy $\Delta(dL/dy)$ [N/m]", fontsize=11)
    ax2.set_title("(b) Loading Discrepancy (Pressure - Wake)", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.7)
    ax2.legend(loc="upper right", fontsize=10)

    # Subplot 3: Centerbody & Root Zoom (eta in [0, 0.35])
    for case_name, res in results.items():
        case = res["case"]
        eta = res["eta"]
        mask = eta <= 0.35
        ax3.plot(eta[mask], res["pressure"][mask], color=case["color"], linestyle="-", linewidth=2.2,
                 label=f"{case_name} Pressure")
        ax3.plot(eta[mask], res["wake"][mask], color=case["color"], linestyle=":", linewidth=2.0,
                 label=f"{case_name} Wake")
    ax3.set_xlabel(r"Centerbody Semi-Span Fraction $\eta = y / (b/2)$", fontsize=11)
    ax3.set_ylabel("Sectional Lift Density $dL/dy$ [N/m]", fontsize=11)
    ax3.set_title("(c) Centerbody / Root Zoom ($\eta \in [0, 0.35]$)", fontsize=12, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.7)
    ax3.legend(loc="lower left", fontsize=9, ncol=2)

    # Subplot 4: Convergence of Metrics vs Spanwise Sections
    spans = [res["case"]["n_span"] for res in results.values()]
    l2_vals = [res["mismatch_l2"] * 100 for res in results.values()]
    max_vals = [res["mismatch_max"] * 100 for res in results.values()]
    root_diff_pct = [abs(res["pressure"][0] - res["wake"][0]) / res["pressure"][0] * 100 for res in results.values()]

    ax4.plot(spans, l2_vals, "o-", color="#1f77b4", linewidth=2.2, markersize=8, label=r"Normalized $L_2$ Mismatch (%)")
    ax4.plot(spans, max_vals, "s--", color="#ff7f0e", linewidth=2.2, markersize=8, label="Max Sectional Mismatch (%)")
    ax4.plot(spans, root_diff_pct, "^-.", color="#2ca02c", linewidth=2.2, markersize=8, label=r"Root $\eta=0$ Mismatch (%)")
    ax4.set_xlabel("Spanwise Analysis Mesh Stations ($N_{\mathrm{span}}$)", fontsize=11)
    ax4.set_ylabel("Relative Mismatch (%)", fontsize=11)
    ax4.set_title("(d) Error Metric Convergence vs. Spanwise Mesh", fontsize=12, fontweight="bold")
    ax4.set_xticks(spans)
    ax4.grid(True, linestyle=":", alpha=0.7)
    ax4.legend(loc="upper right", fontsize=10)

    fig_path = OUTPUT_BASE / "spanwise_wake_consistency_comparison.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Comparison figure saved to: {fig_path}")

    if artifact_dir and Path(artifact_dir).exists():
        import shutil
        art_fig_path = Path(artifact_dir) / "spanwise_wake_consistency_comparison.png"
        shutil.copy2(fig_path, art_fig_path)
        print(f"Copied comparison figure to artifact directory: {art_fig_path}")

    return results


def main():
    x_opt = extract_converged_vector()

    # Run evaluations
    for case in CASES:
        case_name = case["name"]
        case_dir = OUTPUT_BASE / f"eval_{case_name}"
        wake_data_file = case_dir / "wake_load_consistency_data.npz"

        # Check if already evaluated
        if wake_data_file.exists() and os.environ.get("FORCE_RERUN", "0") != "1":
            print(f"Found existing evaluation for {case_name} at {wake_data_file}. Skipping rerun.")
            continue

        success, elapsed = run_forward_evaluation(case, x_opt)
        if not success:
            print(f"Aborting study due to failure in {case_name}")
            return

    artifact_dir = os.environ.get("ARTIFACT_DIR", None)
    if not artifact_dir:
        # Default conversation artifact directory
        candidate = Path("/home/andrewfletcher/.gemini/antigravity/brain/a2f03ba8-1c5a-4fe2-ba15-b4ee2887956e")
        if candidate.exists():
            artifact_dir = str(candidate)

    collect_and_plot(artifact_dir)


if __name__ == "__main__":
    main()

