"""Post-processing diagnostic for near-field versus wake sectional loading."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOW_CL_RUN_NAMES = (
    "2026-09-21_18.59.33.832869",
    "2026-09-21_19.13.02.809357",
    "2026-09-21_19.22.18.066017",
)


def _curve_on_grid(x, values, b, grid, total):
    x = np.asarray(x, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float).reshape(-1)
    order = np.argsort(x)
    x, values = x[order], values[order]
    eta = np.clip(x / b, 0.0, 1.0)
    eta, unique = np.unique(eta, return_index=True)
    values = values[unique]
    eta = np.concatenate(([0.0], eta, [1.0]))
    values = np.concatenate(([values[0]], values, [0.0]))
    curve = np.interp(grid, eta, values)
    integral = b * np.trapz(curve, grid)
    if not np.isfinite(integral) or integral <= 0.0:
        raise ValueError("sectional loading has a non-positive integral")
    return curve * (total / integral)


def aggregate_pressure_loading(panel_centers, panel_forces):
    """Return strip centers, lift density, half-wing lift, and half-span."""
    centers = np.asarray(panel_centers, dtype=float)
    forces = np.asarray(panel_forces, dtype=float)
    if centers.ndim != 2 or forces.ndim != 2 or centers.shape[0] != forces.shape[0]:
        raise ValueError("panel centers and forces must have matching 2-D shapes")
    y_round = np.round(centers[:, 1], 4)
    main_stations = [y for y in np.unique(y_round) if np.count_nonzero(y_round == y) == 40]
    if not main_stations:
        raise ValueError("no 40-panel spanwise stations were found")
    y = np.array([centers[y_round == station, 1].mean() for station in main_stations])
    strip_lift = np.array([forces[y_round == station, 2].sum() for station in main_stations])
    tip_mask = ~np.isin(y_round, main_stations)
    if np.any(tip_mask):
        strip_lift[-1] += forces[tip_mask, 2].sum()
    b = float(np.max(centers[:, 1]))
    bounds = np.concatenate(([0.0], 0.5 * (y[:-1] + y[1:]), [b]))
    return y, strip_lift / np.diff(bounds), float(strip_lift.sum()), b


def aggregate_strip_loading(panel_centers, panel_values):
    """Aggregate a scalar panel quantity with the BWB Fourier strip convention."""
    values = np.asarray(panel_values, dtype=float).reshape(-1)
    centers = np.asarray(panel_centers, dtype=float)
    if centers.ndim != 2 or centers.shape[0] != values.size:
        raise ValueError("panel centers and scalar values must have matching lengths")
    forces = np.zeros((values.size, 3))
    forces[:, 2] = values
    return aggregate_pressure_loading(centers, forces)


def extract_wake_loading(panel_mesh, mu_w, te_edges):
    """Return right-half wake centers, |mu_w| density, raw measure, endpoints."""
    mesh = np.asarray(panel_mesh, dtype=float)
    if mesh.ndim == 3:
        mesh = mesh[0]
    strengths = np.asarray(mu_w, dtype=float)
    if strengths.ndim > 1:
        strengths = strengths[0]
    edges = np.asarray(te_edges, dtype=int)
    if mesh.ndim != 2 or edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("invalid dynamic panel mesh or trailing-edge topology")
    if strengths.size != len(edges):
        raise ValueError("mu_w and trailing-edge edge counts do not match")
    endpoints = mesh[edges]
    y_a, y_b = endpoints[:, 0, 1], endpoints[:, 1, 1]
    y_mid = 0.5 * (y_a + y_b)
    width = np.abs(y_b - y_a)
    right = y_mid > 0.0
    if not np.any(right):
        raise ValueError("no right-half wake segments were found")
    indices = np.where(right)[0][np.argsort(y_mid[right])]
    y_mid, width = y_mid[indices], width[indices]
    density = np.abs(strengths[indices])
    return y_mid, density, float(np.sum(density * width)), endpoints[indices]


def extract_signed_wake_loading(panel_mesh, mu_w, te_edges):
    """Return ordered right-half signed wake strengths and VortexAD abs(dy) widths."""
    mesh = np.asarray(panel_mesh, dtype=float)
    if mesh.ndim == 3:
        mesh = mesh[0]
    strengths = np.asarray(mu_w, dtype=float)
    if strengths.ndim > 1:
        strengths = strengths[0]
    edges = np.asarray(te_edges, dtype=int)
    if mesh.ndim != 2 or edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("invalid dynamic panel mesh or trailing-edge topology")
    if strengths.size != len(edges):
        raise ValueError("mu_w and trailing-edge edge counts do not match")
    endpoints = mesh[edges]
    y_a, y_b = endpoints[:, 0, 1], endpoints[:, 1, 1]
    y_mid = 0.5 * (y_a + y_b)
    right = y_mid > 0.0
    if not np.any(right):
        raise ValueError("no right-half wake segments were found")
    indices = np.where(right)[0][np.argsort(y_mid[right])]
    widths = np.abs(y_b[indices] - y_a[indices])
    if not np.all(np.isfinite(widths)) or np.any(widths <= 0.0):
        raise ValueError("wake contains zero or invalid spanwise segments")
    return y_mid[indices], strengths[indices], widths, endpoints[indices]


def _relative_l2(reference, candidate, b):
    denominator = b * np.trapz(reference ** 2, dx=1.0 / (reference.size - 1))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("reference loading has zero L2 norm")
    return float(np.sqrt(b * np.trapz((reference - candidate) ** 2, dx=1.0 / (reference.size - 1)) / denominator))


def _shape_curve(curve, b):
    total = b * np.trapz(curve, dx=1.0 / (curve.size - 1))
    if not np.isfinite(total) or abs(total) <= 1e-12:
        raise ValueError("sectional loading has a zero signed integral")
    return curve / total


class WakeLoadConsistencyHistory:
    """Accumulate comparisons and write diagnostic artifacts."""

    def __init__(self, grid_size=201):
        self.eta = np.linspace(0.0, 1.0, grid_size)
        self.records: list[dict[str, Any]] = []
        self.final_wake_endpoints = np.empty((0, 2, 3))
        self.final_mu_w = np.empty(0)

    def record(self, iteration, panel_centers, panel_forces, panel_mesh, mu_w,
               te_edges, rho_inf, velocity_inf, cl, cdi_fourier, cdi_trefftz,
               trefftz_lift_ratio=None, sound_speed=None):
        record = dict(iteration=int(iteration), cl=float(cl),
                      cdi_fourier=float(cdi_fourier), cdi_trefftz=float(cdi_trefftz))
        try:
            yp, dp, pressure_lift, bp = aggregate_pressure_loading(panel_centers, panel_forces)
            yw, dw, wake_measure, endpoints = extract_wake_loading(panel_mesh, mu_w, te_edges)
            b = max(bp, float(np.max(yw)))
            pressure = _curve_on_grid(yp, dp, b, self.eta, pressure_lift)
            wake = _curve_on_grid(yw, dw, b, self.eta, 1.0) * pressure_lift
            difference = pressure - wake
            denominator = b * np.trapz(pressure ** 2, self.eta)
            wake_lift_incomp = float(rho_inf * velocity_inf * wake_measure)
            beta = 1.0
            if sound_speed is not None and float(sound_speed) > 0.0:
                beta_sq = 1.0 - (float(velocity_inf) / float(sound_speed)) ** 2
                if np.isfinite(beta_sq) and beta_sq > 0.0:
                    beta = float(np.sqrt(beta_sq))
            wake_lift = wake_lift_incomp / beta
            reconstructed_lift_ratio = (pressure_lift * beta) / (wake_lift_incomp + 1e-8)
            solver_lift_ratio = (
                float(trefftz_lift_ratio)
                if trefftz_lift_ratio is not None and np.isfinite(trefftz_lift_ratio)
                else reconstructed_lift_ratio
            )
            record.update(
                beta=beta,
                pressure_half_lift=pressure_lift,
                wake_raw_measure=wake_measure,
                wake_lift_estimate=wake_lift,
                pressure_wake_lift_ratio=pressure_lift / wake_lift if wake_lift > 0 else np.nan,
                # Exact solver value: L_incomp / (L_wake + 1e-8), before squaring.
                trefftz_lift_ratio_correction=solver_lift_ratio,
                reconstructed_lift_ratio_correction=reconstructed_lift_ratio,
                mismatch_l2=float(np.sqrt(b * np.trapz(difference ** 2, self.eta) / denominator)),
                mismatch_max=float(np.max(np.abs(difference)) / np.max(np.abs(pressure))),
                pressure_curve=pressure,
                wake_curve=wake,
                half_span=b,
                valid=True,
            )
            self.final_wake_endpoints = np.asarray(endpoints)
            strengths = np.asarray(mu_w)
            self.final_mu_w = strengths[0] if strengths.ndim > 1 else strengths
        except (ValueError, FloatingPointError) as exc:
            print(f"Warning: wake-load diagnostic failed at iteration {iteration}: {exc}")
            n = self.eta.size
            record.update(pressure_half_lift=np.nan, wake_raw_measure=np.nan,
                          wake_lift_estimate=np.nan, pressure_wake_lift_ratio=np.nan,
                          trefftz_lift_ratio_correction=np.nan,
                          reconstructed_lift_ratio_correction=np.nan,
                          mismatch_l2=np.nan, mismatch_max=np.nan,
                          pressure_curve=np.full(n, np.nan), wake_curve=np.full(n, np.nan),
                          half_span=np.nan, valid=False)
        self.records.append(record)

    def save_and_plot(self, output_folder):
        output = Path(output_folder)
        output.mkdir(parents=True, exist_ok=True)
        if not self.records:
            raise ValueError("no wake-load diagnostic records were accumulated")
        records = self.records
        arrays = {
            "iterations": np.array([r["iteration"] for r in records]),
            "eta": self.eta,
            "cl": np.array([r["cl"] for r in records]),
            "cdi_fourier": np.array([r["cdi_fourier"] for r in records]),
            "cdi_trefftz": np.array([r["cdi_trefftz"] for r in records]),
            "beta": np.array([r.get("beta", 1.0) for r in records]),
            "pressure_half_lift": np.array([r["pressure_half_lift"] for r in records]),
            "wake_raw_measure": np.array([r["wake_raw_measure"] for r in records]),
            "wake_lift_estimate": np.array([r["wake_lift_estimate"] for r in records]),
            "pressure_wake_lift_ratio": np.array([r["pressure_wake_lift_ratio"] for r in records]),
            "trefftz_lift_ratio_correction": np.array([r["trefftz_lift_ratio_correction"] for r in records]),
            "reconstructed_lift_ratio_correction": np.array([r["reconstructed_lift_ratio_correction"] for r in records]),
            "mismatch_l2": np.array([r["mismatch_l2"] for r in records]),
            "mismatch_max": np.array([r["mismatch_max"] for r in records]),
            "pressure_curve": np.array([r["pressure_curve"] for r in records]),
            "wake_curve_scaled": np.array([r["wake_curve"] for r in records]),
            "valid": np.array([r["valid"] for r in records], dtype=bool),
            "final_wake_endpoints": self.final_wake_endpoints,
            "final_mu_w": self.final_mu_w,
        }
        data_path = output / "wake_load_consistency_data.npz"
        np.savez_compressed(data_path, **arrays)

        import matplotlib.pyplot as plt
        final = records[-1]
        fig, (ax_load, ax_hist) = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
        if final["valid"]:
            ellipse = final["pressure_half_lift"] / (final["half_span"] * np.pi / 4.0) * np.sqrt(np.clip(1.0 - self.eta ** 2, 0.0, 1.0))
        else:
            ellipse = np.full_like(self.eta, np.nan)
        ax_load.plot(self.eta, final["pressure_curve"], label="Pressure-panel loading")
        ax_load.plot(self.eta, final["wake_curve"], label="Scaled wake loading")
        ax_load.plot(self.eta, ellipse, "--", label="Elliptical reference")
        ax_load.set_ylabel("Half-wing loading density")
        ax_load.set_title("Near-field pressure loading vs. Trefftz wake loading")
        ax_load.grid(True, linestyle=":")
        ax_load.legend()

        iterations = arrays["iterations"]
        ax_hist.plot(iterations, arrays["mismatch_l2"], label="Normalized L2 mismatch")
        ax_hist.plot(iterations, arrays["mismatch_max"], label="Maximum normalized mismatch")
        ax_hist.set_xlabel("Optimization iteration")
        ax_hist.set_ylabel("Loading mismatch")
        ax_hist.grid(True, linestyle=":")
        ax_right = ax_hist.twinx()
        ax_right.plot(iterations, arrays["cl"], "k--", label="$C_L$")
        ax_right.plot(iterations, (arrays["cdi_fourier"] - arrays["cdi_trefftz"]) * 1e4, "r-", label="$C_{Di,F}-C_{Di,T}$ [counts]")
        ax_right.set_ylabel("$C_L$ / induced-drag separation")
        h1, l1 = ax_hist.get_legend_handles_labels()
        h2, l2 = ax_right.get_legend_handles_labels()
        ax_hist.legend(h1 + h2, l1 + l2, loc="best")
        if final["valid"]:
            p_excess = np.trapz(np.maximum(final["pressure_curve"] - final["wake_curve"], 0.0), self.eta)
            w_excess = np.trapz(np.maximum(final["wake_curve"] - final["pressure_curve"], 0.0), self.eta)
            dominant = "pressure loading exceeds wake loading" if p_excess >= w_excess else "wake loading exceeds pressure loading"
            ax_load.text(0.02, 0.96, f"Final L2 mismatch = {final['mismatch_l2']:.3g}; {dominant}", transform=ax_load.transAxes, va="top")
        else:
            ax_load.text(0.02, 0.96, "Final diagnostic data invalid", transform=ax_load.transAxes, va="top")
        figure_path = output / "wake_load_consistency.png"
        fig.savefig(figure_path, dpi=180)
        plt.close(fig)
        return data_path, figure_path


class WakeCirculationClosureHistory:
    """Diagnose local closure between pressure lift and signed TE circulation."""

    def __init__(self, grid_size=201):
        self.eta = np.linspace(0.0, 1.0, grid_size)
        self.records: list[dict[str, Any]] = []
        self.final_raw = {}

    def record(self, iteration, panel_centers, panel_forces, panel_lift, panel_mesh,
               mu_w, te_edges, rho_inf, velocity_inf, sound_speed, cl,
               cdi_fourier, cdi_trefftz, trefftz_lift_ratio):
        record = dict(iteration=int(iteration), cl=float(cl), cdi_fourier=float(cdi_fourier),
                      cdi_trefftz=float(cdi_trefftz), valid=False)
        try:
            yp, fz_density, fz_total, bp = aggregate_pressure_loading(panel_centers, panel_forces)
            _, normal_density, normal_total, _ = aggregate_strip_loading(panel_centers, panel_lift)
            yw, signed_mu, widths, endpoints = extract_signed_wake_loading(panel_mesh, mu_w, te_edges)
            beta_sq = 1.0 - (float(velocity_inf) / float(sound_speed)) ** 2
            if not np.isfinite(beta_sq) or beta_sq <= 0.0:
                raise ValueError("invalid Prandtl-Glauert beta")
            beta = float(np.sqrt(beta_sq))
            pressure_incomp_density = beta * normal_density
            pressure_incomp_total = beta * normal_total
            wake_signed_total = float(rho_inf * velocity_inf * np.sum(signed_mu * widths))
            wake_abs_total = float(rho_inf * velocity_inf * np.sum(np.abs(signed_mu) * widths))
            if abs(pressure_incomp_total) <= 1e-10 or abs(wake_signed_total) <= 1e-10:
                raise ValueError("zero or sign-ambiguous pressure/wake lift")
            sign = float(np.sign(pressure_incomp_total * wake_signed_total))
            if sign == 0.0:
                raise ValueError("unable to resolve a global wake sign")
            b = max(bp, float(np.max(yw)))
            fz = _curve_on_grid(yp, fz_density, b, self.eta, fz_total)
            normal = _curve_on_grid(yp, normal_density, b, self.eta, normal_total)
            pressure_incomp = _curve_on_grid(yp, pressure_incomp_density, b, self.eta, pressure_incomp_total)
            wake_raw_density = sign * float(rho_inf) * float(velocity_inf) * signed_mu
            wake_raw = _curve_on_grid(yw, wake_raw_density, b, self.eta, sign * wake_signed_total)
            wake_corrected = float(trefftz_lift_ratio) * wake_raw
            fz_shape = _shape_curve(fz, b)
            normal_shape = _shape_curve(normal, b)
            pressure_shape = _shape_curve(pressure_incomp, b)
            wake_shape = _shape_curve(wake_corrected, b)
            vertical_wake_l2 = _relative_l2(fz_shape, wake_shape, b)
            projection_l2 = _relative_l2(fz_shape, normal_shape, b)
            closure_shape_l2 = _relative_l2(pressure_shape, wake_shape, b)
            closure_l2 = _relative_l2(pressure_incomp, wake_corrected, b)
            closure_max = float(np.max(np.abs(pressure_incomp - wake_corrected)) /
                                np.max(np.abs(pressure_incomp)))
            category = (
                "projection-dominated"
                if closure_shape_l2 <= 0.5 * vertical_wake_l2
                else "wake-pressure-inconsistent"
            )
            record.update(
                beta=beta,
                wake_sign=sign,
                fz_half_lift=fz_total,
                normal_pressure_half_lift=normal_total,
                incompressible_pressure_half_lift=pressure_incomp_total,
                wake_signed_half_lift=wake_signed_total,
                wake_absolute_half_lift=wake_abs_total,
                trefftz_lift_ratio_correction=float(trefftz_lift_ratio),
                vertical_wake_shape_l2=vertical_wake_l2,
                projection_shape_l2=projection_l2,
                closure_shape_l2=closure_shape_l2,
                closure_l2=closure_l2,
                closure_max=closure_max,
                fz_curve=fz,
                normal_curve=normal,
                pressure_incomp_curve=pressure_incomp,
                wake_raw_curve=wake_raw,
                wake_corrected_curve=wake_corrected,
                half_span=b,
                category=category,
                valid=True,
            )
            self.final_raw = dict(
                strip_eta=yp / b,
                strip_fz_density=fz_density,
                strip_normal_lift_density=normal_density,
                wake_eta=yw / b,
                wake_signed_mu=signed_mu,
                wake_widths=widths,
                wake_endpoints=endpoints,
            )
        except (ValueError, FloatingPointError) as exc:
            print(f"Warning: wake-circulation closure failed at iteration {iteration}: {exc}")
            n = self.eta.size
            record.update(**{key: np.nan for key in (
                "beta", "wake_sign", "fz_half_lift", "normal_pressure_half_lift",
                "incompressible_pressure_half_lift", "wake_signed_half_lift",
                "wake_absolute_half_lift", "trefftz_lift_ratio_correction",
                "vertical_wake_shape_l2", "projection_shape_l2", "closure_shape_l2",
                "closure_l2", "closure_max", "half_span")},
                fz_curve=np.full(n, np.nan), normal_curve=np.full(n, np.nan),
                pressure_incomp_curve=np.full(n, np.nan), wake_raw_curve=np.full(n, np.nan),
                wake_corrected_curve=np.full(n, np.nan), category="invalid")
        self.records.append(record)

    def arrays(self):
        if not self.records:
            raise ValueError("no wake-circulation closure records were accumulated")
        scalar_keys = [
            "cl", "cdi_fourier", "cdi_trefftz", "beta", "wake_sign", "fz_half_lift",
            "normal_pressure_half_lift", "incompressible_pressure_half_lift",
            "wake_signed_half_lift", "wake_absolute_half_lift", "trefftz_lift_ratio_correction",
            "vertical_wake_shape_l2", "projection_shape_l2", "closure_shape_l2", "closure_l2",
            "closure_max", "half_span",
        ]
        curve_keys = ["fz_curve", "normal_curve", "pressure_incomp_curve", "wake_raw_curve", "wake_corrected_curve"]
        arrays = {"iterations": np.array([r["iteration"] for r in self.records]), "eta": self.eta,
                  "valid": np.array([r["valid"] for r in self.records], dtype=bool),
                  "category": np.array([r["category"] for r in self.records])}
        arrays.update({key: np.array([r[key] for r in self.records]) for key in scalar_keys + curve_keys})
        arrays.update({f"final_{key}": value for key, value in self.final_raw.items()})
        return arrays

    def save_and_plot(self, output_folder):
        output = Path(output_folder)
        output.mkdir(parents=True, exist_ok=True)
        arrays = self.arrays()
        data_path = output / "wake_circulation_closure_data.npz"
        np.savez_compressed(data_path, **arrays)
        import matplotlib.pyplot as plt
        final = self.records[-1]
        fig, axes = plt.subplots(3, 1, figsize=(10, 11), constrained_layout=True)
        if final["valid"]:
            beta = final.get("beta", 1.0)
            if not np.isfinite(beta) or beta <= 0.0:
                beta = 1.0
            wake_corr_comp = final["wake_corrected_curve"] / beta
            axes[0].plot(self.eta, final["fz_curve"], label="Fourier input $F_z$")
            axes[0].plot(self.eta, final["normal_curve"], label="Pressure lift normal to flow")
            axes[0].plot(self.eta, wake_corr_comp, label="Solver-corrected wake lift (compressible)")
            if beta < 0.999:
                axes[0].plot(self.eta, final["pressure_incomp_curve"], ":", color="gray", label=r"Incompressible pressure lift ($\beta L$)")
            axes[1].plot(self.eta, final["pressure_incomp_curve"] - final["wake_corrected_curve"], color="tab:red", label=r"Incompressible: $\beta L_{\mathrm{norm}} - L_{\mathrm{wake}}$")
            axes[1].axhline(0.0, color="black", linewidth=0.8)
            axes[0].text(0.02, 0.95, f"{final['category']}; max closure residual = {final['closure_max']:.3g}",
                         transform=axes[0].transAxes, va="top")
        else:
            axes[0].text(0.02, 0.95, "Final closure data invalid", transform=axes[0].transAxes, va="top")
        axes[0].set_ylabel("Sectional loading [N/m]")
        axes[1].set_ylabel("Incompressible pressure − wake [N/m]")
        axes[1].set_xlabel(r"Normalized half-span, $\eta$")
        axes[0].legend(); axes[0].grid(True, linestyle=":"); axes[1].grid(True, linestyle=":")
        iterations = arrays["iterations"]
        axes[2].plot(iterations, arrays["projection_shape_l2"], label="$F_z$ vs normal-pressure shape")
        axes[2].plot(iterations, arrays["closure_shape_l2"], label="Pressure vs wake shape")
        axes[2].plot(iterations, arrays["closure_max"], label="Maximum closure residual")
        axes[2].set_xlabel("Optimization iteration"); axes[2].set_ylabel("Normalized mismatch")
        axes[2].grid(True, linestyle=":")
        right = axes[2].twinx()
        right.plot(iterations, arrays["cl"], "k--", label="$C_L$")
        right.plot(iterations, (arrays["cdi_fourier"] - arrays["cdi_trefftz"]) * 1e4,
                   color="tab:purple", label="$C_{Di,F}-C_{Di,T}$ [counts]")
        right.set_ylabel("$C_L$ / drag separation")
        h1, l1 = axes[2].get_legend_handles_labels(); h2, l2 = right.get_legend_handles_labels()
        axes[2].legend(h1 + h2, l1 + l2, loc="best")
        figure_path = output / "wake_circulation_closure.png"
        fig.savefig(figure_path, dpi=180)
        plt.close(fig)
        return data_path, figure_path


def find_latest_output_folder():
    """Return the newest BWB optimization folder containing an x.out history."""
    output_base = REPO_ROOT / "rectangular_wing_to_bwb_aerostructural_optimization_outputs"
    candidates = [path for path in output_base.iterdir() if path.is_dir() and (path / "x.out").is_file()]
    if not candidates:
        raise FileNotFoundError(f"No optimization output folders containing x.out found in {output_base}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _validate_history_width(history, main_script, output):
    expected = sum(int(np.prod(info.variable.shape)) for info in main_script.design_variables.values())
    if history.shape[1] != expected:
        raise ValueError(
            f"{output} has {history.shape[1]} design variables, but the current BWB configuration "
            f"expects {expected}. Historical configuration metadata is required for this run."
        )


def _set_design(main_script, x_scaled):
    cursor = 0
    for dv_info in main_script.design_variables.values():
        size = int(np.prod(dv_info.variable.shape))
        main_script.jax_sim[dv_info.variable] = (x_scaled[cursor:cursor + size] / dv_info.scaler).reshape(dv_info.variable.shape)
        cursor += size


def replay_output_folder(output_folder, run_refinement=True):
    """Replay x.out for an existing run and regenerate diagnostic artifacts."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import importlib
    main_script = importlib.import_module("examples.showcase_examples.rectangular_wing.ex_rectangular_wing_to_bwb")
    output = Path(output_folder)
    history = np.loadtxt(output / "x.out")
    history = history.reshape(1, -1) if history.ndim == 1 else history
    diagnostic = WakeLoadConsistencyHistory()
    closure = WakeCirculationClosureHistory()
    _validate_history_width(history, main_script, output)
    for iteration, x_scaled in enumerate(history):
        _set_design(main_script, x_scaled)
        main_script.jax_sim.run()
        diagnostic.record(
            iteration, np.asarray(main_script.jax_sim[main_script.dynamic_panel_centers_right]),
            np.asarray(main_script.jax_sim[main_script.panel_forces_right_cruise]),
            np.asarray(main_script.jax_sim[main_script.panel_mesh]),
            np.asarray(main_script.jax_sim[main_script.mu_w]), np.asarray(main_script.TE_properties[2]),
            float(np.asarray(main_script.rho_array.value).reshape(-1)[0]),
            float(np.asarray(main_script.cruise_speed.value).reshape(-1)[0]),
            float(np.asarray(main_script.jax_sim[main_script.CL]).reshape(-1)[0]),
            float(np.asarray(main_script.jax_sim[main_script.CDi_Fourier]).reshape(-1)[0]),
            float(np.asarray(main_script.jax_sim[main_script.CDi]).reshape(-1)[0]),
            float(np.asarray(main_script.jax_sim[main_script.trefftz_lift_ratio]).reshape(-1)[0]),
            sound_speed=float(np.asarray(main_script.sos_array.value).reshape(-1)[0]),
        )
        closure.record(
            iteration, np.asarray(main_script.jax_sim[main_script.dynamic_panel_centers_right]),
            np.asarray(main_script.jax_sim[main_script.panel_forces_right_cruise]),
            np.asarray(main_script.jax_sim[main_script.panel_lift_right_cruise]),
            np.asarray(main_script.jax_sim[main_script.panel_mesh]),
            np.asarray(main_script.jax_sim[main_script.mu_w]), np.asarray(main_script.TE_properties[2]),
            float(np.asarray(main_script.rho_array.value).reshape(-1)[0]),
            float(np.asarray(main_script.cruise_speed.value).reshape(-1)[0]),
            float(np.asarray(main_script.sos_array.value).reshape(-1)[0]),
            float(np.asarray(main_script.jax_sim[main_script.CL]).reshape(-1)[0]),
            float(np.asarray(main_script.jax_sim[main_script.CDi_Fourier]).reshape(-1)[0]),
            float(np.asarray(main_script.jax_sim[main_script.CDi]).reshape(-1)[0]),
            float(np.asarray(main_script.jax_sim[main_script.trefftz_lift_ratio]).reshape(-1)[0]),
        )
    paths = diagnostic.save_and_plot(output) + closure.save_and_plot(output)
    if run_refinement:
        _run_refinement_if_needed(output, history, closure)
    return paths


def _run_refinement_worker(output_folder, iteration, mesh_name, result_path):
    """Evaluate one saved fast-FFD design in a clean process on one aero mesh."""
    import importlib
    main_script = importlib.import_module("examples.showcase_examples.rectangular_wing.ex_rectangular_wing_to_bwb")
    output = Path(output_folder)
    history = np.loadtxt(output / "x.out")
    history = history.reshape(1, -1) if history.ndim == 1 else history
    _validate_history_width(history, main_script, output)
    _set_design(main_script, history[int(iteration)])
    main_script.jax_sim.run()
    closure = WakeCirculationClosureHistory()
    closure.record(
        int(iteration), np.asarray(main_script.jax_sim[main_script.dynamic_panel_centers_right]),
        np.asarray(main_script.jax_sim[main_script.panel_forces_right_cruise]),
        np.asarray(main_script.jax_sim[main_script.panel_lift_right_cruise]),
        np.asarray(main_script.jax_sim[main_script.panel_mesh]),
        np.asarray(main_script.jax_sim[main_script.mu_w]), np.asarray(main_script.TE_properties[2]),
        float(np.asarray(main_script.rho_array.value).reshape(-1)[0]),
        float(np.asarray(main_script.cruise_speed.value).reshape(-1)[0]),
        float(np.asarray(main_script.sos_array.value).reshape(-1)[0]),
        float(np.asarray(main_script.jax_sim[main_script.CL]).reshape(-1)[0]),
        float(np.asarray(main_script.jax_sim[main_script.CDi_Fourier]).reshape(-1)[0]),
        float(np.asarray(main_script.jax_sim[main_script.CDi]).reshape(-1)[0]),
        float(np.asarray(main_script.jax_sim[main_script.trefftz_lift_ratio]).reshape(-1)[0]),
    )
    arrays = closure.arrays()
    arrays["mesh_name"] = np.array(mesh_name)
    arrays["te_edge_count"] = np.array(len(main_script.TE_properties[2]))
    np.savez_compressed(result_path, **arrays)


def _worker_measurement(output, iteration, mesh_name):
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as handle:
        result_path = Path(handle.name)
    environment = os.environ.copy()
    environment.setdefault("JAX_PLATFORMS", "cpu")
    environment["BWB_AERO_MESH_FILE"] = mesh_name
    command = [sys.executable, str(Path(__file__)), "--worker", str(output),
               "--iteration", str(iteration), "--worker-output", str(result_path)]
    try:
        subprocess.run(command, check=True, env=environment)
        with np.load(result_path) as result:
            return {key: result[key] for key in result.files}
    finally:
        if result_path.exists():
            result_path.unlink()


def _run_refinement_if_needed(output, history, closure):
    arrays = closure.arrays()
    valid = arrays["valid"]
    if not np.any(valid):
        return None
    final_index = int(arrays["iterations"][-1])
    peak_index = int(arrays["iterations"][np.nanargmax(np.where(valid, arrays["closure_max"], np.nan))])
    trigger = max(float(arrays["closure_max"][-1]), float(np.nanmax(np.where(valid, arrays["closure_max"], np.nan))))
    if trigger < 0.05:
        return None
    selected = [final_index] if peak_index == final_index else [final_index, peak_index]
    measurements = []
    for iteration in selected:
        for label, mesh_name in (("native", "rectangular_wing_naca0012_10ar"),
                                 ("refined", "rectangular_wing_naca0012_35sect")):
            result = _worker_measurement(output, iteration, mesh_name)
            result["mesh_label"] = np.array(label)
            measurements.append(result)
    _save_refinement_artifacts(output, measurements)
    return measurements


def _save_refinement_artifacts(output_folder, measurements):
    output = Path(output_folder)
    cases = sorted({int(m["iterations"][0]) for m in measurements})
    rows = []
    for iteration in cases:
        native = next(m for m in measurements if int(m["iterations"][0]) == iteration and str(m["mesh_label"]) == "native")
        refined = next(m for m in measurements if int(m["iterations"][0]) == iteration and str(m["mesh_label"]) == "refined")
        native_shape, refined_shape = float(native["closure_shape_l2"][0]), float(refined["closure_shape_l2"][0])
        refined_max = float(refined["closure_max"][0])
        verdict = "discretization-sensitive" if refined_shape <= 0.5 * native_shape and refined_max < 0.05 else "persistent-closure-inconsistency"
        rows.append((iteration, native_shape, float(native["closure_max"][0]), refined_shape, refined_max, verdict))
    np.savez_compressed(
        output / "wake_circulation_refinement_data.npz",
        case_iterations=np.array([r[0] for r in rows]),
        native_closure_shape_l2=np.array([r[1] for r in rows]),
        native_closure_max=np.array([r[2] for r in rows]),
        refined_closure_shape_l2=np.array([r[3] for r in rows]),
        refined_closure_max=np.array([r[4] for r in rows]),
        verdict=np.array([r[5] for r in rows]),
        native_te_edge_count=np.array([int(next(m for m in measurements if int(m["iterations"][0]) == r[0] and str(m["mesh_label"]) == "native")["te_edge_count"]) for r in rows]),
        refined_te_edge_count=np.array([int(next(m for m in measurements if int(m["iterations"][0]) == r[0] and str(m["mesh_label"]) == "refined")["te_edge_count"]) for r in rows]),
        eta=np.asarray(measurements[0]["eta"]),
        native_pressure_incomp_curve=np.array([next(m for m in measurements if int(m["iterations"][0]) == r[0] and str(m["mesh_label"]) == "native")["pressure_incomp_curve"][0] for r in rows]),
        native_wake_corrected_curve=np.array([next(m for m in measurements if int(m["iterations"][0]) == r[0] and str(m["mesh_label"]) == "native")["wake_corrected_curve"][0] for r in rows]),
        refined_pressure_incomp_curve=np.array([next(m for m in measurements if int(m["iterations"][0]) == r[0] and str(m["mesh_label"]) == "refined")["pressure_incomp_curve"][0] for r in rows]),
        refined_wake_corrected_curve=np.array([next(m for m in measurements if int(m["iterations"][0]) == r[0] and str(m["mesh_label"]) == "refined")["wake_corrected_curve"][0] for r in rows]),
    )
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(rows), 2, figsize=(12, 4 * len(rows)), squeeze=False, constrained_layout=True)
    for row_index, row in enumerate(rows):
        iteration = row[0]
        for label, axis in zip(("native", "refined"), axes[row_index]):
            measurement = next(m for m in measurements if int(m["iterations"][0]) == iteration and str(m["mesh_label"]) == label)
            eta = measurement["eta"]
            axis.plot(eta, measurement["pressure_incomp_curve"][0], label="Incompressible pressure")
            axis.plot(eta, measurement["wake_corrected_curve"][0], label="Corrected wake")
            axis.set_title(f"iteration {iteration}, {label}, {int(measurement['te_edge_count'])} TE edges")
            axis.set_xlabel("Normalized half-span, $\\eta$"); axis.set_ylabel("Loading [N/m]")
            axis.grid(True, linestyle=":"); axis.legend()
        axes[row_index, 0].text(0.02, 0.95, row[5], transform=axes[row_index, 0].transAxes, va="top")
    figure_path = output / "wake_circulation_refinement.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return output / "wake_circulation_refinement_data.npz", figure_path


def _save_low_cl_comparison(output_folders):
    rows = []
    for folder in output_folders:
        data = np.load(Path(folder) / "wake_circulation_closure_data.npz")
        rows.append((Path(folder).name, float(data["cl"][-1]), float(data["closure_shape_l2"][-1]),
                     float(data["closure_max"][-1]), str(data["category"][-1])))
    output_base = Path(output_folders[0]).parent
    np.savez_compressed(output_base / "wake_circulation_closure_comparison.npz",
                        run_name=np.array([r[0] for r in rows]), cl=np.array([r[1] for r in rows]),
                        closure_shape_l2=np.array([r[2] for r in rows]), closure_max=np.array([r[3] for r in rows]),
                        category=np.array([r[4] for r in rows]))
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    labels = [r[0].split("_")[-1] for r in rows]
    x = np.arange(len(rows))
    axis.bar(x - 0.18, [r[2] for r in rows], width=0.36, label="Closure shape $L^2$")
    axis.bar(x + 0.18, [r[3] for r in rows], width=0.36, label="Maximum closure residual")
    axis.set_xticks(x, labels); axis.set_ylabel("Normalized mismatch"); axis.set_xlabel("Low-$C_L$ run")
    axis.grid(True, axis="y", linestyle=":"); axis.legend()
    fig.savefig(output_base / "wake_circulation_closure_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_folders", nargs="*", help="optimization output folders containing x.out")
    parser.add_argument("--all-low-cl", action="store_true", help="replay the three compatible low-CL runs")
    parser.add_argument("--no-refinement", action="store_true", help="skip automatic native/refined mesh comparison")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--iteration", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if len(args.output_folders) != 1 or args.iteration is None or not args.worker_output:
            parser.error("worker requires one output folder, --iteration, and --worker-output")
        _run_refinement_worker(args.output_folders[0], args.iteration,
                               os.environ.get("BWB_AERO_MESH_FILE", "rectangular_wing_naca0012_10ar"),
                               args.worker_output)
    else:
        if args.all_low_cl:
            output_base = find_latest_output_folder().parent
            output_folders = [output_base / name for name in LOW_CL_RUN_NAMES]
        elif args.output_folders:
            output_folders = [Path(folder) for folder in args.output_folders]
        else:
            output_folders = [find_latest_output_folder()]
        for output_folder in output_folders:
            print(f"Target optimization output directory: {output_folder}")
            paths = replay_output_folder(output_folder, run_refinement=not args.no_refinement)
            print("Saved " + ", ".join(str(path) for path in paths))
        if args.all_low_cl:
            _save_low_cl_comparison(output_folders)
