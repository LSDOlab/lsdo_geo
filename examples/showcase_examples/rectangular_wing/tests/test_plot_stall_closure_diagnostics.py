"""Unit tests for Stratford stall closure & reconciled Trefftz drag diagnostics tool."""

import os
import sys
import tempfile
import pytest
import numpy as np

# Ensure path to codebase
REPO_ROOT = '/home/andrew/optimization/lsdo_geo'
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from examples.showcase_examples.rectangular_wing.optimization_analyses.plot_stall_closure_diagnostics import (
    plot_stall_closure_diagnostics,
    process_stall_closure_fields,
    generate_stall_closure_figure,
    load_stall_closure_telemetry,
    find_latest_output_dir,
    S_CRIT,
)


def test_process_stall_closure_fields_synthetic():
    """Verify data processing and figure generation on synthetic telemetry."""
    num_panels = 280
    num_stations = 14
    half_span = 14.5

    # Build synthetic panel coordinates and forces
    y_vals = np.linspace(0.1, half_span, num_stations)
    panel_y = np.repeat(y_vals, 20)
    panel_x = np.tile(np.linspace(0.0, 3.0, 20), num_stations)
    panel_z = np.zeros(num_panels)
    centers = np.column_stack([panel_x, panel_y, panel_z])

    # Downward / upward forces
    f_cruise = np.zeros((num_panels, 3))
    f_cruise[:, 2] = -50.0  # ~14 kN lift

    f_ss = np.zeros((num_panels, 3))
    f_ss[:, 2] = -125.0  # 2.5g pull up

    synthetic_dict = {
        'stall_model': 'stratford_closure',
        'scale_factor': 7.5,
        'num_stations': 5,
        'f_attached_cruise': np.full(num_stations, 0.92),
        'f_attached_min_cruise': 0.88,
        'f_attached_ss': np.full(num_stations, 0.75),
        'f_attached_min_ss': 0.70,
        'd_separation_cruise': 420.0,
        'di_trefftz_reconciled': 3150.0,
        'cdi_trefftz_reconciled': 0.0125,
        'k_reconcile_cruise': 0.985,
        'res_wake_lift_cruise': -0.008,
        'd_total': 8500.0,
        'd_viscous': 4200.0,
        'd_wave': 730.0,
        'di_trefftz': 3200.0,
        'l_inviscid': 150000.0,
        'l_ss_inviscid': 375000.0,
        'panel_centers_right': centers,
        'panel_forces_right_cruise': f_cruise,
        'panel_forces_corr_cruise': f_cruise * 0.96,
        'panel_forces_right_ss': f_ss,
        'panel_forces_corr_ss': f_ss * 0.88,
        'y_strip_pts': y_vals,
        'strip_tc': np.full(num_stations, 0.12),
        'strip_area': np.full(num_stations, 2.5),
        'local_chord_drag': np.full(num_stations, 0.008),
        'mu_w_inviscid': np.ones((1, num_stations)),
        'mu_w_stall_cruise': np.full((1, num_stations), 0.95),
    }

    proc = process_stall_closure_fields(synthetic_dict)

    assert proc['stall_model'] == 'stratford_closure'
    assert proc['f_min_cruise'] == 0.88
    assert proc['f_min_ss'] == 0.70
    assert proc['d_sep'] == 420.0
    assert proc['di_reconciled'] == 3150.0
    assert proc['k_reconcile'] == 0.985
    assert proc['b_half'] == pytest.approx(half_span, abs=0.1)
    assert len(proc['f_cruise']) == num_stations

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_fig = os.path.join(tmp_dir, 'synthetic_stall_closure.png')
        generate_stall_closure_figure(proc, out_fig)
        assert os.path.exists(out_fig)
        assert os.path.getsize(out_fig) > 50_000  # Generated valid image


def test_plot_stall_closure_diagnostics_on_cached_run():
    """Verify end-to-end execution on real optimization run directory."""
    try:
        latest_run_dir = find_latest_output_dir()
    except FileNotFoundError:
        pytest.skip("No optimization output directory available.")

    with tempfile.TemporaryDirectory() as tmp_art_dir:
        fig_path, npz_path = plot_stall_closure_diagnostics(
            output_folder=latest_run_dir,
            artifact_dir=tmp_art_dir,
        )

        assert os.path.exists(fig_path)
        assert os.path.getsize(fig_path) > 100_000  # At least 100KB PNG

        assert os.path.exists(npz_path)
        data = np.load(npz_path, allow_pickle=True)
        assert 'f_attached_cruise' in data
        assert 'f_attached_min_cruise' in data
        assert 'd_total' in data
        assert 'd_viscous' in data

        # Check artifact copy
        art_fig = os.path.join(tmp_art_dir, 'stall_closure_diagnostics.png')
        assert os.path.exists(art_fig)

