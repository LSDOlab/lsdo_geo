"""Unit tests for viscous drag diagnostics visualization tool."""

import os
import sys
import tempfile
import pytest
import numpy as np

# Ensure path to codebase
REPO_ROOT = '/home/andrew/optimization/lsdo_geo'
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from examples.showcase_examples.rectangular_wing.optimization_analyses.plot_viscous_drag_diagnostics import (
    plot_viscous_drag_diagnostics,
    process_viscous_drag_fields,
    generate_viscous_drag_figure,
    load_viscous_telemetry,
    find_latest_output_dir,
)


def test_process_viscous_drag_fields_synthetic():
    """Verify processing of viscous drag fields and generation of diagnostics figure."""
    num_strips = 100
    num_stations = 14
    half_span = 33.0

    y_strip = np.linspace(0.2, half_span, num_strips)
    c_strip = np.linspace(25.0, 1.8, num_strips)
    dy_strip = np.full(num_strips, half_span / num_strips)
    strip_area = 2.0 * c_strip * dy_strip

    st_y = np.linspace(1.0, half_span, num_stations)
    cd_sections = np.linspace(0.0042, 0.0072, num_stations)
    theta_u = np.linspace(0.038, 0.002, num_stations)
    theta_l = np.linspace(0.032, 0.003, num_stations)

    H_upper_matrix = np.full((20, num_stations), 1.50)

    synthetic_dict = {
        'viscous_drag_mode': 'ibl',
        'scale_factor': 7.5,
        'sref_val': np.sum(strip_area),
        'y_strip_pts': y_strip,
        'local_chord_drag': c_strip,
        'strip_area': strip_area,
        'dy_strip': dy_strip,
        'cd_ibl_section': cd_sections,
        'theta_te_upper': theta_u,
        'theta_te_lower': theta_l,
        'H_upper_matrix': H_upper_matrix,
        'cd_viscous_val': 0.0052,
        'd_viscous_val': 38000.0,
        'd_total_val': 64000.0,
        'cdi_val': 0.0034,
        'cd_wave_val': 0.00016,
    }

    proc = process_viscous_drag_fields(synthetic_dict)

    assert 'cd_strip_visc' in proc
    assert len(proc['cd_strip_visc']) == num_strips
    assert np.all(np.isfinite(proc['cd_strip_visc']))

    assert 'dprime_visc_ibl' in proc
    assert len(proc['dprime_visc_ibl']) == num_strips
    assert np.all(np.isfinite(proc['dprime_visc_ibl']))

    assert 'theta_te_upper' in proc
    assert len(proc['theta_te_upper']) == num_stations

    assert 're_strip' in proc
    assert proc['re_strip'][0] > proc['re_strip'][-1]  # Root Re > Tip Re

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_fig = os.path.join(tmp_dir, 'viscous_drag_diagnostics.png')
        generate_viscous_drag_figure(proc, out_fig)
        assert os.path.exists(out_fig)
        assert os.path.getsize(out_fig) > 10000


def test_find_latest_output_dir_resolution():
    """Verify that find_latest_output_dir handles auto and manual directory queries."""
    # Automatic query should find a directory with telemetry
    latest_auto = find_latest_output_dir()
    assert os.path.isdir(latest_auto)
    assert os.path.exists(os.path.join(latest_auto, 'lift_and_moment_data.npz'))

    # Substring query
    cand = find_latest_output_dir("2026-10-09")
    assert os.path.isdir(cand)
    assert "2026-10-09" in os.path.basename(cand)

