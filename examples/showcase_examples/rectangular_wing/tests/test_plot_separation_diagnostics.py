"""Unit tests for boundary layer separation diagnostics and visualization tool."""

import os
import sys
import tempfile
import pytest
import numpy as np

# Ensure path to codebase
REPO_ROOT = '/home/andrew/optimization/lsdo_geo'
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from examples.showcase_examples.rectangular_wing.optimization_analyses.plot_separation_diagnostics import (
    plot_separation_diagnostics,
    process_separation_fields,
    load_separation_telemetry,
    S_CRIT,
    H_SEP,
)


def test_process_separation_fields_synthetic():
    """Verify 2D surface reconstruction, sectional cuts, and margins on synthetic data."""
    num_ffd = 5
    num_mesh = 14

    synthetic_dict = {
        'num_stations': num_ffd,
        'scale_factor': 7.5,
        'dv_stratford_constraints': np.full((4 * num_ffd,), 0.32),
        'H_section_upper': np.full((num_mesh,), 1.95),
        'dv_H_stations': np.full((num_ffd,), 1.90),
    }

    proc = process_separation_fields(synthetic_dict)

    assert proc['num_ffd_stations'] == num_ffd
    assert proc['num_mesh_stations'] == num_mesh
    assert proc['sc_matrix'].shape == (4, num_ffd)
    assert proc['S_dense'].shape == (60, 80)
    assert proc['H_dense'].shape == (60, 80)
    assert len(proc['sectional_data']) == 5
    assert len(proc['span_margin_S']) == 80
    assert len(proc['span_margin_H']) == 80
    assert proc['global_margin_S'] > 0.0  # S = 0.32 < S_crit = 0.39 -> Feasible
    assert proc['global_margin_H'] > 0.0  # H = 1.95 < H_sep = 2.40 -> Feasible
    assert proc['num_satisfied_S'] == 20


def test_plot_separation_diagnostics_on_cached_run():
    """Verify end-to-end execution on real optimization run directory."""
    latest_run_dir = '/home/andrew/optimization/lsdo_geo/rectangular_wing_to_bwb_aerostructural_optimization_outputs/2026-10-07_16.10.14.701860'
    if not os.path.isdir(latest_run_dir):
        pytest.skip(f"Test run directory {latest_run_dir} not available.")

    with tempfile.TemporaryDirectory() as tmp_art_dir:
        fig_path, npz_path = plot_separation_diagnostics(
            output_folder=latest_run_dir,
            artifact_dir=tmp_art_dir,
            force_rerun=True,
        )

        assert os.path.exists(fig_path)
        assert os.path.getsize(fig_path) > 100_000  # At least 100KB PNG

        assert os.path.exists(npz_path)
        data = np.load(npz_path, allow_pickle=True)
        assert 'S_dense' in data
        assert 'H_dense' in data
        assert 'sc_matrix' in data
        assert 'global_max_S' in data
        assert 'global_margin_S' in data
        assert 'global_max_H' in data
        assert 'global_margin_H' in data

        # Check artifact copies
        art_fig = os.path.join(tmp_art_dir, 'separation_diagnostics.png')
        art_npz = os.path.join(tmp_art_dir, 'separation_diagnostics_data.npz')
        assert os.path.exists(art_fig)
        assert os.path.exists(art_npz)

