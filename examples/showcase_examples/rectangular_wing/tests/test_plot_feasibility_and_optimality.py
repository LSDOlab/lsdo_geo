"""Unit tests for optimization feasibility and optimality history plotting."""

import os
import sys
import tempfile
import pytest
import numpy as np

# Ensure path to codebase
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from examples.showcase_examples.rectangular_wing.optimization_analyses.plot_feasibility_and_optimality import (
    OptimizationConvergenceHistory,
    load_convergence_data,
    generate_convergence_plot,
    find_target_output_dir,
    plot_feasibility_and_optimality,
    CONSTRAINT_STYLE,
    DV_STYLE,
)


def test_optimization_convergence_history_synthetic():
    """Verify in-situ recording, data aggregation, and plotting on synthetic optimization trajectory."""
    dvs_metadata = {
        'chord_stretch_dvs': 2,
        'span_stretch_dv': 1,
        'sweep_dvs': 1,
        'twist_dvs': 1,
        'thickness_stretch_dvs': 1,
        'camber_dvs': 1,
        'ttop_dvs': 2,
        'pitch': 1,
        'payload_center_x': 1,
    }
    history = OptimizationConvergenceHistory(design_variables=dvs_metadata)

    # Simulate 4 iterations
    for it in range(4):
        decay = 0.5 ** it
        x_scaled = np.array([
            1.0 + 0.1 * it, 1.0 - 0.05 * it,  # chord
            1.0 + 0.02 * it,                 # span
            15.0 - 1.0 * it,                 # sweep
            -1.0 + 0.2 * it,                 # twist
            0.15 - 0.01 * it,                # thickness
            0.02 + 0.005 * it,               # camber
            0.01 + 0.001 * it, 0.008,        # ttop
            2.5 - 0.1 * it,                  # pitch
            12.0 + 0.05 * it,                # payload
        ])

        sim_data = {
            'dv_stratford_constraints': np.array([0.38 + 0.1 * decay, 0.35]),
            'stresses_to_enforce': np.array([2.4e8 * (1.0 + 0.2 * decay)]),
            'cl_res': np.array([0.05 * decay]),
            'cm_res': np.array([0.01 * decay]),
            'static_margin': np.array([0.02 - 0.01 * decay]),
            'f_attached_min_ss': np.array([0.70 - 0.1 * decay]),
            'geonic_payload_margin': np.array([-0.05 + 0.1 * decay]),
            'aspect_ratio': np.array([12.0]),
            't_over_c': np.array([0.12]),
        }

        grad = np.ones(11) * (0.1 * decay)
        jac = np.ones((7, 11)) * 0.05
        mult = np.ones(7) * (0.2 * decay)
        history.record(
            iteration=it,
            jax_sim=sim_data,
            x_scaled=x_scaled,
            gradient=grad,
            jacobian=jac,
            multipliers=mult,
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save and plot with metric='both'
        history.save_and_plot(tmp_dir, metric='both')

        npz_file = os.path.join(tmp_dir, 'feasibility_optimality_data.npz')
        assert os.path.exists(npz_file)

        fig_primary = os.path.join(tmp_dir, 'optimization_feasibility_optimality.png')
        fig_max = os.path.join(tmp_dir, 'optimization_feasibility_optimality_max.png')
        fig_norm = os.path.join(tmp_dir, 'optimization_feasibility_optimality_norm.png')

        for fig_path in [fig_primary, fig_max, fig_norm]:
            assert os.path.exists(fig_path)
            assert os.path.getsize(fig_path) > 30_000  # Valid non-empty PNG

        # Test reload
        data = load_convergence_data(tmp_dir)
        assert len(data['iterations']) == 4
        assert data.get('has_kkt_lagrangian') is True

        # Check constraint and DV groups presence
        for c_group in CONSTRAINT_STYLE.keys():
            assert f'con_{c_group}_max' in data
            assert f'con_{c_group}_norm' in data
            assert len(data[f'con_{c_group}_max']) == 4

        for dv_group in DV_STYLE.keys():
            assert f'dv_{dv_group}_max' in data
            assert f'dv_{dv_group}_norm' in data
            assert len(data[f'dv_{dv_group}_max']) == 4


def test_plot_feasibility_and_optimality_on_cached_run():
    """Verify execution on cached optimization run if present."""
    try:
        target_dir = find_target_output_dir()
    except FileNotFoundError:
        pytest.skip("No optimization output directory available.")

    with tempfile.TemporaryDirectory() as tmp_artifact_dir:
        result_fig = plot_feasibility_and_optimality(
            output_folder=target_dir,
            artifact_dir=tmp_artifact_dir,
            metric='max',
        )
        assert os.path.exists(result_fig)
        assert os.path.getsize(result_fig) > 30_000

        artifact_fig = os.path.join(tmp_artifact_dir, 'optimization_feasibility_optimality.png')
        assert os.path.exists(artifact_fig)


def test_kkt_lagrangian_gradient_loading():
    """Verify loading and computing df/dx + lambda * dc/dx from slsqp_recorder.hdf5."""
    import h5py

    with tempfile.TemporaryDirectory() as tmp_dir:
        hdf5_path = os.path.join(tmp_dir, 'slsqp_recorder.hdf5')
        nx = 84
        nc = 10
        num_iters = 3

        with h5py.File(hdf5_path, 'w') as hf:
            for it in range(num_iters):
                grp = hf.create_group(f'iter_{it}')
                grad = np.random.randn(nx) * (0.5 ** it)
                jac = np.random.randn(nc, nx)
                mult = np.random.rand(nc) * (0.5 ** it)
                grp['gradient'] = grad
                grp['jacobian'] = jac
                grp['multipliers'] = mult
                grp['x'] = np.ones(nx) * it
                grp['constraints'] = np.zeros(nc)

        data = load_convergence_data(tmp_dir)
        assert data.get('has_kkt_lagrangian', False) is True
        assert len(data['iterations']) == num_iters
        assert 'dv_total_max' in data
        assert 'dv_total_norm' in data
        for dv_grp in DV_STYLE:
            assert f'dv_{dv_grp}_max' in data
            assert f'dv_{dv_grp}_norm' in data
            # Total must mathematically bound each individual group
            assert np.all(data['dv_total_max'] >= data[f'dv_{dv_grp}_max'] - 1e-12)
            assert np.all(data['dv_total_norm'] >= data[f'dv_{dv_grp}_norm'] - 1e-12)

        out_fig = os.path.join(tmp_dir, 'test_kkt_fig.png')
        generate_convergence_plot(data, out_fig, metric='max')
        assert os.path.exists(out_fig)
        assert os.path.getsize(out_fig) > 30_000
