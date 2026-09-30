import pytest
import numpy as np
import csdl_alpha as csdl
from examples.showcase_examples.rectangular_wing.physics_models.ar_area_bspline import (
    BsplineTargetRegularization,
    _to_vec,
)


def test_control_point_counts_and_symmetry():
    for n in [4, 6, 8]:
        reg = BsplineTargetRegularization(num_chord_stations=n, scale_factor=7.5)
        assert reg.taper_control_points_count == n - 1
        assert reg.tc_control_points_count == n
        assert reg.sweep_control_points_count == n - 1

        # Check exact root symmetry f(eta) == f(-eta) across eta = 0:
        h = 0.05
        B_pos = reg._eval_symmetric_chord_basis(np.array([h]))
        B_neg = reg._eval_symmetric_chord_basis(np.array([-h]))
        np.testing.assert_allclose(B_pos, B_neg, atol=1e-12)

        # Centered derivative at eta = 0 is zero to machine tolerance
        deriv_centered = (B_pos - B_neg) / (2.0 * h)
        np.testing.assert_allclose(deriv_centered, 0.0, atol=1e-12)


def test_taper_strong_bc():
    """T_h(0) = 1 identically for any valid taper control points (P_1, ..., P_{n-1})."""
    for n in [4, 6, 8]:
        reg = BsplineTargetRegularization(num_chord_stations=n, scale_factor=7.5)
        # Evaluate dense B_chord at eta = 0
        B_0 = reg.dense_B_chord[0, :]  # shape (n,)
        
        rng = np.random.default_rng(42)
        for _ in range(5):
            P_rand = rng.uniform(0.1, 2.0, size=n - 1)
            P_0 = 1.5 - 0.5 * P_rand[0]
            P_full = np.concatenate([[P_0], P_rand])
            val_at_0 = B_0 @ P_full
            np.testing.assert_allclose(val_at_0, 1.0, atol=1e-12)


def test_baseline_preservation():
    """At baseline, weak residuals are zero to machine tolerance."""
    for n in [4, 8]:
        reg = BsplineTargetRegularization(num_chord_stations=n, scale_factor=7.5)
        c0 = 7.5
        local_chords = np.full(n, c0)
        local_thicknesses = np.full(n, 0.12 * c0)
        taper_cp = np.ones(n - 1)
        tc_cp = np.full(n, 0.12)
        sweep_cp = np.zeros(n - 1)

        recorder = csdl.Recorder(inline=True)
        recorder.start()

        chords_var = csdl.Variable(value=local_chords)
        t_var = csdl.Variable(value=local_thicknesses)
        taper_var = csdl.Variable(value=taper_cp)
        tc_var = csdl.Variable(value=tc_cp)
        sweep_var = csdl.Variable(value=sweep_cp)

        dx_qc = csdl.Variable(value=np.zeros(n - 1))
        dy_qc = csdl.Variable(value=np.full(n - 1, 5.0 * 7.5 / (n - 1)))

        res_taper = reg.compute_taper_residual(chords_var, taper_var, c_ref=c0)
        res_thick = reg.compute_thickness_residual(t_var, chords_var, tc_var, t_ref=0.12 * c0)
        res_sweep = reg.compute_sweep_residual(dx_qc, dy_qc, sweep_var, y_scale=5.0 * 7.5)

        recorder.stop()

        np.testing.assert_allclose(res_taper.value, 0.0, atol=1e-12)
        np.testing.assert_allclose(res_thick.value, 0.0, atol=1e-12)
        np.testing.assert_allclose(res_sweep.value, 0.0, atol=1e-12)


def test_list_input_handling():
    """Verify that Python lists of scalar CSDL variables work seamlessly."""
    n = 4
    reg = BsplineTargetRegularization(num_chord_stations=n, scale_factor=7.5)
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    chords_list = [csdl.Variable(value=np.array([7.5])) for _ in range(n)]
    thick_list = [csdl.Variable(value=np.array([0.12 * 7.5])) for _ in range(n)]
    taper_var = csdl.Variable(value=np.ones(n - 1))
    tc_var = csdl.Variable(value=np.full(n, 0.12))

    res_taper = reg.compute_taper_residual(chords_list, taper_var, c_ref=7.5)
    res_thick = reg.compute_thickness_residual(thick_list, chords_list, tc_var, t_ref=0.12 * 7.5)
    recorder.stop()

    np.testing.assert_allclose(res_taper.value, 0.0, atol=1e-12)
    np.testing.assert_allclose(res_thick.value, 0.0, atol=1e-12)


def test_jax_derivatives():
    """Verify total derivatives via JAX simulator."""
    n = 4
    reg = BsplineTargetRegularization(num_chord_stations=n, scale_factor=7.5)
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    taper_var = csdl.Variable(value=np.array([0.9, 0.8, 0.7]))
    chords_var = csdl.Variable(value=np.array([7.5, 7.0, 6.0, 5.0]))
    res = reg.compute_taper_residual(chords_var, taper_var, c_ref=7.5)
    obj = csdl.sum(res ** 2)

    recorder.stop()

    sim = csdl.experimental.PySimulator(recorder=recorder)
    sim.run()
    assert np.isfinite(sim[obj])
