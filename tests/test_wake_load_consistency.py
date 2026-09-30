import numpy as np

from examples.showcase_examples.rectangular_wing.optimization_analyses.wake_load_consistency import (
    WakeCirculationClosureHistory,
    WakeLoadConsistencyHistory,
    _curve_on_grid,
    aggregate_pressure_loading,
    extract_wake_loading,
    extract_signed_wake_loading,
)


def _synthetic_pressure(spike=False):
    y = np.repeat([0.375, 1.125], 40)
    centers = np.zeros((y.size, 3))
    centers[:, 1] = y
    force = np.ones(y.size) / 40.0
    if spike:
        force[:4] *= 4.0
    forces = np.zeros((y.size, 3))
    forces[:, 2] = force
    return centers, forces


def _synthetic_wake(reversed_edges=False, signed=False):
    mesh = np.zeros((4, 3))
    mesh[:, 1] = [0.0, 0.75, 0.75, 1.5]
    edges = np.array([[0, 1], [2, 3]])
    if reversed_edges:
        edges = edges[:, ::-1]
    strengths = np.array([1.0, 1.0])
    if signed:
        strengths *= -1.0
    return mesh, strengths, edges


def test_reversed_edges_and_signed_strengths_are_equivalent():
    mesh, strengths, edges = _synthetic_wake()
    reference = extract_wake_loading(mesh, strengths, edges)
    reversed_result = extract_wake_loading(mesh, -strengths, edges[:, ::-1])
    np.testing.assert_allclose(reference[0], reversed_result[0])
    np.testing.assert_allclose(reference[1], reversed_result[1])
    assert reference[2] == reversed_result[2]


def test_localized_inboard_spike_produces_mismatch():
    centers, forces = _synthetic_pressure(spike=True)
    mesh, strengths, edges = _synthetic_wake()
    history = WakeLoadConsistencyHistory()
    history.record(0, centers, forces, mesh, strengths, edges, 1.0, 1.0, 0.2, 0.01, 0.01,
                   trefftz_lift_ratio=1.234)
    assert history.records[0]["valid"]
    assert history.records[0]["mismatch_l2"] > 0.0
    assert history.records[0]["trefftz_lift_ratio_correction"] == 1.234


def test_zero_or_malformed_inputs_are_recorded_as_invalid():
    centers, forces = _synthetic_pressure()
    mesh, strengths, edges = _synthetic_wake()
    history = WakeLoadConsistencyHistory()
    history.record(0, np.empty((0, 3)), np.empty((0, 3)), mesh, strengths, edges, 1.0, 1.0, 0.2, 0.01, 0.01)
    assert not history.records[0]["valid"]
    assert np.isnan(history.records[0]["mismatch_l2"])

    zero_strengths = np.zeros_like(strengths)
    history.record(1, centers, forces, mesh, zero_strengths, edges, 1.0, 1.0, 0.2, 0.01, 0.01)
    assert not history.records[1]["valid"]


def test_identical_curves_have_zero_mismatch_after_normalization():
    grid = np.linspace(0.0, 1.0, 201)
    curve_a = _curve_on_grid([0.2, 0.6], [1.0, 0.5], 1.0, grid, 1.0)
    curve_b = _curve_on_grid([0.2, 0.6], [1.0, 0.5], 1.0, grid, 1.0)
    np.testing.assert_allclose(curve_a, curve_b)


def test_pressure_strip_aggregation_includes_only_40_panel_stations():
    centers, forces = _synthetic_pressure()
    y, density, lift, half_span = aggregate_pressure_loading(centers, forces)
    np.testing.assert_allclose(y, [0.375, 1.125])
    np.testing.assert_allclose(density, [1.0 / 0.75, 1.0 / 0.375])
    assert lift == 2.0
    assert half_span == 1.125


def test_signed_wake_preserves_local_sign_reversals():
    mesh, strengths, edges = _synthetic_wake()
    y, signed, widths, _ = extract_signed_wake_loading(mesh, [1.0, -0.25], edges)
    np.testing.assert_allclose(y, [0.375, 1.125])
    np.testing.assert_allclose(signed, [1.0, -0.25])
    np.testing.assert_allclose(widths, [0.75, 0.75])


def test_closure_matches_identical_pressure_and_wake_shapes():
    centers = np.zeros((80, 3)); centers[:, 1] = np.repeat([0.25, 0.75], 40)
    forces = np.zeros((80, 3)); forces[:, 2] = 1.0 / 40.0
    panel_lift = forces[:, 2].copy()
    mesh = np.zeros((4, 3)); mesh[:, 1] = [0.0, 0.5, 0.5, 1.0]
    edges = np.array([[0, 1], [2, 3]])
    closure = WakeCirculationClosureHistory()
    closure.record(0, centers, forces, panel_lift, mesh, np.array([2.0, 4.0]), edges,
                   1.0, 1.0, 1.e6, 0.2, 0.01, 0.01, 2.0 / 3.0)
    record = closure.records[0]
    assert record["valid"]
    assert record["closure_shape_l2"] < 1.e-8
    assert record["closure_max"] < 1.e-6


def test_closure_separates_vertical_projection_from_wake_loading():
    centers = np.zeros((80, 3)); centers[:, 1] = np.repeat([0.25, 0.75], 40)
    panel_lift = np.ones(80) / 40.0
    forces = np.zeros((80, 3)); forces[:, 2] = np.r_[np.full(40, 2.0 / 40.0), np.full(40, 0.5 / 40.0)]
    mesh = np.zeros((4, 3)); mesh[:, 1] = [0.0, 0.5, 0.5, 1.0]
    closure = WakeCirculationClosureHistory()
    closure.record(0, centers, forces, panel_lift, mesh, np.array([2.0, 4.0]), np.array([[0, 1], [2, 3]]),
                   1.0, 1.0, 1.e6, 0.2, 0.01, 0.01, 2.0 / 3.0)
    record = closure.records[0]
    assert record["valid"]
    assert record["projection_shape_l2"] > 0.1
    assert record["closure_shape_l2"] < 1.e-8
    assert record["category"] == "projection-dominated"
