import pytest
import numpy as np
import csdl_alpha as csdl
import lsdo_geo as lg
from lsdo_geo.optimization.optimization import (
    QuadraticPenaltyOperation,
    QuadraticPenalty,
    quadratic_penalty,
    Optimization,
)


def test_quadratic_penalty_forward_evaluation():
    """Verify forward evaluation: 0 if c <= 0, else c^T rho c."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    # Inactive scalar: c <= 0 -> 0.0
    c_neg = csdl.Variable(value=np.array([-3.5]))
    p_neg = quadratic_penalty(c_neg, penalty_factor=10.0)
    assert p_neg.value[0] == 0.0

    # Active scalar: c > 0 -> rho * c^2
    c_pos = csdl.Variable(value=np.array([2.0]))
    p_pos = quadratic_penalty(c_pos, penalty_factor=5.0)
    assert p_pos.value[0] == 5.0 * (2.0 ** 2)

    # Mixed vector: [2.0, -1.0, 3.0, 0.0]
    # Active: 2.0 (penalty 5*4=20), 3.0 (penalty 5*9=45) -> total = 65.0
    c_vec = csdl.Variable(value=np.array([2.0, -1.0, 3.0, 0.0]))
    p_vec = quadratic_penalty(c_vec, penalty_factor=5.0)
    assert p_vec.value[0] == 65.0

    # Vector rho
    rho_vec = csdl.Variable(value=np.array([2.0, 10.0, 4.0, 8.0]))
    p_vec_rho = quadratic_penalty(c_vec, penalty_factor=rho_vec)
    # Active: 2*2^2 + 4*3^2 = 8 + 36 = 44.0
    assert p_vec_rho.value[0] == 44.0

    recorder.stop()


def test_quadratic_penalty_derivatives_against_finite_differences():
    """Verify 1st and 2nd derivatives against centered finite differences on smooth branches."""
    eps = 1e-6
    c_val = np.array([2.5, -1.5, 3.2])
    rho_val = np.array([4.0])

    def evaluate_model(c_arr, rho_arr):
        rec = csdl.Recorder(inline=True)
        rec.start()
        c = csdl.Variable(value=c_arr)
        rho = csdl.Variable(value=rho_arr)
        p = quadratic_penalty(c, rho)
        d1 = csdl.derivative(p, c)
        d2 = csdl.derivative(d1, c)
        rec.stop()
        return p.value, d1.value.flatten(), d2.value

    p0, g0, H0 = evaluate_model(c_val, rho_val)

    # FD for 1st derivative (gradient)
    fd_g = np.zeros_like(c_val)
    for i in range(len(c_val)):
        c_p = c_val.copy()
        c_p[i] += eps
        c_m = c_val.copy()
        c_m[i] -= eps
        p_p, _, _ = evaluate_model(c_p, rho_val)
        p_m, _, _ = evaluate_model(c_m, rho_val)
        fd_g[i] = (p_p[0] - p_m[0]) / (2.0 * eps)

    np.testing.assert_allclose(g0, fd_g, rtol=1e-5, atol=1e-8)

    # FD for 2nd derivative (Hessian)
    fd_H = np.zeros((len(c_val), len(c_val)))
    for i in range(len(c_val)):
        c_p = c_val.copy()
        c_p[i] += eps
        c_m = c_val.copy()
        c_m[i] -= eps
        _, g_p, _ = evaluate_model(c_p, rho_val)
        _, g_m, _ = evaluate_model(c_m, rho_val)
        fd_H[i, :] = (g_p - g_m) / (2.0 * eps)

    np.testing.assert_allclose(H0, fd_H, rtol=1e-5, atol=1e-8)


def test_quadratic_penalty_higher_order_derivatives_vanish():
    """Verify that 3rd, 4th, and arbitrary higher-order derivatives are identically 0."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    c = csdl.Variable(value=np.array([2.0, -1.0, 3.0]))
    rho = csdl.Variable(value=np.array([10.0]))
    op = QuadraticPenaltyOperation()
    p = op.evaluate(c, rho)

    d1 = csdl.derivative(p, c)
    d2 = csdl.derivative(d1, c)
    d3 = csdl.derivative(d2, c)
    d4 = csdl.derivative(d3, c)

    # 1st deriv: 2 * rho * c * 1_{c > 0} = [40.0, 0.0, 60.0]
    np.testing.assert_allclose(d1.value, np.array([[40.0, 0.0, 60.0]]), atol=1e-12)

    # 2nd deriv: diag(2 * rho * 1_{c > 0}) = diag([20.0, 0.0, 20.0])
    expected_H = np.diag([20.0, 0.0, 20.0])
    np.testing.assert_allclose(d2.value, expected_H, atol=1e-12)

    # 3rd deriv: must be all zeros
    np.testing.assert_allclose(d3.value, np.zeros_like(d3.value), atol=1e-12)

    # 4th deriv: must be all zeros
    np.testing.assert_allclose(d4.value, np.zeros_like(d4.value), atol=1e-12)

    recorder.stop()


def test_quadratic_penalty_vector_penalty_derivatives():
    """Verify derivatives with vector-valued penalty factors (rho as a variable)."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    c = csdl.Variable(value=np.array([2.0, -1.0, 3.0]))
    rho = csdl.Variable(value=np.array([4.0, 5.0, 6.0]))
    p = quadratic_penalty(c, rho)

    # Derivatives w.r.t. rho
    d_rho = csdl.derivative(p, rho)
    # Expected d_rho: c^2 * 1_{c > 0} = [4.0, 0.0, 9.0]
    np.testing.assert_allclose(d_rho.value, np.array([[4.0, 0.0, 9.0]]), atol=1e-12)

    # Cross-derivative: d2 p / (dc drho)
    d_c = csdl.derivative(p, c)
    d2_c_rho = csdl.derivative(d_c, rho)
    # Expected: 2 * c * 1_{c > 0} along diagonal
    np.testing.assert_allclose(d2_c_rho.value, np.diag([4.0, 0.0, 6.0]), atol=1e-12)

    recorder.stop()


def test_optimization_lagrangian_integration():
    """Verify that Optimization.compute_lagrangian() uses the quadratic penalty."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    opt = Optimization()
    x = csdl.Variable(value=np.array([2.0]), name="x")
    opt.add_design_variable(x)
    opt.add_objective(x ** 2)

    # Inactive inequality constraint: (x - 5.0) <= 0  (2 - 5 = -3 <= 0)
    opt.add_inequality_constraint(
        constraint=x - 5.0,
        linear_penalty_factor=0.0,
        quadratic_penalty_factor=100.0,
    )
    # Active inequality constraint: (x - 1.0) <= 0  (2 - 1 = 1 > 0)
    opt.add_inequality_constraint(
        constraint=x - 1.0,
        linear_penalty_factor=0.0,
        quadratic_penalty_factor=50.0,
    )

    lagrangian = opt.compute_lagrangian()
    # Objective = 4.0
    # Inactive constraint penalty = 0.0
    # Active constraint penalty = 50.0 * (1.0)^2 = 50.0
    # Total lagrangian = 4.0 + 50.0 = 54.0
    np.testing.assert_allclose(lagrangian.value[0], 54.0, atol=1e-12)

    grad = opt.compute_lagrangian_gradient(design_variables=[x])
    # d(obj)/dx = 2*x = 4.0
    # d(penalty)/dx = 2 * 50 * 1 = 100.0
    # Total grad = 104.0
    np.testing.assert_allclose(grad[x].value, np.array([[104.0]]), atol=1e-12)

    recorder.stop()


def test_quadratic_penalty_simulators():
    """Verify PySimulator and JaxSimulator execution with QuadraticPenalty."""
    recorder = csdl.Recorder()
    recorder.start()

    c = csdl.Variable(value=np.array([2.0, -1.0]), name="c")
    rho = csdl.Variable(value=np.array([10.0]), name="rho")
    p = quadratic_penalty(c, rho)
    p.add_name("p")
    d1 = csdl.derivative(p, c)
    d1.add_name("d1")
    recorder.stop()

    # PySimulator
    sim_py = csdl.experimental.PySimulator(recorder)
    sim_py.run()
    np.testing.assert_allclose(sim_py[p], np.array([40.0]), atol=1e-12)
    np.testing.assert_allclose(sim_py[d1], np.array([[40.0, 0.0]]), atol=1e-12)

    # JaxSimulator
    sim_jax = csdl.experimental.JaxSimulator(
        recorder,
        additional_inputs=[c, rho],
        additional_outputs=[p, d1],
    )
    sim_jax.run()
    np.testing.assert_allclose(sim_jax[p], np.array([40.0]), atol=1e-12)
    np.testing.assert_allclose(sim_jax[d1], np.array([[40.0, 0.0]]), atol=1e-12)
