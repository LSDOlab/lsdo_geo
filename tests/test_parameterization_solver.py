import pytest
import numpy as np
import csdl_alpha as csdl
import lsdo_geo as lg

def test_geometric_variables_container():
    """Test GeometricVariables container initialization and variable declaration."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    gv = lg.GeometricVariables()
    assert len(gv.computed_value) == 0
    assert len(gv.desired_value) == 0
    assert len(gv.penalty_value) == 0

    v1 = csdl.Variable(value=np.array([1.0]))
    v2 = csdl.Variable(value=np.array([2.0]))
    gv.add_variable(computed_value=v1, desired_value=v2, penalty_value=10.0)

    assert len(gv.computed_value) == 1
    assert len(gv.desired_value) == 1
    assert len(gv.penalty_value) == 1
    assert gv.penalty_value[0] == 10.0

def test_parameterization_solver_equality_constrained_solve():
    """Test ParameterizationSolver solving a constrained quadratic KKT system with NewtonOptimizer."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    solver = lg.ParameterizationSolver()

    # Two state variables: min (x1^2 + x2^2) s.t. x1 + x2 = 2
    # Theoretical optimum: x1 = 1.0, x2 = 1.0
    x1 = csdl.Variable(value=np.array([0.0]), name="x1")
    x2 = csdl.Variable(value=np.array([0.0]), name="x2")

    solver.add_state(x1, cost=1.0)
    solver.add_state(x2, cost=1.0)

    gv = lg.GeometricVariables()
    gv.add_variable(
        computed_value=x1 + x2,
        desired_value=csdl.Variable(value=np.array([2.0])),
    )

    ret = solver.evaluate(gv)
    assert ret is None
    np.testing.assert_allclose(x1.value, np.array([1.0]), atol=1e-4)
    np.testing.assert_allclose(x2.value, np.array([1.0]), atol=1e-4)
    # Check constraint satisfaction: x1 + x2 == 2.0
    np.testing.assert_allclose((x1 + x2).value, np.array([2.0]), atol=1e-4)


def test_parameterization_solver_vector_and_matrix_costs():
    """Test ParameterizationSolver with vector-valued and matrix-valued state costs."""
    # 1. Vector cost: min (x0^2 + 4*x1^2) s.t. x0 + x1 = 5
    # Analytical solution: x0 = 4.0, x1 = 1.0
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    solver = lg.ParameterizationSolver()
    x = csdl.Variable(value=np.zeros(2), name="x")
    cost_vec = np.array([1.0, 4.0])
    solver.add_state(x, cost=cost_vec)

    gv = lg.GeometricVariables()
    gv.add_variable(
        computed_value=csdl.sum(x),
        desired_value=csdl.Variable(value=np.array([5.0])),
    )
    solver.evaluate(gv)
    np.testing.assert_allclose(x.value, np.array([4.0, 1.0]), atol=1e-4)

    # 2. Matrix cost: min y^T [[2, 0], [0, 2]] y s.t. y0 + y1 = 4
    # Analytical solution: y0 = 2.0, y1 = 2.0
    recorder2 = csdl.Recorder(inline=True)
    recorder2.start()

    solver2 = lg.ParameterizationSolver()
    y = csdl.Variable(value=np.zeros(2), name="y")
    cost_mat = np.array([[2.0, 0.0], [0.0, 2.0]])
    solver2.add_state(y, cost=cost_mat)

    gv2 = lg.GeometricVariables()
    gv2.add_variable(
        computed_value=csdl.sum(y),
        desired_value=csdl.Variable(value=np.array([4.0])),
    )
    solver2.evaluate(gv2)
    np.testing.assert_allclose(y.value, np.array([2.0, 2.0]), atol=1e-4)


def test_parameterization_solver_penalty_formulation():
    """Test ParameterizationSolver using penalty value on geometric variables."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    solver = lg.ParameterizationSolver()
    x = csdl.Variable(value=np.array([0.0]), name="x")
    solver.add_state(x, cost=1.0)

    gv = lg.GeometricVariables()
    gv.add_variable(
        computed_value=x,
        desired_value=csdl.Variable(value=np.array([3.0])),
        penalty_value=1e4,
    )
    solver.evaluate(gv)
    np.testing.assert_allclose(x.value, np.array([3.0]), rtol=1e-3)


def test_parameterization_solver_validation_errors():
    """Test ParameterizationSolver and GeometricVariables validation error handling."""
    # GeometricVariables length mismatch
    gv = lg.GeometricVariables()
    gv.computed_value = [csdl.Variable(value=np.array([1.0]))]
    gv.desired_value = []
    with pytest.raises(ValueError, match="same length"):
        gv.__post_init__()

    # State cost vector shape mismatch
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    solver = lg.ParameterizationSolver()
    x = csdl.Variable(value=np.zeros(3), name="x")
    solver.add_state(x, cost=np.ones(2))
    gv_solve = lg.GeometricVariables()
    gv_solve.add_variable(x[0], csdl.Variable(value=np.array([1.0])))
    with pytest.raises(ValueError, match="cost vector must be the same size"):
        solver.evaluate(gv_solve)

    # State cost non-square matrix
    recorder2 = csdl.Recorder(inline=True)
    recorder2.start()

    solver2 = lg.ParameterizationSolver()
    y = csdl.Variable(value=np.zeros(2), name="y")
    solver2.add_state(y, cost=np.ones((2, 3)))
    gv_solve2 = lg.GeometricVariables()
    gv_solve2.add_variable(y[0], csdl.Variable(value=np.array([1.0])))
    with pytest.raises(ValueError, match="cost matrix must be square"):
        solver2.evaluate(gv_solve2)


@pytest.mark.skipif(
    not hasattr(csdl.nonlinear_solvers, "BlockNewton"),
    reason="Requires CSDL BlockNewton solver",
)
def test_parameterization_solver_kkt_symmetry_structure():
    """
    Rigorously verifies that ParameterizationSolver leverages BlockNewton's KKT symmetry:
    1. Multiplier-constraint pairs registered in _kkt_constraints.
    2. Zero block established for d(c)/d(lambda) = 0.
    3. Transpose symmetry established for d(r_primal)/d(lambda) = (d(c)/d(primal))^T.
    4. Exact convergence to analytical solution.
    """
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    solver = lg.ParameterizationSolver()

    # min (x1^2 + x2^2) s.t. 2*x1 + 3*x2 = 13
    # Analytical: x1 = 2*lambda, x2 = 3*lambda => 4*lambda + 9*lambda = 13 => lambda = 1 => x1 = 2, x2 = 3
    x1 = csdl.Variable(value=np.array([0.0]), name="x1")
    x2 = csdl.Variable(value=np.array([0.0]), name="x2")

    solver.add_state(x1, cost=1.0)
    solver.add_state(x2, cost=1.0)

    gv = lg.GeometricVariables()
    gv.add_variable(
        computed_value=2.0 * x1 + 3.0 * x2,
        desired_value=csdl.Variable(value=np.array([13.0])),
    )

    ret = solver.evaluate(gv)
    assert ret is None

    block_solver = solver.optimizer.solver
    assert len(block_solver._kkt_constraints) == 1
    mult, c_res, *_ = block_solver._kkt_constraints[0]

    # Verify zero block: d(c)/d(lambda) = 0
    assert (c_res, mult) in block_solver._zero_blocks

    # Verify transpose blocks: d(r_x)/d(lambda) = (d(c)/d(x))^T
    rx1 = block_solver.state_to_residual_map[x1]
    rx2 = block_solver.state_to_residual_map[x2]
    assert (rx1, mult) in block_solver._transpose_blocks
    assert block_solver._transpose_blocks[(rx1, mult)] == (c_res, x1, 1.0)
    assert (rx2, mult) in block_solver._transpose_blocks
    assert block_solver._transpose_blocks[(rx2, mult)] == (c_res, x2, 1.0)

    # Verify solution values
    np.testing.assert_allclose(x1.value, np.array([2.0]), atol=1e-4)
    np.testing.assert_allclose(x2.value, np.array([3.0]), atol=1e-4)


@pytest.mark.skipif(
    not hasattr(csdl.nonlinear_solvers, "BlockNewton"),
    reason="Requires CSDL BlockNewton solver",
)
def test_parameterization_solver_primal_hessian_symmetry():
    """
    Rigorously verifies that ParameterizationSolver automatically leverages primal-primal Hessian symmetry:
    d(r_pi)/d(pj) = (d(r_pj)/d(pi))^T for all coupled off-diagonal primal state blocks.
    """
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    solver = lg.ParameterizationSolver()

    # 3 primal states: min (x1^2 + x2^2 + x3^2) s.t. x1 + 2*x2 + 3*x3 = 14
    # Analytical: lambda = 14 / (1 + 4 + 9) = 1.0 => x1 = 1, x2 = 2, x3 = 3
    x1 = csdl.Variable(value=np.array([0.0]), name="x1")
    x2 = csdl.Variable(value=np.array([0.0]), name="x2")
    x3 = csdl.Variable(value=np.array([0.0]), name="x3")

    solver.add_state(x1, cost=1.0)
    solver.add_state(x2, cost=1.0)
    solver.add_state(x3, cost=1.0)

    gv = lg.GeometricVariables()
    gv.add_variable(
        computed_value=x1 + 2.0 * x2 + 3.0 * x3,
        desired_value=csdl.Variable(value=np.array([14.0])),
    )

    solver.evaluate(gv)

    block_solver = solver.optimizer.solver
    mult, c_res, *_ = block_solver._kkt_constraints[0]
    rx1 = block_solver.state_to_residual_map[x1]
    rx2 = block_solver.state_to_residual_map[x2]
    rx3 = block_solver.state_to_residual_map[x3]

    # KKT transpose symmetry
    assert (rx1, mult) in block_solver._transpose_blocks
    assert (rx2, mult) in block_solver._transpose_blocks
    assert (rx3, mult) in block_solver._transpose_blocks

    # Verify primal-primal Hessian symmetry pairs (lower triangle resolved as transpose of upper triangle)
    assert (rx2, x1) in block_solver._transpose_blocks
    assert block_solver._transpose_blocks[(rx2, x1)] == (rx1, x2, 1.0)

    assert (rx3, x1) in block_solver._transpose_blocks
    assert block_solver._transpose_blocks[(rx3, x1)] == (rx1, x3, 1.0)

    assert (rx3, x2) in block_solver._transpose_blocks
    assert block_solver._transpose_blocks[(rx3, x2)] == (rx2, x3, 1.0)

    # Verify converged solution
    np.testing.assert_allclose(x1.value, np.array([1.0]), atol=1e-4)
    np.testing.assert_allclose(x2.value, np.array([2.0]), atol=1e-4)
    np.testing.assert_allclose(x3.value, np.array([3.0]), atol=1e-4)


@pytest.mark.skipif(
    not hasattr(csdl.nonlinear_solvers, "BlockNewton"),
    reason="Requires CSDL BlockNewton solver",
)
def test_parameterization_solver_structural_decoupling_sparsity():
    """
    Rigorously verifies automatic DAG reachability sparsity detection across decoupled components.
    Component A: state a s.t. a = 4.0
    Component B: state b s.t. b = 6.0
    Cross-blocks between A and B must be exact zero blocks with 0 AD sweeps.
    """
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    solver = lg.ParameterizationSolver()

    a = csdl.Variable(value=np.array([0.0]), name="comp_a")
    b = csdl.Variable(value=np.array([0.0]), name="comp_b")

    solver.add_state(a, cost=1.0)
    solver.add_state(b, cost=1.0)

    gv = lg.GeometricVariables()
    gv.add_variable(computed_value=a, desired_value=csdl.Variable(value=np.array([4.0])))
    gv.add_variable(computed_value=b, desired_value=csdl.Variable(value=np.array([6.0])))

    solver.evaluate(gv)

    block_solver = solver.optimizer.solver
    mult_a, ca_res, *_ = block_solver._kkt_constraints[0]
    mult_b, cb_res, *_ = block_solver._kkt_constraints[1]

    ra = block_solver.state_to_residual_map[a]
    rb = block_solver.state_to_residual_map[b]

    # Cross-component constraints decoupled: ca does not depend on b, cb does not depend on a
    assert (ca_res, b) in block_solver._zero_blocks
    assert (cb_res, a) in block_solver._zero_blocks
    assert (ca_res, mult_b) in block_solver._zero_blocks
    assert (cb_res, mult_a) in block_solver._zero_blocks

    # Decoupled multipliers have zero block with primal residuals
    assert (ra, mult_b) in block_solver._zero_blocks
    assert (rb, mult_a) in block_solver._zero_blocks

    # Decoupled primal states have zero cross-derivatives
    assert (ra, b) in block_solver._zero_blocks
    assert (rb, a) in block_solver._zero_blocks

    np.testing.assert_allclose(a.value, np.array([4.0]), atol=1e-4)
    np.testing.assert_allclose(b.value, np.array([6.0]), atol=1e-4)


def test_parameterization_solver_arbitrary_symmetric_weighting_matrix():
    """
    Rigorously verifies that ParameterizationSolver naturally accepts and solves with
    arbitrary symmetric (including coupled non-diagonal) weighting matrices.
    """
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    solver = lg.ParameterizationSolver()

    # min x^T W x s.t. x0 + x1 = 4
    # W = [[3, 1], [1, 2]] (symmetric, coupled)
    # L = 3*x0^2 + 2*x0*x1 + 2*x1^2 + lambda*(x0 + x1 - 4)
    # grad_x0 = 6*x0 + 2*x1 + lambda = 0
    # grad_x1 = 2*x0 + 4*x1 + lambda = 0
    # => 4*x0 - 2*x1 = 0 => x1 = 2*x0
    # x0 + 2*x0 = 4 => x0 = 4/3, x1 = 8/3
    x = csdl.Variable(value=np.zeros(2), name="x_state")
    W = np.array([[3.0, 1.0], [1.0, 2.0]])
    solver.add_state(x, cost=W)

    gv = lg.GeometricVariables()
    gv.add_variable(
        computed_value=csdl.sum(x),
        desired_value=csdl.Variable(value=np.array([4.0])),
    )

    solver.evaluate(gv)

    np.testing.assert_allclose(x.value, np.array([4.0 / 3.0, 8.0 / 3.0]), atol=1e-4)


def test_parameterization_solver_jax_total_derivatives():
    """
    Rigorously verifies reverse-mode total derivatives through ParameterizationSolver with JAX backend.
    """
    from csdl_alpha.backends.jax.jax_simulator import JaxSimulator

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    target_var = csdl.Variable(value=np.array([6.0]), name="target_val")

    solver = lg.ParameterizationSolver()
    x1 = csdl.ImplicitVariable((1,), value=np.array([0.0]), name="x1")
    x2 = csdl.ImplicitVariable((1,), value=np.array([0.0]), name="x2")

    solver.add_state(x1, cost=1.0)
    solver.add_state(x2, cost=2.0)

    gv = lg.GeometricVariables()
    gv.add_variable(
        computed_value=x1 + x2,
        desired_value=target_var,
    )
    solver.evaluate(gv)

    # Output objective function
    obj = x1**2 + 2.0 * x2**2
    recorder.stop()

    sim = JaxSimulator(recorder, additional_inputs=[target_var], additional_outputs=[x1, x2, obj])
    sim.run()

    # Analytical solution: min x1^2 + 2*x2^2 s.t. x1 + x2 = T
    # 2*x1 = lambda, 4*x2 = lambda => x1 = 2*x2 => 3*x2 = T => x2 = T/3, x1 = 2T/3
    # obj = (2T/3)^2 + 2*(T/3)^2 = 6*(T/3)^2 = 2/3 * T^2
    # d(obj)/d(T) = 4/3 * T. For T=6: d(obj)/d(T) = 8.0.
    np.testing.assert_allclose(sim[x1], [4.0], atol=1e-4)
    np.testing.assert_allclose(sim[x2], [2.0], atol=1e-4)
    np.testing.assert_allclose(sim[obj], [24.0], atol=1e-4)

    # Check total derivatives
    totals = sim.compute_totals()
    np.testing.assert_allclose(totals[obj, target_var], [[8.0]], rtol=1e-5, atol=1e-5)
    sim.check_totals()


@pytest.mark.skipif(
    not hasattr(csdl.nonlinear_solvers, "BlockNewton"),
    reason="Requires CSDL BlockNewton solver",
)
def test_parameterization_solver_quadratic_objective_linear_constraints_flag():
    """
    Rigorously verifies that quadratic_objective_linear_constraints=True registers constant 2*W
    blocks in BlockNewton._constant_blocks, bypassing reverse-mode sweeps on primal diagonals.
    """
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    solver = lg.ParameterizationSolver(quadratic_objective_linear_constraints=True)
    x = csdl.Variable(value=np.zeros(2), name="x_opt_state")
    cost_vec = np.array([1.5, 3.5])
    solver.add_state(x, cost=cost_vec)

    gv = lg.GeometricVariables()
    gv.add_variable(
        computed_value=csdl.sum(x),
        desired_value=csdl.Variable(value=np.array([5.0])),
    )
    solver.evaluate(gv)

    block_solver = solver.optimizer.solver
    rx = block_solver.state_to_residual_map[x]

    assert (rx, x) in block_solver._constant_blocks
    const_mat = block_solver._constant_blocks[(rx, x)].value
    np.testing.assert_allclose(const_mat, np.array([[3.0, 0.0], [0.0, 7.0]]))


def test_parameterization_solver_constant_kkt_precedence_over_quadratic_linear():
    """
    Verifies that constant_kkt=True takes precedence over quadratic_objective_linear_constraints=True,
    performing a direct single-step pre-factorized solve (K^{-1} R) rather than Newton iteration.
    """
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    # Both set to True: constant_kkt must take precedence
    solver = lg.ParameterizationSolver(
        constant_kkt=True,
        quadratic_objective_linear_constraints=True,
    )
    assert solver.constant_kkt is True
    assert solver.quadratic_objective_linear_constraints is False

    x = csdl.Variable(value=np.zeros(2), name="x_state")
    solver.add_state(x, cost=np.array([1.0, 1.0]))

    gv = lg.GeometricVariables()
    gv.add_variable(
        computed_value=csdl.sum(x),
        desired_value=csdl.Variable(value=np.array([4.0])),
    )
    solver.evaluate(gv)

    assert solver.has_constant_kkt_setup is True
    np.testing.assert_allclose(solver.solved_states["x_state"].value, np.array([2.0, 2.0]), atol=1e-4)


def test_constant_kkt_parameterization_solver_class():
    """
    Verifies ConstantKKTParameterizationSolver.
    """
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    solver1 = lg.ConstantKKTParameterizationSolver()
    assert solver1.constant_kkt is True




