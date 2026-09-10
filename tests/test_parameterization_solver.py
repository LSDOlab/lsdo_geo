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

    outputs = solver.evaluate(gv)

    assert len(outputs) == 1
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

