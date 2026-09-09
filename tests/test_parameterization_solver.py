import pytest
import numpy as np
import csdl_alpha as csdl
import lsdo_geo as lg

def test_geometric_variables_container():
    """Test GeometricVariables container initialization and variable declaration."""
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
