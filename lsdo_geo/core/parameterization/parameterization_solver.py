from lsdo_geo.csdl.optimization import Optimization, NewtonOptimizer
import csdl_alpha as csdl
import numpy as np
from dataclasses import dataclass
from typing import Union

@dataclass
class GeometricVariables:
    computed_values : list[csdl.Variable] = None
    desired_values : list[csdl.Variable] = None

    def __post_init__(self):
        if self.computed_values is None:
            self.computed_values = []
        if self.desired_values is None:
            self.desired_values = []

    def add_geometric_variable(self, computed_value:csdl.Variable, desired_value:csdl.Variable):
        self.computed_values.append(computed_value)
        self.desired_values.append(desired_value)


class ParameterizationSolver:
    '''
    The ParameterizationSolver class is used to solve geometric parameterization problems.
    '''
    def __init__(self) -> None:
        self.optimization = Optimization()
        self.optimizer = NewtonOptimizer()
        self.parameters = []
        self.parameter_costs = []

    def add_geometric_variable(self, computed_value:csdl.Variable, desired_value:csdl.Variable, penalty:Union[float,np.ndarray,csdl.Variable]=None):
        self.optimization.add_constraint(computed_value - desired_value, penalty=penalty)

    def add_parameter(self, parameter:csdl.Variable, cost:Union[float,np.ndarray,csdl.Variable]=1.):
        self.optimization.add_design_variable(parameter)
        self.parameters.append(parameter)
        self.parameter_costs.append(cost)

    def setup(self):
        objective = 0
        for parameter, cost in zip(self.parameters, self.parameter_costs):
            objective += csdl.vdot(parameter, cost*parameter)
        self.optimization.add_objective(objective)
        self.optimizer.add_optimization(self.optimization)

    def evaluate(self, geometric_variables:GeometricVariables) -> list[csdl.Variable]:
        for computed_value, desired_value in zip(geometric_variables.computed_values, geometric_variables.desired_values):
            self.add_geometric_variable(computed_value, desired_value)

        self.setup()
        self.optimizer.run()
        
        # NOTE: Because the solver updates the state variables, the user doesn't actually need to use these outputs
        outputs = []
        for computed_value in geometric_variables.computed_values:
            outputs.append(computed_value.value)
        return outputs
    