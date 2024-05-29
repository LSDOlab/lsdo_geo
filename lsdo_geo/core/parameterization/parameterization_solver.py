from lsdo_geo.csdl.optimization import Optimization, NewtonOptimizer
import csdl_alpha as csdl
import numpy as np
from dataclasses import dataclass
from typing import Union


@dataclass
class GeometricVariables:
    '''
    The GeometricVariables class is a dataclass that contains the computed and desired values of the geometric variables.
    The computed values are the values as computed from the geometry. These quantities will be driven to the desired values by the solver.
    The desired values are the values that the computed values are driven to. These are often the design variables from the optimizer.
    '''
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
        '''
        Add/declare a geometric variable to the parameterization problem.

        Parameters
        ----------
        computed_value : csdl.Variable
            The computed value of the geometric variable. This is the quantity as computed from the geometry.
            This will be driven to the desired value.
        desired_value : csdl.Variable
            The desired value of the geometric variable. This is usually the csdl variable that is the design variable from the optimizer, or can
            be a constant input (like any other csdl variable).
        penalty : Union[float,np.ndarray,csdl.Variable], optional
            The penalty to be applied to the constraint, by default None. If None, lagrange multipliers are used.
            If not None, the penalty is the scaling factor for a quadratic constraint penalty term.
        '''
        self.optimization.add_constraint(computed_value - desired_value, penalty=penalty)


    def add_parameter(self, parameter:csdl.Variable, cost:Union[float,np.ndarray,csdl.Variable]=1.):
        '''
        Add a parameter/dofs to the parameterization problem.

        Parameters
        ----------
        parameter : csdl.Variable
            The parameter to be manipulated in order to help drive the geometric variables to their desired values.
        cost : Union[float,np.ndarray,csdl.Variable], optional
            The cost of the parameter. This is the scaling factor for the quadratic cost/objective function.
        '''
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
        '''
        Evaluate the parameterization solver.

        Parameters
        ----------
        geometric_variables : GeometricVariables
            The geometric variables. This is a dataclass that contains the computed and desired values of the geometric variables.
            The computed values are the values as computed from the geometry, and the desired values are the values that the computed values
            are driven to.

        Returns
        -------
        list[csdl.Variable]
            The computed values of the geometric variables. 
            NOTE: This will return the exact same parameter variables that were added because CSDL updates the state variables in place.
        '''
        for computed_value, desired_value in zip(geometric_variables.computed_values, geometric_variables.desired_values):
            self.add_geometric_variable(computed_value, desired_value)

        self.setup()
        self.optimizer.run()
        
        # NOTE: Because the solver updates the state variables, the user doesn't actually need to use these outputs
        outputs = []
        for computed_value in geometric_variables.computed_values:
            outputs.append(computed_value.value)
        return outputs
    