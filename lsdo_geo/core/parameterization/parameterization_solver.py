from lsdo_geo.csdl.optimization import Optimization, NewtonOptimizer
import csdl_alpha as csdl
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from typing import Union, Optional


@dataclass
class GeometricVariables:
    '''
    The GeometricVariables class is a dataclass that contains the computed and desired values of the geometric variables.
    The computed values are the values as computed from the geometry. These quantities will be driven to the desired values by the solver.
    The desired values are the values that the computed values are driven to. These are often the design variables from the optimizer.
    '''
    computed_value : Optional[list[csdl.Variable]] = None
    desired_value : Optional[list[Union[csdl.Variable, float, npt.NDArray[np.float64]]]] = None
    penalty_value: Optional[list[Union[csdl.Variable, float, npt.NDArray[np.float64], None]]] = None

    def __post_init__(self):
        if self.computed_value is None:
            self.computed_value = []
        if self.desired_value is None:
            self.desired_value = []
        if self.penalty_value is None:
            self.penalty_value = []
        if len(self.computed_value) != len(self.desired_value):
            raise ValueError('The computed and desired value must be the same length.')
        if len(self.computed_value) != len(self.penalty_value):
            raise ValueError('The computed and penalty value must be the same length.')
        if len(self.desired_value) != len(self.penalty_value):
            raise ValueError('The desired and penalty value must be the same length.')

    def add_variable(self, computed_value:csdl.Variable, desired_value:Union[csdl.Variable, float, npt.NDArray[np.float64]], penalty_value:Optional[Union[csdl.Variable,float, npt.NDArray[np.float64]]]=None):
        '''
        Add a geometric variable to the parameterization problem.
        Parameters
        ----------
        computed_value : csdl.Variable
            The computed value of the geometric variable. This is the quantity as computed from the geometry.
            This will be driven to the desired value.
        desired_value : Union[csdl.Variable, npt.NDArray[np.float64]]
            The desired value of the geometric variable. This is usually the csdl variable that is the design variable from the optimizer, or can
            be a constant input (like any other csdl variable).
        penalty_values : Union[float,npt.NDArray[np.float64],csdl.Variable], optional
            The penalty to be applied to the constraint, by default None. If None, lagrange multipliers are used.
            If not None, the penalty is the scaling factor for a quadratic constraint penalty term.
        '''
        if self.computed_value is None:
            self.computed_value = []
        if self.desired_value is None:
            self.desired_value = []
        if self.penalty_value is None:
            self.penalty_value = []

        self.computed_value.append(computed_value)
        self.desired_value.append(desired_value)
        self.penalty_value.append(penalty_value)


class ParameterizationSolver:
    '''
    The ParameterizationSolver class is used to solve geometric parameterization problems.
    '''
    def __init__(self) -> None:
        self.optimization = Optimization()
        self.optimizer = NewtonOptimizer()
        self.parameters : list[csdl.Variable] = []
        self.parameter_costs : list[Union[float,npt.NDArray[np.float64],csdl.Variable]] = []


    def add_variable(self, computed_value:csdl.Variable, desired_value:Union[csdl.Variable, npt.NDArray[np.float64]], penalty:Optional[Union[float,npt.NDArray[np.float64],csdl.Variable]]=None):
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
        # computed_value, desired_value = csdl.backtrack_operations(computed_value, desired_value)

        self.add_constraint(computed_value, desired_value, penalty)

    
    def add_constraint(self, constraint:csdl.Variable, desired_value:Union[csdl.Variable, npt.NDArray[np.float64]], penalty:Optional[Union[float,npt.NDArray[np.float64],csdl.Variable]]=None):
        '''
        Add a constraint to the parameterization problem.

        Parameters
        ----------
        constraint : csdl.Variable
            The constraint to be satisfied. This is the quantity that will be driven to the desired value.
        desired_value : csdl.Variable
            The desired value of the constraint. This is usually the csdl variable that is the design variable from the optimizer, or can
            be a constant input (like any other csdl variable).
        penalty : Union[float,np.ndarray,csdl.Variable], optional
            The penalty to be applied to the constraint, by default None. If None, lagrange multipliers are used.
            If not None, the penalty is the scaling factor for a quadratic constraint penalty term.
        '''
        self.optimization.add_constraint(constraint - desired_value, penalty=penalty)


    def add_state(self, parameter:csdl.Variable, cost:Union[float,npt.NDArray[np.float64],csdl.Variable]=1.):
        '''
        Add a parameter/dofs to the parameterization problem.

        Parameters
        ----------
        parameter : csdl.Variable
            The parameter to be manipulated in order to help drive the geometric variables to their desired values.
        cost : Union[float,np.ndarray,csdl.Variable], optional
            The cost of the parameter. This is the scaling factor for the quadratic cost/objective function.
        '''
        # self.optimization.add_design_variable(parameter)
        self.parameters.append(parameter)
        self.parameter_costs.append(cost)


    def setup(self):
        for parameter in self.parameters:
            self.optimization.add_design_variable(parameter)
        objective = 0
        for parameter, cost in zip(self.parameters, self.parameter_costs):
            if isinstance(cost, (float,int)) or (cost.size==1):
                # Cost is a scalar
                objective = objective + csdl.vdot(parameter, cost*parameter)
            elif len(cost.shape) == 1:
                # Cost is a vector
                if cost.shape[0] != parameter.shape[0]:
                    raise ValueError('The cost vector must be the same size as the parameter. The cost provided, {}, is of shape {}, but the parameter is of shape {}'.format(cost.name, cost.shape, parameter.shape))
                objective = objective + csdl.vdot(parameter, cost*parameter)
            elif len(cost.shape) == 2:
                # Cost is a matrix
                if cost.shape[0] != cost.shape[1]:
                    raise ValueError('The cost matrix must be square. The cost provided, {}, is of shape {}'.format(cost.name, cost.shape))
                if cost.shape[0] != parameter.shape[0]:
                    raise ValueError('The cost matrix must be the same size as the parameter. The cost provided, {}, is of shape {}, but the parameter is of shape {}'.format(cost.name, cost.shape, parameter.shape))
                objective = objective + csdl.vdot(parameter, csdl.matvec(cost, parameter))
            else:
                raise ValueError('The cost must be a scalar, vector, or matrix. The cost provided, {}, is of shape {}'.format(cost.name, cost.shape))
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
        # for computed_value, desired_value in zip(geometric_variables.computed_values, geometric_variables.desired_values):
        for i in range(len(geometric_variables.computed_value)):
            computed_value = geometric_variables.computed_value[i]
            desired_value = geometric_variables.desired_value[i]
            penalty_value = geometric_variables.penalty_value[i]
            self.add_variable(computed_value, desired_value, penalty_value)
        # for parameter, cost in zip(self.parameters, self.parameter_costs):
        #     self.add_parameter(parameter, cost)

        self.setup()
        self.optimizer.run()
        
        # NOTE: Because the solver updates the state variables, the user doesn't actually need to use these outputs
        outputs : list[csdl.Variable] = []
        for computed_value in geometric_variables.computed_value:
            outputs.append(computed_value)
        return outputs
    