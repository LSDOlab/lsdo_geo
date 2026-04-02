from lsdo_geo.csdl.optimization import Optimization, NewtonOptimizer
import csdl_alpha as csdl
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from typing import Union, Optional


@dataclass
class State:
    '''
    The State class is a dataclass that contains the state variable, its initial value, and its cost.
    '''
    state : csdl.Variable
    initial_value : Optional[Union[csdl.Variable, npt.NDArray[np.float64]]] = None
    cost : Union[float, npt.NDArray[np.float64], csdl.Variable] = 1.

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
        self.states : list[State] = []
        # self.state_costs : list[Union[float,npt.NDArray[np.float64],csdl.Variable]] = []


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

        self.add_equality_constraint(computed_value, desired_value, penalty)

    
    def add_equality_constraint(self, constraint:csdl.Variable, desired_value:Union[csdl.Variable, npt.NDArray[np.float64]],
                                penalty:Optional[Union[float,npt.NDArray[np.float64],csdl.Variable]]=None):
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
        # InversionTransform = csdl.transforms.EqualityInversion()
        # inverted_variable, inverted_constant, ops = InversionTransform.apply(lhs=constraint, rhs=desired_value, debug = False, aux_info=True)
        # print(f'Inverted {constraint.name} = {desired_value.name} : {(ops)}')

        equality_constraint = constraint - desired_value
        if isinstance(desired_value, csdl.Variable):
            equality_constraint.add_name(f'{desired_value.name}_enforcement_constraint')
        else:
            equality_constraint.add_name(f'{constraint.name}_enforcement_constraint')
        self.optimization.add_equality_constraint(equality_constraint, penalty=penalty)
        # self.optimization.add_equality_constraint(inverted_variable - inverted_constant, penalty=penalty)

    
    def add_inequality_constraint(self, constraint:csdl.Variable,
                                  linear_penalty_factor:Optional[Union[float,npt.NDArray[np.float64],csdl.Variable]]=None,
                                  linear_activation_factor:Union[float,npt.NDArray[np.float64],csdl.Variable]=10., 
                                  quadratic_penalty_factor:Optional[Union[float,npt.NDArray[np.float64],csdl.Variable]]=None,
                                  quadratic_activation_factor:Union[float,npt.NDArray[np.float64],csdl.Variable]=10.):
        '''
        Add an inequality constraint to the parameterization problem.

        Parameters
        ----------
        constraint : csdl.Variable
            The constraint to be satisfied. This is the quantity that will be driven to be non-positive.
        linear_penalty_factor : Union[float,npt.NDArray[np.float64],csdl.Variable], optional
            The linear penalty to be applied to the constraint, by default None. If None, lagrange multipliers are used.
            If not None, the penalty is the scaling factor for a linear penalty term. The linear penalty term is activated in the infeasible region using mellowmax smoothing.
        linear_activation_factor : Union[float,npt.NDArray[np.float64],csdl.Variable], optional
            The activation factor for the linear penalty term, by default 10. The larger this number, the sharper the (mellowmax) activation. This is a positive number and infinity corresponds to ReLU activation.
        quadratic_penalty_factor : Union[float,npt.NDArray[np.float64],csdl.Variable], optional
            The quadratic penalty to be applied to the constraint, by default None. If None, no quadratic penalty is used.
            If not None, the penalty is the scaling factor for a quadratic penalty term. The quadratic penalty term is activated in the infeasible region using softplus smoothing.
        quadratic_activation_factor : Union[float,npt.NDArray[np.float64],csdl.Variable], optional
            The activation factor for the quadratic penalty term, by default 10. The larger this number, the sharper the (mellowmax) activation. This is a positive number and infinity corresponds to ReLU activation.
        '''
        self.optimization.add_inequality_constraint(constraint, linear_penalty_factor=linear_penalty_factor, quadratic_penalty_factor=quadratic_penalty_factor, linear_activation_factor=linear_activation_factor, quadratic_activation_factor=quadratic_activation_factor)


    def add_state(self, state:csdl.Variable, initial_value:Optional[Union[csdl.Variable,npt.NDArray[np.float64]]]=None, 
                  cost:Union[float,npt.NDArray[np.float64],csdl.Variable]=1.):
        '''
        Add a state to the parameterization problem.

        Parameters
        ----------
        state : csdl.Variable
            The state to be manipulated in order to help drive the geometric variables to their desired values.
        initial_value : Union[csdl.Variable,npt.NDArray[np.float64]], optional
            The initial value of the state, by default None. If None, the state is initialized to value assigned within the state variable.
        cost : Union[float,np.ndarray,csdl.Variable], optional
            The cost of the state. This is the scaling factor for the quadratic cost/objective function.
        '''
        # self.optimization.add_design_variable(parameter)
        state_object = State(state=state, initial_value=initial_value, cost=cost)
        self.states.append(state_object)
        # self.state_costs.append(cost)


    def setup(self):
        for state_object in self.states:
            self.optimization.add_design_variable(design_variable=state_object.state, initial_value=state_object.initial_value)
        objective = 0
        for state_object in self.states:
            state = state_object.state
            cost = state_object.cost
            if isinstance(cost, (float,int)) or (cost.size==1):
                # Cost is a scalar
                objective = objective + csdl.vdot(state, cost*state)
            elif len(cost.shape) == 1:
                # Cost is a vector
                if cost.shape[0] != state.shape[0]:
                    raise ValueError('The cost vector must be the same size as the state. The cost provided, {}, is of shape {}, but the state is of shape {}'.format(cost.name, cost.shape, state.shape))
                objective = objective + csdl.vdot(state, cost*state)
            elif len(cost.shape) == 2:
                # Cost is a matrix
                if cost.shape[0] != cost.shape[1]:
                    raise ValueError('The cost matrix must be square. The cost provided, {}, is of shape {}'.format(cost.name, cost.shape))
                if cost.shape[0] != state.shape[0]:
                    raise ValueError('The cost matrix must be the same size as the state. The cost provided, {}, is of shape {}, but the state is of shape {}'.format(cost.name, cost.shape, state.shape))
                objective = objective + csdl.vdot(state, csdl.matvec(cost, state))
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
    