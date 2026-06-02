import csdl_alpha as csdl
import numpy as np

from typing import Union, Optional
import numpy.typing as npt
from dataclasses import dataclass

# def softplus(x:csdl.Variable, activation_factor:Union[float,npt.NDArray[np.float64],csdl.Variable]=10.) -> csdl.Variable:
#     '''
#     Softplus function for smoothing the max(0, x) (ReLU) function.

#     Parameters
#     ----------
#     x : csdl.Variable
#         The input variable.
#     activation_factor : Union[float,np.ndarray,csdl.Variable], optional
#         The activation factor for the softplus function, by default 10. Higher values make the function closer to max(0, x).

#     Returns
#     -------
#     csdl.Variable
#         The softplus of the input variable.
#     '''
#     return (1./activation_factor)*csdl.log(1.+csdl.exp(activation_factor*x))


# def mellowmax_with_zero(x:csdl.Variable, activation_factor:Union[float,npt.NDArray[np.float64],csdl.Variable]=10.) -> csdl.Variable:
#     '''
#     Mellowmax function for smoothing the max(0, x) (ReLU) function.

#     Parameters
#     ----------
#     x : csdl.Variable
#         The input variable.
#     activation_factor : Union[float,np.ndarray,csdl.Variable], optional
#         The activation factor for the mellowmax function, by default 10. Higher values make the function closer to max(0, x).

#     Returns
#     -------
#     csdl.Variable
#         The mellowmax of the input variable.
#     '''
#     return (1./activation_factor)*csdl.log(1/2*(1.+csdl.exp(activation_factor*x)))

def mellowmax_with_zero(x:csdl.Variable, activation_factor:Union[float,npt.NDArray[np.float64],csdl.Variable]=10.) -> csdl.Variable:
    '''
    Mellowmax function for smoothing the max(0, x) (ReLU) function.

    Parameters
    ----------
    x : csdl.Variable
        The input variable.
    activation_factor : Union[float,np.ndarray,csdl.Variable], optional
        The activation factor for the mellowmax function, by default 10. Higher values make the function closer to max(0, x).

    Returns
    -------
    csdl.Variable
        The mellowmax of the input variable.
    '''
    return 1./activation_factor*csdl.softplus(x*activation_factor) - 1./activation_factor*csdl.log(2.)

def step(x:csdl.Variable) -> csdl.Variable:
    '''
    Step function for smoothing the Heaviside step function.

    Parameters
    ----------
    x : csdl.Variable
        The input variable.

    Returns
    -------
    csdl.Variable
        The step function of the input variable.
    '''
    return csdl.relu(x)/x

@dataclass
class DesignVariable:
    variable:csdl.Variable
    initial_value:Optional[Union[csdl.Variable,npt.NDArray[np.float64],float]]=None

@dataclass
class EqualityConstraint:
    constraint:csdl.Variable
    penalty:Optional[Union[float,npt.NDArray[np.float64],csdl.Variable]]=None

@dataclass
class InequalityConstraint:
    constraint:csdl.Variable
    linear_penalty_factor:Optional[Union[float,npt.NDArray[np.float64],csdl.Variable]]=None
    linear_activation_factor:Union[float,npt.NDArray[np.float64],csdl.Variable]=10.
    quadratic_penalty_factor:Optional[Union[float,npt.NDArray[np.float64],csdl.Variable]]=None
    quadratic_activation_factor:Union[float,npt.NDArray[np.float64],csdl.Variable]=10.

@dataclass
class StateResidualPair:
    state:DesignVariable
    residual:csdl.Variable

@dataclass
class Optimization:
    objective:Optional[csdl.Variable]=None
    design_variables:Optional[list[DesignVariable]]=None
    equality_constraints:Optional[list[EqualityConstraint]]=None
    inequality_constraints:Optional[list[InequalityConstraint]]=None
    state_residual_pairs:Optional[list[StateResidualPair]] = None

    def __post_init__(self):
        if self.design_variables is None:
            self.design_variables : list[DesignVariable] = []
        if self.equality_constraints is None:
            self.equality_constraints : list[EqualityConstraint] = []
        if self.inequality_constraints is None:
            self.inequality_constraints : list[InequalityConstraint] = []
        if self.state_residual_pairs is None:
            self.state_residual_pairs : list[StateResidualPair] = []


    def add_objective(self, objective:csdl.Variable):
        '''
        Add the objective variable to the optimization problem.
        '''
        if objective.size != 1:
            raise ValueError('Objective must be a scalar variable.')
        self.objective = objective


    def add_design_variable(self, design_variable:csdl.Variable, initial_value:Optional[Union[csdl.Variable,float,npt.NDArray[np.float64]]]=None):
        '''
        Add a design variable to the optimization problem.
        '''
        design_variable_object = DesignVariable(variable=design_variable, initial_value=initial_value)
        self.design_variables.append(design_variable_object)
        if initial_value is not None:
            if isinstance(initial_value, csdl.Variable):
                design_variable.set_value(initial_value.value)
            else:
                design_variable.set_value(initial_value)


    def add_equality_constraint(self, constraint:csdl.Variable, penalty:Optional[Union[float,npt.NDArray[np.float64],csdl.Variable]]=None):
        '''
        Add a constraint to the optimization problem.

        Parameters
        ----------
        constraint : csdl.Variable
            The constraint to be added.
        penalty : Union[float,np.ndarray,csdl.Variable], optional
            The penalty for the constraint, by default None. If None, the penalty is treated as a lagrange multiplier.
            If a penalty is given, it is used as the penalty scaling factor for a quadratic constraint penalty.
        '''
        constraint_object = EqualityConstraint(constraint=constraint, penalty=penalty)
        self.equality_constraints.append(constraint_object)

    def add_inequality_constraint(self, constraint:csdl.Variable,
                                  linear_penalty_factor:Optional[Union[float,npt.NDArray[np.float64],csdl.Variable]]=None,
                                  linear_activation_factor:Union[float,npt.NDArray[np.float64],csdl.Variable]=10.,
                                  quadratic_penalty_factor:Optional[Union[float,npt.NDArray[np.float64],csdl.Variable]]=None,
                                  quadratic_activation_factor:Union[float,npt.NDArray[np.float64],csdl.Variable]=10.):
        '''
        Add an inequality constraint to the optimization problem.

        Parameters
        ----------
        constraint : csdl.Variable
            The constraint to be added. Must be in the form C(x) <= 0.
        linear_penalty_factor : Union[float,np.ndarray,csdl.Variable], optional
            The linear penalty factor for the constraint (with mellowmax smoothing), by default None. If None, a Lagrange multiplier is used.
        linear_activation_factor : Union[float,np.ndarray,csdl.Variable], optional
            The linear activation factor for the constraint (with mellowmax smoothing), by default 10.
        quadratic_penalty_factor : Union[float,np.ndarray,csdl.Variable], optional
            The quadratic penalty factor for the constraint (with softplus smoothing), by default None. If None, no quadratic penalty is used.
        quadratic_activation_factor : Union[float,np.ndarray,csdl.Variable], optional
            The quadratic activation factor for the constraint (with softplus smoothing), by default 10.
        '''
        constraint_object = InequalityConstraint(constraint=constraint,
                                                  linear_penalty_factor=linear_penalty_factor,
                                                  linear_activation_factor=linear_activation_factor,
                                                  quadratic_penalty_factor=quadratic_penalty_factor,
                                                  quadratic_activation_factor=quadratic_activation_factor)
        self.inequality_constraints.append(constraint_object)

    def compute_lagrangian(self):
        '''
        Constructs the CSDL variable for the lagrangian of the optimization problem.
        '''
        self.equality_constraint_lagrange_multipliers : list[Optional[csdl.Variable]] = []
        self.inequality_constraint_lagrange_multipliers : list[Optional[csdl.Variable]] = []

        lagrangian = self.objective
        for constraint_object in self.equality_constraints:
            constraint = constraint_object.constraint
            penalty = constraint_object.penalty
            if penalty is not None:
                lagrangian += penalty*csdl.vdot(constraint, constraint)
                # lagrangian += penalty
                self.equality_constraint_lagrange_multipliers.append(None)
            else:
                constraint_lagrange_multipliers = csdl.Variable(shape=(constraint.size,), value=0.,
                                                                        name=f'{constraint.name}_lagrange_multipliers')
                self.equality_constraint_lagrange_multipliers.append(constraint_lagrange_multipliers)
                lagrangian = lagrangian + csdl.vdot(constraint_lagrange_multipliers, constraint)
        for constraint_object in self.inequality_constraints:
            constraint = constraint_object.constraint
            linear_penalty_factor = constraint_object.linear_penalty_factor
            linear_activation_factor = constraint_object.linear_activation_factor
            quadratic_penalty_factor = constraint_object.quadratic_penalty_factor
            quadratic_activation_factor = constraint_object.quadratic_activation_factor
            if linear_penalty_factor is not None:
                # mellowmax smoothing for the linear penalty term
                lagrangian += csdl.vdot(linear_penalty_factor, mellowmax_with_zero(constraint, linear_activation_factor))
                self.inequality_constraint_lagrange_multipliers.append(None)
            else:
                constraint_lagrange_multipliers = csdl.Variable(shape=(constraint.size,), value=0.,
                                                                        name=f'{constraint.name}_inequality_lagrange_multipliers')
                self.inequality_constraint_lagrange_multipliers.append(constraint_lagrange_multipliers)
                smoothed_constraint = mellowmax_with_zero(constraint, linear_activation_factor)
                activated_constraint = csdl.sigmoid(constraint*linear_activation_factor)
                # constrained_lagrange_multiplier = softplus(constraint_lagrange_multipliers, linear_activation_factor)
                positive_lagrange_multipliers = 1./linear_activation_factor*csdl.softplus(constraint_lagrange_multipliers*linear_activation_factor)
                # softplus_negative_lagrange_multiplier = 1/10*csdl.softplus(-10*(constraint_lagrange_multipliers + 1.))
                # softplus_negative_lagrange_multiplier = 0.0001*csdl.softplus(-constraint_lagrange_multipliers)
                # softplus_negative_lagrange_multiplier = csdl.softplus(-constraint_lagrange_multipliers)
                softplus_negative_lagrange_multiplier = 1./linear_activation_factor*csdl.softplus(-constraint_lagrange_multipliers*linear_activation_factor)
                # lagrange_multiplier_regularization = 0.05*csdl.vdot(softplus_negative_lagrange_multiplier, softplus_negative_lagrange_multiplier)
                # lagrange_multiplier_regularization = 1.e-6*csdl.vdot(softplus_negative_lagrange_multiplier, softplus_negative_lagrange_multiplier)
                # lagrange_multiplier_regularization = csdl.sigmoid(-constraint*linear_activation_factor)*csdl.vdot(softplus_negative_lagrange_multiplier, softplus_negative_lagrange_multiplier)
                lagrange_multiplier_regularization = 1.e-6*csdl.sigmoid(-constraint*linear_activation_factor)*csdl.vdot(constraint_lagrange_multipliers, constraint_lagrange_multipliers)

                # Yet another new term to ensure the condition is (mu == 0 AND C(x) <= 0) or c == 0
                new_term = (constraint_lagrange_multipliers - positive_lagrange_multipliers)*csdl.sigmoid(constraint*linear_activation_factor)
                # Alternative new term that uses a softplus instead of mellowmax. By not using the mellowmax for both, the logic is more specific.
                # (mu == 0 AND C(x) <= 0) or (C(x) == 0 AND mu > 0)
                new_new_term = (constraint_lagrange_multipliers - positive_lagrange_multipliers)*csdl.sigmoid(constraint*linear_activation_factor)*(1./linear_activation_factor)*csdl.softplus(constraint*linear_activation_factor)

                lagrangian = lagrangian + csdl.vdot(constraint_lagrange_multipliers, smoothed_constraint) - csdl.vdot(softplus_negative_lagrange_multiplier, softplus_negative_lagrange_multiplier)
                # lagrangian = lagrangian + csdl.vdot(positive_lagrange_multipliers + new_term, smoothed_constraint) - lagrange_multiplier_regularization
                # lagrangian = lagrangian + csdl.vdot(positive_lagrange_multipliers, smoothed_constraint) + new_new_term - lagrange_multiplier_regularization
                # lagrangian = lagrangian + csdl.vdot(positive_lagrange_multipliers, smoothed_constraint) - lagrange_multiplier_regularization
                # lagrangian = lagrangian + csdl.vdot(positive_lagrange_multipliers, csdl.relu(constraint))  - 0.01*csdl.vdot(constraint_lagrange_multipliers, constraint_lagrange_multipliers)
                # lagrangian = lagrangian + csdl.vdot(constraint_lagrange_multipliers, csdl.relu(constraint))  - 1.e-12*csdl.vdot(constraint_lagrange_multipliers, constraint_lagrange_multipliers)
                # lagrangian = lagrangian + csdl.vdot(constraint_lagrange_multipliers, csdl.relu(constraint)) - step(-constraint)*csdl.vdot(constraint_lagrange_multipliers, constraint_lagrange_multipliers)
                # lagrangian = lagrangian + csdl.vdot(constraint_lagrange_multipliers, activated_constraint*smoothed_constraint) + (constraint_lagrange_multipliers - positive_lagrange_multipliers) - lagrange_multiplier_regularization
                # lagrangian = lagrangian + csdl.vdot(constraint_lagrange_multipliers, smoothed_constraint) + csdl.sum(constraint_lagrange_multipliers - positive_lagrange_multipliers)
            if quadratic_penalty_factor is not None:
                # Softplus smoothing for the quadratic penalty term
                # smoothed_constraint = softplus(constraint, quadratic_activation_factor)
                smoothed_constraint = csdl.softplus(constraint)
                lagrangian += csdl.vdot(smoothed_constraint, quadratic_penalty_factor*smoothed_constraint)
                
        lagrangian : csdl.Variable = lagrangian
        self.lagrangian = lagrangian
        return lagrangian


    def compute_objective_gradient(self, objective:Optional[csdl.Variable]=None, design_variables:Optional[list[csdl.Variable]]=None):
        '''
        Constructs the CSDL variable for the objective gradient wrt each design variable.
        '''
        if objective is None:
            objective = self.objective
            if self.objective is None:
                raise ValueError('Objective must be set before computing the objective gradient.')
        if design_variables is None:
            design_variables : list[csdl.Variable] = []
            for dv in self.design_variables:
                design_variables.append(dv.variable)

        df_dx = csdl.derivative(objective, design_variables)
        self.df_dx = df_dx
        return df_dx
    

    def compute_lagrangian_gradient(self, lagrangian:Optional[csdl.Variable]=None, design_variables:Optional[list[csdl.Variable]]=None):
        '''
        Constructs the CSDL variable for the lagrangian gradient wrt each design variable.
        '''
        if lagrangian is None:
            lagrangian = self.lagrangian
        if design_variables is None:
            design_variables : list[csdl.Variable] = []
            for dv in self.design_variables:
                design_variables.append(dv.variable)

        dL_dx = csdl.derivative(lagrangian, design_variables, loop=False)
        dL_dx : dict[csdl.Variable, csdl.Variable] = dL_dx
        self.dL_dx = dL_dx
        return dL_dx


    def compute_constraint_jacobian(self, constraints:Optional[list[csdl.Variable]]=None, design_variables:Optional[list[csdl.Variable]]=None):
        '''
        Constructs the CSDL variables for the jacobian of each constraint wrt each design variable
        '''
        if constraints is None:
            constraints = self.equality_constraints
            if self.equality_constraints is None:
                raise ValueError('Constraints must be set before computing the constraint jacobian.')
        if design_variables is None:
            design_variables : list[csdl.Variable] = []
            for dv in self.design_variables:
                design_variables.append(dv.variable)

        dc_dx = csdl.derivative(constraints, design_variables)
        self.dc_dx = dc_dx
        return dc_dx
    

    def setup(self):
        '''
        Sets up the optimization problem as an implicit model to drive the gradient to 0.
        '''
        if self.objective is not None:
            lagrangian = self.compute_lagrangian()
            design_variables : list[csdl.Variable] = []
            for dv in self.design_variables:
                design_variables.append(dv.variable)
            dL_dx = self.compute_lagrangian_gradient(lagrangian=lagrangian, design_variables=design_variables)

        for constraint_object, constraint_lagrange_multipliers in zip(self.equality_constraints, self.equality_constraint_lagrange_multipliers):
            if constraint_lagrange_multipliers is not None:
                constraint = constraint_object.constraint
                state_residual_pair = StateResidualPair(state=DesignVariable(variable=constraint_lagrange_multipliers), residual=constraint)
                self.state_residual_pairs.append(state_residual_pair)
        for constraint_object, constraint_lagrange_multipliers in zip(self.inequality_constraints, self.inequality_constraint_lagrange_multipliers):
            if constraint_lagrange_multipliers is not None:
                constraint = constraint_object.constraint
                linear_activation_factor = constraint_object.linear_activation_factor
                # smoothed_constraint = mellowmax_with_zero(constraint, linear_activation_factor)
                residual = csdl.derivative(lagrangian, constraint_lagrange_multipliers).flatten()
                state_residual_pair = StateResidualPair(state=DesignVariable(variable=constraint_lagrange_multipliers), residual=residual)
                self.state_residual_pairs.append(state_residual_pair)

        for design_variable_object in self.design_variables:
            design_variable = design_variable_object.variable
            residual = dL_dx[design_variable].reshape((design_variable.size,))
            residual.add_name(f'{design_variable.name}_residual')
            state_residual_pair = StateResidualPair(state=design_variable_object, residual=residual)
            self.state_residual_pairs.append(state_residual_pair)



class NewtonOptimizer:
    '''
    A Newton Optimizer class.

    NOTE: This is a temporary implementation until integration with the CSDL solvers is done
        (the CSDL solvers need the add_optimization functionality)
    '''
    def __init__(self) -> None:
        # self.solver = csdl.nonlinear_solvers.Newton()
        self.solver = csdl.nonlinear_solvers.Newton(residual_jac_kwargs={"loop": True, "concatenate_ofs": True}, tolerance=1.e-12, print_status=True)
        # self.solver = csdl.nonlinear_solvers.Newton(residual_jac_kwargs={"loop": True, "concatenate_ofs": True})
        self.has_been_setup = False

    def add_optimization(self, optimization:Optimization):
        '''
        Add an optimization problem to the optimizer.
        '''
        self.optimization = optimization

    def setup(self):
        self.optimization.setup()
        for state_residual_pair in self.optimization.state_residual_pairs:
            state_object = state_residual_pair.state
            state = state_object.variable
            initial_value = state_object.initial_value
            residual = state_residual_pair.residual
            if initial_value is not None:
                self.solver.add_state(state, residual, initial_value=initial_value)
            else:
                self.solver.add_state(state, residual)
        self.has_been_setup = True

    def run(self):
        '''
        Runs the Newton Optimization.

        NOTE: CSDL state/design variables are already updated and therefore are not needed to be returned.
        '''
        if not self.has_been_setup:
            self.setup()
        self.solver.run()