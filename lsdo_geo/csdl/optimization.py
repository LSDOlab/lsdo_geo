import csdl_alpha as csdl
import numpy as np

from typing import Union
from dataclasses import dataclass

@dataclass
class Optimization:
    objective:csdl.Variable=None
    design_variables:list[csdl.Variable]=None
    constraints:list[csdl.Variable]=None
    constraint_penalties:list[Union[float,np.ndarray]]=None
    state_residual_pairs:list[tuple[csdl.Variable,csdl.Variable]]=None

    def __post_init__(self):
        if self.design_variables is None:
            self.design_variables = []
        if self.constraints is None:
            self.constraints = []
        if self.constraint_penalties is None:
            self.constraint_penalties = []
        if self.state_residual_pairs is None:
            self.state_residual_pairs = []

    def add_objective(self, objective:csdl.Variable):
        '''
        Add the objective variable to the optimization problem.
        '''
        self.objective = objective

    def add_design_variable(self, design_variable:csdl.Variable):
        '''
        Add a design variable to the optimization problem.
        '''
        self.design_variables.append(design_variable)

    def add_constraint(self, constraint:csdl.Variable, penalty:Union[float,np.ndarray,csdl.Variable]=None):
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
        self.constraints.append(constraint)
        self.constraint_penalties.append(penalty)


    def compute_lagrangian(self):
        '''
        Constructs the CSDL variable for the lagrangian of the optimization problem.
        '''
        self.lagrange_multipliers = []
        lagrangian = self.objective
        for constraint, penalty in zip(self.constraints, self.constraint_penalties):
            if penalty is not None:
                lagrangian += penalty*constraint
                self.lagrange_multipliers.append(None)
            else:
                constraint_lagrange_multipliers = csdl.ImplicitVariable(shape=(constraint.size,), value=0.,
                                                                        name=f'{constraint.name}_lagrange_multipliers')
                self.lagrange_multipliers.append(constraint_lagrange_multipliers)
                lagrangian = lagrangian + csdl.vdot(constraint_lagrange_multipliers, constraint)
        self.lagrangian = lagrangian
        return lagrangian

    def compute_objective_gradient(self, objective:csdl.Variable=None, design_variables:list[csdl.Variable]=None):
        '''
        Constructs the CSDL variable for the objective gradient wrt each design variable.
        '''
        if objective is None:
            objective = self.objective
        if design_variables is None:
            design_variables = self.design_variables

        df_dx = csdl.derivative(self.objective, self.design_variables)
        self.df_dx = df_dx
        return df_dx
    
    def compute_lagrangian_gradient(self, lagrangian:csdl.Variable=None, design_variables:list[csdl.Variable]=None):
        '''
        Constructs the CSDL variable for the lagrangian gradient wrt each design variable.
        '''
        if lagrangian is None:
            lagrangian = self.lagrangian
        if design_variables is None:
            design_variables = self.design_variables

        dL_dx = csdl.derivative(self.lagrangian, self.design_variables, loop=False)
        self.dL_dx = dL_dx
        return dL_dx

    def compute_constraint_jacobian(self, constraints:list[csdl.Variable]=None, design_variables:list[csdl.Variable]=None):
        '''
        Constructs the CSDL variables for the jacobian of each constraint wrt each design variable
        '''
        if constraints is None:
            constraints = self.constraints
        if design_variables is None:
            design_variables = self.design_variables

        dc_dx = csdl.derivative(self.constraints, self.design_variables)
        self.dc_dx = dc_dx
        return dc_dx
    
    def setup(self):
        '''
        Sets up the optimization problem as an implicit model to drive the gradient to 0.
        '''
        if self.objective is not None:
            lagrangian = self.compute_lagrangian()
            dL_dx = self.compute_lagrangian_gradient(lagrangian=lagrangian, design_variables=self.design_variables)

        for constraint, constraint_lagrange_multipliers in zip(self.constraints, self.lagrange_multipliers):
            if constraint_lagrange_multipliers is not None:
                self.state_residual_pairs.append((constraint_lagrange_multipliers, constraint))

        for design_variable in self.design_variables:
            residual = dL_dx[design_variable].reshape((design_variable.size,))
            residual.add_name(f'{design_variable.name}_residual')
            self.state_residual_pairs.append((design_variable, residual))



class NewtonOptimizer:
    '''
    A Newton Optimizer class.

    NOTE: This is a temporary implementation until integration with the CSDL solvers is done
        (the CSDL solvers need the add_optimization functionality)
    '''
    def __init__(self) -> None:
        self.solver = csdl.nonlinear_solvers.Newton(residual_jac_kwargs={"loop": False, "concatenate_ofs": True})

    def add_optimization(self, optimization:Optimization):
        '''
        Add an optimization problem to the optimizer.
        '''
        self.optimization = optimization

    def setup(self):
        self.optimization.setup()
        for state, residual in self.optimization.state_residual_pairs:
            self.solver.add_state(state, residual)

    def run(self):
        '''
        Runs the Newton Optimization.

        NOTE: CSDL state/design variables are already updated and therefore are not needed to be returned.
        '''
        self.setup()
        self.solver.run()