from __future__ import annotations
from lsdo_geo.optimization.optimization import Optimization, NewtonOptimizer
import csdl_alpha as csdl
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from typing import Union, Optional, Callable, Dict, List, Sequence
import lsdo_function_spaces as lfs
from lsdo_geo.core.geometry.assembly import Connection, Assembly


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

    @property
    def computed_values(self) -> list[csdl.Variable]:
        return self.computed_value

    @property
    def desired_values(self) -> list[Union[csdl.Variable, float, npt.NDArray[np.float64]]]:
        return self.desired_value

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
    The ParameterizationSolver class solves geometric parameterization problems using
    either direct pre-factored KKT matrix multiplication or implicit Newton-KKT iterations.

    Parameters
    ----------
    optimizer : Optional[NewtonOptimizer], optional
        Custom optimizer instance. If None, instantiates a default NewtonOptimizer.
    inner_parameterization_map : Optional[Callable], optional
        Callable for the inner parameterization map: P_i(alpha_s) -> geometry/assembly.
    constant_kkt : bool, optional
        Set to True when the entire KKT system (both the quadratic cost weighting matrix W
        and the constraint Jacobian A = dc/d(alpha_s)) is constant and completely independent
        of top-level optimization design variables. When True, the KKT matrix is inverted/factored
        once at setup, and every evaluation is performed via a single direct back-substitution step
        (csdl.matvec) with zero Newton iterations. Takes precedence over
        quadratic_objective_linear_constraints. Default is False.
    quadratic_objective_linear_constraints : bool, optional
        Set to True when the objective is quadratic (x^T W x) and the constraints are linear in the
        parameterization states alpha_s (c(alpha_s, d) = A(d) alpha_s - b(d) = 0), but the constraint
        Jacobian A(d) varies across outer optimization iterations with design variables d.
        In this case, the Lagrangian Hessian with respect to alpha_s is mathematically exact 2W (since
        constraint curvature nabla^2 c is zero). Setting this to True injects the constant 2W matrix
        directly into the KKT matrix, completely bypassing reverse-mode autodiff sweeps for the primal
        state residuals, and converges in 1 Newton iteration per optimization step.
        Ignored if constant_kkt is True. Default is False.
    assembly : Optional[Union[Assembly, lfs.FunctionNetwork]], optional
        Assembly or function network containing geometric connections.
    geometry : Optional[Geometry], optional
        Geometry instance to be parameterized.
    '''
    def __init__(
        self,
        optimizer=None,
        inner_parameterization_map: Optional[Callable] = None,
        constant_kkt: bool = False,
        quadratic_objective_linear_constraints: bool = False,
        assembly: Optional[Union[Assembly, lfs.FunctionNetwork]] = None,
        geometry: Optional[Geometry] = None,
    ) -> None:
        self.constant_kkt = constant_kkt
        # constant_kkt takes precedence over quadratic_objective_linear_constraints
        if self.constant_kkt:
            self.quadratic_objective_linear_constraints = False
        else:
            self.quadratic_objective_linear_constraints = quadratic_objective_linear_constraints

        self.optimization = Optimization()
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = NewtonOptimizer(
                quadratic_objective_linear_constraints=self.quadratic_objective_linear_constraints,
            )
        self.states : list[State] = []
        self.inner_parameterization_map = inner_parameterization_map
        self.has_been_setup = False
        self.has_constant_kkt_setup = False
        self.kkt_matrix = None
        self.kkt_matrix_inverse = None
        self.negative_kkt_matrix_inverse = None
        self.solved_states: Dict[str, csdl.Variable] = {}
        self.geometry = geometry
        self.assembly = None

        if assembly is not None:
            self.add_assembly(assembly)

    @property
    def K(self):
        """KKT matrix of the linear parameterization problem."""
        return self.kkt_matrix

    @property
    def K_inv(self):
        """Inverse of the KKT matrix."""
        return self.kkt_matrix_inverse

    @property
    def M(self):
        """Negative inverse of the KKT matrix (- K^{-1})."""
        return self.negative_kkt_matrix_inverse

    def _update_all_representations(self, res_geom: Optional[Any] = None) -> None:
        """Updates markers and representations across returned geometry, stored geometry, and assembly."""
        targets = []
        if res_geom is not None and res_geom not in targets:
            targets.append(res_geom)
        if self.geometry is not None and self.geometry not in targets:
            targets.append(self.geometry)
        if self.assembly is not None and self.assembly not in targets:
            targets.append(self.assembly)

        for target in targets:
            if hasattr(target, "update_representations"):
                target.update_representations()
            elif hasattr(target, "markers") and isinstance(target.markers, dict):
                for m in target.markers.values():
                    m.evaluate(target)

    def set_inner_parameterization_map(self, inner_map: Callable) -> None:
        '''
        Sets the callable for the inner parameterization map: P_i(alpha_s) -> geometry.
        '''
        self.inner_parameterization_map = inner_map


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

    
    def add_equality_constraint(
        self,
        constraint: Union[csdl.Variable, Connection, lfs.InterfaceCondition],
        desired_value: Optional[Union[csdl.Variable, npt.NDArray[np.float64], float]] = None,
        penalty: Optional[Union[float, npt.NDArray[np.float64], csdl.Variable]] = None,
    ):
        '''
        Add a constraint to the parameterization problem.

        Parameters
        ----------
        constraint : Union[csdl.Variable, Connection, lfs.InterfaceCondition]
            The constraint to be satisfied (CSDL Variable or Connection/Condition with evaluate_residual()).
        desired_value : Optional[Union[csdl.Variable, npt.NDArray[np.float64], float]]
            The desired value of the constraint. If None and constraint is a Connection/InterfaceCondition,
            defaults to 0.0.
        penalty : Union[float,np.ndarray,csdl.Variable], optional
            The penalty to be applied to the constraint, by default None. If None, lagrange multipliers are used.
            If not None, the penalty is the scaling factor for a quadratic constraint penalty term.
        '''
        if hasattr(constraint, "evaluate_residual"):
            res = constraint.evaluate_residual()
            if desired_value is None:
                desired_value = 0.0
            constraint = res
        elif desired_value is None:
            desired_value = 0.0

        equality_constraint = constraint - desired_value
        if isinstance(desired_value, csdl.Variable):
            equality_constraint.add_name(f'{desired_value.name}_enforcement_constraint')
        elif hasattr(constraint, "name") and constraint.name is not None:
            equality_constraint.add_name(f'{constraint.name}_enforcement_constraint')
        else:
            equality_constraint.add_name('enforcement_constraint')
        self.optimization.add_equality_constraint(equality_constraint, penalty=penalty)

    def add_connection(
        self,
        connection: Union[Connection, lfs.InterfaceCondition],
        desired_value: Optional[Union[csdl.Variable, npt.NDArray[np.float64], float]] = 0.0,
        penalty: Optional[Union[float, npt.NDArray[np.float64], csdl.Variable]] = None,
    ):
        """Adds a kinematic Connection or InterfaceCondition residual as an equality constraint."""
        self.add_equality_constraint(connection, desired_value=desired_value, penalty=penalty)

    def add_assembly(
        self,
        assembly: Union[Assembly, lfs.FunctionNetwork],
        penalty: Optional[Union[float, npt.NDArray[np.float64], csdl.Variable]] = None,
    ):
        """Adds all condition residuals from an assembly / function network as equality constraints."""
        residuals = assembly.evaluate_residuals()
        self.add_equality_constraint(residuals, desired_value=0.0, penalty=penalty)
        self.assembly = assembly

    
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
            cost_name = getattr(cost, 'name', 'cost')
            if isinstance(cost, (float,int)) or (cost.size==1):
                # Cost is a scalar
                objective = objective + csdl.vdot(state, cost*state)
            elif len(cost.shape) == 1:
                # Cost is a vector
                if cost.shape[0] != state.shape[0]:
                    raise ValueError('The cost vector must be the same size as the state. The cost provided, {}, is of shape {}, but the state is of shape {}'.format(cost_name, cost.shape, state.shape))
                objective = objective + csdl.vdot(state, cost*state)
            elif len(cost.shape) == 2:
                # Cost is a matrix
                if cost.shape[0] != cost.shape[1]:
                    raise ValueError('The cost matrix must be square. The cost provided, {}, is of shape {}'.format(cost_name, cost.shape))
                if cost.shape[0] != state.shape[0]:
                    raise ValueError('The cost matrix must be the same size as the state. The cost provided, {}, is of shape {}, but the state is of shape {}'.format(cost_name, cost.shape, state.shape))
                objective = objective + csdl.vdot(state, csdl.matvec(cost, state))
            else:
                raise ValueError('The cost must be a scalar, vector, or matrix. The cost provided, {}, is of shape {}'.format(cost_name, cost.shape))
        self.optimization.add_objective(objective)
        self.optimizer.add_optimization(self.optimization)


    @staticmethod
    def _state_hessian_block(state_object: State) -> npt.NDArray[np.float64]:
        """Returns the constant 2 * W Hessian matrix block for a single parameterization state."""
        sz = state_object.state.size
        cost = state_object.cost
        if isinstance(cost, (float, int)) or (getattr(cost, "size", 0) == 1):
            c_val = float(np.asarray(cost.value if hasattr(cost, "value") else cost).item())
            return 2.0 * np.eye(sz) * c_val
        elif getattr(cost, "ndim", 0) == 1 or len(getattr(cost, "shape", ())) == 1:
            c_val = np.asarray(cost.value if hasattr(cost, "value") else cost, dtype=np.float64)
            return 2.0 * np.diag(c_val)
        elif getattr(cost, "ndim", 0) == 2 or len(getattr(cost, "shape", ())) == 2:
            c_val = np.asarray(cost.value if hasattr(cost, "value") else cost, dtype=np.float64)
            return 2.0 * c_val
        raise ValueError(f"Unsupported cost shape for state '{state_object.state.name}'")

    def _compute_state_slices(self) -> Dict[str, tuple[int, int, tuple[int, ...]]]:
        """Builds a mapping from state name to (start_idx, end_idx, shape) in the stacked KKT vector."""
        slices: Dict[str, tuple[int, int, tuple[int, ...]]] = {}
        curr_idx = 0
        for state_object in self.states:
            var = state_object.state
            sz = var.size
            slices[var.name] = (curr_idx, curr_idx + sz, var.shape)
            curr_idx += sz
        return slices

    def _extract_solved_states(
        self,
        delta_y: csdl.Variable,
        state_slices: Dict[str, tuple[int, int, tuple[int, ...]]],
    ) -> Dict[str, csdl.Variable]:
        """Extracts primal state increments from stacked KKT step vector and returns updated states."""
        solved_states: Dict[str, csdl.Variable] = {}
        for state_obj in self.states:
            s_var = state_obj.state
            start_i, end_i, shape = state_slices[s_var.name]
            delta_s = delta_y[start_i:end_i]
            if len(shape) > 1 or (len(shape) == 1 and shape[0] > 1):
                delta_s = csdl.reshape(delta_s, shape)
            elif len(shape) == 0:
                delta_s = delta_s[0]
            s_solved = s_var + delta_s
            s_solved.add_name(f"{s_var.name}_solved")
            solved_states[s_var.name] = s_solved
        return solved_states

    def _invoke_inner_map(self, states_dict: Dict[str, csdl.Variable]) -> Optional[Any]:
        """Invokes self.inner_parameterization_map with the supplied states dictionary."""
        if self.inner_parameterization_map is not None and callable(self.inner_parameterization_map):
            import inspect
            sig = inspect.signature(self.inner_parameterization_map)
            if len(sig.parameters) == 1:
                return self.inner_parameterization_map(states_dict)
            try:
                return self.inner_parameterization_map(**states_dict)
            except TypeError:
                return self.inner_parameterization_map(states_dict)
        return None

    def setup_constant_kkt(self, geometric_variables: Optional[GeometricVariables] = None) -> None:
        '''
        Sets up the constant KKT system, computes the exact constant KKT matrix,
        and pre-factors / inverts it for instantaneous single-step back-substitution.
        
        Exploits block structure and symmetry:
        - Top-left block: weighting matrix W = 2 * diag(costs)
        - Bottom-left block: constraint Jacobian A = d c / d alpha_s
        - Top-right block: A.T (exact transpose of bottom-left, no extra derivative calls)
        - Bottom-right block: 0
        '''
        if geometric_variables is not None:
            for i in range(len(geometric_variables.computed_value)):
                self.add_variable(
                    geometric_variables.computed_value[i],
                    geometric_variables.desired_value[i],
                    geometric_variables.penalty_value[i]
                )

        self.setup()
        self.optimization.setup()

        num_dvs = sum(s.state.size for s in self.states)
        num_constraints = sum(c.constraint.size for c in self.optimization.equality_constraints)
        total_size = num_dvs + num_constraints

        kkt_matrix = np.zeros(shape=(total_size, total_size))

        # Top-left block: W = 2 * diag(cost) (from quadratic cost on states)
        counter = 0
        for state_object in self.states:
            sz = state_object.state.size
            kkt_matrix[counter : counter + sz, counter : counter + sz] = self._state_hessian_block(state_object)
            counter += sz

        # Bottom-left block (A) and Top-right block (A.T)
        constraint_counter = 0
        for constraint in self.optimization.equality_constraints:
            c_var = constraint.constraint
            c_size = c_var.size
            dv_counter = 0
            for state_object in self.states:
                dv = state_object.state
                dv_size = dv.size
                jac = csdl.derivative(c_var, dv).value
                jac = np.asarray(jac, dtype=np.float64).reshape((c_size, dv_size))
                # Bottom-left: A
                kkt_matrix[num_dvs + constraint_counter : num_dvs + constraint_counter + c_size,
                           dv_counter : dv_counter + dv_size] = jac
                # Top-right: A.T (exploit symmetry, no additional derivative calls!)
                kkt_matrix[dv_counter : dv_counter + dv_size,
                           num_dvs + constraint_counter : num_dvs + constraint_counter + c_size] = jac.T
                dv_counter += dv_size
            constraint_counter += c_size

        # Lagrangian gradient RHS: [grad_x f; c(x)]
        rhs_parts = []
        for state_object in self.states:
            obj_grad = self.optimization.compute_objective_gradient(design_variables=state_object.state)
            rhs_parts.append(csdl.reshape(obj_grad, (state_object.state.size,)))
        for constraint in self.optimization.equality_constraints:
            rhs_parts.append(csdl.reshape(constraint.constraint, (constraint.constraint.size,)))
        self._R_vec = csdl.concatenate(rhs_parts)

        self.kkt_matrix = kkt_matrix
        try:
            self.kkt_matrix_inverse = np.linalg.inv(kkt_matrix)
        except np.linalg.LinAlgError:
            self.kkt_matrix_inverse = np.linalg.pinv(kkt_matrix)
        self.negative_kkt_matrix_inverse = - self.kkt_matrix_inverse

        self._state_slices = self._compute_state_slices()
        self.has_constant_kkt_setup = True

    def solve_constant_kkt(self, geometric_variables: Optional[GeometricVariables] = None) -> None:
        '''
        Solves the constant KKT system in a single back-substitution (csdl.matvec)
        and automatically reevaluates the inner parameterization map in CSDL.
        Results are saved in self.solved_states.
        '''
        if self.inner_parameterization_map is None:
            raise ValueError(
                "When constant_kkt=True, an inner_parameterization_map must be provided to evaluate and update the parameterized geometry."
            )

        if not self.has_constant_kkt_setup:
            self.setup_constant_kkt(geometric_variables)

        # delta_y = negative_kkt_matrix_inverse @ R_vec (where negative_kkt_matrix_inverse = - K^{-1})
        delta_y = csdl.matvec(self.negative_kkt_matrix_inverse, self._R_vec)

        self.solved_states = self._extract_solved_states(delta_y, self._state_slices)
        res_geom = self._invoke_inner_map(self.solved_states)
        self._update_all_representations(res_geom)
        return None

    def _solve_quadratic_linear_direct(self) -> None:
        """
        Assembles the design-variable-dependent KKT matrix K(d) = [[2W, A(d)^T], [A(d), 0]] in CSDL
        and solves K(d) * delta_y = -R(d) via a single csdl.solve_linear call (no iterative loop).
        """
        from csdl_alpha.src.operations.linalg.blockmat import blockmat

        c_list = [csdl.reshape(c.constraint, (c.constraint.size,)) for c in self.optimization.equality_constraints]
        c_vec = csdl.concatenate(c_list)
        state_vars = [s.state for s in self.states]
        A_jac = csdl.derivative(c_vec, state_vars, as_block=True)

        num_dvs = sum(s.state.size for s in self.states)
        num_constraints = c_vec.size

        w_full = np.zeros((num_dvs, num_dvs), dtype=np.float64)
        counter = 0
        for state_object in self.states:
            sz = state_object.state.size
            w_full[counter : counter + sz, counter : counter + sz] = self._state_hessian_block(state_object)
            counter += sz

        w_var = csdl.Variable(value=w_full, name="param_kkt_W")
        z_var = csdl.Variable(value=np.zeros((num_constraints, num_constraints), dtype=np.float64), name="param_kkt_zero")
        kkt_mat = blockmat([[w_var, A_jac.T()], [A_jac, z_var]])

        rhs_parts = []
        for state_object in self.states:
            obj_grad = self.optimization.compute_objective_gradient(design_variables=state_object.state)
            rhs_parts.append(csdl.reshape(obj_grad, (state_object.state.size,)))
        rhs_parts.append(c_vec)
        rhs = -csdl.concatenate(rhs_parts)

        delta_y = csdl.solve_linear(kkt_mat, rhs)
        state_slices = self._compute_state_slices()
        self.solved_states = self._extract_solved_states(delta_y, state_slices)
        res_geom = self._invoke_inner_map(self.solved_states)
        self._update_all_representations(res_geom)

    def evaluate(
        self,
        geometric_variables: Optional[GeometricVariables] = None,
        constant_kkt: Optional[bool] = None,
    ) -> None:
        '''
        Evaluate the parameterization solver.
        Solves the parameterization and updates all geometry representations in-place.

        Parameters
        ----------
        geometric_variables : Optional[GeometricVariables], optional
            The geometric variables dataclass.
        constant_kkt : Optional[bool], optional
            Whether to use direct pre-factorized constant KKT solve. If None, uses self.constant_kkt.
        '''
        if constant_kkt or (constant_kkt is None and self.constant_kkt):
            self.solve_constant_kkt(geometric_variables)
            return None

        if geometric_variables is not None:
            for i in range(len(geometric_variables.computed_value)):
                computed_value = geometric_variables.computed_value[i]
                desired_value = geometric_variables.desired_value[i]
                penalty_value = geometric_variables.penalty_value[i]
                self.add_variable(computed_value, desired_value, penalty_value)

        self.setup()
        has_penalty_constraints = any(c.penalty is not None for c in self.optimization.equality_constraints)
        has_inequality_constraints = len(self.optimization.inequality_constraints) > 0
        if (
            self.quadratic_objective_linear_constraints
            and self.inner_parameterization_map is not None
            and not has_penalty_constraints
            and not has_inequality_constraints
            and len(self.optimization.equality_constraints) > 0
        ):
            self._solve_quadratic_linear_direct()
            return None

        if (
            self.quadratic_objective_linear_constraints
            and not has_penalty_constraints
            and hasattr(self.optimizer, "solver")
            and hasattr(self.optimizer.solver, "set_block_constant")
        ):
            if not self.optimizer.has_been_setup:
                self.optimizer.setup()
            for state_object in self.states:
                state = state_object.state
                w_mat = self._state_hessian_block(state_object)
                res = self.optimizer.solver.state_to_residual_map[state]
                self.optimizer.solver.set_block_constant(res, state, w_mat)

            if hasattr(self.optimizer.solver, "set_block_zero"):
                for i, si in enumerate(self.states):
                    ri = self.optimizer.solver.state_to_residual_map[si.state]
                    for j, sj in enumerate(self.states):
                        if i != j:
                            self.optimizer.solver.set_block_zero(ri, sj.state)

        self.optimizer.run()

        states_dict = {s.state.name: s.state for s in self.states}
        res_geom = self._invoke_inner_map(states_dict)
        self._update_all_representations(res_geom)
        return None


class ConstantKKTParameterizationSolver(ParameterizationSolver):
    '''
    Automated Constant KKT Geometry Parameterization Solver with pre-factorized KKT matrix.
    Used when the entire KKT system (both weighting matrix W and constraint Jacobian A) is
    constant and completely independent of design variables. Performs an instantaneous
    single-step direct solve (csdl.matvec) with zero Newton iterations.
    '''
    def __init__(
        self,
        inner_parameterization_map: Optional[Callable] = None,
        assembly: Optional[Union[Assembly, lfs.FunctionNetwork]] = None,
        geometry: Optional[Geometry] = None,
    ) -> None:
        super().__init__(
            inner_parameterization_map=inner_parameterization_map,
            constant_kkt=True,
            assembly=assembly,
            geometry=geometry,
        )


    