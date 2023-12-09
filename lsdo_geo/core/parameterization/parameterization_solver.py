import csdl
from python_csdl_backend import Simulator
import m3l
import numpy as np
import scipy.sparse as sps

import time

from dataclasses import dataclass

@dataclass
class ParameterizationSolver:
    '''
    A solver for achieving desired inputs to a model by manipulating the parameterization dofs. The parameterization dofs are minimally
    manipulated in order to achieve the desired inputs.

    Attributes
    ----------
    declared_inputs : dict[str,m3l.Variable] = None
        Dictionary of inputs to the parameterization solver. The keys are the names of the inputs and the values are the desired values for
        the declared quantities.
    residual_penalties : dict[str,m3l.Variable] = None
        Dictionary of penalty factors for the residuals associated with each input. The keys are the names of the inputs and the values are the
        penalty factors. If the penalty factor is None, a lagrange multiplier will be used for exact constraint satisfaction (recommended).
        If a penalty factor is specified, the residual will be enforced with a quadratic penalty method.
    declared_states : dict[str,m3l.Variable] = None
        Dictionary of states to the parameterization solver. The keys are the names of the states and the values are the states that will be
        minimally manipulated to achieve the desired inputs.
    state_penalties : dict[str,m3l.Variable] = None
        Dictionary of penalty factors for the states. The keys are the names of the states and the values are the penalty factors. This is used
        to weight the state in the objective function. States with higher penalties will be used less to achieve the desired inputs.

    Methods
    -------
    declare_input(name:str, input:m3l.Variable, penalty_factor:m3l.Variable=None)
        Declares an input to the parameterization solver.
    declare_state(name:str, state:m3l.Variable, penalty_factor:m3l.Variable=1.)
        Declares a state to the parameterization solver.
    evaluate(inputs:dict[str,m3l.Variable], plot:bool=False, plot_iterations:bool=False)
        Evaluates the parameterization solver.
    plot()
        Plots the results of the parameterization solver.
    '''
    declared_inputs : dict[str,m3l.Variable] = None
    residual_penalties : dict[str,m3l.Variable] = None
    declared_states : dict[str,m3l.Variable] = None
    state_penalties : dict[str,m3l.Variable] = None

    def declare_input(self, name:str, input:m3l.Variable, penalty_factor:m3l.Variable=None):
        '''
        Declares an input to the parameterization solver.

        Parameters
        ----------
        name : str
            Name of the input.
        input : m3l.Variable
            The variable that was calculated using the model. This is the input that will be driven to the desired value.
        penalty_factor : float (optional) = None
            The penalty factor for the residual associated with this input. This is used to weight the residual in the objective function. 
            If the penalty factor is None, a lagrange multiplier will be used for exact constraint satisfaction (recommended). 
            If a penalty factor is specified, the residual will be enforced with a quadratic penalty method.
        '''
        if type(penalty_factor) is not None and (type(penalty_factor) is float or type(penalty_factor) is int):
            penalty_factor = m3l.Variable(value=penalty_factor, shape=input.shape)

        if self.declared_inputs is None:
            self.declared_inputs = {}
        self.declared_inputs[name] = input.copy()

        if penalty_factor is not None:
            if self.residual_penalties is None:
                self.residual_penalties = {}
            self.residual_penalties[name] = penalty_factor

    def declare_state(self, name:str, state:m3l.Variable, penalty_factor:m3l.Variable=1.):
        '''
        Declares a state to the parameterization solver.

        Parameters
        ----------
        state : m3l.Variable
            The state that will be minimally manipulated to achieve the desired inputs.
        penalty_factor : m3l.Variable (optional) = 1.
            The pemalty/cost factor for the state. This is used to weight the state in the objective function. States with higher penalties will
            be used less to achieve the desired inputs.
        '''
        if type(penalty_factor) is float or type(penalty_factor) is int:
            penalty_factor = m3l.Variable(value=penalty_factor, shape=state.shape)

        if self.declared_states is None:
            self.declared_states = {}
        self.declared_states[name] = state.copy()

        if self.state_penalties is None:
            self.state_penalties = {}
        self.state_penalties[name] = penalty_factor

    def evaluate(self, inputs:dict[str,m3l.Variable], plot:bool=False, plot_iterations:bool=False):
        '''
        Evaluates the parameterization solver. The parameterization solver achieves the desired parameterization inputs by minimally manipulating the
        parameterziation states.

        Parameters
        ----------
        inputs : dict[str,m3l.Variable]
            Dictionary of inputs to the parameterization solver. The keys are the names of the inputs and the values are the desired values for 
            the declared quantities.
        plot : bool (optional) = False
            If True, plots the parameterization solver results.
        plot_iterations : bool (optional) = False
            If True, plots the parameterization solver results for each iteration.
        '''
        if self.declared_inputs is None:
            raise ValueError('No inputs declared.')
        if len(inputs) != len(self.declared_inputs):
            raise ValueError('Number of inputs does not match number of declared inputs.')
        for input_name, input in inputs.items():
            if input_name not in self.declared_inputs:
                raise ValueError('Input name not declared.')
            if input.shape != self.declared_inputs[input_name].shape:
                raise ValueError('Input shape does not match declared input shape.')
            
        if self.declared_states is None:
            raise ValueError('No states declared.')

        if self.residual_penalties is None:
            self.residual_penalties = {}
        for input_name, input in self.declared_inputs.items():
            if input_name not in self.residual_penalties:
                self.residual_penalties[input_name] = None
        if self.state_penalties is None:
            self.state_penalties = {}
        for state_name, state in self.declared_states.items():
            if state_name not in self.state_penalties:
                self.state_penalties[state_name] = 1.

        parameterization_solver_operation = ParameterizationSolverM3LOperation(declared_inputs=self.declared_inputs,
                                                                               residual_penalties=self.residual_penalties,
                                                                               declared_states=self.declared_states,
                                                                               state_penalties=self.state_penalties)
        output_variables = parameterization_solver_operation.evaluate(inputs=inputs)
        
        if plot:
            self.plot()

        return output_variables
    
    def plot(self):
        '''
        Plots the results of the parameterization solver.
        '''
        pass

class ParameterizationSolverM3LOperation(m3l.ExplicitOperation):
    '''
    The M3L operation for running the parameterization solver.
    '''
    def initialize(self, kwargs):
        self.parameters.declare('declared_inputs', types=dict)
        self.parameters.declare('residual_penalties', types=dict)
        self.parameters.declare('declared_states', types=dict)
        self.parameters.declare('state_penalties', types=dict)

    def assign_attributes(self):
        self.declared_inputs = self.parameters['declared_inputs'].copy()
        self.residual_penalties = self.parameters['residual_penalties'].copy()
        self.declared_states = self.parameters['declared_states'].copy()
        self.state_penalties = self.parameters['state_penalties'].copy()

    def compute(self):
        csdl_model = ParameterizationSolverCSDL(declared_inputs=self.declared_inputs, residual_penalties=self.residual_penalties,
                                                declared_states=self.declared_states, state_penalties=self.state_penalties,
                                                output_names=self.output_names, arguments=self.arguments)
        return csdl_model

    # def compute_derivates(self):
    #     pass

    def evaluate(self, inputs:dict[str,m3l.Variable]):
        
        input_names = ''
        for input_name, input in inputs.items():
            input_names += f'{input_name}_'
        self.name = f'parameterization_solver_with_inputs_{input_names}'

        self.arguments = {}
        for input_name, input_variable in inputs.items():
            self.arguments[input_variable.name] = input_variable

        output_shapes = {}
        output_variables = {}
        self.output_names = {}
        for output_name, output in self.declared_states.items():
            output_shapes[output_name] = output.shape
            output_variables[output_name] = m3l.Variable(name=output_name, shape=output_shapes[output_name], operation=self)
            self.output_names[output_name] = output.name
        
        # create csdl model for in-line evaluations
        operation_csdl = self.compute()
        sim = Simulator(operation_csdl)
        for input_name, input_variable in inputs.items():
            sim[input_variable.name] = input_variable.value
        sim.run()
        for output_name, output_variable in output_variables.items():
            output_variable.value = sim[output_name]

        return output_variables


'''
IDEA: If solution fails (NaN or max iteration), the constraints are enforced with penalty method instead of lagrange multipliers
    which turns the problem into a regularized least squares problem (Note: There is a nonlinear operation, so can't use pseudo-inverse).
COUNTER IDEA: This is a bit problematic because I don't want to change how it's solved mid-optimization since that can mess with the differentiability.
    Instead, I allow for the user to specify a penalty factor for each input. If the penalty factor is None, a lagrange multiplier will be used for 
    exact. This way, if the solution fails and the user wants to see what's going on, they can switch to a penalty method and see if that works.
'''
class ParameterizationSolverCSDL(csdl.Model):
    '''
    CSDL model for running the parameterization solver. This computes the parameterization dof needed to achieve the desired input parameters.
    The objective is to minimize the weighted norm of the parameterization dof.
    '''
    def initialize(self):
        self.parameters.declare('declared_inputs', types=dict)
        self.parameters.declare('residual_penalties', types=dict)
        self.parameters.declare('declared_states', types=dict)
        self.parameters.declare('state_penalties', types=dict)
        self.parameters.declare('output_names', types=dict)
        self.parameters.declare('arguments', types=dict)

    def define(self):
        self.declared_inputs = self.parameters['declared_inputs'].copy()
        self.residual_penalties = self.parameters['residual_penalties'].copy()
        self.declared_states = self.parameters['declared_states'].copy()
        self.state_penalties = self.parameters['state_penalties'].copy()
        self.output_names = self.parameters['output_names'].copy()
        self.arguments = self.parameters['arguments']
        
        inputs_csdl = []
        for input_name, input in self.arguments.items():
        # for input_name in self.declared_inputs:
            # input = self.arguments[input_name]
            # input_variable_name = input.name    # This will be the M3L variable name instead of the user-declared name
            inputs_csdl.append(self.declare_variable(input_name, val=input.value))

        states_tuple = csdl.custom(*inputs_csdl,
                                op=GeometryParameterizationSolverOperation(declared_inputs=self.declared_inputs, 
                                                                           residual_penalties=self.residual_penalties,
                                                                            declared_states=self.declared_states, 
                                                                            state_penalties=self.state_penalties))

        counter = 0
        for output_name, output in self.declared_states.items():
            output_csdl = states_tuple[counter]
            # output_csdl = states[output.name]   # NOTE: TODO: This is not a dictionary. Can it be a dictionary? 
            # -- Either needs to be a dict or need indexing of the output.
            self.register_output(self.output_names[output_name], output_csdl)
            counter += 1

        for input_name, input in self.declared_inputs.items():
            input_lagrange_multiplier = states_tuple[counter]
            self.register_output(input_name + '_lagrange_multipliers', input_lagrange_multiplier)
            counter += 1

        # input_vector_length = 0
        # for input_name, input in inputs.items():
        #     input_vector_length += np.prod(input.shape)

        # inputs_csdl = self.create_output('parameterization_inputs', shape=(input_vector_length,))

        # starting_index = 0
        # for input_name, input in inputs.items():
        #     input_csdl = self.declare_variable(input.name, val=input.value)

        #     num_flattened_inputs = np.prod(input.shape)
        #     flattened_input_csdl = csdl.reshape(input_csdl, new_shape=(num_flattened_inputs,))

        #     inputs_csdl[starting_index:starting_index+num_flattened_inputs] = flattened_input_csdl
        #     starting_index += num_flattened_inputs

        # self.register_output('states', states)
        # self.register_output('parameterization_lagrange_multipliers', parameterization_lagrange_multipliers)


class GeometryParameterizationSolverOperation(csdl.CustomImplicitOperation):
    """
    CSDL Custom Implicit Operation for computing the parameterization solver. This class actually has the code for computing the residual
    and jacobians for the implicit system.
    """
    def initialize(self):
        self.parameters.declare('declared_inputs', types=dict)
        self.parameters.declare('residual_penalties', types=dict)
        self.parameters.declare('declared_states', types=dict)
        self.parameters.declare('state_penalties', types=dict)

    def define(self):
        self.declared_inputs = self.parameters['declared_inputs'].copy()
        self.residual_penalties = self.parameters['residual_penalties'].copy()
        self.declared_states = self.parameters['declared_states'].copy()
        self.state_penalties = self.parameters['state_penalties'].copy()

        for input_name, input in self.declared_inputs.items():
            self.add_input(name=input_name, val=input.copy().value)

        for state_name, state in self.declared_states.items():
            self.add_output(name=state_name, shape=state.shape, val=state.copy().value)
        for input_name, input in self.declared_inputs.items():
            self.add_output(input_name+'_lagrange_multipliers', shape=input.shape)

        for state_name_1 in self.declared_states:
            for state_name_2 in self.declared_states:
                self.declare_derivatives(state_name_1, state_name_2)
        for state_name in self.declared_states:
            for input_name in self.declared_inputs:
                self.declare_derivatives(state_name, input_name+'_lagrange_multipliers')
        for input_name in self.declared_inputs:
            for state_name in self.declared_states:
                self.declare_derivatives(input_name+'_lagrange_multipliers', state_name)
        for input_name_1 in self.declared_inputs:
            for input_name_2 in self.declared_inputs:
                self.declare_derivatives(input_name_1+'_lagrange_multipliers', input_name_2+'_lagrange_multipliers')

        for state_name, state in self.declared_states.items():
            for input_name, input in self.declared_inputs.items():
                self.declare_derivatives(state_name, input_name)
        for input_name_1 in self.declared_inputs:
            for input_name_2 in self.declared_inputs:
                self.declare_derivatives(input_name_1+'_lagrange_multipliers', input_name_2)

        self.linear_solver = csdl.DirectSolver()
        self.nonlinear_solver = csdl.NewtonSolver(maxiter=10)

        # Construct CSDL model for evaluation
        evaluation_m3l_model = m3l.Model()
        # for forward_model_output_name, forward_model_output in self.declared_inputs.items():
        #     evaluation_m3l_model.register_output(forward_model_output)
        evaluation_m3l_model.register_output(self.declared_inputs)
        evaluation_model = evaluation_m3l_model.assemble()
        self.evaluation_simulator = Simulator(evaluation_model)

        variable_names = []
        for constraint_name, constraint in self.declared_inputs.items():
            if type(constraint.operation) is m3l.Norm:
                variable_that_is_linearly_mapped_to = constraint.operation.arguments['x']
            else:   # Assume it is a linear constraint
                variable_that_is_linearly_mapped_to = constraint
            variable_that_is_linearly_mapped_to_simulator_name = variable_that_is_linearly_mapped_to.operation.name+'.'\
                                                                +variable_that_is_linearly_mapped_to.name
            variable_names.append(variable_that_is_linearly_mapped_to_simulator_name)

        self.evaluation_simulator.run()

        self.linear_derivatives = self.evaluation_simulator.compute_totals(of=variable_names, wrt=list(self.declared_states.keys()))

        # # Precompute linear maps for memory and (definitely NOT time saving)
        # # from m3l import check_if_variable_is_upstream
        # from m3l import compute_mapping_from_upstream_variable
        # self.linear_derivatives = {}
        # for constraint_name, constraint in self.declared_inputs.items():
        #     linear_output = constraint.operation.arguments['x']
        #     for state_name, state in self.declared_states.items():
        #         # # search graph to see if state is in constraint graph
        #         # is_state_upstream = check_if_variable_is_upstream(linear_output, state)
        #         # if not is_state_upstream:
        #         #     continue          

        #         # if state is in constraint graph, then compute linear map
        #         linear_map = compute_mapping_from_upstream_variable(linear_output, state)
        #         if linear_map is None:
        #             linear_map = sps.lil_matrix((linear_output.shape[0], state.shape[0]))
        #         self.linear_derivatives[(constraint_name,state_name)] = linear_map.toarray()


        # Create simulator objects for nonlinear operations
        for constraint_name, constraint in self.declared_inputs.items():
            nonlinear_operation = constraint.operation
            if type(nonlinear_operation) is m3l.Norm:
                nonlinear_operation.compute_derivatives()

        # self.linear_derivatives = {}
        # for constraint_name, constraint in self.declared_inputs.items():
        #     vector = constraint.operation.arguments['x']
        #     geometry_coefficients = vector.operation.arguments['x']
        #     vector_map = vector.operation.map
        #     geometry_coefficients_map = 
        #     for state_name, state in self.declared_states.items():
        #         variable = constraint.operation.arguments['x']
                

        # self.linear_maps = {}
        # for state_name, state in self.declared_states.items():
        #     # Perform graph searching to precompute total linear maps.
        #     current_operation = constraint.operation
        #     while state not in 

        #     self.linear_maps[constraint_name,state_name] = linear_map

        # Construct CSDL model for derivatives
        # NOTE: DON'T DO THIS. IT'S HARD AND WILL BE SLOW AND CSDL WILL DO IT BETTER LATER. INSTEAD, HARD-CODE THE LINEAR+NL STRUCTURE.
        # derivatives_m3l_model = m3l.Model()
        # # for forward_model_output_name, forward_model_output in self.declared_inputs.items():
        #     # derivatives_m3l_model.register_output(forward_model_output)
        # derivatives_m3l_model.register_output(self.declared_inputs)
        # derivatives_model = derivatives_m3l_model.assemble()
        # self.derivative_evaluation_simulator = Simulator(derivatives_model)

        # system_representation = system_parameterization.system_representation

        # geometry_parameterizations = system_parameterization.geometry_parameterizations

        # # Collect the parameterizations which have free dof (one parameterization is like one FFDSet)
        # free_geometry_parameterizations = {}
        # for geometry_parameterization_name, geometry_parameterization in geometry_parameterizations.items():
        #     if geometry_parameterization.num_affine_free_dof != 0:
        #         free_geometry_parameterizations[geometry_parameterization_name] = geometry_parameterization
        #         # NOTE! Hardcoding for one FFDSet as the only geometry parameterization
        #         ffd_set = geometry_parameterization

        # num_affine_free_dof = ffd_set.num_affine_free_dof

        # input_vector_length = 0
        # for input_name, input in system_parameterization.inputs.items():
        #     input_vector_length += np.prod(input.shape)


        # self.add_input('parameterization_inputs', shape=(input_vector_length,))
        # self.add_output('ffd_free_dof', val=np.zeros((num_affine_free_dof,)))
        # self.add_output('parameterization_lagrange_multipliers', val=np.zeros((input_vector_length,)))
        # self.declare_derivatives('ffd_free_dof', 'ffd_free_dof')
        # self.declare_derivatives('ffd_free_dof', 'parameterization_lagrange_multipliers')
        # self.declare_derivatives('parameterization_lagrange_multipliers', 'ffd_free_dof')
        # self.declare_derivatives('parameterization_lagrange_multipliers', 'parameterization_lagrange_multipliers')
        # self.declare_derivatives('ffd_free_dof', 'parameterization_inputs')
        # self.declare_derivatives('parameterization_lagrange_multipliers', 'parameterization_inputs')

        # # self.declare_derivatives('*','*')

        # self.linear_solver = csdl.ScipyKrylov()
        # self.nonlinear_solver = csdl.NewtonSolver(solve_subsystems=True, maxiter=10)

        # # Precompute maps
        # ffd_free_dof_to_local_ffd_control_points = ffd_set.affine_block_deformations_map.dot(ffd_set.free_affine_section_properties_map)
        # # Apply rotation to each map
        # rotation_matrices = []
        # for ffd_block in list(ffd_set.ffd_blocks.values()):
        #     if ffd_block.num_dof == 0:
        #         continue
        #     rotation_matrices.extend([ffd_block.local_to_global_rotation]*ffd_block.num_control_points)
        # local_to_global = sps.block_diag(rotation_matrices)
        # ffd_free_dof_to_ffd_control_points = local_to_global.dot(ffd_free_dof_to_local_ffd_control_points)
        # # Map to embedded entities (geometry control points that are embedded) (need to expand/repeat map from num_ffd_cp to num_ffd_cp*3)
        # NUM_PHYSICAL_DIMENSIONS = 3
        # expanded_ffd_embedded_entity_map = sps.lil_matrix(
        #     (ffd_set.embedded_entities_map.shape[0]*NUM_PHYSICAL_DIMENSIONS,ffd_set.embedded_entities_map.shape[1]*NUM_PHYSICAL_DIMENSIONS))
        # for i in range(NUM_PHYSICAL_DIMENSIONS):
        #     input_indices = np.arange(0,ffd_set.embedded_entities_map.shape[1])*NUM_PHYSICAL_DIMENSIONS + i
        #     output_indices = np.arange(0,ffd_set.embedded_entities_map.shape[0])*NUM_PHYSICAL_DIMENSIONS + i
        #     expanded_ffd_embedded_entity_map[np.ix_(output_indices,input_indices)] = ffd_set.embedded_entities_map
        # expanded_ffd_embedded_entity_map = expanded_ffd_embedded_entity_map.tocsc()
        # ffd_free_dof_to_embedded_entities = expanded_ffd_embedded_entity_map.dot(ffd_free_dof_to_ffd_control_points)
        # # Map from embedded entities to all geometry control points
        # initial_system_representation_geometry = system_representation.spatial_representation.control_points['geometry'].copy()
        # indexing_indices = []
        # for ffd_block in list(geometry_parameterization.active_ffd_blocks.values()):
        #     ffd_block_embedded_primitive_names = list(ffd_block.embedded_entities.keys())
        #     ffd_block_embedded_primitive_indices = []
        #     for primitive_name in ffd_block_embedded_primitive_names:
        #         ffd_block_embedded_primitive_indices.extend(list(
        #             system_representation.spatial_representation.primitive_indices[primitive_name]['geometry']))
        #     indexing_indices.extend(ffd_block_embedded_primitive_indices)

        # indexing_indices_array = np.array(indexing_indices)
        # expanded_indexing_indices = indexing_indices_array*NUM_PHYSICAL_DIMENSIONS
        # row_indices = np.concatenate((expanded_indexing_indices, expanded_indexing_indices + 1, expanded_indexing_indices + 2))
        # num_points = len(indexing_indices)
        # col_indices_array = np.arange(num_points)*NUM_PHYSICAL_DIMENSIONS
        # col_indices = np.concatenate((col_indices_array, col_indices_array + 1, col_indices_array + 2))
        # num_points_system_representation = initial_system_representation_geometry.shape[0]
        # data = np.ones((num_points*3))
        # geometry_assembly_map = sps.coo_matrix((data, (row_indices, col_indices)),
        #                 shape=(num_points_system_representation*NUM_PHYSICAL_DIMENSIONS, num_points*NUM_PHYSICAL_DIMENSIONS))
        # geometry_assembly_map = geometry_assembly_map.tocsc()
        # ffd_free_dof_to_geometry_control_points = geometry_assembly_map.dot(ffd_free_dof_to_embedded_entities)
        # # Map from geometry control points to mapped arrays
        # mapped_array_mappings = []
        # self.mapped_array_indices = {}
        # mapped_array_indices_counter = 0
        # for input_name, parameterization_input in system_parameterization.inputs.items():
        #     if type(parameterization_input.quantity) is am.MappedArray:
        #         mapped_array = parameterization_input.quantity
        #     elif type(parameterization_input.quantity) is am.NonlinearMappedArray:
        #         mapped_array = parameterization_input.quantity.input
        #     num_mapped_array_outputs = mapped_array.linear_map.shape[0]
        #     self.mapped_array_indices[input_name] = \
        #         np.arange(mapped_array_indices_counter*NUM_PHYSICAL_DIMENSIONS, 
        #                   (mapped_array_indices_counter + num_mapped_array_outputs)*NUM_PHYSICAL_DIMENSIONS)
        #     mapped_array_indices_counter += num_mapped_array_outputs

        #     mapped_array_map = mapped_array.linear_map
        #     expanded_mapped_array_map = sps.lil_matrix(
        #         (mapped_array_map.shape[0]*NUM_PHYSICAL_DIMENSIONS,mapped_array_map.shape[1]*NUM_PHYSICAL_DIMENSIONS))
        #     for i in range(NUM_PHYSICAL_DIMENSIONS):
        #         input_indices = np.arange(0,mapped_array_map.shape[1])*NUM_PHYSICAL_DIMENSIONS + i
        #         output_indices = np.arange(0,mapped_array_map.shape[0])*NUM_PHYSICAL_DIMENSIONS + i
        #         expanded_mapped_array_map[np.ix_(output_indices,input_indices)] = mapped_array_map
        #     expanded_mapped_array_map = expanded_mapped_array_map.tocsc()

        #     # mapped_array_mapping = sps.block_diag([mapped_array.linear_map]*NUM_PHYSICAL_DIMENSIONS)
        #     mapped_array_mappings.append(expanded_mapped_array_map)

        # mapped_arrays_map = sps.vstack(mapped_array_mappings).tocsc()

        # self.ffd_free_dof_to_mapped_arrays = mapped_arrays_map.dot(ffd_free_dof_to_geometry_control_points)


    def evaluate_residuals(self, inputs, outputs, residuals):
        constraint_values = {}
        total_derivatives = {}
        for constraint_name, constraint in self.declared_inputs.items():
            if type(constraint.operation) is m3l.Norm:
                # NOTE: NEXT LINE IS PROBLEMATIC IF GEOMETRY WAS DEFORMED WHEN THIS WAS EVALUATED
                vectors_for_constraints = constraint.operation.arguments['x'].value.copy() 

                variable_that_is_linearly_mapped_to = constraint.operation.arguments['x']
                variable_that_is_linearly_mapped_to_simulator_name = variable_that_is_linearly_mapped_to.operation.name+'.'\
                                                                    +variable_that_is_linearly_mapped_to.name
                for state_name, declared_state in self.declared_states.items():
                    if declared_state.operation is not None:
                        state_dict_name = declared_state.operation.name+'.'+state_name
                    else:
                        state_dict_name = state_name

                    vectors_for_constraints += self.linear_derivatives[
                        (variable_that_is_linearly_mapped_to_simulator_name,state_dict_name)].dot(outputs[state_name])
                    # vectors_for_constraints += self.linear_derivatives[
                    #     (constraint_name,state_name)].dot(outputs[state_name])
                    
                nonlinear_operation = constraint.operation
                nonlinear_operation_simulator = nonlinear_operation.sim
                nonlinear_operation_simulator['x'] = vectors_for_constraints
                nonlinear_operation_simulator.run()
                vector_norms_for_constraints = nonlinear_operation_simulator[constraint.name]

                # vector_norms_for_constraints = np.linalg.norm(vectors_for_constraints)

                constraint_values[constraint_name] = vector_norms_for_constraints - inputs[constraint_name]

                nonlinear_derivative = nonlinear_operation_simulator.compute_totals(of=constraint.name, 
                                            wrt='x')[(constraint.name, 'x')]
            else:   # no nonlinear operation. The vector itself is the constraint
                # NOTE: NEXT LINE IS PROBLEMATIC IF GEOMETRY WAS DEFORMED WHEN THIS WAS EVALUATED
                vectors_for_constraints = constraint.value.copy() 

                variable_that_is_linearly_mapped_to = constraint
                variable_that_is_linearly_mapped_to_simulator_name = variable_that_is_linearly_mapped_to.operation.name+'.'\
                                                                    +variable_that_is_linearly_mapped_to.name
                for state_name, declared_state in self.declared_states.items():
                    if declared_state.operation is not None:
                        state_dict_name = declared_state.operation.name+'.'+state_name
                    else:
                        state_dict_name = state_name

                    vectors_for_constraints += self.linear_derivatives[
                        (variable_that_is_linearly_mapped_to_simulator_name,state_dict_name)].dot(outputs[state_name])
                    # vectors_for_constraints += self.linear_derivatives[
                    #     (constraint_name,state_name)].dot(outputs[state_name])

                constraint_values[constraint_name] = vectors_for_constraints - inputs[constraint_name]

                nonlinear_derivative = np.eye(constraint.shape[0])
            
            for state_name, state in self.declared_states.items():
                if declared_state.operation is not None:
                    state_dict_name = declared_state.operation.name+'.'+state_name
                else:
                    state_dict_name = state_name

                total_derivatives[(constraint_name,state_name)] = \
                    nonlinear_derivative.dot(
                        # self.linear_derivatives[(constraint_name,state_name)])
                        self.linear_derivatives[(variable_that_is_linearly_mapped_to_simulator_name,state_dict_name)])

        # NOTE: KEEP THIS METHOD AROUND BEACUSE IT CAN ACCOUNT FOR OTHER NON-FREE DOF PARAMETERS

        # for state_name, declared_state in self.declared_states.items():
        #     self.evaluation_simulator[state_name] = outputs[state_name]

        # self.evaluation_simulator.run()
        # input_variable_names = []
        # for input_name, input in self.declared_inputs.items():
        #     input_variable_names.append(input.operation.name+'.'+input.name)
        # state_variable_names = []
        # for state_name, state in self.declared_states.items():
        #     if state.operation is not None:
        #         state_variable_names.append(state.operation.name+'.'+state.name)
        #     else:
        #         state_variable_names.append(state.name)
        # # derivatives = self.evaluation_simulator.compute_totals(of=list(self.declared_inputs.keys()), wrt=list(self.declared_states.keys()))
        # # derivatives = self.evaluation_simulator.compute_totals(of=input_variable_names, wrt=state_variable_names)
        # nonlinear_derivatives = {}
        # total_derivatives = {}
        # for constraint_name, constraint in self.declared_inputs.items():
        #     nonlinear_operation = constraint.operation
        #     constraint_csdl_name = constraint.operation.name+'.'+constraint.name
        #     nonlinear_derivatives[constraint_name] = self.evaluation_simulator.compute_totals(of=constraint_csdl_name, 
        #                                 wrt=nonlinear_operation.name + '.x')[(constraint_csdl_name, nonlinear_operation.name + '.x')]
        #     # nonlinear_derivatives[constraint_name] = constraint.operation.sim.compute_totals(of=constraint_csdl_name, 
        #     #                             wrt=nonlinear_operation.name + '.x')[(constraint_csdl_name, nonlinear_operation.name + '.x')]
            
        #     variable_that_is_linearly_mapped_to = constraint.operation.arguments['x']
        #     variable_that_is_linearly_mapped_to_simulator_name = variable_that_is_linearly_mapped_to.operation.name+'.'\
        #                                                         +variable_that_is_linearly_mapped_to.name
            
        #     for state_name, state in self.declared_states.items():
        #         if declared_state.operation is not None:
        #             state_dict_name = declared_state.operation.name+'.'+state_name
        #         else:
        #             state_dict_name = state_name
                    
        #         total_derivatives[(constraint_name,state_name)] = \
        #             nonlinear_derivatives[constraint_name].dot(
        #                 self.linear_derivatives[(variable_that_is_linearly_mapped_to_simulator_name,state_dict_name)])

        dobjective_dstate = {}
        for state_name, declared_state in self.declared_states.items():
            state_vector = outputs[state_name]
            # state_penalties[state_name] = self.state_penalties[state_name].value * state_vector.T.dot(state_vector)   # x^T*alpha*x
            dobjective_dstate[state_name] = 2*self.state_penalties[state_name].value * state_vector

        dconstraint_penalty_dx = {}  # dictionary across states (contract different constraints with their lagrange multipliers)
        for state_name, declared_state in self.declared_states.items():
            dconstraint_penalty_dx[state_name] = np.zeros((declared_state.shape[0],))     # Preallocating so I can += the dict element later
            for constraint_name, constraint_value in self.declared_inputs.items():
                constraint_lagrange_multipliers = outputs[constraint_name+'_lagrange_multipliers']
                dc_dx = total_derivatives[(constraint_name, state_name)]
                dconstraint_penalty_dx[state_name] += constraint_lagrange_multipliers.dot(dc_dx)

        # constraint_values = {}
        # for constraint_name, declared_constraint_variable in self.declared_inputs.items():
        #     input_dict_name = declared_constraint_variable.operation.name+'.'+declared_constraint_variable.name
        #     constraint_values[constraint_name] = self.evaluation_simulator[input_dict_name] - inputs[constraint_name]
        
        for state_name, declared_state in self.declared_states.items():
            residuals[state_name] = dobjective_dstate[state_name] + dconstraint_penalty_dx[state_name]
        for constraint_name, constraint_value in self.declared_inputs.items():
            residuals[constraint_name+'_lagrange_multipliers'] = constraint_values[constraint_name]

        print('CONSTRAINT VALUES: ', constraint_values)


    def compute_derivatives(self, inputs, outputs, derivatives):
        total_derivatives = {}
        constraint_nonlinear_second_derivatives_dict = {}
        for constraint_name, constraint in self.declared_inputs.items():
            if type(constraint.operation) is m3l.Norm:
                vectors_for_constraints = constraint.operation.arguments['x'].value.copy()
                variable_that_is_linearly_mapped_to = constraint.operation.arguments['x']
                variable_that_is_linearly_mapped_to_simulator_name = variable_that_is_linearly_mapped_to.operation.name+'.'\
                                                                    +variable_that_is_linearly_mapped_to.name
                for state_name, declared_state in self.declared_states.items():
                    if declared_state.operation is not None:
                        state_dict_name = declared_state.operation.name+'.'+state_name
                    else:
                        state_dict_name = state_name

                    vectors_for_constraints += self.linear_derivatives[
                        (variable_that_is_linearly_mapped_to_simulator_name,state_dict_name)].dot(outputs[state_name])
                    # vectors_for_constraints += self.linear_derivatives[
                    #     (constraint_name,state_name)].dot(outputs[state_name])
                    
                nonlinear_operation = constraint.operation
                nonlinear_operation_simulator = nonlinear_operation.sim
                nonlinear_operation_simulator['x'] = vectors_for_constraints
                nonlinear_operation_simulator.run()

                # nonlinear_derivatives[constraint_name] = nonlinear_operation_simulator.compute_totals(of=constraint.name, 
                #                             wrt='x')[(constraint.name, 'x')]
                
                # for state_name, state in self.declared_states.items():
                #     if declared_state.operation is not None:
                #         state_dict_name = declared_state.operation.name+'.'+state_name
                #     else:
                #         state_dict_name = state_name
                        
                #     total_derivatives[(constraint_name,state_name)] = \
                #         nonlinear_derivatives[constraint_name].dot(
                #             self.linear_derivatives[(variable_that_is_linearly_mapped_to_simulator_name,state_dict_name)])
                    

                # Get second derivatives of nonlinear operations
                nonlinear_derivative_simulator = nonlinear_operation.derivative_sim
                nonlinear_operation_input_value = vectors_for_constraints
                nonlinear_derivative_simulator['x'] = nonlinear_operation_input_value
                nonlinear_derivative_simulator.run()

                nonlinear_derivatives = nonlinear_derivative_simulator[f'd{constraint.name}_dx']

                for state_name, state in self.declared_states.items():
                    if declared_state.operation is not None:
                        state_dict_name = declared_state.operation.name+'.'+state_name
                    else:
                        state_dict_name = state_name
                        
                    total_derivatives[(constraint_name,state_name)] = \
                        nonlinear_derivatives.dot(
                            # self.linear_derivatives[(constraint_name,state_name)])
                            self.linear_derivatives[(variable_that_is_linearly_mapped_to_simulator_name,state_dict_name)])

                nonlinear_second_derivatives = nonlinear_derivative_simulator.compute_totals(of=f'd{constraint.name}_dx',
                                                                                            wrt='x')
                constraint_nonlinear_second_derivatives_dict[constraint_name] = nonlinear_second_derivatives

            
            else:   # Assuming the vector itself is the constraint
                vectors_for_constraints = constraint.value.copy()
                # if type(constraint.operation) is m3l.Norm:
                #     variable_that_is_linearly_mapped_to = constraint.operation.arguments['x']
                # else:   # Assume it is a linear constraint
                variable_that_is_linearly_mapped_to = constraint
                variable_that_is_linearly_mapped_to_simulator_name = variable_that_is_linearly_mapped_to.operation.name+'.'\
                                                                    +variable_that_is_linearly_mapped_to.name
                for state_name, declared_state in self.declared_states.items():
                    if declared_state.operation is not None:
                        state_dict_name = declared_state.operation.name+'.'+state_name
                    else:
                        state_dict_name = state_name

                    vectors_for_constraints += self.linear_derivatives[
                        (variable_that_is_linearly_mapped_to_simulator_name,state_dict_name)].dot(outputs[state_name])
                    # vectors_for_constraints += self.linear_derivatives[
                    #     (constraint_name,state_name)].dot(outputs[state_name])

                    nonlinear_derivatives = np.eye(constraint.shape[0])

                    for state_name, state in self.declared_states.items():
                        if declared_state.operation is not None:
                            state_dict_name = declared_state.operation.name+'.'+state_name
                        else:
                            state_dict_name = state_name
                            
                        total_derivatives[(constraint_name,state_name)] = \
                            nonlinear_derivatives.dot(
                                # self.linear_derivatives[(constraint_name,state_name)])
                                self.linear_derivatives[(variable_that_is_linearly_mapped_to_simulator_name,state_dict_name)])

                    nonlinear_second_derivatives = np.zeros((constraint.shape[0],constraint.shape[0],constraint.shape[0]))
                    constraint_nonlinear_second_derivatives_dict[constraint_name] = {('d'+constraint.name+'_dx','x'): nonlinear_second_derivatives}

        ## Use simulator-computed values to assemble/compute the blocks of the Hessian
        # d^2F_dx^2
        d2objective_dstate2 = {}
        for state_name, declared_state in self.declared_states.items():
            state_vector = outputs[state_name]
            d2objective_dstate2[state_name,state_name] = np.diag(np.ones(declared_state.shape)*2*self.state_penalties[state_name].value)

        # dc_dx
        dc_dx = {}  # dictionary across constraints and states for the off-diagonal entries of the Hessian
        for state_name, declared_state in self.declared_states.items():
            for constraint_name, constraint_value in self.declared_inputs.items():
                constraint_simulator_name = constraint_value.operation.name+'.'+constraint_value.name
                if declared_state.operation is not None:
                    state_simulator_name = declared_state.operation.name+'.'+state_name
                else:
                    state_simulator_name = state_name
                dc_dx[constraint_name, state_name] = total_derivatives[(constraint_name, state_name)]

        # d2c_dx2
        d2constraint_penalty_dx2 = {}  # dictionary across states (contract different constraints with their lagrange multipliers)
        for state_name_i, declared_state_i in self.declared_states.items():
            if declared_state_i.operation is not None:
                state_simulator_name_i = declared_state_i.operation.name+'.'+state_name_i
            else:
                state_simulator_name_i = state_name_i
            for state_name_j, declared_state_j in self.declared_states.items():
                d2constraint_penalty_dx2[state_name_i, state_name_j] = np.zeros((declared_state_i.shape[0],declared_state_j.shape[0]))
                for constraint_name, constraint_value in self.declared_inputs.items():
                    constraint_lagrange_multipliers = outputs[constraint_name+'_lagrange_multipliers']

                    constraint_nonlinear_second_derivatives = constraint_nonlinear_second_derivatives_dict[constraint_name]
                    constraint_simulator_name = constraint_value.operation.name+'.'+constraint_value.name
                    # nonlinear_derivatve_name = 'd'+constraint_simulator_name+'_d'+nonlinear_operation_argument_name # This corresponds to state_i?
                    # nonlinear_derivatve_name = 'd'+constraint_simulator_name+'_dx' # NOTE: Hardcoded for norm operation for now.
                    nonlinear_derivatve_name = 'd'+constraint_value.name+'_dx' # NOTE: Hardcoded for norm operation for now.
                    if declared_state_j.operation is not None:
                        state_simulator_name_j = declared_state_j.operation.name+'.'+state_name_j
                    else:
                        state_simulator_name_j = state_name_j
                    # d2NL_dx2 = constraint_nonlinear_second_derivatives[(nonlinear_derivatve_name, state_simulator_name_j)]
                    d2NL_dx2 = constraint_nonlinear_second_derivatives[(nonlinear_derivatve_name, 'x')] # NOTE: HARDCODED FOR NORM RIGHT NOW
                    # d2NL_dx2 = d2NL_dx2.reshape((constraint_lagrange_multipliers.shape[0], declared_state_i.shape[0], declared_state_j.shape[0]))
                    d2NL_dx2 = d2NL_dx2.reshape((constraint_lagrange_multipliers.shape[0], 3, 3))   # NOTE: HARDCODED FOR NORM (x.shape)

                    if type(constraint_value.operation) is m3l.Norm:
                        variable_that_is_linearly_mapped_to = constraint_value.operation.arguments['x']
                    else:  # Assume it is a linear constraint
                        variable_that_is_linearly_mapped_to = constraint_value

                    # variable_that_is_linearly_mapped_to = constraint_value.operation.arguments['x']
                    variable_that_is_linearly_mapped_to_simulator_name = variable_that_is_linearly_mapped_to.operation.name+'.'\
                        +variable_that_is_linearly_mapped_to.name
                    # linear_map_i = computed_derivatives[(variable_that_is_linearly_mapped_to_simulator_name, state_simulator_name_i)]
                    # linear_map_j = computed_derivatives[(variable_that_is_linearly_mapped_to_simulator_name, state_simulator_name_j)]
                    linear_map_i = self.linear_derivatives[(variable_that_is_linearly_mapped_to_simulator_name, state_simulator_name_i)]
                    linear_map_j = self.linear_derivatives[(variable_that_is_linearly_mapped_to_simulator_name, state_simulator_name_j)]
                    # linear_map_i = self.linear_derivatives[(constraint_name, state_name_i)]
                    # linear_map_j = self.linear_derivatives[(constraint_name, state_name_j)]

                    # linear_map_i = self.linear_maps[constraint_name, state_name_i]
                    # linear_map_j = self.linear_maps[constraint_name, state_name_j]
                    first_term = np.tensordot(d2NL_dx2, linear_map_j, axes=([2, 0]))
                    d2c_dx2 = np.tensordot(linear_map_i, first_term, axes=([0, 1]))

                    d2constraint_penalty_dx2[state_name_i, state_name_j] += np.tensordot(constraint_lagrange_multipliers, d2c_dx2, axes=([0, 1]))

        # -- Assigning Hessian --
        # Assigning upper-left block of Hessian
        for state_name_i, state_i in self.declared_states.items():
            for state_name_j, state_j in self.declared_states.items():
                if state_name_i == state_name_j:
                    derivatives[state_name_i, state_name_j] = d2objective_dstate2[state_name_i, state_name_j] + \
                        d2constraint_penalty_dx2[state_name_i, state_name_j]
                else:
                    derivatives[state_name_i, state_name_j] = d2constraint_penalty_dx2[state_name_i, state_name_j]
        # Assigning the off-block-diagonal entries of the Hessian
        for state_name, state in self.declared_states.items():
            for constraint_name, constraint in self.declared_inputs.items():
                derivatives[state_name, constraint_name+'_lagrange_multipliers'] = dc_dx[constraint_name, state_name].T
                derivatives[constraint_name+'_lagrange_multipliers', state_name] = dc_dx[constraint_name, state_name]
        # Assigning the lower-right block of the Hessian
        for constraint_name_i, constraint_i in self.declared_inputs.items():
            for constraint_name_j, constraint_j in self.declared_inputs.items():
                derivatives[constraint_name_i+'_lagrange_multipliers', constraint_name_j+'_lagrange_multipliers'] = \
                    np.zeros((constraint_i.shape[0],constraint_j.shape[0]))

        # -- Assigning derivatives for adjoint method --
        for state_name, state in self.declared_states.items():
            for input_name, declared_input in self.declared_inputs.items():
                derivatives[state_name, input_name] = np.zeros((state.shape[0],declared_input.shape[0]))
        for constraint_name, constraint in self.declared_inputs.items():
            for input_name, declared_input in self.declared_inputs.items():
                if constraint_name == input_name:
                    derivatives[constraint_name+'_lagrange_multipliers', input_name] = -np.eye(constraint.shape[0])
                else:
                    derivatives[constraint_name+'_lagrange_multipliers', input_name] = np.zeros((constraint.shape[0],declared_input.shape[0]))





        # ## Run simulators to get values and derivatives and second derivatives
        # # Get second derivative of nonlinear operation.
        # for state_name, declared_state in self.declared_states.items():
        #     self.evaluation_simulator[state_name] = outputs[state_name]

        # # Get variable values for computation of second derivatives
        # self.evaluation_simulator.run()

        # # Get dc_dx values
        # input_variable_names = []
        # for input_name, input in self.declared_inputs.items():
        #     input_variable_names.append(input.operation.name+'.'+input.name)
        # state_variable_names = []
        # for state_name, state in self.declared_states.items():
        #     if state.operation is not None:
        #         state_variable_names.append(state.operation.name+'.'+state.name)
        #     else:
        #         state_variable_names.append(state.name)
        # # variables_for_linear_maps = []
        # # for input_name, input in self.declared_inputs.items():
        # #     variable_that_is_linearly_mapped_to = input.operation.arguments['x']
        # #     variables_for_linear_maps.append(variable_that_is_linearly_mapped_to.operation.name+'.'+variable_that_is_linearly_mapped_to.name)
        # # input_variable_names.extend(variables_for_linear_maps)
        # computed_derivatives = self.evaluation_simulator.compute_totals(of=input_variable_names, wrt=state_variable_names)

        # # Get second derivatives of nonlinear operations
        # constraint_nonlinear_second_derivatives_dict = {}
        # for constraint_name, declared_constraint_variable in self.declared_inputs.items():
        #     nonlinear_operation = declared_constraint_variable.operation
        #     nonlinear_operation_derivative_model = nonlinear_operation.compute_derivatives()
        #     nonlinear_derivative_simulator = Simulator(nonlinear_operation_derivative_model)
        #     for nonlinear_operation_argument_name, nonlinear_operation_input in nonlinear_operation.arguments.items():
        #         # Get input value from evaluation simulator
        #         nonlinear_operation_input_name = nonlinear_operation_input.operation.name+'.'+nonlinear_operation_input.name
        #         nonlinear_operation_input_value = self.evaluation_simulator[nonlinear_operation_input_name]
                
        #         # Place input value into derivative simulator
        #         nonlinear_derivative_simulator[nonlinear_operation_argument_name] = nonlinear_operation_input_value
            
        #     nonlinear_derivative_simulator.run()
            
        #     declared_constraint_simulator_name = declared_constraint_variable.operation.name+'.'+declared_constraint_variable.name
        #     derivative_names = []
        #     for nonlinear_operation_argument_name, nonlinear_operation_input in nonlinear_operation.arguments.items():
        #         # derivatve_name = 'd'+declared_constraint_simulator_name+'_d'+nonlinear_operation_argument_name
        #         derivatve_name = 'd'+declared_constraint_variable.name+'_d'+nonlinear_operation_argument_name
        #         derivative_names.append(derivatve_name)

        #     nonlinear_second_derivatives = nonlinear_derivative_simulator.compute_totals(of=derivative_names,
        #                                                                                 wrt=list(nonlinear_operation.arguments.keys()))
        #     constraint_nonlinear_second_derivatives_dict[constraint_name] = nonlinear_second_derivatives
        
        # ## Use simulator-computed values to assemble/compute the blocks of the Hessian
        # # d^2F_dx^2
        # d2objective_dstate2 = {}
        # for state_name, declared_state in self.declared_states.items():
        #     state_vector = outputs[state_name]
        #     d2objective_dstate2[state_name,state_name] = np.diag(np.ones(declared_state.shape)*2*self.state_penalties[state_name].value)

        # # dc_dx
        # dc_dx = {}  # dictionary across constraints and states for the off-diagonal entries of the Hessian
        # for state_name, declared_state in self.declared_states.items():
        #     for constraint_name, constraint_value in self.declared_inputs.items():
        #         constraint_simulator_name = constraint_value.operation.name+'.'+constraint_value.name
        #         if declared_state.operation is not None:
        #             state_simulator_name = declared_state.operation.name+'.'+state_name
        #         else:
        #             state_simulator_name = state_name
        #         dc_dx[constraint_name, state_name] = computed_derivatives[(constraint_simulator_name, state_simulator_name)]

        # # d2c_dx2
        # d2constraint_penalty_dx2 = {}  # dictionary across states (contract different constraints with their lagrange multipliers)
        # for state_name_i, declared_state_i in self.declared_states.items():
        #     if declared_state_i.operation is not None:
        #         state_simulator_name_i = declared_state_i.operation.name+'.'+state_name_i
        #     else:
        #         state_simulator_name_i = state_name_i
        #     for state_name_j, declared_state_j in self.declared_states.items():
        #         d2constraint_penalty_dx2[state_name_i, state_name_j] = np.zeros((declared_state_i.shape[0],declared_state_j.shape[0]))
        #         for constraint_name, constraint_value in self.declared_inputs.items():
        #             constraint_lagrange_multipliers = outputs[constraint_name+'_lagrange_multipliers']

        #             constraint_nonlinear_second_derivatives = constraint_nonlinear_second_derivatives_dict[constraint_name]
        #             constraint_simulator_name = constraint_value.operation.name+'.'+constraint_value.name
        #             # nonlinear_derivatve_name = 'd'+constraint_simulator_name+'_d'+nonlinear_operation_argument_name # This corresponds to state_i?
        #             # nonlinear_derivatve_name = 'd'+constraint_simulator_name+'_dx' # NOTE: Hardcoded for norm operation for now.
        #             nonlinear_derivatve_name = 'd'+constraint_value.name+'_dx' # NOTE: Hardcoded for norm operation for now.
        #             if declared_state_j.operation is not None:
        #                 state_simulator_name_j = declared_state_j.operation.name+'.'+state_name_j
        #             else:
        #                 state_simulator_name_j = state_name_j
        #             # d2NL_dx2 = constraint_nonlinear_second_derivatives[(nonlinear_derivatve_name, state_simulator_name_j)]
        #             d2NL_dx2 = constraint_nonlinear_second_derivatives[(nonlinear_derivatve_name, 'x')] # NOTE: HARDCODED FOR NORM RIGHT NOW
        #             # d2NL_dx2 = d2NL_dx2.reshape((constraint_lagrange_multipliers.shape[0], declared_state_i.shape[0], declared_state_j.shape[0]))
        #             d2NL_dx2 = d2NL_dx2.reshape((constraint_lagrange_multipliers.shape[0], 3, 3))   # NOTE: HARDCODED FOR NORM (x.shape)

        #             variable_that_is_linearly_mapped_to = constraint_value.operation.arguments['x']
        #             variable_that_is_linearly_mapped_to_simulator_name = variable_that_is_linearly_mapped_to.operation.name+'.'\
        #                 +variable_that_is_linearly_mapped_to.name
        #             # linear_map_i = computed_derivatives[(variable_that_is_linearly_mapped_to_simulator_name, state_simulator_name_i)]
        #             # linear_map_j = computed_derivatives[(variable_that_is_linearly_mapped_to_simulator_name, state_simulator_name_j)]
        #             linear_map_i = self.linear_derivatives[(variable_that_is_linearly_mapped_to_simulator_name, state_simulator_name_i)]
        #             linear_map_j = self.linear_derivatives[(variable_that_is_linearly_mapped_to_simulator_name, state_simulator_name_j)]

        #             # linear_map_i = self.linear_maps[constraint_name, state_name_i]
        #             # linear_map_j = self.linear_maps[constraint_name, state_name_j]
        #             first_term = np.tensordot(d2NL_dx2, linear_map_j, axes=([2, 0]))
        #             d2c_dx2 = np.tensordot(linear_map_i, first_term, axes=([0, 1]))

        #             d2constraint_penalty_dx2[state_name_i, state_name_j] += np.tensordot(constraint_lagrange_multipliers, d2c_dx2, axes=([0, 1]))

        # # -- Assigning Hessian --
        # # Assigning upper-left block of Hessian
        # for state_name_i, state_i in self.declared_states.items():
        #     for state_name_j, state_j in self.declared_states.items():
        #         if state_name_i == state_name_j:
        #             derivatives[state_name_i, state_name_j] = d2objective_dstate2[state_name_i, state_name_j] + \
        #                 d2constraint_penalty_dx2[state_name_i, state_name_j]
        #         else:
        #             derivatives[state_name_i, state_name_j] = d2constraint_penalty_dx2[state_name_i, state_name_j]
        # # Assigning the off-block-diagonal entries of the Hessian
        # for state_name, state in self.declared_states.items():
        #     for constraint_name, constraint in self.declared_inputs.items():
        #         derivatives[state_name, constraint_name+'_lagrange_multipliers'] = dc_dx[constraint_name, state_name]
        #         derivatives[constraint_name+'_lagrange_multipliers', state_name] = dc_dx[constraint_name, state_name].T
        # # Assigning the lower-right block of the Hessian
        # for constraint_name_i, constraint_i in self.declared_inputs.items():
        #     for constraint_name_j, constraint_j in self.declared_inputs.items():
        #         derivatives[constraint_name_i+'_lagrange_multipliers', constraint_name_j+'_lagrange_multipliers'] = \
        #             np.zeros((constraint_i.shape[0],constraint_j.shape[0]))

        # # -- Assigning derivatives for adjoint method --
        # for state_name, state in self.declared_states.items():
        #     for input_name, declared_input in self.declared_inputs.items():
        #         derivatives[state_name, input_name] = np.zeros((state.shape[0],declared_input.shape[0]))
        # for constraint_name, constraint in self.declared_inputs.items():
        #     for input_name, declared_input in self.declared_inputs.items():
        #         if constraint_name == input_name:
        #             derivatives[constraint_name+'_lagrange_multipliers', input_name] = -np.eye(constraint.shape[0])
        #         else:
        #             derivatives[constraint_name+'_lagrange_multipliers', input_name] = np.zeros((constraint.shape[0],declared_input.shape[0]))


        ## NOTE: OLD ATTEMPT

        # for state_name in self.declared_states:
        #     self.derivative_evaluation_simulator[state_name] = outputs[state_name]

        # self.derivative_evaluation_simulator.run()
        # input_variable_names = []
        # for input_name, input in self.declared_inputs.items():
        #     input_variable_names.append(input.operation.name+'.'+input.name)
        # state_variable_names = []
        # for state_name, state in self.declared_states.items():
        #     if state.operation is not None:
        #         state_variable_names.append(state.operation.name+'.'+state.name)
        #     else:
        #         state_variable_names.append(state.name)
        # second_derivatives = self.derivative_evaluation_simulator.compute_totals(of=input_variable_names, 
        #                                                                          wrt=state_variable_names)

        # d2objective_dstate2 = {}
        # for state_name, declared_state in self.declared_states.items():
        #     state_vector = outputs[state_name]
        #     # state_penalties[state_name] = self.state_penalties[state_name].value * state_vector.T.dot(state_vector)   # x^T*alpha*x
        #     d2objective_dstate2[state_name,state_name] = np.diag(np.ones(declared_state.shape)*2*self.state_penalties[state_name].value)

        # dc_dx = {}  # dictionary across constraints and states for the off-diagonal entries of the Hessian
        # for state_name, declared_state in self.declared_states.items():
        #     for constraint_name, constraint_value in self.declared_inputs.items():
        #         csdl_variable_name = constraint_value.operation.name+'.'+constraint_value.name
        #         dc_dx[constraint_name, state_name] = self.derivative_evaluation_simulator[csdl_variable_name]

        # d2constraint_penalty_dx2 = {}  # dictionary across states (contract different constraints with their lagrange multipliers)
        # for state_name_i, declared_state_i in self.declared_states.items():
        #     for state_name_j, declared_state_j in self.declared_states.items():
        #         d2constraint_penalty_dx2[state_name_i, state_name_j] = np.zeros((declared_state_i.shape[0],declared_state_j.shape[0]))
        #         for constraint_name, constraint_value in self.declared_inputs.items():
        #             constraint_lagrange_multipliers = outputs[constraint_name+'_lagrange_multipliers']
        #             input_dict_name = constraint_value.operation.name+'.'+constraint_value.name

        #             # TODO: FIGURE OUT HOW TO GET SECOND DERIVATIVE FROM FIRST DERIVATIVE CSDL MODEL
        #             if declared_state.operation is not None:
        #                 state_dict_name = declared_state.operation.name+'.'+state_name
        #             else:
        #                 state_dict_name = state_name
        #             d2c_dx2 = second_derivatives[(input_dict_name, state_dict_name)]


        #             d2constraint_penalty_dx2[state_name_i, state_name_j] += constraint_lagrange_multipliers.dot(d2c_dx2)

        # # -- Assigning Hessian --
        # # Assigning upper-left block of Hessian
        # for state_name_i, state_i in self.declared_states.items():
        #     for state_name_j, state_j in self.declared_states.items():
        #         if state_name_i == state_name_j:
        #             derivatives[state_name_i, state_name_j] = d2objective_dstate2[state_name_i, state_name_j] + \
        #                 d2constraint_penalty_dx2[state_name_i, state_name_j]
        #         else:
        #             derivatives[state_name_i, state_name_j] = d2c_dx2[state_name_i, state_name_j]
        # # Assigning the off-block-diagonal entries of the Hessian
        # for state_name, state in self.declared_states.items():
        #     for constraint_name, constraint in self.declared_inputs.items():
        #         derivatives[state_name, constraint_name+'_lagrange_multipliers'] = dc_dx[constraint_name, state_name]
        #         derivatives[constraint_name+'_lagrange_multipliers', state_name] = dc_dx[constraint_name, state_name].T
        # # Assigning the lower-right block of the Hessian
        # for constraint_name_i, constraint_i in self.declared_inputs.items():
        #     for constraint_name_j, constraint_j in self.declared_inputs.items():
        #         derivatives[constraint_name_i+'_lagrange_multipliers', constraint_name_j+'_lagrange_multipliers'] = 0.

        # # -- Assigning derivatives for adjoint method --
        # for state_name, state in self.declared_states.items():
        #     for input_name, declared_input in self.declared_inputs.items():
        #         derivatives[state_name, input_name] = 0.
        # for constraint_name, constraint in self.declared_inputs.items():
        #     for input_name, declared_input in self.declared_inputs.items():
        #         if constraint_name == input_name:
        #             derivatives[constraint_name+'_lagrange_multipliers', input_name] = -np.eye(constraint.shape[0])
        #         else:
        #             derivatives[constraint_name+'_lagrange_multipliers', input_name] = 0.


    '''
    Might be able to take advantage of convexity here.
    '''
    # def solve_residual_equations(self, inputs, outputs):
    #     pass


# if __name__ == "__main__":
#     import csdl
#     from python_csdl_backend import Simulator
#     # from csdl_om import Simulator
#     import numpy as np
#     from src.caddee.concept.geometry.geometry import Geometry
#     from src.caddee.concept.geometry.geocore.component import Component
#     from src.caddee.concept.geometry.geocore.geometric_calculations import GeometricOuputs, MagnitudeCalculation
#     from src.caddee.concept.geometry.geocore.ffd import FFDParameter, FFDTranslationXParameter, FFDTranslationYParameter, FFDTranslationZParameter, FFDScaleYParameter, FFDScaleZParameter

#     from src import STP_FILES_FOLDER
#     from vedo import Points, Plotter

#     # # Single FFD block
#     # stp_path = STP_FILES_FOLDER / 'rect_wing.stp'
#     # geo = Geometry()
#     # geo.read_file(file_name=stp_path)
#     # wing_comp = Component(stp_entity_names=['RectWing'], name='wing')       # Creating a wing component and naming it wing
#     # geo.add_component(wing_comp)

#     # top_wing_surface_names = [
#     #     'RectWing, 0, 3',
#     #     'RectWing, 1, 9',
#     #     ]

#     # bot_wing_surface_names = [
#     #     'RectWing, 0, 2',
#     #     'RectWing, 1, 8',
#     #     ]

#     # up_direction = np.array([0., 0., 1.])
#     # down_direction = np.array([0., 0., -1.])

#     # left_lead_point = np.array([0., -9000., 2000.])/1000
#     # left_trail_point = np.array([4000.0, -9000.0, 2000.])/1000
#     # right_lead_point = np.array([0.0, 9000.0, 2000.])/1000
#     # right_trail_point = np.array([4000.0, 9000.0, 2000.])/1000

#     # '''Project points'''
#     # wing_lead_left, wing_lead_left_coord = geo.project_points(left_lead_point, projection_direction = down_direction, projection_targets_names=["wing"])
#     # wing_trail_left, wing_trail_left_coord = geo.project_points(left_trail_point, projection_direction = down_direction, projection_targets_names=["wing"])
#     # wing_lead_right, wing_lead_right_coord = geo.project_points(right_lead_point, projection_direction = down_direction, projection_targets_names=["wing"])
#     # wing_trail_right, wing_trail_right_coord = geo.project_points(right_trail_point, projection_direction = down_direction, projection_targets_names=["wing"])
#     # wing_lead_mid, _ = geo.project_points(np.array([0., 0., 2.]), projection_direction = down_direction, projection_targets_names=["wing"])
#     # wing_trail_mid, _ = geo.project_points(np.array([4., 0., 2.]), projection_direction = down_direction, projection_targets_names=["wing"])

#     # chord = geo.subtract_pointsets(wing_lead_mid, wing_trail_mid)
#     # span = geo.subtract_pointsets(wing_lead_right, wing_lead_left)

#     # # Adding dof to the wing FFD block
#     # # wing_comp.add_ffd_parameter(parameter_type='scale_y', degree=1, num_dof=2, cost_factor=1.)  # without object
#     # wing_comp.add_ffd_parameter(FFDScaleYParameter(degree=1, num_dof=3, cost_factor=1.))    # with object
#     # wing_comp.add_ffd_parameter(FFDScaleYParameter(degree=2, num_dof=4, cost_factor=100.))
#     # wing_comp.add_ffd_parameter(FFDTranslationXParameter(degree=1, num_dof=2, cost_factor=1.))
#     # wing_comp.add_ffd_parameter(FFDRotationXParameter(degree=1, num_dof=3, value=np.array([0., -0.2, 0.])))
#     # wing_comp.add_ffd_parameter(FFDScaleZParameter(degree=1, num_dof=3, value=np.array([0., 10., 0.])))
#     # # wing_comp.add_ffd_parameter()
#     # # from src.utils.constants import SCALE_Y
#     # # wing_comp.add_ffd_parameter(parameter_type=nasa_uli_tc1.SCALE_Y, degree=1, num_dof=2, cost_factor=1.)

#     # # Adding inputs to the geometry model
#     # # geo.add_input(MagnitudeCalculation(pointset=chord))
#     # # geo.add_input(MagnitudeCalculation(pointset=span), connection_name='span')
#     # geo.add_input(MagnitudeCalculation(pointset=chord), connection_name='chord')
#     # # geo.add_constraint(MagnitudeCalculation(pointset=chord), value=15.)
#     # # geo.add_input(calculation_type=nasa_uli_tc1.MAGNITUDE, pointset=chord, connection_name='chord')

#     # inner_opt_implicit_model = InnerOptimizationModel(geometry=geo)

#     # test_model = csdl.Model()
#     # test_model.create_input('chord', val=10.)
#     # # test_model.create_input('span', val=30.)

#     # test_model.add(submodel=inner_opt_implicit_model, name='InnerOptimizationModel', promotes=[])

#     # for geometric_input in geo.inputs:
#     #     test_model.connect(f'{geometric_input.connection_name}', f'InnerOptimizationModel.{geometric_input.connection_name}')

#     # # sim = Simulator(inner_opt_implicit_model)
#     # sim = Simulator(test_model)
#     # sim.run()
#     # # sim.visualize_implementation()
#     # # sim.prob.check_totals(of='InnerOptimizationModel.ffd_free_dof', wrt='chord')
#     # # print('ffd_free_dof', sim['ffd_free_dof'].reshape((NUM_PROPERTIES,NUM_DV_PER_PARAMETER)))


#     # Multiple FFD blocks
#     stp_path = STP_FILES_FOLDER / 'dw_with_nacelles.stp'
#     geo = Geometry()
#     geo.read_file(file_name=stp_path)

#     wing_comp = Component(stp_entity_names=['Wing'], name='wing', nxp=2, nyp=3, nzp=2)  # Creating a wing component and naming it wing
#     tail_comp = Component(stp_entity_names=['Tail'], name='tail', nxp=2, nyp=3, nzp=2)
#     front_left_nacelle_comp = Component(stp_entity_names=['LiftNacelleFrontLeft'], name='lift_nacelle_front_left')
#     front_right_nacelle_comp = Component(stp_entity_names=['LiftNacelleFrontRight'], name='lift_nacelle_front_right')
#     geo.add_component(wing_comp)
#     geo.add_component(tail_comp)
#     geo.add_component(front_left_nacelle_comp)
#     geo.add_component(front_right_nacelle_comp)

#     wing_comp.add_ffd_parameter(FFDScaleYParameter(degree=1, num_dof=3, cost_factor=1.))
#     wing_comp.add_ffd_parameter(FFDScaleZParameter(degree=1, num_dof=3, cost_factor=1.))
#     wing_comp.add_ffd_parameter(FFDRotationXParameter(degree=1, num_dof=3, value=np.array([0., -1., 0.])))
#     wing_comp.add_ffd_parameter(FFDTranslationXParameter(degree=1, num_dof=2, cost_factor=1.))
#     tail_comp.add_ffd_parameter(FFDScaleZParameter(degree=1, num_dof=3, value=np.array([0., 1., 0.])))
#     # tail_comp.add_ffd_parameter(FFDScaleYParameter(degree=1, num_dof=3))
#     tail_comp.add_ffd_parameter(FFDTranslationXParameter(degree=0, num_dof=1))
#     tail_comp.add_ffd_parameter(FFDTranslationYParameter(degree=0, num_dof=1))
#     tail_comp.add_ffd_parameter(FFDTranslationZParameter(degree=0, num_dof=1))
#     # front_left_nacelle_comp.add_ffd_parameter(FFDScaleYParameter(degree=2, num_dof=5, cost_factor=1.))
#     # front_right_nacelle_comp.add_ffd_parameter(FFDTranslationXParameter(degree=1, num_dof=2, connection_name='hanging_input'))

#     up_direction = np.array([0., 0., 1.])
#     down_direction = np.array([0., 0., -1.])

#     left_lead_point = np.array([0., -9000., 2000.])/1000
#     left_trail_point = np.array([2000.0, -9000.0, 2000.])/1000
#     right_lead_point = np.array([0.0, 9000.0, 2000.])/1000
#     right_trail_point = np.array([2000.0, 9000.0, 2000.])/1000
#     mid_lead_point = np.array([0., 0., 2.])
#     mid_trail_point = np.array([2., 0., 2.])

#     tail_lead_mid_point = np.array([2., 0., 2.])

#     '''Project points'''
#     wing_lead_left, wing_lead_left_coord = geo.project_points(left_lead_point, projection_direction = down_direction, projection_targets_names=["wing"])
#     wing_trail_left, wing_trail_left_coord = geo.project_points(left_trail_point, projection_direction = down_direction, projection_targets_names=["wing"])
#     wing_lead_right, wing_lead_right_coord = geo.project_points(right_lead_point, projection_direction = down_direction, projection_targets_names=["wing"])
#     wing_trail_right, wing_trail_right_coord = geo.project_points(right_trail_point, projection_direction = down_direction, projection_targets_names=["wing"])
#     wing_lead_mid, _ = geo.project_points(mid_lead_point, projection_direction = down_direction, projection_targets_names=["wing"])
#     wing_trail_mid, _ = geo.project_points(mid_trail_point, projection_direction = down_direction, projection_targets_names=["wing"])

#     tail_lead_mid, _ = geo.project_points(tail_lead_mid_point, projection_targets_names=['tail'])

#     root_chord = geo.subtract_pointsets(wing_lead_mid, wing_trail_mid)
#     left_tip_chord = geo.subtract_pointsets(wing_lead_left, wing_trail_left)
#     right_tip_chord = geo.subtract_pointsets(wing_lead_right, wing_trail_right)
#     span = geo.subtract_pointsets(wing_trail_right, wing_trail_left)
#     wing_to_tail_displacement = geo.subtract_pointsets(tail_lead_mid, wing_trail_mid)

#     # Adding inputs to the geometry model
#     geo.add_input(MagnitudeCalculation(pointset=root_chord), connection_name='chord')
#     # geo.add_constraint(MagnitudeCalculation(pointset=left_tip_chord))
#     # geo.add_constraint(MagnitudeCalculation(pointset=right_tip_chord))
#     geo.add_input(MagnitudeCalculation(pointset=span), connection_name='span')
#     # geo.add_input(DisplacementCalculation(wing_to_tail_displacement), connection_name='wing_to_tail_displacement')
#     geo.add_constraint(DisplacementCalculation(wing_to_tail_displacement))

#     inner_opt_implicit_model = InnerOptimizationModel(geometry=geo)

#     test_model = csdl.Model()
#     test_model.create_input('chord', val=6.)
#     test_model.create_input('span', val=20.)
#     # test_model.create_input('wing_to_tail_displacement', val=np.array([10., 0., 0.]))

#     test_model.add(submodel=inner_opt_implicit_model, name='InnerOptimizationModel', promotes=[])

#     for geometric_input in geo.inputs:
#         test_model.connect(f'{geometric_input.connection_name}', f'InnerOptimizationModel.{geometric_input.connection_name}')


#     sim = Simulator(test_model)
#     sim.run()
#     sim.check_totals(of='InnerOptimizationModel.ffd_free_dof', wrt='chord')
#     sim.check_totals(of='InnerOptimizationModel.ffd_free_dof', wrt='span')
#     # sim.check_totals(of='InnerOptimizationModel.ffd_free_dof', wrt='wing_to_tail_displacement')



#     geometry = geo
#     ffd_set = geometry.ffd_set

#     ffd_free_dof = sim['InnerOptimizationModel.ffd_free_dof']     # output
#     parameterization_lagrange_multipliers = sim['InnerOptimizationModel.parameterization_lagrange_multipliers']   # output

#     print('ffd_free_dof', ffd_free_dof)
#     print('norm(ffd_free_dof)', np.linalg.norm(sim['InnerOptimizationModel.ffd_free_dof']))
#     print('lagrange multipliers', parameterization_lagrange_multipliers)

#     ffd_set = geometry.ffd_set
#     ffd_blocks = ffd_set.ffd_blocks

#     cost_matrix = ffd_set.cost_matrix
#     free_section_properties_map = ffd_set.free_section_properties_map
#     prescribed_section_properties_map = ffd_set.prescribed_section_properties_map
#     ffd_control_points_map = ffd_set.ffd_control_points_map
#     ffd_control_points_x_map = ffd_set.ffd_control_points_x_map
#     ffd_control_points_y_map = ffd_set.ffd_control_points_y_map
#     ffd_control_points_z_map = ffd_set.ffd_control_points_z_map
#     sectional_rotations_map = ffd_set.sectional_rotations_map
#     # rotated_ffd_control_points_map = ffd_set.rotated_ffd_control_points_map   # not real map since nonlinear. Must be constructed from prescribed rotational dof
#     geometry_control_points_map = ffd_set.geometry_control_points_map
#     unchanged_geometry_indexing_map = ffd_set.unchanged_geometry_indexing_map
#     pointset_map = geometry.eval_map

#     num_affine_free_dof = ffd_set.num_affine_free_dof
#     num_affine_section_properties = ffd_set.num_affine_section_properties
#     num_affine_ffd_control_points = ffd_set.num_affine_ffd_control_points
#     num_affine_free_ffd_control_points = ffd_set.num_affine_free_ffd_control_points
#     num_embedded_points = ffd_set.num_embedded_points
#     num_geometry_control_points = ffd_set.num_geometry_control_points

#     # ffd section properties evaluation
#     ffd_free_section_properties_without_initial = free_section_properties_map.dot(ffd_free_dof)

#     # An initial value of 1 must be added to the scaling section properties. This is not differentiated because it's a constant 1.
#     ffd_section_properties = np.zeros((num_affine_section_properties,)) # Section properties model output!
#     ffd_block_starting_index = 0
#     for ffd_block in ffd_blocks:
#         if ffd_block.num_affine_dof == 0:
#             continue

#         ffd_block_ending_index = ffd_block_starting_index + ffd_block.num_sections * ffd_block.num_affine_properties
#         NUM_SCALING_PROPERTIES = 3
#         ffd_block_scaling_properties_starting_index = ffd_block_starting_index + ffd_block.num_sections*(ffd_block.num_affine_properties-NUM_SCALING_PROPERTIES)    #The last 2 properties are scaling
#         ffd_block_scaling_properties_ending_index = ffd_block_scaling_properties_starting_index + ffd_block.num_sections*(NUM_SCALING_PROPERTIES)

#         # Use calculated values for non-scaling parameters
#         ffd_section_properties[ffd_block_starting_index:ffd_block_scaling_properties_starting_index] = \
#             ffd_free_section_properties_without_initial[ffd_block_starting_index:ffd_block_scaling_properties_starting_index] # -3 is because scale_y and z are the last 2 properties

#         # Add 1 to scaling parameters to make initial scaling=1.
#         ffd_section_properties[ffd_block_scaling_properties_starting_index:ffd_block_scaling_properties_ending_index] = \
#             ffd_free_section_properties_without_initial[ffd_block_scaling_properties_starting_index:ffd_block_scaling_properties_ending_index] + 1.  # adding 1 which is initial scale value

#         ffd_block_starting_index = ffd_block_ending_index


#     # ffd control points evaluation (affine)
#     # NOTE: x, y, and z are split up to take advantage of scipy sparse. numpy tensor will likely be used for derivative for simplicity
#     ffd_block_starting_index = 0
#     affine_free_ffd_control_points_x_map = None
#     for ffd_block in ffd_blocks:
#         ffd_block_ending_index = ffd_block_starting_index + ffd_block.num_ffd_control_points

#         if ffd_block.num_affine_free_dof != 0:
#             if affine_free_ffd_control_points_x_map is None:
#                 affine_free_ffd_control_points_x_map = ffd_control_points_x_map[ffd_block_starting_index:ffd_block_ending_index, :]
#                 affine_free_ffd_control_points_y_map = ffd_control_points_y_map[ffd_block_starting_index:ffd_block_ending_index, :]
#                 affine_free_ffd_control_points_z_map = ffd_control_points_z_map[ffd_block_starting_index:ffd_block_ending_index, :]
#             else:
#                 affine_free_ffd_control_points_x_map = sps.vstack((affine_free_ffd_control_points_x_map, ffd_control_points_x_map[ffd_block_starting_index:ffd_block_ending_index, :]))
#                 affine_free_ffd_control_points_y_map = sps.vstack((affine_free_ffd_control_points_y_map, ffd_control_points_y_map[ffd_block_starting_index:ffd_block_ending_index, :]))
#                 affine_free_ffd_control_points_z_map = sps.vstack((affine_free_ffd_control_points_z_map, ffd_control_points_z_map[ffd_block_starting_index:ffd_block_ending_index, :]))

#         ffd_block_starting_index = ffd_block_ending_index

#     ffd_control_points_x = affine_free_ffd_control_points_x_map.dot(ffd_section_properties)
#     ffd_control_points_y = affine_free_ffd_control_points_y_map.dot(ffd_section_properties)
#     ffd_control_points_z = affine_free_ffd_control_points_z_map.dot(ffd_section_properties)

#     affine_free_ffd_control_points_map = np.zeros((num_affine_free_ffd_control_points, 3, num_affine_section_properties))
#     affine_free_ffd_control_points_map[:,0,:] = affine_free_ffd_control_points_x_map.toarray()
#     affine_free_ffd_control_points_map[:,1,:] = affine_free_ffd_control_points_y_map.toarray()
#     affine_free_ffd_control_points_map[:,2,:] = affine_free_ffd_control_points_z_map.toarray()

#     # Combine x,y,z components back to list of points
#     affine_ffd_control_points = np.zeros((num_affine_free_ffd_control_points, 3))
#     affine_ffd_control_points[:,0] = ffd_control_points_x
#     affine_ffd_control_points[:,1] = ffd_control_points_y
#     affine_ffd_control_points[:,2] = ffd_control_points_z


#     # Construct ffd control points vector from ffd blocks with affine and/or rotational dof
#     ffd_control_points_local_frame = np.zeros((num_affine_free_ffd_control_points,3))
#     ffd_control_points_starting_index = 0
#     affine_ffd_control_points_starting_index = 0
#     for ffd_block in ffd_blocks:
#         if ffd_block.num_affine_free_dof == 0:
#             continue

#         ffd_block_num_control_points = ffd_block.nxp * ffd_block.nyp * ffd_block.nzp
#         ffd_control_points_ending_index = ffd_control_points_starting_index + ffd_block_num_control_points

#         affine_ffd_control_points_ending_index = affine_ffd_control_points_starting_index + ffd_block_num_control_points
#         ffd_control_points_local_frame[ffd_control_points_starting_index:ffd_control_points_ending_index] = affine_ffd_control_points[affine_ffd_control_points_starting_index:affine_ffd_control_points_ending_index]
#         affine_ffd_control_points_starting_index = affine_ffd_control_points_ending_index

#         ffd_control_points_starting_index = ffd_control_points_ending_index


#     # transformation back into global frame construction
#     global_frame_rotation_map = np.zeros((num_affine_free_ffd_control_points, 3, num_affine_free_ffd_control_points, 3))
#     ffd_block_starting_index = 0
#     for ffd_block in ffd_blocks:
#         if ffd_block.num_affine_free_dof == 0:
#             continue

#         ffd_block_rotation_matrix_tensor = np.zeros((ffd_block.num_ffd_control_points, 3, ffd_block.num_ffd_control_points, 3))
#         for i in range(ffd_block_rotation_matrix_tensor.shape[0]):
#             ffd_block_rotation_matrix_tensor[i,:,i,:] = ffd_block.rotation_matrix

#         ffd_block_ending_index = ffd_block_starting_index + ffd_block.num_ffd_control_points
#         global_frame_rotation_map[ffd_block_starting_index:ffd_block_ending_index,:,ffd_block_starting_index:ffd_block_ending_index,:] = ffd_block_rotation_matrix_tensor
#         ffd_block_starting_index = ffd_block_ending_index

#     # Transformation back into global frame evaluation
#     ffd_control_points_without_origin = np.tensordot(global_frame_rotation_map, ffd_control_points_local_frame)
#     ffd_control_points = ffd_control_points_without_origin
#     ffd_block_starting_index = 0
#     for ffd_block in ffd_blocks:
#         if ffd_block.num_affine_free_dof == 0:
#             continue
#         ffd_block_num_control_points = ffd_block.nxp * ffd_block.nyp * ffd_block.nzp
#         ffd_block_ending_index = ffd_block_starting_index + ffd_block_num_control_points
#         ffd_control_points[ffd_block_starting_index:ffd_block_ending_index] += np.repeat(ffd_block.section_origins, ffd_block.nyp*ffd_block.nzp, axis=0)
#         ffd_block_starting_index = ffd_block_ending_index

#     # vp_init = Plotter()
#     # vps = []
#     # vps1 = Points(ffd_control_points, r=8, c = 'blue')
#     # vps2 = Points(ffd_control_points_without_origin, r=7, c='cyan')
#     # vps3 = Points(ffd_control_points_local_frame, r=7, c='red')
#     # vps4 = Points(affine_ffd_control_points, r=7, c='magenta')
#     # vps.append(vps1)
#     # vps.append(vps2)
#     # vps.append(vps3)
#     # vps.append(vps4)

#     # vp_init.show(vps, 'FFD Changes', axes=1, viewup="z", interactive = True)

#     # Geometry control points evaluation
#     if num_embedded_points != num_geometry_control_points:  #
#         # Get unchanged geometry control points (points not included in FFD)
#         initial_geometry_control_points = geometry.total_cntrl_pts_vector
#         unchanged_geometry_control_points = unchanged_geometry_indexing_map.dot(initial_geometry_control_points)

#     # --Construct map (ffd control points (in ffd blocks with affine free dof) --> geometry control points)
#     affine_geometry_control_points_map = None
#     ffd_block_starting_index = 0
#     for ffd_block in ffd_blocks:
#         if ffd_block.num_affine_free_dof == 0:
#             continue

#         ffd_block_num_control_points = ffd_block.nxp * ffd_block.nyp * ffd_block.nzp
#         ffd_block_ending_index = ffd_block_starting_index + ffd_block_num_control_points

#         if affine_geometry_control_points_map is None:
#             affine_geometry_control_points_map = geometry_control_points_map[:,ffd_block_starting_index:ffd_block_ending_index]
#         else:
#             affine_geometry_control_points_map = sps.hstack((affine_geometry_control_points_map, geometry_control_points_map[:,ffd_block_starting_index:ffd_block_ending_index]))

#         ffd_block_starting_index = ffd_block_ending_index

#     # --Evaluate updated geometry control points using map
#     updated_geometry_control_points = affine_geometry_control_points_map.dot(ffd_control_points)

#     # --Combine updated and unchanged portions of the geometry to complete the geometry
#     if num_embedded_points != num_geometry_control_points:
#         geometry_control_points = updated_geometry_control_points + unchanged_geometry_control_points
#     else:
#         geometry_control_points = updated_geometry_control_points

#     ''' Plotting results of Application model '''
#     pts_shape = wing_comp.ffd_control_points.shape
#     nxp = pts_shape[0]
#     nyp = pts_shape[1]
#     nzp = pts_shape[2]

#     ffd_pts_reshape_updated = np.reshape(wing_comp.ffd_control_points, (nxp * nyp * nzp, 3))
#     wing_ffd = list(geo.components_ffd_dict.values())[0]
#     cp = wing_ffd.evaluate(ffd_pts_reshape_updated)

#     vp_init = Plotter()
#     vps = []
#     vps1 = Points(ffd_control_points, r=8, c = 'blue')
#     # vps2 = Points(np.reshape(initial_ffd_block_control_points, (nxp * nyp * nzp, 3)), r=9, c = 'red')
#     vps3 = Points(np.reshape(ffd_blocks[0].control_points, (nxp * nyp * nzp, 3)), r=9, c = 'red')
#     vps4 = Points(geo.total_cntrl_pts_vector, r=6, c='black')
#     vps5 = Points(geometry_control_points, r=7, c='cyan')
#     vps.append(vps1)
#     vps.append(vps3)
#     vps.append(vps4)
#     vps.append(vps5)

#     vp_init.show(vps, 'FFD Changes', axes=1, viewup="z", interactive = True)
