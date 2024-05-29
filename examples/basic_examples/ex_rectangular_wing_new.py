import csdl_alpha as csdl
import numpy as np
import lsdo_function_spaces as lfs

from lsdo_geo.core.parameterization.free_form_deformation_functions import (
    construct_tight_fit_ffd_block,
)
from lsdo_geo.core.parameterization.volume_sectional_parameterization import (
    VolumeSectionalParameterization,
    VolumeSectionalParameterizationInputs
)


import lsdo_geo

'''
TODO:
1. Set up inner optimization class
2. Ruff Ruff

'''


recorder = csdl.Recorder(inline=True)
recorder.start()

geometry = lsdo_geo.import_geometry(
    "lsdo_geo/splines/b_splines/sample_geometries/rectangular_wing.stp",
    parallelize=False,
)
# geometry.refit(parallelize=False) # New API if you want to do this!
# geometry.plot()


# region Key locations
leading_edge_left = geometry.project(np.array([0.0, -4.0, 0.0]))
leading_edge_right = geometry.project(np.array([0.0, 4.0, 0.0]))
trailing_edge_left = geometry.project(np.array([1.0, -4.0, 0.0]))
trailing_edge_right = geometry.project(np.array([1.0, 4.0, 0.0]))
leading_edge_center = geometry.project(np.array([0.0, 0.0, 0.0]))
trailing_edge_center = geometry.project(np.array([1.0, 0.0, 0.0]))
quarter_chord_left = geometry.project(np.array([0.25, -4.0, 0.0]))
quarter_chord_right = geometry.project(np.array([0.25, 4.0, 0.0]))
quarter_chord_center = geometry.project(np.array([0.25, 0.0, 0.0]))
# endregion

# region Mesh definitions
# region Wing Camber Surface
num_spanwise = 11
num_chordwise = 4
points_to_project_on_leading_edge = np.linspace(np.array([0., -4., 1.]), np.array([0., 4., 1.]), num_spanwise)
points_to_project_on_trailing_edge = np.linspace(np.array([1., -4., 1.]), np.array([1., 4., 1.]), num_spanwise)

leading_edge_parametric = geometry.project(points_to_project_on_leading_edge, direction=np.array([0., 0., -1.]), plot=False)
leading_edge_physical = geometry.evaluate(leading_edge_parametric, plot=False)
trailing_edge_parametric = geometry.project(points_to_project_on_trailing_edge, direction=np.array([0., 0., -1.]), plot=False)
trailing_edge_physical = geometry.evaluate(trailing_edge_parametric)

chord_surface = csdl.linear_combination(leading_edge_physical, trailing_edge_physical, num_chordwise).value.reshape((num_chordwise, num_spanwise, 3))
upper_surface_wireframe_parametric = geometry.project(chord_surface + np.array([0., 0., 1]), direction=np.array([0., 0., -1.]), plot=False)
lower_surface_wireframe_parametric = geometry.project(chord_surface - np.array([0., 0., 1]), direction=np.array([0., 0., -1.]), plot=False)
upper_surface_wireframe = geometry.evaluate(upper_surface_wireframe_parametric, plot=False)
lower_surface_wireframe = geometry.evaluate(lower_surface_wireframe_parametric)
camber_surface = csdl.linear_combination(upper_surface_wireframe, lower_surface_wireframe, 1).reshape((num_chordwise, num_spanwise, 3))
# geometry.plot_meshes([camber_surface])
# endregion

# endregion

# region Parameterization

# region Create Parameterization Objects

num_ffd_sections = 3
num_wing_secctions = 2
ffd_block = construct_tight_fit_ffd_block(entities=geometry, num_coefficients=(2, (num_ffd_sections // num_wing_secctions + 1), 2), degree=(1,1,1))
# ffd_block = construct_tight_fit_ffd_block(entities=geometry, num_coefficients=(2, 3, 2), degree=(1,1,1))
# ffd_block.plot()

ffd_sectional_parameterization = VolumeSectionalParameterization(
    name="ffd_sectional_parameterization",
    parameterized_points=ffd_block.coefficients,
    principal_parametric_dimension=1,
)
# ffd_sectional_parameterization.plot()

space_of_linear_3_dof_b_splines = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(3,))
space_of_linear_2_dof_b_splines = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))

chord_stretching_b_spline = lfs.Function(space=space_of_linear_3_dof_b_splines,
                                         coefficients=csdl.ImplicitVariable(shape=(3,), value=np.array([-0.8, 3., -0.8])), name='chord_stretching_b_spline_coefficients')

wingspan_stretching_b_spline = lfs.Function(space=space_of_linear_2_dof_b_splines,
                                             coefficients=csdl.ImplicitVariable(shape=(2,), value=np.array([-4., 4.])), name='wingspan_stretching_b_spline_coefficients')

sweep_translation_b_spline = lfs.Function(space=space_of_linear_3_dof_b_splines,
                                            coefficients=csdl.ImplicitVariable(shape=(3,), value=np.array([4.0, 0.0, 4.0])), name='sweep_translation_b_spline_coefficients')
# sweep_translation_b_spline.plot()

twist_b_spline = lfs.Function(space=space_of_linear_3_dof_b_splines,
                                coefficients=csdl.Variable(shape=(3,), value=np.array([15, 0., 15])))

# endregion

# region Evaluate Parameterization To Define Parameterization Forward Model For Parameterization Solver
parametric_b_spline_inputs = np.linspace(0.0, 1.0, num_ffd_sections).reshape((-1, 1))
chord_stretch_sectional_parameters = chord_stretching_b_spline.evaluate(
    parametric_b_spline_inputs
)
wingspan_stretch_sectional_parameters = wingspan_stretching_b_spline.evaluate(
    parametric_b_spline_inputs
)
sweep_translation_sectional_parameters = sweep_translation_b_spline.evaluate(
    parametric_b_spline_inputs
)

twist_sectional_parameters = twist_b_spline.evaluate(
    parametric_b_spline_inputs
)


sectional_parameters = VolumeSectionalParameterizationInputs()
sectional_parameters.add_sectional_stretch(axis=0, stretch=chord_stretch_sectional_parameters)
sectional_parameters.add_sectional_translation(axis=1, translation=wingspan_stretch_sectional_parameters)
sectional_parameters.add_sectional_translation(axis=0, translation=sweep_translation_sectional_parameters)

ffd_coefficients = ffd_sectional_parameterization.evaluate(sectional_parameters, plot=False)    # TODO: Fix plot function

geometry_coefficients = ffd_block.evaluate_ffd(ffd_coefficients, plot=False)
geometry.set_coefficients(geometry_coefficients)
# geometry.plot()


wingspan = csdl.norm(
    geometry.evaluate(leading_edge_right) - geometry.evaluate(leading_edge_left)
)
root_chord = csdl.norm(
    geometry.evaluate(trailing_edge_center) - geometry.evaluate(leading_edge_center)
)
tip_chord_left = csdl.norm(
    geometry.evaluate(trailing_edge_left) - geometry.evaluate(leading_edge_left)
)
tip_chord_right = csdl.norm(
    geometry.evaluate(trailing_edge_right) - geometry.evaluate(leading_edge_right)
)

spanwise_direction_left = geometry.evaluate(quarter_chord_left) - geometry.evaluate(quarter_chord_center)
spanwise_direction_right = geometry.evaluate(quarter_chord_right) - geometry.evaluate(quarter_chord_center)
# sweep_angle = csdl.arccos(csdl.vdot(spanwise_direction, np.array([0., -1., 0.])) / csdl.norm(spanwise_direction))
sweep_angle_left = csdl.arccos(-spanwise_direction_left[1] / csdl.norm(spanwise_direction_left))
sweep_angle_right = csdl.arccos(spanwise_direction_right[1] / csdl.norm(spanwise_direction_right))

print("Wingspan: ", wingspan.value)
print("Root Chord: ", root_chord.value)
print("Tip Chord Left: ", tip_chord_left.value)
print("Tip Chord Right: ", tip_chord_right.value)
print("Sweep Angle Left: ", sweep_angle_left.value*180/np.pi)
print("Sweep Angle Right: ", sweep_angle_right.value*180/np.pi)

# Create Newton solver for inner optimization
from typing import Union
class NewtonOptimizer:
    def __init__(self, objective:csdl.Variable=None, design_variables:list[csdl.Variable]=None, 
                 constraints:list[csdl.Variable]=None, constraint_penalties:list[Union[float,np.ndarray]]=None):
        self.objective = objective
        self.design_variables = design_variables
        self.constraints = constraints
        self.constraint_penalties = constraint_penalties

        if design_variables is None:
            self.design_variables = []
        if constraints is None:
            self.constraints = []
        if constraint_penalties is None:
            self.constraint_penalties = []

        self.solver = csdl.nonlinear_solvers.Newton()

    def add_objective(self, objective:csdl.Variable):
        self.objective = objective

    def add_design_variable(self, design_variable:csdl.Variable):
        self.design_variables.append(design_variable)

    def add_constraint(self, constraint:csdl.Variable, penalty:Union[float,np.ndarray]=None):
        self.constraints.append(constraint)
        self.constraint_penalties.append(penalty)


    def compute_lagrangian(self):
        '''
        Constructs the CSDL variable for the lagrangian of the optimization problem.
        '''
        lagrangian = self.objective
        for constraint, penalty in zip(self.constraints, self.constraint_penalties):
            if penalty is not None:
                lagrangian += penalty*constraint
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

        dL_dx = csdl.derivative(self.lagrangian, self.design_variables)
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
        dc_dx = self.compute_constraint_jacobian(constraints=self.constraints, design_variables=self.design_variables)

        lagrange_multipliers = []
        for constraint, penalty in zip(self.constraints, self.constraint_penalties):
            if penalty is None:
                constraint_lagrange_multipliers = csdl.ImplicitVariable(shape=(constraint.shape[0],), value=0.,
                                                                        name=f'{constraint.name}_lagrange_multipliers')
                lagrange_multipliers.append(constraint_lagrange_multipliers)
                self.solver.add_state(constraint_lagrange_multipliers, constraint)
            else:
                lagrange_multipliers.append(None)
                

        for design_variable in self.design_variables:
            residual = dL_dx[design_variable].reshape((design_variable.size,))
            for constraint, constraint_lagrange_multipliers in zip(self.constraints, lagrange_multipliers):
                if constraint_lagrange_multipliers is not None:
                    constraint_jacobian = dc_dx[constraint,design_variable]
                    residual = residual + csdl.tensordot(constraint_lagrange_multipliers, constraint_jacobian, axes=([0],[0]))
            residual.add_name(f'{design_variable.name}_residual')
            self.solver.add_state(design_variable, residual)

    def run(self):
        self.setup()
        self.solver.run()

chord_stretching_b_spline.coefficients.add_name('chord_stretching_b_spline_coefficients')
wingspan_stretching_b_spline.coefficients.add_name('wingspan_stretching_b_spline_coefficients')
sweep_translation_b_spline.coefficients.add_name('sweep_translation_b_spline_coefficients')


# solver = csdl.nonlinear_solvers.Newton()
objective = (csdl.vdot(chord_stretching_b_spline.coefficients, chord_stretching_b_spline.coefficients)
            + csdl.vdot(wingspan_stretching_b_spline.coefficients, wingspan_stretching_b_spline.coefficients)
            + csdl.vdot(sweep_translation_b_spline.coefficients, sweep_translation_b_spline.coefficients))
# vdot doesn't have derivative right now
# objective = (csdl.sum(chord_stretching_b_spline.coefficients*chord_stretching_b_spline.coefficients)
#             + csdl.sum(wingspan_stretching_b_spline.coefficients*wingspan_stretching_b_spline.coefficients)
#             + csdl.sum(sweep_translation_b_spline.coefficients*sweep_translation_b_spline.coefficients))

# df_dx = csdl.derivative(objective, 
#                                 [chord_stretching_b_spline.coefficients, 
#                                  wingspan_stretching_b_spline.coefficients, 
#                                  sweep_translation_b_spline.coefficients])
# df_d_chord_stretch = df_dx[chord_stretching_b_spline.coefficients].flatten()
# df_d_wingspan_stretch = df_dx[wingspan_stretching_b_spline.coefficients].flatten()
# df_d_sweep_translation = df_dx[sweep_translation_b_spline.coefficients].flatten()

wingspan_outer_dv = csdl.Variable(shape=(1,), value=np.array([6.0]))
root_chord_outer_dv = csdl.Variable(shape=(1,), value=np.array([2.0]))
tip_chord_outer_dv = csdl.Variable(shape=(1,), value=np.array([0.5]))
sweep_angle_outer_dv = csdl.Variable(shape=(1,), value=np.array([45*np.pi/180]))

constraint1 = wingspan - wingspan_outer_dv
constraint1.add_name('wingspan_constraint')
constraint2 = root_chord - root_chord_outer_dv
constraint2.add_name('root_chord_constraint')
constraint3 = tip_chord_left - tip_chord_outer_dv
constraint3.add_name('tip_chord_left_constraint')
constraint4 = tip_chord_right - tip_chord_outer_dv
constraint4.add_name('tip_chord_right_constraint')
constraint5 = sweep_angle_left - sweep_angle_outer_dv
constraint5.add_name('sweep_angle_left_constraint')
constraint6 = sweep_angle_right - sweep_angle_outer_dv
constraint6.add_name('sweep_angle_right_constraint')
# num_constraints = 6
# constraints_vector = csdl.Variable(shape=(num_constraints,), value=0.)
# constraints_vector = constraints_vector.set(csdl.slice[0], constraint1)
# constraints_vector = constraints_vector.set(csdl.slice[1], constraint2)
# constraints_vector = constraints_vector.set(csdl.slice[2], constraint3)
# constraints_vector = constraints_vector.set(csdl.slice[3], constraint4)
# constraints_vector = constraints_vector.set(csdl.slice[4], constraint5)
# constraints_vector = constraints_vector.set(csdl.slice[5], constraint6)

# lagrange_multipliers = csdl.ImplicitVariable(shape=(num_constraints,), value=0., name='lagrange_multipliers')

# dc_dx = csdl.derivative(constraints_vector,[chord_stretching_b_spline.coefficients,
#                                                     wingspan_stretching_b_spline.coefficients,
#                                                     sweep_translation_b_spline.coefficients])
# dc_d_chord_stretch = dc_dx[chord_stretching_b_spline.coefficients]
# dc_d_wingspan_stretch = dc_dx[wingspan_stretching_b_spline.coefficients]
# dc_d_sweep_translation = dc_dx[sweep_translation_b_spline.coefficients]

# chord_stretch_residual = df_d_chord_stretch + lagrange_multipliers @ dc_d_chord_stretch
# chord_stretch_residual = df_d_chord_stretch + csdl.tensordot(lagrange_multipliers, dc_d_chord_stretch, axes=([0],[0]))
# wingspan_stretch_residual = df_d_wingspan_stretch + lagrange_multipliers @ dc_d_wingspan_stretch
# wingspan_stretch_residual = df_d_wingspan_stretch + csdl.tensordot(lagrange_multipliers, dc_d_wingspan_stretch, axes=([0],[0]))
# sweep_translation_residual = df_d_sweep_translation + lagrange_multipliers @ dc_d_sweep_translation
# sweep_translation_residual = df_d_sweep_translation + csdl.tensordot(lagrange_multipliers, dc_d_sweep_translation, axes=([0],[0]))

# geometry_model = csdl.Model()
# geometry_model.add_objective(...)
# geometry_model.add_design_variable(...)
# ...
# geometry_optimizer = NewtonOptimizer(geometry_model)
# geometry_optimizer.run()

# geometry_optimizer = Optimization()
geometry_optimizer = NewtonOptimizer()
geometry_optimizer.add_objective(objective)
geometry_optimizer.add_design_variable(chord_stretching_b_spline.coefficients)
geometry_optimizer.add_design_variable(wingspan_stretching_b_spline.coefficients)
geometry_optimizer.add_design_variable(sweep_translation_b_spline.coefficients)
geometry_optimizer.add_constraint(constraint1)
geometry_optimizer.add_constraint(constraint2)
geometry_optimizer.add_constraint(constraint3)
geometry_optimizer.add_constraint(constraint4)
geometry_optimizer.add_constraint(constraint5)
geometry_optimizer.add_constraint(constraint6)

# states, residuals = geometry_optimizer.get_first_order_optimality()
# geometry_solver = csdl.nonlinear_solvers.Newton()
# geometry_solver.add_state(states, residuals)
# geometry_solver.add_optimization(geometry_optimizer)
geometry_optimizer.run()

# outer_optimizer.add_design_variables(states)
# outer_optimizer.add_constraints(residuals)
# outer_optimizer.add_nested_optimization(geometry_optimizer)

geometry.plot()


print("Wingspan: ", wingspan.value)
print("Root Chord: ", root_chord.value)
print("Tip Chord Left: ", tip_chord_left.value)
print("Tip Chord Right: ", tip_chord_right.value)
print("Sweep Angle Left: ", sweep_angle_left.value*180/np.pi)
print("Sweep Angle Right: ", sweep_angle_right.value*180/np.pi)
print("Chord Stretching: ", chord_stretching_b_spline.coefficients.value)
print("Wingspan Stretching: ", wingspan_stretching_b_spline.coefficients.value)
print("Sweep Translation: ", sweep_translation_b_spline.coefficients.value)
# print('Chord Stretching Residual: ', chord_stretch_residual.value)
# print('Wingspan Stretching Residual: ', wingspan_stretch_residual.value)
# print('Sweep Translation Residual: ', sweep_translation_residual.value)
# print('Constraints: ', constraints_vector.value)

# solver.add_state(chord_stretching_b_spline.coefficients, chord_stretch_residual)
# solver.add_state(wingspan_stretching_b_spline.coefficients, wingspan_stretch_residual)
# solver.add_state(sweep_translation_b_spline.coefficients, sweep_translation_residual)
# solver.add_state(lagrange_multipliers, constraints_vector, tolerance=1e-8)
# solver.run()

geometry_solver.run()
geometry.plot()

print("Wingspan: ", wingspan.value)
print("Root Chord: ", root_chord.value)
print("Tip Chord Left: ", tip_chord_left.value)
print("Tip Chord Right: ", tip_chord_right.value)
print("Sweep Angle Left: ", sweep_angle_left.value*180/np.pi)
print("Sweep Angle Right: ", sweep_angle_right.value*180/np.pi)
print("Chord Stretching: ", chord_stretching_b_spline.coefficients.value)
print("Wingspan Stretching: ", wingspan_stretching_b_spline.coefficients.value)
print("Sweep Translation: ", sweep_translation_b_spline.coefficients.value)
# print('Chord Stretching Residual: ', chord_stretch_residual.value)
# print('Wingspan Stretching Residual: ', wingspan_stretch_residual.value)
# print('Sweep Translation Residual: ', sweep_translation_residual.value)
# print('Constraints: ', constraints_vector.value)

# d_wingspan_dx = csdl.derivative.reverse(wingspan, [wingspan_stretching_b_spline.coefficients, chord_stretching_b_spline.coefficients])
# print(d_wingspan_dx)
# for key, value in d_wingspan_dx.items():
#     print(key, value.value)
# # from csdl_alpha.src.operations.derivative.utils import verify_derivatives_inline
# # verify_derivatives_inline([wingspan], [wingspan_stretching_b_spline.coefficients, chord_stretching_b_spline.coefficients])
# # d_root_chord_dx = csdl.derivative.reverse(root_chord, [wingspan_stretching_b_spline.coefficients, chord_stretching_b_spline.coefficients])
# # print(d_root_chord_dx)
# # for key, value in d_root_chord_dx.items():
# #     print(key, value.value)

# d_wingspan_d_wingspan_stretch = d_wingspan_dx[wingspan_stretching_b_spline.coefficients]
# test_output = d_wingspan_d_wingspan_stretch*10
# print(test_output.value)

# # objective = csdl.tensordot(wingspan_stretching_b_spline.coefficients, wingspan_stretching_b_spline.coefficients, axes=([0],[0]))
# # objective = csdl.sum(wingspan_stretching_b_spline.coefficients*wingspan_stretching_b_spline.coefficients)
# # objective = wingspan_stretching_b_spline.coefficients @ wingspan_stretching_b_spline.coefficients
# # residual = csdl.derivative.reverse(objective, [wingspan_stretching_b_spline.coefficients])[wingspan_stretching_b_spline.coefficients]
# # residual = residual.flatten()

# # derivatives = csdl.derivative([objective, f2], [wingspan_stretching_b_spline.coefficients, x2], mode=reverse)
# # df_dwingspan = derivatives[objective, wingspan_stretching_b_spline.coefficients]
# # df_dwingspan = derivatives[objective][wingspan_stretching_b_spline.coefficients]
# # df_dwingspan = derivatives[0,0]

# # gradient = csdl.derivative([lagrangian], [design_variables, lagrange_multipliers], return_type='block')
# # hessian = csdl.derivative([gradient], [design_variables, langrange_multiplers], return_type='block')
# # delta_x = csdl.solve_linear(hessian, -gradient)


# # df_dx = csdl.derivative(objective, [design_variables, lagrange_multipliers])[objective, design_variables]



# # solver = csdl.nonlinear_solvers.Newton()
# # solver.add_state(wingspan_stretching_b_spline.coefficients, residual, tolerance=1e-8, initial_value=1.)
# # solver.run()

# # print(wingspan_stretching_b_spline.coefficients.value)
# # print(residual.value)

# # csdl.get_current_recorder().print_graph_structure()
# # csdl.get_current_recorder().visualize_graph('my_graph')
# # exit()
# # parameterization_solver.declare_input(name="wingspan", input=wingspan)
# # parameterization_solver.declare_input(name="root_chord", input=root_chord)
# # parameterization_solver.declare_input(name="tip_chord_left", input=tip_chord_left)
# # parameterization_solver.declare_input(name="tip_chord_right", input=tip_chord_right)
# # endregion

# # # region Evaluate Parameterization Solver
# # parameterization_inputs["wingspan"] = csdl.Variable(
# #     name="wingspan", shape=(1,), value=np.array([6.0])
# # )
# # parameterization_inputs["root_chord"] = csdl.Variable(
# #     name="root_chord", shape=(1,), value=np.array([2.0])
# # )
# # parameterization_inputs["tip_chord_left"] = csdl.Variable(
# #     name="tip_chord_left", shape=(1,), value=np.array([0.5])
# # )
# # parameterization_inputs["tip_chord_right"] = csdl.Variable(
# #     name="tip_chord_right", shape=(1,), value=np.array([0.5])
# # )
# # # geometry.plot()

# # parameterization_solver_states = parameterization_solver.evaluate(
# #     inputs=parameterization_inputs
# # )
# # # endregion


# # region Evaluate Parameterization Forward Model Using Solver States
# # parametric_b_spline_inputs = np.linspace(0.0, 1.0, num_ffd_sections).reshape((-1, 1))
# # chord_stretching_b_spline.coefficients = parameterization_solver_states[
# #     "chord_stretching_b_spline_coefficients"
# # ]
# # wingspan_stretching_b_spline.coefficients = parameterization_solver_states[
# #     "wingspan_stretching_b_spline_coefficients"
# # ]

# chord_stretch_sectional_parameters = chord_stretching_b_spline.evaluate(
#     parametric_b_spline_inputs
# )
# wingspan_stretch_sectional_parameters = wingspan_stretching_b_spline.evaluate(
#     parametric_b_spline_inputs
# )
# sweep_translation_sectional_parameters = sweep_translation_b_spline.evaluate(
#     parametric_b_spline_inputs
# )

# # sectional_parameters = {
# #     "chord_stretching": chord_stretch_sectional_parameters,
# #     "wingspan_stretching": wingspan_stretch_sectional_parameters,
# #     "sweep_translation": sweep_translation_sectional_parameters,
# # }
# sectional_parameters = VolumeSectionalParameterizationInputs()
# sectional_parameters.add_sectional_stretch(axis=0, stretch=chord_stretch_sectional_parameters)
# sectional_parameters.add_sectional_translation(axis=1, translation=wingspan_stretch_sectional_parameters)
# sectional_parameters.add_sectional_translation(axis=0, translation=sweep_translation_sectional_parameters)
# sectional_parameters.add_sectional_rotation(axis=1, rotation=twist_sectional_parameters)

# ffd_coefficients = ffd_sectional_parameterization.evaluate(sectional_parameters)

# geometry_coefficients = ffd_block.evaluate(ffd_coefficients)

# geometry.assign_coefficients(geometry_coefficients)

# rotation_axis_origin = geometry.evaluate(geometry.project(np.array([0.5, 0., 0.5])))
# rotation_axis_vector = geometry.evaluate(geometry.project(np.array([0.5, 1., 0.5])))
# rotation_angles = 45
# geometry.rotate(rotation_axis_origin, rotation_axis_vector, rotation_angles)

# parameterization_inputs = {}

# wingspan = csdl.norm(
#     geometry.evaluate(leading_edge_right) - geometry.evaluate(leading_edge_left)
# )
# root_chord = csdl.norm(
#     geometry.evaluate(trailing_edge_center) - geometry.evaluate(leading_edge_center)
# )
# tip_chord_left = csdl.norm(
#     geometry.evaluate(trailing_edge_left) - geometry.evaluate(leading_edge_left)
# )
# tip_chord_right = csdl.norm(
#     geometry.evaluate(trailing_edge_right) - geometry.evaluate(leading_edge_right)
# )

# upper_surface_wireframe = geometry.evaluate(upper_surface_wireframe_parametric)
# lower_surface_wireframe = geometry.evaluate(lower_surface_wireframe_parametric)
# camber_surface = csdl.linear_combination(
#     upper_surface_wireframe, lower_surface_wireframe, 1
# ).reshape((num_chordwise, num_spanwise, 3))
# endregion

# region Print and Plot Geometric Outputs
# geometry.plot()
# geometry.plot_meshes([camber_surface])



# endregion

# endregion

# recorder.visualize_graph('my_graph')
# csdl.save_all_variables()
# # csdl.inline_save('variables')
# recorder.save_graph('graph')