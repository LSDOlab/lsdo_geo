import lsdo_geo
from lsdo_geo.splines.b_splines.b_spline import BSpline
import pickle
import meshio
import numpy as np
import m3l
import vedo

from lsdo_geo.core.parameterization.parameterization_solver_old import ParameterizationSolver
from lsdo_geo.core.parameterization.free_form_deformation_functions import construct_ffd_block_around_entities
from lsdo_geo.core.parameterization.volume_sectional_parameterization import VolumeSectionalParameterization
from lsdo_geo.splines.b_splines.b_spline_space import BSplineSpace

'''
Objective: Maximize tail tip displacement
Design variable: Length
Subject to: Length <= 0.5
'''

# region Import and Setup
def import_geometry() -> BSpline:
    with open("examples/advanced_examples/robotic_fish/pickle_files/fishy_volume_geometry.pickle", 'rb') as handle:
        fishy = pickle.load(handle)
        return fishy

fishy = import_geometry()
fishy.name = 'fishy'

# fishy.plot(opacity=0.3)
# region -Structural Mesh Projection
structural_mesh = meshio.read("examples/advanced_examples/robotic_fish/meshes/module_v1.msh")
structural_mesh_nodes = structural_mesh.points/1000 + np.array([0.08 + 0.04, 0, 0.])   # Shift the mesh to the right to make it the front module
structural_mesh_parametric = fishy.project(points=structural_mesh_nodes, grid_search_density=50, max_iterations=500, plot=False)
# endregion -Structural Mesh Projection

# region -Projections for Design Variables (Parameterization Solver Inputs)
fishy_nose_parametric = fishy.project(points=np.array([[0.3, 0., 0.]]), grid_search_density=100, plot=False)
fishy_tail_tip_parametric = fishy.project(points=np.array([[-0.2, 0., 0.]]), grid_search_density=100, plot=False)
# endregion -Projections for Design Variables (Parameterization Solver Inputs)

# endregion Import and Setup

system_model = m3l.Model()

# region Geometry Parameterization
# region -Create Parameterization Objects
geometry_parameterization_solver = ParameterizationSolver()

num_ffd_sections = 2
ffd_block = construct_ffd_block_around_entities(name='ffd_block', entities=fishy, num_coefficients=(num_ffd_sections,2,2), order=(2,2,2))
# ffd_block.plot()

ffd_sectional_parameterization = VolumeSectionalParameterization(
    name='ffd_sectional_parameterization',
    parameterized_points=ffd_block.coefficients,
    principal_parametric_dimension=0,
    parameterized_points_shape=ffd_block.coefficients_shape,
)
ffd_sectional_parameterization.add_sectional_translation(name='length_sectional_translations', axis=0)

linear_2_dof_space = BSplineSpace(name='linear_2_dof_space', order=2, parametric_coefficients_shape=(2,))


length_sectional_translations_b_spline_coefficients = m3l.Variable(
    name='length_sectional_translations_b_spline_coefficients',
    shape=(2,),
    value=np.array([0., 0.]),
)
length_sectional_translations_b_spline_parameterization = BSpline(
    name='length_sectional_translations_b_spline_parameterization',
    space=linear_2_dof_space,
    coefficients=length_sectional_translations_b_spline_coefficients,
    num_physical_dimensions=1,
)

geometry_parameterization_solver.declare_state(name='length_sectional_translations_b_spline_coefficients', 
                                               state=length_sectional_translations_b_spline_coefficients)
# endregion -Create Parameterization Objects

# region -Define Parameterization For Solver
parameterization_b_spline_input = np.linspace(0., 1., num_ffd_sections).reshape((-1,1))

length_sectional_translations = length_sectional_translations_b_spline_parameterization.evaluate(parameterization_b_spline_input)

sectional_parameters = {'length_sectional_translations':length_sectional_translations}
ffd_block_coefficients = ffd_sectional_parameterization.evaluate(sectional_parameters, plot=False)

fishy_coefficients = ffd_block.evaluate(coefficients=ffd_block_coefficients, plot=False)

fishy.coefficients = fishy_coefficients
# fishy.plot()

# region -Evaluate Parameterization Solver Inputs
fishy_nose = fishy.evaluate(fishy_nose_parametric)
fishy_tail_tip = fishy.evaluate(fishy_tail_tip_parametric)
fishy_length = m3l.norm(fishy_nose - fishy_tail_tip)

geometry_parameterization_solver.declare_input(name='fishy_length', input=fishy_length)

# endregion -Evaluate Parameterization Solver Inputs

# endregion -Evaluate Parameterization For Solver

# region -Evaluate Parameterization
optimization_inputs = {'fishy_length':m3l.Variable(name='fishy_length', shape=(1,), value=0.5)}
parameterization_solver_states = geometry_parameterization_solver.evaluate(optimization_inputs)

length_sectional_translations_b_spline_parameterization.coefficients = parameterization_solver_states['length_sectional_translations_b_spline_coefficients']

length_sectional_translations = length_sectional_translations_b_spline_parameterization.evaluate(parameterization_b_spline_input)

sectional_parameters = {'length_sectional_translations':length_sectional_translations}
ffd_block_coefficients = ffd_sectional_parameterization.evaluate(sectional_parameters, plot=False)

fishy_coefficients = ffd_block.evaluate(coefficients=ffd_block_coefficients, plot=False)
fishy.coefficients = fishy_coefficients
fishy.plot()
# endregion Geometry Parameterization

# region Evaluate Meshes
fishy_length = m3l.norm(fishy.evaluate(fishy_nose_parametric) - fishy.evaluate(fishy_tail_tip_parametric))
print(fishy_length)

structural_mesh = fishy.evaluate(structural_mesh_parametric).reshape((-1,3))

geometry_plot = fishy.plot(show=False, opacity=0.3)
mesh_points = vedo.Points(structural_mesh.value, r=5, c='gold').opacity(0.6)
plotter = vedo.Plotter()
plotter.show([geometry_plot, mesh_points], axes=1, viewup='y')

front_point_in_structural_solver = np.array([0.08, 0., 0.])
structural_mesh_displacements = structural_mesh - structural_mesh_nodes # + front_point_in_structural_solver - current_front_point
np.savetxt('structural_mesh_displacements.txt', structural_mesh_displacements.value)
# endregion Evaluate Meshes

# region Structural Solver

# endregion Structural Solver