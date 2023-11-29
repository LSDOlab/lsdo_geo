from lsdo_geo.core.geometry.geometry_functions import import_geometry
import numpy as np
import m3l
import time
import scipy.sparse as sps

# var1 = m3l.Variable('var1', shape=(2,3), value=np.array([[1., 2., 3.], [4., 5., 6.]]))
# var2 = m3l.Variable('var2', shape=(2,3), value=np.array([[1., 2., 3.], [4., 5., 6.]]))
# var3 = var1 + var2
# print(var3)

import vedo
camera = dict(
    # position=(-35, -30, 35.),
    position=(-45, -40, 45.),
    focal_point=(15., 0, 5.),
    viewup=(0, 0, 1),
    distance=0,
)

t1 = time.time()
geometry = import_geometry('lsdo_geo/splines/b_splines/sample_geometries/lift_plus_cruise_final.stp')
t2 = time.time()
geometry.refit(parallelize=True)
t3 = time.time()
# geometry.find_connections() # NOTE: This is really really slow for large geometries. Come back to this.
t4 = time.time()
# plotting_elements = geometry.plot(show=False)
# plotter = vedo.Plotter(size=(4200,2000),offscreen=False)
# plotter = vedo.Plotter(size=(1920,1200),offscreen=False)
# plotter = vedo.Plotter(size=(2920,2200),offscreen=False)
# plotter.show(plotting_elements, camera=camera)
geometry.plot(show=False)
t5 = time.time()
print('Import time: ', t2-t1)
print('Refit time: ', t3-t2)
# print('Find connections time: ', t4-t3)
print('Plot time: ', t5-t4)

exit()

wing_copy = geometry.declare_component(component_name='wing', b_spline_search_names=['Wing'])
# wing.plot()
horizontal_stabilizer = geometry.declare_component(component_name='horizontal_stabilizer', b_spline_search_names=['Tail_1'])
# horizontal_stabilizer.plot()

# geometry3 = geometry.copy()
# axis_origin = geometry.evaluate(geometry.project(np.array([0.5, -10., 0.5])))
# axis_vector = geometry.evaluate(geometry.project(np.array([0.5, 10., 0.5]))) - axis_origin
# angles = 45
# geometry3.coefficients = m3l.rotate(points=geometry3.coefficients.reshape((-1,3)), axis_origin=axis_origin, axis_vector=axis_vector,
#                                     angles=angles, units='degrees').reshape((-1,))
# # geometry3.plot()

# leading_edge_parametric_coordinates = [
#     ('WingGeom, 0, 3', np.array([1.,  0.])),
#     ('WingGeom, 0, 3', np.array([0.777, 0.])),
#     ('WingGeom, 0, 3', np.array([0.555, 0.])),
#     ('WingGeom, 0, 3', np.array([0.333, 0.])),
#     ('WingGeom, 0, 3', np.array([0.111, 0.])),
#     ('WingGeom, 1, 8', np.array([0.111 , 0.])),
#     ('WingGeom, 1, 8', np.array([0.333, 0.])),
#     ('WingGeom, 1, 8', np.array([0.555, 0.])),
#     ('WingGeom, 1, 8', np.array([0.777, 0.])),
#     ('WingGeom, 1, 8', np.array([1., 0.])),
# ]

# trailing_edge_parametric_coordinates = [
#     ('WingGeom, 0, 3', np.array([1.,  1.])),
#     ('WingGeom, 0, 3', np.array([0.777, 1.])),
#     ('WingGeom, 0, 3', np.array([0.555, 1.])),
#     ('WingGeom, 0, 3', np.array([0.333, 1.])),
#     ('WingGeom, 0, 3', np.array([0.111, 1.])),
#     ('WingGeom, 1, 8', np.array([0.111 , 1.])),
#     ('WingGeom, 1, 8', np.array([0.333, 1.])),
#     ('WingGeom, 1, 8', np.array([0.555, 1.])),
#     ('WingGeom, 1, 8', np.array([0.777, 1.])),
#     ('WingGeom, 1, 8', np.array([1., 1.])),
# ]

# geometry4 = geometry.copy()

# leading_edge = geometry4.evaluate(leading_edge_parametric_coordinates, plot=False).reshape((-1,3))
# trailing_edge = geometry4.evaluate(trailing_edge_parametric_coordinates, plot=False).reshape((-1,3))
# chord_surface = m3l.linspace(leading_edge, trailing_edge, num_steps=4)

# geometry4.plot_meshes(meshes=chord_surface, mesh_plot_types=['wireframe'], mesh_opacity=1., mesh_color='#F5F0E6',)

# # geometry4.rotate(axis_origin=axis_origin, axis_vector=axis_vector, angles=angles, units='degrees')
# left_wing_transition = geometry4.declare_component(component_name='left_wing', b_spline_search_names=['WingGeom, 1'])
# left_wing_transition.rotate(axis_origin=axis_origin, axis_vector=axis_vector, angles=angles, units='degrees')
# geometry4.plot()

# leading_edge = geometry4.evaluate(leading_edge_parametric_coordinates, plot=False).reshape((-1,3))
# trailing_edge = geometry4.evaluate(trailing_edge_parametric_coordinates, plot=False).reshape((-1,3))
# chord_surface = m3l.linspace(leading_edge, trailing_edge, num_steps=4)

# geometry4.plot_meshes(meshes=chord_surface, mesh_plot_types=['wireframe'], mesh_opacity=1., mesh_color='#F5F0E6',)

geometry5 = geometry.copy()
wing = geometry5.declare_component(component_name='wing', b_spline_search_names=['Wing'])
from lsdo_geo.core.parameterization.free_form_deformation_functions import construct_ffd_block_around_entities
wing_ffd_block = construct_ffd_block_around_entities(name='wing_ffd_block', entities=wing, num_coefficients=(2, 11, 2))
# plotting_elements = wing_ffd_block.plot(plot_embedded_points=False, show=False, additional_plotting_elements=plotting_elements)
horizontal_stabilizer_ffd_block = construct_ffd_block_around_entities(name='horizontal_stabilizer_ffd_block', entities=horizontal_stabilizer, num_coefficients=(2, 11, 2))
# plotting_elements = horizontal_stabilizer_ffd_block.plot(plot_embedded_points=False, show=False, additional_plotting_elements=plotting_elements)

from lsdo_geo.core.parameterization.volume_sectional_parameterization import VolumeSectionalParameterization
wing_ffd_block_sectional_parameterization = VolumeSectionalParameterization(name='wing_ffd_block_sectional_parameterization',
                                                                                    principal_parametric_dimension=1,
                                                                                    parameterized_points=wing_ffd_block.coefficients,
                                                        parameterized_points_shape=wing_ffd_block.coefficients_shape)

# wing_ffd_block_sectional_parameterization.add_sectional_translation(name='wing_sweep', axis=0)
wing_ffd_block_sectional_parameterization.add_sectional_translation(name='wing_diheral', axis=2)
# wing_ffd_block_sectional_parameterization.add_sectional_translation(name='wing_wingspan_stretch', axis=1)
# wing_ffd_block_sectional_parameterization.add_sectional_stretch(name='wing_chord_stretch', axis=0)
# wing_ffd_block_sectional_parameterization.add_sectional_rotation(name='wing_twist', axis=1)
# plotting_elements = wing_ffd_block_sectional_parameterization.plot(additional_plotting_elements=plotting_elements, show=False)

import lsdo_geo.splines.b_splines as bsp
linear_b_spline_curve_2_dof_space = bsp.BSplineSpace(name='linear_b_spline_curve_2_dof_space', order=2, parametric_coefficients_shape=(2,))
linear_b_spline_curve_3_dof_space = bsp.BSplineSpace(name='linear_b_spline_curve_3_dof_space', order=2, parametric_coefficients_shape=(3,))
cubic_b_spline_curve_5_dof_space = bsp.BSplineSpace(name='cubic_b_spline_curve_5_dof_space', order=4, parametric_coefficients_shape=(5,))

wing_sweep_coefficients = m3l.Variable(name='wing_sweep_coefficients', shape=(2,), value=np.array([1., 0.]))
wing_sweep_b_spline = bsp.BSpline(name='wing_sweep_b_spline', space=linear_b_spline_curve_2_dof_space, 
                                        coefficients=wing_sweep_coefficients, num_physical_dimensions=1)

wing_dihedral_coefficients = m3l.Variable(name='wing_dihedral_coefficients', shape=(2,), value=np.array([1., 0.]))
wing_dihedral_b_spline = bsp.BSpline(name='wing_dihedral_b_spline', space=linear_b_spline_curve_2_dof_space, 
                                        coefficients=wing_dihedral_coefficients, num_physical_dimensions=1)

wing_wingspan_stretch_coefficients = m3l.Variable(name='wingspan_stretch_coefficients', shape=(2,), value=np.array([-1., 0.]))
wing_wingspan_stretch_b_spline = bsp.BSpline(name='wing_wingspan_stretch_b_spline', space=linear_b_spline_curve_2_dof_space,
                                                    coefficients=wing_wingspan_stretch_coefficients, num_physical_dimensions=1)

wing_chord_stretch_coefficients = m3l.Variable(name='wing_chord_stretch_coefficients', shape=(3,), 
                                                    value=np.array([-0.5, -0.1, 0.]))
wing_chord_stretch_b_spline = bsp.BSpline(name='wing_chord_stretch_b_spline', space=linear_b_spline_curve_3_dof_space,
                                                coefficients=wing_chord_stretch_coefficients, num_physical_dimensions=1)

wing_twist_coefficients = m3l.Variable(name='wing_twist_coefficients', shape=(5,),
                                            value=np.array([0., 30., 20., 10., 0.]))
wing_twist_b_spline = bsp.BSpline(name='wing_twist_b_spline', space=cubic_b_spline_curve_5_dof_space,
                                            coefficients=wing_twist_coefficients, num_physical_dimensions=1)



section_parametric_coordinates = np.linspace(0., 1., wing_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
wing_sectional_sweep = wing_sweep_b_spline.evaluate(section_parametric_coordinates)
wing_sectional_diheral = wing_dihedral_b_spline.evaluate(section_parametric_coordinates)
wing_wingspan_stretch = wing_wingspan_stretch_b_spline.evaluate(section_parametric_coordinates)
wing_sectional_chord_stretch = wing_chord_stretch_b_spline.evaluate(section_parametric_coordinates)
wing_sectional_twist = wing_twist_b_spline.evaluate(section_parametric_coordinates)
# parameter_value_sweep = np.hstack((np.linspace(0, 2, 10), np.linspace(2, 0, 10)))
# video = vedo.Video('translation_w_parameter_1.mp4', duration=2, backend="cv")
# for i, parameter_value in enumerate(parameter_value_sweep):
#     print(i, parameter_value)
#     parameter_values = np.zeros((wing_ffd_block_sectional_parameterization.num_sections,))
#     parameter_values[-1] = parameter_value

#     wing_ffd_block = construct_ffd_block_around_entities(name='wing_ffd_block', entities=wing, num_coefficients=(2, 11, 2))

#     wing_ffd_block_sectional_parameterization = VolumeSectionalParameterization(name='wing_ffd_block_sectional_parameterization',
#                                                                                     principal_parametric_dimension=1,
#                                                                                     parameterized_points=wing_ffd_block.coefficients,
#                                                         parameterized_points_shape=wing_ffd_block.coefficients_shape)
#     wing_ffd_block_sectional_parameterization.add_sectional_translation(name='wing_diheral', axis=2)

#     # wing_sweep = m3l.Variable('wing_sweep', 
#     #                                                    shape=(wing_ffd_block_sectional_parameterization.num_sections,),
#     #                                                     value=np.linspace(1., 0., wing_ffd_block_sectional_parameterization.num_sections))
#     wing_diheral = m3l.Variable(name='wing_diheral', shape=(wing_ffd_block_sectional_parameterization.num_sections,),
#                                                         value=parameter_values)
#     # wing_chord_stretch = m3l.Variable('wing_chord_stretch',
#     #                                                      shape=(wing_ffd_block_sectional_parameterization.num_sections,),
#     #                                                       value=np.linspace(-.5, 0., wing_ffd_block_sectional_parameterization.num_sections))
#     # wing_twist = m3l.Variable('wing_twist',
#     #                                                      shape=(wing_ffd_block_sectional_parameterization.num_sections,),
#     #                                                       value=np.linspace(-30., 0., wing_ffd_block_sectional_parameterization.num_sections))
#     sectional_parameters = {
#         # 'wing_sweep':wing_sectional_sweep, 
#         'wing_diheral':wing_diheral,
#         # 'wing_wingspan_stretch':wing_wingspan_stretch,
#         # 'wing_chord_stretch':wing_sectional_chord_stretch,
#         # 'wing_twist':wing_sectional_twist,
#                             }


#     wing_ffd_block_coefficients = wing_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)

#     wing_coefficients = wing_ffd_block.evaluate(wing_ffd_block_coefficients, plot=False)

#     # wing_ffd_block_sectional_parameterization.plot()
#     geometry5.assign_coefficients(coefficients=wing_coefficients, b_spline_names=wing.b_spline_names)

#     plotting_elements = geometry5.plot(show=False)
#     plotting_elements = wing_ffd_block_sectional_parameterization.plot(show=False, additional_plotting_elements=plotting_elements)

#     plotter = vedo.Plotter(size=(4200,2000),offscreen=True)
#     plotter.show(plotting_elements, camera=camera)

#     video.add_frame()

#     geometry5.assign_coefficients(coefficients=wing_copy.get_coefficients(), b_spline_names=wing.b_spline_names)
# video.close()

# parameter_value_sweep = np.hstack((np.linspace(0, -3, 10), np.linspace(-3, 0, 10)))
# video = vedo.Video('stretch_v_parameter_1.mp4', duration=2, backend="cv")
# for i, parameter_value in enumerate(parameter_value_sweep):
#     print(i, parameter_value)
#     parameter_values = np.zeros((wing_ffd_block_sectional_parameterization.num_sections,))
#     parameter_values[0] = parameter_value

#     wing_ffd_block = construct_ffd_block_around_entities(name='wing_ffd_block', entities=wing, num_coefficients=(2, 11, 2))

#     wing_ffd_block_sectional_parameterization = VolumeSectionalParameterization(name='wing_ffd_block_sectional_parameterization',
#                                                                                     principal_parametric_dimension=1,
#                                                                                     parameterized_points=wing_ffd_block.coefficients,
#                                                         parameterized_points_shape=wing_ffd_block.coefficients_shape)
#     wing_ffd_block_sectional_parameterization.add_sectional_stretch(name='wing_chord_stretch', axis=0)

#     # wing_sweep = m3l.Variable('wing_sweep', 
#     #                                                    shape=(wing_ffd_block_sectional_parameterization.num_sections,),
#     #                                                     value=np.linspace(1., 0., wing_ffd_block_sectional_parameterization.num_sections))
#     # wing_diheral = m3l.Variable(name='wing_diheral', shape=(wing_ffd_block_sectional_parameterization.num_sections,),
#     #                                                     value=parameter_values)
#     wing_chord_stretch = m3l.Variable(name='wing_chord_stretch',
#                                                          shape=(wing_ffd_block_sectional_parameterization.num_sections,),
#                                                           value=parameter_values)
#     # wing_twist = m3l.Variable('wing_twist',
#     #                                                      shape=(wing_ffd_block_sectional_parameterization.num_sections,),
#     #                                                       value=np.linspace(-30., 0., wing_ffd_block_sectional_parameterization.num_sections))
#     sectional_parameters = {
#         # 'wing_sweep':wing_sectional_sweep, 
#         # 'wing_diheral':wing_diheral,
#         # 'wing_wingspan_stretch':wing_wingspan_stretch,
#         'wing_chord_stretch':wing_chord_stretch,
#         # 'wing_twist':wing_sectional_twist,
#                             }


#     wing_ffd_block_coefficients = wing_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)

#     wing_coefficients = wing_ffd_block.evaluate(wing_ffd_block_coefficients, plot=False)

#     # wing_ffd_block_sectional_parameterization.plot()
#     geometry5.assign_coefficients(coefficients=wing_coefficients, b_spline_names=wing.b_spline_names)

#     plotting_elements = geometry5.plot(show=False)
#     plotting_elements = wing_ffd_block_sectional_parameterization.plot(show=False, additional_plotting_elements=plotting_elements)

#     plotter = vedo.Plotter(size=(4200,2000),offscreen=True)
#     plotter.show(plotting_elements, camera=camera)

#     video.add_frame()

#     geometry5.assign_coefficients(coefficients=wing_copy.get_coefficients(), b_spline_names=wing.b_spline_names)
# video.close()

# parameter_value_sweep = np.hstack((np.linspace(0, 4, 10), np.linspace(4, 0, 10)))
# video = vedo.Video('stretch_v_parameter_5.mp4', duration=2, backend="cv")
# for i, parameter_value in enumerate(parameter_value_sweep):
#     print(i, parameter_value)
#     parameter_values = np.zeros((wing_ffd_block_sectional_parameterization.num_sections,))
#     parameter_values[5] = parameter_value

#     wing_ffd_block = construct_ffd_block_around_entities(name='wing_ffd_block', entities=wing, num_coefficients=(2, 11, 2))

#     wing_ffd_block_sectional_parameterization = VolumeSectionalParameterization(name='wing_ffd_block_sectional_parameterization',
#                                                                                     principal_parametric_dimension=1,
#                                                                                     parameterized_points=wing_ffd_block.coefficients,
#                                                         parameterized_points_shape=wing_ffd_block.coefficients_shape)
#     wing_ffd_block_sectional_parameterization.add_sectional_stretch(name='wing_chord_stretch', axis=0)

#     # wing_sweep = m3l.Variable('wing_sweep', 
#     #                                                    shape=(wing_ffd_block_sectional_parameterization.num_sections,),
#     #                                                     value=np.linspace(1., 0., wing_ffd_block_sectional_parameterization.num_sections))
#     # wing_diheral = m3l.Variable(name='wing_diheral', shape=(wing_ffd_block_sectional_parameterization.num_sections,),
#     #                                                     value=parameter_values)
#     wing_chord_stretch = m3l.Variable(name='wing_chord_stretch',
#                                                          shape=(wing_ffd_block_sectional_parameterization.num_sections,),
#                                                           value=parameter_values)
#     # wing_twist = m3l.Variable('wing_twist',
#     #                                                      shape=(wing_ffd_block_sectional_parameterization.num_sections,),
#     #                                                       value=np.linspace(-30., 0., wing_ffd_block_sectional_parameterization.num_sections))
#     sectional_parameters = {
#         # 'wing_sweep':wing_sectional_sweep, 
#         # 'wing_diheral':wing_diheral,
#         # 'wing_wingspan_stretch':wing_wingspan_stretch,
#         'wing_chord_stretch':wing_chord_stretch,
#         # 'wing_twist':wing_sectional_twist,
#                             }


#     wing_ffd_block_coefficients = wing_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)

#     wing_coefficients = wing_ffd_block.evaluate(wing_ffd_block_coefficients, plot=False)

#     # wing_ffd_block_sectional_parameterization.plot()
#     geometry5.assign_coefficients(coefficients=wing_coefficients, b_spline_names=wing.b_spline_names)

#     plotting_elements = geometry5.plot(show=False)
#     plotting_elements = wing_ffd_block_sectional_parameterization.plot(show=False, additional_plotting_elements=plotting_elements)

#     plotter = vedo.Plotter(size=(4200,2000),offscreen=True)
#     plotter.show(plotting_elements, camera=camera)

#     video.add_frame()

#     geometry5.assign_coefficients(coefficients=wing_copy.get_coefficients(), b_spline_names=wing.b_spline_names)
# video.close()


# parameter_value_sweep = np.hstack((np.linspace(0, 40, 10), np.linspace(40, 0, 10)))
# video = vedo.Video('twist_parameter_3.mp4', duration=2, backend="cv")
# for i, parameter_value in enumerate(parameter_value_sweep):
#     print(i, parameter_value)
#     parameter_values = np.zeros((wing_ffd_block_sectional_parameterization.num_sections,))
#     parameter_values[3] = parameter_value

#     wing_ffd_block = construct_ffd_block_around_entities(name='wing_ffd_block', entities=wing, num_coefficients=(2, 11, 2))

#     wing_ffd_block_sectional_parameterization = VolumeSectionalParameterization(name='wing_ffd_block_sectional_parameterization',
#                                                                                     principal_parametric_dimension=1,
#                                                                                     parameterized_points=wing_ffd_block.coefficients,
#                                                         parameterized_points_shape=wing_ffd_block.coefficients_shape)
#     wing_ffd_block_sectional_parameterization.add_sectional_rotation(name='wing_twist', axis=1)

#     # wing_sweep = m3l.Variable('wing_sweep', 
#     #                                                    shape=(wing_ffd_block_sectional_parameterization.num_sections,),
#     #                                                     value=np.linspace(1., 0., wing_ffd_block_sectional_parameterization.num_sections))
#     # wing_diheral = m3l.Variable(name='wing_diheral', shape=(wing_ffd_block_sectional_parameterization.num_sections,),
#     #                                                     value=parameter_values)
#     # wing_chord_stretch = m3l.Variable(name='wing_chord_stretch',
#     #                                                      shape=(wing_ffd_block_sectional_parameterization.num_sections,),
#     #                                                       value=parameter_values)
#     wing_twist = m3l.Variable(name='wing_twist',
#                                                          shape=(wing_ffd_block_sectional_parameterization.num_sections,),
#                                                           value=parameter_values)
#     sectional_parameters = {
#         # 'wing_sweep':wing_sectional_sweep, 
#         # 'wing_diheral':wing_diheral,
#         # 'wing_wingspan_stretch':wing_wingspan_stretch,
#         # 'wing_chord_stretch':wing_chord_stretch,
#         'wing_twist':wing_twist,
#                             }


#     wing_ffd_block_coefficients = wing_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)

#     wing_coefficients = wing_ffd_block.evaluate(wing_ffd_block_coefficients, plot=False)

#     # wing_ffd_block_sectional_parameterization.plot()
#     geometry5.assign_coefficients(coefficients=wing_coefficients, b_spline_names=wing.b_spline_names)

#     plotting_elements = geometry5.plot(show=False)
#     plotting_elements = wing_ffd_block_sectional_parameterization.plot(show=False, additional_plotting_elements=plotting_elements)

#     plotter = vedo.Plotter(size=(4200,2000),offscreen=True)
#     plotter.show(plotting_elements, camera=camera)

#     video.add_frame()

#     geometry5.assign_coefficients(coefficients=wing_copy.get_coefficients(), b_spline_names=wing.b_spline_names)
# video.close()


# # Wingspan stretch -- Good to generate a video of this
# parameter_value_sweep = np.hstack((np.linspace(0, 10, 10), np.linspace(10, 0, 10)))
# video = vedo.Video('wingspan_stretch.mp4', duration=2, backend="cv")
# for i, parameter_value in enumerate(parameter_value_sweep):
#     print(i, parameter_value)

#     # wing_sweep_coefficients = m3l.Variable(name='wing_sweep_coefficients', shape=(2,), value=np.array([1., 0.]))
#     # wing_dihedral_coefficients = m3l.Variable(name='wing_dihedral_coefficients', shape=(2,), value=np.array([1., 0.]))
#     wing_wingspan_stretch_coefficients = m3l.Variable(name='wingspan_stretch_coefficients', shape=(2,), value=np.array([-parameter_value, parameter_value]))
#     # wing_chord_stretch_coefficients = m3l.Variable(name='wing_chord_stretch_coefficients', shape=(3,), value=np.array([-0.5, -0.1, 0.]))
#     # wing_twist_coefficients = m3l.Variable(name='wing_twist_coefficients', shape=(5,), value=np.array([0., 30., 20., 10., 0.]))

#     wing_wingspan_stretch_b_spline = bsp.BSpline(name='wing_wingspan_stretch_b_spline', space=linear_b_spline_curve_2_dof_space,
#                                                     coefficients=wing_wingspan_stretch_coefficients, num_physical_dimensions=1)

#     section_parametric_coordinates = np.linspace(0., 1., wing_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
#     # wing_sectional_sweep = wing_sweep_b_spline.evaluate(section_parametric_coordinates)
#     # wing_sectional_diheral = wing_dihedral_b_spline.evaluate(section_parametric_coordinates)
#     wing_wingspan_stretch = wing_wingspan_stretch_b_spline.evaluate(section_parametric_coordinates)
#     # wing_sectional_chord_stretch = wing_chord_stretch_b_spline.evaluate(section_parametric_coordinates)
#     # wing_sectional_twist = wing_twist_b_spline.evaluate(section_parametric_coordinates)

#     wing_ffd_block = construct_ffd_block_around_entities(name='wing_ffd_block', entities=wing, num_coefficients=(2, 11, 2))

#     wing_ffd_block_sectional_parameterization = VolumeSectionalParameterization(name='wing_ffd_block_sectional_parameterization',
#                                                                                     principal_parametric_dimension=1,
#                                                                                     parameterized_points=wing_ffd_block.coefficients,
#                                                         parameterized_points_shape=wing_ffd_block.coefficients_shape)
#     wing_ffd_block_sectional_parameterization.add_sectional_translation(name='wingspan_stretch', axis=1)


#     sectional_parameters = {
#         # 'wing_sweep':wing_sectional_sweep, 
#         # 'wing_diheral':wing_diheral,
#         'wingspan_stretch':wing_wingspan_stretch,
#         # 'wing_chord_stretch':wing_chord_stretch,
#         # 'wing_twist':wing_twist,
#                             }


#     wing_ffd_block_coefficients = wing_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)

#     wing_coefficients = wing_ffd_block.evaluate(wing_ffd_block_coefficients, plot=False)

#     # wing_ffd_block_sectional_parameterization.plot()
#     geometry5.assign_coefficients(coefficients=wing_coefficients, b_spline_names=wing.b_spline_names)

#     plotting_elements = geometry5.plot(show=False)
#     plotting_elements = wing_ffd_block_sectional_parameterization.plot(show=False, additional_plotting_elements=plotting_elements)

#     plotter = vedo.Plotter(size=(3920,2200),offscreen=True)
#     plotter.show(plotting_elements, camera=camera)

#     video.add_frame()

#     geometry5.assign_coefficients(coefficients=wing_copy.get_coefficients(), b_spline_names=wing.b_spline_names)
# video.close()


# # Linear root chord stretch -- Good to generate a video of this
# parameter_value_sweep = np.hstack((np.linspace(0, 6, 10), np.linspace(6, 0, 10)))
# video = vedo.Video('chord_stretch.mp4', duration=2, backend="cv")
# for i, parameter_value in enumerate(parameter_value_sweep):
#     print(i, parameter_value)

#     # wing_sweep_coefficients = m3l.Variable(name='wing_sweep_coefficients', shape=(2,), value=np.array([1., 0.]))
#     # wing_dihedral_coefficients = m3l.Variable(name='wing_dihedral_coefficients', shape=(2,), value=np.array([1., 0.]))
#     # wing_wingspan_stretch_coefficients = m3l.Variable(name='wingspan_stretch_coefficients', shape=(2,), value=np.array([-parameter_value, parameter_value]))
#     # wing_chord_stretch_coefficients = m3l.Variable(name='wing_chord_stretch_coefficients', shape=(3,), value=np.array([-0., parameter_value, 0.]))
#     # wing_twist_coefficients = m3l.Variable(name='wing_twist_coefficients', shape=(5,), value=np.array([0., 30., 20., 10., 0.]))

#     wing_chord_stretch_coefficients = m3l.Variable(name='wing_chord_stretch_coefficients', shape=(3,), 
#                                                     value=np.array([-0., parameter_value, 0.]))
#     wing_chord_stretch_b_spline = bsp.BSpline(name='wing_chord_stretch_b_spline', space=linear_b_spline_curve_3_dof_space,
#                                                 coefficients=wing_chord_stretch_coefficients, num_physical_dimensions=1)

#     section_parametric_coordinates = np.linspace(0., 1., wing_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
#     # wing_sectional_sweep = wing_sweep_b_spline.evaluate(section_parametric_coordinates)
#     # wing_sectional_diheral = wing_dihedral_b_spline.evaluate(section_parametric_coordinates)
#     # wing_wingspan_stretch = wing_wingspan_stretch_b_spline.evaluate(section_parametric_coordinates)
#     wing_sectional_chord_stretch = wing_chord_stretch_b_spline.evaluate(section_parametric_coordinates)
#     # wing_sectional_twist = wing_twist_b_spline.evaluate(section_parametric_coordinates)

#     wing_ffd_block = construct_ffd_block_around_entities(name='wing_ffd_block', entities=wing, num_coefficients=(2, 11, 2))

#     wing_ffd_block_sectional_parameterization = VolumeSectionalParameterization(name='wing_ffd_block_sectional_parameterization',
#                                                                                     principal_parametric_dimension=1,
#                                                                                     parameterized_points=wing_ffd_block.coefficients,
#                                                         parameterized_points_shape=wing_ffd_block.coefficients_shape)
#     wing_ffd_block_sectional_parameterization.add_sectional_stretch(name='chord_stretch', axis=0)


#     sectional_parameters = {
#         # 'wing_sweep':wing_sectional_sweep, 
#         # 'wing_diheral':wing_diheral,
#         # 'wing_wingspan_stretch':wing_wingspan_stretch,
#         'chord_stretch':wing_sectional_chord_stretch,
#         # 'wing_twist':wing_twist,
#                             }


#     wing_ffd_block_coefficients = wing_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)

#     wing_coefficients = wing_ffd_block.evaluate(wing_ffd_block_coefficients, plot=False)

#     # wing_ffd_block_sectional_parameterization.plot()
#     geometry5.assign_coefficients(coefficients=wing_coefficients, b_spline_names=wing.b_spline_names)

#     plotting_elements = geometry5.plot(show=False)
#     plotting_elements = wing_ffd_block_sectional_parameterization.plot(show=False, additional_plotting_elements=plotting_elements)

#     plotter = vedo.Plotter(size=(1920,1200),offscreen=True)
#     plotter.show(plotting_elements, camera=camera)

#     video.add_frame()

#     geometry5.assign_coefficients(coefficients=wing_copy.get_coefficients(), b_spline_names=wing.b_spline_names)
# video.close()


# Linear sweep translation  -- Good to generate a video of this
parameter_value_sweep = np.hstack((np.linspace(0, 15, 10), np.linspace(15, 0, 10)))
video = vedo.Video('sweep_translation.mp4', duration=2, backend="cv")
for i, parameter_value in enumerate(parameter_value_sweep):
    print(i, parameter_value)

    # wing_sweep_coefficients = m3l.Variable(name='wing_sweep_coefficients', shape=(2,), value=np.array([1., 0.]))
    # wing_dihedral_coefficients = m3l.Variable(name='wing_dihedral_coefficients', shape=(2,), value=np.array([1., 0.]))
    # wing_wingspan_stretch_coefficients = m3l.Variable(name='wingspan_stretch_coefficients', shape=(2,), value=np.array([-parameter_value, parameter_value]))
    # wing_chord_stretch_coefficients = m3l.Variable(name='wing_chord_stretch_coefficients', shape=(3,), value=np.array([-0.5, -0.1, 0.]))
    # wing_twist_coefficients = m3l.Variable(name='wing_twist_coefficients', shape=(5,), value=np.array([0., 30., 20., 10., 0.]))

    wing_sweep_coefficients = m3l.Variable(name='wing_sweep_coefficients', shape=(3,), value=np.array([parameter_value, 0., parameter_value]))
    wing_sweep_b_spline = bsp.BSpline(name='wing_sweep_b_spline', space=linear_b_spline_curve_3_dof_space, 
                                        coefficients=wing_sweep_coefficients, num_physical_dimensions=1)

    section_parametric_coordinates = np.linspace(0., 1., wing_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
    wing_sectional_sweep = wing_sweep_b_spline.evaluate(section_parametric_coordinates)
    # wing_sectional_diheral = wing_dihedral_b_spline.evaluate(section_parametric_coordinates)
    # wing_wingspan_stretch = wing_wingspan_stretch_b_spline.evaluate(section_parametric_coordinates)
    # wing_sectional_chord_stretch = wing_chord_stretch_b_spline.evaluate(section_parametric_coordinates)
    # wing_sectional_twist = wing_twist_b_spline.evaluate(section_parametric_coordinates)

    wing_ffd_block = construct_ffd_block_around_entities(name='wing_ffd_block', entities=wing, num_coefficients=(2, 11, 2))

    wing_ffd_block_sectional_parameterization = VolumeSectionalParameterization(name='wing_ffd_block_sectional_parameterization',
                                                                                    principal_parametric_dimension=1,
                                                                                    parameterized_points=wing_ffd_block.coefficients,
                                                        parameterized_points_shape=wing_ffd_block.coefficients_shape)
    wing_ffd_block_sectional_parameterization.add_sectional_translation(name='wing_sweep', axis=0)


    sectional_parameters = {
        'wing_sweep':wing_sectional_sweep, 
        # 'wing_diheral':wing_diheral,
        # 'wing_wingspan_stretch':wing_wingspan_stretch,
        # 'wing_chord_stretch':wing_chord_stretch,
        # 'wing_twist':wing_twist,
                            }


    wing_ffd_block_coefficients = wing_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)

    wing_coefficients = wing_ffd_block.evaluate(wing_ffd_block_coefficients, plot=False)

    # wing_ffd_block_sectional_parameterization.plot()
    geometry5.assign_coefficients(coefficients=wing_coefficients, b_spline_names=wing.b_spline_names)

    plotting_elements = geometry5.plot(show=False)
    plotting_elements = wing_ffd_block_sectional_parameterization.plot(show=False, additional_plotting_elements=plotting_elements)

    plotter = vedo.Plotter(size=(3920,2200),offscreen=True)
    plotter.show(plotting_elements, camera=camera)

    video.add_frame()

    geometry5.assign_coefficients(coefficients=wing_copy.get_coefficients(), b_spline_names=wing.b_spline_names)
video.close()


# # Linear twist
# parameter_value_sweep = np.hstack((np.linspace(0, 40, 10), np.linspace(40, 0, 10)))
# video = vedo.Video('twist.mp4', duration=2, backend="cv")
# for i, parameter_value in enumerate(parameter_value_sweep):
#     print(i, parameter_value)

#     # wing_sweep_coefficients = m3l.Variable(name='wing_sweep_coefficients', shape=(2,), value=np.array([1., 0.]))
#     # wing_dihedral_coefficients = m3l.Variable(name='wing_dihedral_coefficients', shape=(2,), value=np.array([1., 0.]))
#     # wing_wingspan_stretch_coefficients = m3l.Variable(name='wingspan_stretch_coefficients', shape=(2,), value=np.array([-parameter_value, parameter_value]))
#     # wing_chord_stretch_coefficients = m3l.Variable(name='wing_chord_stretch_coefficients', shape=(3,), value=np.array([-0.5, -0.1, 0.]))
#     # wing_twist_coefficients = m3l.Variable(name='wing_twist_coefficients', shape=(5,), value=np.array([0., 30., 20., 10., 0.]))

#     wing_twist_coefficients = m3l.Variable(name='wing_twist_coefficients', shape=(5,),
#                                             value=np.array([0., 0., parameter_value, 0., 0.]))
#     wing_twist_b_spline = bsp.BSpline(name='wing_twist_b_spline', space=cubic_b_spline_curve_5_dof_space,
#                                                 coefficients=wing_twist_coefficients, num_physical_dimensions=1)

#     section_parametric_coordinates = np.linspace(0., 1., wing_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
#     # wing_sectional_sweep = wing_sweep_b_spline.evaluate(section_parametric_coordinates)
#     # wing_sectional_diheral = wing_dihedral_b_spline.evaluate(section_parametric_coordinates)
#     # wing_wingspan_stretch = wing_wingspan_stretch_b_spline.evaluate(section_parametric_coordinates)
#     # wing_sectional_chord_stretch = wing_chord_stretch_b_spline.evaluate(section_parametric_coordinates)
#     wing_twist = wing_twist_b_spline.evaluate(section_parametric_coordinates)

#     wing_ffd_block = construct_ffd_block_around_entities(name='wing_ffd_block', entities=wing, num_coefficients=(2, 11, 2))

#     wing_ffd_block_sectional_parameterization = VolumeSectionalParameterization(name='wing_ffd_block_sectional_parameterization',
#                                                                                     principal_parametric_dimension=1,
#                                                                                     parameterized_points=wing_ffd_block.coefficients,
#                                                         parameterized_points_shape=wing_ffd_block.coefficients_shape)
#     wing_ffd_block_sectional_parameterization.add_sectional_rotation(name='wing_twist', axis=1)


#     sectional_parameters = {
#         # 'wing_sweep':wing_sectional_sweep, 
#         # 'wing_diheral':wing_diheral,
#         # 'wing_wingspan_stretch':wing_wingspan_stretch,
#         # 'wing_chord_stretch':wing_chord_stretch,
#         'wing_twist':wing_twist,
#                             }


#     wing_ffd_block_coefficients = wing_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)

#     wing_coefficients = wing_ffd_block.evaluate(wing_ffd_block_coefficients, plot=False)

#     # wing_ffd_block_sectional_parameterization.plot()
#     geometry5.assign_coefficients(coefficients=wing_coefficients, b_spline_names=wing.b_spline_names)

#     plotting_elements = geometry5.plot(show=False)
#     plotting_elements = wing_ffd_block_sectional_parameterization.plot(show=False, additional_plotting_elements=plotting_elements)

#     plotter = vedo.Plotter(size=(1920,1200),offscreen=True)
#     plotter.show(plotting_elements, camera=camera)

#     video.add_frame()

#     geometry5.assign_coefficients(coefficients=wing_copy.get_coefficients(), b_spline_names=wing.b_spline_names)
# video.close()


print('hi')