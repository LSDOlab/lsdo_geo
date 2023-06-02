import csdl
from python_csdl_backend import Simulator
import numpy as np
import array_mapper as am

from caddee.caddee_core.system_representation.system_representation import SystemRepresentation
from caddee.caddee_core.system_parameterization.system_parameterization import SystemParameterization

system_representation = SystemRepresentation()
spatial_rep = system_representation.spatial_representation
file_path = 'models/stp/'
file_name = 'lift_plus_cruise.stp'
# file_path = 'models/stp/'
spatial_rep.import_file(file_name=file_path+file_name)
# spatial_rep.import_file(file_name=file_path+'lift_plus_cruise_final_3.stp')
spatial_rep.refit_geometry(num_control_points=15, fit_resolution=30, file_name=file_path + file_name)

import vedo
plotting_elements = spatial_rep.plot(surface_texture='metallic')

light1 = vedo.Light([-5,0,0], c='w', intensity=1)
light2 = vedo.Light([0,0,20], c='w', intensity=2)
light3 = vedo.Light([0,-5,-15], c='w', intensity=3)
camera = dict(
    position=(-35, -30, 35),
    focal_point=(15, 0, 5),
    viewup=(0, 0, 1),
    distance=0,
)
plotting_elements.extend([light1, light2, light3])
plotter = vedo.Plotter(size=(3200,1000))
plotter.show(plotting_elements, camera=camera)

system_parameterization = SystemParameterization(system_representation=system_representation)

# Create Components
from caddee.caddee_core.system_representation.component.component import LiftingSurface, Component
wing_primitive_names = list(spatial_rep.get_primitives(search_names=['Wing']).keys())
wing = LiftingSurface(name='wing', spatial_representation=spatial_rep, primitive_names=wing_primitive_names)
tail_primitive_names = list(spatial_rep.get_primitives(search_names=['Tail_1']).keys())
horizontal_stabilizer = LiftingSurface(name='tail', spatial_representation=spatial_rep, primitive_names=tail_primitive_names)
fuselage_primitive_names = list(spatial_rep.get_primitives(search_names=['Fuselage']))
fuselage = Component(name='fuselage', spatial_representation=spatial_rep, primitive_names=fuselage_primitive_names)

rotor1_primitive_names = list(spatial_rep.get_primitives(search_names=['Rotor_1']))
rotor_1 = Component(name='rotor_1', spatial_representation=spatial_rep, primitive_names=rotor1_primitive_names)
rotor2_primitive_names = list(spatial_rep.get_primitives(search_names=['Rotor_2']))
rotor_2 = Component(name='rotor_2', spatial_representation=spatial_rep, primitive_names=rotor2_primitive_names)
rotor3_primitive_names = list(spatial_rep.get_primitives(search_names=['Rotor_3']))
rotor_3 = Component(name='rotor_3', spatial_representation=spatial_rep, primitive_names=rotor3_primitive_names)
rotor4_primitive_names = list(spatial_rep.get_primitives(search_names=['Rotor_4']))
rotor_4 = Component(name='rotor_4', spatial_representation=spatial_rep, primitive_names=rotor4_primitive_names)
rotor5_primitive_names = list(spatial_rep.get_primitives(search_names=['Rotor_5']))
rotor_5 = Component(name='rotor_5', spatial_representation=spatial_rep, primitive_names=rotor5_primitive_names)
rotor6_primitive_names = list(spatial_rep.get_primitives(search_names=['Rotor_6']))
rotor_6 = Component(name='rotor_6', spatial_representation=spatial_rep, primitive_names=rotor6_primitive_names)
rotor7_primitive_names = list(spatial_rep.get_primitives(search_names=['Rotor_7']))
rotor_7 = Component(name='rotor_7', spatial_representation=spatial_rep, primitive_names=rotor7_primitive_names)
rotor8_primitive_names = list(spatial_rep.get_primitives(search_names=['Rotor_8']))
rotor_8 = Component(name='rotor_8', spatial_representation=spatial_rep, primitive_names=rotor8_primitive_names)

system_representation.add_component(wing)
system_representation.add_component(horizontal_stabilizer)
system_representation.add_component(fuselage)
system_representation.add_component(rotor_1)
system_representation.add_component(rotor_2)
system_representation.add_component(rotor_3)
system_representation.add_component(rotor_4)
system_representation.add_component(rotor_5)
system_representation.add_component(rotor_6)
system_representation.add_component(rotor_7)
system_representation.add_component(rotor_8)

# Define actautor power_systems_architecture
# Fixed component: Fuselage
# Relative Actuations:
# 1. Tail actuates relative to fuselage
# -- Actuation axis
# -- -- NOTE: Definitely want to use quaternions where the axis is a MappedArray.
# -- Actuation value
# -- Do we want to specify an actuator object that has a location on the fuselage and another location on the tail?
# -- -- I think this is more intuitive from the perspective of a system configuration. The location can still be a DV, but this needs to work with param.
# -- -- In the case of something like a motor, the location on each component specifies the axis? Only if the points are not directly stacked.
# -- -- -- It seems like we may want to separately along axis specification for cases like tilt-win where the axis of rotation should really only dependent
#           on the wing and not on its position along the fuselage.
# -- -- -- Should locations on each component be optional, and if so, they are auto-detected? If specified, can they be DV and/or move in the parameterization?
# -- 
# 2. Wing actuates relative to fuselage

# NOTE: An actuator is a component (converts energy from electrical to mechanical) with two joints (for example fuselage<-->act<-->tail, these 2 joints are fixed)
# -- However, there is also a joint within the actuator

# tail_actuator = Motor(name='', stiffness=0., damping=0.) # NOTE: Motor is a component that is specifically an actuator
# CADDEE api
# tail_actuator = Component(name='motor', spatial_representation=spatial_rep, primitive_names=[])
# system_representation.add_actuator()

# actuation_solver = PrescribedActuation(component=tail_actuator)

# System Configuration api that System model will call.


# Meshes definitions
num_spanwise_vlm = 21
num_chordwise_vlm = 5
leading_edge = wing.project(am.linspace(am.array([8., -26., 7.5]), am.array([8., 26., 7.5]), num_spanwise_vlm), direction=am.array([0., 0., -1.]), plot=True)  # returns MappedArray
trailing_edge = wing.project(np.linspace(np.array([15., -26., 7.5]), np.array([15., 26., 7.5]), num_spanwise_vlm), direction=np.array([0., 0., -1.]))   # returns MappedArray
chord_surface = am.linspace(leading_edge, trailing_edge, num_chordwise_vlm)
wing_upper_surface_wireframe = wing.project(chord_surface.value + np.array([0., 0., 1.]), direction=np.array([0., 0., -1.]), grid_search_n=15)
wing_lower_surface_wireframe = wing.project(chord_surface.value - np.array([0., 0., 1.]), direction=np.array([0., 0., 1.]), grid_search_n=15)
wing_camber_surface = am.linspace(wing_upper_surface_wireframe, wing_lower_surface_wireframe, 1) # this linspace will return average when n=1
wing_camber_surface = wing_camber_surface.reshape((num_chordwise_vlm, num_spanwise_vlm, 3))
system_representation.add_output(name='chord_distribution', quantity=am.norm(leading_edge-trailing_edge))

spatial_rep.plot_meshes([leading_edge])

num_spanwise_vlm = 11
num_chordwise_vlm = 3
leading_edge = horizontal_stabilizer.project(np.linspace(np.array([27., -6.5, 6.]), np.array([27., 6.75, 6.]), num_spanwise_vlm), direction=np.array([0., 0., -1.]), grid_search_n=15)  # returns MappedArray
trailing_edge = horizontal_stabilizer.project(np.linspace(np.array([31.5, -6.5, 6.]), np.array([31.5, 6.75, 6.]), num_spanwise_vlm), direction=np.array([0., 0., -1.]), grid_search_n=15)   # returns MappedArray
chord_surface = am.linspace(leading_edge, trailing_edge, num_chordwise_vlm)
horizontal_stabilizer_upper_surface_wireframe = horizontal_stabilizer.project(chord_surface.value + np.array([0., 0., 1.]), direction=np.array([0., 0., -1.]), grid_search_n=15)
horizontal_stabilizer_lower_surface_wireframe = horizontal_stabilizer.project(chord_surface.value - np.array([0., 0., 1.]), direction=np.array([0., 0., 1.]), grid_search_n=15)
horizontal_stabilizer_camber_surface = am.linspace(horizontal_stabilizer_upper_surface_wireframe, horizontal_stabilizer_lower_surface_wireframe, 1) # this linspace will return average when n=1
horizontal_stabilizer_camber_surface = horizontal_stabilizer_camber_surface.reshape((num_chordwise_vlm, num_spanwise_vlm, 3))

system_representation.add_output('wing_camber_surface', wing_camber_surface)
system_representation.add_output('horizontal_stabilizer_camber_surface', horizontal_stabilizer_camber_surface)
starboard_trailing_tip = wing.project(np.array([15., 26., 7.5]), direction=np.array([0., 0., -1.]))
port_trailing_tip = wing.project(np.array([15., -26., 7.5]), direction=np.array([0., 0., -1.]))
wingspan_vector = starboard_trailing_tip - port_trailing_tip
wingspan = am.norm(wingspan_vector)     # NOTE: Nonlinear operations don't return MappedArrays. They return NonlinearMappedarrays
system_representation.add_output(name='wingspan', quantity=wingspan)
root_leading = wing.project(np.array([9., 0., 7.5]), direction=np.array([0., 0., -1.]))
root_trailing = wing.project(np.array([15., 0., 7.5]), direction=np.array([0., 0., -1.]))
root_chord_vector = root_leading - root_trailing
root_chord = am.norm(root_chord_vector)     # NOTE: Nonlinear operations don't return MappedArrays. They return NonlinearMappedarrays
system_representation.add_output(name='wing_root_chord', quantity=root_chord)

# # Parameterization
from caddee.caddee_core.system_parameterization.free_form_deformation.ffd_functions import create_cartesian_enclosure_volume
from caddee.caddee_core.system_parameterization.free_form_deformation.ffd_block import SRBGFFDBlock

wing_geometry_primitives = wing.get_geometry_primitives()
wing_ffd_bspline_volume = create_cartesian_enclosure_volume(wing_geometry_primitives, num_control_points=(11, 2, 2), order=(4,2,2), xyz_to_uvw_indices=(1,0,2))
wing_ffd_block = SRBGFFDBlock(name='wing_ffd_block', primitive=wing_ffd_bspline_volume, embedded_entities=wing_geometry_primitives)
wing_ffd_block.add_scale_v(name='linear_chord_distribution', order=2, num_dof=3, value=np.array([0., 1., 0.]))
# wing_ffd_block.add_scale_w(name='constant_thickness_scaling', order=1, num_dof=1, value=np.array([0.5]))
# wing_ffd_block.add_rotation_u(name='twist_distribution', order=4, num_dof=10, value=-1/4*np.array([0., 0.11, 0.22, 0.33, 0.44, 0.44, 0.33, 0.22, 0.11, 0.]))
wing_ffd_block.add_translation_u(name='wingspan_dof', order=2, num_dof=2, value=np.array([-30., 30]))

horizontal_stabilizer_geometry_primitives = horizontal_stabilizer.get_geometry_primitives()
horizontal_stabilizer_ffd_bspline_volume = create_cartesian_enclosure_volume(horizontal_stabilizer_geometry_primitives, num_control_points=(11, 2, 2), order=(4,2,2), xyz_to_uvw_indices=(1,0,2))
horizontal_stabilizer_ffd_block = SRBGFFDBlock(name='horizontal_stabilizer_ffd_block', primitive=horizontal_stabilizer_ffd_bspline_volume, embedded_entities=horizontal_stabilizer_geometry_primitives)
# horizontal_stabilizer_ffd_block.add_scale_v(name='horizontal_stabilizer_linear_taper', order=2, num_dof=3, value=np.array([0.5, 0.5, 0.5]), cost_factor=1.)
# horizontal_stabilizer_ffd_block.add_rotation_u(name='horizontal_stabilizer_twist_distribution', order=1, num_dof=1, value=np.array([np.pi/10]))
horizontal_stabilizer_ffd_block.add_translation_v('tail_moment_arm', order=1, num_dof=1, value=np.array([0.]))

rotor_1_geometry_primitives = rotor_1.get_geometry_primitives()
rotor_1_ffd_bspline_volume = create_cartesian_enclosure_volume(rotor_1_geometry_primitives, num_control_points=(2, 2, 2), order=(2,2,2), xyz_to_uvw_indices=(2,0,1))
rotor_1_ffd_block = SRBGFFDBlock(name='rotor_1_ffd_block', primitive=rotor_1_ffd_bspline_volume, embedded_entities=rotor_1_geometry_primitives)
rotor_1_ffd_block.add_scale_v(name='rotor_1_x_scaling', order=1, num_dof=1, value=np.array([2.]))
rotor_1_ffd_block.add_scale_w(name='rotor_1_y_scaling', order=1, num_dof=1, value=np.array([2.]))
rotor_1_ffd_block.add_translation_v(name='rotor_1_translation_x', order=1, num_dof=1, value=np.array([10.]))
rotor_1_ffd_block.add_translation_w(name='rotor_1_translation_y', order=1, num_dof=1, value=np.array([2.]))

rotor_2_geometry_primitives = rotor_2.get_geometry_primitives()
rotor_2_ffd_bspline_volume = create_cartesian_enclosure_volume(rotor_2_geometry_primitives, num_control_points=(2, 2, 2), order=(2,2,2), xyz_to_uvw_indices=(2,0,1))
rotor_2_ffd_block = SRBGFFDBlock(name='rotor_2_ffd_block', primitive=rotor_2_ffd_bspline_volume, embedded_entities=rotor_2_geometry_primitives)
rotor_2_ffd_block.add_scale_v(name='rotor_2_x_scaling', order=1, num_dof=1, value=np.array([2.]))
rotor_2_ffd_block.add_scale_w(name='rotor_2_y_scaling', order=1, num_dof=1, value=np.array([2.]))
rotor_2_ffd_block.add_translation_v(name='rotor_2_translation_x', order=1, num_dof=1, value=np.array([10.]))
rotor_2_ffd_block.add_translation_w(name='rotor_2_translation_y', order=1, num_dof=1, value=np.array([2.]))

rotor_3_geometry_primitives = rotor_3.get_geometry_primitives()
rotor_3_ffd_bspline_volume = create_cartesian_enclosure_volume(rotor_3_geometry_primitives, num_control_points=(2, 2, 2), order=(2,2,2), xyz_to_uvw_indices=(2,0,1))
rotor_3_ffd_block = SRBGFFDBlock(name='rotor_3_ffd_block', primitive=rotor_3_ffd_bspline_volume, embedded_entities=rotor_3_geometry_primitives)
rotor_3_ffd_block.add_scale_v(name='rotor_3_x_scaling', order=1, num_dof=1, value=np.array([2.]))
rotor_3_ffd_block.add_scale_w(name='rotor_3_y_scaling', order=1, num_dof=1, value=np.array([2.]))
rotor_3_ffd_block.add_translation_v(name='rotor_3_translation_x', order=1, num_dof=1, value=np.array([10.]))
rotor_3_ffd_block.add_translation_w(name='rotor_3_translation_y', order=1, num_dof=1, value=np.array([2.]))

rotor_4_geometry_primitives = rotor_4.get_geometry_primitives()
rotor_4_ffd_bspline_volume = create_cartesian_enclosure_volume(rotor_4_geometry_primitives, num_control_points=(2, 2, 2), order=(2,2,2), xyz_to_uvw_indices=(2,0,1))
rotor_4_ffd_block = SRBGFFDBlock(name='rotor_4_ffd_block', primitive=rotor_4_ffd_bspline_volume, embedded_entities=rotor_4_geometry_primitives)
rotor_4_ffd_block.add_scale_v(name='rotor_4_x_scaling', order=1, num_dof=1, value=np.array([2.]))
rotor_4_ffd_block.add_scale_w(name='rotor_4_y_scaling', order=1, num_dof=1, value=np.array([2.]))
rotor_4_ffd_block.add_translation_v(name='rotor_4_translation_x', order=1, num_dof=1, value=np.array([10.]))
rotor_4_ffd_block.add_translation_w(name='rotor_4_translation_y', order=1, num_dof=1, value=np.array([2.]))

rotor_5_geometry_primitives = rotor_5.get_geometry_primitives()
rotor_5_ffd_bspline_volume = create_cartesian_enclosure_volume(rotor_5_geometry_primitives, num_control_points=(2, 2, 2), order=(2,2,2), xyz_to_uvw_indices=(2,0,1))
rotor_5_ffd_block = SRBGFFDBlock(name='rotor_5_ffd_block', primitive=rotor_5_ffd_bspline_volume, embedded_entities=rotor_5_geometry_primitives)
rotor_5_ffd_block.add_scale_v(name='rotor_5_x_scaling', order=1, num_dof=1, value=np.array([2.]))
rotor_5_ffd_block.add_scale_w(name='rotor_5_y_scaling', order=1, num_dof=1, value=np.array([2.]))
rotor_5_ffd_block.add_translation_v(name='rotor_5_translation_x', order=1, num_dof=1, value=np.array([10.]))
rotor_5_ffd_block.add_translation_w(name='rotor_5_translation_y', order=1, num_dof=1, value=np.array([2.]))

rotor_6_geometry_primitives = rotor_6.get_geometry_primitives()
rotor_6_ffd_bspline_volume = create_cartesian_enclosure_volume(rotor_6_geometry_primitives, num_control_points=(2, 2, 2), order=(2,2,2), xyz_to_uvw_indices=(2,0,1))
rotor_6_ffd_block = SRBGFFDBlock(name='rotor_6_ffd_block', primitive=rotor_6_ffd_bspline_volume, embedded_entities=rotor_6_geometry_primitives)
rotor_6_ffd_block.add_scale_v(name='rotor_6_x_scaling', order=1, num_dof=1, value=np.array([2.]))
rotor_6_ffd_block.add_scale_w(name='rotor_6_y_scaling', order=1, num_dof=1, value=np.array([2.]))
rotor_6_ffd_block.add_translation_v(name='rotor_6_translation_x', order=1, num_dof=1, value=np.array([10.]))
rotor_6_ffd_block.add_translation_w(name='rotor_6_translation_y', order=1, num_dof=1, value=np.array([2.]))

rotor_7_geometry_primitives = rotor_7.get_geometry_primitives()
rotor_7_ffd_bspline_volume = create_cartesian_enclosure_volume(rotor_7_geometry_primitives, num_control_points=(2, 2, 2), order=(2,2,2), xyz_to_uvw_indices=(2,0,1))
rotor_7_ffd_block = SRBGFFDBlock(name='rotor_7_ffd_block', primitive=rotor_7_ffd_bspline_volume, embedded_entities=rotor_7_geometry_primitives)
rotor_7_ffd_block.add_scale_v(name='rotor_7_x_scaling', order=1, num_dof=1, value=np.array([2.]))
rotor_7_ffd_block.add_scale_w(name='rotor_7_y_scaling', order=1, num_dof=1, value=np.array([2.]))
rotor_7_ffd_block.add_translation_v(name='rotor_7_translation_x', order=1, num_dof=1, value=np.array([10.]))
rotor_7_ffd_block.add_translation_w(name='rotor_7_translation_y', order=1, num_dof=1, value=np.array([2.]))

rotor_8_geometry_primitives = rotor_8.get_geometry_primitives()
rotor_8_ffd_bspline_volume = create_cartesian_enclosure_volume(rotor_8_geometry_primitives, num_control_points=(2, 2, 2), order=(2,2,2), xyz_to_uvw_indices=(2,0,1))
rotor_8_ffd_block = SRBGFFDBlock(name='rotor_8_ffd_block', primitive=rotor_8_ffd_bspline_volume, embedded_entities=rotor_8_geometry_primitives)
rotor_8_ffd_block.add_scale_v(name='rotor_8_x_scaling', order=1, num_dof=1, value=np.array([2.]))
rotor_8_ffd_block.add_scale_w(name='rotor_8_y_scaling', order=1, num_dof=1, value=np.array([2.]))
rotor_8_ffd_block.add_translation_v(name='rotor_8_translation_x', order=1, num_dof=1, value=np.array([10.]))
rotor_8_ffd_block.add_translation_w(name='rotor_8_translation_y', order=1, num_dof=1, value=np.array([2.]))

# plotting_elements = rotor_1_ffd_block.plot_sections(plot_embedded_entities=False, show=False)
# plotting_elements = rotor_2_ffd_block.plot_sections(plot_embedded_entities=False, additional_plotting_elements=plotting_elements, show=False)
# plotting_elements = rotor_3_ffd_block.plot_sections(plot_embedded_entities=False, additional_plotting_elements=plotting_elements, show=False)
# plotting_elements = rotor_4_ffd_block.plot_sections(plot_embedded_entities=False, additional_plotting_elements=plotting_elements, show=False)
# plotting_elements = rotor_5_ffd_block.plot_sections(plot_embedded_entities=False, additional_plotting_elements=plotting_elements, show=False)
# plotting_elements = rotor_6_ffd_block.plot_sections(plot_embedded_entities=False, additional_plotting_elements=plotting_elements, show=False)
# plotting_elements = rotor_7_ffd_block.plot_sections(plot_embedded_entities=False, additional_plotting_elements=plotting_elements, show=False)
# plotting_elements = rotor_8_ffd_block.plot_sections(plot_embedded_entities=False, additional_plotting_elements=plotting_elements, show=False)
# spatial_rep.plot(additional_plotting_elements=plotting_elements)


from caddee.caddee_core.system_parameterization.free_form_deformation.ffd_set import SRBGFFDSet
ffd_set = SRBGFFDSet(name='ffd_set', ffd_blocks={wing_ffd_block.name : wing_ffd_block, horizontal_stabilizer_ffd_block.name : horizontal_stabilizer_ffd_block, \
                                                 rotor_1_ffd_block.name: rotor_1_ffd_block, rotor_2_ffd_block.name: rotor_2_ffd_block, \
                                                 rotor_3_ffd_block.name: rotor_3_ffd_block, rotor_4_ffd_block.name: rotor_4_ffd_block, \
                                                 rotor_5_ffd_block.name: rotor_5_ffd_block, rotor_6_ffd_block.name: rotor_6_ffd_block, \
                                                 rotor_7_ffd_block.name: rotor_7_ffd_block, rotor_8_ffd_block.name: rotor_8_ffd_block})

system_parameterization.add_geometry_parameterization(ffd_set)
system_parameterization.setup()

system_representation_model = system_representation.assemble_csdl()
system_parameterization_model = system_parameterization.assemble_csdl()

my_model = csdl.Model()
my_model.add(system_parameterization_model, 'system_parameterization')
my_model.add(system_representation_model, 'system_representation')
# my_model.add(solver_model, 'solver_model_name')

# initial_guess_linear_taper = np.array([0., 2., 0.])
# my_model.create_input('linear_taper', val=initial_guess_linear_taper)
# my_model.add_design_variable('linear_taper')
# my_model.add_objective('wing_camber_surface')
# my_model.add_objective('chord_distribution')

# sim = Simulator(my_model)
# sim.run()
# sim.compute_total_derivatives()
# sim.check_totals()

# affine_section_properties = ffd_set.evaluate_affine_section_properties(prescribed_affine_dof=np.append(initial_guess_linear_taper, np.zeros(4,)))
# affine_section_properties = ffd_set.evaluate_affine_section_properties()
# rotational_section_properties = ffd_set.evaluate_rotational_section_properties()
# affine_deformed_ffd_control_points = ffd_set.evaluate_affine_block_deformations()
# rotated_ffd_control_points = ffd_set.evaluate_rotational_block_deformations()
# ffd_control_points = ffd_set.evaluate_control_points()
# ffd_embedded_entities = ffd_set.evaluate_embedded_entities()

# updated_primitives_names = wing.primitive_names.copy()
# updated_primitives_names.extend(horizontal_stabilizer.primitive_names.copy())
# updated_primitives_names.extend(rotor_1.primitive_names.copy())
# updated_primitives_names.extend(rotor_2.primitive_names.copy())
# updated_primitives_names.extend(rotor_3.primitive_names.copy())
# updated_primitives_names.extend(rotor_4.primitive_names.copy())
# updated_primitives_names.extend(rotor_5.primitive_names.copy())
# updated_primitives_names.extend(rotor_6.primitive_names.copy())
# updated_primitives_names.extend(rotor_7.primitive_names.copy())
# updated_primitives_names.extend(rotor_8.primitive_names.copy())
# spatial_rep.update(ffd_embedded_entities, updated_primitives_names)

# wing_camber_surface.evaluate(spatial_rep.control_points['geometry'])
# horizontal_stabilizer_camber_surface.evaluate(spatial_rep.control_points['geometry'])

# spatial_rep.plot()

# print('wingspan', sim['wingspan'])
# print("Python and CSDL difference: wingspan", np.linalg.norm(wingspan.value - sim['wingspan']))
# print('wing root chord', sim['wing_root_chord'])
# print("Python and CSDL difference: wing root chord", np.linalg.norm(root_chord.value - sim['wing_root_chord']))

# wing_camber_surface_csdl = sim['wing_camber_surface']
# horizontal_stabilizer_camber_surface_csdl = sim['horizontal_stabilizer_camber_surface']
# print("Python and CSDL difference: wing camber surface", np.linalg.norm(wing_camber_surface_csdl - wing_camber_surface.value))
# print("Python and CSDL difference: horizontal stabilizer camber surface", np.linalg.norm(horizontal_stabilizer_camber_surface_csdl - horizontal_stabilizer_camber_surface.value))

# spatial_rep.plot_meshes([wing_camber_surface, horizontal_stabilizer_camber_surface], mesh_plot_types=['wireframe'], mesh_opacity=1.)
# spatial_rep.plot_meshes([wing_camber_surface_csdl, horizontal_stabilizer_camber_surface_csdl], mesh_plot_types=['wireframe'], mesh_opacity=1.)

# taper_ratio = 0.45
# wing_area = np.linspace()
# wingspan = np.linspace()
num_iterations = 50
root_chord_dof = np.linspace(0., 0.05, num_iterations)
tip_chord_dof = np.linspace(0., 0.05, num_iterations)
translation_dof_left = -np.linspace(0., 5., num_iterations)
tail_moment_arm = np.linspace(0., 0.5, num_iterations)
radius_shrink = np.linspace(0., -0.2, num_iterations)
translation_y_outer = np.linspace(0., -4., num_iterations)
translation_y_inner = np.linspace(0., -2., num_iterations)
translation_x_front = np.linspace(0., 1., num_iterations)
translation_x_back = np.linspace(0., -1., num_iterations)

affine_dof = np.zeros((num_iterations, 6+4*8))
affine_dof[:,0] = tip_chord_dof
affine_dof[:,1] = root_chord_dof
affine_dof[:,2] = tip_chord_dof
affine_dof[:,3] = translation_dof_left
affine_dof[:,4] = -translation_dof_left
affine_dof[:,5] = tail_moment_arm
for i in range(8):
    affine_dof[:,6+4*i] = radius_shrink
    affine_dof[:,6+4*i+1] = radius_shrink
affine_dof[:,9] = -translation_y_outer
affine_dof[:,13] = -translation_y_outer
affine_dof[:,17] = -translation_y_inner
affine_dof[:,21] = -translation_y_inner
affine_dof[:,25] = translation_y_inner
affine_dof[:,29] = translation_y_inner
affine_dof[:,33] = translation_y_outer
affine_dof[:,37] = translation_y_outer

affine_dof[:,8] = translation_x_front
affine_dof[:,12] = translation_x_back
affine_dof[:,16] = translation_x_front
affine_dof[:,20] = translation_x_back
affine_dof[:,24] = translation_x_front
affine_dof[:,28] = translation_x_back
affine_dof[:,32] = translation_x_front
affine_dof[:,36] = translation_x_back

initial_geometry_plot = spatial_rep.plot_meshes([wing_camber_surface, horizontal_stabilizer_camber_surface], primitives_opacity=0.20, mesh_opacity=0.15)

affine_section_properties = ffd_set.evaluate_affine_section_properties(prescribed_affine_dof=affine_dof[-1,:])
rotational_section_properties = ffd_set.evaluate_rotational_section_properties()
affine_deformed_ffd_control_points = ffd_set.evaluate_affine_block_deformations()
rotated_ffd_control_points = ffd_set.evaluate_rotational_block_deformations()
ffd_control_points = ffd_set.evaluate_control_points()
ffd_embedded_entities = ffd_set.evaluate_embedded_entities()
updated_primitives_names = list(ffd_set.embedded_entities.keys())
spatial_rep.update(ffd_embedded_entities, updated_primitives_names)
wing_camber_surface.evaluate(spatial_rep.control_points['geometry'])
horizontal_stabilizer_camber_surface.evaluate(spatial_rep.control_points['geometry'])
combined_geometry_plot = spatial_rep.plot_meshes([wing_camber_surface, horizontal_stabilizer_camber_surface], primitives_opacity=0.9, mesh_opacity=1., additional_plotting_elements=initial_geometry_plot)
camera = dict(
    position=(-35, -30, 35),
    focal_point=(15, 0, 5),
    viewup=(0, 0, 1),
    distance=0,
)
plotter = vedo.Plotter(size=(3200,1000))
plotter.show(combined_geometry_plot, camera=camera)

from vedo import Video
video = Video('test.mp4', duration=None, fps=10, backend="cv")

for iter in range(num_iterations):
    affine_section_properties = ffd_set.evaluate_affine_section_properties(prescribed_affine_dof=affine_dof[iter,:])
    rotational_section_properties = ffd_set.evaluate_rotational_section_properties()
    affine_deformed_ffd_control_points = ffd_set.evaluate_affine_block_deformations()
    rotated_ffd_control_points = ffd_set.evaluate_rotational_block_deformations()
    ffd_control_points = ffd_set.evaluate_control_points()
    ffd_embedded_entities = ffd_set.evaluate_embedded_entities()
    updated_primitives_names = list(ffd_set.embedded_entities.keys())
    spatial_rep.update(ffd_embedded_entities, updated_primitives_names)
    wing_camber_surface.evaluate(spatial_rep.control_points['geometry'])
    horizontal_stabilizer_camber_surface.evaluate(spatial_rep.control_points['geometry'])

    iteration_plot = spatial_rep.plot_meshes([wing_camber_surface, horizontal_stabilizer_camber_surface], primitives_opacity=0.75, 
                     mesh_plot_types=['wireframe'], mesh_opacity=1., additional_plotting_elements=initial_geometry_plot, show=False)
    
    plotter = vedo.Plotter(size=(3200,1000),offscreen=False)
    plotter.show(iteration_plot, camera=camera, interactive=False)

    video.add_frame()

video.close()




# index = 0
# tref_e,tref_t = 0,0
# for t in np.arange(0, n, 1):
#     m1.rotate_y(-1*np.rad2deg(tie[index] - tref_e))
#     m2.rotate_y(-1*np.rad2deg(tit[index] - tref_t))
#     tref_e = tie[index]
#     tref_t = tit[index]
#     m1.pos(xie[index], 0, 1.5*hie[index])
#     m2.pos(xit[index], 0, 1.5*hit[index])
#     index += 1

#     plt.show(m1,m2, __doc__, axes=0, viewup="z",camera=cam,rate=2000,bg=[240,248,255])
#     video.add_frame()

# #video.action(cameras=cam)
# video.close() 
# plt.interactive().close()
