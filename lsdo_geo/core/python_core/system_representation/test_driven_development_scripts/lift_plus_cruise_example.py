import caddee as cd 
import numpy as np
import array_mapper as am

# evtol = cd.CADDEE()
# from caddee.caddee_core.caddee import CADDEE
# evtol = CADDEE()
# evtol.set_units('SI')

# evtol.system_representation = system_representation = cd.SystemRepresentation()
# evtol.system_paramaterization = system_parameterization = cd.SystemParameterization()
# evtol.system_model = system_model = cd.SystemModel()
from caddee.caddee_core.system_representation.system_representation import SystemRepresentation
system_representation = SystemRepresentation()
from caddee.caddee_core.system_parameterization.system_parameterization import SystemParameterization
system_parameterization = SystemParameterization()

# Geometry + Material properties
file_path = 'models/stp/'
# geo = system_representation.import_geometry(file_name=file_path+'rect_wing.stp')
# geo = system_representation.import_geometry(file_name=file_path+'c172_wings_and_nacelles_3.stp')
# geo = system_representation.import_geometry(file_name=file_path+'no_rotor_no_people.stp')
geo = system_representation.import_geometry(file_name=file_path+'lift_plus_cruise_final_3.stp')

geo.plot(plot_types=['mesh'])
# geo.plot(plot_type='wireframe')
# geo.plot(plot_type='point_cloud')

# Create Components
from caddee.caddee_core.system_representation.component.component import LiftingSurface, Component
wing = LiftingSurface(name='wing', geometry=geo, geometry_primitive_names=['Wing'])  # TODO add material arguments
horizontal_stabilizer = LiftingSurface(name='horizontal_stabilizer', geometry=geo, geometry_primitive_names=['Tail_1'])
rotor_hub_1 = Component(name='rotor_1_hub', geometry=geo, geometry_primitive_names=['Rotor_1_Hub'])
rotor_hub_2 = Component(name='rotor_2_hub', geometry=geo, geometry_primitive_names=['Rotor_2_Hub'])
rotor_hub_3 = Component(name='rotor_3_hub', geometry=geo, geometry_primitive_names=['Rotor_3_Hub'])
rotor_hub_4 = Component(name='rotor_4_hub', geometry=geo, geometry_primitive_names=['Rotor_4_Hub'])
rotor_hub_5 = Component(name='rotor_5_hub', geometry=geo, geometry_primitive_names=['Rotor_5_Hub'])
rotor_hub_6 = Component(name='rotor_6_hub', geometry=geo, geometry_primitive_names=['Rotor_6_Hub'])
rotor_hub_7 = Component(name='rotor_7_hub', geometry=geo, geometry_primitive_names=['Rotor_7_Hub'])
rotor_hub_8 = Component(name='rotor_8_hub', geometry=geo, geometry_primitive_names=['Rotor_8_Hub'])
rotor_hub_9 = Component(name='rotor_9_hub', geometry=geo, geometry_primitive_names=['Rotor_9_Hub'])
rotor_1_disk = Component(name='rotor_1_disk', geometry=geo, geometry_primitive_names=['Rotor_1_disk'])
# horizontal_stabilizer.plot()
system_representation.add_component(wing)
system_representation.add_component(horizontal_stabilizer)

# from caddee.caddee_core.system_representation.component.component import Rotor
# rotor1 = Rotor(name='rotor1', geometry=geo, geometry_primitive_names=['Rotor, 0'])
# system_representation.add_component(rotor1)

# Key locations identificatiions (common variables that are used repeatedly)
starboard_tip_quarter_chord = wing.project(np.array([12., 26., 10.])) # returns a MappedArray
port_tip_quarter_chord = wing.project(np.array([12., -26., 10.])) # returns a MappedArray
wing_root_leading_edge = wing.project(np.array([8., 0., 7.5]))
wing_root_trailing_edge = wing.project(np.array([15., 0., 7.5]))
wing_starboard_tip_leading_edge = wing.project(np.array([8., 26., 7.5]))
wing_starboard_tip_trailing_edge = wing.project(np.array([15., 24.5, 7.5]))
wing_port_tip_leading_edge = wing.project(np.array([8., -26., 7.5]))
wing_port_tip_trailing_edge = wing.project(np.array([15., -24.5, 7.5]))
tail_root_leading_edge = horizontal_stabilizer.project(np.array([27., 0., 6.]))
tail_root_trailing_edge = horizontal_stabilizer.project(np.array([31.5, 0., 6.]))
tail_starboard_tip_leading_edge = horizontal_stabilizer.project(np.array([27., 6.75, 6.]))
tail_starboard_tip_trailing_edge = horizontal_stabilizer.project(np.array([31.5, 6.75, 6.]))
tail_port_tip_leading_edge = horizontal_stabilizer.project(np.array([27., -6.75, 6.]))
tail_port_tip_trailing_edge = horizontal_stabilizer.project(np.array([31.5, -6.75, 6.]))
rotor_1_starboard_side = rotor_1_disk.project(np.array([5.07, -10.75, 6.73]))
rotor_1_port_side = rotor_1_disk.project(np.array([5.07, -25.75, 6.73]))

wing_span_vector = starboard_tip_quarter_chord - port_tip_quarter_chord
wing_root_chord_vector = wing_root_trailing_edge - wing_root_leading_edge
wing_starboard_tip_chord_vector = wing_starboard_tip_trailing_edge - wing_starboard_tip_leading_edge
wing_port_tip_chord_vector = wing_port_tip_trailing_edge - wing_port_tip_leading_edge

plotting_points = [starboard_tip_quarter_chord.value, port_tip_quarter_chord.value, wing_root_leading_edge.value, wing_root_trailing_edge.value, 
                wing_starboard_tip_leading_edge.value, wing_starboard_tip_trailing_edge.value, wing_port_tip_leading_edge.value, 
                wing_port_tip_trailing_edge.value, tail_root_leading_edge.value, tail_root_trailing_edge.value, tail_starboard_tip_leading_edge.value,
                tail_starboard_tip_trailing_edge.value, tail_port_tip_leading_edge.value, tail_port_tip_trailing_edge.value,
                rotor_1_starboard_side.value, rotor_1_port_side.value]
geo.plot_meshes(plotting_points, mesh_plot_types=['point_cloud'], primitives_opacity=0.25)

''' TODO: Finish addressing code below starting from this point '''
# TODO Redo actuations to use the kinematic optimization.
# # Actuator
# # # Example: tilt-wing
# rotation_axis = starboard_tip_quarter_chord - port_tip_quarter_chord
# rotation_origin = wing.project_points([1., 0., 2.])
# tilt_wing_actuator = cd.Actuator(actuating_components=[wing, rotor1], rotation_origin=rotation_origin, rotation_axis=rotation_axis)
# tilt_wing_actuator = cd.Actuator(actuating_components=['wing', 'rotor1'], rotation_origin=rotation_origin, rotation_axis=rotation_axis)
# system_parameterization.add_actuator(tilt_wing_actuator)

# TODO Go back and implement Joints after the kinematics optimization (make MBD optimization (energy minimization probably))
# Joints
# Example: rotor mounted to boom
# rotor_to_boom_connection_point = np.array([6., 18., 2.])
# rotor_to_boom_connection_point_on_rotor = rotor1.project(rotor_to_boom_connection_point)
# rotor_to_boom_connection_point_on_boom = boom.project(rotor_to_boom_connection_point)
# rotor_boom_joint = cd.Joint()


# Note: Powertrain and material definitions have been skipped for the sake of time in this iteration.


# Parameterization
from caddee.caddee_core.system_parameterization.free_form_deformation.ffd_functions import create_cartesian_enclosure_volume
from caddee.caddee_core.system_parameterization.free_form_deformation.ffd_block import SRBGFFDBlock

wing_ffd_bspline_volume = create_cartesian_enclosure_volume(wing.geometry_primitives, num_control_points=(2, 10, 2), order=(2,4,2))
wing_ffd_block = SRBGFFDBlock(name='wing_ffd_block', primitive=wing_ffd_bspline_volume, embedded_entities=wing.geometry_primitives)
wing_ffd_block.plot(plot_embedded_entites=True)

# wing_ffd_block.add_scale_y(num_dof=3, parameter_degree=1, value=np.array([0., 1., 0.]), cost_factor=1.)
# wing_ffd_block.add_rotation_x(num_dof=10, parameter_degree=3, value=np.array([0., 1., 0.]))

# # rotor1_translator = TranslatingFFDBlock(component=rotor1)
# # horizontal_stabilizer_translator = TranslatingFFDBlock(component=horizontal_stabilizer)
# # horizontal_stabilizer_translator.set_value(np.array([10., 0., 0.]))

# # motor_ffd_block = PolarFFDBlock(component='motor1')

# ''' Note: Do we want to generalize ffd to design_parameterization so we are not tied to ffd? '''
# system_parameterization.add_ffd(wing_ffd_block)
# # system_parameterization.add_ffd(rotor1_translator)
# # system_parameterization.add_ffd(horizontal_stabilizer_translator)
# # system_parameterization.add_ffd(motor_ffd_block)


# Geometric inputs, outputs, and constraints
''' Note: Where to add geometric inputs, outputs, and constraints? '''
''' How do we want to add parameters? We can try to use the convention from the spreadsheet, but there are too many possibilities? '''
''' Fundamental issue: In most solvers, you know exactly what the inputs need to be. For geometry, we don't. '''
# wing_to_tail_vector =  horizontal_stabilizer.project(np.array([0., 0., 0.])) - wing_root_trailing_edge

# spatial_representation.add_variable('wing_root_chord', computed_upstream=False, dv=True, quantity=MagnitudeCalculation(wing_root_chord_vector))   # geometric design variable
# spatial_representation.add_variable('wing_starboard_tip_chord', computed_upstream=True, connection_name='wing_tip_chord', quantity=MagnitudeCalculation(wing_starboard_tip_chord_vector))   # geometric input
# spatial_representation.add_variable('wing_port_tip_chord', computed_upstream=True, connection_name='wing_tip_chord', quantity=MagnitudeCalculation(wing_port_tip_chord_vector))   # geometric input
# # Note: This will throw an error because CSDL does not allow for a single variable to conenct to multiple (wing_tip_chord --> wing_starboard_tip_chord and wing_tip_chord --> wing_port_tip_chord)

# spatial_representation.add_variable('wing_to_tail_distance', computed_upstream=False, val=10, quantity=MagnitudeCalculation(wing_port_tip_chord_vector))    # Geometric constraint (very similar to geometric input)

# spatial_representation.add_variable('wing_span', output_name='wingspan', quantity=MagnitudeCalculation(wing_span_vector))   # geometric output


# Mesh definitions
# rotor1_thrust_vector_origin = rotor1.project(np.array([5., 18., 0.]))
# rotor1_thurst_vector_tip = rotor1.project(np.array([5., 18., 2.]))
# rotor1_thurst_direction_vector = rotor1_thurst_vector_tip - rotor1_thrust_vector_origin
# rotor1_bem_mesh = cd.BemMesh(rotor1_thurst_direction_vector)    # Note: not immediately sure if this object should be a part of CADDEE (cd.)

num_spanwise_vlm = 21
num_chordwise_vlm = 4
leading_edge = wing.project(np.linspace(np.array([8., -26., 7.5]), np.array([8., 26., 7.5]), num_spanwise_vlm), direction=np.array([0., 0., -1.]))  # returns MappedArray
trailing_edge = wing.project(np.linspace(np.array([15., -26., 7.5]), np.array([15., 26., 7.5]), num_spanwise_vlm), direction=np.array([0., 0., -1.]))   # returns MappedArray
chord_surface = am.linspace(leading_edge, trailing_edge, num_chordwise_vlm)
wing_upper_surface_wireframe = wing.project(chord_surface.value + np.array([0., 0., 1.]), direction=np.array([0., 0., -1.]), grid_search_n=75)
wing_lower_surface_wireframe = wing.project(chord_surface.value - np.array([0., 0., 1.]), direction=np.array([0., 0., 1.]), grid_search_n=75)
wing_camber_surface = am.linspace(wing_upper_surface_wireframe, wing_lower_surface_wireframe, 1) # this linspace will return average when n=1

num_spanwise_vlm = 11
num_chordwise_vlm = 3
leading_edge = horizontal_stabilizer.project(np.linspace(np.array([27., -6.5, 6.]), np.array([27., 6.75, 6.]), num_spanwise_vlm), direction=np.array([0., 0., -1.]))  # returns MappedArray
trailing_edge = horizontal_stabilizer.project(np.linspace(np.array([31.5, -6.5, 6.]), np.array([31.5, 6.75, 6.]), num_spanwise_vlm), direction=np.array([0., 0., -1.]))   # returns MappedArray
chord_surface = am.linspace(leading_edge, trailing_edge, num_chordwise_vlm)
horizontal_stabilizer_upper_surface_wireframe = horizontal_stabilizer.project(chord_surface.value + np.array([0., 0., 1.]), direction=np.array([0., 0., -1.]), grid_search_n=25)
horizontal_stabilizer_lower_surface_wireframe = horizontal_stabilizer.project(chord_surface.value - np.array([0., 0., 1.]), direction=np.array([0., 0., 1.]), grid_search_n=25)
horizontal_stabilizer_camber_surface = am.linspace(horizontal_stabilizer_upper_surface_wireframe, horizontal_stabilizer_lower_surface_wireframe, 1) # this linspace will return average when n=1


rotor_hub_1_bot_point = rotor_hub_1.project(np.array([5.07, -18.75, 6.73]))
rotor_hub_2_bot_point = rotor_hub_2.project(np.array([19.02, -18.75, 9.01]))
rotor_hub_3_bot_point = rotor_hub_3.project(np.array([4.63, -8.45, 7.04]))
rotor_hub_4_bot_point = rotor_hub_4.project(np.array([18.76, -8.45, 9.3]))
rotor_hub_5_bot_point = rotor_hub_5.project(np.array([4.63, 8.45, 7.04]))
rotor_hub_6_bot_point = rotor_hub_6.project(np.array([18.76, 8.45, 9.3]))
rotor_hub_7_bot_point = rotor_hub_7.project(np.array([5.07, 18.75, 6.73]))
rotor_hub_8_bot_point = rotor_hub_8.project(np.array([19.2, 18.75, 9.01]))
rotor_hub_9_bot_point = rotor_hub_9.project(np.array([31.94, 0., 7.79]))
rotor_hub_1_top_point = rotor_hub_1.project(np.array([5.07, -18.75, 7.23]))
rotor_hub_2_top_point = rotor_hub_2.project(np.array([19.02, -18.75, 9.51]))
rotor_hub_3_top_point = rotor_hub_3.project(np.array([4.63, -8.65, 7.54]))
rotor_hub_4_top_point = rotor_hub_4.project(np.array([18.76, -8.65, 9.8]))
rotor_hub_5_top_point = rotor_hub_5.project(np.array([4.63, 8.65, 7.54]))
rotor_hub_6_top_point = rotor_hub_6.project(np.array([18.76, 8.65, 9.8]))
rotor_hub_7_top_point = rotor_hub_7.project(np.array([5.07, 18.75, 7.23]))
rotor_hub_8_top_point = rotor_hub_8.project(np.array([19.2, 18.75, 9.51]))
rotor_hub_9_top_point = rotor_hub_9.project(np.array([32.44, 0., 7.79]))
rotor_hub_1_thrust_vector_stop = am.linear_combination(rotor_hub_1_bot_point, rotor_hub_1_top_point, num_steps=1, start_weights=[-5.], stop_weights=[6.])
rotor_hub_2_thrust_vector_stop = am.linear_combination(rotor_hub_2_bot_point, rotor_hub_2_top_point, num_steps=1, start_weights=[-5.], stop_weights=[6.])
rotor_hub_3_thrust_vector_stop = am.linear_combination(rotor_hub_3_bot_point, rotor_hub_3_top_point, num_steps=1, start_weights=[-5.], stop_weights=[6.])
rotor_hub_4_thrust_vector_stop = am.linear_combination(rotor_hub_4_bot_point, rotor_hub_4_top_point, num_steps=1, start_weights=[-5.], stop_weights=[6.])
rotor_hub_5_thrust_vector_stop = am.linear_combination(rotor_hub_5_bot_point, rotor_hub_5_top_point, num_steps=1, start_weights=[-5.], stop_weights=[6.])
rotor_hub_6_thrust_vector_stop = am.linear_combination(rotor_hub_6_bot_point, rotor_hub_6_top_point, num_steps=1, start_weights=[-5.], stop_weights=[6.])
rotor_hub_7_thrust_vector_stop = am.linear_combination(rotor_hub_7_bot_point, rotor_hub_7_top_point, num_steps=1, start_weights=[-5.], stop_weights=[6.])
rotor_hub_8_thrust_vector_stop = am.linear_combination(rotor_hub_8_bot_point, rotor_hub_8_top_point, num_steps=1, start_weights=[-5.], stop_weights=[6.])
rotor_hub_9_thrust_vector_stop = am.linear_combination(rotor_hub_9_bot_point, rotor_hub_9_top_point, num_steps=1, start_weights=[-5.], stop_weights=[6.])
rotor_1_thrust_vector = rotor_hub_1_thrust_vector_stop - rotor_hub_1_top_point
rotor_2_thrust_vector = rotor_hub_2_thrust_vector_stop - rotor_hub_2_top_point
rotor_3_thrust_vector = rotor_hub_3_thrust_vector_stop - rotor_hub_3_top_point
rotor_4_thrust_vector = rotor_hub_4_thrust_vector_stop - rotor_hub_4_top_point
rotor_5_thrust_vector = rotor_hub_5_thrust_vector_stop - rotor_hub_5_top_point
rotor_6_thrust_vector = rotor_hub_6_thrust_vector_stop - rotor_hub_6_top_point
rotor_7_thrust_vector = rotor_hub_7_thrust_vector_stop - rotor_hub_7_top_point
rotor_8_thrust_vector = rotor_hub_8_thrust_vector_stop - rotor_hub_8_top_point
rotor_9_thrust_vector = rotor_hub_9_thrust_vector_stop - rotor_hub_9_top_point

plotting_points = [rotor_hub_1_bot_point.value, rotor_hub_2_bot_point.value, rotor_hub_3_bot_point.value, rotor_hub_4_bot_point.value, 
                rotor_hub_5_bot_point.value, rotor_hub_6_bot_point.value, rotor_hub_7_bot_point.value, 
                rotor_hub_8_bot_point.value, rotor_hub_9_bot_point.value,
                rotor_hub_1_top_point.value, rotor_hub_2_top_point.value, rotor_hub_3_top_point.value, rotor_hub_4_top_point.value, 
                rotor_hub_5_top_point.value, rotor_hub_6_top_point.value, rotor_hub_7_top_point.value, 
                rotor_hub_8_top_point.value, rotor_hub_9_top_point.value, (rotor_hub_1_top_point.value, rotor_hub_1_thrust_vector_stop.value-rotor_hub_1_top_point.value)]
geo.plot_meshes(plotting_points, mesh_plot_types=['point_cloud'], primitives_opacity=0.8)

plotting_meshes = [wing_camber_surface.value, horizontal_stabilizer_camber_surface.value, (rotor_hub_1_top_point.value, rotor_1_thrust_vector.value),
        (rotor_hub_2_top_point.value, rotor_2_thrust_vector.value), (rotor_hub_3_top_point.value, rotor_3_thrust_vector.value),
        (rotor_hub_4_top_point.value, rotor_4_thrust_vector.value), (rotor_hub_5_top_point.value, rotor_5_thrust_vector.value),
        (rotor_hub_6_top_point.value, rotor_6_thrust_vector.value), (rotor_hub_7_top_point.value, rotor_7_thrust_vector.value),
        (rotor_hub_8_top_point.value, rotor_8_thrust_vector.value), (rotor_hub_9_top_point.value, rotor_9_thrust_vector.value)]

geo.plot_meshes(plotting_meshes, mesh_plot_types=['wireframe'])

plotting_elements = geo.plot_meshes(plotting_meshes, mesh_plot_types=['wireframe'], primitives_opacity=0.5, show=False)
wing_ffd_block.plot(plot_embedded_entites=False, opacity=0.3, additional_plotting_elements=plotting_elements)
# geo.plot_meshes([wing_camber_surface.value], mesh_plot_type='wireframe', mesh_opacity=1.)

wing_vlm_mesh = VLMMesh(meshes=[wing_camber_surface, horizontal_stabilizer_camber_surface])


# Define solvers: After defining solver meshes we define solvers 
# bem = BEM(meshes=[rotor1_bem_mesh])
vlm = VLM(meshes=[wing_vlm_mesh])
# imga = IMGA(imga_mesh)
# motor_model = MotorSolverTC1(some_motor_mesh) # this should still take in a notion of a mesh in order to be associated with a node in the powertrain

# TODO Will come back to SISR after January. Note: May need some basic data transfer for just forward passes though.
# # Define coupled-analyses
# ''' Note: Need more time to iron out API and deisgn of this. A particularly challenging case for API is 3-way coupling (bem-wake_solver-vlm coupling) '''
# a_s_transfer_mesh = cd.TransferMesh(component=wing, mesh_attributes='place_holder')
# a_s_coupling = CoupledGroup(transfer_mesh=a_s_transfer_mesh)
# a_s_coupling.add_models(solvers=[vlm, imga])
# p = a_s_coupling.vlm.get_output()    # pressures
# u = a_s_coupling.imga.get_output()   # displacements
# a_s_coupling.vlm.set_input(u)
# a_s_coupling.imga.set_input(p)
# a_s_coupling.nonlinear_solver = NLSolvers.Newton(
#     max_iter=50,
#     tol=1e-5,
# )

