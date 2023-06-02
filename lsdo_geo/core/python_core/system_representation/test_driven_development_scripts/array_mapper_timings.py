import caddee as cd 
import numpy as np
import array_mapper as am

# evtol = cd.CADDEE()
# from caddee.caddee_core.caddee import CADDEE
# evtol = CADDEE()
# evtol.set_units('SI')

from caddee.caddee_core.system_representation.system_representation import SystemRepresentation
system_representation = SystemRepresentation()
from caddee.caddee_core.system_parameterization.system_parameterization import SystemParameterization
system_parameterization = SystemParameterization(system_representation=system_representation)

file_path = 'models/stp/'
spatial_rep = system_representation.spatial_representation
spatial_rep.import_file(file_name=file_path+'lift_plus_cruise_final_3.stp')

spatial_rep.plot(plot_types=['mesh'])

# Create Components
from caddee.caddee_core.system_representation.component.component import LiftingSurface
wing_primitive_names = list(spatial_rep.get_primitives(search_names=['Wing']).keys())
wing = LiftingSurface(name='wing', spatial_representation=spatial_rep, primitive_names=wing_primitive_names)  # TODO add material arguments
tail_primitive_names = list(spatial_rep.get_primitives(search_names=['Tail_1']).keys())
horizontal_stabilizer = LiftingSurface(name='tail', spatial_representation=spatial_rep, primitive_names=tail_primitive_names)  # TODO add material arguments
system_representation.add_component(wing)
system_representation.add_component(horizontal_stabilizer)

# # Key locations identificatiions (common variables that are used repeatedly)
# starboard_tip_quarter_chord = wing.project(np.array([12., 26., 10.])) # returns a MappedArray
# port_tip_quarter_chord = wing.project(np.array([12., -26., 10.])) # returns a MappedArray
# wing_root_leading_edge = wing.project(np.array([8., 0., 7.5]))
# wing_root_trailing_edge = wing.project(np.array([15., 0., 7.5]))
# wing_starboard_tip_leading_edge = wing.project(np.array([8., 26., 7.5]))
# wing_starboard_tip_trailing_edge = wing.project(np.array([15., 24.5, 7.5]))
# wing_port_tip_leading_edge = wing.project(np.array([8., -26., 7.5]))
# wing_port_tip_trailing_edge = wing.project(np.array([15., -24.5, 7.5]))

# wing_span_vector = starboard_tip_quarter_chord - port_tip_quarter_chord
# wing_root_chord_vector = wing_root_trailing_edge - wing_root_leading_edge
# wing_starboard_tip_chord_vector = wing_starboard_tip_trailing_edge - wing_starboard_tip_leading_edge
# wing_port_tip_chord_vector = wing_port_tip_trailing_edge - wing_port_tip_leading_edge

''' TODO: Skip Joints/Actuators for now '''
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

# # Parameterization
from caddee.caddee_core.system_parameterization.free_form_deformation.ffd_functions import create_cartesian_enclosure_volume
from caddee.caddee_core.system_parameterization.free_form_deformation.ffd_block import SRBGFFDBlock

wing_geometry_primitives = wing.get_geometry_primitives()
wing_ffd_bspline_volume = create_cartesian_enclosure_volume(wing_geometry_primitives, num_control_points=(11, 2, 2), order=(4,2,2), xyz_to_uvw_indices=(1,0,2))
wing_ffd_block = SRBGFFDBlock(name='wing_ffd_block', primitive=wing_ffd_bspline_volume, embedded_entities=wing_geometry_primitives)
wing_ffd_block.add_scale_v(name='linear_taper', order=2, num_dof=3, value=np.array([0., 1., 0.]), cost_factor=1.)
wing_ffd_block.add_rotation_u(name='twist_distribution', order=4, num_dof=10, value=-1/2*np.array([0., 0.11, 0.22, 0.33, 0.44, 0.44, 0.33, 0.22, 0.11, 0.]))

horizontal_stabilizer_geometry_primitives = horizontal_stabilizer.get_geometry_primitives()
horizontal_stabilizer_ffd_bspline_volume = create_cartesian_enclosure_volume(horizontal_stabilizer_geometry_primitives, num_control_points=(11, 2, 2), order=(4,2,2), xyz_to_uvw_indices=(1,0,2))
horizontal_stabilizer_ffd_block = SRBGFFDBlock(name='horizontal_stabilizer_ffd_block', primitive=horizontal_stabilizer_ffd_bspline_volume, embedded_entities=horizontal_stabilizer_geometry_primitives)
horizontal_stabilizer_ffd_block.add_scale_v(name='horizontal_stabilizer_linear_taper', order=2, num_dof=3, value=np.array([0.5, 0.5, 0.5]), cost_factor=1.)
horizontal_stabilizer_ffd_block.add_rotation_u(name='horizontal_stabilizer_twist_distribution', order=1, num_dof=1, value=np.array([np.pi/10]))

plotting_elements = wing_ffd_block.plot(plot_embedded_entities=False, show=False)
plotting_elements = horizontal_stabilizer_ffd_block.plot(plot_embedded_entities=False, show=False, additional_plotting_elements=plotting_elements)
spatial_rep.plot(additional_plotting_elements=plotting_elements)

from caddee.caddee_core.system_parameterization.free_form_deformation.ffd_set import SRBGFFDSet
ffd_set = SRBGFFDSet(name='ffd_set', ffd_blocks={wing_ffd_block.name : wing_ffd_block, horizontal_stabilizer_ffd_block.name : horizontal_stabilizer_ffd_block})

''' TODO Finish addressing code starting from below. '''

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
num_spanwise_vlm = 21
num_chordwise_vlm = 5
leading_edge = wing.project(np.linspace(np.array([8., -26., 7.5]), np.array([8., 26., 7.5]), num_spanwise_vlm), direction=np.array([0., 0., -1.]), plot=True)  # returns MappedArray
trailing_edge = wing.project(np.linspace(np.array([15., -26., 7.5]), np.array([15., 26., 7.5]), num_spanwise_vlm), direction=np.array([0., 0., -1.]), plot=True)   # returns MappedArray
chord_surface = am.linspace(leading_edge, trailing_edge, num_chordwise_vlm)
spatial_rep.plot_meshes([chord_surface])
wing_upper_surface_wireframe = wing.project(chord_surface.value + np.array([0., 0., 1.]), direction=np.array([0., 0., -1.]), grid_search_n=15, plot=True)
wing_lower_surface_wireframe = wing.project(chord_surface.value - np.array([0., 0., 1.]), direction=np.array([0., 0., 1.]), grid_search_n=15, plot=True)
wing_camber_surface = am.linspace(wing_upper_surface_wireframe, wing_lower_surface_wireframe, 1) # this linspace will return average when n=1
spatial_rep.plot_meshes([wing_camber_surface])



# '''
# Mapped Arrays Example
# '''
# upper_surface_map = wing_upper_surface_wireframe.linear_map
# lower_surface_map = wing_lower_surface_wireframe.linear_map
# def numpy_implementation(control_points):
#     upper_surface_grid = upper_surface_map.dot(control_points)
#     lower_surface_grid = lower_surface_map.dot(control_points)
#     wing_grids = np.linspace(upper_surface_grid, lower_surface_grid, 3)
#     wing_camber_surface = wing_grids[1,:,:]
#     return wing_camber_surface

# import time
# num_iters = [1, 10, 50, 100, 1000]
# for num_iter in num_iters:
#     control_points = spatial_rep.control_points.copy()

#     t1 = time.time()
#     for i in range(num_iter):
#         control_points += 0.1/num_iter
#         output = numpy_implementation(control_points)

#     t2 = time.time()
#     numpy_time = t2-t1
#     # print(f'numpy_time with num_iter={num_iter}', numpy_time)

#     control_points = spatial_rep.control_points.copy()
#     t3 = time.time()
#     for i in range(num_iter):
#         control_points += 0.1/num_iter
#         output = wing_camber_surface.evaluate(control_points)
#     t4 = time.time()
#     array_mapper_time = t4-t3
#     # print(f'array_mapper_time with num_iter={num_iter}', array_mapper_time)
#     print(f'improvement with num_iter={num_iter}', numpy_time/array_mapper_time)
# '''
# '''

# '''
# MappedArrays Example 2: Ribs and Spars
# '''
# num_ribs = 21
# num_chordwise_points = 11
# num_rib_side_points = 4
# forward_spar_upper = wing.project(np.linspace(np.array([11., -26., 10.5]), np.array([11., 26., 10.5]), num_ribs), direction=np.array([0., 0., -1.]))
# forward_spar_lower = wing.project(np.linspace(np.array([11., -26., 6.5]), np.array([11., 26., 6.5]), num_ribs), direction=np.array([0., 0., 1.]))
# rear_spar_upper = wing.project(np.linspace(np.array([13., -26., 10.5]), np.array([13., 26., 10.5]), num_ribs), direction=np.array([0., 0., -1.]))
# rear_spar_lower = wing.project(np.linspace(np.array([13., -26., 6.5]), np.array([13., 26., 6.5]), num_ribs), direction=np.array([0., 0., 1.]))
# forward_spar_surface = am.linspace(forward_spar_upper, forward_spar_lower, num_rib_side_points)   # Only want top and bottom therefore linspace wants 2
# rear_spar_surface = am.linspace(rear_spar_upper, rear_spar_lower, num_rib_side_points)   # Only want top and bottom therefore linspace wants 2
# # spatial_rep.plot_meshes([forward_spar_surface, rear_spar_surface])
# # rib_upper_curves = # Evaluate B-spline at parametric values or project
# # rib_lower_curves = # Evaluate B-spline at parametric values or project
# # ribs = am.transfinite_interpolation(forward_spar_surface, rear_spar_surface, rib_lower_curves, rib_upper_curves) # this linspace will return average when n=1
# ribs = am.linspace(forward_spar_surface, rear_spar_surface, num_chordwise_points)
# plotting_elements = []
# for i in range(ribs.shape[2]):
#     # if i < ribs.shape[2]-1:
#     plotting_elements = spatial_rep.plot_meshes([ribs.value[:,:,i,:]], primitives=['none'], additional_plotting_elements=plotting_elements, show=False)
#     # else:
#         # spatial_rep.plot_meshes([ribs.value[:,:,i,:]], additional_plotting_elements=plotting_elements)
# spatial_rep.plot_meshes([forward_spar_surface, rear_spar_surface], additional_plotting_elements=plotting_elements)

# forward_spar_upper_map = forward_spar_upper.linear_map
# forward_spar_lower_map = forward_spar_lower.linear_map
# rear_spar_upper_map = rear_spar_upper.linear_map
# rear_spar_lower_map = rear_spar_lower.linear_map
# def numpy_implementation2(control_points):
#     forward_spar_upper = forward_spar_upper_map.dot(control_points)
#     forward_spar_lower = forward_spar_lower_map.dot(control_points)
#     rear_spar_upper = rear_spar_upper_map.dot(control_points)
#     rear_spar_lower = rear_spar_lower_map.dot(control_points)
#     forward_spar_surface = np.linspace(forward_spar_upper, forward_spar_lower, num_rib_side_points)
#     rear_spar_surface = np.linspace(rear_spar_upper, rear_spar_lower, num_rib_side_points)
#     ribs = np.linspace(rear_spar_surface, forward_spar_surface, num_chordwise_points)
#     return ribs

# import time
# num_iters = [1, 10, 50, 100, 1000]
# for num_iter in num_iters:
#     control_points = spatial_rep.control_points.copy()

#     t1 = time.time()
#     for i in range(num_iter):
#         control_points += 0.1/num_iter
#         output = numpy_implementation2(control_points)

#     t2 = time.time()
#     numpy_time = t2-t1
#     # print(f'numpy_time with num_iter={num_iter}', numpy_time)

#     control_points = spatial_rep.control_points.copy()
#     t3 = time.time()
#     for i in range(num_iter):
#         control_points += 0.1/num_iter
#         output = ribs.evaluate(control_points)
#     t4 = time.time()
#     array_mapper_time = t4-t3
#     # print(f'array_mapper_time with num_iter={num_iter}', array_mapper_time)
#     print(f'improvement with num_iter={num_iter}', numpy_time/array_mapper_time)
# '''
# '''

num_spanwise_vlm = 11
num_chordwise_vlm = 3
leading_edge = horizontal_stabilizer.project(np.linspace(np.array([27., -6.5, 6.]), np.array([27., 6.75, 6.]), num_spanwise_vlm), direction=np.array([0., 0., -1.]), grid_search_n=15)  # returns MappedArray
trailing_edge = horizontal_stabilizer.project(np.linspace(np.array([31.5, -6.5, 6.]), np.array([31.5, 6.75, 6.]), num_spanwise_vlm), direction=np.array([0., 0., -1.]), grid_search_n=15)   # returns MappedArray
chord_surface = am.linspace(leading_edge, trailing_edge, num_chordwise_vlm)
horizontal_stabilizer_upper_surface_wireframe = horizontal_stabilizer.project(chord_surface.value + np.array([0., 0., 1.]), direction=np.array([0., 0., -1.]), grid_search_n=15)
horizontal_stabilizer_lower_surface_wireframe = horizontal_stabilizer.project(chord_surface.value - np.array([0., 0., 1.]), direction=np.array([0., 0., 1.]), grid_search_n=15)
horizontal_stabilizer_camber_surface = am.linspace(horizontal_stabilizer_upper_surface_wireframe, horizontal_stabilizer_lower_surface_wireframe, 1) # this linspace will return average when n=1

plotting_meshes = [wing_camber_surface, horizontal_stabilizer_camber_surface]
spatial_rep.plot_meshes(plotting_meshes, mesh_plot_types=['wireframe'], mesh_opacity=1.)

ffd_set.setup()

affine_section_properties = ffd_set.evaluate_affine_section_properties()
rotational_section_properties = ffd_set.evaluate_rotational_section_properties()
affine_ffd_control_points_local_frame = ffd_set.evaluate_affine_block_deformations(plot=True)
ffd_control_points_local_frame = ffd_set.evaluate_rotational_block_deformations(plot=True)
ffd_control_points = ffd_set.evaluate_control_points(plot=True)
updated_geometry = ffd_set.evaluate_embedded_entities(plot=True)
updated_primitives_names = wing.primitive_names.copy()
updated_primitives_names.extend(horizontal_stabilizer.primitive_names.copy())


print('Sample evaluation: affine section properties: \n', affine_section_properties)
print('Sample evaluation: rotational section properties: \n', rotational_section_properties)

# Performing assembly of geometry from different parameterizations which usually happens in SystemParameterization
spatial_rep.update(updated_geometry, updated_primitives_names)

wing_camber_surface.evaluate(spatial_rep.control_points)
horizontal_stabilizer_camber_surface.evaluate(spatial_rep.control_points)
spatial_rep.plot_meshes([wing_camber_surface, horizontal_stabilizer_camber_surface], mesh_plot_types=['wireframe'], mesh_opacity=1.)

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

