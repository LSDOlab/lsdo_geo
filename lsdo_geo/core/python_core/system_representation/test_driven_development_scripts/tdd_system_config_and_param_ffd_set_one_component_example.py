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
spatial_rep.import_file(file_name=file_path+'rect_wing.stp')

spatial_rep.plot(plot_types=['mesh'])

# Create Components
from caddee.caddee_core.system_representation.component.component import LiftingSurface, Component
wing_primitive_names = list(spatial_rep.get_primitives(search_names=['Wing']).keys())
wing = LiftingSurface(name='wing', spatial_representation=spatial_rep, primitive_names=wing_primitive_names)  # TODO add material arguments
system_representation.add_component(wing)

# Key locations identificatiions (common variables that are used repeatedly)
starboard_tip_quarter_chord = wing.project(np.array([1., 9., 1.])) # returns a MappedArray
port_tip_quarter_chord = wing.project(np.array([1., -9., 1.])) # returns a MappedArray
wing_root_leading_edge = wing.project(np.array([0., 0., 0.]))
wing_root_trailing_edge = wing.project(np.array([4., 0., 0.]))
wing_starboard_tip_leading_edge = wing.project(np.array([0., 9., 0.]))
wing_starboard_tip_trailing_edge = wing.project(np.array([4., 9., 0.]))
wing_port_tip_leading_edge = wing.project(np.array([0., -9., 0.]))
wing_port_tip_trailing_edge = wing.project(np.array([4., -9., 0.]))

wing_span_vector = starboard_tip_quarter_chord - port_tip_quarter_chord
wing_root_chord_vector = wing_root_trailing_edge - wing_root_leading_edge
wing_starboard_tip_chord_vector = wing_starboard_tip_trailing_edge - wing_starboard_tip_leading_edge
wing_port_tip_chord_vector = wing_port_tip_trailing_edge - wing_port_tip_leading_edge

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

from caddee.caddee_core.system_parameterization.free_form_deformation.ffd_set import SRBGFFDSet
ffd_set = SRBGFFDSet(name='ffd_set', ffd_blocks={wing_ffd_block.name : wing_ffd_block})

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
leading_edge = wing.project(np.linspace(np.array([0., -9., 0.]), np.array([0., 9., 0.]), num_spanwise_vlm), direction=np.array([0., 0., -1.]))  # returns MappedArray
trailing_edge = wing.project(np.linspace(np.array([4., -9., 0.]), np.array([4., 9., 0.]), num_spanwise_vlm), direction=np.array([0., 0., -1.]))   # returns MappedArray
chord_surface = am.linspace(leading_edge, trailing_edge, num_chordwise_vlm)
wing_upper_surface_wireframe = wing.project(chord_surface.value + np.array([0., 0., 1.]), direction=np.array([0., 0., -1.]), grid_search_n=25)
wing_lower_surface_wireframe = wing.project(chord_surface.value - np.array([0., 0., 1.]), direction=np.array([0., 0., 1.]), grid_search_n=25)
wing_camber_surface = am.linspace(wing_upper_surface_wireframe, wing_lower_surface_wireframe, 1) # this linspace will return average when n=1

plotting_meshes = [wing_camber_surface]
spatial_rep.plot_meshes(plotting_meshes, mesh_plot_types=['wireframe'], mesh_opacity=1.)

ffd_set.setup()

affine_section_properties = ffd_set.evaluate_affine_section_properties()
rotational_section_properties = ffd_set.evaluate_rotational_section_properties()
affine_ffd_control_points_local_frame = ffd_set.evaluate_affine_block_deformations(plot=True)
ffd_control_points_local_frame = ffd_set.evaluate_rotational_block_deformations(plot=True)
ffd_control_points = ffd_set.evaluate_control_points(plot=True)
updated_wing = ffd_set.evaluate_embedded_entities(plot=True)

print('Sample evaluation: affine section properties: \n', affine_section_properties)
print('Sample evaluation: rotational section properties: \n', rotational_section_properties)

updated_camber_mesh = wing_camber_surface.evaluate(updated_wing)
spatial_rep.plot_meshes([updated_camber_mesh], mesh_plot_types=['wireframe'], mesh_opacity=1.)

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

