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
file_path = '/models/stp/'
# system_representation.spatial_representation = spatial_representation = cd.SpatialRepresentation(import_file = file_path + 'rect_wing.stp')
from caddee.caddee_core.system_representation.spatial_representation.spatial_representation import SpatialRepresentation
system_representation.spatial_representation = spatial_representation = SpatialRepresentation()
spatial_representation.import_geometry(file_name=file_path+'rect_wing.stp')
''' TODO: Several methods in mechanical structure need to be filled in. '''

''' TODO: Components and their relationship with SpatialRepresentation (and geometry and material) needs to be straightened out. '''
# # rotor1 = cd.Rotor(name='rotor1', file_search_name='rotor, 0')
# # wing = cd.LiftingSurface(name='wing', file_search_name='Wing')
# # horizontal_stabilizer = cd.LiftingSurface(name='horizontal_stabilizer', file_search_name='tail, 0')
# from caddee.caddee_core.system_representation.spatial_representation.component.lifting_surface import LiftingSurface
# wing = LiftingSurface(name='wing', primitive_names=['Wing']) # NOTE: WARNING: ODDITY!! This doesn't have enough info.
# # In order to have enough information, component needs a pointer to the SpatialRepresentation object which is weird and bad.
# # What the example above does is the Component only stores the string until it's added to the SpatialRepresentation.
# #   -- This works, but it makes it impossible to do projections on the Component before this adding happens.
# #   -- This Component object feels like it is breaking rules about being a standalone object.
# #   -- Is a SpatialRepresentation a collection of Components? Are Components partitionings of SpatialRepresentation? Should a Component be its own object?
# #       -- I feel like it's a partitioning more than a MS is a collection of compoennts. I do feel it should be an object though so we can add vars.
# #       -- Perhaps a general Component object can have its primitives added directly, but our use case is we essentially "import" geometry from the MS.
# #           -- This doesn't feel great since we are using the standard use case of a Component being a component of a larger object.

# # rotor1.add_variable('RPM', condition=cruise, dv=True, computed_upstream=False)    # computed_upstream is False by default

# # spatial_representation.add_component(rotor1)
# spatial_representation.add_component(wing)    # If using the API from above, this adding really really should happen immediately after creating the Component.
# # spatial_representation.add_component(horizontal_stabilizer)

# wing = spatial_representation.partition_component(name='wing', primitive_names=['Wing'], type='lifting surface')  # This returns a LiftingSurface component object.
# # NOTE: I feel that this is logically more consistent, but perhaps not as scalable.
# # -- An alternative idea is to have the user pass in the object, and this method adds the primitives.

## Alternative
from caddee.caddee_core.system_representation.spatial_representation.component.component import LiftingSurface
wing_no_geometry = LiftingSurface()
wing = spatial_representation.partition_component(name='wing', primitive_names=['Wing'], obj=wing_no_geometry)

''' TODO: Finish addressing code below starting from this point '''

# Parameterization
wing_ffd_block = RectangularFFDBlock(component=wing)

wing_ffd_block.add_scale_y(num_dof=3, parameter_degree=1, value=np.array([0., 1., 0.]), cost_factor=1.)
wing_ffd_block.add_rotation_x(num_dof=10, parameter_degree=3, value=np.array([0., 1., 0.]))

# rotor1_translator = TranslatingFFDBlock(component=rotor1)
# horizontal_stabilizer_translator = TranslatingFFDBlock(component=horizontal_stabilizer)
# horizontal_stabilizer_translator.set_value(np.array([10., 0., 0.]))

# motor_ffd_block = PolarFFDBlock(component='motor1')

''' Note: Do we want to generalize ffd to design_parameterization so we are not tied to ffd? '''
system_parameterization.add_ffd(wing_ffd_block)
# system_parameterization.add_ffd(rotor1_translator)
# system_parameterization.add_ffd(horizontal_stabilizer_translator)
# system_parameterization.add_ffd(motor_ffd_block)


# Geometric inputs, outputs, and constraints
''' Note: Where to add geometric inputs, outputs, and constraints? '''
''' How do we want to add parameters? We can try to use the convention from the spreadsheet, but there are too many possibilities? '''
''' Fundamental issue: In most solvers, you know exactly what the inputs need to be. For geometry, we don't. '''
starboard_tip_quarter_chord = wing.project(np.array([1., 20., 2.])) # returns a MappedArray
port_tip_quarter_chord = wing.project(np.array([1., -20., 2.])) # returns a MappedArray
wing_root_leading_edge = wing.project([0., 0., 0.])
wing_root_trailing_edge = wing.project([5., 0., 0.])
wing_starboard_tip_leading_edge = wing.project([0., 20., 0.])
wing_starboard_tip_trailing_edge = wing.project([5., 20., 0.])
wing_port_tip_leading_edge = wing.project([0., -20., 0.])
wing_port_tip_trailing_edge = wing.project([50., 20., 0.])

wing_span_vector = starboard_tip_quarter_chord - port_tip_quarter_chord
wing_root_chord_vector = wing_root_trailing_edge - wing_root_leading_edge
wing_starboard_tip_chord_vector = wing_starboard_tip_trailing_edge - wing_starboard_tip_leading_edge
wing_port_tip_chord_vector = wing_port_tip_trailing_edge - wing_port_tip_leading_edge

# wing_to_tail_vector =  horizontal_stabilizer.project(np.array([0., 0., 0.])) - wing_root_trailing_edge

# spatial_representation.add_variable('wing_root_chord', computed_upstream=False, dv=True, quantity=MagnitudeCalculation(wing_root_chord_vector))   # geometric design variable
# spatial_representation.add_variable('wing_starboard_tip_chord', computed_upstream=True, connection_name='wing_tip_chord', quantity=MagnitudeCalculation(wing_starboard_tip_chord_vector))   # geometric input
# spatial_representation.add_variable('wing_port_tip_chord', computed_upstream=True, connection_name='wing_tip_chord', quantity=MagnitudeCalculation(wing_port_tip_chord_vector))   # geometric input
# # Note: This will throw an error because CSDL does not allow for a single variable to conenct to multiple (wing_tip_chord --> wing_starboard_tip_chord and wing_tip_chord --> wing_port_tip_chord)

# spatial_representation.add_variable('wing_to_tail_distance', computed_upstream=False, val=10, quantity=MagnitudeCalculation(wing_port_tip_chord_vector))    # Geometric constraint (very similar to geometric input)

# spatial_representation.add_variable('wing_span', output_name='wingspan', quantity=MagnitudeCalculation(wing_span_vector))   # geometric output

# TODO Redo actuations to use the kinematic optimization.
# Actuator
# # Example: tilt-wing
# rotation_axis = starboard_tip_quarter_chord - port_tip_quarter_chord
# rotation_origin = wing.project_points([1., 0., 2.])
# tilt_wing_actuator = cd.Actuator(actuating_components=[wing, rotor1], rotation_origin=rotation_origin, rotation_axis=rotation_axis)
# tilt_wing_actuator = cd.Actuator(actuating_components=['wing', 'rotor1'], rotation_origin=rotation_origin, rotation_axis=rotation_axis)
# system_parameterization.add_actuator(tilt_wing_actuator)

# TODO Go back and implement Joins after the kinematics optimization (make MBD optimization (energy minimization probably))
# Joints
# Example: rotor mounted to boom
# rotor_to_boom_connection_point = np.array([6., 18., 2.])
# rotor_to_boom_connection_point_on_rotor = rotor1.project(rotor_to_boom_connection_point)
# rotor_to_boom_connection_point_on_boom = boom.project(rotor_to_boom_connection_point)
# rotor_boom_joint = cd.Joint()


# Note: Powertrain and material definitions have been skipped for the sake of time in this iteration.

# Mesh definitions
# rotor1_thrust_vector_origin = rotor1.project(np.array([5., 18., 0.]))
# rotor1_thurst_vector_tip = rotor1.project(np.array([5., 18., 2.]))
# rotor1_thurst_direction_vector = rotor1_thurst_vector_tip - rotor1_thrust_vector_origin
# rotor1_bem_mesh = cd.BemMesh(rotor1_thurst_direction_vector)    # Note: not immediately sure if this object should be a part of CADDEE (cd.)

num_spanwise_vlm = 20
num_chordwise_vlm = 4
leading_edge = wing.project(np.linspace(np.array([0., -10., 0.]), np.array([0., 10., 0.]), num_spanwise_vlm))  # returns MappedArray
trailing_edge = wing.project(np.linspace(np.array([10., -10., 0.]), np.array([10., 10., 0.]), num_spanwise_vlm))   # returns MappedArray
chord_surface = am.linspace(leading_edge, trailing_edge, num_chordwise_vlm)
wing_upper_surface_wireframe = wing.project(chord_surface + np.array([0., 0., 10.]), direction=np.array([0., 0., -1.]))
wing_lower_surface_wireframe = wing.project(chord_surface - np.array([0., 0., 10.]), direction=np.array([0., 0., 1.]))
wing_camber_surface = am.linspace(wing_upper_surface_wireframe, wing_lower_surface_wireframe, 1) # this linspace will return average when n=1
wing_vlm_mesh = cd.VLMMesh(wing_camber_surface)    # Note: not immediately sure if this object should be a part of CADDEE (cd.)


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

