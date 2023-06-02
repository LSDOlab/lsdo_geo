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
spatial_rep.plot(point_types=['evaluated_points'])
# spatial_rep.plot(point_types=['control_points'])
system_parameterization = SystemParameterization(system_representation=system_representation)

# Create Components
from caddee.caddee_core.system_representation.component.component import LiftingSurface, Component
wing_primitive_names = list(spatial_rep.get_primitives(search_names=['Wing']).keys())
wing = LiftingSurface(name='wing', spatial_representation=spatial_rep, primitive_names=wing_primitive_names)
tail_primitive_names = list(spatial_rep.get_primitives(search_names=['Tail_1']).keys())
horizontal_stabilizer = LiftingSurface(name='tail', spatial_representation=spatial_rep, primitive_names=tail_primitive_names)
fuselage_primitive_names = list(spatial_rep.get_primitives(search_names=['Fuselage']))
fuselage = Component(name='fuselage', spatial_representation=spatial_rep, primitive_names=fuselage_primitive_names)

system_representation.add_component(wing)
system_representation.add_component(horizontal_stabilizer)
system_representation.add_component(fuselage)

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
# -- However, there is also a joint within the actuator. I want to impose a rule that joints can never add energy into the system or convert energy types. 

# tail_actuator = Motor(name='', stiffness=0., damping=0.) # NOTE: Motor is a component that is specifically an actuator
# CADDEE api
tail_actuator = Component(name='tail_actuator_motor', spatial_representation=spatial_rep, primitive_names=[])
system_representation.add_component(tail_actuator)
point_on_horizontal_stabilizer = horizontal_stabilizer.project(np.array([28., 0., 7.5]))
point_on_fuselage = fuselage.project(np.array([28., 0., 9]))
system_representation.connect(component1=tail_actuator,
                             component2=fuselage, region_on_component2=point_on_fuselage, type='mechanical')
system_representation.connect(component1=tail_actuator, 
                             component2=horizontal_stabilizer, region_on_component2=point_on_horizontal_stabilizer, type='mechanical')
# should this be system_representation.power_systems_architecture.connect(...)? I like calling just system_representation better.
# I think maybe in general, the user should never directly interact with SpatialRepresentation or power_systems_architectureRepresentation


# Meshes definitions
num_spanwise_vlm = 21
num_chordwise_vlm = 5
leading_edge = wing.project(am.linspace(am.array([8., -26., 7.5]), am.array([8., 26., 7.5]), num_spanwise_vlm),
                            direction=am.array([0., 0., -1.]))  # returns MappedArray
trailing_edge = wing.project(np.linspace(np.array([15., -26., 7.5]), np.array([15., 26., 7.5]), num_spanwise_vlm),
                             direction=np.array([0., 0., -1.]))   # returns MappedArray
chord_surface = am.linspace(leading_edge, trailing_edge, num_chordwise_vlm)
wing_upper_surface_wireframe = wing.project(chord_surface.value + np.array([0., 0., 1.]), direction=np.array([0., 0., -1.]), grid_search_n=15)
wing_lower_surface_wireframe = wing.project(chord_surface.value - np.array([0., 0., 1.]), direction=np.array([0., 0., 1.]), grid_search_n=15)
wing_camber_surface = am.linspace(wing_upper_surface_wireframe, wing_lower_surface_wireframe, 1) # this linspace will return average when n=1
wing_camber_surface = wing_camber_surface.reshape((num_chordwise_vlm, num_spanwise_vlm, 3))
system_representation.add_output(name='chord_distribution', quantity=am.norm(leading_edge-trailing_edge))


num_spanwise_vlm = 11
num_chordwise_vlm = 3
leading_edge = horizontal_stabilizer.project(np.linspace(np.array([27., -6.5, 6.]), np.array([27., 6.75, 6.]), num_spanwise_vlm),
                                             direction=np.array([0., 0., -1.]), grid_search_n=15)
trailing_edge = horizontal_stabilizer.project(np.linspace(np.array([31.5, -6.5, 6.]), np.array([31.5, 6.75, 6.]), num_spanwise_vlm),
                                              direction=np.array([0., 0., -1.]), grid_search_n=15)
chord_surface = am.linspace(leading_edge, trailing_edge, num_chordwise_vlm)
horizontal_stabilizer_upper_surface_wireframe = horizontal_stabilizer.project(chord_surface.value + np.array([0., 0., 1.]),
                                                                              direction=np.array([0., 0., -1.]), grid_search_n=15)
horizontal_stabilizer_lower_surface_wireframe = horizontal_stabilizer.project(chord_surface.value - np.array([0., 0., 1.]),
                                                                              direction=np.array([0., 0., 1.]), grid_search_n=15)
horizontal_stabilizer_camber_surface = am.linspace(horizontal_stabilizer_upper_surface_wireframe, horizontal_stabilizer_lower_surface_wireframe, 1)
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
wing_ffd_bspline_volume = create_cartesian_enclosure_volume(wing_geometry_primitives, num_control_points=(11, 2, 2), order=(4,2,2),
                                                            xyz_to_uvw_indices=(1,0,2))
wing_ffd_block = SRBGFFDBlock(name='wing_ffd_block', primitive=wing_ffd_bspline_volume, embedded_entities=wing_geometry_primitives)
wing_ffd_block.add_scale_v(name='linear_taper', order=2, num_dof=3, value=np.array([0., 1., 0.]))
wing_ffd_block.add_scale_w(name='constant_thickness_scaling', order=1, num_dof=1, value=np.array([0.5]))
wing_ffd_block.add_rotation_u(name='twist_distribution', order=4, num_dof=10,
                              value=1/4*np.array([0., 0.11, 0.22, 0.33, 0.44, 0.44, 0.33, 0.22, 0.11, 0.]))

horizontal_stabilizer_geometry_primitives = horizontal_stabilizer.get_geometry_primitives()
horizontal_stabilizer_ffd_bspline_volume = create_cartesian_enclosure_volume(horizontal_stabilizer_geometry_primitives,num_control_points=(11, 2, 2),
                                                                             order=(4,2,2), xyz_to_uvw_indices=(1,0,2))
horizontal_stabilizer_ffd_block = SRBGFFDBlock(name='horizontal_stabilizer_ffd_block', primitive=horizontal_stabilizer_ffd_bspline_volume,
                                               embedded_entities=horizontal_stabilizer_geometry_primitives)
horizontal_stabilizer_ffd_block.add_scale_v(name='horizontal_stabilizer_linear_taper', order=2, num_dof=3, value=np.array([0.5, 0.5, 0.5]),
                                            cost_factor=1.)
horizontal_stabilizer_ffd_block.add_rotation_u(name='horizontal_stabilizer_twist_distribution', order=1, num_dof=1, value=np.array([np.pi/10]))

from caddee.caddee_core.system_parameterization.free_form_deformation.ffd_set import SRBGFFDSet
ffd_blocks = {wing_ffd_block.name : wing_ffd_block, horizontal_stabilizer_ffd_block.name : horizontal_stabilizer_ffd_block}
ffd_set = SRBGFFDSet(name='ffd_set', ffd_blocks=ffd_blocks)
system_parameterization.add_geometry_parameterization(ffd_set)
system_parameterization.setup()


# Configuration creations and prescribed actuations
''' Implement everything below this point. '''

# For modularity and cleanliness, want to have the functionality so that CADDEE is not absolutely needed in order to perform the functionalities
# (CADDEE calls these functionalities and automates it). One aspect is the changing configuration over time/life of the system. Long term, I want
# to support a continuous representation of the system w.r.t. time. This would likely use control points. This method should still work for the
# long term goal because these discrete representations can serve as the control points in a continuous representation. Just to include this wacky idea,
# if we think about the system as a collection of 2D manifolds (pick B-spline surfaces as an example), and we want to continuous represent the system
# over a dynamic segment, then the system can be represented as a collection of 3D manifolds (pick B-spline volumes as an example), where the additional
# parametric axis corresponds to the time axis. Since the geometry itself can include volumes and we want to also continously represent the properties,
# this would require the use of n-dimensional manifolds to represent the system.
configuration_names = ["hover_configuration", "cruise_configuration"]
system_representations = system_representation.create_instances(names=configuration_names)
hover_configuration = system_representations['hover_configuration']
cruise_configuration = system_representations['cruise_configuration']
# NOTE: having multiple instances of system configuration specifically also implied multiple instances of the power_systems_architecture representation. This is beneficial
# because it allows for dynamically changing power_systems_architecture where connections are made or not made (think of transformers as a funny example). This makes
# things a little more complicated because now the TES may have to handle discretely changing power_systems_architecture rather than solving a single power_systems_architecture over time.
# -- Because the long term idea is to have a continous representation of the configuration over time, this means that we would want to have a 
#       continous representation of the power_systems_architecture over time (this is a challenge because a power_systems_architecture is inherently discrete). A potential solution to this
#       could be to use ideas from topology optimization (assign a topological density to each power_systems_architecture connection that can vary with time).
# NOTE: Perhaps there can be a separate method or keyword for a dynamic instances. This could help with specifying a dynamic actuation profile as a
#           vector rather than individually an unreasonable number of times. Perhaps there can be a subsequent method that is used to "string together"
#           configurations into a dynamic configuration. I think I like this second idea.


# Want to try to not create separate components because a single "thing" can be a "lifting surface" and also a "rotor" and a "thermal insulator", etc.
# -- Instead, the solvers see the components through a certain lens and perform their analysis.
# -- With this philosophy, it seems that actuators should be passed into a PrescribedRotation solver or PrescribedTranslation or PrescribedDisplacement
# -- I then know that I want to have a single PrescribedActuation "solver", so perhaps these sub-solvers go into the larger solver?
# As a note, one motivation for this is to stay independent whether its a prescribed actuation or solved using E-M, etc.
# To further this, this is recognizing that no part of the system really has inherent meaning. As the end of the day, each component is an arbitrary
#   collection of atoms. In life as we perceive it, we assign meaning to objects and treat them as such. For instance, an aerodynamicist may see an object
#   as a wing, while a thermodynamicist may see it as a thermal insulator, while a soldier sees it as a shield, and an alien sees it as something else 
#   entirely. For extensibility, this recognition is important because the user is tasked with assigning meaning to components of the system through
#   deciding to perform specialized analysis that is specific to the meaning. If we try to create a type of component for every type of way each thing
#   can be perceived, we will end up with an incredibly large library of components, and a single component will simultaneously have multiple definitions.
#   (for instance, a wing component may be redefined as a black body for heat transfer within the same script/representation).
horizontal_stabilizer_quarter_chord_port = horizontal_stabilizer.project(np.array([28.5, -10., 8.]))
horizontal_stabilizer_quarter_chord_starboard = horizontal_stabilizer.project(np.array([28.5, 10., 8.]))
horizontal_stabilizer_acutation_axis = horizontal_stabilizer_quarter_chord_starboard - horizontal_stabilizer_quarter_chord_port
from caddee.caddee_core.system_representation.prescribed_actuations import PrescribedRotation
horizontal_stabilizer_actuator_solver = PrescribedRotation(component=tail_actuator, axis=horizontal_stabilizer_acutation_axis)
# horizontal_stabilizer_actuator_solver.set_input(name='cruise_tail_actuation', units='radians', value=0.25)  # one attempt at note below.
horizontal_stabilizer_actuator_solver.set_rotation(name='cruise_tail_actuation', value=0.25 , units='radians')  # another attempt at note below.
# TODO NOTE: Probably separate out the name and value to a separate set_input call to be like other solvers? what if another one of these "solvers"
#     requires multiple inputs rather than having one parameter per solver?
# horizontal_stabilizer_actuator_solver.set_module_input('rotation', val=0.25, dv_flag=False, lower=0.5, upper=0.5, scaler=1.)
cruise_configuration.transform(horizontal_stabilizer_actuator_solver)   # The representation's version of add_solver. CADDEE may or may not call this.
# The nature of this operation is that it is changing the actual system in a prescribed manner. The canonical case is prescribed actuation although
# we can also consider prescribed displacement (is that actuation in that sense?). I wonder if there are perhaps changes that could be in materials
# or properties that could influence the name of this method.
# It seems that also want an API for separately making connections (building the power_systems_architecture). Perhaps the connections by default apply to every connection
#   but if the user wants, they may manually specify a subset of configurations if they only want them to apply then.
# -- NOTE: We want to automate most of the connections like phyiscal connections (mechanical work transfer) anyways.

# NOTE: In a sense, each rotation/translation/displacement/etc. is a solver. I'm leaning towards that each solver is added individually to the 
# configuration, then in the backend, all those changes are organized a vectorized.


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

sim = Simulator(my_model)
sim.run()
# sim.compute_total_derivatives()
# sim.check_totals()

# affine_section_properties = ffd_set.evaluate_affine_section_properties(prescribed_affine_dof=np.append(initial_guess_linear_taper, np.zeros(4,)))
affine_section_properties = ffd_set.evaluate_affine_section_properties()
rotational_section_properties = ffd_set.evaluate_rotational_section_properties()
affine_deformed_ffd_control_points = ffd_set.evaluate_affine_block_deformations()
rotated_ffd_control_points = ffd_set.evaluate_rotational_block_deformations()
ffd_control_points = ffd_set.evaluate_control_points()
ffd_embedded_entities = ffd_set.evaluate_embedded_entities()

updated_primitives_names = wing.primitive_names.copy()
updated_primitives_names.extend(horizontal_stabilizer.primitive_names.copy())
spatial_rep.update(ffd_embedded_entities, updated_primitives_names)

wing_camber_surface.evaluate(spatial_rep.control_points['geometry'])
horizontal_stabilizer_camber_surface.evaluate(spatial_rep.control_points['geometry'])

print('wingspan', sim['wingspan'])
print("Python and CSDL difference: wingspan", np.linalg.norm(wingspan.value - sim['wingspan']))
print('wing root chord', sim['wing_root_chord'])
print("Python and CSDL difference: wing root chord", np.linalg.norm(root_chord.value - sim['wing_root_chord']))

wing_camber_surface_csdl = sim['wing_camber_surface']
horizontal_stabilizer_camber_surface_csdl = sim['horizontal_stabilizer_camber_surface']
print("Python and CSDL difference: wing camber surface", np.linalg.norm(wing_camber_surface_csdl - wing_camber_surface.value))
print("Python and CSDL difference: horizontal stabilizer camber surface", 
      np.linalg.norm(horizontal_stabilizer_camber_surface_csdl - horizontal_stabilizer_camber_surface.value))

spatial_rep.plot_meshes([wing_camber_surface_csdl, horizontal_stabilizer_camber_surface_csdl], mesh_plot_types=['wireframe'], mesh_opacity=1.)