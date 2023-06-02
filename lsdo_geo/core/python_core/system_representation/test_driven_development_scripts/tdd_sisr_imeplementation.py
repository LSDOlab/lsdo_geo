import csdl
from python_csdl_backend import Simulator
import numpy as np
import array_mapper as am

from caddee.caddee_core.system_representation.system_representation import SystemRepresentation
from caddee.caddee_core.system_parameterization.system_parameterization import SystemParameterization

system_representation = SystemRepresentation()
spatial_rep = system_representation.spatial_representation
file_path = 'models/stp/'
file_name = 'pegasus_no_rotors.stp'
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
tail_primitive_names = list(spatial_rep.get_primitives(search_names=['HT']).keys())
horizontal_stabilizer = LiftingSurface(name='tail', spatial_representation=spatial_rep, primitive_names=tail_primitive_names)
fuselage_primitive_names = list(spatial_rep.get_primitives(search_names=['Fuselage']))
fuselage = Component(name='fuselage', spatial_representation=spatial_rep, primitive_names=fuselage_primitive_names)

system_representation.add_component(wing)
system_representation.add_component(horizontal_stabilizer)
system_representation.add_component(fuselage)

# CADDEE api
tail_actuator = Component(name='tail_actuator_motor', spatial_representation=spatial_rep, primitive_names=[])
system_representation.add_component(tail_actuator)
point_on_horizontal_stabilizer = horizontal_stabilizer.project(np.array([28., 0., 7.5]))
point_on_fuselage = fuselage.project(np.array([28., 0., 9]))
system_representation.connect(component1=tail_actuator,
                             component2=fuselage, region_on_component2=point_on_fuselage, type='mechanical')
system_representation.connect(component1=tail_actuator, 
                             component2=horizontal_stabilizer, region_on_component2=point_on_horizontal_stabilizer, type='mechanical')

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
configuration_names = ["hover_configuration", "cruise_configuration"]
system_representations = system_representation.create_instances(names=configuration_names)
hover_configuration = system_representations['hover_configuration']
cruise_configuration = system_representations['cruise_configuration']

horizontal_stabilizer_quarter_chord_port = horizontal_stabilizer.project(np.array([28.5, -10., 8.]))
horizontal_stabilizer_quarter_chord_starboard = horizontal_stabilizer.project(np.array([28.5, 10., 8.]))
horizontal_stabilizer_acutation_axis = horizontal_stabilizer_quarter_chord_starboard - horizontal_stabilizer_quarter_chord_port
from caddee.caddee_core.system_representation.prescribed_actuations import PrescribedRotation
horizontal_stabilizer_actuator_solver = PrescribedRotation(component=tail_actuator, axis=horizontal_stabilizer_acutation_axis)
horizontal_stabilizer_actuator_solver.set_rotation(name='cruise_tail_actuation', value=0.25 , units='radians')
cruise_configuration.transform(horizontal_stabilizer_actuator_solver)

system_representation_model = system_representation.assemble_csdl()
system_parameterization_model = system_parameterization.assemble_csdl()

my_model = csdl.Model()
my_model.add(system_parameterization_model, 'system_parameterization')
my_model.add(system_representation_model, 'system_representation')


# some_model =  Model()

# # Starting Here


# system_model = cd.SystemModel()
# system_model.sizing_group = sizing_group = cd.SizingGroup()
# dc = cd.DesignScenario(name='recon_mission')
# system_model.add_design_scenario(dc)
# ha_cruise = cd.AircraftCondition(
#     name='high_altitude_cruise',
#     stability_flag=False,
#     dynamic_flag=False,)
# ha_cruise.set_module_input('cruise_speed')
# dc.add_design_condition(ha_cruise)



# system_model.connect(_from=ha_cruise, _to=some_model, 'cruise_speed')


# # UFL is unified form language
# # csdl: computational system design language (Unified Computational System Design Language)
# # macl: multidisciplinary analysis coupling language
# # idea M3L: unified multifidelity multidisciplinary modeling language
# # idea MICL: unified multifidelity interdisciplinary coupling language

# # option A
# pressure_mesh = ...     # Pressure mesh will be something like a B-spline class instance or something like that

# pressure = m3l.StateVariable('pressure', pressure_mesh)
# displacement = m3l.StateVariable('displacement', displacement_mesh)

# nodal_forces = m3l.framework_pressure_to_nodal_forces(pressure, nodal_forces_mesh)
# structural_solver = BeamSolver(beam_mesh)
# nodal_displacements = structural_solver.evaluate(nodal_forces, nodal_displacement_mesh)
# displacement_output = m3l.nodal_to_framework(nodal_displacements, displacement_mesh, nodal_forces_mesh, structural_solver)     # TODO think of other ways the 2 maps can be passed

# nodal_displacements_for_aero = m3l.evaluate_field(displacement, nodal_forces_mesh_for_aero)
# aero_solver = VLM(vlm_mesh)
# nodal_forces_from_aero = aero_solver.evaluate(nodal_displacements_for_aero, nodal_forces_from_aero_mesh)
# pressure_output = m3l.nodal_to_framework(nodal_forces_from_aero, pressure.mesh, aero_solver)

# # m3l.implcit_solution(displacement_output, pressure_output, solver=Newton)
# pressure.equals(pressure_output)
# displacement.equals(displacement_output)

# coupled_model = m3l.ModelGroup(pressure, displacement, solver=m3l.Netwon(max_iter=20,...))      # Perhaps it uses a Newton class from elsewhere?

# # ha_cruise.add_model_group(model_group)
# ha_cruise.mechanics_group.add_model(coupled_model)


# # Option 1 for nomenclature
# StateVariable()   # sublass of CSDL Variable()
# DisciplineModel() # sublass of CSDL Model()
# StateType()
# StateOperation()  # sublass of CSDL Operation()
# MultidisciplinaryModel()      # another subclass of CSDL Model()

# # Option 2 for nomenclature   # Start with option 2
# State()
# StateType()
# StateMap()
# Model()
# ModelGroup()
# # For vectorization, 


# # Insert API for specifying implicit csdl model


# # Option B
# pressure_mesh = ...     # Pressure mesh will be something like a B-spline class instance or something like that

# pressure = cd.StateVariable('pressure', pressure_mesh)
# # ha_cruise.add_state(pressure)     # Don't think
# # displacement = cd.StateVariable('displacement', displacement_mesh)
# # ha_cruise.add_state(displacement)

# nodal_forces = xfer.pressure_to_force(pressure, nodal_forces_mesh)
# structural_solver = BeamSolver(fea_mesh)
# nodal_displacements = structural_solver.evaluate(nodal_forces, nodal_displacement_mesh)
# displacement = xfer.nodal_to_framework(nodal_displacements, displacement_mesh, structural_solver)     # TODO think of other ways the 2 maps can be passed

# nodal_displacements_for_aero = xfer.evaluate_field(displacement, nodal_forces_mesh_for_aero)
# aero_solver = VLM(vlm_mesh)
# nodal_forces_from_aero = aero_solver.evaluate(nodal_displacements_for_aero, nodal_forces_from_aero_mesh)
# pressure = xfer.nodal_to_framework(nodal_forces_from_aero, pressure.mesh, aero_solver)
# modified_pressure = pressure * 0.9

# # Insert API for specifying implicit csdl model






# # option B
# # Put this on hold because we want to start with the user specifying which map will be computed from the others (if they want to conserve energy)
# pressure = cd.StateVariable('pressure', pressure_mesh)
# structural_solver_module = BeamSolverModule(
#     mesh=fea_mesh,
#     pressure=pressure,
#     displacement_mesh=displacement_mesh,
#     pressure_to_nodal_forces_map or nodal_forces_mesh)
# displacements = structural_solver_module.evaluate(nodal_forces, nodal_displacement_mesh)


# insert aero module and api for specifying implicit csdl model

# note: do we want to include the component?
# I think as of this very moment, it doesn't seem like it is necessarily needed. However, in some longer term ideas, I like the interpretation of a 
# component as the analysis domain in regards to the physical system.


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