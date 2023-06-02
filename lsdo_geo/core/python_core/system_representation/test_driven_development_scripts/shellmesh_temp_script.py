import caddee as cd 
import numpy as np
import array_mapper as am

from caddee.caddee_core.system_representation.system_representation import SystemRepresentation
system_representation = SystemRepresentation()
from caddee.caddee_core.system_parameterization.system_parameterization import SystemParameterization
system_parameterization = SystemParameterization(system_representation=system_representation)

file_path = 'models/stp/'
spatial_rep = system_representation.spatial_representation
spatial_rep.import_file(file_name=file_path+'tilt_duct_no_rotors_no_people.stp')

spatial_rep.plot(plot_types=['mesh'])

# Create Components
from caddee.caddee_core.system_representation.component.component import LiftingSurface
wing_primitive_names = list(spatial_rep.get_primitives(search_names=['MainWing']).keys())
wing = LiftingSurface(name='wing', spatial_representation=spatial_rep, primitive_names=wing_primitive_names)
system_representation.add_component(wing)
horizontal_tail_primitive_names = list(spatial_rep.get_primitives(search_names=['HorzTail']).keys())
horizontal_tail = LiftingSurface(name='horizontal_tail', spatial_representation=spatial_rep, primitive_names=horizontal_tail_primitive_names)
system_representation.add_component(horizontal_tail)
vertical_tail_primitive_names = list(spatial_rep.get_primitives(search_names=['Tail_1']).keys())
vertical_tail = LiftingSurface(name='vertical_tail', spatial_representation=spatial_rep, primitive_names=vertical_tail_primitive_names)
system_representation.add_component(vertical_tail)
canard_primitive_names = list(spatial_rep.get_primitives(search_names=['Canard']).keys())
canard = LiftingSurface(name='canard', spatial_representation=spatial_rep, primitive_names=canard_primitive_names)
system_representation.add_component(canard)
fuselage_primitive_names = list(spatial_rep.get_primitives(search_names=['Fuselage']).keys())
fuselage = LiftingSurface(name='fuselage', spatial_representation=spatial_rep, primitive_names=fuselage_primitive_names)
system_representation.add_component(fuselage)

right_wing_vane0_primitive_names = list(spatial_rep.get_primitives(search_names=['HVane_0']).keys())
right_wing_vane0 = LiftingSurface(name='right_wing_vane0', spatial_representation=spatial_rep, primitive_names=right_wing_vane0_primitive_names)
system_representation.add_component(right_wing_vane0)
left_wing_vane1_primitive_names = list(spatial_rep.get_primitives(search_names=['HVane_1']).keys())
left_wing_vane1 = LiftingSurface(name='left_wing_vane1', spatial_representation=spatial_rep, primitive_names=left_wing_vane1_primitive_names)
system_representation.add_component(left_wing_vane1)
right_horizontal_tail_vane2_primitive_names = list(spatial_rep.get_primitives(search_names=['HVane_2']).keys())
right_horizontal_tail_vane2 = LiftingSurface(name='right_horizontal_tail_vane2', spatial_representation=spatial_rep, primitive_names=right_horizontal_tail_vane2_primitive_names)
system_representation.add_component(right_horizontal_tail_vane2)
left_horizontal_tail_vane3_primitive_names = list(spatial_rep.get_primitives(search_names=['HVane_3']).keys())
left_horizontal_tail_vane3 = LiftingSurface(name='vane3', spatial_representation=spatial_rep, primitive_names=left_horizontal_tail_vane3_primitive_names)
system_representation.add_component(left_horizontal_tail_vane3)
right_canard_vane4_primitive_names = list(spatial_rep.get_primitives(search_names=['HVane_4']).keys())
right_canard_vane4 = LiftingSurface(name='right_canard_vane4', spatial_representation=spatial_rep, primitive_names=right_canard_vane4_primitive_names)
system_representation.add_component(right_canard_vane4)
left_canard_vane5_primitive_names = list(spatial_rep.get_primitives(search_names=['HVane_5']).keys())
left_canard_vane5 = LiftingSurface(name='left_canard_vane5', spatial_representation=spatial_rep, primitive_names=left_canard_vane5_primitive_names)
system_representation.add_component(left_canard_vane5)


wing.plot()
horizontal_tail.plot()
vertical_tail.plot()
canard.plot()
fuselage.plot()

right_wing_vane0.plot()
left_wing_vane1.plot()
right_horizontal_tail_vane2.plot()
left_horizontal_tail_vane3.plot()
right_canard_vane4.plot()
left_canard_vane5.plot()


# Mesh definitions
# num_ribs = 21
# num_chordwise = 5
# forward_spar_upper = wing.project(np.linspace(np.array([11., -14., 7.5]), np.array([11., 14., 7.5]), num_ribs), direction=np.array([0., 0., -1.]), plot=True)  # returns MappedArray
# forward_spar_lower = wing.project(np.linspace(np.array([11., -14., -7.5]), np.array([11., 14., -7.5]), num_ribs), direction=np.array([0., 0., 1.]), plot=True)  # returns MappedArray
# rear_spar_upper = wing.project(np.linspace(np.array([13., -14., 7.5]), np.array([13., 14., 7.5]), num_ribs), direction=np.array([0., 0., -1.]), plot=True)   # returns MappedArray
# rear_spar_lower = wing.project(np.linspace(np.array([13., -14., -7.5]), np.array([13., 14., -7.5]), num_ribs), direction=np.array([0., 0., 1.]), plot=True)   # returns MappedArray
# forward_spar = am.linspace(forward_spar_upper, forward_spar_lower, 2)
# rear_spar = am.linspace(rear_spar_upper, rear_spar_lower, 2)

# plotting_meshes = [forward_spar, rear_spar]
# spatial_rep.plot_meshes(plotting_meshes, mesh_plot_types=['wireframe'], mesh_opacity=1.)

