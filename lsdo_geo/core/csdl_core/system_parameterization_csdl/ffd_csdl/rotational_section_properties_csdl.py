import csdl
import numpy as np
import scipy.sparse as sps

'''
NOTE: Going to start with it assuming even space between sections!!
'''

class RotationalSectionPropertiesCSDL(csdl.Model):

    def initialize(self):
        self.parameters.declare('ffd_set')

    def define(self):
        ffd_set = self.parameters['ffd_set']
        ffd_blocks = ffd_set.ffd_blocks
        
        # Create variables for free and prescribed dof
        if ffd_set.num_rotational_dof != 0:
            ffd_rotational_dof = self.create_output('ffd_rotational_dof', val=ffd_set.rotational_dof)
        else:   # Purely so CSDL doesn't throw an error for the model not doing anything
            self.create_input('dummy_input_rotational_section_properties', val=0.)
        rotational_section_properties_map = ffd_set.rotational_section_properties_map

        # Connect in prescribed dof whether they are inputs or connections.
        parameter_starting_index = 0
        for ffd_block in list(ffd_blocks.values()):
            if ffd_block.num_rotational_dof == 0:
                continue

            for parameter_name, parameter in ffd_block.parameters.items():
                if parameter.property_type != 'rotation_u' \
                    and parameter.property_type != 'rotation_v' \
                    and parameter.property_type != 'rotation_w':
                    continue

                if parameter.connection_name is not None:
                    if parameter.value is not None:
                        dof = self.declare_variable(parameter.connection_name, val=parameter.value) # SystemParameterization makes this connection.
                    else:
                        dof = self.declare_variable(parameter.connection_name, shape=(parameter.num_dof,))
                else:
                    if parameter.value is not None:
                        dof = self.create_input(parameter_name, val=parameter.value)
                        # dof = self.create_input(f'{ffd_block.name}_order_{parameter.order}_{parameter.property_type}', val=parameter.value)
                    else:   # no connection name and no value means it's not prescribed (it's free)
                        dof = self.create_input(parameter_name, shape=(parameter.num_dof,))

                
                parameter_ending_index = parameter_starting_index + parameter.num_dof
                ffd_rotational_dof[parameter_starting_index:parameter_ending_index] = dof

                parameter_starting_index = parameter_ending_index

        # Evaluate and register
        if ffd_set.num_rotational_dof != 0:
            rotational_section_properties = csdl.matvec(rotational_section_properties_map, ffd_rotational_dof)
            self.register_output('ffd_rotational_section_properties', rotational_section_properties)


if __name__ == "__main__":
    import csdl
    # from csdl_om import Simulator
    from python_csdl_backend import Simulator
    import numpy as np
    from vedo import Points, Plotter

    from lsdo_geo.caddee_core.system_representation.system_representation import SystemRepresentation
    system_representation = SystemRepresentation()
    spatial_rep = system_representation.spatial_representation
    # from lsdo_geo.caddee_core.system_parameterization.system_parameterization import SystemParameterization
    # system_parameterization = SystemParameterization()

    '''
    Single FFD Block
    '''
    file_path = 'models/stp/'
    spatial_rep.import_file(file_name=file_path+'rect_wing.stp')

    # Create Components
    from lsdo_geo.caddee_core.system_representation.component.component import LiftingSurface, Component
    wing_primitive_names = list(spatial_rep.get_primitives(search_names=['Wing']).keys())
    wing = LiftingSurface(name='wing', spatial_representation=spatial_rep, primitive_names=wing_primitive_names)  # TODO add material arguments
    system_representation.add_component(wing)

    # # Parameterization
    from lsdo_geo.caddee_core.system_parameterization.free_form_deformation.ffd_functions import create_cartesian_enclosure_volume
    from lsdo_geo.caddee_core.system_parameterization.free_form_deformation.ffd_block import SRBGFFDBlock

    wing_geometry_primitives = wing.get_geometry_primitives()
    wing_ffd_b_spline_volume = create_cartesian_enclosure_volume(wing_geometry_primitives, num_coefficients=(11, 2, 2), order=(4,2,2), xyz_to_uvw_indices=(1,0,2))
    wing_ffd_block = SRBGFFDBlock(name='wing_ffd_block', primitive=wing_ffd_b_spline_volume, embedded_entities=wing_geometry_primitives)

    wing_ffd_block.add_rotation_u(name='twist_distribution', order=4, num_dof=10, value=-1/2*np.array([0., 0.11, 0.22, 0.33, 0.44, 0.44, 0.33, 0.22, 0.11, 0.]))
    wing_ffd_block.add_rotation_v(name='wingtip_twist', order=4, num_dof=10, value=-1/2*np.array([np.pi/2, 0., 0., 0., 0., 0., 0., 0., 0., -np.pi/2]))

    from lsdo_geo.caddee_core.system_parameterization.free_form_deformation.ffd_set import SRBGFFDSet
    ffd_set = SRBGFFDSet(name='ffd_set', ffd_blocks={wing_ffd_block.name : wing_ffd_block})

    ffd_set.setup(project_embedded_entities=False)
    rotational_section_properties = wing_ffd_block.evaluate_rotational_section_properties()
    print('Python evaluation: rotational section properties: \n', rotational_section_properties)

    sim = Simulator(RotationalSectionPropertiesCSDL(ffd_set=ffd_set))
    sim.run()
    # sim.visualize_implementation()        # Only csdl_om can do this

    print('CSDL evaluation: rotational section properties: \n', sim['ffd_rotational_section_properties'])
    print("Python and CSDL difference", np.linalg.norm(sim['ffd_rotational_section_properties'] - rotational_section_properties))

    '''
    Multiple FFD blocks
    '''
    system_representation = SystemRepresentation()
    spatial_rep = system_representation.spatial_representation
    file_path = 'models/stp/'
    spatial_rep.import_file(file_name=file_path+'lift_plus_cruise_final_3.stp')

    # Create Components
    from lsdo_geo.caddee_core.system_representation.component.component import LiftingSurface
    wing_primitive_names = list(spatial_rep.get_primitives(search_names=['Wing']).keys())
    wing = LiftingSurface(name='wing', spatial_representation=spatial_rep, primitive_names=wing_primitive_names)  # TODO add material arguments
    tail_primitive_names = list(spatial_rep.get_primitives(search_names=['Tail_1']).keys())
    horizontal_stabilizer = LiftingSurface(name='tail', spatial_representation=spatial_rep, primitive_names=tail_primitive_names)  # TODO add material arguments
    system_representation.add_component(wing)
    system_representation.add_component(horizontal_stabilizer)

    # # Parameterization
    from lsdo_geo.caddee_core.system_parameterization.free_form_deformation.ffd_functions import create_cartesian_enclosure_volume
    from lsdo_geo.caddee_core.system_parameterization.free_form_deformation.ffd_block import SRBGFFDBlock

    wing_geometry_primitives = wing.get_geometry_primitives()
    wing_ffd_b_spline_volume = create_cartesian_enclosure_volume(wing_geometry_primitives, num_coefficients=(11, 2, 2), order=(4,2,2), xyz_to_uvw_indices=(1,0,2))
    wing_ffd_block = SRBGFFDBlock(name='wing_ffd_block', primitive=wing_ffd_b_spline_volume, embedded_entities=wing_geometry_primitives)
    wing_ffd_block.add_rotation_u(name='twist_distribution', order=4, num_dof=10, value=-1/2*np.array([0., 0.11, 0.22, 0.33, 0.44, 0.44, 0.33, 0.22, 0.11, 0.]))

    horizontal_stabilizer_geometry_primitives = horizontal_stabilizer.get_geometry_primitives()
    horizontal_stabilizer_ffd_b_spline_volume = create_cartesian_enclosure_volume(horizontal_stabilizer_geometry_primitives, num_coefficients=(11, 2, 2), order=(4,2,2), xyz_to_uvw_indices=(1,0,2))
    horizontal_stabilizer_ffd_block = SRBGFFDBlock(name='horizontal_stabilizer_ffd_block', primitive=horizontal_stabilizer_ffd_b_spline_volume, embedded_entities=horizontal_stabilizer_geometry_primitives)
    horizontal_stabilizer_ffd_block.add_rotation_u(name='horizontal_stabilizer_twist_distribution', order=1, num_dof=1, value=np.array([np.pi/10]))

    # plotting_elements = wing_ffd_block.plot(plot_embedded_entities=False, show=False)
    # plotting_elements = horizontal_stabilizer_ffd_block.plot(plot_embedded_entities=False, show=False, additional_plotting_elements=plotting_elements)
    # spatial_rep.plot(additional_plotting_elements=plotting_elements)

    from lsdo_geo.caddee_core.system_parameterization.free_form_deformation.ffd_set import SRBGFFDSet
    ffd_set = SRBGFFDSet(name='ffd_set', ffd_blocks={wing_ffd_block.name : wing_ffd_block, horizontal_stabilizer_ffd_block.name : horizontal_stabilizer_ffd_block})

    ffd_set.setup(project_embedded_entities=False)
    rotational_section_properties = ffd_set.evaluate_rotational_section_properties()
    print('Python evaluation: rotational section properties: \n', rotational_section_properties)

    sim = Simulator(RotationalSectionPropertiesCSDL(ffd_set=ffd_set))
    sim.run()
    # sim.visualize_implementation()    $ Only usable with csdl_om

    print('CSDL evaluation: rotational section properties: \n', sim['ffd_rotational_section_properties'])
    print("Python and CSDL difference", np.linalg.norm(sim['ffd_rotational_section_properties'] - rotational_section_properties))
                
