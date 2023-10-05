import csdl
import numpy as np
import scipy.sparse as sps

'''
NOTE: Going to start with it assuming even space between sections!!
'''

class AffineSectionPropertiesCSDL(csdl.Model):

    def initialize(self):
        self.parameters.declare('ffd_set')

    def define(self):
        ffd_set = self.parameters['ffd_set']
        ffd_blocks = ffd_set.active_ffd_blocks

        # ffd_set.setup(project_points=False)       I want SystemParam to perform setup as CSDL models are instantiated.
        # num_affine_dof = ffd_set.num_affine_dof
        num_affine_free_dof = ffd_set.num_affine_free_dof
        num_affine_prescribed_dof = ffd_set.num_affine_prescribed_dof
        num_affine_section_properties = ffd_set.num_affine_section_properties
        num_affine_sections = ffd_set.num_affine_sections

        # Create variables for free and prescribed dof
        # Get sparse Python maps for (free dof --> section properties) and (prescribed dof --> section_properties)
        if num_affine_free_dof != 0:
            ffd_free_dof = self.declare_variable('ffd_free_dof', val=ffd_set.free_affine_dof)
            free_section_properties_map = ffd_set.free_section_properties_map
        if num_affine_prescribed_dof != 0:
            ffd_prescribed_dof = self.create_output(f'ffd_prescribed_dof', val=ffd_set.prescribed_affine_dof)
            prescribed_affine_section_properties_map = ffd_set.prescribed_affine_section_properties_map

        # Connect in prescribed dof whether they are inputs or connections.
        parameter_starting_index = 0
        for ffd_block in list(ffd_blocks.values()):
            if ffd_block.num_affine_prescribed_dof == 0:
                continue

            for parameter_name, parameter in ffd_block.parameters.items():
                if parameter.property_type == 'rotation_u' \
                    or parameter.property_type == 'rotation_v' \
                    or parameter.property_type == 'rotation_w':
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
                ffd_prescribed_dof[parameter_starting_index:parameter_ending_index] = dof

                parameter_starting_index = parameter_ending_index

        # Evaluate maps
        if num_affine_free_dof != 0:
            ffd_free_section_properties_without_initial = csdl.matvec(free_section_properties_map, ffd_free_dof)
        else:
            ffd_free_section_properties_without_initial = self.create_input('ffd_free_section_properties_without_initial', val=np.zeros((num_affine_section_properties,)))

        if num_affine_prescribed_dof != 0:
            ffd_prescribed_section_properties_without_initial = csdl.matvec(prescribed_affine_section_properties_map, ffd_prescribed_dof)
        else:
            ffd_prescribed_section_properties_without_initial = self.create_input('ffd_prescribed_section_properties_without_initial', val=np.zeros((num_affine_section_properties,)))


        # Add initial value of 1 to the scaling properties
        if ffd_set.num_affine_section_properties != 0:
            num_active_ffd_block_affine_section_properties = 0
            for ffd_block in list(ffd_blocks.values()):
                num_active_ffd_block_affine_section_properties += ffd_block.num_sections*6  # 6 is number of affine section properties
            # ffd_section_properties = self.create_output(name=f'ffd_affine_section_properties', shape=(num_active_ffd_block_affine_section_properties,))
            ffd_section_properties = self.create_output(name=f'ffd_affine_section_properties', val=ffd_set.affine_section_properties)
        else:
            return
        ffd_block_starting_index = 0
        ffd_block_starting_index_for_properties = 0
        for ffd_block in list(ffd_blocks.values()):
            ffd_block_ending_index = ffd_block_starting_index + ffd_block.num_sections * ffd_block.num_affine_properties
            NUM_SCALING_PROPERTIES = 3
            ffd_block_scaling_properties_starting_index = ffd_block_starting_index + ffd_block.num_sections*(ffd_block.num_affine_properties-NUM_SCALING_PROPERTIES)    #The last 2 properties are scaling
            ffd_block_scaling_properties_ending_index = ffd_block_scaling_properties_starting_index + ffd_block.num_sections*(NUM_SCALING_PROPERTIES)

            if ffd_block.num_affine_dof != 0:
                ffd_block_ending_index_for_properties = ffd_block_starting_index_for_properties + ffd_block.num_sections * ffd_block.num_affine_properties
                ffd_block_scaling_properties_starting_index_for_properties = ffd_block_starting_index + ffd_block.num_sections*(ffd_block.num_affine_properties-NUM_SCALING_PROPERTIES)    #The last 2 properties are scaling
                ffd_block_scaling_properties_ending_index_for_properties = ffd_block_scaling_properties_starting_index + ffd_block.num_sections*(NUM_SCALING_PROPERTIES)

                # Use calculated values for non-scaling parameters
                ffd_section_properties[ffd_block_starting_index:ffd_block_scaling_properties_starting_index] = \
                    ffd_free_section_properties_without_initial[ffd_block_starting_index_for_properties:ffd_block_scaling_properties_starting_index_for_properties] \
                    + ffd_prescribed_section_properties_without_initial[ffd_block_starting_index_for_properties:ffd_block_scaling_properties_starting_index_for_properties]
                
                # Add 1 to scaling parameters to make initial scaling=1.
                ffd_section_properties[ffd_block_scaling_properties_starting_index:ffd_block_scaling_properties_ending_index] = \
                    ffd_free_section_properties_without_initial[ffd_block_scaling_properties_starting_index_for_properties:ffd_block_scaling_properties_ending_index_for_properties] \
                    + ffd_prescribed_section_properties_without_initial[ffd_block_scaling_properties_starting_index_for_properties:ffd_block_scaling_properties_ending_index_for_properties] \
                    + 1.  # adding 1 which is initial scale value
                
                ffd_block_starting_index_for_properties = ffd_block_ending_index_for_properties

            # else:
            #     ffd_section_properties[ffd_block_starting_index:ffd_block_scaling_properties_starting_index] = 0.
            #     ffd_section_properties[ffd_block_scaling_properties_starting_index:ffd_block_scaling_properties_ending_index] = 1.


            ffd_block_starting_index = ffd_block_ending_index

        # Get dedicated translations vector to allow rotations model to rotate around correct origin
        if ffd_set.num_affine_sections != 0:
            ffd_translations = self.create_output('ffd_translations', val=np.zeros_like(ffd_set.translations))   # 3 is number of translational properties
        translations_starting_index = 0
        section_property_starting_index = 0
        for ffd_block in list(ffd_blocks.values()):
            if ffd_block.num_affine_dof == 0:
                continue

            translations_ending_index = translations_starting_index + ffd_block.num_sections*3
            section_property_ending_index = section_property_starting_index + ffd_block.num_sections*3
            ffd_translations[translations_starting_index:translations_ending_index] = ffd_section_properties[section_property_starting_index:section_property_ending_index]

            translations_starting_index = translations_ending_index
            section_property_starting_index = section_property_starting_index + ffd_block.num_sections * ffd_block.num_affine_properties


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

    wing_ffd_block.add_scale_v(name='linear_taper', order=2, num_dof=3, value=np.array([0., 1., 0.]), cost_factor=1.)
    wing_ffd_block.add_translation_w(name='wingtip_translation', order=4, num_dof=10, value=-1/2*np.array([-2., 0., 0., 0., 0., 0., 0., 0., 0., -2.]))

    from lsdo_geo.caddee_core.system_parameterization.free_form_deformation.ffd_set import SRBGFFDSet
    ffd_set = SRBGFFDSet(name='ffd_set', ffd_blocks={wing_ffd_block.name : wing_ffd_block})

    ffd_set.setup(project_embedded_entities=False)
    affine_section_properties = wing_ffd_block.evaluate_affine_section_properties()
    print('Python evaluation: affine section properties: \n', affine_section_properties)

    sim = Simulator(AffineSectionPropertiesCSDL(ffd_set=ffd_set))
    sim.run()
    # sim.visualize_implementation()        # Only csdl_om can do this

    print('CSDL evaluation: affine section properties: \n', sim['ffd_affine_section_properties'])
    print("Python and CSDL difference", np.linalg.norm(sim['ffd_affine_section_properties'] - affine_section_properties))

    '''
    Multiple FFD blocks
    '''
    # system_representation = SystemRepresentation()
    # spatial_rep = system_representation.spatial_representation
    # file_path = 'models/stp/'
    # spatial_rep.import_file(file_name=file_path+'lift_plus_cruise_final_3.stp')

    # # Create Components
    # from lsdo_geo.caddee_core.system_representation.component.component import LiftingSurface
    # wing_primitive_names = list(spatial_rep.get_primitives(search_names=['Wing']).keys())
    # wing = LiftingSurface(name='wing', spatial_representation=spatial_rep, primitive_names=wing_primitive_names)  # TODO add material arguments
    # tail_primitive_names = list(spatial_rep.get_primitives(search_names=['Tail_1']).keys())
    # horizontal_stabilizer = LiftingSurface(name='tail', spatial_representation=spatial_rep, primitive_names=tail_primitive_names)  # TODO add material arguments
    # system_representation.add_component(wing)
    # system_representation.add_component(horizontal_stabilizer)

    # # # Parameterization
    # from lsdo_geo.caddee_core.system_parameterization.free_form_deformation.ffd_functions import create_cartesian_enclosure_volume
    # from lsdo_geo.caddee_core.system_parameterization.free_form_deformation.ffd_block import SRBGFFDBlock

    # wing_geometry_primitives = wing.get_geometry_primitives()
    # wing_ffd_b_spline_volume = create_cartesian_enclosure_volume(wing_geometry_primitives, num_coefficients=(11, 2, 2), order=(4,2,2), xyz_to_uvw_indices=(1,0,2))
    # wing_ffd_block = SRBGFFDBlock(name='wing_ffd_block', primitive=wing_ffd_b_spline_volume, embedded_entities=wing_geometry_primitives)
    # wing_ffd_block.add_scale_v(name='linear_taper', order=2, num_dof=3, value=np.array([0., 1., 0.]), cost_factor=1.)

    # horizontal_stabilizer_geometry_primitives = horizontal_stabilizer.get_geometry_primitives()
    # horizontal_stabilizer_ffd_b_spline_volume = create_cartesian_enclosure_volume(horizontal_stabilizer_geometry_primitives, num_coefficients=(11, 2, 2), order=(4,2,2), xyz_to_uvw_indices=(1,0,2))
    # horizontal_stabilizer_ffd_block = SRBGFFDBlock(name='horizontal_stabilizer_ffd_block', primitive=horizontal_stabilizer_ffd_b_spline_volume, embedded_entities=horizontal_stabilizer_geometry_primitives)
    # horizontal_stabilizer_ffd_block.add_scale_v(name='horizontal_stabilizer_linear_taper', order=2, num_dof=3, value=np.array([0.5, 0.5, 0.5]), cost_factor=1.)

    # # plotting_elements = wing_ffd_block.plot(plot_embedded_entities=False, show=False)
    # # plotting_elements = horizontal_stabilizer_ffd_block.plot(plot_embedded_entities=False, show=False, additional_plotting_elements=plotting_elements)
    # # spatial_rep.plot(additional_plotting_elements=plotting_elements)

    # from lsdo_geo.caddee_core.system_parameterization.free_form_deformation.ffd_set import SRBGFFDSet
    # ffd_set = SRBGFFDSet(name='ffd_set', ffd_blocks={wing_ffd_block.name : wing_ffd_block, horizontal_stabilizer_ffd_block.name : horizontal_stabilizer_ffd_block})

    # ffd_set.setup(project_embedded_entities=False)
    # affine_section_properties = ffd_set.evaluate_affine_section_properties()
    # print('Python evaluation: affine section properties: \n', affine_section_properties)

    # sim = Simulator(AffineSectionPropertiesCSDL(ffd_set=ffd_set))
    # sim.run()
    # # sim.visualize_implementation()    $ Only usable with csdl_om

    # print('CSDL evaluation: affine section properties: \n', sim['ffd_affine_section_properties'])
    # print("Python and CSDL difference", np.linalg.norm(sim['ffd_affine_section_properties'] - affine_section_properties))
                
