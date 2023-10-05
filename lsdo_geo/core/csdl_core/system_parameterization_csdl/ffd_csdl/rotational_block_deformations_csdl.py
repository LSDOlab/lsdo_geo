import csdl
from csdl_om import Simulator
import numpy as np

class RotationalBlockDeformationsCSDL(csdl.Model):
    '''
    Performs the sectional rotations based on the rotational section properties.
    '''

    def initialize(self):
        self.parameters.declare('ffd_set')

    def define(self):
        ffd_set = self.parameters['ffd_set']

        rotational_section_properties = self.declare_variable('ffd_rotational_section_properties', val=ffd_set.rotational_section_properties)
        translations_flattened = self.declare_variable('ffd_translations', val=ffd_set.translations)
        # translations = csdl.reshape(translations_flattened, (ffd_set.num_sections, 3))  # 3 translational properties # NOTE: This reshapes in wrong order.
        translations_opposite_shape = csdl.reshape(translations_flattened, (3,ffd_set.num_sections))    # 3 translational properties
        translations = csdl.reorder_axes(translations_opposite_shape, 'ij->ji')
        affine_deformed_coefficients = self.declare_variable('affine_deformed_ffd_coefficients', val=ffd_set.affine_deformed_coefficients)

        # For simplicity for now (and to avoid storage/construction of a huge, dense rotation tensor, call each FFD rotation separately)
        NUM_PARAMETRIC_DIMENSIONS = 3
        if ffd_set.num_dof != 0:
            rotated_coefficients = self.create_output('rotated_ffd_coefficients', shape=(ffd_set.num_coefficients,NUM_PARAMETRIC_DIMENSIONS))
        else:   # Purely so CSDL doesn't throw an error for the model not doing anything
            self.create_input('dummy_input_rotational_block_deformations', val=0.)
        starting_index_coefficients = 0
        starting_index_sections = 0
        starting_index_section_properties = 0
        for ffd_block in list(ffd_set.active_ffd_blocks.values()):
            ending_index_coefficients = starting_index_coefficients + ffd_block.num_coefficients
            ending_index_sections = starting_index_sections + ffd_block.num_sections
            ending_index_section_properties = starting_index_section_properties + ffd_block.num_sections*ffd_block.num_rotational_properties

            if ffd_block.num_rotational_dof == 0:
                rotated_coefficients[starting_index_coefficients:ending_index_coefficients,:] = affine_deformed_coefficients[starting_index_coefficients:ending_index_coefficients,:]*1.
                starting_index_coefficients = ending_index_coefficients
                starting_index_sections = ending_index_sections
                continue

            ffd_block_affine_deformed_coefficients = affine_deformed_coefficients[starting_index_coefficients:ending_index_coefficients,:]
            ffd_block_translations = translations[starting_index_sections:ending_index_sections,:]
            ffd_block_rotational_section_properties = rotational_section_properties[starting_index_section_properties:ending_index_section_properties]

            # Undo translations so section origin is at origin
            ffd_block_translations_expanded = csdl.expand(ffd_block_translations, ffd_block.primitive.shape, 'ij->iklj')
            affine_coefficients_local_frame_reshaped = csdl.reshape(ffd_block_affine_deformed_coefficients, ffd_block.primitive.shape)
            affine_coefficients_local_frame_without_translations = affine_coefficients_local_frame_reshaped - ffd_block_translations_expanded

            # Calculate rotation matrices (rotation_matrices_u, rotation_matrices_v, rotation_matrices_w), each (num_section,3,3)
            rotation_u = ffd_block_rotational_section_properties[:ffd_block.num_sections]
            rotation_v = ffd_block_rotational_section_properties[ffd_block.num_sections:2*ffd_block.num_sections]
            rotation_w = ffd_block_rotational_section_properties[2*ffd_block.num_sections:3*ffd_block.num_sections]

            sin_rotations_u = csdl.sin(rotation_u)
            cos_rotations_u = csdl.cos(rotation_u)
            sin_rotations_v = csdl.sin(rotation_v)
            cos_rotations_v = csdl.cos(rotation_v)
            sin_rotations_w = csdl.sin(rotation_w)
            cos_rotations_w = csdl.cos(rotation_w)

            # Rotation tensors are numpy arrays because they are very dense and the method used has fastest runtime.
            # -- The final matmtul we are doing is ijkl, ijlm --> ijkm
            # ---- i corresponds to sections, so we want element-wise operation
            # ---- j corresponds to points per section, so this is a repeated operation (like an extra column in dot)
            # ---- l is summed over corresponding to the matmul when normally applying matrix multiplication
            # -- For efficiency, we leave j axis out of map because it's the same map. We also loop over sections instead of tensordot for speed.
            identity_per_section = np.tile(np.eye(NUM_PARAMETRIC_DIMENSIONS), (ffd_block.num_sections,1,1))
            rotation_tensor_u = self.create_output(f'{ffd_block.name}_rotation_tensor_u', val=identity_per_section.copy())
            rotation_tensor_v = self.create_output(f'{ffd_block.name}_rotation_tensor_v', val=identity_per_section.copy())
            rotation_tensor_w = self.create_output(f'{ffd_block.name}_rotation_tensor_w', val=identity_per_section.copy())

            rotation_tensor_u[:,1,1] = csdl.expand(cos_rotations_u, cos_rotations_u.shape + (1,1), 'i->ijk')
            rotation_tensor_u[:,1,2] = csdl.expand(sin_rotations_u, sin_rotations_u.shape + (1,1), 'i->ijk')
            rotation_tensor_u[:,2,1] = csdl.expand(-sin_rotations_u, sin_rotations_u.shape + (1,1), 'i->ijk')
            rotation_tensor_u[:,2,2] = csdl.expand(cos_rotations_u, cos_rotations_u.shape + (1,1), 'i->ijk')

            rotation_tensor_v[:,0,0] = csdl.expand(cos_rotations_v, cos_rotations_v.shape + (1,1), 'i->ijk')
            rotation_tensor_v[:,0,2] = csdl.expand(-sin_rotations_v, sin_rotations_v.shape + (1,1), 'i->ijk')
            rotation_tensor_v[:,2,0] = csdl.expand(sin_rotations_v, sin_rotations_v.shape + (1,1), 'i->ijk')
            rotation_tensor_v[:,2,2] = csdl.expand(cos_rotations_v, cos_rotations_v.shape + (1,1), 'i->ijk')

            rotation_tensor_w[:,0,0] = csdl.expand(cos_rotations_w, cos_rotations_w.shape + (1,1), 'i->ijk')
            rotation_tensor_w[:,0,1] = csdl.expand(sin_rotations_w, sin_rotations_w.shape + (1,1), 'i->ijk')
            rotation_tensor_w[:,1,0] = csdl.expand(-sin_rotations_w, sin_rotations_w.shape + (1,1), 'i->ijk')
            rotation_tensor_w[:,1,1] = csdl.expand(cos_rotations_w, cos_rotations_w.shape + (1,1), 'i->ijk')

            rotated_coefficients_local_frame_without_translations = self.create_output(f'{ffd_block.name}_rotated_coefficients_local_frame_without_translations', shape=ffd_block.primitive.shape)
            for i in range(ffd_block.num_sections):
                # Combine x,y,z rotation maps
                section_rotation_matrix_u = csdl.reshape(rotation_tensor_u[i,:,:], rotation_tensor_u.shape[1:])
                section_rotation_matrix_v = csdl.reshape(rotation_tensor_v[i,:,:], rotation_tensor_v.shape[1:])
                section_rotation_matrix_w = csdl.reshape(rotation_tensor_w[i,:,:], rotation_tensor_w.shape[1:])
                section_rotation_matrix = csdl.matmat(section_rotation_matrix_u, csdl.matmat(section_rotation_matrix_v, section_rotation_matrix_w))

                # Apply rotation to section
                affine_coefficients_local_frame_without_translations_flattened = csdl.reshape(affine_coefficients_local_frame_without_translations[i,:,:,:], 
                                                                                    (ffd_block.num_coefficients_per_section, NUM_PARAMETRIC_DIMENSIONS))
                
                ffd_block_rotated_points_flattened = csdl.matmat(affine_coefficients_local_frame_without_translations_flattened, csdl.transpose(section_rotation_matrix))
                rotated_coefficients_local_frame_without_translations[i,:,:,:] = csdl.reshape(ffd_block_rotated_points_flattened, (1,)+ffd_block.primitive.shape[1:])

            # Add back on translations from the affine transformation
            rotated_coefficients_local_frame_reshaped = rotated_coefficients_local_frame_without_translations + ffd_block_translations_expanded
            ffd_block_rotated_coefficients = csdl.reshape(rotated_coefficients_local_frame_reshaped, (ffd_block.num_coefficients,NUM_PARAMETRIC_DIMENSIONS))

            rotated_coefficients[starting_index_coefficients:ending_index_coefficients,:] = ffd_block_rotated_coefficients
            starting_index_coefficients = ending_index_coefficients
            starting_index_sections = ending_index_sections
            starting_index_section_properties = ending_index_section_properties


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
    wing_ffd_block.add_rotation_v(name='wingtip_twist', order=4, num_dof=10, value=-np.array([np.pi/2, 0., 0., 0., 0., 0., 0., 0., 0., -np.pi/2]))
    wing_ffd_block.add_translation_w(name='wingtip_translation', order=4, num_dof=10, value=np.array([2., 0., 0., 0., 0., 0., 0., 0., 0., 2.]))

    from lsdo_geo.caddee_core.system_parameterization.free_form_deformation.ffd_set import SRBGFFDSet
    ffd_set = SRBGFFDSet(name='ffd_set', ffd_blocks={wing_ffd_block.name : wing_ffd_block})

    ffd_set.setup(project_embedded_entities=False)
    affine_section_properties = ffd_set.evaluate_affine_section_properties()
    affine_deformed_ffd_coefficients = ffd_set.evaluate_affine_block_deformations()
    rotational_section_properties = ffd_set.evaluate_rotational_section_properties()
    rotated_ffd_coefficients = ffd_set.evaluate_rotational_block_deformations()
    print('Python evaluation: rotational deformed FFD coefficients: \n', rotated_ffd_coefficients)

    sim = Simulator(RotationalBlockDeformationsCSDL(ffd_set=ffd_set))
    sim.run()
    # sim.visualize_implementation()        # Only csdl_om can do this

    print('CSDL evaluation: rotational deformed FFD coefficients: \n', sim['rotated_ffd_coefficients'])
    print("Python and CSDL difference", np.linalg.norm(sim['rotated_ffd_coefficients'] - rotated_ffd_coefficients))

    wing_ffd_block.plot_sections(coefficients=sim['rotated_ffd_coefficients'].reshape(wing_ffd_b_spline_volume.shape), offset_sections=True, plot_embedded_entities=False, opacity=0.75, show=True)
    wing_ffd_block.plot_sections(coefficients=rotated_ffd_coefficients.reshape(wing_ffd_b_spline_volume.shape), offset_sections=True, plot_embedded_entities=False, opacity=0.75, show=True)
    # wing_ffd_b_spline_volume.coefficients = sim['rotated_ffd_coefficients'].reshape(wing_ffd_b_spline_volume.shape)
    # wing_ffd_b_spline_volume.plot()
    

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
    affine_section_properties = ffd_set.evaluate_affine_section_properties()
    rotational_section_properties = ffd_set.evaluate_rotational_section_properties()
    rotated_ffd_coefficients = ffd_set.evaluate_rotational_block_deformations()
    print('Python evaluation: rotational deformed FFD coefficients: \n', rotated_ffd_coefficients)

    sim = Simulator(RotationalBlockDeformationsCSDL(ffd_set=ffd_set))
    sim.run()
    # sim.visualize_implementation()    $ Only usable with csdl_om

    print('CSDL evaluation: rotational deformed FFD coefficients: \n', sim['rotated_ffd_coefficients'])
    print("Python and CSDL difference", np.linalg.norm(sim['rotated_ffd_coefficients'] - rotated_ffd_coefficients))

    wing_ffd_block.plot_sections(coefficients=(sim['rotated_ffd_coefficients'][0:11*2*2,:]).reshape(wing_ffd_b_spline_volume.shape), offset_sections=True, plot_embedded_entities=False, opacity=0.75, show=True)
    horizontal_stabilizer_ffd_block.plot_sections(coefficients=(sim['rotated_ffd_coefficients'][11*2*2:,:]).reshape(horizontal_stabilizer_ffd_b_spline_volume.shape), offset_sections=True, plot_embedded_entities=False, opacity=0.75, show=True)


    # wing_ffd_b_spline_volume.coefficients = (sim['rotated_ffd_coefficients'][0:11*2*2,:]).reshape(wing_ffd_b_spline_volume.shape)
    # wing_ffd_b_spline_volume.plot()
    # horizontal_stabilizer_ffd_b_spline_volume.coefficients = (sim['rotated_ffd_coefficients'][11*2*2:,:]).reshape(horizontal_stabilizer_ffd_b_spline_volume.shape)
    # horizontal_stabilizer_ffd_b_spline_volume.plot()
