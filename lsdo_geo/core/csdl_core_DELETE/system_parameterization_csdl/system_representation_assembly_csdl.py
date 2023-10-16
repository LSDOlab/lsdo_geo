import csdl
from csdl_om import Simulator
import numpy as np
import scipy.sparse as sps

from lsdo_geo.csdl_core.system_parameterization_csdl.ffd_csdl.ffd_csdl import FFDCSDL
from lsdo_geo.caddee_core.system_parameterization.free_form_deformation.ffd_set import FFDSet
from lsdo_geo.caddee_core.system_parameterization.free_form_deformation.ffd_set import SRBGFFDSet

class SystemRepresentationAssemblyCSDL(csdl.Model):
    '''
    Assembles the System Configuration from all of the System Parameterizations
    '''

    def initialize(self):
        self.parameters.declare('system_parameterization')

    def define(self):
        system_parameterization = self.parameters['system_parameterization']
        system_representation = system_parameterization.system_representation

        # system_representation_geometry = self.create_output('system_representation_geometry', val=system_representation.spatial_representation.coefficients)
        initial_system_representation_geometry = self.create_input('initial_system_representation_geometry', val=system_representation.spatial_representation.coefficients['geometry'])

        geometry_parameterizations = system_parameterization.geometry_parameterizations
        for geometry_parameterization in list(geometry_parameterizations.values()):
            if type(geometry_parameterization) is FFDSet or type(geometry_parameterization) is SRBGFFDSet:
                # NOTE: If there are multiple FFDSets, then there must be some details figured out in the variable names.
                parameterization_output = self.declare_variable('ffd_embedded_entities', val=geometry_parameterization.embedded_points)

                parameterization_indices = []
                for ffd_block in list(geometry_parameterization.active_ffd_blocks.values()):
                    ffd_block_embedded_primitive_names = list(ffd_block.embedded_entities.keys())
                    ffd_block_embedded_primitive_indices = []
                    for primitive_name in ffd_block_embedded_primitive_names:
                        ffd_block_embedded_primitive_indices.extend(list(system_representation.spatial_representation.primitive_indices[primitive_name]['geometry']))
                    parameterization_indices.extend(ffd_block_embedded_primitive_indices)

                # # system_representation_geometry[parameterization_indices] = parameterization_output   # NOTE: Approach 1
                # for i, index in enumerate(parameterization_indices):      # NOTE: Approach 2
                #     system_representation_geometry[int(index),:] = parameterization_output[i,:]

                # NOTE: Approach 3: TODO!!! This doesn't work for multiple parameterizations!!
                num_points_system_representation = initial_system_representation_geometry.shape[0]
                data = np.ones((len(parameterization_indices)))
                indexing_map = sps.coo_matrix((data, (np.array(parameterization_indices), np.arange(len(parameterization_indices)))),
                                              shape=(num_points_system_representation, len(parameterization_indices)))
                indexing_map = indexing_map.tocsc()
                if len(parameterization_indices) != 0:
                    updated_geometry_component = csdl.sparsematmat(parameterization_output, sparse_mat=indexing_map)

                num_unchanged_points = num_points_system_representation - len(parameterization_indices)
                if num_unchanged_points == 0:
                    system_representation_geometry = updated_geometry_component
                elif len(parameterization_indices) == 0:
                    system_representation_geometry = initial_system_representation_geometry*1
                else:
                    data = np.ones((num_unchanged_points,))
                    unchanged_indices = np.delete(np.arange(num_points_system_representation), parameterization_indices)
                    indexing_map = sps.coo_matrix((data, (unchanged_indices, unchanged_indices)),
                                                shape=(num_points_system_representation, num_points_system_representation))
                    unchanged_indexing_map = indexing_map.tocsc()
                    initial_geometry_component = csdl.sparsematmat(initial_system_representation_geometry, sparse_mat=unchanged_indexing_map)
                    system_representation_geometry = updated_geometry_component + initial_geometry_component
                self.register_output('system_representation_geometry', system_representation_geometry)
            else:
                continue

        # Similar type of assembly for material parameterization


if __name__ == "__main__":
    import csdl
    # from csdl_om import Simulator
    from python_csdl_backend import Simulator
    import numpy as np

    from lsdo_geo.caddee_core.system_representation.system_representation import SystemRepresentation
    from lsdo_geo.caddee_core.system_parameterization.system_parameterization import SystemParameterization
    system_representation = SystemRepresentation()
    spatial_rep = system_representation.spatial_representation
    system_parameterization = SystemParameterization(system_representation=system_representation)

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

    wing_ffd_set_primitives = wing.get_geometry_primitives()
    wing_ffd_b_spline_volume = create_cartesian_enclosure_volume(wing_ffd_set_primitives, num_coefficients=(11, 2, 2), order=(4,2,2), xyz_to_uvw_indices=(1,0,2))
    wing_ffd_block = SRBGFFDBlock(name='wing_ffd_block', primitive=wing_ffd_b_spline_volume, embedded_entities=wing_ffd_set_primitives)

    wing_ffd_block.add_rotation_u(name='twist_distribution', order=4, num_dof=10, value=-1/2*np.array([0., 0.11, 0.22, 0.33, 0.44, 0.44, 0.33, 0.22, 0.11, 0.]))
    wing_ffd_block.add_rotation_v(name='wingtip_twist', order=4, num_dof=10, value=-np.array([np.pi/2, 0., 0., 0., 0., 0., 0., 0., 0., -np.pi/2]))
    wing_ffd_block.add_translation_w(name='wingtip_translation', order=4, num_dof=10, value=np.array([2., 0., 0., 0., 0., 0., 0., 0., 0., 2.]))

    from lsdo_geo.caddee_core.system_parameterization.free_form_deformation.ffd_set import SRBGFFDSet
    ffd_set = SRBGFFDSet(name='ffd_set', ffd_blocks={wing_ffd_block.name : wing_ffd_block})
    system_parameterization.add_geometry_parameterization(ffd_set)
    system_parameterization.setup()

    affine_section_properties = ffd_set.evaluate_affine_section_properties()
    affine_deformed_ffd_coefficients = ffd_set.evaluate_affine_block_deformations()
    rotational_section_properties = ffd_set.evaluate_rotational_section_properties()
    rotated_ffd_coefficients = ffd_set.evaluate_rotational_block_deformations()
    ffd_coefficients = ffd_set.evaluate_coefficients()
    embedded_entities = ffd_set.evaluate_embedded_entities()
    # print('Python evaluation: embedded entities: \n', embedded_entities)

    sim = Simulator(SystemRepresentationAssemblyCSDL(system_parameterization=system_parameterization))
    sim.run()
    # sim.visualize_implementation()        # Only csdl_om can do this

    # print('CSDL evaluation: ffd embedded entities: \n', sim['ffd_embedded_entities'])
    # print("Python and CSDL difference", np.linalg.norm(sim['ffd_embedded_entities'] - embedded_entities))

    spatial_rep.update(sim['system_representation_geometry'])
    spatial_rep.plot()

    '''
    Multiple FFD blocks
    '''
    from lsdo_geo.caddee_core.system_representation.system_representation import SystemRepresentation
    from lsdo_geo.caddee_core.system_parameterization.system_parameterization import SystemParameterization
    system_representation = SystemRepresentation()
    spatial_rep = system_representation.spatial_representation
    system_parameterization = SystemParameterization(system_representation=system_representation)
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

    wing_ffd_set_primitives = wing.get_geometry_primitives()
    wing_ffd_b_spline_volume = create_cartesian_enclosure_volume(wing_ffd_set_primitives, num_coefficients=(11, 2, 2), order=(4,2,2), xyz_to_uvw_indices=(1,0,2))
    wing_ffd_block = SRBGFFDBlock(name='wing_ffd_block', primitive=wing_ffd_b_spline_volume, embedded_entities=wing_ffd_set_primitives)
    wing_ffd_block.add_rotation_u(name='twist_distribution', order=4, num_dof=10, value=-1/2*np.array([0., 0.11, 0.22, 0.33, 0.44, 0.44, 0.33, 0.22, 0.11, 0.]))
    wing_ffd_block.add_scale_v(name="chord_distribution_scaling", order=2, num_dof=3, value=np.array([0.5, 1.5, 0.5]))

    horizontal_stabilizer_ffd_set_primitives = horizontal_stabilizer.get_geometry_primitives()
    horizontal_stabilizer_ffd_b_spline_volume = create_cartesian_enclosure_volume(horizontal_stabilizer_ffd_set_primitives, num_coefficients=(11, 2, 2), order=(4,2,2), xyz_to_uvw_indices=(1,0,2))
    horizontal_stabilizer_ffd_block = SRBGFFDBlock(name='horizontal_stabilizer_ffd_block', primitive=horizontal_stabilizer_ffd_b_spline_volume, embedded_entities=horizontal_stabilizer_ffd_set_primitives)
    horizontal_stabilizer_ffd_block.add_rotation_u(name='horizontal_stabilizer_twist_distribution', order=1, num_dof=1, value=np.array([np.pi/10]))
    horizontal_stabilizer_ffd_block.add_scale_v(name="chord_distribution_scaling", order=2, num_dof=3, value=np.array([-0.5, 0.5, -0.5]))

    from lsdo_geo.caddee_core.system_parameterization.free_form_deformation.ffd_set import SRBGFFDSet
    ffd_set = SRBGFFDSet(name='ffd_set', ffd_blocks={wing_ffd_block.name : wing_ffd_block, horizontal_stabilizer_ffd_block.name : horizontal_stabilizer_ffd_block})
    system_parameterization.add_geometry_parameterization(ffd_set)
    system_parameterization.setup()
    
    affine_section_properties = ffd_set.evaluate_affine_section_properties()
    rotational_section_properties = ffd_set.evaluate_rotational_section_properties()
    affine_deformed_ffd_coefficients = ffd_set.evaluate_affine_block_deformations()
    rotated_ffd_coefficients = ffd_set.evaluate_rotational_block_deformations()
    ffd_coefficients = ffd_set.evaluate_coefficients()
    ffd_embedded_entities = ffd_set.evaluate_embedded_entities()
    # print('Python evaluation: FFD embedded entities: \n', rotated_ffd_coefficients)

    sim = Simulator(SystemRepresentationAssemblyCSDL(system_parameterization=system_parameterization))
    sim.run()
    # sim.visualize_implementation()    $ Only usable with csdl_om

    # print('CSDL evaluation: FFD embedded entities: \n', sim['ffd_embedded_entities'])
    # print("Python and CSDL difference", np.linalg.norm(sim['ffd_embedded_entities'] - ffd_embedded_entities))

    # updated_primitives_names = wing.primitive_names.copy()
    # updated_primitives_names.extend(horizontal_stabilizer.primitive_names.copy())
    # spatial_rep.update(sim['ffd_embedded_entities'], updated_primitives_names)
    spatial_rep.update(sim['system_representation_geometry'])

    spatial_rep.plot()
