import csdl
from csdl_om import Simulator
import numpy as np
import scipy.sparse as sps

# from lsdo_geo.csdl_core.system_parameterization_csdl.ffd_csdl.affine_section_properties_csdl import AffineSectionPropertiesCSDL
from lsdo_geo.csdl_core.system_parameterization_csdl.ffd_csdl.affine_section_properties_csdl import AffineSectionPropertiesCSDL
from lsdo_geo.csdl_core.system_parameterization_csdl.ffd_csdl.rotational_section_properties_csdl import RotationalSectionPropertiesCSDL
from lsdo_geo.csdl_core.system_parameterization_csdl.ffd_csdl.affine_block_deformations_csdl import AffineBlockDeformationsCSDL
from lsdo_geo.csdl_core.system_parameterization_csdl.ffd_csdl.rotational_block_deformations_csdl import RotationalBlockDeformationsCSDL
from lsdo_geo.csdl_core.system_parameterization_csdl.ffd_csdl.local_to_global_csdl import LocalToGlobalCSDL
from lsdo_geo.csdl_core.system_parameterization_csdl.ffd_csdl.ffd_evaluation_csdl import FFDEvaluationCSDL


class FFDCSDL(csdl.Model):
    '''
    Performs the scalings and translations based on the affine section properties.
    '''

    def initialize(self):
        self.parameters.declare('ffd_set')

    def define(self):
        ffd_set = self.parameters['ffd_set']

        # region Affine Section Properties Model
        affine_sectional_properties_model = AffineSectionPropertiesCSDL(ffd_set=ffd_set)
        self.add(submodel=affine_sectional_properties_model, name='affine_section_properties_model')
        # endregion

        # region Rotational Section Properties Model
        if ffd_set.num_rotational_dof != 0:
            rotational_section_properties_model = RotationalSectionPropertiesCSDL(ffd_set=ffd_set) 
            self.add(submodel=rotational_section_properties_model, name='rotational_section_properties_model')
        # endregion

        # region Affine Block Deformations Model
        affine_block_deformations_model = AffineBlockDeformationsCSDL(ffd_set=ffd_set)
        self.add(submodel=affine_block_deformations_model, name='affine_block_deformations_model')
        # endregion

        # region Rotational Block Deformations Model
        rotational_block_deformations_model = RotationalBlockDeformationsCSDL(ffd_set=ffd_set) 
        self.add(submodel=rotational_block_deformations_model, name='rotational_block_deformations_model')
        # endregion

        # region Local To Global Transformation Model
        local_to_global_transformation_model = LocalToGlobalCSDL(ffd_set=ffd_set) 
        self.add(submodel=local_to_global_transformation_model, name='local_to_global_transformation_model')
        # endregion

        # region FFD Evaluation Model
        ffd_evaluation_model = FFDEvaluationCSDL(ffd_set=ffd_set) 
        self.add(submodel=ffd_evaluation_model, name='ffd_evaluation_model')
        # endregion


if __name__ == "__main__":
    import csdl
    # from csdl_om import Simulator
    from python_csdl_backend import Simulator
    import numpy as np

    from lsdo_geo.caddee_core.system_representation.system_representation import SystemRepresentation
    system_representation = SystemRepresentation()
    spatial_rep = system_representation.spatial_representation

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

    ffd_set.setup(project_embedded_entities=True)
    # affine_section_properties = ffd_set.evaluate_affine_section_properties()
    # affine_deformed_ffd_coefficients = ffd_set.evaluate_affine_block_deformations()
    # rotational_section_properties = ffd_set.evaluate_rotational_section_properties()
    # rotated_ffd_coefficients = ffd_set.evaluate_rotational_block_deformations()
    # ffd_coefficients = ffd_set.evaluate_coefficients()
    # embedded_entities = ffd_set.evaluate_embedded_entities()
    # print('Python evaluation: embedded entities: \n', embedded_entities)

    sim = Simulator(FFDCSDL(ffd_set=ffd_set))
    sim.run()
    # sim.visualize_implementation()        # Only csdl_om can do this

    # print('CSDL evaluation: ffd embedded entities: \n', sim['ffd_embedded_entities'])
    # print("Python and CSDL difference", np.linalg.norm(sim['ffd_embedded_entities'] - embedded_entities))

    spatial_rep.update(sim['ffd_embedded_entities'])
    plotting_elements = wing_ffd_block.plot_sections(coefficients=sim['ffd_coefficients'].reshape(wing_ffd_b_spline_volume.shape), plot_embedded_entities=False, opacity=0.75, show=False)
    spatial_rep.plot(additional_plotting_elements=plotting_elements)

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

    ffd_set.setup(project_embedded_entities=True)
    # affine_section_properties = ffd_set.evaluate_affine_section_properties()
    # rotational_section_properties = ffd_set.evaluate_rotational_section_properties()
    # affine_deformed_ffd_coefficients = ffd_set.evaluate_affine_block_deformations()
    # rotated_ffd_coefficients = ffd_set.evaluate_rotational_block_deformations()
    # ffd_coefficients = ffd_set.evaluate_coefficients()
    # ffd_embedded_entities = ffd_set.evaluate_embedded_entities()
    # print('Python evaluation: FFD embedded entities: \n', rotated_ffd_coefficients)

    sim = Simulator(FFDCSDL(ffd_set=ffd_set))
    sim.run()
    # sim.visualize_implementation()    $ Only usable with csdl_om

    # print('CSDL evaluation: FFD embedded entities: \n', sim['ffd_embedded_entities'])
    # print("Python and CSDL difference", np.linalg.norm(sim['ffd_embedded_entities'] - ffd_embedded_entities))

    updated_primitives_names = wing.primitive_names.copy()
    updated_primitives_names.extend(horizontal_stabilizer.primitive_names.copy())
    spatial_rep.update(sim['ffd_embedded_entities'], updated_primitives_names)
    
    plotting_elements = wing_ffd_block.plot_sections(
        coefficients=(sim['ffd_coefficients'][:wing_ffd_block.num_coefficients,:]).reshape(wing_ffd_b_spline_volume.shape),
        plot_embedded_entities=False, opacity=0.75, show=False)
    plotting_elements = horizontal_stabilizer_ffd_block.plot_sections(
        coefficients=(sim['ffd_coefficients'][wing_ffd_block.num_coefficients:,:]).reshape(wing_ffd_b_spline_volume.shape), 
        plot_embedded_entities=False, opacity=0.75, additional_plotting_elements=plotting_elements, show=False)
    spatial_rep.plot(additional_plotting_elements=plotting_elements)
