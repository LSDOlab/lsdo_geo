import csdl
from csdl_om import Simulator
import numpy as np
import scipy.sparse as sps

class FFDEvaluationCSDL(csdl.Model):
    '''
    Performs the scalings and translations based on the affine section properties.
    '''

    def initialize(self):
        self.parameters.declare('ffd_set')

    def define(self):
        ffd_set = self.parameters['ffd_set']

        # Declare input
        ffd_control_points = self.declare_variable('ffd_control_points', val=ffd_set.control_points)

        # Get maps (section properties --> ffd control points)
        embedded_entities_map = ffd_set.embedded_entities_map

        # Evaluate maps to get FFD control points
        if ffd_set.num_dof != 0:
            embedded_entities = csdl.sparsematmat(ffd_control_points, sparse_mat=embedded_entities_map)
            self.register_output("ffd_embedded_entities", embedded_entities)
        else:   # Purely so CSDL doesn't throw an error for the model not doing anything
            self.create_input('dummy_input_ffd_evaluation', val=0.)


if __name__ == "__main__":
    import csdl
    # from csdl_om import Simulator
    from python_csdl_backend import Simulator
    import numpy as np
    from vedo import Points, Plotter

    from caddee.caddee_core.system_representation.system_representation import SystemRepresentation
    system_representation = SystemRepresentation()
    spatial_rep = system_representation.spatial_representation
    # from caddee.caddee_core.system_parameterization.system_parameterization import SystemParameterization
    # system_parameterization = SystemParameterization()

    '''
    Single FFD Block
    '''
    file_path = 'models/stp/'
    spatial_rep.import_file(file_name=file_path+'rect_wing.stp')

    # Create Components
    from caddee.caddee_core.system_representation.component.component import LiftingSurface, Component
    wing_primitive_names = list(spatial_rep.get_primitives(search_names=['Wing']).keys())
    wing = LiftingSurface(name='wing', spatial_representation=spatial_rep, primitive_names=wing_primitive_names)  # TODO add material arguments
    system_representation.add_component(wing)

    # # Parameterization
    from caddee.caddee_core.system_parameterization.free_form_deformation.ffd_functions import create_cartesian_enclosure_volume
    from caddee.caddee_core.system_parameterization.free_form_deformation.ffd_block import SRBGFFDBlock

    wing_geometry_primitives = wing.get_geometry_primitives()
    wing_ffd_bspline_volume = create_cartesian_enclosure_volume(wing_geometry_primitives, num_control_points=(11, 2, 2), order=(4,2,2), xyz_to_uvw_indices=(1,0,2))
    wing_ffd_block = SRBGFFDBlock(name='wing_ffd_block', primitive=wing_ffd_bspline_volume, embedded_entities=wing_geometry_primitives)

    wing_ffd_block.add_rotation_u(name='twist_distribution', order=4, num_dof=10, value=-1/2*np.array([0., 0.11, 0.22, 0.33, 0.44, 0.44, 0.33, 0.22, 0.11, 0.]))
    wing_ffd_block.add_rotation_v(name='wingtip_twist', order=4, num_dof=10, value=-np.array([np.pi/2, 0., 0., 0., 0., 0., 0., 0., 0., -np.pi/2]))
    wing_ffd_block.add_translation_w(name='wingtip_translation', order=4, num_dof=10, value=np.array([2., 0., 0., 0., 0., 0., 0., 0., 0., 2.]))

    from caddee.caddee_core.system_parameterization.free_form_deformation.ffd_set import SRBGFFDSet
    ffd_set = SRBGFFDSet(name='ffd_set', ffd_blocks={wing_ffd_block.name : wing_ffd_block})

    ffd_set.setup(project_embedded_entities=True)
    affine_section_properties = ffd_set.evaluate_affine_section_properties()
    affine_deformed_ffd_control_points = ffd_set.evaluate_affine_block_deformations()
    rotational_section_properties = ffd_set.evaluate_rotational_section_properties()
    rotated_ffd_control_points = ffd_set.evaluate_rotational_block_deformations()
    ffd_control_points = ffd_set.evaluate_control_points()
    embedded_entities = ffd_set.evaluate_embedded_entities()
    # print('Python evaluation: embedded entities: \n', embedded_entities)

    sim = Simulator(FFDEvaluationCSDL(ffd_set=ffd_set))
    sim.run()
    # sim.visualize_implementation()        # Only csdl_om can do this

    # print('CSDL evaluation: ffd embedded entities: \n', sim['ffd_embedded_entities'])
    print("Python and CSDL difference", np.linalg.norm(sim['ffd_embedded_entities'] - embedded_entities))

    spatial_rep.update(sim['ffd_embedded_entities'])
    plotting_elements = wing_ffd_block.plot_sections(control_points=ffd_control_points.reshape(wing_ffd_bspline_volume.shape), plot_embedded_entities=False, opacity=0.75, show=False)
    spatial_rep.plot(additional_plotting_elements=plotting_elements)

    '''
    Multiple FFD blocks
    '''
    system_representation = SystemRepresentation()
    spatial_rep = system_representation.spatial_representation
    file_path = 'models/stp/'
    spatial_rep.import_file(file_name=file_path+'lift_plus_cruise_final_3.stp')

    # Create Components
    from caddee.caddee_core.system_representation.component.component import LiftingSurface
    wing_primitive_names = list(spatial_rep.get_primitives(search_names=['Wing']).keys())
    wing = LiftingSurface(name='wing', spatial_representation=spatial_rep, primitive_names=wing_primitive_names)  # TODO add material arguments
    tail_primitive_names = list(spatial_rep.get_primitives(search_names=['Tail_1']).keys())
    horizontal_stabilizer = LiftingSurface(name='tail', spatial_representation=spatial_rep, primitive_names=tail_primitive_names)  # TODO add material arguments
    system_representation.add_component(wing)
    system_representation.add_component(horizontal_stabilizer)

    # # Parameterization
    from caddee.caddee_core.system_parameterization.free_form_deformation.ffd_functions import create_cartesian_enclosure_volume
    from caddee.caddee_core.system_parameterization.free_form_deformation.ffd_block import SRBGFFDBlock

    wing_geometry_primitives = wing.get_geometry_primitives()
    wing_ffd_bspline_volume = create_cartesian_enclosure_volume(wing_geometry_primitives, num_control_points=(11, 2, 2), order=(4,2,2), xyz_to_uvw_indices=(1,0,2))
    wing_ffd_block = SRBGFFDBlock(name='wing_ffd_block', primitive=wing_ffd_bspline_volume, embedded_entities=wing_geometry_primitives)
    wing_ffd_block.add_rotation_u(name='twist_distribution', order=4, num_dof=10, value=-1/2*np.array([0., 0.11, 0.22, 0.33, 0.44, 0.44, 0.33, 0.22, 0.11, 0.]))

    horizontal_stabilizer_geometry_primitives = horizontal_stabilizer.get_geometry_primitives()
    horizontal_stabilizer_ffd_bspline_volume = create_cartesian_enclosure_volume(horizontal_stabilizer_geometry_primitives, num_control_points=(11, 2, 2), order=(4,2,2), xyz_to_uvw_indices=(1,0,2))
    horizontal_stabilizer_ffd_block = SRBGFFDBlock(name='horizontal_stabilizer_ffd_block', primitive=horizontal_stabilizer_ffd_bspline_volume, embedded_entities=horizontal_stabilizer_geometry_primitives)
    horizontal_stabilizer_ffd_block.add_rotation_u(name='horizontal_stabilizer_twist_distribution', order=1, num_dof=1, value=np.array([np.pi/10]))

    from caddee.caddee_core.system_parameterization.free_form_deformation.ffd_set import SRBGFFDSet
    ffd_set = SRBGFFDSet(name='ffd_set', ffd_blocks={wing_ffd_block.name : wing_ffd_block, horizontal_stabilizer_ffd_block.name : horizontal_stabilizer_ffd_block})

    ffd_set.setup(project_embedded_entities=True)
    affine_section_properties = ffd_set.evaluate_affine_section_properties()
    rotational_section_properties = ffd_set.evaluate_rotational_section_properties()
    affine_deformed_ffd_control_points = ffd_set.evaluate_affine_block_deformations()
    rotated_ffd_control_points = ffd_set.evaluate_rotational_block_deformations()
    ffd_control_points = ffd_set.evaluate_control_points()
    ffd_embedded_entities = ffd_set.evaluate_embedded_entities()
    # print('Python evaluation: FFD embedded entities: \n', rotated_ffd_control_points)

    sim = Simulator(FFDEvaluationCSDL(ffd_set=ffd_set))
    sim.run()
    # sim.visualize_implementation()    $ Only usable with csdl_om

    # print('CSDL evaluation: FFD embedded entities: \n', sim['ffd_embedded_entities'])
    print("Python and CSDL difference", np.linalg.norm(sim['ffd_embedded_entities'] - ffd_embedded_entities))

    updated_primitives_names = wing.primitive_names.copy()
    updated_primitives_names.extend(horizontal_stabilizer.primitive_names.copy())
    spatial_rep.update(sim['ffd_embedded_entities'], updated_primitives_names)
    
    plotting_elements = wing_ffd_block.plot_sections(
        control_points=(ffd_control_points[:wing_ffd_block.num_control_points,:]).reshape(wing_ffd_bspline_volume.shape),
        plot_embedded_entities=False, opacity=0.75, show=False)
    plotting_elements = horizontal_stabilizer_ffd_block.plot_sections(
        control_points=(ffd_control_points[wing_ffd_block.num_control_points:,:]).reshape(wing_ffd_bspline_volume.shape), 
        plot_embedded_entities=False, opacity=0.75, additional_plotting_elements=plotting_elements, show=False)
    spatial_rep.plot(additional_plotting_elements=plotting_elements)
