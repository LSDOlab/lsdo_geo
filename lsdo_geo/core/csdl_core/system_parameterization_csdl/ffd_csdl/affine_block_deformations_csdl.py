import csdl
from csdl_om import Simulator
import numpy as np
import scipy.sparse as sps

class AffineBlockDeformationsCSDL(csdl.Model):
    '''
    Performs the scalings and translations based on the affine section properties.
    '''

    def initialize(self):
        self.parameters.declare('ffd_set')

    def define(self):
        ffd_set = self.parameters['ffd_set']

        # ffd_set.setup(project_points=False)

        # Declare input
        section_properties = self.declare_variable('ffd_affine_section_properties', val=ffd_set.affine_section_properties)

        # Get maps (section properties --> ffd control points)
        affine_block_deformations_map = ffd_set.affine_block_deformations_map

        # Evaluate maps to get FFD control points
        if ffd_set.num_dof != 0:
            affine_deformed_ffd_control_points_flattened = csdl.matvec(affine_block_deformations_map, section_properties)
            NUM_PARAMETRIC_DIMENSIONS = 3
            affine_deformed_ffd_control_points = csdl.reshape(affine_deformed_ffd_control_points_flattened, new_shape=(ffd_set.num_control_points, NUM_PARAMETRIC_DIMENSIONS))
            self.register_output("affine_deformed_ffd_control_points", affine_deformed_ffd_control_points)
        else:   # Purely so CSDL doesn't throw an error for the model not doing anything
            self.create_input('dummy_input_affine_block_deformations', val=0.)


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

    wing_ffd_block.add_scale_v(name='linear_taper', order=2, num_dof=3, value=np.array([0., 1., 0.]), cost_factor=1.)
    wing_ffd_block.add_translation_w(name='wingtip_translation', order=4, num_dof=10, value=-1/2*np.array([-2., 0., 0., 0., 0., 0., 0., 0., 0., -2.]))

    from caddee.caddee_core.system_parameterization.free_form_deformation.ffd_set import SRBGFFDSet
    ffd_set = SRBGFFDSet(name='ffd_set', ffd_blocks={wing_ffd_block.name : wing_ffd_block})

    ffd_set.setup(project_embedded_entities=False)
    affine_section_properties = ffd_set.evaluate_affine_section_properties()
    affine_deformed_ffd_control_points = ffd_set.evaluate_affine_block_deformations()
    print('Python evaluation: affine deformed FFD control_points: \n', affine_deformed_ffd_control_points)

    sim = Simulator(AffineBlockDeformationsCSDL(ffd_set=ffd_set))
    sim.run()
    # sim.visualize_implementation()        # Only csdl_om can do this

    print('CSDL evaluation: affine deformed FFD control_points: \n', sim['affine_deformed_ffd_control_points'])
    print("Python and CSDL difference", np.linalg.norm(sim['affine_deformed_ffd_control_points'] - affine_deformed_ffd_control_points))

    wing_ffd_block.plot_sections(control_points=sim['affine_deformed_ffd_control_points'].reshape(wing_ffd_bspline_volume.shape), offset_sections=True, plot_embedded_entities=False, opacity=0.75, show=True)
    # wing_ffd_bspline_volume.control_points = sim['affine_deformed_ffd_control_points'].reshape(wing_ffd_bspline_volume.shape)
    # wing_ffd_bspline_volume.plot()

    

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
    wing_ffd_block.add_scale_v(name='linear_taper', order=2, num_dof=3, value=np.array([0., 1., 0.]), cost_factor=1.)

    horizontal_stabilizer_geometry_primitives = horizontal_stabilizer.get_geometry_primitives()
    horizontal_stabilizer_ffd_bspline_volume = create_cartesian_enclosure_volume(horizontal_stabilizer_geometry_primitives, num_control_points=(11, 2, 2), order=(4,2,2), xyz_to_uvw_indices=(1,0,2))
    horizontal_stabilizer_ffd_block = SRBGFFDBlock(name='horizontal_stabilizer_ffd_block', primitive=horizontal_stabilizer_ffd_bspline_volume, embedded_entities=horizontal_stabilizer_geometry_primitives)
    horizontal_stabilizer_ffd_block.add_scale_v(name='horizontal_stabilizer_linear_taper', order=2, num_dof=3, value=np.array([0.5, 0.5, 0.5]), cost_factor=1.)

    # plotting_elements = wing_ffd_block.plot(plot_embedded_entities=False, show=False)
    # plotting_elements = horizontal_stabilizer_ffd_block.plot(plot_embedded_entities=False, show=False, additional_plotting_elements=plotting_elements)
    # spatial_rep.plot(additional_plotting_elements=plotting_elements)

    from caddee.caddee_core.system_parameterization.free_form_deformation.ffd_set import SRBGFFDSet
    ffd_set = SRBGFFDSet(name='ffd_set', ffd_blocks={wing_ffd_block.name : wing_ffd_block, horizontal_stabilizer_ffd_block.name : horizontal_stabilizer_ffd_block})

    ffd_set.setup(project_embedded_entities=False)
    affine_section_properties = ffd_set.evaluate_affine_section_properties()
    affine_deformed_ffd_control_points = ffd_set.evaluate_affine_block_deformations()
    print('Python evaluation: affine deformed FFD control_points: \n', affine_deformed_ffd_control_points)

    sim = Simulator(AffineBlockDeformationsCSDL(ffd_set=ffd_set))
    sim.run()
    # sim.visualize_implementation()    $ Only usable with csdl_om

    print('CSDL evaluation: affine deformed FFD control_points: \n', sim['affine_deformed_ffd_control_points'])
    print("Python and CSDL difference", np.linalg.norm(sim['affine_deformed_ffd_control_points'] - affine_deformed_ffd_control_points))

    wing_ffd_block.plot_sections(control_points=(sim['affine_deformed_ffd_control_points'][0:11*2*2,:]).reshape(wing_ffd_bspline_volume.shape), offset_sections=True, plot_embedded_entities=False, opacity=0.75, show=True)
    horizontal_stabilizer_ffd_block.plot_sections(control_points=(sim['affine_deformed_ffd_control_points'][11*2*2:,:]).reshape(horizontal_stabilizer_ffd_bspline_volume.shape), offset_sections=True, plot_embedded_entities=False, opacity=0.75, show=True)


    # wing_ffd_bspline_volume.control_points = (sim['affine_deformed_ffd_control_points'][0:11*2*2,:]).reshape(wing_ffd_bspline_volume.shape)
    # wing_ffd_bspline_volume.plot()
    # horizontal_stabilizer_ffd_bspline_volume.control_points = (sim['affine_deformed_ffd_control_points'][11*2*2:,:]).reshape(horizontal_stabilizer_ffd_bspline_volume.shape)
    # horizontal_stabilizer_ffd_bspline_volume.plot()
