import numpy as np
import pickle
from dataclasses import dataclass
from pathlib import Path
import pickle
import m3l
from lsdo_geo.splines.b_splines.b_spline_set import BSplineSet
from lsdo_geo.splines.b_splines.b_spline_sub_set import BSplineSubSet
from typing import Union

@dataclass
class Geometry(BSplineSet):

    def get_function_space(self):
        return self.space
    
    def copy(self):
        '''
        Creates a copy of the geometry that does not point to this geometry.
        '''
        if self.connections is not None:
            return Geometry(self.name+'_copy', self.space, self.coefficients.copy(), self.num_physical_dimensions.copy(),
                    self.coefficient_indices.copy(), self.connections.copy())
        else:
            return Geometry(self.name+'_copy', self.space, self.coefficients.copy(), self.num_physical_dimensions.copy(),
                    self.coefficient_indices.copy())

    def declare_component(self, component_name:str, b_spline_names:list[str]=None, b_spline_search_names:list[str]=None) -> BSplineSubSet:
        '''
        Declares a component. This component will point to a sub-set of the entire geometry.

        Parameters
        ----------
        component_name : str
            The name of the component.
        b_spline_names : list[str]
            The names of the B-splines that make up the component.
        b_spline_search_names : list[str], optional
            The names of the B-splines to search for. Names of B-splines will be returned for each B-spline that INCLUDES the search name.
        '''
        if b_spline_names is None:
            b_spline_names_input = []
        else:
            b_spline_names_input = b_spline_names.copy()

        if b_spline_search_names is not None:
            b_spline_names_input += self.space.search_b_spline_names(b_spline_search_names)

        sub_set = self.declare_sub_set(sub_set_name=component_name, b_spline_names=b_spline_names_input)
        return sub_set
    
    def create_sub_geometry(self, sub_geometry_name:str, b_spline_names:list[str]=None, b_spline_search_names:list[str]=None) -> BSplineSubSet:
        '''
        Creates a new geometry that is a subset of the current geometry.

        Parameters
        ----------
        component_name : str
            The name of the component.
        b_spline_names : list[str]
            The names of the B-splines that make up the component.
        b_spline_search_names : list[str], optional
            The names of the B-splines to search for. Names of B-splines will be returned for each B-spline that INCLUDES the search name.
        '''
        if b_spline_names is None:
            b_spline_names_input = []
        else:
            b_spline_names_input = b_spline_names.copy()

        if b_spline_search_names is not None:
            b_spline_names_input += self.space.search_b_spline_names(b_spline_search_names)

        sub_set = self.create_sub_set(sub_set_name=sub_geometry_name, b_spline_names=b_spline_names_input)
        return sub_set

    def import_geometry(self, file_name:str):
        '''
        Imports geometry from a file.

        Parameters
        ----------
        file_name : str
            The name of the file (with path) that containts the geometric information.
        '''
        from lsdo_geo.splines.b_splines.b_spline_functions import import_file, create_b_spline_set
        b_splines = import_file(file_name)
        b_spline_set = create_b_spline_set(self.name, b_splines)

        self.space = b_spline_set.space
        self.coefficients = b_spline_set.coefficients
        self.num_physical_dimensions = b_spline_set.num_physical_dimensions
        self.coefficient_indices = b_spline_set.coefficient_indices
        self.connections = b_spline_set.connections

    def refit(self, num_coefficients:tuple=(20,20), fit_resolution:tuple=(50,50), order:tuple=(4,4), parallelize:bool=True):
        '''
        Evaluates a grid over the geometry and finds the best set of coefficients/control points at the desired resolution to fit the geometry.

        Parameters
        ----------
        num_coefficients : tuple, optional
            The number of coefficients to use in each direction.
        fit_resolution : tuple, optional
            The number of points to evaluate in each direction for each B-spline to fit the geometry.
        order : tuple, optional
            The order of the B-splines to use in each direction.
        '''
        from lsdo_geo.splines.b_splines.b_spline_functions import refit_b_spline_set
        from lsdo_geo import REFIT_FOLDER

        saved_refit_file = Path(REFIT_FOLDER / f'{self.name}_{num_coefficients}_{order}_{fit_resolution}_refit_dict.pickle')
        if saved_refit_file.is_file():
            with open(saved_refit_file, 'rb') as handle:
                refit_dict = pickle.load(handle)
                b_spline_set = refit_dict['b_spline_set']

        else:
            refit_dict = {}
            b_spline_set = refit_b_spline_set(self, order, num_coefficients, fit_resolution, parallelize=parallelize)
            refit_dict['b_spline_set'] = b_spline_set
            with open(REFIT_FOLDER / f'{self.name}_{num_coefficients}_{order}_{fit_resolution}_refit_dict.pickle', 'wb+') as handle:
                pickle.dump(refit_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


        self.debugging_b_spline_set = b_spline_set

        self.space = b_spline_set.space
        self.coefficients = b_spline_set.coefficients
        self.coefficients = b_spline_set.coefficients
        self.num_physical_dimensions = b_spline_set.num_physical_dimensions
        self.coefficient_indices = b_spline_set.coefficient_indices
        self.connections = b_spline_set.connections

        self.num_coefficients = b_spline_set.num_coefficients

    
    def plot_meshes(self, meshes:list[m3l.Variable], mesh_plot_types:list[str]=['wireframe'], mesh_opacity:float=1., mesh_color:str='#F5F0E6',
                b_splines:list[str]=None, b_splines_plot_types:list[str]=['surface'], b_splines_opacity:float=0.25, b_splines_color:str='#00629B',
                b_splines_surface_texture:str="", additional_plotting_elements:list=[], camera:dict=None, show:bool=True):
        '''
        Plots a mesh over the geometry.

        Parameters
        ----------
        meshes : list
            A list of meshes to plot.
        mesh_plot_types : list, optional
            A list of plot types for each mesh. Options are 'wireframe', 'surface', and 'points'.
        mesh_opacity : float, optional
            The opacity of the mesh.
        mesh_color : str, optional
            The color of the mesh.
        b_splines : list, optional
            A list of b_splines to plot.
        b_splines_plot_types : list, optional
            A list of plot types for each primitive. Options are 'wireframe', 'surface', and 'points'.
        b_splines_opacity : float, optional
            The opacity of the b_splines.
        b_splines_color : str, optional
            The color of the b_splines.
        b_splines_surface_texture : str {"", "metallic", "glossy", "ambient",... see Vedo for more options}
            The surface texture for the primitive surfaces. (determines how light bounces off)
            More options: https://github.com/marcomusy/vedo/blob/master/examples/basic/lightings.py
        additional_plotting_elements : list, optional
            A list of additional plotting elements to plot.
        camera : dict, optional
            A dictionary of camera parameters. see Vedo documentation for more information.
        show : bool, optional
            Whether or not to show the plot.

        Returns
        -------
        plotting_elements : list
            A list of the vedo plotting elements.
        '''
        import vedo
        plotting_elements = additional_plotting_elements.copy()

        if not isinstance(meshes, list) and not isinstance(meshes, tuple):
            meshes = [meshes]

        # Create plotting meshes for b_splines
        plotting_elements = self.plot(b_splines=b_splines, plot_types=b_splines_plot_types, opacity=b_splines_opacity,
                                      color=b_splines_color, surface_texture=b_splines_surface_texture,
                                      additional_plotting_elements=plotting_elements,show=False)

        for mesh in meshes:
            if type(mesh) is m3l.Variable:
                points = mesh.value
            else:
                points = mesh

            if isinstance(mesh, tuple):
                # Is vector, so draw an arrow
                processed_points = ()
                for point in mesh:
                    if type(point) is m3l.Variable:
                        processed_points = processed_points + (point.value,)
                    else:
                        processed_points = processed_points + (point,)
                arrow = vedo.Arrow(tuple(processed_points[0].reshape((-1,))), 
                                   tuple((processed_points[0] + processed_points[1]).reshape((-1,))), s=0.05)
                plotting_elements.append(arrow)
                continue

            if 'point_cloud' in mesh_plot_types:
                num_points = np.cumprod(points.shape[:-1])[-1]
                plotting_elements.append(vedo.Points(points.reshape((num_points,-1)), r=4).color('#00C6D7'))

            if points.shape[0] == 1:
                points = points.reshape((points.shape[1:]))

            if len(points.shape) == 2:  # If it's a curve
                from vedo import Line
                plotting_elements.append(Line(points).color(mesh_color).linewidth(3))
                
                if 'wireframe' in mesh_plot_types:
                    num_points = np.cumprod(points.shape[:-1])[-1]
                    plotting_elements.append(vedo.Points(points.reshape((num_points,-1)), r=12).color(mesh_color))
                continue

            if ('surface' in mesh_plot_types or 'wireframe' in mesh_plot_types) and len(points.shape) == 3: # If it's a surface
                num_points_u = points.shape[0]
                num_points_v = points.shape[1]
                num_points = num_points_u*num_points_v
                vertices = []
                faces = []
                for u_index in range(num_points_u):
                    for v_index in range(num_points_v):
                        vertex = tuple(points[u_index,v_index,:])
                        vertices.append(vertex)
                        if u_index != 0 and v_index != 0:
                            face = tuple((
                                (u_index-1)*num_points_v+(v_index-1),
                                (u_index-1)*num_points_v+(v_index),
                                (u_index)*num_points_v+(v_index),
                                (u_index)*num_points_v+(v_index-1),
                            ))
                            faces.append(face)

                plotting_mesh = vedo.Mesh([vertices, faces]).opacity(mesh_opacity).color('lightblue')
            if 'surface' in mesh_plot_types:
                plotting_elements.append(plotting_mesh)
            if 'wireframe' in mesh_plot_types:
                # plotting_mesh = vedo.Mesh([vertices, faces]).opacity(mesh_opacity).color('blue')
                plotting_mesh = vedo.Mesh([vertices, faces]).opacity(mesh_opacity).color(mesh_color) # Default is UCSD Sand
                plotting_elements.append(plotting_mesh.wireframe().linewidth(3))
            
        if show:
            plotter = vedo.Plotter()
            plotter.show(plotting_elements, 'Meshes', axes=1, viewup="z", interactive=True, camera=camera)

        return plotting_elements
    

    def plot_2d_mesh(self, mesh):
        pass



if __name__ == "__main__":
    from lsdo_geo.core.geometry.geometry_functions import import_geometry
    # import array_mapper as am
    import m3l
    import time
    import scipy.sparse as sps

    # var1 = m3l.Variable('var1', shape=(2,3), value=np.array([[1., 2., 3.], [4., 5., 6.]]))
    # var2 = m3l.Variable('var2', shape=(2,3), value=np.array([[1., 2., 3.], [4., 5., 6.]]))
    # var3 = var1 + var2
    # print(var3)

    t1 = time.time()
    geometry = import_geometry('lsdo_geo/splines/b_splines/sample_geometries/rectangular_wing.stp', parallelize=False)
    # geometry = import_geometry('lsdo_geo/splines/b_splines/sample_geometries/lift_plus_cruise_final.stp')
    t2 = time.time()
    geometry.refit(parallelize=False)
    t3 = time.time()
    # geometry.find_connections() # NOTE: This is really really slow for large geometries. Come back to this.
    t4 = time.time()
    # geometry.plot()
    t5 = time.time()
    print('Import time: ', t2-t1)
    print('Refit time: ', t3-t2)
    # print('Find connections time: ', t4-t3)
    print('Plot time: ', t5-t4)

    projected_points1_coordinates = geometry.project(np.array([[0.2, 1., 10.], [0.5, 1., 1.]]), plot=False, direction=np.array([0., 0., -1.]))
    projected_points2_coordinates = geometry.project(np.array([[0.2, 0., 10.], [0.5, 1., 1.]]), plot=False, max_iterations=100)

    projected_points1 = geometry.evaluate(projected_points1_coordinates, plot=False)
    projected_points2 = geometry.evaluate(projected_points2_coordinates, plot=False)

    test_linspace = m3l.linspace(projected_points1, projected_points2)

    import vedo
    plotter = vedo.Plotter()
    plotting_points = vedo.Points(test_linspace.value.reshape(-1,3))
    geometry_plot = geometry.plot(show=False)
    # plotter.show(geometry_plot, plotting_points, interactive=True, axes=1)

    right_wing = geometry.declare_component(component_name='right_wing', b_spline_search_names=['WingGeom, 0'])
    # right_wing.plot()

    projected_points1_on_right_wing = right_wing.project(np.array([[0.2, 1., 10.], [0.5, 1., 1.]]), plot=False, direction=np.array([0., 0., -1.]))
    projected_points2_on_right_wing = right_wing.project(np.array([[0.2, 0., 1.], [0.5, 1., 1.]]), plot=False, max_iterations=100)

    left_wing = geometry.declare_component(component_name='left_wing', b_spline_search_names=['WingGeom, 1'])
    # left_wing.plot()

    # geometry2 = geometry.copy()
    # geometry2.coefficients = geometry.coefficients.copy()*2
    # geometry2.plot()
    # geometry.plot()

    # print(projected_points1_on_right_wing.evaluate(geometry2.coefficients))

    # left_wing_pressure_space = geometry.space.create_sub_space(sub_space_name='left_wing_pressure_space', b_spline_names=left_wing.b_spline_names)
    
    # # Manually creating a pressure distribution
    # pressure_coefficients = np.zeros((0,))
    # for b_spline_name in left_wing.b_spline_names:
    #     left_wing_geometry_coefficients = geometry.coefficients.value[geometry.coefficient_indices[b_spline_name]].reshape((-1,3))
    #     b_spline_pressure_coefficients = \
    #         -4*8.1/(np.pi*8.1)*np.sqrt(1 - (2*left_wing_geometry_coefficients[:,1]/8.1)**2) \
    #         * np.sqrt(1 - (2*left_wing_geometry_coefficients[:,0]/4)**2) \
    #         * (left_wing_geometry_coefficients[:,2]+0.05)
    #     pressure_coefficients = np.concatenate((pressure_coefficients, b_spline_pressure_coefficients.flatten()))

    # left_wing_pressure_function = left_wing_pressure_space.create_function(name='left_wing_pressure_function', 
    #                                                                        coefficients=pressure_coefficients, num_physical_dimensions=1)
    # left_wing.plot(color=left_wing_pressure_function)

    left_wing_pressure_space = geometry.space.create_sub_space(sub_space_name='left_wing_pressure_space',
                                                                            b_spline_names=left_wing.b_spline_names)
    # Manually creating a pressure distribution mesh
    pressure_mesh = np.zeros((0,))
    pressure_parametric_coordinates = []
    for b_spline_name in left_wing.b_spline_names:
        left_wing_geometry_coefficients = geometry.coefficients.value[geometry.coefficient_indices[b_spline_name]].reshape((-1,3))
        b_spline_pressure_coefficients = \
            -4*8.1/(np.pi*8.1)*np.sqrt(1 - (2*left_wing_geometry_coefficients[:,1]/8.1)**2) \
            * np.sqrt(1 - (2*left_wing_geometry_coefficients[:,0]/4)**2) \
            * (left_wing_geometry_coefficients[:,2]+0.05)
        pressure_mesh = np.concatenate((pressure_mesh, b_spline_pressure_coefficients.flatten()))

        # parametric_coordinates = left_wing.project(points=left_wing_geometry_coefficients, targets=[b_spline_name])
        # pressure_parametric_coordinates.extend(parametric_coordinates)

        b_spline_space = left_wing_pressure_space.spaces[left_wing_pressure_space.b_spline_to_space[b_spline_name]]

        b_spline_num_coefficients_u = b_spline_space.parametric_coefficients_shape[0]
        b_spline_num_coefficients_v = b_spline_space.parametric_coefficients_shape[1]

        u_vec = np.einsum('i,j->ij', np.linspace(0., 1., b_spline_num_coefficients_u), np.ones(b_spline_num_coefficients_u)).flatten()
        v_vec = np.einsum('i,j->ij', np.ones(b_spline_num_coefficients_v), np.linspace(0., 1., b_spline_num_coefficients_v)).flatten()
        parametric_coordinates = np.hstack((u_vec.reshape((-1,1)), v_vec.reshape((-1,1))))

        for i in range(len(parametric_coordinates)):
            pressure_parametric_coordinates.append(tuple((b_spline_name, parametric_coordinates[i,:])))

    pressure_coefficients = left_wing_pressure_space.fit_b_spline_set(fitting_points=pressure_mesh.reshape((-1,1)),
                                                                                  fitting_parametric_coordinates=pressure_parametric_coordinates,
                                                                                #   regularization_parameter=1.e-3)
                                                                                  regularization_parameter=0.)

    left_wing_pressure_function = left_wing_pressure_space.create_function(name='left_wing_pressure_function', 
                                                                           coefficients=pressure_coefficients, num_physical_dimensions=1)
    
    # left_wing.plot(color=left_wing_pressure_function)

    geometry3 = geometry.copy()
    axis_origin = geometry.evaluate(geometry.project(np.array([0.5, -10., 0.5])))
    axis_vector = geometry.evaluate(geometry.project(np.array([0.5, 10., 0.5]))) - axis_origin
    angles = 45
    # geometry3.coefficients = m3l.rotate(points=geometry3.coefficients.reshape((-1,3)), axis_origin=axis_origin, axis_vector=axis_vector,
    #                                     angles=angles, units='degrees').reshape((-1,))
    # geometry3.plot()

    leading_edge_parametric_coordinates = [
        ('WingGeom, 0, 3', np.array([1.,  0.])),
        ('WingGeom, 0, 3', np.array([0.777, 0.])),
        ('WingGeom, 0, 3', np.array([0.555, 0.])),
        ('WingGeom, 0, 3', np.array([0.333, 0.])),
        ('WingGeom, 0, 3', np.array([0.111, 0.])),
        ('WingGeom, 1, 8', np.array([0.111 , 0.])),
        ('WingGeom, 1, 8', np.array([0.333, 0.])),
        ('WingGeom, 1, 8', np.array([0.555, 0.])),
        ('WingGeom, 1, 8', np.array([0.777, 0.])),
        ('WingGeom, 1, 8', np.array([1., 0.])),
    ]

    trailing_edge_parametric_coordinates = [
        ('WingGeom, 0, 3', np.array([1.,  1.])),
        ('WingGeom, 0, 3', np.array([0.777, 1.])),
        ('WingGeom, 0, 3', np.array([0.555, 1.])),
        ('WingGeom, 0, 3', np.array([0.333, 1.])),
        ('WingGeom, 0, 3', np.array([0.111, 1.])),
        ('WingGeom, 1, 8', np.array([0.111 , 1.])),
        ('WingGeom, 1, 8', np.array([0.333, 1.])),
        ('WingGeom, 1, 8', np.array([0.555, 1.])),
        ('WingGeom, 1, 8', np.array([0.777, 1.])),
        ('WingGeom, 1, 8', np.array([1., 1.])),
    ]

    # geometry4 = geometry.copy()

    # leading_edge = geometry4.evaluate(leading_edge_parametric_coordinates, plot=False).reshape((-1,3))
    # trailing_edge = geometry4.evaluate(trailing_edge_parametric_coordinates, plot=False).reshape((-1,3))
    # chord_surface = m3l.linspace(leading_edge, trailing_edge, num_steps=4)

    # geometry4.plot_meshes(meshes=chord_surface, mesh_plot_types=['wireframe'], mesh_opacity=1., mesh_color='#F5F0E6',)

    # # geometry4.rotate(axis_origin=axis_origin, axis_vector=axis_vector, angles=angles, units='degrees')
    # left_wing_transition = geometry4.declare_component(component_name='left_wing', b_spline_search_names=['WingGeom, 1'])
    # left_wing_transition.rotate(axis_origin=axis_origin, axis_vector=axis_vector, angles=angles, units='degrees')
    # geometry4.plot()

    # leading_edge = geometry4.evaluate(leading_edge_parametric_coordinates, plot=False).reshape((-1,3))
    # trailing_edge = geometry4.evaluate(trailing_edge_parametric_coordinates, plot=False).reshape((-1,3))
    # chord_surface = m3l.linspace(leading_edge, trailing_edge, num_steps=4)

    # geometry4.plot_meshes(meshes=chord_surface, mesh_plot_types=['wireframe'], mesh_opacity=1., mesh_color='#F5F0E6',)

    geometry5 = geometry.copy()
    from lsdo_geo.core.parameterization.free_form_deformation_functions import construct_ffd_block_around_entities
    left_wing_ffd_block = construct_ffd_block_around_entities(name='left_wing_ffd_block', entities=left_wing, num_coefficients=(3, 30, 2))
    # left_wing_ffd_block = construct_ffd_block_around_entities(name='left_wing_ffd_block', entities=left_wing, num_coefficients=(5, 5, 5))
    # left_wing_ffd_block = construct_ffd_block_around_entities(name='left_wing_ffd_block', entities=left_wing, num_coefficients=(10, 5, 5))

    # left_wing_ffd_block.plot()

    # scaling_matrix = sps.eye(left_wing_ffd_block.num_coefficients)*2
    # scaling_matrix = scaling_matrix.tocsc()

    # new_left_wing_ffd_block_coefficients = m3l.matvec(scaling_matrix, left_wing_ffd_block.coefficients)
    # left_wing_coefficients = left_wing_ffd_block.evaluate(new_left_wing_ffd_block_coefficients, plot=True)
    # geometry5.assign_coefficients(coefficients=left_wing_coefficients, b_spline_names=left_wing.b_spline_names)

    from lsdo_geo.core.parameterization.volume_sectional_parameterization import VolumeSectionalParameterization
    left_wing_ffd_block_sectional_parameterization = VolumeSectionalParameterization(name='left_wing_ffd_block_sectional_parameterization',
                                                                                     principal_parametric_dimension=1,
                                                                                     parameterized_points=left_wing_ffd_block.coefficients,
                                                            parameterized_points_shape=left_wing_ffd_block.coefficients_shape)

    left_wing_ffd_block_sectional_parameterization.add_sectional_translation(name='left_wing_sweep', axis=0)
    left_wing_ffd_block_sectional_parameterization.add_sectional_translation(name='left_wing_diheral', axis=2)
    left_wing_ffd_block_sectional_parameterization.add_sectional_translation(name='left_wing_wingspan_stretch', axis=1)
    left_wing_ffd_block_sectional_parameterization.add_sectional_stretch(name='left_wing_chord_stretch', axis=0)
    left_wing_ffd_block_sectional_parameterization.add_sectional_rotation(name='left_wing_twist', axis=1)
    # left_wing_ffd_block_sectional_parameterization.plot()

    import lsdo_geo.splines.b_splines as bsp
    linear_b_spline_curve_2_dof_space = bsp.BSplineSpace(name='linear_b_spline_curve_2_dof_space', order=2, parametric_coefficients_shape=(2,))
    linear_b_spline_curve_3_dof_space = bsp.BSplineSpace(name='linear_b_spline_curve_3_dof_space', order=2, parametric_coefficients_shape=(3,))
    cubic_b_spline_curve_5_dof_space = bsp.BSplineSpace(name='cubic_b_spline_curve_5_dof_space', order=4, parametric_coefficients_shape=(5,))

    left_wing_sweep_coefficients = m3l.Variable(name='left_wing_sweep_coefficients', shape=(2,), value=np.array([1., 0.]))
    left_wing_sweep_b_spline = bsp.BSpline(name='left_wing_sweep_b_spline', space=linear_b_spline_curve_2_dof_space, 
                                           coefficients=left_wing_sweep_coefficients, num_physical_dimensions=1)
    
    left_wing_dihedral_coefficients = m3l.Variable(name='left_wing_dihedral_coefficients', shape=(2,), value=np.array([1., 0.]))
    left_wing_dihedral_b_spline = bsp.BSpline(name='left_wing_dihedral_b_spline', space=linear_b_spline_curve_2_dof_space, 
                                           coefficients=left_wing_dihedral_coefficients, num_physical_dimensions=1)
    
    left_wing_wingspan_stretch_coefficients = m3l.Variable(name='left_wing_wingspan_stretch_coefficients', shape=(2,), value=np.array([-1., 0.]))
    left_wing_wingspan_stretch_b_spline = bsp.BSpline(name='left_wing_wingspan_stretch_b_spline', space=linear_b_spline_curve_2_dof_space,
                                                     coefficients=left_wing_wingspan_stretch_coefficients, num_physical_dimensions=1)
    
    left_wing_chord_stretch_coefficients = m3l.Variable(name='left_wing_chord_stretch_coefficients', shape=(3,), 
                                                        value=np.array([-0.5, -0.1, 0.]))
    left_wing_chord_stretch_b_spline = bsp.BSpline(name='left_wing_chord_stretch_b_spline', space=linear_b_spline_curve_3_dof_space,
                                                    coefficients=left_wing_chord_stretch_coefficients, num_physical_dimensions=1)
    
    left_wing_twist_coefficients = m3l.Variable(name='left_wing_twist_coefficients', shape=(5,),
                                                value=np.array([0., 30., 20., 10., 0.]))
    left_wing_twist_b_spline = bsp.BSpline(name='left_wing_twist_b_spline', space=cubic_b_spline_curve_5_dof_space,
                                             coefficients=left_wing_twist_coefficients, num_physical_dimensions=1)


    
    section_parametric_coordinates = np.linspace(0., 1., left_wing_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
    left_wing_sectional_sweep = left_wing_sweep_b_spline.evaluate(section_parametric_coordinates)
    left_wing_sectional_diheral = left_wing_dihedral_b_spline.evaluate(section_parametric_coordinates)
    left_wing_wingspan_stretch = left_wing_wingspan_stretch_b_spline.evaluate(section_parametric_coordinates)
    left_wing_sectional_chord_stretch = left_wing_chord_stretch_b_spline.evaluate(section_parametric_coordinates)
    left_wing_sectional_twist = left_wing_twist_b_spline.evaluate(section_parametric_coordinates)

    # left_wing_sweep = m3l.Variable('left_wing_sweep', 
    #                                                    shape=(left_wing_ffd_block_sectional_parameterization.num_sections,),
    #                                                     value=np.linspace(1., 0., left_wing_ffd_block_sectional_parameterization.num_sections))
    # left_wing_diheral = m3l.Variable('left_wing_diheral', 
    #                                                    shape=(left_wing_ffd_block_sectional_parameterization.num_sections,),
    #                                                     value=np.linspace(0.5, 0., left_wing_ffd_block_sectional_parameterization.num_sections))
    # left_wing_chord_stretch = m3l.Variable('left_wing_chord_stretch',
    #                                                      shape=(left_wing_ffd_block_sectional_parameterization.num_sections,),
    #                                                       value=np.linspace(-.5, 0., left_wing_ffd_block_sectional_parameterization.num_sections))
    # left_wing_twist = m3l.Variable('left_wing_twist',
    #                                                      shape=(left_wing_ffd_block_sectional_parameterization.num_sections,),
    #                                                       value=np.linspace(-30., 0., left_wing_ffd_block_sectional_parameterization.num_sections))
    sectional_parameters = {
        'left_wing_sweep':left_wing_sectional_sweep, 
        'left_wing_diheral':left_wing_sectional_diheral,
        'left_wing_wingspan_stretch':left_wing_wingspan_stretch,
        'left_wing_chord_stretch':left_wing_sectional_chord_stretch,
        'left_wing_twist':left_wing_sectional_twist,
                            }
    

    left_wing_ffd_block_coefficients = left_wing_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)

    left_wing_coefficients = left_wing_ffd_block.evaluate(left_wing_ffd_block_coefficients, plot=False)

    # left_wing_ffd_block_sectional_parameterization.plot()
    geometry5.assign_coefficients(coefficients=left_wing_coefficients, b_spline_names=left_wing.b_spline_names)

    geometry5.rotate(axis_origin=axis_origin, axis_vector=axis_vector, angles=angles, units='degrees')

    # leading_edge = geometry5.evaluate(leading_edge_parametric_coordinates, plot=False).reshape((-1,3))
    # trailing_edge = geometry5.evaluate(trailing_edge_parametric_coordinates, plot=False).reshape((-1,3))
    # chord_surface = m3l.linspace(leading_edge, trailing_edge, num_steps=4)

    # geometry5.plot_meshes(meshes=chord_surface, mesh_plot_types=['wireframe'], mesh_opacity=1., mesh_color='#F5F0E6',)
    # geometry5.plot()

    left_wing_root_quarter_chord_ish = geometry5.evaluate([('WingGeom, 1, 8', np.array([0., 0.6]),)], plot=False)
    left_wing_tip_quarter_chord_ish = geometry5.evaluate([('WingGeom, 1, 8', np.array([1., 0.6]),)], plot=False)
    half_wingspan = m3l.norm(left_wing_tip_quarter_chord_ish - left_wing_root_quarter_chord_ish)    # NOTE: Consider adding dot operation to m3l

    left_wing_root_leading_edge = geometry5.evaluate([('WingGeom, 1, 8', np.array([0., 0.]),)], plot=False)
    left_wing_root_trailing_edge = geometry5.evaluate([('WingGeom, 1, 8', np.array([0., 1.]),)], plot=False)
    left_wing_tip_leading_edge = geometry5.evaluate([('WingGeom, 1, 8', np.array([1., 0.]),)], plot=False)
    left_wing_tip_trailing_edge = geometry5.evaluate([('WingGeom, 1, 8', np.array([1., 1.]),)], plot=False)
    left_wing_mid_leading_edge = geometry5.evaluate([('WingGeom, 1, 8', np.array([0.5, 0.]),)], plot=True)
    left_wing_mid_trailing_edge = geometry5.evaluate([('WingGeom, 1, 8', np.array([0.5, 1.]),)], plot=False)
    root_chord = m3l.norm(left_wing_root_leading_edge - left_wing_root_trailing_edge)
    tip_chord = m3l.norm(left_wing_tip_leading_edge - left_wing_tip_trailing_edge)
    mid_chord = m3l.norm(left_wing_mid_leading_edge - left_wing_mid_trailing_edge)

    from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver
    left_wing_parameterization_solver = ParameterizationSolver()

    left_wing_parameterization_solver.declare_state('left_wing_wingspan_stretch_coefficients', left_wing_wingspan_stretch_coefficients)
    left_wing_parameterization_solver.declare_state('left_wing_chord_stretch_coefficients', left_wing_chord_stretch_coefficients)

    left_wing_parameterization_solver.declare_input(name='half_wingspan', input=half_wingspan)
    left_wing_parameterization_solver.declare_input(name='root_chord', input=root_chord)
    left_wing_parameterization_solver.declare_input(name='tip_chord', input=tip_chord)
    left_wing_parameterization_solver.declare_input(name='mid_chord', input=mid_chord)

    half_wingspan_input = m3l.Variable(name='half_wingspan', shape=(1,), value=np.array([10.]))
    root_chord_input = m3l.Variable(name='root_chord', shape=(1,), value=np.array([4.]))
    tip_chord_input = m3l.Variable(name='tip_chord', shape=(1,), value=np.array([0.5]))
    mid_chord_input = m3l.Variable(name='mid_chord', shape=(1,), value=np.array([2.5]))

    parameterization_inputs = {
        'half_wingspan':half_wingspan_input,
        'root_chord':root_chord_input,
        'tip_chord':tip_chord_input,
        'mid_chord':mid_chord_input
    }
    outputs_dict = left_wing_parameterization_solver.evaluate(inputs=parameterization_inputs, plot=False)
    
    # print(outputs_dict['left_wing_wingspan_stretch_coefficients'])

    # left_wing_ffd_block = construct_ffd_block_around_entities(name='left_wing_ffd_block', entities=left_wing, num_coefficients=(3, 30, 2))

    # from lsdo_geo.core.parameterization.volume_sectional_parameterization import VolumeSectionalParameterization
    # left_wing_ffd_block_sectional_parameterization = VolumeSectionalParameterization(name='left_wing_ffd_block_sectional_parameterization',
    #                                                                                  principal_parametric_dimension=1,
    #                                                                                  parameterized_points=left_wing_ffd_block.coefficients,
    #                                                         parameterized_points_shape=left_wing_ffd_block.coefficients_shape)

    left_wing_dihedral_coefficients = m3l.Variable(name='left_wing_dihedral_coefficients', shape=(2,), value=np.array([1., 0.]))
    left_wing_dihedral_b_spline = bsp.BSpline(name='left_wing_dihedral_b_spline', space=linear_b_spline_curve_2_dof_space, 
                                           coefficients=left_wing_dihedral_coefficients, num_physical_dimensions=1)
    
    left_wing_wingspan_stretch_coefficients = outputs_dict['left_wing_wingspan_stretch_coefficients']
    left_wing_wingspan_stretch_b_spline = bsp.BSpline(name='left_wing_wingspan_stretch_b_spline', space=linear_b_spline_curve_2_dof_space,
                                                     coefficients=left_wing_wingspan_stretch_coefficients, num_physical_dimensions=1)
    
    left_wing_chord_stretch_coefficients = outputs_dict['left_wing_chord_stretch_coefficients']
    left_wing_chord_stretch_b_spline = bsp.BSpline(name='left_wing_chord_stretch_b_spline', space=linear_b_spline_curve_3_dof_space,
                                                    coefficients=left_wing_chord_stretch_coefficients, num_physical_dimensions=1)
    
    left_wing_twist_coefficients = m3l.Variable(name='left_wing_twist_coefficients', shape=(5,),
                                                value=np.array([0., 30., 20., 10., 0.]))
    left_wing_twist_b_spline = bsp.BSpline(name='left_wing_twist_b_spline', space=cubic_b_spline_curve_5_dof_space,
                                             coefficients=left_wing_twist_coefficients, num_physical_dimensions=1)


    
    section_parametric_coordinates = np.linspace(0., 1., left_wing_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
    left_wing_sectional_sweep = left_wing_sweep_b_spline.evaluate(section_parametric_coordinates)
    left_wing_sectional_diheral = left_wing_dihedral_b_spline.evaluate(section_parametric_coordinates)
    left_wing_wingspan_stretch = left_wing_wingspan_stretch_b_spline.evaluate(section_parametric_coordinates)
    left_wing_sectional_chord_stretch = left_wing_chord_stretch_b_spline.evaluate(section_parametric_coordinates)
    left_wing_sectional_twist = left_wing_twist_b_spline.evaluate(section_parametric_coordinates)

    sectional_parameters = {
        'left_wing_sweep':left_wing_sectional_sweep, 
        'left_wing_diheral':left_wing_sectional_diheral,
        'left_wing_wingspan_stretch':left_wing_wingspan_stretch,
        'left_wing_chord_stretch':left_wing_sectional_chord_stretch,
        'left_wing_twist':left_wing_sectional_twist,
                            }
    

    left_wing_ffd_block_coefficients = left_wing_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=True)

    left_wing_coefficients = left_wing_ffd_block.evaluate(left_wing_ffd_block_coefficients, plot=True)

    geometry5.assign_coefficients(coefficients=left_wing_coefficients, b_spline_names=left_wing.b_spline_names)

    geometry5.rotate(axis_origin=axis_origin, axis_vector=axis_vector, angles=angles, units='degrees')

    leading_edge = geometry5.evaluate(leading_edge_parametric_coordinates, plot=False).reshape((-1,3))
    trailing_edge = geometry5.evaluate(trailing_edge_parametric_coordinates, plot=False).reshape((-1,3))
    chord_surface = m3l.linspace(leading_edge, trailing_edge, num_steps=4)

    # geometry5.plot_meshes(meshes=chord_surface, mesh_plot_types=['wireframe'], mesh_opacity=1., mesh_color='#F5F0E6',)
    # geometry5.plot()

    left_wing_root_quarter_chord_ish = geometry5.evaluate([('WingGeom, 1, 8', np.array([0., 0.6]),)], plot=False)
    left_wing_tip_quarter_chord_ish = geometry5.evaluate([('WingGeom, 1, 8', np.array([1., 0.6]),)], plot=False)
    half_wingspan = m3l.norm(left_wing_tip_quarter_chord_ish - left_wing_root_quarter_chord_ish)    # NOTE: Consider adding dot operation to m3l

    left_wing_root_leading_edge = geometry5.evaluate([('WingGeom, 1, 8', np.array([0., 0.]),)], plot=False)
    left_wing_root_trailing_edge = geometry5.evaluate([('WingGeom, 1, 8', np.array([0., 1.]),)], plot=False)
    left_wing_tip_leading_edge = geometry5.evaluate([('WingGeom, 1, 8', np.array([1., 0.]),)], plot=False)
    left_wing_tip_trailing_edge = geometry5.evaluate([('WingGeom, 1, 8', np.array([1., 1.]),)], plot=False)
    root_chord = m3l.norm(left_wing_root_leading_edge - left_wing_root_trailing_edge)
    tip_chord = m3l.norm(left_wing_tip_leading_edge - left_wing_tip_trailing_edge)

    print('new half wingspan', half_wingspan)
    print('new root chord', root_chord)
    # THEN REVISIT ACTUATIONS
    # -- Do I want to create the framework of creating actuators, etc.?


    # print('hi')