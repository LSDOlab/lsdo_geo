from __future__ import annotations

import numpy as np
import pickle
from dataclasses import dataclass
from pathlib import Path
# import pickle
import csdl_alpha as csdl
# from lsdo_geo.splines.b_splines.b_spline_set import BSplineSet
# from lsdo_geo.splines.b_splines.b_spline_sub_set import BSplineSubSet
import lsdo_function_spaces as lfs
import lsdo_geo as lg

@dataclass
class Geometry(lfs.FunctionSet):

    def get_function_space(self):
        return self.space
    

    def declare_component(self, function_indices:list[int]=None, function_search_names:list[str]=None, name:str=None) -> lg.Geometry:
        '''
        Declares a component. This component will point to a sub-set of the entire geometry.

        Parameters
        ----------
        function_names : list[str]
            The names of the functions that make up the component.
        function_search_names : list[str], optional
            The names of the functions to search for. Names of functions will be returned for each B-spline that INCLUDES the search name.
        name : str
            The name of the component.
        '''
        function_set = self.create_subset(function_indices=function_indices, function_search_names=function_search_names, name=name)

        component = lg.Geometry(functions=function_set.functions, function_names=function_set.function_names, name=name, 
                                space=function_set.space)
        return component
    
    def create_component_copy(self, function_indices:list[int]=None, function_search_names:list[str]=None, name:str=None) -> lg.Geometry:
        '''
        Declares a component. This component will point to a sub-set of the entire geometry.

        Parameters
        ----------
        function_names : list[str]
            The names of the functions that make up the component.
        function_search_names : list[str], optional
            The names of the functions to search for. Names of functions will be returned for each B-spline that INCLUDES the search name.
        name : str
            The name of the component.
        '''
        component = self.create_subset(function_indices=function_indices, function_search_names=function_search_names, name=name)
        component_copy = component.copy()
        return component_copy


    # def import_geometry(self, file_name:str):
    #     '''
    #     Imports geometry from a file.

    #     Parameters
    #     ----------
    #     file_name : str
    #         The name of the file (with path) that containts the geometric information.
    #     '''
    #     from lsdo_geo.splines.b_splines.b_spline_functions import import_file, create_b_spline_set
    #     b_splines = import_file(file_name)
    #     b_spline_set = create_b_spline_set(self.name, b_splines)

    #     self.space = b_spline_set.space
    #     self.coefficients = b_spline_set.coefficients
    #     self.num_physical_dimensions = b_spline_set.num_physical_dimensions
    #     self.coefficient_indices = b_spline_set.coefficient_indices
    #     self.connections = b_spline_set.connections


    def rotate(self, axis_origin:csdl.Variable, axis_vector:csdl.Variable, angles:csdl.Variable, function_indices:list[int]=None,
                units:str='degrees'):
        '''
        Rotates the B-spline set about an axis.

        Parameters
        -----------
        axis_origin : csdl.Variable
            The origin of the axis of rotation.
        axis_vector : csdl.Variable
            The vector of the axis of rotation.
        angles : csdl.Variable
            The angle of rotation.
        function_indices : list[int]
            The indices of the functions to rotate.
        units : str
            The units of the angle of rotation. {degrees, radians}
        '''
        from lsdo_geo.core.geometry.geometry_functions import rotate as rotate_function
        if units == 'degrees':
            angles = angles * np.pi / 180.
            units = 'radians'
        elif units == 'radians':
            pass
        else:
            raise ValueError(f'Invalid units {units}.')
        
        if function_indices is None:
            function_indices = list(self.functions.keys())
        if isinstance(function_indices, int):
            function_indices = [function_indices]
        if not isinstance(function_indices, list):
            raise ValueError(f'The function indices must be a list of int, received {type(function_indices)}')
        
        if isinstance(axis_origin, np.ndarray):
            axis_origin = csdl.Variable(shape=axis_origin.shape, value=axis_origin)
        if type(axis_vector) is np.ndarray:
            axis_vector = csdl.Variable(shape=axis_vector.shape, value=axis_vector)
        if type(angles) is np.ndarray:
            angles = csdl.Variable(shape=angles.shape, value=angles)

        for function_index in function_indices:
            function = self.functions[function_index]
            rotated_coefficients = rotate_function(
                points=function.coefficients.reshape((function.coefficients.size // function.coefficients.shape[-1], function.coefficients.shape[-1])), 
                axis_origin=axis_origin, axis_vector=axis_vector, angles=angles, units=units
            )
            function.coefficients = rotated_coefficients.reshape(function.coefficients.shape)

    
    def plot_meshes(self, meshes:list[csdl.Variable], mesh_plot_types:list[str]=['wireframe'], mesh_opacity:float=1., mesh_color:str='#F5F0E6',
                mesh_color_map='jet', mesh_line_width:float=3.,
                function_indices:list[str]=None, function_plot_types:list[str]=['function'], function_opacity:float=0.25, function_color:str='#00629B',
                function_color_map:str='jet', function_surface_texture:str="",
                additional_plotting_elements:list=[], camera:dict=None, show:bool=True):
        '''
        Plots a mesh over the geometry.

        Parameters
        ----------
        meshes : list
            A list of meshes to plot.
        mesh_plot_types : list, optional = ['wireframe']
            A list of plot types for each mesh. Options are 'wireframe', 'function', and 'points'.
        mesh_opacity : float, optional = 1.
            The opacity of the mesh.
        mesh_color : str, optional = '#F5F0E6'
            The color of the mesh.
        mesh_color_map : str, optional = 'jet'
            The color map for the mesh.
        mesh_line_width : float, optional = 3.
            The line width of the mesh.
        function_indices : list, optional = None
            A list of indices for which functions to plot.
        function_plot_types : list, optional = ['function']
            A list of plot types for each primitive. Options are 'wireframe', 'function', and 'points'.
        function_opacity : float, optional = 0.25
            The opacity of the function.
        function_color : str, optional = '#00629B'
            The color of the function.
        function_color_map : str, optional = 'jet'
            The color map for the function.
        function_surface_texture : str {"", "metallic", "glossy", "ambient",... see Vedo for more options}
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
        import lsdo_function_spaces.utils.plotting_functions as pf
        plotting_elements = additional_plotting_elements.copy()

        if not isinstance(meshes, list) and not isinstance(meshes, tuple):
            meshes = [meshes]

        # Create plotting meshes for the functions/geometry
        plotting_elements = self.plot(point_types=['evaluated_points'], plot_types=function_plot_types, opacity=function_opacity,
                                      color=function_color, color_map=function_color_map, surface_texture=function_surface_texture,
                                      additional_plotting_elements=plotting_elements, show=False)

        for mesh in meshes:
            if type(mesh) is csdl.Variable:
                points = mesh.value
            else:
                points = mesh

            if isinstance(mesh, tuple):
                # Is vector, so draw an arrow
                processed_points = ()
                for point in mesh:
                    if type(point) is csdl.Variable:
                        processed_points = processed_points + (point.value,)
                    else:
                        processed_points = processed_points + (point,)
                arrow = vedo.Arrow(tuple(processed_points[0].reshape((-1,))), 
                                   tuple((processed_points[0] + processed_points[1]).reshape((-1,))), s=0.05)
                plotting_elements.append(arrow)
                continue

            if 'point_cloud' in mesh_plot_types:
                plotting_elements = pf.plot_points(points, opacity=mesh_opacity, color=mesh_color, color_map=mesh_color_map, 
                                                   additional_plotting_elements=plotting_elements, show=False)

            if points.shape[0] == 1:
                points = points.reshape((points.shape[1:]))

            if len(points.shape) == 2:  # If it's a curve
                from vedo import Line
                plotting_elements.append(Line(points).color(mesh_color).linewidth(mesh_line_width))
                
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
                plotting_elements.append(plotting_mesh.wireframe().linewidth(mesh_line_width))
            
        if show:
            plotter = vedo.Plotter()
            plotter.show(plotting_elements, 'Meshes', axes=1, viewup="z", interactive=True, camera=camera)

        return plotting_elements
    

    def plot_2d_mesh(self, mesh):
        pass



# if __name__ == "__main__":
#     from lsdo_geo.core.geometry.geometry_functions import import_geometry
#     # import array_mapper as am
#     import time

#     recorder = csdl.Recorder(inline=True)
#     recorder.start()

#     # import sys
#     # sys.setrecursionlimit(10000)

#     # var1 = csdl.Variable('var1', shape=(2,3), value=np.array([[1., 2., 3.], [4., 5., 6.]]))
#     # var2 = csdl.Variable('var2', shape=(2,3), value=np.array([[1., 2., 3.], [4., 5., 6.]]))
#     # var3 = var1 + var2
#     # print(var3)

#     t1 = time.time()
#     geometry = import_geometry('lsdo_geo/splines/b_splines/sample_geometries/rectangular_wing.stp', parallelize=False)
#     # geometry = import_geometry('lsdo_geo/splines/b_splines/sample_geometries/lift_plus_cruise_final.stp')
#     t2 = time.time()
#     geometry.refit(parallelize=False, fit_resolution=(50,50))
#     t3 = time.time()
#     # geometry.find_connections() # NOTE: This is really really slow for large geometries. Come back to this.
#     t4 = time.time()
#     # geometry.plot()
#     t5 = time.time()
#     print('Import time: ', t2-t1)
#     print('Refit time: ', t3-t2)
#     # print('Find connections time: ', t4-t3)
#     print('Plot time: ', t5-t4)


#     projected_points1_coordinates = geometry.project(np.array([[0.2, 1., 10.], [0.5, 1., 1.]]), plot=False, direction=np.array([0., 0., -1.]))
#     projected_points2_coordinates = geometry.project(np.array([[0.2, 0., 10.], [0.5, 1., 1.]]), plot=False, max_iterations=100)

#     projected_points1 = geometry.evaluate(projected_points1_coordinates, plot=False)
#     projected_points2 = geometry.evaluate(projected_points2_coordinates, plot=False)

#     test_linspace = csdl.linear_combination(projected_points1, projected_points2, 10)

#     import vedo
#     plotter = vedo.Plotter()
#     plotting_points = vedo.Points(test_linspace.value.reshape(-1,3))
#     # geometry_plot = geometry.plot(show=False)
#     # plotter.show(geometry_plot, plotting_points, interactive=True, axes=1)

#     right_wing = geometry.declare_component(component_name='right_wing', b_spline_search_names=['WingGeom, 0'])
#     # right_wing.plot()

#     projected_points1_on_right_wing = right_wing.project(np.array([[0.2, 1., 10.], [0.5, 1., 1.]]), plot=False, direction=np.array([0., 0., -1.]))
#     projected_points2_on_right_wing = right_wing.project(np.array([[0.2, 0., 1.], [0.5, 1., 1.]]), plot=False, max_iterations=100)

#     left_wing = geometry.declare_component(component_name='left_wing', b_spline_search_names=['WingGeom, 1'])
#     # left_wing.plot()

#     # geometry2 = geometry.copy()
#     # geometry2.coefficients = geometry.coefficients.copy()*2
#     # # geometry2.plot()
#     # # geometry.plot()

#     # # print(projected_points1_on_right_wing.evaluate(geometry2.coefficients))

#     # # left_wing_pressure_space = geometry.space.create_sub_space(sub_space_name='left_wing_pressure_space', b_spline_names=left_wing.b_spline_names)
    
#     # # # Manually creating a pressure distribution
#     # # pressure_coefficients = np.zeros((0,))
#     # # for b_spline_name in left_wing.b_spline_names:
#     # #     left_wing_geometry_coefficients = geometry.coefficients.value[geometry.coefficient_indices[b_spline_name]].reshape((-1,3))
#     # #     b_spline_pressure_coefficients = \
#     # #         -4*8.1/(np.pi*8.1)*np.sqrt(1 - (2*left_wing_geometry_coefficients[:,1]/8.1)**2) \
#     # #         * np.sqrt(1 - (2*left_wing_geometry_coefficients[:,0]/4)**2) \
#     # #         * (left_wing_geometry_coefficients[:,2]+0.05)
#     # #     pressure_coefficients = np.concatenate((pressure_coefficients, b_spline_pressure_coefficients.flatten()))

#     # # left_wing_pressure_function = left_wing_pressure_space.create_function(name='left_wing_pressure_function', 
#     # #                                                                        coefficients=pressure_coefficients, num_physical_dimensions=1)
#     # # left_wing.plot(color=left_wing_pressure_function)

#     # left_wing_pressure_space = geometry.space.create_sub_space(sub_space_name='left_wing_pressure_space',
#     #                                                                         b_spline_names=left_wing.b_spline_names)
#     # # Manually creating a pressure distribution mesh
#     # pressure_mesh = np.zeros((0,))
#     # pressure_parametric_coordinates = []
#     # for b_spline_name in left_wing.b_spline_names:
#     #     left_wing_geometry_coefficients = geometry.coefficients.value[geometry.coefficient_indices[b_spline_name]].reshape((-1,3))
#     #     b_spline_pressure_coefficients = \
#     #         -4*8.1/(np.pi*8.1)*np.sqrt(1 - (2*left_wing_geometry_coefficients[:,1]/8.1)**2) \
#     #         * np.sqrt(1 - (2*left_wing_geometry_coefficients[:,0]/4)**2) \
#     #         * (left_wing_geometry_coefficients[:,2]+0.05)
#     #     pressure_mesh = np.concatenate((pressure_mesh, b_spline_pressure_coefficients.flatten()))

#     #     # parametric_coordinates = left_wing.project(points=left_wing_geometry_coefficients, targets=[b_spline_name])
#     #     # pressure_parametric_coordinates.extend(parametric_coordinates)

#     #     b_spline_space = left_wing_pressure_space.spaces[left_wing_pressure_space.b_spline_to_space[b_spline_name]]

#     #     b_spline_num_coefficients_u = b_spline_space.parametric_coefficients_shape[0]
#     #     b_spline_num_coefficients_v = b_spline_space.parametric_coefficients_shape[1]

#     #     u_vec = np.einsum('i,j->ij', np.linspace(0., 1., b_spline_num_coefficients_u), np.ones(b_spline_num_coefficients_u)).flatten()
#     #     v_vec = np.einsum('i,j->ij', np.ones(b_spline_num_coefficients_v), np.linspace(0., 1., b_spline_num_coefficients_v)).flatten()
#     #     parametric_coordinates = np.hstack((u_vec.reshape((-1,1)), v_vec.reshape((-1,1))))

#     #     for i in range(len(parametric_coordinates)):
#     #         pressure_parametric_coordinates.append(tuple((b_spline_name, parametric_coordinates[i,:])))

#     # pressure_coefficients = left_wing_pressure_space.fit_b_spline_set(fitting_points=pressure_mesh.reshape((-1,1)),
#     #                                                                               fitting_parametric_coordinates=pressure_parametric_coordinates,
#     #                                                                             #   regularization_parameter=1.e-3)
#     #                                                                               regularization_parameter=0.)

#     # left_wing_pressure_function = left_wing_pressure_space.create_function(name='left_wing_pressure_function', 
#     #                                                                        coefficients=pressure_coefficients, num_physical_dimensions=1)
    
#     # # left_wing.plot(color=left_wing_pressure_function)

#     geometry3 = geometry.copy()
#     axis_origin = geometry.evaluate(geometry.project(np.array([0.5, -10., 0.5])))
#     axis_vector = geometry.evaluate(geometry.project(np.array([0.5, 10., 0.5]))) - axis_origin
#     angles = 45
#     # geometry3.coefficients = csdl.rotate(points=geometry3.coefficients.reshape((-1,3)), axis_origin=axis_origin, axis_vector=axis_vector,
#     #                                     angles=angles, units='degrees').reshape((-1,))
#     # # geometry3.plot()

#     leading_edge_parametric_coordinates = [
#         ('WingGeom, 0, 3', np.array([1.,  0.])),
#         ('WingGeom, 0, 3', np.array([0.777, 0.])),
#         ('WingGeom, 0, 3', np.array([0.555, 0.])),
#         ('WingGeom, 0, 3', np.array([0.333, 0.])),
#         ('WingGeom, 0, 3', np.array([0.111, 0.])),
#         ('WingGeom, 1, 8', np.array([0.111 , 0.])),
#         ('WingGeom, 1, 8', np.array([0.333, 0.])),
#         ('WingGeom, 1, 8', np.array([0.555, 0.])),
#         ('WingGeom, 1, 8', np.array([0.777, 0.])),
#         ('WingGeom, 1, 8', np.array([1., 0.])),
#     ]

#     trailing_edge_parametric_coordinates = [
#         ('WingGeom, 0, 3', np.array([1.,  1.])),
#         ('WingGeom, 0, 3', np.array([0.777, 1.])),
#         ('WingGeom, 0, 3', np.array([0.555, 1.])),
#         ('WingGeom, 0, 3', np.array([0.333, 1.])),
#         ('WingGeom, 0, 3', np.array([0.111, 1.])),
#         ('WingGeom, 1, 8', np.array([0.111 , 1.])),
#         ('WingGeom, 1, 8', np.array([0.333, 1.])),
#         ('WingGeom, 1, 8', np.array([0.555, 1.])),
#         ('WingGeom, 1, 8', np.array([0.777, 1.])),
#         ('WingGeom, 1, 8', np.array([1., 1.])),
#     ]

#     # geometry4 = geometry.copy()

#     # leading_edge = geometry4.evaluate(leading_edge_parametric_coordinates, plot=False).reshape((-1,3))
#     # trailing_edge = geometry4.evaluate(trailing_edge_parametric_coordinates, plot=False).reshape((-1,3))
#     # chord_surface = csdl.linspace(leading_edge, trailing_edge, num_steps=4)

#     # # geometry4.plot_meshes(meshes=chord_surface, mesh_plot_types=['wireframe'], mesh_opacity=1., mesh_color='#F5F0E6',)

#     # # geometry4.rotate(axis_origin=axis_origin, axis_vector=axis_vector, angles=angles, units='degrees')
#     # left_wing_transition = geometry4.declare_component(component_name='left_wing', b_spline_search_names=['WingGeom, 1'])
#     # left_wing_transition.rotate(axis_origin=axis_origin, axis_vector=axis_vector, angles=angles, units='degrees')
#     # # geometry4.plot()

#     # leading_edge = geometry4.evaluate(leading_edge_parametric_coordinates, plot=False).reshape((-1,3))
#     # trailing_edge = geometry4.evaluate(trailing_edge_parametric_coordinates, plot=False).reshape((-1,3))
#     # chord_surface = csdl.linspace(leading_edge, trailing_edge, num_steps=4)

#     # # geometry4.plot_meshes(meshes=chord_surface, mesh_plot_types=['wireframe'], mesh_opacity=1., mesh_color='#F5F0E6',)

#     geometry5 = geometry.copy()
#     from lsdo_geo.core.parameterization.free_form_deformation_functions import construct_ffd_block_around_entities
#     right_wing_ffd_block = construct_ffd_block_around_entities(name='right_wing_ffd_block', entities=right_wing, num_coefficients=(2, 2, 2))
#     right_wing_ffd_block.coefficients.name = 'right_wing_ffd_block_coefficients'
#     from lsdo_geo.core.parameterization.volume_sectional_parameterization import VolumeSectionalParameterization, VolumeSectionalParameterizationInputs
#     right_wing_ffd_block_sectional_parameterization = VolumeSectionalParameterization(name='right_wing_ffd_block_sectional_parameterization',
#                                                                                      principal_parametric_dimension=1,
#                                                                                      parameterized_points=right_wing_ffd_block.coefficients,
#                                                             parameterized_points_shape=right_wing_ffd_block.coefficients_shape)
#     # right_wing_ffd_block_sectional_parameterization.add_sectional_translation(name='right_wing_rigid_body_translation_x', axis=0)
#     # right_wing_ffd_block_sectional_parameterization.add_sectional_translation(name='right_wing_rigid_body_translation_y', axis=1)
#     # right_wing_ffd_block_sectional_parameterization.add_sectional_translation(name='right_wing_rigid_body_translation_z', axis=2)
#     import lsdo_geo.splines.b_splines as bsp
#     constant_b_spline_curve_1_dof_space = bsp.BSplineSpace(name='constant_b_spline_curve_1_dof_space', order=1, parametric_coefficients_shape=(1,))
#     right_wing_rigid_body_translation_x_coefficients = csdl.Variable(name='right_wing_rigid_body_translation_x_coefficients', shape=(1,), 
#                                                                     value=np.array([0.,]))
#     right_wing_rigid_body_translation_x_b_spline = bsp.BSpline(name='right_wing_rigid_body_translation_x_b_spline', 
#                                                                space=constant_b_spline_curve_1_dof_space, 
#                                            coefficients=right_wing_rigid_body_translation_x_coefficients, num_physical_dimensions=1)
#     right_wing_rigid_body_translation_y_coefficients = csdl.Variable(name='right_wing_rigid_body_translation_y_coefficients', shape=(1,), 
#                                                                     value=np.array([0.,]))
#     right_wing_rigid_body_translation_y_b_spline = bsp.BSpline(name='right_wing_rigid_body_translation_y_b_spline', 
#                                                                space=constant_b_spline_curve_1_dof_space, 
#                                            coefficients=right_wing_rigid_body_translation_y_coefficients, num_physical_dimensions=1)
#     right_wing_rigid_body_translation_z_coefficients = csdl.Variable(name='right_wing_rigid_body_translation_z_coefficients', shape=(1,), 
#                                                                     value=np.array([0.,]))
#     right_wing_rigid_body_translation_z_b_spline = bsp.BSpline(name='right_wing_rigid_body_translation_z_b_spline',
#                                                                space=constant_b_spline_curve_1_dof_space, 
#                                            coefficients=right_wing_rigid_body_translation_z_coefficients, num_physical_dimensions=1)
    
#     section_parametric_coordinates = np.linspace(0., 1., right_wing_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
#     right_wing_sectional_rigid_body_translation_x = right_wing_rigid_body_translation_x_b_spline.evaluate(section_parametric_coordinates)
#     right_wing_sectional_rigid_body_translation_y = right_wing_rigid_body_translation_y_b_spline.evaluate(section_parametric_coordinates)
#     right_wing_sectional_rigid_body_translation_z = right_wing_rigid_body_translation_z_b_spline.evaluate(section_parametric_coordinates)

#     # sectional_parameters = {
#     #     'right_wing_rigid_body_translation_x':right_wing_sectional_rigid_body_translation_x,
#     #     'right_wing_rigid_body_translation_y':right_wing_sectional_rigid_body_translation_y,
#     #     'right_wing_rigid_body_translation_z':right_wing_sectional_rigid_body_translation_z,
#     #                         }
#     sectional_parameters = VolumeSectionalParameterizationInputs()
#     sectional_parameters.add_sectional_translation(axis=0, translation=right_wing_sectional_rigid_body_translation_x)
#     sectional_parameters.add_sectional_translation(axis=1, translation=right_wing_sectional_rigid_body_translation_y)
#     sectional_parameters.add_sectional_translation(axis=2, translation=right_wing_sectional_rigid_body_translation_z)

#     right_wing_ffd_block_coefficients = right_wing_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)
#     right_wing_coefficients = right_wing_ffd_block.evaluate(right_wing_ffd_block_coefficients, plot=False)
#     geometry5.assign_coefficients(coefficients=right_wing_coefficients, b_spline_names=right_wing.b_spline_names)

#     from lsdo_geo.core.parameterization.free_form_deformation_functions import construct_ffd_block_around_entities
#     left_wing_ffd_block = construct_ffd_block_around_entities(name='left_wing_ffd_block', entities=left_wing, num_coefficients=(2, 30, 2))
#     left_wing_ffd_block.coefficients.name = 'left_wing_ffd_block_coefficients'
#     # left_wing_ffd_block = construct_ffd_block_around_entities(name='left_wing_ffd_block', entities=left_wing, num_coefficients=(5, 5, 5))
#     # left_wing_ffd_block = construct_ffd_block_around_entities(name='left_wing_ffd_block', entities=left_wing, num_coefficients=(10, 5, 5))

#     # left_wing_ffd_block.plot()

#     # scaling_matrix = sps.eye(left_wing_ffd_block.num_coefficients)*2
#     # scaling_matrix = scaling_matrix.tocsc()

#     # new_left_wing_ffd_block_coefficients = csdl.matvec(scaling_matrix, left_wing_ffd_block.coefficients)
#     # left_wing_coefficients = left_wing_ffd_block.evaluate(new_left_wing_ffd_block_coefficients, plot=True)
#     # geometry5.assign_coefficients(coefficients=left_wing_coefficients, b_spline_names=left_wing.b_spline_names)

#     from lsdo_geo.core.parameterization.volume_sectional_parameterization import VolumeSectionalParameterization, VolumeSectionalParameterizationInputs
#     left_wing_ffd_block_sectional_parameterization = VolumeSectionalParameterization(name='left_wing_ffd_block_sectional_parameterization',
#                                                                                      principal_parametric_dimension=1,
#                                                                                      parameterized_points=left_wing_ffd_block.coefficients,
#                                                             parameterized_points_shape=left_wing_ffd_block.coefficients_shape)

#     # left_wing_ffd_block_sectional_parameterization.add_sectional_translation(name='left_wing_sweep', axis=0)
#     # left_wing_ffd_block_sectional_parameterization.add_sectional_translation(name='left_wing_diheral', axis=2)
#     # left_wing_ffd_block_sectional_parameterization.add_sectional_translation(name='left_wing_wingspan_stretch', axis=1)
#     # left_wing_ffd_block_sectional_parameterization.add_sectional_stretch(name='left_wing_chord_stretch', axis=0)
#     # # left_wing_ffd_block_sectional_parameterization.add_sectional_rotation(name='left_wing_twist', axis=1)
#     # # left_wing_ffd_block_sectional_parameterization.plot()

#     import lsdo_geo.splines.b_splines as bsp
#     linear_b_spline_curve_2_dof_space = bsp.BSplineSpace(name='linear_b_spline_curve_2_dof_space', order=2, parametric_coefficients_shape=(2,))
#     linear_b_spline_curve_3_dof_space = bsp.BSplineSpace(name='linear_b_spline_curve_3_dof_space', order=2, parametric_coefficients_shape=(3,))
#     cubic_b_spline_curve_5_dof_space = bsp.BSplineSpace(name='cubic_b_spline_curve_5_dof_space', order=4, parametric_coefficients_shape=(5,))

#     left_wing_sweep_coefficients = csdl.Variable(name='left_wing_sweep_coefficients', shape=(2,), value=np.array([0., 0.]))
#     left_wing_sweep_b_spline = bsp.BSpline(name='left_wing_sweep_b_spline', space=linear_b_spline_curve_2_dof_space, 
#                                            coefficients=left_wing_sweep_coefficients, num_physical_dimensions=1)
    
#     left_wing_dihedral_coefficients = csdl.Variable(name='left_wing_dihedral_coefficients', shape=(2,), value=np.array([0., 0.]))
#     left_wing_dihedral_b_spline = bsp.BSpline(name='left_wing_dihedral_b_spline', space=linear_b_spline_curve_2_dof_space, 
#                                            coefficients=left_wing_dihedral_coefficients, num_physical_dimensions=1)
    
#     left_wing_wingspan_stretch_coefficients_state = csdl.Variable(name='left_wing_wingspan_stretch_coefficients_state', shape=(2,), 
#                                                                  value=np.array([0., 0.]))
#     left_wing_wingspan_stretch_b_spline = bsp.BSpline(name='left_wing_wingspan_stretch_b_spline', space=linear_b_spline_curve_2_dof_space,
#                                                      coefficients=left_wing_wingspan_stretch_coefficients_state, num_physical_dimensions=1)
    
#     left_wing_chord_stretch_coefficients_state = csdl.Variable(name='left_wing_chord_stretch_coefficients_state', shape=(3,), 
#                                                         value=np.array([0., 0., 0.]))
#     left_wing_chord_stretch_b_spline = bsp.BSpline(name='left_wing_chord_stretch_b_spline', space=linear_b_spline_curve_3_dof_space,
#                                                     coefficients=left_wing_chord_stretch_coefficients_state, num_physical_dimensions=1)
    
#     left_wing_twist_coefficients = csdl.Variable(name='left_wing_twist_coefficients', shape=(5,),
#                                                 value=np.array([0., 30., 20., 10., 0.]))
#     # left_wing_twist_b_spline = bsp.BSpline(name='left_wing_twist_b_spline', space=cubic_b_spline_curve_5_dof_space,
#     #                                          coefficients=left_wing_twist_coefficients, num_physical_dimensions=1)

#     section_parametric_coordinates = np.linspace(0., 1., left_wing_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
#     left_wing_sectional_sweep = left_wing_sweep_b_spline.evaluate(section_parametric_coordinates)
#     left_wing_sectional_diheral = left_wing_dihedral_b_spline.evaluate(section_parametric_coordinates)
#     left_wing_wingspan_stretch = left_wing_wingspan_stretch_b_spline.evaluate(section_parametric_coordinates)
#     left_wing_sectional_chord_stretch = left_wing_chord_stretch_b_spline.evaluate(section_parametric_coordinates)
#     # left_wing_sectional_twist = left_wing_twist_b_spline.evaluate(section_parametric_coordinates)

#     # sectional_parameters = {
#     #     'left_wing_sweep':left_wing_sectional_sweep, 
#     #     'left_wing_diheral':left_wing_sectional_diheral,
#     #     'left_wing_wingspan_stretch':left_wing_wingspan_stretch,
#     #     'left_wing_chord_stretch':left_wing_sectional_chord_stretch,
#     #     # 'left_wing_twist':left_wing_sectional_twist,
#     #                         }
#     sectional_parameters = VolumeSectionalParameterizationInputs()
#     sectional_parameters.add_sectional_translation(axis=0, translation=left_wing_sectional_sweep)
#     sectional_parameters.add_sectional_translation(axis=2, translation=left_wing_sectional_diheral)
#     sectional_parameters.add_sectional_translation(axis=1, translation=left_wing_wingspan_stretch)
#     sectional_parameters.add_sectional_stretch(axis=0, stretch=left_wing_sectional_chord_stretch)
#     # sectional_parameters.add_sectional_rotation(axis=1, rotation=left_wing_sectional_twist)
    

#     left_wing_ffd_block_coefficients = left_wing_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)

#     left_wing_coefficients = left_wing_ffd_block.evaluate(left_wing_ffd_block_coefficients, plot=False)

#     # left_wing_ffd_block_sectional_parameterization.plot()
#     geometry5.assign_coefficients(coefficients=left_wing_coefficients, b_spline_names=left_wing.b_spline_names)

#     # geometry5.rotate(axis_origin=axis_origin, axis_vector=axis_vector, angles=angles, units='degrees')

#     # leading_edge = geometry5.evaluate(leading_edge_parametric_coordinates, plot=False).reshape((-1,3))
#     # trailing_edge = geometry5.evaluate(trailing_edge_parametric_coordinates, plot=False).reshape((-1,3))
#     # chord_surface = csdl.linspace(leading_edge, trailing_edge, num_steps=4)

#     # geometry5.plot_meshes(meshes=chord_surface, mesh_plot_types=['wireframe'], mesh_opacity=1., mesh_color='#F5F0E6',)
#     # geometry5.plot()

#     left_wing_root_quarter_chord_ish = geometry5.evaluate([('WingGeom, 1, 8', np.array([0., 0.6]),)], plot=False)
#     left_wing_tip_quarter_chord_ish = geometry5.evaluate([('WingGeom, 1, 8', np.array([1., 0.6]),)], plot=False)
#     half_wingspan = csdl.norm(left_wing_tip_quarter_chord_ish - left_wing_root_quarter_chord_ish)    # NOTE: Consider adding dot operation to m3l

#     left_wing_root_leading_edge = geometry5.evaluate([('WingGeom, 1, 8', np.array([0., 0.]),)], plot=False)
#     left_wing_root_trailing_edge = geometry5.evaluate([('WingGeom, 1, 8', np.array([0., 1.]),)], plot=False)
#     left_wing_tip_leading_edge = geometry5.evaluate([('WingGeom, 1, 8', np.array([1., 0.]),)], plot=False)
#     left_wing_tip_trailing_edge = geometry5.evaluate([('WingGeom, 1, 8', np.array([1., 1.]),)], plot=False)
#     left_wing_mid_leading_edge = geometry5.evaluate([('WingGeom, 1, 8', np.array([0.5, 0.]),)], plot=False)
#     left_wing_mid_trailing_edge = geometry5.evaluate([('WingGeom, 1, 8', np.array([0.5, 1.]),)], plot=False)
#     root_chord = csdl.norm(left_wing_root_leading_edge - left_wing_root_trailing_edge)
#     tip_chord = csdl.norm(left_wing_tip_leading_edge - left_wing_tip_trailing_edge)
#     mid_chord = csdl.norm(left_wing_mid_leading_edge - left_wing_mid_trailing_edge)

#     right_wing_root_quarter_chord_ish = geometry5.evaluate([('WingGeom, 0, 3', np.array([0., 0.6]),)], plot=False)
#     left_wing_right_wing_connection = left_wing_root_quarter_chord_ish - right_wing_root_quarter_chord_ish

#     # from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver
#     # parameterization_solver = ParameterizationSolver()

#     # parameterization_solver.declare_state('left_wing_wingspan_stretch_coefficients_state', left_wing_wingspan_stretch_coefficients_state)
#     # parameterization_solver.declare_state('left_wing_chord_stretch_coefficients_state', left_wing_chord_stretch_coefficients_state)
#     # parameterization_solver.declare_state('right_wing_rigid_body_translation_x_coefficients', right_wing_rigid_body_translation_x_coefficients)
#     # parameterization_solver.declare_state('right_wing_rigid_body_translation_y_coefficients', right_wing_rigid_body_translation_y_coefficients)
#     # parameterization_solver.declare_state('right_wing_rigid_body_translation_z_coefficients', right_wing_rigid_body_translation_z_coefficients)

#     # parameterization_solver.declare_input(name='half_wingspan', input=half_wingspan)
#     # parameterization_solver.declare_input(name='root_chord', input=root_chord)
#     # parameterization_solver.declare_input(name='tip_chord', input=tip_chord)
#     # parameterization_solver.declare_input(name='mid_chord', input=mid_chord)
#     # parameterization_solver.declare_input(name='left_wing_right_wing_connection', input=left_wing_right_wing_connection)
#     # # parameterization_solver.declare_input(name='left_wing_right_wing_connection', input=csdl.norm(left_wing_right_wing_connection))
    

#     half_wingspan_input = csdl.Variable(name='half_wingspan', shape=(1,), value=np.array([10.]))
#     root_chord_input = csdl.Variable(name='root_chord', shape=(1,), value=np.array([4.]))
#     tip_chord_input = csdl.Variable(name='tip_chord', shape=(1,), value=np.array([0.5]))
#     mid_chord_input = csdl.Variable(name='mid_chord', shape=(1,), value=np.array([2.5]))

#     left_wing_right_wing_connection_input = csdl.Variable(name='left_wing_right_wing_connection', shape=(3,), value=left_wing_right_wing_connection.value)
#     # left_wing_right_wing_connection_input = csdl.Variable(name='left_wing_right_wing_connection', shape=(1,), 
#     #                                                      value=csdl.norm(left_wing_right_wing_connection).value)

#     # parameterization_inputs = {
#     #     'half_wingspan':half_wingspan_input,
#     #     'root_chord':root_chord_input,
#     #     'tip_chord':tip_chord_input,
#     #     'mid_chord':mid_chord_input,
#     #     'left_wing_right_wing_connection':left_wing_right_wing_connection_input
#     # }
#     # outputs_dict = parameterization_solver.evaluate(inputs=parameterization_inputs, plot=False)




#     # right_wing_rigid_body_translation_x_coefficients = outputs_dict['right_wing_rigid_body_translation_x_coefficients']
#     # right_wing_rigid_body_translation_x_b_spline = bsp.BSpline(name='right_wing_rigid_body_translation_x_b_spline', 
#     #                                                            space=constant_b_spline_curve_1_dof_space, 
#     #                                        coefficients=right_wing_rigid_body_translation_x_coefficients, num_physical_dimensions=1)
#     # right_wing_rigid_body_translation_y_coefficients = outputs_dict['right_wing_rigid_body_translation_y_coefficients']
#     # right_wing_rigid_body_translation_y_b_spline = bsp.BSpline(name='right_wing_rigid_body_translation_y_b_spline', 
#     #                                                            space=constant_b_spline_curve_1_dof_space, 
#     #                                        coefficients=right_wing_rigid_body_translation_y_coefficients, num_physical_dimensions=1)
#     # right_wing_rigid_body_translation_z_coefficients = outputs_dict['right_wing_rigid_body_translation_z_coefficients']
#     # right_wing_rigid_body_translation_z_b_spline = bsp.BSpline(name='right_wing_rigid_body_translation_z_b_spline',
#     #                                                            space=constant_b_spline_curve_1_dof_space, 
#     #                                        coefficients=right_wing_rigid_body_translation_z_coefficients, num_physical_dimensions=1)
    
#     section_parametric_coordinates = np.linspace(0., 1., right_wing_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
#     right_wing_sectional_rigid_body_translation_x = right_wing_rigid_body_translation_x_b_spline.evaluate(section_parametric_coordinates)
#     right_wing_sectional_rigid_body_translation_y = right_wing_rigid_body_translation_y_b_spline.evaluate(section_parametric_coordinates)
#     right_wing_sectional_rigid_body_translation_z = right_wing_rigid_body_translation_z_b_spline.evaluate(section_parametric_coordinates)

#     # sectional_parameters = {
#     #     'right_wing_rigid_body_translation_x':right_wing_sectional_rigid_body_translation_x,
#     #     'right_wing_rigid_body_translation_y':right_wing_sectional_rigid_body_translation_y,
#     #     'right_wing_rigid_body_translation_z':right_wing_sectional_rigid_body_translation_z,
#     #                         }
    
#     sectional_parameters = VolumeSectionalParameterizationInputs()
#     sectional_parameters.add_sectional_translation(axis=0, translation=right_wing_sectional_rigid_body_translation_x)
#     sectional_parameters.add_sectional_translation(axis=1, translation=right_wing_sectional_rigid_body_translation_y)
#     sectional_parameters.add_sectional_translation(axis=2, translation=right_wing_sectional_rigid_body_translation_z)

#     right_wing_ffd_block_coefficients = right_wing_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)
#     right_wing_coefficients = right_wing_ffd_block.evaluate(right_wing_ffd_block_coefficients, plot=False)
#     geometry5.assign_coefficients(coefficients=right_wing_coefficients, b_spline_names=right_wing.b_spline_names)
    
#     # print(outputs_dict['left_wing_wingspan_stretch_coefficients'])

#     # left_wing_ffd_block = construct_ffd_block_around_entities(name='left_wing_ffd_block', entities=left_wing, num_coefficients=(3, 30, 2))

#     # from lsdo_geo.core.parameterization.volume_sectional_parameterization import VolumeSectionalParameterization
#     # left_wing_ffd_block_sectional_parameterization = VolumeSectionalParameterization(name='left_wing_ffd_block_sectional_parameterization',
#     #                                                                                  principal_parametric_dimension=1,
#     #                                                                                  parameterized_points=left_wing_ffd_block.coefficients,
#     #                                                         parameterized_points_shape=left_wing_ffd_block.coefficients_shape)

#     left_wing_dihedral_coefficients = csdl.Variable(name='left_wing_dihedral_coefficients', shape=(2,), value=np.array([1., 0.]))
#     left_wing_dihedral_b_spline = bsp.BSpline(name='left_wing_dihedral_b_spline', space=linear_b_spline_curve_2_dof_space, 
#                                            coefficients=left_wing_dihedral_coefficients, num_physical_dimensions=1)
    
#     # left_wing_wingspan_stretch_coefficients = outputs_dict['left_wing_wingspan_stretch_coefficients_state'].copy()
#     # left_wing_wingspan_stretch_b_spline = bsp.BSpline(name='left_wing_wingspan_stretch_b_spline', space=linear_b_spline_curve_2_dof_space,
#     #                                                  coefficients=left_wing_wingspan_stretch_coefficients, num_physical_dimensions=1)
    
#     # left_wing_chord_stretch_coefficients = outputs_dict['left_wing_chord_stretch_coefficients_state'].copy()
#     # left_wing_chord_stretch_b_spline = bsp.BSpline(name='left_wing_chord_stretch_b_spline', space=linear_b_spline_curve_3_dof_space,
#     #                                                 coefficients=left_wing_chord_stretch_coefficients, num_physical_dimensions=1)
    
#     # left_wing_ffd_block_sectional_parameterization.add_sectional_rotation(name='left_wing_twist', axis=1)
#     # left_wing_twist_coefficients = csdl.Variable(name='left_wing_twist_coefficients', shape=(5,),
#     #                                             value=np.array([0., 30., 20., 10., 0.]))
#     # left_wing_twist_b_spline = bsp.BSpline(name='left_wing_twist_b_spline', space=cubic_b_spline_curve_5_dof_space,
#     #                                          coefficients=left_wing_twist_coefficients, num_physical_dimensions=1)


    
#     section_parametric_coordinates = np.linspace(0., 1., left_wing_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
#     left_wing_sectional_sweep = left_wing_sweep_b_spline.evaluate(section_parametric_coordinates)
#     left_wing_sectional_diheral = left_wing_dihedral_b_spline.evaluate(section_parametric_coordinates)
#     left_wing_wingspan_stretch = left_wing_wingspan_stretch_b_spline.evaluate(section_parametric_coordinates)
#     left_wing_sectional_chord_stretch = left_wing_chord_stretch_b_spline.evaluate(section_parametric_coordinates)
#     # left_wing_sectional_twist = left_wing_twist_b_spline.evaluate(section_parametric_coordinates)

#     # sectional_parameters = {
#     #     'left_wing_sweep':left_wing_sectional_sweep, 
#     #     'left_wing_diheral':left_wing_sectional_diheral,
#     #     'left_wing_wingspan_stretch':left_wing_wingspan_stretch,
#     #     'left_wing_chord_stretch':left_wing_sectional_chord_stretch,
#     #     'left_wing_twist':left_wing_sectional_twist,
#     #                         }
    
#     sectional_parameters = VolumeSectionalParameterizationInputs()
#     sectional_parameters.add_sectional_translation(axis=0, translation=left_wing_sectional_sweep)
#     sectional_parameters.add_sectional_translation(axis=2, translation=left_wing_sectional_diheral)
#     sectional_parameters.add_sectional_translation(axis=1, translation=left_wing_wingspan_stretch)
#     sectional_parameters.add_sectional_stretch(axis=0, stretch=left_wing_sectional_chord_stretch)
#     # sectional_parameters.add_sectional_rotation(axis=1, rotation=left_wing_sectional_twist)
    
#     left_wing_ffd_block_coefficients = left_wing_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)

#     left_wing_coefficients = left_wing_ffd_block.evaluate(left_wing_ffd_block_coefficients, plot=False)

#     geometry5.assign_coefficients(coefficients=left_wing_coefficients, b_spline_names=left_wing.b_spline_names)

#     # geometry5.rotate(axis_origin=axis_origin, axis_vector=axis_vector, angles=angles, units='degrees')

#     leading_edge = geometry5.evaluate(leading_edge_parametric_coordinates, plot=False).reshape((-1,3))
#     trailing_edge = geometry5.evaluate(trailing_edge_parametric_coordinates, plot=False).reshape((-1,3))
#     chord_surface = csdl.linear_combination(leading_edge, trailing_edge, num_steps=4)

#     # geometry5.plot_meshes(meshes=chord_surface, mesh_plot_types=['wireframe'], mesh_opacity=1., mesh_color='#F5F0E6',)
#     # geometry5.plot()

#     left_wing_root_quarter_chord_ish = geometry5.evaluate([('WingGeom, 1, 8', np.array([0., 0.6]),)], plot=False)
#     left_wing_tip_quarter_chord_ish = geometry5.evaluate([('WingGeom, 1, 8', np.array([1., 0.6]),)], plot=False)
#     half_wingspan_output = csdl.norm(left_wing_tip_quarter_chord_ish - left_wing_root_quarter_chord_ish)    # NOTE: Consider adding dot operation to m3l

#     left_wing_root_leading_edge = geometry5.evaluate([('WingGeom, 1, 8', np.array([0., 0.]),)], plot=False)
#     left_wing_root_trailing_edge = geometry5.evaluate([('WingGeom, 1, 8', np.array([0., 1.]),)], plot=False)
#     left_wing_tip_leading_edge = geometry5.evaluate([('WingGeom, 1, 8', np.array([1., 0.]),)], plot=False)
#     left_wing_tip_trailing_edge = geometry5.evaluate([('WingGeom, 1, 8', np.array([1., 1.]),)], plot=False)
#     root_chord_output = csdl.norm(left_wing_root_leading_edge - left_wing_root_trailing_edge)
#     tip_chord_output = csdl.norm(left_wing_tip_leading_edge - left_wing_tip_trailing_edge)

#     geometry5.plot()
#     '''
#     # NOTE: No CSDL simulator yet.

#     m3l_model.register_output(half_wingspan_output)

#     csdl_model = m3l_model.assemble()
#     from python_csdl_backend import Simulator
#     simulator = Simulator(csdl_model)
#     simulator.run()
#     simulator.check_totals(of=[half_wingspan_output.operation.name + '.' +  half_wingspan_output.name],
#                             wrt=['half_wingspan', 'root_chord', 'tip_chord', 'mid_chord', 'left_wing_right_wing_connection'])

#     print('new half wingspan', half_wingspan_output)
#     print('new root chord', root_chord_output)
#     # THEN REVISIT ACTUATIONS
#     # -- Do I want to create the framework of creating actuators, etc.?
#     '''


#     print('hi')