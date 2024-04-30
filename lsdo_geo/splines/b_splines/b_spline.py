from __future__ import annotations
from typing import Union
import m3l
import csdl

import numpy as np
import scipy.sparse as sps
# import array_mapper as am
import vedo
from pathlib import Path
import pickle

from lsdo_b_splines_cython.cython.basis_matrix_surface_py import get_basis_surface_matrix
from lsdo_b_splines_cython.cython.surface_projection_py import compute_surface_projection
from lsdo_b_splines_cython.cython.volume_projection_py import compute_volume_projection

from lsdo_geo.splines.b_splines.b_spline_space import BSplineSpace

from dataclasses import dataclass

# TODO: I'm going to leave this class as surface for now, but I want to generalize to n-dimensional.

@dataclass
class BSpline(m3l.Function):
    '''
    B-spline class

    Attributes
    ----------
    name : str
        The name of the B-spline.
    space : BSplineSpace
        The space that the B-spline is in.
    coefficients : m3l.Variable
        The coefficients of the B-spline.
    num_physical_dimensions : int
        The number of physical dimensions that the B-spline is in.

    Methods
    -------
    evaluate(parametric_coordinates:np.ndarray, parametric_derivative_order:tuple=None) -> m3l.Variable
        Evaluates the B-spline at the given parametric coordinates.
    compute_evaluation_map(parametric_coordinates:np.ndarray, parametric_derivative_order:tuple=None,
                           expand_map_for_physical:bool=True) -> sps.csc_matrix
        Computes the evaluation map for the B-spline.
    project(points:np.ndarray, direction:np.ndarray=None, grid_search_density:int=50,
            max_iterations:int=100, return_parametric_coordinates:bool=False, plot:bool=False)
        Projects the given points onto the B-spline.
    plot(point_types:list=['evaluated_points', 'coefficients'], plot_types:list=['mesh'],
         opacity:float=1., color:str='#00629B', surface_texture:str="", additional_plotting_elements:list=[], show:bool=True)
        Plots the B-spline Surface.


    Notes
    -----
    The B-spline is defined by the following equation:

    .. math::

        \\mathbf{P}(u,v) = \\sum_{i=0}^{n_u-1} \\sum_{j=0}^{n_v-1} \\mathbf{C}_{i,j} N_{i,p}(u) N_{j,q}(v)

    where :math:`\\mathbf{P}(u,v)` is the B-spline surface, :math:`\\mathbf{C}_{i,j}` are the coefficients, :math:`N_{i,p}(u)` are the basis functions
    in the u-direction, and :math:`N_{j,q}(v)` are the basis functions in the v-direction.

    The basis functions are defined by the Cox-de Boor recursion formula:

    .. math::

        N_{i,0}(u) = \\begin{cases}
            1 & \\text{if } u_i \\leq u < u_{i+1} \\\\
            0 & \\text{otherwise}
        \\end{cases}

        N_{i,p}(u) = \\frac{u-u_i}{u_{i+p}-u_i} N_{i,p-1}(u) + \\frac{u_{i+p+1}-u}{u_{i+p+1}-u_{i+1}} N_{i+1,p-1}(u)

    where :math:`u_i` are the knots.

    The evaluation map is defined by the following equation:

    .. math::

        \\mathbf{P}(u,v) = \\mathbf{B}(u,v) \\mathbf{C}

    where :math:`\\mathbf{B}(u,v)` is the evaluation map and :math:`\\mathbf{C}` are the coefficients.
    '''
    space : BSplineSpace    # Just overwriting the type hint for the space attribute
    num_physical_dimensions : int

    def __post_init__(self):
        self.coefficients_shape = self.space.parametric_coefficients_shape + (self.num_physical_dimensions,)
        self.num_coefficients = np.prod(self.coefficients_shape)
        self.num_coefficient_elements = self.space.num_coefficient_elements

        if len(self.coefficients) != self.num_coefficients:
            if np.prod(self.coefficients.shape) == np.prod(self.coefficients_shape):
                self.coefficients = self.coefficients.reshape((-1,))
            else:
                raise Exception("Coefficients size doesn't match the function space's coefficients shape.")
            
        if type(self.coefficients) is np.ndarray:
            self.coefficients = m3l.Variable(name=f'{self.name}_coefficients', shape=self.coefficients.shape, value=self.coefficients)

        # Promote attributes to make this object a bit more intuitive
        # Not doing this for now to make objects more lightweight
        # self.order = self.space.order
        # self.knots = self.space.knots
        # self.num_coefficients = self.space.num_coefficients
        # self.num_parametric_dimensions = self.space.num_parametric_dimensions

    def copy(self):
        copied_name = f'{self.name}_copy'
        copied_coefficients = self.coefficients.copy()
        space = self.space
        return BSpline(name=copied_name, space=space, coefficients=copied_coefficients, num_physical_dimensions=self.num_physical_dimensions)

    
    def evaluate(self, parametric_coordinates:np.ndarray, parametric_derivative_order:tuple=None, plot:bool=False) -> m3l.Variable:
        # basis_matrix = self.compute_evaluation_map(parametric_coordinates, parametric_derivative_order)
        # output = basis_matrix.dot(self.coefficients)

        evaluation_map = self.compute_evaluation_map(parametric_coordinates=parametric_coordinates,
                                                     parametric_derivative_order=parametric_derivative_order)
        
        # evaluation_map = m3l.Variable(name=f'evaluation_map', shape=evaluation_map.shape, operation=None, value=evaluation_map)

        if type(self.coefficients) is m3l.Variable:
            output = m3l.matvec(evaluation_map, self.coefficients)
        else:
            output = evaluation_map.dot(self.coefficients)
            output = m3l.Variable(name=f'{self.name}_evaluated_points', shape=output.shape, value=output)

        if plot:
            plotter = vedo.Plotter()
            b_spline_meshes = self.plot(opacity=0.25, show=False)
            # Plot 
            plotting_points = []
            # TODO This will break if geometry is not one of the properties. Fix this.
            flattened_projected_points = (output.value).reshape((-1, 3)) # last axis usually has length 3 for x,y,z
            plotting_projected_points = vedo.Points(flattened_projected_points, r=12, c='#00C6D7')  # TODO make this (1,3) instead of (3,)
            plotting_points.append(plotting_projected_points)
            plotter.show(b_spline_meshes, plotting_points, 'Projected Points', axes=1, viewup="z", interactive=True)

        return output
    
    
    def compute_evaluation_map(self, parametric_coordinates:np.ndarray, parametric_derivative_order:tuple=None,
                               expand_map_for_physical:bool=True) -> sps.csc_matrix:
        '''
        Computes the evaluation map for the B-spline.

        Parameters
        ----------
        parametric_coordinates : np.ndarray
            The parametric coordinates to evaluate the B-spline at.
        parametric_derivative_order : tuple
            The order of the parametric derivative to evaluate the B-spline at. 0 is regular evaluation, 1 is first derivative, etc.
        expand_map_for_physical : bool
            Whether to expand the map for physical dimensions. For example, instead of the map being used to multiply with coefficients in shape
            (nu*nv,3), the map is expanded to be used to multiply with coefficients in shape (nu*nv*3,) where 3 is 
            the number of physical dimensions (most commonly x,y,z).

        Returns
        -------
        map : sps.csc_matrix
            The evaluation map.
        '''
        from lsdo_geo.splines.b_splines.b_spline_functions import compute_evaluation_map

        if expand_map_for_physical:
            expansion_factor = self.num_physical_dimensions
        else:
            expansion_factor = 1

        map = compute_evaluation_map(
            parametric_coordinates=parametric_coordinates, order=self.space.order,
            parametric_coefficients_shape=self.space.parametric_coefficients_shape,
            knots=self.space.knots,
            parametric_derivative_order=parametric_derivative_order,
            expansion_factor=expansion_factor,
        )

        return map
    

    def project(self, points:np.ndarray, direction:np.ndarray=None, grid_search_density:int=50,
                    max_iterations:int=100, plot:bool=False) -> m3l.Variable:
        if len(points.shape) == 1:
            points = points.reshape((-1,self.num_physical_dimensions))

        if type(points) is m3l.Variable:
            points = points.value
        
        input_shape = points.shape
        flattened_points = points.flatten()
        if len(points.shape) > 1:
            num_points = np.cumprod(points.shape[:-1])[-1]
        else:
            num_points = 1

        if direction is None:
            direction = np.zeros((num_points*points.shape[-1],))
        else:
            direction = np.tile(direction, num_points)

        # check if projection is stored
        projections_folder = "stored_files/projections"
        name_space_file_path = projections_folder + '/name_space_dict.pickle'
        
        name_space_dict_file_path = Path(name_space_file_path)
        if name_space_dict_file_path.is_file():
            with open(name_space_file_path, 'rb') as handle:
                name_space_dict = pickle.load(handle)
        else:
            Path(projections_folder).mkdir(parents=True, exist_ok=True)
            name_space_dict = {}

        long_name_space = f'{self.name}_{self.coefficients.value}_{str(points)}_{str(direction)}_{grid_search_density}_{max_iterations}'
        if long_name_space in name_space_dict.keys():
            short_name_space = name_space_dict[long_name_space]
            saved_projections_file = projections_folder + f'/{short_name_space}.pickle'
            with open(saved_projections_file, 'rb') as handle:
                parametric_coordinates = pickle.load(handle)

            # if plot:
            #     self.evaluate(parametric_coordinates=parametric_coordinates, plot=plot)
            
        else:
            # Path(projections_folder).mkdir(parents=True, exist_ok=True)
            # stored_projection_file_name_points_part = f'{np.linalg.norm(points, axis=0)}_{direction[:3]}_{grid_search_density}_{max_iterations}_'
            # stored_projection_file_name_b_spline_part = f'{self.name}_{np.linalg.norm(self.coefficients.value)}_{self.space.order}_projection.pickle'
            # stored_projection_filename = stored_projection_file_name_points_part + stored_projection_file_name_b_spline_part
            # stored_projection_file_path = Path(projections_folder + '/' + stored_projection_filename)
            # if stored_projection_file_path.is_file():
            #     with open(projections_folder + f'/{stored_projection_filename}', 'rb') as handle:
            #         parametric_coordinates = pickle.load(handle)
            #         if plot:
            #             # Plot the surfaces that are projected onto
            #             plotter = vedo.Plotter()
            #             b_spline_meshes = self.plot(plot_types=['surface'], opacity=0.25, show=False)
            #             # Plot 
            #             plotting_points = []
            #             projected_points = self.evaluate(parametric_coordinates=parametric_coordinates)
            #             flattened_projected_points = (projected_points.value).reshape((num_points, -1)) # last axis usually has length 3 for x,y,z
            #             plotting_b_spline_coefficients = vedo.Points(flattened_projected_points, r=12, c='blue')  # TODO make this (1,3) instead of (3,)
            #             plotting_points.append(plotting_b_spline_coefficients)
            #             plotter.show(b_spline_meshes, plotting_points, 'Projected Points', axes=1, viewup="z", interactive=True)
            #         return parametric_coordinates

            # If projection is not stored, continue with projection
            if self.space.num_parametric_dimensions == 2:
                u_vec_flattened = np.zeros(num_points)
                v_vec_flattened = np.zeros(num_points)

                num_surfaces = 1

                compute_surface_projection(
                    np.array([self.space.order[0]], dtype=np.int32), np.array([self.coefficients_shape[0]], dtype=np.int32),
                    np.array([self.space.order[1]], dtype=np.int32), np.array([self.coefficients_shape[1]], dtype=np.int32),
                    num_points, max_iterations,
                    flattened_points, 
                    self.coefficients.value.reshape((-1,)),
                    self.space.knots[self.space.knot_indices[0]].copy(), self.space.knots[self.space.knot_indices[1]].copy(),
                    u_vec_flattened, v_vec_flattened, grid_search_density,
                    direction.reshape((-1,)), np.zeros((num_points,), dtype=np.int32), num_surfaces
                )

                parametric_coordinates = np.hstack((u_vec_flattened.reshape((-1,1)), v_vec_flattened.reshape((-1,1))))

            elif self.space.num_parametric_dimensions == 3:
                u_vec_flattened = np.zeros(num_points)
                v_vec_flattened = np.zeros(num_points)
                w_vec_flattened = np.zeros(num_points)

                compute_volume_projection(
                    np.array([self.space.order[0]]), np.array([self.coefficients_shape[0]]),
                    np.array([self.space.order[1]]), np.array([self.coefficients_shape[1]]),
                    np.array([self.space.order[2]]), np.array([self.coefficients_shape[2]]),
                    num_points, max_iterations,
                    flattened_points, 
                    self.coefficients.value.reshape((-1,)),
                    self.space.knots[self.space.knot_indices[0]].copy(), self.space.knots[self.space.knot_indices[1]].copy(),
                    self.space.knots[self.space.knot_indices[2]].copy(),
                    u_vec_flattened, v_vec_flattened, w_vec_flattened, grid_search_density, direction.reshape((-1,))
                )

                parametric_coordinates = np.hstack((u_vec_flattened.reshape((-1,1)), v_vec_flattened.reshape((-1,1)), w_vec_flattened.reshape((-1,1))))

            import string 
            import random
            characters = string.ascii_letters + string.digits  # Alphanumeric characters
            # Generate a random string of the specified length
            short_name_space = ''.join(random.choice(characters) for _ in range(6))
            name_space_dict[long_name_space] = short_name_space
            with open(name_space_file_path, 'wb+') as handle:
                pickle.dump(name_space_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(projections_folder + f'/{short_name_space}.pickle', 'wb+') as handle:
                pickle.dump(parametric_coordinates, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # map = self.compute_evaluation_map(parametric_coordinates)
        # projected_points = am.array(input=self.coefficients, linear_map=map, shape=input_shape)

        if plot:
            self.evaluate(parametric_coordinates=parametric_coordinates, plot=plot)

        # if plot:
        #     # Plot the surfaces that are projected onto
        #     plotter = vedo.Plotter()
        #     b_spline_meshes = self.plot(plot_types=['surface'], opacity=0.25, show=False)
        #     # Plot 
        #     plotting_points = []
        #     projected_points = self.evaluate(parametric_coordinates=parametric_coordinates)
        #     flattened_projected_points = (projected_points.value).reshape((num_points, -1)) # last axis usually has length 3 for x,y,z
        #     plotting_b_spline_coefficients = vedo.Points(flattened_projected_points, r=12, c='blue')  # TODO make this (1,3) instead of (3,)
        #     plotting_points.append(plotting_b_spline_coefficients)
        #     plotter.show(b_spline_meshes, plotting_points, 'Projected Points', axes=1, viewup="z", interactive=True)

        # u_vec = u_vec_flattened.reshape(tuple(input_shape[:-1],)+(1,))
        # v_vec = v_vec_flattened.reshape(tuple(input_shape[:-1],)+(1,))
        # parametric_coordinates = np.concatenate((u_vec, v_vec), axis=-1)
        # with open(f'stored_files/projections/{stored_projection_filename}', 'wb+') as handle:
        #     pickle.dump(parametric_coordinates, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return parametric_coordinates

        # if return_parametric_coordinates:
            # return parametric_coordinates
            # return (u_vec_flattened, v_vec_flattened)
            # return np.hstack((u_vec_flattened.reshape((-1,1)), v_vec_flattened.reshape((-1,1))))
        # else:
        #     return projected_points


    def plot(self, point_types:list=['evaluated_points'], plot_types:list=['surface'],
              opacity:float=1., color:Union[str,BSpline]='#00629B', surface_texture:str="", additional_plotting_elements:list=[], show:bool=True):
        '''
        Plots the B-spline Surface.

        Parameters
        -----------
        points_type : list
            The type of points to be plotted. {evaluated_points, coefficients}
        plot_types : list
            The type of plot {surface, wireframe, point_cloud}
        opactity : float
            The opacity of the plot. 0 is fully transparent and 1 is fully opaque.
        color : str
            The 6 digit color code to plot the B-spline as.
        surface_texture : str = "" {"metallic", "glossy", ...}, optional
            The surface texture to determine how light bounces off the surface.
            See https://github.com/marcomusy/vedo/blob/master/examples/basic/lightings.py for options.
        additional_plotting_elemets : list
            Vedo plotting elements that may have been returned from previous plotting functions that should be plotted with this plot.
        show : bool
            A boolean on whether to show the plot or not. If the plot is not shown, the Vedo plotting element is returned.
        '''
        if self.space.num_parametric_dimensions == 1:
            return self.plot_curve(point_types=point_types, plot_types=plot_types, opacity=opacity, color=color,
                                    additional_plotting_elements=additional_plotting_elements, show=show)
        elif self.space.num_parametric_dimensions == 2:
            return self.plot_surface(point_types=point_types, plot_types=plot_types, opacity=opacity, color=color, 
                                     surface_texture=surface_texture, additional_plotting_elements=additional_plotting_elements, show=show)
        elif self.space.num_parametric_dimensions == 3:
            return self.plot_volume(point_types=point_types, plot_types=plot_types, opacity=opacity, color=color,
                                    surface_texture=surface_texture, additional_plotting_elements=additional_plotting_elements, show=show)
        
    def plot_curve(self, point_types:list=['evaluated_points'], plot_types:list=['wireframe'],
              opacity:float=1., color:Union[str,BSpline]='#00629B', additional_plotting_elements:list=[], show:bool=True):
        '''
        Plots the B-spline Surface.

        Parameters
        -----------
        points_type : list
            The type of points to be plotted. {evaluated_points, coefficients}
        plot_types : list
            The type of plot {curve, point_cloud}
        opactity : float
            The opacity of the plot. 0 is fully transparent and 1 is fully opaque.
        color : str
            The 6 digit color code to plot the B-spline as.
        additional_plotting_elemets : list
            Vedo plotting elements that may have been returned from previous plotting functions that should be plotted with this plot.
        show : bool
            A boolean on whether to show the plot or not. If the plot is not shown, the Vedo plotting element is returned.
        '''
        
        plotting_elements = additional_plotting_elements.copy()

        num_physical_dimensions = self.num_physical_dimensions

        for point_type in point_types:
            if point_type == 'evaluated_points':
                num_points_u = 15
                u_vec = np.linspace(0., 1., num_points_u)
                parametric_coordinates = u_vec.reshape((-1,1))
                plotting_points = self.evaluate(parametric_coordinates=parametric_coordinates).value

                if num_physical_dimensions == 1:    # plot against the parametric coordinate
                    # scale u axis to be more visually clear based on scaling of parameter
                    u_axis_scaling = np.max(plotting_points) - np.min(plotting_points)
                    if u_axis_scaling != 0:
                        parametric_coordinates = u_vec * u_axis_scaling
                    plotting_points = np.hstack((parametric_coordinates.reshape((-1,1)), plotting_points.reshape((-1,1)), np.zeros((num_points_u,1))))
                elif num_physical_dimensions == 2:  # plot against the parametric coordinate
                    # scale u axis to be more visually clear based on scaling of parameter
                    u_axis_scaling = np.max(plotting_points) - np.min(plotting_points)
                    parametric_coordinates = u_vec * u_axis_scaling
                    plotting_points = np.hstack((parametric_coordinates.reshape((-1,1)), plotting_points.reshape((-1,2))))

                if type(color) is BSpline:
                    plotting_colors = color.evaluate(parametric_coordinates=parametric_coordinates).value

            elif point_type == 'coefficients':
                plotting_points_shape = self.coefficients_shape
                # num_plotting_points = np.cumprod(plotting_points_shape[:-1])[-1]
                plotting_points = self.coefficients.value.reshape((-1,num_physical_dimensions))

            if 'point_cloud' in plot_types:
                plotting_elements.append(vedo.Points(plotting_points, r=6).opacity(opacity).color('darkred'))

            if 'curve' in plot_types or 'wireframe' in plot_types or 'surface' in plot_types:
                from vedo import Line
                plotting_line = Line(plotting_points).color(color).linewidth(3)
                
                if 'wireframe' in plot_types:
                    num_points = np.cumprod(plotting_points.shape[:-1])[-1]
                    plotting_elements.append(vedo.Points(plotting_points.reshape((num_points,-1)), r=12).color(color))
                
                if type(color) is str:
                    plotting_line.color(color)
                elif type(color) is BSpline:
                    plotting_line.cmap('jet', plotting_colors)

                plotting_elements.append(plotting_line)

        if show:
            plotter = vedo.Plotter()
            if num_physical_dimensions == 1:
                viewup = "y"
            else:
                viewup = "z"
            plotter.show(plotting_elements, f'B-spline Curve: {self.name}', axes=1, viewup=viewup, interactive=True)
            return plotting_elements
        else:
            return plotting_elements


    def plot_surface(self, point_types:list=['evaluated_points'], plot_types:list=['surface'],
              opacity:float=1., color:Union[str,BSpline]='#00629B', surface_texture:str="", additional_plotting_elements:list=[], show:bool=True):
        '''
        Plots the B-spline Surface.

        Parameters
        -----------
        points_type : list
            The type of points to be plotted. {evaluated_points, coefficients}
        plot_types : list
            The type of plot {surface, wireframe, point_cloud}
        opactity : float
            The opacity of the plot. 0 is fully transparent and 1 is fully opaque.
        color : str
            The 6 digit color code to plot the B-spline as.
        surface_texture : str = "" {"metallic", "glossy", ...}, optional
            The surface texture to determine how light bounces off the surface.
            See https://github.com/marcomusy/vedo/blob/master/examples/basic/lightings.py for options.
        additional_plotting_elemets : list
            Vedo plotting elements that may have been returned from previous plotting functions that should be plotted with this plot.
        show : bool
            A boolean on whether to show the plot or not. If the plot is not shown, the Vedo plotting element is returned.
        '''
        
        plotting_elements = additional_plotting_elements.copy()

        num_physical_dimensions = self.num_physical_dimensions

        for point_type in point_types:
            if point_type == 'evaluated_points':
                num_points_u = 15
                num_points_v = 15
                u_vec = np.einsum('i,j->ij', np.linspace(0., 1., num_points_u), np.ones(num_points_v)).reshape((-1,1))
                v_vec = np.einsum('i,j->ij', np.ones(num_points_u), np.linspace(0., 1., num_points_v)).reshape((-1,1))
                parametric_coordinates = np.hstack((u_vec, v_vec))
                plotting_points = self.evaluate(parametric_coordinates=parametric_coordinates).value
                plotting_points_shape = (num_points_u, num_points_v, num_physical_dimensions)

                if type(color) is BSpline:
                    plotting_colors = color.evaluate(parametric_coordinates=parametric_coordinates).value
            elif point_type == 'coefficients':
                plotting_points_shape = self.coefficients_shape
                # num_plotting_points = np.cumprod(plotting_points_shape[:-1])[-1]
                plotting_points = self.coefficients.value.reshape((-1,num_physical_dimensions))

            if 'point_cloud' in plot_types:
                plotting_elements.append(vedo.Points(plotting_points, r=6).opacity(opacity).color('darkred'))

            if 'surface' in plot_types or 'wireframe' in plot_types:
                num_plot_u = plotting_points_shape[0]
                num_plot_v = plotting_points_shape[1]

                vertices = []
                faces = []
                plotting_points_reshaped = plotting_points.reshape(plotting_points_shape)
                for u_index in range(num_plot_u):
                    for v_index in range(num_plot_v):
                        vertex = tuple(plotting_points_reshaped[u_index, v_index, :])
                        vertices.append(vertex)
                        if u_index != 0 and v_index != 0:
                            face = tuple((
                                (u_index-1)*num_plot_v+(v_index-1),
                                (u_index-1)*num_plot_v+(v_index),
                                (u_index)*num_plot_v+(v_index),
                                (u_index)*num_plot_v+(v_index-1),
                            ))
                            faces.append(face)

                mesh = vedo.Mesh([vertices, faces]).opacity(opacity).lighting(surface_texture)
                if type(color) is str:
                    mesh.color(color)
                elif type(color) is BSpline:
                    mesh.cmap('jet', plotting_colors)
            if 'surface' in plot_types:
                plotting_elements.append(mesh)
            if 'wireframe' in plot_types:
                mesh = vedo.Mesh([vertices, faces]).opacity(opacity)
                plotting_elements.append(mesh.wireframe())

        if show:
            plotter = vedo.Plotter()
            # from vedo import Light
            # light = Light([-1,0,0], c='w', intensity=1)
            # plotter = vedo.Plotter(size=(3200,1000))
            # plotter.show(plotting_elements, light, f'B-spline Surface: {self.name}', axes=1, viewup="z", interactive=True)
            plotter.show(plotting_elements, f'B-spline Surface: {self.name}', axes=1, viewup="z", interactive=True)
            return plotting_elements
        else:
            return plotting_elements

    
    def plot_volume(self, point_types:list=['evaluated_points'], plot_types:list=['surface'],
              opacity:float=1., color:str='#00629B', surface_texture:str="", additional_plotting_elements:list=[], show:bool=True):
        '''
        Plots the B-spline Surface.

        Parameters
        -----------
        points_type : list
            The type of points to be plotted. {evaluated_points, coefficients}
        plot_types : list
            The type of plot {surface, wireframe, point_cloud}
        opactity : float
            The opacity of the plot. 0 is fully transparent and 1 is fully opaque.
        color : str
            The 6 digit color code to plot the B-spline as.
        surface_texture : str = "" {"metallic", "glossy", ...}, optional
            The surface texture to determine how light bounces off the surface.
            See https://github.com/marcomusy/vedo/blob/master/examples/basic/lightings.py for options.
        additional_plotting_elemets : list
            Vedo plotting elements that may have been returned from previous plotting functions that should be plotted with this plot.
        show : bool
            A boolean on whether to show the plot or not. If the plot is not shown, the Vedo plotting element is returned.
        '''
        # Plot the 6 sides of the volume
        plotting_elements = additional_plotting_elements.copy()

        coefficients = self.coefficients.value.reshape(self.space.parametric_coefficients_shape + (self.num_physical_dimensions,))

        for point_type in point_types:
            if point_type == 'evaluated_points':
                num_points_per_dimension = 50
                linspace_dimension = np.linspace(0., 1., num_points_per_dimension)
                linspace_meshgrid = np.meshgrid(linspace_dimension, linspace_dimension)
                linspace_dimension1 = linspace_meshgrid[0].reshape((-1,1))
                linspace_dimension2 = linspace_meshgrid[1].reshape((-1,1))
                zeros_dimension = np.zeros((num_points_per_dimension**2,)).reshape((-1,1))
                ones_dimension = np.ones((num_points_per_dimension**2,)).reshape((-1,1))

                parametric_coordinates = []
                parametric_coordinates.append(np.column_stack((linspace_dimension1, linspace_dimension2, zeros_dimension)))
                parametric_coordinates.append(np.column_stack((linspace_dimension1, linspace_dimension2, ones_dimension)))
                parametric_coordinates.append(np.column_stack((linspace_dimension1, zeros_dimension, linspace_dimension2)))
                parametric_coordinates.append(np.column_stack((linspace_dimension1, ones_dimension, linspace_dimension2)))
                parametric_coordinates.append(np.column_stack((zeros_dimension, linspace_dimension1, linspace_dimension2)))
                parametric_coordinates.append(np.column_stack((ones_dimension, linspace_dimension1, linspace_dimension2)))

                num_points_u = num_points_per_dimension
                num_points_v = num_points_per_dimension
                plotting_points_shape = []
                for i in range(6):
                    plotting_points_shape.append((num_points_u, num_points_v, self.num_physical_dimensions))

                plotting_points = []
                for parametric_coordinate_set in parametric_coordinates:
                    evaluation_map = self.compute_evaluation_map(parametric_coordinates=parametric_coordinate_set, expand_map_for_physical=False)
                    plotting_points.append(evaluation_map.dot(self.coefficients.value.reshape((-1,3))))

                plotting_colors = []
                if type(color) is BSpline:
                    for parametric_coordinate_set in parametric_coordinates:
                        plotting_colors.append(color.evaluate(parametric_coordinate_set).value)
            
            elif point_type == 'coefficients':
                plotting_points = []
                plotting_points.append(coefficients[0,:,:].reshape((-1, self.num_physical_dimensions)))
                plotting_points.append(coefficients[-1,:,:].reshape((-1, self.num_physical_dimensions)))
                plotting_points.append(coefficients[:,0,:].reshape((-1, self.num_physical_dimensions)))
                plotting_points.append(coefficients[:,-1,:].reshape((-1, self.num_physical_dimensions)))
                plotting_points.append(coefficients[:,:,0].reshape((-1, self.num_physical_dimensions)))
                plotting_points.append(coefficients[:,:,-1].reshape((-1, self.num_physical_dimensions)))

                plotting_points_shape = []
                plotting_points_shape.append(coefficients[0,:,:].shape)
                plotting_points_shape.append(coefficients[-1,:,:].shape)
                plotting_points_shape.append(coefficients[:,0,:].shape)
                plotting_points_shape.append(coefficients[:,-1,:].shape)
                plotting_points_shape.append(coefficients[:,:,0].shape)
                plotting_points_shape.append(coefficients[:,:,-1].shape)


            for i in range(6):
                if 'point_cloud' in plot_types:
                    plotting_elements.append(vedo.Points(plotting_points[i], r=6).opacity(opacity).color('darkred'))

                if 'surface' in plot_types or 'wireframe' in plot_types:
                    num_plot_u = plotting_points_shape[i][0]
                    num_plot_v = plotting_points_shape[i][1]

                    vertices = []
                    faces = []
                    plotting_points_reshaped = plotting_points[i].reshape(plotting_points_shape[i])
                    for u_index in range(num_plot_u):
                        for v_index in range(num_plot_v):
                            vertex = tuple(plotting_points_reshaped[u_index, v_index, :])
                            vertices.append(vertex)
                            if u_index != 0 and v_index != 0:
                                face = tuple((
                                    (u_index-1)*num_plot_v+(v_index-1),
                                    (u_index-1)*num_plot_v+(v_index),
                                    (u_index)*num_plot_v+(v_index),
                                    (u_index)*num_plot_v+(v_index-1),
                                ))
                                faces.append(face)

                    mesh = vedo.Mesh([vertices, faces]).opacity(opacity).lighting(surface_texture)
                    if type(color) is str:
                        mesh.color(color)
                    elif type(color) is BSpline:
                        mesh.cmap('jet', plotting_colors[i])

                if 'surface' in plot_types:
                    plotting_elements.append(mesh)
                if 'wireframe' in plot_types:
                    mesh = vedo.Mesh([vertices, faces]).opacity(opacity)
                    plotting_elements.append(mesh.wireframe())

        
        if show:
            plotter = vedo.Plotter()
            plotter.show(plotting_elements, f'B-spline Volume: {self.name}', axes=1, viewup="y", interactive=True)
            return plotting_elements
        else:
            return plotting_elements

        
    # def plot_volume(self, point_types:list=['evaluated_points'], plot_types:list=['surface'],
    #           opacity:float=1., color:str='#00629B', surface_texture:str="", additional_plotting_elements:list=[], show:bool=True):
    #     '''
    #     Plots the B-spline Surface.

    #     Parameters
    #     -----------
    #     points_type : list
    #         The type of points to be plotted. {evaluated_points, coefficients}
    #     plot_types : list
    #         The type of plot {surface, wireframe, point_cloud}
    #     opactity : float
    #         The opacity of the plot. 0 is fully transparent and 1 is fully opaque.
    #     color : str
    #         The 6 digit color code to plot the B-spline as.
    #     surface_texture : str = "" {"metallic", "glossy", ...}, optional
    #         The surface texture to determine how light bounces off the surface.
    #         See https://github.com/marcomusy/vedo/blob/master/examples/basic/lightings.py for options.
    #     additional_plotting_elemets : list
    #         Vedo plotting elements that may have been returned from previous plotting functions that should be plotted with this plot.
    #     show : bool
    #         A boolean on whether to show the plot or not. If the plot is not shown, the Vedo plotting element is returned.
    #     '''
    #     # Plot the 6 sides of the volume
    #     plotting_elements = additional_plotting_elements.copy()

    #     coefficients = self.coefficients.value.reshape(self.space.parametric_coefficients_shape + (self.num_physical_dimensions,))
        
    #     plotting_elements = self.plot_section(coefficients[:,:,0], plot_types=plot_types, opacity=opacity, color=color,
    #                                           additional_plotting_elements=plotting_elements, show=False)
    #     plotting_elements = self.plot_section(coefficients[:,:,-1], plot_types=plot_types, opacity=opacity, color=color,
    #                                           additional_plotting_elements=plotting_elements, show=False)
    #     plotting_elements = self.plot_section(coefficients[:,0,:], plot_types=plot_types, opacity=opacity, color=color,
    #                                           additional_plotting_elements=plotting_elements, show=False)
    #     plotting_elements = self.plot_section(coefficients[:,-1,:], plot_types=plot_types, opacity=opacity, color=color,
    #                                           additional_plotting_elements=plotting_elements, show=False)
    #     plotting_elements = self.plot_section(coefficients[0,:,:], plot_types=plot_types, opacity=opacity, color=color,
    #                                           additional_plotting_elements=plotting_elements, show=False)
    #     plotting_elements = self.plot_section(coefficients[-1,:,:], plot_types=plot_types, opacity=opacity, color=color,
    #                                           additional_plotting_elements=plotting_elements, show=False)

    #     if show:
    #         plotter = vedo.Plotter()
    #         plotter.show(plotting_elements, f'B-spline Volume: {self.name}', axes=1, viewup="y", interactive=True)
    #         return plotting_elements
    #     else:
    #         return plotting_elements

    def plot_section(self, points:np.ndarray, point_types:list=['evaluated_points'], plot_types:list=['surface'],
              opacity:float=1., color:str='#00629B', surface_texture:str="", additional_plotting_elements:list=[], show:bool=True):
        '''
        Plots the B-spline Surface.

        Parameters
        -----------
        point_types : list  NOTE: Doesn't use this. Only plots coefficients.
            The type of points to be plotted. {evaluated_points, coefficients}
        plot_types : list
            The type of plot {surface, wireframe, point_cloud}
        opactity : float
            The opacity of the plot. 0 is fully transparent and 1 is fully opaque.
        color : str
            The 6 digit color code to plot the B-spline as.
        surface_texture : str = "" {"metallic", "glossy", ...}, optional
            The surface texture to determine how light bounces off the surface.
            See https://github.com/marcomusy/vedo/blob/master/examples/basic/lightings.py for options.
        additional_plotting_elemets : list
            Vedo plotting elements that may have been returned from previous plotting functions that should be plotted with this plot.
        show : bool
            A boolean on whether to show the plot or not. If the plot is not shown, the Vedo plotting element is returned.
        '''
        plotting_elements = additional_plotting_elements.copy()
        
        if 'point_cloud' in plot_types:
            num_points = points.shape[0]*points.shape[1]
            if 'surface' in plot_types:
                point_opacity = (0.75*opacity + 0.25*1.)
            else:
                point_opacity = opacity
            plotting_elements.append(vedo.Points(points.reshape((num_points,-1)), r=6).opacity(point_opacity).color('darkred'))

        if 'surface' in plot_types or 'wireframe' in plot_types:
            num_control_points_u = points.shape[0]
            num_control_points_v = points.shape[1]
            vertices = []
            faces = []
            for u_index in range(num_control_points_u):
                for v_index in range(num_control_points_v):
                    vertex = tuple(points[u_index, v_index, :])
                    vertices.append(vertex)
                    if u_index != 0 and v_index != 0:
                        face = tuple((
                            (u_index-1)*num_control_points_v+(v_index-1),
                            (u_index-1)*num_control_points_v+(v_index),
                            (u_index)*num_control_points_v+(v_index),
                            (u_index)*num_control_points_v+(v_index-1),
                        ))
                        faces.append(face)

            

            mesh = vedo.Mesh([vertices, faces]).opacity(opacity).color(color)
        if 'surface' in plot_types:
            plotting_elements.append(mesh)
        if 'wireframe' in plot_types:
            mesh = vedo.Mesh([vertices, faces]).opacity(opacity).color(color)
            plotting_elements.append(mesh.wireframe().color(color))

        if show:
            plotter = vedo.Plotter()
            plotter.show(plotting_elements, 'Surface', axes=1, viewup="z", interactive=True)
            return plotting_elements
        else:
            return plotting_elements


if __name__ == "__main__":
    from lsdo_geo.splines.b_splines.b_spline_space import BSplineSpace

    num_coefficients = 10
    num_physical_dimensions = 3
    order = 4
    space_of_cubic_b_spline_surfaces_with_10_cp = BSplineSpace(name='cubic_b_spline_surfaces_10_cp', order=(order,order),
                                                              parametric_coefficients_shape=(num_coefficients,num_coefficients))

    # coefficients_line = np.zeros((num_coefficients,))
    # coefficients_line[order//2:-order//2] = np.linspace(0., 1., num_coefficients-order)
    # coefficients_line[-order//2:] = 1.
    coefficients_line = np.linspace(0., 1., num_coefficients)
    coefficients_x, coefficients_y = np.meshgrid(coefficients_line,coefficients_line)
    coefficients = np.stack((coefficients_x, coefficients_y, 0.1*np.random.rand(num_coefficients,num_coefficients)), axis=-1)

    b_spline = BSpline(name='test_b_spline', space=space_of_cubic_b_spline_surfaces_with_10_cp, coefficients=coefficients,
                        num_physical_dimensions=num_physical_dimensions)

    plotting_elements = b_spline.plot(point_types=['evaluated_points'], plot_types=['surface'])

    parametric_coordinates = np.array([
        [0., 0.],
        [0., 1.],
        [1., 0.],
        [1., 1.],
        [0.5, 0.5],
        [0.25, 0.75]
    ])

    print('points: ', b_spline.evaluate(parametric_coordinates=parametric_coordinates, parametric_derivative_order=(0,0)))
    print('derivative wrt u:', b_spline.evaluate(parametric_coordinates=parametric_coordinates, parametric_derivative_order=(1,0)))
    print('second derivative wrt u: ', b_spline.evaluate(parametric_coordinates=parametric_coordinates, parametric_derivative_order=(2,0)))

    projecting_points_z = np.zeros((6,))
    projecting_points = np.stack((parametric_coordinates[:,0], parametric_coordinates[:,1], projecting_points_z), axis=-1)

    b_spline.project(points=projecting_points, plot=True)

    num_fitting_points = 25
    u_vec = np.einsum('i,j->ij', np.linspace(0., 1., num_fitting_points), np.ones(num_fitting_points)).flatten().reshape((-1,1))
    v_vec = np.einsum('i,j->ij', np.ones(num_fitting_points), np.linspace(0., 1., num_fitting_points)).flatten().reshape((-1,1))
    parametric_coordinates = np.hstack((u_vec, v_vec))

    grid_points = b_spline.evaluate(parametric_coordinates=parametric_coordinates, parametric_derivative_order=(0,0), plot=True
                                    ).value.reshape((num_fitting_points,num_fitting_points,3))

    from lsdo_geo.splines.b_splines.b_spline_functions import fit_b_spline

    new_b_spline = fit_b_spline(fitting_points=grid_points, parametric_coordinates=parametric_coordinates, num_coefficients=(15,),
                                order=(5,), regularization_parameter=1.e-3)
    new_b_spline.plot()