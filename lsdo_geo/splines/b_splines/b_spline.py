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

            # elif self.space.num_parametric_dimensions == 3:
            #     u_vec_flattened = np.zeros(num_points)
            #     v_vec_flattened = np.zeros(num_points)
            #     w_vec_flattened = np.zeros(num_points)

            #     compute_volume_projection(
            #         np.array([self.space.order[0]]), np.array([self.coefficients_shape[0]]),
            #         np.array([self.space.order[1]]), np.array([self.coefficients_shape[1]]),
            #         np.array([self.space.order[2]]), np.array([self.coefficients_shape[2]]),
            #         num_points, max_iterations,
            #         flattened_points, 
            #         self.coefficients.value.reshape((-1,)),
            #         self.space.knots[self.space.knot_indices[0]].copy(), self.space.knots[self.space.knot_indices[1]].copy(),
            #         self.space.knots[self.space.knot_indices[2]].copy(),
            #         u_vec_flattened, v_vec_flattened, w_vec_flattened, grid_search_density, direction.reshape((-1,))
            #     )

            #     parametric_coordinates = np.hstack((u_vec_flattened.reshape((-1,1)), v_vec_flattened.reshape((-1,1)), w_vec_flattened.reshape((-1,1))))
            elif self.space.num_parametric_dimensions == 3:
                parametric_coordinates = self._experimental_projection(points=points, direction=direction, 
                                                                       grid_search_density_parameter=grid_search_density, 
                                                                       max_newton_iterations=max_iterations, plot=False)


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

    
    def _experimental_projection(self, points:np.ndarray, direction:np.ndarray=None, grid_search_density_parameter:int=1, 
                max_newton_iterations:int=100, newton_tolerance:float=1e-6, plot:bool=False) -> csdl.Variable:
        '''
        Projects a set of points onto the function. The points to project must be provided. If a direction is provided, the projection will find
        the points on the function that are closest to the axis defined by the direction. If no direction is provided, the projection will find the
        points on the function that are closest to the points to project. The grid search density parameter controls the density of the grid search
        used to find the initial guess for the Newton iterations. The max newton iterations and newton tolerance control the convergence of the
        Newton iterations. If plot is True, a plot of the projection will be displayed.

        NOTE: Distance is measured by the 2-norm.

        Parameters
        ----------
        points : np.ndarray -- shape=(num_points, num_phyiscal_dimensions)
            The points to project onto the function.
        direction : np.ndarray = None -- shape=(num_parametric_dimensions,)
            The direction of the projection.
        grid_search_density_parameter : int = 1
            The density of the grid search used to find the initial guess for the Newton iterations.
        max_newton_iterations : int = 100
            The maximum number of Newton iterations.
        newton_tolerance : float = 1e-6
            The tolerance for the Newton iterations.
        plot : bool = False
            Whether or not to plot the projection.
        '''
        num_physical_dimensions = points.shape[-1]

        points = points.reshape((-1, num_physical_dimensions))

        grid_search_resolution = 10*grid_search_density_parameter//self.space.num_parametric_dimensions + 1
        if isinstance(grid_search_resolution, int) or len(grid_search_resolution) == 1:
            grid_search_resolution = (grid_search_resolution,)*self.num_parametric_dimensions

        mesh_grid_input = []
        for dimension_index in range(self.space.num_parametric_dimensions):
            mesh_grid_input.append(np.linspace(0., 1., grid_search_resolution[dimension_index]))

        parametric_coordinates_tuple = np.meshgrid(*mesh_grid_input, indexing='ij')
        for dimensions_index in range(self.space.num_parametric_dimensions):
            parametric_coordinates_tuple[dimensions_index] = parametric_coordinates_tuple[dimensions_index].reshape((-1,1))

        parametric_grid_search = np.hstack(parametric_coordinates_tuple)
        # grid_search_resolution = 100

        # Generate parametric grid
        # Evaluate grid of points
        grid_search_values = self.evaluate(parametric_grid_search, coefficients=self.coefficients.value)
        points_expanded = np.repeat(points[:,np.newaxis,:], grid_search_values.shape[0], axis=1)
        grid_search_displacements = grid_search_values - points_expanded
        grid_search_distances = np.linalg.norm(grid_search_displacements, axis=2)

        # Perform a grid search
        if direction is None:
            # If no direction is provided, the projection will find the points on the function that are closest to the points to project.
            # The grid search will be used to find the initial guess for the Newton iterations
            
            # Find closest point on function to each point to project
            # closest_point_indices = np.argmin(np.linalg.norm(grid_search_values - points, axis=1))
            closest_point_indices = np.argmin(grid_search_distances, axis=1)

        else:
            # If a direction is provided, the projection will find the points on the function that are closest to the axis defined by the direction.
            # The grid search will be used to find the initial guess for the Newton iterations
            rho = 1e-3
            grid_search_distances_along_axis = np.dot(grid_search_displacements, direction)
            grid_search_distances_from_axis_squared = (1 + rho)*grid_search_distances**2 - grid_search_distances_along_axis**2
            closest_point_indices = np.argmin(grid_search_distances_from_axis_squared, axis=1)

        # Use the parametric coordinate corresponding to each closest point as the initial guess for the Newton iterations
        initial_guess = parametric_grid_search[closest_point_indices]

        # Experimental implementation that does all the Newton optimizations at once to vectorize many of the computations
        current_guess = initial_guess.copy()
        points_left_to_converge = np.arange(points.shape[0])
        for j in range(max_newton_iterations):
            # Perform B-spline evaluations needed for gradient and hessian (0th, 1st, and 2nd order derivatives needed)
            function_values = self.evaluate(current_guess[points_left_to_converge], coefficients=self.coefficients.value)
            displacements = (points[points_left_to_converge] - function_values).reshape(points_left_to_converge.shape[0], num_physical_dimensions)
            
            d_displacement_d_parametric = np.zeros((points_left_to_converge.shape[0], num_physical_dimensions, self.space.num_parametric_dimensions))
            d2_displacement_d_parametric2 = np.zeros((points_left_to_converge.shape[0], num_physical_dimensions, 
                                                      self.space.num_parametric_dimensions, self.space.num_parametric_dimensions))

            for k in range(self.space.num_parametric_dimensions):
                parametric_derivative_orders = np.zeros((self.space.num_parametric_dimensions,), dtype=int)
                parametric_derivative_orders[k] = 1
                # d_displacement_d_parametric[:, :, k] = -np.tensordot(
                #     self.space.compute_basis_matrix(current_guess, parametric_derivative_orders=parametric_derivative_orders),
                #     self.coefficients.value.reshape(-1, num_physical_dimensions), axes=[1,0])
                d_displacement_d_parametric[:, :, k] = -self.compute_evaluation_map(current_guess[points_left_to_converge], 
                                                                                        parametric_derivative_orders=parametric_derivative_orders,
                                                                                        expand_map_for_physical=False).dot(
                                                                    self.coefficients.value.reshape(-1, num_physical_dimensions))
                    # NOTE on indices: i=points, j=coefficients, k=physical dimensions

                for m in range(self.space.num_parametric_dimensions):
                    parametric_derivative_orders = np.zeros((self.space.num_parametric_dimensions,))
                    if m == k:
                        parametric_derivative_orders[m] = 2
                    else:
                        parametric_derivative_orders[k] = 1
                        parametric_derivative_orders[m] = 1
                    # d2_displacement_d_parametric2[:, :, k, m] = -np.einsum(
                    #     self.space.compute_basis_matrix(current_guess, parametric_derivative_orders=parametric_derivative_orders),
                    #     self.coefficients.value.reshape((-1, num_physical_dimensions)), 'ij,jk->ik')
                    d2_displacement_d_parametric2[:, :, k, m] = -self.compute_evaluation_map(current_guess[points_left_to_converge], 
                                                                            parametric_derivative_orders=parametric_derivative_orders,
                                                                            expand_map_for_physical=False).dot(
                                                                        self.coefficients.value.reshape((-1, num_physical_dimensions)))
                        # NOTE on indices: i=points, j=coefficients, k=physical dimensions

            # Construct the gradient and hessian
            if direction is None:
                gradient = 2 * np.einsum('ij,ijk->ik', displacements, d_displacement_d_parametric)
                hessian = 2 * (np.einsum('ijk,ijm->ikm', d_displacement_d_parametric, d_displacement_d_parametric)
                            + np.einsum('ij,ijkm->ikm', displacements, d2_displacement_d_parametric2))
            else:
                displacement_dot_d_displacement_d_parametric = np.einsum('ij,ijk->ik', displacements, d_displacement_d_parametric)
                direction_dot_displacement = np.einsum('j,ij->i', direction, displacements)
                direction_dot_d_displacement_d_parametric = np.einsum('j,ijk->ik', direction, d_displacement_d_parametric)
                direction_dot_d2_displacement_d_parametric2 = np.einsum('j,ijkm->ikm', direction, d2_displacement_d_parametric2)
                gradient = 2 * ((1 + rho)*displacement_dot_d_displacement_d_parametric 
                                - direction_dot_displacement[:, np.newaxis] * direction_dot_d_displacement_d_parametric)
                hessian = 2 * ( (1 + rho)*(
                    np.einsum('ijk,ijm->ikm', d_displacement_d_parametric, d_displacement_d_parametric)
                    + np.einsum('ij,ijkm->ikm', displacements, d2_displacement_d_parametric2))
                    - np.einsum('ik,im->ikm', direction_dot_d_displacement_d_parametric, direction_dot_d_displacement_d_parametric)
                    - np.einsum('i,ikm->ikm', direction_dot_displacement, direction_dot_d2_displacement_d_parametric2)
                )

            # Remove dof that are on constrant boundary and want to leave (active subspace method)
            coorinates_to_remove_on_lower_boundary = np.logical_and(current_guess[points_left_to_converge] == 0, gradient > 0)
            coorinates_to_remove_on_upper_boundary = np.logical_and(current_guess[points_left_to_converge] == 1, gradient < 0)
            coorinates_to_remove_boolean = np.logical_or(coorinates_to_remove_on_lower_boundary, coorinates_to_remove_on_upper_boundary)
            coordinates_to_keep_boolean = np.logical_not(coorinates_to_remove_boolean)
            indices_to_keep = []
            for i in range(points_left_to_converge.shape[0]):
                indices_to_keep.append(np.arange(self.space.num_parametric_dimensions)[coordinates_to_keep_boolean[i]])

            reduced_gradients = []
            reduced_hessians = []
            total_gradient_norm = 0.
            counter = 0
            for i in range(points_left_to_converge.shape[0]):
                reduced_gradient = gradient[i, indices_to_keep[counter]]

                if np.linalg.norm(reduced_gradient) < newton_tolerance:
                    points_left_to_converge = np.delete(points_left_to_converge, counter)
                    del indices_to_keep[counter]
                    continue

                # This is after check so it doesn't throw error
                reduced_hessian = hessian[np.ix_(np.array([i]), indices_to_keep[counter], indices_to_keep[counter])][0]    

                reduced_gradients.append(reduced_gradient)
                reduced_hessians.append(reduced_hessian)
                total_gradient_norm += np.linalg.norm(reduced_gradient)
                counter += 1

            # Check for convergence
            if np.linalg.norm(total_gradient_norm) < newton_tolerance:
                break

            # Solve the linear systems
            for i, index in enumerate(points_left_to_converge):
                delta = np.linalg.solve(reduced_hessians[i], -reduced_gradients[i])

                # Update the initial guess
                current_guess[index, indices_to_keep[i]] += delta

            # If any of the coordinates are outside the bounds, set them to the bounds
            current_guess[points_left_to_converge] = np.clip(current_guess[points_left_to_converge], 0., 1.)


        if plot:
            # Use original plotting implementation
            pass
            
            # projection_results = self.evaluate(current_guess).value
            # plotting_elements = []
            # plotting_elements.append(lfs.plot_points(points, color='#00629B', size=10, show=False))
            # plotting_elements.append(lfs.plot_points(projection_results, color='#F5F0E6', size=10, show=False))
            # self.plot(opacity=0.8, additional_plotting_elements=plotting_elements, show=True)

        return current_guess

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