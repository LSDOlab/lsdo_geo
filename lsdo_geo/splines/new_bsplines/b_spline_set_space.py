import m3l
import csdl

import array_mapper as am
import numpy as np
import scipy.sparse as sps

from dataclasses import dataclass

@dataclass
class BSplineSetSpace(m3l.FunctionSpace):
    name : str
    b_spline_spaces : dict
    order : tuple = None
    knots : tuple = None

    def __post_init__(self):
        self.assemble()

    def assemble(self):
        self.num_control_points = 0
        self.coefficient_indices = {}

        for b_spline_space_name, b_spline_space in self.b_spline_spaces.items():
            if self.order is None:
                self.order = list(b_spline_space.order)
                self.knots = list(b_spline_space.knots)
            for i in range(len(b_spline_space.order)):

                self.order[i] = np.vstack((self.order[i], b_spline_space.order[i]))
                self.knots[i] = np.vstack((self.knots[i], b_spline_space.knots[i]))

            self.coefficient_indices[b_spline_space_name] = \
                np.arange(self.num_control_points, self.num_control_points + b_spline_space.num_control_points)
            self.num_control_points += b_spline_space.num_control_points

        self.order = tuple(self.order)
        self.knots = tuple(self.knots)
        self.num_coefficients = self.num_control_points

    def evaluate(self, coefficients:csdl.Variable, mesh:am.MappedArray):
        '''
        Evaluates a function at the mesh values.

        Parameters
        ----------
        coefficients : csdl.Variable
            The coefficients that define function within the function space.
        mesh : am.MappedArray
            The mesh nodes where the function will be evaluated.

        Returns
        -------
        evaluated_function_values : csdl.Variable
            The values of the function evaluated at the mesh nodes.
        '''
        # TODO: As a temporary method, have states use the same function space as geometry and just use the MappedArray map without having to reproject.
        evaluation_map = self.project(mesh)
        evaluated_function_values = csdl.sparsematmat(coefficients, sparse_mat=evaluation_map)
        return evaluated_function_values


    def project(self, points:np.ndarray, grid_search_n:int=25, max_iterations=100, plot:bool=False) -> sps.csc_matrix:
        '''
        Projects points onto the system.

        Parameters
        -----------
        points : {np.ndarray, am.MappedArray}
            The points to be projected onto the system.
        grid_search_n : int, optional
            The resolution of the grid search prior to the Newton iteration for solving the optimization problem.
        max_iterations : int, optional
            The maximum number of iterations for the Newton iteration.
        plot : bool
            A boolean on whether or not to plot the projection result.

        Return
        ------
        linear_map : sps.csc_matrix
            The linear map that will compute the projected points given an input of coefficients.
        '''
        #  TODO Consider parallelizing using Numba, or using the FFD method or in Cython.

        targets = list(self.b_spline_spaces.values())

        if type(points) is am.MappedArray:
            points = points.value

        if len(points.shape) == 1:
            points = points.reshape((1, -1))    # Last axis is reserved for dimensionality of physical space
        
        num_targets = len(targets)
        projected_points_on_each_target = []
        # Project all points onto each target
        for target in targets:   # TODO Parallelize this for loop
            target_projected_points = {}
            target_projected_points['parametric_coordinates'] = target.project(points=points, 
                        grid_search_n=grid_search_n,max_iter=max_iterations, return_parametric_coordinates=True)
            target_projected_points['geometry'] = target.project(points=points,
                        grid_search_n=grid_search_n,max_iter=max_iterations, return_parametric_coordinates=False)

                    # properties are not passed in here because we NEED geometry
            projected_points_on_each_target.append(target_projected_points)

        projected_points_on_each_target_numpy = np.zeros(tuple((num_targets,)) + points.shape)
        for i in range(num_targets):
                projected_points_on_each_target_numpy[i] = projected_points_on_each_target[i]['geometry'].value

        # Compare results across targets to keep best result
        distances = np.linalg.norm(projected_points_on_each_target_numpy - points, axis=-1)   # Computes norm across spatial axis
        closest_surfaces_indices = np.argmin(distances, axis=0) # Take argmin across surfaces
        if len(points.shape) == 1:
            num_points = 1
        else:
            num_points = np.cumprod(points.shape[:-1])[-1]
        flattened_surface_indices = closest_surfaces_indices.flatten()

        projection_receiving_primitives = []

        linear_map = sps.lil_array((num_points, self.num_control_points))
        for i in range(num_points):
            target_index = flattened_surface_indices[i]
            receiving_target = targets[target_index]
            receiving_target_control_point_indices = self.coefficient_indices[receiving_target.name]
            point_parametric_coordinates = projected_points_on_each_target[target_index]['parametric_coordinates']

            point_map_on_receiving_target = receiving_target.compute_evaluation_map(u_vec=np.array([point_parametric_coordinates[0][i]]), 
                                                                                    v_vec=np.array([point_parametric_coordinates[1][i]]))

            linear_map[i, receiving_target_control_point_indices] = point_map_on_receiving_target

        linear_map = linear_map.tocsc()
        return linear_map

        property_shape = points.shape[:-1] + (3,)   # TODO: Figure out how to get this info out.
        property_mapped_array = am.array(self.num_control_points, linear_map=linear_map.tocsc(), shape=property_shape)

        projection_receiving_primitives = list(targets)

        if plot:
            # Plot the surfaces that are projected onto
            plotter = vedo.Plotter()
            primitive_meshes = self.plot(primitives=projection_receiving_primitives, opacity=0.25, show=False)
            # Plot 
            plotting_points = []
            # TODO This will break if geometry is not one of the properties. Fix this.
            flattened_projected_points = (projection_outputs['geometry'].value).reshape((num_points, -1)) # last axis usually has length 3 for x,y,z
            plotting_primitive_control_points = vedo.Points(flattened_projected_points, r=12, c='#00C6D7')  # TODO make this (1,3) instead of (3,)
            plotting_points.append(plotting_primitive_control_points)
            plotter.show(primitive_meshes, plotting_points, 'Projected Points', axes=1, viewup="z", interactive=True)

        return property_mapped_array
    
    def inverse_evaluate(self, points:csdl.Variable) -> csdl.Variable:
        '''
        Performs a fitting operation to get the coefficients to define the function that best fits the evaluated points.

        Parameters
        ----------
        points : csdl.Variable
            The evaluated points that the coefficients will be fit to.

        Returns
        -------
        coefficients : csdl.Variable
            The coefficients of the best-fit function.
        '''
        pass
        # TODO: For now, just use geometry subspace for states, so use the Mappedarray map directly for fitting  (fits to find all control points simultaneously)
        #   -- I think this approach is the same as fitting each individually since the a point (a row in the eval matrix) should only have values to map from a single B-spline.
        #   -- For better or for worse, this may actually be a better type of approach for meshes not on geometry like a camber surface. However, I think we want to avoid that
        #       anyway.

