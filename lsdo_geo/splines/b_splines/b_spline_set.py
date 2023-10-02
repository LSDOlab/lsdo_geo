import m3l
import csdl

import numpy as np
import scipy.sparse as sps
import array_mapper as am
import vedo

from lsdo_geo.cython.basis_matrix_surface_py import get_basis_surface_matrix
from lsdo_geo.cython.surface_projection_py import compute_surface_projection

from lsdo_geo.splines.b_splines.b_spline_set_space import BSplineSetSpace
from lsdo_geo.splines.b_splines.b_spline import BSpline

from dataclasses import dataclass

# TODO: I'm going to leave this class as surface for now, but I want to generalize to n-dimensional.

# TODO: NOTE: I want to include connectivity information (architecture/graph/network) in the B-spline set space.
# -- Can perform projections that span multiple B-splines.
# -- Can have a sliding connection point (important for something like a strut/jury)
# -- Can make projections more efficient by not performing full projection on every B-spline.
# -- Use a B-spline of the higher dimension of the pair because intersection also includes one b-spline embedded in another.

# Geometry-centric design is a well established method for multidisciplinary-multifidelity design optimization.
# Geoemtry-centric design works by including an information-dense, solver-independent geometric representation of the real system.
# By taking this approach, the solver-specific simplying assumptions are all made starting from a consistent representation.
# This ensures consistency between solvers of varying discipline and fidelity. This also helps simplify MDO problem setup.
# This idea has been expanded to include solver states to facilitate solver-independent field transfer, greatly decreasing the number
#  of solver interfaces that must be developed, and while maximizing the information retained in the transfer and ensuring conservation. [cite]
# However, missing from this solver-independent representation of the real system is the connectivity information.
# As of now, the geometry is simply a set of B-spline patches that individually exist in space.
# This work seeks to expand the ideas of solver-independent geometry-centric design to include connectivity information.
# Similar to geometry-centric design and SIFR, this will help facilitate the optimization problem setup by
# -- Speeding up projections
# -- Facilitate continuous sliding across B-splines
# -- Automate/supply graph/network/architecture creation for things like MBD
# -- Automate things like structural joints (e.g. the connection of beam nodes on the wing to the fuselage)
# ---- This idea will be combined with immersed methods to allow for differentability in the optimization.


@dataclass
class BSplineSet(m3l.Function):
    '''
    B-spline set class.

    Parameters
    ----------
    name : str
        The name of the B-spline.
    space : BSplineSetSpace
        The space that the B-spline set is a member of.
    coefficients : np.ndarray
        The coefficients of the B-spline set.
    num_physical_dimensions : dict[str, int]
        A dictionary of the number of physical dimensions for each B-spline.
    coefficient_indices : dict[str, np.ndarray] = None
        A dictionary of coefficient indices. The keys are the names of the B-spline.
    connections : list[tuple[str, str]] = None
        A list of connections between B-splines. The ordering of the strings does not matter.
    '''
    space : BSplineSetSpace
    # b_splines : dict[str, BSpline] = None # NOTE: DON'T DO THIS. A B-spline set is a function picked from the space of B-spline set functions.
    # This functionality can be achieved by creating a function that will create this BSplineSet from a dictionary of B-splines.
    # -- I think what I'm trying to say here is that it shouldn't explicitly store the b-spline objects, but rather aggregated vectors for cp, etc.

    num_physical_dimensions : dict[str, int]
    coefficient_indices : dict[str, np.ndarray] = None
    # Fow now, I'm going to put off this idea of connections and start simple to allow more time to think about how to best implement this.
    # If we store 2 connection B-splines between every B-spline, we'll have 2*n^2 connection B-splines which is probably intractable.
    connections : dict[str, list[str]] = None  # Outer dict has key of B-spline name, inner dict has key of connected space name
    # NOTE: These are connections in physical space. We can have a "disctontinuous" B-spline. The BSplineSetSpace  has the parametric connections.

    def __post_init__(self):
        self.control_points = self.coefficients

        self.num_coefficients = len(self.coefficients)
        self.num_control_points = self.num_coefficients

        self.coefficient_indices = {}
        coefficients_counter = 0
        for b_spline_name, space_name in self.space.b_spline_to_space_dict.items():
            b_spline_num_coefficients = self.space.spaces[space_name].num_coefficients*self.num_physical_dimensions[b_spline_name]
            self.coefficient_indices[b_spline_name] = np.arange(coefficients_counter, coefficients_counter + b_spline_num_coefficients)
            coefficients_counter += b_spline_num_coefficients

        # NOTE: For aggregation, see not on BSplineSetSpace (it should come later probably)
        # # Promote attributes to make this object a bit more intuitive?
        # self.knots = self.space.knots
        # self.num_knots = self.space.num_knots

        # Find connections!
        # self.find_connections()



    def evaluate(self, b_spline_name:str, parametric_coordinates:np.ndarray, parametric_derivative_order:tuple=None) -> am.MappedArray:
        b_spline_basis = self.space.compute_evaluation_map(
            b_spline_name=b_spline_name, parametric_coordinates=parametric_coordinates, parametric_derivative_order=parametric_derivative_order)
        
        num_control_points = self.space.spaces[self.space.b_spline_to_space_dict[b_spline_name]].num_control_points
        num_physical_dimensions = self.num_physical_dimensions[b_spline_name]
        
        # Do I merge b_spline name and parametric coordinates into one coordinates input to encourage the user to think of it that way?
        #   -- Seems a bit inconvenient to work with, but working with sets like this is not easy anyway.
        #   -- Doing this would allow to have one dictionary/list/? of coordinates instead of one for each B-spline.
        #   -- My reaction right now is this is fairly intuitive.
        #       -- Hesitancy is what object would this be? Each coordinate should have [str,float,float,float] for b_spline_name, u, v, w
        # Is this even worth the effort? When would someone use this?

        b_spline_control_points = self.control_points[self.coefficient_indices[b_spline_name]].reshape((num_control_points, num_physical_dimensions))
        # points = basis0.dot(b_spline_control_points.reshape((num_control_points, self.num_physical_dimensions)))
        points = b_spline_basis.dot(b_spline_control_points)

        # NOTE: This should probably return a MappedArray (Actually, M3L so this is a valid M3L operation)
        return points


    def project(self, points:np.ndarray, targets:list[str]=None, direction:np.ndarray=None,
                grid_search_density_parameter:int=10, max_iterations=100, plot:bool=False):
        '''
        Projects points onto the B-spline set.

        Parameters
        -----------
        points : {np.ndarray, am.MappedArray}
            The points to be projected onto the system.
        targets : list, optional
            The list of primitives to project onto.
        direction : {np.ndarray, am.MappedArray}, optional
            An axis for perfoming projection along an axis. The projection will return the closest point to the axis.
        grid_search_density : int, optional
            The resolution of the grid search prior to the Newton iteration for solving the optimization problem.
        max_iterations : int, optional
            The maximum number of iterations for the Newton iteration.
        plot : bool
            A boolean on whether or not to plot the projection result.
        '''
        # NOTE/TODO: Consider changing the algorithm to use the connection information to reduce the number of dense grid-searches and
        #   individual projection Newton optimizations.
        #  TODO Consider parallelizing using Numba, or using the FFD method or in Cython.

        # steps:
        # 1. Evaluates a coarse grid on every B-spline to find potentially close B-splines for each point
        # 2. Evaluate a dense grid over ALL the potentially close B-splines and use it to find the most likely closest B-spline for each point
        # 3. For each point, perform a projection on the closest B-spline with a very dense grid and a Newton iteration
        #   -- If the solution is on the edge of the B-spline, then the projections will be performed on the connected B-splines
        #   -- -- For each connected B-spline, if the objetive goes up, the branch is cut off.
        #   -- -- If the objective goes down, the branch is followed / logic is repeeated until there are no more branches.

        coarse_grid_search_density_parameter = 5
        fine_grid_search_density_parameter = 25*grid_search_density_parameter
        projection_grid_search_density_parameter = 10*grid_search_density_parameter

        points_shape = points.shape

        if type(points) is am.MappedArray:
            points = points.value
        if len(points.shape) == 1:
            points = points.reshape((1, -1))    # Last axis is reserved for dimensionality of physical space
        
        num_points = np.prod(points.shape[:-1])
        num_physical_dimensions = points.shape[-1]

        if targets is None:
            targets = list(self.coefficient_indices.keys())

        if type(direction) is am.MappedArray:
            direction = direction.value
        if direction is None:
            direction = np.zeros((num_points, num_physical_dimensions))
        if type(direction.shape[0]) is int: # checks if only one direction vector is given
            direction = np.tile(direction, (points.shape[0], 1))
        
        # Evaluate a coarse grid on every B-spline
        coarse_grid_points_indices = {}
        coarse_grid_points = np.zeros((0, num_physical_dimensions))
        i = 0
        for b_spline_name in targets:
            num_parametric_dimensions = self.space.spaces[self.space.b_spline_to_space_dict[b_spline_name]].num_parametric_dimensions
            dimension_linspace = np.linspace(0., 1., coarse_grid_search_density_parameter)

            mesh_grid_input = []
            for dimension_index in range(num_parametric_dimensions):
                mesh_grid_input.append(dimension_linspace)

            parametric_coordinates_tuple = np.meshgrid(*mesh_grid_input, indexing='ij')
            for dimensions_index in range(num_parametric_dimensions):
                parametric_coordinates_tuple[dimensions_index] = parametric_coordinates_tuple[dimensions_index].reshape((-1,1))

            parametric_coordinates = np.hstack(parametric_coordinates_tuple)
            b_spline_grid_points = self.evaluate(b_spline_name=b_spline_name, parametric_coordinates=parametric_coordinates)
            coarse_grid_points = np.vstack((coarse_grid_points, b_spline_grid_points))
            coarse_grid_points_indices[b_spline_name] = np.arange(coarse_grid_search_density_parameter**num_parametric_dimensions) + i
            i += coarse_grid_search_density_parameter**num_parametric_dimensions

        # Get a length scale parameter to determine a tolerance for each point
        points_expanded_coarse = np.repeat(points[:,np.newaxis,:], coarse_grid_points.shape[0], axis=1)
        coarse_distances = np.linalg.norm(points_expanded_coarse - coarse_grid_points, axis=2)
        length_scales = np.min(coarse_distances, axis=1)
        system_length_scales = np.zeros((3,))
        for i in range(3):
            system_length_scales[i] = np.max(coarse_grid_points[:,i]) - np.min(coarse_grid_points[:,i])
        system_length_scale = np.max(system_length_scales)
        projection_tolerances = length_scales + 0.01*length_scales*system_length_scale

        # Check which b_splines are close to which points
        close_targets = []    # total list of close targets
        close_targets_for_this_point = []   # list of close targets for this point. This is a list of lists
        for i in range(num_points):
            close_targets_for_this_point.append([])
        close_b_spline_indices = np.argwhere(coarse_distances < projection_tolerances[:,np.newaxis])

        for i in range(close_b_spline_indices.shape[0]):    # i corresponds to the index looping over the list of close target/point pairs
            point_index = close_b_spline_indices[i,0]
            close_grid_point_index = close_b_spline_indices[i,1]
            for target_name, point_indices in coarse_grid_points_indices.items():
                if close_grid_point_index in point_indices:
                    close_b_spline_name = target_name
                    break
            if close_b_spline_name not in close_targets:
                close_targets.append(close_b_spline_name)
            if close_b_spline_name not in close_targets_for_this_point[point_index]:
                close_targets_for_this_point[point_index].append(close_b_spline_name)

        # Evaluate a fine grid on every B-spline that is close to at least one point
        fine_grid_points_indices = {}
        fine_grid_points = np.zeros((0, num_physical_dimensions))
        b_spline_length_scales_dict = {}
        i = 0
        for b_spline_name in close_targets:
            num_parametric_dimensions = self.space.spaces[self.space.b_spline_to_space_dict[b_spline_name]].num_parametric_dimensions
            b_spline_length_scales = np.zeros((3,))
            for j in range(3):
                b_spline_length_scales[j] = np.max(coarse_grid_points[coarse_grid_points_indices[b_spline_name],j]) \
                    - np.min(coarse_grid_points[coarse_grid_points_indices[b_spline_name],j])
            b_spline_length_scale = np.linalg.norm(b_spline_length_scales)
            b_spline_length_scales_dict[b_spline_name] = b_spline_length_scale
            fine_grid_search_density = int(np.ceil(fine_grid_search_density_parameter*b_spline_length_scale/system_length_scale
                                               /num_parametric_dimensions))
            dimension_linspace = np.linspace(0., 1., fine_grid_search_density)

            mesh_grid_input = []
            for dimension_index in range(num_parametric_dimensions):
                mesh_grid_input.append(dimension_linspace)

            parametric_coordinates_tuple = np.meshgrid(*mesh_grid_input, indexing='ij')
            for dimensions_index in range(num_parametric_dimensions):
                parametric_coordinates_tuple[dimensions_index] = parametric_coordinates_tuple[dimensions_index].reshape((-1,1))

            parametric_coordinates = np.hstack(parametric_coordinates_tuple)
            b_spline_grid_points = self.evaluate(b_spline_name=b_spline_name, parametric_coordinates=parametric_coordinates)
            fine_grid_points = np.vstack((fine_grid_points, b_spline_grid_points))
            fine_grid_points_indices[b_spline_name] = np.arange(fine_grid_search_density**num_parametric_dimensions) + i
            i += fine_grid_search_density**num_parametric_dimensions

        # Find the closest fine grid point to each point
        points_expanded_fine = np.repeat(points[:,np.newaxis,:], fine_grid_points.shape[0], axis=1)
        fine_distances = np.linalg.norm(points_expanded_fine - fine_grid_points, axis=2)
        closest_fine_grid_point_indices = np.argmin(fine_distances, axis=1)
        closest_fine_grid_point_b_spline_names = []
        for i in range(num_points):
            close_grid_point_index = closest_fine_grid_point_indices[i]
            for target_name, point_indices in fine_grid_points_indices.items():
                if close_grid_point_index in point_indices:
                    close_b_spline_name = target_name
                    break
            closest_fine_grid_point_b_spline_names.append(target_name)

        # For each point, perform a projection on the closest B-spline with a very dense grid and a Newton iteration
        # -- If the solution is on the edge of the B-spline, then the projections will be performed on the connected B-splines
        # -- -- For each connected B-spline, if the objetive goes up, the branch is cut off.
        # -- -- If the objective goes down, the branch is followed / logic is repeeated until there are no more branches.
        closest_b_splines = {}
        for i in closest_fine_grid_point_b_spline_names:    # This is a bit tacky, but create B-spline objects to make projections easier
            b_spline = BSpline(name=i, space=self.space.spaces[self.space.b_spline_to_space_dict[i]], 
                               coefficients=self.coefficients[self.coefficient_indices[i]], num_physical_dimensions=self.num_physical_dimensions[i])
            closest_b_splines[i] = b_spline

        projected_points_list = []
        # TODO: Insert boundary checking and projection onto connected B-splines
        for i in range(num_points):
            point = points[i]
            direction_vector = direction[i].copy()
            closest_b_spline_name = closest_fine_grid_point_b_spline_names[i]
            
            b_spline = closest_b_splines[closest_b_spline_name]
            projection_grid_search_density = int(np.ceil(projection_grid_search_density_parameter
                                                         *b_spline_length_scales_dict[close_b_spline_name]/system_length_scale
                                                         /num_parametric_dimensions
                                                         *int(np.prod(np.array(b_spline.order)-1))))
            if projection_grid_search_density == 0:
                projection_grid_search_density = 1
            projected_point_on_b_spline = b_spline.project(points=point, direction=direction_vector, 
                                                           grid_search_density=projection_grid_search_density,max_iterations=max_iterations,
                                                           plot=True)
            
            # take map of projected point and add columns to fit whole cp vector
            point_map = sps.lil_matrix((self.num_physical_dimensions[close_b_spline_name], self.num_coefficients))
            # Same map for all physical dimensions (like x, y, and z in 3d case).
            num_physical_dimensions = self.num_physical_dimensions[closest_b_spline_name]
            for dimension_index in range(num_physical_dimensions):
                point_map[dimension_index, self.coefficient_indices[closest_b_spline_name][::num_physical_dimensions] + dimension_index] = \
                    projected_point_on_b_spline.linear_map
            point_map = point_map.tocsc()
            projected_point_on_b_spline_set = am.MappedArray(input=self.coefficients, 
                                                             linear_map=point_map, shape=(1, self.num_physical_dimensions[closest_b_spline_name]))
            
            projected_points_list.append(projected_point_on_b_spline_set)            

        projected_points = am.vstack(tuple(projected_points_list))
        projected_points.shape = points_shape


        if plot:
            # Plot the surfaces that are projected onto
            plotter = vedo.Plotter()
            b_spline_meshes = self.plot(b_splines=list(closest_b_splines.keys()), opacity=0.25, show=False)
            # Plot 
            plotting_points = []
            # TODO This will break if geometry is not one of the properties. Fix this.
            flattened_projected_points = (projected_points.value).reshape((num_points, -1)) # last axis usually has length 3 for x,y,z
            plotting_projected_points = vedo.Points(flattened_projected_points, r=12, c='#00C6D7')  # TODO make this (1,3) instead of (3,)
            plotting_points.append(plotting_projected_points)
            plotter.show(b_spline_meshes, plotting_points, 'Projected Points', axes=1, viewup="z", interactive=True)

        projected_points


    def plot(self, b_splines:list[str]=None, point_types:list=['evaluated_points', 'control_points'], plot_types:list=['mesh'],
              opacity:float=1., color:str='#00629B', surface_texture:str="", additional_plotting_elements:list=[], show:bool=True):
        '''
        Plots the B-spline Surface.

        Parameters
        -----------
        b_splines : list[str]
            The B-splines to be plotted. If None, all B-splines are plotted.
        points_type : list
            The type of points to be plotted. {evaluated_points, control_points}
        plot_types : list
            The type of plot {mesh, wireframe, point_cloud}
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

        if b_splines is None:
            b_splines = list(self.coefficient_indices.keys())


        for point_type in point_types:
            for b_spline_name in b_splines:
                b_spline_coefficient_indices = self.coefficient_indices[b_spline_name]
                if point_type == 'evaluated_points':
                        num_points_u = 25
                        num_points_v = 25
                        u_vec = np.einsum('i,j->ij', np.linspace(0., 1., num_points_u), np.ones(num_points_v)).reshape((-1,1))
                        v_vec = np.einsum('i,j->ij', np.ones(num_points_u), np.linspace(0., 1., num_points_v)).reshape((-1,1))
                        parametric_coordinates = np.hstack((u_vec, v_vec))
                        num_plotting_points = num_points_u * num_points_v

                        plotting_points = self.evaluate(b_spline_name=b_spline_name, parametric_coordinates=parametric_coordinates)
                        plotting_points_shape = (num_points_u, num_points_v, self.num_physical_dimensions[b_spline_name])
                elif point_type == 'control_points':
                    plotting_points_shape = self.space.spaces[self.space.b_spline_to_space_dict[b_spline_name]].control_points_shape + (self.num_physical_dimensions[b_spline_name],)
                    num_plotting_points = np.prod(plotting_points_shape[:-1])
                    plotting_points = self.control_points[b_spline_coefficient_indices].reshape((num_plotting_points,-1))
                else:
                    raise NotImplementedError(f'Point type {point_type} is not implemented.')

                if 'point_cloud' in plot_types:
                    for b_spline_name in self.coefficient_indices.keys():
                        plotting_elements.append(vedo.Points(plotting_points).opacity(opacity).color('darkred'))

                if 'mesh' in plot_types or 'wireframe' in plot_types:
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

                    mesh = vedo.Mesh([vertices, faces]).opacity(opacity).color(color).lighting(surface_texture)
                if 'mesh' in plot_types:
                    plotting_elements.append(mesh)
                if 'wireframe' in plot_types:
                    mesh = vedo.Mesh([vertices, faces]).opacity(opacity)
                    plotting_elements.append(mesh.wireframe())

        if show:
            plotter = vedo.Plotter()
            from vedo import Light
            # light = Light([-1,0,0], c='w', intensity=1)
            # plotter.show(plotting_elements, light, f'B-spline Surface: {self.name}', axes=1, viewup="z", interactive=True)
            plotter.show(plotting_elements, f'Geometry: {self.name}', axes=1, viewup="z", interactive=True)
            return plotting_elements
        else:
            return plotting_elements
                
    
    def find_connections(self, b_spline_names:list=None) -> dict[str, list[str]]:
        '''
        Finds the connections between B-splines.

        Parameters
        ----------
        b_spline_names : list[str] = None
            A list of B-spline names to find connections for. If None, all B-splines are checked.

        Returns
        -------
        connections : dict[str, list[str]]
            A dictionary of connections between B-splines.
        '''
        if self.connections is None:
            self.connections = {}
            for b_spline_name in self.space.b_spline_to_space_dict.keys():
                self.connections[b_spline_name] = []

        if b_spline_names is None:
            b_spline_names = list(self.coefficient_indices.keys())
        
        # Run this for each number of dimensions
        min_num_physical_dimensions = np.min(list(self.num_physical_dimensions.values()))
        max_num_physical_dimensions = np.max(list(self.num_physical_dimensions.values()))
        for num_physical_dimensions in range(min_num_physical_dimensions, max_num_physical_dimensions+1):
            # Get a list of B-splines with the current number of dimensions
            b_spline_names_with_same_dimension = []
            for b_spline_name, space_name in self.space.b_spline_to_space_dict.items():
                if self.num_physical_dimensions[b_spline_name] == num_physical_dimensions:
                    b_spline_names_with_same_dimension.append(b_spline_name)

            # If there are no B-splines with the current number of dimensions, continue
            if len(b_spline_names_with_same_dimension) == 0:
                continue

            # If there is only one B-spline with the current number of dimensions, there are no connections to be made
            if len(b_spline_names_with_same_dimension) == 1:
                continue

            # If there are multiple B-splines with the current number of dimensions, find connections
            self.find_connections_with_same_num_physical_dimensions(b_spline_names=b_spline_names_with_same_dimension)

        return self.connections

    def find_connections_with_same_num_physical_dimensions(self, b_spline_names:list) -> None:
        '''
        Finds the connections between B-splines.

        Parameters
        ----------
        b_spline_names : list[str]
            A list of B-spline names to find connections for. If None, all B-splines are checked.

        Returns
        -------
        connections : dict[str, list[str]]
            A dictionary of connections between B-splines.
        '''
        num_physical_dimensions = self.num_physical_dimensions[b_spline_names[0]]

        # Evaluate a grid on every B-spline
        coarse_grid_points_indices = {}
        coarse_grid_points = np.zeros((0, num_physical_dimensions))
        i = 0
        for b_spline_name in b_spline_names:
            num_points_u = 5
            num_points_v = 5
            u_vec = np.einsum('i,j->ij', np.linspace(0., 1., num_points_u), np.ones(num_points_v)).reshape((-1,1))
            v_vec = np.einsum('i,j->ij', np.ones(num_points_u), np.linspace(0., 1., num_points_v)).reshape((-1,1))
            parametric_coordinates = np.hstack((u_vec, v_vec))
            b_spline_grid_points = self.evaluate(b_spline_name=b_spline_name, parametric_coordinates=parametric_coordinates)
            coarse_grid_points = np.vstack((coarse_grid_points, b_spline_grid_points))
            coarse_grid_points_indices[b_spline_name] = np.arange(num_points_u*num_points_v) + i
            i += num_points_u*num_points_v

        # Get a length scale parameter to determine a large tolerance and small tolerance
        length_scales = np.zeros((3,))
        for i in range(3):
            length_scales[i] = np.max(coarse_grid_points[:,i]) - np.min(coarse_grid_points[:,i])
        length_scale = np.linalg.norm(length_scales)
        large_tolerance = 0.1*length_scale
        small_tolerance = 1e-3*length_scale

        # If distance between points on grid is below large tolerance, store pairs of possible connections
        close_b_splines = []
        i = 0
        for b_spline_name in b_spline_names:
            b_spline_grid_points = coarse_grid_points[coarse_grid_points_indices[b_spline_name],:]
            j = 0
            for other_b_spline_name in b_spline_names:
                if j <= i:   # Don't need to check for connections with B-splines that have already been checked
                    j += 1
                    continue
                # if other_b_spline_name != b_spline_name:    # The > i check accomplishes this
                other_b_spline_grid_points = coarse_grid_points[coarse_grid_points_indices[other_b_spline_name],:]
                distances = np.linalg.norm(b_spline_grid_points - other_b_spline_grid_points, axis=-1)
                if np.any(distances < large_tolerance):
                    # print('Performing projection on tighter grid')
                    close_b_splines.append((b_spline_name, other_b_spline_name))
            i += 1

        # Perform simultaneous projection on tighter grid on those B-splines, and below small tolerance, add connection
        connections = {}    # This is created and returned purely to help with debugging
        for b_spline_name in b_spline_names:
            connections[b_spline_name] = []
        for connection in close_b_splines:
            distance = self.find_distance(b_spline_1=connection[0], b_spline_2=connection[1], tolerance=small_tolerance)
            if distance < small_tolerance:
                # print('Adding connection between B-splines')
                self.connections[connection[0]].append(connection[1])
                self.connections[connection[1]].append(connection[0])

                connections[connection[0]].append(connection[1])
                connections[connection[1]].append(connection[0])

        return connections


    def find_distance(self, b_spline_1:BSpline, b_spline_2:BSpline, tolerance:float=1.e-6) -> float:
        '''
        Finds the distance between two B-splines.

        Parameters
        ----------
        b_spline_1 : {BSpline, str}
            The first B-spline or the name of the first BSpline.
        b_spline_2 : {BSpline, str}
            The second B-spline or the name of the second BSpline.

        Returns
        -------
        distance : float
            The distance between the two B-splines.
        '''
        # This is currently implemented as a Co-op game with two separate projection optimizations going back and forth.
        # - The goal of the game is to get their points as close as possible.
        # -- In the future, this should be implemented as a single optimization problem manipulating the parametric coordinates of both B-splines.
        player_1 = b_spline_1
        player_2 = b_spline_2
        num_physical_dimensions = self.num_physical_dimensions[player_1]
        if type(player_1) is str:
            # player_1 = self.b_splines[player_1]   # Not storing B-spline objects in this set.
            player_1_space = self.space.spaces[self.space.b_spline_to_space_dict[player_1]]
            player_1 = BSpline(name=player_1, space=player_1_space, coefficients=self.coefficients[self.coefficient_indices[player_1]],
                               num_physical_dimensions=num_physical_dimensions)
        if type(player_2) is str:
            player_2_space = self.space.spaces[self.space.b_spline_to_space_dict[player_2]]
            player_2 = BSpline(name=player_2, space=player_2_space, coefficients=self.coefficients[self.coefficient_indices[player_2]],
                               num_physical_dimensions=num_physical_dimensions)

        max_turns = 3
        player_2_point = player_2.evaluate(parametric_coordinates=np.array([[0.5, 0.5]]))
        for i in range(max_turns):
            # Player 1's turn
            player_1_point = player_1.project(points=player_2_point)
            # Player 2's turn
            player_2_point = player_2.project(points=player_1_point)

            distance = np.linalg.norm(player_1_point.value - player_2_point.value)
            if distance < tolerance:
                break
        
        return distance
        
        
        



if __name__ == "__main__":
    from lsdo_geo.splines.b_splines.b_spline_space import BSplineSpace
    from lsdo_geo.cython.get_open_uniform_py import get_open_uniform

    # ''' Creating B-spline set manually '''

    # num_control_points1 = 10
    # order1 = 4
    # num_control_points2 = 5
    # order2 = 3
    
    # space_of_cubic_b_spline_surfaces_with_10_cp = BSplineSpace(name='cubic_b_spline_surfaces_10_cp', order=(order1,order1),
    #                                                           control_points_shape=(num_control_points1,num_control_points1))
    # space_of_quadratic_b_spline_surfaces_with_5_cp = BSplineSpace(name='quadratic_b_spline_surfaces_5_cp', order=(order2,order2),
    #                                                           control_points_shape=(num_control_points2,num_control_points2))
    # b_spline_spaces = {space_of_cubic_b_spline_surfaces_with_10_cp.name : space_of_cubic_b_spline_surfaces_with_10_cp,
    #                    space_of_quadratic_b_spline_surfaces_with_5_cp.name : space_of_quadratic_b_spline_surfaces_with_5_cp}
    # b_spline_set_space = BSplineSetSpace(name='my_b_spline_set', spaces=b_spline_spaces, 
    #                                      b_spline_to_space_dict={'my_b_spline_1':space_of_cubic_b_spline_surfaces_with_10_cp.name,
    #                                                              'my_b_spline_2':space_of_quadratic_b_spline_surfaces_with_5_cp.name})


    # coefficients = np.zeros(((num_control_points1*num_control_points1 + num_control_points2*num_control_points2)*3))
    # coefficients[:num_control_points1*num_control_points1*3] = 0.
    # coefficients[num_control_points1*num_control_points1*3:] = 1.

    # # Connection
    # # coefficients[num_control_points1*num_control_points1*3 - 1] = 1.
    # b_spline_set = BSplineSet(name='my_b_spline_set', space=b_spline_set_space, coefficients=coefficients,
    #                           num_physical_dimensions={'my_b_spline_1':3, 'my_b_spline_2':3})

    ''' Importing B-spline set '''
    from lsdo_geo.splines.b_splines.b_spline_functions import import_file
    from lsdo_geo.splines.b_splines.b_spline_functions import create_b_spline_set
    from lsdo_geo.splines.b_splines.b_spline_functions import refit_b_spline_set

    b_splines = import_file('lsdo_geo/splines/b_splines/sample_geometries/rectangular_wing.stp')
    for b_spline in list(b_splines.values()):
        print(b_spline.name)
        # b_spline.plot(plot_types=['mesh'], show=True)
    
    b_spline_set = create_b_spline_set(name='sample_wing', b_splines=b_splines)
    b_spline_set = refit_b_spline_set(b_spline_set=b_spline_set, num_control_points=(25,10), order=(4,3))
    b_spline_set.find_connections()
    # b_spline_set.plot()

    # projected_points1 = b_spline_set.project(np.array([[0.2, 1., 10.], [0.5, 1., 1.]]), plot=True, direction=np.array([0., 0., -1.]))
    projected_points2 = b_spline_set.project(np.array([[0.2, 0., 1.], [0.5, 1., 1.]]), plot=True, max_iterations=100)

    print('hi')