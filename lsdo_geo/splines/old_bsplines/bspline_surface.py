import numpy as np
import scipy.sparse as sps
import array_mapper as am
import vedo

from lsdo_b_splines_cython.cython.basis_matrix_surface_py import get_basis_surface_matrix
from lsdo_b_splines_cython.cython.surface_projection_py import compute_surface_projection

from lsdo_geo.primitives.b_splines.b_spline import BSpline

class BSplineSurface(BSpline):
    def __init__(self, name:str, order_u:int, order_v:int, knots_u:np.ndarray, knots_v:np.ndarray, shape:tuple, coefficients:np.ndarray):
        self.name = name
        self.order_u = order_u
        self.order_v = order_v
        self.knots_u = knots_u
        self.knots_v = knots_v
        self.shape = shape
        self.coefficients = coefficients

    def compute_evaluation_map(self, u_vec, v_vec):
        data = np.zeros(len(u_vec) * self.order_u * self.order_v)
        row_indices = np.zeros(len(data), np.int32)
        col_indices = np.zeros(len(data), np.int32)

        num_coefficients = self.shape[0] * self.shape[1]

        get_basis_surface_matrix(self.order_u, self.shape[0], 0, u_vec, self.knots_u, 
            self.order_v, self.shape[1], 0, v_vec, self.knots_v, 
            len(u_vec), data, row_indices, col_indices)

        basis0 = sps.csc_matrix((data, (row_indices, col_indices)), shape=(len(u_vec), num_coefficients) )
        
        return basis0

    def compute_derivative_evaluation_map(self, u_vec, v_vec):
        data = np.zeros(len(u_vec) * self.order_u * self.order_v)
        row_indices = np.zeros(len(data), np.int32)
        col_indices = np.zeros(len(data), np.int32)        

        num_coefficients = self.shape[0] * self.shape[1]

        get_basis_surface_matrix(self.order_u, self.coefficients.shape[0], 1, u_vec, self.knots_u, 
            self.order_v, self.coefficients.shape[1], 1, v_vec, self.knots_v, 
            len(u_vec), data, row_indices, col_indices)

        basis1 = sps.csc_matrix((data, (row_indices, col_indices)), shape=(len(u_vec), num_coefficients) )
        
        return basis1

    def compute_second_derivative_evaluation_map(self, u_vec, v_vec):
        data = np.zeros(len(u_vec) * self.order_u * self.order_v)
        row_indices = np.zeros(len(data), np.int32)
        col_indices = np.zeros(len(data), np.int32)      

        num_coefficients = self.shape[0] * self.shape[1]
        
        get_basis_surface_matrix(self.order_u, self.coefficients.shape[0], 2, u_vec, self.knots_u, 
            self.order_v, self.coefficients.shape[1], 2, v_vec, self.knots_v, 
            len(u_vec), data, row_indices, col_indices)

        basis2 = sps.csc_matrix((data, (row_indices, col_indices)), shape=(len(u_vec), num_coefficients) )
        
        return basis2

    def evaluate_points(self, u_vec, v_vec):       
        num_coefficients = self.shape[0] * self.shape[1]
        
        basis0 = self.compute_evaluation_map(u_vec, v_vec)
        points = basis0.dot(self.coefficients.reshape((num_coefficients, self.shape[2])))

        return points

    def evaluate_derivative(self, u_vec, v_vec):
        num_coefficients = self.shape[0] * self.shape[1]
        
        basis1 = self.compute_derivative_evaluation_map(u_vec, v_vec)
        derivs1 = basis1.dot(self.coefficients.reshape((num_coefficients, self.shape[2])))

        return derivs1 

    def evaluate_second_derivative(self, u_vec, v_vec):
        num_coefficients = self.shape[0] * self.shape[1]
        
        basis2 = self.compute_second_derivative_evaluation_map(u_vec, v_vec)
        derivs2 = basis2.dot(self.coefficients.reshape((num_coefficients, self.shape[2])))

        return derivs2


    def project(self, points:np.ndarray, direction:np.ndarray=None, grid_search_density:int=50,
                    max_iter:int=100, return_parametric_coordinates:bool=False, plot:bool=False):
        
        if type(points) is am.MappedArray:
            points = points.value
        
        input_shape = points.shape
        flattened_points = points.flatten()
        if len(points.shape) > 1:
            num_points = np.cumprod(points.shape[:-1])[-1]
        else:
            num_points = 1

        if direction is None:
            direction = np.zeros((num_points*np.cumprod(points.shape)[-1],))
        else:
            direction = np.tile(direction, num_points)

        
        u_vec_flattened = np.zeros(num_points)
        v_vec_flattened = np.zeros(num_points)
        num_coefficients = np.cumprod(self.shape[:-1])[-1]

        compute_surface_projection(
            np.array([self.order_u]), np.array([self.shape[0]]),
            np.array([self.order_v]), np.array([self.shape[1]]),
            num_points, max_iter,
            flattened_points, 
            self.coefficients.reshape((-1,)),
            self.knots_u, self.knots_v,
            u_vec_flattened, v_vec_flattened, grid_search_density,
            direction.reshape((-1,)), np.zeros((num_points,), dtype=int), self.coefficients.reshape((num_coefficients, -1))
        )

        map = self.compute_evaluation_map(u_vec_flattened, v_vec_flattened)
        projected_points = am.array(input=self.coefficients.reshape((num_coefficients,-1)), linear_map=map, shape=input_shape)

        if plot:
            # Plot the surfaces that are projected onto
            plotter = vedo.Plotter()
            primitive_meshes = self.plot(plot_types=['mesh'], opacity=0.25, show=False)
            # Plot 
            plotting_points = []
            flattened_projected_points = (projected_points.value).reshape((num_points, -1)) # last axis usually has length 3 for x,y,z
            plotting_primitive_coefficients = vedo.Points(flattened_projected_points, r=12, c='blue')  # TODO make this (1,3) instead of (3,)
            plotting_points.append(plotting_primitive_coefficients)
            plotter.show(primitive_meshes, plotting_points, 'Projected Points', axes=1, viewup="z", interactive=True)

        # u_vec = u_vec_flattened.reshape(tuple(input_shape[:-1],)+(1,))
        # v_vec = v_vec_flattened.reshape(tuple(input_shape[:-1],)+(1,))
        # parametric_coordinates = np.concatenate((u_vec, v_vec), axis=-1)

        if return_parametric_coordinates:
            # return parametric_coordinates
            return (u_vec_flattened, v_vec_flattened)
        else:
            return projected_points


    def plot(self, point_types:list=['evaluated_points', 'coefficients'], plot_types:list=['mesh'],
              opacity:float=1., color:str='#00629B', surface_texture:str="", additional_plotting_elements:list=[], show:bool=True):
        '''
        Plots the B-spline Surface.

        Parameters
        -----------
        points_type : list
            The type of points to be plotted. {evaluated_points, coefficients}
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

        for point_type in point_types:
            if point_type == 'evaluated_points':
                num_points_u = 25
                num_points_v = 25
                u_vec = np.einsum('i,j->ij', np.linspace(0., 1., num_points_u), np.ones(num_points_v)).flatten()
                v_vec = np.einsum('i,j->ij', np.ones(num_points_u), np.linspace(0., 1., num_points_v)).flatten()
                num_plotting_points = num_points_u * num_points_v
                plotting_points = self.evaluate_points(u_vec=u_vec, v_vec=v_vec)
                plotting_points_shape = (num_points_u, num_points_v, plotting_points.shape[-1])
            elif point_type == 'coefficients':
                plotting_points_shape = self.shape
                num_plotting_points = np.cumprod(plotting_points_shape[:-1])[-1]
                plotting_points = self.coefficients.reshape((num_plotting_points,-1))

            if 'point_cloud' in plot_types:
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
            light = Light([-1,0,0], c='w', intensity=1)
            plotter = vedo.Plotter(size=(3200,1000))
            plotter.show(plotting_elements, light, f'B-spline Surface: {self.name}', axes=1, viewup="z", interactive=True)
            return plotting_elements
        else:
            return plotting_elements
