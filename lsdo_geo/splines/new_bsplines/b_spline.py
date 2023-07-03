import m3l
import csdl

import numpy as np
import scipy.sparse as sps
import array_mapper as am
import vedo

from lsdo_geo.cython.basis_matrix_surface_py import get_basis_surface_matrix
from lsdo_geo.cython.surface_projection_py import compute_surface_projection

from lsdo_geo.splines.new_bsplines.b_spline_space import BSplineSpace

from dataclasses import dataclass

# TODO: I'm going to leave this class as surface for now, but I want to generalize to n-dimensional.

@dataclass
class BSpline(m3l.Function):
    '''
    B-spline class
    '''

    def __post_init__(self):
        # self.coefficients = self.control_points
        self.control_points = self.coefficients
        self.num_physical_dimensions = self.control_points.shape[-1]

        self.control_points_shape = self.space.control_points_shape

        # Promote attributes to make this object a bit more intuitive
        self.order = self.space.order
        self.knots = self.space.knots
        self.num_control_points = self.space.num_control_points
        self.num_parametric_dimensions = self.space.num_parametric_dimensions


    def evaluate_points(self, parametric_coordinates:np.ndarray) -> sps.csc_matrix:       
        num_control_points = self.num_control_points
        
        basis0 = self.space.compute_evaluation_map(parametric_coordinates)
        points = basis0.dot(self.control_points.reshape((num_control_points, self.num_physical_dimensions)))

        return points

    def evaluate_derivative(self, parametric_coordinates:np.ndarray) -> sps.csc_matrix:
        num_control_points = self.num_control_points
        
        basis1 = self.space.compute_derivative_evaluation_map(parametric_coordinates)
        derivs1 = basis1.dot(self.control_points.reshape((num_control_points, self.num_physical_dimensions)))

        return derivs1 

    def evaluate_second_derivative(self, parametric_coordinates:np.ndarray) -> sps.csc_matrix:
        num_control_points = self.num_control_points
        
        basis2 = self.space.compute_second_derivative_evaluation_map(parametric_coordinates)
        derivs2 = basis2.dot(self.control_points.reshape((num_control_points, self.num_physical_dimensions)))

        return derivs2


    def project(self, points:np.ndarray, direction:np.ndarray=None, grid_search_n:int=50,
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
        num_control_points = self.num_control_points

        compute_surface_projection(
            np.array([self.order[0]]), np.array([self.control_points_shape[0]]),
            np.array([self.order[1]]), np.array([self.control_points_shape[1]]),
            num_points, max_iter,
            flattened_points, 
            self.control_points.reshape((-1,)),
            self.knots[0], self.knots[1],
            u_vec_flattened, v_vec_flattened, grid_search_n,
            direction.reshape((-1,)), np.zeros((num_points,), dtype=int), self.control_points.reshape((num_control_points, -1))
        )

        parametric_coordinates = np.stack((u_vec_flattened, v_vec_flattened), axis=-1)
        map = self.space.compute_evaluation_map(parametric_coordinates)
        projected_points = am.array(input=self.control_points.reshape((num_control_points,-1)), linear_map=map, shape=input_shape)

        if plot:
            # Plot the surfaces that are projected onto
            plotter = vedo.Plotter()
            primitive_meshes = self.plot(plot_types=['mesh'], opacity=0.25, show=False)
            # Plot 
            plotting_points = []
            flattened_projected_points = (projected_points.value).reshape((num_points, -1)) # last axis usually has length 3 for x,y,z
            plotting_primitive_control_points = vedo.Points(flattened_projected_points, r=12, c='blue')  # TODO make this (1,3) instead of (3,)
            plotting_points.append(plotting_primitive_control_points)
            plotter.show(primitive_meshes, plotting_points, 'Projected Points', axes=1, viewup="z", interactive=True)

        # u_vec = u_vec_flattened.reshape(tuple(input_shape[:-1],)+(1,))
        # v_vec = v_vec_flattened.reshape(tuple(input_shape[:-1],)+(1,))
        # parametric_coordinates = np.concatenate((u_vec, v_vec), axis=-1)

        if return_parametric_coordinates:
            # return parametric_coordinates
            return (u_vec_flattened, v_vec_flattened)
        else:
            return projected_points


    def plot(self, point_types:list=['evaluated_points', 'control_points'], plot_types:list=['mesh'],
              opacity:float=1., color:str='#00629B', surface_texture:str="", additional_plotting_elements:list=[], show:bool=True):
        '''
        Plots the B-spline Surface.

        Parameters
        -----------
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

        for point_type in point_types:
            if point_type == 'evaluated_points':
                num_points_u = 25
                num_points_v = 25
                u_vec = np.einsum('i,j->ij', np.linspace(0., 1., num_points_u), np.ones(num_points_v)).reshape((-1,1))
                v_vec = np.einsum('i,j->ij', np.ones(num_points_u), np.linspace(0., 1., num_points_v)).reshape((-1,1))
                parametric_coordinates = np.hstack((u_vec, v_vec))
                num_plotting_points = num_points_u * num_points_v
                plotting_points = self.evaluate_points(parametric_coordinates=parametric_coordinates)
                plotting_points_shape = (num_points_u, num_points_v, plotting_points.shape[-1])
            elif point_type == 'control_points':
                plotting_points_shape = self.control_points.shape
                num_plotting_points = np.cumprod(plotting_points_shape[:-1])[-1]
                plotting_points = self.control_points.reshape((num_plotting_points,-1))

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


if __name__ == "__main__":
    from lsdo_geo.splines.new_bsplines.b_spline_space import BSplineSpace

    num_control_points = 10
    order = 4
    space_of_cubic_bspline_surfaces_with_10_cp = BSplineSpace(name='cubic_bspline_surfaces_10_cp', order=(order,order),
                                                              control_points_shape=(num_control_points,num_control_points))

    control_points_line = np.linspace(0., 1., num_control_points)
    control_points_x, control_points_y = np.meshgrid(control_points_line,control_points_line)
    control_points = np.stack((control_points_x, control_points_y, 0.1*np.random.rand(10,10)), axis=-1)

    b_spline = BSpline(name='test_b_spline', space=space_of_cubic_bspline_surfaces_with_10_cp, coefficients=control_points)

    plotting_elements = b_spline.plot(point_types=['evaluated_points'], plot_types=['mesh'])

    parametric_coordinates = np.array([
        [0., 0.],
        [0., 1.],
        [1., 0.],
        [1., 1.],
        [0.5, 0.5],
        [0.25, 0.75]
    ])

    print('points: ', b_spline.evaluate_points(parametric_coordinates=parametric_coordinates))
    print('derivative:', b_spline.evaluate_derivative(parametric_coordinates=parametric_coordinates))
    print('second derivative: ', b_spline.evaluate_second_derivative(parametric_coordinates=parametric_coordinates))

    projecting_points_z = np.zeros((6,))
    projecting_points = np.stack((parametric_coordinates[:,0], parametric_coordinates[:,1], projecting_points_z), axis=-1)

    b_spline.project(points=projecting_points, plot=True)
