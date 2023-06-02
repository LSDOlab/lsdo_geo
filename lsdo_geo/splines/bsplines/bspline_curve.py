import numpy as np
import array_mapper as am
import scipy.sparse as sps
import vedo
import vedo.pyplot as plt

from caddee.cython.basis_matrix_curve_py import get_basis_curve_matrix
from caddee.cython.curve_projection_py import compute_curve_projection
from caddee.cython.get_open_uniform_py import get_open_uniform

from caddee.primitives.bsplines.bspline import BSpline

class BSplineCurve(BSpline):
    def __init__(self, name, control_points, order_u=4, knots_u=None):
        
        self.name = name
        self.order_u = order_u
        self.shape = control_points.shape
        self.control_points = control_points
        self.knots_u = knots_u
        if np.isscalar(self.shape):
            self.num_control_points = self.shape
        elif (len(self.shape) == 1):
            self.num_control_points = self.shape[0]
            self.num_physical_dimenensions = 1
        elif len(self.shape == 2):
            self.num_control_points = self.shape[0]
            self.num_physical_dimenensions = self.shape[1]
        else:
            raise Exception("Control points should have a proper shape for a curve.")

        if self.knots_u is None:
            self.knots_u = np.zeros(self.shape[0] + self.order_u)
            get_open_uniform(self.order_u, self.shape[0], self.knots_u)

    def compute_evaluation_map(self, u_vec):
        data = np.zeros(len(u_vec) * self.order_u)
        row_indices = np.zeros(len(data), np.int32)
        col_indices = np.zeros(len(data), np.int32)

        get_basis_curve_matrix(self.order_u, self.shape[0], 0, u_vec, self.knots_u, 
            len(u_vec), data, row_indices, col_indices)

        basis0 = sps.csc_matrix((data, (row_indices, col_indices)), shape=(len(u_vec), self.num_control_points))
        
        return basis0

    def compute_derivative_evaluation_map(self, u_vec):
        data = np.zeros(len(u_vec) * self.order_u)
        row_indices = np.zeros(len(data), np.int32)
        col_indices = np.zeros(len(data), np.int32)

        get_basis_curve_matrix(self.order_u, self.shape[0], 1, u_vec, self.knots_u, 
            len(u_vec), data, row_indices, col_indices)

        basis1 = sps.csc_matrix((data, (row_indices, col_indices)), shape=(len(u_vec), self.num_control_points))
        
        return basis1

    def compute_second_derivative_evaluation_map(self, u_vec):
        data = np.zeros(len(u_vec) * self.order_u)
        row_indices = np.zeros(len(data), np.int32)
        col_indices = np.zeros(len(data), np.int32)

        get_basis_curve_matrix(self.order_u, self.shape[0], 2, u_vec, self.knots_u, 
            len(u_vec), data, row_indices, col_indices)

        basis2 = sps.csc_matrix((data, (row_indices, col_indices)), shape=(len(u_vec), self.num_control_points))
        
        return basis2

    def evaluate_points(self, u_vec):
        basis0 = self.compute_evaluation_map(u_vec)
        points = basis0.dot(self.control_points.reshape((self.num_control_points, -1)))

        return points

    def evaluate_derivative(self, u_vec):
        basis1 = self.compute_derivative_evaluation_map(u_vec)
        derivs1 = basis1.dot(self.control_points.reshape((self.num_control_points, -1)))

        return derivs1 

    def evaluate_second_derivative(self, u_vec):
        basis2 = self.compute_second_derivative_evaluation_map(u_vec)
        derivs2 = basis2.dot(self.control_points.reshape((self.num_control_points, -1)))

        return derivs2


    def project(self, points:np.ndarray, grid_search_n:int=50,
                    max_iter:int=100, return_parametric_coordinates:bool=False, plot:bool=False):
    

        num_points = len(points)
        num_control_points = self.shape[0]

        u_vec = np.zeros((num_points,))

        compute_curve_projection(
            self.order_u, num_control_points,
            num_points, max_iter,
            points, 
            self.control_points,
            u_vec, grid_search_n
        )

        map = self.compute_evaluation_map(u_vec)
        projected_points = am.array(input=self.control_points.reshape((num_control_points,-1)), linear_map=map, shape=points.shape)

        if plot:
            # Plot the surfaces that are projected onto
            plotter = vedo.Plotter()
            primitive_meshes = self.plot(plot_types=['mesh'], opacity=0.25, show=False)
            # Plot 
            plotting_points = []
            plotting_primitive_control_points = vedo.Points(projected_points.value, r=12, c='#C69214')  # TODO make this (1,3) instead of (3,)
            plotting_points.append(plotting_primitive_control_points)
            plotter.show(primitive_meshes, plotting_points, 'Projected Points', axes=1, viewup="z", interactive=True)

        if return_parametric_coordinates:
            # return parametric_coordinates
            return (u_vec,)
        else:
            return projected_points


    def plot(self, plot_types:list=['mesh'], point_types:list=['evaluated_points', 'control_points'],
             opacity:float=1., color:str='#00629B',
             additional_plotting_elements:list=[], show:bool=True):
        '''
        Plots the B-spline Curve.
        
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
        additional_plotting_elemets : list
            Vedo plotting elements that may have been returned from previous plotting functions that should be plotted with this plot.
        show : bool
            A boolean on whether to show the plot or not. If the plot is not shown, the Vedo plotting element is returned.
        '''
        plotting_elements = additional_plotting_elements.copy()

        for point_type in point_types:
            if point_type == 'evaluated_points':
                u_vec = np.linspace(0,1,100)
                points = self.evaluate_points(u_vec).reshape((-1, self.num_physical_dimenensions))
                plotting_points = np.zeros((points.shape[0], 3))  # Vedo does 3D plotting
                plotting_points[:,0:points.shape[1]] = points
            elif point_type == 'control_points':
                points = self.control_points.reshape((self.num_control_points, self.num_physical_dimenensions))
                plotting_points = np.zeros((points.shape[0], 3))  # Vedo does 3D plotting
                plotting_points[:,0:points.shape[1]] = points

            if 'point_cloud' in plot_types:
                plotting_elements.append(vedo.Points(plotting_points).opacity(opacity).color(color))

            if 'mesh' in plot_types or 'wireframe' in plot_types:
                plotting_elements.append(vedo.Line(plotting_points).opacity(opacity).color(color).linewidth(3))
                if 'wireframe' in plot_types:   # If it is specifically wireframe, then add point markers
                    plotting_elements.append(vedo.Points(plotting_points).opacity(opacity).color(color))

            # evaluated_points = None
            # fig = None
            # if 'evaluated_points' in point_types:
            #     u_vec = np.linspace(0,1,100)
            #     evaluated_points = self.evaluate_points(u_vec)
            #     # fig = plt.plot(u_vec, evaluated_points, '-r', label='Evaluated Points', title=f'{self.name}', xtitle='u', ytitle="f(u)")
            #     # fig += plt.plot(kh, forward_kh, '-b', like=fig, label='foward method')
            #     # fig += plt.plot(kh, centered_kh, '-g', like=fig, label='centered method')
            #     # fig += plt.plot(kh, second_derivative_centered_kh, '-p', like=fig, label='centered second derivative')
            #     # fig.add_legend()
            #     # fig.show()
            
            # if 'point_cloud' in plot_types:
            #     if 'evaluated_points' in point_types:
            #         u_vec = np.linspace(0,1,100)
            #         evaluated_points = self.evaluate_points(u_vec)
            #         if fig is None:
            #             fig = plt.plot(u_vec, evaluated_points, 'bo', label='Evaluated Points', title=f'{self.name}', xtitle='u', ytitle="f(u)")
            #         else:
            #             fig += plt.plot(u_vec, evaluated_points, 'bo', like=fig, label='Evaluated Points')
            #     if 'control_points' in point_types:
            #         u_vec = np.linspace(0, 1, self.control_points.shape[0])
            #         if fig is None:
            #             fig = plt.plot(u_vec, self.control_points, 'ro', label='Control Points', title=f'{self.name}', xtitle='u', ytitle="f(u)")
            #         else:
            #             fig += plt.plot(u_vec, self.control_points, 'bo', like=fig, label='Evaluated Points')

            #     # num_points = self.shape[0]*self.shape[1]
            #     # plotting_elements.append(vedo.Points(self.control_points.reshape((num_points,-1))).opacity(opacity).color('green'))

            # if 'mesh' in plot_types or 'wireframe' in plot_types:
            #     if 'evaluated_points' in point_types:
            #         u_vec = np.linspace(0,1,100)
            #         evaluated_points = self.evaluate_points(u_vec)
            #         if fig is None:
            #             fig = plt.plot(u_vec, evaluated_points, '-b', label='Evaluated Points', title=f'{self.name}', xtitle='u', ytitle="f(u)")
            #         else:
            #             fig += plt.plot(u_vec, evaluated_points, '-b', like=fig, label='Evaluated Points')
            #     if 'control_points' in point_types:
            #         u_vec = np.linspace(0, 1, self.control_points.shape[0])
            #         if fig is None:
            #             fig = plt.plot(u_vec, self.control_points, '-r', label='Control Points', title=f'{self.name}', xtitle='u', ytitle="f(u)")
            #         else:
            #             fig += plt.plot(u_vec, self.control_points, '-b', like=fig, label='Evaluated Points')

            # plotting_elements.append(fig)

        if show:
            plotter = vedo.Plotter()
            plotter.show(plotting_elements, f'B-spline Surface: {self.name}', axes=1, viewup="z", interactive=True)
            return plotting_elements
        else:
            return plotting_elements
