import numpy as np
import array_mapper as am
import scipy.sparse as sps

from caddee.cython.basis_matrix_volume_py import get_basis_volume_matrix
from caddee.cython.volume_projection_py import compute_volume_projection

from caddee.primitives.bsplines.bspline import BSpline

import vedo

class BSplineVolume(BSpline):
    def __init__(self, name, order_u, order_v, order_w, knots_u, knots_v, knots_w, shape, control_points):
        self.name = name
        self.order_u = order_u
        self.knots_u = knots_u
        self.order_v = order_v
        self.knots_v = knots_v
        self.order_w = order_w
        self.knots_w = knots_w
        self.shape = shape
        self.control_points = control_points
        self.num_control_points = shape[0] * shape[1] * shape[2]

    def compute_evaluation_map(self, u_vec, v_vec, w_vec):
        num_points = len(u_vec) # = len(v_vec) = len(w_vec), they correspond

        data = np.zeros(num_points * self.order_u * self.order_v * self.order_w) 
        row_indices = np.zeros(len(data), np.int32)
        col_indices = np.zeros(len(data), np.int32)

        get_basis_volume_matrix(self.order_u, self.shape[0], 0, u_vec, self.knots_u, 
            self.order_v, self.shape[1], 0, v_vec, self.knots_v,
            self.order_w, self.shape[2], 0, w_vec, self.knots_w, 
            num_points, data, row_indices, col_indices)

        basis0 = sps.csc_matrix((data, (row_indices, col_indices)), shape=(len(u_vec), self.num_control_points) )
        
        return basis0


    def compute_derivative_evaluation_map(self, u_vec, v_vec, w_vec):
        data = np.zeros(len(u_vec) * self.order_u * self.order_v * self.order_w)
        row_indices = np.zeros(len(data), np.int32)
        col_indices = np.zeros(len(data), np.int32)

        get_basis_volume_matrix(self.order_u, self.shape[0], 1, u_vec, self.knots_u, 
            self.order_v, self.shape[1], 1, v_vec, self.knots_v,
            self.order_w, self.shape[2], 1, w_vec, self.knots_w, 
            len(u_vec), data, row_indices, col_indices)

        basis1 = sps.csc_matrix((data, (row_indices, col_indices)), shape=(len(u_vec), self.num_control_points) )
        
        return basis1


    def compute_second_derivative_evaluation_map(self, u_vec, v_vec, w_vec):
        data = np.zeros(len(u_vec) * self.order_u * self.order_v * self.order_w)
        row_indices = np.zeros(len(data), np.int32)
        col_indices = np.zeros(len(data), np.int32)

        get_basis_volume_matrix(self.order_u, self.shape[0], 2, u_vec, self.knots_u, 
            self.order_v, self.shape[1], 2, v_vec, self.knots_v,
            self.order_w, self.shape[2], 2, w_vec, self.knots_w, 
            len(u_vec), data, row_indices, col_indices)

        basis2 = sps.csc_matrix((data, (row_indices, col_indices)), shape=(len(u_vec), self.num_control_points) )
        
        return basis2


    def evaluate_points(self, u_vec, v_vec, w_vec):

        basis0 = self.compute_evaluation_map(u_vec, v_vec, w_vec)
        points = basis0.dot(self.control_points.reshape((self.num_control_points, 3)))

        return points


    def evaluate_derivative(self, u_vec, v_vec, w_vec):

        basis1 = self.compute_derivative_evaluation_map(u_vec, v_vec, w_vec)
        derivs1 = basis1.dot(self.control_points.reshape((self.num_control_points, 3)))

        return derivs1

    def evaluate_second_derivative(self, u_vec, v_vec, w_vec):

        basis2 = self.compute_second_derivative_evaluation_map(u_vec, v_vec, w_vec)
        derivs2 = basis2.dot(self.control_points.reshape((self.num_control_points, 3)))

        return derivs2


    def project(self, points:np.ndarray, direction:np.ndarray=None, grid_search_n:int=15,
                    max_iter:int=100, return_parametric_coordinates:bool=False, plot:bool=False):
        input_shape = points.shape
        flattened_points = points.flatten()
        if len(points.shape) > 1:
            num_points = np.cumprod(points.shape[:-1])[-1]
        else:
            num_points = 1

        if direction is None:
            direction = np.zeros((np.cumprod(points.shape)[-1],))
        else:
            direction = np.tile(direction, num_points)

        u_vec_flattened = np.zeros(num_points)
        v_vec_flattened = np.zeros(num_points)
        w_vec_flattened = np.zeros(num_points)
        num_control_points = np.cumprod(self.shape[:-1])[-1]

        compute_volume_projection(
            self.order_u, self.shape[0],
            self.order_v, self.shape[1],
            self.order_w, self.shape[2],
            num_points, max_iter,
            flattened_points, 
            self.control_points.reshape(self.num_control_points * 3),
            self.knots_u, self.knots_v, self.knots_w,
            u_vec_flattened, v_vec_flattened, w_vec_flattened, grid_search_n, direction.reshape((-1,))
        )

        map = self.compute_evaluation_map(u_vec_flattened, v_vec_flattened, w_vec_flattened)
        projected_points = am.array(input=self.control_points.reshape((num_control_points,-1)), linear_map=map, shape=input_shape)

        if plot:
            # Plot the surfaces that are projected onto
            plotter = vedo.Plotter()
            primitive_meshes = self.plot(plot_types=['mesh'], opacity=0.3, show=False)
            # Plot 
            plotting_points = []
            flattened_projected_points = (projected_points.value).reshape((num_points, -1)) # last axis usually has length 3 for x,y,z
            plotting_primitive_control_points = vedo.Points(flattened_projected_points, r=12, c='blue')  # TODO make this (1,3) instead of (3,)
            plotting_points.append(plotting_primitive_control_points)
            plotter.show(primitive_meshes, plotting_points, 'Projected Points', axes=1, viewup="z", interactive=True)

        u_vec = u_vec_flattened.reshape(tuple(input_shape[:-1],)+(1,))
        v_vec = v_vec_flattened.reshape(tuple(input_shape[:-1],)+(1,))
        w_vec = w_vec_flattened.reshape(tuple(input_shape[:-1],)+(1,))
        parametric_coordinates = np.concatenate((u_vec, v_vec, w_vec), axis=-1)

        if return_parametric_coordinates:
            return parametric_coordinates
        else:
            return projected_points


    def compute_projection_evalaluation_map(self, points_to_project, max_iter=100):
        u_vec, v_vec, w_vec = self.project(points_to_project, max_iter=max_iter)

        basis0 = self.compute_evaluation_map(u_vec, v_vec, w_vec)
        
        return basis0


    def plot(self, plot_types:list=['mesh','point_cloud'], opacity:float=0.3, additional_plotting_elements:list=[], show:bool=True):
        '''
        Plots the B-bpline volume.
        # TODO Make sure this works. (this was transferred from FFD)
        '''

        plotting_elements = additional_plotting_elements.copy()

        # TODO Currently plotting 6 outer surfaces. Decide if we want to show inside as well.
        # -- The 6 outer surfaces are plotted as meshes.
        plotting_elements = self.plot_surface(self.control_points[:,:,0], plot_types=plot_types, opacity=opacity, additional_plotting_elements=plotting_elements)
        plotting_elements = self.plot_surface(self.control_points[:,:,-1], plot_types=plot_types, opacity=opacity, additional_plotting_elements=plotting_elements)
        plotting_elements = self.plot_surface(self.control_points[:,0,:], plot_types=plot_types, opacity=opacity, additional_plotting_elements=plotting_elements)
        plotting_elements = self.plot_surface(self.control_points[:,-1,:], plot_types=plot_types, opacity=opacity, additional_plotting_elements=plotting_elements)
        plotting_elements = self.plot_surface(self.control_points[0,:,:], plot_types=plot_types, opacity=opacity, additional_plotting_elements=plotting_elements)
        plotting_elements = self.plot_surface(self.control_points[-1,:,:], plot_types=plot_types, opacity=opacity, additional_plotting_elements=plotting_elements)

        if show:
            plotter = vedo.Plotter()
            plotter.show(plotting_elements, f'B-spline Volume: {self.name}', axes=1, viewup="z", interactive=True)
            return plotting_elements
        else:
            return plotting_elements

    def plot_surface(self, points:np.ndarray, plot_types:list=['mesh', 'point_cloud'], opacity:float=0.5, additional_plotting_elements:list=[], show:bool=False):
        plotting_elements = additional_plotting_elements.copy()
        
        if 'point_cloud' in plot_types:
            num_points = points.shape[0]*points.shape[1]
            if 'mesh' in plot_types:
                point_opacity = (0.75*opacity + 0.25*1.)
            else:
                point_opacity = opacity
            plotting_elements.append(vedo.Points(points.reshape((num_points,-1)), r=5).opacity(point_opacity).color('darkred'))

        if 'mesh' in plot_types or 'wireframe' in plot_types:
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

            

            mesh = vedo.Mesh([vertices, faces]).opacity(opacity).color('lightblue')
        if 'mesh' in plot_types:
            plotting_elements.append(mesh)
        if 'wireframe' in plot_types:
            mesh = vedo.Mesh([vertices, faces]).opacity(opacity).color('lightblue')
            plotting_elements.append(mesh.wireframe().color('lightblue'))

        if show:
            plotter = vedo.Plotter()
            plotter.show(plotting_elements, 'Surface', axes=1, viewup="z", interactive=True)
            return plotting_elements
        else:
            return plotting_elements

