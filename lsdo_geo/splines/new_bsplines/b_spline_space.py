from dataclasses import dataclass

import m3l
import numpy as np
import array_mapper as am
import scipy.sparse as sps

from lsdo_geo.cython.basis_matrix_surface_py import get_basis_surface_matrix
from lsdo_geo.cython.get_open_uniform_py import get_open_uniform


@dataclass
class BSplineSpace(m3l.FunctionSpace):
    name : str
    order : tuple
    control_points_shape : tuple
    knots : tuple = None

    def __post_init__(self):
        # control_points_per_dimensions = []
        # for dimension in range(len(self.knots)):
        #     knots_i = self.knots[dimension]
        #     order_i = self.order[dimension]

        #     control_points_per_dimensions.append(len(knots_i) - order_i)

        # self.control_points_shape = tuple(control_points_per_dimensions)

        self.coefficients_shape = self.control_points_shape
        self.num_control_points = np.prod(self.control_points_shape)
        self.num_coefficients = self.num_control_points

        self.num_parametric_dimensions = len(self.order)

        if self.knots is None:
            self.knots = []
            for i in range(self.num_parametric_dimensions):
                num_knots = self.order[i] + self.control_points_shape[i]
                knots_i = np.zeros((num_knots,))
                get_open_uniform(order=self.order[i], num_control_points=self.control_points_shape[i], knot_vector=knots_i)
                self.knots.append(knots_i)


    def compute_evaluation_map(self, parametric_coordinates:np.ndarray, parametric_derivative_order:tuple=None) -> sps.csc_matrix:
        # NOTE: parametric coordinates are in shape (np,3) where 3 corresponds to u,v,w
        num_parametric_coordinates = parametric_coordinates.shape[-1]
        if parametric_derivative_order is None:
            parametric_derivative_order = (0,)*num_parametric_coordinates
        if type(parametric_derivative_order) is int:
            parametric_derivative_order = (parametric_derivative_order,)*num_parametric_coordinates
        elif len(parametric_derivative_order) == 1 and num_parametric_coordinates != 1:
            parametric_derivative_order = parametric_derivative_order*num_parametric_coordinates

        num_points = np.prod(parametric_coordinates.shape)
        order_multiplied = 1
        for i in range(len(self.order)):
            order_multiplied *= self.order[i]

        data = np.zeros(num_points * order_multiplied) 
        row_indices = np.zeros(len(data), np.int32)
        col_indices = np.zeros(len(data), np.int32)

        num_control_points = self.num_control_points

        if self.num_parametric_dimensions == 2:
            u_vec = parametric_coordinates[:,0].copy()
            v_vec = parametric_coordinates[:,1].copy()
            order_u = self.order[0]
            order_v = self.order[1]
            knots_u = self.knots[0]
            knots_v = self.knots[1]
            get_basis_surface_matrix(order_u, self.control_points_shape[0], parametric_derivative_order[0], u_vec, knots_u, 
                order_v, self.control_points_shape[1], parametric_derivative_order[1], v_vec, knots_v, 
                len(u_vec), data, row_indices, col_indices)

        basis0 = sps.csc_matrix((data, (row_indices, col_indices)), shape=(len(u_vec), num_control_points))
        
        return basis0

    # def compute_derivative_evaluation_map(self, parametric_coordinates:np.ndarray) -> sps.csc_matrix:
    #     num_points = np.prod(parametric_coordinates.shape[:-1])
    #     order_multiplied = 1
    #     for i in range(len(self.order)):
    #         order_multiplied *= self.order[i]

    #     data = np.zeros(num_points * order_multiplied) 
    #     row_indices = np.zeros(len(data), np.int32)
    #     col_indices = np.zeros(len(data), np.int32)

    #     num_control_points = self.num_control_points

    #     if self.num_parametric_dimensions == 2:
    #         u_vec = parametric_coordinates[:,0].copy()
    #         v_vec = parametric_coordinates[:,1].copy()
    #         order_u = self.order[0]
    #         order_v = self.order[1]
    #         knots_u = self.knots[0]
    #         knots_v = self.knots[1]
    #         get_basis_surface_matrix(order_u, self.control_points_shape[0], 1, u_vec, knots_u, 
    #             order_v, self.control_points_shape[1], 1, v_vec, knots_v, 
    #             len(u_vec), data, row_indices, col_indices)

    #     basis1 = sps.csc_matrix((data, (row_indices, col_indices)), shape=(len(u_vec), num_control_points))
        
    #     return basis1

    # def compute_second_derivative_evaluation_map(self, parametric_coordinates:np.ndarray) -> sps.csc_matrix:
    #     num_points = np.prod(parametric_coordinates.shape[:-1])
    #     order_multiplied = 1
    #     for i in range(len(self.order)):
    #         order_multiplied *= self.order[i]

    #     data = np.zeros(num_points * order_multiplied) 
    #     row_indices = np.zeros(len(data), np.int32)
    #     col_indices = np.zeros(len(data), np.int32)

    #     num_control_points = self.num_control_points

    #     if self.num_parametric_dimensions == 2:
    #         u_vec = parametric_coordinates[:,0].copy()
    #         v_vec = parametric_coordinates[:,1].copy()
    #         order_u = self.order[0]
    #         order_v = self.order[1]
    #         knots_u = self.knots[0]
    #         knots_v = self.knots[1]
    #         get_basis_surface_matrix(order_u, self.control_points_shape[0], 2, u_vec, knots_u, 
    #             order_v, self.control_points_shape[1], 2, v_vec, knots_v, 
    #             len(u_vec), data, row_indices, col_indices)

    #     basis2 = sps.csc_matrix((data, (row_indices, col_indices)), shape=(len(u_vec), num_control_points))
        
    #     return basis2


if __name__ == "__main__":
    from lsdo_geo.splines.new_bsplines.b_spline_space import BSplineSpace
    from lsdo_geo.cython.get_open_uniform_py import get_open_uniform

    num_control_points = 10
    order = 4
    # knots_u = np.zeros((num_control_points + order))
    # knots_v = np.zeros((num_control_points + order))
    # get_open_uniform(order=order, num_control_points=num_control_points, knot_vector=knots_u)
    # get_open_uniform(order=order, num_control_points=num_control_points, knot_vector=knots_v)
    # space_of_cubic_bspline_surfaces_with_10_cp = BSplineSpace(name='cubic_bspline_surfaces_10_cp', order=(order,order), knots=(knots_u,knots_v))
    space_of_cubic_bspline_surfaces_with_10_cp = BSplineSpace(name='cubic_bspline_surfaces_10_cp', order=(order,order),
                                                              control_points_shape=(num_control_points,num_control_points))

    parametric_coordinates = np.array([
        [0., 0.],
        [0., 1.],
        [1., 0.],
        [1., 1.],
        [0.5, 0.5],
        [0.25, 0.75]
    ])
    eval_map = \
        space_of_cubic_bspline_surfaces_with_10_cp.compute_evaluation_map(parametric_coordinates=parametric_coordinates)
    derivative_map = \
        space_of_cubic_bspline_surfaces_with_10_cp.compute_derivative_evaluation_map(parametric_coordinates=parametric_coordinates)
    second_derivative_map = \
        space_of_cubic_bspline_surfaces_with_10_cp.compute_second_derivative_evaluation_map(parametric_coordinates=parametric_coordinates)