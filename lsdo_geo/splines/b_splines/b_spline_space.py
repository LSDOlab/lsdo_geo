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
    order : tuple[int]
    control_points_shape : tuple
    knots : np.ndarray = None
    knot_indices : list[np.ndarray] = None  # outer list is for parametric dimensions, inner list is for knot indices

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
            self.knots = np.array([])
            self.knot_indices = []
            for i in range(self.num_parametric_dimensions):
                num_knots = self.order[i] + self.control_points_shape[i]
                knots_i = np.zeros((num_knots,))
                get_open_uniform(order=self.order[i], num_control_points=self.control_points_shape[i], knot_vector=knots_i)
                self.knot_indices.append(np.arange(len(self.knots), len(self.knots) + num_knots))
                self.knots = np.hstack((self.knots, knots_i))
        else:
            self.knot_indices = []
            knot_index = 0
            for i in range(self.num_parametric_dimensions):
                num_knots_i = self.order[i] + self.control_points_shape[i]
                self.knot_indices.append(np.arange(knot_index, knot_index + num_knots_i))
                knot_index += num_knots_i


    def compute_evaluation_map(self, parametric_coordinates:np.ndarray, parametric_derivative_order:tuple=None) -> sps.csc_matrix:
        # NOTE: parametric coordinates are in shape (np,3) where 3 corresponds to u,v,w
        num_parametric_coordinates = parametric_coordinates.shape[-1]
        if parametric_derivative_order is None:
            parametric_derivative_order = (0,)*num_parametric_coordinates
        if type(parametric_derivative_order) is int:
            parametric_derivative_order = (parametric_derivative_order,)*num_parametric_coordinates
        elif len(parametric_derivative_order) == 1 and num_parametric_coordinates != 1:
            parametric_derivative_order = parametric_derivative_order*num_parametric_coordinates

        num_points = np.prod(parametric_coordinates.shape[:-1])
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
            knots_u = self.knots[self.knot_indices[0]].copy()
            knots_v = self.knots[self.knot_indices[1]].copy()
            get_basis_surface_matrix(order_u, self.control_points_shape[0], parametric_derivative_order[0], u_vec, knots_u, 
                order_v, self.control_points_shape[1], parametric_derivative_order[1], v_vec, knots_v, 
                len(u_vec), data, row_indices, col_indices)

        basis0 = sps.csc_matrix((data, (row_indices, col_indices)), shape=(len(u_vec), num_control_points))
        
        return basis0


if __name__ == "__main__":
    from lsdo_geo.splines.b_splines.b_spline_space import BSplineSpace
    from lsdo_geo.cython.get_open_uniform_py import get_open_uniform

    num_control_points = 10
    order = 4
    # knots_u = np.zeros((num_control_points + order))
    # knots_v = np.zeros((num_control_points + order))
    # get_open_uniform(order=order, num_control_points=num_control_points, knot_vector=knots_u)
    # get_open_uniform(order=order, num_control_points=num_control_points, knot_vector=knots_v)
    # space_of_cubic_b_spline_surfaces_with_10_cp = BSplineSpace(name='cubic_b_spline_surfaces_10_cp', order=(order,order), knots=(knots_u,knots_v))
    space_of_cubic_b_spline_surfaces_with_10_cp = BSplineSpace(name='cubic_b_spline_surfaces_10_cp', order=(order,order),
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
        space_of_cubic_b_spline_surfaces_with_10_cp.compute_evaluation_map(parametric_coordinates=parametric_coordinates)
    derivative_map = \
        space_of_cubic_b_spline_surfaces_with_10_cp.compute_evaluation_map(
            parametric_coordinates=parametric_coordinates, parametric_derivative_order=(1,1))
    second_derivative_map = \
        space_of_cubic_b_spline_surfaces_with_10_cp.compute_evaluation_map(
            parametric_coordinates=parametric_coordinates, parametric_derivative_order=(2,2))