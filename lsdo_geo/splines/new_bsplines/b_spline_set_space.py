from dataclasses import dataclass

import m3l
import numpy as np
import array_mapper as am
import scipy.sparse as sps

from lsdo_geo.cython.basis_matrix_surface_py import get_basis_surface_matrix
from lsdo_geo.cython.get_open_uniform_py import get_open_uniform

from lsdo_geo.splines.new_bsplines.b_spline_space import BSplineSpace

@dataclass
class BSplineSetSpace(m3l.FunctionSpace):
    name : str
    spaces : dict[str, BSplineSpace]

    # aggregated quantities
    # knots : tuple = None

    def __post_init__(self):
        # Counting num coefficients to avoid error
        self.num_coefficients = 0
        for space_name, space in self.spaces.items():
            self.num_coefficients += space.num_coefficients

        # self.knots = []
        # for space_name, space in self.spaces.items():
        # TODO: Down the line, potentially try to aggregate knot vectors.
        # -- NOTE: If I do this, then this seems like it should be a continuous manifold (with boundaries)
        #     and not support combinations of B-splines of different dimensions (such as surfaces + volumes)


    def compute_evaluation_map(self, space_name:str, parametric_coordinates:np.ndarray) -> sps.csc_matrix:
        return self.spaces[space_name].compute_evaluation_map(parametric_coordinates=parametric_coordinates)

    def compute_derivative_evaluation_map(self, space_name:str, parametric_coordinates:np.ndarray) -> sps.csc_matrix:
        return self.spaces[space_name].compute_derivative_evaluation_map(parametric_coordinates=parametric_coordinates)

    def compute_second_derivative_evaluation_map(self, space_name:str, parametric_coordinates:np.ndarray) -> sps.csc_matrix:
        return self.spaces[space_name].compute_second_derivative_evaluation_map(parametric_coordinates=parametric_coordinates)


if __name__ == "__main__":
    from lsdo_geo.splines.new_bsplines.b_spline_space import BSplineSpace
    from lsdo_geo.cython.get_open_uniform_py import get_open_uniform

    num_control_points = 10
    order = 4
    
    space_of_cubic_bspline_surfaces_with_10_cp = BSplineSpace(name='cubic_bspline_surfaces_10_cp', order=(order,order),
                                                              control_points_shape=(num_control_points,num_control_points))
    space_of_quadratic_bspline_surfaces_with_5_cp = BSplineSpace(name='quadratic_bspline_surfaces_5_cp', order=(3,3),
                                                              control_points_shape=(5,5))
    b_spline_spaces = {space_of_cubic_bspline_surfaces_with_10_cp.name : space_of_cubic_bspline_surfaces_with_10_cp,
                       space_of_quadratic_bspline_surfaces_with_5_cp.name : space_of_quadratic_bspline_surfaces_with_5_cp}
    b_spline_set_space = BSplineSetSpace(name='my_b_spline_set', spaces=b_spline_spaces)

    parametric_coordinates = np.array([
        [0., 0.],
        [0., 1.],
        [1., 0.],
        [1., 1.],
        [0.5, 0.5],
        [0.25, 0.75]
    ])
    eval_map = \
        b_spline_set_space.compute_evaluation_map(space_of_cubic_bspline_surfaces_with_10_cp.name,
                                                  parametric_coordinates=parametric_coordinates)
    eval_map = \
        b_spline_set_space.compute_evaluation_map(space_of_quadratic_bspline_surfaces_with_5_cp.name,
                                                  parametric_coordinates=parametric_coordinates)
    derivative_map = \
        b_spline_set_space.compute_derivative_evaluation_map(space_of_cubic_bspline_surfaces_with_10_cp.name,
                                                             parametric_coordinates=parametric_coordinates)
    derivative_map = \
        b_spline_set_space.compute_derivative_evaluation_map(space_of_quadratic_bspline_surfaces_with_5_cp.name,
                                                             parametric_coordinates=parametric_coordinates)
    second_derivative_map = \
        b_spline_set_space.compute_second_derivative_evaluation_map(space_of_cubic_bspline_surfaces_with_10_cp.name,
                                                                    parametric_coordinates=parametric_coordinates)
    second_derivative_map = \
        b_spline_set_space.compute_second_derivative_evaluation_map(space_of_quadratic_bspline_surfaces_with_5_cp.name,
                                                                    parametric_coordinates=parametric_coordinates)