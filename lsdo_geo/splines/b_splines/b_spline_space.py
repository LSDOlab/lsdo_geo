from dataclasses import dataclass

import m3l
import numpy as np
# import array_mapper as am
import scipy.sparse as sps

from lsdo_b_splines_cython.cython.basis_matrix_surface_py import get_basis_surface_matrix
from lsdo_b_splines_cython.cython.get_open_uniform_py import get_open_uniform

# from lsdo_geo.splines.b_splines.b_spline import BSpline   # Can't do this. Circular import.


@dataclass
class BSplineSpace(m3l.FunctionSpace):
    name : str
    order : tuple[int]
    parametric_coefficients_shape : tuple[int]
    knots : np.ndarray = None
    knot_indices : list[np.ndarray] = None  # outer list is for parametric dimensions, inner list is for knot indices

    def __post_init__(self):

        self.num_coefficient_elements = np.prod(self.parametric_coefficients_shape)

        self.num_parametric_dimensions = len(self.parametric_coefficients_shape)

        if type(self.order) is int:
            self.order = tuple([self.order] * self.num_parametric_dimensions)
        if len(self.order) == 1:
            self.order = tuple([self.order[0]] * self.num_parametric_dimensions)

        if self.knots is None:
            self.knots = np.array([])
            self.knot_indices = []
            for i in range(self.num_parametric_dimensions):
                num_knots = self.order[i] + self.parametric_coefficients_shape[i]
                knots_i = np.zeros((num_knots,))
                get_open_uniform(order=self.order[i], num_coefficients=self.parametric_coefficients_shape[i], knot_vector=knots_i)
                self.knot_indices.append(np.arange(len(self.knots), len(self.knots) + num_knots))
                self.knots = np.hstack((self.knots, knots_i))
        else:
            self.knot_indices = []
            knot_index = 0
            for i in range(self.num_parametric_dimensions):
                num_knots_i = self.order[i] + self.parametric_coefficients_shape[i]
                self.knot_indices.append(np.arange(knot_index, knot_index + num_knots_i))
                knot_index += num_knots_i

    
    def create_function(self, name:str, coefficients:np.ndarray) -> m3l.Function:
        '''
        Creates a function in this function space.

        Parameters
        ----------
        name : str
            The name of the function.
        coefficients : np.ndarray
            The coefficients of the function.
        '''
        # TODO: Automatically determine the number of physical dimensions from the shape of the coefficients array
        # and parametric_coefficients_shape.
        num_coefficient_elements = np.prod(self.parametric_coefficients_shape)

        if len(coefficients.shape) == 1:
            if np.mod(len(coefficients), num_coefficient_elements) != 0:
                raise ValueError('Invalid number of coefficients.')
            num_physical_dimensions = len(coefficients) // num_coefficient_elements
        elif len(coefficients.shape) == 2:
            if coefficients.shape[0] != num_coefficient_elements:
                raise ValueError('Invalid number or shape of coefficients.')
            num_physical_dimensions = coefficients.shape[1]
        elif len(coefficients.shape) == (len(self.parametric_coefficients_shape)+1):
            if coefficients.shape[:-1] != self.parametric_coefficients_shape:
                raise ValueError('Invalid number or shape of coefficients.')
            num_physical_dimensions = coefficients.shape[-1]
        else:
            raise ValueError('Invalid shape of coefficients. Please pass in a shape of (num_coeffs,)' + \
                            'or (num_coeffs, num_physical_dimensions) or parametric_coefficients_shape + (num_physical_dimensions,).')

        from lsdo_geo.splines.b_splines.b_spline import BSpline
        return BSpline(name=name, space=self, coefficients=coefficients, num_physical_dimensions=num_physical_dimensions)


if __name__ == "__main__":
    from lsdo_geo.splines.b_splines.b_spline_space import BSplineSpace
    from lsdo_b_splines_cython.cython.get_open_uniform_py import get_open_uniform

    num_coefficients = 10
    order = 4
    # knots_u = np.zeros((num_coefficients + order))
    # knots_v = np.zeros((num_coefficients + order))
    # get_open_uniform(order=order, num_coefficients=num_coefficients, knot_vector=knots_u)
    # get_open_uniform(order=order, num_coefficients=num_coefficients, knot_vector=knots_v)
    # space_of_cubic_b_spline_surfaces_with_10_cp = BSplineSpace(name='cubic_b_spline_surfaces_10_cp', order=(order,order), knots=(knots_u,knots_v))
    # NOTE: If passing in the knot vectors like this, the indices are also needed!

    space_of_cubic_b_spline_surfaces_with_10_cp = BSplineSpace(name='cubic_b_spline_surfaces_10_cp', order=(order,order),
                                                              parametric_coefficients_shape=(num_coefficients,num_coefficients))

    coefficients_line = np.linspace(0., 1., num_coefficients)
    coefficients_x, coefficients_y = np.meshgrid(coefficients_line,coefficients_line)
    coefficients = np.stack((coefficients_x, coefficients_y, 0.1*np.random.rand(10,10)), axis=-1)

    my_cubic_b_spline_surface = space_of_cubic_b_spline_surfaces_with_10_cp.create_function(
        name='my_cubic_b_spline_surface', coefficients=coefficients)
    my_cubic_b_spline_surface.plot()