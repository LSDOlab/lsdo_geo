import m3l
from dataclasses import dataclass

import numpy as np

@dataclass
class BSplineSpace(m3l.FunctionSpace):
    name : str
    order : tuple
    knots : tuple

    def __post_init__(self):
        control_points_per_dimensions = []
        for dimension in range(len(self.knots)):
            knots_i = self.knots[dimension]
            order_i = self.order[dimension]

            control_points_per_dimensions.append(len(knots_i) - 2*order_i)

        self.control_points_shape = tuple(control_points_per_dimensions)
        self.coefficients_shape = self.control_points_shape
        self.num_control_points = np.prod(self.control_points_shape)
        self.num_coefficients = self.num_control_points