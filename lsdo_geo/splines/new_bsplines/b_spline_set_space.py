import m3l

import numpy as np

from dataclasses import dataclass

@dataclass
class BSplineSetSpace(m3l.FunctionSpace):
    name : str
    b_spline_spaces : dict
    order : tuple = None
    knots : tuple = None

    def __post_init__(self):
        self.assemble()

    def assemble(self):
        self.num_control_points = 0

        for b_spline_space_name, b_spline_space in self.b_spline_spaces.items():
            if self.order is None:
                self.order = list(b_spline_space.order)
                self.knots = list(b_spline_space.knots)
            for i in range(len(b_spline_space.order)):

                self.order[i] = np.vstack((self.order[i], b_spline_space.order[i]))
                self.knots[i] = np.vstack((self.knots[i], b_spline_space.knots[i]))

            self.num_control_points += b_spline_space.num_control_points

        self.order = tuple(self.order)
        self.knots = tuple(self.knots)
        self.num_coefficients = self.num_control_points