import numpy as np
import pickle
from dataclasses import dataclass

import m3l
from lsdo_geo.splines.b_splines.b_spline_set import BSplineSet

@dataclass
class Geometry(BSplineSet):
    
    # def __init__(self, function_space, coefficients):
    #     self.function_space = function_space
    #     self.coefficients = coefficients

    def get_function_space(self):
        return self.space
    
    def define_component(self):
        pass

    def import_geometry(self, file_name:str):
        '''
        Imports geometry from a file.

        Parameters
        ----------
        file_name : str
            The name of the file (with path) that containts the geometric information.
        '''
        from lsdo_geo.splines.b_splines.b_spline_functions import import_file, create_b_spline_set
        b_splines = import_file(file_name)
        b_spline_set = create_b_spline_set(self.name, b_splines)

        self.space = b_spline_set.space
        self.coefficients = b_spline_set.coefficients
        self.num_physical_dimensions = b_spline_set.num_physical_dimensions
        self.coefficient_indices = b_spline_set.coefficient_indices
        self.connections = b_spline_set.connections

    def refit(self, num_control_points:tuple=(25,25), fit_resolution:tuple=(100,100), order:tuple=(4,4)):
        from lsdo_geo.splines.b_splines.b_spline_functions import refit_b_spline_set
        b_spline_set = refit_b_spline_set(self, num_control_points, fit_resolution, order)

        self.debugging_b_spline_set = b_spline_set

        self.space = b_spline_set.space
        self.coefficients = b_spline_set.coefficients
        self.control_points = b_spline_set.control_points
        self.num_physical_dimensions = b_spline_set.num_physical_dimensions
        self.coefficient_indices = b_spline_set.coefficient_indices
        self.connections = b_spline_set.connections



if __name__ == "__main__":
    from lsdo_geo.core.python_core.geometry.geometry_functions import import_geometry
    geometry = import_geometry('lsdo_geo/splines/b_splines/sample_geometries/rectangular_wing.stp')
    geometry.refit()

    geometry.find_connections()
    geometry.plot()
    print('hi')