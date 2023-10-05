import numpy as np
import pickle
from dataclasses import dataclass

import m3l
from lsdo_geo.splines.b_splines.b_spline_set import BSplineSet
from lsdo_geo.splines.b_splines.b_spline_sub_set import BSplineSubSet

@dataclass
class Geometry(BSplineSet):
    
    # def __init__(self, function_space, coefficients):
    #     self.function_space = function_space
    #     self.coefficients = coefficients

    def get_function_space(self):
        return self.space
    
    def copy(self):
        '''
        Creates a copy of the geometry that does not point to this geometry.
        '''
        return Geometry(self.name+'_copy', self.space, self.coefficients.copy(), self.num_physical_dimensions.copy(),
                  self.coefficient_indices.copy(), self.connections.copy())
    
    def declare_component(self, component_name:str, b_spline_names:list[str]=None, b_spline_search_names:list[str]=None) -> BSplineSubSet:
        '''
        Declares a component. This component will point to a sub-set of the entire geometry.

        Parameters
        ----------
        component_name : str
            The name of the component.
        b_spline_names : list[str]
            The names of the B-splines that make up the component.
        b_spline_search_names : list[str], optional
            The names of the B-splines to search for. Names of B-splines will be returned for each B-spline that INCLUDES the search name.
        '''
        if b_spline_names is None:
            b_spline_names_input = []
        else:
            b_spline_names_input = b_spline_names.copy()

        if b_spline_search_names is not None:
            b_spline_names_input += self.space.search_b_spline_names(b_spline_search_names)

        sub_set = self.declare_sub_set(sub_set_name=component_name, b_spline_names=b_spline_names_input)
        return sub_set
    
    def create_sub_geometry(self, sub_geometry_name:str, b_spline_names:list[str]=None, b_spline_search_names:list[str]=None) -> BSplineSubSet:
        '''
        Creates a new geometry that is a subset of the current geometry.

        Parameters
        ----------
        component_name : str
            The name of the component.
        b_spline_names : list[str]
            The names of the B-splines that make up the component.
        b_spline_search_names : list[str], optional
            The names of the B-splines to search for. Names of B-splines will be returned for each B-spline that INCLUDES the search name.
        '''
        if b_spline_names is None:
            b_spline_names_input = []
        else:
            b_spline_names_input = b_spline_names.copy()

        if b_spline_search_names is not None:
            b_spline_names_input += self.space.search_b_spline_names(b_spline_search_names)

        sub_set = self.create_sub_set(sub_set_name=sub_geometry_name, b_spline_names=b_spline_names_input)
        return sub_set

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

    def refit(self, num_coefficients:tuple=(25,25), fit_resolution:tuple=(100,100), order:tuple=(4,4)):
        '''
        Evaluates a grid over the geometry and finds the best set of coefficients/control points at the desired resolution to fit the geometry.

        Parameters
        ----------
        num_coefficients : tuple, optional
            The number of coefficients to use in each direction.
        fit_resolution : tuple, optional
            The number of points to evaluate in each direction for each B-spline to fit the geometry.
        order : tuple, optional
            The order of the B-splines to use in each direction.
        '''
        from lsdo_geo.splines.b_splines.b_spline_functions import refit_b_spline_set
        b_spline_set = refit_b_spline_set(self, num_coefficients, fit_resolution, order)

        self.debugging_b_spline_set = b_spline_set

        self.space = b_spline_set.space
        self.coefficients = b_spline_set.coefficients
        self.coefficients = b_spline_set.coefficients
        self.num_physical_dimensions = b_spline_set.num_physical_dimensions
        self.coefficient_indices = b_spline_set.coefficient_indices
        self.connections = b_spline_set.connections

        self.num_coefficients = b_spline_set.num_coefficients



if __name__ == "__main__":
    from lsdo_geo.core.python_core.geometry.geometry_functions import import_geometry
    import array_mapper as am
    import m3l

    # var1 = m3l.Variable('var1', shape=(2,3), value=np.array([[1., 2., 3.], [4., 5., 6.]]))
    # var2 = m3l.Variable('var2', shape=(2,3), value=np.array([[1., 2., 3.], [4., 5., 6.]]))
    # var3 = var1 + var2
    # print(var3)


    geometry = import_geometry('lsdo_geo/splines/b_splines/sample_geometries/rectangular_wing.stp')
    geometry.refit()

    geometry.find_connections()
    geometry.plot()

    projected_points1 = geometry.project(np.array([[0.2, 1., 10.], [0.5, 1., 1.]]), plot=True, direction=np.array([0., 0., -1.]))
    projected_points2 = geometry.project(np.array([[0.2, 0., 10.], [0.5, 1., 1.]]), plot=True, max_iterations=100)

    test_linspace = am.linspace(projected_points1, projected_points2)
    # print(test_linspace)

    right_wing = geometry.declare_component(component_name='right_wing', b_spline_search_names=['WingGeom, 0'])
    right_wing.plot()

    projected_points1_on_right_wing = right_wing.project(np.array([[0.2, 1., 10.], [0.5, 1., 1.]]), plot=True, direction=np.array([0., 0., -1.]))
    projected_points2_on_right_wing = right_wing.project(np.array([[0.2, 0., 1.], [0.5, 1., 1.]]), plot=True, max_iterations=100)

    left_wing = geometry.declare_component(component_name='left_wing', b_spline_search_names=['WingGeom, 1'])
    left_wing.plot()

    geometry2 = geometry.copy()
    geometry2.coefficients = geometry.coefficients.copy()*2
    geometry2.plot()
    geometry.plot()

    # print(projected_points1_on_right_wing.evaluate(geometry2.coefficients))

    left_wing_pressure_space = geometry.space.create_sub_space(sub_space_name='left_wing_pressure_space', b_spline_names=left_wing.b_spline_names)
    
    # Manually creating a pressure distribution
    pressure_coefficients = np.zeros((0,))
    for b_spline_name in left_wing.b_spline_names:
        left_wing_geometry_coefficients = geometry.coefficients[geometry.coefficient_indices[b_spline_name]].reshape((-1,3))
        b_spline_pressure_coefficients = \
            -4*8.1/(np.pi*8.1)*np.sqrt(1 - (2*left_wing_geometry_coefficients[:,1]/8.1)**2) \
            * np.sqrt(1 - (2*left_wing_geometry_coefficients[:,0]/4)**2) \
            * (left_wing_geometry_coefficients[:,2]+0.05)
        pressure_coefficients = np.concatenate((pressure_coefficients, b_spline_pressure_coefficients.flatten()))

    left_wing_pressure_function = left_wing_pressure_space.create_function(name='left_wing_pressure_function', 
                                                                           coefficients=pressure_coefficients, num_physical_dimensions=1)
    
    left_wing.plot(color=left_wing_pressure_function)

    # DO ACTUATIONS NEXT
    # THEN DO PYTHON FFD, PYTHON FFD SECTIONAL PARAMETERIZATION, THEN PYTHON FFD B-SPLINE SECTIONAL PARAMETERIZATION
    # NOTE: THE B-SPLINE SECTIONAL PARAMETERIZATION SHOULD JUST BE A STRAIGHT B-SPLINE PARAMETERIZATION, NOTHING SPECIFIC TO FFD
    # THEN DO INNER OPTIMIZATION
    # I guess, as doing each one, should just do the CSDL models and M3L operation at the same time.


    print('hi')