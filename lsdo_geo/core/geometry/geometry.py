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
        if self.connections is not None:
            return Geometry(self.name+'_copy', self.space, self.coefficients.copy(), self.num_physical_dimensions.copy(),
                    self.coefficient_indices.copy(), self.connections.copy())
        else:
            return Geometry(self.name+'_copy', self.space, self.coefficients.copy(), self.num_physical_dimensions.copy(),
                    self.coefficient_indices.copy())

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

    def refit(self, num_coefficients:tuple=(20,20), fit_resolution:tuple=(50,50), order:tuple=(4,4), parallelize:bool=True):
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
        b_spline_set = refit_b_spline_set(self, order, num_coefficients, fit_resolution, parallelize=parallelize)

        self.debugging_b_spline_set = b_spline_set

        self.space = b_spline_set.space
        self.coefficients = b_spline_set.coefficients
        self.coefficients = b_spline_set.coefficients
        self.num_physical_dimensions = b_spline_set.num_physical_dimensions
        self.coefficient_indices = b_spline_set.coefficient_indices
        self.connections = b_spline_set.connections

        self.num_coefficients = b_spline_set.num_coefficients



if __name__ == "__main__":
    from lsdo_geo.core.geometry.geometry_functions import import_geometry
    import array_mapper as am
    import m3l
    import time

    # var1 = m3l.Variable('var1', shape=(2,3), value=np.array([[1., 2., 3.], [4., 5., 6.]]))
    # var2 = m3l.Variable('var2', shape=(2,3), value=np.array([[1., 2., 3.], [4., 5., 6.]]))
    # var3 = var1 + var2
    # print(var3)

    t1 = time.time()
    geometry = import_geometry('lsdo_geo/splines/b_splines/sample_geometries/rectangular_wing.stp', parallelize=False)
    # geometry = import_geometry('lsdo_geo/splines/b_splines/sample_geometries/lift_plus_cruise_final.stp')
    t2 = time.time()
    geometry.refit(parallelize=False)
    t3 = time.time()
    # geometry.find_connections() # NOTE: This is really really slow for large geometries. Come back to this.
    t4 = time.time()
    # geometry.plot()
    t5 = time.time()
    print('Import time: ', t2-t1)
    print('Refit time: ', t3-t2)
    print('Find connections time: ', t4-t3)
    print('Plot time: ', t5-t4)

    # projected_points1_coordinates = geometry.project(np.array([[0.2, 1., 10.], [0.5, 1., 1.]]), plot=True, direction=np.array([0., 0., -1.]))
    # projected_points2_coordinates = geometry.project(np.array([[0.2, 0., 10.], [0.5, 1., 1.]]), plot=True, max_iterations=100)

    # projected_points1 = geometry.evaluate(projected_points1_coordinates, plot=True)
    # projected_points2 = geometry.evaluate(projected_points2_coordinates, plot=True)

    # test_linspace = m3l.linspace(projected_points1, projected_points2)

    # import vedo
    # plotter = vedo.Plotter()
    # plotting_points = vedo.Points(test_linspace.value.reshape(-1,3))
    # geometry_plot = geometry.plot(show=False)
    # plotter.show(geometry_plot, plotting_points, interactive=True, axes=1)

    # right_wing = geometry.declare_component(component_name='right_wing', b_spline_search_names=['WingGeom, 0'])
    # right_wing.plot()

    # projected_points1_on_right_wing = right_wing.project(np.array([[0.2, 1., 10.], [0.5, 1., 1.]]), plot=True, direction=np.array([0., 0., -1.]))
    # projected_points2_on_right_wing = right_wing.project(np.array([[0.2, 0., 1.], [0.5, 1., 1.]]), plot=True, max_iterations=100)

    left_wing = geometry.declare_component(component_name='left_wing', b_spline_search_names=['WingGeom, 1'])
    # left_wing.plot()

    # geometry2 = geometry.copy()
    # geometry2.coefficients = geometry.coefficients.copy()*2
    # geometry2.plot()
    # geometry.plot()

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

    # left_wing_pressure_space = geometry.space.create_sub_space(sub_space_name='left_wing_pressure_space',
    #                                                                         b_spline_names=left_wing.b_spline_names)
    # # Manually creating a pressure distribution mesh
    # pressure_mesh = np.zeros((0,))
    # pressure_parametric_coordinates = []
    # for b_spline_name in left_wing.b_spline_names:
    #     left_wing_geometry_coefficients = geometry.coefficients[geometry.coefficient_indices[b_spline_name]].reshape((-1,3))
    #     b_spline_pressure_coefficients = \
    #         -4*8.1/(np.pi*8.1)*np.sqrt(1 - (2*left_wing_geometry_coefficients[:,1]/8.1)**2) \
    #         * np.sqrt(1 - (2*left_wing_geometry_coefficients[:,0]/4)**2) \
    #         * (left_wing_geometry_coefficients[:,2]+0.05)
    #     pressure_mesh = np.concatenate((pressure_mesh, b_spline_pressure_coefficients.flatten()))

    #     # parametric_coordinates = left_wing.project(points=left_wing_geometry_coefficients, targets=[b_spline_name])
    #     # pressure_parametric_coordinates.extend(parametric_coordinates)

    #     b_spline_space = left_wing_pressure_space.spaces[left_wing_pressure_space.b_spline_to_space[b_spline_name]]

    #     b_spline_num_coefficients_u = b_spline_space.parametric_coefficients_shape[0]
    #     b_spline_num_coefficients_v = b_spline_space.parametric_coefficients_shape[1]

    #     u_vec = np.einsum('i,j->ij', np.linspace(0., 1., b_spline_num_coefficients_u), np.ones(b_spline_num_coefficients_u)).flatten()
    #     v_vec = np.einsum('i,j->ij', np.ones(b_spline_num_coefficients_v), np.linspace(0., 1., b_spline_num_coefficients_v)).flatten()
    #     parametric_coordinates = np.hstack((u_vec.reshape((-1,1)), v_vec.reshape((-1,1))))

    #     for i in range(len(parametric_coordinates)):
    #         pressure_parametric_coordinates.append(tuple((b_spline_name, parametric_coordinates[i,:])))

    # pressure_coefficients = left_wing_pressure_space.fit_b_spline_set(fitting_points=pressure_mesh.reshape((-1,1)),
    #                                                                               fitting_parametric_coordinates=pressure_parametric_coordinates)

    # left_wing_pressure_function = left_wing_pressure_space.create_function(name='left_wing_pressure_function', 
    #                                                                        coefficients=pressure_coefficients, num_physical_dimensions=1)
    
    # left_wing.plot(color=left_wing_pressure_function)

    geometry3 = geometry.copy()
    axis_origin = geometry.evaluate(geometry.project(np.array([0.5, -10., 0.5])))
    axis_vector = geometry.evaluate(geometry.project(np.array([0.5, 10., 0.5]))) - axis_origin
    angles = 45
    geometry3.coefficients = m3l.rotate(points=geometry3.coefficients.reshape((-1,3)), axis_origin=axis_origin, axis_vector=axis_vector, angles=angles,
                                        units='degrees').reshape((-1,))
    geometry3.plot()

    geometry4 = geometry.copy()
    # left_wing_transition = geometry.declare_component(component_name='left_wing', b_spline_search_names=['WingGeom, 1'])
    # actuated_left_wing_transition = m3l.rotate(left_wing_transition.coefficients.reshape((-1,3)), axis_origin=axis_origin,
    #                                            axis_vector=axis_vector, angles=angles, units='degrees').reshape((-1,))
    geometry4.rotate(axis_origin=axis_origin, axis_vector=axis_vector, angles=angles, units='degrees')


    # DO ACTUATIONS NEXT
    # -- Should be able to actuate meshes/control/points/FFD, whatever. It should just take in points and rotate them around an axis.
    # ---- Considering this, it seems like actuate should be a general function (geometry_functions.py)
    # -- Do I want to create the framework of creating actuators, etc. or just rotate around / translate across an arbitrary axis?
    # THEN DO PYTHON FFD, PYTHON FFD SECTIONAL PARAMETERIZATION, THEN PYTHON FFD B-SPLINE SECTIONAL PARAMETERIZATION
    # NOTE: THE B-SPLINE SECTIONAL PARAMETERIZATION SHOULD JUST BE A STRAIGHT B-SPLINE PARAMETERIZATION, NOTHING SPECIFIC TO FFD
    # THEN DO INNER OPTIMIZATION
    # I guess, as doing each one, should just do the CSDL models and M3L operation at the same time.


    print('hi')