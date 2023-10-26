import m3l
import csdl

import numpy as np
import scipy.sparse as sps
# import array_mapper as am
import vedo

from lsdo_geo.splines.b_splines.b_spline_set import BSplineSet

from dataclasses import dataclass
from typing import Union

@dataclass
class BSplineSubSet:
    '''
    B-spline sub set class. This points to a larger B-spline set and is a sub-set of it.

    Parameters
    ----------
    name : str
        The name of the B-spline sub-set.
    b_spline_set : BSplineSet
        The B-spline set that this is a sub-set of.
    b_spline_names : list[str]
        The names of the B-splines whose spaces form the B-spline set space.
    '''
    name : str
    b_spline_set : BSplineSet
    b_spline_names : list[str]

    def get_coefficients(self) -> m3l.Variable:
        '''
        Gets the coefficients of the B-spline sub-set.

        Returns
        -------
        coefficients : m3l.Variable
            The coefficients of the B-spline sub-set.
        '''

        return self.b_spline_set.get_coefficients(b_spline_names=self.b_spline_names, name=self.name + '_coefficients')
    
    def assign_coefficients(self, coefficients:m3l.Variable):
        '''
        Assigns the coefficients of the B-spline sub-set.

        Parameters
        ----------
        coefficients : m3l.Variable
            The coefficients of the B-spline sub-set.
        '''

        return self.b_spline_set.assign_coefficients(coefficients=coefficients, b_spline_names=self.b_spline_names)


    def evaluate(self, parametric_coordinates:np.ndarray, parametric_derivative_order:tuple[int]=None) -> m3l.Variable:

        return self.b_spline_set.evaluate(parametric_coordinates=parametric_coordinates,
                                           parametric_derivative_order=parametric_derivative_order)


    def project(self, points:np.ndarray, targets:list[str]=None, direction:np.ndarray=None,
                grid_search_density_parameter:int=10, max_iterations=100, plot:bool=False):
        '''
        Projects points onto the B-spline set.

        Parameters
        -----------
        points : {np.ndarray, m3l.Variable}
            The points to be projected onto the system.
        targets : list, optional
            The list of primitives to project onto.
        direction : {np.ndarray, m3l.Variable}, optional
            An axis for perfoming projection along an axis. The projection will return the closest point to the axis.
        grid_search_density : int, optional
            The resolution of the grid search prior to the Newton iteration for solving the optimization problem.
        max_iterations : int, optional
            The maximum number of iterations for the Newton iteration.
        plot : bool
            A boolean on whether or not to plot the projection result.
        '''

        if targets is None:
            targets = list(self.b_spline_names)

        return self.b_spline_set.project(points=points, targets=targets, direction=direction,
                                         grid_search_density_parameter=grid_search_density_parameter, max_iterations=max_iterations, plot=plot)
    

    def rotate(self, axis_origin:m3l.Variable, axis_vector:m3l.Variable, angles:m3l.Variable, b_splines:list[str]=None, units:str='degrees'):
        '''
        Rotates the B-spline set about an axis.

        Parameters
        -----------
        b_splines : list[str]
            The B-splines to rotate.
        axis_origin : m3l.Variable
            The origin of the axis of rotation.
        axis_vector : m3l.Variable
            The vector of the axis of rotation.
        angles : m3l.Variable
            The angle of rotation.
        units : str
            The units of the angle of rotation. {degrees, radians}
        '''
        if b_splines is None:
            b_splines = list(self.b_spline_names)

        return self.b_spline_set.rotate(axis_origin=axis_origin, axis_vector=axis_vector, angles=angles, b_splines=b_splines, units=units)


    def plot(self, b_splines:list[str]=None, point_types:list=['evaluated_points'], plot_types:list=['surface'],
              opacity:float=1., color:Union[str,BSplineSet] ='#00629B', surface_texture:str="", additional_plotting_elements:list=[], show:bool=True):
        '''
        Plots the B-spline Surface.

        Parameters
        -----------
        b_splines : list[str]
            The B-splines to be plotted. If None, all B-splines are plotted.
        points_type : list
            The type of points to be plotted. {evaluated_points, coefficients}
        plot_types : list
            The type of plot {mesh, wireframe, point_cloud}
        opactity : float
            The opacity of the plot. 0 is fully transparent and 1 is fully opaque.
        color : str
            The 6 digit color code to plot the B-spline as.
        surface_texture : str = "" {"metallic", "glossy", ...}, optional
            The surface texture to determine how light bounces off the surface.
            See https://github.com/marcomusy/vedo/blob/master/examples/basic/lightings.py for options.
        additional_plotting_elemets : list
            Vedo plotting elements that may have been returned from previous plotting functions that should be plotted with this plot.
        show : bool
            A boolean on whether to show the plot or not. If the plot is not shown, the Vedo plotting element is returned.
        '''
        
        plotting_elements = additional_plotting_elements.copy()

        if b_splines is None:
            b_splines = list(self.b_spline_names)

        return self.b_spline_set.plot(b_splines=b_splines, point_types=point_types, plot_types=plot_types,
                                      opacity=opacity, color=color, surface_texture=surface_texture,
                                      additional_plotting_elements=plotting_elements, show=show)


if __name__ == "__main__":
    from lsdo_geo.splines.b_splines.b_spline_space import BSplineSpace
    from lsdo_b_splines_cython.cython.get_open_uniform_py import get_open_uniform

    # ''' Creating B-spline set manually '''

    # num_coefficients1 = 10
    # order1 = 4
    # num_coefficients2 = 5
    # order2 = 3
    
    # space_of_cubic_b_spline_surfaces_with_10_cp = BSplineSpace(name='cubic_b_spline_surfaces_10_cp', order=(order1,order1),
    #                                                           coefficients_shape=(num_coefficients1,num_coefficients1))
    # space_of_quadratic_b_spline_surfaces_with_5_cp = BSplineSpace(name='quadratic_b_spline_surfaces_5_cp', order=(order2,order2),
    #                                                           coefficients_shape=(num_coefficients2,num_coefficients2))
    # b_spline_spaces = {space_of_cubic_b_spline_surfaces_with_10_cp.name : space_of_cubic_b_spline_surfaces_with_10_cp,
    #                    space_of_quadratic_b_spline_surfaces_with_5_cp.name : space_of_quadratic_b_spline_surfaces_with_5_cp}
    # b_spline_set_space = BSplineSetSpace(name='my_b_spline_set', spaces=b_spline_spaces, 
    #                                      b_spline_to_space={'my_b_spline_1':space_of_cubic_b_spline_surfaces_with_10_cp.name,
    #                                                              'my_b_spline_2':space_of_quadratic_b_spline_surfaces_with_5_cp.name})


    # coefficients = np.zeros(((num_coefficients1*num_coefficients1 + num_coefficients2*num_coefficients2)*3))
    # coefficients[:num_coefficients1*num_coefficients1*3] = 0.
    # coefficients[num_coefficients1*num_coefficients1*3:] = 1.

    # # Connection
    # # coefficients[num_coefficients1*num_coefficients1*3 - 1] = 1.
    # b_spline_set = BSplineSet(name='my_b_spline_set', space=b_spline_set_space, coefficients=coefficients,
    #                           num_physical_dimensions={'my_b_spline_1':3, 'my_b_spline_2':3})

    ''' Importing B-spline set '''
    from lsdo_geo.splines.b_splines.b_spline_functions import import_file
    from lsdo_geo.splines.b_splines.b_spline_functions import create_b_spline_set
    from lsdo_geo.splines.b_splines.b_spline_functions import refit_b_spline_set

    b_splines = import_file('lsdo_geo/splines/b_splines/sample_geometries/rectangular_wing.stp')
    for b_spline in list(b_splines.values()):
        print(b_spline.name)
        # b_spline.plot(plot_types=['mesh'], show=True)
    
    b_spline_set = create_b_spline_set(name='sample_wing', b_splines=b_splines)
    b_spline_set = refit_b_spline_set(b_spline_set=b_spline_set, num_coefficients=(25,10), order=(4,3))
    b_spline_set.find_connections()
    # b_spline_set.plot()

    # projected_points1 = b_spline_set.project(np.array([[0.2, 1., 10.], [0.5, 1., 1.]]), plot=True, direction=np.array([0., 0., -1.]))
    projected_points2 = b_spline_set.project(np.array([[0.2, 0., 1.], [0.5, 1., 1.]]), plot=True, max_iterations=100)

    print('hi')