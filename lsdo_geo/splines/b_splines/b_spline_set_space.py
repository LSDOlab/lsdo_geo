from dataclasses import dataclass
from typing import Union

import m3l
import numpy as np
# import array_mapper as am
import scipy.sparse as sps

from lsdo_geo.splines.b_splines.b_spline_space import BSplineSpace
# from lsdo_geo.splines.b_splines.b_spline_set import BSplineSet    # Can't do this. Circular Import

# @dataclass
# This should perhaps be ConnectionSpace since the B-spline set space doesn't have the coefficients to know what connections exist.
# Fow now, I'm going to put off this idea of connections and start simple to allow more time to think about how to best implement this.
# If we store a Connection between every B-spline, we'll have n^2 connection B-splines which is probably intractable.
# class Connection:
#     connected_from : str
#     connected_to : str
# #    region_space : BSplineSpace    # A level set B-spline in parametric space


@dataclass
class BSplineSetSpace(m3l.FunctionSpace):
    '''
    A class for representing a function space for a set of B-splines.

    Attributes
    ----------
    name : str
        The name of the B-spline set space.
    spaces : dict[str, BSplineSpace]
        A dictionary of B-spline spaces. The keys are the names of the B-spline spaces.
    b_spline_to_space : dict[str, str]
        A dictionary of B-spline to space mappings. The keys are the names of the B-splines. The values are the names of the B-spline spaces.
    connections : dict[str, str] = None
        A dictionary of connections. The keys are the names of the B-spline spaces. The values are the names of the connected B-spline spaces.
    knots : np.ndarray = None
        The knots of the B-spline set space. The knots are flattened into one vector.
    knot_indices : dict[str, list[np.ndarray]] = None
        A dictionary of knot indices. The keys are the names of the B-spline.
    '''
    name : str
    spaces : dict[str, BSplineSpace]
    b_spline_to_space : dict[str, str]
    # connections : dict[str, dict[str, Connection]] = None  # Outer dict has key of B-spline name, inner dict has key of connected B-spline name
    # -- This dictionary nesting lends towards having every B-spline have a topological B-spline for every other (2 per connection).
    # -- This is in contrast to having one for every connection. One B-spline per connection can't do parametric topology though.
    # connections : dict[str, dict[str, BSplineSpace]] = None  # Outer dict has key of B-spline name, inner dict has key of connected B-spline name
    connections : dict[str, list[str]] = None  # key is name of B-spline, value is a list of names of connected B-splines
    # THIS IS CONNECTION IN PARAMETRIC SPACE. We can have a "disctontinuous" B-spline 
    #   set function where parametric spaces connect, but not in physical space.
    # -- For now, just use string for discrete whether or not these manifolds are connected.
    knots : np.ndarray = None   # Doesn't support different dimensions unless if all the knots_u,v,w are flattened into one vector
    knot_indices : dict[str, list[np.ndarray]] = None   # list index = dimension index
    b_spline_index_to_name : dict[int, str] = None

    def __post_init__(self):
        self.knots = []
        self.knot_indices = {}

        # self.num_coefficients = 0
        self.num_knots = 0
        b_spline_index = 0
        for b_spline_name, space_name in self.b_spline_to_space.items():
            space = self.spaces[space_name]
            self.knot_indices[b_spline_name] = []
            for i in range(space.num_parametric_dimensions):
                dimension_num_knots = len(space.knot_indices[i])
                self.knot_indices[b_spline_name].append(np.arange(self.num_knots, self.num_knots + dimension_num_knots))
                self.knots.append(space.knots[i])
                self.num_knots += dimension_num_knots

            # self.num_coefficients += space.num_coefficients
            # self.b_spline_index_to_name[b_spline_index] = b_spline_name
            b_spline_index += 1

        # self.num_coefficients = self.num_coefficients
        self.knots = np.hstack(self.knots)
            

    def create_function(self, name:str, coefficients:np.ndarray=None, num_physical_dimensions:Union[dict[str,int],int]=None) -> m3l.Function:
        '''
        Creates a function in the B-spline set space.

        Parameters
        ----------
        name : str
            The name of the function.
        coefficients : np.ndarray = None
            The coefficients of the function.
        num_physical_dimensions : {dict[str,int], int} = None
            The number of physical dimensions for each B-spline in the set.
        '''
        if num_physical_dimensions is None:
            set_num_coefficient_elements = 0
            for b_spline_name in self.b_spline_to_space:
                b_spline_space = self.spaces[b_spline_name]
                b_spline_num_coefficient_elements = b_spline_space.num_coefficient_elements
                set_num_coefficient_elements += b_spline_num_coefficient_elements

            if len(coefficients.shape) == 1:
                if np.mod(len(coefficients), set_num_coefficient_elements) != 0:
                    raise ValueError('Invalid number of coefficients.')
                num_physical_dimensions = int(len(coefficients) / set_num_coefficient_elements)
            elif len(coefficients.shape) == 2:
                if coefficients.shape[0] != set_num_coefficient_elements:
                    raise ValueError('Invalid number or shape of coefficients.')
                num_physical_dimensions = coefficients.shape[1]
            else:
                raise ValueError('Invalid shape of coefficients without passing in num_physical dimensions.' + 
                                 'Please pass in a shape of (num_coeffs,) or (-1,num_physical_dimesions).' + \
                                  'If number of physical dimensions changes across B-splines, please pass in the dictionary argument.')
             


        from lsdo_geo.splines.b_splines.b_spline_set import BSplineSet
        num_physical_dimensions_dict = {}
        if type(num_physical_dimensions) is int:
            for b_spline_name in self.b_spline_to_space:
                num_physical_dimensions_dict[b_spline_name] = num_physical_dimensions
            num_physical_dimensions = num_physical_dimensions_dict

        if coefficients is None:
            coefficients = np.zeros((0,))
            for b_spline_name in self.b_spline_to_space:
                b_spline_num_coefficients = self.spaces[self.b_spline_to_space[b_spline_name]].num_coefficient_elements \
                    * num_physical_dimensions[b_spline_name]
                coefficients = np.hstack((coefficients, np.zeros((b_spline_num_coefficients))))

        return BSplineSet(name=name, space=self, coefficients=coefficients, num_physical_dimensions=num_physical_dimensions)
    

    def fit_b_spline_set(self, fitting_points:np.ndarray, fitting_parametric_coordinates:list[tuple[str,np.ndarray]],
                         regularization_parameter:float=0.) -> np.ndarray:
        '''
        Fits a B-spline set to a mesh.

        Parameters
        ----------
        name : str
            The name of the B-spline set.
        fitting_points : np.ndarray
            The points to fit the B-spline set to. The points should have the number of the physical dimension as the last axis (e.g. (num_points, 3))
        fitting_parametric_coordinates : list[tuple[str,np.ndarray]]
            A list of tuples of the B-spline names and the parametric coordinates of the fitting points.
        regularization_parameter : float = 0.
            The regularization parameter to use when fitting the B-spline set to the mesh.

        Returns
        -------
        coefficients : np.ndarray
            The coefficients (control points) that choose the function that best fit the points within this function spcae.
        '''
        from lsdo_geo.splines.b_splines.b_spline_functions import fit_b_spline_set
        coefficients = fit_b_spline_set(fitting_points=fitting_points, fitting_parametric_coordinates=fitting_parametric_coordinates, 
                                        b_spline_set_space=self, regularization_parameter=regularization_parameter)
        return coefficients
        
    

    def search_b_spline_names(self, b_spline_search_names:list[str]):
        '''
        Searches for B-splines names in the B-spline set space.

        Parameters
        ----------
        b_spline_search_names : list[str]
            The names of the B-splines to search for. Names of B-splines will be returned for each B-spline that INCLUDES the search name.
        '''
        returned_b_spline_names = []
        for b_spline_name in self.b_spline_to_space:
            for b_spline_search_name in b_spline_search_names:
                if b_spline_search_name in b_spline_name:
                    returned_b_spline_names.append(b_spline_name)
                    break
        return returned_b_spline_names

    def declare_sub_space(self, sub_space_name:str, b_spline_names:list[str]=None, b_spline_search_names:list[str]=None):
        '''
        Creates a B-spline set sub space. This object points to a sub-set of the larger B-spline set space.

        Parameters
        ----------
        sub_space_name : str
            The name of the B-spline set sub space.
        b_spline_names : list[str]
            The names of the B-splines whose spaces form the B-spline set space.
        b_spline_search_names : list[str]
            The names of the B-splines to search for. Names of B-splines will be returned for each B-spline that INCLUDES the search name.

        Returns
        -------
        BSplineSetSubSpace
            The B-spline set sub space.
        '''
        if b_spline_names is None:
            b_spline_names_input = []
        else:
            b_spline_names_input = b_spline_names.copy()

        if b_spline_search_names is not None:
            b_spline_names_input += self.search_b_spline_names(b_spline_search_names)


        from lsdo_geo.splines.b_splines.b_spline_set_sub_space import BSplineSetSubSpace
        return BSplineSetSubSpace(name=sub_space_name, b_spline_set_space=self, b_spline_names=b_spline_names_input)


    def create_sub_space(self, sub_space_name:str, b_spline_names:list[str], b_spline_search_names:list[str]=None):
        '''
        Creates a new B-spline space that is a subset of the current B-spline space.

        Parameters
        ----------
        sub_space_name : str
            The name of the new B-spline set space.
        b_spline_names : list[str]
            The names of the B-splines whose space will form the new B-spline set space.
        b_spline_search_names : list[str]
            The names of the B-splines to search for. Names of B-splines will be returned for each B-spline that INCLUDES the search name.
        '''
        if b_spline_names is None:
            b_spline_names_input = []
        else:
            b_spline_names_input = b_spline_names.copy()

        if b_spline_search_names is not None:
            b_spline_names_input += self.search_b_spline_names(b_spline_search_names)

        spaces = {}
        b_spline_to_space = {}
        if self.connections is not None:
            connections = {}
        else:
            connections = None
        knots = []
        knot_indices = {}
        for b_spline_name in b_spline_names_input:
            space_name = self.b_spline_to_space[b_spline_name]
            spaces[space_name] = self.spaces[space_name]
            b_spline_to_space[b_spline_name] = space_name
            if self.connections is not None:
                connections[b_spline_name] = self.connections[b_spline_name]
            for i in range(self.spaces[space_name].num_parametric_dimensions):
                knot_indices[b_spline_name] = self.knot_indices[b_spline_name]
                knots.append(self.spaces[space_name].knots[i])

        knots = np.hstack(knots)
        sub_space = BSplineSetSpace(name=sub_space_name, spaces=spaces, b_spline_to_space=b_spline_to_space,
                                    connections=connections, knots=knots, knot_indices=knot_indices)
        
        return sub_space
    

    def copy(self):
        spaces = {}
        b_spline_to_space = {}
        if self.connections is not None:
            connections = {}
        else:
            connections = None
        knots = []
        knot_indices = {}
        for b_spline_name in list(self.b_spline_to_space.keys()):
            space_name = self.b_spline_to_space[b_spline_name]
            spaces[space_name] = self.spaces[space_name]
            b_spline_to_space[b_spline_name] = space_name
            if self.connections is not None:
                connections[b_spline_name] = self.connections[b_spline_name]
            for i in range(self.spaces[space_name].num_parametric_dimensions):
                knot_indices[b_spline_name] = self.knot_indices[b_spline_name]
                knots.append(self.spaces[space_name].knots[i])

        knots = np.hstack(knots)
        copied_space = BSplineSetSpace(name=self.name+'_copy', spaces=spaces, b_spline_to_space=b_spline_to_space,
                                    connections=connections, knots=knots, knot_indices=knot_indices)
        return copied_space


if __name__ == "__main__":
    from lsdo_geo.splines.b_splines.b_spline_space import BSplineSpace

    num_coefficients1 = 10
    num_coefficients2 = 5
    order1 = 4
    order2 = 3
    
    space_of_cubic_b_spline_surfaces_with_10_cp = BSplineSpace(name='cubic_b_spline_surfaces_10_cp', order=(order1,order1),
                                                              parametric_coefficients_shape=(num_coefficients1,num_coefficients1))
    space_of_quadratic_b_spline_surfaces_with_5_cp = BSplineSpace(name='quadratic_b_spline_surfaces_5_cp', order=(order2,order2),
                                                              parametric_coefficients_shape=(num_coefficients2,num_coefficients2))
    b_spline_spaces = {space_of_cubic_b_spline_surfaces_with_10_cp.name : space_of_cubic_b_spline_surfaces_with_10_cp,
                       space_of_quadratic_b_spline_surfaces_with_5_cp.name : space_of_quadratic_b_spline_surfaces_with_5_cp}
    
    b_spline_to_space = {space_of_cubic_b_spline_surfaces_with_10_cp.name : space_of_cubic_b_spline_surfaces_with_10_cp.name,
                                space_of_quadratic_b_spline_surfaces_with_5_cp.name : space_of_quadratic_b_spline_surfaces_with_5_cp.name}
    b_spline_set_space = BSplineSetSpace(name='my_b_spline_set', spaces=b_spline_spaces, b_spline_to_space=b_spline_to_space)

    coefficients_line = np.linspace(0., 1., num_coefficients1)
    coefficients_x, coefficients_y = np.meshgrid(coefficients_line,coefficients_line)
    coefficients1 = np.stack((coefficients_x, coefficients_y, 0.1*np.random.rand(num_coefficients1,num_coefficients1)), axis=-1)

    coefficients_line = np.linspace(0., 1., num_coefficients2)
    coefficients_x, coefficients_y = np.meshgrid(coefficients_line,coefficients_line)
    coefficients_y += 1.5
    coefficients2 = np.stack((coefficients_x, coefficients_y, 0.1*np.random.rand(num_coefficients2,num_coefficients2)), axis=-1)

    coefficients = np.hstack((coefficients1.reshape((-1,)), coefficients2.reshape((-1,))))

    my_b_spline_surface_set = b_spline_set_space.create_function(
        name='my_b_spline_set', coefficients=coefficients)
    my_b_spline_surface_set.plot()