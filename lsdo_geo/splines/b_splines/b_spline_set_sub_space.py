from dataclasses import dataclass

import m3l
import numpy as np
import array_mapper as am
import scipy.sparse as sps

from lsdo_geo.splines.b_splines.b_spline_set_space import BSplineSetSpace

# @dataclass
# This should perhaps be ConnectionSpace since the B-spline set space doesn't have the coefficients to know what connections exist.
# Fow now, I'm going to put off this idea of connections and start simple to allow more time to think about how to best implement this.
# If we store a Connection between every B-spline, we'll have n^2 connection B-splines which is probably intractable.
# class Connection:
#     connected_from : str
#     connected_to : str
# #    region_space : BSplineSpace    # A level set B-spline in parametric space


@dataclass
class BSplineSetSubSpace(m3l.FunctionSpace):
    '''
    A class for representing a function space for a sub-set of B-splines. This object points to a sub-set of the larger B-spline set space.

    Attributes
    ----------
    name : str
        The name of the B-spline set space.
    b_spline_set_space : BSplineSetSpace
        The B-spline set space that this is a sub-space of.
    b_spline_names : list[str]
        The names of the B-splines whose spaces form the B-spline set space.
    '''
    name : str
    b_spline_set_space : BSplineSetSpace
    b_spline_names : list[str]
    # connections : dict[str, dict[str, Connection]] = None  # Outer dict has key of B-spline name, inner dict has key of connected B-spline name

    # def __post_init__(self):
    #     pass
        
    def compute_evaluation_map(self, b_spline_name:str, parametric_coordinates:np.ndarray,
                               parametric_derivative_order:tuple[int]=None) -> sps.csc_matrix:
        '''
        Computes the evaluation map for a B-spline in the B-spline set space.

        Parameters
        ----------
        b_spline_name : str
            The name of the B-spline.

        parametric_coordinates : np.ndarray
            The parametric coordinates at which to evaluate the B-spline.

        parametric_derivative_order : tuple[int]
            The order of the parametric derivative to evaluate. The length of the tuple is the number of parametric dimensions.
        '''
        return self.b_spline_set_space.compute_evaluation_map(b_spline_name=b_spline_name,
                                                              parametric_coordinates=parametric_coordinates,
                                                              parametric_derivative_order=parametric_derivative_order)

if __name__ == "__main__":
    pass