import array_mapper as am
import numpy as np
import scipy.sparse as sps

class Mesh(am.MappedArray):
    '''
    An instance of a Mesh object. This is an array that stores an affine mapping to calculate itself.

    Parameters
    ----------
    input: array_like
        The input into the map to calculate the array

    linear_map: numpy.ndarray or scipy.sparse matrix
        Linear map for evaluating the array
    
    offset: numpy.ndarray
        Offset for evaluating the array

    shape: tuple
        The shape of the MappedArray

    parametric_coordinates : tuple = None
        The tuple (i,u,v,...) storing the parametric coordinates of the geometry function space that the mesh is defined on.
    '''
    
    def __init__(self, input=None, linear_map=None, offset=None, shape:tuple=None, parametric_coordinates:tuple=None) -> None:
        '''
        Creates an instance of a MappedArray object.

        Parameters
        ----------
        input: array_like
            The input into the map to calculate the array

        linear_map: numpy.ndarray or scipy.sparse matrix
            Linear map for evaluating the array
        
        offset: numpy.ndarray
            Offset for evaluating the array

        shape: tuple
            The shape of the MappedArray
            
        parametric_coordinates : tuple = None
            The tuple (i,u,v,...) storing the parametric coordinates of the geometry function space that the mesh is defined on.
        '''
        
        # Listing list of attributes
        self.input = input
        self.linear_map = linear_map
        self.offset_map = offset
        self.shape = shape
        self.parametric_coordinates = parametric_coordinates
        self.value = None

        if type(input) is np.ndarray:
            self.input = input
            self.value = input
            self.shape = input.shape
        elif type(input) is list or type(input) is tuple:
            self.input = np.array(input)
            self.value = np.array(input)
            self.shape = np.array(input).shape
        elif type(input) is am.MappedArray and linear_map is not None:
            raise Exception("Can't instantiate the input with a MappedArray while specifying a linear map."
            "Please use the array_mapper.dot function.")
        elif type(input) is am.MappedArray:   # Creates a copy of the input MappedArray
            self.linear_map = input.linear_map
            self.offset_map = input.offset_map
            self.shape = input.shape
            self.input = input.input
            self.value = input.value

        if self.linear_map is None and self.input is not None:
            self.linear_map = sps.eye(self.input.shape[0])

        if type(shape) is list or type(shape) is tuple:
            self.shape = shape

        self.evaluate()