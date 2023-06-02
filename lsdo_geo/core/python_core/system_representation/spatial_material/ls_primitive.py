import numpy as np

# from caddee.caddee_core.system_representation.system_representation import SystemRepresentation
# from caddee.caddee_core.system_representation.component.component import Component
from caddee.caddee_core.system_representation.utils.material import Material
from caddee.primitives.primitive import Primitive
from caddee.primitives.bsplines import bspline_surface, bspline_functions
import scipy.sparse as sps
import vedo
import array_mapper as am


class LSPrimitive:
    def __init__(self, primitive:Primitive=None, h=1e-4):
        self.primitive=primitive
        self.h = h

    def _heaviside(self, x):
        if x < -self.h:
            return 0
        elif x > self.h:
            return 1
        else: 
            return -1/4*(x/self.h)**3+3/4*(x/self.h)+1/2
    
    def _heaviside_derivative(self, x):
        if x<-self.h or x > self.h:
            return 0
        return -3/4*(x/self.h)**2/self.h+3/4/self.h


    def fit(self, parametric_coordinates, function_value):
        '''
        Fits new primitive to input data
        '''
        pass

    def evaluate_points(self, u_vec, v_vec):
        '''
        Evaluates the level set function at the parametric coordinates.
        '''
        points=self.primitive.evaluate_points(u_vec, v_vec)

        for i in range(0,points.shape[0]):
            points[i] = self._heaviside(points[i])

        return points

    def evaluate_derivative(self, u_vec, v_vec):
        '''
        Evaluates the derivative of the level set function at the parametric coordinates.
        '''
        # num_control_points = self.shape[0] * self.shape[1]
        
        # basis1 = self.compute_derivative_evaluation_map(u_vec, v_vec)
        # derivs1 = basis1.dot(self.control_points.reshape((num_control_points, 3)))

        # return derivs1 
        pass

    def evaluate_second_derivative(self, u_vec, v_vec):
        '''
        Evaluates the second derivative of the level set function at the parametric coordinates.
        '''
        # num_control_points = self.shape[0] * self.shape[1]
        
        # basis2 = self.compute_second_derivative_evaluation_map(u_vec, v_vec)
        # derivs2 = basis2.dot(self.control_points.reshape((num_control_points, 3)))

        # return derivs2
        pass