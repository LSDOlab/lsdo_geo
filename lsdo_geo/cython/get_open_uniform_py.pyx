import numpy as np
cimport numpy as np

from caddee.cython.get_open_uniform cimport get_open_uniform

def get_open_uniform(int order, int num_control_points, np.ndarray[double] knot_vector):
  get_open_uniform(order, num_control_points, &knot_vector[0])