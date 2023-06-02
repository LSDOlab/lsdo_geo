from libc.stdlib cimport malloc, free


cdef int get_basis2(int order, int num_control_points, double u, double* knot_vector, double* basis2)