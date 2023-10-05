from libc.stdlib cimport malloc, free


cdef int get_basis2(int order, int num_coefficients, double u, double* knot_vector, double* basis2)