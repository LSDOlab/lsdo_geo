cdef get_open_uniform(int order, int num_coefficients, double* knot_vector):
    cdef int i
    cdef double den = num_coefficients - order + 1

    for i in range(order):
        knot_vector[i] = 0.

    for i in range(order, num_coefficients):
        knot_vector[i] = (i - order + 1) / den

    for i in range(num_coefficients, num_coefficients + order):
        knot_vector[i] = 1.
