cdef get_open_uniform(int order, int num_control_points, double* knot_vector):
    cdef int i
    cdef double den = num_control_points - order + 1

    for i in range(order):
        knot_vector[i] = 0.

    for i in range(order, num_control_points):
        knot_vector[i] = (i - order + 1) / den

    for i in range(num_control_points, num_control_points + order):
        knot_vector[i] = 1.
