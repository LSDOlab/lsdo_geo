cdef int get_basis2(int order, int num_control_points, double u, double* knot_vector, double* basis2):
    cdef int i, j1, j2, l, n
    cdef double den, b0_a, b0_b, b1_a, b1_b, b2_a, b2_b

    cdef double *basis0 = <double *> malloc(order * sizeof(double))
    cdef double *basis1 = <double *> malloc(order * sizeof(double))

    # Find the knot interval
    cdef int i_start = -1
    for i in range(order - 1, num_control_points):
        if (knot_vector[i] <= u) and (u < knot_vector[i + 1]):
            i_start = i - order + 1

    # Initialize the basis0 to (0., ..., 0., 1.)
    for i in range(order - 1):
        basis0[i] = 0.

    basis0[order - 1] = 1.

    # If parameter is at the maximum of the knot vector, set the i_start appropriately
    if abs(u - knot_vector[order + num_control_points - 1]) < 1e-14:
        i_start = num_control_points - order

    for i in range(order):
        basis1[i] = 0.
        basis2[i] = 0.

    # Recursion loop over the order index
    for i in range(1, order):
        j1 = order - i
        j2 = order

        for j in range(j1 - 1, j2):
            n = i_start + j + 1
            if knot_vector[n + i - 1] != knot_vector[n - 1]:
                den = knot_vector[n + i - 1] - knot_vector[n - 1]
                b0_a = ( u - knot_vector[n - 1] ) / den * basis0[j]
                b1_a = ( basis0[j] + ( u - knot_vector[n - 1] ) * basis1[j] ) / den
                b2_a = ( 2 * basis1[j] + ( u - knot_vector[n - 1] ) * basis2[j] ) / den
            else:
                b0_a = 0.
                b1_a = 0.
                b2_a = 0.
            if j != j2 - 1 and  knot_vector[n + i] != knot_vector[n]:
                den = knot_vector[n + i] - knot_vector[n]
                b0_b = ( knot_vector[n + i] - u ) / den * basis0[j + 1]
                b1_b = ( ( knot_vector[n + i] - u ) * basis1[j + 1] - basis0[j + 1] ) / den
                b2_b = ( ( knot_vector[n + i] - u ) * basis2[j + 1] - 2 * basis1[j + 1] ) / den
            else:
                b0_b = 0.
                b1_b = 0.
                b2_b = 0.

            basis0[j] = b0_a + b0_b
            basis1[j] = b1_a + b1_b
            if i > 1:
                basis2[j] = b2_a + b2_b

    free(basis0)
    free(basis1)

    return i_start
