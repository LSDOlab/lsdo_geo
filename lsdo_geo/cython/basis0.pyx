cdef int get_basis0(int order, int num_control_points, double u, double* knot_vector, double* basis0):
    cdef int i, j1, j2, l, n



    # Find the knot interval
    cdef int i_start = -1
    # print('order: ', order)
    # print('num_control_points: ', num_control_points)
    for i in range(order - 1, num_control_points):
        # print('i:', i)
        # print('u: ', u)
        # print('knot vector[i]: ', knot_vector[i])
        # print('knot vector[i+1]: ', knot_vector[i+1])

        if (knot_vector[i] <= u) and (u < knot_vector[i + 1]):
            i_start = i - order + 1
            # print('i_start: ', i_start)
            # print('------------------------------')

    # Initialize the basis0 to (0., ..., 0., 1.)
    for i in range(order - 1):
        basis0[i] = 0.

    basis0[order - 1] = 1.

    # If parameter is at the maximum of the knot vector, set the i_start appropriately
    if abs(u - knot_vector[order + num_control_points - 1]) < 1e-14:
        i_start = num_control_points - order
        # print('i_start MAX: ', i_start)


    # print('BEFORE LOOP i_start: ', i_start)
    # Recursion loop over the order index
    for i in range(1, order):
        j1 = order - i
        j2 = order
        n = i_start + j1
        
        # print('i_start: ', i_start)
        # print('i:', i)
        # print('order: ', order)
        # print('n:', n)
        # print('-----------------------------')

        if knot_vector[n + i] != knot_vector[n]:
            basis0[j1 - 1] = ( knot_vector[n + i] - u ) / ( knot_vector[n + i] - knot_vector[n] ) * basis0[j1]
            # print('1. i_start: ', i_start)
            # print('-----------------------------')

        else:
            basis0[j1 - 1] = 0.
            # print('2. i_start: ', i_start)
            # print('-----------------------------')


        for j in range(j1, j2 - 1):
            n = i_start + j + 1
            if knot_vector[n + i - 1] != knot_vector[n - 1]:
                basis0[j] = ( u - knot_vector[n - 1] ) / ( knot_vector[n + i - 1] - knot_vector[n - 1] ) * basis0[j]
            else:
                basis0[j] = 0.
            if knot_vector[n + i] != knot_vector[n]:
                basis0[j] += ( knot_vector[n + i] - u ) / ( knot_vector[n + i] - knot_vector[n] ) * basis0[j + 1]

        n = i_start + j2

        if knot_vector[n + i - 1] != knot_vector[n - 1]:
            basis0[j2 - 1] = ( u - knot_vector[n - 1] ) / ( knot_vector[n + i - 1] - knot_vector[n - 1] ) * basis0[j2 - 1]
        else:
            basis0[j2 - 1] = 0.



    # print('AFTER LOOP i_start: ', i_start)

    return i_start
