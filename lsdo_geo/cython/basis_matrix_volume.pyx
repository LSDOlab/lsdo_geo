ctypedef int (*get_basis_func)(int order, int num_control_points, double u, double* knot_vector, double* basis)


cdef get_basis_volume_matrix(
    int order_u, int num_control_points_u, int u_der, double* u_vec, double* knot_vector_u,
    int order_v, int num_control_points_v, int v_der, double* v_vec, double* knot_vector_v,
    int order_w, int num_control_points_w, int w_der, double* w_vec, double* knot_vector_w,
    int num_points, double* data, int* row_indices, int* col_indices,
):
    cdef int i_pt, i_order_u, i_order_v, i_order_w, i_start_u, i_start_v, i_start_w, i_nz

    cdef double *basis_u = <double *> malloc(order_u * sizeof(double))
    cdef double *basis_v = <double *> malloc(order_v * sizeof(double))
    cdef double *basis_w = <double *> malloc(order_w * sizeof(double))

    # cdef double *knot_vector_u = <double *> malloc((order_u + num_control_points_u) * sizeof(double))
    # cdef double *knot_vector_v = <double *> malloc((order_v + num_control_points_v) * sizeof(double))

    cdef get_basis_func get_basis_u, get_basis_v, get_basis_w

    if u_der == 0:
        get_basis_u = get_basis0
       # print('U_DER == 0')
    elif u_der == 1:
        get_basis_u = get_basis1
        #print('U_DER == 1')
    elif u_der == 2:
        get_basis_u = get_basis2
        # print('U_DER == 2')

    if v_der == 0:
        get_basis_v = get_basis0
    elif v_der == 1:
        get_basis_v = get_basis1
    elif v_der == 2:
        get_basis_v = get_basis2

    if w_der == 0:
        get_basis_w = get_basis0
    elif w_der == 1:
        get_basis_w = get_basis1
    elif w_der == 2:
        get_basis_w = get_basis2

    # get_open_uniform(order_u, num_control_points_u, knot_vector_u)
    # get_open_uniform(order_v, num_control_points_v, knot_vector_v)

    i_nz = 0
    for i_pt in range(num_points):
        i_start_u = get_basis_u(order_u, num_control_points_u, u_vec[i_pt], knot_vector_u, basis_u)
        i_start_v = get_basis_v(order_v, num_control_points_v, v_vec[i_pt], knot_vector_v, basis_v)
        i_start_w = get_basis_w(order_w, num_control_points_w, w_vec[i_pt], knot_vector_w, basis_w)

        #print('================================================')

       
        #print('order_u: ', order_u)
        #print('i_start_u: ', i_start_u)
        #print('num_control_points_u: ', num_control_points_u)

        #print('\n')

        #print('order_v: ', order_v)
        #print('i_start_v: ', i_start_v)
        #print('num_control_points_v: ', num_control_points_v)

        #print('\n')

        #print('order_w: ', order_w)
        #print('i_start_w: ', i_start_w)
        #print('num_control_points_w: ', num_control_points_w)

       # print('\n')

       # print('-----------------------------------------------')

        for i_order_u in range(order_u):
            for i_order_v in range(order_v):
                for i_order_w in range(order_w):

                    data[i_nz] = basis_u[i_order_u] * basis_v[i_order_v] * basis_w[i_order_w] 

                   # print('basis_u: ', basis_u[i_order_u])
                    #print('basis_v: ', basis_v[i_order_v])
                    #print('basis_w: ', basis_w[i_order_w])

                    row_indices[i_nz] = i_pt

                    col_indices[i_nz] = num_control_points_w * (num_control_points_v * (i_start_u + i_order_u)   +   (i_start_v + i_order_v) ) + (i_start_w + i_order_w)
                    
                    # Idea: If column index is returning negative, check that there is enough control points for the given order, and check to make sure u > 0 (u>=0 might be fine?)


                    # print('data: ', data[i_nz])
                    # print('row indices: ', row_indices[i_nz])
                    # print('col indices: ', col_indices[i_nz])

                    # print('\n')

                    i_nz += 1

    free(basis_u)
    free(basis_v)
    free(basis_w)
    # free(knot_vector_u)
    # free(knot_vector_v)

