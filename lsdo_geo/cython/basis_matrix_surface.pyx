ctypedef int (*get_basis_func)(int order, int num_control_points, double u, double* knot_vector, double* basis)


cdef get_basis_surface_matrix(
    int order_u, int num_control_points_u, int u_der, double* u_vec, double* knot_vector_u,
    int order_v, int num_control_points_v, int v_der, double* v_vec, double* knot_vector_v,
    int num_points, double* data, int* row_indices, int* col_indices,
):
    cdef int i_pt, i_order_u, i_order_v, i_start_u, i_start_v, i_nz

    cdef double *basis_u = <double *> malloc(order_u * sizeof(double))
    cdef double *basis_v = <double *> malloc(order_v * sizeof(double))

    # cdef double *knot_vector_u = <double *> malloc((order_u + num_control_points_u) * sizeof(double))
    # cdef double *knot_vector_v = <double *> malloc((order_v + num_control_points_v) * sizeof(double))

    cdef get_basis_func get_basis_u, get_basis_v

    if u_der == 0:
        get_basis_u = get_basis0
    elif u_der == 1:
        get_basis_u = get_basis1
    elif u_der == 2:
        get_basis_u = get_basis2

    if v_der == 0:
        get_basis_v = get_basis0
    elif v_der == 1:
        get_basis_v = get_basis1
    elif v_der == 2:
        get_basis_v = get_basis2

    # get_open_uniform(order_u, num_control_points_u, knot_vector_u)
    # get_open_uniform(order_v, num_control_points_v, knot_vector_v)

    i_nz = 0
    for i_pt in range(num_points):
        i_start_u = get_basis_u(order_u, num_control_points_u, u_vec[i_pt], knot_vector_u, basis_u)
        i_start_v = get_basis_v(order_v, num_control_points_v, v_vec[i_pt], knot_vector_v, basis_v)

        for i_order_u in range(order_u):
            for i_order_v in range(order_v):
                data[i_nz] = basis_u[i_order_u] * basis_v[i_order_v]
                row_indices[i_nz] = i_pt
                col_indices[i_nz] = num_control_points_v * (i_start_u + i_order_u) + (i_start_v + i_order_v)

                i_nz += 1

    free(basis_u)
    free(basis_v)
    # free(knot_vector_u)
    # free(knot_vector_v)