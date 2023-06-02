ctypedef int (*get_basis_func)(int order, int num_control_points, double u, double* knot_vector, double* basis)


cdef get_basis_curve_matrix(
    int order, int num_control_points, int u_der, double* u_vec, double* knot_vector,
    int num_points, double* data, int* row_indices, int* col_indices,
):
    cdef int i_pt, i_order, i_start, i_nz

    cdef double *basis = <double *> malloc(order * sizeof(double))

    cdef get_basis_func get_basis

    if u_der == 0:
        get_basis = get_basis0
    elif u_der == 1:
        get_basis = get_basis1
    elif u_der == 2:
        get_basis = get_basis2

    # get_open_uniform(order, num_control_points, knot_vector)

    i_nz = 0
    for i_pt in range(num_points):
        i_start = get_basis(order, num_control_points, u_vec[i_pt], knot_vector, basis)

        for i_order in range(order):
            data[i_nz] = basis[i_order]
            row_indices[i_nz] = i_pt
            col_indices[i_nz] = i_start + i_order

            i_nz += 1

    free(basis)
    #free(knot_vector)