# F(u) = \sum_k 0.5 * [ \sum_i  B_i(u) C_ik - P_k ] ** 2
# dF/du = [ \sum_i B_i(u) C_ik - P_k ]
#         [ \sum_i B_i'(u) C_ik ]

# F(u) = \sum_k 0.5 * [ P_k(u,v) - Q_k ] ** 2
# dF/du = [ P_k(u,v) - Q_k ] dP_k/du
# d2F/du2 = dP_k/du dP_k/du + [ P_k(u) - Q_k ] d2P_k/du2



cdef compute_curve_projection(
    int order_u, int num_control_points_u,
    int num_points, int max_iter,
    double* pts, double* cps,
    double* u_vec, double* knot_vector,
    int n_guesses,
):
    cdef int i_pt, i_order_u, i_start_u
    cdef int i_iter, k, index
    cdef double u
    cdef double P00[3], P10[3], P20[3]
    cdef double C[3], D[3], P[3]
    cdef double x, dx, G, H, N, det

    cdef double *basis_u0 = <double *> malloc(order_u * sizeof(double))
    cdef double *basis_u1 = <double *> malloc(order_u * sizeof(double))
    cdef double *basis_u2 = <double *> malloc(order_u * sizeof(double))

    #cdef double *knot_vector = <double *> malloc(
    #    (order_u + num_control_points_u) * sizeof(double))



    #tom's initial guess vars
    cdef double temp_distance
    cdef double distance
    cdef double u_closest



    # get_open_uniform(order_u, num_control_points_u, knot_vector)


    #tom's initial guess implementation
    for i in range(num_points) :
        u_closest = .5
        distance = 1000.

        for k in range(3):
            P[k] = pts[3 * i + k]


        for a in range(n_guesses) :
                
            x = a/n_guesses

            i_start_u = get_basis0(
                order_u, num_control_points_u, x, knot_vector, basis_u0)

            for k in range(3): 
                P00[k] = 0.
                for i_order_u in range(order_u):
                        index = 3  * (i_start_u + i_order_u) \
                            + k
                        C[k] = cps[index]

                        P00[k] = P00[k] + basis_u0[i_order_u] * C[k]
            for k in range(3) :
                D[k] = P[k]-P00[k]

            temp_distance = norm(3,D)

            if temp_distance < distance :
                u_closest = x
                distance = temp_distance
        
        u_vec[i] = u_closest




    for i_pt in range(num_points):
        x = u_vec[i_pt]

        for k in range(3):
            P[k] = pts[3 * i_pt + k]

        for i_iter in range(max_iter):

            i_start_u = get_basis0(
                order_u, num_control_points_u, x, knot_vector, basis_u0)
            i_start_u = get_basis1(
                order_u, num_control_points_u, x, knot_vector, basis_u1)
            i_start_u = get_basis2(
                order_u, num_control_points_u, x, knot_vector, basis_u2)

            for k in range(3):
                P00[k] = 0.
                P10[k] = 0.
                P20[k] = 0.
        
                for i_order_u in range(order_u):
                    index = 3 * (i_start_u + i_order_u) \
                        + k
                    C[k] = cps[index]

                    P00[k] = P00[k] + basis_u0[i_order_u] * C[k]
                    P10[k] = P10[k] + basis_u1[i_order_u] * C[k]
                    P20[k] = P20[k] + basis_u2[i_order_u] * C[k]

                D[k] = P00[k] - P[k]

            G = dot(3, D, P10)
            H = dot(3, P10, P10) + dot(3, D, P20)

            if (
                (abs(x - 0) < 1e-14) and (G > 0) or
                (abs(x - 1) < 1e-14) and (G < 0)
            ):
                G = 0.
                H = 1.

            N = 1/H

            # matvec(1, 1, N, G, dx)

            dx = G*N

            dx = -dx

            if x + dx < 0:
                dx = -x
            elif x + dx > 1:
                dx = 1 - x

            # print(i_iter, norm_G)

            if abs(G) < 1e-14 or abs(dx) < 1e-14:
                break

            x = x + dx

        u_vec[i_pt] = x

    free(basis_u0)
    free(basis_u1)
    free(basis_u2)

    free(knot_vector)



cdef double dot(int size, double* a, double* b):
    cdef int i
    cdef double result = 0

    for i in range(size):
        result = result + a[i] * b[i]

    return result

cdef double norm(int size, double* a):
    return dot(size, a, a) ** 0.5
