from __future__ import print_function
cdef compute_volume_projection(
    int order_u, int num_control_points_u,
    int order_v, int num_control_points_v,
    int order_w, int num_control_points_w,
    int num_points, int max_iter,
    double* pts, double* cps,
    double* knot_vector_u, double* knot_vector_v, double* knot_vector_w,
    double* u_vec, double* v_vec, double* w_vec,
    int guess_grid_n,
    double* axis,
):
    cdef int i_pt, i_order_u, i_order_v, i_order_w, i_start_u, i_start_v, i_start_w
    cdef int i_iter, k, index
    cdef double u, v, w, alpha
    cdef double P000[3], P100[3], P010[3], P001[3], P110[3], P011[3], P101[3],  P200[3], P020[3] , P002[3], 
    cdef double C[3], D[3], P[3], A[3], D2[3]
    cdef double n1[3], n2[3], nD[3], nA[3], nP10[3], nP01[3], nP20[3], nP11[3], nP02[3], f[3], fu[3], fv[3], fuu[3], fuv[3], fvv[3]
    cdef double x[3], dx[3], G[3], H[9], N[9], det, tem_dx[3], M[9], Adj[9]
    cdef double tao

    cdef double *basis_u0 = <double *> malloc(order_u * sizeof(double))
    cdef double *basis_u1 = <double *> malloc(order_u * sizeof(double))
    cdef double *basis_u2 = <double *> malloc(order_u * sizeof(double))
    cdef double *basis_v0 = <double *> malloc(order_v * sizeof(double))
    cdef double *basis_v1 = <double *> malloc(order_v * sizeof(double))
    cdef double *basis_v2 = <double *> malloc(order_v * sizeof(double))
    cdef double *basis_w0 = <double *> malloc(order_w * sizeof(double))
    cdef double *basis_w1 = <double *> malloc(order_w * sizeof(double))
    cdef double *basis_w2 = <double *> malloc(order_w * sizeof(double))

    # cdef double *knot_vector_u = <double *> malloc(
    #     (order_u + num_control_points_u) * sizeof(double))
    # cdef double *knot_vector_v = <double *> malloc(
    #     (order_v + num_control_points_v) * sizeof(double))
    cdef double norm_D2, dot_nA_D, dot_nA_P10, dot_nA_P01, dot_nA_P20, dot_nA_P11, dot_nA_P02, dot_D_P10, dot_D_P01 , dot_D_P20, dot_D_P11, dot_D_P02, dot_P10_P10, dot_P01_P01, dot_P01_P10                    
    cdef double ddu_normD, ddu2_normD, ddv_normD, ddv2_normD
    cdef double grad_u_numer, grad_denom, ddu_grad_u_numer, ddu_grad_denom, grad_v_numer, ddv_grad_v_numer, ddv_grad_denom, ddv_grad_u_numer

    #tom's initial guess vars
    cdef double temp_distance
    cdef double distance
    cdef double u_closest
    cdef double v_closest
    cdef double w_closest
    cdef double distance2
    cdef double tem_distance2
    
    cdef double size_norm_factor
    cdef double size_norm_factor_avg
    cdef double size_norm_factor_lower
    cdef double size_norm_factor_upper
    cdef double size_norm_factor_upper_lower_avg

    # normalizing vars
    cdef double P_corner000[3], P_corner100[3], P_corner010[3], P_corner001[3], P_corner110[3], P_corner011[3], P_corner101[3], P_corner111[3],
    cdef double diag_1[3]
    cdef double diag_2[3]

    cdef double diag_1_upper[3]
    cdef double diag_1_lower[3]
    cdef double diag_2_upper[3]
    cdef double diag_2_lower[3]

    cdef double height1[3]
    cdef double height2[3]
    cdef double height3[3]
    cdef double height4[3]

    cdef double height1_norm
    cdef double height2_norm
    cdef double height3_norm
    cdef double height4_norm

    cdef double diag_dist_1
    cdef double diag_dist_2
                
    # get_open_uniform(order_u, num_control_points_u, knot_vector_u)
    # get_open_uniform(order_v, num_control_points_v, knot_vector_v)

    # Getting normalizing factor measuring volume size
    x[0] = 0.
    x[1] = 0.
    x[2] = 0.
    i_start_u = get_basis0(
        order_u, num_control_points_u, x[0], knot_vector_u, basis_u0)
    i_start_v = get_basis0(
        order_v, num_control_points_v, x[1], knot_vector_v, basis_v0)
    i_start_w = get_basis0(
        order_w, num_control_points_w, x[2], knot_vector_w, basis_w0)
    for k in range(3): 
        P_corner000[k] = 0.
        for i_order_u in range(order_u):
            for i_order_v in range(order_v):
                for i_order_w in range(order_w):
                    index = 3 * ( num_control_points_w * num_control_points_v * (i_start_u + i_order_u) \
                        + num_control_points_w * (i_start_v + i_order_v) + (i_start_w + i_order_w) ) + k  
                    C[k] = cps[index]

                    P_corner000[k] = P_corner000[k] + basis_u0[i_order_u] * basis_v0[i_order_v] * basis_w0[i_order_w] * C[k]

    x[0] = 1.
    x[1] = 1.
    x[2] = 1.
    i_start_u = get_basis0(
        order_u, num_control_points_u, x[0], knot_vector_u, basis_u0)
    i_start_v = get_basis0(
        order_v, num_control_points_v, x[1], knot_vector_v, basis_v0)
    i_start_w = get_basis0(
        order_w, num_control_points_w, x[2], knot_vector_w, basis_w0)
    for k in range(3): 
        P_corner111[k] = 0.
        for i_order_u in range(order_u):
            for i_order_v in range(order_v):
                for i_order_w in range(order_w):
                    index = 3 * ( num_control_points_w * num_control_points_v * (i_start_u + i_order_u) \
                        + num_control_points_w * (i_start_v + i_order_v) + (i_start_w + i_order_w) ) + k  
                    C[k] = cps[index]

                    P_corner111[k] = P_corner111[k] + basis_u0[i_order_u] * basis_v0[i_order_v] * basis_w0[i_order_w] * C[k]

    x[0] = 0.
    x[1] = 1.
    x[2] = 1.
    i_start_u = get_basis0(
        order_u, num_control_points_u, x[0], knot_vector_u, basis_u0)
    i_start_v = get_basis0(
        order_v, num_control_points_v, x[1], knot_vector_v, basis_v0)
    i_start_w = get_basis0(
        order_w, num_control_points_w, x[2], knot_vector_w, basis_w0)
    for k in range(3): 
        P_corner011[k] = 0.
        for i_order_u in range(order_u):
            for i_order_v in range(order_v):
                for i_order_w in range(order_w):
                    index = 3 * ( num_control_points_w * num_control_points_v * (i_start_u + i_order_u) \
                        + num_control_points_w * (i_start_v + i_order_v) + (i_start_w + i_order_w) ) + k  
                    C[k] = cps[index]

                    P_corner011[k] = P_corner011[k] + basis_u0[i_order_u] * basis_v0[i_order_v] * basis_w0[i_order_w] * C[k]

    x[0] = 1.
    x[1] = 1.
    x[2] = 0.
    i_start_u = get_basis0(
        order_u, num_control_points_u, x[0], knot_vector_u, basis_u0)
    i_start_v = get_basis0(
        order_v, num_control_points_v, x[1], knot_vector_v, basis_v0)
    i_start_w = get_basis0(
        order_w, num_control_points_w, x[2], knot_vector_w, basis_w0)
    for k in range(3): 
        P_corner110[k] = 0.
        for i_order_u in range(order_u):
            for i_order_v in range(order_v):
                for i_order_w in range(order_w):
                    index = 3 * ( num_control_points_w * num_control_points_v * (i_start_u + i_order_u) \
                        + num_control_points_w * (i_start_v + i_order_v) + (i_start_w + i_order_w) ) + k  
                    C[k] = cps[index]

                    P_corner110[k] = P_corner110[k] + basis_u0[i_order_u] * basis_v0[i_order_v] * basis_w0[i_order_w] * C[k]
    x[0] = 1.
    x[1] = 0.
    x[2] = 1.
    i_start_u = get_basis0(
        order_u, num_control_points_u, x[0], knot_vector_u, basis_u0)
    i_start_v = get_basis0(
        order_v, num_control_points_v, x[1], knot_vector_v, basis_v0)
    i_start_w = get_basis0(
        order_w, num_control_points_w, x[2], knot_vector_w, basis_w0)
    for k in range(3): 
        P_corner101[k] = 0.
        for i_order_u in range(order_u):
            for i_order_v in range(order_v):
                for i_order_w in range(order_w):
                    index = 3 * ( num_control_points_w * num_control_points_v * (i_start_u + i_order_u) \
                        + num_control_points_w * (i_start_v + i_order_v) + (i_start_w + i_order_w) ) + k  
                    C[k] = cps[index]

                    P_corner101[k] = P_corner101[k] + basis_u0[i_order_u] * basis_v0[i_order_v] * basis_w0[i_order_w] * C[k]

    x[0] = 0.
    x[1] = 0.
    x[2] = 1.
    i_start_u = get_basis0(
        order_u, num_control_points_u, x[0], knot_vector_u, basis_u0)
    i_start_v = get_basis0(
        order_v, num_control_points_v, x[1], knot_vector_v, basis_v0)
    i_start_w = get_basis0(
        order_w, num_control_points_w, x[2], knot_vector_w, basis_w0)
    for k in range(3): 
        P_corner001[k] = 0.
        for i_order_u in range(order_u):
            for i_order_v in range(order_v):
                for i_order_w in range(order_w):
                    index = 3 * ( num_control_points_w * num_control_points_v * (i_start_u + i_order_u) \
                        + num_control_points_w * (i_start_v + i_order_v) + (i_start_w + i_order_w) ) + k  
                    C[k] = cps[index]

                    P_corner001[k] = P_corner001[k] + basis_u0[i_order_u] * basis_v0[i_order_v] * basis_w0[i_order_w] * C[k]

    x[0] = 1.
    x[1] = 0.
    x[2] = 0.
    i_start_u = get_basis0(
        order_u, num_control_points_u, x[0], knot_vector_u, basis_u0)
    i_start_v = get_basis0(
        order_v, num_control_points_v, x[1], knot_vector_v, basis_v0)
    i_start_w = get_basis0(
        order_w, num_control_points_w, x[2], knot_vector_w, basis_w0)
    for k in range(3): 
        P_corner100[k] = 0.
        for i_order_u in range(order_u):
            for i_order_v in range(order_v):
                for i_order_w in range(order_w):
                    index = 3 * ( num_control_points_w * num_control_points_v * (i_start_u + i_order_u) \
                        + num_control_points_w * (i_start_v + i_order_v) + (i_start_w + i_order_w) ) + k  
                    C[k] = cps[index]

                    P_corner100[k] = P_corner100[k] + basis_u0[i_order_u] * basis_v0[i_order_v] * basis_w0[i_order_w] * C[k]

    x[0] = 0.
    x[1] = 1.
    x[2] = 0.
    i_start_u = get_basis0(
        order_u, num_control_points_u, x[0], knot_vector_u, basis_u0)
    i_start_v = get_basis0(
        order_v, num_control_points_v, x[1], knot_vector_v, basis_v0)
    i_start_w = get_basis0(
        order_w, num_control_points_w, x[2], knot_vector_w, basis_w0)
    for k in range(3): 
        P_corner010[k] = 0.
        for i_order_u in range(order_u):
            for i_order_v in range(order_v):
                for i_order_w in range(order_w):
                    index = 3 * ( num_control_points_w * num_control_points_v * (i_start_u + i_order_u) \
                        + num_control_points_w * (i_start_v + i_order_v) + (i_start_w + i_order_w) ) + k  
                    C[k] = cps[index]

                    P_corner010[k] = P_corner010[k] + basis_u0[i_order_u] * basis_v0[i_order_v] * basis_w0[i_order_w] * C[k]

    for k in range(3):
        diag_1_upper[k] = P_corner111 - P_corner001
        diag_2_upper[k] = P_corner011 - P_corner101

        diag_1_lower[k] = P_corner110 - P_corner000
        diag_2_lower[k] = P_corner010 - P_corner100

        height1[k] = P_corner001 - P_corner000
        height2[k] = P_corner011 - P_corner010
        height3[k] = P_corner111 - P_corner110
        height4[k] = P_corner101 - P_corner100


    diag_dist_1_upper = norm(3, diag_1_upper)
    diag_dist_2_upper = norm(3, diag_2_upper)

    diag_dist_1_lower = norm(3, diag_1_lower)
    diag_dist_2_lower = norm(3, diag_2_lower)

    height1_norm = norm(3, height1)
    height2_norm = norm(3, height2)
    height3_norm = norm(3, height3)
    height4_norm = norm(3, height4)

    size_norm_factor_upper = diag_dist_1_upper*diag_dist_2_upper
    size_norm_factor_lower = diag_dist_1_lower*diag_dist_2_lower
    size_norm_factor_upper_lower_avg = (size_norm_factor_upper + size_norm_factor_lower)/2
    
    size_norm_factor_avg = (size_norm_factor_upper_lower_avg * (height1_norm + height2_norm + height3_norm + height4_norm) ) / 4


    #tom's initial guess implementation
    if not (guess_grid_n == 0) :
        for i in range(num_points) :
            u_closest = .5
            v_closest = .5
            w_closest = .5
            distance = 100000.
            distance2 = 100000.

            for k in range(3):
                P[k] = pts[3 * i + k]
                A[k] = axis[3 * i + k]

            for a in range(guess_grid_n) :
                for b in range(guess_grid_n) :
                    for c in range(guess_grid_n) :
                        x[0] = a/(guess_grid_n - 1)
                        x[1] = b/(guess_grid_n - 1)
                        x[2] = c/(guess_grid_n - 1)
                        i_start_u = get_basis0(
                            order_u, num_control_points_u, x[0], knot_vector_u, basis_u0)
                        i_start_v = get_basis0(
                            order_v, num_control_points_v, x[1], knot_vector_v, basis_v0)
                        i_start_w = get_basis0(
                            order_w, num_control_points_w, x[2], knot_vector_w, basis_w0)
                        #print('x0',x[0],'x1',x[1])
                        #print('i_start_u',i_start_u,'i_start_v',i_start_v)
                        for k in range(3): 
                            P000[k] = 0.
                            for i_order_u in range(order_u):
                                for i_order_v in range(order_v):
                                    for i_order_w in range(order_w):

                                        index = 3 * ( num_control_points_w * num_control_points_v * (i_start_u + i_order_u) \
                                            + num_control_points_w * (i_start_v + i_order_v) + (i_start_w + i_order_w) ) + k  
                                        C[k] = cps[index]

                                        P000[k] = P000[k] + basis_u0[i_order_u] * basis_v0[i_order_v] * basis_w0[i_order_w] * C[k]
                        for k in range(3) :
                            D[k] = P000[k] - P[k]
                        temp_distance = 0
                        temp_product = 0

                        if norm(3,A) < 1e-8:
                            temp_distance = norm(3,D)
                            if temp_distance < distance :
                                u_closest = x[0]
                                v_closest = x[1]
                                w_closest = x[2]
                                distance = temp_distance
                        else:
                            for k in range(3):
                                nA[k] = A[k]/norm(3,A)
                            #print(i,'n1:',n1,'n2',n2)
                            tao = 1e-5
                            temp_distance2 = dot(3,D,D) - dot(3,D,nA)**2 + tao*dot(3,D,D)#dot(3, n1, n2)
                            temp_distance2 = temp_distance2/size_norm_factor_avg
                            #print(i, temp_product)
                            if temp_distance2 < distance2 :
                                #print('test',i, temp_product)
                                u_closest = x[0]
                                v_closest = x[1]
                                w_closest = x[2]
                                distance2 = temp_distance2
            #print(i, product)  
              
            u_vec[i] = u_closest
            v_vec[i] = v_closest
            w_vec[i] = w_closest
            #print(i)
            # print(i, u_vec[i],v_vec[i])
    #exit()    

    for i_pt in range(num_points):
        x[0] = u_vec[i_pt]
        x[1] = v_vec[i_pt]
        x[2] = w_vec[i_pt]

        for k in range(3):
            P[k] = pts[3 * i_pt + k]
            A[k] = axis[3 * i_pt + k]

        for i_iter in range(max_iter):
            i_start_u = get_basis0(
                order_u, num_control_points_u, x[0], knot_vector_u, basis_u0)
            i_start_u = get_basis1(
                order_u, num_control_points_u, x[0], knot_vector_u, basis_u1)
            i_start_u = get_basis2(
                order_u, num_control_points_u, x[0], knot_vector_u, basis_u2)
            
            i_start_v = get_basis0(
                order_v, num_control_points_v, x[1], knot_vector_v, basis_v0)
            i_start_v = get_basis1(
                order_v, num_control_points_v, x[1], knot_vector_v, basis_v1)
            i_start_v = get_basis2(
                order_v, num_control_points_v, x[1], knot_vector_v, basis_v2)

            i_start_w = get_basis0(
                order_w, num_control_points_w, x[2], knot_vector_w, basis_w0)
            i_start_w = get_basis1(
                order_w, num_control_points_w, x[2], knot_vector_w, basis_w1)
            i_start_w = get_basis2(
                order_w, num_control_points_w, x[2], knot_vector_w, basis_w2)

            for k in range(3):
                P000[k] = 0.
                P100[k] = 0.
                P010[k] = 0.
                P001[k] = 0.
                P110[k] = 0.
                P011[k] = 0.
                P101[k] = 0.
                P200[k] = 0.
                P020[k] = 0.
                P002[k] = 0.

                for i_order_u in range(order_u):
                    for i_order_v in range(order_v):
                        for i_order_w in range(order_w):

                            index = 3 * ( num_control_points_w * num_control_points_v * (i_start_u + i_order_u) \
                                + num_control_points_w * (i_start_v + i_order_v) + (i_start_w + i_order_w) ) + k  
                            C[k] = cps[index]
                            #print(i_pt, i_iter,i_order_u,i_order_v,'C',C[k])
                            #print(basis_u1[i_order_u],basis_v0[i_order_v],basis_u2[i_order_u],basis_v0[i_order_v])
                            P000[k] = P000[k] + basis_u0[i_order_u] * basis_v0[i_order_v] * basis_w0[i_order_w] * C[k]

                            P100[k] = P100[k] + basis_u1[i_order_u] * basis_v0[i_order_v] * basis_w0[i_order_w] * C[k]
                            P010[k] = P010[k] + basis_u0[i_order_u] * basis_v1[i_order_v] * basis_w0[i_order_w]* C[k]
                            P001[k] = P001[k] + basis_u0[i_order_u] * basis_v0[i_order_v] * basis_w1[i_order_w]* C[k]

                            P110[k] = P110[k] + basis_u1[i_order_u] * basis_v1[i_order_v] * basis_w0[i_order_w]* C[k]
                            P011[k] = P011[k] + basis_u0[i_order_u] * basis_v1[i_order_v] * basis_w1[i_order_w]* C[k]
                            P101[k] = P101[k] + basis_u1[i_order_u] * basis_v0[i_order_v] * basis_w1[i_order_w]* C[k]

                            P200[k] = P200[k] + basis_u2[i_order_u] * basis_v0[i_order_v] * basis_w0[i_order_w]* C[k]
                            P020[k] = P020[k] + basis_u0[i_order_u] * basis_v2[i_order_v] * basis_w0[i_order_w]* C[k]
                            P002[k] = P002[k] + basis_u0[i_order_u] * basis_v0[i_order_v] * basis_w2[i_order_w]* C[k]

                D[k] = P000[k] - P[k]               
            '''min F:= (||D[k]||)^2'''
            #print(i_pt, i_iter,'D',D,'P00',P00,'P10',P10,'P01',P01,'P20',P20,'P02',P02)
            if norm(3,A) < 1e-8:
                alpha = 1
                G[0] = 2 * dot(3, D, P100)
                G[1] = 2 * dot(3, D, P010)
                G[2] = 2 * dot(3, D, P001)

                H[0] = 2 * (dot(3, P100, P100) + dot(3, D, P200))
                H[1] = 2 * (dot(3, P100, P010) + dot(3, D, P110))
                H[2] = 2 * (dot(3, P100, P001) + dot(3, D, P101))

                H[3] = H[1]
                H[4] = 2 * (dot(3, P010, P010) + dot(3, D, P020))
                H[5] = 2 * (dot(3, P010, P001) + dot(3, D, P011))

                H[6] = H[2]
                H[7] = H[5]
                H[8] = 2 * (dot(3, P001, P001) + dot(3, D, P002))


            else: 
                #print(i_pt, P00, P10, P01, P)
                # for k in range(3):                                  
                #     nD[k] = D[k]/norm(3,D)
                #     nA[k] = A[k]/norm(3,A)
                #     D2[k] = 2*D[k]
                alpha = 1
                norm_D = norm(3,D)
                norm_D2 = dot(3,D,D)

                dot_nA_D = dot(3,nA,D)
                dot_nA_P100 = dot(3,nA,P100)
                dot_nA_P010 = dot(3,nA,P010)
                dot_nA_P001 = dot(3,nA,P001)
                dot_nA_P110 = dot(3,nA,P110)
                dot_nA_P011 = dot(3,nA,P011)
                dot_nA_P101 = dot(3,nA,P101)

                dot_nA_P200 = dot(3,nA,P200)
                dot_nA_P020 = dot(3,nA,P020)
                dot_nA_P002 = dot(3,nA,P002)
                
                dot_D_P100 = dot(3,D,P100)
                dot_D_P010 = dot(3,D,P010)
                dot_D_P001 = dot(3,D,P001)
                dot_D_P110 = dot(3,D,P110)
                dot_D_P011 = dot(3,D,P011)
                dot_D_P101 = dot(3,D,P101)

                dot_D_P200 = dot(3,D,P200)
                dot_D_P020 = dot(3,D,P020)
                dot_D_P002 = dot(3,D,P002)

                dot_P100_P100 = dot(3,P100,P100)
                dot_P010_P010 = dot(3,P010,P010)
                dot_P001_P001 = dot(3,P001,P001)

                dot_P100_P010 = dot(3,P100,P010)
                dot_P100_P001 = dot(3,P100,P001)
                dot_P010_P001 = dot(3,P010,P001)

                #ddu_normD2 = 2*dot_D_P10
                #ddu_normD = 0.5*(norm_D2**(-0.5))*2*dot_D_P10 
                #ddu2_normD = -norm_D2**(-3/2)*dot_D_P10*dot_D_P10 + norm_D2**(-0.5)*dot_P10_P10 + norm_D2**(-0.5)*dot_D_P20
                #ddv_normD = 0.5*(norm_D2**(-0.5))*2*dot_D_P01
                #ddv2_normD = -norm_D2**(-3/2)*dot_D_P01*dot_D_P01 + norm_D2**(-0.5)*dot_P01_P01 + norm_D2**(-0.5)*dot_D_P02
                #dduddv_normD = -norm_D2**(-3/2)*dot_D_P01*dot_D_P10 + norm_D2**(-0.5)*dot_P01_P10 + norm_D2**(-0.5)*dot_D_P11

                #grad_u_numer = dot_nA_P10*norm_D - dot_nA_D*ddu_normD 
                #grad_denom = norm_D2 
                #ddu_grad_u_numer = dot_nA_P10*ddu_normD + dot_nA_P20*norm_D - (dot_nA_P10*ddu_normD + ddu2_normD*dot_nA_D) 
                #ddu_grad_denom = 2*norm_D*ddu_normD

                #grad_v_numer = dot_nA_P01*norm_D - dot_nA_D*ddv_normD 
                #ddv_grad_v_numer = dot_D_P01*ddv_normD + dot_nA_P02*norm_D - (dot_nA_P01*ddv_normD + ddv2_normD*dot_nA_D)
                #ddv_grad_denom = 2*norm_D*ddv_normD

                #ddv_grad_u_numer = dot_nA_P10*ddv_normD + dot_nA_P11*norm_D - (dot_nA_P01*ddu_normD + dduddv_normD*dot_nA_D)
                # G[0] = -(grad_u_numer) / grad_denom
                # G[1] = -(grad_v_numer) / grad_denom
                # H[0] = -(ddu_grad_u_numer*grad_denom - ddu_grad_denom*grad_u_numer) / (grad_denom**2)
                # H[1] = -(ddv_grad_u_numer*grad_denom - ddv_grad_denom*grad_u_numer) / (grad_denom**2)
                # H[2] = H[1] #-(ddu_grad_v_numer*grad_denom - ddu_grad_denom*grad_v_numer) / (grad_denom**2) #
                # H[3] = -(ddv_grad_v_numer*grad_denom - ddv_grad_denom*grad_v_numer) / (grad_denom**2)
                '''                     
                min F:= ||P00 - P||**2 - ((P00 - P)*nA)**2 + tao*||P00 - P||**2
                '''
                # print('D', D, 'P10', P10, 'P01', P10)
                tao = 1e-5
                G[0] = ( 2 * dot(3, D, P100) - 2 * dot(3, D, nA) * dot(3, P100, nA) + tao*2*dot(3, D, P100) ) /size_norm_factor_avg
                G[1] = (2 * dot(3, D, P010) - 2 * dot(3, D, nA) * dot(3, P010, nA) + tao*2*dot(3, D, P010) ) /size_norm_factor_avg
                G[2] = (2 * dot(3, D, P001) - 2 * dot(3, D, nA) * dot(3, P001, nA) + tao*2*dot(3, D, P001) ) /size_norm_factor_avg

                H[0] = (2 * dot(3, P100, P100) + 2 * dot(3, D, P200) - 2 * dot(3, P100, nA) * dot(3, P100, nA) - 2 * dot(3, D, nA) * dot(3, P200, nA) \
                    + tao*(2 * dot(3, P100, P100) + 2 * dot(3, D, P200)))/size_norm_factor_avg

                H[1] = (2 * dot(3, P100, P010) + 2 * dot(3, D, P110) - 2 * dot(3, P100, nA) * dot(3, P010, nA) - 2 * dot(3, D, nA) * dot(3, P110, nA) \
                    + tao*(2 * dot(3, P100, P010) + 2 * dot(3, D, P110)))/size_norm_factor_avg

                H[2] = (2 * dot(3, P100, P001) + 2 * dot(3, D, P101) - 2 * dot(3, P001, nA) * dot(3, P100, nA) - 2 * dot(3, D, nA) * dot(3, P101, nA) \
                    + tao*(2 * dot(3, P100, P001) + 2 * dot(3, D, P101)))/size_norm_factor_avg

                H[3] = H[1]

                H[4] = (2 * dot(3, P010, P010) + 2 * dot(3, D, P020) - 2 * dot(3, P010, nA) * dot(3, P010, nA) - 2 * dot(3, D, nA) * dot(3, P020, nA) \
                    + tao*(2 * dot(3, P010, P010) + 2 * dot(3, D, P020)))/size_norm_factor_avg

                H[5] = (2 * dot(3, P010, P001) + 2 * dot(3, D, P011) - 2 * dot(3, P001, nA) * dot(3, P010, nA) - 2 * dot(3, D, nA) * dot(3, P011, nA) \
                    + tao*(2 * dot(3, P010, P001) + 2 * dot(3, D, P011)))/size_norm_factor_avg

                H[6] = H[2]

                H[7] = H[5]

                H[8] = (2 * dot(3, P001, P001) + 2 * dot(3, D, P002) - 2 * dot(3, P001, nA) * dot(3, P001, nA) - 2 * dot(3, D, nA) * dot(3, P002, nA) \
                    + tao*(2 * dot(3, P001, P001) + 2 * dot(3, D, P002)))/size_norm_factor_avg
            

                # print('H0', H[0])
                # print('H1',H[1])
                # print('H2',H[2])
                # print('H3',H[3])
                # print('H4',H[4])
                # print('H5',H[5])
                # print('H6',H[6])
                # print('H7',H[7])
                # print('H8',H[8])

                # print('x', x)

                '''                     
                min F:= -nA*nD = -nA*(P00 - P)/norm_D
                '''
            #print(i_pt, i_iter, H[0], H[1], H[2], H[3],G[0], G[1])
            # if norm(3,A) == 0:
            for k in range(3):
                if (
                    (x[k] < 1e-14) and (G[k] > -1e-16) or #add condition of G=0 to solve the zerodivision error with det(H) = 0
                    (x[k] > (1 - 1e-14)) and (G[k] < 1e-16)
                ):
                    #print('test1',i_pt, i_iter, x[k],G[k],H[0], H[1], H[2], H[3])
                    G[k] = 0.
                    
                    if (k == 0): 
                        H[1] = 0.
                        H[2] = 0.
                        H[3] = 0.
                        H[6] = 0.

                    elif (k == 1):
                        H[1] = 0.
                        H[3] = 0.
                        H[5] = 0.
                        H[7] = 0.

                    elif (k == 2):
                        H[2] = 0.
                        H[5] = 0.
                        H[6] = 0.
                        H[7] = 0.

                    H[k * 4] = 1.
            
            # else:
            #for k in range(3):
             #   if (
              #      (abs(x[k] - 0) < 1e-14) and (G[k] > -1e-16) or
               #     (abs(x[k] - 1) < 1e-14) and (G[k] < 1e-16)
                #):
                    #print('test2', i_pt, i_iter, x[k],G[k],H[0], H[1], H[2], H[3],)
                 #   G[k] = 0.
                  #  H[1] = 0.
                   # H[2] = 0.
                    #H[k * 3] = 1.
                        #print('test3', i_pt, i_iter, G[0],G[1])
            # print(i_pt, i_iter, H[0], H[1], H[2], H[3])
            det = H[0] * (H[4]*H[8] - H[5]*H[7]) - H[1]*(H[3]*H[8] - H[5]*H[6]) - H[2] * (H[3]*H[7] - H[4]*H[6])
            
            # if det == 0:
            #     print(i_pt, i_iter, 'H', H[0], H[1], H[2], H[3])
            #     print('G', G)
            #     print(x)
            for k in range(3):#wedge case
                if (abs(H[k * 3] - 0) < 1e-16):
                    H[k * 3] = 1.

            # TODO: Compute a 3x3 matrix inverse and assign values to indices manually 

            # Create the minor matrix
            M[0] = H[4]*H[8] - H[5]*H[7]
            M[1] = H[3]*H[8] - H[5]*H[6]
            M[2] = H[3]*H[7] - H[4]*H[6]
            M[3] = H[1]*H[8] - H[2]*H[7]
            M[4] = H[0]*H[8] - H[2]*H[6]
            M[5] = H[0]*H[7] - H[1]*H[6]
            M[6] = H[1]*H[5] - H[2]*H[4]
            M[7] = H[0]*H[5] - H[2]*H[3]
            M[8] = H[0]*H[4] - H[3]*H[1]

            # Minor Matrix become cofactor matrix when you dot by [+ - +; - + -; + - +]
            M[1] = -M[1]
            M[3] = -M[3]
            M[5] = -M[5]
            M[7] = -M[7]

            # Take the Transpose of the Cofactor matrix to get the adjugate matrix
            Adj[0] = M[0]
            Adj[1] = M[3]
            Adj[2] = M[6]
            Adj[3] = M[1]
            Adj[4] = M[4]
            Adj[5] = M[7]
            Adj[6] = M[2]
            Adj[7] = M[5]
            Adj[8] = M[8]

            # Multiply 1/det(A) * Adj(A) to get the inverse
            N[0] = (1/det) * Adj[0]
            N[1] = (1/det) * Adj[1]
            N[2] = (1/det) * Adj[2]
            N[3] = (1/det) * Adj[3]
            N[4] = (1/det) * Adj[4]
            N[5] = (1/det) * Adj[5]
            N[6] = (1/det) * Adj[6]
            N[7] = (1/det) * Adj[7]
            N[8] = (1/det) * Adj[8]
            
            
            matvec(3, 3, N, G, dx)
            #print('dx',dx)
            if norm(3,A) == 0:
                for k in range(3):
                    dx[k] = -dx[k]
                for k in range(3):
                    if x[k] + dx[k] < 0:                        
                        dx[k] = -x[k]
                    elif x[k] + dx[k] > 1:
                        dx[k] = 1 - x[k]
            else:
                for k in range(3):
                    dx[k] = -dx[k]            
                for k in range(3):                                  
                    if x[k] + alpha*dx[k] < 0:                   
                        dx[k] = (-x[k])/alpha
                    elif x[k] + alpha*dx[k] > 1:
                        #print('yes2', x[k]+dx[k],x[k],dx[k])
                        dx[k] = (1 - x[k])/alpha#- x[k]+ (2 - x[k] - dx[k])#- x[k] + (x[k] + dx[k]-1)

            norm_G = norm(3, G)
            norm_dx = norm(3, dx)

            if norm(3,A) == 0:
                if norm_G < 1e-16 or norm_dx < 1e-16:
                    # print("solution found",norm_G,norm_dx)
                    break
            else:
                for k in range(3):
                    dx[k] = alpha*dx[k]
                norm_dx = norm(3, dx)
                if norm_G < 1e-16 or norm_dx < 1e-16:#(1 - temp_product) < 1e-6 or (temp_product+1) < 1e-6:#norm_dx < 8e-4:# norm_G < 1e-14 or  or # i_iter ==0 or 
                    #print("solution found",norm_G,norm_dx,i_iter)
                    break
            for k in range(3):
                x[k] = x[k] + dx[k]
            #
        #print(i_pt,'iter', i_iter, 'norm_G',norm_G, 'norm_dx',norm_dx)
        #print(i_pt,'iter',i_iter,'G',G,'H',H,'dx',dx)

        u_vec[i_pt] = x[0]
        v_vec[i_pt] = x[1]
        w_vec[i_pt] = x[2]

        # print(x[0])
        # print(x[1])
        # print(x[2])

    # cdef double P000[3], P100[3], P010[3], P001[3], P110[3], P011[3], P101[3],  P200[3], P020[3] , P002[3], 
    # cdef double C[3], D[3], P[3], A[3], D2[3]
    # cdef double n1[3], n2[3], nD[3], nA[3], nP10[3], nP01[3], nP20[3], nP11[3], nP02[3], f[3], fu[3], fv[3], fuu[3], fuv[3], fvv[3]
    # cdef double x[3], dx[3], G[3], H[9], N[9], det, tem_dx[3], M[9], Adj[9]

    # for i in range(9):
        # print('H: ', H[i])
        # print('N: ', H[i])
        # print('M: ', H[i])
        # print('Adj: ', H[i])
        

    # for i in range(3):
    #     print('p000:', P000[i])
    #     print('p100:', P100[i])
    #     print('p010:',P010[i])
    #     print('p001:',P001[i])
    #     print('p110:',P110[i])
    #     print('p011:',P011[i])
    #     print('p101:',P101[i])
    #     print('p200:',P200[i])
    #     print('p020:',P020[i])
    #     print('p002:',P002[i])
    #     print('C:',C[i])
    #     print('D:',D[i])
    #     print('P:',P[i])
    #     print('A:',A[i])
    #     print('D2:',D2[i])
    #     print('n1:',n1[i])
    #     print('n2:',n2[i])
    #     print('nD:',nD[i])
    #     print('nA:',nA[i])
    #     print('np10:',nP10[i])
    #     print('np01:',nP01[i])
    #     print('np20:',nP20[i])
    #     print('np11:',nP11[i])
    #     print('np02:',nP02[i])
    #     print('f:',f[i])
    #     print('fu:',fu[i])
    #     print('fv:',fv[i])
    #     print('fuu:',fuu[i])
    #     print('fuv:',fuv[i])
    #     print('fvv:',fvv[i])
    #     print('x:',x[i])
    #     print('dx:',dx[i])
    #     print('G:',G[i])
    #     print('tem_dx:',tem_dx[i])


    free(basis_u0)
    free(basis_u1)
    free(basis_u2)
    free(basis_v0)
    free(basis_v1)
    free(basis_v2)
    free(basis_w0)
    free(basis_w1)
    free(basis_w2)

    # free(knot_vector_u)
    # free(knot_vector_v)


cdef double dot(int size, double* a, double* b):
    cdef int i
    cdef double result = 0

    for i in range(size):
        result = result + a[i] * b[i]

    return result


cdef double norm(int size, double* a):
    return dot(size, a, a) ** 0.5


cdef matvec(int size_i, int size_j, double* A, double* x, double *y):
    cdef int i, j

    for i in range(size_i):
        y[i] = 0.
        for j in range(size_j):
            y[i] = y[i] + A[i * size_j + j] * x[j]