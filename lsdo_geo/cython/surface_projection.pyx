from __future__ import print_function
cdef compute_surface_projection(
    long[:] surfs_order_u, long[:] surfs_num_control_points_u,
    long[:] surfs_order_v, long[:] surfs_num_control_points_v,
    int num_points, int max_iter,
    double* pts, double* cps,
    double* knot_vector_u, double* knot_vector_v,
    double* u_vec, double* v_vec,
    int guess_grid_n,
    double* axis,
    long[:] surfs_index,
    double[:,:] surfs_cps,
    #double* u_vec_initial, double* v_vec_initial,
):
    cdef int i_pt, i_order_u, i_order_v, i_start_u, i_start_v
    cdef int i_iter, k, index
    cdef double u, v, alpha
    cdef double P00[3], P10[3], P01[3], P20[3], P11[3], P02[3]
    cdef double C[3], D[3], P[3], A[3], D2[3]
    cdef double n1[3], n2[3], nD[3], nA[3], nP10[3], nP01[3], nP20[3], nP11[3], nP02[3], f[3], fu[3], fv[3], fuu[3], fuv[3], fvv[3]
    cdef double x[2], dx[2], G[2], H[4], N[4], det, tem_dx[2]
    cdef double tao
    
    cdef int order_u, order_v
    order_u = surfs_order_u[0]
    order_v = surfs_order_v[0]
    cdef double *basis_u0 = <double *> malloc(order_u * sizeof(double))
    cdef double *basis_u1 = <double *> malloc(order_u * sizeof(double))
    cdef double *basis_u2 = <double *> malloc(order_u * sizeof(double))
    cdef double *basis_v0 = <double *> malloc(order_v * sizeof(double))
    cdef double *basis_v1 = <double *> malloc(order_v * sizeof(double))
    cdef double *basis_v2 = <double *> malloc(order_v * sizeof(double))

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
    cdef double distance2
    cdef double tem_distance2

    # normalizing vars
    cdef double P_corner00[3], P_corner11[3], P_corner10[3], P_corner01[3]
    cdef double diag_1[3]
    cdef double diag_2[3]
    cdef double diag_dist_1
    cdef double diag_dist_2
    cdef size_norm_factor
                
    # get_open_uniform(order_u, num_control_points_u, knot_vector_u)
    # get_open_uniform(order_v, num_control_points_v, knot_vector_v)
     
    # multi surfs
    cdef int num_surfs, num_control_points_u, num_control_points_v
    cdef int surf_num_points = num_points
    import numpy as np
    cdef double[:] surfs_dist = np.empty(num_points, dtype=np.double)
    cdef double[:] u_vec_ini = np.empty(num_points, dtype=np.double)
    cdef double[:] v_vec_ini = np.empty(num_points, dtype=np.double)

    for i in range(num_points) :
        surfs_dist[i] = 1e16
    num_surfs = len(surfs_order_u)
    #print('num_surf',num_surfs)
    for ns in range(num_surfs):

        order_u = surfs_order_u[ns]
        order_v = surfs_order_v[ns]
        num_control_points_u = surfs_num_control_points_u[ns]
        num_control_points_v = surfs_num_control_points_v[ns]
        # get_open_uniform(order_u, num_control_points_u, knot_vector_u)
        # get_open_uniform(order_v, num_control_points_v, knot_vector_v)
        for i in range(len(surfs_cps[ns,:])):
            cps[i] = surfs_cps[ns,i]

        # Getting normalizing factor measuring surface size

        x[0] = 0.
        x[1] = 0.
        i_start_u = get_basis0(
            order_u, num_control_points_u, x[0], knot_vector_u, basis_u0)
        i_start_v = get_basis0(
            order_v, num_control_points_v, x[1], knot_vector_v, basis_v0)
        for k in range(3): 
            P_corner00[k] = 0.
            for i_order_u in range(order_u):
                for i_order_v in range(order_v):
                    index = 3 * num_control_points_v * (i_start_u + i_order_u) \
                        + 3 * (i_start_v + i_order_v) + k
                    C[k] = cps[index]

                    P_corner00[k] = P_corner00[k] + basis_u0[i_order_u] * basis_v0[i_order_v] * C[k]

        x[0] = 1.
        x[1] = 1.
        i_start_u = get_basis0(
            order_u, num_control_points_u, x[0], knot_vector_u, basis_u0)
        i_start_v = get_basis0(
            order_v, num_control_points_v, x[1], knot_vector_v, basis_v0)
        for k in range(3): 
            P_corner11[k] = 0.
            for i_order_u in range(order_u):
                for i_order_v in range(order_v):
                    index = 3 * num_control_points_v * (i_start_u + i_order_u) \
                        + 3 * (i_start_v + i_order_v) + k
                    C[k] = cps[index]

                    P_corner11[k] = P_corner11[k] + basis_u0[i_order_u] * basis_v0[i_order_v] * C[k]

        x[0] = 0.
        x[1] = 1.
        i_start_u = get_basis0(
            order_u, num_control_points_u, x[0], knot_vector_u, basis_u0)
        i_start_v = get_basis0(
            order_v, num_control_points_v, x[1], knot_vector_v, basis_v0)
        for k in range(3): 
            P_corner01[k] = 0.
            for i_order_u in range(order_u):
                for i_order_v in range(order_v):
                    index = 3 * num_control_points_v * (i_start_u + i_order_u) \
                        + 3 * (i_start_v + i_order_v) + k
                    C[k] = cps[index]

                    P_corner01[k] = P_corner01[k] + basis_u0[i_order_u] * basis_v0[i_order_v] * C[k]

        x[0] = 1.
        x[1] = 0.
        i_start_u = get_basis0(
            order_u, num_control_points_u, x[0], knot_vector_u, basis_u0)
        i_start_v = get_basis0(
            order_v, num_control_points_v, x[1], knot_vector_v, basis_v0)
        for k in range(3): 
            P_corner10[k] = 0.
            for i_order_u in range(order_u):
                for i_order_v in range(order_v):
                    index = 3 * num_control_points_v * (i_start_u + i_order_u) \
                        + 3 * (i_start_v + i_order_v) + k
                    C[k] = cps[index]

                    P_corner10[k] = P_corner10[k] + basis_u0[i_order_u] * basis_v0[i_order_v] * C[k]

        for k in range(3):
            diag_1[k] = P_corner11 - P_corner00
            diag_2[k] = P_corner01 - P_corner10

        diag_dist_1 = norm(3, diag_1)
        diag_dist_2 = norm(3, diag_2)
        size_norm_factor = diag_dist_1*diag_dist_2

        #tom's initial guess implementation
        if not (guess_grid_n == 0) :
            #print('initial guess implementation')
            for i in range(num_points) :
                u_closest = .5
                v_closest = .5
                distance = 1000.
                distance2 = 1000.

                for k in range(3):
                    P[k] = pts[3 * i + k]
                    A[k] = axis[3 * i + k]


                for a in range(guess_grid_n) :
                    for b in range(guess_grid_n) :
                        
                        x[0] = a/(guess_grid_n - 1)
                        x[1] = b/(guess_grid_n - 1)

                        i_start_u = get_basis0(
                            order_u, num_control_points_u, x[0], knot_vector_u, basis_u0)
                        i_start_v = get_basis0(
                            order_v, num_control_points_v, x[1], knot_vector_v, basis_v0)
                        #print('x0',x[0],'x1',x[1])
                        #print('i_start_u',i_start_u,'i_start_v',i_start_v)
                        for k in range(3): 
                            P00[k] = 0.
                            for i_order_u in range(order_u):
                                for i_order_v in range(order_v):
                                    index = 3 * num_control_points_v * (i_start_u + i_order_u) \
                                        + 3 * (i_start_v + i_order_v) + k
                                    C[k] = cps[index]

                                    P00[k] = P00[k] + basis_u0[i_order_u] * basis_v0[i_order_v] * C[k]
                        for k in range(3) :
                            D[k] = P00[k] - P[k]
                        temp_distance = 0
                        temp_product = 0
                        if norm(3,A) == 0:
                            temp_distance = norm(3,D)
                            if temp_distance < distance :
                                u_closest = x[0]
                                v_closest = x[1]
                                distance = temp_distance
                        else:                       
                            for k in range(3):
                                nA[k] = A[k]/norm(3,A)
                            #print(i,'n1:',n1,'n2',n2)
                            tao = 1e-5
                            temp_distance2 = dot(3,D,D) - dot(3,D,nA)**2 + tao*dot(3,D,D)#dot(3, n1, n2)
                            temp_distance2 = temp_distance2/size_norm_factor
                            #print(i, temp_product)
                            if temp_distance2 < distance2 :
                                #print('test',i, temp_product)
                                u_closest = x[0]
                                v_closest = x[1]
                                distance2 = temp_distance2
                #print(i, product)  
                
                u_vec_ini[i] = u_closest
                v_vec_ini[i] = v_closest
                #print(i)
                #print('uv_vec_ini',i, u_vec[i],v_vec[i])
                #print('uv_vec_initial',i, u_vec_initial[i],v_vec_initial[i])
        #exit()    

        #for i in range(num_points) :
            #u_vec[i] = u_vec_initial[i]
            #v_vec[i] = v_vec_initial[i]










        for i_pt in range(num_points):
            x[0] = u_vec_ini[i_pt]
            x[1] = v_vec_ini[i_pt]

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
                for k in range(3):
                    P00[k] = 0.
                    P10[k] = 0.
                    P01[k] = 0.
                    P20[k] = 0.
                    P11[k] = 0.
                    P02[k] = 0.

                    for i_order_u in range(order_u):
                        for i_order_v in range(order_v):
                            index = 3 * num_control_points_v * (i_start_u + i_order_u) \
                                + 3 * (i_start_v + i_order_v) + k
                            C[k] = cps[index]
                            #print(i_pt, i_iter,i_order_u,i_order_v,'C',C[k])
                            #print(basis_u1[i_order_u],basis_v0[i_order_v],basis_u2[i_order_u],basis_v0[i_order_v])
                            P00[k] = P00[k] + basis_u0[i_order_u] * basis_v0[i_order_v] * C[k]
                            P10[k] = P10[k] + basis_u1[i_order_u] * basis_v0[i_order_v] * C[k]
                            P01[k] = P01[k] + basis_u0[i_order_u] * basis_v1[i_order_v] * C[k]
                            P20[k] = P20[k] + basis_u2[i_order_u] * basis_v0[i_order_v] * C[k]
                            P11[k] = P11[k] + basis_u1[i_order_u] * basis_v1[i_order_v] * C[k]
                            P02[k] = P02[k] + basis_u0[i_order_u] * basis_v2[i_order_v] * C[k]
                    '''
                    if norm(3,A) == 0:
                        for k in range(3) :
                            D[k] = P00[k] - P[k]
                    else:
                        for k in range(3) :
                            D[k] = P00[k] - P[k]
                    '''
                    D[k] = P00[k] - P[k]
                '''min F:= (||D[k]||)^2'''
                #print(i_pt, i_iter,'D',D,'P00',P00,'P10',P10,'P01',P01,'P20',P20,'P02',P02)
                if norm(3,A) == 0:
                    alpha = 1
                    G[0] = 2 * dot(3, D, P10)
                    G[1] = 2 * dot(3, D, P01)
                    H[0] = 2 * dot(3, P10, P10) + 2 * dot(3, D, P20)
                    H[1] = 2 * dot(3, P10, P01) + 2 * dot(3, D, P11)
                    H[2] = H[1]
                    H[3] = 2 * dot(3, P01, P01) + 2 * dot(3, D, P02)

                else: 
                    #print(i_pt, P00, P10, P01, P)
                    for k in range(3):                                  
                        #nD[k] = D[k]/norm(3,D)
                        nA[k] = A[k]/norm(3,A)
                        #D2[k] = 2*D[k]
                    alpha = 1
                    norm_D = norm(3,D)
                    norm_D2 = dot(3,D,D)
                    dot_nA_D = dot(3,nA,D)
                    dot_nA_P10 = dot(3,nA,P10)
                    dot_nA_P01 = dot(3,nA,P01)
                    dot_nA_P20 = dot(3,nA,P20)
                    dot_nA_P11 = dot(3,nA,P11)
                    dot_nA_P02 = dot(3,nA,P02)
                    dot_D_P10 = dot(3,D,P10)
                    dot_D_P01 = dot(3,D,P01)
                    dot_D_P20 = dot(3,D,P20)
                    dot_D_P11 = dot(3,D,P11)
                    dot_D_P02 = dot(3,D,P02)
                    dot_P10_P10 = dot(3,P10,P10)
                    dot_P01_P01 = dot(3,P01,P01)
                    dot_P01_P10 = dot(3,P01,P10)
                    #ddu_normD2 = 2*dot_D_P10
                    ddu_normD = 0.5*(norm_D2**(-0.5))*2*dot_D_P10 
                    ddu2_normD = -norm_D2**(-3/2)*dot_D_P10*dot_D_P10 + norm_D2**(-0.5)*dot_P10_P10 + norm_D2**(-0.5)*dot_D_P20
                    ddv_normD = 0.5*(norm_D2**(-0.5))*2*dot_D_P01
                    ddv2_normD = -norm_D2**(-3/2)*dot_D_P01*dot_D_P01 + norm_D2**(-0.5)*dot_P01_P01 + norm_D2**(-0.5)*dot_D_P02
                    dduddv_normD = -norm_D2**(-3/2)*dot_D_P01*dot_D_P10 + norm_D2**(-0.5)*dot_P01_P10 + norm_D2**(-0.5)*dot_D_P11

                    grad_u_numer = dot_nA_P10*norm_D - dot_nA_D*ddu_normD 
                    grad_denom = norm_D2 
                    ddu_grad_u_numer = dot_nA_P10*ddu_normD + dot_nA_P20*norm_D - (dot_nA_P10*ddu_normD + ddu2_normD*dot_nA_D) 
                    ddu_grad_denom = 2*norm_D*ddu_normD

                    grad_v_numer = dot_nA_P01*norm_D - dot_nA_D*ddv_normD 
                    ddv_grad_v_numer = dot_D_P01*ddv_normD + dot_nA_P02*norm_D - (dot_nA_P01*ddv_normD + ddv2_normD*dot_nA_D)
                    ddv_grad_denom = 2*norm_D*ddv_normD

                    ddv_grad_u_numer = dot_nA_P10*ddv_normD + dot_nA_P11*norm_D - (dot_nA_P01*ddu_normD + dduddv_normD*dot_nA_D)
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
                    G[0] = (2 * dot(3, D, P10) - 2 * dot(3, D, nA) * dot(3, P10, nA) + tao*2*dot(3, D, P10))/size_norm_factor
                    G[1] = (2 * dot(3, D, P01) - 2 * dot(3, D, nA) * dot(3, P01, nA) + tao*2*dot(3, D, P01))/size_norm_factor
                    H[0] = (2 * dot(3, P10, P10) + 2 * dot(3, D, P20) - 2 * dot(3, P10, nA) * dot(3, P10, nA) - 2 * dot(3, D, nA) * dot(3, P20, nA) \
                        + tao*(2 * dot(3, P10, P10) + 2 * dot(3, D, P20)))/size_norm_factor
                    H[1] = (2 * dot(3, P10, P01) + 2 * dot(3, D, P11) - 2 * dot(3, P10, nA) * dot(3, P01, nA) - 2 * dot(3, D, nA) * dot(3, P11, nA) \
                        + tao*(2 * dot(3, P10, P01) + 2 * dot(3, D, P11)))/size_norm_factor
                    H[2] = (2 * dot(3, P10, P01) + 2 * dot(3, D, P11) - 2 * dot(3, P01, nA) * dot(3, P10, nA) - 2 * dot(3, D, nA) * dot(3, P11, nA) \
                        + tao*(2 * dot(3, P10, P01) + 2 * dot(3, D, P11)))/size_norm_factor
                    H[3] = (2 * dot(3, P01, P01) + 2 * dot(3, D, P02) - 2 * dot(3, P01, nA) * dot(3, P01, nA) - 2 * dot(3, D, nA) * dot(3, P02, nA) \
                        + tao*(2 * dot(3, P01, P01) + 2 * dot(3, D, P02)))/size_norm_factor
                    '''                     
                    min F:= -nA*nD = -nA*(P00 - P)/norm_D
                    '''
                #print(i_pt, i_iter, H[0], H[1], H[2], H[3],G[0], G[1])
                for k in range(2):
                    if (
                        (x[k] < 1e-14) and (G[k] > -1e-16) or #add condition of G=0 to solve the zerodivision error with det(H) = 0
                        (x[k] > (1 - 1e-14)) and (G[k] < 1e-16)
                    ):
                        #print('test1',i_pt, i_iter, x[k],G[k],H[0], H[1], H[2], H[3])
                        G[k] = 0.
                        H[1] = 0.
                        H[2] = 0.
                        H[k * 3] = 1.
                
                # print(i_pt, i_iter, H[0], H[1], H[2], H[3])
                det = H[0] * H[3] - H[1] * H[2]
                # if det == 0:
                #     print(i_pt, i_iter, 'H', H[0], H[1], H[2], H[3])
                #     print('G', G)
                #     print(x)
                for k in range(2):#wedge case
                    if (abs(H[k * 3] - 0) < 1e-16):
                        H[k * 3] = 1.
                N[0] = H[3] / det
                N[1] = -H[1] / det
                N[2] = N[1]
                N[3] = H[0] / det
                #print('test4',G)
                matvec(2, 2, N, G, dx)
                #print('dx',dx)
                if norm(3,A) == 0:
                    for k in range(2):
                        dx[k] = -dx[k]
                    for k in range(2):
                        if x[k] + dx[k] < 0:                        
                            dx[k] = -x[k]
                        elif x[k] + dx[k] > 1:
                            dx[k] = 1 - x[k]
                else:
                    for k in range(2):
                        dx[k] = -dx[k]            
                    for k in range(2):                                  
                        if x[k] + alpha*dx[k] < 0:                   
                            dx[k] = (-x[k])/alpha
                        elif x[k] + alpha*dx[k] > 1:
                            #print('yes2', x[k]+dx[k],x[k],dx[k])
                            dx[k] = (1 - x[k])/alpha#- x[k]+ (2 - x[k] - dx[k])#- x[k] + (x[k] + dx[k]-1)

                norm_G = norm(2, G)
                norm_dx = norm(2, dx)

                if norm(3,A) == 0:
                    if norm_G < 1e-16 or norm_dx < 1e-16:
                        #print("solution found",norm_G,norm_dx)
                        break
                else:
                    for k in range(2):
                        dx[k] = alpha*dx[k]
                    norm_dx = norm(2, dx)
                    if norm_G < 1e-16 or norm_dx < 1e-16:#(1 - temp_product) < 1e-6 or (temp_product+1) < 1e-6:#norm_dx < 8e-4:# norm_G < 1e-14 or  or # i_iter ==0 or 
                        #print(ns,i_pt,'iter', i_iter, 'norm_G',norm_G, 'norm_dx',norm_dx)
                        break
                for k in range(2):
                    x[k] = x[k] + dx[k]
                #
            #print(ns,i_pt,'iter', i_iter, 'norm_G',norm_G, 'norm_dx',norm_dx)
            #print(i_pt,'iter',i_iter,'G',G,'H',H,'dx',dx)
            
            #u_vec[i_pt] = x[0]
            #v_vec[i_pt] = x[1]

            #print('u',i_pt,u_vec[i_pt])
            #print('v',i_pt,v_vec[i_pt])
            #print("solution found",norm_G,norm_dx)
            i_start_u = get_basis0(
                order_u, num_control_points_u, x[0], knot_vector_u, basis_u0)
            i_start_v = get_basis0(
                order_v, num_control_points_v, x[1], knot_vector_v, basis_v0)

            for k in range(3): 
                P00[k] = 0.
                for i_order_u in range(order_u):
                    for i_order_v in range(order_v):
                        index = 3 * num_control_points_v * (i_start_u + i_order_u) \
                            + 3 * (i_start_v + i_order_v) + k
                        C[k] = cps[index]
                        P00[k] = P00[k] + basis_u0[i_order_u] * basis_v0[i_order_v] * C[k]
            for k in range(3) :
                D[k] = P00[k] - P[k]
            temp_distance = 0
            surfs_dist_temp = norm(3,D)
            #print(ns,i_pt,"solution found2",x[0],x[1],surfs_dist_temp)
            if surfs_dist_temp < surfs_dist[i_pt]:
                #print(ns,i_pt,"solution found3",x[0],x[1])
                u_vec[i_pt] = x[0]
                v_vec[i_pt] = x[1]
                surfs_dist[i_pt] = surfs_dist_temp
                surfs_index[i_pt] = ns
            #print(ns,i_pt,"solution found4",u_vec[i_pt],v_vec[i_pt])
            #print('basis_u0')
            for i_order_u in range(order_u):
                for i_order_v in range(order_v):
                    pass#print(basis_u0[i_order_u], end=' ')
            #print()
            #print('basis_v0')
            for i_order_u in range(order_u):
                for i_order_v in range(order_v):
                    pass#print(basis_v0[i_order_v], end=' ')
    '''
    print('u_vec111')
    for i in range(num_points):
        print(u_vec[i], end=' ')
    print()
    print('v_vec111')
    for i in range(num_points):
        print(v_vec[i], end=' ')
    '''
    free(basis_u0)
    free(basis_u1)
    free(basis_u2)
    free(basis_v0)
    free(basis_v1)
    free(basis_v2)

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