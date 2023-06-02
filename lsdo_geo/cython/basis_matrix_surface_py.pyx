import numpy as np
cimport numpy as np

from caddee.cython.basis_matrix_surface cimport get_basis_surface_matrix


def get_basis_surface_matrix(
        int order_u, int num_control_points_u, int u_der, np.ndarray[double] u_vec, np.ndarray[double] knot_vector_u,
        int order_v, int num_control_points_v, int v_der, np.ndarray[double] v_vec, np.ndarray[double] knot_vector_v,
        int num_points, 
        np.ndarray[double] data, np.ndarray[int] row_indices, np.ndarray[int] col_indices):
    get_basis_surface_matrix(
        order_u, num_control_points_u, u_der, &u_vec[0], &knot_vector_u[0],
        order_v, num_control_points_v, v_der, &v_vec[0], &knot_vector_v[0],
        num_points, &data[0], &row_indices[0], &col_indices[0],
    )