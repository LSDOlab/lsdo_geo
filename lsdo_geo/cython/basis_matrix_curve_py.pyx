import numpy as np
cimport numpy as np

from lsdo_geo.cython.basis_matrix_curve cimport get_basis_curve_matrix


def get_basis_curve_matrix(
        int order, int num_coefficients, int u_der, np.ndarray[double] u_vec, np.ndarray[double] knot_vector,
        int num_points, 
        np.ndarray[double] data, np.ndarray[int] row_indices, np.ndarray[int] col_indices):
    get_basis_curve_matrix(
        order, num_coefficients, u_der, &u_vec[0], &knot_vector[0],
        num_points, &data[0], &row_indices[0], &col_indices[0],
    )