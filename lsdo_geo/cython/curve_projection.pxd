from libc.stdlib cimport malloc, free

from caddee.cython.get_open_uniform cimport get_open_uniform
from caddee.cython.basis0 cimport get_basis0
from caddee.cython.basis1 cimport get_basis1
from caddee.cython.basis2 cimport get_basis2
from caddee.cython.basis_matrix_curve cimport get_basis_curve_matrix


cdef compute_curve_projection(
    int order_u, int num_control_points_u,
    int num_points, int max_iter,
    double* pts, double* cps,
    double* u_vec, double* knot_vector,
    int n_guesses,
)