from libc.stdlib cimport malloc, free

from caddee.cython.get_open_uniform cimport get_open_uniform
from caddee.cython.basis0 cimport get_basis0
from caddee.cython.basis1 cimport get_basis1
from caddee.cython.basis2 cimport get_basis2
from caddee.cython.basis_matrix_volume cimport get_basis_volume_matrix


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
)