from libc.stdlib cimport malloc, free

from lsdo_geo.cython.get_open_uniform cimport get_open_uniform
from lsdo_geo.cython.basis0 cimport get_basis0
from lsdo_geo.cython.basis1 cimport get_basis1
from lsdo_geo.cython.basis2 cimport get_basis2
from lsdo_geo.cython.basis_matrix_surface cimport get_basis_surface_matrix


cdef compute_surface_projection(
    long[:] surfs_order_u, long[:] surfs_num_coefficients_u,
    long[:] surfs_order_v, long[:] surfs_num_coefficients_v,
    int num_points, int max_iter,
    double* pts, double* cps,
    double* knot_vector_u, double* knot_vector_v,
    double* u_vec, double* v_vec,
    int guess_grid_n,
    double* axis,
    long[:] surfs_index,
    double[:,:] surfs_cps,
    #double* u_vec_initial, double* v_vec_initial,
)