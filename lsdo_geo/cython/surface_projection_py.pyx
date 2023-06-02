import numpy as np
cimport numpy as np

from caddee.cython.surface_projection cimport compute_surface_projection


def compute_surface_projection(
    np.ndarray[np.int64_t] surfs_order_u, np.ndarray[np.int64_t] surfs_num_control_points_u,
    np.ndarray[np.int64_t] surfs_order_v, np.ndarray[np.int64_t] surfs_num_control_points_v,
    int num_points, int max_iter,
    np.ndarray[double] pts,  np.ndarray[double] cps,
    np.ndarray[double] knot_vector_u, np.ndarray[double] knot_vector_v,
    np.ndarray[double] u_vec, np.ndarray[double] v_vec,
    int guess_grid_n,
    np.ndarray[double] axis,
    np.ndarray[np.int64_t] surfs_index,
    np.ndarray[double, ndim=2] surfs_cps,
):
    compute_surface_projection(
        surfs_order_u, surfs_num_control_points_u,
        surfs_order_u, surfs_num_control_points_v,
        num_points, max_iter,
        &pts[0], &cps[0],
        &knot_vector_u[0], &knot_vector_v[0],
        &u_vec[0], &v_vec[0],
        guess_grid_n,
        &axis[0],
        surfs_index,
        surfs_cps,
        #&u_vec_initial[0], &v_vec_initial[0],
    )