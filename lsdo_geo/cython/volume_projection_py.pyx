import numpy as np
cimport numpy as np

from caddee.cython.volume_projection cimport compute_volume_projection


def compute_volume_projection(
    int order_u, int num_control_points_u, 
    int order_v, int num_control_points_v,
    int order_w, int num_control_points_w,
    int num_points, int max_iter,
    np.ndarray[double] pts,  np.ndarray[double] cps,
    np.ndarray[double] knot_vector_u, np.ndarray[double] knot_vector_v, np.ndarray[double] knot_vector_w,
    np.ndarray[double] u_vec, np.ndarray[double] v_vec, np.ndarray[double] w_vec,
    int guess_grid_n,
    np.ndarray[double] axis,
):
    compute_volume_projection(
        order_u, num_control_points_u,
        order_v, num_control_points_v,
        order_w, num_control_points_w,
        num_points, max_iter,
        &pts[0], &cps[0],
        &knot_vector_u[0], &knot_vector_v[0], &knot_vector_w[0],
        &u_vec[0], &v_vec[0], &w_vec[0],
        guess_grid_n,
        &axis[0],
    )