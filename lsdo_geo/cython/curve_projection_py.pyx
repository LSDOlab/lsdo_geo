import numpy as np
cimport numpy as np

from caddee.cython.curve_projection cimport compute_curve_projection


def compute_curve_projection(
    int order_u, int num_control_points_u,
    int num_points, int max_iter,
    np.ndarray[double] pts,  np.ndarray[double] cps,
    np.ndarray[double] u_vec, np.ndarray[double] knot_vector,
    int n_guesses,
):
    compute_curve_projection(
        order_u, num_control_points_u,
        num_points, max_iter,
        &pts[0], &cps[0],
        &u_vec[0], &knot_vector[0],
        n_guesses,
    )