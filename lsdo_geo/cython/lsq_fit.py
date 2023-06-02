import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve

from caddee.cython.basis_matrix_curve_py import get_basis_curve_matrix


def get_lsq_fit(
        pts_ndarray, num_control_points
        ):

    order = 4
    num_points = pts_ndarray[:,0].shape[0]
    u_vec = np.linspace(0., 1., num_points)

    data = np.zeros(num_points * order)
    row_indices = np.zeros(num_points * order, np.int32)
    col_indices = np.zeros(num_points * order, np.int32)

    get_basis_curve_matrix(order, num_control_points, 0, u_vec, num_points, data, row_indices, col_indices)
    basis0 = sps.csc_matrix((data, (row_indices, col_indices)), shape=(num_points, num_control_points))

    bTb = basis0.T.dot(basis0)
    rhs = basis0.T.dot(pts_ndarray)
    
    cp_array = sps.linalg.spsolve(bTb, rhs)

    return cp_array

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # data = [
    # 1.00000,     0.00105,
    # 0.95041,     0.00990,
    # 0.90067,     0.01816,
    # 0.80097,     0.03296,
    # 0.70102,     0.04551,
    # 0.60085,     0.05580,
    # 0.50049,     0.06356,
    # 0.40000,     0.06837,
    # 0.29875,     0.06875,
    # 0.24814,     0.06668,
    # 0.19761,     0.06276,
    # 0.14722,     0.05665,
    # 0.09710,     0.04766,
    # 0.07217,     0.04169,
    # 0.04742,     0.03420,
    # 0.02297,     0.02411,
    # 0.01098,     0.01694,
    # 0.00000,     0.00000
    # ]

    # pts = np.zeros((round(len(data)/2),2))
    # for i in np.arange(round(len(data)/2)):
    #     for k in np.arange(2):
    #         pts[i, k] = data[i*2+k]

    pts = np.array([
        [1.00000,     0.00105],
        [0.95041,     0.00990],
        [0.90067,     0.01816],
        [0.80097,     0.03296],
        [0.70102,     0.04551],
        [0.60085,     0.05580],
        [0.50049,     0.06356],
        [0.40000,     0.06837],
        [0.29875,     0.06875],
        [0.24814,     0.06668],
        [0.19761,     0.06276],
        [0.14722,     0.05665],
        [0.09710,     0.04766],
        [0.07217,     0.04169],
        [0.04742,     0.03420],
        [0.02297,     0.02411],
        [0.01098,     0.01694],
        [0.00000,     0.00000],
        [0.01402,     -0.01448],
        [0.02703,     -0.01927],
        [0.05258,     -0.02482],
        [0.07783,     -0.02809],
        [0.10290,     -0.03016],
        [0.15278,     -0.03227],
        [0.20239,     -0.03276],
        [0.25186,     -0.03230],
        [0.30125,     -0.03125],
        [0.40000,     -0.02837],
        [0.49951,     -0.02468],
        [0.59915,     -0.02024],
        [0.69898,     -0.01551],
        [0.79903,     -0.01074],
        [0.89933,     -0.00594],
        [0.94959,     -0.00352],
        [1.00000,     -0.00105],
    ])
    num_control_points = 15

    cp_array = get_lsq_fit(pts,num_control_points)
    # print(cp_array)
    plt.plot(pts[:, 0], pts[:, 1], 'ok-')
    plt.plot(cp_array[:, 0], cp_array[:, 1], 'or')
    plt.title('Control pts given set points')

    plt.show()