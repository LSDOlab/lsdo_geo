import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps

from basis_matrix_curve_py import get_basis_curve_matrix
from get_open_uniform_py import get_open_uniform


num_points = 100
order = 4
num_coefficients = 20

u_vec = np.linspace(0., 1., num_points)

data = np.zeros(num_points * order)
row_indices = np.zeros(num_points * order, np.int32)
col_indices = np.zeros(num_points * order, np.int32)

knot_vector = np.zeros(num_coefficients+order)
get_open_uniform(order, num_coefficients, knot_vector)

get_basis_curve_matrix(order, num_coefficients, 0, u_vec, knot_vector, num_points, data, row_indices, col_indices)
basis0 = sps.csc_matrix((data, (row_indices, col_indices)), shape=(num_points, num_coefficients))

get_basis_curve_matrix(order, num_coefficients, 1, u_vec, knot_vector, num_points, data, row_indices, col_indices)
basis1 = sps.csc_matrix((data, (row_indices, col_indices)), shape=(num_points, num_coefficients))

get_basis_curve_matrix(order, num_coefficients, 2, u_vec, knot_vector, num_points, data, row_indices, col_indices)
basis2 = sps.csc_matrix((data, (row_indices, col_indices)), shape=(num_points, num_coefficients))

cps = np.zeros((num_coefficients, 2))
cps[:, 0] = np.linspace(0., 1., num_coefficients)
cps[:, 1] = np.linspace(0., 1., num_coefficients)

cps[int(num_coefficients/2), 1] += 0.2

pts = basis0.dot(cps)
derivs1 = basis1.dot(cps)
derivs2 = basis2.dot(cps)

plt.subplot(2, 1, 1)
plt.plot(pts[:, 0], pts[:, 1], 'ok-')
plt.plot(cps[:, 0], cps[:, 1], 'or')
plt.title('Curve')

plt.subplot(2, 1, 2)
plt.plot(pts[:, 0], derivs1[:, 1] / derivs1[:, 0], 'ok-')
plt.title('dy/dx')

plt.show()

